import os
import json
import argparse
import collections
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from peft import (
    LoraConfig,
    get_peft_model,
    load_peft_weights,
    set_peft_model_state_dict,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MultimodalCBIRDataset(Dataset):
    MAX_TOKENS: int = 77

    def __init__(self, json_path: str, img_dir: str, processor: CLIPProcessor):
        with open(json_path, "r") as f:
            raw = json.load(f)

        self.img_dir = img_dir
        self.processor = processor

        models = raw["metadata"]["models_evaluated"]
        if len(models) != 1:
            raise ValueError(
                f"Expected exactly 1 model in 'models_evaluated', got: {models}. "
                "Edit the JSON metadata to contain only the model you want to use."
            )
        self.model_key = models[0]

        # Build a stable class→int mapping so label integers are consistent
        # across runs (sorted alphabetically).
        all_class_labels = sorted(
            {meta["class_label"] for meta in raw["images"].values()}
        )
        self.class_to_int: dict[str, int] = {
            cls: i for i, cls in enumerate(all_class_labels)
        }

        self.items = []
        self.class_labels: list[str] = []  # string labels (for stratified split)
        self.class_ints: list[int] = []  # integer labels (fed into UniCL loss)
        truncated_count = 0

        print("[INFO] Pre-tokenising captions...")
        for rel_path, meta in raw["images"].items():
            caps = meta["captions"][self.model_key]
            text = f"{caps['primary']}, {caps['extended']}"

            encoded = processor.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.MAX_TOKENS,
                return_tensors="pt",
            )

            token_ids = processor.tokenizer.encode(text)
            if len(token_ids) > self.MAX_TOKENS:
                truncated_count += 1

            self.items.append(
                {
                    "rel_path": rel_path,
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                }
            )
            cls_str = meta["class_label"]
            self.class_labels.append(cls_str)
            self.class_ints.append(self.class_to_int[cls_str])

        if truncated_count:
            print(
                f"[WARNING] {truncated_count}/{len(self.items)} captions exceed "
                f"{self.MAX_TOKENS} tokens and will be truncated."
            )
        n_classes = len(self.class_to_int)
        print(
            f"[INFO] Pre-tokenisation complete. {len(self.items)} items, "
            f"{n_classes} classes cached."
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = os.path.join(self.img_dir, item["rel_path"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image at '{img_path}'. "
                "Check that the file exists and is a valid image."
            ) from exc

        pixel_values = self.processor.image_processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            # ← NEW: integer class label for UniCL loss
            "label": torch.tensor(self.class_ints[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def stratified_split(
    dataset: MultimodalCBIRDataset,
    val_fraction: float,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    rng = random.Random(seed)

    class_to_indices: dict[str, list[int]] = collections.defaultdict(list)
    for idx, label in enumerate(dataset.class_labels):
        class_to_indices[label].append(idx)

    train_indices, val_indices = [], []
    for label, indices in class_to_indices.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_val = max(1, math.ceil(val_fraction * len(shuffled)))
        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])

    print(
        f"[INFO] Stratified split: {len(train_indices)} train / "
        f"{len(val_indices)} val across {len(class_to_indices)} classes."
    )

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


# ---------------------------------------------------------------------------
# UniCL loss  (Yang et al., CVPR 2022 — Algorithm 1)
#
# Implements bidirectional contrastive loss in image-text-label space.
# Unlike plain InfoNCE (CLIP), same-class samples within a batch are treated
# as POSITIVE pairs, not negatives. This directly satisfies all three goals:
#   1. Same-class images → positive (via shared text alignment)
#   2. Same-class texts  → positive (via shared image alignment)
#   3. Matched image-text pairs → positive (the standard CLIP objective)
#
# When every item in the batch has a unique label (i.e. no same-class pairs),
# the loss reduces exactly to symmetric InfoNCE / CLIP loss.
# ---------------------------------------------------------------------------


def _soft_cross_entropy(
    logits: torch.Tensor, soft_targets: torch.Tensor
) -> torch.Tensor:
    """
    Soft-target cross-entropy following the SoftCE function in Algorithm 1.

    logits       : (B, B)  raw similarity scores (already scaled by temperature)
    soft_targets : (B, B)  non-negative matrix; rows need not sum to 1 yet
                            (we normalise here, matching the paper's
                            loss/target.sum(dim=-1) formulation)
    """
    log_probs = F.log_softmax(logits, dim=-1)  # (B, B)
    # Normalise each row of the target so it sums to 1.
    # Rows with no positives (shouldn't happen) are guarded with clamp.
    target_norm = soft_targets / soft_targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    loss = -(target_norm * log_probs).sum(dim=-1)  # (B,)
    return loss.mean()


def unicl_loss(
    image_embeds: torch.Tensor,  # (B, D)  L2-normalised
    text_embeds: torch.Tensor,  # (B, D)  L2-normalised
    labels: torch.Tensor,  # (B,)    integer class indices
    logit_scale: torch.Tensor,  # scalar  learnable log-temperature
) -> torch.Tensor:
    """
    Unified Contrastive Learning loss (Yang et al., CVPR 2022).

    Positive mask: target[i, j] = 1  iff  labels[i] == labels[j]
    This turns the hard diagonal positives of InfoNCE into a soft multi-positive
    target, which is what makes the loss aware of class structure across both
    modalities.
    """
    # Clamp log-temperature and cast to embedding dtype so the temperature
    # scaling is numerically consistent under autocast (float16 on CUDA).
    # Ceiling of 50 (not 100): CLIP literature finds optimal scale is 20-50
    # on small/medium datasets; 100 collapses the softmax and causes overfitting.
    scale = logit_scale.exp().clamp(max=50.0).to(image_embeds.dtype)

    # (B, B) cosine similarity matrix
    logits = scale * (image_embeds @ text_embeds.T)

    # Build binary positive mask from class labels.
    # target[i, j] = 1 iff labels[i] == labels[j]  (row i = query i's positives)
    # unsqueeze(1) gives (B,1), unsqueeze(0) gives (1,B) → broadcasts to (B,B)
    # This matches P(i) = {k | y_k == y_i} from Algorithm 1 of the paper.
    target = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()  # (B, B)

    # Bidirectional loss: image→text rows, text→image columns
    loss_i2t = _soft_cross_entropy(logits, target)
    loss_t2i = _soft_cross_entropy(logits.T, target.T)

    return (loss_i2t + loss_t2i) / 2.0


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def build_lora_model(
    model_id: str, lora_rank: int, lora_dropout: float, lora_mlp: bool
):
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)

    # alpha == rank keeps the effective scale of the adapter output constant
    # regardless of the rank chosen, making rank a pure capacity knob.
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "visual_projection",
        "text_projection",
    ]
    if lora_mlp:
        # CLIP's vision and text transformer MLP layers. Adding these gives the
        # encoders more room to adapt representations beyond attention alone.
        # Only enable when the dataset is large enough — watch the train/val gap.
        target_modules += ["fc1", "fc2"]

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,  # alpha == rank: scale is rank-independent
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, processor


def resolve_logit_scale(model: nn.Module, reset: bool = True) -> nn.Parameter:
    """
    Finds logit_scale in the (possibly LoRA-wrapped) model, forces it trainable,
    optionally resets it to the healthy CLIP pretraining start value, and returns it.

    Why the reset matters
    ---------------------
    CLIP's pretrained logit_scale is ln(100) ~= 4.605, giving exp ~= 100 — right
    at the clamp ceiling. With scale frozen at 100 the temperature is 0.01, making
    the softmax extremely peaked and driving overfitting on small datasets.
    Resetting to ln(1/0.07) ~= 2.659 (exp ~= 14.3) — the value CLIP used at the
    START of its own pretraining — gives the optimiser a meaningful range to work
    in. This is standard practice for CLIP fine-tuning.

    reset=False is used on resume: the checkpoint value is restored separately
    by the resume block, so we must not overwrite it here.

    Why requires_grad may be False after LoRA wrapping
    ---------------------------------------------------
    PEFT freezes all base-model parameters before injecting adapters. logit_scale
    is not a weight matrix so no LoRA adapter covers it, leaving it frozen.
    We must explicitly re-enable its gradient here.
    """
    named_params = dict(model.named_parameters())
    candidate_keys = [
        "base_model.model.logit_scale",
        "base_model.model.model.logit_scale",
        "logit_scale",
    ]
    for key in candidate_keys:
        if key in named_params:
            param = named_params[key]

            # Re-enable gradient — PEFT freezes all base params by default.
            param.requires_grad = True

            if reset:
                # Reset to CLIP's pretraining start value so the learnable
                # temperature has room to evolve rather than sitting at the
                # clamp ceiling from epoch 1.
                with torch.no_grad():
                    param.fill_(2.659)  # ln(1 / 0.07) — exp ~= 14.3
                print(
                    f"[INFO] logit_scale found at '{key}'. "
                    f"Reset to {param.item():.4f} (exp={param.exp().item():.2f}). "
                    f"requires_grad={param.requires_grad}"
                )
            else:
                print(
                    f"[INFO] logit_scale found at '{key}'. "
                    f"Current value: {param.item():.4f} (exp={param.exp().item():.2f}). "
                    f"requires_grad={param.requires_grad} "
                    f"(value will be restored from checkpoint)"
                )
            return param

    raise RuntimeError(
        "Could not locate logit_scale in model parameters. "
        f"Searched keys: {candidate_keys}. "
        f"Available keys containing 'logit': "
        f"{[k for k in named_params if 'logit' in k]}"
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args):
    # ── Resolve output dir before any work ────────────────────────────────────
    if args.output_dir is None:
        args.output_dir = f"./output/models/{args.model_id.replace('/', '_')}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    if args.grad_accum < 1:
        raise ValueError(f"--grad-accum must be >= 1, got {args.grad_accum}.")

    effective_batch = args.batch_size * args.grad_accum
    print(
        f"[INFO] Batch size: {args.batch_size}  |  "
        f"Grad accum steps: {args.grad_accum}  |  "
        f"Effective batch size: {effective_batch}"
    )

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    model, processor = build_lora_model(
        args.model_id,
        lora_rank=args.lora_rank,
        lora_dropout=args.lora_dropout,
        lora_mlp=args.lora_mlp,
    )
    model.to(device)

    # ── Resolve logit_scale BEFORE torch.compile ──────────────────────────────
    # torch.compile wraps the model in OptimizedModule which may not expose
    # named_parameters() the same way, making resolve_logit_scale unreliable.
    # reset=False on resume: the checkpoint value is restored in the resume
    # block below and must not be overwritten by the fresh-start reset.
    is_resuming = (
        args.resume
        and os.path.exists(os.path.join(args.output_dir, "latest"))
        and os.path.exists(os.path.join(args.output_dir, "latest", "trainer_state.pt"))
    )
    logit_scale = resolve_logit_scale(model, reset=not is_resuming)

    if args.compile:
        if hasattr(torch, "compile"):
            print("[INFO] Compiling model with torch.compile...")
            model = torch.compile(model)
        else:
            print(
                "[WARNING] --compile requested but torch.compile is not available "
                "(requires PyTorch >= 2.0). Skipping."
            )

    use_amp = device == "cuda"
    # ── FIX: use torch.amp instead of deprecated torch.cuda.amp ──────────────
    scaler = GradScaler(device, enabled=use_amp)

    full_dataset = MultimodalCBIRDataset(args.json_path, args.img_dir, processor)
    train_ds, val_ds = stratified_split(full_dataset, args.val_split, seed=42)

    if len(val_ds) < 2:
        raise ValueError(
            f"Validation set has only {len(val_ds)} sample(s). "
            "InfoNCE/UniCL loss requires at least 2 samples per batch. "
            "Reduce --val-split or use a larger dataset."
        )

    default_workers = 0 if os.name == "nt" else 8
    num_workers = args.num_workers if args.num_workers is not None else default_workers
    persistent = num_workers > 0
    pin = device == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        drop_last=True,  # InfoNCE / UniCL require uniform batch sizes
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        drop_last=False,  # FIX: was True — val should use every sample
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Optimizer steps happen every grad_accum micro-batches.
    steps_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / args.grad_accum)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=optimizer_steps_per_epoch,
        epochs=args.epochs,
        pct_start=0.1,  # ~2–3 epochs warm-up at 30-epoch budget
        anneal_strategy="cos",
        final_div_factor=1000,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 0
    latest_dir = os.path.join(args.output_dir, "latest")
    trainer_state_file = os.path.join(latest_dir, "trainer_state.pt")
    training_log_file = os.path.join(args.output_dir, "training_log.json")
    training_log: list[dict] = []

    if (
        args.resume
        and os.path.exists(latest_dir)
        and os.path.exists(trainer_state_file)
    ):
        print(f"\n[INFO] Found checkpoint at '{latest_dir}'. Resuming...")

        peft_weights = load_peft_weights(latest_dir)
        set_peft_model_state_dict(model, peft_weights)

        state = torch.load(trainer_state_file, map_location=device, weights_only=False)
        start_epoch = state["epoch"] + 1
        best_val_loss = state["best_val_loss"]
        patience_counter = state["patience_counter"]
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])

        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])

        if "logit_scale" in state:
            with torch.no_grad():
                logit_scale.copy_(torch.tensor(state["logit_scale"], device=device))

        saved_steps = state.get("optimizer_steps_per_epoch")
        if saved_steps is not None and saved_steps != optimizer_steps_per_epoch:
            raise RuntimeError(
                f"Dataset size or --grad-accum changed since the last checkpoint: "
                f"saved optimizer_steps_per_epoch={saved_steps}, "
                f"current optimizer_steps_per_epoch={optimizer_steps_per_epoch}. "
                "Delete the checkpoint and retrain from scratch."
            )

        saved_epochs = state.get("total_epochs")
        if saved_epochs is not None and saved_epochs != args.epochs:
            raise RuntimeError(
                f"--epochs changed since the last checkpoint: "
                f"saved total_epochs={saved_epochs}, current --epochs={args.epochs}. "
                "Restore --epochs to its original value or delete the checkpoint."
            )

        if os.path.exists(training_log_file):
            with open(training_log_file, "r") as f:
                training_log = json.load(f)
            print(
                f"[INFO] Loaded {len(training_log)} existing log entries from "
                f"'{training_log_file}'."
            )

        print(
            f"[INFO] Resumed successfully. Starting from Epoch {start_epoch + 1}. "
            f"Current best val loss: {best_val_loss:.4f}\n"
        )
    elif args.resume:
        print(
            "\n[WARNING] --resume flag set but no valid checkpoint found. "
            "Starting from scratch.\n"
        )

    # ── Main training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]")
        ):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)  # ← NEW

            with autocast(device, enabled=use_amp):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                img_emb = outputs.image_embeds / outputs.image_embeds.norm(
                    dim=-1, keepdim=True
                )
                text_emb = outputs.text_embeds / outputs.text_embeds.norm(
                    dim=-1, keepdim=True
                )

                # ── UniCL loss replaces symmetric_infonce_loss ─────────────
                loss = unicl_loss(img_emb, text_emb, labels, logit_scale)

            scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()

            total_train_loss += loss.item()

            is_last_micro_batch = (step + 1) % args.grad_accum == 0
            is_last_batch = (step + 1) == len(train_loader)

            if is_last_micro_batch or is_last_batch:
                scaler.unscale_(optimizer)
                # Include logit_scale explicitly: it may not appear in
                # model.parameters() in all PEFT/compile configurations,
                # and an unclipped temperature gradient can cause divergence.
                params_to_clip = list(model.parameters()) + [logit_scale]
                torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        avg_train = total_train_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]  "
            ):
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)  # ← NEW

                # UniCL requires ≥ 2 samples; skip undersized last batch.
                if pixel_values.size(0) < 2:
                    continue

                # Skip mono-class batches: when all labels are identical every
                # target row is all-ones, log_softmax collapses to uniform, and
                # the loss is 0. This silently deflates avg_val with shuffle=False.
                if labels.unique().numel() == 1:
                    continue

                with autocast(device, enabled=use_amp):
                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                    img_emb = outputs.image_embeds / outputs.image_embeds.norm(
                        dim=-1, keepdim=True
                    )
                    text_emb = outputs.text_embeds / outputs.text_embeds.norm(
                        dim=-1, keepdim=True
                    )
                    loss = unicl_loss(img_emb, text_emb, labels, logit_scale)

                total_val_loss += loss.item()
                val_batches += 1

        avg_val = total_val_loss / max(val_batches, 1)

        if val_batches == 0:
            raise RuntimeError(
                "All validation batches were skipped (every batch was either "
                "size < 2 or mono-class). This should not happen with Corel-10K "
                "and val_split=0.1. Check your dataset or reduce --val-split."
            )
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train loss: {avg_train:.4f} | "
            f"val loss: {avg_val:.4f} | "
            f"lr: {current_lr:.2e} | "
            f"logit_scale: {logit_scale.exp().item():.2f}"
        )

        improved = avg_val < (best_val_loss - args.min_delta)
        if improved:
            best_val_loss = avg_val
            patience_counter = 0
        else:
            patience_counter += 1

        training_log.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(avg_train, 6),
                "val_loss": round(avg_val, 6),
                "lr": round(current_lr, 8),
                "logit_scale": round(logit_scale.exp().item(), 4),
                "best_val_loss": round(best_val_loss, 6),
                "improved": improved,
            }
        )
        with open(training_log_file, "w") as f:
            json.dump(training_log, f, indent=2)

        os.makedirs(latest_dir, exist_ok=True)
        model.save_pretrained(latest_dir)
        processor.save_pretrained(latest_dir)
        torch.save(
            {
                "epoch": epoch,
                "total_epochs": args.epochs,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "logit_scale": logit_scale.item(),
            },
            trainer_state_file,
        )

        if improved:
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print(f"  ✓ New best checkpoint saved (val loss: {best_val_loss:.4f})")
        else:
            print(
                f"  ! No significant improvement (threshold={args.min_delta}). "
                f"Patience: {patience_counter}/{args.patience}"
            )

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model weights saved in: {args.output_dir}")
    print(f"Training log saved to: {training_log_file}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune CLIP ViT-B/32 with LoRA + UniCL loss for CBIR"
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    p.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    p.add_argument(
        "--json-path",
        default=os.path.join("output/captions", "Corel-10K_captions.json"),
    )
    p.add_argument("--img-dir", default=os.path.join("data", "Corel-10K"))
    p.add_argument(
        "--output-dir",
        default=None,  # FIX: resolved in train() to avoid recursive p.parse_args()
        help=(
            "Path to save the best checkpoint. "
            "Defaults to ./output/models/<model-id>."
        ),
    )

    # ── Training schedule ─────────────────────────────────────────────────────
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help=(
            "Per-GPU micro-batch size. "
            "Reduced from 128: UniCL already leverages same-class pairs as "
            "implicit positives, so the contrastive signal is richer per sample. "
            "This also reduces the chance of a batch being all same-class after "
            "drop_last, which would make the loss trivially 0."
        ),
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help=(
            "Micro-batches to accumulate before an optimiser step. "
            "Effective batch size = batch-size * grad-accum = 256. "
            "Larger effective batches increase the expected number of "
            "same-class pairs in each UniCL similarity matrix, which "
            "directly strengthens the multi-positive contrastive signal."
        ),
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=30,
        help=(
            "Total training epochs. "
            "Reduced from 50: UniCL's stronger label-aware signal converges "
            "faster than plain InfoNCE, and early stopping handles the rest."
        ),
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help=(
            "Peak learning rate. "
            "Raised from 2e-5 to 1e-4: UniCL uses a richer positive signal "
            "and the LoRA adapters have relatively few parameters, so a higher "
            "LR is well-tolerated. Compensated by stronger weight decay."
        ),
    )
    p.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data reserved for validation (stratified).",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=8,
        help=(
            "Early-stopping patience in epochs without improvement. "
            "Increased from 7 to 8: UniCL loss can plateau slightly before "
            "improving when new same-class pairs appear in batches."
        ),
    )
    p.add_argument(
        "--min-delta",
        type=float,
        default=0.002,
        help=(
            "Minimum val-loss improvement to reset patience. "
            "Slightly reduced from 0.003: UniCL loss values are generally "
            "lower than InfoNCE (more positives in numerator), so meaningful "
            "improvements appear as smaller absolute deltas."
        ),
    )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    p.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help=(
            "LoRA rank r. "
            "Increased from 4 to 8: UniCL's richer training signal can "
            "productively use more adapter capacity without overfitting, "
            "because the label-aware loss provides additional regularisation "
            "beyond what the InfoNCE diagonal alone offers."
        ),
    )
    p.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help=(
            "Dropout inside LoRA adapters. "
            "Slightly reduced from 0.15 to 0.1: UniCL's label supervision "
            "already acts as a regulariser; excessive dropout fights the signal."
        ),
    )
    p.add_argument(
        "--lora-mlp",
        action="store_true",
        default=False,
        help=(
            "Also apply LoRA to MLP layers (fc1, fc2). "
            "Keep False unless the train/val gap is small and the dataset "
            "is large enough to support the additional parameters."
        ),
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.15,
        help=(
            "AdamW weight decay. "
            "Slightly raised from 0.1 to 0.15 to compensate for the higher "
            "learning rate. Keeps adapter weights small and prevents the "
            "logit_scale from diverging."
        ),
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--resume", action="store_true")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Compile the model with torch.compile (PyTorch >= 2.0). "
            "First epoch will be slow while tracing; subsequent epochs are faster."
        ),
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
