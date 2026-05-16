import os
import json
import argparse
import collections
import math
import random
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
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

        self.items = []
        self.class_labels = []
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
            self.class_labels.append(meta["class_label"])

        if truncated_count:
            print(
                f"[WARNING] {truncated_count}/{len(self.items)} captions exceed "
                f"{self.MAX_TOKENS} tokens and will be truncated."
            )
        print(f"[INFO] Pre-tokenisation complete. {len(self.items)} items cached.")

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
        }


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


def symmetric_infonce_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    scale = torch.clamp(logit_scale.exp(), max=100.0)
    logits = scale * (image_embeds @ text_embeds.T)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = nn.functional.cross_entropy(logits, labels)
    loss_t2i = nn.functional.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2.0


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


def resolve_logit_scale(model: nn.Module) -> nn.Parameter:
    named_params = dict(model.named_parameters())
    candidate_keys = [
        "base_model.model.logit_scale",
        "base_model.model.model.logit_scale",
        "logit_scale",
    ]
    for key in candidate_keys:
        if key in named_params:
            param = named_params[key]
            param.requires_grad = True
            return param

    raise RuntimeError(
        "Could not locate logit_scale in model parameters. "
        f"Searched keys: {candidate_keys}. "
        f"Available keys containing 'logit': "
        f"{[k for k in named_params if 'logit' in k]}"
    )


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Validate grad_accum early so misconfiguration fails before any work is done.
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

    if args.compile:
        if hasattr(torch, "compile"):
            print("[INFO] Compiling model with torch.compile...")
            model = torch.compile(model)
        else:
            print(
                "[WARNING] --compile requested but torch.compile is not available "
                "(requires PyTorch >= 2.0). Skipping."
            )

    logit_scale = resolve_logit_scale(model)

    use_amp = device == "cuda"
    scaler = GradScaler(enabled=use_amp)

    full_dataset = MultimodalCBIRDataset(args.json_path, args.img_dir, processor)
    train_ds, val_ds = stratified_split(full_dataset, args.val_split, seed=42)

    if len(val_ds) < 2:
        raise ValueError(
            f"Validation set has only {len(val_ds)} sample(s). "
            "InfoNCE loss requires at least 2 samples per batch. "
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
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        drop_last=True,
    )

    # Higher weight_decay (0.1 vs original 0.01) is a strong and cheap
    # regulariser for contrastive models — it penalises large adapter weights
    # directly and consistently improves generalisation on small datasets.
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Optimizer steps happen every grad_accum micro-batches, so the scheduler
    # must count optimizer steps, not micro-batch steps.
    steps_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / args.grad_accum)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=optimizer_steps_per_epoch,
        epochs=args.epochs,
        pct_start=0.05,  # shorter warm-up (was 0.1): ~1 epoch at 30 total
        anneal_strategy="cos",
        final_div_factor=1000,  # stronger end-of-run decay (was default 1e4/10000)
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

        state = torch.load(trainer_state_file, map_location=device)
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
                f"[INFO] Loaded {len(training_log)} existing log entries from '{training_log_file}'."
            )

        print(
            f"[INFO] Resumed successfully. Starting from Epoch {start_epoch + 1}. "
            f"Current best val loss: {best_val_loss:.4f}\n"
        )
    elif args.resume:
        print(
            "\n[WARNING] --resume flag set but no valid checkpoint found. Starting from scratch.\n"
        )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()  # zero once before the epoch starts

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]")
        ):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast(enabled=use_amp):
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
                loss = symmetric_infonce_loss(img_emb, text_emb, logit_scale)

            # Scale loss by accum steps so gradients are averaged, not summed,
            # across micro-batches — keeps gradient magnitude independent of
            # the accumulation factor.
            scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()

            total_train_loss += loss.item()

            is_last_micro_batch = (step + 1) % args.grad_accum == 0
            is_last_batch = (step + 1) == len(train_loader)

            if is_last_micro_batch or is_last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        avg_train = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]  "
            ):
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with autocast(enabled=use_amp):
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
                    loss = symmetric_infonce_loss(img_emb, text_emb, logit_scale)

                total_val_loss += loss.item()

        avg_val = total_val_loss / len(val_loader)
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune CLIP ViT-B/32 with LoRA for CBIR"
    )
    p.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    p.add_argument(
        "--json-path",
        default=os.path.join("output/captions", "Corel-10K_captions.json"),
    )
    p.add_argument("--img-dir", default=os.path.join("data", "Corel-10K"))
    p.add_argument(
        "--output-dir",
        default=f"./output/models/{p.parse_args().model_id.replace('/', '_')}",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help=(
            "Number of micro-batches to accumulate before an optimizer step. "
            "Effective batch size = batch-size * grad-accum. "
            "Default 4 gives an effective batch of 512 at batch-size 128, "
            "which significantly improves InfoNCE contrastive signal without "
            "extra VRAM cost."
        ),
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Peak learning rate. Lowered from 5e-5 to 3e-5 to reduce overfitting.",
    )
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early-stopping patience in epochs without improvement.",
    )
    p.add_argument(
        "--min-delta",
        type=float,
        default=0.003,
        help=(
            "Minimum val-loss improvement to reset patience. "
            "Raised from 0.001 to 0.005 to avoid counting noise as progress."
        ),
    )
    p.add_argument(
        "--lora-rank",
        type=int,
        default=4,
        help=(
            "LoRA rank r. Lower = fewer trainable parameters = less overfitting. "
            "Default 8 (was 16). Try 4 if the train/val gap is still large."
        ),
    )
    p.add_argument(
        "--lora-dropout",
        type=float,
        default=0.15,
        help=(
            "Dropout applied inside LoRA adapters. "
            "Raised from 0.05 to 0.1 for better regularisation."
        ),
    )
    p.add_argument(
        "--lora-mlp",
        action="store_true",
        default=False,
        help=(
            "Also apply LoRA to the MLP layers (fc1, fc2) in addition to "
            "attention and projection layers. Increases adapter capacity — "
            "only enable this if the train/val gap is small and you have "
            "enough data to support more parameters."
        ),
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help=(
            "AdamW weight decay. Raised from 0.01 to 0.1 — a strong and "
            "cheap regulariser for contrastive models on small datasets."
        ),
    )
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
