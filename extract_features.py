import os
import json
import argparse
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

MAX_TOKENS: int = 77  # Must match the value used during training


def collect_image_paths_and_captions(
    img_dir: str, json_path: str
) -> tuple[list[str], list[str], list[str]]:
    """
    Returns (rel_paths, abs_paths, captions).

    Paths and captions are read from the caption JSON (same format used during
    training) to preserve the original order and include only images that were
    part of the dataset.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    models = raw["metadata"]["models_evaluated"]
    if len(models) != 1:
        raise ValueError(
            f"Expected exactly 1 model in 'models_evaluated', got: {models}. "
            "Edit the JSON metadata to contain only the model you want to use."
        )
    model_key = models[0]

    rel_paths, captions = [], []
    for rel_path, meta in raw["images"].items():
        caps = meta["captions"][model_key]
        caption = f"{caps['primary']}, {caps['extended']}"
        rel_paths.append(rel_path)
        captions.append(caption)

    print(f"[INFO] Loaded {len(rel_paths)} paths + captions from JSON: '{json_path}'")

    abs_paths = [os.path.join(img_dir, p) for p in rel_paths]
    return rel_paths, abs_paths, captions


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_dir: str, device: str):
    """
    Loads the fine-tuned CLIP model (LoRA adapter on top of the base model).
    Returns the full model so both encoders are accessible.
    """
    print(f"[INFO] Loading base model + LoRA adapter from '{model_dir}'...")

    base_model = CLIPModel.from_pretrained(model_dir)
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    model.to(device)

    processor = CLIPProcessor.from_pretrained(model_dir)

    print("[INFO] Model loaded successfully.")
    return model, processor


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_features(
    model,
    processor: CLIPProcessor,
    abs_paths: list[str],
    captions: list[str],
    batch_size: int,
    device: str,
    fusion_mode: str,
    text_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs both CLIP encoders over all images and their captions in batches.

    Returns
    -------
    image_features  : (N, D)   L2-normalised image embeddings
    text_features   : (N, D)   L2-normalised text embeddings
    fused_features  : (N, D*)  combined representation, ready for KNN
                                shape depends on --fusion-mode:
                                  'concat'  → (N, 2D)
                                  'avg'     → (N, D)
                                  'weighted'→ (N, D)
    """
    use_amp = device == "cuda"
    all_img_embeds, all_txt_embeds = [], []
    failed_indices = []

    for batch_start in tqdm(
        range(0, len(abs_paths), batch_size),
        desc="Extracting features",
    ):
        batch_paths = abs_paths[batch_start : batch_start + batch_size]
        batch_caps = captions[batch_start : batch_start + batch_size]
        images = []

        # ── image loading ────────────────────────────────────────────────────
        for i, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as exc:
                global_idx = batch_start + i
                print(
                    f"\n[WARNING] Could not load image [{global_idx}] '{path}': {exc}"
                )
                failed_indices.append(global_idx)
                images.append(Image.new("RGB", (224, 224)))

        pixel_values = processor.image_processor(
            images=images,
            return_tensors="pt",
        )[
            "pixel_values"
        ].to(device)

        # ── text tokenisation ─────────────────────────────────────────────────
        text_inputs = processor.tokenizer(
            batch_caps,
            padding="max_length",
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        # ── forward passes ────────────────────────────────────────────────────
        with torch.autocast(device_type=device, enabled=use_amp):
            image_embeds = model.get_image_features(pixel_values=pixel_values)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            text_embeds = model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        all_img_embeds.append(image_embeds.cpu().float().numpy())
        all_txt_embeds.append(text_embeds.cpu().float().numpy())

    if failed_indices:
        print(
            f"\n[WARNING] {len(failed_indices)} image(s) failed to load and were "
            f"replaced with blank placeholders. Indices: {failed_indices}"
        )

    image_features = np.concatenate(all_img_embeds, axis=0)  # (N, D)
    text_features = np.concatenate(all_txt_embeds, axis=0)  # (N, D)

    # ── fusion ────────────────────────────────────────────────────────────────
    if fusion_mode == "concat":
        # (N, 2D) — keeps both modalities fully independent; KNN operates in
        # the joint space. No re-normalisation needed because each half is
        # already L2-normalised and equally scaled.
        fused_features = np.concatenate([image_features, text_features], axis=1)

    elif fusion_mode == "avg":
        # (N, D) — element-wise mean then re-normalise. Equivalent to the
        # midpoint on the unit hypersphere; simple and effective when both
        # modalities are equally reliable.
        fused_features = image_features + text_features
        norms = np.linalg.norm(fused_features, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # guard against zero vectors
        fused_features = fused_features / norms

    elif fusion_mode == "weighted":
        # (N, D) — weighted sum then re-normalise. Use --text-weight to tune
        # how much the text branch contributes (0 = image only, 1 = text only,
        # 0.5 = equal). Useful when one modality is noisier than the other.
        img_weight = 1.0 - text_weight
        fused_features = img_weight * image_features + text_weight * text_features
        norms = np.linalg.norm(fused_features, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        fused_features = fused_features / norms

    else:
        raise ValueError(
            f"Unknown --fusion-mode '{fusion_mode}'. "
            "Choose from: concat | avg | weighted"
        )

    return image_features, text_features, fused_features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running on: {device}")
    print(
        f"[INFO] Fusion mode: '{args.fusion_mode}'"
        + (
            f"  (text weight: {args.text_weight})"
            if args.fusion_mode == "weighted"
            else ""
        )
    )

    model, processor = load_model(args.model_dir, device)

    rel_paths, abs_paths, captions = collect_image_paths_and_captions(
        args.img_dir, args.json_path
    )

    if len(rel_paths) == 0:
        raise RuntimeError(
            "No images found. Check --img-dir and --json-path arguments."
        )

    image_features, text_features, fused_features = extract_features(
        model,
        processor,
        abs_paths,
        captions,
        args.batch_size,
        device,
        args.fusion_mode,
        args.text_weight,
    )

    n = len(rel_paths)
    assert image_features.shape[0] == n, (
        f"image_features count ({image_features.shape[0]}) != path count ({n}). "
        "This is a bug — please report it."
    )
    assert text_features.shape[0] == n, (
        f"text_features count ({text_features.shape[0]}) != path count ({n}). "
        "This is a bug — please report it."
    )
    assert fused_features.shape[0] == n, (
        f"fused_features count ({fused_features.shape[0]}) != path count ({n}). "
        "This is a bug — please report it."
    )

    print(f"\n[INFO] image_features shape : {image_features.shape}")
    print(f"[INFO] text_features shape  : {text_features.shape}")
    print(f"[INFO] fused_features shape : {fused_features.shape}  ← use this for KNN")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(
        args.output,
        # ── individual modality embeddings ──────────────────────────────────
        image_features=image_features,  # float32 (N, D)  L2-normalised
        text_features=text_features,  # float32 (N, D)  L2-normalised
        # ── fused embedding — feed this array to your KNN index ─────────────
        fused_features=fused_features,  # float32 (N, D or 2D) depends on fusion_mode
        # ── metadata ────────────────────────────────────────────────────────
        paths=np.array(rel_paths),  # (N,) relative paths, aligned with features
        fusion_mode=np.array(args.fusion_mode),
    )

    print(f"\n[INFO] Saved features to '{args.output}.npz'")
    print(f"       Keys:")
    print(f"         'image_features'  {image_features.shape}")
    print(f"         'text_features'   {text_features.shape}")
    print(f"         'fused_features'  {fused_features.shape}  ← KNN index input")
    print(f"         'paths'           ({n},)")
    print(f"         'fusion_mode'     '{args.fusion_mode}'")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Extract CLIP image + text encoder features and save to a .npz file. "
            "The fused_features array is ready to be loaded into a KNN index."
        )
    )
    p.add_argument(
        "--model-dir",
        default="./output/models/openai_clip-vit-base-patch32",
        help="Path to the directory containing the best saved LoRA checkpoint.",
    )
    p.add_argument(
        "--img-dir",
        default=os.path.join("data", "Corel-10K"),
        help="Root directory of the image dataset.",
    )
    p.add_argument(
        "--json-path",
        default="output\\captions\\Corel-10K_captions.json",
        help=(
            "Path to the caption JSON used during training. "
            "Captions are encoded by the text encoder and fused with the image "
            "embeddings to produce the final KNN feature vectors."
        ),
    )
    p.add_argument(
        "--output",
        default="./output/features/clip_multimodal_features",
        help="Output path (without extension). A .npz file will be created here.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Samples per forward pass. Increase on a large GPU, decrease if OOM.",
    )
    p.add_argument(
        "--fusion-mode",
        choices=["concat", "avg", "weighted"],
        default="avg",
        help=(
            "How to combine image and text embeddings into the KNN feature vector:\n"
            "  concat   — concatenate → (N, 2D). Keeps both modalities fully "
            "independent. Best default choice.\n"
            "  avg      — element-wise mean then L2-normalise → (N, D). "
            "Equal weight, smaller index.\n"
            "  weighted — weighted sum then L2-normalise → (N, D). "
            "Control the text/image balance with --text-weight."
        ),
    )
    p.add_argument(
        "--text-weight",
        type=float,
        default=0.5,
        help=(
            "Weight assigned to the text embedding when --fusion-mode=weighted. "
            "Image weight = 1 - text_weight. Range: [0.0, 1.0]. Default: 0.5."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
