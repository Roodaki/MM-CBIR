import os
import json
import argparse
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel, PeftConfig
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

MAX_TOKENS: int = 77  # Must match the value used during training

# All fusion strategies that will be computed and saved in a single run.
# Each entry: (key_name_in_npz, fusion_mode, text_weight_or_None)
WEIGHTED_TEXT_WEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

FUSION_VARIANTS: list[tuple[str, str, float | None]] = [
    ("fused_concat", "concat", None),
    ("fused_avg", "avg", None),
] + [
    # round() before int() guards against IEEE-754 imprecision
    # (e.g. 0.1 * 10 == 0.9999... in floating point → int() would give 0)
    (f"fused_weighted_{round(w * 10):02d}", "weighted", w)
    for w in WEIGHTED_TEXT_WEIGHTS
]
# e.g. keys saved: fused_concat, fused_avg,
#                  fused_weighted_01 … fused_weighted_09


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

    config = PeftConfig.from_pretrained(model_dir)
    base_model_name = config.base_model_name_or_path

    base_model = CLIPModel.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    model.to(device)

    processor = CLIPProcessor.from_pretrained(base_model_name)

    print("[INFO] Model loaded successfully.")
    return model, processor


# ---------------------------------------------------------------------------
# Feature extraction  (single forward pass, returns raw embeddings)
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_raw_embeddings(
    model,
    processor: CLIPProcessor,
    abs_paths: list[str],
    captions: list[str],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs both CLIP encoders over all images + captions in batches.

    Returns
    -------
    image_features : (N, D)  L2-normalised image embeddings  (float32)
    text_features  : (N, D)  L2-normalised text embeddings   (float32)
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

        text_inputs = processor.tokenizer(
            batch_caps,
            padding="max_length",
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        with torch.amp.autocast(device_type=device, enabled=use_amp):
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
    return image_features, text_features


# ---------------------------------------------------------------------------
# Fusion helpers
# ---------------------------------------------------------------------------


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def fuse(
    image_features: np.ndarray,
    text_features: np.ndarray,
    fusion_mode: str,
    text_weight: float | None = None,
) -> np.ndarray:
    """
    Combines L2-normalised image and text embeddings into a single vector.

    concat   → (N, 2D)  — concatenation; each half already L2-normalised
    avg      → (N, D)   — element-wise mean, then re-normalise
    weighted → (N, D)   — weighted sum, then re-normalise
                          text_weight ∈ [0, 1]; image_weight = 1 - text_weight
    """
    if fusion_mode == "concat":
        return np.concatenate([image_features, text_features], axis=1)

    elif fusion_mode == "avg":
        return _l2_normalise(image_features + text_features)

    elif fusion_mode == "weighted":
        if text_weight is None:
            raise ValueError("text_weight must be provided for fusion_mode='weighted'")
        img_weight = 1.0 - text_weight
        return _l2_normalise(img_weight * image_features + text_weight * text_features)

    else:
        raise ValueError(
            f"Unknown fusion_mode '{fusion_mode}'. "
            "Choose from: concat | avg | weighted"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running on: {device}")

    model, processor = load_model(args.model_dir, device)

    rel_paths, abs_paths, captions = collect_image_paths_and_captions(
        args.img_dir, args.json_path
    )

    if not rel_paths:
        raise RuntimeError(
            "No images found. Check --img-dir and --json-path arguments."
        )

    # ── single forward pass for both encoders ────────────────────────────────
    image_features, text_features = extract_raw_embeddings(
        model, processor, abs_paths, captions, args.batch_size, device
    )

    n = len(rel_paths)
    if image_features.shape[0] != n:
        raise RuntimeError(
            f"image_features row count ({image_features.shape[0]}) != path count ({n}). "
            "This is a bug — please report it."
        )
    if text_features.shape[0] != n:
        raise RuntimeError(
            f"text_features row count ({text_features.shape[0]}) != path count ({n}). "
            "This is a bug — please report it."
        )

    print(f"\n[INFO] image_features shape : {image_features.shape}")
    print(f"[INFO] text_features shape  : {text_features.shape}")

    # ── compute every fusion variant ─────────────────────────────────────────
    fused_arrays: dict[str, np.ndarray] = {}
    print("\n[INFO] Computing all fusion variants:")
    for npz_key, fusion_mode, text_weight in FUSION_VARIANTS:
        arr = fuse(image_features, text_features, fusion_mode, text_weight)
        fused_arrays[npz_key] = arr
        label = (
            f"text_weight={text_weight}" if fusion_mode == "weighted" else fusion_mode
        )
        print(f"       {npz_key:<30}  shape={arr.shape}  [{label}]")

    # ── build fusion metadata (saved alongside arrays for retrieval.py) ──────
    fusion_meta = {}
    for npz_key, fusion_mode, text_weight in FUSION_VARIANTS:
        fusion_meta[npz_key] = {
            "fusion_mode": fusion_mode,
            "text_weight": text_weight,  # None for concat / avg
            "dim": int(fused_arrays[npz_key].shape[1]),
        }

    # ── save ─────────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_dict = {
        # raw modality embeddings
        "image_features": image_features,  # float32 (N, D)
        "text_features": text_features,  # float32 (N, D)
        # metadata
        "paths": np.array(rel_paths),
        "fusion_meta": np.array(json.dumps(fusion_meta)),  # JSON string scalar
    }
    # add every fused variant
    save_dict.update(fused_arrays)

    np.savez(args.output, **save_dict)

    print(f"\n[INFO] Saved all features to '{args.output}.npz'")
    print(
        f"       Fixed keys   : image_features {image_features.shape}, "
        f"text_features {text_features.shape}, paths ({n},), fusion_meta"
    )
    print(f"       Fused keys   ({len(fused_arrays)}):")
    for npz_key, arr in fused_arrays.items():
        print(f"         '{npz_key}'  {arr.shape}")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Extract CLIP image + text encoder features and save ALL fusion "
            "variants (concat, avg, weighted×9) to a single .npz file. "
            "The fused_* arrays are ready to be loaded into a KNN index."
        )
    )
    p.add_argument(
        "--model-dir",
        default="./output/models/openai_clip-vit-base-patch32",
        help="Path to the directory containing the best saved LoRA checkpoint.",
    )
    p.add_argument(
        "--img-dir",
        default=os.path.join("data", "GHIM-10K"),
        help="Root directory of the image dataset.",
    )
    p.add_argument(
        "--json-path",
        default="output\\captions\\GHIM-10K_captions.json",
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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
