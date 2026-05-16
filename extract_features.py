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


def collect_image_paths(
    img_dir: str, json_path: str | None
) -> tuple[list[str], list[str]]:
    """
    Returns (rel_paths, abs_paths).

    If a JSON caption file is provided (same format used during training),
    paths and class labels are read from it to preserve the original order
    and include only images that were part of the dataset.

    Otherwise, every image found recursively under img_dir is included.
    """
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if json_path is not None:
        with open(json_path, "r") as f:
            raw = json.load(f)
        rel_paths = list(raw["images"].keys())
        print(f"[INFO] Loaded {len(rel_paths)} paths from JSON: '{json_path}'")
    else:
        rel_paths = []
        for root, _, files in os.walk(img_dir):
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() in supported_exts:
                    rel = os.path.relpath(os.path.join(root, fname), img_dir)
                    rel_paths.append(rel)
        print(f"[INFO] Found {len(rel_paths)} images by scanning '{img_dir}'")

    abs_paths = [os.path.join(img_dir, p) for p in rel_paths]
    return rel_paths, abs_paths


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_dir: str, device: str):
    """
    Loads the fine-tuned CLIP model (LoRA adapter on top of the base model).
    Returns only the vision components needed for feature extraction.
    """
    print(f"[INFO] Loading base model + LoRA adapter from '{model_dir}'...")

    # Load the full CLIP model first, then apply the saved LoRA adapter.
    # CLIPProcessor saved alongside the checkpoint handles image preprocessing.
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
    batch_size: int,
    device: str,
) -> np.ndarray:
    """
    Runs the CLIP vision encoder over all images in batches.
    Returns an (N, D) float32 numpy array of L2-normalised embeddings.
    """
    use_amp = device == "cuda"
    all_embeddings = []
    failed_indices = []

    for batch_start in tqdm(
        range(0, len(abs_paths), batch_size),
        desc="Extracting features",
    ):
        batch_paths = abs_paths[batch_start : batch_start + batch_size]
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
                # Use a blank placeholder so the batch stays aligned.
                images.append(Image.new("RGB", (224, 224)))

        pixel_values = processor.image_processor(
            images=images,
            return_tensors="pt",
        )[
            "pixel_values"
        ].to(device)

        with torch.autocast(device_type=device, enabled=use_amp):
            # Call only the vision encoder — the text encoder is never touched.
            image_embeds = model.get_image_features(pixel_values=pixel_values)

        # L2-normalise so cosine similarity == dot product at retrieval time.
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        all_embeddings.append(image_embeds.cpu().float().numpy())

    if failed_indices:
        print(
            f"\n[WARNING] {len(failed_indices)} image(s) failed to load and were "
            f"replaced with blank placeholders. Indices: {failed_indices}"
        )

    return np.concatenate(all_embeddings, axis=0)  # (N, D)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running on: {device}")

    model, processor = load_model(args.model_dir, device)

    rel_paths, abs_paths = collect_image_paths(args.img_dir, args.json_path)

    if len(rel_paths) == 0:
        raise RuntimeError(
            "No images found. Check --img-dir and --json-path arguments."
        )

    features = extract_features(model, processor, abs_paths, args.batch_size, device)

    assert features.shape[0] == len(rel_paths), (
        f"Feature count ({features.shape[0]}) != path count ({len(rel_paths)}). "
        "This is a bug — please report it."
    )

    print(f"\n[INFO] Extracted feature matrix shape: {features.shape}")  # (N, D)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(
        args.output,
        features=features,  # float32 (N, D) — L2-normalised embeddings
        paths=np.array(rel_paths),  # (N,) relative paths, aligned with features
    )
    print(f"[INFO] Saved features to '{args.output}.npz'")
    print(f"       Keys: 'features' {features.shape}, " f"'paths' ({len(rel_paths)},)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract CLIP image encoder features and save to a .npz file."
    )
    p.add_argument(
        "--model-dir",
        default="./output/models/openai_clip-vit-base-patch32",
        help="Path to the directory containing the best saved LoRA checkpoint "
        "(the one written to --output-dir during training, NOT the 'latest' subfolder).",
    )
    p.add_argument(
        "--img-dir",
        default=os.path.join("data", "Corel-10K"),
        help="Root directory of the image dataset.",
    )
    p.add_argument(
        "--json-path",
        default=None,
        help="(Optional) Path to the caption JSON used during training. "
        "When provided, only those images are processed and their original "
        "order is preserved. When omitted, all images under --img-dir are used.",
    )
    p.add_argument(
        "--output",
        default="./output/features/clip_image_features",
        help="Output path (without extension). A .npz file will be created here.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Images per forward pass. Increase on a large GPU, decrease if OOM.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
