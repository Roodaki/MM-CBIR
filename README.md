# Multimodal Content-Based Image Retrieval with VLM-Generated Captions and Fine-Tuned CLIP

> A research pipeline that augments visual image retrieval with LLM-generated semantic captions, fine-tunes CLIP via LoRA, and performs a systematic ablation of multimodal fusion strategies.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)
- [Citation](#citation)

  
## Overview

This repository presents a complete, end-to-end research system for **Content-Based Image Retrieval (CBIR)** that combines vision and language modalities. The core hypothesis is that enriching CLIP's image embeddings with dense, machine-generated semantic captions improves retrieval precision, recall, and MAP compared to image-only or text-only baselines.

The pipeline covers every stage of the research workflow:

1. **Caption generation** — dense semantic tags are produced for each image using a vision-language model (Llama 4 Scout via Groq), guided by a carefully engineered prompt optimised for CLIP token weighting.
2. **Fine-tuning** — `openai/clip-vit-base-patch32` is fine-tuned on the captioned dataset using **LoRA** adapters and a **UniCL-style InfoNCE loss** with mixed-precision training.
3. **Feature extraction** — image and text encoders are run in a single forward pass; 11 fusion variants (concat, average, weighted ×9) are computed and saved together.
4. **Retrieval evaluation** — a FAISS exact cosine-similarity index is built for each variant; Precision@K, Recall@K, F1@K, and MAP@K are computed for K ∈ {10, 20, …, 100}.
5. **Visualisation** — training loss curves, retrieval metric plots, and t-SNE embeddings are generated for qualitative and quantitative analysis.

---

## Repository Structure

```
.
├── generate_caption.py      # Stage 1 — LLM-based image captioning (async, key-rotating)
├── train_clip.py            # Stage 2 — LoRA fine-tuning of CLIP with InfoNCE loss
├── extract_features.py      # Stage 3 — Dual-encoder feature extraction + fusion
├── retrieval.py             # Stage 4 — FAISS retrieval & metric evaluation
├── plot_results.py          # Visualisation — Precision/Recall/F1/MAP@K curves
├── plot_training.py         # Visualisation — Train/Val loss curves
├── visualize_tsne.py        # Visualisation — t-SNE embedding visualisation
├── prompt.md                # CLIP-optimised captioning prompt (v5)
├── config.yaml              # Centralised configuration (paths, models, rate limits)
└── output/
    ├── captions/            # <dataset>_captions.json
    ├── models/              # LoRA checkpoints + training_log.json
    ├── features/            # clip_multimodal_features.npz
    └── results/             # retrieval_results.json + metric PNGs
```

---

## Methodology

### Caption Generation

Images are captioned using **Llama 4 Scout 17B** (via Groq API) with a bespoke prompt designed to produce ultra-dense, machine-readable semantic tags rather than natural language descriptions. The prompt enforces:

- Strict JSON output with `primary` (≤10 tokens, taxonomy-first) and `extended` (≤40 tokens, visual modifiers) fields.
- Front-loaded noun-adjective ordering to maximise CLIP token weighting.
- Zero redundancy between the two fields.

An asynchronous, key-rotating rate limiter handles Groq's per-key RPM quotas across a pool of API keys, enabling high-throughput captioning of large datasets (e.g., GHIM-10K).

### Fine-Tuning

CLIP is fine-tuned using **LoRA** (Low-Rank Adaptation) adapters applied to both the vision and text transformer encoders, keeping the number of trainable parameters small while allowing the model to specialise on the target distribution.

The training objective is a **symmetric InfoNCE (contrastive) loss** — the same formulation used in the original CLIP paper — computed over image–caption pairs within each mini-batch. Key training details:

- Optimiser: AdamW with **OneCycleLR** scheduler
- Mixed precision: `torch.amp.autocast` (CUDA)
- Stratified train/validation split
- Best checkpoint saved by minimum validation loss

### Feature Extraction & Fusion

After fine-tuning, both encoders are run once over the full dataset. The resulting L2-normalised image and text embeddings are combined via 11 fusion strategies saved into a single `.npz` file:

| Key | Strategy | Description |
|-----|----------|-------------|
| `image_features` | Image-only | Baseline — visual embeddings only |
| `text_features` | Text-only | Baseline — caption embeddings only |
| `fused_concat` | Concatenation | `[img ‖ txt]` — dimension doubles to 2D |
| `fused_avg` | Average | Element-wise mean, re-normalised |
| `fused_weighted_01` … `_09` | Weighted sum | img weight ∈ {0.9, …, 0.1}, re-normalised |

### Retrieval Evaluation

Retrieval is performed using **FAISS `IndexFlatIP`** (exact inner product search on L2-normalised vectors, equivalent to cosine similarity). For every query image, the system excludes the query itself and ranks the remaining gallery by similarity.

Four metrics are computed at each K:

- **Precision@K** — fraction of top-K results that share the query's class label
- **Recall@K** — fraction of all relevant images recovered in the top K
- **F1@K** — harmonic mean of P@K and R@K
- **MAP@K** — Mean Average Precision, accounting for rank order of relevant results

Results are aggregated over all queries and exported to JSON for downstream plotting.

---

## Datasets

### Original Image Datasets

| Dataset | Images | Categories | Split | Kaggle |
|---------|--------|------------|-------|--------|
| **GHIM-10K** | 10,000 | 20 | Captioning + fine-tuning | [↗ Download](https://www.kaggle.com/datasets/guohey/ghim10k) |
| **Corel-10K** | 10,000 | 100 | Retrieval evaluation | [↗ Download](https://www.kaggle.com/datasets/amirhosseinroodaki/corel-1k-corel-5k-and-corel-10k-datasets) |

The dataset root is configured via `paths.dataset` in `config.yaml`.

### Multimodal Datasets (MM-*)

As part of this research, enriched versions of both datasets have been released — each image is paired with its LLM-generated semantic caption in the format consumed by this pipeline.

| Dataset | Based on | Kaggle |
|---------|----------|--------|
| **MM-GHIM-10K** | GHIM-10K + Llama 4 Scout captions | [↗ Download](https://www.kaggle.com/datasets/amirhosseinroodaki/mm-ghim-10k) |
| **MM-Corel-10K** | Corel-10K + Llama 4 Scout captions | [↗ Download](https://www.kaggle.com/datasets/amirhosseinroodaki/mm-corel-10k) |

These MM-* datasets allow skipping Stage 1 (caption generation) entirely and proceeding directly to fine-tuning or feature extraction.

### Pre-Extracted Features

To skip Stages 1–3, pre-computed `.npz` feature files are available for direct download. Each file contains all 13 arrays described in the [Feature Extraction & Fusion](#feature-extraction--fusion) section.

| Dataset | Host | Download |
|---------|------|----------|
| **Corel-10K** features | MEGA | [↗ Download](https://mega.nz/file/dx53xSDS#f93DJY0JRPCXHHvW1XJYBBLYxHwxV9DJnb5fWW0xpzg) |
| **GHIM-10K** features | MEGA | [↗ Download](https://mega.nz/file/11plWYSI#ZS97_bZwicOwsapNxKcINIDcSscJjU_ztqpOuDFa0Pc) |

#### `.npz` File Structure

Each feature file contains the following arrays:

```
clip_multimodal_features.npz
│
├── image_features          float32  (N, 512)   L2-normalised image embeddings
├── text_features           float32  (N, 512)   L2-normalised caption embeddings
├── paths                   str      (N,)        Relative image paths (used as IDs)
├── fusion_meta             str      scalar      JSON string — fusion config per variant
│
├── fused_concat            float32  (N, 1024)  Concatenation of image + text
├── fused_avg               float32  (N, 512)   Element-wise average, re-normalised
├── fused_weighted_01       float32  (N, 512)   img×0.9 + txt×0.1, re-normalised
├── fused_weighted_02       float32  (N, 512)   img×0.8 + txt×0.2, re-normalised
├── fused_weighted_03       float32  (N, 512)   img×0.7 + txt×0.3, re-normalised
├── fused_weighted_04       float32  (N, 512)   img×0.6 + txt×0.4, re-normalised
├── fused_weighted_05       float32  (N, 512)   img×0.5 + txt×0.5, re-normalised
├── fused_weighted_06       float32  (N, 512)   img×0.4 + txt×0.6, re-normalised
├── fused_weighted_07       float32  (N, 512)   img×0.3 + txt×0.7, re-normalised
├── fused_weighted_08       float32  (N, 512)   img×0.2 + txt×0.8, re-normalised
└── fused_weighted_09       float32  (N, 512)   img×0.1 + txt×0.9, re-normalised
```

Load and inspect with:

```python
import numpy as np, json

data = np.load("clip_multimodal_features.npz", allow_pickle=True)
print(list(data.keys()))                          # all available arrays
print(data["image_features"].shape)               # (N, 512)
fusion_meta = json.loads(data["fusion_meta"].item())
print(fusion_meta["fused_weighted_03"])
# {'fusion_mode': 'weighted', 'text_weight': 0.3, 'dim': 512}
```

---

## Installation

```bash
# Clone
git clone https://github.com/Roodaki/multimodal-cbir.git
cd auto-captioning-cbir

# Install dependencies
pip install torch torchvision transformers peft faiss-cpu \
            pillow tqdm groq scikit-learn matplotlib numpy pyyaml
```

> For GPU-accelerated FAISS replace `faiss-cpu` with `faiss-gpu`.

---

## Usage

Run each stage in order. All paths and hyperparameters are controlled through `config.yaml`.

### 1 — Generate Captions

```bash
python generate_caption.py
```

Produces `output/captions/<dataset>_captions.json`.

### 2 — Fine-Tune CLIP

```bash
python train_clip.py \
  --json-path output/captions/GHIM-10K_captions.json \
  --img-dir data/GHIM-10K \
  --output-dir output/models/openai_clip-vit-base-patch32
```

### 3 — Extract Features

```bash
python extract_features.py \
  --model-dir output/models/openai_clip-vit-base-patch32 \
  --img-dir data/GHIM-10K \
  --json-path output/captions/GHIM-10K_captions.json \
  --output output/features/clip_multimodal_features
```

### 4 — Evaluate Retrieval

```bash
python retrieval.py \
  --npz-path output/features/clip_multimodal_features.npz \
  --ks 10 20 30 40 50 \
  --output output/results/retrieval_results.json
```

### 5 — Plot Results

```bash
python plot_results.py      # Metric curves (requires results JSON)
python plot_training.py     # Loss curve
python visualize_tsne.py    # t-SNE (--feature-key fused_avg, etc.)
```

---

## Dependencies

| Package | Role |
|---------|------|
| `torch` | Model training and inference |
| `transformers` | CLIP model and processor |
| `peft` | LoRA adapter fine-tuning |
| `faiss-cpu` / `faiss-gpu` | Approximate/exact nearest-neighbour search |
| `groq` | Llama 4 Scout API for captioning |
| `scikit-learn` | t-SNE dimensionality reduction |
| `matplotlib` | Metric and embedding visualisation |
| `pillow`, `tqdm`, `numpy` | Image I/O, progress bars, numerics |

---

## License

This project is released under the terms of the [LICENSE](LICENSE) file in this repository.

---

## Citation

If you use this code or methodology in your research, please cite this repository:

```bibtex
TBA
```

---
