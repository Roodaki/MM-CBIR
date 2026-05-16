import os
import argparse
import json
import numpy as np
import faiss
from collections import defaultdict
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIMILARITIES = ["cosine", "l2", "dot"]

# Keys that must exist in the .npz produced by extract_features.py
FEATURE_VARIANTS = {
    "image_only": "image_features",
    "text_only": "text_features",
    "fused": "fused_features",
}

# ---------------------------------------------------------------------------
# FAISS index factory
# ---------------------------------------------------------------------------


def build_index(features: np.ndarray, similarity: str, use_gpu: bool):
    """
    Builds an exact FAISS index for the requested similarity metric.

    cosine  — L2-normalises features, then uses IndexFlatIP (dot == cosine)
    l2      — IndexFlatL2, exact Euclidean distance
    dot     — IndexFlatIP on raw (unnormalised) features

    Returns (index, feats) where feats is the (possibly normalised) float32
    array that was added to the index.
    """
    dim = features.shape[1]

    if similarity == "cosine":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        feats = features / np.where(norms == 0, 1.0, norms)
        index = faiss.IndexFlatIP(dim)
    elif similarity == "l2":
        feats = features.copy()
        index = faiss.IndexFlatL2(dim)
    elif similarity == "dot":
        feats = features.copy()
        index = faiss.IndexFlatIP(dim)
    else:
        raise ValueError(
            f"Unknown similarity '{similarity}'. Choose from {SIMILARITIES}."
        )

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as e:
            print(f"[WARNING] GPU FAISS failed ({e}). Falling back to CPU.")

    index.add(feats.astype(np.float32))
    return index, feats.astype(np.float32)


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------


def extract_labels(paths: np.ndarray) -> list[str]:
    return [p.split(os.sep)[0] if os.sep in p else p.split("/")[0] for p in paths]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def precision_at_k(relevant: set, retrieved: list, k: int) -> float:
    return sum(1 for idx in retrieved[:k] if idx in relevant) / k


def recall_at_k(relevant: set, retrieved: list, k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for idx in retrieved[:k] if idx in relevant) / len(relevant)


def f1_at_k(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def average_precision_at_k(relevant: set, retrieved: list, k: int) -> float:
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for rank, idx in enumerate(retrieved[:k], start=1):
        if idx in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    index: faiss.Index,
    features: np.ndarray,
    labels: list[str],
    ks: list[int],
) -> dict:
    n = len(labels)
    max_k = max(ks)
    k_search = max_k + 1  # +1 to exclude the query itself

    label_to_indices: dict[str, set[int]] = defaultdict(set)
    for i, lbl in enumerate(labels):
        label_to_indices[lbl].add(i)

    sum_p = {k: 0.0 for k in ks}
    sum_r = {k: 0.0 for k in ks}
    sum_f1 = {k: 0.0 for k in ks}
    sum_ap = {k: 0.0 for k in ks}

    batch_size = 512
    for batch_start in range(0, n, batch_size):
        batch_feats = features[batch_start : batch_start + batch_size]
        _, indices = index.search(batch_feats, k_search)

        for local_i, neighbours in enumerate(indices):
            query_idx = batch_start + local_i
            relevant = label_to_indices[labels[query_idx]] - {query_idx}
            retrieved = [idx for idx in neighbours.tolist() if idx != query_idx]

            for k in ks:
                p = precision_at_k(relevant, retrieved, k)
                r = recall_at_k(relevant, retrieved, k)
                sum_p[k] += p
                sum_r[k] += r
                sum_f1[k] += f1_at_k(p, r)
                sum_ap[k] += average_precision_at_k(relevant, retrieved, k)

    return {
        k: {
            "precision": sum_p[k] / n,
            "recall": sum_r[k] / n,
            "f1": sum_f1[k] / n,
            "map": sum_ap[k] / n,
        }
        for k in ks
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_variant_results(
    variant: str, similarity: str, results: dict, ks: list[int]
) -> None:
    title = f"  [{variant.upper()}]  Similarity: {similarity.upper()}  "
    border = "=" * max(len(title), 58)
    header = f"{'K':>5}  {'P@K':>8}  {'R@K':>8}  {'F1@K':>8}  {'MAP@K':>8}"

    print(f"\n{border}")
    print(title)
    print(border)
    print(header)
    print("-" * len(header))
    for k in ks:
        r = results[k]
        print(
            f"{k:>5}  "
            f"{r['precision']:>8.4f}  "
            f"{r['recall']:>8.4f}  "
            f"{r['f1']:>8.4f}  "
            f"{r['map']:>8.4f}"
        )
    print(border)


def print_summary(
    all_results: dict, ks: list[int], variants: list[str], similarities: list[str]
) -> None:
    """
    Condensed comparison table at the median K showing every
    variant × similarity combination side-by-side.
    """
    rep_k = ks[len(ks) // 2]
    col_w = 10

    # Build column headers: one per (variant, similarity) pair
    combos = [(v, s) for v in variants for s in similarities]
    header_top = f"{'':>5}  " + "  ".join(f"{v+'/'+s:>{col_w*4+6}}" for v, s in combos)
    header_mid = f"{'K':>5}  " + "  ".join(
        f"{'P':>{col_w}} {'R':>{col_w}} {'F1':>{col_w}} {'MAP':>{col_w}}"
        for _ in combos
    )
    sep = "=" * len(header_mid)

    print(f"\n{sep}")
    print(f"  SUMMARY — all variants & similarities @ K={rep_k}")
    print(sep)
    print(header_top)
    print(header_mid)
    print("-" * len(header_mid))

    row = f"{rep_k:>5}  "
    for variant, sim in combos:
        r = all_results[variant][sim][rep_k]
        row += (
            f"{r['precision']:>{col_w}.4f} "
            f"{r['recall']:>{col_w}.4f} "
            f"{r['f1']:>{col_w}.4f} "
            f"{r['map']:>{col_w}.4f}  "
        )
    print(row)
    print(sep)

    # Also print per-metric best at rep_k
    print(f"\n  Best per metric @ K={rep_k}:")
    for metric in ("precision", "recall", "f1", "map"):
        best_val = -1.0
        best_name = ""
        for variant, sim in combos:
            val = all_results[variant][sim][rep_k][metric]
            if val > best_val:
                best_val = val
                best_name = f"{variant}/{sim}"
        print(f"    {metric.upper():>10}: {best_name:30s} {best_val:.4f}")
    print()


def save_results(all_results: dict, output_path: str) -> None:
    # all_results[variant][similarity][k] → {precision, recall, f1, map}
    serialisable = {
        variant: {
            sim: {str(k): metrics for k, metrics in res.items()}
            for sim, res in sim_results.items()
        }
        for variant, sim_results in all_results.items()
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"[INFO] Results saved to '{output_path}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    print(f"[INFO] Loading features from '{args.npz_path}'...")
    data = np.load(args.npz_path)

    # Validate that the npz was produced by the multimodal extract_features.py
    missing = [key for key in FEATURE_VARIANTS.values() if key not in data]
    if missing:
        raise KeyError(
            f"The .npz file is missing keys: {missing}. "
            "Make sure you generated it with the multimodal extract_features.py "
            "(not the image-only version). Expected keys: "
            f"{list(FEATURE_VARIANTS.values())}."
        )

    paths = data["paths"]
    labels = extract_labels(paths)
    n = len(labels)
    n_cls = len(set(labels))

    # Load all three feature arrays
    feature_arrays: dict[str, np.ndarray] = {
        variant: data[npz_key].astype(np.float32)
        for variant, npz_key in FEATURE_VARIANTS.items()
    }

    for variant, feats in feature_arrays.items():
        print(
            f"[INFO] {variant:>12}: {feats.shape[0]} embeddings | "
            f"dim {feats.shape[1]} | {n_cls} classes"
        )

    similarities = args.similarities if args.similarities else SIMILARITIES
    variants = list(FEATURE_VARIANTS.keys())  # image_only, text_only, fused

    # all_results[variant][similarity] → {k: {precision, recall, f1, map}}
    all_results: dict[str, dict[str, dict]] = {v: {} for v in variants}

    for variant in variants:
        feats = feature_arrays[variant]
        for sim in similarities:
            print(
                f"\n[INFO] [{variant.upper()}] Building index — similarity: {sim.upper()} ..."
            )
            index, indexed_feats = build_index(feats, sim, use_gpu=args.gpu)

            print(f"[INFO] [{variant.upper()}] Evaluating ...")
            results = evaluate(index, indexed_feats, labels, ks=args.ks)

            print_variant_results(variant, sim, results, ks=args.ks)
            all_results[variant][sim] = results

    if len(similarities) > 1 or len(variants) > 1:
        print_summary(
            all_results, ks=args.ks, variants=variants, similarities=similarities
        )

    if args.output:
        save_results(all_results, args.output)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "CBIR evaluation over image-only / text-only / fused features, "
            "across cosine / L2 / dot-product similarity metrics and all K values."
        )
    )
    p.add_argument(
        "--npz-path",
        default="./output/features/clip_multimodal_features.npz",
        help=(
            "Path to the .npz file produced by extract_features.py. "
            "Must contain 'image_features', 'text_features', 'fused_features', and 'paths'."
        ),
    )
    p.add_argument(
        "--similarities",
        nargs="+",
        choices=SIMILARITIES,
        default=None,
        help=f"Subset of similarities to evaluate. Default: all {SIMILARITIES}",
    )
    p.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=list(range(10, 101, 10)),
        help="Values of K to evaluate at. Default: 10 20 30 ... 100",
    )
    p.add_argument(
        "--output",
        default="./output/results/retrieval_results.json",
        help="Path to save the JSON results file.",
    )
    p.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU-accelerated FAISS if available.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
