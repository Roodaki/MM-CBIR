import os
import argparse
import json
import numpy as np
import faiss
from collections import defaultdict
from tqdm import tqdm

# ---------------------------------------------------------------------------
# FAISS index factory
# ---------------------------------------------------------------------------

SIMILARITIES = ["cosine", "l2", "dot"]


def build_index(features: np.ndarray, similarity: str, use_gpu: bool) -> faiss.Index:
    """
    Builds an exact FAISS index for the requested similarity metric.

    cosine  — L2-normalises features, then uses IndexFlatIP (dot == cosine)
    l2      — IndexFlatL2, exact Euclidean distance
    dot     — IndexFlatIP on raw (unnormalised) features
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
        feats = features.copy()  # raw, unnormalised
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
    k_search = max_k + 1  # +1 to account for the query itself

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


def print_results(similarity: str, results: dict, ks: list[int]) -> None:
    title = f"  Similarity: {similarity.upper()}  "
    border = "=" * max(len(title), 52)
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


def print_summary(all_results: dict, ks: list[int]) -> None:
    """Prints a condensed comparison table at a single representative K."""
    rep_k = ks[len(ks) // 2]  # median K as the representative
    print(f"\n{'=' * 62}")
    print(f"  SUMMARY — best similarity per metric @ K={rep_k}")
    print(f"{'=' * 62}")
    print(f"{'Similarity':>12}  {'P@K':>8}  {'R@K':>8}  {'F1@K':>8}  {'MAP@K':>8}")
    print(f"{'-' * 62}")
    for sim, results in all_results.items():
        r = results[rep_k]
        print(
            f"{sim:>12}  "
            f"{r['precision']:>8.4f}  "
            f"{r['recall']:>8.4f}  "
            f"{r['f1']:>8.4f}  "
            f"{r['map']:>8.4f}"
        )
    print(f"{'=' * 62}\n")


def save_results(all_results: dict, output_path: str) -> None:
    serialisable = {
        sim: {str(k): v for k, v in res.items()} for sim, res in all_results.items()
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
    features = data["features"].astype(np.float32)
    paths = data["paths"]
    labels = extract_labels(paths)

    print(
        f"[INFO] {features.shape[0]} embeddings | dim {features.shape[1]} | {len(set(labels))} classes"
    )

    similarities = args.similarities if args.similarities else SIMILARITIES
    all_results = {}

    for sim in similarities:
        print(f"\n[INFO] Building index for similarity: {sim.upper()} ...")
        index, feats = build_index(features, sim, use_gpu=args.gpu)

        print(f"[INFO] Evaluating ...")
        results = evaluate(index, feats, labels, ks=args.ks)

        print_results(sim, results, ks=args.ks)
        all_results[sim] = results

    if len(similarities) > 1:
        print_summary(all_results, ks=args.ks)

    if args.output:
        save_results(all_results, args.output)


def parse_args():
    p = argparse.ArgumentParser(
        description="CBIR ablation study over cosine / L2 / dot-product similarity."
    )
    p.add_argument(
        "--npz-path",
        default="./output/features/clip_image_features.npz",
    )
    p.add_argument(
        "--similarities",
        nargs="+",
        choices=SIMILARITIES,
        default=None,
        help=f"Subset of similarities to run. Default: all {SIMILARITIES}",
    )
    p.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=list(range(10, 101, 10)),
        help="Values of K. Default: 10 20 30 ... 100",
    )
    p.add_argument(
        "--output",
        default="./output/results/ablation_similarity.json",
    )
    p.add_argument("--gpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
