import os
import argparse
import json
import numpy as np
import faiss
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Similarity used for every index — cosine is the natural choice for
# L2-normalised CLIP embeddings.  Concat vectors are a special case handled
# in build_index (each half is already normalised so we use IP directly).
SIMILARITY = "cosine"

# ---------------------------------------------------------------------------
# FAISS index factory
# ---------------------------------------------------------------------------


def build_index(features: np.ndarray, is_concat: bool, use_gpu: bool):
    """
    Builds an exact cosine-similarity FAISS index.

    For standard (non-concat) vectors the features are L2-normalised before
    being added, so Inner Product == cosine similarity.

    For concat vectors the two halves are already independently L2-normalised;
    re-normalising the combined 2D vector would distort magnitudes, so we add
    them as-is.  The resulting IP score equals the sum of the two per-modality
    cosine similarities, which is still a valid ranking criterion.

    Returns (index, indexed_feats).
    """
    dim = features.shape[1]

    if is_concat:
        feats = features.copy().astype(np.float32)
    else:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        feats = (features / np.where(norms == 0, 1.0, norms)).astype(np.float32)

    index = faiss.IndexFlatIP(dim)

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as e:
            print(f"[WARNING] GPU FAISS failed ({e}). Falling back to CPU.")

    index.add(feats)
    return index, feats


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------


def extract_labels(paths: np.ndarray) -> list[str]:
    return [p.replace("\\", "/").split("/")[0] for p in paths]


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
            # FAISS pads with -1 when k_search > n; exclude those and the
            # query itself.  Checking idx >= 0 must come before idx != query_idx
            # because -1 would pass the != check and then silently hit
            # labels[-1] via Python's negative indexing.
            retrieved = [
                idx for idx in neighbours.tolist() if idx >= 0 and idx != query_idx
            ]

            for k in ks:
                p = precision_at_k(relevant, retrieved, k)
                r = recall_at_k(relevant, retrieved, k)
                sum_p[k] += p
                sum_r[k] += r
                sum_f1[k] += f1_at_k(p, r)
                sum_ap[k] += average_precision_at_k(relevant, retrieved, k)

    return {
        k: {
            "precision": round(sum_p[k] / n, 6),
            "recall": round(sum_r[k] / n, 6),
            "f1": round(sum_f1[k] / n, 6),
            "map": round(sum_ap[k] / n, 6),
        }
        for k in ks
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _fusion_label(meta: dict) -> str:
    """Human-readable label for a fusion variant."""
    mode = meta["fusion_mode"]
    if mode == "weighted":
        tw = meta["text_weight"]
        iw = round(1.0 - tw, 1)
        return f"weighted  img={iw}  txt={tw}"
    return mode


def print_variant_results(
    npz_key: str, label: str, results: dict, ks: list[int]
) -> None:
    title = f"  [{npz_key}]  {label}  "
    border = "=" * max(len(title), 62)
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


def print_summary(all_results: dict, fusion_meta: dict, ks: list[int]) -> None:
    """
    Prints two compact summaries that stay readable regardless of the number
    of fusion variants:

    1. Best-per-metric table — for every K, show which variant wins on each
       of P, R, F1, MAP and its score.
    2. Per-variant MAP@K table — one row per variant, one column per K,
       sorted descending by MAP at the median K so the best strategies float
       to the top.
    """
    keys = list(all_results.keys())
    metrics = ("precision", "recall", "f1", "map")
    rep_k = ks[len(ks) // 2]
    col_w = 9  # width for score columns
    var_w = 30  # width for variant-name column

    # ── 1. Best-per-metric table ─────────────────────────────────────────────
    k_col_w = 6

    header = f"{'K':>{k_col_w}}  " + "  ".join(
        f"{m.upper():^{var_w + col_w}}" for m in metrics
    )
    sep = "=" * len(header)

    print(f"\n{sep}")
    print("  BEST VARIANT PER METRIC AT EACH K — cosine similarity")
    print(sep)
    sub = f"{'K':>{k_col_w}}  " + "  ".join(
        f"{'variant':<{var_w}} {'score':>{col_w}}" for _ in metrics
    )
    print(sub)
    print("-" * len(sub))

    for k in ks:
        row = f"{k:>{k_col_w}}  "
        for metric in metrics:
            best_val = -1.0
            best_name = ""
            for npz_key in keys:
                val = all_results[npz_key][k][metric]
                if val > best_val:
                    best_val = val
                    best_name = _fusion_label(fusion_meta[npz_key])
            row += f"{best_name:<{var_w}} {best_val:>{col_w}.4f}  "
        print(row)
    print(sep)

    # ── 2. MAP@K table (all variants, sorted by MAP at median K) ────────────
    sorted_keys = sorted(
        keys,
        key=lambda k_: all_results[k_][rep_k]["map"],
        reverse=True,
    )
    k_headers = "  ".join(f"{k:>{col_w}}" for k in ks)
    var_header = f"{'Variant':<{var_w}}"
    header2 = f"{var_header}  {k_headers}"
    sep2 = "=" * len(header2)

    print(f"\n{sep2}")
    print(f"  MAP@K — ALL VARIANTS (sorted by MAP@{rep_k}, best first)")
    print(sep2)
    print(f"  {'':>{var_w}}  " + "  ".join(f"{'K='+str(k):>{col_w}}" for k in ks))
    print("-" * len(header2))

    for npz_key in sorted_keys:
        lbl = _fusion_label(fusion_meta[npz_key])
        scores = "  ".join(f"{all_results[npz_key][k]['map']:>{col_w}.4f}" for k in ks)
        marker = "  ◄ best" if npz_key == sorted_keys[0] else ""
        print(f"  {lbl:<{var_w}}  {scores}{marker}")
    print(sep2)
    print()


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


def build_output(
    all_results: dict,
    fusion_meta: dict,
    ks: list[int],
) -> dict:
    """
    Builds the full JSON-serialisable results dict.

    Structure
    ---------
    {
      "similarity": "cosine",
      "k_values": [...],
      "variants": {
        "<npz_key>": {
          "fusion_mode":  "weighted",
          "text_weight":  0.3,
          "dim":          512,
          "results": {
            "10": {"precision": ..., "recall": ..., "f1": ..., "map": ...},
            ...
          }
        },
        ...
      },
      "best_per_metric": {
        "<median_k>": {
          "precision": {"variant": ..., "label": ..., "value": ...},
          ...
        }
      }
    }
    """
    rep_k = ks[len(ks) // 2]

    best_per_metric: dict[str, dict] = {}
    for metric in ("precision", "recall", "f1", "map"):
        best_val = -1.0
        best_key = ""
        for npz_key in all_results:
            val = all_results[npz_key][rep_k][metric]
            if val > best_val:
                best_val = val
                best_key = npz_key
        best_per_metric[metric] = {
            "variant": best_key,
            "label": _fusion_label(fusion_meta[best_key]),
            "value": round(best_val, 6),
        }

    variants_out = {}
    for npz_key, results_by_k in all_results.items():
        meta = fusion_meta[npz_key]
        variants_out[npz_key] = {
            "fusion_mode": meta["fusion_mode"],
            "text_weight": meta["text_weight"],  # None for concat/avg
            "dim": meta["dim"],
            "results": {str(k): results_by_k[k] for k in ks},
        }

    return {
        "similarity": SIMILARITY,
        "k_values": ks,
        "variants": variants_out,
        "best_per_metric": {str(rep_k): best_per_metric},
    }


def save_results(output: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[INFO] Results saved to '{output_path}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    print(f"[INFO] Loading features from '{args.npz_path}'...")
    data = np.load(args.npz_path, allow_pickle=True)

    if "fusion_meta" not in data:
        raise KeyError(
            "The .npz file is missing the 'fusion_meta' key. "
            "Re-run extract_features.py to regenerate the feature file."
        )

    fusion_meta: dict = json.loads(data["fusion_meta"].item())

    paths = data["paths"]
    labels = extract_labels(paths)
    n = len(labels)
    n_cls = len(set(labels))

    print(f"[INFO] Dataset: {n} embeddings | {n_cls} classes")
    print(f"[INFO] Fusion variants found in .npz: {list(fusion_meta.keys())}")

    # Also include image-only and text-only baselines if present
    baselines = {}
    for bname, npz_key in [
        ("image_only", "image_features"),
        ("text_only", "text_features"),
    ]:
        if npz_key in data:
            baselines[bname] = npz_key

    # Extend fusion_meta with baseline entries now, before any variant validation
    for bname, npz_key in baselines.items():
        arr = data[npz_key].astype(np.float32)
        fusion_meta[bname] = {
            "fusion_mode": bname,
            "text_weight": None,
            "dim": int(arr.shape[1]),
        }

    # Optionally restrict to a subset of variants (validated against the full
    # set including baselines, which are now in fusion_meta)
    all_known = set(fusion_meta.keys())
    if args.variants:
        unknown = [v for v in args.variants if v not in all_known]
        if unknown:
            raise ValueError(
                f"Requested variant(s) not found in .npz: {unknown}. "
                f"Available: {sorted(all_known)}"
            )
        # Preserve baselines at the front, then requested fused variants
        baseline_keys = [k for k in baselines if k in args.variants]
        fused_keys = [v for v in args.variants if v not in baselines]
        eval_keys = baseline_keys + fused_keys
    else:
        # Default: baselines first, then all fused variants from fusion_meta
        fused_keys = [k for k in fusion_meta if k not in baselines]
        eval_keys = list(baselines.keys()) + fused_keys

    all_results: dict[str, dict] = {}

    for variant_key in eval_keys:
        # Resolve which array to load
        if variant_key in baselines:
            npz_array_key = baselines[variant_key]
            is_concat = False
        else:
            npz_array_key = variant_key
            is_concat = fusion_meta[variant_key]["fusion_mode"] == "concat"

        feats = data[npz_array_key].astype(np.float32)
        label = _fusion_label(fusion_meta[variant_key])

        print(
            f"\n[INFO] Building index for '{variant_key}'  [{label}]  dim={feats.shape[1]} ..."
        )
        index, indexed_feats = build_index(feats, is_concat=is_concat, use_gpu=args.gpu)

        print(f"[INFO] Evaluating '{variant_key}' ...")
        results = evaluate(index, indexed_feats, labels, ks=args.ks)

        print_variant_results(variant_key, label, results, ks=args.ks)
        all_results[variant_key] = results

    print_summary(all_results, fusion_meta, ks=args.ks)

    if args.output:
        output_doc = build_output(all_results, fusion_meta, ks=args.ks)
        save_results(output_doc, args.output)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "CBIR evaluation ablating fusion strategies "
            "(image-only, text-only, concat, avg, weighted×9) "
            "using cosine similarity across all requested K values."
        )
    )
    p.add_argument(
        "--npz-path",
        default=r"output\features\corel10k_infonce_clip_multimodal_features.npz",
        help=(
            "Path to the .npz file produced by extract_features.py. "
            "Must contain 'fusion_meta' and all fused_* arrays."
        ),
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help=(
            "Subset of fusion variants to evaluate (by npz key, e.g. "
            "'fused_concat fused_avg fused_weighted_05'). "
            "Defaults to all variants found in the .npz file."
        ),
    )
    p.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=list(range(10, 101, 10)),
        help="Values of K to evaluate at. Default: 10 20 30 … 100",
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
