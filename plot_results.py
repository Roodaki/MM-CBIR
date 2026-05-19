"""
plot_retrieval_metrics.py
--------------------------
Generates four publication-quality PNGs:
  precision_at_k.png  |  map_at_k.png  |  recall_at_k.png  |  f1_at_k.png

All 13 variants are treated equally — same linewidth, each with a unique
color + marker combination so the reader can distinguish every line.
"""

import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── 0. Load data ──────────────────────────────────────────────────────────────
DATA_PATH = "output\\results\\Corel-10K\\corel10k_retrieval_results.json"

with open(DATA_PATH) as f:
    data = json.load(f)

variants = data["variants"]
k_values = sorted(data["k_values"])

# ── 1. Build a flat ordered list of all variants ──────────────────────────────
VARIANT_ORDER = [
    "image_only",
    "text_only",
    "fused_concat",
    "fused_avg",
    "fused_weighted_01",
    "fused_weighted_02",
    "fused_weighted_03",
    "fused_weighted_04",
    "fused_weighted_05",
    "fused_weighted_06",
    "fused_weighted_07",
    "fused_weighted_08",
    "fused_weighted_09",
]

SHORT_LABELS = {
    "image_only": "Image-only",
    "text_only": "Text-only",
    "fused_concat": "Concat",
    "fused_avg": "Average",
    "fused_weighted_01": "Weighted 0.9/0.1",
    "fused_weighted_02": "Weighted 0.8/0.2",
    "fused_weighted_03": "Weighted 0.7/0.3",
    "fused_weighted_04": "Weighted 0.6/0.4",
    "fused_weighted_05": "Weighted 0.5/0.5",
    "fused_weighted_06": "Weighted 0.4/0.6",
    "fused_weighted_07": "Weighted 0.3/0.7",
    "fused_weighted_08": "Weighted 0.2/0.8",
    "fused_weighted_09": "Weighted 0.1/0.9",
}

# 13 maximally-distinct colors (Okabe-Ito extended + Kelly's set)
COLORS = [
    "#0072B2",  # strong blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # pink
    "#E69F00",  # amber
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow (dark edge makes it visible)
    "#000000",  # black
    "#8B0000",  # dark red
    "#4B0082",  # indigo
    "#008080",  # teal
    "#FF69B4",  # hot pink
    "#6B8E23",  # olive
]

# 13 distinct marker shapes
MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "p", "h", "<", ">", "H"]

LW = 1.7  # uniform linewidth
MS = 6.0  # uniform marker size
MEW = 0.6  # marker edge width

# ── 2. Build style dict ───────────────────────────────────────────────────────
STYLES = {}
for i, key in enumerate(VARIANT_ORDER):
    STYLES[key] = dict(
        label=SHORT_LABELS[key],
        color=COLORS[i],
        marker=MARKERS[i],
    )

# ── 3. Global rcParams ────────────────────────────────────────────────────────
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia"],
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8.8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ── 4. Helper ─────────────────────────────────────────────────────────────────
def get_series(key, metric):
    res = variants[key]["results"]
    return [res[str(k)][metric] for k in k_values]


# ── 5. Plot function ──────────────────────────────────────────────────────────
def make_plot(metric, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    all_vals = []
    for key in VARIANT_ORDER:
        s = STYLES[key]
        vals = get_series(key, metric)
        all_vals.extend(vals)
        edge = "#888800" if s["color"] == "#F0E442" else "white"
        ax.plot(
            k_values,
            vals,
            color=s["color"],
            linestyle="-",
            linewidth=LW,
            marker=s["marker"],
            markersize=MS,
            markerfacecolor=s["color"],
            markeredgecolor=edge,
            markeredgewidth=MEW,
            label=s["label"],
            zorder=3,
        )

    # Auto-compute tight y-limits with 10% padding around the data range
    data_min, data_max = min(all_vals), max(all_vals)
    data_range = data_max - data_min if data_max != data_min else data_max * 0.1
    pad = data_range * 0.10
    ylim = (data_min - pad, data_max + pad)

    ax.set_xlabel("$K$", labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_xlim(k_values[0] - 3, k_values[-1] + 3)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(mticker.AutoLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.grid(which="major", linestyle="--", linewidth=0.45, color="#cccccc", zorder=0)
    ax.grid(which="minor", linestyle=":", linewidth=0.30, color="#e8e8e8", zorder=0)

    handles = [
        Line2D(
            [0],
            [0],
            color=STYLES[k]["color"],
            linestyle="-",
            linewidth=LW,
            marker=STYLES[k]["marker"],
            markersize=MS,
            markerfacecolor=STYLES[k]["color"],
            markeredgecolor="#888800" if STYLES[k]["color"] == "#F0E442" else "white",
            markeredgewidth=MEW,
            label=STYLES[k]["label"],
        )
        for k in VARIANT_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        framealpha=0.95,
        edgecolor="#bbbbbb",
        borderpad=0.7,
        labelspacing=0.45,
        handlelength=2.2,
        handletextpad=0.5,
        fancybox=False,
        fontsize=8.8,
        ncol=1,
        title="img/txt weight",
        title_fontsize=8.5,
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ── 6. Generate all four plots ────────────────────────────────────────────────
make_plot("precision", "Precision@$K$", "precision_at_k.png")
make_plot("map", "MAP@$K$", "map_at_k.png")
make_plot("recall", "Recall@$K$", "recall_at_k.png")
make_plot("f1", "F1@$K$", "f1_at_k.png")
