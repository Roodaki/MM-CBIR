"""
plot_precision_at_k.py
----------------------
Generates two separate publication-quality PNGs:
  1. precision_at_k_cosine.png  — Precision@K
  2. map_at_k_cosine.png        — MAP@K
"""

import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── 0. Load data ──────────────────────────────────────────────────────────────
DATA_PATH = "./output\\results\\corel10k_retrieval_results.json"
OUT_PRECISION = "./output\\results\\precision_at_k_cosine.png"
OUT_MAP = "./output\\results\\map_at_k_cosine.png"

with open(DATA_PATH) as f:
    data = json.load(f)

# ── 1. Extract cosine values ──────────────────────────────────────────────────
STRATEGIES = {
    "image_only": "Image-Only",
    "text_only": "Text-Only",
    "fused": "Fused (Image + Text)",
}

series = {}
for key, label in STRATEGIES.items():
    cos = data[key]["cosine"]
    ks = sorted(int(k) for k in cos.keys())
    precision = [cos[str(k)]["precision"] for k in ks]
    map_vals = [cos[str(k)]["map"] for k in ks]
    series[label] = (ks, precision, map_vals)

# ── 2. Style / theme ──────────────────────────────────────────────────────────
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
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

PALETTE = {
    "Image-Only": "#0072B2",
    "Text-Only": "#D55E00",
    "Fused (Image + Text)": "#009E73",
}
MARKERS = {
    "Image-Only": "o",
    "Text-Only": "s",
    "Fused (Image + Text)": "^",
}
OFFSETS = {
    "Image-Only": (0, 6),
    "Text-Only": (0, 6),
    "Fused (Image + Text)": (0, -11),
}


# ── 3. Reusable plot function ─────────────────────────────────────────────────
def make_plot(metric_idx, ylabel, ylim, y_major, y_minor, title, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for label, (ks, precision, map_vals) in series.items():
        vals = precision if metric_idx == 0 else map_vals
        ax.plot(
            ks,
            vals,
            color=PALETTE[label],
            linestyle="-",
            linewidth=1.8,
            marker=MARKERS[label],
            markersize=5.5,
            markerfacecolor=PALETTE[label],
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=label,
            zorder=3,
        )
        dx, dy = OFFSETS[label]
        for k, v in zip(ks, vals):
            ax.annotate(
                f"{v:.3f}",
                xy=(k, v),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=6.2,
                color=PALETTE[label],
                ha="center",
                va="bottom" if dy >= 0 else "top",
            )

    ax.set_xlabel("$K$", labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_title(title, pad=10)
    ax.set_xlim(5, 105)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_minor))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(which="major", linestyle="--", linewidth=0.45, color="#cccccc", zorder=0)
    ax.grid(which="minor", linestyle=":", linewidth=0.30, color="#e8e8e8", zorder=0)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE[lbl],
            linestyle="-",
            linewidth=1.8,
            marker=MARKERS[lbl],
            markersize=5.5,
            markerfacecolor=PALETTE[lbl],
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=lbl,
        )
        for lbl in STRATEGIES.values()
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        framealpha=0.92,
        edgecolor="#bbbbbb",
        borderpad=0.6,
        labelspacing=0.4,
        handlelength=2.2,
        fancybox=False,
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ── 4. Generate both plots ────────────────────────────────────────────────────
make_plot(
    metric_idx=0,
    ylabel="Precision@$K$",
    ylim=(0.68, 0.96),
    y_major=0.05,
    y_minor=0.025,
    title="",
    out_path=OUT_PRECISION,
)

make_plot(
    metric_idx=1,
    ylabel="MAP@$K$",
    ylim=(0.62, 0.95),
    y_major=0.05,
    y_minor=0.025,
    title="",
    out_path=OUT_MAP,
)
