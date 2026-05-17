"""
plot_precision_at_k.py
----------------------
Plots Precision@K (cosine similarity) for all retrieval strategies
across all available K values. Publication-quality output saved as PNG.
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── 0. Load data ──────────────────────────────────────────────────────────────
DATA_PATH = "output\\results\\corel10k_retrieval_results.json"
OUTPUT_PNG = "output\\results\\precision_at_k_cosine.png"

with open(DATA_PATH) as f:
    data = json.load(f)

# ── 1. Extract cosine / precision values ──────────────────────────────────────
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
    series[label] = (ks, precision)

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
        "pdf.fonttype": 42,  # embeds fonts for journal submission
        "ps.fonttype": 42,
    }
)

# Accessible, colourblind-safe palette (Wong 2011)
PALETTE = {
    "Image-Only": "#0072B2",  # blue
    "Text-Only": "#D55E00",  # vermillion
    "Fused (Image + Text)": "#009E73",  # green
}
MARKERS = {
    "Image-Only": "o",
    "Text-Only": "s",
    "Fused (Image + Text)": "^",
}
LINESTYLES = {
    "Image-Only": "-",
    "Text-Only": "-",
    "Fused (Image + Text)": "-",
}

# ── 3. Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.2))

for label, (ks, precision) in series.items():
    ax.plot(
        ks,
        precision,
        color=PALETTE[label],
        linestyle=LINESTYLES[label],
        linewidth=1.8,
        marker=MARKERS[label],
        markersize=5.5,
        markerfacecolor=PALETTE[label],
        markeredgecolor="white",
        markeredgewidth=0.6,
        label=label,
        zorder=3,
    )

# ── 4. Reference annotations ──────────────────────────────────────────────────
# Annotate every data point with its precision value
# Alternate above/below per strategy to avoid overlap
OFFSETS = {
    "Image-Only": (0, 6),
    "Text-Only": (0, 6),
    "Fused (Image + Text)": (0, -11),
}
for label, (ks, precision) in series.items():
    dx, dy = OFFSETS[label]
    for k, p in zip(ks, precision):
        ax.annotate(
            f"{p:.3f}",
            xy=(k, p),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.2,
            color=PALETTE[label],
            ha="center",
            va="bottom" if dy >= 0 else "top",
        )

# ── 5. Axes / labels ──────────────────────────────────────────────────────────
ax.set_xlabel("Number of Retrieved Documents  $K$", labelpad=6)
ax.set_ylabel("Precision@$K$  (Cosine Similarity)", labelpad=6)
ax.set_title("Retrieval Precision@$K$ by Strategy  —  Cosine Similarity", pad=10)

ax.set_xlim(5, 105)
ax.set_ylim(0.68, 0.96)

ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.025))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

ax.grid(which="major", linestyle="--", linewidth=0.45, color="#cccccc", zorder=0)
ax.grid(which="minor", linestyle=":", linewidth=0.30, color="#e8e8e8", zorder=0)

# ── 6. Legend ─────────────────────────────────────────────────────────────────
legend_handles = [
    Line2D(
        [0],
        [0],
        color=PALETTE[lbl],
        linestyle=LINESTYLES[lbl],
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

# ── 7. Save ───────────────────────────────────────────────────────────────────
fig.tight_layout()
fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUTPUT_PNG}")
plt.close(fig)
