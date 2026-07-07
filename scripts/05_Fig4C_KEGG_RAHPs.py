"""
Marano et al.

Figure 4C

Generate the KEGG pathway enrichment dot plot shown in Figure 4C.

The enrichment analysis was performed using DAVID.
This script generates the publication-ready visualization from the
selected enrichment results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"

FIGURE_DIR.mkdir(exist_ok=True)

# ============================================================================
# Selected KEGG pathways reported in Figure 4C.
# Enrichment statistics (P-value and Fold Enrichment) were obtained using DAVID.
# ============================================================================

kegg_results = {
    "Term": [
        "Fatty acid metabolism",
        "Biosynthesis of unsaturated fatty acids",
        "Peroxisome",
        "Protein digestion and absorption",
        "Steroid biosynthesis",
        "Metabolic pathways",
        "ECM-receptor interaction",
    ],
    "P": [
        4.8e-5,
        5.0e-5,
        1.4e-4,
        8.9e-4,
        2.4e-3,
        2.9e-3,
        3.0e-3,
    ],
    "FE": [
        7.8,
        13.5,
        6.6,
        7.6,
        13.9,
        1.6,
        5.9,
    ],
}

df = pd.DataFrame(kegg_results)

# ============================================================================
# Convert P-values to -log10(P) and assign discrete colors
# ============================================================================

eps = 1e-300

df["neglog10P"] = -np.log10(np.clip(df["P"], eps, None))

p_bins = [0, 2, 4, 6, 8, np.inf]

p_colors = [
    "#c7e9b4",
    "#7fcdbb",
    "#41b6c4",
    "#2c7fb8",
    "#253494",
]

p_labels = ["2", "4", "6"]

p_idx = np.digitize(df["neglog10P"], p_bins, right=False) - 1
p_idx = np.clip(p_idx, 0, len(p_colors) - 1)

df["color"] = [p_colors[i] for i in p_idx]

# ============================================================================
# Plot
# ============================================================================

terms = list(df["Term"])
df["y"] = df["Term"].apply(lambda t: terms.index(t))

fig, ax = plt.subplots(figsize=(7.5, 7.5))

ax.scatter(
    df["FE"],
    df["y"],
    s=80,
    c=df["color"],
    marker="o",
    alpha=0.9,
    edgecolors="white",
    linewidth=0.7,
)

ax.set_title("KEGG pathway enrichment of RAHPs", fontsize=13, pad=10)

ax.set_xlabel("Fold enrichment")

ax.set_yticks(range(len(terms)))
ax.set_yticklabels(terms)
ax.invert_yaxis()

ax.set_xlim(0, max(df["FE"]) + 6)

ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)

# ============================================================================
# Legend
# ============================================================================

color_handles = [
    plt.scatter([], [], s=80, color=c, edgecolors="white")
    for c in p_colors
]

ax.legend(
    color_handles,
    p_labels,
    title="−log10(P)",
    loc="upper left",
    bbox_to_anchor=(1.02, 0.95),
    frameon=True,
)

# ============================================================================
# Optional annotations
# ============================================================================

labels = {
    # Example:
    # "Peroxisome": "PEX1, PEX3, PEX5",
}

offset = {}

for _, row in df.iterrows():

    if row["Term"] in labels:

        dx, dy = offset.get(row["Term"], (8, -6))

        ax.annotate(
            labels[row["Term"]],
            (row["FE"], row["y"]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="left",
            va="center",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="none",
                alpha=0.8,
            ),
        )

plt.subplots_adjust(right=0.78, left=0.35)

output = FIGURE_DIR / "Fig4C_KEGG_RAHPs.pdf"

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight",
)

print(f"Saved: {output}")

plt.show()
