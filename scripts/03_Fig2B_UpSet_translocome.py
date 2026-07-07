"""
Generate an UpSet-like plot comparing translocome, proteome,
host-factor, phosphosite and interactor datasets.

Input:
    data/Upset_plot_translocome_vs_host_interaction_phospho.txt

Outputs:
    results/upset_intersections.tsv
    figures/upset_translocome_host_phospho.png
"""

from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

input_file = DATA_DIR / "Upset_plot_translocome_vs_host_interaction_phospho.txt"

sep = "\t"
min_intersection_size = 15

break_lower = 500
break_upper = 1000

desired_order = [
    "TRANSLOCOME",
    "PROTEOME",
    "HOST FACTORS",
    "PHOSPHOSITES",
    "ALL INTERACTORS + NSP4-6",
]


df = pd.read_csv(input_file, sep=sep)

df.columns = [c.split("(")[0].strip() for c in df.columns]

rename_columns = {
    "TRASLOCOME": "TRANSLOCOME",
    "PROTEOMA": "PROTEOME",
    "PHOSPOSITES": "PHOSPHOSITES",
    "TUTTI GLI INTERATTORI + NSP4-6": "ALL INTERACTORS + NSP4-6",
}

df = df.rename(columns=rename_columns)

datasets = desired_order

gene_sets = {
    col: set(df[col].dropna().astype(str))
    for col in datasets
}

set_sizes = {
    col: len(genes)
    for col, genes in gene_sets.items()
}

comb_list = []

for r in range(1, len(datasets) + 1):
    for comb in itertools.combinations(datasets, r):
        intersection = set.intersection(*(gene_sets[c] for c in comb))
        size = len(intersection)

        if size >= min_intersection_size and len(comb) > 1:
            comb_list.append(
                {
                    "sets": comb,
                    "size": size,
                    "genes": sorted(intersection),
                }
            )

comb_list.sort(key=lambda x: x["size"], reverse=True)

df_intersections = pd.DataFrame(
    [
        {
            "Combination": " & ".join(c["sets"]),
            "Size": c["size"],
            "Genes": ";".join(c["genes"]),
        }
        for c in comb_list
    ]
)

df_intersections.to_csv(
    RESULTS_DIR / "upset_intersections.tsv",
    sep="\t",
    index=False,
)

fig = plt.figure(figsize=(10, 4))

gs = gridspec.GridSpec(
    2,
    2,
    width_ratios=[1.2, 4],
    height_ratios=[1.3, 1],
    hspace=0.05,
    wspace=0.1,
)

ax_sets = fig.add_subplot(gs[:, 0])

y_pos = np.arange(len(datasets))

ax_sets.barh(
    y_pos,
    [set_sizes[d] for d in datasets],
    color="black",
)

ax_sets.set_yticks(y_pos)
ax_sets.set_yticklabels(datasets)
ax_sets.invert_yaxis()
ax_sets.set_xlabel("Total proteins\nannotated in each dataset")

for i, dataset in enumerate(datasets):
    value = set_sizes[dataset]
    ax_sets.text(
        value + max(set_sizes.values()) * 0.01,
        i,
        str(value),
        va="center",
        ha="left",
        fontsize=8,
    )

sizes = [c["size"] for c in comb_list]
x_pos = np.arange(len(comb_list))
max_size = max(sizes) if sizes else 0

gs_top = gridspec.GridSpecFromSubplotSpec(
    2,
    1,
    subplot_spec=gs[0, 1],
    height_ratios=[2, 1],
    hspace=0.05,
)

ax_top_low = fig.add_subplot(gs_top[0, 0])
ax_top_high = fig.add_subplot(gs_top[1, 0], sharex=ax_top_low)

ax_top_low.bar(x_pos, sizes, color="black")
ax_top_high.bar(x_pos, sizes, color="black")

ax_top_low.set_ylim(0, break_lower)
ax_top_high.set_ylim(break_upper, max_size * 1.05)

ax_top_low.set_ylabel("Number of\nproteins in set", labelpad=5)
ax_top_high.set_xticks([])
ax_top_low.set_xticks([])

kwargs = dict(
    transform=ax_top_low.transAxes,
    color="black",
    clip_on=False,
    linewidth=0.7,
)

ax_top_low.plot((-0.015, +0.015), (1, 1), **kwargs)
ax_top_low.plot((0.985, 1.015), (1, 1), **kwargs)

kwargs = dict(
    transform=ax_top_high.transAxes,
    color="black",
    clip_on=False,
    linewidth=0.7,
)

ax_top_high.plot((-0.015, +0.015), (0, 0), **kwargs)
ax_top_high.plot((0.985, 1.015), (0, 0), **kwargs)

for i, combination in enumerate(comb_list):
    value = combination["size"]
    axis = ax_top_high if value > break_lower else ax_top_low

    axis.text(
        i,
        value + max_size * 0.01,
        str(value),
        ha="center",
        va="bottom",
        fontsize=7,
    )

ax_mat = fig.add_subplot(gs[1, 1], sharex=ax_top_low)

for xi, combination in enumerate(comb_list):
    present_sets = combination["sets"]

    for yi, dataset in enumerate(datasets):
        ax_mat.scatter(
            xi,
            yi,
            s=20,
            color="lightgrey",
            alpha=0.3,
        )

    y_present = []

    for yi, dataset in enumerate(datasets):
        if dataset in present_sets:
            ax_mat.scatter(
                xi,
                yi,
                s=25,
                color="black",
            )
            y_present.append(yi)

    if len(y_present) > 1:
        ax_mat.vlines(
            xi,
            min(y_present),
            max(y_present),
            color="black",
            linewidth=1,
        )

ax_mat.set_yticks(np.arange(len(datasets)))
ax_mat.set_yticklabels([])
ax_mat.invert_yaxis()
ax_mat.set_xlabel("")
ax_mat.tick_params(axis="x", length=0)

plt.tight_layout()

fig.savefig(
    FIGURE_DIR / "upset_translocome_host_phospho.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    FIGURE_DIR / "upset_translocome_host_phospho.pdf",
    bbox_inches="tight",
)

plt.show()