"""
Generate UMAP visualization of the uninfected organelle proteomics dataset.

Input:
    data/MOCK_rowdata_umap.csv

Output:
    figures/umap_uninfected.pdf
    figures/umap_uninfected.png
    results/umap_model.joblib
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import umap
from joblib import dump, load


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

FIGURE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

csv_path = DATA_DIR / "MOCK_rowdata_umap.csv"
model_path = RESULTS_DIR / "umap_model.joblib"

data = pd.read_csv(csv_path)
numeric_data = data.select_dtypes(include=["float64", "int64"])

if model_path.exists():
    umap_model = load(model_path)
else:
    umap_model = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    umap_model.fit(numeric_data)
    dump(umap_model, model_path)
    print("UMAP model generated and saved.")

data["Localizzazione finale"] = data["Localizzazione finale"].replace(
    ["LD", "Peroxisome", "endosome-vesicle"],
    "Others",
)
data["Localizzazione finale"] = data["Localizzazione finale"].replace(
    "protein-complex",
    "Protein-complex",
)

umap_results = umap_model.transform(numeric_data)

localization = data["Localizzazione finale"].fillna("Unknown").astype("category")

if "Others" in localization.cat.categories:
    localization = localization.cat.reorder_categories(
        [cat for cat in localization.cat.categories if cat != "Others"] + ["Others"],
        ordered=True,
    )

unique_categories = localization.cat.categories

color_palette = {
    "Cytoplasm": mcolors.TABLEAU_COLORS["tab:blue"],
    "ER": mcolors.TABLEAU_COLORS["tab:green"],
    "Golgi apparatus": mcolors.TABLEAU_COLORS["tab:orange"],
    "Lysosome": "#B8860B",
    "Mitochondria": "#800000",
    "Nucleus": mcolors.TABLEAU_COLORS["tab:purple"],
    "PM": mcolors.TABLEAU_COLORS["tab:pink"],
    "Protein-complex": "#ADD8E6",
    "Translation": "darkgray",
    "Others": "lightgray",
    "Unknown": "lightgray",
}

colors = localization.map(color_palette)

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(
    umap_results[:, 0],
    umap_results[:, 1],
    s=10,
    alpha=0.7,
    c=colors,
)

ax.set_title("UNINFECTED")
ax.set_xlabel("UMAP_1")
ax.set_ylabel("UMAP_2")

proteins_to_highlight = [
    "NUP93",
    "PSMB5",
    "RPS6",
    "CANX",
    "PDIA3",
    "CALR",
    "GAPDH",
    "GOLGA2",
    "CTSB",
    "CS",
    "CD55",
]

label_offsets = {
    "NUP93": (30, 40),
    "PSMB5": (-50, 20),
    "RPS6": (-40, 10),
    "CANX": (-20, -40),
    "PDIA3": (30, 25),
    "CALR": (-40, 25),
    "GAPDH": (10, 20),
    "GOLGA2": (30, -30),
    "CTSB": (20, 20),
    "CS": (15, 45),
    "CD55": (40, -40),
}

for protein in proteins_to_highlight:
    protein_data = data[data["T: T: Genes"] == protein]

    if not protein_data.empty:
        idx = protein_data.index[0]
        x, y = umap_results[idx, 0], umap_results[idx, 1]
        protein_color = color_palette.get(localization.iloc[idx], "black")
        offset_x, offset_y = label_offsets.get(protein, (10, 10))

        ax.annotate(
            protein,
            (x, y),
            textcoords="offset points",
            xytext=(offset_x, offset_y),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=protein_color,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
        )

handles = [
    plt.Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        label=cat,
        markersize=10,
        markerfacecolor=color_palette.get(cat, "lightgray"),
    )
    for cat in unique_categories
]

ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()

fig.savefig(FIGURE_DIR / "umap_uninfected.pdf", bbox_inches="tight")
fig.savefig(FIGURE_DIR / "umap_uninfected.png", dpi=300, bbox_inches="tight")

plt.show()