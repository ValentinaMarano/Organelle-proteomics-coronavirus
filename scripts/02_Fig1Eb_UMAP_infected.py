"""
Generate UMAP visualization of the infected organelle proteomics dataset
with manually positioned HCoV-OC43 viral proteins.

Input:
    data/INFECTED_rowdata_umap.csv
    results/umap_model.joblib

Output:
    figures/umap_infected_with_viral_proteins.pdf
    figures/umap_infected_with_viral_proteins.png
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

FIGURE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

csv_path = DATA_DIR / "INFECTED_rowdata_umap.csv"
model_path = RESULTS_DIR / "umap_model.joblib"


def get_high_density_point(cluster_data):
    """Return the point with the highest local density within a UMAP cluster."""
    if len(cluster_data) < 2:
        return cluster_data.mean(axis=0)

    kde = gaussian_kde(cluster_data.T)
    densities = kde(cluster_data.T)
    return cluster_data[np.argmax(densities)]


data = pd.read_csv(csv_path)

viral_proteins = pd.DataFrame(
    {
        "T: T: Genes": [
            "HEMA",
            "MEM",
            "NP16",
            "NS2A",
            "NSP1",
            "NSP10",
            "NSP12",
            "NSP13",
            "NSP14",
            "NSP15",
            "NSP2",
            "NSP3",
            "NSP5",
            "NSP7",
            "NSP8",
            "NSP9",
            "NUC",
            "ORF",
            "ORF12",
            "SPIKE",
        ],
        "Localizzazione finale": [
            "Golgi apparatus",
            "Others",
            "ER",
            "Cytoplasm",
            "Translation",
            "ER",
            "ER",
            "ER",
            "ER",
            "ER",
            "Lysosome",
            "ER",
            "Mitochondria",
            "ER",
            "ER",
            "ER",
            "Nucleus",
            "Lysosome",
            "Nucleus",
            "ER",
        ],
    }
)

viral_proteins["Localizzazione finale"] = viral_proteins[
    "Localizzazione finale"
].replace(["LD", "Peroxisome", "endosome-vesicle"], "Others")

data["Localizzazione finale"] = data["Localizzazione finale"].replace(
    ["LD", "Peroxisome", "endosome-vesicle"],
    "Others",
)
data["Localizzazione finale"] = data["Localizzazione finale"].replace(
    "protein-complex",
    "Protein-complex",
)

umap_model = load(model_path)

numeric_data = data.select_dtypes(include=["float64", "int64"])
umap_results = umap_model.transform(numeric_data)

data["UMAP1"] = umap_results[:, 0]
data["UMAP2"] = umap_results[:, 1]

cluster_densities = {}

for cluster, group in data.groupby("Localizzazione finale"):
    cluster_points = group[["UMAP1", "UMAP2"]].values
    cluster_densities[cluster] = get_high_density_point(cluster_points)

np.random.seed(42)
jitter = np.random.uniform(-0.2, 0.2, size=(len(viral_proteins), 2))

viral_proteins["UMAP1"] = (
    viral_proteins["Localizzazione finale"].map(lambda loc: cluster_densities[loc][0])
    + jitter[:, 0]
)

viral_proteins["UMAP2"] = (
    viral_proteins["Localizzazione finale"].map(lambda loc: cluster_densities[loc][1])
    + jitter[:, 1]
)

combined_data = pd.concat([data, viral_proteins], ignore_index=True)

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
}

viral_color = "red"

colors = combined_data["Localizzazione finale"].map(color_palette)
viral_mask = combined_data["T: T: Genes"].isin(viral_proteins["T: T: Genes"])
colors.loc[viral_mask] = viral_color

fig, ax = plt.subplots(figsize=(10, 7))

ax.set_title("INFECTED with viral proteins", fontsize=14, fontweight="bold")
ax.set_xlabel("UMAP_1")
ax.set_ylabel("UMAP_2")

ax.scatter(
    combined_data["UMAP1"],
    combined_data["UMAP2"],
    s=10,
    alpha=0.7,
    c=colors,
)

label_offsets = {
    "HEMA": (20, 10),
    "MEM": (25, -20),
    "NP16": (15, -25),
    "NS2A": (10, -30),
    "NSP1": (30, 20),
    "NSP10": (-5, -25),
    "NSP12": (25, 5),
    "NSP13": (-5, 30),
    "NSP14": (15, -20),
    "NSP15": (-5, -30),
    "NSP2": (20, -15),
    "NSP3": (20, 20),
    "NSP5": (25, 20),
    "NSP7": (5, -20),
    "NSP8": (30, -10),
    "NSP9": (15, 30),
    "NUC": (10, 15),
    "ORF": (10, 20),
    "ORF12": (20, 15),
    "SPIKE": (15, 10),
}

for _, row in viral_proteins.iterrows():
    offset_x, offset_y = label_offsets.get(row["T: T: Genes"], (10, 10))

    ax.annotate(
        row["T: T: Genes"],
        (row["UMAP1"], row["UMAP2"]),
        textcoords="offset points",
        xytext=(offset_x, offset_y),
        ha="center",
        fontsize=8,
        fontweight="bold",
        color=viral_color,
        arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
    )

unique_categories = list(color_palette.keys()) + ["HCoV-OC43 viral proteins"]

handles = [
    plt.Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        label=cat,
        markersize=10,
        markerfacecolor=(
            viral_color if cat == "HCoV-OC43 viral proteins" else color_palette[cat]
        ),
    )
    for cat in unique_categories
]

ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()

fig.savefig(FIGURE_DIR / "umap_infected_with_viral_proteins.pdf", bbox_inches="tight")
fig.savefig(
    FIGURE_DIR / "umap_infected_with_viral_proteins.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()