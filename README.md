# Organelle proteomics reveals coronavirus-induced remodeling of cellular organelles

This repository contains the Python scripts used to generate the analyses and figures associated with the manuscript:

**Marano et al.**

## Repository structure

```
scripts/    Python scripts used for data analysis and figure generation
data/       Input data (not included)
results/    Generated output tables
figures/    Generated figures
```

## Scripts

| Figure | Description | Script |
|--------|-------------|--------|
| Fig. 1E(a) | UMAP projection of the uninfected spatial proteome | `01_Fig1Ea_UMAP_uninfected.py` |
| Fig. 1E(b) | UMAP projection of the infected spatial proteome with viral proteins | `02_Fig1Eb_UMAP_infected.py` |
| Fig. 2B | UpSet plot comparing the Translocome with published host factor, phosphosite and interaction datasets | `03_Fig2B_UpSet_translocome.py` |
| Fig. 4B | UpSet plot of RAHP overlap with SARS-CoV-2 nsp interaction datasets | `04_Fig4B_UpSet_RAHPs_nsp_interactors.py` |
| Fig. 4C | KEGG pathway enrichment dot plot of RAHPs | `05_Fig4C_KEGG_RAHPs.py` |
| Fig. 4D | Quantification of RAHP re-localization to SARS-CoV-2 replication sites | `06_Fig4D_RAHP_dsRNA_colocalization.py` |

## Data availability

The datasets used in this study are not included in this repository.

Input datasets are not distributed with this repository.

The quantitative proteomics datasets and microscopy images generated in this study are available from the corresponding authors upon reasonable request.
