"""
Marano et al.

Figure 4D

Quantification of RAHP re-localization to SARS-CoV-2 replication sites.

For each infected cell, the script calculates the re-localization ratio as:

    Mean fluorescence intensity within the dsRNA mask
    -------------------------------------------------
      Mean fluorescence intensity within the ER mask

The same analysis is performed for PDI as an ER marker control.

Statistical significance is assessed using a two-sided Mann–Whitney U test.

Input
-----
data/Fig4D_images/*.tif

Outputs
-------
results/Fig4D_colocalization/results.csv
results/Fig4D_colocalization/qc/
figures/Fig4D_colocalization_plot.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from skimage import io, filters, morphology, measure, segmentation, feature
import warnings
warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data" / "Fig4D_images"

RESULTS_DIR = ROOT / "results" / "Fig4D_colocalization"

FIGURE_DIR = ROOT / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

# ============================================================================
# Channels
# ============================================================================

CH_DAPI = 0
CH_RAHP = 2
CH_PDI = 1
CH_DSRNA = 3

RAHP_NAME = "HSD17B11"   # Change according to the analyzed RAHP

NUCLEUS_MIN_DIST = 60
MIN_NUCLEUS_AREA = 1000
CELL_EXPANSION = 80
DSRNA_MIN_AREA = 500
ER_PERCENTILE = 30
ER_DILATION = 10

# ─────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────

def load_image(path):
    img = io.imread(str(path))
    if img.ndim == 3 and img.shape[2] <= 5:
        img = np.moveaxis(img, -1, 0)
    return img.astype(np.float32)

def normalize(img):
    mn, mx = img.min(), img.max()
    if mx == mn:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)

def otsu_mask(channel, extra_mask=None):
    """
    Soglia Otsu globale sull intera immagine.
    Se extra_mask e fornita, applica la maschera al risultato.
    """
    blurred = filters.gaussian(channel, sigma=2)
    thresh  = filters.threshold_otsu(blurred)
    binary  = blurred > thresh
    if extra_mask is not None:
        binary = binary & extra_mask
    return binary

# ─────────────────────────────────────────────
# SEGMENTAZIONE
# ─────────────────────────────────────────────

def segment_nuclei(dapi_ch):
    """Nuclei dal DAPI con watershed → label map."""
    blurred  = filters.gaussian(dapi_ch, sigma=3)
    thresh   = filters.threshold_otsu(blurred)
    binary   = blurred > thresh
    binary   = morphology.remove_small_holes(binary, area_threshold=2000)
    binary   = morphology.remove_small_objects(binary, min_size=MIN_NUCLEUS_AREA)
    distance = ndimage.distance_transform_edt(binary)
    coords   = feature.peak_local_max(distance, min_distance=NUCLEUS_MIN_DIST, labels=binary)
    peaks    = np.zeros(distance.shape, dtype=bool)
    peaks[tuple(coords.T)] = True
    markers, _ = ndimage.label(peaks)
    labels     = segmentation.watershed(-distance, markers, mask=binary)
    return labels

def expand_to_cell(nucleus_labels, pdi_ch):
    """
    Espande ogni nucleo dentro la zona PDI usando il gradiente del PDI
    come superficie di watershed. Il confine tra cellule vicine viene
    messo dove il segnale PDI scende — cioè nella zona buia tra i citoplasmi.
    """
    blurred   = filters.gaussian(pdi_ch, sigma=3)
    thresh    = filters.threshold_otsu(blurred)
    cell_zone = blurred > thresh
    cell_zone = morphology.remove_small_holes(cell_zone, area_threshold=5000)
    cell_zone = morphology.remove_small_objects(cell_zone, min_size=2000)
    full_zone = cell_zone | (nucleus_labels > 0)

    # usa il negativo del PDI come superficie:
    # i picchi (massimo PDI) vengono assegnati prima,
    # i confini vengono messi nei minimi locali (zona buia tra cellule)
    surface     = -blurred
    cell_labels = segmentation.watershed(surface, nucleus_labels, mask=full_zone)
    return cell_labels

# ─────────────────────────────────────────────
# QC
# ─────────────────────────────────────────────

def save_qc(img_path, img, nucleus_labels, cell_labels,
            er_masks, dsrna_masks, infected_ids, results_per_cell,
            er_global, dsrna_global):
    from skimage.segmentation import find_boundaries

    qc_dir = RESULTS_DIR / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    er_all = np.zeros(img[CH_PDI].shape, dtype=bool)
    for m in er_masks.values():
        er_all |= m
    ds_all = np.zeros(img[CH_DSRNA].shape, dtype=bool)
    for m in dsrna_masks.values():
        ds_all |= m

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))

    # ── riga 1: canali grezzi ──
    for ax, (ch_idx, title, cmap) in zip(axes[0], [
        (CH_DAPI,   "DAPI",   "Blues"),
        (CH_PDI,    "PDI",    "Greens"),
        (CH_RAHP, RAHP_NAME, "Purples"),
        (CH_DSRNA,  "dsRNA",  "Oranges"),
    ]):
        ax.imshow(normalize(img[ch_idx]), cmap=cmap)
        ax.set_title(f"Grezzo: {title}", fontsize=11)
        ax.axis("off")

    # ── riga 2: maschere globali (prima di restringere per cellula) ──
    # pan 1: nuclei
    ax = axes[1][0]
    ov = np.stack([normalize(img[CH_DAPI])]*3, axis=-1)
    ov[find_boundaries(nucleus_labels, mode="outer")] = [1, 1, 0]
    ax.imshow(np.clip(ov, 0, 1))
    ax.set_title(f"Nuclei segmentati ({nucleus_labels.max()})", fontsize=11)
    ax.axis("off")

    # pan 2: ER globale (maschera PDI grezza su tutta immagine)
    ax = axes[1][1]
    ov = np.stack([normalize(img[CH_PDI])]*3, axis=-1)
    ov[er_global] = np.clip(ov[er_global] * 0.4 + [0, 0.6, 0.2], 0, 1)
    ov[find_boundaries(er_global.astype(np.uint8), mode="outer")] = [0, 1, 0]
    ax.imshow(np.clip(ov, 0, 1))
    n_er = er_global.sum()
    ax.set_title(f"ER globale (verde) — {n_er:,} px", fontsize=11)
    ax.axis("off")

    # pan 3: dsRNA globale (maschera dsRNA grezza su tutta immagine)
    ax = axes[1][2]
    ov = np.stack([normalize(img[CH_DSRNA])]*3, axis=-1)
    ov[dsrna_global] = np.clip(ov[dsrna_global] * 0.3 + [0.7, 0, 0], 0, 1)
    ov[find_boundaries(dsrna_global.astype(np.uint8), mode="outer")] = [1, 0.2, 0.2]
    ax.imshow(np.clip(ov, 0, 1))
    n_ds = dsrna_global.sum()
    ax.set_title(f"dsRNA globale (rosso) — {n_ds:,} px", fontsize=11)
    ax.axis("off")

    # pan 4: sovrapposizione ER e dsRNA globali
    ax = axes[1][3]
    ov = np.stack([normalize(img[CH_PDI])]*3, axis=-1)
    ov[er_global]    = np.clip(ov[er_global]    * 0.4 + [0, 0.5, 0.5], 0, 1)
    ov[dsrna_global] = [1, 0.2, 0.2]
    overlap = er_global & dsrna_global
    ov[overlap] = [1, 1, 0]   # giallo = sovrapposizione ER+dsRNA
    ax.imshow(np.clip(ov, 0, 1))
    ax.set_title("Sovrapposizione: ER(ciano) dsRNA(rosso) overlap(giallo)", fontsize=9)
    ax.axis("off")

    # ── riga 3: maschere per cellula + misure ──
    # pan 1: ER per cellula (dopo esclusione dsRNA)
    ax = axes[2][0]
    ov = np.stack([normalize(img[CH_PDI])]*3, axis=-1)
    ov[er_all] = np.clip(ov[er_all] * 0.4 + [0, 0.6, 0.2], 0, 1)
    ov[find_boundaries(er_all.astype(np.uint8), mode="outer")] = [0, 1, 0]
    ax.imshow(np.clip(ov, 0, 1))
    ax.set_title("ER per cellula (dsRNA escluso) — usato per misura", fontsize=10)
    ax.axis("off")

    # pan 2: dsRNA per cellula
    ax = axes[2][1]
    ov = np.stack([normalize(img[CH_DSRNA])]*3, axis=-1)
    ov[ds_all] = np.clip(ov[ds_all] * 0.3 + [0.7, 0, 0], 0, 1)
    ov[find_boundaries(ds_all.astype(np.uint8), mode="outer")] = [1, 0.2, 0.2]
    ax.imshow(np.clip(ov, 0, 1))
    ax.set_title(f"dsRNA per cellula — {len(infected_ids)} cellule infette", fontsize=10)
    ax.axis("off")

    # pan 3: RAHP con maschere sovrapposte
    ax = axes[2][2]
    ov = np.stack([normalize(img[CH_RAHP])]*3, axis=-1)
    ov[er_all] = np.clip(ov[er_all] * 0.5 + [0, 0.3, 0.3], 0, 1)
    ov[ds_all] = np.clip(ov[ds_all] * 0.4 + [0.6, 0, 0], 0, 1)
    ax.imshow(np.clip(ov, 0, 1))
    ax.set_title("RAHP con ER(ciano) e dsRNA(rosso)", fontsize=10)
    ax.axis("off")

    # pan 4: finale completo
    ax = axes[2][3]
    rgb = np.zeros((*img[CH_DAPI].shape, 3), dtype=np.float32)
    rgb[..., 0] = normalize(img[CH_RAHP])
    rgb[..., 1] = normalize(img[CH_PDI])
    rgb[..., 2] = normalize(img[CH_DAPI])
    for cid in infected_ids:
        rgb[find_boundaries(cell_labels == cid, mode="outer")] = [1, 1, 0]
    rgb[er_all] = np.clip(rgb[er_all] * 0.5 + [0, 0.5, 0.5], 0, 1)
    rgb[ds_all] = [1, 0.15, 0.15]
    ax.imshow(np.clip(rgb, 0, 1))
    ax.set_title("Finale: bordi(giallo) ER(ciano) dsRNA(rosso)", fontsize=11)
    ax.axis("off")

    # ── tabella numeri per cellula infetta ──
    if infected_ids and results_per_cell:
        fig2, ax2 = plt.subplots(figsize=(12, max(2, len(infected_ids) * 0.7 + 1.5)))
        ax2.axis("off")
        cols = ["cell", "ER area (px)", "dsRNA area (px)",
                "RAHP in ER", "RAHP in dsRNA", "ratio RAHP",
                "PDI in ER", "PDI in dsRNA", "ratio PDI"]
        rows = []
        for r in results_per_cell:
            rows.append([
                r["cell_id"],
                f"{r['er_area_px']:,}",
                f"{r['dsrna_area_px']:,}",
                f"{r['RAHP_in_er']:.1f}",
                f"{r['RAHP_in_dsrna']:.1f}",
                f"{r['ratio_RAHP']:.3f}",
                f"{r['pdi_in_er']:.1f}",
                f"{r['pdi_in_dsrna']:.1f}",
                f"{r['ratio_PDI']:.3f}",
            ])
        tbl = ax2.table(cellText=rows, colLabels=cols,
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.8)
        # colora header
        for j in range(len(cols)):
            tbl[0, j].set_facecolor("#DDDDDD")
            tbl[0, j].set_text_props(fontweight="bold")
        # colora ratio RAHP e PDI
        for i in range(1, len(rows)+1):
            tbl[i, 5].set_facecolor("#D4B8F0")  # ratio RAHP viola
            tbl[i, 8].set_facecolor("#A8DFC4")  # ratio PDI verde
        ax2.set_title(f"{img_path.name} — misure per cellula infetta", fontsize=11, pad=20)
        fig2.tight_layout()
        out2 = qc_dir / (img_path.stem + "_qc_numeri.png")
        fig2.savefig(out2, dpi=120, bbox_inches="tight")
        plt.close(fig2)
        print(f"    → QC numeri: {out2}")

    plt.suptitle(img_path.name, fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = qc_dir / (img_path.stem + "_qc.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    → QC: {out}")

# ─────────────────────────────────────────────
# ANALISI PER IMMAGINE
# ─────────────────────────────────────────────

def analyze_image(img_path):
    img    = load_image(img_path)
    dapi   = normalize(img[CH_DAPI])
    RAHP = img[CH_RAHP]
    pdi    = img[CH_PDI]
    dsrna  = img[CH_DSRNA]

    # segmentazione globale
    nucleus_labels = segment_nuclei(dapi)
    cell_labels    = expand_to_cell(nucleus_labels, pdi)

    # debug scale canali grezzi
    print(f"    DAPI   min={img[CH_DAPI].min():.1f}  max={img[CH_DAPI].max():.1f}  mean={img[CH_DAPI].mean():.1f}")
    print(f"    RAHP min={img[CH_RAHP].min():.1f}  max={img[CH_RAHP].max():.1f}  mean={img[CH_RAHP].mean():.1f}")
    print(f"    PDI    min={img[CH_PDI].min():.1f}  max={img[CH_PDI].max():.1f}  mean={img[CH_PDI].mean():.1f}")
    print(f"    dsRNA  min={img[CH_DSRNA].min():.1f}  max={img[CH_DSRNA].max():.1f}  mean={img[CH_DSRNA].mean():.1f}")

    nucleus_mask  = nucleus_labels > 0
    cyto_mask     = ~nucleus_mask

    # ER: soglia al percentile 30 globale → include anche segnali PDI deboli in periferia
    pdi_blurred  = filters.gaussian(pdi, sigma=2)
    er_thresh    = np.percentile(pdi_blurred[cyto_mask], ER_PERCENTILE)
    er_global    = (pdi_blurred > er_thresh) & cyto_mask
    er_global    = morphology.remove_small_holes(er_global, area_threshold=500)

    # dsRNA: Otsu globale semplice + rimozione oggetti piccoli
    # Otsu funziona bene perche il canale dsRNA ha background basso e cluster brillanti
    dsrna_blurred = filters.gaussian(dsrna, sigma=2)
    dsrna_thresh  = filters.threshold_otsu(dsrna_blurred)
    dsrna_global  = (dsrna_blurred > dsrna_thresh) & cyto_mask
    dsrna_global  = morphology.remove_small_holes(dsrna_global, area_threshold=500)
    dsrna_global  = morphology.remove_small_objects(dsrna_global, min_size=DSRNA_MIN_AREA)

    results      = []
    er_masks     = {}
    dsrna_masks  = {}
    infected_ids = []

    for cell_id in np.unique(cell_labels):
        if cell_id == 0:
            continue
        cell_mask = cell_labels == cell_id

        # maschera ER e dsRNA per questa cellula
        er_cell   = er_global   & cell_mask
        dsrna_cell = dsrna_global & cell_mask

        # skip se non c e dsRNA → cellula mock
        if dsrna_cell.sum() < DSRNA_MIN_AREA:
            print(f"    cell {cell_id}: no dsRNA → mock, skip")
            continue

        # skip se non c e ER
        if er_cell.sum() == 0:
            print(f"    cell {cell_id}: ER non trovato, skip")
            continue

        # escludi la zona dsRNA dall ER per evitare sovrapposizione
        er_only = er_cell & ~dsrna_cell
        # dilata l ER per includere zona periferica debole
        if ER_DILATION > 0:
            er_only = morphology.dilation(er_only, morphology.disk(ER_DILATION))
            er_only = er_only & cell_mask & ~dsrna_cell & ~nucleus_mask  # dentro cellula, fuori dsRNA e nucleo

        if er_only.sum() == 0:
            print(f"    cell {cell_id}: ER vuoto dopo dilatazione, skip")
            continue

        # misura intensita medie
        RAHP_er    = RAHP[er_only].mean()
        RAHP_dsrna = RAHP[dsrna_cell].mean()
        pdi_er       = pdi[er_only].mean()
        pdi_dsrna    = pdi[dsrna_cell].mean()

        ratio_RAHP = RAHP_dsrna / (RAHP_er + 1e-9)
        ratio_pdi    = pdi_dsrna    / (pdi_er    + 1e-9)

        print(f"    cell {cell_id}: RAHP(dsRNA/ER)={ratio_RAHP:.2f} | PDI(dsRNA/ER)={ratio_pdi:.2f}")

        results.append({
            "image":          img_path.name,
            "cell_id":        cell_id,
            "ratio_RAHP":   ratio_RAHP,
            "ratio_PDI":      ratio_pdi,
            "RAHP_in_er":   RAHP_er,
            "RAHP_in_dsrna":RAHP_dsrna,
            "pdi_in_er":      pdi_er,
            "pdi_in_dsrna":   pdi_dsrna,
            "er_area_px":     er_only.sum(),
            "dsrna_area_px":  dsrna_cell.sum(),
        })

        er_masks[cell_id]    = er_only
        dsrna_masks[cell_id] = dsrna_cell
        infected_ids.append(cell_id)

    save_qc(img_path, img, nucleus_labels, cell_labels,
            er_masks, dsrna_masks, infected_ids, results,
            er_global, dsrna_global)
    return results

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────

def plot_results(df, plot_path=Path("plot.pdf")):
    from scipy import stats

    fig, ax = plt.subplots(figsize=(7, 6))
    proteins = ["RAHP", "PDI"]
    colors   = {"RAHP": "#7F77DD", "PDI": "#1D9E75"}

    for pos, protein in enumerate(proteins, 1):
        col  = f"ratio_{protein}"
        vals = df[col].dropna().values

        parts = ax.violinplot([vals], positions=[pos],
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(colors[protein])
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2.5)

        rng    = np.random.default_rng(pos)
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(pos + jitter, vals, color=colors[protein], s=25,
                   alpha=0.7, edgecolors="white", linewidths=0.4, zorder=3)

        med = np.median(vals)
        ax.text(pos, 0.02, f"n={len(vals)}\nmed={med:.2f}",
                ha="center", va="bottom", fontsize=9,
                color=colors[protein], transform=ax.get_xaxis_transform())

    # p-value
    v1 = df["ratio_RAHP"].dropna().values
    v2 = df["ratio_PDI"].dropna().values
    if len(v1) > 1 and len(v2) > 1:
        _, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
        ymax = max(v1.max(), v2.max())
        h    = ymax * 0.04
        ax.plot([1, 1, 2, 2], [ymax+h, ymax+2*h, ymax+2*h, ymax+h], lw=1, color="black")
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(1.5, ymax+2.5*h, f"{stars}  p={p:.2e}",
                ha="center", va="bottom", fontsize=10)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["RAHP", "PDI"], fontsize=12)
    ax.axhline(1, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, label="ratio=1")
    ax.set_ylabel("Intensità media dsRNA / Intensità media ER", fontsize=10)
    ax.set_title("Concentrazione nell'ER vs dsRNA\n(solo cellule infette)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"→ {plot_path} salvato")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    files = sorted(IMAGE_DIR.glob("*.tif")) + sorted(IMAGE_DIR.glob("*.tiff"))
    print(f"Trovate {len(files)} immagini in '{IMAGE_DIR}'")
    if not files:
        raise FileNotFoundError(f"Nessun .tif trovato in {IMAGE_DIR.resolve()}")

    all_results = []
    for f in files:
        print(f"\n  Analizzando {f.name}...")
        r = analyze_image(f)
        all_results.extend(r)
        print(f"  → {len(r)} cellule infette")

    if not all_results:
        print("\nNessuna cellula infetta! Prova ad abbassare DSRNA_MIN_AREA.")
        return

    df = pd.DataFrame(all_results)
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / "results.csv"
    plot_path = FIGURE_DIR / "Fig4D_colocalization_plot.pdf"
    df.to_csv(csv_path, index=False)
    print(f"\nTotale cellule infette: {len(df)}")
    print(f"Mediana ratio RAHP (dsRNA/ER): {df.ratio_RAHP.median():.3f}")
    print(f"Mediana ratio PDI    (dsRNA/ER): {df.ratio_PDI.median():.3f}")
    print(f"→ {csv_path} salvato")
    plot_results(df, plot_path)

if __name__ == "__main__":
    main()
