# A Compact RNA-Seq Cancer Classifier and a Panel-Stability Analysis

Code and analysis for the study *"High Classification Accuracy Does Not Guarantee a Reproducible
Gene Signature: A Stability Analysis of a Compact RNA-Seq Cancer Classifier."*

A 20-gene RNA-seq classifier separates five tumor types (breast, kidney clear-cell, colon,
lung, prostate) with high accuracy and transfers to four independent CPTAC types — but the
specific gene panel is largely **unstable** across resampling (only 7 of 20 genes are reliably
selected). The central message: high accuracy does not imply a reproducible gene signature.

## What's here

| File | Purpose |
|---|---|
| `run_analysis.py` | Main pipeline: TCGA load (UCSC Xena), nested CV with in-fold gene selection, single held-out test, external CPTAC validation (two normalization schemes), Wilson/bootstrap CIs. Writes `outputs/`. |
| `panel_stability.py` | Re-derives the top-20 panel across 60 resamples and reports per-gene selection frequency + Jaccard overlap. |
| `core_panel_test.py` | Tests whether the stable "core" genes retain accuracy. |
| `score_prostate_gse54460.py` | Standalone cross-platform prostate test (GSE54460) — documents the transfer failure. |
| `make_figures.py` | Builds the manuscript figures from `outputs/`. |
| `make_supplementary.py` | Builds Supplementary Table S1 (all genes + frequencies) from the stability run. |
| `requirements.txt` / `environment.yml` | Dependencies. |
| `final_panel.csv` | The 20 selected genes. |
| `panel_stability.csv` | Per-gene selection frequencies across resamples. |
| `Supplementary_Table_S1.csv` | All 79 genes that entered the top-20 across resamples. |

## Reproduce

```bash
# option A: conda
conda env create -f environment.yml && conda activate rnaseq-panel
# option B: pip
pip install -r requirements.txt
```

Then (the scripts download TCGA from UCSC Xena automatically; place the CPTAC and GSE54460
files alongside the scripts for external validation):

```bash
python run_analysis.py          # -> outputs/ (metrics, panel, frozen models)
python panel_stability.py       # -> outputs/panel_stability.csv
python core_panel_test.py       # core-gene accuracy
python make_figures.py          # -> outputs/figures/
python make_supplementary.py    # -> Supplementary Table S1
python score_prostate_gse54460.py
```

## Data sources (cited, not redistributed)

- **TCGA** RNA-seq (training/internal): UCSC Xena — https://xenabrowser.net
- **CPTAC** RNA-seq (external, 4 types): LinkedOmicsKB — https://kb.linkedomics.org
- **Prostate** (external): GEO accession **GSE54460** — https://www.ncbi.nlm.nih.gov/geo/

Raw expression data are not included here; download them from the sources above.

## Key results

- Held-out test (n = 586): RF 99.32%, SVM 99.49% accuracy; lowest class recall 97% (LUAD/RF).
- External CPTAC (4 types, training-fixed): ~99.6–99.8% accuracy.
- Prostate GSE54460: frozen panel does **not** transfer (cross-platform scale mismatch).
- Panel stability (60 resamples): 7/20 core genes; mean Jaccard 0.44; 79 distinct genes seen.
- Seven-gene core classifier: ~95% held-out accuracy.

## Citation

If you use this code, please cite the article (citation to be added on publication) and this
repository: Zenodo, DOI 10.5281/zenodo.20806355.

## License

MIT License — see the `LICENSE` file.
