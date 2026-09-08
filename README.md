
# A Compact RNA-Seq Cancer Classifier and a Selection-Stability Analysis

Code and analysis for the study *"High Classification Accuracy Does Not Guarantee a Stable Gene Signature: A Selection-Stability Analysis of a Compact RNA-Seq Cancer Classifier."*

A 75-gene RNA-seq panel separates five tumor types (breast, kidney clear-cell, colon, lung, prostate) at essentially whole-transcriptome accuracy and transfers to an independent CPTAC cohort under batch-level normalization. Its membership, however, is largely **not reproducible**: across 60 resamples of the same development data, only 14 of the 75 genes were reselected in at least 80% of runs, and 381 distinct genes entered the panel at least once. The central message is that high classification accuracy does not imply a stable gene signature.

## Headline results

| Quantity | Value |
| --- | --- |
| Held-out test accuracy (n = 586) | 99.32% random forest, 99.66% linear SVM |
| Whole-transcriptome benchmark (20,289 genes) | 99.83% ± 0.30% cross-validated |
| External CPTAC (440 tumors, pooled-batch normalization) | 97.73% random forest, 98.18% linear SVM |
| Stable core at k = 75 (≥ 80% of resamples) | 14 of 75 genes |
| Distinct genes entering the top 75 across 60 resamples | 381 |
| Mean pairwise Jaccard overlap between resampled panels | 0.319 (Kuncheva index 0.480) |
| Median correlation between a dropped gene and its replacement | 0.058 |

Transfer to CPTAC depends entirely on normalization. Supplied with raw values the frozen model reaches only 32.05% (random forest) and 40.23% (SVM); a within-sample rank transform gives 65.91% and 81.14%. The pooled-batch figures above require the target batch in hand, which is possible for a set of specimens and impossible for a single prospective sample.

## What's here

| File | Purpose |
| --- | --- |
| `run_analysis.py` | Main pipeline: TCGA load (UCSC Xena), nested cross-validation (10 outer × 5 inner) with the random-forest ranking **refitted independently inside every fold**, one-standard-error panel-size rule, single held-out test, whole-transcriptome benchmark, Wilson/bootstrap CIs. Writes `outputs/`. |
| `external_validation.py` | Applies the frozen panel to 440 CPTAC tumors under three normalization schemes: raw values, pooled-batch standardization, within-sample rank. |
| `panel_stability.py` | Re-derives the panel across 60 resamples at k = 75 and k = 20; reports per-gene selection frequency, mean pairwise Jaccard, and the chance-corrected Kuncheva index. |
| `substitution_analysis.py` | For each substitution event, records the dropped gene, its replacement, and their Pearson correlation — the evidence that instability reflects rank noise rather than redundancy. |
| `permutation_sensitivity.py` | Repeats the stability analysis using permutation importance in place of Gini importance (30 resamples, k = 20, top 500 genes permuted). |
| `class_balance_sensitivity.py` | Repeats the accuracy-versus-panel-size analysis with all five cohorts downsampled to n = 286. |
| `score_prostate_gse54460.py` | Standalone cross-platform prostate test (GSE54460), retained from an earlier version of the study; documents the transfer failure that motivated the normalization comparison. Not carried forward to the 75-gene panel. |
| `make_figures.py` | Builds Figures 1–5 from `outputs/`. |
| `make_supplementary.py` | Builds Supplementary Tables S1 and S2 from the stability run. |
| `final_panel.csv` | The 75 selected genes, ranked by random-forest importance. |
| `panel_stability.csv` | Per-gene selection frequency across the 60 resamples. |
| `wilcoxon_pvalues.csv` | Complete pairwise p-value matrix for the panel-size comparisons (paired Wilcoxon signed-rank across the 10 outer folds), referenced in Table 2 of the manuscript. |
| `Supplementary_Table_S1.csv` | All 381 genes that entered the top 75 across resamples, with selection frequencies. |
| `Supplementary_Table_S2.csv` | The reported 75-gene panel with selection frequencies. |
| `requirements.txt` / `environment.yml` | Dependencies. |

## Reproduce

```
# option A: conda
conda env create -f environment.yml && conda activate rnaseq-panel
# option B: pip
pip install -r requirements.txt
```

The scripts download TCGA from UCSC Xena automatically. Place the CPTAC and GSE54460 files alongside the scripts for the external analyses.

```
python run_analysis.py                  # -> outputs/ (metrics, panel, frozen models)
python external_validation.py           # -> CPTAC results under three normalizations
python panel_stability.py               # -> outputs/panel_stability.csv
python substitution_analysis.py         # -> substitution events and correlations
python permutation_sensitivity.py       # -> permutation-importance comparison
python class_balance_sensitivity.py     # -> balanced-cohort comparison
python make_figures.py                  # -> outputs/figures/
python make_supplementary.py            # -> Supplementary Tables S1 and S2
python score_prostate_gse54460.py
```

A note on the pipeline. The panel-size curve must be generated with the gene ranking refitted inside each cross-validation fold. Ranking once on the full development set and then cross-validating on folds of that same set lets the ranking see every validation fold, which inflates the apparent accuracy of small panels. An earlier version of this analysis did exactly that; correcting it is why the reported panel is 75 genes rather than 20.

## Data sources (cited, not redistributed)

- **TCGA** RNA-seq (training/internal): UCSC Xena — <https://xenabrowser.net>
- **CPTAC** RNA-seq (external, 4 types): LinkedOmicsKB — <https://kb.linkedomics.org>
- **Prostate** (cross-platform, earlier version): GEO accession **GSE54460** — <https://www.ncbi.nlm.nih.gov/geo/>

Raw expression data are not included here; download them from the sources above. Preprocessing retains primary solid tumors only (sample-type code 01), giving 2,928 tumors across five cohorts, and removes 241 zero-variance genes to leave 20,289. The development/test split is 80/20 stratified with seed 42 (2,342 development, 586 test).

## Stability at two panel sizes

| Panel size | Core genes (≥ 80%) | Distinct genes entering top *k* | Mean pairwise Jaccard | Kuncheva index |
| --- | --- | --- | --- | --- |
| 20 | 3 | 149 | 0.215 | 0.349 |
| 75 | 14 | 381 | 0.319 | 0.480 |

Under permutation importance at k = 20, no gene reaches the 80% threshold, 305 distinct genes enter the panel, and mean Jaccard falls to 0.053 (Kuncheva 0.087).

## Citation

If you use this code, please cite the article (citation to be added on publication) and this repository (Zenodo DOI to be added).

## License

MIT License — see the `LICENSE` file.
