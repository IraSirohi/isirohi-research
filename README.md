A Compact RNA-Seq Cancer Classifier and a Selection-Stability Analysis
Code and analysis outputs for "High Classification Accuracy Does Not Guarantee a Stable Gene Signature: A Selection-Stability Analysis of a Compact RNA-Seq Cancer Classifier."
A 75-gene RNA-seq panel separates five tumor types (breast, kidney clear-cell, colon, lung, prostate) at essentially whole-transcriptome accuracy and transfers to an independent CPTAC cohort under batch-level normalization. Its membership, however, is largely not reproducible: across 60 resamples of the same development data, only 14 of the 75 genes were reselected in at least 80% of runs, and 381 distinct genes entered the panel at least once. High classification accuracy does not imply a stable gene signature.
Headline results
Quantity	Value
Held-out test accuracy (n = 586)	99.32% random forest (4 errors), 99.66% linear SVM (2 errors)
Whole-transcriptome benchmark (20,289 genes)	99.83% ± 0.30% cross-validated
External CPTAC (440 tumors, pooled-batch normalization)	97.73% random forest, 98.18% linear SVM
Stable core at k = 75 (≥ 80% of resamples)	14 of 75 genes
Distinct genes entering the top 75 across 60 resamples	381
Mean pairwise Jaccard between resampled panels	0.319 (Kuncheva index 0.480)
Median correlation between a dropped gene and its replacement	0.058
Transfer to CPTAC depends entirely on normalization. Supplied with raw values the frozen model reaches 32.05% (random forest) and 40.23% (SVM); a within-sample rank transform gives 65.91% and 81.14%. The pooled-batch figures above require the target batch in hand, which is possible for a set of specimens and impossible for a single prospective sample.
Scripts
Run `run_analysis.py` first — the others depend on its outputs. All scripts download TCGA from UCSC Xena themselves and write to `outputs/`.
Script	What it does	Writes
`run_analysis.py`	Nested cross-validation (10 outer × 5 inner) with the ranking refitted inside every fold, one-standard-error panel-size rule, final panel, single held-out evaluation	`results.json`, `final_panel.csv`, `per_type_expression.csv`, `nested_cv_folds.csv`, `models.joblib`
`fig4_and_benchmarks.py`	Leakage-free accuracy-versus-panel-size curve, paired Wilcoxon fold differences, whole-transcriptome benchmark, class-balanced sensitivity	`fig4_curve.csv`, `fig4_paired.csv`, `fig4.png`, `benchmarks.json`
`panel_stability_v2.py`	Selection stability across 60 resamples at each `--panel-sizes`; Jaccard and Kuncheva; `--correlate` adds the substitution analysis; `--importance permutation` is the sensitivity run	`stability_k{K}.csv`, `substitutions_k{K}.csv`, `stability_summary.json`
`preprocessing_and_errors.py`	Gene intersection and zero-variance counts; held-out misclassification breakdown by direction	`preprocessing_and_errors.json`
`external_validation.py`	Applies the frozen classifiers to CPTAC under three normalization schemes	`external_validation.json`, `cptac_confusion_{model}.csv`
`collect_for_manuscript.py`	Bundles every output into one file and reports what is missing	`MANUSCRIPT_VALUES.json`
Reproduce
```
pip install -r requirements.txt

python run_analysis.py --final-k 75
python fig4_and_benchmarks.py --reference-k 75
python panel_stability_v2.py --panel-sizes 75 20 --correlate
python panel_stability_v2.py --importance permutation --panel-sizes 20 --n-splits 20 --out outputs/perm
python preprocessing_and_errors.py
python external_validation.py --expression cptac.csv --labels cptac_labels.csv
python collect_for_manuscript.py
```
Expect hours, not minutes. The whole-transcriptome benchmark fits a linear SVM on all 20,289 genes ten times, and the stability analysis fits 120 random forests on the full matrix.
The results above have been verified end to end: the panel size, the panel membership (identical held-out errors), both benchmarks, the stability metrics at both panel sizes, and the preprocessing counts all reproduce the published values.
A note on the pipeline
The panel-size curve must be generated with the gene ranking refitted inside each cross-validation fold. Ranking once on the full development set and then cross-validating on folds of that same set lets the ranking see every validation fold, which inflates the apparent accuracy of small panels — most severely at small k, where the ranking matters most. An earlier version of this analysis did exactly that. Correcting it is why the reported panel is 75 genes rather than 20, and the header of `fig4_and_benchmarks.py` documents the original error in full.
`run_analysis.py` is a reconstruction. The original was lost, and this version was rebuilt from the published Methods to match the conventions of the other scripts. It reproduces the published panel exactly — the same 75 genes, the same four held-out misclassifications — and the same non-convergence of the inner loop across outer folds. One difference is worth stating: applied to its own outer-fold accuracies, the one-standard-error rule selects 50 rather than 75, because 50 and 75 sit within hundredths of a percentage point of the threshold. Applied to the `fig4_and_benchmarks.py` curve, which is original code, the rule selects 75 as published. The reported panel size follows the latter.
Data sources (cited, not redistributed)
TCGA RNA-seq (training and internal validation): UCSC Xena — https://xenabrowser.net
CPTAC RNA-seq (external, 4 tumor types): LinkedOmicsKB — https://kb.linkedomics.org
Prostate (cross-platform, earlier version of the study): GEO accession GSE54460
Raw expression data are not included here. Preprocessing retains primary solid tumors only (sample-type code 01), giving 2,928 tumors across five cohorts, and removes 241 zero-variance genes from 20,530 to leave 20,289. The development/test split is 80/20 stratified with seed 42 (2,342 development, 586 held out).
Data files
File	Contents
`final_panel.csv`	The 75 selected genes, ranked by random-forest Gini importance
`panel_stability.csv`	Per-gene selection frequency across the 60 resamples
`Supplementary_Table_S1.csv`	All 381 genes entering the top 75, with selection frequencies
`Supplementary_Table_S2.csv`	The reported 75-gene panel with selection frequencies
Stability at two panel sizes
Panel size	Core genes (≥ 80%)	Distinct genes entering top k	Mean pairwise Jaccard	Kuncheva index
20	3	149	0.215	0.349
75	14	381	0.319	0.480
Under permutation importance at k = 20, no gene reaches the 80% threshold, 305 distinct genes enter the panel, and mean Jaccard falls to 0.053 (Kuncheva 0.087).
Citation
If you use this code, please cite the article (citation to be added on publication).
License
MIT License — see `LICENSE`.
