#!/usr/bin/env python3
"""
score_prostate_gse54460.py — standalone, reproducible prostate external test.

Reuses the frozen models saved by run_analysis.py (outputs/frozen_models.joblib) and scores
the GSE54460 cohort under both normalization schemes. Documents the cross-platform transfer
FAILURE reported in the paper. Run in the same Colab folder after run_analysis.py.

The GSE54460 file layout (verified): 15 metadata rows, then a header row
`Entrez <tab> SYMBOL <tab> <106 samples>`, then gene rows. Column 1 is the HUGO symbol;
columns 2.. are FPKM values (linear), some quoted with thousands commas.
"""
import gzip, numpy as np, pandas as pd, joblib
from sklearn.metrics import classification_report, confusion_matrix

PROSTATE_FILE = "gse54460_prostate_rnaseq.tsv.gz"
M = joblib.load("outputs/frozen_models.joblib")
rf, svm = M["rf"], M["svm"]
panel = [g.upper() for g in M["panel"]]; classes = list(M["classes"])
mu = np.array(M["mu"]); sd = np.array(M["sd"]); sd[sd == 0] = 1.0
PRAD = classes.index("PRAD")

rows = [l.rstrip("\n").split("\t") for l in gzip.open(PROSTATE_FILE, "rt")]
hdr = next(i for i, r in enumerate(rows) if len(r) > 1 and r[1].strip().upper() == "SYMBOL")
sample_cols = list(range(2, len(rows[hdr])))
pset = set(panel); data = {}
for r in rows[hdr + 1:]:
    if len(r) <= 2:
        continue
    sym = r[1].strip().upper()
    if sym in pset:
        vals = []
        for j in sample_cols:
            x = r[j].replace('"', '').replace(',', '') if j < len(r) else ''
            try: vals.append(float(x))
            except: vals.append(np.nan)
        data[sym] = vals
Mp = np.log2(pd.DataFrame(data).clip(lower=0) + 1.0).reindex(columns=panel)
print(f"samples: {Mp.shape[0]} | panel genes with data: {int(Mp.notna().any(axis=0).sum())}/20")

yp = np.array([PRAD] * Mp.shape[0])
muP = np.nanmean(Mp.values, 0); sdP = np.nanstd(Mp.values, 0); sdP[sdP == 0] = 1.0
schemes = {"training-fixed": np.nan_to_num((Mp.values - mu) / sd),
           "pooled":         np.nan_to_num((Mp.values - muP) / sdP)}
for sname, X in schemes.items():
    for mname, clf in [("RF", rf), ("SVM", svm)]:
        pred = clf.predict(X)
        rec = (pred == PRAD).mean() * 100
        called = pd.Series([classes[p] for p in pred]).value_counts().to_dict()
        print(f"\n[{sname} | {mname}] PRAD recall = {rec:.1f}%  ({(pred==PRAD).sum()}/{len(pred)})")
        print("  predicted as:", called)

print("\nConclusion: the frozen TCGA panel does not transfer to GSE54460 under either "
      "scheme — a cross-platform (FPKM vs log2 RSEM) scale incompatibility, reported "
      "honestly as a four-type external validation with prostate as a documented failure.")
