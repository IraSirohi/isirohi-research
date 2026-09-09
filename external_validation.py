#!/usr/bin/env python3
"""
external_validation.py — apply the frozen classifiers to the CPTAC cohort
without retraining, under three normalization schemes.

RECONSTRUCTED alongside run_analysis.py. Implements Methods 2.7.

The three schemes are reported together so that the sensitivity of transfer to
this choice is visible rather than hidden behind a single number:

    raw            values supplied directly, no alignment
    pooled_batch   each gene standardized once across all 440 samples together,
                   cohort labels never used. This is the primary result and
                   reflects the prospectively realistic setting in which an
                   unlabeled batch is normalized against itself.
    rank           each sample's panel genes converted to percentile ranks,
                   requiring no batch statistics at all

A per-cohort scheme is deliberately not offered. Each CPTAC cohort contains a
single class, so centring within cohorts removes the between-class differences
the classifier depends on.

Run AFTER run_analysis.py (needs outputs/models.joblib).

Writes:
    external_validation.json     accuracy and per-class recall by scheme
    cptac_confusion_{model}.csv  confusion matrices under pooled_batch

Usage:
    python external_validation.py --expression cptac.csv --labels labels.csv

  --expression  samples x genes matrix, HUGO symbols as column headers
  --labels      two columns: sample id, cohort label matching the TCGA classes

Requires: pandas numpy scikit-learn joblib
"""
import argparse, json, os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

OUT = "outputs"


def normalize(df, scheme):
    if scheme == "raw":
        return df
    if scheme == "pooled_batch":
        return (df - df.mean(0)) / df.std(0).replace(0, 1)
    if scheme == "rank":
        return df.rank(axis=1, pct=True)
    raise ValueError(scheme)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expression", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bundle = joblib.load(f"{args.out}/models.joblib")
    panel, classes = bundle["panel"], bundle["classes"]

    X = pd.read_csv(args.expression, index_col=0)
    X.columns = [str(c).upper() for c in X.columns]
    y = pd.read_csv(args.labels, index_col=0).squeeze("columns").loc[X.index]

    missing = [g for g in panel if g not in X.columns]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(panel)} panel genes absent from the "
            f"external matrix: {missing[:10]}"
        )
    X = X[panel]
    print(f"  {len(X)} tumors, all {len(panel)} panel genes recovered")
    print(f"  cohorts present: {sorted(y.unique())}")

    # Classes absent from the external cohort (CPTAC has no prostate cohort)
    # are simply never predicted correctly; recall is reported per class.
    absent = [c for c in classes if c not in set(y)]
    if absent:
        print(f"  no external cohort for: {absent}")

    out = {"n_samples": int(len(X)), "absent_classes": absent, "schemes": {}}
    for scheme in ("raw", "pooled_batch", "rank"):
        Z = normalize(X, scheme)
        row = {}
        for name in ("rf", "svm"):
            pred = bundle[name].predict(Z.values)
            pred = np.array([classes[i] for i in pred])
            row[name] = {
                "accuracy": round(100 * accuracy_score(y, pred), 3),
                "recall_by_class": {
                    c: round(100 * r, 3) for c, r in zip(
                        sorted(y.unique()),
                        recall_score(y, pred, average=None,
                                     labels=sorted(y.unique())))
                },
            }
            if scheme == "pooled_batch":
                labels = sorted(y.unique())
                pd.DataFrame(confusion_matrix(y, pred, labels=labels),
                             index=labels, columns=labels).to_csv(
                    f"{args.out}/cptac_confusion_{name}.csv")
        out["schemes"][scheme] = row
        print(f"  {scheme:<13} RF {row['rf']['accuracy']:6.2f}%   "
              f"SVM {row['svm']['accuracy']:6.2f}%")

    with open(f"{args.out}/external_validation.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {args.out}/external_validation.json")


if __name__ == "__main__":
    main()
