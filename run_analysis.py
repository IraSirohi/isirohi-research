#!/usr/bin/env python3
"""
run_analysis.py — main pipeline: nested cross-validation, panel selection,
held-out evaluation.

RECONSTRUCTED. The original file was lost. This version was rebuilt from
Methods 2.3-2.10 of the published manuscript and follows the conventions of
fig4_and_benchmarks.py and panel_stability_v2.py (same loader, same
LabelEncoder, same numpy layout), so the scripts interoperate.

The earlier version of this file contained the feature-selection leak described
in the header of fig4_and_benchmarks.py: it ranked genes once on the whole
development set and then cross-validated on folds of that same set. That is not
reproduced here. Every ranking in this file is fitted on training data only.

Run first. The other scripts depend on its outputs.

Writes (to ./outputs/):
    results.json            nested-CV choices, one-SE rule, held-out metrics
    final_panel.csv         the k=75 panel, ranked by Gini importance
    per_type_expression.csv standardized mean expression per tumor type
    nested_cv_folds.csv     per-fold accuracy at every candidate size
    models.joblib           frozen classifiers, for external_validation.py

Usage:  python run_analysis.py [--final-k 75] [--outer 10] [--inner 5]

Requires: pandas numpy scikit-learn joblib requests
"""
import argparse, io, gzip, json, os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             recall_score)

RS = 42
OUT = "outputs"

PANEL_SIZES = [2, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200]

COHORTS = {
    "BRCA": "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz",
    "KIRC": "https://tcga.xenahubs.net/download/TCGA.KIRC.sampleMap/HiSeqV2.gz",
    "LUAD": "https://tcga.xenahubs.net/download/TCGA.LUAD.sampleMap/HiSeqV2.gz",
    "PRAD": "https://tcga.xenahubs.net/download/TCGA.PRAD.sampleMap/HiSeqV2.gz",
    "COAD": "https://tcga.xenahubs.net/download/TCGA.COAD.sampleMap/HiSeqV2.gz",
}


def load_xena():
    """Identical to the loader in the other scripts, so sample sets match."""
    import requests
    frames = []
    for lab, url in COHORTS.items():
        print(f"  [load] {lab} ...")
        txt = gzip.decompress(requests.get(url, timeout=600).content).decode("utf-8")
        df = pd.read_csv(io.StringIO(txt), sep="\t", index_col=0).T
        df = df.loc[[s for s in df.index if str(s)[13:15] == "01"]]
        df.columns = [str(c).upper() for c in df.columns]
        df["__y__"] = lab
        frames.append(df)
    shared = set(frames[0].columns)
    for f in frames[1:]:
        shared &= set(f.columns)
    shared.discard("__y__")
    shared = sorted(shared)
    X = pd.concat([f[shared] for f in frames], axis=0).astype(float).fillna(0.0)
    y = pd.concat([f["__y__"] for f in frames], axis=0)
    X = X.loc[:, X.var(0) > 0]
    return X, y


def svm_pipe():
    return Pipeline(
        [("sc", StandardScaler()), ("svm", SVC(kernel="linear", C=1.0, random_state=RS))]
    )


def rank_order(Xv, yv, n_estimators=300, seed=RS):
    """Gene column indices by decreasing Gini importance. TRAINING DATA ONLY.

    Fitting this on a full development set and then cross-validating on folds
    of that same set is the leak the revision exists to correct. Every call in
    this file passes a training subset.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1
    ).fit(Xv, yv)
    return np.argsort(rf.feature_importances_)[::-1]


# --------------------------------------------------------------------------
# nested cross-validation (Methods 2.6)
# --------------------------------------------------------------------------
def inner_select(Xtr, ytr, sizes, n_inner, n_estimators, seed):
    """Inner loop: the panel size with the highest mean inner-fold accuracy."""
    cv = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
    acc = np.zeros((n_inner, len(sizes)))
    for fi, (tr, va) in enumerate(cv.split(Xtr, ytr)):
        order = rank_order(Xtr[tr], ytr[tr], n_estimators, seed)
        for ki, k in enumerate(sizes):
            cols = order[:k]
            clf = svm_pipe().fit(Xtr[tr][:, cols], ytr[tr])
            acc[fi, ki] = accuracy_score(ytr[va], clf.predict(Xtr[va][:, cols]))
    return sizes[int(acc.mean(0).argmax())]


def nested_cv(Xdev, ydev, sizes, n_outer, n_inner, n_estimators):
    """Outer loop. Returns outer-fold accuracies and per-fold size choices."""
    cv = StratifiedKFold(n_outer, shuffle=True, random_state=RS)
    outer_acc = np.zeros((n_outer, len(sizes)))
    choices = []

    for fi, (tr, va) in enumerate(cv.split(Xdev, ydev)):
        order = rank_order(Xdev[tr], ydev[tr], n_estimators, RS + fi)
        for ki, k in enumerate(sizes):
            cols = order[:k]
            clf = svm_pipe().fit(Xdev[tr][:, cols], ydev[tr])
            outer_acc[fi, ki] = accuracy_score(
                ydev[va], clf.predict(Xdev[va][:, cols])
            )
        chosen = inner_select(Xdev[tr], ydev[tr], sizes, n_inner,
                              n_estimators, RS + fi)
        choices.append(int(chosen))
        print(f"    outer fold {fi + 1}/{n_outer}: inner loop chose k={chosen}")

    return outer_acc, choices


def one_standard_error_rule(outer_acc, sizes):
    """Smallest size within one SE of the best-performing size (Methods 2.6).

    Applied to the outer folds, without reference to the held-out test set.
    """
    v = outer_acc * 100
    means = v.mean(0)
    sems = v.std(0, ddof=1) / np.sqrt(v.shape[0])

    best = int(means.argmax())
    threshold = means[best] - sems[best]
    eligible = [k for ki, k in enumerate(sizes) if means[ki] >= threshold]

    return {
        "best_size": sizes[best],
        "best_mean_accuracy": round(float(means[best]), 3),
        "best_sem": round(float(sems[best]), 3),
        "threshold": round(float(threshold), 3),
        "selected_size": int(min(eligible)),
        "mean_by_size": {str(k): round(float(means[ki]), 3)
                         for ki, k in enumerate(sizes)},
    }


def per_type_expression(X, y, genes):
    """Standardized mean expression of each panel gene by tumor type (Fig 4)."""
    sub = X[genes]
    z = (sub - sub.mean(0)) / sub.std(0).replace(0, 1)
    return z.groupby(y.values).mean().T


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-k", type=int, default=75)
    ap.add_argument("--sizes", type=int, nargs="+", default=PANEL_SIZES)
    ap.add_argument("--outer", type=int, default=10)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--skip-nested", action="store_true",
                    help="skip the nested loop (slow) and use --final-k directly")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sizes = sorted(set(args.sizes + [args.final_k]))

    X, y = load_xena()
    genes = np.array(X.columns)
    le = LabelEncoder()
    yi = le.fit_transform(y)
    classes = list(le.classes_)
    print(f"Matrix: {X.shape[0]} tumors x {X.shape[1]} genes")

    dev_idx, test_idx = next(
        StratifiedShuffleSplit(1, test_size=0.20, random_state=RS).split(X.values, yi)
    )
    Xdev, ydev = X.values[dev_idx], yi[dev_idx]
    Xte, yte = X.values[test_idx], yi[test_idx]
    print(f"Development {len(ydev)}, held-out test {len(yte)}")

    results = {
        "config": {
            "seed": RS,
            "n_estimators_ranking": args.n_estimators,
            "n_estimators_classifier": 500,
            "svm_C": 1.0,
            "n_dev": int(len(ydev)),
            "n_test": int(len(yte)),
            "n_genes": int(X.shape[1]),
            "classes": classes,
        }
    }

    if not args.skip_nested:
        print(f"\nNested cross-validation ({args.outer} outer x {args.inner} inner) ...")
        outer_acc, choices = nested_cv(
            Xdev, ydev, sizes, args.outer, args.inner, args.n_estimators
        )
        pd.DataFrame(outer_acc * 100, columns=[str(k) for k in sizes]).to_csv(
            f"{args.out}/nested_cv_folds.csv", index_label="outer_fold"
        )
        rule = one_standard_error_rule(outer_acc, sizes)
        results["nested_cv"] = {
            "per_fold_choices": choices,
            "distinct_choices": sorted(set(choices)),
            "converged": len(set(choices)) == 1,
            "one_standard_error_rule": rule,
        }
        print(f"  per-fold choices: {choices}")
        print(f"  one-SE rule selects k = {rule['selected_size']} "
              f"(best {rule['best_size']} at {rule['best_mean_accuracy']}%, "
              f"threshold {rule['threshold']}%)")

    # ---- final panel: ranked on the development set, evaluated once --------
    k = args.final_k
    order = rank_order(Xdev, ydev, args.n_estimators, RS)
    cols = order[:k]
    panel = list(genes[cols])

    rf_full = RandomForestClassifier(
        n_estimators=args.n_estimators, random_state=RS, n_jobs=-1
    ).fit(Xdev, ydev)
    pd.DataFrame({
        "rank": range(1, k + 1),
        "gene": panel,
        "gini_importance": rf_full.feature_importances_[cols],
    }).to_csv(f"{args.out}/final_panel.csv", index=False)

    rf = RandomForestClassifier(n_estimators=500, random_state=RS, n_jobs=-1)
    rf.fit(Xdev[:, cols], ydev)
    sv = svm_pipe().fit(Xdev[:, cols], ydev)

    print(f"\nHeld-out evaluation (k = {k}) ...")
    heldout = {}
    for name, model in [("random_forest", rf), ("linear_svm", sv)]:
        pred = model.predict(Xte[:, cols])
        heldout[name] = {
            "accuracy": round(100 * accuracy_score(yte, pred), 3),
            "macro_f1": round(100 * f1_score(yte, pred, average="macro"), 3),
            "n_errors": int((pred != yte).sum()),
            "recall_by_class": {
                c: round(100 * r, 3)
                for c, r in zip(classes, recall_score(yte, pred, average=None))
            },
            "confusion_matrix": confusion_matrix(yte, pred).tolist(),
        }
        print(f"  {name}: {heldout[name]['accuracy']}% "
              f"({heldout[name]['n_errors']} errors)")

    results["heldout"] = heldout
    results["final_panel"] = {"k": k, "genes": panel}

    per_type_expression(X, y, panel).to_csv(f"{args.out}/per_type_expression.csv")

    joblib.dump(
        {"rf": rf, "svm": sv, "panel": panel, "classes": classes,
         "cols": cols.tolist()},
        f"{args.out}/models.joblib",
    )

    with open(f"{args.out}/results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {args.out}/results.json, final_panel.csv, "
          f"per_type_expression.csv, models.joblib")


if __name__ == "__main__":
    main()
