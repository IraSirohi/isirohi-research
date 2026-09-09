#!/usr/bin/env python3
"""
fig4_and_benchmarks.py — recomputes Figure 4 without leakage and adds the two
benchmarks the reviewer asked for.

Why this file exists. In run_analysis.py lines 138-144 the panel-size curve was
built like this:

    order_dev = rank_genes(Xdev.values, ydev)          # ranks on ALL of dev
    for k in sizes:
        cols = order_dev[:k]
        curve[k] = cross_val_score(svm_pipe(), Xdev.values[:, cols], ydev, cv=cv)

The ranking was fitted on the whole development set and then scored by
cross-validation on folds of that same set, so every validation fold had already
been seen by the gene ranker. That is feature-selection leakage and it inflates
the curve, most severely at small k where the ranking matters most. This is
exactly what must-fix 3 asks about, and the answer is yes, it leaked.

What this script produces (to ./outputs/):
  fig4_curve.csv        leakage-free accuracy vs panel size, mean + 95% CI,
                        ranking recomputed independently inside every fold
  fig4_paired.csv       paired per-fold differences between each k and the
                        reference k, with a Wilcoxon p-value, replacing the
                        comparison against a pooled standard deviation
  fig4.png              the figure
  benchmarks.json       whole-transcriptome (all 20,289 genes) accuracy, and
                        class-balanced sensitivity accuracy

  [must-fix 3]  leakage-free curve; extended size grid; per-size CIs; paired
                fold differences instead of a pooled SD comparison
  [must-fix 3]  whole-transcriptome benchmark, which the manuscript compares
                against but never actually fitted
  [must-fix 5]  class-balanced run (all cohorts downsampled to n_min) to test
                whether performance depends on the cohort size imbalance

Requires: pandas numpy scipy scikit-learn matplotlib requests
"""
import argparse, io, gzip, json, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

RS = 42
OUT = "outputs"

COHORTS = {
    "BRCA": "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz",
    "KIRC": "https://tcga.xenahubs.net/download/TCGA.KIRC.sampleMap/HiSeqV2.gz",
    "LUAD": "https://tcga.xenahubs.net/download/TCGA.LUAD.sampleMap/HiSeqV2.gz",
    "PRAD": "https://tcga.xenahubs.net/download/TCGA.PRAD.sampleMap/HiSeqV2.gz",
    "COAD": "https://tcga.xenahubs.net/download/TCGA.COAD.sampleMap/HiSeqV2.gz",
}


def load_xena():
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


def rank_order(Xv, yv, n_estimators, seed=RS):
    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1
    ).fit(Xv, yv)
    return np.argsort(rf.feature_importances_)[::-1]


def curve_no_leakage(Xdev, ydev, sizes, n_folds, n_estimators):
    """Per-fold accuracy at each k, ranking refitted inside each training fold.

    Returns array (n_folds, len(sizes)). The ranking is computed ONCE per fold
    on that fold's training half and then truncated to each k, so folds are
    comparable across k while no validation sample ever informs the ranking.
    """
    cv = StratifiedKFold(n_folds, shuffle=True, random_state=RS)
    acc = np.zeros((n_folds, len(sizes)))
    for fi, (tr, va) in enumerate(cv.split(Xdev, ydev)):
        order = rank_order(Xdev[tr], ydev[tr], n_estimators)
        for ki, k in enumerate(sizes):
            cols = order[:k]
            clf = svm_pipe().fit(Xdev[tr][:, cols], ydev[tr])
            acc[fi, ki] = accuracy_score(ydev[va], clf.predict(Xdev[va][:, cols]))
        print(f"    fold {fi + 1}/{n_folds} done")
    return acc


def summarise_curve(acc, sizes, reference_k):
    n = acc.shape[0]
    rows = []
    for ki, k in enumerate(sizes):
        v = acc[:, ki] * 100
        sem = v.std(ddof=1) / np.sqrt(n)
        rows.append(
            {
                "panel_size": k,
                "mean_accuracy": round(v.mean(), 3),
                "sd": round(v.std(ddof=1), 3),
                "ci_low": round(v.mean() - 1.96 * sem, 3),
                "ci_high": round(v.mean() + 1.96 * sem, 3),
            }
        )
    curve = pd.DataFrame(rows)

    ref = acc[:, sizes.index(reference_k)]
    prs = []
    for ki, k in enumerate(sizes):
        if k == reference_k:
            continue
        d = (acc[:, ki] - ref) * 100
        try:
            p = wilcoxon(acc[:, ki], ref).pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = 1.0
        sem = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        prs.append(
            {
                "panel_size": k,
                "reference_k": reference_k,
                "mean_paired_diff": round(d.mean(), 3),
                "ci_low": round(d.mean() - 1.96 * sem, 3),
                "ci_high": round(d.mean() + 1.96 * sem, 3),
                "wilcoxon_p": round(float(p), 4),
            }
        )
    return curve, pd.DataFrame(prs)


def plot_curve(curve, reference_k, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve.panel_size, curve.mean_accuracy, marker="o", lw=1.5, color="#1f77b4")
    ax.fill_between(
        curve.panel_size, curve.ci_low, curve.ci_high, alpha=0.2, color="#1f77b4"
    )
    ax.axvline(reference_k, ls="--", color="grey", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(curve.panel_size)
    ax.set_xticklabels(curve.panel_size)
    ax.set_xlabel("Panel size (number of genes)")
    ax.set_ylabel("Cross-validated accuracy (%)")
    ax.set_title("Accuracy vs panel size (ranking refitted within each fold)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"  wrote {path}")


def whole_transcriptome(Xdev, ydev, n_folds):
    """The benchmark the manuscript compares against but never fitted."""
    cv = StratifiedKFold(n_folds, shuffle=True, random_state=RS)
    accs = []
    for tr, va in cv.split(Xdev, ydev):
        clf = svm_pipe().fit(Xdev[tr], ydev[tr])
        accs.append(accuracy_score(ydev[va], clf.predict(Xdev[va])))
    a = np.array(accs) * 100
    return {
        "n_genes": int(Xdev.shape[1]),
        "mean_accuracy": round(a.mean(), 3),
        "sd": round(a.std(ddof=1), 3),
        "per_fold": [round(x, 3) for x in a],
    }


def balanced_sensitivity(X, yi, classes, sizes, n_folds, n_estimators, seed=RS):
    """Downsample every class to the smallest class and repeat the core result."""
    rng = np.random.default_rng(seed)
    n_min = min(int((yi == i).sum()) for i in range(len(classes)))
    keep = np.concatenate(
        [rng.choice(np.where(yi == i)[0], n_min, replace=False) for i in range(len(classes))]
    )
    Xb, yb = X[keep], yi[keep]
    acc = curve_no_leakage(Xb, yb, sizes, n_folds, n_estimators)
    return {
        "n_per_class": int(n_min),
        "n_total": int(len(yb)),
        "accuracy_by_size": {
            str(k): round(float(acc[:, ki].mean() * 100), 3) for ki, k in enumerate(sizes)
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes", type=int, nargs="+",
        default=[2, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200],
    )
    ap.add_argument("--reference-k", type=int, default=75)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--skip-whole", action="store_true")
    ap.add_argument("--skip-balanced", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.reference_k not in args.sizes:
        args.sizes = sorted(set(args.sizes + [args.reference_k]))

    X, y = load_xena()
    le = LabelEncoder()
    yi = le.fit_transform(y)
    classes = list(le.classes_)

    dev_idx, _ = next(
        StratifiedShuffleSplit(1, test_size=0.20, random_state=RS).split(X.values, yi)
    )
    Xdev, ydev = X.values[dev_idx], yi[dev_idx]
    print(f"Development set: {Xdev.shape[0]} x {Xdev.shape[1]}")

    print("Panel-size curve (leakage-free) ...")
    acc = curve_no_leakage(Xdev, ydev, args.sizes, args.folds, args.n_estimators)
    curve, paired = summarise_curve(acc, args.sizes, args.reference_k)
    curve.to_csv(f"{args.out}/fig4_curve.csv", index=False)
    paired.to_csv(f"{args.out}/fig4_paired.csv", index=False)
    plot_curve(curve, args.reference_k, f"{args.out}/fig4.png")
    print(curve.to_string(index=False))

    bench = {"panel_size_curve": curve.to_dict("records")}

    if not args.skip_whole:
        print("\nWhole-transcriptome benchmark ...")
        bench["whole_transcriptome"] = whole_transcriptome(Xdev, ydev, args.folds)
        print(f"  {bench['whole_transcriptome']}")

    if not args.skip_balanced:
        print("\nClass-balanced sensitivity ...")
        bench["class_balanced"] = balanced_sensitivity(
            X.values, yi, classes, args.sizes, args.folds, args.n_estimators
        )
        print(f"  {bench['class_balanced']}")

    with open(f"{args.out}/benchmarks.json", "w") as fh:
        json.dump(bench, fh, indent=2)
    print(f"\nWrote {args.out}/benchmarks.json")


if __name__ == "__main__":
    main()
