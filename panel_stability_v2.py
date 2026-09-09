#!/usr/bin/env python3
"""
panel_stability_v2.py — selection stability at multiple panel sizes.

Replaces panel_stability.py. Changes, each tied to a reviewer comment:

  [must-fix 4]  Runs at every k in PANEL_SIZES (default 20 and 75) so the
                75-gene panel selected by the one-standard-error rule has a
                stability result, not just the 20-gene parsimony subset.
  [must-fix 3]  Reports a size-corrected overlap (Kuncheva consistency index)
                alongside raw Jaccard, so "is a bigger panel more stable?" is
                answered by a statistic that is comparable across k rather
                than asserted.
  [must-fix 4]  Resamples the DEVELOPMENT split only. The original script
                resampled all 2,928 samples, which put the held-out test set
                inside the stability loop.
  [rec. 5]      --importance {gini,permutation}. Gini importance is biased
                toward correlated and high-cardinality predictors, which is
                exactly the regime under study; permutation importance is the
                sensitivity check.
  [rec. 5]      --n-estimators is explicit and is recorded in the output, so
                the manuscript can state the ranking forest size without
                contradicting the classifier forest size.
  [rec. 7]      --correlate writes, for each gene in the reported panel, the
                genes that most often replaced it and their correlation, which
                turns "instability reflects redundancy" from an assertion into
                a measurement.

  RF seeding: by default the ranking forest is re-seeded per resample
  (--vary-forest-seed, on) so the reported instability includes the estimator's
  own randomness. Pass --no-vary-forest-seed to reproduce the original
  fixed-seed behaviour, which isolates data-driven instability and will report
  slightly HIGHER stability. State whichever you use in the Methods.

Outputs (to ./outputs/):
  stability_k{K}.csv          per-gene selection frequency at each k
  stability_summary.json      core size, Jaccard, Kuncheva, n distinct genes
  substitutions_k{K}.csv      (--correlate) swap partners and correlations

Requires: pandas numpy scikit-learn requests
"""
import argparse, io, gzip, json, os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

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
    """Unchanged from the original pipeline, so the sample set matches exactly."""
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


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------
def rank_order(Xv, yv, n_estimators, importance, seed):
    """Return gene column indices ordered by decreasing importance."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1
    ).fit(Xv, yv)
    if importance == "gini":
        imp = rf.feature_importances_
    elif importance == "permutation":
        # Permutation importance on 20k features is expensive; restrict to the
        # top 500 Gini genes and permute only those. Genes outside that pool
        # cannot plausibly enter a top-75 panel, so the ordering of the head is
        # unaffected while the cost drops by ~40x.
        pool = np.argsort(rf.feature_importances_)[::-1][:500]
        rf_pool = RandomForestClassifier(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1
        ).fit(Xv[:, pool], yv)
        r = permutation_importance(
            rf_pool, Xv[:, pool], yv, n_repeats=5, random_state=seed, n_jobs=-1
        )
        imp = np.full(Xv.shape[1], -np.inf)
        imp[pool] = r.importances_mean
    else:
        raise ValueError(importance)
    return np.argsort(imp)[::-1]


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def kuncheva(a, b, k, p):
    """Consistency index: overlap corrected for the overlap expected by chance.

    IC = (r - k^2/p) / (k - k^2/p), where r = |a & b|. Ranges to 1 for identical
    sets and ~0 for independent draws, and unlike Jaccard is comparable across
    different k. With p ~ 20,000 the correction is small (0.02 at k=20, 0.28 at
    k=75) but it is the standard statistic and removes the objection outright.
    """
    r = len(a & b)
    exp = (k * k) / p
    denom = k - exp
    return (r - exp) / denom if denom > 0 else float("nan")


# --------------------------------------------------------------------------
def resample_indices(n, yi, n_splits):
    """10 stratified CV folds + n_splits repeated 80/20 splits, on dev data."""
    idx = np.arange(n)
    for tr, _ in StratifiedKFold(10, shuffle=True, random_state=RS).split(idx, yi):
        yield tr
    for seed in range(n_splits):
        tr, _ = next(
            StratifiedShuffleSplit(1, test_size=0.20, random_state=seed).split(idx, yi)
        )
        yield tr


def run_for_k(Xdev, ydev, genes, k, args):
    p = Xdev.shape[1]

    reported_order = rank_order(
        Xdev, ydev, args.n_estimators, args.importance, RS
    )
    reported = set(genes[reported_order[:k]])

    counter = Counter()
    panels = []
    for i, tr in enumerate(resample_indices(len(ydev), ydev, args.n_splits)):
        seed = (RS + i) if args.vary_forest_seed else RS
        order = rank_order(Xdev[tr], ydev[tr], args.n_estimators, args.importance, seed)
        panel = set(genes[order[:k]])
        counter.update(panel)
        panels.append(panel)
        if (i + 1) % 10 == 0:
            print(f"    k={k}: {i + 1} resamples done")

    n_rep = len(panels)
    jac_vs_reported = [jaccard(pl, reported) for pl in panels]
    # pairwise stability across resamples — independent of the arbitrary choice
    # of which single panel is called "reported"
    pair_j, pair_ic = [], []
    for i in range(n_rep):
        for j in range(i + 1, n_rep):
            pair_j.append(jaccard(panels[i], panels[j]))
            pair_ic.append(kuncheva(panels[i], panels[j], k, p))

    rows = [
        {
            "gene": g,
            "selected": c,
            "n_resamples": n_rep,
            "frequency": round(c / n_rep, 3),
            "in_reported_panel": g in reported,
        }
        for g, c in counter.most_common()
    ]
    df = pd.DataFrame(rows).sort_values(
        ["in_reported_panel", "frequency"], ascending=[False, False]
    )
    df.to_csv(f"{OUT}/stability_k{k}.csv", index=False)

    core = df[(df.in_reported_panel) & (df.frequency >= 0.8)]
    summary = {
        "k": k,
        "n_resamples": n_rep,
        "reported_panel": sorted(reported),
        "core_genes": sorted(core.gene.tolist()),
        "core_size": int(len(core)),
        "core_fraction": round(len(core) / k, 3),
        "distinct_genes_entering_topk": int(len(df)),
        "mean_jaccard_vs_reported": round(float(np.mean(jac_vs_reported)), 3),
        "min_jaccard_vs_reported": round(float(np.min(jac_vs_reported)), 3),
        "max_jaccard_vs_reported": round(float(np.max(jac_vs_reported)), 3),
        "mean_pairwise_jaccard": round(float(np.mean(pair_j)), 3),
        "mean_pairwise_kuncheva": round(float(np.mean(pair_ic)), 3),
        "expected_random_jaccard": round(
            (k * k / p) / (2 * k - k * k / p), 5
        ),
    }
    print(
        f"\n  [k={k}] core {summary['core_size']}/{k} "
        f"({summary['core_fraction']:.0%})  "
        f"distinct={summary['distinct_genes_entering_topk']}  "
        f"Jaccard={summary['mean_pairwise_jaccard']}  "
        f"Kuncheva={summary['mean_pairwise_kuncheva']}"
    )

    if args.correlate:
        write_substitutions(Xdev, genes, reported, panels, k)

    return summary


def write_substitutions(Xdev, genes, reported, panels, k):
    """For each reported gene, which genes entered when it dropped out?"""
    gidx = {g: i for i, g in enumerate(genes)}
    swaps = defaultdict(Counter)
    for pl in panels:
        missing = reported - pl
        entered = pl - reported
        for m in missing:
            swaps[m].update(entered)

    rows = []
    for g, ctr in swaps.items():
        if g not in gidx:
            continue
        gv = Xdev[:, gidx[g]]
        for partner, n in ctr.most_common(5):
            pv = Xdev[:, gidx[partner]]
            if gv.std() == 0 or pv.std() == 0:
                r = np.nan
            else:
                r = float(np.corrcoef(gv, pv)[0, 1])
            rows.append(
                {
                    "dropped_gene": g,
                    "replacement": partner,
                    "co_occurrences": n,
                    "pearson_r": round(r, 3) if r == r else None,
                }
            )
    pd.DataFrame(rows).to_csv(f"{OUT}/substitutions_k{k}.csv", index=False)
    print(f"  [k={k}] substitution table written")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-sizes", type=int, nargs="+", default=[20, 75])
    ap.add_argument("--n-splits", type=int, default=50)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--importance", choices=["gini", "permutation"], default="gini")
    ap.add_argument("--correlate", action="store_true")
    ap.add_argument(
        "--vary-forest-seed", dest="vary_forest_seed", action="store_true", default=True
    )
    ap.add_argument(
        "--no-vary-forest-seed", dest="vary_forest_seed", action="store_false"
    )
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    OUT = args.out
    os.makedirs(OUT, exist_ok=True)

    X, y = load_xena()
    genes = np.array(X.columns)
    le = LabelEncoder()
    yi = le.fit_transform(y)

    # Hold out the SAME 20% test set as run_analysis.py, and resample only the
    # remaining 80%. This is the fix for the test set appearing inside the
    # stability loop in the original script.
    dev_idx, test_idx = next(
        StratifiedShuffleSplit(1, test_size=0.20, random_state=RS).split(X.values, yi)
    )
    Xdev, ydev = X.values[dev_idx], yi[dev_idx]
    print(f"Development set: {Xdev.shape[0]} samples x {Xdev.shape[1]} genes")

    summaries = [run_for_k(Xdev, ydev, genes, k, args) for k in args.panel_sizes]

    out = {
        "config": {
            "n_estimators": args.n_estimators,
            "importance": args.importance,
            "vary_forest_seed": args.vary_forest_seed,
            "n_splits": args.n_splits,
            "n_dev_samples": int(Xdev.shape[0]),
            "n_genes": int(Xdev.shape[1]),
            "seed": RS,
        },
        "by_panel_size": summaries,
    }
    with open(f"{OUT}/stability_summary.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUT}/stability_summary.json — populate the manuscript from this file only.")


if __name__ == "__main__":
    main()
