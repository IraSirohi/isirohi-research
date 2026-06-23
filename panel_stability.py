#!/usr/bin/env python3
"""
panel_stability.py — how stable is the 20-gene panel?

Re-derives the top-20 panel many times under resampling and counts how often each gene is
selected. Reports selection frequency, the Jaccard overlap of each panel with the reported
panel, and how many of the reported 20 genes are "core" (selected in >=80% of repeats).

Two resampling schemes (both reported):
  (A) 10 outer CV folds  — the same folds the paper's nested CV uses.
  (B) N repeated 80/20 stratified splits with different seeds.

Run in Colab (needs Xena). Writes outputs/panel_stability.csv and prints a table you can
paste straight into the manuscript. NOTHING is hardcoded; selection is on training data only.

Requires: pandas numpy scikit-learn requests
"""
import io, os, gzip, json, requests
import numpy as np, pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

RS = 42
K = 20                      # panel size to report stability for
N_SPLITS = 50               # scheme B: number of repeated 80/20 splits
OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
COHORTS = {
    "BRCA": "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz",
    "KIRC": "https://tcga.xenahubs.net/download/TCGA.KIRC.sampleMap/HiSeqV2.gz",
    "LUAD": "https://tcga.xenahubs.net/download/TCGA.LUAD.sampleMap/HiSeqV2.gz",
    "PRAD": "https://tcga.xenahubs.net/download/TCGA.PRAD.sampleMap/HiSeqV2.gz",
    "COAD": "https://tcga.xenahubs.net/download/TCGA.COAD.sampleMap/HiSeqV2.gz",
}

def load_xena():
    frames = []
    for lab, url in COHORTS.items():
        print(f"  [load] {lab} ...")
        txt = gzip.decompress(requests.get(url, timeout=600).content).decode("utf-8")
        df = pd.read_csv(io.StringIO(txt), sep="\t", index_col=0).T
        df = df.loc[[s for s in df.index if str(s)[13:15] == "01"]]
        df.columns = [str(c).upper() for c in df.columns]
        df["__y__"] = lab; frames.append(df)
    shared = set(frames[0].columns)
    for f in frames[1:]: shared &= set(f.columns)
    shared.discard("__y__"); shared = sorted(shared)
    X = pd.concat([f[shared] for f in frames], axis=0).astype(float).fillna(0.0)
    y = pd.concat([f["__y__"] for f in frames], axis=0)
    X = X.loc[:, X.var(0) > 0]
    return X, y

def top_panel(Xv, yv, genes, k=K):
    rf = RandomForestClassifier(300, random_state=RS, n_jobs=-1).fit(Xv, yv)
    order = np.argsort(rf.feature_importances_)[::-1][:k]
    return set(genes[order])

def jaccard(a, b): return len(a & b) / len(a | b)

def main():
    print("Loading TCGA (Xena) ...")
    X, y = load_xena(); genes = np.array(X.columns)
    le = LabelEncoder(); yi = le.fit_transform(y)

    # reported panel = top-K on the full development split (seed 42), to compare against
    dev_idx, _ = next(StratifiedShuffleSplit(1, test_size=0.20, random_state=RS).split(X.values, yi))
    reported = top_panel(X.values[dev_idx], yi[dev_idx], genes)
    print(f"\nReported panel ({len(reported)} genes): {sorted(reported)}")

    counter = Counter(); jacc = []; n_repeats = 0

    # Scheme A: 10 outer CV folds (train side of each fold)
    for tr, _ in StratifiedKFold(10, shuffle=True, random_state=RS).split(X.values, yi):
        panel = top_panel(X.values[tr], yi[tr], genes)
        counter.update(panel); jacc.append(jaccard(panel, reported)); n_repeats += 1

    # Scheme B: repeated 80/20 splits
    for seed in range(N_SPLITS):
        tr, _ = next(StratifiedShuffleSplit(1, test_size=0.20, random_state=seed).split(X.values, yi))
        panel = top_panel(X.values[tr], yi[tr], genes)
        counter.update(panel); jacc.append(jaccard(panel, reported)); n_repeats += 1

    # build the table: every gene ever selected, frequency out of n_repeats
    rows = []
    for g, c in counter.most_common():
        rows.append({"gene": g, "selected": c, "n_repeats": n_repeats,
                     "frequency": round(c / n_repeats, 3), "in_reported_panel": g in reported})
    df = pd.DataFrame(rows).sort_values(["in_reported_panel", "frequency"], ascending=[False, False])
    df.to_csv(os.path.join(OUT, "panel_stability.csv"), index=False)

    core = df[(df.in_reported_panel) & (df.frequency >= 0.8)]
    print(f"\nRepeats: {n_repeats} ({10} CV folds + {N_SPLITS} random splits)")
    print(f"Mean Jaccard overlap with reported panel: {np.mean(jacc):.2f} "
          f"(min {np.min(jacc):.2f}, max {np.max(jacc):.2f})")
    print(f"Reported-panel genes selected in >=80% of repeats (core): {len(core)}/{len(reported)}")
    print(f"Distinct genes ever entering the top-{K}: {len(df)}")

    print("\n=== Selection frequency of the reported panel genes (paste into the manuscript) ===")
    print(f"{'Gene':<10}{'Selection frequency':<22}")
    for _, r in df[df.in_reported_panel].iterrows():
        print(f"{r.gene:<10}{int(r.selected)}/{n_repeats}")
    print("\nFull table (incl. genes outside the reported panel) saved to outputs/panel_stability.csv")

if __name__ == "__main__":
    main()
