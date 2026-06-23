#!/usr/bin/env python3
"""
core_panel_test.py — does the STABLE core of the panel retain accuracy?

Reads outputs/panel_stability.csv, takes the genes selected in >=80% of resamples (the core),
and evaluates a classifier restricted to just those genes on the same held-out 20% test split
(seed 42) used in the main analysis. Reports core-panel accuracy vs. the full 20-gene panel,
so the paper can state whether a small stable subset carries the signal.

Run in Colab after run_analysis.py and panel_stability.py.
Requires: pandas numpy scikit-learn requests
"""
import io, os, gzip, math, requests
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

RS = 42
COHORTS = {
    "BRCA": "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz",
    "KIRC": "https://tcga.xenahubs.net/download/TCGA.KIRC.sampleMap/HiSeqV2.gz",
    "LUAD": "https://tcga.xenahubs.net/download/TCGA.LUAD.sampleMap/HiSeqV2.gz",
    "PRAD": "https://tcga.xenahubs.net/download/TCGA.PRAD.sampleMap/HiSeqV2.gz",
    "COAD": "https://tcga.xenahubs.net/download/TCGA.COAD.sampleMap/HiSeqV2.gz",
}

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return round(p*100, 2), round((c-h)*100, 2), round((c+h)*100, 2)

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
    return X.loc[:, X.var(0) > 0], y

st = pd.read_csv("outputs/panel_stability.csv")
core = [g for g in st[(st.in_reported_panel) & (st.frequency >= 0.8)].gene]
print(f"Core genes (>=80% selection): {len(core)} -> {core}")

X, y = load_xena()
le = LabelEncoder(); yi = le.fit_transform(y); classes = list(le.classes_)
present = [g for g in core if g in X.columns]
if len(present) < len(core):
    print("  note: not all core genes present in matrix:", set(core) - set(present))

Xtr, Xte, ytr, yte = train_test_split(X.values, yi, test_size=0.20, stratify=yi, random_state=RS)
cols = [list(X.columns).index(g) for g in present]
sc = StandardScaler().fit(Xtr[:, cols])
for name, clf, Xt in [
    ("RF",  RandomForestClassifier(500, random_state=RS, n_jobs=-1).fit(Xtr[:, cols], ytr), Xte[:, cols]),
    ("SVM", SVC(kernel="linear", C=1.0, random_state=RS).fit(sc.transform(Xtr[:, cols]), ytr), sc.transform(Xte[:, cols])),
]:
    p = clf.predict(Xt); k = int((p == yte).sum()); n = len(yte)
    a = wilson(k, n)
    print(f"\n[CORE {len(present)}-gene | {name}] accuracy = {a[0]}% (95% CI {a[1]}-{a[2]})  ({k}/{n})")
    print(classification_report(yte, p, target_names=classes, zero_division=0))

print("\nCompare to the full 20-gene held-out accuracy (RF 99.32% / SVM 99.49%).")
print("Report whatever this shows: if the core retains accuracy, a small stable subset carries "
      "the signal; if not, even the stable core is insufficient alone.")
