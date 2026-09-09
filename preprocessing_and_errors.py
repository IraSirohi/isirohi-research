#!/usr/bin/env python3
"""
preprocessing_and_errors.py — produces three of the five values the other
scripts cannot: the gene intersection size, the zero-variance count, and the
held-out misclassification breakdown.

Run AFTER run_analysis.py (it needs outputs/final_panel.csv for the panel).

Writes outputs/preprocessing_and_errors.json, which collect_for_manuscript.py
picks up automatically.

Fills these manuscript placeholders:
    [[N_INTERSECT]]        Methods 2.2
    [[N_ZEROVAR]]          Methods 2.2
    [[N_MISCLASSIFIED]]    Results 3.2
    Results 3.2 narrative  (which classes, which direction, shared errors)
"""
import io, gzip, json, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit

RS = 42
OUT = "outputs"

COHORTS = {
    "BRCA": "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz",
    "KIRC": "https://tcga.xenahubs.net/download/TCGA.KIRC.sampleMap/HiSeqV2.gz",
    "LUAD": "https://tcga.xenahubs.net/download/TCGA.LUAD.sampleMap/HiSeqV2.gz",
    "PRAD": "https://tcga.xenahubs.net/download/TCGA.PRAD.sampleMap/HiSeqV2.gz",
    "COAD": "https://tcga.xenahubs.net/download/TCGA.COAD.sampleMap/HiSeqV2.gz",
}


def load_with_counts():
    """Same loader as the pipeline, but records the counts lost at each step."""
    import requests
    frames, per_cohort = [], {}
    for lab, url in COHORTS.items():
        print(f"  [load] {lab} ...")
        txt = gzip.decompress(requests.get(url, timeout=600).content).decode("utf-8")
        df = pd.read_csv(io.StringIO(txt), sep="\t", index_col=0).T
        n_all = df.shape[0]
        df = df.loc[[s for s in df.index if str(s)[13:15] == "01"]]
        df.columns = [str(c).upper() for c in df.columns]
        per_cohort[lab] = {
            "samples_all_types": int(n_all),
            "samples_primary_tumor": int(df.shape[0]),
            "genes_in_cohort": int(df.shape[1]),
        }
        df["__y__"] = lab
        frames.append(df)

    union = set()
    shared = set(frames[0].columns)
    for f in frames:
        union |= set(f.columns)
        shared &= set(f.columns)
    shared.discard("__y__")
    union.discard("__y__")
    shared = sorted(shared)

    X = pd.concat([f[shared] for f in frames], axis=0).astype(np.float32).fillna(0.0)
    y = pd.concat([f["__y__"] for f in frames], axis=0)

    n_intersect = X.shape[1]
    var = X.var(0)
    n_zerovar = int((var <= 0).sum())
    X = X.loc[:, var > 0]

    counts = {
        "genes_union_across_cohorts": int(len(union)),
        "genes_intersection_across_cohorts": int(n_intersect),
        "genes_dropped_by_intersection": int(len(union) - n_intersect),
        "genes_zero_variance_removed": n_zerovar,
        "genes_final": int(X.shape[1]),
        "additional_filters_applied": "none",
        "samples_final": int(X.shape[0]),
        "per_cohort": per_cohort,
    }
    return X, y, counts


def main():
    os.makedirs(OUT, exist_ok=True)
    X, y, counts = load_with_counts()
    print(json.dumps(counts, indent=2)[:800])

    panel_path = f"{OUT}/final_panel.csv"
    if not os.path.exists(panel_path):
        print(f"\n{panel_path} not found — run run_analysis.py first. Writing counts only.")
        with open(f"{OUT}/preprocessing_and_errors.json", "w") as fh:
            json.dump({"preprocessing": counts}, fh, indent=2)
        return

    panel = pd.read_csv(panel_path)
    gene_col = "gene" if "gene" in panel.columns else panel.columns[0]
    genes = [g for g in panel[gene_col].astype(str) if g in X.columns]
    print(f"\nPanel: {len(genes)} genes matched")

    le = LabelEncoder()
    yi = le.fit_transform(y)
    classes = list(le.classes_)
    dev, test = next(
        StratifiedShuffleSplit(1, test_size=0.20, random_state=RS).split(X.values, yi)
    )
    Xp = X[genes].values
    Xtr, ytr, Xte, yte = Xp[dev], yi[dev], Xp[test], yi[test]

    rf = RandomForestClassifier(n_estimators=500, random_state=RS, n_jobs=-1).fit(Xtr, ytr)
    sv = Pipeline(
        [("sc", StandardScaler()), ("svm", SVC(kernel="linear", C=1.0, random_state=RS))]
    ).fit(Xtr, ytr)

    preds = {"random_forest": rf.predict(Xte), "linear_svm": sv.predict(Xte)}
    report = {}
    wrong_sets = {}
    for name, p in preds.items():
        wrong = np.where(p != yte)[0]
        wrong_sets[name] = set(wrong.tolist())
        errs = {}
        for i in wrong:
            key = f"{classes[yte[i]]} -> {classes[p[i]]}"
            errs[key] = errs.get(key, 0) + 1
        report[name] = {
            "n_test": int(len(yte)),
            "n_misclassified": int(len(wrong)),
            "accuracy_pct": round(100 * (1 - len(wrong) / len(yte)), 3),
            "errors_by_direction": dict(sorted(errs.items(), key=lambda kv: -kv[1])),
            "misclassified_sample_ids": [str(X.index[test][i]) for i in wrong],
        }

    a, b = wrong_sets["random_forest"], wrong_sets["linear_svm"]
    shared = a & b
    report["shared_errors"] = {
        "n_missed_by_both": int(len(shared)),
        "n_missed_by_rf_only": int(len(a - b)),
        "n_missed_by_svm_only": int(len(b - a)),
        "sample_ids_missed_by_both": [str(X.index[test][i]) for i in sorted(shared)],
        "interpretation_hint": (
            "Samples missed by BOTH models are candidates for intrinsically "
            "ambiguous specimens; samples missed by only one are model-specific "
            "error. Report both counts in Results 3.2."
        ),
    }

    with open(f"{OUT}/preprocessing_and_errors.json", "w") as fh:
        json.dump({"preprocessing": counts, "misclassification": report}, fh, indent=2)

    print("\n--- misclassification ---")
    for name in preds:
        r = report[name]
        print(f"{name}: {r['n_misclassified']}/{r['n_test']} wrong ({r['accuracy_pct']}%)")
        for k, v in r["errors_by_direction"].items():
            print(f"    {k}: {v}")
    print(f"missed by both models: {report['shared_errors']['n_missed_by_both']}")
    print(f"\nWrote {OUT}/preprocessing_and_errors.json")


if __name__ == "__main__":
    main()
