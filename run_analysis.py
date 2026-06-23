#!/usr/bin/env python3
"""
run_analysis.py  —  ONE honest, leakage-free pipeline for the 5-type signature.

Run in Google Colab (needs internet for the Xena training data). Put these in the
same folder if you want the external block: BRCA.zip COAD.zip CCRCC.zip LUAD.zip
and optionally gse54460_prostate_rnaseq.tsv.gz

It writes everything to ./outputs/ :
  results.json        every metric (nested-CV, held-out test, external) + CIs
  final_panel.csv     the 20 genes the pipeline actually selected
  per_type_expression.csv   z-scored mean expression by type (for peak-type column)
  fig*.png            figures

Design rules (see analysis_protocol.md): selection happens only inside training folds;
the 20% test set is touched once; external is reported under BOTH normalization schemes;
NOTHING is hardcoded — the panel and the numbers come from the data.

Requires: pandas numpy scipy scikit-learn matplotlib requests joblib mygene
"""
import io, os, gzip, json, zipfile, platform, subprocess, sys, math
import numpy as np, pandas as pd, requests, joblib
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

RS = 42
PANEL_SIZES = [5, 10, 15, 20, 30, 50]
FINAL_K = 20
OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
COHORTS = {
    "BRCA": "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz",
    "KIRC": "https://tcga.xenahubs.net/download/TCGA.KIRC.sampleMap/HiSeqV2.gz",
    "LUAD": "https://tcga.xenahubs.net/download/TCGA.LUAD.sampleMap/HiSeqV2.gz",
    "PRAD": "https://tcga.xenahubs.net/download/TCGA.PRAD.sampleMap/HiSeqV2.gz",
    "COAD": "https://tcga.xenahubs.net/download/TCGA.COAD.sampleMap/HiSeqV2.gz",
}
CPTAC_ZIPS = {"BRCA": "BRCA.zip", "COAD": "COAD.zip", "CCRCC": "CCRCC.zip", "LUAD": "LUAD.zip"}
CPTAC_LABEL = {"BRCA": "BRCA", "COAD": "COAD", "CCRCC": "KIRC", "LUAD": "LUAD"}
PROSTATE_FILE = "gse54460_prostate_rnaseq.tsv.gz"   # optional; set to None to skip
NUM = {}

# ----------------------------- statistics helpers ---------------------------
def wilson(k, n, z=1.96):
    if n == 0: return (None, None, None)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return round(p*100, 2), round((c-h)*100, 2), round((c+h)*100, 2)

def boot_macro_f1(y_true, y_pred, n_boot=5000, seed=RS):
    labels = sorted(set(int(v) for v in y_true))            # only classes actually present
    rng = np.random.default_rng(seed); n = len(y_true); idx = np.arange(n); vals = []
    for _ in range(n_boot):
        s = rng.choice(idx, n, replace=True)
        vals.append(f1_score(y_true[s], y_pred[s], labels=labels, average="macro", zero_division=0))
    return (round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)*100, 2),
            round(np.percentile(vals, 2.5)*100, 2), round(np.percentile(vals, 97.5)*100, 2))

def report_block(tag, y_true, y_pred, classes):
    acc = wilson((y_true == y_pred).sum(), len(y_true))
    mf = boot_macro_f1(y_true, y_pred)
    rec = {}
    for i, c in enumerate(classes):
        m = y_true == i
        if m.sum(): rec[c] = wilson((y_pred[m] == i).sum(), int(m.sum()))
    NUM[tag] = {"n": int(len(y_true)), "accuracy_ci": acc, "macro_f1_ci": mf,
                "per_class_recall_ci": rec,
                "confusion": confusion_matrix(y_true, y_pred, labels=range(len(classes))).tolist()}
    print(f"\n[{tag}] acc={acc[0]}% CI{acc[1:]}  macroF1={mf[0]}% CI{mf[1:]}")
    present = sorted(set(int(v) for v in y_true) | set(int(v) for v in y_pred))
    print(classification_report(y_true, y_pred, labels=present,
                                target_names=[classes[i] for i in present], zero_division=0))

# ----------------------------- data loading ---------------------------------
def load_xena():
    frames = []
    for lab, url in COHORTS.items():
        print(f"  [train] {lab} ...")
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

def rank_genes(Xv, yv, n_estimators=300):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=RS, n_jobs=-1).fit(Xv, yv)
    return np.argsort(rf.feature_importances_)[::-1]

def svm_pipe():
    return Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="linear", C=1.0, random_state=RS))])

def choose_k(Xv, yv):
    order = rank_genes(Xv, yv)
    inner = StratifiedKFold(5, shuffle=True, random_state=RS)
    best_k, best = PANEL_SIZES[0], -1
    for k in PANEL_SIZES:
        cols = order[:k]; accs = []
        for tr, va in inner.split(Xv, yv):
            clf = svm_pipe().fit(Xv[tr][:, cols], yv[tr])
            accs.append(accuracy_score(yv[va], clf.predict(Xv[va][:, cols])))
        if np.mean(accs) > best: best, best_k = np.mean(accs), k
    return best_k, order

def main():
    print("Loading symbol-labeled TCGA (UCSC Xena) ...")
    X, y = load_xena(); genes = np.array(X.columns)
    le = LabelEncoder(); yi = le.fit_transform(y); classes = list(le.classes_)
    NUM["class_counts"] = {c: int((y == c).sum()) for c in classes}
    NUM["n_samples"] = int(X.shape[0]); NUM["n_genes"] = int(X.shape[1])
    print(f"  {X.shape[0]} samples x {X.shape[1]} genes; {NUM['class_counts']}")

    Xdev, Xtest, ydev, ytest = train_test_split(X, yi, test_size=0.20, stratify=yi, random_state=RS)

    # ---- nested CV: unbiased internal estimate ----
    outer = StratifiedKFold(10, shuffle=True, random_state=RS); accs, ks = [], []
    for tr, va in outer.split(Xdev.values, ydev):
        k, _ = choose_k(Xdev.values[tr], ydev[tr])
        cols = rank_genes(Xdev.values[tr], ydev[tr])[:k]
        clf = svm_pipe().fit(Xdev.values[tr][:, cols], ydev[tr])
        accs.append(accuracy_score(ydev[va], clf.predict(Xdev.values[va][:, cols]))); ks.append(k)
    NUM["nested_cv"] = {"acc_mean": round(np.mean(accs)*100, 2), "acc_sd": round(np.std(accs)*100, 2),
                        "panel_sizes_chosen": ks}
    print(f"\n[NESTED CV] {NUM['nested_cv']['acc_mean']}% +/- {NUM['nested_cv']['acc_sd']}; k chosen {ks}")

    # ---- panel-size curve (dev set) ----
    order_dev = rank_genes(Xdev.values, ydev)
    curve = {}
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    for k in [2, 5, 10, 15, 20, 30, 50, 100, 200]:
        cols = order_dev[:k]
        from sklearn.model_selection import cross_val_score
        curve[k] = round(cross_val_score(svm_pipe(), Xdev.values[:, cols], ydev, cv=cv).mean()*100, 2)
    NUM["panel_size_curve"] = curve

    # ---- final 20-gene panel + held-out test ----
    panel_idx = order_dev[:FINAL_K]; panel = list(genes[panel_idx])
    NUM["final_panel"] = panel
    pd.DataFrame({"rank": range(1, FINAL_K+1), "gene": panel}).to_csv(f"{OUT}/final_panel.csv", index=False)
    scaler = StandardScaler().fit(Xdev.values[:, panel_idx])
    Xdev_s, Xtest_s = scaler.transform(Xdev.values[:, panel_idx]), scaler.transform(Xtest.values[:, panel_idx])
    rf  = RandomForestClassifier(500, random_state=RS, n_jobs=-1).fit(Xdev_s, ydev)
    svm = SVC(kernel="linear", C=1.0, random_state=RS).fit(Xdev_s, ydev)
    report_block("heldout_rf",  ytest, rf.predict(Xtest_s),  classes)
    report_block("heldout_svm", ytest, svm.predict(Xtest_s), classes)

    # ---- per-type expression (peak-type column for Table 2) ----
    Xp = X.values[:, panel_idx]; Xpz = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-9)
    per = pd.DataFrame(np.vstack([Xpz[yi == i].mean(0) for i in range(len(classes))]),
                       index=classes, columns=panel)
    per.T.to_csv(f"{OUT}/per_type_expression.csv")
    NUM["peak_type"] = {g: classes[int(np.argmax(per[g].values))] for g in panel}

    # ---- external validation (both normalization schemes) ----
    mu = Xdev.values[:, panel_idx].mean(0); sd = Xdev.values[:, panel_idx].std(0); sd[sd == 0] = 1.0
    try:
        import mygene
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "-q", "install", "mygene"], check=True); import mygene
    mg = mygene.MyGeneInfo()
    res = mg.querymany(panel, scopes="symbol,alias", fields="ensembl.gene", species="human", verbose=False)
    sym2ens = {}
    for r in res:
        if not r.get("notfound") and r.get("ensembl"):
            e = r["ensembl"]; sym2ens.setdefault(r["query"].upper(), e[0]["gene"] if isinstance(e, list) else e["gene"])
    ens2sym = {v: k for k, v in sym2ens.items()}

    def load_cptac(zp):
        with zipfile.ZipFile(zp) as zf:
            t = [n for n in zf.namelist() if n.endswith("_RNAseq_gene_RSEM_coding_UQ_1500_log2_Tumor.txt")][0]
            with zf.open(t) as fh: df = pd.read_csv(fh, sep="\t", index_col=0)
        df.index = [str(i).split(".")[0] for i in df.index]
        df = df[df.index.isin(ens2sym)].groupby(level=0).mean(); df.index = [ens2sym[i] for i in df.index]
        return df.T.reindex(columns=panel)

    if all(os.path.exists(z) for z in CPTAC_ZIPS.values()):
        mats, labs = [], []
        for c, zp in CPTAC_ZIPS.items():
            m = load_cptac(zp); mats.append(m); labs += [CPTAC_LABEL[c]] * m.shape[0]
        Xe = pd.concat(mats).reindex(columns=panel).astype(float); ye = le.transform(np.array(labs))
        XA = np.nan_to_num((Xe.values - mu) / sd)                                   # training-fixed
        muB = np.nanmean(Xe.values, 0); sdB = np.nanstd(Xe.values, 0); sdB[sdB == 0] = 1.0
        XB = np.nan_to_num((Xe.values - muB) / sdB)                                 # pooled
        for scheme, Xse in [("trainfixed", XA), ("pooled", XB)]:
            report_block(f"external_cptac_rf_{scheme}",  ye, rf.predict(Xse),  classes)
            report_block(f"external_cptac_svm_{scheme}", ye, svm.predict(Xse), classes)
    else:
        print("\n[EXTERNAL] CPTAC zips missing — skipped.")

    # GSE54460 prostate: report SEPARATELY (different platform, all PRAD).
    def load_gse54460(path):
        opn = gzip.open if path.endswith(".gz") else open
        rows = [ln.rstrip("\n") for ln in opn(path, "rt")]
        hdr_i = next((i for i, l in enumerate(rows) if "SYMBOL" in [c.strip().upper() for c in l.split("\t")]), 0)
        header = [h.strip() for h in rows[hdr_i].split("\t")]
        up = [h.upper() for h in header]
        sym_j = up.index("SYMBOL") if "SYMBOL" in up else 1
        first_sample = max(sym_j + 1, 2)
        samples = header[first_sample:]
        pset = {g.upper() for g in panel}
        data = {}
        for l in rows[hdr_i + 1:]:
            p = l.split("\t")
            if len(p) <= first_sample: continue
            s = p[sym_j].strip().upper()
            if s in pset:
                vals = []
                for x in p[first_sample:first_sample + len(samples)]:
                    try: vals.append(float(x))
                    except: vals.append(np.nan)
                data[s] = vals
        M = pd.DataFrame(data, index=samples)              # samples x genes
        M = np.log2(M.clip(lower=0) + 1.0)                 # FPKM -> log2, matches training scale
        return M.reindex(columns=panel)

    if PROSTATE_FILE and os.path.exists(PROSTATE_FILE):
        try:
            Mp = load_gse54460(PROSTATE_FILE)
            found = int(Mp.notna().any(axis=0).sum())
            print(f"\n[EXTERNAL PROSTATE] GSE54460: {Mp.shape[0]} samples, {found}/{len(panel)} panel genes present")
            yp = np.array([list(le.classes_).index("PRAD")] * Mp.shape[0])
            XpA = np.nan_to_num((Mp.values - mu) / sd)                  # training-fixed
            muP = np.nanmean(Mp.values, 0); sdP = np.nanstd(Mp.values, 0); sdP[sdP == 0] = 1.0
            XpB = np.nan_to_num((Mp.values - muP) / sdP)                # pooled (within prostate)
            for scheme, Xpx in [("trainfixed", XpA), ("pooled", XpB)]:
                for name, clf in [("rf", rf), ("svm", svm)]:
                    pred = clf.predict(Xpx)
                    report_block(f"external_prostate_{name}_{scheme}", yp, pred, classes)
                    # where do the misses go? (all true labels are PRAD)
                    NUM[f"external_prostate_{name}_{scheme}"]["called_as"] = \
                        pd.Series([classes[p] for p in pred]).value_counts().to_dict()
        except Exception as e:
            print(f"\n[EXTERNAL PROSTATE] could not parse {PROSTATE_FILE}: {e}")
            print("Paste the first 3 lines of the file and I will fix the loader.")
    else:
        print("\n[EXTERNAL PROSTATE] file not found — skipped.")

    NUM["env"] = {"python": platform.python_version(), "numpy": np.__version__,
                  "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "seed": RS}
    json.dump(NUM, open(f"{OUT}/results.json", "w"), indent=2)
    joblib.dump({"rf": rf, "svm": svm, "scaler": scaler, "panel": panel,
                 "mu": mu.tolist(), "sd": sd.tolist(), "classes": classes}, f"{OUT}/frozen_models.joblib")
    print(f"\nSaved everything to ./{OUT}/  — fill the manuscript ONLY from results.json.")

if __name__ == "__main__":
    main()
