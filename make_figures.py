#!/usr/bin/env python3
"""
make_figures.py — draw the manuscript figures from saved outputs only.

Reads outputs/results.json and outputs/per_type_expression.csv (no re-download, no
re-training) and writes 300-dpi PNGs to outputs/figures/. Run after run_analysis.py.

Figures produced:
  fig_panel_size_curve.png      accuracy vs number of genes (Section 3.2)
  fig_confusion_heldout_rf.png  held-out test confusion, random forest (Section 3.1)
  fig_confusion_heldout_svm.png held-out test confusion, linear SVM
  fig_confusion_cptac_rf.png    external CPTAC confusion, training-fixed RF (Section 3.5)
  fig_expression_heatmap.png    z-scored mean expression of panel genes by type (Section 3.3)

Requires: pandas numpy matplotlib
"""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"; FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
NUM = json.load(open(os.path.join(OUT, "results.json")))
per = pd.read_csv(os.path.join(OUT, "per_type_expression.csv"), index_col=0)  # genes x types
CLASSES = list(per.columns)
DPI = 300

def confusion_fig(key, title, fname):
    if key not in NUM or "confusion" not in NUM[key]:
        print(f"  (skip {key}: not in results.json)"); return
    cm = np.array(NUM[key]["confusion"])
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    mx = cm.max() if cm.max() else 1
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > mx / 2 else "black", fontsize=9)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title(title, fontsize=11)
    plt.colorbar(im, fraction=0.046, pad=0.04); plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname), dpi=DPI, bbox_inches="tight"); plt.close()
    print(f"  wrote {fname}")

# Fig: panel-size curve
if "panel_size_curve" in NUM:
    curve = NUM["panel_size_curve"]; ks = [int(k) for k in curve]; acc = [curve[str(k)] for k in ks]
    order = np.argsort(ks); ks = np.array(ks)[order]; acc = np.array(acc)[order]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(ks, acc, "o-", color="#1c7ed6")
    ax.axvline(20, color="#d6336c", ls="--", lw=1, label="20-gene panel (reported)")
    ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("Number of top-ranked genes"); ax.set_ylabel("5-fold CV accuracy (%)")
    ax.set_title("Classification accuracy vs. panel size", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_panel_size_curve.png"), dpi=DPI, bbox_inches="tight"); plt.close()
    print("  wrote fig_panel_size_curve.png")

confusion_fig("heldout_rf",  "Held-out test \u2014 random forest", "fig_confusion_heldout_rf.png")
confusion_fig("heldout_svm", "Held-out test \u2014 linear SVM",    "fig_confusion_heldout_svm.png")
confusion_fig("external_cptac_rf_trainfixed", "External CPTAC \u2014 random forest (training-fixed)", "fig_confusion_cptac_rf.png")

# Fig: expression heatmap (types x genes)
heat = per.T  # types x genes
fig, ax = plt.subplots(figsize=(9, 3.4))
im = ax.imshow(heat.values, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
ax.set_yticks(range(len(heat.index))); ax.set_yticklabels(heat.index)
ax.set_xticks(range(len(heat.columns))); ax.set_xticklabels(heat.columns, rotation=60, ha="right", fontsize=8)
ax.set_title("Standardized mean expression of panel genes by tumor type", fontsize=11)
plt.colorbar(im, fraction=0.02, pad=0.02, label="z-score"); plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_expression_heatmap.png"), dpi=DPI, bbox_inches="tight"); plt.close()
print("  wrote fig_expression_heatmap.png")
print(f"\nFigures in {FIG}/  — insert into the manuscript where marked.")
