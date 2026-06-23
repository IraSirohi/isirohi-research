#!/usr/bin/env python3
"""
make_supplementary.py — build Supplementary Table S1 (all 79 genes) from the stability run.

Reads outputs/panel_stability.csv (written by panel_stability.py) and produces:
  outputs/Supplementary_Table_S1.csv   — Gene, Selection frequency (X/60), In reported panel
  outputs/Supplementary_Table_S1.md    — a clean Markdown table to paste into the SI document

Run in Colab after panel_stability.py. Requires: pandas
"""
import pandas as pd, os
df = pd.read_csv("outputs/panel_stability.csv")
n = int(df["n_repeats"].iloc[0])
df = df.sort_values(["selected", "gene"], ascending=[False, True]).reset_index(drop=True)

out = df[["gene", "selected", "in_reported_panel"]].copy()
out.columns = ["Gene", f"Selection frequency (/{n})", "In reported 20-gene panel"]
out.to_csv("outputs/Supplementary_Table_S1.csv", index=False)

with open("outputs/Supplementary_Table_S1.md", "w") as f:
    f.write(f"# Supplementary Table S1. Selection frequency of all genes entering the top-20 "
            f"panel across {n} resamples (10 CV folds + 50 repeated 80/20 splits).\n\n")
    f.write(f"| Gene | Selection frequency (/{n}) | In reported 20-gene panel |\n")
    f.write("|---|---|---|\n")
    for _, r in out.iterrows():
        f.write(f"| {r['Gene']} | {int(r[out.columns[1]])} | {'yes' if r[out.columns[2]] else 'no'} |\n")

print(f"Wrote Supplementary_Table_S1.csv and .md  ({len(out)} genes, frequencies out of {n}).")
print("Paste the .md table into your Supplementary Materials file, or upload the .csv.")
