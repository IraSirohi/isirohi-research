#!/usr/bin/env python3
"""
collect_for_manuscript.py — bundle every analysis output into ONE file.

Run this last, after:
    python run_analysis.py                      # with FINAL_K set to the nested optimum
    python fig4_and_benchmarks.py
    python panel_stability_v2.py --correlate
    python panel_stability_v2.py --importance permutation --panel-sizes <K> --out outputs/perm

It reads everything in ./outputs/ and writes:

    MANUSCRIPT_VALUES.json

Upload that single file. Every key matches a [[PLACEHOLDER]] in the manuscript.
Missing inputs are reported as "MISSING — did you run X?" rather than silently
skipped, so you can see at a glance what still needs running.

Usage:  python collect_for_manuscript.py [--outputs outputs] [--perm outputs/perm]
"""
import argparse, json, os, sys

import pandas as pd


def load_json(path, label):
    if not os.path.exists(path):
        return {"__missing__": f"MISSING — {label}"}
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as e:  # noqa: BLE001
        return {"__error__": f"could not parse {path}: {e}"}


def load_csv(path, label):
    if not os.path.exists(path):
        return {"__missing__": f"MISSING — {label}"}
    try:
        return pd.read_csv(path).to_dict("records")
    except Exception as e:  # noqa: BLE001
        return {"__error__": f"could not parse {path}: {e}"}


def find_stability_files(out):
    """Return {k: path} for every stability_k*.csv present."""
    found = {}
    if not os.path.isdir(out):
        return found
    for fn in os.listdir(out):
        if fn.startswith("stability_k") and fn.endswith(".csv"):
            try:
                k = int(fn[len("stability_k"):-len(".csv")])
            except ValueError:
                continue
            found[k] = os.path.join(out, fn)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--perm", default="outputs/perm")
    ap.add_argument("--dest", default="MANUSCRIPT_VALUES.json")
    args = ap.parse_args()

    out = args.outputs
    if not os.path.isdir(out):
        sys.exit(f"No such directory: {out}. Run the analysis scripts first.")

    bundle = {}

    # ---- main pipeline -----------------------------------------------
    bundle["results_json"] = load_json(
        os.path.join(out, "results.json"), "run_analysis.py has not been run"
    )
    bundle["final_panel"] = load_csv(
        os.path.join(out, "final_panel.csv"), "run_analysis.py has not been run"
    )
    bundle["per_type_expression"] = load_csv(
        os.path.join(out, "per_type_expression.csv"),
        "run_analysis.py has not been run",
    )

    # ---- figure 4 + benchmarks ---------------------------------------
    bundle["benchmarks"] = load_json(
        os.path.join(out, "benchmarks.json"), "fig4_and_benchmarks.py has not been run"
    )
    bundle["fig4_curve"] = load_csv(
        os.path.join(out, "fig4_curve.csv"), "fig4_and_benchmarks.py has not been run"
    )
    bundle["fig4_paired"] = load_csv(
        os.path.join(out, "fig4_paired.csv"), "fig4_and_benchmarks.py has not been run"
    )

    # ---- stability, every k that was run -----------------------------
    bundle["stability_summary"] = load_json(
        os.path.join(out, "stability_summary.json"),
        "panel_stability_v2.py has not been run",
    )
    stab = find_stability_files(out)
    bundle["stability_tables"] = {
        str(k): load_csv(p, "") for k, p in sorted(stab.items())
    } or {"__missing__": "MISSING — no stability_k*.csv found"}

    subs = {}
    for k in sorted(stab):
        p = os.path.join(out, f"substitutions_k{k}.csv")
        if os.path.exists(p):
            subs[str(k)] = load_csv(p, "")
    bundle["substitutions"] = subs or {
        "__missing__": "MISSING — rerun panel_stability_v2.py with --correlate"
    }

    # ---- permutation-importance sensitivity --------------------------
    bundle["permutation_sensitivity"] = load_json(
        os.path.join(args.perm, "stability_summary.json"),
        "permutation-importance run has not been done",
    )

    # ---- preprocessing counts + misclassification --------------------
    bundle["preprocessing_and_errors"] = load_json(
        os.path.join(out, "preprocessing_and_errors.json"),
        "preprocessing_and_errors.py has not been run",
    )

    # ---- things only the author can supply ---------------------------
    bundle["author_must_supply"] = {
        "enrichment_result": "g:Profiler result for the NEW panel — terms and adjusted p-values (Results 3.5)",
        "citation_audit": "Result of checking each remaining reference against the claim it supports (Response letter, Must Fix 7)",
    }

    # ---- completeness report -----------------------------------------
    missing = []

    def walk(node, path=""):
        if isinstance(node, dict):
            if "__missing__" in node:
                missing.append(f"{path}: {node['__missing__']}")
                return
            if "__error__" in node:
                missing.append(f"{path}: {node['__error__']}")
                return
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)

    for key in [
        "results_json", "final_panel", "per_type_expression", "benchmarks",
        "fig4_curve", "fig4_paired", "stability_summary", "stability_tables",
        "substitutions", "permutation_sensitivity", "preprocessing_and_errors",
    ]:
        walk(bundle[key], key)

    bundle["_completeness"] = {
        "complete": not missing,
        "missing": missing,
    }

    with open(args.dest, "w") as fh:
        json.dump(bundle, fh, indent=2, default=str)

    print(f"Wrote {args.dest}")
    if missing:
        print("\nStill missing — the manuscript cannot be fully filled without these:")
        for m in missing:
            print(f"  - {m}")
    else:
        print("\nAll analysis outputs present. Upload this file.")
    print(
        "\nAlso supply by hand: "
        + ", ".join(bundle["author_must_supply"].keys())
    )


if __name__ == "__main__":
    main()
