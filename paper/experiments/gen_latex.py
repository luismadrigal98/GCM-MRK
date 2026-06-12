"""Regenerate the paper's LaTeX result files from saved benchmark outputs.

This reuses ``write_latex`` from run_experiments.py but reads the previously
computed ``results/benchmark.json`` and ``results/empirical_modules.csv`` instead
of re-running the genetic algorithm.  Running the full experiment driver also
writes these files; this script is a fast path when only the .tex needs
refreshing.

Run from the repo root:  PYTHONPATH=. python3 paper/experiments/gen_latex.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from run_experiments import write_latex, RESULTS  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    data = json.loads((RESULTS / "benchmark.json").read_text())
    rows, meta = data["rows"], data["meta"]
    scan_hard = meta["model_selection_scan_hard"]
    kbic = min(scan_hard, key=lambda s: s["loglik_bic"])["k_found"]
    kaic = min(scan_hard, key=lambda s: s["loglik_aic"])["k_found"]

    module_rows = []
    mpath = RESULTS / "empirical_modules.csv"
    if mpath.exists():
        with open(mpath) as fh:
            for r in csv.DictReader(fh):
                row = {
                    "module": int(r["module"]),
                    "size": int(r["size"]),
                    "within_abs_corr": float(r["within_abs_corr"]),
                    "eigengene_var_explained": float(r["eigengene_var_explained"]),
                    "tumor_normal_t": float(r["tumor_normal_t"]),
                    "tumor_normal_p": float(r["tumor_normal_p"]),
                    "example_genes": r["example_genes"],
                }
                if "tumor_normal_q" in r:
                    row["tumor_normal_q"] = float(r["tumor_normal_q"])
                module_rows.append(row)

    write_latex(rows, meta, module_rows or None, scan_hard, kbic, kaic)
    print(f"Regenerated LaTeX files in {RESULTS}: "
          "results_macros.tex, benchmark_table.tex, empirical_table.tex")


if __name__ == "__main__":
    main()
