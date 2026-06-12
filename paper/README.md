# GCM-MRK paper

LaTeX source and reproduction code for the GCM-MRK methodological paper,
formatted for *PLOS Computational Biology* using the official PLOS template.

## Files

- `gcmrk_paper.tex` — the manuscript.
- `references.bib` — bibliography.
- `plos2025.bst` — PLOS BibTeX style (required by the template).
- `experiments/run_experiments.py` — runs all benchmarks and writes the results
  (`results/`), figures (`figures/`), **and the LaTeX the manuscript `\input`s**
  (`results/results_macros.tex`, `results/benchmark_table.tex`,
  `results/empirical_table.tex`). Re-running it updates every number and table in
  the paper automatically.
- `experiments/gen_latex.py` — fast path: regenerate only the LaTeX files from the
  already-saved `results/benchmark.json` and `results/empirical_modules.csv`
  (no GA re-run).
- `data/GSE183947_fpkm.csv` — empirical breast-cancer RNA-seq dataset.
- `results/benchmark.csv`, `results/benchmark.json` — generated results.
- `figures/*.pdf` — generated figures.

## Reproduce the experiments

From the repository root (so `gcmrk` is importable):

```bash
PYTHONPATH=. python3 paper/experiments/run_experiments.py
```

Dependencies beyond the package itself: `scikit-learn` (Iris dataset + ARI)
and `matplotlib` (figures).

## Build the PDF

```bash
cd paper
pdflatex gcmrk_paper
bibtex   gcmrk_paper
pdflatex gcmrk_paper
pdflatex gcmrk_paper
```

## How auto-update works

`run_experiments.py` writes `\newcommand` macros and full `tabular` bodies into
`results/*.tex`. The manuscript reads them via `\InputIfFileExists` / `\input`
(with `\providecommand` defaults so it still compiles before the first run). To
refresh the paper after changing the analysis: re-run the experiments (or
`gen_latex.py`), then recompile. No manual number edits.

## Status / TODO

- [x] Tool description and metric definitions (incl. penalised correlation criteria)
- [x] Two synthetic datasets (easy / hard) + Iris benchmark
- [x] Model selection: penalised `loglik_aic`/`loglik_bic` + NSGA-II demo
- [x] Empirical dataset 1: GSE183947 breast-cancer co-expression modules
- [x] Auto-generated, re-runnable results macros/tables
- [ ] Empirical dataset 2 (a second set to be added)
- [ ] Finalise author list and affiliations
