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

## Experiments

- `experiments/decisive.py` — the replicated benchmark (30 seeds): noise sweep
  (memetic vs ablation vs hierarchical avg/complete vs k-means), model selection
  with unknown k, and noise-gene robustness. Writes `results/decisive.json`,
  `results/perf_table.tex`, `results/modelsel_table.tex`,
  `results/decisive_macros.tex`, and `figures/perf_vs_noise.pdf`,
  `figures/ablation.pdf`.
- `experiments/run_experiments.py` — single-dataset illustrative table, the
  generative-model figures (convergence, model selection, Pareto), Iris, and the
  GSE183947 empirical analysis (BH-corrected). Writes `results/benchmark*.{csv,json}`,
  `results/results_macros.tex`, `results/benchmark_table.tex`,
  `results/empirical_table.tex`, and the corresponding figures.

- `experiments/iris_validation.py` — Iris as optimiser validation: writes
  `results/iris_macros.tex` with the silhouette values showing GCM-MRK reaches the
  k-means k=2 optimum and that the true 3-species labelling has a lower silhouette
  (so the k=2 outcome is the index's doing, not the optimiser's).

Run all three from the repo root with `PYTHONPATH=.`, then compile.

## Status / TODO

- [x] Generative model + likelihood derivation (S1 Appendix)
- [x] Principled correlation BIC/AIC; memetic local search
- [x] Decisive replicated benchmark with baselines + ablation (CIs)
- [x] Model selection and noise-gene robustness experiments
- [x] Iris generality benchmark
- [x] Empirical dataset 1: GSE183947, BH-corrected module-phenotype association
- [x] Auto-generated, re-runnable results macros/tables/figures
- [ ] Empirical dataset 2 (a second set to be added)
- [ ] Real citation for GSE183947; finalise author list and affiliations
