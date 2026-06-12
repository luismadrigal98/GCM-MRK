# GCM-MRK paper

LaTeX source and reproduction code for the GCM-MRK methodological paper,
formatted for *PLOS Computational Biology* using the official PLOS template.

## Files

- `gcmrk_paper.tex` — the manuscript.
- `references.bib` — bibliography.
- `plos2025.bst` — PLOS BibTeX style (required by the template).
- `experiments/run_experiments.py` — runs all benchmarks and writes the
  results table (`results/`) and figures (`figures/`).
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

## Status / TODO

- [x] Tool description and metric definitions
- [x] Two synthetic datasets (easy / hard) + Iris benchmark
- [x] Model-selection (NSGA-II loglik vs BIC) demonstration
- [ ] Two empirical expression datasets (placeholders in the Results section)
- [ ] Finalise author list and affiliations
