# GCM-MRK paper

LaTeX source and reproduction code for the GCM-MRK methodological paper,
targeting **PeerJ**.

> **Status (2026-09-01):** desk-rejected by PLOS Comp Biol (2026-08-29) for
> lacking a state-of-the-art comparison. That comparison now exists, along with a
> model-misspecification analysis, a fix (the rank-q factor model), a scaling
> study and a comparative empirical analysis. **Venue: PeerJ.**

## Two wrappers, one manuscript

The scientific content lives in shared files that both venue wrappers include, so
the two cannot drift apart:

| file | contents |
|---|---|
| `gcmrk_macros.tex` | `\tool`, `\previewfig`, and all `\InputIfFileExists` of generated results |
| `gcmrk_abstract.tex` | abstract body (no sectioning command) |
| `gcmrk_authorsummary.tex` | PLOS-only author summary |
| `gcmrk_body.tex` | Introduction through Acknowledgments |
| **`gcmrk_paper_peerj.tex`** | **PeerJ wrapper (`wlpeerj.cls`) — the submission target** |
| `gcmrk_paper.tex` | PLOS wrapper, retained for reference |

Build either with the usual `pdflatex / bibtex / pdflatex / pdflatex` cycle.

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

Run each from the repo root with `PYTHONPATH=.`, then compile. The first three
are the originals; the last three were added for the 2026-08 revision and
provide the state-of-the-art comparison, the misspecification analysis and the
scaling table.

- `experiments/decisive.py` — the original replicated benchmark (30 seeds):
  noise sweep (memetic vs ablation vs hierarchical avg/complete vs k-means),
  model selection with unknown k, and noise-gene robustness. Superseded for the
  manuscript's main table by `benchmark_baselines.py`, but retained because it
  is far cheaper to run.
- `experiments/run_experiments.py` — single-dataset illustrative table, the
  generative-model figures (convergence, model selection, Pareto), Iris, and the
  GSE183947 empirical analysis (BH-corrected).
- `experiments/iris_validation.py` — Iris as optimizer validation.

- `experiments/baselines.py` — **not a driver**; wraps the competing methods
  (Leiden, Louvain, spectral, MCL, and real WGCNA via `Rscript`) behind one
  signature, plus the fairness protocol (resolution/inflation bisected to the
  true k; WGCNA reported at default *and* oracle-tuned settings).
- `experiments/simulators.py` — **not a driver**; the five data-generating
  processes that violate the one-factor block model (multi-factor, hub,
  overlapping, negative-binomial counts, LFR network).
- `experiments/benchmark_baselines.py` — the manuscript's main comparison
  (~2.5 h). Writes `results/baselines.json`, `results/baselines_table.tex`,
  `results/baselines_modelsel.tex`, `results/baselines_macros.tex` and
  `figures/baselines_vs_noise.pdf`.
- `experiments/misspecification.py` — recovery under model misspecification
  (~1 h per noise level). Run **twice**, tagged, because difficulty varies by
  generator and sigma=1.5 is the discriminating level:

  ```bash
  PYTHONPATH=. python3 paper/experiments/misspecification.py --noise 1.0 --tag _n10
  PYTHONPATH=. python3 paper/experiments/misspecification.py --noise 1.5 --tag _n15
  ```

  The manuscript inputs the `_n15` set. `--generators a,b` restricts the sweep.
- `experiments/scaling.py` — runtime and ARI vs gene count (~20 min). **Run it
  on an otherwise idle machine**: concurrent jobs distort the timings badly
  enough to change the fitted growth exponent. Note the table reports the
  *constrained* objective; the factor model is far more expensive on real data
  (300 genes: 14 s vs 463 s).
- `experiments/empirical_comparison.py` — GSE183947 comparison of all methods by
  coherence, GO enrichment (Enrichr, needs network), phenotype association and
  split-half reproducibility (~5 min at `--n-top 200`).
- `experiments/saelens_benchmark.py` — the real-data benchmark with curated
  ground truth (Saelens et al. 2018). Needs `data/saelens2018/data/`, unzipped
  from Zenodo record 5532578 (`data.zip`, 1.2 GB). Scores recovery/relevance/F1
  as the benchmark's authors defined them; `--list` shows available datasets.
  **~5 h for six datasets at `--n-top 500`**, almost entirely the factor arm.

### Extra dependencies for the new experiments

The package itself still depends only on NumPy/SciPy/pandas. The benchmark
harness additionally needs `scikit-learn`, `python-igraph`, `leidenalg`,
`markov_clustering`, `matplotlib`, and an R installation with `WGCNA`. A
virtualenv at the repo root (`.venv`) holds the Python side.

## Status / TODO

- [x] Generative model + likelihood derivation (S1 Appendix)
- [x] Principled correlation BIC/AIC; memetic local search
- [x] Decisive replicated benchmark with baselines + ablation (CIs)
- [x] Model selection and noise-gene robustness experiments
- [x] Iris generality benchmark
- [x] Empirical dataset 1: GSE183947, BH-corrected module-phenotype association
- [x] Auto-generated, re-runnable results macros/tables/figures
- [x] Comparison against Leiden/Louvain/spectral/MCL/WGCNA with a fairness
      protocol, paired Wilcoxon tests (2026-08 revision)
- [x] Model-misspecification analysis over six data-generating processes
- [x] Rank-q free-loading factor block model with BIC rank selection
- [x] Runtime scaling table
- [ ] Empirical dataset 2 (a second set to be added)
- [x] Comparative empirical analysis on GSE183947 (coherence, GO enrichment,
      split-half reproducibility) — **result is a wash; reported as such**
- [ ] Real citation for GSE183947; finalise author list and affiliations
- [x] Venue decided (PeerJ) and ported to `wlpeerj.cls`
- [x] Real-data benchmark with ground truth: Saelens et al. 2018, 6 datasets /
      3 organisms — GCM (`loglik_aic`) has the best mean recovery, wins 4/6,
      beats Leiden and hierarchical on 6/6, ties WGCNA
- [x] Lead method reverted to `loglik_aic`: the factor model loses on all 7 real
      datasets despite winning 6/6 synthetic ones
