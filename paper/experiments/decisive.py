"""Decisive multi-seed benchmark for the GCM-MRK paper.

Establishes, with replication and confidence intervals, where memetic global
optimization of the correlation block-likelihood beats greedy alternatives.

Experiments
-----------
1. Noise sweep (known k): ARI vs noise for
   memetic GCM-MRK, GCM-MRK without local search (ablation),
   hierarchical average / complete linkage on 1-|R|, and k-means.
2. Model selection (unknown k): memetic loglik_aic vs hierarchical + penalized
   cut; recovered k and ARI.
3. Noise-gene robustness: modules embedded among unstructured "noise" genes;
   does the objective isolate the noise rather than pollute real modules?

All conditions are replicated over many random seeds; we report mean and SD.

Outputs (../results, ../figures):
  results/decisive.json                full results + metadata
  results/perf_table.tex               LaTeX table body (noise sweep)
  results/modelsel_table.tex           LaTeX table body (model selection)
  results/decisive_macros.tex          scalar macros for the manuscript
  figures/perf_vs_noise.pdf            ARI vs noise with CI bands
  figures/ablation.pdf                 contribution of local search

Run from repo root:  PYTHONPATH=. python3 paper/experiments/decisive.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import FIG_WIDTH  # noqa: E402  (sets pdf.fonttype=42)

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score as ARI

from gcmrk import cluster, simulate_modular_data
from gcmrk.data import normalize_data, pearson_correlation
from gcmrk.metrics import loglik_aic
from gcmrk.seeding import kmeans

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGURES = HERE.parent / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

N_SEEDS = 30
MODULES = [15, 20, 25, 30, 10]          # 5 unequal modules
K_TRUE = len(MODULES)
N_SAMPLES = 50
NOISE_LEVELS = [0.7, 1.0, 1.5, 2.0]
GA = dict(generations=100, population_size=150, seed=0)   # GA seed fixed; data seed varies


# --------------------------------------------------------------------------- #
def abscor_distance(cor):
    d = 1.0 - np.abs(cor)
    np.fill_diagonal(d, 0.0)
    return (d + d.T) / 2.0


def hierarchical(cor, k, method):
    Z = linkage(squareform(abscor_distance(cor), checks=False), method=method)
    return fcluster(Z, t=k, criterion="maxclust")


def hierarchical_select(cor, n_samples, g_max, method):
    """Cut the dendrogram at the k minimizing the penalized correlation AIC."""
    Z = linkage(squareform(abscor_distance(cor), checks=False), method=method)
    best_k, best_lab, best_ic = None, None, np.inf
    for k in range(2, g_max + 1):
        lab = fcluster(Z, t=k, criterion="maxclust")
        ic = loglik_aic(cor, lab, n_samples)
        if ic < best_ic:
            best_ic, best_k, best_lab = ic, k, lab
    return best_lab


def make_data(noise, seed, noise_genes=0):
    X, t = simulate_modular_data(MODULES, n_samples=N_SAMPLES, noise=noise,
                                 latent_per_module=1, seed=seed)
    if noise_genes:
        rng = np.random.default_rng(10_000 + seed)
        extra = rng.normal(0.0, 1.0, size=(noise_genes, N_SAMPLES))
        extra = (extra - extra.mean(1, keepdims=True)) / extra.std(1, keepdims=True)
        X = np.vstack([X, extra])
        t = np.concatenate([t, np.zeros(noise_genes, dtype=int)])  # 0 = background
    return X, t


# --------------------------------------------------------------------------- #
def exp_noise_sweep():
    methods = ["memetic", "no_local", "hier_avg", "hier_comp", "kmeans"]
    out = {m: {nl: [] for nl in NOISE_LEVELS} for m in methods}
    for noise in NOISE_LEVELS:
        for seed in range(N_SEEDS):
            X, t = make_data(noise, seed)
            Xn = normalize_data(X, by_sample=True)
            cor = pearson_correlation(Xn, rowvar=True)
            out["memetic"][noise].append(
                ARI(t, cluster(X, targets=["loglik"], g_max=K_TRUE, **GA).labels))
            out["no_local"][noise].append(
                ARI(t, cluster(X, targets=["loglik"], g_max=K_TRUE,
                               local_search=False, **GA).labels))
            out["hier_avg"][noise].append(ARI(t, hierarchical(cor, K_TRUE, "average")))
            out["hier_comp"][noise].append(ARI(t, hierarchical(cor, K_TRUE, "complete")))
            out["kmeans"][noise].append(
                ARI(t, kmeans(Xn, K_TRUE, np.random.default_rng(0), n_init=10)))
    return out


def exp_model_selection(noise=1.5, g_max=10):
    rows = {"memetic_aic": {"k": [], "ari": []},
            "hier_avg_sel": {"k": [], "ari": []}}
    for seed in range(N_SEEDS):
        X, t = make_data(noise, seed)
        cor = pearson_correlation(normalize_data(X, by_sample=True), rowvar=True)
        r = cluster(X, targets=["loglik_aic"], g_max=g_max, **GA)
        rows["memetic_aic"]["k"].append(len(set(int(x) for x in r.labels)))
        rows["memetic_aic"]["ari"].append(ARI(t, r.labels))
        lab = hierarchical_select(cor, N_SAMPLES, g_max, "average")
        rows["hier_avg_sel"]["k"].append(len(set(lab.tolist())))
        rows["hier_avg_sel"]["ari"].append(ARI(t, lab))
    return rows


def exp_noise_genes(noise=1.0, noise_genes=40):
    """Modules embedded among unstructured genes; loose g_max so methods must
    isolate the noise.  ARI over all genes (background is its own truth class)."""
    out = {"memetic": [], "hier_avg": [], "hier_comp": []}
    g_max = K_TRUE + 4
    for seed in range(N_SEEDS):
        X, t = make_data(noise, seed, noise_genes=noise_genes)
        cor = pearson_correlation(normalize_data(X, by_sample=True), rowvar=True)
        out["memetic"].append(
            ARI(t, cluster(X, targets=["loglik_aic"], g_max=g_max, **GA).labels))
        out["hier_avg"].append(ARI(t, hierarchical_select(cor, N_SAMPLES, g_max, "average")))
        out["hier_comp"].append(ARI(t, hierarchical_select(cor, N_SAMPLES, g_max, "complete")))
    return out


# --------------------------------------------------------------------------- #
def ms(x):
    return float(np.mean(x)), float(np.std(x))


def main():
    t0 = time.time()
    meta = {"n_seeds": N_SEEDS, "modules": MODULES, "n_samples": N_SAMPLES,
            "noise_levels": NOISE_LEVELS, "ga": GA}

    sweep = exp_noise_sweep()
    modelsel = exp_model_selection()
    noisegene = exp_noise_genes()

    # ---- console summary ---- #
    print("\n=== Noise sweep (known k), ARI mean(sd) ===")
    label = {"memetic": "GCM-MRK (memetic)", "no_local": "GCM-MRK (no local search)",
             "hier_avg": "hierarchical avg", "hier_comp": "hierarchical complete",
             "kmeans": "k-means"}
    for m in sweep:
        cells = " ".join(f"{np.mean(sweep[m][nl]):.2f}({np.std(sweep[m][nl]):.2f})"
                         for nl in NOISE_LEVELS)
        print(f"  {label[m]:28s} {cells}")
    print("\n=== Model selection (true k=5), unknown k ===")
    for m in modelsel:
        km, ks = ms(modelsel[m]["k"]); am, asd = ms(modelsel[m]["ari"])
        print(f"  {m:16s} k={km:.1f}({ks:.1f}) ARI={am:.3f}({asd:.3f})")
    print("\n=== Noise-gene robustness ===")
    for m in noisegene:
        am, asd = ms(noisegene[m])
        print(f"  {m:12s} ARI={am:.3f}({asd:.3f})")

    data = {"meta": meta, "noise_sweep": sweep, "model_selection": modelsel,
            "noise_genes": noisegene}
    (RESULTS / "decisive.json").write_text(json.dumps(data, indent=1))

    write_latex(data)
    fig_perf_vs_noise(sweep)
    fig_ablation(sweep)
    print(f"\nDone in {time.time()-t0:.0f}s. Results -> {RESULTS}, figures -> {FIGURES}")


# --------------------------------------------------------------------------- #
def _mac(name, val):
    return f"\\newcommand{{\\{name}}}{{{val}}}\n"


def write_latex(data):
    sweep, modelsel, ng = data["noise_sweep"], data["model_selection"], data["noise_genes"]
    label = {"memetic": "\\tool{} (memetic)", "no_local": "\\tool{} (no local search)",
             "hier_avg": "hierarchical (average)", "hier_comp": "hierarchical (complete)",
             "kmeans": "k-means"}
    order = ["memetic", "no_local", "hier_avg", "hier_comp", "kmeans"]
    nls = data["meta"]["noise_levels"]

    # performance table
    rows = ["\\begin{tabular}{@{}l" + "r" * len(nls) + "@{}}", "\\hline",
            "Method & " + " & ".join(f"$\\sigma={nl}$" for nl in nls) + " \\\\", "\\hline"]
    for m in order:
        cells = " & ".join(f"{np.mean(sweep[m][str(nl) if str(nl) in sweep[m] else nl]):.2f}"
                           f"$\\pm${np.std(sweep[m][str(nl) if str(nl) in sweep[m] else nl]):.2f}"
                           for nl in nls)
        rows.append(f"{label[m]} & {cells} \\\\")
    rows += ["\\hline", "\\end{tabular}"]
    (RESULTS / "perf_table.tex").write_text("\n".join(rows) + "\n")

    # model-selection table
    mrows = ["\\begin{tabular}{@{}lrr@{}}", "\\hline",
             "Method & $k$ recovered & ARI \\\\", "\\hline"]
    mlabel = {"memetic_aic": "\\tool{} (\\texttt{loglik\\_aic})",
              "hier_avg_sel": "hierarchical + penalized cut"}
    for m in ["memetic_aic", "hier_avg_sel"]:
        km, ks = ms(modelsel[m]["k"]); am, asd = ms(modelsel[m]["ari"])
        mrows.append(f"{mlabel[m]} & {km:.1f}$\\pm${ks:.1f} & {am:.2f}$\\pm${asd:.2f} \\\\")
    mrows += ["\\hline", "\\end{tabular}"]
    (RESULTS / "modelsel_table.tex").write_text("\n".join(mrows) + "\n")

    # scalar macros
    M = ""
    M += _mac("NSeeds", data["meta"]["n_seeds"])
    hi = max(nls)
    def cell(m, nl): return np.mean(sweep[m][nl if nl in sweep[m] else str(nl)])
    M += _mac("MemeticHardARI", f"{cell('memetic', 1.5):.2f}")
    M += _mac("NoLocalHardARI", f"{cell('no_local', 1.5):.2f}")
    M += _mac("HierAvgHardARI", f"{cell('hier_avg', 1.5):.2f}")
    M += _mac("KmeansHardARI", f"{cell('kmeans', 1.5):.2f}")
    km, _ = ms(modelsel["memetic_aic"]["k"]); am, _ = ms(modelsel["memetic_aic"]["ari"])
    M += _mac("MemeticSelK", f"{km:.1f}") + _mac("MemeticSelARI", f"{am:.2f}")
    hk, _ = ms(modelsel["hier_avg_sel"]["k"]); ha, _ = ms(modelsel["hier_avg_sel"]["ari"])
    M += _mac("HierSelK", f"{hk:.1f}") + _mac("HierSelARI", f"{ha:.2f}")
    nm, _ = ms(ng["memetic"]); nh, _ = ms(ng["hier_avg"])
    M += _mac("NoiseGeneMemeticARI", f"{nm:.2f}") + _mac("NoiseGeneHierARI", f"{nh:.2f}")
    (RESULTS / "decisive_macros.tex").write_text(M)


def _series(sweep, m, nl):
    """Fetch one cell, tolerating float or string keys.

    Keys are floats in a live run but become strings after a JSON round-trip, so
    the figures can be regenerated from ``decisive.json`` without a rerun.
    """
    cell = sweep[m]
    return cell[nl] if nl in cell else cell[str(nl)]


def fig_perf_vs_noise(sweep):
    nls = NOISE_LEVELS
    style = {"memetic": ("GCM-MRK (memetic)", "o-", "C0"),
             "no_local": ("GCM-MRK (no local search)", "s--", "C1"),
             "hier_avg": ("hierarchical (average)", "^-", "C2"),
             "hier_comp": ("hierarchical (complete)", "v-", "C3"),
             "kmeans": ("k-means", "d-", "C4")}
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.2))
    for m, (lab, ls, c) in style.items():
        mean = np.array([np.mean(_series(sweep, m, nl)) for nl in nls])
        sd = np.array([np.std(_series(sweep, m, nl)) for nl in nls])
        ax.plot(nls, mean, ls, color=c, label=lab, lw=1.6, ms=4)
        ax.fill_between(nls, mean - sd, mean + sd, color=c, alpha=0.15)
    ax.set_xlabel("noise level $\\sigma$")
    ax.set_ylabel("adjusted Rand index")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "perf_vs_noise.pdf"); plt.close(fig)


def fig_ablation(sweep):
    nls = NOISE_LEVELS
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.2))
    x = np.arange(len(nls)); w = 0.38
    mem = [np.mean(_series(sweep, "memetic", nl)) for nl in nls]
    nol = [np.mean(_series(sweep, "no_local", nl)) for nl in nls]
    mem_sd = [np.std(_series(sweep, "memetic", nl)) for nl in nls]
    nol_sd = [np.std(_series(sweep, "no_local", nl)) for nl in nls]
    ax.bar(x - w/2, mem, w, yerr=mem_sd, capsize=3, label="with local search")
    ax.bar(x + w/2, nol, w, yerr=nol_sd, capsize=3, label="GA only")
    ax.set_xticks(x); ax.set_xticklabels([f"{nl}" for nl in nls])
    ax.set_xlabel("noise level $\\sigma$"); ax.set_ylabel("adjusted Rand index")
    # the bars fill every corner, so the legend goes above the axes
    ax.legend(frameon=False, fontsize=8, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout(); fig.savefig(FIGURES / "ablation.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
