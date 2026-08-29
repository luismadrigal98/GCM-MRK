"""Extended benchmark: GCM-MRK against the state-of-the-art module detectors.

``decisive.py`` compares GCM-MRK only with agglomerative linkage and k-means.
That is the comparison an editor will call incomplete, because the methods that
actually dominate module detection in practice --- Louvain, Leiden, spectral
clustering, Markov clustering and WGCNA itself --- are absent.  This script adds
them on exactly the same datasets, seeds and metrics, so the two sets of numbers
are directly comparable.

Fairness protocol
-----------------
Every method is given the most favourable setting we can construct:

* Methods that accept a cluster count (k-means, spectral, hierarchical) are
  handed the true ``k``.
* Resolution-based methods (Louvain, Leiden) and Markov clustering do not accept
  a cluster count, so their resolution / inflation is bisected until they return
  the true ``k``.
* WGCNA is reported twice: at the settings its own ``pickSoftThreshold``
  recommends (what a practitioner gets), and at the best of a
  ``power x minModuleSize`` grid *scored against ground truth* --- an oracle no
  real user has.  If GCM-MRK still leads the oracle arm, the margin cannot be
  blamed on WGCNA hyper-parameters.

Experiments
-----------
1. Noise sweep with ``k`` known.
2. Model selection with ``k`` unknown (every method runs free).
3. Robustness to unstructured background genes.

Outputs (../results, ../figures):
  results/baselines.json            full results + metadata
  results/baselines_table.tex       LaTeX table body (noise sweep, all methods)
  results/baselines_modelsel.tex    LaTeX table body (unknown k)
  results/baselines_macros.tex      scalar macros for the manuscript
  figures/baselines_vs_noise.pdf    ARI vs noise, all methods

Run from repo root:
    PYTHONPATH=. python3 paper/experiments/benchmark_baselines.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score as ARI

from gcmrk import cluster, simulate_modular_data
from gcmrk.data import normalize_data, pearson_correlation
from gcmrk.metrics import loglik_aic

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as B  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGURES = HERE.parent / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

N_SEEDS = 30
MODULES = [15, 20, 25, 30, 10]
K_TRUE = len(MODULES)
N_SAMPLES = 50
NOISE_LEVELS = [0.7, 1.0, 1.5, 2.0]
GA = dict(generations=100, population_size=150, seed=0)

#: methods evaluated with the true k supplied (or bisected to)
KNOWN_K_METHODS = {
    "hier_avg": B.hierarchical_average,
    "hier_comp": B.hierarchical_complete,
    "kmeans": B.kmeans,
    "spectral": B.spectral,
    "louvain": B.louvain,
    "leiden": B.leiden,
    "mcl": B.mcl,
}

LABELS = {
    "memetic": "\\tool{} (memetic)",
    "no_local": "\\tool{} (no local search)",
    "hier_avg": "hierarchical (average)",
    "hier_comp": "hierarchical (complete)",
    "kmeans": "k-means",
    "spectral": "spectral",
    "louvain": "Louvain",
    "leiden": "Leiden",
    "mcl": "MCL",
    "wgcna_default": "WGCNA (default settings)",
    "wgcna_oracle": "WGCNA (oracle-tuned)",
}

ORDER = ["memetic", "no_local", "hier_avg", "hier_comp", "kmeans", "spectral",
         "louvain", "leiden", "mcl", "wgcna_default", "wgcna_oracle"]


# --------------------------------------------------------------------------- #
def make_data(noise, seed, noise_genes=0):
    X, t = simulate_modular_data(MODULES, n_samples=N_SAMPLES, noise=noise,
                                 latent_per_module=1, seed=seed)
    if noise_genes:
        rng = np.random.default_rng(10_000 + seed)
        extra = rng.normal(0.0, 1.0, size=(noise_genes, N_SAMPLES))
        extra = (extra - extra.mean(1, keepdims=True)) / extra.std(1, keepdims=True)
        X = np.vstack([X, extra])
        t = np.concatenate([t, np.zeros(noise_genes, dtype=int)])
    return X, t


def hierarchical_select(cor, n_samples, g_max, method):
    """Cut the dendrogram at the k minimizing the penalized correlation AIC."""
    Z = linkage(squareform(B.abscor_distance(cor), checks=False), method=method)
    best_lab, best_ic = None, np.inf
    for k in range(2, g_max + 1):
        lab = fcluster(Z, t=k, criterion="maxclust")
        ic = loglik_aic(cor, lab, n_samples)
        if ic < best_ic:
            best_ic, best_lab = ic, lab
    return best_lab


def _nk(labels):
    return int(len(np.unique(labels)))


# --------------------------------------------------------------------------- #
def exp_noise_sweep(n_seeds, noise_levels):
    """ARI vs noise with k known, for every method."""
    out = {m: {nl: [] for nl in noise_levels} for m in ORDER}
    wgcna_cfg = {nl: [] for nl in noise_levels}

    for noise in noise_levels:
        print(f"  noise sigma={noise} ...", flush=True)
        datasets = [make_data(noise, s) for s in range(n_seeds)]

        # WGCNA for every seed at this noise level, in one R process
        t0 = time.time()
        wg = B.wgcna_batch([X for X, _ in datasets])
        print(f"    WGCNA batch: {time.time() - t0:.0f}s", flush=True)

        for i, (X, t) in enumerate(datasets):
            Xn = normalize_data(X, by_sample=True)
            cor = pearson_correlation(Xn, rowvar=True)
            rng = np.random.default_rng(0)

            out["memetic"][noise].append(
                ARI(t, cluster(X, targets=["loglik"], g_max=K_TRUE, **GA).labels))
            out["no_local"][noise].append(
                ARI(t, cluster(X, targets=["loglik"], g_max=K_TRUE,
                               local_search=False, **GA).labels))

            for name, fn in KNOWN_K_METHODS.items():
                out[name][noise].append(ARI(t, fn(cor, Xn, K_TRUE,
                                                  np.random.default_rng(0))))

            out["wgcna_default"][noise].append(ARI(t, B.wgcna_default(wg[i])))
            lab, cfg = B.wgcna_oracle(wg[i], t, ARI)
            out["wgcna_oracle"][noise].append(ARI(t, lab))
            wgcna_cfg[noise].append({"power": cfg["power"],
                                     "min_module_size": cfg["min_module_size"],
                                     "auto_power": wg[i]["auto_power"]})
    return out, wgcna_cfg


def exp_model_selection(n_seeds, noise=1.5, g_max=10):
    """Every method runs free: no ground-truth k, no oracle tuning."""
    names = ["memetic_aic", "hier_avg_sel", "louvain_free", "leiden_free",
             "mcl_free", "wgcna_default"]
    rows = {m: {"k": [], "ari": []} for m in names}

    datasets = [make_data(noise, s) for s in range(n_seeds)]
    wg = B.wgcna_batch([X for X, _ in datasets])

    for i, (X, t) in enumerate(datasets):
        Xn = normalize_data(X, by_sample=True)
        cor = pearson_correlation(Xn, rowvar=True)

        r = cluster(X, targets=["loglik_aic"], g_max=g_max, **GA)
        rows["memetic_aic"]["k"].append(_nk(r.labels))
        rows["memetic_aic"]["ari"].append(ARI(t, r.labels))

        lab = hierarchical_select(cor, N_SAMPLES, g_max, "average")
        rows["hier_avg_sel"]["k"].append(_nk(lab))
        rows["hier_avg_sel"]["ari"].append(ARI(t, lab))

        for name, fn in [("louvain_free", B.louvain), ("leiden_free", B.leiden),
                         ("mcl_free", B.mcl)]:
            lab = fn(cor, Xn, None, np.random.default_rng(0))
            rows[name]["k"].append(_nk(lab))
            rows[name]["ari"].append(ARI(t, lab))

        lab = B.wgcna_default(wg[i])
        rows["wgcna_default"]["k"].append(_nk(lab))
        rows["wgcna_default"]["ari"].append(ARI(t, lab))
    return rows


def exp_noise_genes(n_seeds, noise=1.0, noise_genes=40):
    """Modules embedded among unstructured genes, loose g_max."""
    names = ["memetic", "hier_avg", "hier_comp", "leiden_free", "wgcna_default"]
    out = {m: [] for m in names}
    g_max = K_TRUE + 4

    datasets = [make_data(noise, s, noise_genes=noise_genes) for s in range(n_seeds)]
    wg = B.wgcna_batch([X for X, _ in datasets])

    for i, (X, t) in enumerate(datasets):
        Xn = normalize_data(X, by_sample=True)
        cor = pearson_correlation(Xn, rowvar=True)
        out["memetic"].append(
            ARI(t, cluster(X, targets=["loglik_aic"], g_max=g_max, **GA).labels))
        out["hier_avg"].append(ARI(t, hierarchical_select(cor, N_SAMPLES, g_max, "average")))
        out["hier_comp"].append(ARI(t, hierarchical_select(cor, N_SAMPLES, g_max, "complete")))
        out["leiden_free"].append(ARI(t, B.leiden(cor, Xn, None, np.random.default_rng(0))))
        out["wgcna_default"].append(ARI(t, B.wgcna_default(wg[i])))
    return out


# --------------------------------------------------------------------------- #
def ms(x):
    return float(np.mean(x)), float(np.std(x))


def _mac(name, val):
    return f"\\newcommand{{\\{name}}}{{{val}}}\n"


def write_latex(data):
    sweep = data["noise_sweep"]
    nls = data["meta"]["noise_levels"]

    def cell(m, nl):
        vals = sweep[m][nl if nl in sweep[m] else str(nl)]
        return np.mean(vals), np.std(vals)

    rows = ["\\begin{tabular}{@{}l" + "r" * len(nls) + "@{}}", "\\hline",
            "Method & " + " & ".join(f"$\\sigma={nl}$" for nl in nls) + " \\\\", "\\hline"]
    for m in ORDER:
        cells = " & ".join(f"{cell(m, nl)[0]:.2f}$\\pm${cell(m, nl)[1]:.2f}" for nl in nls)
        rows.append(f"{LABELS[m]} & {cells} \\\\")
    rows += ["\\hline", "\\end{tabular}"]
    (RESULTS / "baselines_table.tex").write_text("\n".join(rows) + "\n")

    msel = data["model_selection"]
    mlabel = {"memetic_aic": "\\tool{} (\\texttt{loglik\\_aic})",
              "hier_avg_sel": "hierarchical + penalized cut",
              "louvain_free": "Louvain (default resolution)",
              "leiden_free": "Leiden (default resolution)",
              "mcl_free": "MCL (default inflation)",
              "wgcna_default": "WGCNA (default settings)"}
    mrows = ["\\begin{tabular}{@{}lrr@{}}", "\\hline",
             "Method & $k$ recovered & ARI \\\\", "\\hline"]
    for m in ["memetic_aic", "hier_avg_sel", "louvain_free", "leiden_free",
              "mcl_free", "wgcna_default"]:
        km, ks = ms(msel[m]["k"])
        am, asd = ms(msel[m]["ari"])
        mrows.append(f"{mlabel[m]} & {km:.1f}$\\pm${ks:.1f} & {am:.2f}$\\pm${asd:.2f} \\\\")
    mrows += ["\\hline", "\\end{tabular}"]
    (RESULTS / "baselines_modelsel.tex").write_text("\n".join(mrows) + "\n")

    M = ""
    macro_name = {"spectral": "Spectral", "louvain": "Louvain", "leiden": "Leiden",
                  "mcl": "MCL", "wgcna_default": "WgcnaDef", "wgcna_oracle": "WgcnaOra"}
    for m, nm in macro_name.items():
        M += _mac(f"{nm}HardARI", f"{cell(m, 1.5)[0]:.2f}")
    ng = data["noise_genes"]
    M += _mac("NoiseGeneLeidenARI", f"{ms(ng['leiden_free'])[0]:.2f}")
    M += _mac("NoiseGeneWgcnaARI", f"{ms(ng['wgcna_default'])[0]:.2f}")
    for m, nm in [("leiden_free", "LeidenSel"), ("wgcna_default", "WgcnaSel")]:
        M += _mac(f"{nm}K", f"{ms(msel[m]['k'])[0]:.1f}")
        M += _mac(f"{nm}ARI", f"{ms(msel[m]['ari'])[0]:.2f}")
    (RESULTS / "baselines_macros.tex").write_text(M)


def fig_vs_noise(sweep, nls):
    style = {
        "memetic": ("GCM-MRK (memetic)", "o-", "C0"),
        "no_local": ("GCM-MRK (no local search)", "s--", "C1"),
        "hier_avg": ("hierarchical (average)", "^-", "C2"),
        "kmeans": ("k-means", "d-", "C4"),
        "spectral": ("spectral", "P-", "C5"),
        "leiden": ("Leiden", "X-", "C6"),
        "louvain": ("Louvain", "*-", "C7"),
        "mcl": ("MCL", "<-", "C8"),
        "wgcna_oracle": ("WGCNA (oracle-tuned)", ">-", "C9"),
    }
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for m, (lab, ls, c) in style.items():
        mean = np.array([np.mean(sweep[m][nl]) for nl in nls])
        sd = np.array([np.std(sweep[m][nl]) for nl in nls])
        ax.plot(nls, mean, ls, color=c, label=lab, lw=1.5, ms=4)
        ax.fill_between(nls, mean - sd, mean + sd, color=c, alpha=0.10)
    ax.set_xlabel("noise level $\\sigma$")
    ax.set_ylabel("adjusted Rand index")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "baselines_vs_noise.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="3 seeds and 2 noise levels, for a smoke run")
    args = ap.parse_args()

    n_seeds = 3 if args.quick else N_SEEDS
    noise_levels = [1.0, 1.5] if args.quick else NOISE_LEVELS

    t0 = time.time()
    print(f"Environment: {json.dumps(B.available())}", flush=True)
    print(f"Noise sweep ({n_seeds} seeds x {len(noise_levels)} levels)", flush=True)
    sweep, wgcna_cfg = exp_noise_sweep(n_seeds, noise_levels)
    print("Model selection (unknown k)", flush=True)
    modelsel = exp_model_selection(n_seeds)
    print("Noise-gene robustness", flush=True)
    noisegene = exp_noise_genes(n_seeds)

    print("\n=== Noise sweep (known k), ARI mean(sd) ===")
    for m in ORDER:
        cells = " ".join(f"{np.mean(sweep[m][nl]):.2f}({np.std(sweep[m][nl]):.2f})"
                         for nl in noise_levels)
        print(f"  {LABELS[m].replace(chr(92) + 'tool{}', 'GCM-MRK'):30s} {cells}")

    print("\n=== Model selection (true k=5), unknown k ===")
    for m, r in modelsel.items():
        km, ks = ms(r["k"])
        am, asd = ms(r["ari"])
        print(f"  {m:16s} k={km:.1f}({ks:.1f}) ARI={am:.3f}({asd:.3f})")

    print("\n=== Noise-gene robustness ===")
    for m, v in noisegene.items():
        am, asd = ms(v)
        print(f"  {m:16s} ARI={am:.3f}({asd:.3f})")

    data = {
        "meta": {"n_seeds": n_seeds, "modules": MODULES, "n_samples": N_SAMPLES,
                 "noise_levels": noise_levels, "ga": GA, "quick": args.quick,
                 "wgcna_powers": list(B.WGCNA_POWERS),
                 "wgcna_min_sizes": list(B.WGCNA_MIN_SIZES)},
        "noise_sweep": sweep,
        "wgcna_config": {str(k): v for k, v in wgcna_cfg.items()},
        "model_selection": modelsel,
        "noise_genes": noisegene,
    }
    (RESULTS / "baselines.json").write_text(json.dumps(data, indent=1))

    if not args.quick:
        write_latex(data)
        fig_vs_noise(sweep, noise_levels)
    print(f"\nDone in {time.time() - t0:.0f}s. Results -> {RESULTS}")


if __name__ == "__main__":
    main()
