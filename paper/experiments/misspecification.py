"""Does GCM-MRK's advantage survive when its generative model is wrong?

The manuscript's benchmark simulates data from a one-factor block model with
:math:`\\pm 1` loadings --- precisely the model whose profile log-likelihood the
``loglik`` objective maximizes.  Winning there shows the optimizer reaches the
likelihood optimum; it does not show the model describes real co-expression
data.  This script re-runs the comparison on five generators that each violate
an assumption of that model (see :mod:`simulators`), holding every method,
seed and metric fixed.

Reading the result
------------------
Leading on ``onefactor`` alone would mean the advantage is an artifact of the
simulator; leading throughout would mean the objective is robust to
misspecification.  The informative outcome is between the two, since it
identifies *which* structural assumptions matter to a reader deciding whether the
objective suits their data.  The reported within- minus between-module
correlation contrast bounds how much any generator can discriminate.

Outputs (../results, ../figures):
  results/misspecification.json         full results + metadata
  results/misspec_table.tex             LaTeX table body (generator x method)
  results/misspec_macros.tex            scalar macros for the manuscript
  figures/misspecification.pdf          grouped bars, ARI by generator

Run from repo root:
    PYTHONPATH=. python3 paper/experiments/misspecification.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import FIG_WIDTH  # noqa: E402  (sets pdf.fonttype=42)

from sklearn.metrics import adjusted_rand_score as ARI

from gcmrk import cluster, cluster_factor_auto, simulate_modular_data
from gcmrk.data import normalize_data, pearson_correlation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as B  # noqa: E402
import simulators as S  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGURES = HERE.parent / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

N_SEEDS = 20
TAG = ""          # set from --tag; suffixes the output filenames
MODULES = [15, 20, 25, 30, 10]
N_SAMPLES = 50
GA = dict(generations=100, population_size=150, seed=0)

#: a reduced WGCNA grid keeps this sweep tractable; the full grid is used in
#: benchmark_baselines.py, and the extra powers never won there.
WGCNA_POWERS = (1, 2, 3, 4, 6)
WGCNA_MIN_SIZES = (5, 10)

#: `noise` is calibrated per generator so that the within-minus-between
#: absolute-correlation contrast is broadly comparable across generators
#: (0.18-0.47 at noise=1.0); difficulty is not identical, but no generator is
#: trivially easy or impossibly hard.
GENERATORS = {
    "onefactor": lambda sizes, noise, seed: simulate_modular_data(
        sizes, n_samples=N_SAMPLES, noise=noise, latent_per_module=1, seed=seed),
    "multifactor": lambda sizes, noise, seed: S.simulate_multifactor(
        sizes, n_samples=N_SAMPLES, noise=noise, seed=seed),
    "hub": lambda sizes, noise, seed: S.simulate_hub(
        sizes, n_samples=N_SAMPLES, noise=noise, seed=seed),
    "overlapping": lambda sizes, noise, seed: S.simulate_overlapping(
        sizes, n_samples=N_SAMPLES, noise=noise, seed=seed),
    "counts": lambda sizes, noise, seed: S.simulate_counts(
        sizes, n_samples=N_SAMPLES, noise=noise, seed=seed),
    "network": lambda sizes, noise, seed: S.simulate_network(
        sizes, n_samples=N_SAMPLES, noise=noise, seed=seed),
}

GEN_ORDER = ["onefactor", "multifactor", "hub", "overlapping", "counts", "network"]

GEN_LABELS = {
    "onefactor": "one-factor $\\pm$ (GCM's model)",
    "multifactor": "multi-factor, continuous loadings",
    "hub": "hub-and-spoke",
    "overlapping": "overlapping modules",
    "counts": "negative-binomial counts",
    "network": "network GMRF (LFR)",
}

#: candidate ranks for the factor branch; BIC picks among them per dataset
FACTOR_RANKS = (1, 2, 3)

METHOD_LABELS = {
    "memetic": "\\tool{} (memetic)",
    "factor_auto": "\\tool{} (\\texttt{loglik\\_factor}, rank by BIC)",
    "hier_avg": "hierarchical (average)",
    "kmeans": "k-means",
    "spectral": "spectral",
    "louvain": "Louvain",
    "leiden": "Leiden",
    "mcl": "MCL",
    "wgcna_oracle": "WGCNA (oracle-tuned)",
}

FIG_LABELS = {
    "memetic": "GCM (memetic)",
    "factor_auto": "GCM (loglik_factor, BIC rank)",
    "hier_avg": "hierarchical (average)",
    "kmeans": "k-means",
    "spectral": "spectral",
    "louvain": "Louvain",
    "leiden": "Leiden",
    "mcl": "MCL",
    "wgcna_oracle": "WGCNA (oracle-tuned)",
}

METHOD_ORDER = ["memetic", "factor_auto", "hier_avg", "kmeans", "spectral",
                "louvain", "leiden", "mcl", "wgcna_oracle"]

KNOWN_K = {
    "hier_avg": B.hierarchical_average,
    "kmeans": B.kmeans,
    "spectral": B.spectral,
    "louvain": B.louvain,
    "leiden": B.leiden,
    "mcl": B.mcl,
}


# --------------------------------------------------------------------------- #
def contrast(X, t):
    """Within- minus between-module mean absolute correlation."""
    cor = pearson_correlation(normalize_data(X, by_sample=True), rowvar=True)
    same = t[:, None] == t[None, :]
    np.fill_diagonal(same, False)
    off = ~np.eye(len(t), dtype=bool)
    return float(np.abs(cor[same]).mean() - np.abs(cor[off & ~same]).mean())


def run_generator(name, n_seeds, noise):
    """Every method on ``n_seeds`` datasets from one generator.

    The number of planted modules is read from each dataset's own labels: the
    network generator does not always produce exactly ``len(MODULES)``
    communities, and every k-consuming method must be given the truth for that
    dataset rather than a nominal constant.
    """
    per_method = {m: [] for m in METHOD_ORDER}
    ranks_picked = []
    ks, contrasts = [], []

    datasets = []
    for seed in range(n_seeds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X, t = GENERATORS[name](MODULES, noise, seed)
        datasets.append((X, t))
        ks.append(int(len(np.unique(t))))
        contrasts.append(contrast(X, t))

    t0 = time.time()
    wg = B.wgcna_batch([X for X, _ in datasets],
                       powers=WGCNA_POWERS, min_sizes=WGCNA_MIN_SIZES)
    print(f"    WGCNA batch: {time.time() - t0:.0f}s", flush=True)

    for i, (X, t) in enumerate(datasets):
        k_true = int(len(np.unique(t)))
        Xn = normalize_data(X, by_sample=True)
        cor = pearson_correlation(Xn, rowvar=True)

        per_method["memetic"].append(
            ARI(t, cluster(X, targets=["loglik"], g_max=k_true, **GA).labels))
        fa = cluster_factor_auto(X, ranks=FACTOR_RANKS, g_max=k_true, **GA)
        per_method["factor_auto"].append(ARI(t, fa.labels))
        ranks_picked.append(int(fa.scores["factor_rank"]))
        for m, fn in KNOWN_K.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                per_method[m].append(ARI(t, fn(cor, Xn, k_true,
                                               np.random.default_rng(0))))
        lab, _ = B.wgcna_oracle(wg[i], t, ARI)
        per_method["wgcna_oracle"].append(ARI(t, lab))

    return {"ari": per_method, "k_true": ks, "contrast": contrasts,
            "factor_rank": ranks_picked}


# --------------------------------------------------------------------------- #
def ms(x):
    return float(np.mean(x)), float(np.std(x))


def write_latex(data):
    res = data["results"]
    gens = [g for g in GEN_ORDER if g in res]

    rows = ["\\begin{tabular}{@{}l" + "r" * len(gens) + "@{}}", "\\hline",
            "Method & " + " & ".join("\\rotatebox{60}{" + GEN_LABELS[g] + "}"
                                     for g in gens) + " \\\\", "\\hline"]
    for m in METHOD_ORDER:
        cells = []
        for g in gens:
            mu, sd = ms(res[g]["ari"][m])
            best = max(ms(res[g]["ari"][x])[0] for x in METHOD_ORDER)
            cell = f"{mu:.2f}$\\pm${sd:.2f}"
            if mu >= best - 1e-9:
                cell = "\\textbf{" + cell + "}"
            cells.append(cell)
        rows.append(f"{METHOD_LABELS[m]} & " + " & ".join(cells) + " \\\\")
    rows += ["\\hline", "\\end{tabular}"]
    (RESULTS / f"misspec_table{TAG}.tex").write_text("\n".join(rows) + "\n")

    M = ""
    short = {"onefactor": "OneFac", "multifactor": "MultiFac", "hub": "Hub",
             "overlapping": "Overlap", "counts": "Counts", "network": "Network"}
    # "best rival" means the best *competing* method: both of our own arms are
    # excluded, so the manuscript never reports GCM as its own strongest rival.
    ours = {"memetic", "factor_auto"}
    n_wins = 0
    for g in gens:
        gm, _ = ms(res[g]["ari"]["memetic"])
        rival = max((ms(res[g]["ari"][m])[0], m)
                    for m in METHOD_ORDER if m not in ours)
        M += f"\\newcommand{{\\Misspec{short[g]}GCM}}{{{gm:.2f}}}\n"
        M += f"\\newcommand{{\\Misspec{short[g]}Best}}{{{rival[0]:.2f}}}\n"
        # prose form, so the name reads naturally inside a sentence
        prose = {"wgcna_oracle": "oracle-tuned WGCNA"}.get(rival[1], METHOD_LABELS[rival[1]])
        M += f"\\newcommand{{\\Misspec{short[g]}BestName}}{{{prose}}}\n"
        if "factor_auto" in res[g]["ari"]:
            fa, _ = ms(res[g]["ari"]["factor_auto"])
            M += f"\\newcommand{{\\Misspec{short[g]}Factor}}{{{fa:.2f}}}\n"
        if g in res and "factor_rank" in res[g]:
            # ranks are integers; print one only when every seed agreed
            ranks = res[g]["factor_rank"]
            r = (f"{ranks[0]}" if len(set(ranks)) == 1
                 else f"{np.mean(ranks):.1f} on average")
            M += f"\\newcommand{{\\Misspec{short[g]}Rank}}{{{r}}}\n"
        if gm >= rival[0]:
            n_wins += 1
    M += f"\\newcommand{{\\MisspecWins}}{{{n_wins}}}\n"
    M += f"\\newcommand{{\\MisspecTotal}}{{{len(gens)}}}\n"
    M += f"\\newcommand{{\\MisspecSeeds}}{{{data['meta']['n_seeds']}}}\n"
    (RESULTS / f"misspec_macros{TAG}.tex").write_text(M)


def fig_misspec(data):
    res = data["results"]
    gens = [g for g in GEN_ORDER if g in res]
    show = ["memetic", "factor_auto", "hier_avg", "spectral", "leiden",
            "wgcna_oracle"]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.8))
    x = np.arange(len(gens))
    w = 0.8 / len(show)
    for j, m in enumerate(show):
        mu = [ms(res[g]["ari"][m])[0] for g in gens]
        sd = [ms(res[g]["ari"][m])[1] for g in gens]
        ax.bar(x + (j - len(show) / 2 + 0.5) * w, mu, w, yerr=sd, capsize=2,
               label=FIG_LABELS.get(m, m))
    ax.set_xticks(x)
    ax.set_xticklabels([GEN_LABELS[g].replace("$\\pm$", "±") for g in gens],
                       rotation=20, ha="right", fontsize=7.5)
    ax.set_ylabel("adjusted Rand index")
    ax.set_ylim(0, 1.08)
    ax.axvline(0.5, color="0.6", lw=0.8, ls=":")
    ax.text(0.02, 0.98, "model correct →| ← model misspecified",
            transform=ax.transAxes, fontsize=7, va="top", color="0.35")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    fig.savefig(FIGURES / f"misspecification{TAG}.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 seeds, for a smoke run")
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--generators", default="",
                    help="comma-separated subset of generators to run "
                         "(default: all)")
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames, to keep runs at different "
                         "noise levels side by side")
    args = ap.parse_args()
    global TAG
    TAG = args.tag
    n_seeds = 3 if args.quick else N_SEEDS

    t0 = time.time()
    gens = ([g.strip() for g in args.generators.split(",") if g.strip()]
            if args.generators else GEN_ORDER)
    results = {}
    for g in gens:
        print(f"generator: {g} ({n_seeds} seeds)", flush=True)
        results[g] = run_generator(g, n_seeds, args.noise)

    print("\n=== ARI by generator (mean over seeds) ===")
    header = f"{'method':28s}" + "".join(f"{g[:11]:>12s}" for g in gens)
    print(header)
    for m in METHOD_ORDER:
        line = f"{METHOD_LABELS[m].replace(chr(92) + 'tool{}', 'GCM-MRK'):28s}"
        line += "".join(f"{ms(results[g]['ari'][m])[0]:12.3f}" for g in gens)
        print(line)
    print(f"\n{'factor rank picked (mean)':28s}"
          + "".join(f"{np.mean(results[g]['factor_rank']):12.1f}" for g in gens))
    print(f"{'contrast (within-between |r|)':28s}"
          + "".join(f"{np.mean(results[g]['contrast']):12.3f}" for g in gens))
    print(f"{'planted k (mean)':28s}"
          + "".join(f"{np.mean(results[g]['k_true']):12.1f}" for g in gens))

    data = {"meta": {"n_seeds": n_seeds, "modules": MODULES, "noise": args.noise,
                     "n_samples": N_SAMPLES, "ga": GA, "quick": args.quick,
                     "wgcna_powers": list(WGCNA_POWERS),
                     "wgcna_min_sizes": list(WGCNA_MIN_SIZES)},
            "results": results}
    (RESULTS / f"misspecification{TAG}.json").write_text(json.dumps(data, indent=1))

    if not args.quick:
        write_latex(data)
        fig_misspec(data)
    print(f"\nDone in {time.time() - t0:.0f}s. Results -> {RESULTS}")


if __name__ == "__main__":
    main()
