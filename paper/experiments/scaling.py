"""Runtime and recovery as the number of genes grows.

WGCNA users routinely cluster 5,000-20,000 genes.  The manuscript pre-filters to
a few hundred and notes the :math:`O(n^2)` correlation cost only in the
Discussion, which invites the obvious question: does this method work at the
scale its intended users actually operate at?  This script answers it with
measured wall-clock times rather than an asymptotic argument.

Each method gets the same datasets and the true number of modules.  A method
that exceeds ``--timeout`` on a given size is recorded as ``None`` and omitted
from later sizes, so a single slow configuration cannot stall the sweep.

Outputs (../results, ../figures):
  results/scaling.json          runtimes + ARI per method per size
  results/scaling_table.tex     LaTeX table body
  results/scaling_macros.tex    scalar macros for the manuscript
  figures/scaling.pdf           runtime vs genes (log-log) and ARI vs genes

Run from repo root:
    PYTHONPATH=. python3 paper/experiments/scaling.py [--sizes 200,500,1000,2000]
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

from gcmrk import cluster, simulate_modular_data
from gcmrk.data import normalize_data, pearson_correlation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as B  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGURES = HERE.parent / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

DEFAULT_SIZES = [200, 500, 1000, 2000, 4000]
N_SAMPLES = 50
N_MODULES = 10
NOISE = 1.0
N_SEEDS = 3
GA = dict(generations=100, population_size=150, seed=0)

METHOD_LABELS = {
    "memetic": "\\tool{} (memetic)",
    "hier_avg": "hierarchical (average)",
    "leiden": "Leiden",
    "spectral": "spectral",
    "wgcna_default": "WGCNA",
}
METHOD_ORDER = ["memetic", "hier_avg", "leiden", "spectral", "wgcna_default"]


def make_data(n_genes, seed):
    """``N_MODULES`` roughly equal modules totalling ``n_genes`` rows."""
    base = n_genes // N_MODULES
    sizes = [base] * N_MODULES
    sizes[-1] += n_genes - sum(sizes)
    return simulate_modular_data(sizes, n_samples=N_SAMPLES, noise=NOISE,
                                 latent_per_module=1, seed=seed)


def timed(fn):
    t0 = time.time()
    out = fn()
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--timeout", type=float, default=3600.0,
                    help="seconds; a method exceeding this is dropped from "
                         "larger sizes")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    results = {m: {} for m in METHOD_ORDER}
    dropped = set()

    for n_genes in sizes:
        print(f"\n=== {n_genes} genes ===", flush=True)
        datasets = [make_data(n_genes, s) for s in range(args.seeds)]

        if "wgcna_default" not in dropped:
            try:
                wg, dt = timed(lambda: B.wgcna_batch(
                    [X for X, _ in datasets], powers=(6,), min_sizes=(20,)))
                per = dt / len(datasets)
                aris = [ARI(t, B.wgcna_default(wg[i]))
                        for i, (_, t) in enumerate(datasets)]
                results["wgcna_default"][n_genes] = {
                    "seconds": per, "ari": float(np.mean(aris))}
                print(f"  {'WGCNA':24s} {per:8.1f}s  ARI={np.mean(aris):.3f}", flush=True)
                if per > args.timeout:
                    dropped.add("wgcna_default")
            except Exception as exc:
                print(f"  WGCNA failed: {type(exc).__name__}: {str(exc)[:200]}")
                dropped.add("wgcna_default")

        for name in ["memetic", "hier_avg", "leiden", "spectral"]:
            if name in dropped:
                continue
            times, aris = [], []
            failed = False
            for X, t in datasets:
                k = int(len(np.unique(t)))
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if name == "memetic":
                            out, dt = timed(lambda: cluster(
                                X, targets=["loglik"], g_max=k, **GA).labels)
                        else:
                            Xn = normalize_data(X, by_sample=True)
                            cor = pearson_correlation(Xn, rowvar=True)
                            fn = {"hier_avg": B.hierarchical_average,
                                  "leiden": B.leiden,
                                  "spectral": B.spectral}[name]
                            out, dt = timed(lambda: fn(cor, Xn, k,
                                                       np.random.default_rng(0)))
                except (MemoryError, np.linalg.LinAlgError) as exc:
                    print(f"  {name:24s} failed: {type(exc).__name__}")
                    failed = True
                    break
                times.append(dt)
                aris.append(ARI(t, out))
                if sum(times) > args.timeout:
                    break
            if failed:
                dropped.add(name)
                continue
            per = float(np.mean(times))
            results[name][n_genes] = {"seconds": per, "ari": float(np.mean(aris))}
            print(f"  {METHOD_LABELS[name].replace(chr(92) + 'tool{}', 'GCM-MRK'):24s} "
                  f"{per:8.1f}s  ARI={np.mean(aris):.3f}", flush=True)
            if per > args.timeout:
                print(f"    -> exceeds {args.timeout:.0f}s budget; "
                      f"dropped from larger sizes", flush=True)
                dropped.add(name)

    data = {"meta": {"sizes": sizes, "n_samples": N_SAMPLES,
                     "n_modules": N_MODULES, "noise": NOISE,
                     "seeds": args.seeds, "ga": GA,
                     "timeout": args.timeout,
                     "dropped": sorted(dropped)},
            "results": results}
    (RESULTS / "scaling.json").write_text(json.dumps(data, indent=1))
    write_latex(data)
    fig_scaling(data)
    print(f"\nResults -> {RESULTS / 'scaling.json'}")


def write_latex(data):
    res, sizes = data["results"], data["meta"]["sizes"]
    rows = ["\\begin{tabular}{@{}l" + "r" * len(sizes) + "@{}}", "\\hline",
            "Method & " + " & ".join(f"{n}" for n in sizes) + " \\\\",
            "\\hline"]
    for m in METHOD_ORDER:
        cells = []
        for n in sizes:
            entry = res[m].get(str(n), res[m].get(n))
            cells.append("---" if entry is None else f"{entry['seconds']:.1f}")
        rows.append(f"{METHOD_LABELS[m]} & " + " & ".join(cells) + " \\\\")
    rows += ["\\hline", "\\end{tabular}"]
    (RESULTS / "scaling_table.tex").write_text("\n".join(rows) + "\n")

    M = ""
    mem = data["results"]["memetic"]
    done = sorted(int(k) for k in mem)
    if done:
        largest = max(done)
        M += f"\\newcommand{{\\ScaleMaxGenes}}{{{largest}}}\n"
        entry = mem.get(str(largest), mem.get(largest))
        M += f"\\newcommand{{\\ScaleMaxSeconds}}{{{entry['seconds']:.0f}}}\n"
        M += f"\\newcommand{{\\ScaleMaxARI}}{{{entry['ari']:.2f}}}\n"
        # empirical growth exponent over 1000 genes and up, where fixed
        # overheads no longer dominate
        big = [n for n in done if n >= 1000]
        if len(big) >= 2:
            secs = [mem.get(str(n), mem.get(n))["seconds"] for n in big]
            slope = np.polyfit(np.log(big), np.log(secs), 1)[0]
            M += f"\\newcommand{{\\ScaleExponent}}{{{slope:.2f}}}\n"
            M += f"\\newcommand{{\\ScaleExponentFrom}}{{{min(big)}}}\n"
    (RESULTS / "scaling_macros.tex").write_text(M)


def fig_scaling(data):
    res, sizes = data["results"], data["meta"]["sizes"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3.0))
    for m in METHOD_ORDER:
        xs = sorted(int(k) for k in res[m])
        if not xs:
            continue
        ys = [res[m].get(str(n), res[m].get(n))["seconds"] for n in xs]
        aa = [res[m].get(str(n), res[m].get(n))["ari"] for n in xs]
        lab = METHOD_LABELS[m].replace("\\tool{}", "GCM")
        ax1.plot(xs, ys, "o-", label=lab, lw=1.5, ms=4)
        ax2.plot(xs, aa, "o-", label=lab, lw=1.5, ms=4)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("genes"); ax1.set_ylabel("seconds per dataset")
    ax2.set_xscale("log")
    ax2.set_xlabel("genes"); ax2.set_ylabel("adjusted Rand index")
    ax1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "scaling.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
