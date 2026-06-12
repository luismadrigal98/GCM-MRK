"""Reproduce all benchmark experiments reported in the GCM-MRK paper.

Runs metric-guided clustering (GCM-MRK) and a k-means baseline on:

* Synthetic-Easy   -- 3 well-separated correlated modules.
* Synthetic-Hard   -- 5 unequal, weakly-separated correlated modules.
* Iris             -- the classic 3-class geometric benchmark.

For every (dataset, method) pair it reports the number of clusters recovered,
the adjusted Rand index (ARI) and Hungarian-matched accuracy against the ground
truth, and wall-clock runtime.  It also runs an NSGA-II model-selection demo
(loglik vs BIC) on the synthetic data and saves convergence / Pareto figures.

Outputs (written next to this script under ../results and ../figures):
    results/benchmark.csv          tidy results table
    results/benchmark.json         same, with run metadata
    figures/convergence.pdf        GA fitness trajectory (synthetic-easy)
    figures/pareto.pdf             loglik-vs-BIC Pareto front (synthetic-hard)
    figures/iris_silhouette.pdf    silhouette profile of the recovered iris clusters

Run:  python3 run_experiments.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score

from gcmrk import cluster, simulate_modular_data
from gcmrk.data import normalize_data, pearson_correlation
from gcmrk.seeding import kmeans

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGURES = HERE.parent / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SEED = 20240117
GA_KW = dict(generations=120, population_size=200, seed=SEED, kmeans_restarts=2)


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def matched_accuracy(truth: np.ndarray, pred: np.ndarray) -> float:
    """Clustering accuracy under the optimal label permutation (Hungarian)."""
    truth = np.asarray(truth)
    pred = np.asarray(pred)
    t_lab = np.unique(truth)
    p_lab = np.unique(pred)
    cost = np.zeros((t_lab.size, p_lab.size), dtype=int)
    for i, t in enumerate(t_lab):
        for j, p in enumerate(p_lab):
            cost[i, j] = np.sum((truth == t) & (pred == p))
    row, col = linear_sum_assignment(-cost)
    return cost[row, col].sum() / truth.size


def n_clusters(labels: np.ndarray) -> int:
    return int(np.unique(np.asarray(labels)[np.asarray(labels) != 0]).size)


def evaluate(truth, pred, runtime, dataset, method, target, k_setting):
    return {
        "dataset": dataset,
        "method": method,
        "target": target,
        "k_setting": k_setting,
        "k_found": n_clusters(pred),
        "ari": round(float(adjusted_rand_score(truth, pred)), 4),
        "accuracy": round(float(matched_accuracy(truth, pred)), 4),
        "runtime_s": round(float(runtime), 2),
    }


def kmeans_baseline(X, k, truth, dataset):
    """k-means with the data row-standardised the same way GCM-MRK normalises it."""
    Xn = normalize_data(X, by_sample=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    labels = kmeans(Xn, k, rng, n_init=10, max_iter=100)
    dt = time.perf_counter() - t0
    return evaluate(truth, labels, dt, dataset, "k-means", "inertia", f"k={k}")


def gcmrk_run(X, targets, g_max, truth, dataset, mode="weighted", by_sample=True):
    t0 = time.perf_counter()
    res = cluster(X, targets=list(targets), g_max=g_max, mode=mode,
                  by_sample=by_sample, **GA_KW)
    dt = time.perf_counter() - t0
    row = evaluate(truth, res.labels, dt, dataset,
                   "GCM-MRK" + ("/nsga2" if mode == "nsga2" else ""),
                   "+".join(targets), f"g_max={g_max}")
    return row, res


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def make_synthetic_easy():
    # Single shared latent factor per module ("one regulator drives the module"):
    # within-module genes are strongly correlated (in absolute value), between
    # modules they are not.  Well separated, low noise.
    X, truth = simulate_modular_data([25, 25, 25], n_samples=80, noise=0.7,
                                     latent_per_module=1, seed=SEED)
    return X, truth, 3


def make_synthetic_hard():
    # Five unequal, weakly separated modules: fewer samples and higher noise
    # shrink within-module correlation toward the between-module level.
    X, truth = simulate_modular_data([15, 20, 25, 30, 10], n_samples=50, noise=1.5,
                                     latent_per_module=1, seed=SEED)
    return X, truth, 5


def make_iris():
    iris = load_iris()
    return iris.data.astype(float), iris.target, 3


def module_separation(X, truth):
    """Mean within- vs between-module absolute correlation (difficulty proxy)."""
    cor = pearson_correlation(X, rowvar=True)
    n = len(truth)
    same = truth[:, None] == truth[None, :]
    iu = np.triu_indices(n, 1)
    c = np.abs(cor[iu])
    s = same[iu]
    return float(c[s].mean()), float(c[~s].mean())


# --------------------------------------------------------------------------- #
# Main experiment driver
# --------------------------------------------------------------------------- #
def main():
    rows = []
    meta = {"seed": SEED, "ga": GA_KW}

    # ---- Synthetic-Easy --------------------------------------------------- #
    Xe, te, ke = make_synthetic_easy()
    w, b = module_separation(Xe, te)
    meta["synthetic_easy"] = {"shape": list(Xe.shape), "k_true": ke,
                              "within_r": round(w, 3), "between_r": round(b, 3)}
    rows.append(kmeans_baseline(Xe, ke, te, "Synthetic-Easy"))
    r, res_easy = gcmrk_run(Xe, ["loglik"], ke, te, "Synthetic-Easy")
    rows.append(r)
    rows.append(gcmrk_run(Xe, ["silhouette"], ke, te, "Synthetic-Easy")[0])

    # ---- Synthetic-Hard --------------------------------------------------- #
    Xh, th, kh = make_synthetic_hard()
    w, b = module_separation(Xh, th)
    meta["synthetic_hard"] = {"shape": list(Xh.shape), "k_true": kh,
                              "within_r": round(w, 3), "between_r": round(b, 3)}
    rows.append(kmeans_baseline(Xh, kh, th, "Synthetic-Hard"))
    rows.append(gcmrk_run(Xh, ["loglik"], kh, th, "Synthetic-Hard")[0])
    rows.append(gcmrk_run(Xh, ["silhouette"], kh, th, "Synthetic-Hard")[0])
    # Model selection without knowing k: NSGA-II loglik vs BIC, loose g_max.
    r_ms, res_ms = gcmrk_run(Xh, ["loglik", "bic"], 10, th, "Synthetic-Hard",
                             mode="nsga2")
    rows.append(r_ms)

    # ---- Iris ------------------------------------------------------------- #
    Xi, ti, ki = make_iris()
    meta["iris"] = {"shape": list(Xi.shape), "k_true": ki}
    rows.append(kmeans_baseline(Xi, ki, ti, "Iris"))
    # Iris is geometric/low-dimensional -> use geometric targets.
    r_sil, res_iris = gcmrk_run(Xi, ["silhouette"], ki, ti, "Iris", by_sample=False)
    rows.append(r_sil)
    rows.append(gcmrk_run(Xi, ["calinski_harabasz"], ki, ti, "Iris",
                          by_sample=False)[0])
    # Model selection on iris: silhouette vs DB, loose g_max.
    rows.append(gcmrk_run(Xi, ["silhouette", "davies_bouldin"], 8, ti, "Iris",
                          mode="nsga2", by_sample=False)[0])

    # ---- Write tidy results ---------------------------------------------- #
    fieldnames = ["dataset", "method", "target", "k_setting", "k_found",
                  "ari", "accuracy", "runtime_s"]
    import csv
    with open(RESULTS / "benchmark.csv", "w", newline="") as fh:
        w_csv = csv.DictWriter(fh, fieldnames=fieldnames)
        w_csv.writeheader()
        w_csv.writerows(rows)
    with open(RESULTS / "benchmark.json", "w") as fh:
        json.dump({"meta": meta, "rows": rows}, fh, indent=2)

    print(f"{'dataset':16s} {'method':14s} {'target':22s} {'k':>3s} "
          f"{'ARI':>6s} {'acc':>6s} {'t(s)':>6s}")
    for r in rows:
        print(f"{r['dataset']:16s} {r['method']:14s} {r['target']:22s} "
              f"{r['k_found']:3d} {r['ari']:6.3f} {r['accuracy']:6.3f} "
              f"{r['runtime_s']:6.1f}")

    # ---- Figures ---------------------------------------------------------- #
    make_convergence_figure(res_easy)
    make_pareto_figure(res_ms)
    make_iris_silhouette_figure(Xi, res_iris.labels)
    print(f"\nWrote results to {RESULTS} and figures to {FIGURES}")


def make_convergence_figure(res):
    gens = [h["gen"] for h in res.history]
    best = [h["max"] for h in res.history]
    avg = [h["avg"] for h in res.history]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(gens, best, label="best", lw=1.8)
    ax.plot(gens, avg, label="population mean", lw=1.2, ls="--")
    ax.set_xlabel("generation")
    ax.set_ylabel("weighted fitness (loglik)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "convergence.pdf")
    plt.close(fig)


def make_pareto_figure(res):
    if not res.pareto_front:
        return
    ll = [s["scores"]["loglik"] for s in res.pareto_front]
    bic = [s["scores"]["bic"] for s in res.pareto_front]
    kk = [n_clusters(s["labels"]) for s in res.pareto_front]
    order = np.argsort(ll)
    ll = np.array(ll)[order]; bic = np.array(bic)[order]; kk = np.array(kk)[order]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    sc = ax.scatter(ll, bic, c=kk, cmap="viridis", s=40, edgecolor="k", lw=0.4)
    ax.set_xlabel("correlation log-likelihood (maximise)")
    ax.set_ylabel("BIC (minimise)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("clusters")
    fig.tight_layout()
    fig.savefig(FIGURES / "pareto.pdf")
    plt.close(fig)


def make_iris_silhouette_figure(X, labels):
    from scipy.spatial.distance import squareform, pdist
    Xn = normalize_data(X, by_sample=False)
    D = squareform(pdist(Xn))
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    sil = np.zeros(len(labels))
    for i in range(len(labels)):
        own = labels == labels[i]
        a = D[i, own].sum() / max(1, own.sum() - 1)
        b = min(D[i, labels == c].mean() for c in uniq if c != labels[i])
        sil[i] = (b - a) / max(a, b)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    y = 0
    for c in uniq:
        vals = np.sort(sil[labels == c])
        ax.barh(range(y, y + len(vals)), vals, height=1.0)
        y += len(vals) + 5
    ax.axvline(sil.mean(), color="k", ls="--", lw=1,
               label=f"mean = {sil.mean():.2f}")
    ax.set_xlabel("silhouette coefficient")
    ax.set_ylabel("samples (grouped by cluster)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "iris_silhouette.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
