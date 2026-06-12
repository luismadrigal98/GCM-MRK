"""Reproduce all benchmark experiments reported in the GCM-MRK paper.

Datasets
--------
* Synthetic-Easy  -- 3 well-separated correlated modules (single latent factor).
* Synthetic-Hard  -- 5 unequal, weakly-separated correlated modules.
* Iris            -- the classic 3-class geometric benchmark.
* GSE183947       -- breast-cancer RNA-seq (30 tumor / 30 paired normal); gene
                     co-expression modules and their association with phenotype.

For the synthetic data and Iris it reports, per method, the number of clusters
recovered, the adjusted Rand index (ARI) and Hungarian-matched accuracy against
ground truth, and runtime.  It covers three regimes:

  1. Recovery with the number of clusters known (single-objective).
  2. Over-segmentation when ``loglik`` is optimized under a loose ``g_max``.
  3. Model selection with k unknown, via the penalized correlation criteria
     (``loglik_bic`` / ``loglik_aic``, single objective) and via NSGA-II.

Outputs (under ../results and ../figures):
  results/benchmark.csv / .json      tidy results + metadata
  results/empirical_modules.csv      gene-module summary for GSE183947
  figures/convergence.pdf            GA fitness trajectory (Synthetic-Easy)
  figures/model_selection.pdf        loglik & penalized criteria vs k (Synthetic-Hard)
  figures/pareto.pdf                 NSGA-II loglik-vs-BIC front (Synthetic-Hard)
  figures/iris_silhouette.pdf        silhouette profile of the Iris clustering
  figures/empirical_eigengenes.pdf   module eigengenes, tumor vs normal

Run from the repo root:  PYTHONPATH=. python3 paper/experiments/run_experiments.py
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_ind

from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score

from gcmrk import cluster, simulate_modular_data
from gcmrk.data import normalize_data, pearson_correlation
from gcmrk.metrics import loglik_bic, loglik_aic, log_likelihood_correlation
from gcmrk.seeding import kmeans

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGURES = HERE.parent / "figures"
DATA = HERE.parent / "data"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SEED = 20240117
GA_KW = dict(generations=120, population_size=200, seed=SEED, kmeans_restarts=2)


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def matched_accuracy(truth, pred) -> float:
    truth = np.asarray(truth); pred = np.asarray(pred)
    t_lab = np.unique(truth); p_lab = np.unique(pred)
    cost = np.zeros((t_lab.size, p_lab.size), dtype=int)
    for i, t in enumerate(t_lab):
        for j, p in enumerate(p_lab):
            cost[i, j] = np.sum((truth == t) & (pred == p))
    row, col = linear_sum_assignment(-cost)
    return cost[row, col].sum() / truth.size


def n_clusters(labels) -> int:
    a = np.asarray(labels)
    return int(np.unique(a[a != 0]).size)


def evaluate(truth, pred, runtime, dataset, method, target, k_setting):
    return {
        "dataset": dataset, "method": method, "target": target,
        "k_setting": k_setting, "k_found": n_clusters(pred),
        "ari": round(float(adjusted_rand_score(truth, pred)), 4),
        "accuracy": round(float(matched_accuracy(truth, pred)), 4),
        "runtime_s": round(float(runtime), 2),
    }


def kmeans_baseline(X, k, truth, dataset, by_sample=True):
    Xn = normalize_data(X, by_sample=by_sample)
    t0 = time.perf_counter()
    labels = kmeans(Xn, k, np.random.default_rng(SEED), n_init=10, max_iter=100)
    return evaluate(truth, labels, time.perf_counter() - t0, dataset,
                    "k-means", "WCSS", f"k={k}")


def gcmrk_run(X, targets, g_max, truth, dataset, mode="weighted", by_sample=True,
              k_setting=None):
    t0 = time.perf_counter()
    res = cluster(X, targets=list(targets), g_max=g_max, mode=mode,
                  by_sample=by_sample, **GA_KW)
    dt = time.perf_counter() - t0
    method = "GCM-MRK/nsga2" if mode == "nsga2" else "GCM-MRK"
    row = evaluate(truth, res.labels, dt, dataset, method, "+".join(targets),
                   k_setting or f"g_max={g_max}")
    return row, res


def module_separation(X, truth):
    cor = pearson_correlation(normalize_data(X, by_sample=True), rowvar=True)
    n = len(truth)
    same = truth[:, None] == truth[None, :]
    iu = np.triu_indices(n, 1)
    c = np.abs(cor[iu]); s = same[iu]
    return float(c[s].mean()), float(c[~s].mean())


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def make_synthetic_easy():
    X, truth = simulate_modular_data([25, 25, 25], n_samples=80, noise=0.7,
                                     latent_per_module=1, seed=SEED)
    return X, truth, 3


def make_synthetic_hard():
    X, truth = simulate_modular_data([15, 20, 25, 30, 10], n_samples=50, noise=1.5,
                                     latent_per_module=1, seed=SEED)
    return X, truth, 5


def make_iris():
    iris = load_iris()
    return iris.data.astype(float), iris.target, 3


# --------------------------------------------------------------------------- #
# Experiment blocks
# --------------------------------------------------------------------------- #
def synthetic_block(X, truth, k_true, name, rows, meta_key, meta, res_store):
    w, b = module_separation(X, truth)
    meta[meta_key] = {"shape": list(X.shape), "k_true": k_true,
                      "within_r": round(w, 3), "between_r": round(b, 3)}
    # baseline
    rows.append(kmeans_baseline(X, k_true, truth, name))
    # recovery with known k
    r_rec, res_rec = gcmrk_run(X, ["loglik"], k_true, truth, name,
                               k_setting=f"known k={k_true}")
    rows.append(r_rec)
    res_store[name + "_recovery"] = res_rec
    # over-segmentation: loglik under a loose bound
    rows.append(gcmrk_run(X, ["loglik"], 10, truth, name,
                          k_setting="loose g_max=10")[0])
    # model selection, single objective, loose bound
    rows.append(gcmrk_run(X, ["loglik_bic"], 10, truth, name,
                          k_setting="model sel. g_max=10")[0])
    rows.append(gcmrk_run(X, ["loglik_aic"], 10, truth, name,
                          k_setting="model sel. g_max=10")[0])


def model_selection_scan(X, truth, name, k_range=range(2, 11)):
    """Single-objective loglik at each k; report loglik and penalized criteria."""
    cor = pearson_correlation(normalize_data(X, by_sample=True), rowvar=True)
    d = X.shape[1]
    out = []
    for k in k_range:
        res = cluster(X, targets=["loglik"], g_max=k, generations=100,
                      population_size=150, seed=SEED, kmeans_restarts=2)
        out.append({
            "k": k, "k_found": n_clusters(res.labels),
            "loglik": float(log_likelihood_correlation(cor, res.labels)),
            "loglik_bic": float(loglik_bic(cor, res.labels, d)),
            "loglik_aic": float(loglik_aic(cor, res.labels, d)),
            "ari": float(adjusted_rand_score(truth, res.labels)),
        })
    return out


# --------------------------------------------------------------------------- #
# Empirical: GSE183947 breast-cancer RNA-seq
# --------------------------------------------------------------------------- #
def empirical_block(rows, meta, n_top=200, g_max=12):
    path = DATA / "GSE183947_fpkm.csv"
    if not path.exists():
        meta["empirical"] = {"status": "skipped (data file absent)"}
        return None, None
    df = pd.read_csv(path, index_col=0)
    genes = df.index.to_numpy()
    is_tumor = np.array([c.startswith("CA.") for c in df.columns])
    L = np.log2(df.values + 1.0)                      # genes x samples
    # top-variable genes
    v = L.var(axis=1)
    idx = np.argsort(v)[::-1][:n_top]
    G = L[idx]                                        # n_top genes x samples
    gnames = genes[idx]
    cor = pearson_correlation(normalize_data(G, by_sample=True), rowvar=True)
    iu = np.triu_indices(n_top, 1)
    bg = float(np.abs(cor[iu]).mean())

    # model-selection by the penalized correlation criterion
    res = cluster(G, targets=["loglik_aic"], g_max=g_max, by_sample=True, **GA_KW)
    labels = np.asarray(res.labels)
    k = n_clusters(labels)

    # per-module coherence and eigengene-phenotype association
    Gs = normalize_data(G, by_sample=True)
    module_rows = []
    eigengenes = {}
    for c in sorted(set(labels)):
        ii = np.where(labels == c)[0]
        if ii.size < 2:
            continue
        sub = np.abs(cor[np.ix_(ii, ii)])
        within = (sub.sum() - ii.size) / (ii.size ** 2 - ii.size)
        # eigengene = first principal component across samples
        block = Gs[ii]                                # genes_in_module x samples
        block = block - block.mean(axis=1, keepdims=True)
        u, s, vt = np.linalg.svd(block, full_matrices=False)
        eig = vt[0]                                   # length = n_samples
        # orient so tumor mean >= normal mean for interpretability
        if eig[is_tumor].mean() < eig[~is_tumor].mean():
            eig = -eig
        t_stat, p_val = ttest_ind(eig[is_tumor], eig[~is_tumor], equal_var=False)
        eigengenes[int(c)] = eig
        module_rows.append({
            "module": int(c), "size": int(ii.size),
            "within_abs_corr": round(float(within), 3),
            "eigengene_var_explained": round(float(s[0] ** 2 / (s ** 2).sum()), 3),
            "tumor_normal_t": round(float(t_stat), 2),
            "tumor_normal_p": float(p_val),
            "example_genes": ",".join(map(str, gnames[ii[:5]])),
        })

    # Benjamini-Hochberg FDR correction across the module tests.
    pvals = np.array([m["tumor_normal_p"] for m in module_rows])
    order = np.argsort(pvals)
    m_tests = len(pvals)
    q = np.empty(m_tests)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):           # largest p first
        i = m_tests - rank                              # 1-based rank from top
        prev = min(prev, pvals[idx] * m_tests / i)
        q[idx] = prev
    for mrow, qv in zip(module_rows, q):
        mrow["tumor_normal_q"] = float(qv)

    with open(RESULTS / "empirical_modules.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(module_rows[0].keys()))
        wr.writeheader(); wr.writerows(module_rows)

    n_sig = sum(1 for m in module_rows if m["tumor_normal_q"] < 0.05)
    meta["empirical"] = {
        "dataset": "GSE183947", "shape_full": list(df.shape),
        "n_top_genes": n_top, "g_max": g_max, "k_modules": k,
        "background_abs_corr": round(bg, 3),
        "mean_within_abs_corr": round(
            float(np.mean([m["within_abs_corr"] for m in module_rows])), 3),
        "modules_assoc_phenotype_q05": n_sig,
        "n_modules": len(module_rows),
    }
    rows.append({
        "dataset": "GSE183947", "method": "GCM-MRK", "target": "loglik_aic",
        "k_setting": f"top{n_top}, g_max={g_max}", "k_found": k,
        "ari": "", "accuracy": "", "runtime_s": "",
    })
    return module_rows, eigengenes


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_convergence(res):
    gens = [h["gen"] for h in res.history]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(gens, [h["max"] for h in res.history], label="best", lw=1.8)
    ax.plot(gens, [h["avg"] for h in res.history], label="population mean",
            lw=1.2, ls="--")
    ax.set_xlabel("generation"); ax.set_ylabel("weighted fitness (loglik)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "convergence.pdf"); plt.close(fig)


def fig_model_selection(scan, k_true):
    ks = [s["k_found"] for s in scan]
    fig, ax1 = plt.subplots(figsize=(4.6, 3.0))
    ax1.plot(ks, [s["loglik"] for s in scan], "o-", color="C0",
             label="loglik (fit)")
    ax1.set_xlabel("number of clusters k")
    ax1.set_ylabel("correlation log-likelihood", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.plot(ks, [s["loglik_bic"] for s in scan], "s--", color="C1",
             label="loglik_bic")
    ax2.plot(ks, [s["loglik_aic"] for s in scan], "^--", color="C2",
             label="loglik_aic")
    ax2.set_ylabel("penalized criterion (lower = better)")
    kbic = min(scan, key=lambda s: s["loglik_bic"])["k_found"]
    kaic = min(scan, key=lambda s: s["loglik_aic"])["k_found"]
    ax2.axvline(kaic, color="C2", lw=0.8, alpha=0.6)
    ax1.axvline(k_true, color="k", ls=":", lw=1, label=f"true k={k_true}")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=7,
               loc="center right")
    fig.tight_layout(); fig.savefig(FIGURES / "model_selection.pdf"); plt.close(fig)
    return kbic, kaic


def fig_pareto(res):
    if not res.pareto_front:
        return
    ll = np.array([s["scores"]["loglik"] for s in res.pareto_front])
    bic = np.array([s["scores"]["bic"] for s in res.pareto_front])
    kk = np.array([n_clusters(s["labels"]) for s in res.pareto_front])
    order = np.argsort(ll)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    sc = ax.scatter(ll[order], bic[order], c=kk[order], cmap="viridis", s=40,
                    edgecolor="k", lw=0.4)
    ax.set_xlabel("correlation log-likelihood (maximize)")
    ax.set_ylabel("BIC (minimize)")
    fig.colorbar(sc, ax=ax, label="clusters")
    fig.tight_layout(); fig.savefig(FIGURES / "pareto.pdf"); plt.close(fig)


def fig_iris_silhouette(X, labels):
    from scipy.spatial.distance import pdist, squareform
    Xn = normalize_data(X, by_sample=False)
    D = squareform(pdist(Xn))
    labels = np.asarray(labels); uniq = np.unique(labels)
    sil = np.zeros(len(labels))
    for i in range(len(labels)):
        own = labels == labels[i]
        a = D[i, own].sum() / max(1, own.sum() - 1)
        b = min(D[i, labels == c].mean() for c in uniq if c != labels[i])
        sil[i] = (b - a) / max(a, b)
    fig, ax = plt.subplots(figsize=(4.2, 3.0)); y = 0
    for c in uniq:
        vals = np.sort(sil[labels == c]); ax.barh(range(y, y + len(vals)), vals,
                                                   height=1.0)
        y += len(vals) + 5
    ax.axvline(sil.mean(), color="k", ls="--", lw=1,
               label=f"mean = {sil.mean():.2f}")
    ax.set_xlabel("silhouette coefficient")
    ax.set_ylabel("samples (grouped by cluster)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGURES / "iris_silhouette.pdf"); plt.close(fig)


def fig_empirical_eigengenes(module_rows, eigengenes, is_tumor):
    if not eigengenes:
        return
    # show the modules most associated with phenotype
    order = sorted(module_rows, key=lambda m: m["tumor_normal_q"])[:6]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    pos = 0; ticks = []; ticklab = []
    for m in order:
        eig = eigengenes[m["module"]]
        for grp, mask, off, col in [("T", is_tumor, 0, "C3"),
                                    ("N", ~is_tumor, 1, "C0")]:
            ax.boxplot(eig[mask], positions=[pos + off], widths=0.6,
                       patch_artist=True,
                       boxprops=dict(facecolor=col, alpha=0.6),
                       medianprops=dict(color="k"), showfliers=False)
        ticks.append(pos + 0.5)
        star = "*" if m["tumor_normal_q"] < 0.05 else ""
        ticklab.append(f"M{m['module']}{star}")
        pos += 3
    ax.set_xticks(ticks); ax.set_xticklabels(ticklab)
    ax.set_ylabel("module eigengene")
    ax.set_xlabel("module (T=tumor red, N=normal blue; * BH $q<0.05$)")
    fig.tight_layout(); fig.savefig(FIGURES / "empirical_eigengenes.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# LaTeX auto-generation: the paper \input's these so re-running updates numbers.
# --------------------------------------------------------------------------- #
def _fmt(x, nd=2):
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float) or isinstance(x, (np.floating,)):
        return f"{float(x):.{nd}f}"
    return str(x)


def _find(rows, dataset, target=None, contains=None):
    for r in rows:
        if r["dataset"] != dataset:
            continue
        if target is not None and r["target"] != target:
            continue
        if contains is not None and contains not in r["k_setting"]:
            continue
        return r
    return None


def _macro(name, value):
    # LaTeX command names must be letters only.
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def write_latex(rows, meta, module_rows, scan_hard, kbic, kaic):
    """Emit results_macros.tex (scalars) and benchmark_table.tex (table body)."""
    M = ""
    se, sh, ir = meta["synthetic_easy"], meta["synthetic_hard"], meta["iris"]
    M += _macro("EasyGenes", se["shape"][0]) + _macro("EasySamples", se["shape"][1])
    M += _macro("EasyWithinR", _fmt(se["within_r"])) + _macro("EasyBetweenR", _fmt(se["between_r"]))
    M += _macro("HardGenes", sh["shape"][0]) + _macro("HardSamples", sh["shape"][1])
    M += _macro("HardWithinR", _fmt(sh["within_r"])) + _macro("HardBetweenR", _fmt(sh["between_r"]))
    M += _macro("IrisSamples", ir["shape"][0]) + _macro("IrisFeatures", ir["shape"][1])

    def ari(row):
        return _fmt(row["ari"]) if row and isinstance(row["ari"], float) else "NA"

    M += _macro("EasyLoglikARI", ari(_find(rows, "Synthetic-Easy", "loglik", "known")))
    M += _macro("EasyKmeansARI", ari(_find(rows, "Synthetic-Easy", "WCSS")))
    M += _macro("HardLoglikARI", ari(_find(rows, "Synthetic-Hard", "loglik", "known")))
    M += _macro("HardKmeansARI", ari(_find(rows, "Synthetic-Hard", "WCSS")))

    el = _find(rows, "Synthetic-Easy", "loglik", "loose")
    hl = _find(rows, "Synthetic-Hard", "loglik", "loose")
    M += _macro("EasyLooseK", el["k_found"] if el else "NA")
    M += _macro("HardLooseK", hl["k_found"] if hl else "NA")

    for tag, ds in [("Easy", "Synthetic-Easy"), ("Hard", "Synthetic-Hard")]:
        rb = _find(rows, ds, "loglik_bic")
        ra = _find(rows, ds, "loglik_aic")
        if rb:
            M += _macro(tag + "BicK", rb["k_found"]) + _macro(tag + "BicARI", ari(rb))
        if ra:
            M += _macro(tag + "AicK", ra["k_found"]) + _macro(tag + "AicARI", ari(ra))

    M += _macro("ScanHardKbic", kbic) + _macro("ScanHardKaic", kaic)

    M += _macro("IrisKmeansARI", ari(_find(rows, "Iris", "WCSS")))
    rs = _find(rows, "Iris", "silhouette")
    rc = _find(rows, "Iris", "calinski_harabasz")
    if rs:
        M += _macro("IrisSilK", rs["k_found"]) + _macro("IrisSilARI", ari(rs))
    if rc:
        M += _macro("IrisCHK", rc["k_found"]) + _macro("IrisCHARI", ari(rc))

    emp = meta.get("empirical", {})
    if "k_modules" in emp:
        M += _macro("EmpFullGenes", emp["shape_full"][0])
        M += _macro("EmpSamples", emp["shape_full"][1])
        M += _macro("EmpNtop", emp["n_top_genes"])
        M += _macro("EmpKmodules", emp["k_modules"])
        M += _macro("EmpNmodules", emp["n_modules"])
        M += _macro("EmpWithinR", _fmt(emp["mean_within_abs_corr"]))
        M += _macro("EmpBgR", _fmt(emp["background_abs_corr"]))
        M += _macro("EmpNsig", emp["modules_assoc_phenotype_q05"])

    (RESULTS / "results_macros.tex").write_text(M)

    # ----- benchmark table (full tabular; \input as a whole) ----- #
    # The full environment is emitted (not just rows) so the paper can \input it
    # at top level: \input-ing a row fragment *inside* a tabular breaks under the
    # template's global \raggedright, whereas \input-ing a complete tabular works.
    def cell(v):
        if v == "" or v is None:
            return "--"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    body = ["\\begin{tabular}{@{}llllrrr@{}}", "\\hline",
            "Dataset & Method & Target & Setting & $k$ & ARI & Acc. \\\\", "\\hline"]
    last_ds = None
    for r in rows:
        ds = r["dataset"] if r["dataset"] != last_ds else ""
        if ds and last_ds is not None:
            body.append("\\hline")
        last_ds = r["dataset"]
        method = r["method"].replace("GCM-MRK", "\\tool{}").replace("_", "\\_")
        target = r["target"].replace("_", "\\_")
        setting = r["k_setting"].replace("_", "\\_")
        body.append(
            f"{ds} & {method} & \\texttt{{{target}}} & {setting} & "
            f"{r['k_found']} & {cell(r['ari'])} & {cell(r['accuracy'])} \\\\"
        )
    body += ["\\hline", "\\end{tabular}"]
    (RESULTS / "benchmark_table.tex").write_text("\n".join(body) + "\n")

    # ----- empirical module table (full tabular) ----- #
    if module_rows:
        def fmt_p(p):
            return "$<$0.001" if p < 1e-3 else f"{p:.3f}"
        mb = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\hline",
              "Module & Size & Within $\\lvert r\\rvert$ & Eig.\\ var. & "
              "$t$ (T vs N) & $q$ (BH) \\\\", "\\hline"]
        for m in sorted(module_rows, key=lambda x: x["tumor_normal_q"]):
            mb.append(
                f"M{m['module']} & {m['size']} & {m['within_abs_corr']:.2f} & "
                f"{m['eigengene_var_explained']:.2f} & {m['tumor_normal_t']:.1f} & "
                f"{fmt_p(m['tumor_normal_q'])} \\\\"
            )
        mb += ["\\hline", "\\end{tabular}"]
        (RESULTS / "empirical_table.tex").write_text("\n".join(mb) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    rows = []; meta = {"seed": SEED, "ga": GA_KW}; res_store = {}

    Xe, te, ke = make_synthetic_easy()
    synthetic_block(Xe, te, ke, "Synthetic-Easy", rows, "synthetic_easy", meta,
                    res_store)
    Xh, th, kh = make_synthetic_hard()
    synthetic_block(Xh, th, kh, "Synthetic-Hard", rows, "synthetic_hard", meta,
                    res_store)

    # NSGA-II multi-objective demo on the hard data (for the Pareto figure/row).
    r_mo, res_mo = gcmrk_run(Xh, ["loglik", "bic"], 10, th, "Synthetic-Hard",
                             mode="nsga2", k_setting="nsga2 g_max=10")
    rows.append(r_mo)

    # Iris (geometric).
    Xi, ti, ki = make_iris()
    meta["iris"] = {"shape": list(Xi.shape), "k_true": ki}
    rows.append(kmeans_baseline(Xi, ki, ti, "Iris", by_sample=False))
    r_sil, res_iris = gcmrk_run(Xi, ["silhouette"], 6, ti, "Iris",
                                by_sample=False, k_setting="g_max=6")
    rows.append(r_sil)
    rows.append(gcmrk_run(Xi, ["calinski_harabasz"], 6, ti, "Iris",
                          by_sample=False, k_setting="g_max=6")[0])
    rows.append(gcmrk_run(Xi, ["silhouette", "davies_bouldin"], 8, ti, "Iris",
                          mode="nsga2", by_sample=False,
                          k_setting="nsga2 g_max=8")[0])

    # Model-selection scan (for the figure + selected-k table values).
    scan_hard = model_selection_scan(Xh, th, "Synthetic-Hard")
    meta["model_selection_scan_hard"] = scan_hard

    # Empirical.
    module_rows, eigengenes = empirical_block(rows, meta)

    # ---- write tidy results ---- #
    fieldnames = ["dataset", "method", "target", "k_setting", "k_found",
                  "ari", "accuracy", "runtime_s"]
    with open(RESULTS / "benchmark.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fieldnames); wr.writeheader()
        wr.writerows(rows)
    with open(RESULTS / "benchmark.json", "w") as fh:
        json.dump({"meta": meta, "rows": rows}, fh, indent=2)

    print(f"{'dataset':16s} {'method':14s} {'target':18s} {'setting':20s} "
          f"{'k':>2s} {'ARI':>6s} {'acc':>6s}")
    for r in rows:
        ari = r["ari"]; acc = r["accuracy"]
        ari = f"{ari:6.3f}" if isinstance(ari, float) else f"{str(ari):>6s}"
        acc = f"{acc:6.3f}" if isinstance(acc, float) else f"{str(acc):>6s}"
        print(f"{r['dataset']:16s} {r['method']:14s} {r['target']:18s} "
              f"{r['k_setting']:20s} {r['k_found']:2d} {ari} {acc}")

    # ---- figures ---- #
    fig_convergence(res_store["Synthetic-Easy_recovery"])
    kbic, kaic = fig_model_selection(scan_hard, kh)
    print(f"\nmodel-selection scan (hard): loglik_bic picks k={kbic}, "
          f"loglik_aic picks k={kaic} (true {kh})")
    fig_pareto(res_mo)
    fig_iris_silhouette(Xi, res_iris.labels)
    if module_rows is not None:
        df = pd.read_csv(DATA / "GSE183947_fpkm.csv", index_col=0, nrows=1)
        is_tumor = np.array([c.startswith("CA.") for c in df.columns])
        fig_empirical_eigengenes(module_rows, eigengenes, is_tumor)
        print(f"empirical: {meta['empirical']}")

    # Auto-generate the LaTeX the paper \input's, so re-running updates the paper.
    write_latex(rows, meta, module_rows, scan_hard, kbic, kaic)

    print(f"\nWrote results to {RESULTS} and figures to {FIGURES}")
    print(f"Wrote LaTeX macros + table bodies to {RESULTS}/*.tex")


if __name__ == "__main__":
    main()
