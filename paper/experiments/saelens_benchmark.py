"""Module recovery on the Saelens et al. (2018) benchmark: real data, real modules.

Every decisive result in this manuscript so far comes from data we generated
ourselves.  The misspecification sweep mitigates that -- those generators
deliberately break our own model -- but they are still simulations, and the one
empirical dataset analysed (GSE183947) cannot separate any of the methods.  This
script closes the gap using the benchmark of Saelens, Cannoodt and Saeys
(Nat Commun 9:1090, 2018), which pairs real expression compendia with curated
module definitions derived from regulatory networks.

Ground truth here is *not* a partition.  Known modules are regulons: they
overlap, and they do not cover every gene.  Adjusted Rand index is therefore the
wrong metric, and we use the one the benchmark's authors defined:

    recovery  = mean over known modules   of the best Jaccard against any predicted module
    relevance = mean over predicted modules of the best Jaccard against any known module
    F1rr      = harmonic mean of the two

Recovery alone is maximized by predicting one huge module; relevance alone by
predicting many tiny ones.  F1rr is the score to read.

Deviations from the original study, all documented because they matter for
interpretation:

* We analyse the ``--n-top`` most variable genes rather than whole compendia.
  The factor objective costs minutes per fit at this scale (see the scaling
  section of the manuscript), so whole-compendium runs are not tractable here.
  Known modules are intersected with the retained gene set and those left with
  fewer than ``--min-known`` genes are dropped.
* Because of that subsetting our absolute scores are not comparable with the
  published numbers in Saelens et al.  They remain comparable *between methods*,
  which is what the manuscript claims.
* Unassigned genes (label 0, e.g. WGCNA's grey module) are excluded from the
  predicted module set rather than treated as a module.

Outputs (../results, ../figures):
  results/saelens.json            per dataset per method scores
  results/saelens_table.tex       LaTeX table body
  results/saelens_macros.tex      scalar macros for the manuscript
  figures/saelens.pdf             F1rr by dataset and method

Run from repo root:
    PYTHONPATH=. python3 paper/experiments/saelens_benchmark.py \\
        --datasets ecoli_colombos,yeast_gpl2529,human_tcga --n-top 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from gcmrk import cluster, cluster_factor_auto
from gcmrk.data import normalize_data, pearson_correlation
from gcmrk.metrics import loglik_aic

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as B  # noqa: E402

HERE = Path(__file__).resolve().parent
SAELENS = HERE.parent / "data" / "saelens2018"
RESULTS = HERE.parent / "results"
FIGURES = HERE.parent / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

#: The human `regcircuit` ground truth is published at nine stringency
#: thresholds. The permissive ones are unusable as module definitions here --
#: on human_tcga the median "module" at threshold 001 spans 1295 of 5888 genes,
#: a quarter of the transcriptome -- so for the human compendia we use the most
#: stringent variant only (median size 9). E. coli and yeast sources are all
#: module-scale and are used in full.
PREFERRED_SOURCES = {
    "human_gtex": ["human_regcircuit_5"],
    "human_tcga": ["human_regcircuit_5"],
    "human_seek_gpl5175": ["human_regcircuit_5"],
    "human_seek_gpl8300": ["human_regcircuit_5"],
}

GA = dict(generations=60, population_size=100, seed=0)
FACTOR_RANKS = (1, 2, 3)
G_MAX = 20

METHOD_LABELS = {
    "gcm_factor": "\\tool{} (\\texttt{loglik\\_factor}, two-stage)",
    "gcm_loglik": "\\tool{} (\\texttt{loglik\\_aic})",
    "wgcna": "WGCNA",
    "leiden": "Leiden",
    "spectral": "spectral",
    "hier_avg": "hierarchical (average)",
}
METHOD_ORDER = ["gcm_loglik", "gcm_factor", "wgcna", "leiden", "spectral", "hier_avg"]


# --------------------------------------------------------------------------- #
# the benchmark's own scoring
# --------------------------------------------------------------------------- #
def memberships(modules, genes):
    """Boolean gene x module matrix for a list of gene-name lists."""
    index = {g: i for i, g in enumerate(genes)}
    M = np.zeros((len(genes), len(modules)), dtype=bool)
    for j, mod in enumerate(modules):
        for g in mod:
            i = index.get(g)
            if i is not None:
                M[i, j] = True
    return M


def jaccard_matrix(A, B):
    """Pairwise Jaccard between the columns of two boolean membership matrices."""
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    inter = A.T @ B
    a = A.sum(0)[:, None]
    b = B.sum(0)[None, :]
    union = a + b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(union > 0, inter / union, 0.0)
    return J


def score_modules(known, predicted, genes):
    """Recovery, relevance and their harmonic mean (Saelens et al. 2018)."""
    A = memberships(known, genes)
    Bm = memberships(predicted, genes)
    if A.shape[1] == 0 or Bm.shape[1] == 0:
        return {"recovery": 0.0, "relevance": 0.0, "f1rr": 0.0}
    J = jaccard_matrix(A, Bm)
    recovery = float(J.max(axis=1).mean())
    relevance = float(J.max(axis=0).mean())
    f1 = (0.0 if min(recovery, relevance) <= 0
          else 2 * recovery * relevance / (recovery + relevance))
    return {"recovery": recovery, "relevance": relevance, "f1rr": f1}


def labels_to_modules(labels, genes):
    """Hard labels -> list of gene-name lists, dropping the unassigned label."""
    out = []
    for m in np.unique(labels):
        if m == 0:
            continue
        out.append([genes[i] for i in np.where(labels == m)[0]])
    return out


# --------------------------------------------------------------------------- #
def load_dataset(name, n_top, min_known, max_known_frac=0.25):
    """Expression matrix (genes x samples) plus known modules, both subsetted."""
    root = SAELENS / "data" / name
    e_path = root / "E.tsv"
    if not e_path.exists():
        raise FileNotFoundError(f"{e_path} not found; is data.zip extracted?")

    # E.tsv is written as samples x genes by the benchmark's own tooling
    E = pd.read_csv(e_path, sep="\t", index_col=0)
    X = E.values.T                                     # genes x samples
    genes = np.asarray(E.columns)

    keep = np.argsort(np.nanvar(X, axis=1))[::-1][:n_top]
    X, genes = X[keep], genes[keep]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    km_root = root / "knownmodules"
    wanted = PREFERRED_SOURCES.get(name)
    max_size = max(int(max_known_frac * len(genes)), min_known + 1)
    sources = {}
    if km_root.exists():
        for src in sorted(p for p in km_root.iterdir() if p.is_dir()):
            if wanted is not None and src.name not in wanted:
                continue
            f = src / "minimal.json"
            if not f.exists():
                continue
            mods = json.loads(f.read_text())
            gene_set = set(genes.tolist())
            mods = [[g for g in m if g in gene_set] for m in mods]
            # drop modules too small to score and those so large they are not
            # module-scale relative to the analysed gene set
            mods = [m for m in mods if min_known <= len(m) <= max_size]
            if mods:
                sources[src.name] = mods
    return X, genes, sources


def hierarchical_select(cor, n_samples, g_max):
    Z = linkage(squareform(B.abscor_distance(cor), checks=False), method="average")
    best, best_ic = None, np.inf
    for k in range(2, g_max + 1):
        lab = fcluster(Z, t=k, criterion="maxclust")
        ic = loglik_aic(cor, lab, n_samples)
        if ic < best_ic:
            best_ic, best = ic, lab
    return np.asarray(best)


def fit_all(X):
    """Every method on one matrix; returns (labels, seconds) keyed by method."""
    Xn = normalize_data(X, by_sample=True)
    cor = pearson_correlation(Xn, rowvar=True)
    out, secs = {}, {}

    t0 = time.time()
    r = cluster(X, targets=["loglik_aic"], g_max=G_MAX, **GA)
    out["gcm_loglik"] = np.asarray(r.labels)
    secs["gcm_loglik"] = time.time() - t0
    k_hat = len(np.unique(out["gcm_loglik"]))
    print(f"    gcm_loglik k={k_hat} ({secs['gcm_loglik']:.0f}s)", flush=True)

    t0 = time.time()
    fa = cluster_factor_auto(X, ranks=FACTOR_RANKS, g_max=k_hat, **GA)
    out["gcm_factor"] = np.asarray(fa.labels)
    secs["gcm_factor"] = time.time() - t0      # the factor stage alone
    print(f"    gcm_factor k={len(np.unique(out['gcm_factor']))} "
          f"q={fa.scores['factor_rank']} ({secs['gcm_factor']:.0f}s)", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["leiden"] = np.asarray(B.leiden(cor, Xn, None, np.random.default_rng(0)))
        out["spectral"] = np.asarray(B.spectral(cor, Xn, k_hat,
                                                np.random.default_rng(0)))
    out["hier_avg"] = hierarchical_select(cor, X.shape[1], G_MAX)

    try:
        wg = B.wgcna_batch([X], powers=(6,), min_sizes=(10,))
        out["wgcna"] = np.asarray(B.wgcna_default(wg[0]))
    except Exception as exc:                            # pragma: no cover
        print(f"    WGCNA failed: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        out["wgcna"] = None
    return out, secs


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ecoli_colombos,yeast_gpl2529,human_tcga")
    ap.add_argument("--n-top", type=int, default=500)
    ap.add_argument("--min-known", type=int, default=5)
    ap.add_argument("--max-known-frac", type=float, default=0.25,
                    help="drop known modules spanning more than this fraction "
                         "of the analysed genes (they are not module-scale)")
    ap.add_argument("--list", action="store_true",
                    help="list available datasets and exit")
    args = ap.parse_args()

    if args.list:
        root = SAELENS / "data"
        if not root.exists():
            print(f"{root} not found; extract data.zip first")
            return
        for d in sorted(p.name for p in root.iterdir() if p.is_dir()):
            km = root / d / "knownmodules"
            srcs = ([p.name for p in km.iterdir() if p.is_dir()]
                    if km.exists() else [])
            has_e = (root / d / "E.tsv").exists()
            print(f"  {d:28s} E.tsv={'y' if has_e else 'n'}  knownmodules={srcs}")
        return

    t_start = time.time()
    all_results = {}
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        print(f"\n=== {name} ===", flush=True)
        try:
            X, genes, sources = load_dataset(name, args.n_top, args.min_known,
                                             args.max_known_frac)
        except FileNotFoundError as exc:
            print(f"  skipped: {exc}", flush=True)
            continue
        if not sources:
            print("  skipped: no known modules survive the gene subset", flush=True)
            continue
        print(f"  {X.shape[0]} genes x {X.shape[1]} samples; known-module sources: "
              + ", ".join(f"{k} ({len(v)} modules)" for k, v in sources.items()),
              flush=True)

        labels, secs = fit_all(X)
        ds = {"n_genes": int(X.shape[0]), "n_samples": int(X.shape[1]),
              "sources": {k: len(v) for k, v in sources.items()},
              "seconds": {k: round(v, 1) for k, v in secs.items()},
              "methods": {}}
        for m in METHOD_ORDER:
            lab = labels.get(m)
            if lab is None:
                ds["methods"][m] = {"status": "failed"}
                continue
            pred = labels_to_modules(lab, genes)
            per_source = {src: score_modules(known, pred, genes)
                          for src, known in sources.items()}
            # average across known-module sources when a dataset has several
            f1 = float(np.mean([v["f1rr"] for v in per_source.values()]))
            rec = float(np.mean([v["recovery"] for v in per_source.values()]))
            rel = float(np.mean([v["relevance"] for v in per_source.values()]))
            ds["methods"][m] = {"k": int(len(pred)), "recovery": rec,
                                "relevance": rel, "f1rr": f1,
                                "per_source": per_source}
            print(f"    {m:12s} k={len(pred):3d} recovery={rec:.3f} "
                  f"relevance={rel:.3f} F1rr={f1:.3f}", flush=True)
        all_results[name] = ds

    data = {"meta": {"benchmark": "Saelens et al. 2018 (Nat Commun 9:1090)",
                     "n_top": args.n_top, "min_known": args.min_known,
                     "max_known_frac": args.max_known_frac,
                     "preferred_sources": PREFERRED_SOURCES,
                     "ga": GA, "g_max": G_MAX,
                     "factor_ranks": list(FACTOR_RANKS),
                     "seconds": round(time.time() - t_start, 1)},
            "results": all_results}
    (RESULTS / "saelens.json").write_text(json.dumps(data, indent=1))
    if all_results:
        write_latex(data)
        fig_saelens(data)
    print(f"\nDone in {time.time()-t_start:.0f}s -> {RESULTS/'saelens.json'}")


#: two-line column headers: organism over compendium
DISPLAY = {"ecoli_colombos": ("\\emph{E. coli}", "COLOMBOS"),
           "ecoli_precise2": ("\\emph{E. coli}", "PRECISE2"),
           "ecoli_dream5": ("\\emph{E. coli}", "DREAM5"),
           "yeast_gpl2529": ("yeast", "GPL2529"),
           "yeast_dream5": ("yeast", "DREAM5"),
           "human_tcga": ("human", "TCGA"),
           "human_gtex": ("human", "GTEx"),
           "human_seek_gpl5175": ("human", "SEEK 5175"),
           "human_seek_gpl8300": ("human", "SEEK 8300")}


def write_latex(data):
    res = data["results"]
    names = list(res)
    head = " & ".join("\\shortstack{%s\\\\%s}" % DISPLAY.get(n, (n.replace("_", "\\_"), ""))
                      for n in names)
    rows = ["\\begin{tabular}{@{}l" + "r" * len(names) + "@{}}", "\\hline",
            "Method & " + head + " \\\\", "\\hline"]
    best = {n: max((res[n]["methods"][m].get("f1rr", -1) for m in METHOD_ORDER
                    if m in res[n]["methods"]), default=-1) for n in names}
    for m in METHOD_ORDER:
        cells = []
        for n in names:
            v = res[n]["methods"].get(m, {})
            if v.get("status") == "failed":
                cells.append("---"); continue
            c = f"{v['f1rr']:.3f}"
            if v["f1rr"] >= best[n] - 1e-9:
                c = "\\textbf{" + c + "}"
            cells.append(c)
        rows.append(f"{METHOD_LABELS[m]} & " + " & ".join(cells) + " \\\\")
    rows += ["\\hline", "\\end{tabular}"]
    (RESULTS / "saelens_table.tex").write_text("\n".join(rows) + "\n")

    M = ""
    def mac(n, v): return f"\\newcommand{{\\{n}}}{{{v}}}\n"
    # NB: LaTeX control sequences may not contain digits, so these are named
    # ...Score rather than ...F1.
    M += mac("SaelensNTop", data["meta"]["n_top"])
    M += mac("SaelensNDatasets", len(names))
    for m, short in [("gcm_loglik", "Loglik"), ("gcm_factor", "Factor"),
                     ("wgcna", "Wgcna")]:
        wins = sum(1 for n in names
                   if res[n]["methods"].get(m, {}).get("f1rr", -1) >= best[n] - 1e-9)
        M += mac(f"Saelens{short}Wins", wins)
    # paired sign tests against the lead objective, across datasets
    from scipy.stats import binomtest
    lead = np.array([res[n]["methods"]["gcm_loglik"]["f1rr"] for n in names])
    for m, short in [("gcm_factor", "Factor"), ("wgcna", "Wgcna"),
                     ("spectral", "Spectral"), ("leiden", "Leiden"),
                     ("hier_avg", "Hier")]:
        other = np.array([res[n]["methods"][m]["f1rr"] for n in names])
        pos = int((lead - other > 0).sum())
        M += mac(f"Saelens{short}Beaten", f"{pos}/{len(names)}")
        M += mac(f"Saelens{short}P",
                 f"{binomtest(pos, len(names), 0.5).pvalue:.3f}")
    for m, short in [("gcm_factor", "Factor"), ("gcm_loglik", "Loglik"),
                     ("wgcna", "Wgcna"), ("leiden", "Leiden"),
                     ("spectral", "Spectral"), ("hier_avg", "Hier")]:
        vals = [res[n]["methods"][m]["f1rr"] for n in names
                if res[n]["methods"].get(m, {}).get("status") != "failed"]
        if vals:
            M += mac(f"Saelens{short}Score", f"{np.mean(vals):.3f}")
    secs = [res[n].get("seconds", {}) for n in names]
    if all("gcm_loglik" in t and "gcm_factor" in t for t in secs):
        lo = [t["gcm_loglik"] for t in secs]
        fa = [t["gcm_factor"] for t in secs]
        ratio = [f / l for f, l in zip(fa, lo)]
        M += mac("SaelensLoglikSecMin", f"{min(lo):.0f}")
        M += mac("SaelensLoglikSecMax", f"{max(lo):.0f}")
        M += mac("SaelensFactorSecMin", f"{min(fa):.0f}")
        M += mac("SaelensFactorSecMax", f"{max(fa):.0f}")
        # rounded to the nearest ten: the ratio is a summary, not a measurement
        M += mac("SaelensCostRatioMin", f"{10 * round(min(ratio) / 10):.0f}")
        M += mac("SaelensCostRatioMax", f"{10 * round(max(ratio) / 10):.0f}")
    all_f1 = [res[n]["methods"][m]["f1rr"] for n in names for m in METHOD_ORDER
              if res[n]["methods"].get(m, {}).get("status") != "failed"]
    M += mac("SaelensFMin", f"{min(all_f1):.2f}")
    M += mac("SaelensFMax", f"{max(all_f1):.2f}")
    (RESULTS / "saelens_macros.tex").write_text(M)


def fig_saelens(data):
    res = data["results"]
    names = list(res)
    plain = {m: METHOD_LABELS[m].replace("\\tool{}", "GCM")
                                 .replace("\\texttt{", "").replace("}", "")
                                 .replace("\\_", "_")
             for m in METHOD_ORDER}
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(names))
    w = 0.8 / len(METHOD_ORDER)
    for j, m in enumerate(METHOD_ORDER):
        vals = [res[n]["methods"].get(m, {}).get("f1rr", np.nan) for n in names]
        ax.bar(x + (j - len(METHOD_ORDER) / 2 + 0.5) * w, vals, w, label=plain[m])
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n", 1) for n in names], fontsize=8)
    ax.set_ylabel("F1 (recovery, relevance)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "saelens.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
