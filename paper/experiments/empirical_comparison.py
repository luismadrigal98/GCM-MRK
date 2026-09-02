"""Comparative empirical analysis on GSE183947 breast-cancer RNA-seq.

The manuscript's original empirical section applied GCM-MRK alone and reported
that its module eigengenes separate tumour from normal tissue.  That is close to
unfalsifiable: almost any partition of the most variable genes in a tumour/normal
dataset will produce some eigengenes that differ between the groups.  This script
replaces it with a comparison in which every method is scored on the same data by
the same criteria, none of which uses the tumour label to fit anything.

Methods
-------
``gcm_factor``   two-stage: ``loglik_aic`` selects K, then the rank-q factor model
                 is fitted with K held there (the workflow the paper recommends).
``gcm_loglik``   the constrained objective alone, via ``loglik_aic``.
``wgcna``        the standard pipeline at its own ``pickSoftThreshold`` settings.
                 There is no ground truth on real data, so the oracle-tuned arm
                 used in the synthetic benchmarks has no analogue here; this is
                 what a practitioner actually gets.
``leiden``       default resolution, on the soft-thresholded correlation graph.
``hier_avg``     average linkage on 1-|R|, cut at the K minimizing loglik_aic.

Criteria
--------
1. *Coherence* -- mean within-module |r|, and the variance each module's
   eigengene explains within its own module.
2. *Phenotype association* -- how many module eigengenes separate tumour from
   normal (Welch t-test, Benjamini-Hochberg).  Reported for context, but note it
   is the weakest criterion: the tumour/normal axis dominates this dataset, so
   most partitions score something.
3. *Functional enrichment* -- GO Biological Process over-representation per
   module (Enrichr), against the analysed gene set as background.  This is the
   criterion that actually discriminates: it asks whether a module corresponds to
   a coherent biological programme, which coherence alone does not guarantee.
4. *Reproducibility* -- samples are split in half (stratified by tumour status),
   each half is clustered independently, and the two partitions are compared by
   adjusted Rand index.  A method whose modules are an artefact of sampling
   noise scores poorly here regardless of how coherent its modules look.

Outputs (../results, ../figures):
  results/empirical_comparison.json    full per-method results
  results/empirical_comparison.tex     LaTeX table body
  results/empirical_macros.tex         scalar macros for the manuscript
  figures/empirical_comparison.pdf     coherence, enrichment and reproducibility

Run from repo root:
    PYTHONPATH=. python3 paper/experiments/empirical_comparison.py [--n-top 200]
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

from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score as ARI

from gcmrk import cluster, cluster_factor_auto
from gcmrk.data import normalize_data, pearson_correlation
from gcmrk.metrics import loglik_aic

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as B  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULTS = HERE.parent / "results"
FIGURES = HERE.parent / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

#: The factor objective is far more expensive on real data than on the synthetic
#: designs (more modules, higher selected rank), so the GA budget is reduced for
#: every GCM arm equally.  Both arms use the same settings, so the comparison
#: between them is unaffected.
GA = dict(generations=60, population_size=100, seed=0)
FACTOR_RANKS = (1, 2, 3)
G_MAX = 15
MIN_MODULE_FOR_ENRICHMENT = 8

METHOD_LABELS = {
    "gcm_factor": "\\tool{} (\\texttt{loglik\\_factor}, two-stage)",
    "gcm_loglik": "\\tool{} (\\texttt{loglik\\_aic})",
    "wgcna": "WGCNA (default settings)",
    "leiden": "Leiden (default resolution)",
    "hier_avg": "hierarchical (average) + penalized cut",
}
METHOD_ORDER = ["gcm_factor", "gcm_loglik", "wgcna", "leiden", "hier_avg"]


# --------------------------------------------------------------------------- #
def load_data(n_top):
    df = pd.read_csv(DATA / "GSE183947_fpkm.csv", index_col=0)
    genes = df.index.to_numpy()
    is_tumor = np.array([c.startswith("CA.") for c in df.columns])
    L = np.log2(df.values + 1.0)
    idx = np.argsort(L.var(axis=1))[::-1][:n_top]
    return L[idx], genes[idx], is_tumor


def hierarchical_select(cor, n_samples, g_max):
    Z = linkage(squareform(B.abscor_distance(cor), checks=False), method="average")
    best, best_ic = None, np.inf
    for k in range(2, g_max + 1):
        lab = fcluster(Z, t=k, criterion="maxclust")
        ic = loglik_aic(cor, lab, n_samples)
        if ic < best_ic:
            best_ic, best = ic, lab
    return np.asarray(best)


def fit_all(G, seed=0):
    """Every method on one expression matrix (genes x samples)."""
    Xn = normalize_data(G, by_sample=True)
    cor = pearson_correlation(Xn, rowvar=True)
    d = G.shape[1]
    out = {}

    t0 = time.time()
    r_aic = cluster(G, targets=["loglik_aic"], g_max=G_MAX, **GA)
    out["gcm_loglik"] = np.asarray(r_aic.labels)
    k_hat = len(np.unique(out["gcm_loglik"]))
    print(f"    gcm_loglik: k={k_hat} ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    fa = cluster_factor_auto(G, ranks=FACTOR_RANKS, g_max=k_hat, **GA)
    out["gcm_factor"] = np.asarray(fa.labels)
    print(f"    gcm_factor: k={len(np.unique(out['gcm_factor']))} "
          f"q={fa.scores['factor_rank']} ({time.time()-t0:.0f}s)", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["leiden"] = np.asarray(B.leiden(cor, Xn, None, np.random.default_rng(seed)))
    out["hier_avg"] = hierarchical_select(cor, d, G_MAX)

    try:
        wg = B.wgcna_batch([G], powers=(6,), min_sizes=(10,))
        out["wgcna"] = np.asarray(B.wgcna_default(wg[0]))
    except Exception as exc:                     # pragma: no cover
        print(f"    WGCNA failed: {type(exc).__name__}: {str(exc)[:150]}")
        out["wgcna"] = None
    return out, cor


# --------------------------------------------------------------------------- #
def module_stats(labels, G, cor, is_tumor):
    """Coherence, eigengene variance explained, and phenotype association."""
    rows = []
    for m in np.unique(labels):
        if m == 0:                                # WGCNA grey / unassigned
            continue
        idx = np.where(labels == m)[0]
        if idx.size < 3:
            continue
        block = cor[np.ix_(idx, idx)]
        iu = np.triu_indices(idx.size, 1)
        coherence = float(np.abs(block[iu]).mean())

        sub = G[idx] - G[idx].mean(axis=1, keepdims=True)
        u, s, vt = np.linalg.svd(sub, full_matrices=False)
        pve = float(s[0] ** 2 / (s ** 2).sum())
        eig = vt[0]
        t, p = stats.ttest_ind(eig[is_tumor], eig[~is_tumor], equal_var=False)
        rows.append({"module": int(m), "size": int(idx.size),
                     "coherence": coherence, "pve": pve, "t": float(t),
                     "p": float(p)})
    # Benjamini-Hochberg across this method's modules
    if rows:
        ps = np.array([r["p"] for r in rows])
        order = np.argsort(ps)
        q = np.empty_like(ps)
        n = ps.size
        prev = 1.0
        for rank, i in enumerate(order[::-1]):
            prev = min(prev, ps[i] * n / (n - rank))
            q[i] = prev
        for r, qq in zip(rows, q):
            r["q"] = float(qq)
    return rows


def enrichment(labels, gene_names, background, library="GO_Biological_Process_2023"):
    """Per-module GO over-representation via Enrichr.

    Returns one record per tested module.  Network failures are recorded rather
    than raised: the rest of the analysis does not depend on them.
    """
    import gseapy

    recs = []
    for m in np.unique(labels):
        if m == 0:
            continue
        idx = np.where(labels == m)[0]
        if idx.size < MIN_MODULE_FOR_ENRICHMENT:
            continue
        genes = [str(g) for g in gene_names[idx]]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = gseapy.enrichr(gene_list=genes, gene_sets=library,
                                     background=list(background), outdir=None,
                                     no_plot=True).results
            adj = res["Adjusted P-value"].astype(float).values
            best = float(adj.min()) if adj.size else 1.0
            n_sig = int((adj < 0.05).sum())
            top = str(res.iloc[int(np.argmin(adj))]["Term"]) if adj.size else ""
        except Exception as exc:                  # pragma: no cover
            print(f"      enrichr failed for module {m}: "
                  f"{type(exc).__name__}: {str(exc)[:90]}", flush=True)
            best, n_sig, top = float("nan"), 0, "(request failed)"
        recs.append({"module": int(m), "size": int(idx.size),
                     "best_adj_p": best, "n_significant": n_sig, "top_term": top})
        time.sleep(0.5)                           # be polite to Enrichr
    return recs


def summarize(labels, G, cor, is_tumor, gene_names, do_enrich):
    stats_rows = module_stats(labels, G, cor, is_tumor)
    n_mod = len(stats_rows)
    assigned = int((labels != 0).sum())
    summary = {
        "k": int(len(np.unique(labels[labels != 0]))),
        "n_modules_scored": n_mod,
        "assigned_fraction": assigned / labels.size,
        "median_coherence": float(np.median([r["coherence"] for r in stats_rows]))
                            if n_mod else float("nan"),
        "median_pve": float(np.median([r["pve"] for r in stats_rows]))
                      if n_mod else float("nan"),
        "n_phenotype_sig": int(sum(r["q"] < 0.05 for r in stats_rows)) if n_mod else 0,
        "modules": stats_rows,
    }
    if do_enrich:
        recs = enrichment(labels, gene_names, gene_names)
        summary["enrichment"] = recs
        valid = [r for r in recs if np.isfinite(r["best_adj_p"])]
        summary["n_enriched_modules"] = int(sum(r["n_significant"] > 0 for r in valid))
        summary["n_enrichment_tested"] = len(valid)
        summary["median_best_adj_p"] = (float(np.median([r["best_adj_p"] for r in valid]))
                                        if valid else float("nan"))
    return summary


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-top", type=int, default=200)
    ap.add_argument("--no-enrichment", action="store_true")
    ap.add_argument("--no-split", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    G, gene_names, is_tumor = load_data(args.n_top)
    print(f"GSE183947: {G.shape[0]} genes x {G.shape[1]} samples "
          f"({is_tumor.sum()} tumour / {(~is_tumor).sum()} normal)", flush=True)

    print("Fitting all methods on the full dataset", flush=True)
    labels_full, cor = fit_all(G)

    results = {}
    for m in METHOD_ORDER:
        lab = labels_full.get(m)
        if lab is None:
            results[m] = {"status": "failed"}
            continue
        print(f"  scoring {m}", flush=True)
        results[m] = summarize(lab, G, cor, is_tumor, gene_names,
                               do_enrich=not args.no_enrichment)

    # ---- reproducibility: cluster each half of the samples independently ---- #
    if not args.no_split:
        print("Split-half reproducibility", flush=True)
        rng = np.random.default_rng(0)
        half_a = np.zeros(G.shape[1], dtype=bool)
        for grp in (is_tumor, ~is_tumor):         # stratify by phenotype
            gi = np.where(grp)[0]
            half_a[rng.permutation(gi)[: gi.size // 2]] = True
        fits = {}
        for name, mask in (("A", half_a), ("B", ~half_a)):
            print(f"  half {name} ({mask.sum()} samples)", flush=True)
            fits[name], _ = fit_all(G[:, mask], seed=1)
        for m in METHOD_ORDER:
            la, lb = fits["A"].get(m), fits["B"].get(m)
            if la is None or lb is None or m not in results:
                continue
            results[m]["split_half_ari"] = float(ARI(la, lb))
            results[m]["split_half_k"] = [int(len(np.unique(la[la != 0]))),
                                          int(len(np.unique(lb[lb != 0])))]

    data = {"meta": {"dataset": "GSE183947", "n_top": args.n_top,
                     "n_samples": int(G.shape[1]), "ga": GA,
                     "g_max": G_MAX, "factor_ranks": list(FACTOR_RANKS),
                     "enrichment_library": "GO_Biological_Process_2023",
                     "seconds": round(time.time() - t_start, 1)},
            "results": results}
    (RESULTS / "empirical_comparison.json").write_text(json.dumps(data, indent=1))

    print_summary(data)
    write_latex(data)
    fig_comparison(data)
    print(f"\nDone in {time.time()-t_start:.0f}s -> {RESULTS}")


def print_summary(data):
    r = data["results"]
    print("\n=== GSE183947 comparative summary ===")
    hdr = (f"{'method':16} {'k':>3} {'assigned':>9} {'coher':>7} {'PVE':>6} "
           f"{'pheno':>6} {'enrich':>8} {'splitARI':>9}")
    print(hdr)
    for m in METHOD_ORDER:
        v = r.get(m, {})
        if v.get("status") == "failed":
            print(f"{m:16} FAILED"); continue
        er = (f"{v.get('n_enriched_modules','-')}/{v.get('n_enrichment_tested','-')}")
        print(f"{m:16} {v['k']:3d} {v['assigned_fraction']:9.2f} "
              f"{v['median_coherence']:7.3f} {v['median_pve']:6.2f} "
              f"{v['n_phenotype_sig']:3d}/{v['n_modules_scored']:<2d} {er:>8} "
              f"{v.get('split_half_ari', float('nan')):9.3f}")


def write_latex(data):
    r = data["results"]
    rows = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\hline",
            "Method & $K$ & assigned & coherence & eigengene & GO-enriched & "
            "split-half \\\\",
            " & & & (med.\\ $|r|$) & PVE & modules & ARI \\\\", "\\hline"]
    for m in METHOD_ORDER:
        v = r.get(m, {})
        if v.get("status") == "failed":
            continue
        er = f"{v.get('n_enriched_modules','--')}/{v.get('n_enrichment_tested','--')}"
        rows.append(
            f"{METHOD_LABELS[m]} & {v['k']} & {v['assigned_fraction']:.2f} & "
            f"{v['median_coherence']:.3f} & {v['median_pve']:.2f} & {er} & "
            f"{v.get('split_half_ari', float('nan')):.2f} \\\\")
    rows += ["\\hline", "\\end{tabular}"]
    (RESULTS / "empirical_comparison.tex").write_text("\n".join(rows) + "\n")

    M = ""
    def mac(n, v): return f"\\newcommand{{\\{n}}}{{{v}}}\n"
    M += mac("EmpCompNTop", data["meta"]["n_top"])
    M += mac("EmpCompLibrary", "GO Biological Process")
    for m, short in [("gcm_factor", "Factor"), ("gcm_loglik", "Loglik"),
                     ("wgcna", "Wgcna"), ("leiden", "Leiden"), ("hier_avg", "Hier")]:
        v = r.get(m, {})
        if v.get("status") == "failed":
            continue
        M += mac(f"EmpComp{short}K", v["k"])
        M += mac(f"EmpComp{short}Coh", f"{v['median_coherence']:.3f}")
        M += mac(f"EmpComp{short}Enr", v.get("n_enriched_modules", 0))
        M += mac(f"EmpComp{short}Tested", v.get("n_enrichment_tested", 0))
        M += mac(f"EmpComp{short}Split", f"{v.get('split_half_ari', float('nan')):.2f}")
    (RESULTS / "empirical_macros.tex").write_text(M)


def fig_comparison(data):
    r = data["results"]
    ms = [m for m in METHOD_ORDER if r.get(m, {}).get("status") != "failed"]
    labels = [METHOD_LABELS[m].replace("\\tool{}", "GCM-MRK")
              .replace("\\texttt{", "").replace("}", "") for m in ms]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1))
    panels = [
        ("median_coherence", "median within-module $|r|$"),
        (None, "GO-enriched modules (fraction)"),
        ("split_half_ari", "split-half ARI"),
    ]
    for ax, (key, title) in zip(axes, panels):
        if key is None:
            vals = [(r[m].get("n_enriched_modules", 0)
                     / max(r[m].get("n_enrichment_tested", 1), 1)) for m in ms]
        else:
            vals = [r[m].get(key, np.nan) for m in ms]
        ax.barh(np.arange(len(ms)), vals, color=["C0"] + ["0.7"] * (len(ms) - 1))
        ax.set_yticks(np.arange(len(ms)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel(title, fontsize=7)
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES / "empirical_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
