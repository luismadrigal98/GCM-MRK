"""Third-party clustering baselines, behind a single interface.

Module detection in practice is dominated by community-detection, spectral and
network methods as much as by the agglomerative clustering at the core of
co-expression pipelines.  This module wraps all of them behind one signature so
that a driver can call them interchangeably:

    labels = method(cor, X, k_true, rng)     # 1-based integer labels

Methods
-------
``hierarchical_average`` / ``hierarchical_complete``
    Agglomerative linkage on the correlation distance :math:`1-|R|`.
``kmeans``
    Euclidean k-means on the standardized data (via :mod:`gcmrk.seeding`).
``spectral``
    Normalized spectral clustering on the affinity :math:`|R|^{\\beta}`.
``louvain`` / ``leiden``
    Modularity maximization on the weighted co-expression graph.  Neither takes
    a cluster count, so for the known-:math:`k` comparison we bisect the
    resolution parameter until the method returns :math:`k` communities --- the
    most favourable setting we can give them.
``mcl``
    Markov clustering; granularity bisected on the inflation parameter in the
    same way.
``wgcna``
    The real pipeline (soft threshold, TOM, dynamic hybrid tree cut) called out
    to R.  It selects its own module count by construction and assigns
    unclustered genes to the grey module, which we map to label 0.

Graph construction follows WGCNA's unsigned adjacency, :math:`a_{ij}=|r_{ij}|^\\beta`
with ``SOFT_POWER`` = 6, so the graph methods operate on the same notion of
co-expression the correlation objective uses.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from gcmrk.seeding import kmeans as _kmeans_seed

__all__ = [
    "SOFT_POWER",
    "abscor_distance",
    "adjacency",
    "hierarchical",
    "hierarchical_average",
    "hierarchical_complete",
    "kmeans",
    "spectral",
    "louvain",
    "leiden",
    "mcl",
    "wgcna_batch",
    "wgcna_default",
    "wgcna_oracle",
    "available",
]

SOFT_POWER = 6          # WGCNA's default unsigned soft-thresholding power
_BISECT_ITERS = 40      # resolution / inflation bisection budget


# --------------------------------------------------------------------------- #
# shared graph construction
# --------------------------------------------------------------------------- #
def abscor_distance(cor: np.ndarray) -> np.ndarray:
    """Correlation distance :math:`1-|R|`, symmetrized with a zero diagonal."""
    d = 1.0 - np.abs(cor)
    np.fill_diagonal(d, 0.0)
    return (d + d.T) / 2.0


def adjacency(cor: np.ndarray, power: int = SOFT_POWER) -> np.ndarray:
    """WGCNA-style unsigned soft-thresholded adjacency :math:`|r|^{\\beta}`."""
    a = np.abs(cor) ** power
    np.fill_diagonal(a, 0.0)
    return (a + a.T) / 2.0


def _relabel(labels) -> np.ndarray:
    """Map arbitrary cluster ids to consecutive 1-based integers."""
    labels = np.asarray(labels)
    out = np.empty(labels.shape[0], dtype=int)
    for new, old in enumerate(np.unique(labels), start=1):
        out[labels == old] = new
    return out


def _n_clusters(labels) -> int:
    return len(np.unique(labels))


def _bisect_parameter(fit, k_target, lo, hi, iters=_BISECT_ITERS):
    """Find a parameter in ``[lo, hi]`` for which ``fit`` yields ``k_target``
    clusters, searching in log space.  ``fit`` must be monotone increasing in
    cluster count.  Returns the labelling whose count is closest to the target.

    Used to give the resolution-based methods the true ``k``, which is the most
    generous setting available to them.
    """
    lo, hi = np.log(lo), np.log(hi)
    best_lab, best_gap = None, np.inf
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lab = fit(float(np.exp(mid)))
        k = _n_clusters(lab)
        gap = abs(k - k_target)
        if gap < best_gap:
            best_lab, best_gap = lab, gap
        if k == k_target:
            return lab
        if k < k_target:
            lo = mid
        else:
            hi = mid
    return best_lab


# --------------------------------------------------------------------------- #
# agglomerative / geometric
# --------------------------------------------------------------------------- #
def hierarchical(cor, k, method):
    Z = linkage(squareform(abscor_distance(cor), checks=False), method=method)
    return fcluster(Z, t=k, criterion="maxclust")


def hierarchical_average(cor, X, k, rng):
    return hierarchical(cor, k, "average")


def hierarchical_complete(cor, X, k, rng):
    return hierarchical(cor, k, "complete")


def kmeans(cor, X, k, rng):
    return _kmeans_seed(X, k, rng, n_init=10)


# --------------------------------------------------------------------------- #
# spectral
# --------------------------------------------------------------------------- #
def spectral(cor, X, k, rng):
    from sklearn.cluster import SpectralClustering

    affinity = adjacency(cor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            n_init=10,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        return _relabel(sc.fit_predict(affinity))


# --------------------------------------------------------------------------- #
# community detection on the weighted co-expression graph
# --------------------------------------------------------------------------- #
def _igraph_from_adjacency(a):
    import igraph as ig

    g = ig.Graph.Weighted_Adjacency(a.tolist(), mode="undirected", loops=False)
    return g


def louvain(cor, X, k, rng):
    """Louvain (igraph ``community_multilevel``), resolution bisected to ``k``."""
    g = _igraph_from_adjacency(adjacency(cor))
    w = g.es["weight"]

    def fit(resolution):
        return _relabel(g.community_multilevel(weights=w, resolution=resolution).membership)

    if k is None:
        return fit(1.0)
    return _bisect_parameter(fit, k, 1e-3, 1e3)


def leiden(cor, X, k, rng):
    """Leiden (RBConfiguration), resolution bisected to ``k``."""
    import leidenalg as la

    g = _igraph_from_adjacency(adjacency(cor))
    w = g.es["weight"]
    seed = int(rng.integers(0, 2**31 - 1))

    def fit(resolution):
        part = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights=w,
            resolution_parameter=resolution,
            seed=seed,
            n_iterations=5,
        )
        return _relabel(part.membership)

    if k is None:
        return fit(1.0)
    return _bisect_parameter(fit, k, 1e-3, 1e3)


def mcl(cor, X, k, rng):
    """Markov clustering, inflation bisected to ``k`` clusters."""
    import markov_clustering as mc

    a = adjacency(cor)

    def fit(inflation):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = mc.run_mcl(a, inflation=float(inflation))
            clusters = mc.get_clusters(result)
        lab = np.zeros(a.shape[0], dtype=int)
        for cid, members in enumerate(clusters, start=1):
            lab[list(members)] = cid
        # MCL can leave vertices out of every cluster; give each its own label
        nxt = len(clusters) + 1
        for i in np.flatnonzero(lab == 0):
            lab[i] = nxt
            nxt += 1
        return _relabel(lab)

    if k is None:
        return fit(2.0)
    return _bisect_parameter(fit, k, 1.1, 20.0, iters=25)


# --------------------------------------------------------------------------- #
# WGCNA (called out to R)
# --------------------------------------------------------------------------- #
#: soft-thresholding powers and minimum module sizes explored for WGCNA.
WGCNA_POWERS = (1, 2, 3, 4, 6, 8, 12)
WGCNA_MIN_SIZES = (5, 10, 20)

_WGCNA_R = r"""
suppressMessages(library(WGCNA))
options(stringsAsFactors = FALSE)
args <- commandArgs(trailingOnly = TRUE)
manifest <- args[1]; outFile <- args[2]
powers <- as.integer(strsplit(args[3], ",")[[1]])
minSizes <- as.integer(strsplit(args[4], ",")[[1]])

files <- readLines(manifest)
out <- list()

for (i in seq_along(files)) {
  X <- as.matrix(read.csv(files[i], header = FALSE))
  datExpr <- t(X)                       # rows=genes on disk; WGCNA wants samples x genes

  # WGCNA's own recommended power, for the "default settings" arm
  sft <- try(pickSoftThreshold(datExpr, powerVector = 1:20, verbose = 0,
                               networkType = "unsigned"), silent = TRUE)
  autoPower <- 6
  if (!inherits(sft, "try-error") && !is.na(sft$powerEstimate))
    autoPower <- sft$powerEstimate

  runs <- list()
  for (p in unique(c(autoPower, powers))) {
    for (mms in minSizes) {
      net <- try(blockwiseModules(datExpr, power = p, networkType = "unsigned",
                 TOMType = "unsigned", minModuleSize = mms, deepSplit = 2,
                 reassignThreshold = 0, mergeCutHeight = 0.25,
                 numericLabels = TRUE, pamRespectsDendro = FALSE,
                 maxBlockSize = 20000, verbose = 0), silent = TRUE)
      if (inherits(net, "try-error")) next
      runs[[length(runs) + 1]] <- list(power = p, minModuleSize = mms,
                                       labels = as.integer(net$colors))
    }
  }
  out[[i]] <- list(file = files[i], autoPower = autoPower, runs = runs)
}

cat(jsonlite::toJSON(out, auto_unbox = TRUE), file = outFile)
"""

_WGCNA_SCRIPT = None


def _wgcna_script() -> Path:
    global _WGCNA_SCRIPT
    if _WGCNA_SCRIPT is None or not _WGCNA_SCRIPT.exists():
        path = Path(tempfile.gettempdir()) / "gcmrk_wgcna_baseline.R"
        path.write_text(_WGCNA_R)
        _WGCNA_SCRIPT = path
    return _WGCNA_SCRIPT


def wgcna_batch(datasets, powers=WGCNA_POWERS, min_sizes=WGCNA_MIN_SIZES):
    """Run the full WGCNA pipeline over many datasets in a single R process.

    ``datasets`` is a sequence of ``(genes x samples)`` arrays.  For each one we
    run ``blockwiseModules`` (soft threshold, TOM, dynamic hybrid tree cut) at
    every ``(power, min_module_size)`` combination, plus the power WGCNA's own
    ``pickSoftThreshold`` recommends.

    Returns one dict per dataset::

        {"auto_power": int,
         "runs": [{"power": int, "min_module_size": int, "labels": ndarray}, ...]}

    Grey-module genes keep label 0, which matches the background class used in
    the noise-gene experiment.  Batching matters: one R startup instead of one
    per dataset turns hours of process overhead into seconds.
    """
    datasets = list(datasets)
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, X in enumerate(datasets):
            p = os.path.join(tmp, f"expr_{i:04d}.csv")
            np.savetxt(p, X, delimiter=",")
            paths.append(p)
        manifest = os.path.join(tmp, "manifest.txt")
        Path(manifest).write_text("\n".join(paths) + "\n")
        out_file = os.path.join(tmp, "wgcna.json")

        proc = subprocess.run(
            ["Rscript", "--vanilla", str(_wgcna_script()), manifest, out_file,
             ",".join(map(str, powers)), ",".join(map(str, min_sizes))],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not os.path.exists(out_file):
            raise RuntimeError(f"WGCNA batch failed:\n{proc.stderr[-3000:]}")
        raw = json.loads(Path(out_file).read_text())

    results = []
    for entry in raw:
        runs = [
            {"power": int(r["power"]),
             "min_module_size": int(r["minModuleSize"]),
             "labels": np.asarray(r["labels"], dtype=int)}
            for r in entry["runs"]
        ]
        results.append({"auto_power": int(entry["autoPower"]), "runs": runs})
    return results


def wgcna_default(entry):
    """The labelling WGCNA produces at its own recommended power, minModuleSize 10.

    This is what a practitioner following the standard tutorial would get.
    """
    for r in entry["runs"]:
        if r["power"] == entry["auto_power"] and r["min_module_size"] == 10:
            return r["labels"]
    return entry["runs"][0]["labels"]


def wgcna_oracle(entry, truth, score):
    """The best labelling WGCNA achieves over the whole tuning grid.

    Selecting the configuration by its score *against ground truth* gives WGCNA
    an oracle advantage no real user has.  We report it so that any margin over
    WGCNA cannot be attributed to unlucky hyper-parameters.
    """
    best, best_s = None, -np.inf
    for r in entry["runs"]:
        s = score(truth, r["labels"])
        if s > best_s:
            best, best_s = r, s
    return best["labels"], best


# --------------------------------------------------------------------------- #
def available() -> dict:
    """Report which optional baselines can actually run in this environment."""
    status = {}
    for name, probe in [
        ("spectral", lambda: __import__("sklearn.cluster", fromlist=["SpectralClustering"])),
        ("louvain", lambda: __import__("igraph")),
        ("leiden", lambda: __import__("leidenalg")),
        ("mcl", lambda: __import__("markov_clustering")),
    ]:
        try:
            probe()
            status[name] = True
        except Exception:
            status[name] = False
    try:
        proc = subprocess.run(
            ["Rscript", "--vanilla", "-e",
             'cat(requireNamespace("WGCNA", quietly=TRUE))'],
            capture_output=True, text=True, timeout=300,
        )
        status["wgcna"] = proc.stdout.strip().endswith("TRUE")
    except Exception:
        status["wgcna"] = False
    return status


if __name__ == "__main__":
    print(json.dumps(available(), indent=1))
