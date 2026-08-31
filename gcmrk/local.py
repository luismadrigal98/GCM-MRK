"""Memetic local search for the correlation-based objective.

The genetic operators (crossover, mutation) explore the partition space globally
but coarsely; on their own they leave the optimizer short of the optimum, even
when the seed is a good hierarchical clustering.  This module adds a greedy
hill-climb that repeatedly performs the single element reassignment that most
improves the objective, until no improving move remains.  Crucially this is a
move the greedy *agglomerative* clusterers used for seeding cannot make: once
they merge two elements they can never separate them, whereas reassignment can
undo early mistakes.  Hybridising the GA with this refinement (a memetic /
Lamarckian GA) is what lets the optimizer reach partitions that hierarchical
clustering cannot.

The objective is the correlation family (``loglik`` and its penalized variants),
all of which are functions of the per-cluster log-likelihood terms and the number
of coherent clusters ``K``.  The gain of a single move touches only the source
and target clusters, so it is evaluated incrementally in ``O(cluster size)``
rather than recomputing the whole score; a full sweep is ``O(n^2)``.
"""

from __future__ import annotations

from math import isfinite as _isfinite, log as _log

import numpy as np

from .metrics import _PERFECT_FIT_CAP, _factor_block_term, factor_rank
from .metrics import FACTOR_METRICS
from .partition import consolidate_labels

__all__ = ["CorrelationRefiner", "FactorRefiner"]


def _term(ns: int, cs: float) -> float:
    """Single-cluster contribution to the (pre-0.5) correlation log-likelihood.

    Mirrors :func:`gcmrk.metrics.log_likelihood_correlation` exactly so that the
    incremental score matches the metric the GA reports.

    Uses :mod:`math` rather than numpy: these are Python scalars, and this is the
    hottest function in the package (millions of calls per run), where numpy's
    per-call scalar overhead dominates.
    """
    if ns <= 1:
        return 0.0
    denom = float(ns) * ns - cs
    if denom <= 0.0 or cs <= 0.0:
        return _PERFECT_FIT_CAP
    try:
        log_term = _log(ns / cs) + (ns - 1) * _log((float(ns) * ns - ns) / denom)
    except (ValueError, ZeroDivisionError):
        return _PERFECT_FIT_CAP
    if _isfinite(log_term):
        return log_term
    return _PERFECT_FIT_CAP


class CorrelationRefiner:
    """Greedy reassignment refinement for a correlation-family objective.

    Parameters
    ----------
    abscor:
        ``|R|`` -- the element-by-element absolute correlation matrix (diagonal
        ones).
    metric_name:
        ``"loglik"``, ``"loglik_bic"`` or ``"loglik_aic"``; sets the cluster-count
        penalty so the hill-climb optimizes exactly the reported objective.
    n_samples:
        Number of samples (columns); calibrates the BIC/AIC penalty.
    g_max, all_in_clusters:
        Partition constraints to respect (never drop below two clusters; never
        assign the unassigned label).
    max_pass:
        Maximum number of full sweeps over all elements.
    """

    def __init__(self, abscor: np.ndarray, metric_name: str, n_samples: int,
                 g_max: int, all_in_clusters: bool, max_pass: int = 25):
        self.A = np.asarray(abscor, dtype=float)
        self.n = self.A.shape[0]
        self.g_max = g_max
        self.all_in_clusters = all_in_clusters
        self.max_pass = max_pass
        d = int(n_samples)
        if metric_name == "loglik":
            self.alpha, self.beta = 1.0, 0.0           # maximize loglik
        elif metric_name == "loglik_bic":
            self.alpha, self.beta = 2.0 * d, float(np.log(max(d, 2)))
        elif metric_name == "loglik_aic":
            self.alpha, self.beta = 2.0 * d, 2.0
        else:
            raise ValueError(f"local search not defined for metric {metric_name!r}")

    @staticmethod
    def supports(metric_names) -> bool:
        return all(m in ("loglik", "loglik_bic", "loglik_aic") for m in metric_names)

    def _loglik(self, sizes: dict, C: dict) -> float:
        return 0.5 * sum(_term(sizes[c], C[c]) for c in sizes)

    def _k(self, sizes: dict) -> int:
        return sum(1 for c in sizes if sizes[c] > 1)

    def refine(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels, dtype=int).copy()
        A = self.A
        clusters = [int(c) for c in np.unique(labels) if c != 0]
        if len(clusters) < 2:
            return labels

        col = {c: j for j, c in enumerate(clusters)}
        sizes = {c: int(np.sum(labels == c)) for c in clusters}
        C = {}
        for c in clusters:
            idx = np.where(labels == c)[0]
            C[c] = float(A[np.ix_(idx, idx)].sum())

        # Affinity matrix: S[i, j] is the summed |corr| between element i and
        # every member of cluster j.  Computing it once as a single matmul and
        # updating it in O(n) per accepted move replaces the O(n) masked sum
        # this loop previously did for every (element, candidate cluster) pair,
        # which dominated the runtime.
        Z = np.zeros((self.n, len(clusters)))
        assigned = labels != 0
        Z[np.where(assigned)[0], [col[int(c)] for c in labels[assigned]]] = 1.0
        S = A @ Z

        for _ in range(self.max_pass):
            improved = False
            for i in range(self.n):
                a = int(labels[i])
                if a == 0:
                    continue
                r_a = float(S[i, col[a]]) - A[i, i]
                term_a_old = _term(sizes[a], C[a])
                na_new = sizes[a] - 1
                Ca_new = C[a] - 2.0 * r_a - A[i, i]
                term_a_new = _term(na_new, Ca_new)
                # Don't let the partition collapse below two clusters.
                emptying_a = na_new == 0
                live_clusters = sum(1 for c in sizes if sizes[c] > 0)
                if emptying_a and live_clusters <= 2:
                    continue

                best_gain = 1e-9
                best_b = None
                best_pack = None
                for b in sizes:
                    if b == a or sizes[b] == 0:
                        continue
                    r_b = float(S[i, col[b]])
                    term_b_old = _term(sizes[b], C[b])
                    nb_new = sizes[b] + 1
                    Cb_new = C[b] + 2.0 * r_b + A[i, i]
                    term_b_new = _term(nb_new, Cb_new)
                    dll = 0.5 * ((term_a_new - term_a_old) + (term_b_new - term_b_old))
                    dk = ((1 if na_new > 1 else 0) - (1 if sizes[a] > 1 else 0)
                          + (1 if nb_new > 1 else 0) - (1 if sizes[b] > 1 else 0))
                    gain = self.alpha * dll - self.beta * dk
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b
                        best_pack = (Ca_new, na_new, Cb_new, nb_new)

                if best_b is not None:
                    Ca_new, na_new, Cb_new, nb_new = best_pack
                    labels[i] = best_b
                    sizes[a] = na_new
                    C[a] = Ca_new
                    sizes[best_b] = nb_new
                    C[best_b] = Cb_new
                    # Element i left a and joined best_b: shift its column of
                    # affinities between the two clusters.
                    S[:, col[a]] -= A[:, i]
                    S[:, col[best_b]] += A[:, i]
                    improved = True
            if not improved:
                break

        return np.asarray(consolidate_labels(labels), dtype=int)


class FactorRefiner:
    """Greedy reassignment refinement for the ``loglik_factor`` family.

    The exchangeable objective decomposes into per-cluster ``(size, |corr| sum)``
    terms, which :class:`CorrelationRefiner` updates in ``O(cluster size)`` per
    move.  The factor model has no such decomposition: moving one element changes
    the block's eigenvalues, so the affected blocks must be re-solved.  We
    re-solve only the two clusters a move touches, and only for their top ``q``
    eigenvalues, which keeps a full sweep at ``O(n K q n_m^2)``.

    That is materially heavier than the exchangeable refiner.  Above
    ``max_elements`` rows the refinement is skipped rather than allowed to
    dominate the run; the GA still optimizes the factor objective, just without
    the hill-climb.

    Parameters
    ----------
    cor:
        The **signed** correlation matrix.  The factor model represents
        anti-correlated module members through negative loadings, so unlike the
        exchangeable objective it must not be given ``|R|``.
    metric_name:
        ``"loglik_factor"``, ``"loglik_factor_bic"`` or ``"loglik_factor_aic"``,
        optionally with an ``@q`` rank suffix.
    n_samples:
        Number of samples (columns); scales the likelihood against the penalty.
    """

    def __init__(self, cor: np.ndarray, metric_name: str, n_samples: int,
                 g_max: int, all_in_clusters: bool, max_pass: int = 10,
                 max_elements: int = 1500, n_candidates: int = 3):
        self.R = np.asarray(cor, dtype=float)
        self.absR = np.abs(self.R)
        self.n_candidates = max(int(n_candidates), 1)
        self.n = self.R.shape[0]
        self.q = factor_rank(metric_name)
        self.d = int(n_samples)
        self.g_max = g_max
        self.all_in_clusters = all_in_clusters
        self.max_pass = max_pass
        self.max_elements = max_elements

        base = metric_name.partition("@")[0]
        if base == "loglik_factor":
            self.half_penalty = 0.0
        elif base == "loglik_factor_bic":
            self.half_penalty = 0.5 * float(np.log(max(self.d, 2)))
        elif base == "loglik_factor_aic":
            self.half_penalty = 1.0
        else:
            raise ValueError(f"factor local search not defined for {metric_name!r}")

    @staticmethod
    def supports(metric_names) -> bool:
        return all(m.partition("@")[0] in FACTOR_METRICS for m in metric_names)

    def _term(self, idx: np.ndarray) -> float:
        """Log-likelihood contribution of the block spanned by ``idx``."""
        if idx.size <= 1:
            return 0.0
        return _factor_block_term(self.R[np.ix_(idx, idx)], self.q, self.d)

    def _params(self, n_m: int) -> int:
        if n_m <= 1:
            return 0
        qe = min(self.q, n_m - 1)
        if qe < 1:
            return 0
        return n_m * qe - qe * (qe - 1) // 2 + 1

    def _affinity(self, labels, cluster_ids):
        """Mean |correlation| between every element and every cluster.

        Used only to rank candidate destinations, never to score a move: a full
        eigen-solve still decides. Screening to the ``n_candidates`` most
        plausible targets is what keeps the sweep affordable, since a gene is
        essentially never best placed in a module it correlates weakly with.
        """
        ind = np.zeros((self.n, len(cluster_ids)))
        for j, c in enumerate(cluster_ids):
            ind[labels == c, j] = 1.0
        counts = np.maximum(ind.sum(axis=0), 1.0)
        return (self.absR @ ind) / counts

    def refine(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels, dtype=int).copy()
        if self.n > self.max_elements:
            return labels
        clusters = [int(c) for c in np.unique(labels) if c != 0]
        if len(clusters) < 2:
            return labels

        members = {c: np.where(labels == c)[0] for c in clusters}
        term = {c: self._term(members[c]) for c in clusters}

        for _ in range(self.max_pass):
            improved = False
            live = [c for c in members if members[c].size > 0]
            if len(live) < 2:
                break
            aff = self._affinity(labels, live)
            order = np.argsort(-aff, axis=1)[:, :self.n_candidates]

            for i in range(self.n):
                a = int(labels[i])
                if a == 0:
                    continue
                idx_a_new = members[a][members[a] != i]
                if idx_a_new.size == 0 and sum(
                        1 for c in members if members[c].size > 0) <= 2:
                    continue
                term_a_new = self._term(idx_a_new)
                d_params_a = self._params(idx_a_new.size) - self._params(members[a].size)

                best_gain, best_b, best_pack = 1e-9, None, None
                for j in order[i]:
                    b = live[j]
                    if b == a or members[b].size == 0:
                        continue
                    idx_b_new = np.append(members[b], i)
                    term_b_new = self._term(idx_b_new)
                    d_params = d_params_a + (self._params(idx_b_new.size)
                                             - self._params(members[b].size))
                    gain = ((term_a_new - term[a]) + (term_b_new - term[b])
                            - self.half_penalty * d_params)
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b
                        best_pack = (idx_a_new, term_a_new, idx_b_new, term_b_new)

                if best_b is not None:
                    idx_a_new, term_a_new, idx_b_new, term_b_new = best_pack
                    labels[i] = best_b
                    members[a], term[a] = idx_a_new, term_a_new
                    members[best_b], term[best_b] = idx_b_new, term_b_new
                    improved = True
            if not improved:
                break

        return np.asarray(consolidate_labels(labels), dtype=int)
