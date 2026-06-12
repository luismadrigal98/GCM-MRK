"""Memetic local search for the correlation-based objective.

The genetic operators (crossover, mutation) explore the partition space globally
but coarsely; on their own they leave the optimiser short of the optimum, even
when the seed is a good hierarchical clustering.  This module adds a greedy
hill-climb that repeatedly performs the single element reassignment that most
improves the objective, until no improving move remains.  Crucially this is a
move the greedy *agglomerative* clusterers used for seeding cannot make: once
they merge two elements they can never separate them, whereas reassignment can
undo early mistakes.  Hybridising the GA with this refinement (a memetic /
Lamarckian GA) is what lets the optimiser reach partitions that hierarchical
clustering cannot.

The objective is the correlation family (``loglik`` and its penalised variants),
all of which are functions of the per-cluster log-likelihood terms and the number
of coherent clusters ``K``.  The gain of a single move touches only the source
and target clusters, so it is evaluated incrementally in ``O(cluster size)``
rather than recomputing the whole score; a full sweep is ``O(n^2)``.
"""

from __future__ import annotations

import numpy as np

from .metrics import _PERFECT_FIT_CAP
from .partition import consolidate_labels

__all__ = ["CorrelationRefiner"]


def _term(ns: int, cs: float) -> float:
    """Single-cluster contribution to the (pre-0.5) correlation log-likelihood.

    Mirrors :func:`gcmrk.metrics.log_likelihood_correlation` exactly so that the
    incremental score matches the metric the GA reports.
    """
    if ns <= 1:
        return 0.0
    denom = float(ns) * ns - cs
    if denom <= 0.0:
        return _PERFECT_FIT_CAP
    log_term = np.log(ns / cs) + (ns - 1) * np.log((float(ns) * ns - ns) / denom)
    if np.isfinite(log_term):
        return float(log_term)
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
        penalty so the hill-climb optimises exactly the reported objective.
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
            self.alpha, self.beta = 1.0, 0.0           # maximise loglik
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
        clusters = [c for c in np.unique(labels) if c != 0]
        if len(clusters) < 2:
            return labels

        # Per-cluster member counts and within-cluster |corr| sums (incl diagonal).
        sizes = {c: int(np.sum(labels == c)) for c in clusters}
        C = {}
        for c in clusters:
            idx = np.where(labels == c)[0]
            C[c] = float(A[np.ix_(idx, idx)].sum())

        score_ll = self._loglik(sizes, C)
        score_k = self._k(sizes)

        for _ in range(self.max_pass):
            improved = False
            for i in range(self.n):
                a = int(labels[i])
                if a == 0:
                    continue
                r_a = float(A[i, labels == a].sum()) - A[i, i]   # excl. self
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
                    r_b = float(A[i, labels == b].sum())
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
                    improved = True
            if not improved:
                break

        return np.asarray(consolidate_labels(labels), dtype=int)
