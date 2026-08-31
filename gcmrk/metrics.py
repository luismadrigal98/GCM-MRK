"""Internal cluster-validity metrics used as optimization targets.

Every metric reduces a clustering (a 1-D vector of integer labels) to a single
scalar score.  These scalars are the "performance targets" that drive the
genetic algorithm: the user chooses which ones to optimize and in which
direction.

A label of ``0`` denotes an *unassigned* element.  Metrics ignore unassigned
elements when measuring cluster quality, but an optional ``unassigned_penalty``
nudges the score so that leaving elements out is not free.

Conventions
-----------
Each public metric is registered in :data:`METRICS` together with its natural
optimization ``direction`` (``"max"`` or ``"min"``).  The genetic algorithm
turns a direction into a fitness weight, so metric functions never need to know
whether they are being maximized or minimized.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from math import isfinite as _isfinite, log as _log
from typing import Callable, Dict

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

__all__ = [
    "log_likelihood_correlation",
    "loglik_bic",
    "loglik_aic",
    "factor_loglik",
    "factor_bic",
    "factor_aic",
    "factor_rank",
    "FACTOR_METRICS",
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
    "gaussian_log_likelihood",
    "bic",
    "aic",
    "MetricSpec",
    "METRICS",
    "get_metric",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
# Finite stand-in for the +inf contributed by a perfectly correlated cluster.
# Matches the magnitude used to penalize degenerate partitions in the GA's
# weighted-sum normalization, so a perfect cluster is strongly (but finitely)
# rewarded rather than crashing the metric or being dropped.
_PERFECT_FIT_CAP = 1e6


def _as_labels(labels) -> np.ndarray:
    """Return labels as a 1-D integer numpy array."""
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError("labels must be a 1-D array or list")
    return arr.astype(int)


def _assigned_mask(labels: np.ndarray) -> np.ndarray:
    """Boolean mask of elements that belong to a real cluster (label != 0)."""
    return labels != 0


def _n_clusters(labels: np.ndarray) -> int:
    """Number of non-zero clusters present in ``labels``."""
    assigned = labels[_assigned_mask(labels)]
    return int(np.unique(assigned).size)


def _unassigned_fraction(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    return float(np.sum(~_assigned_mask(labels)) / labels.size)


# --------------------------------------------------------------------------- #
# The novel correlation-based log-likelihood (preserved from the original work)
# --------------------------------------------------------------------------- #
def log_likelihood_correlation(cor: np.ndarray, labels) -> float:
    """Correlation-based log-likelihood of a clustering solution.

    This is the bespoke objective at the heart of the project.  It rewards
    partitions whose within-cluster elements are strongly correlated.  ``cor``
    is the (n, n) sample-by-sample correlation matrix; ``labels`` assigns each
    sample to a cluster.  Higher is better.

    Unassigned elements (label ``0``) form their own singleton-like group and
    contribute little, so they are simply included as one of the unique labels.
    """
    cor = np.asarray(cor, dtype=float)
    labels = _as_labels(labels)
    if cor.ndim != 2 or cor.shape[0] != cor.shape[1]:
        raise ValueError("cor must be a square 2-D matrix")
    if cor.shape[0] != labels.size:
        raise ValueError("cor and labels must have compatible shapes")

    log_likelihood = 0.0
    for s in np.unique(labels):
        idx = np.where(labels == s)[0]
        ns = int(idx.size)
        if ns <= 1:
            # A singleton carries no within-cluster correlation information.
            continue
        # Sum of absolute correlations within the cluster (includes the
        # diagonal of ones, i.e. each element's self-correlation).  Because the
        # diagonal contributes ``ns`` ones, ``cs`` is bounded in ``[ns, ns**2]``.
        # Indexing the block directly allocates ns**2 rather than the n**2
        # boolean mask a `np.outer` selection would build, which matters a great
        # deal on genome-scale inputs.
        cs = float(np.abs(cor[np.ix_(idx, idx)]).sum())
        # ``denom`` -> 0 as the cluster becomes perfectly correlated (cs -> ns**2).
        # In that limit the log term diverges to +inf, i.e. the *best* possible
        # fit.  We must NOT let this raise (a Python-scalar division by zero is
        # not caught by ``np.errstate``) nor silently drop the cluster: instead
        # we award a large finite cap so a perfect cluster is rewarded, not lost.
        denom = float(ns) * ns - cs
        if denom <= 0.0 or cs <= 0.0:
            log_likelihood += _PERFECT_FIT_CAP
            continue
        try:
            log_term = _log(ns / cs) + (ns - 1) * _log(
                (float(ns) * ns - ns) / denom
            )
        except (ValueError, ZeroDivisionError):
            log_likelihood += _PERFECT_FIT_CAP
            continue
        if _isfinite(log_term):
            log_likelihood += log_term
        else:
            # Numerical overflow in the limit -> treat as a perfect-fit cluster.
            log_likelihood += _PERFECT_FIT_CAP

    return 0.5 * log_likelihood


def _n_coherent_clusters(labels: np.ndarray) -> int:
    """Number of clusters with at least two members (size-1 clusters carry no
    coherence parameter and do not contribute to the log-likelihood)."""
    assigned = labels[_assigned_mask(labels)]
    if assigned.size == 0:
        return 0
    _, counts = np.unique(assigned, return_counts=True)
    return int(np.sum(counts > 1))


def _penalized_correlation_loglik(cor: np.ndarray, labels, n_samples: int,
                                  *, penalty_per_cluster: float) -> float:
    r"""Information criterion for the equal-magnitude one-factor block model.

    Under that generative model (each module driven by one latent factor with
    equal-magnitude :math:`\pm` loadings; modules mutually independent), the
    Gaussian profile log-likelihood of a partition is, up to a
    partition-independent constant,

    .. math:: \ell(\ell) = d\,\mathcal{L}(\ell),

    where :math:`d` = ``n_samples`` and :math:`\mathcal{L}` is
    :func:`log_likelihood_correlation` (the maximum-likelihood common coherence
    of a module equals its mean absolute off-diagonal correlation, at which the
    trace term of the likelihood collapses to the dimension; see the paper's
    appendix for the derivation).  Each module contributes a single coherence
    parameter, so a partition with ``K`` non-singleton modules is scored

    .. math:: -2\,\ell(\ell) + \text{penalty\_per\_cluster}\cdot K ,

    minimized.  ``penalty_per_cluster = log(d)`` gives BIC, ``2`` gives AIC.
    Crucially the likelihood scales with ``d`` while the penalty does not, so
    more samples correctly support more modules.
    """
    cor = np.asarray(cor, dtype=float)
    labels = _as_labels(labels)
    ll = log_likelihood_correlation(cor, labels)
    if not np.isfinite(ll):
        return np.inf
    k = _n_coherent_clusters(labels)
    if k < 1:
        return np.inf
    d = int(n_samples)
    return -2.0 * d * ll + penalty_per_cluster * k


def loglik_bic(cor: np.ndarray, labels, n_samples: int) -> float:
    """BIC for the one-factor block model (lower is better).

    Penalty ``log(d)`` per module.  Selects the number of correlation modules in
    a single objective without leaving correlation space.  See
    :func:`_penalized_correlation_loglik`.
    """
    d = max(int(n_samples), 2)
    return _penalized_correlation_loglik(cor, labels, n_samples,
                                         penalty_per_cluster=float(np.log(d)))


def loglik_aic(cor: np.ndarray, labels, n_samples: int) -> float:
    """AIC for the one-factor block model (lower is better).

    Penalty ``2`` per module: a lighter complexity penalty than
    :func:`loglik_bic`, so it tolerates a few more modules.
    """
    return _penalized_correlation_loglik(cor, labels, n_samples,
                                         penalty_per_cluster=2.0)


# --------------------------------------------------------------------------- #
# Free-loading factor block model (the `loglik_factor` branch)
# --------------------------------------------------------------------------- #
# Smallest eigenvalue / residual variance we will take a logarithm of.
_EIG_FLOOR = 1e-9


def _top_eigenvalues(block: np.ndarray, q: int) -> np.ndarray:
    """The ``q`` largest eigenvalues of a symmetric block, descending."""
    n = block.shape[0]
    q = min(q, n - 1)
    if q < 1:
        return np.empty(0)
    if n <= 64:
        # For small blocks the full solver beats the subset driver's overhead.
        e = np.linalg.eigvalsh(block)[::-1][:q]
    else:
        from scipy.linalg import eigh
        e = eigh(block, eigvals_only=True, subset_by_index=[n - q, n - 1])[::-1]
    return np.maximum(e, _EIG_FLOOR)


def _factor_block_term(block: np.ndarray, q: int, d: int) -> float:
    r"""Probabilistic-PCA log-likelihood of one module's correlation block.

    Models the block as :math:`q` free-loading factors plus isotropic residual,
    :math:`\Sigma = \Lambda\Lambda^{\top} + \sigma^2 I`.  At the maximum-likelihood
    solution (Tipping & Bishop) the profile log-likelihood over ``d`` samples
    depends on the block only through its leading eigenvalues:

    .. math::
        \ell_m = -\tfrac{d}{2}\Big[\sum_{i\le q}\log e_i
                 + (n_m-q)\log\sigma^2 + n_m\Big],
        \qquad \sigma^2 = \frac{n_m - \sum_{i\le q} e_i}{n_m - q},

    using :math:`\operatorname{tr}(R_m) = n_m` for a correlation block.  Only the
    top ``q`` eigenvalues are needed, which is what keeps this affordable inside
    the GA's inner loop.

    Unlike :func:`log_likelihood_correlation`, this uses the **signed**
    correlation matrix: a factor with mixed-sign loadings reproduces an
    anti-correlated module natively, so no absolute value is required.  It also
    drops the exchangeability assumption --- module members may carry different
    loading magnitudes, which is what the equal-magnitude model cannot express.
    """
    n = block.shape[0]
    if n <= q or n <= 1:
        return 0.0
    e = _top_eigenvalues(block, q)
    if e.size == 0:
        return 0.0
    qe = int(e.size)
    resid = float(n) - float(e.sum())
    if resid <= _EIG_FLOOR:
        # Perfect low-rank fit: the likelihood diverges to +inf.  Award the same
        # large finite value the exchangeable objective uses, so a perfect module
        # is strongly rewarded rather than crashing the metric.
        return _PERFECT_FIT_CAP
    sigma2 = resid / (n - qe)
    term = float(np.log(e).sum()) + (n - qe) * float(np.log(sigma2)) + n
    if not np.isfinite(term):
        return _PERFECT_FIT_CAP
    return -0.5 * d * term


def factor_loglik(cor: np.ndarray, labels, n_samples: int, q: int = 1) -> float:
    """Log-likelihood of the rank-``q`` free-loading block model (maximize).

    The ``loglik`` objective assumes each module is one latent factor with
    equal-magnitude :math:`\\pm` loadings, so it sees a block only through its
    mean absolute correlation and treats all member pairs as exchangeable.  That
    is efficient when the assumption holds and actively misleading when it does
    not: modules whose members carry unequal loadings (hub-and-spoke) or that
    span several factors score worse than tighter spurious groupings.

    This objective replaces that block model with ``q`` freely-loaded factors,
    fitted per module by :func:`_factor_block_term`.  Use ``q=1`` for modules
    with one regulator but heterogeneous response strength, and larger ``q``
    where modules are driven by several factors.  ``q`` is selectable through the
    metric name, e.g. ``loglik_factor@3``.
    """
    cor = np.asarray(cor, dtype=float)
    labels = _as_labels(labels)
    if cor.ndim != 2 or cor.shape[0] != cor.shape[1]:
        raise ValueError("cor must be a square 2-D matrix")
    if cor.shape[0] != labels.size:
        raise ValueError("cor and labels must have compatible shapes")

    q = max(int(q), 1)
    d = int(n_samples)
    total = 0.0
    for s in np.unique(labels[_assigned_mask(labels)]):
        idx = np.where(labels == s)[0]
        if idx.size <= 1:
            continue
        total += _factor_block_term(cor[np.ix_(idx, idx)], q, d)
    return total


def _factor_n_params(labels: np.ndarray, q: int) -> int:
    r"""Free parameters of the rank-``q`` block model.

    A module of :math:`n_m` variables contributes :math:`n_m q - q(q-1)/2`
    loadings (discounting rotational redundancy) plus one residual variance.
    Unlike the exchangeable model's single parameter per module, this **scales
    with module size**, so the information criteria below penalize splitting a
    module far more realistically.
    """
    total = 0
    for s in np.unique(labels[_assigned_mask(labels)]):
        n_m = int(np.sum(labels == s))
        if n_m <= 1:
            continue
        qe = min(int(q), n_m - 1)
        if qe < 1:
            continue
        total += n_m * qe - qe * (qe - 1) // 2 + 1
    return int(total)


def _penalized_factor_loglik(cor, labels, n_samples, *, penalty_per_param,
                             q) -> float:
    cor = np.asarray(cor, dtype=float)
    labels = _as_labels(labels)
    ll = factor_loglik(cor, labels, n_samples, q=q)
    if not np.isfinite(ll):
        return np.inf
    p = _factor_n_params(labels, q)
    if p < 1:
        return np.inf
    return -2.0 * ll + penalty_per_param * p


def factor_bic(cor: np.ndarray, labels, n_samples: int, q: int = 1) -> float:
    """BIC for the rank-``q`` free-loading block model (lower is better)."""
    d = max(int(n_samples), 2)
    return _penalized_factor_loglik(cor, labels, n_samples,
                                    penalty_per_param=float(np.log(d)), q=q)


def factor_aic(cor: np.ndarray, labels, n_samples: int, q: int = 1) -> float:
    """AIC for the rank-``q`` free-loading block model (lower is better)."""
    return _penalized_factor_loglik(cor, labels, n_samples,
                                    penalty_per_param=2.0, q=q)


# --------------------------------------------------------------------------- #
# Standard internal validity indices (numpy / scipy implementations)
# --------------------------------------------------------------------------- #
def silhouette(X: np.ndarray, labels, metric: str = "euclidean",
               unassigned_penalty: float = 0.0) -> float:
    """Mean silhouette coefficient over assigned samples (higher is better).

    Range is ``[-1, 1]``.  Requires at least two clusters; otherwise returns
    ``-1`` (the worst possible value) so the GA avoids degenerate partitions.
    """
    X = np.asarray(X, dtype=float)
    labels = _as_labels(labels)
    mask = _assigned_mask(labels)
    Xa, la = X[mask], labels[mask]

    if _n_clusters(labels) < 2 or Xa.shape[0] < 2:
        return -1.0

    D = squareform(pdist(Xa, metric=metric)) if Xa.shape[0] > 1 else np.zeros((1, 1))
    unique = np.unique(la)
    sil = np.zeros(Xa.shape[0], dtype=float)

    for i in range(Xa.shape[0]):
        own = la == la[i]
        own_count = np.sum(own) - 1
        if own_count <= 0:
            sil[i] = 0.0  # singleton cluster -> silhouette defined as 0
            continue
        a_i = np.sum(D[i, own]) / own_count
        b_i = np.inf
        for c in unique:
            if c == la[i]:
                continue
            other = la == c
            b_i = min(b_i, np.mean(D[i, other]))
        sil[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0

    score = float(np.mean(sil))
    return score - unassigned_penalty * _unassigned_fraction(labels)


def davies_bouldin(X: np.ndarray, labels, unassigned_penalty: float = 0.0) -> float:
    """Davies-Bouldin index (lower is better).

    Mean over clusters of the worst-case ratio between within-cluster scatter
    and between-cluster centroid distance.  Returns ``inf`` when fewer than two
    clusters are present, so the GA is pushed away from degenerate partitions.
    """
    X = np.asarray(X, dtype=float)
    labels = _as_labels(labels)
    mask = _assigned_mask(labels)
    Xa, la = X[mask], labels[mask]

    unique = np.unique(la)
    k = unique.size
    if k < 2:
        return np.inf

    centroids = np.array([Xa[la == c].mean(axis=0) for c in unique])
    scatter = np.array([
        np.mean(np.linalg.norm(Xa[la == c] - centroids[idx], axis=1))
        for idx, c in enumerate(unique)
    ])

    centroid_dist = squareform(pdist(centroids))
    db = 0.0
    for i in range(k):
        ratios = [
            (scatter[i] + scatter[j]) / centroid_dist[i, j]
            for j in range(k) if j != i and centroid_dist[i, j] > 0
        ]
        db += max(ratios) if ratios else 0.0
    score = db / k
    return score + unassigned_penalty * _unassigned_fraction(labels)


def calinski_harabasz(X: np.ndarray, labels, unassigned_penalty: float = 0.0) -> float:
    """Calinski-Harabasz index / variance-ratio criterion (higher is better)."""
    X = np.asarray(X, dtype=float)
    labels = _as_labels(labels)
    mask = _assigned_mask(labels)
    Xa, la = X[mask], labels[mask]

    unique = np.unique(la)
    k = unique.size
    n = Xa.shape[0]
    if k < 2 or n <= k:
        return 0.0

    overall = Xa.mean(axis=0)
    between, within = 0.0, 0.0
    for c in unique:
        members = Xa[la == c]
        centroid = members.mean(axis=0)
        between += members.shape[0] * np.sum((centroid - overall) ** 2)
        within += np.sum((members - centroid) ** 2)
    if within == 0:
        return 0.0
    score = (between / (k - 1)) / (within / (n - k))
    return score - unassigned_penalty * _unassigned_fraction(labels)


def gaussian_log_likelihood(X: np.ndarray, labels) -> float:
    """Log-likelihood of the data under a spherical-Gaussian-per-cluster model.

    Used as the basis for :func:`bic` and :func:`aic`.  Each assigned cluster is
    modelled as an isotropic Gaussian; the variance is pooled across clusters
    (the standard k-means / x-means assumption), which keeps the estimate stable
    for small clusters.
    """
    X = np.asarray(X, dtype=float)
    labels = _as_labels(labels)
    mask = _assigned_mask(labels)
    Xa, la = X[mask], labels[mask]

    n, d = Xa.shape
    unique = np.unique(la)
    k = unique.size
    if n == 0 or k == 0:
        return float("-inf")

    # Pooled within-cluster sum of squares.
    wcss = 0.0
    for c in unique:
        members = Xa[la == c]
        centroid = members.mean(axis=0)
        wcss += np.sum((members - centroid) ** 2)

    if n - k <= 0:
        return float("-inf")
    variance = wcss / (d * (n - k)) if (n - k) > 0 else 0.0
    if variance <= 0:
        # Perfect fit; return a large finite likelihood to avoid +inf.
        variance = 1e-12

    counts = np.array([np.sum(la == c) for c in unique])
    ll = 0.0
    for cnt in counts:
        ll += (
            cnt * np.log(cnt / n)
            - cnt * d / 2.0 * np.log(2 * np.pi * variance)
            - (cnt - 1) * d / 2.0
        )
    return float(ll)


def _n_free_parameters(labels: np.ndarray, n_features: int) -> int:
    """Free parameters for a spherical-Gaussian mixture: k centroids + k weights
    + 1 pooled variance."""
    k = _n_clusters(labels)
    return k * n_features + (k - 1) + 1


def bic(X: np.ndarray, labels, unassigned_penalty: float = 0.0) -> float:
    """Bayesian Information Criterion (lower is better)."""
    X = np.asarray(X, dtype=float)
    labels = _as_labels(labels)
    ll = gaussian_log_likelihood(X, labels)
    n = int(np.sum(_assigned_mask(labels)))
    p = _n_free_parameters(labels, X.shape[1])
    if n <= 0 or not np.isfinite(ll):
        return np.inf
    score = -2.0 * ll + p * np.log(n)
    return score + unassigned_penalty * _unassigned_fraction(labels) * abs(score)


def aic(X: np.ndarray, labels, unassigned_penalty: float = 0.0) -> float:
    """Akaike Information Criterion (lower is better)."""
    X = np.asarray(X, dtype=float)
    labels = _as_labels(labels)
    ll = gaussian_log_likelihood(X, labels)
    p = _n_free_parameters(labels, X.shape[1])
    if not np.isfinite(ll):
        return np.inf
    score = -2.0 * ll + 2.0 * p
    return score + unassigned_penalty * _unassigned_fraction(labels) * abs(score)


# --------------------------------------------------------------------------- #
# Metric registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetricSpec:
    """Metadata describing one optimization target.

    Attributes
    ----------
    name:
        Identifier used on the command line and in results.
    direction:
        ``"max"`` if larger scores are better, ``"min"`` otherwise.
    needs:
        Which precomputed input the metric consumes -- ``"X"`` for the data
        matrix or ``"cor"`` for the correlation matrix.
    func:
        Callable ``func(payload, labels, **kwargs) -> float``.
    description:
        Human-readable summary shown in CLI help.
    wants_n:
        If true, the metric also receives ``n_samples`` (the number of samples /
        columns) -- needed by the correlation information criteria, whose penalty
        is calibrated against a likelihood that scales with the sample size.
    """

    name: str
    direction: str
    needs: str
    func: Callable
    description: str
    wants_n: bool = False

    @property
    def weight(self) -> float:
        """Fitness weight: +1 for maximization, -1 for minimization."""
        return 1.0 if self.direction == "max" else -1.0


METRICS: Dict[str, MetricSpec] = {
    "loglik": MetricSpec(
        "loglik", "max", "cor", log_likelihood_correlation,
        "Correlation-based log-likelihood (rewards correlated clusters).",
    ),
    "loglik_bic": MetricSpec(
        "loglik_bic", "min", "cor", loglik_bic,
        "BIC for the one-factor block model (selects module count).",
        wants_n=True,
    ),
    "loglik_aic": MetricSpec(
        "loglik_aic", "min", "cor", loglik_aic,
        "AIC for the one-factor block model (selects module count).",
        wants_n=True,
    ),
    "silhouette": MetricSpec(
        "silhouette", "max", "X", silhouette,
        "Mean silhouette coefficient in [-1, 1].",
    ),
    "davies_bouldin": MetricSpec(
        "davies_bouldin", "min", "X", davies_bouldin,
        "Davies-Bouldin index (cluster separation / compactness).",
    ),
    "calinski_harabasz": MetricSpec(
        "calinski_harabasz", "max", "X", calinski_harabasz,
        "Calinski-Harabasz variance-ratio criterion.",
    ),
    "bic": MetricSpec(
        "bic", "min", "X", bic,
        "Bayesian Information Criterion (Gaussian mixture).",
    ),
    "aic": MetricSpec(
        "aic", "min", "X", aic,
        "Akaike Information Criterion (Gaussian mixture).",
    ),
    "loglik_factor": MetricSpec(
        "loglik_factor", "max", "cor", factor_loglik,
        "Rank-q free-loading block model (use loglik_factor@q to set q).",
        wants_n=True,
    ),
    "loglik_factor_bic": MetricSpec(
        "loglik_factor_bic", "min", "cor", factor_bic,
        "BIC for the rank-q free-loading block model.",
        wants_n=True,
    ),
    "loglik_factor_aic": MetricSpec(
        "loglik_factor_aic", "min", "cor", factor_aic,
        "AIC for the rank-q free-loading block model.",
        wants_n=True,
    ),
}

#: Metrics whose name accepts a trailing ``@q`` to set the factor rank.
FACTOR_METRICS = ("loglik_factor", "loglik_factor_bic", "loglik_factor_aic")


def factor_rank(name: str) -> int:
    """The factor rank encoded in a metric name (``loglik_factor@3`` -> 3)."""
    base, sep, arg = name.partition("@")
    if sep and base in FACTOR_METRICS:
        return max(int(arg), 1)
    return 1


def get_metric(name: str) -> MetricSpec:
    """Look up a metric by name, with a helpful error on typos.

    Factor metrics accept a rank suffix: ``loglik_factor@3`` optimizes the
    rank-3 model.  Without a suffix the rank is 1.
    """
    base, sep, arg = name.partition("@")
    if sep:
        if base not in FACTOR_METRICS:
            raise KeyError(
                f"Metric {base!r} takes no '@' argument. "
                f"Parameterizable metrics: {', '.join(FACTOR_METRICS)}"
            )
        try:
            q = int(arg)
        except ValueError:
            raise KeyError(f"Factor rank in {name!r} must be an integer")
        if q < 1:
            raise KeyError(f"Factor rank in {name!r} must be >= 1")
        spec = METRICS[base]
        return MetricSpec(name, spec.direction, spec.needs,
                          partial(spec.func, q=q), spec.description,
                          wants_n=spec.wants_n)
    try:
        return METRICS[name]
    except KeyError:
        raise KeyError(
            f"Unknown metric {name!r}. Available: {', '.join(sorted(METRICS))}"
        )
