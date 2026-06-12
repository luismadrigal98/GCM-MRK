"""Lightweight k-means used to seed the genetic algorithm.

The original project seeded its population from external clustering tools
(k-means, WGCNA, Clust) via the ``InpPartition`` class.  This module provides a
dependency-free (numpy-only) k-means so the GA can bootstrap itself with strong
candidate partitions without requiring scikit-learn.

For row-standardized data, Euclidean k-means is monotonically related to the
correlation structure, so these seeds help the correlation-based objective just
as much as the geometric ones.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

__all__ = ["kmeans", "kmeans_seeds", "correlation_seeds"]


def kmeans(X: np.ndarray, k: int, rng: np.random.Generator,
           n_init: int = 3, max_iter: int = 50) -> np.ndarray:
    """k-means clustering returning 1-based labels.

    Uses k-means++ initialization and keeps the best of ``n_init`` restarts by
    within-cluster sum of squares.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    k = min(k, n)
    if k <= 1:
        return np.ones(n, dtype=int)

    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        centroids = _kpp_init(X, k, rng)
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            dists = _sqdist(X, centroids)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            for c in range(k):
                members = X[labels == c]
                if members.size:
                    centroids[c] = members.mean(axis=0)
                else:
                    centroids[c] = X[rng.integers(0, n)]
        inertia = float(np.sum(np.min(_sqdist(X, centroids), axis=1)))
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels

    return best_labels + 1  # 1-based labels


def _sqdist(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between rows of X and each centroid."""
    return np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)


def _kpp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ centroid initialization."""
    n = X.shape[0]
    centroids = np.empty((k, X.shape[1]))
    centroids[0] = X[rng.integers(0, n)]
    closest = np.sum((X - centroids[0]) ** 2, axis=1)
    for i in range(1, k):
        total = closest.sum()
        if total == 0:
            centroids[i] = X[rng.integers(0, n)]
        else:
            probs = closest / total
            centroids[i] = X[rng.choice(n, p=probs)]
        closest = np.minimum(closest, np.sum((X - centroids[i]) ** 2, axis=1))
    return centroids


def kmeans_seeds(X: np.ndarray, g_max: int, rng: np.random.Generator,
                 restarts: int = 2) -> list:
    """Generate k-means seed partitions for every k in ``2..g_max``.

    Returns a list of 1-based label vectors (with ``restarts`` partitions per k)
    to inject into the GA's initial population.
    """
    seeds = []
    for k in range(2, max(2, g_max) + 1):
        for _ in range(restarts):
            seeds.append(kmeans(X, k, rng))
    return seeds


def correlation_seeds(cor: np.ndarray, g_max: int) -> list:
    """Seed partitions from hierarchical clustering of the correlation matrix.

    For a correlation-based objective, Euclidean k-means is a poor seeder:
    elements that are strongly *anti*-correlated (and so coherent under the
    absolute-correlation log-likelihood) are far apart in Euclidean space and
    get split.  This routine instead clusters on the correlation distance
    ``d = 1 - |R|`` with agglomerative linkage -- the classical co-expression
    clustering strategy -- which produces seeds that already respect the
    correlation structure the GA is asked to optimize.

    Two linkage methods (average and complete) are used for diversity; for each,
    one partition per ``k`` in ``2..g_max`` is returned as a 1-based label
    vector.
    """
    cor = np.asarray(cor, dtype=float)
    n = cor.shape[0]
    if n < 2:
        return [np.ones(n, dtype=int)]
    dist = 1.0 - np.abs(cor)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, None)  # symmetric, non-negative
    condensed = squareform(dist, checks=False)

    seeds = []
    for method in ("average", "complete"):
        Z = linkage(condensed, method=method)
        for k in range(2, max(2, g_max) + 1):
            seeds.append(fcluster(Z, t=k, criterion="maxclust").astype(int))
    return seeds
