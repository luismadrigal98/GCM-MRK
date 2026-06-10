"""Lightweight k-means used to seed the genetic algorithm.

The original project seeded its population from external clustering tools
(k-means, WGCNA, Clust) via the ``InpPartition`` class.  This module provides a
dependency-free (numpy-only) k-means so the GA can bootstrap itself with strong
candidate partitions without requiring scikit-learn.

For row-standardised data, Euclidean k-means is monotonically related to the
correlation structure, so these seeds help the correlation-based objective just
as much as the geometric ones.
"""

from __future__ import annotations

import numpy as np

__all__ = ["kmeans", "kmeans_seeds"]


def kmeans(X: np.ndarray, k: int, rng: np.random.Generator,
           n_init: int = 3, max_iter: int = 50) -> np.ndarray:
    """k-means clustering returning 1-based labels.

    Uses k-means++ initialisation and keeps the best of ``n_init`` restarts by
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
    """k-means++ centroid initialisation."""
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
