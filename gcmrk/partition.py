"""Partition representation and repair operators.

A *partition* is a 1-D integer vector assigning each of ``n`` elements to a
cluster.  Two initialisation strategies are provided:

* random partitions (:func:`random_partition`), and
* partitions seeded from an external clustering file (:func:`read_partitions`),
  e.g. the output of k-means, WGCNA or Clust.

When ``all_in_clusters`` is true every element must belong to a cluster
(labels are ``1..k``).  Otherwise label ``0`` marks an unassigned element.

The repair helpers keep partitions valid during evolution: they relabel to
consecutive integers and guarantee at least two clusters so that the validity
metrics remain defined.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "random_partition",
    "read_partitions",
    "consolidate_labels",
    "repair",
]


def consolidate_labels(labels):
    """Relabel to consecutive integers, preserving the unassigned marker.

    If ``0`` (unassigned) is present, output labels start at ``0``; otherwise
    they start at ``1``.  Cluster identity is preserved, only the numbering is
    compacted.

    >>> consolidate_labels([6, 2, 0, 2, 1])
    [3, 2, 0, 2, 1]
    >>> consolidate_labels([9, 9, 2, 2, 4])
    [3, 3, 1, 1, 2]
    """
    labels = list(labels)
    if not all(isinstance(x, (int, np.integer)) or
               (isinstance(x, float) and float(x).is_integer())
               for x in labels):
        raise TypeError("Labels must be integers.")
    has_unassigned = 0 in labels

    if has_unassigned:
        # Map 0 -> 0, then the rest to 1, 2, ...
        others = sorted(set(labels) - {0})
        mapping = {0: 0}
        mapping.update({old: new for new, old in enumerate(others, start=1)})
    else:
        mapping = {old: new for new, old in enumerate(sorted(set(labels)), start=1)}

    try:
        return [mapping[label] for label in labels]
    except (KeyError, TypeError):
        raise TypeError("Labels must be integers.")


def random_partition(n_elements: int, g_max: int, all_in_clusters: bool,
                     rng: np.random.Generator) -> np.ndarray:
    """Generate a random partition of ``n_elements`` into up to ``g_max`` clusters.

    The realised number of clusters is itself random (between 2 and ``g_max``),
    which seeds the population with diverse cluster counts.
    """
    if not isinstance(n_elements, (int, np.integer)) or n_elements <= 0:
        raise ValueError("n_elements must be a positive integer")
    if not isinstance(g_max, (int, np.integer)) or g_max <= 0:
        raise ValueError("g_max must be a positive integer")

    low = 1 if all_in_clusters else 0
    # Ensure room for at least two distinct cluster ids.
    high = max(low + 1, int(rng.integers(low + 1, g_max + 1)) if g_max > low else low + 1)
    labels = rng.integers(low, high + 1, size=n_elements)
    return repair(labels, g_max, all_in_clusters, rng)


def read_partitions(path: str, sep: str = ",", expected_len: int | None = None):
    """Read seed partitions from a file: one partition per line.

    Returns a list of integer numpy arrays.  Non-integer tokens trigger a clear
    error.  Lines whose length does not match ``expected_len`` (when given) are
    skipped with no error so a partly-matching seed file is still usable.
    """
    partitions = []
    with open(path, "r") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            tokens = line.split(sep) if sep in line else line.split()
            try:
                vec = np.array([int(round(float(tok))) for tok in tokens], dtype=int)
            except ValueError:
                raise ValueError(
                    f"{path!r} line {lineno}: partition contains non-numeric values"
                )
            if expected_len is not None and vec.size != expected_len:
                continue
            partitions.append(vec)
    return partitions


def repair(labels, g_max: int, all_in_clusters: bool,
           rng: np.random.Generator) -> np.ndarray:
    """Return a valid partition derived from ``labels``.

    Guarantees:

    * labels are consecutive integers (via :func:`consolidate_labels`);
    * the number of distinct clusters is between 2 and ``g_max``;
    * when ``all_in_clusters`` is true there are no ``0`` (unassigned) labels.

    Clusters are merged at random if there are more than ``g_max``; a singleton
    second cluster is created at random if a partition collapses to one cluster.
    """
    labels = np.asarray(labels, dtype=int).copy()
    n = labels.size

    if all_in_clusters:
        # Reassign any unassigned elements to a random existing/available cluster.
        zero_mask = labels == 0
        if zero_mask.any():
            labels[zero_mask] = rng.integers(1, max(2, g_max) + 1, size=int(zero_mask.sum()))

    labels = np.asarray(consolidate_labels(labels), dtype=int)

    # Cap the number of clusters at g_max by merging extras into existing ones.
    assigned = labels[labels != 0]
    uniques = np.unique(assigned)
    if uniques.size > g_max:
        keep = set(rng.choice(uniques, size=g_max, replace=False).tolist())
        for c in uniques:
            if c not in keep:
                target = rng.choice(list(keep))
                labels[labels == c] = target
        labels = np.asarray(consolidate_labels(labels), dtype=int)

    # Guarantee at least two clusters so validity metrics stay defined.
    assigned = labels[labels != 0]
    if np.unique(assigned).size < 2 and n >= 2:
        low = 1 if all_in_clusters else 0
        # Flip a random non-empty subset of elements to a fresh cluster id.
        new_id = (labels.max() + 1) if labels.size else 1
        idx = rng.choice(n, size=max(1, n // 2), replace=False)
        labels[idx] = new_id
        labels = np.asarray(consolidate_labels(labels), dtype=int)

    return labels
