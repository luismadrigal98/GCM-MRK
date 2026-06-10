"""Synthetic data generation for testing and demonstration.

Generates modular ("block-correlated") data: groups of correlated rows that a
good clustering should recover.  Adapted from the project's original
latent-variable simulator, rewritten around a single numpy ``Generator`` for
reproducibility.
"""

from __future__ import annotations

import numpy as np

__all__ = ["simulate_modular_data"]


def simulate_modular_data(
    genes_per_module: list,
    n_samples: int = 200,
    latent_per_module: int = 4,
    noise: float = 1.0,
    seed: int | None = None,
):
    """Simulate modular expression-like data.

    Each module is driven by a set of shared latent variables, so rows within a
    module are mutually correlated while rows in different modules are not.

    Parameters
    ----------
    genes_per_module:
        Number of rows (genes/elements) in each module; ``len`` is the number of
        true clusters.
    n_samples:
        Number of columns (samples/conditions).
    latent_per_module:
        Latent factors shared within each module.
    noise:
        Std of the per-entry gaussian noise.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    data : ndarray (sum(genes_per_module), n_samples)
        Row-standardised data matrix.
    labels : ndarray
        Ground-truth module id (1-based) for each row.
    """
    rng = np.random.default_rng(seed)
    total = int(sum(genes_per_module))
    data = np.zeros((total, n_samples))

    row = 0
    labels = np.empty(total, dtype=int)
    for module, n_genes in enumerate(genes_per_module, start=1):
        latents = rng.normal(0.0, 1.0, size=(latent_per_module, n_samples))
        for _ in range(n_genes):
            coeff = rng.choice([-1.0, 1.0], size=latent_per_module)
            data[row] = coeff @ latents + rng.normal(0.0, noise, size=n_samples)
            labels[row] = module
            row += 1

    # Row-standardise (mean 0, unit variance per row).
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    data = (data - mean) / std
    return data, labels
