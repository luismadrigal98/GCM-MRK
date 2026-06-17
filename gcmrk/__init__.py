"""GCM-MRK: genetic-algorithm clustering guided by performance-target metrics.

Evolve a partition of your data so that it directly optimizes the clustering
quality metric(s) you care about -- correlation-based log-likelihood,
silhouette, Davies-Bouldin, Calinski-Harabasz, BIC or AIC -- either as a single
weighted objective or as a multi-objective Pareto search.

Quick start
-----------
>>> from gcmrk import cluster, simulate_modular_data
>>> data, truth = simulate_modular_data([20, 20, 20], n_samples=50, seed=0)
>>> result = cluster(data, targets=["silhouette"], g_max=5,
...                   generations=20, population_size=60, seed=0)
>>> result.labels            # best partition found  # doctest: +SKIP
"""

from .core import cluster
from .ga import GAConfig, GAResult, evolve
from .metrics import METRICS, MetricSpec, get_metric
from .partition import consolidate_labels, random_partition, read_partitions, repair
from .data import load_matrix, normalize_data, pearson_correlation
from .simulate import simulate_modular_data
from .visualize import (
    available_methods,
    plot_clusters,
    plot_embedding_grid,
    reduce_dimensions,
)

__version__ = "1.0.0"

__all__ = [
    "cluster",
    "evolve",
    "GAConfig",
    "GAResult",
    "METRICS",
    "MetricSpec",
    "get_metric",
    "consolidate_labels",
    "random_partition",
    "read_partitions",
    "repair",
    "load_matrix",
    "normalize_data",
    "pearson_correlation",
    "simulate_modular_data",
    "available_methods",
    "plot_clusters",
    "plot_embedding_grid",
    "reduce_dimensions",
    "__version__",
]
