"""High-level entry point tying data, metrics and the GA together."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .data import load_matrix, normalize_data, pearson_correlation
from .ga import GAConfig, GAResult, evolve
from .metrics import get_metric
from .partition import read_partitions

__all__ = ["cluster"]


def cluster(
    data,
    targets: Sequence[str] = ("loglik",),
    g_max: int = 10,
    *,
    mode: str = "weighted",
    all_in_clusters: bool = True,
    normalize: bool = True,
    by_sample: bool = True,
    sep: str = ",",
    seed_file: Optional[str] = None,
    unassigned_penalty: float = 0.0,
    population_size: int = 200,
    generations: int = 100,
    tournament_size: int = 3,
    crossover_prob: float = 0.8,
    mutation_prob: float = 0.2,
    mutation_intensity: float = 0.05,
    elitism: int = 5,
    kmeans_restarts: int = 2,
    local_search: bool = True,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> GAResult:
    """Cluster ``data`` by optimising one or more performance-target metrics.

    Parameters
    ----------
    data:
        File path or array-like; rows are elements to cluster, columns features.
    targets:
        Metric names to optimise (see :data:`gcmrk.metrics.METRICS`).  More than
        one target combined with ``mode="nsga2"`` performs multi-objective
        optimisation.
    g_max:
        Maximum number of clusters allowed.
    mode:
        ``"weighted"`` (single combined fitness) or ``"nsga2"`` (Pareto).
    all_in_clusters:
        If true, every element must be assigned (no label ``0``).
    normalize / by_sample:
        Standardise the data first; ``by_sample`` normalises each row, which the
        correlation log-likelihood expects.
    seed_file:
        Optional file of seed partitions (one per line) to inject.
    unassigned_penalty:
        Penalty strength for leaving elements unassigned.
    Remaining keyword arguments are GA hyper-parameters.

    Returns
    -------
    GAResult
        Best partition, its metric scores, the run history, and (for nsga2) the
        Pareto front.
    """
    X = load_matrix(data, sep=sep)
    if normalize:
        X = normalize_data(X, by_sample=by_sample)

    specs = [get_metric(t) for t in targets]
    needs_cor = any(s.needs == "cor" for s in specs)
    cor = pearson_correlation(X, rowvar=True) if needs_cor else None

    seeds: Optional[List[np.ndarray]] = None
    if seed_file:
        seeds = read_partitions(seed_file, sep=sep, expected_len=X.shape[0])

    config = GAConfig(
        g_max=g_max,
        all_in_clusters=all_in_clusters,
        population_size=population_size,
        generations=generations,
        tournament_size=tournament_size,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        mutation_intensity=mutation_intensity,
        elitism=elitism,
        kmeans_restarts=kmeans_restarts,
        local_search=local_search,
        mode=mode,
        seed=seed,
        verbose=verbose,
    )

    return evolve(X, cor, specs, config, seeds=seeds,
                  unassigned_penalty=unassigned_penalty)
