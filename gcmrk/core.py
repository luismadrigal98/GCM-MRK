"""High-level entry point tying data, metrics and the GA together."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .data import load_matrix, normalize_data, pearson_correlation
from .ga import GAConfig, GAResult, evolve
from .metrics import get_metric
from .partition import read_partitions

__all__ = ["cluster", "cluster_factor_auto"]


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
    """Cluster ``data`` by optimizing one or more performance-target metrics.

    Parameters
    ----------
    data:
        File path or array-like; rows are elements to cluster, columns features.
    targets:
        Metric names to optimize (see :data:`gcmrk.metrics.METRICS`).  More than
        one target combined with ``mode="nsga2"`` performs multi-objective
        optimization.
    g_max:
        Maximum number of clusters allowed.
    mode:
        ``"weighted"`` (single combined fitness) or ``"nsga2"`` (Pareto).
    all_in_clusters:
        If true, every element must be assigned (no label ``0``).
    normalize / by_sample:
        Standardize the data first; ``by_sample`` normalizes each row, which the
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


def cluster_factor_auto(data, ranks: Sequence[int] = (1, 2, 3),
                        criterion: str = "bic", **kwargs):
    """Cluster with the factor block model, choosing the rank ``q`` from the data.

    The rank-``q`` objective is powerful but sensitive: setting ``q`` above the
    true number of factors per module is worse than leaving it at 1.  Because the
    factor model's parameter count scales with module size, its information
    criteria are comparable *across* ranks, so ``q`` can simply be selected by
    running each candidate and keeping the best-scoring fit.

    This runs one full optimization per rank in ``ranks`` and returns the result
    minimizing ``loglik_factor_bic`` (or ``loglik_factor_aic``).  Prefer ``bic``:
    ``aic`` tends to select one rank too many.

    .. warning::
       **Set ``g_max`` to a known or externally justified cluster count.**  This
       function selects the *rank* well but must not be used to select the
       *number of clusters*.  Summing the per-module parameter count over ``k``
       modules gives ``n*q - k*q(q-1)/2 + k``, which increases with ``k`` only at
       ``q = 1``; it is flat at ``q = 2`` and decreases for ``q >= 3``.  At higher
       ranks the criterion therefore charges *less* for more modules and the
       search runs to whatever ``g_max`` allows.  Use ``loglik_aic`` /
       ``loglik_bic`` when the module count is unknown.

    Parameters
    ----------
    data:
        File path or array-like; rows are elements, columns features.
    ranks:
        Candidate factor ranks to compare.
    criterion:
        ``"bic"`` (default) or ``"aic"``.
    Remaining keyword arguments are passed through to :func:`cluster`; do not
    pass ``targets``, which this function sets itself.

    Returns
    -------
    GAResult
        The winning run, with two extra attributes recorded in ``scores``:
        ``factor_rank`` (the selected ``q``) and the criterion value.
    """
    if criterion not in ("bic", "aic"):
        raise ValueError("criterion must be 'bic' or 'aic'")
    if "targets" in kwargs:
        raise TypeError("cluster_factor_auto sets `targets` itself; "
                        "pass `ranks` to control the search instead")
    ranks = [int(q) for q in ranks]
    if not ranks or any(q < 1 for q in ranks):
        raise ValueError("ranks must be a non-empty sequence of integers >= 1")

    metric = f"loglik_factor_{criterion}"
    best, best_score, best_q = None, np.inf, None
    for q in ranks:
        result = cluster(data, targets=[f"{metric}@{q}"], **kwargs)
        score = float(result.scores[f"{metric}@{q}"])
        if score < best_score:
            best, best_score, best_q = result, score, q

    best.scores["factor_rank"] = best_q
    return best
