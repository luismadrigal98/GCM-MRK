"""Self-contained genetic algorithm for metric-guided clustering.

The algorithm evolves a population of partitions (label vectors).  Each
partition is scored by one or more validity metrics (the "performance
targets").  Two optimisation modes are supported:

``weighted``
    Combine the chosen metrics into a single fitness via direction-aware
    weights and run elitist tournament selection.

``nsga2``
    Treat the metrics as competing objectives and run NSGA-II (fast
    non-dominated sorting + crowding distance), returning the Pareto front.

The implementation depends only on numpy + scipy so it runs anywhere those are
available -- no DEAP, no scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .metrics import MetricSpec
from .partition import random_partition, repair
from .seeding import kmeans_seeds

__all__ = ["GAConfig", "GAResult", "evolve"]


@dataclass
class GAConfig:
    """Genetic-algorithm hyper-parameters."""

    g_max: int = 10
    all_in_clusters: bool = True
    population_size: int = 200
    generations: int = 100
    tournament_size: int = 3
    crossover_prob: float = 0.8
    mutation_prob: float = 0.2
    mutation_intensity: float = 0.05  # per-gene flip probability when mutating
    elitism: int = 5
    mode: str = "weighted"  # "weighted" or "nsga2"
    kmeans_restarts: int = 2  # k-means seed partitions per k (0 disables seeding)
    seed: Optional[int] = None
    verbose: bool = False


@dataclass
class GAResult:
    """Outcome of an evolutionary run."""

    labels: np.ndarray                       # best partition (consolidated)
    scores: dict                             # metric name -> value for `labels`
    fitness: float                           # combined fitness of `labels`
    history: List[dict] = field(default_factory=list)  # per-generation stats
    pareto_front: Optional[List[dict]] = None          # nsga2 only


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
class _Evaluator:
    """Scores partitions against the selected metrics, with a small cache."""

    def __init__(self, metrics: Sequence[MetricSpec], X, cor,
                 unassigned_penalty: float):
        self.metrics = list(metrics)
        self.X = X
        self.cor = cor
        self.penalty = unassigned_penalty
        self._cache: dict = {}

    def raw_scores(self, labels: np.ndarray) -> np.ndarray:
        key = labels.tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        out = np.empty(len(self.metrics), dtype=float)
        for i, spec in enumerate(self.metrics):
            payload = self.cor if spec.needs == "cor" else self.X
            if spec.needs == "cor":
                out[i] = spec.func(payload, labels)
            else:
                out[i] = spec.func(payload, labels, unassigned_penalty=self.penalty)
        self._cache[key] = out
        return out

    def weighted_fitness(self, labels: np.ndarray) -> float:
        raw = self.raw_scores(labels)
        weights = np.array([m.weight for m in self.metrics])
        norm = _normalise_for_sum(raw, self.metrics)
        return float(np.dot(weights, norm))

    def named_scores(self, labels: np.ndarray) -> dict:
        return {m.name: float(v) for m, v in zip(self.metrics, self.raw_scores(labels))}


def _normalise_for_sum(raw: np.ndarray, metrics: Sequence[MetricSpec]) -> np.ndarray:
    """Squash each metric to a comparable scale before weighting.

    Metrics live on wildly different scales (silhouette in [-1, 1], BIC in the
    thousands).  For the single-objective weighted sum we map each through a
    bounded transform so no single metric dominates purely by magnitude.
    """
    out = np.empty_like(raw)
    for i, (val, m) in enumerate(zip(raw, metrics)):
        if not np.isfinite(val):
            out[i] = -1e6 * m.weight  # heavily penalise degenerate partitions
        elif m.name == "silhouette":
            out[i] = val  # already in [-1, 1]
        else:
            # Smooth, monotonic, bounded squashing that preserves ordering.
            out[i] = np.tanh(val / (abs(val) + 1.0)) if val != 0 else 0.0
    return out


# --------------------------------------------------------------------------- #
# Genetic operators
# --------------------------------------------------------------------------- #
def _crossover(p1: np.ndarray, p2: np.ndarray, rng) -> tuple:
    """Two-point crossover on label vectors."""
    n = p1.size
    if n < 2:
        return p1.copy(), p2.copy()
    a, b = sorted(rng.integers(0, n, size=2).tolist())
    if a == b:
        b = min(n, a + 1)
    c1, c2 = p1.copy(), p2.copy()
    c1[a:b], c2[a:b] = p2[a:b], p1[a:b]
    return c1, c2


def _mutate(ind: np.ndarray, g_max: int, all_in_clusters: bool,
            intensity: float, rng) -> np.ndarray:
    """Randomly reassign a fraction of elements to other clusters."""
    out = ind.copy()
    low = 1 if all_in_clusters else 0
    mask = rng.random(out.size) < intensity
    n_flip = int(np.sum(mask))
    if n_flip:
        out[mask] = rng.integers(low, g_max + 1, size=n_flip)
    return out


# --------------------------------------------------------------------------- #
# NSGA-II support
# --------------------------------------------------------------------------- #
def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if objective vector ``a`` Pareto-dominates ``b`` (maximisation)."""
    return np.all(a >= b) and np.any(a > b)


def _fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """Partition indices into Pareto fronts (objectives are maximised)."""
    n = objectives.shape[0]
    S = [[] for _ in range(n)]
    dom_count = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objectives[p], objectives[q]):
                S[p].append(q)
            elif _dominates(objectives[q], objectives[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def _crowding_distance(objectives: np.ndarray, front: List[int]) -> np.ndarray:
    """Crowding distance for the members of a single front."""
    m = objectives.shape[1]
    dist = np.zeros(len(front))
    pts = objectives[front]
    for k in range(m):
        order = np.argsort(pts[:, k])
        dist[order[0]] = dist[order[-1]] = np.inf
        vmin, vmax = pts[order[0], k], pts[order[-1], k]
        span = vmax - vmin
        if span == 0:
            continue
        for idx in range(1, len(front) - 1):
            dist[order[idx]] += (
                pts[order[idx + 1], k] - pts[order[idx - 1], k]
            ) / span
    return dist


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def evolve(X, cor, metrics: Sequence[MetricSpec], config: GAConfig,
           seeds: Optional[Sequence[np.ndarray]] = None,
           unassigned_penalty: float = 0.0) -> GAResult:
    """Run the genetic algorithm and return the best partition found.

    Parameters
    ----------
    X:
        Data matrix (n_samples, n_features); ``None`` is allowed only if every
        chosen metric consumes the correlation matrix instead.
    cor:
        Sample-by-sample correlation matrix (n_samples, n_samples) or ``None``.
    metrics:
        The optimisation targets.
    config:
        Hyper-parameters.
    seeds:
        Optional list of seed partitions injected into the initial population.
    unassigned_penalty:
        Strength of the penalty for leaving elements unassigned.
    """
    if not metrics:
        raise ValueError("At least one metric must be provided")
    n = X.shape[0] if X is not None else cor.shape[0]
    rng = np.random.default_rng(config.seed)
    evaluator = _Evaluator(metrics, X, cor, unassigned_penalty)

    # --- initial population ------------------------------------------------ #
    population: List[np.ndarray] = []
    if seeds:
        for s in seeds:
            if s.size == n:
                population.append(repair(s, config.g_max, config.all_in_clusters, rng))
    # Bootstrap with k-means partitions across candidate cluster counts.
    if config.kmeans_restarts > 0 and X is not None:
        for s in kmeans_seeds(X, config.g_max, rng, restarts=config.kmeans_restarts):
            population.append(repair(s, config.g_max, config.all_in_clusters, rng))
    while len(population) < config.population_size:
        population.append(
            random_partition(n, config.g_max, config.all_in_clusters, rng)
        )
    population = population[: config.population_size]

    if config.mode == "nsga2":
        return _run_nsga2(population, evaluator, metrics, config, rng)
    return _run_weighted(population, evaluator, config, rng)


def _run_weighted(population, evaluator, config: GAConfig, rng) -> GAResult:
    history = []
    fits = np.array([evaluator.weighted_fitness(ind) for ind in population])

    for gen in range(config.generations):
        # Elitism: carry over the best individuals untouched.
        elite_idx = np.argsort(fits)[::-1][: config.elitism]
        elites = [population[i].copy() for i in elite_idx]

        # Produce offspring to fill the rest of the next generation.
        offspring = []
        while len(offspring) < config.population_size - len(elites):
            p1 = _tournament(population, fits, config.tournament_size, rng)
            p2 = _tournament(population, fits, config.tournament_size, rng)
            if rng.random() < config.crossover_prob:
                c1, c2 = _crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            for child in (c1, c2):
                if rng.random() < config.mutation_prob:
                    child = _mutate(child, config.g_max, config.all_in_clusters,
                                    config.mutation_intensity, rng)
                child = repair(child, config.g_max, config.all_in_clusters, rng)
                offspring.append(child)
                if len(offspring) >= config.population_size - len(elites):
                    break

        population = elites + offspring
        fits = np.array([evaluator.weighted_fitness(ind) for ind in population])

        best = float(np.max(fits))
        record = {"gen": gen, "max": best, "avg": float(np.mean(fits)),
                  "min": float(np.min(fits))}
        history.append(record)
        if config.verbose:
            print(f"gen {gen:4d} | best {best:.4f} | avg {record['avg']:.4f}")

    best_idx = int(np.argmax(fits))
    best_labels = population[best_idx]
    return GAResult(
        labels=best_labels,
        scores=evaluator.named_scores(best_labels),
        fitness=float(fits[best_idx]),
        history=history,
    )


def _tournament(population, fits, k, rng) -> np.ndarray:
    contenders = rng.integers(0, len(population), size=k)
    best = contenders[int(np.argmax(fits[contenders]))]
    return population[best].copy()


def _run_nsga2(population, evaluator, metrics, config: GAConfig, rng) -> GAResult:
    weights = np.array([m.weight for m in metrics])

    def objectives_of(pop):
        # Convert every metric to a maximisation objective via its weight.
        return np.array([evaluator.raw_scores(ind) * weights for ind in pop])

    history = []
    obj = objectives_of(population)

    for gen in range(config.generations):
        offspring = _make_offspring(population, obj, config, rng)
        combined = population + offspring
        obj_combined = objectives_of(combined)

        fronts = _fast_non_dominated_sort(obj_combined)
        new_pop, new_idx = [], []
        for front in fronts:
            if len(new_pop) + len(front) <= config.population_size:
                new_pop.extend(combined[i] for i in front)
                new_idx.extend(front)
            else:
                cd = _crowding_distance(obj_combined, front)
                order = np.argsort(cd)[::-1]
                remaining = config.population_size - len(new_pop)
                for j in order[:remaining]:
                    new_pop.append(combined[front[j]])
                    new_idx.append(front[j])
                break

        population = new_pop
        obj = obj_combined[new_idx]

        front0 = _fast_non_dominated_sort(obj)[0]
        record = {"gen": gen, "front_size": len(front0),
                  "best_per_obj": obj[front0].max(axis=0).tolist()}
        history.append(record)
        if config.verbose:
            print(f"gen {gen:4d} | front {len(front0)} | "
                  f"best {record['best_per_obj']}")

    # Build the Pareto front of the final population.
    front0 = _fast_non_dominated_sort(obj)[0]
    pareto = []
    for i in front0:
        labels = population[i]
        pareto.append({"labels": labels, "scores": evaluator.named_scores(labels)})

    # Pick a single representative: the knee point maximising the sum of
    # min-max normalised objectives across the front.
    front_obj = obj[front0]
    span = front_obj.max(axis=0) - front_obj.min(axis=0)
    span[span == 0] = 1.0
    norm = (front_obj - front_obj.min(axis=0)) / span
    rep_local = int(np.argmax(norm.sum(axis=1)))
    rep_labels = population[front0[rep_local]]

    return GAResult(
        labels=rep_labels,
        scores=evaluator.named_scores(rep_labels),
        fitness=float(norm.sum(axis=1)[rep_local]),
        history=history,
        pareto_front=pareto,
    )


def _make_offspring(population, obj, config: GAConfig, rng) -> list:
    """Generate an offspring population for NSGA-II via binary tournaments on
    non-domination rank."""
    fronts = _fast_non_dominated_sort(obj)
    rank = np.empty(len(population), dtype=int)
    for r, front in enumerate(fronts):
        for i in front:
            rank[i] = r

    def tourn():
        a, b = rng.integers(0, len(population), size=2)
        return population[a] if rank[a] <= rank[b] else population[b]

    offspring = []
    while len(offspring) < config.population_size:
        p1, p2 = tourn(), tourn()
        if rng.random() < config.crossover_prob:
            c1, c2 = _crossover(p1, p2, rng)
        else:
            c1, c2 = p1.copy(), p2.copy()
        for child in (c1, c2):
            if rng.random() < config.mutation_prob:
                child = _mutate(child, config.g_max, config.all_in_clusters,
                                config.mutation_intensity, rng)
            child = repair(child, config.g_max, config.all_in_clusters, rng)
            offspring.append(child)
            if len(offspring) >= config.population_size:
                break
    return offspring
