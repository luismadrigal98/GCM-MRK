# GCM-MRK — metric-guided clustering by genetic algorithm

GCM-MRK clusters your data by **directly optimising the cluster-validity metric
you care about**. Instead of running a fixed algorithm (k-means, hierarchical,
…) and then *measuring* quality afterwards, GCM-MRK evolves a partition so that
the chosen *performance target* is as good as it can be.

It is built for settings — such as gene co-expression analysis — where the
notion of a "good" cluster is defined by the evaluation criterion itself, and
where that criterion (e.g. a correlation-based likelihood) is not what
off-the-shelf clusterers optimise.

The implementation depends only on **numpy, scipy and pandas**. There is no
dependency on DEAP or scikit-learn.

## Installation

```bash
pip install -e .          # from the repository root
```

or simply run it in place with `python -m gcmrk ...` (the package directory must
be on `PYTHONPATH`, e.g. run from the repo root).

## Concepts

| Term | Meaning |
|------|---------|
| **Element** | One row of the data matrix — the thing being clustered. |
| **Partition** | A vector of integer labels assigning each element to a cluster. Label `0` (optional) marks an *unassigned* element. |
| **Performance target / metric** | A scalar score measuring partition quality. The GA optimises it. |
| **Mode** | `weighted` (one combined objective) or `nsga2` (multi-objective Pareto search). |

### Available metrics

| Name | Direction | Consumes | Notes |
|------|-----------|----------|-------|
| `loglik` | maximise | correlation matrix | Correlation-based log-likelihood; rewards within-cluster correlation. The project's bespoke objective, suited to co-expression / modular data. |
| `silhouette` | maximise | data matrix | Mean silhouette in `[-1, 1]`; geometric (Euclidean) separation. |
| `davies_bouldin` | minimise | data matrix | Compactness vs separation ratio. |
| `calinski_harabasz` | maximise | data matrix | Variance-ratio criterion. |
| `bic` | minimise | data matrix | Gaussian-mixture BIC; rewards parsimony. |
| `aic` | minimise | data matrix | Gaussian-mixture AIC; rewards parsimony. |

Run `gcmrk metrics` for this list at any time.

## Command-line usage

```bash
# 1. Generate a synthetic modular dataset to experiment with
gcmrk simulate --modules 20 20 20 --samples 50 --seed 5 \
    --out sim.csv --truth-out truth.txt

# 2. Cluster it by maximising the correlation-based log-likelihood.
#    --g-max sets the maximum number of clusters.
gcmrk cluster --data sim.csv --targets loglik --g-max 3 \
    --generations 100 --population 200 --seed 0 --out labels.txt

# 3. Multi-objective: trade off fit (loglik) against compactness (Davies-Bouldin)
gcmrk cluster --data sim.csv --targets loglik davies_bouldin --mode nsga2 \
    --g-max 8 --out labels.txt --pareto-out pareto.txt
```

Cluster labels are written to `--out` (one per line). Progress and a metric
summary are printed to stderr. For `nsga2`, `--pareto-out` saves every
non-dominated solution (one partition per line).

### Useful options

- `--allow-unassigned` — permit elements to be left out (label `0`), with
  `--unassigned-penalty` controlling how strongly that is discouraged.
- `--seed-file FILE` — inject seed partitions (one per line), e.g. the output of
  another clustering tool, into the initial population.
- `--by-feature` — normalise per column instead of per row.
- `--no-normalize` — skip standardisation entirely.
- `--kmeans-restarts N` — number of k-means seed partitions per candidate
  cluster count (`0` disables k-means seeding).

## Python API

```python
from gcmrk import cluster, simulate_modular_data

data, truth = simulate_modular_data([20, 20, 20], n_samples=50, seed=5)

result = cluster(
    data,
    targets=["loglik"],
    g_max=3,
    generations=100,
    population_size=200,
    seed=0,
)

print(result.labels)   # best partition (numpy array of labels)
print(result.scores)   # {'loglik': ...}
print(result.history)  # per-generation statistics

# Multi-objective
mo = cluster(data, targets=["loglik", "bic"], mode="nsga2", g_max=8)
for sol in mo.pareto_front:
    print(sol["scores"], "->", sol["labels"])
```

## How it works

1. **Initialisation** — the population is seeded with k-means partitions across
   a range of cluster counts (`2 … g_max`), any user-supplied seed partitions,
   and random partitions for diversity.
2. **Evaluation** — each partition is scored on the chosen metric(s).
3. **Selection / variation** — elitist tournament selection (weighted mode) or
   NSGA-II non-dominated sorting + crowding distance (multi-objective mode),
   followed by two-point crossover and reassignment mutation.
4. **Repair** — after every operator, partitions are relabelled to consecutive
   integers and constrained to have between 2 and `g_max` clusters so the
   validity metrics stay well-defined.

## Choosing the number of clusters (important)

The correlation-based `loglik` is a *goodness-of-fit* score with **no built-in
parsimony**: splitting a good cluster into smaller well-correlated pieces tends
to increase it. Optimising `loglik` alone with a loose `--g-max` therefore
over-segments. Two ways to get the right number of clusters:

- **Constrain `--g-max`** to the number of clusters you expect (or scan a few
  values and compare). On the synthetic modular data above this recovers the
  ground-truth modules with ~0.85 accuracy.
- **Add a parsimony objective** — combine `loglik` with `bic` or `aic` under
  `--mode nsga2` and inspect the Pareto front for the knee. Note that BIC/AIC
  here assume Euclidean-Gaussian clusters, so they are most informative when the
  clusters are also geometrically compact.

## Repository layout

```
gcmrk/
  metrics.py     cluster-validity metrics (the optimisation targets)
  partition.py   partition representation, repair, label consolidation
  seeding.py     numpy k-means used to seed the population
  ga.py          genetic algorithm engine (weighted + NSGA-II)
  data.py        loading, normalisation, correlation
  simulate.py    synthetic modular-data generator
  core.py        cluster() — the high-level entry point
  cli.py         command-line interface
tests/           unittest suite (run: python -m unittest discover -s tests)
legacy/          the original exploratory scripts, kept for reference
```

## Running the tests

```bash
python -m unittest discover -s tests -v
# or, if pytest is available:
pytest
```
