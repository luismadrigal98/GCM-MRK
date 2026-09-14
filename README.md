# GCM-MRK — metric-guided clustering by genetic algorithm

GCM-MRK clusters your data by **directly optimizing the cluster-validity metric
you care about**. Instead of running a fixed algorithm (k-means, hierarchical,
…) and then *measuring* quality afterwards, GCM-MRK evolves a partition so that
the chosen *performance target* is as good as it can be.

It is built for settings — such as gene co-expression analysis — where the
notion of a "good" cluster is defined by the evaluation criterion itself, and
where that criterion (e.g. a correlation-based likelihood) is not what
off-the-shelf clusterers optimize.

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
| **Performance target / metric** | A scalar score measuring partition quality. The GA optimizes it. |
| **Mode** | `weighted` (one combined objective) or `nsga2` (multi-objective Pareto search). |

### Available metrics

| Name | Direction | Consumes | Notes |
|------|-----------|----------|-------|
| `loglik` | maximize | correlation matrix | Correlation-based log-likelihood; rewards within-cluster correlation. The project's bespoke objective, suited to co-expression / modular data. |
| `loglik_bic` | minimize | correlation matrix | `loglik` penalized by `K·log(n)`; selects the number of modules within correlation space. |
| `loglik_aic` | minimize | correlation matrix | `loglik` penalized by `2K`; lighter penalty, tolerates a few more modules. |
| `loglik_factor` | maximize | correlation matrix | Rank-`q` **free-loading** block model. Unlike `loglik`, module members may carry unequal loadings. Set the rank with `loglik_factor@q`. |
| `loglik_factor_bic` | minimize | correlation matrix | BIC for the rank-`q` factor model; parameter count scales with module size. |
| `loglik_factor_aic` | minimize | correlation matrix | AIC for the rank-`q` factor model. |
| `silhouette` | maximize | data matrix | Mean silhouette in `[-1, 1]`; geometric (Euclidean) separation. |
| `davies_bouldin` | minimize | data matrix | Compactness vs separation ratio. |
| `calinski_harabasz` | maximize | data matrix | Variance-ratio criterion. |
| `bic` | minimize | data matrix | Gaussian-mixture BIC; rewards parsimony. |
| `aic` | minimize | data matrix | Gaussian-mixture AIC; rewards parsimony. |

Run `gcmrk metrics` for this list at any time.

### Choosing between `loglik` and `loglik_factor`

The two correlation objectives assume different things about what a module *is*.

`loglik` models a module as **one latent factor with equal-magnitude ± loadings**.
It sees a module only through its mean absolute correlation and treats all member
pairs as interchangeable. `loglik_factor` models a module as **`q` freely-loaded
factors**: members may carry different loading magnitudes, and anti-correlated
members are represented by negative loadings rather than an absolute value. The
first is the constrained special case of the second.

**Start with `loglik_aic`.** It is much faster, needs no rank selection, and on
real expression data we have found it to match or beat the factor model. Reach
for `loglik_factor` when there is specific reason to expect that module members
respond to a shared regulator at *different strengths* (hub-and-spoke structure),
or that modules are driven by *several* factors.

| situation | use |
|---|---|
| general use; cluster count unknown | `loglik_aic` (or `loglik_bic`) |
| unequal member strengths or several regulators, count known | `cluster_factor_auto(..., g_max=K)` |
| as above, count unknown | two-stage: `loglik_aic` picks K, then `cluster_factor_auto(g_max=K)` |
| large inputs | `loglik` / `loglik_aic` |

The rank matters, and setting it too high is worse than leaving it at 1, so let
the information criterion choose it — its parameter count scales with module
size, unlike the exchangeable model's one-per-module:

```python
from gcmrk import cluster, cluster_factor_auto

# cluster count unknown: pick it with the constrained criterion first
k_hat = len(set(cluster(data, targets=["loglik_aic"], g_max=15).labels))
result = cluster_factor_auto(data, ranks=(1, 2, 3), g_max=k_hat)
print(result.scores["factor_rank"])   # the q that was selected
```

or from the command line, keeping the rank with the lowest criterion:

```bash
for q in 1 2 3; do
    gcmrk cluster --data expr.csv --targets loglik_factor_bic@$q \
        --g-max 8 --out labels_q$q.txt
done
```

**Give `cluster_factor_auto` a cluster count you trust.** It selects the rank
reliably but cannot select the number of clusters: the summed parameter count is
`n*q - k*q(q-1)/2 + k`, which grows with `k` only at `q = 1`, is flat at `q = 2`
and shrinks for `q >= 3`. At higher ranks the criterion charges less for more
modules, so with a loose `g_max` the search runs to the ceiling. Use
`loglik_aic` / `loglik_bic` when the count is unknown.

`loglik_factor` costs several times more than `loglik` per run — its local search
re-solves the affected blocks' leading eigenvalues per candidate move, where the
exchangeable objective updates in constant time — and the gap widens on real data,
where more modules and a higher selected rank are typical. Above ~1500 elements
the hill-climb is skipped automatically and the GA optimizes without refinement.

## Command-line usage

```bash
# 1. Generate a synthetic modular dataset to experiment with
gcmrk simulate --modules 20 20 20 --samples 50 --seed 5 \
    --out sim.csv --truth-out truth.txt

# 2. Cluster it by maximizing the correlation-based log-likelihood.
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
- `--by-feature` — normalize per column instead of per row.
- `--no-normalize` — skip standardization entirely.
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

## Visualization

GCM-MRK includes built-in tools for visualizing clustering results in a
reduced-dimensional embedding.  Four methods are supported:

| Method | Dependency | Character |
|--------|-----------|-----------|
| `pca` | numpy (always available) | Linear; preserves global variance |
| `cor_mds` | scipy (always available) | Classical MDS on `1 − |R|` — the correlation distance native to the log-likelihood objective |
| `tsne` | scikit-learn (optional) | Non-linear; preserves local neighbourhoods |
| `umap` | umap-learn (optional) | Non-linear; balances local and global structure |

Install optional backends with `pip install gcmrk[viz]`.

### Command-line visualization

```bash
# Visualize an existing clustering result
gcmrk visualize --data sim.csv --labels labels.txt \
    --methods pca cor_mds --out clusters.pdf

# Or auto-generate a plot immediately after clustering
gcmrk cluster --data sim.csv --targets loglik --g-max 3 \
    --out labels.txt --plot --plot-out clusters.pdf
```

### Python API

```python
from gcmrk import cluster, simulate_modular_data
from gcmrk.visualize import reduce_dimensions, plot_clusters, plot_embedding_grid

data, truth = simulate_modular_data([20, 20, 20], n_samples=50, seed=5)
result = cluster(data, targets=["loglik"], g_max=3,
                 generations=100, population_size=200, seed=0)

# Single-method embedding
X_2d = reduce_dimensions(data, result.labels, method="cor_mds")
fig = plot_clusters(X_2d, result.labels, method="cor_mds",
                    title="Correlation MDS", truth=truth)
fig.savefig("clusters_mds.pdf")

# Multi-panel grid comparing all available methods
fig = plot_embedding_grid(data, result.labels, truth=truth)
fig.savefig("clusters_grid.pdf")
```

## How it works

1. **Initialization** — the population is seeded with k-means partitions and
   correlation-distance hierarchical partitions (average/complete linkage on
   `1 − |R|`, the classical co-expression strategy) across a range of cluster
   counts (`2 … g_max`), any user-supplied seed partitions, and random partitions
   for diversity. The correlation-aware seeds matter for `loglik`: Euclidean
   k-means splits coherent but anti-correlated genes.
2. **Evaluation** — each partition is scored on the chosen metric(s).
3. **Selection / variation** — elitist tournament selection (weighted mode) or
   NSGA-II non-dominated sorting + crowding distance (multi-objective mode),
   followed by two-point crossover and reassignment mutation.
4. **Memetic local search** — for the correlation objective, seeds and elites are
   refined by a greedy hill-climb that reassigns the single gene most improving
   the objective until none remains (computed incrementally, `O(n²)` per sweep).
   This is the step that lets the optimizer beat the hierarchical clustering it is
   seeded from — agglomerative clustering can never separate two genes once
   merged. Disable with `local_search=False` (Python API).
5. **Repair** — after every operator, partitions are relabelled to consecutive
   integers and constrained to have between 2 and `g_max` clusters so the
   validity metrics stay well-defined.

The `loglik` objective is, up to a constant and the sample-size factor, the
profile log-likelihood of a block-diagonal one-factor Gaussian model (each module
= one regulator with equal-magnitude ± loadings); `loglik_bic`/`loglik_aic` are
that model's BIC/AIC. See `paper/` for the derivation and benchmarks.

## Choosing the number of clusters (important)

The correlation-based `loglik` is a *goodness-of-fit* score with **no built-in
parsimony**: splitting a good cluster into smaller well-correlated pieces tends
to increase it. Optimizing `loglik` alone with a loose `--g-max` therefore
over-segments. Two ways to get the right number of clusters:

- **Optimize a penalized correlation criterion** (recommended) — use
  `--targets loglik_aic` (or `loglik_bic`) as a single objective with a loose
  `--g-max`. These add a cluster-count penalty to `loglik` *within correlation
  space*, so the search both fits and selects the number of modules. `loglik_aic`
  recovers the true module count on the synthetic data; `loglik_bic` is more
  conservative and favours fewer, stronger modules when separation is weak.
- **Constrain `--g-max`** to the number of clusters you expect (or scan a few
  values and pick the one minimizing `loglik_aic`/`loglik_bic`).
- **Add a parsimony objective** — combine `loglik` with `bic` or `aic` under
  `--mode nsga2` and inspect the Pareto front. Note that the *geometric* BIC/AIC
  assume Euclidean-Gaussian clusters, so they conflict with the
  absolute-correlation objective when modules contain anti-correlated genes; the
  penalized correlation criteria above avoid that assumption.

## Repository layout

```
gcmrk/
  metrics.py     cluster-validity metrics (the optimization targets)
  partition.py   partition representation, repair, label consolidation
  seeding.py     numpy k-means used to seed the population
  ga.py          genetic algorithm engine (weighted + NSGA-II)
  data.py        loading, normalization, correlation
  simulate.py    synthetic modular-data generator
  visualize.py   dimensionality-reduction visualization (PCA, MDS, t-SNE, UMAP)
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
