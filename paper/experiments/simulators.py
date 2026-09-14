"""Data-generating processes that GCM-MRK's likelihood does *not* assume.

The benchmark in the manuscript uses :func:`gcmrk.simulate_modular_data`, which
draws each module from a single latent factor with random :math:`\\pm 1` loadings.
That is exactly the model whose profile log-likelihood the ``loglik`` objective
maximizes.  Recovering those modules therefore demonstrates that the optimizer
reaches the likelihood optimum; it cannot, on its own, demonstrate that the model
is a good description of real co-expression data.  Testing that requires data the
model does not assume.

Each generator here violates at least one assumption of the one-factor block
model, while keeping the planted module structure that defines ground truth:

``multifactor``
    Modules driven by several latent factors with *continuous* Gaussian
    loadings.  Breaks the single-factor, equal-magnitude assumption; within-module
    correlation becomes heterogeneous across gene pairs.
``hub``
    Each module has a hub gene; members correlate with the hub at geometrically
    decaying strength.  Breaks the exchangeable-block assumption --- module
    members are not mutually equivalent.
``overlapping``
    A fraction of genes load on two modules.  Breaks the hard-partition
    assumption; ground truth records the dominant module.
``counts``
    Negative-binomial counts driven by a latent factor, then
    :math:`\\log_2(x+1)`.  Breaks Gaussianity and adds a mean-variance trend, as
    real RNA-seq does.
``network``
    A Gaussian graphical model whose precision matrix is the adjacency of an LFR
    benchmark graph: planted communities, scale-free degrees, and correlation
    that arises from *conditional* dependence rather than a shared factor.  This
    is the structure WGCNA's scale-free topology criterion assumes and the one
    furthest from GCM-MRK's model.

Every generator returns ``(X, labels)`` with ``X`` of shape
``(n_genes, n_samples)``, row-standardized, and 1-based integer ``labels``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "simulate_multifactor",
    "simulate_hub",
    "simulate_overlapping",
    "simulate_counts",
    "simulate_network",
    "GENERATORS",
]


def _standardize(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def _labels_from_sizes(sizes) -> np.ndarray:
    return np.concatenate([np.full(n, m, dtype=int)
                           for m, n in enumerate(sizes, start=1)])


# --------------------------------------------------------------------------- #
def simulate_multifactor(sizes, n_samples=50, noise=1.0, n_factors=3, seed=None):
    """Modules driven by several latent factors with continuous loadings.

    GCM-MRK's model says a module is one factor with :math:`\\pm 1` loadings.
    Here each module is a ``n_factors``-dimensional subspace and loadings are
    ``N(0, 1)``, so within-module correlations vary continuously in magnitude and
    some same-module pairs are only weakly related.
    """
    rng = np.random.default_rng(seed)
    labels = _labels_from_sizes(sizes)
    X = np.empty((len(labels), n_samples))
    row = 0
    for n_genes in sizes:
        latents = rng.normal(0.0, 1.0, size=(n_factors, n_samples))
        for _ in range(n_genes):
            coeff = rng.normal(0.0, 1.0, size=n_factors)
            X[row] = coeff @ latents + rng.normal(0.0, noise, size=n_samples)
            row += 1
    return _standardize(X), labels


def simulate_hub(sizes, n_samples=50, noise=1.0, floor=0.45, seed=None):
    """Hub-and-spoke modules with decaying membership strength.

    Within a module of :math:`n` genes, gene :math:`j` carries the hub signal at
    weight ``floor ** (j / (n - 1))``, i.e. a geometric decay from 1 down to
    ``floor``.  Peripheral genes are genuinely weaker members rather than pure
    noise, so the module is recoverable but its members are not exchangeable ---
    which is precisely what the block model assumes they are.
    """
    rng = np.random.default_rng(seed)
    labels = _labels_from_sizes(sizes)
    X = np.empty((len(labels), n_samples))
    row = 0
    for n_genes in sizes:
        hub = rng.normal(0.0, 1.0, size=n_samples)
        for j in range(n_genes):
            weight = floor ** (j / max(n_genes - 1, 1))
            sign = rng.choice([-1.0, 1.0])
            X[row] = sign * weight * hub + rng.normal(0.0, noise, size=n_samples)
            row += 1
    return _standardize(X), labels


def simulate_overlapping(sizes, n_samples=50, noise=1.0, overlap=0.2,
                         secondary=0.6, seed=None):
    """A fraction of genes co-load on a second module.

    ``overlap`` is the fraction of genes carrying a secondary loading of relative
    magnitude ``secondary``.  Ground truth is the dominant module, so a perfect
    hard partition is no longer attainable --- which is the situation real
    regulatory programs present.
    """
    rng = np.random.default_rng(seed)
    labels = _labels_from_sizes(sizes)
    n_modules = len(sizes)
    latents = rng.normal(0.0, 1.0, size=(n_modules, n_samples))
    X = np.empty((len(labels), n_samples))
    for i, primary in enumerate(labels):
        sign = rng.choice([-1.0, 1.0])
        signal = sign * latents[primary - 1]
        if rng.random() < overlap and n_modules > 1:
            others = [m for m in range(n_modules) if m != primary - 1]
            other = rng.choice(others)
            signal = signal + secondary * rng.choice([-1.0, 1.0]) * latents[other]
        X[i] = signal + rng.normal(0.0, noise, size=n_samples)
    return _standardize(X), labels


def simulate_counts(sizes, n_samples=50, noise=1.0, dispersion=0.3,
                    base_mean=200.0, seed=None):
    """Negative-binomial counts driven by a latent factor, then log-transformed.

    The latent factor shifts each gene's expected count multiplicatively; counts
    are drawn NB with the given dispersion and returned as :math:`\\log_2(x+1)`.
    Gaussianity fails, and the mean-variance trend of RNA-seq is present.
    """
    rng = np.random.default_rng(seed)
    labels = _labels_from_sizes(sizes)
    n_modules = len(sizes)
    latents = rng.normal(0.0, 1.0, size=(n_modules, n_samples))
    counts = np.empty((len(labels), n_samples))
    for i, module in enumerate(labels):
        sign = rng.choice([-1.0, 1.0])
        gene_mean = base_mean * rng.lognormal(0.0, 0.5)
        # latent modulates log-expression; the 0.8 factor scales `noise` onto
        # the log-expression scale so that a given sigma yields a
        # within-minus-between correlation contrast comparable to the other
        # generators; without it this generator stays saturated.
        log_mu = np.log(gene_mean) + 0.8 * sign * latents[module - 1] \
            + rng.normal(0.0, 0.8 * noise, size=n_samples)
        mu = np.exp(np.clip(log_mu, -20, 20))
        # NB via gamma-Poisson mixture
        shape = 1.0 / dispersion
        lam = rng.gamma(shape=shape, scale=mu / shape)
        counts[i] = rng.poisson(lam)
    X = np.log2(counts + 1.0)
    return _standardize(X), labels


def simulate_network(sizes, n_samples=50, noise=1.0, tau1=2.5, tau2=1.6,
                     mu=0.03, diffusion=0.95, avg_degree=10.0, seed=None):
    """Correlated Gaussian data from an LFR benchmark graph.

    Builds a graph with planted communities and a power-law degree distribution,
    then derives the covariance by diffusion over that graph,
    :math:`\\Sigma \\propto (I - c\\,D^{-1/2} A D^{-1/2})^{-1}`, which couples
    genes according to how well-connected they are *through* the network rather
    than through a shared latent factor.  Per-gene sign flips (a diagonal
    :math:`\\pm 1` similarity, which preserves positive definiteness) put
    anti-correlated pairs inside modules, as real co-expression data show.

    Neither the one-factor structure nor the equal-loading assumption holds here,
    and module members have heterogeneous connectivity by construction.  This is
    the generator furthest from GCM-MRK's model, and the one closest to the
    scale-free topology WGCNA assumes.

    ``noise`` adds independent measurement error; ``diffusion`` (``c``, in
    :math:`[0, 1)`) sets how far correlation spreads along the graph.
    """
    import networkx as nx

    rng = np.random.default_rng(seed)
    n_genes = int(sum(sizes))

    # mu (the fraction of a node's edges leaving its community) is the knob that
    # decides whether module structure survives diffusion: above ~0.1 the graph
    # is too well mixed for any method to recover communities from marginal
    # correlation, so we keep it assortative and vary difficulty via `noise`.
    graph = None
    for _ in range(80):
        try:
            graph = nx.LFR_benchmark_graph(
                n=n_genes, tau1=tau1, tau2=tau2, mu=mu,
                average_degree=avg_degree,
                min_community=max(10, min(sizes)),
                max_community=max(sizes) + 10,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            break
        except (nx.ExceededMaxIterations, nx.NetworkXError):
            continue
    if graph is None:
        raise RuntimeError("LFR graph generation failed; relax tau/mu/degree")

    communities = {}
    for node in graph.nodes():
        key = frozenset(graph.nodes[node]["community"])
        communities.setdefault(key, len(communities) + 1)
    labels = np.array([communities[frozenset(graph.nodes[n]["community"])]
                       for n in range(n_genes)], dtype=int)

    A = nx.to_numpy_array(graph, nodelist=range(n_genes))
    deg = np.maximum(A.sum(axis=1), 1.0)
    A_norm = A / np.sqrt(np.outer(deg, deg))          # spectral radius <= 1

    Sigma = np.linalg.inv(np.eye(n_genes) - diffusion * A_norm)
    Sigma = (Sigma + Sigma.T) / 2.0

    signs = rng.choice([-1.0, 1.0], size=n_genes)     # per-gene flip keeps PD
    Sigma = Sigma * np.outer(signs, signs)

    d = np.sqrt(np.diag(Sigma))
    Sigma = Sigma / np.outer(d, d)

    eigmin = np.linalg.eigvalsh(Sigma).min()
    jitter = max(1e-8, 1e-6 - min(eigmin, 0.0))
    L = np.linalg.cholesky(Sigma + jitter * np.eye(n_genes))
    X = L @ rng.normal(0.0, 1.0, size=(n_genes, n_samples))
    X = X + rng.normal(0.0, noise * 0.5, size=X.shape)
    return _standardize(X), labels


#: name -> generator, for the misspecification sweep
GENERATORS = {
    "onefactor": None,          # filled by the caller with gcmrk's own simulator
    "multifactor": simulate_multifactor,
    "hub": simulate_hub,
    "overlapping": simulate_overlapping,
    "counts": simulate_counts,
    "network": simulate_network,
}
