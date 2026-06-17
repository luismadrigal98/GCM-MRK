"""Dimensionality-reduction visualization for clustering results.

Project recovered (or ground-truth) cluster labels into a 2-D or 3-D embedding
so users can visually assess separation, overlap, and structure.

Four embedding methods are supported:

``pca``
    Principal-component analysis (NumPy SVD).  Linear, fast, preserves global
    variance.  Always available.

``cor_mds``
    Classical (metric) multidimensional scaling on the correlation distance
    matrix ``1 − |R|``, where ``R`` is the element-by-element Pearson
    correlation.  This is the natural embedding for the correlation objective:
    it uses the same distance space the log-likelihood operates in.  Always
    available (scipy only).

``tsne``
    t-distributed Stochastic Neighbor Embedding via scikit-learn.  Non-linear,
    preserves local neighbourhood structure.  Requires ``scikit-learn``.

``umap``
    Uniform Manifold Approximation and Projection.  Non-linear, balances local
    and global structure.  Requires ``umap-learn``.

Quick start
-----------
>>> from gcmrk import cluster, simulate_modular_data
>>> from gcmrk.visualize import reduce_dimensions, plot_clusters
>>> data, truth = simulate_modular_data([20, 20, 20], n_samples=50, seed=0)
>>> result = cluster(data, targets=["loglik"], g_max=3,
...                  generations=20, population_size=60, seed=0)
>>> X_2d = reduce_dimensions(data, result.labels, method="pca")
>>> fig = plot_clusters(X_2d, result.labels)       # doctest: +SKIP
>>> fig.savefig("clusters.pdf")                    # doctest: +SKIP
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .data import normalize_data, pearson_correlation

__all__ = [
    "METHODS",
    "available_methods",
    "reduce_dimensions",
    "plot_clusters",
    "plot_embedding_grid",
]

# Registry of supported methods and whether they need optional packages.
METHODS: Dict[str, dict] = {
    "pca": {
        "label": "PCA",
        "package": None,
        "description": "Principal Component Analysis (linear, always available).",
    },
    "cor_mds": {
        "label": "Correlation MDS",
        "package": None,
        "description": "Classical MDS on 1−|R| (correlation distance).",
    },
    "tsne": {
        "label": "t-SNE",
        "package": "sklearn",
        "description": "t-SNE (requires scikit-learn).",
    },
    "umap": {
        "label": "UMAP",
        "package": "umap",
        "description": "UMAP (requires umap-learn).",
    },
}


# --------------------------------------------------------------------------- #
# Availability check
# --------------------------------------------------------------------------- #
def available_methods() -> List[str]:
    """Return the names of embedding methods available in this environment."""
    out = []
    for name, info in METHODS.items():
        pkg = info["package"]
        if pkg is None:
            out.append(name)
        else:
            try:
                __import__(pkg)
                out.append(name)
            except ImportError:
                pass
    return out


# --------------------------------------------------------------------------- #
# Dimensionality reduction
# --------------------------------------------------------------------------- #
def reduce_dimensions(
    X: np.ndarray,
    labels: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    *,
    cor: Optional[np.ndarray] = None,
    by_sample: bool = True,
    normalize: bool = False,
    seed: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Embed the data into ``n_components`` dimensions.

    Parameters
    ----------
    X:
        Data matrix (n_elements, n_features).
    labels:
        Integer cluster labels for each element (used only by methods that
        benefit from label-aware colouring; the embedding itself is
        unsupervised).
    method:
        One of ``"pca"``, ``"cor_mds"``, ``"tsne"``, ``"umap"``.
    n_components:
        Target dimensionality (2 or 3).
    cor:
        Pre-computed element-by-element correlation matrix.  Required for
        ``cor_mds``; computed automatically if ``None``.
    by_sample:
        If ``normalize`` is True, standardize per row (True) or per column
        (False).
    normalize:
        Standardize the data before embedding.
    seed:
        Random seed for stochastic methods (t-SNE, UMAP).
    **kwargs:
        Extra keyword arguments forwarded to the underlying implementation
        (e.g. ``perplexity`` for t-SNE, ``n_neighbors`` for UMAP).

    Returns
    -------
    X_embed : ndarray (n_elements, n_components)
        The low-dimensional embedding.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D matrix (n_elements, n_features)")
    if labels.size != X.shape[0]:
        raise ValueError("labels length must equal the number of rows in X")
    if method not in METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Available: {', '.join(METHODS)}"
        )
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")

    if normalize:
        X = normalize_data(X, by_sample=by_sample)

    if method == "pca":
        return _pca(X, n_components)
    if method == "cor_mds":
        return _cor_mds(X, n_components, cor=cor, by_sample=by_sample)
    if method == "tsne":
        return _tsne(X, n_components, seed=seed, **kwargs)
    if method == "umap":
        return _umap(X, n_components, seed=seed, **kwargs)

    raise RuntimeError(f"Unhandled method {method!r}")  # pragma: no cover


def _pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """PCA via truncated SVD (numpy only)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :n_components] * S[:n_components])


def _cor_mds(
    X: np.ndarray, n_components: int, *, cor: Optional[np.ndarray], by_sample: bool
) -> np.ndarray:
    """Classical MDS on the correlation distance matrix 1 − |R|."""
    if cor is None:
        Xn = normalize_data(X, by_sample=by_sample)
        cor = pearson_correlation(Xn, rowvar=True)
    D = 1.0 - np.abs(cor)
    np.fill_diagonal(D, 0.0)
    # Double-centre the squared-distance matrix (Torgersen).
    n = D.shape[0]
    D2 = D ** 2
    row_mean = D2.mean(axis=1, keepdims=True)
    col_mean = D2.mean(axis=0, keepdims=True)
    grand = D2.mean()
    B = -0.5 * (D2 - row_mean - col_mean + grand)
    # Eigendecomposition — take the top positive eigenvalues.
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:n_components]
    lam = np.maximum(eigvals[idx], 0.0)
    return eigvecs[:, idx] * np.sqrt(lam)


def _tsne(X: np.ndarray, n_components: int, seed: Optional[int], **kwargs) -> np.ndarray:
    """t-SNE via scikit-learn (optional dependency)."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError(
            "t-SNE requires scikit-learn.  Install it with:  "
            "pip install scikit-learn   (or:  pip install gcmrk[viz])"
        )
    defaults = {"perplexity": min(30, max(5, X.shape[0] // 4)), "init": "pca",
                "learning_rate": "auto"}
    defaults.update(kwargs)
    tsne = TSNE(n_components=n_components, random_state=seed, **defaults)
    return tsne.fit_transform(X)


def _umap(X: np.ndarray, n_components: int, seed: Optional[int], **kwargs) -> np.ndarray:
    """UMAP via umap-learn (optional dependency)."""
    try:
        import umap as umap_mod
    except ImportError:
        raise ImportError(
            "UMAP requires umap-learn.  Install it with:  "
            "pip install umap-learn   (or:  pip install gcmrk[viz])"
        )
    defaults = {"n_neighbors": min(15, max(2, X.shape[0] // 5))}
    defaults.update(kwargs)
    reducer = umap_mod.UMAP(n_components=n_components, random_state=seed, **defaults)
    return reducer.fit_transform(X)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
# Default palette: perceptually distinct, colour-blind friendly selection.
_DEFAULT_COLOURS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
]
_UNASSIGNED_COLOUR = "#BDBDBD"


def plot_clusters(
    X_embed: np.ndarray,
    labels: np.ndarray,
    *,
    method: str = "",
    ax=None,
    title: Optional[str] = None,
    truth: Optional[np.ndarray] = None,
    show_centroids: bool = True,
    show_legend: bool = True,
    colours: Optional[Sequence[str]] = None,
    point_size: float = 30.0,
    alpha: float = 0.75,
    figsize: Tuple[float, float] = (5.0, 4.0),
):
    """Scatter-plot cluster assignments in a reduced-dimensional embedding.

    Parameters
    ----------
    X_embed:
        (n, 2) or (n, 3) embedding coordinates.
    labels:
        Integer cluster labels.
    method:
        Embedding method name (used in the axis label).
    ax:
        Matplotlib ``Axes`` to draw on; a new figure is created if ``None``.
    title:
        Plot title.
    truth:
        Optional ground-truth labels to overlay as marker shapes.
    show_centroids:
        Plot cluster centroids as large diamonds.
    show_legend:
        Include a legend.
    colours:
        Custom colour list; defaults to a curated palette.
    point_size:
        Marker size.
    alpha:
        Marker opacity.
    figsize:
        Figure size if ``ax`` is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot (same as ``ax.figure`` if ``ax`` was
        provided).
    """
    import matplotlib.pyplot as plt

    X_embed = np.asarray(X_embed, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_dim = X_embed.shape[1]
    is_3d = n_dim == 3
    colours = list(colours) if colours else list(_DEFAULT_COLOURS)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    unique = np.unique(labels)
    has_unassigned = 0 in unique
    cluster_ids = sorted(c for c in unique if c != 0)

    # Truth markers (optional): different shapes per true class.
    marker_map = {}
    if truth is not None:
        truth = np.asarray(truth, dtype=int)
        truth_classes = sorted(np.unique(truth))
        shapes = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]
        for i, tc in enumerate(truth_classes):
            marker_map[tc] = shapes[i % len(shapes)]

    def _scatter(mask, colour, label, marker="o", edge="white", alpha_=alpha):
        coords = [X_embed[mask, d] for d in range(n_dim)]
        ax.scatter(
            *coords, c=colour, s=point_size, marker=marker,
            edgecolors=edge, linewidths=0.3, alpha=alpha_,
            label=label, zorder=2,
        )

    # Unassigned (label 0).
    if has_unassigned:
        mask = labels == 0
        _scatter(mask, _UNASSIGNED_COLOUR, "unassigned", alpha_=0.35)

    # Each cluster.
    for idx, c in enumerate(cluster_ids):
        col = colours[idx % len(colours)]
        mask = labels == c
        if truth is not None:
            # Sub-scatter per truth class for marker differentiation.
            for tc in sorted(np.unique(truth[mask])):
                sub = mask & (truth == tc)
                _scatter(sub, col, f"C{c}" if tc == sorted(np.unique(truth[mask]))[0] else None,
                         marker=marker_map.get(tc, "o"))
        else:
            _scatter(mask, col, f"C{c}")

        # Centroid.
        if show_centroids:
            centroid = X_embed[mask].mean(axis=0)
            coords = [centroid[d] for d in range(n_dim)]
            ax.scatter(
                *coords, c=col, s=point_size * 4, marker="D",
                edgecolors="black", linewidths=1.0, zorder=3,
            )

    # Axis labels.
    method_label = METHODS.get(method, {}).get("label", method) if method else ""
    dim_prefix = f"{method_label} " if method_label else ""
    ax.set_xlabel(f"{dim_prefix}dim 1", fontsize=9)
    ax.set_ylabel(f"{dim_prefix}dim 2", fontsize=9)
    if is_3d:
        ax.set_zlabel(f"{dim_prefix}dim 3", fontsize=9)

    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")

    if show_legend:
        handles, lbls = ax.get_legend_handles_labels()
        # Deduplicate.
        seen = {}
        for h, l in zip(handles, lbls):
            if l and l not in seen:
                seen[l] = h
        if seen:
            ax.legend(
                seen.values(), seen.keys(),
                fontsize=7, frameon=True, framealpha=0.7,
                loc="best", markerscale=0.8,
            )

    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def plot_embedding_grid(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    cor: Optional[np.ndarray] = None,
    methods: Optional[Sequence[str]] = None,
    truth: Optional[np.ndarray] = None,
    n_components: int = 2,
    by_sample: bool = True,
    normalize: bool = False,
    seed: Optional[int] = None,
    suptitle: Optional[str] = None,
    figsize_per_panel: Tuple[float, float] = (4.2, 3.5),
    **kwargs,
):
    """Produce a multi-panel figure comparing embeddings from several methods.

    Parameters
    ----------
    X, labels, cor, n_components, by_sample, normalize, seed:
        Forwarded to :func:`reduce_dimensions`.
    methods:
        Which embedding methods to include.  Defaults to all available methods.
    truth:
        Optional ground-truth labels passed to :func:`plot_clusters`.
    suptitle:
        Figure super-title.
    figsize_per_panel:
        (width, height) per subplot panel.
    **kwargs:
        Extra keyword arguments forwarded to each embedding method.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The grid figure.
    """
    import matplotlib.pyplot as plt

    if methods is None:
        methods = available_methods()
    methods = [m for m in methods if m in available_methods()]
    if not methods:
        raise RuntimeError(
            "No embedding methods are available.  Install scikit-learn or "
            "umap-learn for t-SNE / UMAP."
        )

    n = len(methods)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fw = figsize_per_panel[0] * ncols
    fh = figsize_per_panel[1] * nrows
    is_3d = n_components == 3

    fig = plt.figure(figsize=(fw, fh))
    for i, method in enumerate(methods):
        if is_3d:
            ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        else:
            ax = fig.add_subplot(nrows, ncols, i + 1)
        X_embed = reduce_dimensions(
            X, labels, method=method, n_components=n_components,
            cor=cor, by_sample=by_sample, normalize=normalize,
            seed=seed, **kwargs,
        )
        plot_clusters(
            X_embed, labels, method=method, ax=ax,
            title=METHODS[method]["label"], truth=truth,
            show_legend=(i == 0),
        )

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig
