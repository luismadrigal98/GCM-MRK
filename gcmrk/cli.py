"""Command-line interface for GCM-MRK.

Examples
--------
List the available performance-target metrics::

    gcmrk metrics

Cluster a CSV, maximizing the silhouette score, into at most 6 clusters::

    gcmrk cluster --data expr.csv --targets silhouette --g-max 6 \
        --generations 100 --population 200 --out labels.txt

Multi-objective (Pareto) clustering on two competing targets::

    gcmrk cluster --data expr.csv --targets loglik davies_bouldin \
        --mode nsga2 --g-max 8 --out labels.txt

Generate synthetic modular data to experiment with::

    gcmrk simulate --modules 30 30 40 --samples 100 --out sim.csv

Visualize a clustering result in a reduced-dimensional embedding::

    gcmrk visualize --data expr.csv --labels labels.txt \
        --methods pca cor_mds --out clusters.pdf
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from .core import cluster
from .data import load_matrix, normalize_data, pearson_correlation
from .metrics import METRICS
from .simulate import simulate_modular_data
from .visualize import (
    METHODS as VIZ_METHODS,
    available_methods,
    plot_clusters,
    plot_embedding_grid,
    reduce_dimensions,
)


def _add_cluster_parser(sub):
    p = sub.add_parser(
        "cluster",
        help="Cluster a dataset by optimizing performance-target metrics.",
        description="Evolve a clustering that optimizes the chosen metric(s).",
    )
    p.add_argument("--data", required=True,
                   help="Path to a CSV/TSV file. Rows = elements, columns = features.")
    p.add_argument("--targets", nargs="+", default=["loglik"],
                   metavar="METRIC",
                   help="Metric(s) to optimize. See `gcmrk metrics`. "
                        "Default: loglik.")
    p.add_argument("--g-max", type=int, default=10,
                   help="Maximum number of clusters (default 10).")
    p.add_argument("--mode", choices=["weighted", "nsga2"], default="weighted",
                   help="Single weighted objective or NSGA-II Pareto search.")
    p.add_argument("--sep", default=",", help="Field separator (default ',').")
    p.add_argument("--allow-unassigned", action="store_true",
                   help="Permit elements to be left unassigned (label 0).")
    p.add_argument("--no-normalize", action="store_true",
                   help="Do not standardize the data before clustering.")
    p.add_argument("--by-feature", action="store_true",
                   help="Normalize per feature (column) instead of per sample (row).")
    p.add_argument("--unassigned-penalty", type=float, default=0.0,
                   help="Penalty strength for unassigned elements (default 0).")
    p.add_argument("--seed-file", default=None,
                   help="File of seed partitions (one per line) to inject.")
    p.add_argument("--population", type=int, default=200,
                   help="Population size (default 200).")
    p.add_argument("--generations", type=int, default=100,
                   help="Number of generations (default 100).")
    p.add_argument("--tournament-size", type=int, default=3)
    p.add_argument("--crossover-prob", type=float, default=0.8)
    p.add_argument("--mutation-prob", type=float, default=0.2)
    p.add_argument("--mutation-intensity", type=float, default=0.05,
                   help="Per-gene reassignment probability when mutating.")
    p.add_argument("--elitism", type=int, default=5)
    p.add_argument("--kmeans-restarts", type=int, default=2,
                   help="k-means seed partitions per candidate cluster count "
                        "(0 disables k-means seeding).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility.")
    p.add_argument("--out", default=None,
                   help="Write the resulting labels to this file (one per line). "
                        "If omitted, labels are printed to stdout.")
    p.add_argument("--pareto-out", default=None,
                   help="(nsga2) Write the full Pareto front of label vectors "
                        "to this file, one partition per line.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-generation progress output.")
    # Visualization shortcut: auto-generate an embedding plot after clustering.
    p.add_argument("--plot", action="store_true",
                   help="Generate a dimensionality-reduction scatter plot of "
                        "the resulting clusters (PCA + correlation MDS).")
    p.add_argument("--plot-out", default=None,
                   help="Output path for the --plot figure (default: "
                        "<out_basename>_viz.pdf or clusters_viz.pdf).")
    p.add_argument("--plot-methods", nargs="+", default=None,
                   metavar="METHOD",
                   help="Embedding methods for --plot (default: pca cor_mds). "
                        f"Available: {', '.join(VIZ_METHODS)}.")
    return p


def _add_simulate_parser(sub):
    p = sub.add_parser(
        "simulate",
        help="Generate synthetic modular data for testing.",
        description="Write a synthetic block-correlated dataset and its truth.",
    )
    p.add_argument("--modules", nargs="+", type=int, required=True,
                   metavar="SIZE", help="Rows per module, e.g. --modules 30 30 40.")
    p.add_argument("--samples", type=int, default=200,
                   help="Number of columns/samples (default 200).")
    p.add_argument("--latent", type=int, default=4,
                   help="Latent factors per module (default 4).")
    p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", required=True, help="Output CSV path for the data.")
    p.add_argument("--truth-out", default=None,
                   help="Optional path to write ground-truth labels.")
    return p


def _add_visualize_parser(sub):
    p = sub.add_parser(
        "visualize",
        help="Visualize clustering results in a reduced-dimensional embedding.",
        description="Project data into 2-D (or 3-D) space using PCA, "
                    "correlation MDS, t-SNE, or UMAP and produce a scatter "
                    "plot coloured by cluster labels.",
    )
    p.add_argument("--data", required=True,
                   help="Path to the data matrix (CSV/TSV).")
    p.add_argument("--labels", required=True,
                   help="Path to a label file (one integer per line).")
    p.add_argument("--methods", nargs="+", default=["pca", "cor_mds"],
                   metavar="METHOD",
                   help="Embedding method(s) to use. "
                        f"Available: {', '.join(VIZ_METHODS)}. "
                        "Default: pca cor_mds.")
    p.add_argument("--truth", default=None,
                   help="Optional path to ground-truth labels for overlay.")
    p.add_argument("--out", default=None,
                   help="Output figure path (PDF/PNG/SVG). "
                        "If omitted, the figure is displayed interactively.")
    p.add_argument("--n-components", type=int, default=2, choices=[2, 3],
                   help="Dimensionality of the embedding (default 2).")
    p.add_argument("--title", default=None, help="Custom figure title.")
    p.add_argument("--sep", default=",", help="Field separator (default ',').")
    p.add_argument("--no-normalize", action="store_true",
                   help="Do not standardize the data.")
    p.add_argument("--by-feature", action="store_true",
                   help="Normalize per feature (column) instead of per sample (row).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for stochastic embeddings (t-SNE, UMAP).")
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gcmrk",
        description="Genetic-algorithm clustering guided by performance-target "
                    "metrics.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_cluster_parser(sub)
    _add_simulate_parser(sub)
    _add_visualize_parser(sub)
    sub.add_parser("metrics", help="List available performance-target metrics.")
    return parser


def _run_cluster(args) -> int:
    result = cluster(
        args.data,
        targets=args.targets,
        g_max=args.g_max,
        mode=args.mode,
        all_in_clusters=not args.allow_unassigned,
        normalize=not args.no_normalize,
        by_sample=not args.by_feature,
        sep=args.sep,
        seed_file=args.seed_file,
        unassigned_penalty=args.unassigned_penalty,
        population_size=args.population,
        generations=args.generations,
        tournament_size=args.tournament_size,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        mutation_intensity=args.mutation_intensity,
        elitism=args.elitism,
        kmeans_restarts=args.kmeans_restarts,
        seed=args.seed,
        verbose=not args.quiet,
    )

    n_clusters = len(set(result.labels.tolist()) - {0})
    print(f"\nFound {n_clusters} clusters over {result.labels.size} elements.",
          file=sys.stderr)
    print("Metric scores:", file=sys.stderr)
    for name, value in result.scores.items():
        print(f"  {name:18s} {value:.6g}", file=sys.stderr)

    labels_text = "\n".join(str(int(x)) for x in result.labels)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(labels_text + "\n")
        print(f"Labels written to {args.out}", file=sys.stderr)
    else:
        print(labels_text)

    if args.mode == "nsga2" and args.pareto_out and result.pareto_front:
        with open(args.pareto_out, "w") as fh:
            for entry in result.pareto_front:
                fh.write(",".join(str(int(x)) for x in entry["labels"]) + "\n")
        print(f"Pareto front ({len(result.pareto_front)} solutions) written to "
              f"{args.pareto_out}", file=sys.stderr)

    # Auto-visualization when --plot is given.
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")

        X = load_matrix(args.data, sep=args.sep)
        if not args.no_normalize:
            X = normalize_data(X, by_sample=not args.by_feature)

        methods = args.plot_methods or ["pca", "cor_mds"]
        fig = plot_embedding_grid(
            X, result.labels, methods=methods,
            by_sample=not args.by_feature,
            suptitle="Cluster visualization",
        )

        if args.plot_out:
            plot_path = args.plot_out
        elif args.out:
            base, _ = os.path.splitext(args.out)
            plot_path = base + "_viz.pdf"
        else:
            plot_path = "clusters_viz.pdf"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"Visualization written to {plot_path}", file=sys.stderr)

    return 0


def _run_simulate(args) -> int:
    data, labels = simulate_modular_data(
        args.modules, n_samples=args.samples, latent_per_module=args.latent,
        noise=args.noise, seed=args.seed,
    )
    np.savetxt(args.out, data, delimiter=",", fmt="%.6g")
    print(f"Wrote {data.shape[0]}x{data.shape[1]} data matrix to {args.out}",
          file=sys.stderr)
    if args.truth_out:
        np.savetxt(args.truth_out, labels, fmt="%d")
        print(f"Wrote ground-truth labels to {args.truth_out}", file=sys.stderr)
    return 0


def _run_visualize(args) -> int:
    """Execute the ``gcmrk visualize`` subcommand."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X = load_matrix(args.data, sep=args.sep)
    if not args.no_normalize:
        X = normalize_data(X, by_sample=not args.by_feature)

    labels = np.loadtxt(args.labels, dtype=int)
    if labels.size != X.shape[0]:
        print(f"Error: label file has {labels.size} entries but data has "
              f"{X.shape[0]} rows.", file=sys.stderr)
        return 1

    truth = None
    if args.truth:
        truth = np.loadtxt(args.truth, dtype=int)

    methods = [m for m in args.methods if m in available_methods()]
    unavailable = [m for m in args.methods if m not in available_methods()]
    if unavailable:
        print(f"Warning: skipping unavailable methods: {', '.join(unavailable)}",
              file=sys.stderr)
    if not methods:
        print("Error: no embedding methods available.", file=sys.stderr)
        return 1

    fig = plot_embedding_grid(
        X, labels, methods=methods, truth=truth,
        n_components=args.n_components,
        by_sample=not args.by_feature,
        seed=args.seed,
        suptitle=args.title,
    )

    if args.out:
        fig.savefig(args.out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved to {args.out}", file=sys.stderr)
    else:
        plt.show()
    return 0


def _run_metrics() -> int:
    print("Available performance-target metrics:\n")
    for name, spec in METRICS.items():
        arrow = "maximize" if spec.direction == "max" else "minimize"
        print(f"  {name:18s} [{arrow:8s}] {spec.description}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "cluster":
        return _run_cluster(args)
    if args.command == "simulate":
        return _run_simulate(args)
    if args.command == "visualize":
        return _run_visualize(args)
    if args.command == "metrics":
        return _run_metrics()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
