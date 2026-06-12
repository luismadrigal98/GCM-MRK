"""Iris as optimiser validation, not a recovery benchmark.

On Iris the internal validity indices prefer k=2 (merging the overlapping
versicolor/virginica), a textbook fact: the three-species labelling has a
*lower* silhouette than the two-cluster split. The point of including Iris is
therefore to show that the GA reaches the same index optimum a standard method
does --- i.e. that it is a faithful, competent optimiser of whichever geometric
index it is given --- with the k=2 outcome attributed entirely to the index.

Writes results/iris_macros.tex with the silhouette values that make this explicit.

Run from repo root:  PYTHONPATH=. python3 paper/experiments/iris_validation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris

from gcmrk import cluster, metrics
from gcmrk.data import normalize_data
from gcmrk.seeding import kmeans

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 20240117


def main():
    X = load_iris().data.astype(float)
    t = load_iris().target
    Xn = normalize_data(X, by_sample=False)

    def sil(lab):
        return metrics.silhouette(Xn, lab)

    true_sil = sil(t + 1)
    km2 = kmeans(Xn, 2, np.random.default_rng(SEED), n_init=20)
    km3 = kmeans(Xn, 3, np.random.default_rng(SEED), n_init=20)
    g = cluster(X, targets=["silhouette"], g_max=6, by_sample=False,
                generations=120, population_size=200, seed=SEED)

    macros = {
        "IrisTrueSil": f"{true_sil:.2f}",
        "IrisKmTwoSil": f"{sil(km2):.2f}",
        "IrisKmThreeSil": f"{sil(km3):.2f}",
        "IrisGcmrkSil": f"{sil(g.labels):.2f}",
    }
    text = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
    (RESULTS / "iris_macros.tex").write_text(text)
    print("Iris silhouette validation:")
    print(f"  true 3 species   {true_sil:.3f}")
    print(f"  k-means k=2      {sil(km2):.3f}")
    print(f"  k-means k=3      {sil(km3):.3f}")
    print(f"  GCM-MRK (sil)    {sil(g.labels):.3f}  (k={len(set(g.labels))})")
    print(f"Wrote {RESULTS/'iris_macros.tex'}")


if __name__ == "__main__":
    main()
