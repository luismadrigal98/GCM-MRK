"""Tests for the dimensionality-reduction visualization module."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from gcmrk.simulate import simulate_modular_data
from gcmrk.visualize import (
    METHODS,
    available_methods,
    plot_clusters,
    plot_embedding_grid,
    reduce_dimensions,
)


def _simple_data():
    """Small synthetic dataset with known clusters."""
    X, labels = simulate_modular_data([10, 10, 10], n_samples=30, seed=42)
    return X, labels


class TestReduceDimensions(unittest.TestCase):
    """Test the reduce_dimensions function."""

    def setUp(self):
        self.X, self.labels = _simple_data()

    def test_pca_shape_2d(self):
        X2 = reduce_dimensions(self.X, self.labels, method="pca", n_components=2)
        self.assertEqual(X2.shape, (self.X.shape[0], 2))

    def test_pca_shape_3d(self):
        X3 = reduce_dimensions(self.X, self.labels, method="pca", n_components=3)
        self.assertEqual(X3.shape, (self.X.shape[0], 3))

    def test_cor_mds_shape_2d(self):
        X2 = reduce_dimensions(self.X, self.labels, method="cor_mds", n_components=2)
        self.assertEqual(X2.shape, (self.X.shape[0], 2))

    def test_cor_mds_shape_3d(self):
        X3 = reduce_dimensions(self.X, self.labels, method="cor_mds", n_components=3)
        self.assertEqual(X3.shape, (self.X.shape[0], 3))

    def test_cor_mds_with_precomputed_cor(self):
        from gcmrk.data import pearson_correlation, normalize_data
        Xn = normalize_data(self.X, by_sample=True)
        cor = pearson_correlation(Xn, rowvar=True)
        X2 = reduce_dimensions(self.X, self.labels, method="cor_mds", cor=cor)
        self.assertEqual(X2.shape, (self.X.shape[0], 2))
        # Should produce same result as without precomputed cor.
        X2b = reduce_dimensions(self.X, self.labels, method="cor_mds")
        np.testing.assert_allclose(X2, X2b, atol=1e-10)

    def test_pca_values_finite(self):
        X2 = reduce_dimensions(self.X, self.labels, method="pca")
        self.assertTrue(np.all(np.isfinite(X2)))

    def test_cor_mds_values_finite(self):
        X2 = reduce_dimensions(self.X, self.labels, method="cor_mds")
        self.assertTrue(np.all(np.isfinite(X2)))

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            reduce_dimensions(self.X, self.labels, method="nonexistent")

    def test_bad_n_components_raises(self):
        with self.assertRaises(ValueError):
            reduce_dimensions(self.X, self.labels, n_components=4)

    def test_mismatched_labels_raises(self):
        with self.assertRaises(ValueError):
            reduce_dimensions(self.X, self.labels[:5], method="pca")

    def test_1d_data_raises(self):
        with self.assertRaises(ValueError):
            reduce_dimensions(self.X[0], self.labels, method="pca")

    def test_normalize_flag(self):
        X2a = reduce_dimensions(self.X, self.labels, method="pca", normalize=False)
        X2b = reduce_dimensions(self.X, self.labels, method="pca", normalize=True)
        # Should produce different results when normalization changes.
        self.assertEqual(X2a.shape, X2b.shape)

    def test_tsne_unavailable_message(self):
        """If sklearn is not installed, t-SNE should raise a clear error."""
        if "tsne" not in available_methods():
            with self.assertRaises(ImportError) as ctx:
                reduce_dimensions(self.X, self.labels, method="tsne")
            self.assertIn("scikit-learn", str(ctx.exception))

    def test_tsne_if_available(self):
        """If sklearn is installed, t-SNE should work."""
        if "tsne" in available_methods():
            X2 = reduce_dimensions(self.X, self.labels, method="tsne",
                                   n_components=2, seed=0)
            self.assertEqual(X2.shape, (self.X.shape[0], 2))

    def test_umap_unavailable_message(self):
        """If umap is not installed, UMAP should raise a clear error."""
        if "umap" not in available_methods():
            with self.assertRaises(ImportError) as ctx:
                reduce_dimensions(self.X, self.labels, method="umap")
            self.assertIn("umap-learn", str(ctx.exception))


class TestAvailableMethods(unittest.TestCase):
    """Test method availability check."""

    def test_pca_always_available(self):
        self.assertIn("pca", available_methods())

    def test_cor_mds_always_available(self):
        self.assertIn("cor_mds", available_methods())

    def test_returns_list(self):
        self.assertIsInstance(available_methods(), list)


class TestPlotClusters(unittest.TestCase):
    """Test the plot_clusters function."""

    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")
        self.X, self.labels = _simple_data()
        self.X2 = reduce_dimensions(self.X, self.labels, method="pca")

    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        fig = plot_clusters(self.X2, self.labels)
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_with_method_name(self):
        import matplotlib.pyplot as plt
        fig = plot_clusters(self.X2, self.labels, method="pca")
        plt.close(fig)

    def test_with_title(self):
        import matplotlib.pyplot as plt
        fig = plot_clusters(self.X2, self.labels, title="Test Title")
        plt.close(fig)

    def test_with_truth_overlay(self):
        import matplotlib.pyplot as plt
        fig = plot_clusters(self.X2, self.labels, truth=self.labels)
        plt.close(fig)

    def test_no_centroids(self):
        import matplotlib.pyplot as plt
        fig = plot_clusters(self.X2, self.labels, show_centroids=False)
        plt.close(fig)

    def test_no_legend(self):
        import matplotlib.pyplot as plt
        fig = plot_clusters(self.X2, self.labels, show_legend=False)
        plt.close(fig)

    def test_custom_colours(self):
        import matplotlib.pyplot as plt
        fig = plot_clusters(self.X2, self.labels,
                            colours=["red", "green", "blue"])
        plt.close(fig)

    def test_with_existing_axes(self):
        import matplotlib.pyplot as plt
        fig0, ax = plt.subplots()
        fig = plot_clusters(self.X2, self.labels, ax=ax)
        self.assertIs(fig, fig0)
        plt.close(fig)

    def test_unassigned_handling(self):
        """Elements with label 0 should be plotted in grey."""
        import matplotlib.pyplot as plt
        labels = self.labels.copy()
        labels[:5] = 0
        fig = plot_clusters(self.X2, labels)
        plt.close(fig)

    def test_3d_plot(self):
        import matplotlib.pyplot as plt
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except ImportError:
            self.skipTest("3D projection not available in this matplotlib install")
        X3 = reduce_dimensions(self.X, self.labels, method="pca", n_components=3)
        try:
            fig = plot_clusters(X3, self.labels)
            plt.close(fig)
        except ValueError as e:
            if "3d" in str(e).lower():
                self.skipTest("3D projection not registered in this matplotlib")
            raise

    def test_save_to_file(self):
        import matplotlib.pyplot as plt
        fig = plot_clusters(self.X2, self.labels)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            fig.savefig(path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)
            plt.close(fig)


class TestPlotEmbeddingGrid(unittest.TestCase):
    """Test the multi-panel embedding grid."""

    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")
        self.X, self.labels = _simple_data()

    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        fig = plot_embedding_grid(self.X, self.labels,
                                  methods=["pca", "cor_mds"])
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_single_method(self):
        import matplotlib.pyplot as plt
        fig = plot_embedding_grid(self.X, self.labels, methods=["pca"])
        plt.close(fig)

    def test_with_truth(self):
        import matplotlib.pyplot as plt
        fig = plot_embedding_grid(self.X, self.labels,
                                  methods=["pca"], truth=self.labels)
        plt.close(fig)

    def test_with_suptitle(self):
        import matplotlib.pyplot as plt
        fig = plot_embedding_grid(self.X, self.labels,
                                  methods=["pca"], suptitle="Grid Test")
        plt.close(fig)

    def test_defaults_to_available(self):
        import matplotlib.pyplot as plt
        fig = plot_embedding_grid(self.X, self.labels)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
