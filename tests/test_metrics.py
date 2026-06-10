import unittest

import numpy as np

from gcmrk import metrics
from gcmrk.simulate import simulate_modular_data


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Two well-separated blobs in 2-D.
        self.X = np.array([
            [0.0, 0.0], [0.1, 0.1], [-0.1, 0.0],
            [10.0, 10.0], [10.1, 9.9], [9.9, 10.1],
        ])
        self.good = np.array([1, 1, 1, 2, 2, 2])
        self.bad = np.array([1, 2, 1, 2, 1, 2])

    def test_silhouette_prefers_good_partition(self):
        good = metrics.silhouette(self.X, self.good)
        bad = metrics.silhouette(self.X, self.bad)
        self.assertGreater(good, bad)
        self.assertGreater(good, 0.8)

    def test_silhouette_single_cluster_is_worst(self):
        self.assertEqual(metrics.silhouette(self.X, np.ones(6, dtype=int)), -1.0)

    def test_davies_bouldin_lower_for_good(self):
        good = metrics.davies_bouldin(self.X, self.good)
        bad = metrics.davies_bouldin(self.X, self.bad)
        self.assertLess(good, bad)

    def test_davies_bouldin_single_cluster_is_inf(self):
        self.assertEqual(metrics.davies_bouldin(self.X, np.ones(6, dtype=int)), np.inf)

    def test_calinski_harabasz_higher_for_good(self):
        good = metrics.calinski_harabasz(self.X, self.good)
        bad = metrics.calinski_harabasz(self.X, self.bad)
        self.assertGreater(good, bad)

    def test_bic_aic_finite_and_prefer_good(self):
        for fn in (metrics.bic, metrics.aic):
            good = fn(self.X, self.good)
            bad = fn(self.X, self.bad)
            self.assertTrue(np.isfinite(good))
            self.assertLess(good, bad)  # lower is better

    def test_loglik_prefers_true_modules(self):
        data, truth = simulate_modular_data([15, 15, 15], n_samples=40, seed=1)
        cor = np.corrcoef(data, rowvar=True)
        good = metrics.log_likelihood_correlation(cor, truth)
        rng = np.random.default_rng(0)
        shuffled = rng.permutation(truth)
        bad = metrics.log_likelihood_correlation(cor, shuffled)
        self.assertGreater(good, bad)

    def test_unassigned_penalty_applies(self):
        labels = self.good.copy()
        labels[0] = 0  # leave one element unassigned
        base = metrics.silhouette(self.X, labels, unassigned_penalty=0.0)
        penalised = metrics.silhouette(self.X, labels, unassigned_penalty=1.0)
        self.assertLess(penalised, base)

    def test_registry_directions(self):
        self.assertEqual(metrics.get_metric("silhouette").weight, 1.0)
        self.assertEqual(metrics.get_metric("davies_bouldin").weight, -1.0)
        with self.assertRaises(KeyError):
            metrics.get_metric("nonexistent")


if __name__ == "__main__":
    unittest.main()
