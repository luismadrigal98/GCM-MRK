"""Tests for the generative-model objective and the memetic local search."""

import unittest

import numpy as np

from gcmrk import cluster, metrics
from gcmrk.data import normalize_data, pearson_correlation
from gcmrk.local import CorrelationRefiner
from gcmrk.simulate import simulate_modular_data


def _equicorr_logdet(rho, n):
    return np.log(1 + (n - 1) * rho) + (n - 1) * np.log(1 - rho)


class TestGenerativeModelIdentity(unittest.TestCase):
    """`loglik` equals -1/2 sum of equicorrelation log-determinants."""

    def test_loglik_equals_equicorr_logdet(self):
        X, t = simulate_modular_data([10, 12, 8], n_samples=40, noise=1.0,
                                     latent_per_module=1, seed=3)
        cor = pearson_correlation(normalize_data(X, by_sample=True), rowvar=True)
        expected = 0.0
        for s in np.unique(t):
            idx = np.where(t == s)[0]
            n = idx.size
            sub = np.abs(cor[np.ix_(idx, idx)])
            rho = (sub.sum() - n) / (n * n - n)
            expected += -0.5 * _equicorr_logdet(rho, n)
        got = metrics.log_likelihood_correlation(cor, t)
        self.assertAlmostEqual(got, expected, places=6)


class TestPenalizedCriteria(unittest.TestCase):
    def test_bic_scaling_uses_samples_and_cluster_count(self):
        X, t = simulate_modular_data([10, 10], n_samples=30, noise=1.0,
                                     latent_per_module=1, seed=1)
        cor = pearson_correlation(normalize_data(X, by_sample=True), rowvar=True)
        d = X.shape[1]
        ll = metrics.log_likelihood_correlation(cor, t)
        k = 2
        self.assertAlmostEqual(metrics.loglik_bic(cor, t, d),
                               -2 * d * ll + k * np.log(d), places=6)
        self.assertAlmostEqual(metrics.loglik_aic(cor, t, d),
                               -2 * d * ll + 2 * k, places=6)

    def test_bic_prefers_true_over_split(self):
        X, t = simulate_modular_data([12, 12, 12], n_samples=60, noise=0.7,
                                     latent_per_module=1, seed=2)
        cor = pearson_correlation(normalize_data(X, by_sample=True), rowvar=True)
        d = X.shape[1]
        split = t.copy()
        m = np.where(t == 1)[0]
        split[m[: m.size // 2]] = t.max() + 1   # over-segment one module
        self.assertLess(metrics.loglik_bic(cor, t, d),
                        metrics.loglik_bic(cor, split, d))


class TestLocalSearch(unittest.TestCase):
    def test_refine_never_decreases_loglik(self):
        X, t = simulate_modular_data([15, 20, 10], n_samples=50, noise=1.5,
                                     latent_per_module=1, seed=7)
        cor = pearson_correlation(normalize_data(X, by_sample=True), rowvar=True)
        rng = np.random.default_rng(0)
        refiner = CorrelationRefiner(np.abs(cor), "loglik", X.shape[1],
                                     g_max=8, all_in_clusters=True)
        start = rng.integers(1, 6, size=len(t))
        before = metrics.log_likelihood_correlation(cor, start)
        after = metrics.log_likelihood_correlation(cor, refiner.refine(start))
        self.assertGreaterEqual(after + 1e-9, before)

    def test_memetic_beats_no_local_search_on_hard_data(self):
        from sklearn.metrics import adjusted_rand_score as ari
        X, t = simulate_modular_data([15, 20, 25, 30, 10], n_samples=50,
                                     noise=1.5, latent_per_module=1, seed=0)
        mem = cluster(X, targets=["loglik"], g_max=5, generations=60,
                      population_size=100, seed=0)
        nol = cluster(X, targets=["loglik"], g_max=5, generations=60,
                      population_size=100, seed=0, local_search=False)
        self.assertGreaterEqual(ari(t, mem.labels), ari(t, nol.labels) - 1e-9)


if __name__ == "__main__":
    unittest.main()
