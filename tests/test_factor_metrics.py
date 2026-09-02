"""Tests for the rank-q free-loading factor block model.

These cover the properties that motivated adding the branch: it must handle
modules whose members carry *unequal* loadings (which the exchangeable
``loglik`` cannot), must treat anti-correlated members through negative loadings
rather than an absolute value, and must charge a parameter cost that grows with
module size.
"""

import unittest

import numpy as np

from gcmrk import metrics
from gcmrk.local import FactorRefiner


def _corr(X):
    X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    return np.corrcoef(X)


def _one_factor(n, d, rng, weights=None, noise=0.3):
    """One latent factor, optionally with per-gene loading magnitudes."""
    z = rng.normal(size=d)
    if weights is None:
        weights = np.ones(n)
    signs = rng.choice([-1.0, 1.0], size=n)
    return np.outer(signs * weights, z) + rng.normal(0, noise, size=(n, d))


class TestFactorMetric(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.d = 60

    def test_prefers_true_partition_over_scrambled(self):
        rng = self.rng
        X = np.vstack([_one_factor(10, self.d, rng), _one_factor(10, self.d, rng)])
        cor = _corr(X)
        truth = np.array([1] * 10 + [2] * 10)
        scrambled = np.array([1, 2] * 10)
        self.assertGreater(metrics.factor_loglik(cor, truth, self.d),
                           metrics.factor_loglik(cor, scrambled, self.d))

    def test_handles_anticorrelated_members(self):
        """A module with mixed-sign loadings must score as a good module.

        The factor model represents this with negative loadings, so it needs no
        absolute value; a naive signed-correlation-sum objective would not.
        """
        rng = np.random.default_rng(1)
        z = rng.normal(size=self.d)
        signs = np.array([1.0] * 5 + [-1.0] * 5)
        block = np.outer(signs, z) + rng.normal(0, 0.2, size=(10, self.d))
        other = rng.normal(size=(10, self.d))
        cor = _corr(np.vstack([block, other]))
        truth = np.array([1] * 10 + [2] * 10)
        split = np.array([1] * 5 + [3] * 5 + [2] * 10)  # splits the module by sign
        self.assertGreater(metrics.factor_loglik(cor, truth, self.d),
                           metrics.factor_loglik(cor, split, self.d))

    def test_unequal_loadings_favour_factor_over_exchangeable(self):
        """The case the branch exists for.

        With geometrically decaying loadings the true module has heterogeneous
        within-module correlation.  The factor model should still rank the truth
        above a partition that drops the weak members into another cluster.
        """
        rng = np.random.default_rng(2)
        weights = 0.5 ** np.linspace(0, 1, 12)
        mod = _one_factor(12, self.d, rng, weights=weights, noise=0.35)
        other = _one_factor(12, self.d, rng, noise=0.35)
        cor = _corr(np.vstack([mod, other]))
        truth = np.array([1] * 12 + [2] * 12)
        # move the four weakest members of module 1 into module 2
        mangled = truth.copy()
        mangled[8:12] = 2
        self.assertGreater(metrics.factor_loglik(cor, truth, self.d, q=1),
                           metrics.factor_loglik(cor, mangled, self.d, q=1))

    def test_rank_q_captures_multifactor_module(self):
        """A 3-factor module is better explained at q=3 than at q=1."""
        rng = np.random.default_rng(3)
        latents = rng.normal(size=(3, self.d))
        loadings = rng.normal(size=(15, 3))
        X = loadings @ latents + rng.normal(0, 0.3, size=(15, self.d))
        cor = _corr(X)
        labels = np.ones(15, dtype=int)
        per_param_1 = metrics.factor_loglik(cor, labels, self.d, q=1)
        per_param_3 = metrics.factor_loglik(cor, labels, self.d, q=3)
        self.assertGreater(per_param_3, per_param_1)

    def test_parameter_count_scales_with_module_size(self):
        small = np.array([1] * 5 + [2] * 5)
        large = np.array([1] * 50 + [2] * 50)
        p_small = metrics._factor_n_params(small, q=1)
        p_large = metrics._factor_n_params(large, q=1)
        self.assertGreater(p_large, p_small)
        # exchangeable model charges the same for both, which is the flaw the
        # factor parameter count fixes
        self.assertEqual(metrics._n_coherent_clusters(small),
                         metrics._n_coherent_clusters(large))

    def test_parameter_count_formula(self):
        labels = np.array([1] * 10 + [2] * 6)
        # n*q - q(q-1)/2 + 1, summed: (10*2-1+1) + (6*2-1+1) = 20 + 12
        self.assertEqual(metrics._factor_n_params(labels, q=2), 32)

    def test_singletons_contribute_nothing(self):
        rng = np.random.default_rng(4)
        cor = _corr(rng.normal(size=(6, self.d)))
        with_singleton = np.array([1, 1, 1, 2, 2, 3])
        without = np.array([1, 1, 1, 2, 2, 2])
        # a singleton cluster adds no parameters
        self.assertEqual(metrics._factor_n_params(with_singleton, q=1),
                         metrics._factor_n_params(np.array([1, 1, 1, 2, 2, 0]), q=1))
        self.assertTrue(np.isfinite(metrics.factor_loglik(cor, without, self.d)))

    def test_information_criteria_are_finite_and_minimized(self):
        rng = np.random.default_rng(5)
        X = np.vstack([_one_factor(10, self.d, rng), _one_factor(10, self.d, rng)])
        cor = _corr(X)
        truth = np.array([1] * 10 + [2] * 10)
        over = np.array([1] * 5 + [3] * 5 + [2] * 5 + [4] * 5)
        for fn in (metrics.factor_bic, metrics.factor_aic):
            self.assertTrue(np.isfinite(fn(cor, truth, self.d)))
            # lower is better: the truth should beat the over-segmented split
            self.assertLess(fn(cor, truth, self.d), fn(cor, over, self.d))

    def test_bic_penalises_more_than_aic(self):
        rng = np.random.default_rng(6)
        cor = _corr(np.vstack([_one_factor(10, self.d, rng),
                               _one_factor(10, self.d, rng)]))
        labels = np.array([1] * 10 + [2] * 10)
        # same likelihood, heavier penalty => larger criterion value
        self.assertGreater(metrics.factor_bic(cor, labels, self.d),
                           metrics.factor_aic(cor, labels, self.d))


class TestFactorMetricRegistry(unittest.TestCase):
    def test_rank_suffix_parsing(self):
        self.assertEqual(metrics.factor_rank("loglik_factor@3"), 3)
        self.assertEqual(metrics.factor_rank("loglik_factor"), 1)
        self.assertEqual(metrics.factor_rank("loglik"), 1)

    def test_get_metric_with_rank(self):
        spec = metrics.get_metric("loglik_factor@2")
        self.assertEqual(spec.name, "loglik_factor@2")
        self.assertEqual(spec.direction, "max")
        self.assertEqual(spec.needs, "cor")
        self.assertTrue(spec.wants_n)

    def test_get_metric_rank_changes_score(self):
        rng = np.random.default_rng(7)
        d = 60
        latents = rng.normal(size=(3, d))
        X = rng.normal(size=(15, 3)) @ latents + rng.normal(0, 0.3, size=(15, d))
        cor = _corr(X)
        labels = np.ones(15, dtype=int)
        s1 = metrics.get_metric("loglik_factor@1").func(cor, labels, d)
        s3 = metrics.get_metric("loglik_factor@3").func(cor, labels, d)
        self.assertNotAlmostEqual(s1, s3)

    def test_rejects_bad_rank_suffix(self):
        with self.assertRaises(KeyError):
            metrics.get_metric("loglik_factor@x")
        with self.assertRaises(KeyError):
            metrics.get_metric("loglik_factor@0")
        with self.assertRaises(KeyError):
            metrics.get_metric("silhouette@2")

    def test_factor_metrics_registered(self):
        for name in metrics.FACTOR_METRICS:
            self.assertIn(name, metrics.METRICS)


class TestFactorRefiner(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(8)
        self.d = 60
        X = np.vstack([_one_factor(12, self.d, self.rng, noise=0.4),
                       _one_factor(12, self.d, self.rng, noise=0.4)])
        self.cor = _corr(X)
        self.truth = np.array([1] * 12 + [2] * 12)

    def test_supports_only_factor_metrics(self):
        self.assertTrue(FactorRefiner.supports(["loglik_factor"]))
        self.assertTrue(FactorRefiner.supports(["loglik_factor_aic@2"]))
        self.assertFalse(FactorRefiner.supports(["loglik"]))
        self.assertFalse(FactorRefiner.supports(["silhouette"]))

    def test_refine_does_not_worsen_objective(self):
        refiner = FactorRefiner(self.cor, "loglik_factor", self.d,
                                g_max=4, all_in_clusters=True)
        start = np.array([1, 2] * 12)          # deliberately scrambled
        before = metrics.factor_loglik(self.cor, start, self.d)
        after_labels = refiner.refine(start)
        after = metrics.factor_loglik(self.cor, after_labels, self.d)
        self.assertGreaterEqual(after, before)

    def test_refine_recovers_structure_from_scrambled_start(self):
        from sklearn.metrics import adjusted_rand_score
        refiner = FactorRefiner(self.cor, "loglik_factor", self.d,
                                g_max=4, all_in_clusters=True)
        start = np.array([1, 2] * 12)
        out = refiner.refine(start)
        self.assertGreater(adjusted_rand_score(self.truth, out), 0.5)

    def test_refine_preserves_length_and_assignment(self):
        refiner = FactorRefiner(self.cor, "loglik_factor", self.d,
                                g_max=4, all_in_clusters=True)
        out = refiner.refine(np.array([1, 2] * 12))
        self.assertEqual(out.size, self.truth.size)
        self.assertTrue(np.all(out != 0))

    def test_skips_refinement_above_size_cap(self):
        """Large problems fall back to no hill-climb rather than stalling."""
        refiner = FactorRefiner(self.cor, "loglik_factor", self.d,
                                g_max=4, all_in_clusters=True, max_elements=5)
        start = np.array([1, 2] * 12)
        np.testing.assert_array_equal(refiner.refine(start), start)

    def test_uses_signed_correlation(self):
        """The refiner must not be handed |R| -- signs carry loading direction."""
        refiner = FactorRefiner(self.cor, "loglik_factor", self.d,
                                g_max=4, all_in_clusters=True)
        self.assertTrue(np.any(refiner.R < 0))


if __name__ == "__main__":
    unittest.main()


class TestClusterFactorAuto(unittest.TestCase):
    """Rank selection: the user should not have to guess q."""

    def setUp(self):
        rng = np.random.default_rng(11)
        self.d = 50
        self.X = np.vstack([_one_factor(12, self.d, rng, noise=0.4),
                            _one_factor(12, self.d, rng, noise=0.4)])
        self.truth = np.array([1] * 12 + [2] * 12)
        self.kw = dict(g_max=2, generations=25, population_size=40, seed=0)

    def test_selects_rank_one_for_one_factor_data(self):
        from gcmrk import cluster_factor_auto
        res = cluster_factor_auto(self.X, ranks=(1, 2, 3), **self.kw)
        self.assertEqual(res.scores["factor_rank"], 1)

    def test_recovers_structure(self):
        from gcmrk import cluster_factor_auto
        from sklearn.metrics import adjusted_rand_score
        res = cluster_factor_auto(self.X, ranks=(1, 2), **self.kw)
        self.assertGreater(adjusted_rand_score(self.truth, res.labels), 0.7)

    def test_rejects_targets_kwarg(self):
        from gcmrk import cluster_factor_auto
        with self.assertRaises(TypeError):
            cluster_factor_auto(self.X, targets=["loglik"], **self.kw)

    def test_rejects_bad_criterion_and_ranks(self):
        from gcmrk import cluster_factor_auto
        with self.assertRaises(ValueError):
            cluster_factor_auto(self.X, criterion="mdl", **self.kw)
        with self.assertRaises(ValueError):
            cluster_factor_auto(self.X, ranks=(), **self.kw)
        with self.assertRaises(ValueError):
            cluster_factor_auto(self.X, ranks=(0, 1), **self.kw)

    def test_accepts_aic_criterion(self):
        from gcmrk import cluster_factor_auto
        res = cluster_factor_auto(self.X, ranks=(1, 2), criterion="aic", **self.kw)
        self.assertIn(res.scores["factor_rank"], (1, 2))


class TestFactorParameterCountLimitation(unittest.TestCase):
    """Pin the known limitation of the factor parameter count.

    Summed over k modules totalling n elements the count is
    ``n*q - k*q(q-1)/2 + k``, so it grows with k only at q=1.  At q>=2 the
    information criteria therefore cannot penalise partition complexity, which is
    why `cluster_factor_auto` must be given a constrained `g_max`.  These tests
    exist so the property is documented and cannot regress silently.
    """

    def _params(self, n, k, q):
        return metrics._factor_n_params(np.repeat(np.arange(1, k + 1), n // k), q=q)

    def test_rank_one_penalises_splitting(self):
        self.assertLess(self._params(100, 2, 1), self._params(100, 20, 1))

    def test_rank_two_is_flat_in_k(self):
        self.assertEqual(self._params(100, 2, 2), self._params(100, 20, 2))

    def test_rank_three_rewards_splitting(self):
        # the documented failure mode: more modules cost fewer parameters
        self.assertGreater(self._params(100, 2, 3), self._params(100, 20, 3))

    def test_matches_closed_form(self):
        n, k, q = 120, 6, 2
        expected = n * q - k * q * (q - 1) // 2 + k
        self.assertEqual(self._params(n, k, q), expected)
