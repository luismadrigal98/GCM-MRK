import unittest

import numpy as np

from gcmrk import cluster
from gcmrk.ga import GAConfig, evolve, _fast_non_dominated_sort, _dominates
from gcmrk.metrics import get_metric
from gcmrk.simulate import simulate_modular_data


def _accuracy(truth, pred):
    """Best-match accuracy between two labelings (cluster ids are arbitrary)."""
    from itertools import permutations
    truth = np.asarray(truth)
    pred = np.asarray(pred)
    t_ids = np.unique(truth)
    p_ids = np.unique(pred)
    if p_ids.size > 6 or t_ids.size > 6:
        # Fall back to a greedy contingency match for larger label sets.
        best = 0
        for p in p_ids:
            mask = pred == p
            if mask.any():
                best += np.bincount(truth[mask]).max()
        return best / truth.size
    best = 0.0
    for perm in permutations(p_ids, min(len(p_ids), len(t_ids))):
        mapping = {p: t for p, t in zip(perm, t_ids)}
        mapped = np.array([mapping.get(p, -1) for p in pred])
        best = max(best, np.mean(mapped == truth))
    return best


class TestNonDominatedSort(unittest.TestCase):
    def test_dominance(self):
        self.assertTrue(_dominates(np.array([2, 2]), np.array([1, 1])))
        self.assertFalse(_dominates(np.array([2, 0]), np.array([1, 1])))

    def test_fronts(self):
        obj = np.array([[3, 3], [2, 2], [1, 1], [3, 1], [1, 3]])
        fronts = _fast_non_dominated_sort(obj)
        self.assertIn(0, fronts[0])  # [3,3] dominates all -> front 0


class TestEvolveWeighted(unittest.TestCase):
    def test_recovers_blobs_with_silhouette(self):
        X = np.vstack([
            np.random.default_rng(0).normal([0, 0], 0.3, (15, 2)),
            np.random.default_rng(1).normal([8, 8], 0.3, (15, 2)),
            np.random.default_rng(2).normal([0, 8], 0.3, (15, 2)),
        ])
        truth = np.array([1] * 15 + [2] * 15 + [3] * 15)
        config = GAConfig(g_max=5, population_size=80, generations=60,
                          seed=0, mode="weighted")
        res = evolve(X, None, [get_metric("silhouette")], config)
        self.assertGreaterEqual(_accuracy(truth, res.labels), 0.9)
        self.assertEqual(len(res.history), 60)

    def test_fitness_improves(self):
        data, _ = simulate_modular_data([15, 15], n_samples=40, seed=3)
        cor = np.corrcoef(data, rowvar=True)
        config = GAConfig(g_max=4, population_size=60, generations=40, seed=0)
        res = evolve(data, cor, [get_metric("loglik")], config)
        first = res.history[0]["max"]
        last = res.history[-1]["max"]
        self.assertGreaterEqual(last, first)


class TestEvolveNSGA2(unittest.TestCase):
    def test_returns_pareto_front(self):
        data, truth = simulate_modular_data([15, 15, 15], n_samples=40, seed=4)
        cor = np.corrcoef(data, rowvar=True)
        config = GAConfig(g_max=5, population_size=60, generations=40,
                          seed=0, mode="nsga2")
        res = evolve(data, cor, [get_metric("loglik"),
                                 get_metric("davies_bouldin")], config)
        self.assertIsNotNone(res.pareto_front)
        self.assertGreaterEqual(len(res.pareto_front), 1)
        self.assertEqual(res.labels.size, data.shape[0])


class TestClusterAPI(unittest.TestCase):
    def test_loglik_recovers_correlation_modules(self):
        # The correlation-based log-likelihood is the metric designed for
        # modular (co-expression) data; with g_max at the true cluster count it
        # recovers the modules.
        data, truth = simulate_modular_data([20, 20, 20], n_samples=50, seed=5)
        res = cluster(data, targets=["loglik"], g_max=3,
                      generations=50, population_size=80, seed=0, verbose=False)
        self.assertEqual(res.labels.size, 60)
        self.assertGreaterEqual(_accuracy(truth, res.labels), 0.8)

    def test_multi_target_weighted_runs(self):
        data, truth = simulate_modular_data([15, 15], n_samples=40, seed=6)
        res = cluster(data, targets=["loglik", "silhouette"], g_max=4,
                      generations=30, population_size=60, seed=0, verbose=False)
        self.assertIn("loglik", res.scores)
        self.assertIn("silhouette", res.scores)
        self.assertEqual(res.labels.size, 30)
        self.assertGreaterEqual(len(set(res.labels.tolist())), 2)


if __name__ == "__main__":
    unittest.main()
