import os
import tempfile
import unittest

import numpy as np

from gcmrk.cli import main
from gcmrk.simulate import simulate_modular_data


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data_path = os.path.join(self.tmp, "data.csv")
        data, truth = simulate_modular_data([15, 15], n_samples=30, seed=0)
        np.savetxt(self.data_path, data, delimiter=",", fmt="%.6g")
        self.truth = truth

    def test_metrics_command(self):
        self.assertEqual(main(["metrics"]), 0)

    def test_simulate_command(self):
        out = os.path.join(self.tmp, "sim.csv")
        truth_out = os.path.join(self.tmp, "truth.txt")
        rc = main(["simulate", "--modules", "10", "10", "--samples", "20",
                   "--seed", "0", "--out", out, "--truth-out", truth_out])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(out))
        self.assertTrue(os.path.exists(truth_out))
        loaded = np.loadtxt(out, delimiter=",")
        self.assertEqual(loaded.shape, (20, 20))

    def test_cluster_command_writes_labels(self):
        out = os.path.join(self.tmp, "labels.txt")
        rc = main(["cluster", "--data", self.data_path, "--targets", "silhouette",
                   "--g-max", "4", "--generations", "20", "--population", "40",
                   "--seed", "0", "--quiet", "--out", out])
        self.assertEqual(rc, 0)
        labels = np.loadtxt(out, dtype=int)
        self.assertEqual(labels.size, 30)
        self.assertGreaterEqual(len(set(labels.tolist())), 2)

    def test_cluster_nsga2_pareto_out(self):
        out = os.path.join(self.tmp, "labels2.txt")
        pareto = os.path.join(self.tmp, "pareto.txt")
        rc = main(["cluster", "--data", self.data_path,
                   "--targets", "loglik", "davies_bouldin", "--mode", "nsga2",
                   "--g-max", "4", "--generations", "20", "--population", "40",
                   "--seed", "0", "--quiet", "--out", out, "--pareto-out", pareto])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(pareto))


if __name__ == "__main__":
    unittest.main()
