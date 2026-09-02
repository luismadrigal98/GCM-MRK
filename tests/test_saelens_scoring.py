"""Validate the reimplementation of the Saelens et al. (2018) module scores.

The manuscript's real-data results depend on these three numbers, and the
original implementation is Cython that we deliberately did not vendor, so the
reimplementation is pinned here against cases whose values can be worked out by
hand.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper" / "experiments"))
import saelens_benchmark as SB  # noqa: E402


GENES = np.array([f"g{i}" for i in range(10)])


class TestJaccardMatrix(unittest.TestCase):
    def test_identical_sets_are_one(self):
        A = SB.memberships([["g0", "g1"]], GENES)
        self.assertAlmostEqual(SB.jaccard_matrix(A, A)[0, 0], 1.0)

    def test_disjoint_sets_are_zero(self):
        A = SB.memberships([["g0", "g1"]], GENES)
        B = SB.memberships([["g2", "g3"]], GENES)
        self.assertAlmostEqual(SB.jaccard_matrix(A, B)[0, 0], 0.0)

    def test_half_overlap(self):
        # |A n B| = 1, |A u B| = 3  ->  1/3
        A = SB.memberships([["g0", "g1"]], GENES)
        B = SB.memberships([["g1", "g2"]], GENES)
        self.assertAlmostEqual(SB.jaccard_matrix(A, B)[0, 0], 1 / 3)

    def test_empty_module_does_not_divide_by_zero(self):
        A = SB.memberships([[]], GENES)
        B = SB.memberships([["g0"]], GENES)
        self.assertTrue(np.isfinite(SB.jaccard_matrix(A, B)).all())

    def test_genes_outside_the_universe_are_ignored(self):
        A = SB.memberships([["g0", "not_present"]], GENES)
        self.assertEqual(int(A.sum()), 1)


class TestScoreModules(unittest.TestCase):
    def test_perfect_recovery(self):
        mods = [["g0", "g1"], ["g2", "g3"]]
        s = SB.score_modules(mods, mods, GENES)
        self.assertAlmostEqual(s["recovery"], 1.0)
        self.assertAlmostEqual(s["relevance"], 1.0)
        self.assertAlmostEqual(s["f1rr"], 1.0)

    def test_split_module_halves_recovery(self):
        known = [["g0", "g1", "g2", "g3"]]
        pred = [["g0", "g1"], ["g2", "g3"]]
        s = SB.score_modules(known, pred, GENES)
        # best Jaccard for the known module is 2/4 = 0.5
        self.assertAlmostEqual(s["recovery"], 0.5)
        # each predicted module's best Jaccard against the known one is also 0.5
        self.assertAlmostEqual(s["relevance"], 0.5)

    def test_recovery_and_relevance_are_asymmetric(self):
        known = [["g0", "g1"], ["g2", "g3"]]
        pred = [["g0", "g1"]]                     # finds one module, misses one
        s = SB.score_modules(known, pred, GENES)
        self.assertAlmostEqual(s["recovery"], 0.5)   # (1.0 + 0.0) / 2
        self.assertAlmostEqual(s["relevance"], 1.0)  # its only module is perfect

    def test_f1_is_the_harmonic_mean(self):
        known = [["g0", "g1"], ["g2", "g3"]]
        pred = [["g0", "g1"]]
        s = SB.score_modules(known, pred, GENES)
        expected = 2 * 0.5 * 1.0 / (0.5 + 1.0)
        self.assertAlmostEqual(s["f1rr"], expected)

    def test_f1_is_zero_when_either_side_is_zero(self):
        s = SB.score_modules([["g0"]], [["g5"]], GENES)
        self.assertAlmostEqual(s["f1rr"], 0.0)

    def test_empty_prediction_scores_zero(self):
        s = SB.score_modules([["g0", "g1"]], [], GENES)
        self.assertEqual(s["f1rr"], 0.0)

    def test_one_giant_module_trades_relevance_for_recovery(self):
        known = [["g0", "g1"], ["g2", "g3"]]
        giant = [[f"g{i}" for i in range(10)]]
        s = SB.score_modules(known, giant, GENES)
        # each known module: |int|=2, |union|=10 -> 0.2
        self.assertAlmostEqual(s["recovery"], 0.2)
        self.assertAlmostEqual(s["relevance"], 0.2)
        self.assertLess(s["f1rr"], 0.5)


class TestLabelsToModules(unittest.TestCase):
    def test_unassigned_label_is_dropped(self):
        labels = np.array([0, 0, 1, 1, 2])
        mods = SB.labels_to_modules(labels, GENES[:5])
        self.assertEqual(len(mods), 2)
        self.assertNotIn("g0", [g for m in mods for g in m])

    def test_modules_partition_the_assigned_genes(self):
        labels = np.array([1, 1, 2, 2, 3])
        mods = SB.labels_to_modules(labels, GENES[:5])
        flat = [g for m in mods for g in m]
        self.assertEqual(sorted(flat), sorted(GENES[:5].tolist()))
