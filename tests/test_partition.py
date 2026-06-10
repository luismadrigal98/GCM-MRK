import unittest

import numpy as np

from gcmrk import partition


class TestConsolidate(unittest.TestCase):
    def test_basic_consolidation(self):
        self.assertEqual(partition.consolidate_labels([9, 9, 2, 2, 4]), [3, 3, 1, 1, 2])

    def test_preserves_unassigned_zero(self):
        self.assertEqual(partition.consolidate_labels([6, 2, 0, 2, 1]), [3, 2, 0, 2, 1])

    def test_non_integer_raises(self):
        with self.assertRaises(TypeError):
            partition.consolidate_labels(["a", "b"])


class TestRandomPartition(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_length_and_range(self):
        p = partition.random_partition(20, 5, True, self.rng)
        self.assertEqual(p.size, 20)
        self.assertGreaterEqual(p.min(), 1)  # all assigned
        self.assertLessEqual(np.unique(p).size, 5)

    def test_at_least_two_clusters(self):
        for _ in range(20):
            p = partition.random_partition(10, 4, True, self.rng)
            self.assertGreaterEqual(np.unique(p[p != 0]).size, 2)

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            partition.random_partition(0, 5, True, self.rng)
        with self.assertRaises(ValueError):
            partition.random_partition(10, 0, True, self.rng)


class TestRepair(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_caps_cluster_count(self):
        labels = np.arange(1, 21)  # 20 singleton clusters
        repaired = partition.repair(labels, 4, True, self.rng)
        self.assertLessEqual(np.unique(repaired).size, 4)

    def test_removes_unassigned_when_required(self):
        labels = np.zeros(10, dtype=int)
        repaired = partition.repair(labels, 5, True, self.rng)
        self.assertNotIn(0, repaired.tolist())
        self.assertGreaterEqual(np.unique(repaired).size, 2)

    def test_breaks_single_cluster(self):
        labels = np.ones(10, dtype=int)
        repaired = partition.repair(labels, 5, True, self.rng)
        self.assertGreaterEqual(np.unique(repaired).size, 2)


class TestReadPartitions(unittest.TestCase):
    def test_reads_and_filters_by_length(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            fh.write("1,1,2,2\n")
            fh.write("1,2,3\n")        # wrong length -> skipped
            fh.write("2,2,1,1\n")
            path = fh.name
        try:
            parts = partition.read_partitions(path, sep=",", expected_len=4)
            self.assertEqual(len(parts), 2)
            self.assertEqual(parts[0].tolist(), [1, 1, 2, 2])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
