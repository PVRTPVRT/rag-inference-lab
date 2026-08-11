import unittest

from evaluation.bootstrap import holm_adjust, paired_bootstrap


class PairedBootstrapTests(unittest.TestCase):
    def test_positive_paired_difference_is_reproducible(self):
        first = paired_bootstrap(
            [0.0, 0.2, 0.4, 0.6],
            [0.1, 0.3, 0.5, 0.7],
            samples=2_000,
            seed=7,
        )
        second = paired_bootstrap(
            [0.0, 0.2, 0.4, 0.6],
            [0.1, 0.3, 0.5, 0.7],
            samples=2_000,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["mean_difference"], 0.1)
        self.assertGreater(first["ci_lower"], 0)
        self.assertEqual(first["query_wins"], 4)
        self.assertEqual(first["query_losses"], 0)

    def test_identical_pairs_have_zero_interval(self):
        result = paired_bootstrap([0.1, 0.4], [0.1, 0.4], samples=100)
        self.assertEqual(result["mean_difference"], 0.0)
        self.assertEqual(result["ci_lower"], 0.0)
        self.assertEqual(result["ci_upper"], 0.0)
        self.assertEqual(result["two_sided_bootstrap_p"], 1.0)
        self.assertEqual(result["query_ties"], 2)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([], [])
        with self.assertRaises(ValueError):
            paired_bootstrap([0.1], [0.1, 0.2])
        with self.assertRaises(ValueError):
            paired_bootstrap([0.1], [0.2], samples=0)
        with self.assertRaises(ValueError):
            paired_bootstrap([0.1], [0.2], confidence=1.0)


class HolmAdjustmentTests(unittest.TestCase):
    def test_step_down_adjustment_is_monotone(self):
        adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04})
        self.assertEqual(adjusted, {"a": 0.03, "b": 0.06, "c": 0.06})

    def test_empty_family(self):
        self.assertEqual(holm_adjust({}), {})

    def test_rejects_invalid_p_values(self):
        with self.assertRaises(ValueError):
            holm_adjust({"bad": -0.1})
        with self.assertRaises(ValueError):
            holm_adjust({"bad": 1.1})
