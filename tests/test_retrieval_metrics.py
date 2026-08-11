import unittest

from evaluation.retrieval_metrics import rank_of_expected, summarize_ranks


class RetrievalMetricTests(unittest.TestCase):
    def test_rank_of_expected_uses_one_based_rank(self):
        self.assertEqual(rank_of_expected(["a", "b", "c"], "b"), 2)
        self.assertIsNone(rank_of_expected(["a", "b"], "c"))

    def test_hit_rate_and_mrr_include_misses(self):
        summary = summarize_ranks([1, 2, None, 4], top_k=3)
        self.assertEqual(summary["queries"], 4)
        self.assertEqual(summary["hits"], 2)
        self.assertEqual(summary["hit_rate_at_3"], 0.5)
        self.assertEqual(summary["mean_reciprocal_rank"], 0.375)

    def test_empty_input_is_explicit(self):
        summary = summarize_ranks([], top_k=3)
        self.assertIsNone(summary["hit_rate_at_3"])
        self.assertIsNone(summary["mean_reciprocal_rank"])


if __name__ == "__main__":
    unittest.main()
