import json
import tempfile
import unittest
from pathlib import Path

from evaluation.beir_metrics import aggregate_metrics, query_metrics
from rag.scifact import (
    document_text,
    exact_cosine_top_k,
    load_scifact,
)


class BeirMetricTests(unittest.TestCase):
    def test_binary_ranked_metrics(self):
        metrics = query_metrics(
            ["miss", "relevant-a", "relevant-b"],
            {"relevant-a": 1, "relevant-b": 1},
            cutoff=3,
        )
        self.assertAlmostEqual(metrics["mrr_at_3"], 0.5)
        self.assertAlmostEqual(metrics["recall_at_3"], 1.0)
        self.assertAlmostEqual(metrics["map_at_3"], (0.5 + 2 / 3) / 2)

    def test_graded_ndcg_and_aggregation(self):
        ideal = query_metrics(["high", "low"], {"high": 2, "low": 1}, cutoff=2)
        reversed_row = query_metrics(
            ["low", "high"], {"high": 2, "low": 1}, cutoff=2
        )
        self.assertEqual(ideal["ndcg_at_2"], 1.0)
        self.assertLess(reversed_row["ndcg_at_2"], 1.0)
        aggregate = aggregate_metrics(
            {"q1": ["high", "low"], "q2": ["low", "high"]},
            {"q1": {"high": 2, "low": 1}, "q2": {"high": 2, "low": 1}},
            cutoffs=(2,),
        )
        self.assertAlmostEqual(
            aggregate["ndcg_at_2"],
            round((ideal["ndcg_at_2"] + reversed_row["ndcg_at_2"]) / 2, 6),
        )

    def test_rejects_invalid_cutoff(self):
        with self.assertRaises(ValueError):
            query_metrics([], {}, cutoff=0)


class ExactCosineTests(unittest.TestCase):
    def test_exact_ranking_and_document_id_tie_break(self):
        rankings = exact_cosine_top_k(
            [[1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
            ["z", "b", "a"],
            top_k=3,
        )
        self.assertEqual(
            [row["doc_id"] for row in rankings[0]],
            ["a", "b", "z"],
        )
        self.assertEqual(rankings[0][0]["dense_score"], 1.0)
        self.assertEqual(rankings[0][2]["dense_score"], 0.0)

    def test_exact_ranking_validates_shape_and_cutoff(self):
        with self.assertRaises(ValueError):
            exact_cosine_top_k([[1.0]], [[1.0]], ["d"], top_k=2)
        with self.assertRaises(ValueError):
            exact_cosine_top_k([[1.0, 0.0]], [[1.0]], ["d"], top_k=1)


class SciFactLoaderTests(unittest.TestCase):
    def test_loads_only_test_queries_and_qrels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "qrels").mkdir()
            corpus = [
                {"_id": "d1", "title": "Title", "text": "Abstract"},
                {"_id": "d2", "title": "", "text": "Other"},
            ]
            queries = [
                {"_id": "q-train", "text": "train"},
                {"_id": "q-test", "text": "test"},
            ]
            (root / "corpus.jsonl").write_text(
                "\n".join(json.dumps(row) for row in corpus), encoding="utf-8"
            )
            (root / "queries.jsonl").write_text(
                "\n".join(json.dumps(row) for row in queries), encoding="utf-8"
            )
            (root / "qrels" / "test.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq-test\td1\t1\n",
                encoding="utf-8",
            )
            loaded_corpus, loaded_queries, qrels = load_scifact(root)
            self.assertEqual(set(loaded_corpus), {"d1", "d2"})
            self.assertEqual(loaded_queries, {"q-test": "test"})
            self.assertEqual(qrels, {"q-test": {"d1": 1}})
            self.assertEqual(document_text(loaded_corpus["d1"]), "Title\nAbstract")


if __name__ == "__main__":
    unittest.main()
