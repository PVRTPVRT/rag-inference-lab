import json
import tempfile
import unittest
from pathlib import Path

from rag.arguana import load_arguana, remove_query_self_matches


class ArguAnaLoaderTests(unittest.TestCase):
    def test_loads_test_qrels_and_preserves_query_document_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "qrels").mkdir()
            corpus = [
                {"_id": "q1", "title": "", "text": "query argument"},
                {"_id": "d1", "title": "", "text": "counterargument"},
            ]
            queries = [
                {"_id": "q1", "text": "query argument"},
                {"_id": "unused", "text": "not in test qrels"},
            ]
            (root / "corpus.jsonl").write_text(
                "\n".join(json.dumps(row) for row in corpus), encoding="utf-8"
            )
            (root / "queries.jsonl").write_text(
                "\n".join(json.dumps(row) for row in queries), encoding="utf-8"
            )
            (root / "qrels" / "test.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
                encoding="utf-8",
            )
            loaded_corpus, loaded_queries, qrels = load_arguana(root)
            self.assertEqual(set(loaded_corpus), {"q1", "d1"})
            self.assertEqual(loaded_queries, {"q1": "query argument"})
            self.assertEqual(qrels, {"q1": {"d1": 1}})


class SelfMatchTests(unittest.TestCase):
    def test_removes_query_id_and_retains_top_k(self):
        filtered = remove_query_self_matches(
            {
                "q1": [
                    {"doc_id": "q1", "dense_score": 1.0},
                    {"doc_id": "d1", "dense_score": 0.8},
                    {"doc_id": "d2", "dense_score": 0.7},
                ]
            },
            top_k=2,
        )
        self.assertEqual(
            [row["doc_id"] for row in filtered["q1"]], ["d1", "d2"]
        )

    def test_does_not_remove_other_query_ids(self):
        filtered = remove_query_self_matches(
            {"q1": [{"doc_id": "q2"}, {"doc_id": "d1"}]}, top_k=2
        )
        self.assertEqual(len(filtered["q1"]), 2)
