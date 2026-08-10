import unittest
from unittest.mock import patch

import rag.retriever as retriever
from rag.retriever import build_context, retrieve


class FakeVector(list):
    def tolist(self):
        return list(self)


class FakeModel:
    def encode(self, texts, batch_size, max_length):
        return {"dense_vecs": [FakeVector([0.1, 0.2])]}


class FakeCollection:
    def query(self, **kwargs):
        return {
            "documents": [["dense a", "dense b"]],
            "metadatas": [[
                {"source": "A", "chunk_idx": 1},
                {"source": "B", "chunk_idx": 1},
            ]],
            "distances": [[0.1, 0.2]],
        }

class FakeParentChildCollection:
    def query(self, **kwargs):
        return {
            "documents": [["child a1", "child a2", "child b1"]],
            "metadatas": [[
                {
                    "source": "A", "parent_chunk_idx": 4,
                    "child_chunk_idx": 1, "parent_text": "parent a",
                },
                {
                    "source": "A", "parent_chunk_idx": 4,
                    "child_chunk_idx": 2, "parent_text": "parent a",
                },
                {
                    "source": "B", "parent_chunk_idx": 2,
                    "child_chunk_idx": 0, "parent_text": "parent b",
                },
            ]],
            "distances": [[0.1, 0.15, 0.2]],
        }



class RetrievalModeTests(unittest.TestCase):
    @patch("rag.retriever._get_model")
    @patch("rag.retriever._sparse_retrieve")
    def test_sparse_mode_does_not_load_embedding_model(self, sparse, get_model):
        sparse.return_value = [{
            "source": "A", "chunk_idx": 1, "text": "a",
            "score": 2.0, "sparse_score": 2.0,
        }]
        chunks = retrieve("query", retrieval="sparse", top_k=1)
        get_model.assert_not_called()
        sparse.assert_called_once_with("query", 1)
        self.assertEqual(chunks[0]["sparse_score"], 2.0)

    @patch("rag.retriever.chromadb.PersistentClient")
    def test_collection_loader_returns_and_caches_collection(self, client):
        previous = retriever._col
        sentinel = object()
        client.return_value.get_collection.return_value = sentinel
        try:
            retriever._col = None
            self.assertIs(retriever._get_collection(), sentinel)
            self.assertIs(retriever._get_collection(), sentinel)
            client.assert_called_once_with(path=retriever.CHROMA_PATH)
            client.return_value.get_collection.assert_called_once_with(
                retriever.COLLECTION_NAME
            )
        finally:
            retriever._col = previous

    @patch("rag.retriever._sparse_retrieve")
    @patch("rag.retriever._get_collection")
    @patch("rag.retriever._get_model")
    def test_hybrid_fuses_dense_and_sparse_candidates(
        self, get_model, get_collection, sparse
    ):
        get_model.return_value = FakeModel()
        get_collection.return_value = FakeCollection()
        sparse.return_value = [
            {
                "source": "B", "chunk_idx": 1, "text": "dense b",
                "score": 3.0, "sparse_score": 3.0,
            },
            {
                "source": "C", "chunk_idx": 1, "text": "sparse c",
                "score": 2.0, "sparse_score": 2.0,
            },
        ]
        chunks = retrieve(
            "query", retrieval="hybrid", top_k=2, candidate_k=2, rrf_k=60
        )
        self.assertEqual((chunks[0]["source"], chunks[0]["chunk_idx"]), ("B", 1))
        self.assertEqual(chunks[0]["dense_rank"], 2)
        self.assertEqual(chunks[0]["sparse_rank"], 1)
        self.assertEqual(chunks[0]["score"], chunks[0]["rrf_score"])
        self.assertEqual(chunks[0]["candidate_max_dense_score"], 0.9091)

    @patch("rag.retriever._get_parent_child_collection")
    @patch("rag.retriever._get_model")
    def test_parent_child_deduplicates_parents_and_returns_parent_text(
        self, get_model, get_collection
    ):
        get_model.return_value = FakeModel()
        get_collection.return_value = FakeParentChildCollection()
        chunks = retrieve(
            "query", retrieval="parent_child", top_k=2, candidate_k=3
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], "parent a")
        self.assertEqual(chunks[0]["chunk_idx"], 4)
        self.assertEqual(chunks[0]["child_chunk_idx"], 1)
        self.assertEqual(chunks[0]["child_rank"], 1)
        self.assertEqual(chunks[1]["text"], "parent b")
        self.assertEqual(chunks[1]["child_rank"], 3)
        self.assertEqual(chunks[1]["candidate_max_dense_score"], 0.9091)

    def test_rejects_incompatible_or_unknown_modes(self):
        with self.assertRaises(ValueError):
            retrieve("query", retrieval="unknown")
        with self.assertRaises(ValueError):
            retrieve("query", retrieval="hybrid", rerank=True)

    def test_context_names_hybrid_component_scores(self):
        context = build_context([{
            "source": "A",
            "chunk_idx": 2,
            "text": "evidence",
            "score": 0.03,
            "dense_score": 0.8,
            "sparse_score": 4.2,
            "rrf_score": 0.03,
        }])
        self.assertIn("Dense score: 0.8", context)
        self.assertIn("BM25 score: 4.2", context)
        self.assertIn("RRF score: 0.03", context)


if __name__ == "__main__":
    unittest.main()
