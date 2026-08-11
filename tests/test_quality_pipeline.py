import unittest
from unittest.mock import patch

from rag.prompting import INSUFFICIENT_EVIDENCE, build_rag_prompt, should_abstain
from rag.reranker import _as_score_list, rerank_chunks
from rag.retriever import build_context


class PromptingTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"source": "Paper A", "chunk_idx": 7, "score": 0.7, "text": "alpha"},
            {"source": "Paper B", "chunk_idx": 2, "score": 0.6, "text": "beta"},
        ]

    def test_context_has_stable_source_ids(self):
        context = build_context(self.chunks)
        self.assertIn("[S1 | Source: Paper A | Chunk: 7 | Dense score: 0.7]", context)
        self.assertIn("[S2 | Source: Paper B | Chunk: 2 | Dense score: 0.6]", context)

    def test_prompt_requires_citations_and_exact_refusal(self):
        prompt = build_rag_prompt("question", self.chunks)
        self.assertIn("Every factual claim must cite", prompt)
        self.assertIn(INSUFFICIENT_EVIDENCE, prompt)
        self.assertIn("[S1 | Source: Paper A", prompt)

    def test_abstention_uses_best_dense_score(self):
        self.assertFalse(should_abstain(self.chunks, 0.65))
        self.assertTrue(should_abstain(self.chunks, 0.71))
        self.assertTrue(should_abstain([], 0.1))

    def test_abstention_uses_candidate_pool_confidence_after_reranking(self):
        chunks = [{
            "source": "Paper A", "text": "alpha", "score": 0.55,
            "candidate_max_dense_score": 0.72,
        }]
        self.assertFalse(should_abstain(chunks, 0.70))


class RerankerTests(unittest.TestCase):
    def test_scalar_score_is_normalized(self):
        self.assertEqual(_as_score_list(0.25), [0.25])

    @patch("rag.reranker._get_model")
    def test_reranking_orders_and_preserves_dense_scores(self, get_model):
        get_model.return_value.compute_score.return_value = [0.1, 0.9]
        chunks = [
            {"source": "A", "text": "first", "score": 0.8},
            {"source": "B", "text": "second", "score": 0.7},
        ]
        ranked = rerank_chunks("query", chunks, top_k=1)
        self.assertEqual(ranked[0]["source"], "B")
        self.assertEqual(ranked[0]["score"], 0.7)
        self.assertEqual(ranked[0]["rerank_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
