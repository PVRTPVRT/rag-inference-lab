import unittest

from rag.hybrid import BM25Index, lexical_tokens, reciprocal_rank_fusion


class BM25Tests(unittest.TestCase):
    def test_tokenizer_preserves_technical_identifiers(self):
        self.assertEqual(
            lexical_tokens("PagedAttention KV-cache Qwen2.5"),
            ["pagedattention", "kv", "cache", "qwen2", "5"],
        )

    def test_exact_rare_term_ranks_relevant_document_first(self):
        index = BM25Index([
            "generic cache memory",
            "radixattention radix tree kv cache",
            "generic model memory cache",
        ])
        ranked = index.rank("How does RadixAttention reuse cache?", top_k=3)
        self.assertEqual(ranked[0][0], 1)
        self.assertGreater(ranked[0][1], ranked[1][1])

    def test_no_overlap_returns_no_candidates(self):
        index = BM25Index(["alpha beta", "gamma delta"])
        self.assertEqual(index.rank("unseen", top_k=2), [])

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            BM25Index(["text"], k1=0)
        with self.assertRaises(ValueError):
            BM25Index(["text"], b=1.1)
        with self.assertRaises(ValueError):
            BM25Index(["text"]).rank("text", 0)


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_document_present_in_both_rankings_is_promoted(self):
        dense = [
            {"source": "A", "chunk_idx": 1, "text": "a", "dense_score": 0.9},
            {"source": "B", "chunk_idx": 1, "text": "b", "dense_score": 0.8},
        ]
        sparse = [
            {"source": "B", "chunk_idx": 1, "text": "b", "sparse_score": 4.0},
            {"source": "C", "chunk_idx": 1, "text": "c", "sparse_score": 3.0},
        ]
        fused = reciprocal_rank_fusion({"dense": dense, "sparse": sparse})
        self.assertEqual((fused[0]["source"], fused[0]["chunk_idx"]), ("B", 1))
        self.assertEqual(fused[0]["dense_rank"], 2)
        self.assertEqual(fused[0]["sparse_rank"], 1)
        self.assertIn("dense_score", fused[0])
        self.assertIn("sparse_score", fused[0])

    def test_custom_document_key_controls_generic_tie_break(self):
        dense = [{"doc_id": "b", "dense_score": 0.9}]
        sparse = [{"doc_id": "a", "sparse_score": 2.0}]
        fused = reciprocal_rank_fusion(
            {"dense": dense, "sparse": sparse},
            key=lambda row: (row["doc_id"],),
        )
        self.assertEqual([row["doc_id"] for row in fused], ["a", "b"])
        self.assertEqual(fused[0]["rrf_score"], fused[1]["rrf_score"])

    def test_rejects_negative_rrf_constant(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion({}, rrf_k=-1)


if __name__ == "__main__":
    unittest.main()
