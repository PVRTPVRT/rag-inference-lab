import unittest

from backends.metrics import (
    client_tokens_per_second,
    nanoseconds_to_milliseconds,
    percentile,
    summarize,
    tokens_per_second,
)
from rag.chunking import chunk_text, chunk_token_ids
from rag.parent_child import build_parent_child_records


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in token_ids)


class MetricsTests(unittest.TestCase):
    def test_ollama_server_rate_uses_nanoseconds(self):
        self.assertEqual(tokens_per_second(50, 2_000_000_000), 25.0)
        self.assertEqual(nanoseconds_to_milliseconds(2_500_000), 2.5)

    def test_client_rate_excludes_time_to_first_token(self):
        self.assertEqual(client_tokens_per_second(90, 4000.0, 1000.0), 30.0)
        self.assertIsNone(client_tokens_per_second(None, 4000.0, 1000.0))

    def test_percentiles_and_summary(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)
        result = summarize([1, 2, None, 3])
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["p95"], 2.9)


class ChunkingTests(unittest.TestCase):
    def test_token_overlap(self):
        self.assertEqual(
            chunk_token_ids(list(range(10)), size=4, overlap=1),
            [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]],
        )

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_token_ids([1, 2], size=2, overlap=2)

    def test_text_chunking_uses_tokenizer(self):
        chunks = chunk_text("a b c d e", FakeTokenizer(), size=3, overlap=1)
        self.assertEqual(chunks, ["0 1 2", "2 3 4"])

    def test_parent_child_records_preserve_parent_mapping(self):
        records = build_parent_child_records(
            "a b c d e f g h i j",
            FakeTokenizer(),
            parent_size=6,
            parent_overlap=2,
            child_size=3,
            child_overlap=1,
        )
        parent_indices = [row["parent_chunk_idx"] for row in records]
        self.assertEqual(parent_indices, [0, 0, 0, 1, 1, 1])
        self.assertEqual(records[0]["parent_text"], "0 1 2 3 4 5")
        self.assertEqual(records[3]["child_text"], "0 1 2")


if __name__ == "__main__":
    unittest.main()
