import unittest

from rag.entailment import (
    NLIResult, extract_citation_ids, extract_claims, focus_premise,
    verify_cited_answer,
)


class FakeVerifier:
    def predict(self, premise, hypothesis):
        if "unsupported" in hypothesis:
            return NLIResult(
                "neutral",
                {"contradiction": 0.05, "entailment": 0.05, "neutral": 0.9},
                1.0,
            )
        if "opposite" in hypothesis:
            return NLIResult(
                "contradiction",
                {"contradiction": 0.9, "entailment": 0.05, "neutral": 0.05},
                1.0,
            )
        return NLIResult(
            "entailment",
            {"contradiction": 0.02, "entailment": 0.93, "neutral": 0.05},
            1.0,
        )


class ClaimExtractionTests(unittest.TestCase):
    def test_extracts_bullets_sentences_and_citations(self):
        answer = "- First claim [S1]. Second claim [S2][S3].\n2. Final claim [S1]."
        self.assertEqual(
            extract_claims(answer),
            [
                {"claim": "First claim.", "citation_ids": [1]},
                {"claim": "Second claim.", "citation_ids": [2, 3]},
                {"claim": "Final claim.", "citation_ids": [1]},
            ],
        )

    def test_accepts_adjacent_and_comma_separated_citation_blocks(self):
        self.assertEqual(extract_citation_ids("A [S1][S2, S3]."), [1, 2, 3])
        self.assertEqual(extract_claims("A claim [S2, S3].")[0]["citation_ids"], [2, 3])


class PremiseFocusingTests(unittest.TestCase):
    def test_keeps_relevant_sentences_in_document_order(self):
        premise = (
            "Scheduling details are unrelated. "
            "PagedAttention stores the KV cache in non-contiguous blocks. "
            "Measurements are reported later. "
            "This reduces memory waste caused by fragmentation."
        )
        focused = focus_premise(
            premise,
            "PagedAttention reduces KV cache fragmentation with non-contiguous blocks.",
            max_sentences=2,
        )
        self.assertEqual(
            focused,
            "PagedAttention stores the KV cache in non-contiguous blocks. "
            "This reduces memory waste caused by fragmentation.",
        )

    def test_zero_disables_focusing(self):
        premise = "First sentence. Second sentence."
        self.assertEqual(focus_premise(premise, "Second", 0), premise)

    def test_no_lexical_overlap_keeps_full_premise(self):
        premise = "First sentence. Second sentence. Third sentence. Fourth sentence."
        self.assertEqual(focus_premise(premise, "Unrelated hypothesis", 2), premise)


class AnswerVerificationTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"source": "A", "text": "supporting premise"},
            {"source": "B", "text": "other premise"},
        ]

    def test_reports_supported_uncited_and_invalid_claims(self):
        answer = (
            "Supported statement [S1].\n"
            "An unsupported statement [S2].\n"
            "Uncited statement.\n"
            "Invalid citation [S9]."
        )
        result = verify_cited_answer(answer, self.chunks, verifier=FakeVerifier())
        summary = result["summary"]
        self.assertEqual(summary["claim_count"], 4)
        self.assertEqual(summary["cited_claim_count"], 3)
        self.assertEqual(summary["supported_claim_count"], 1)
        self.assertEqual(summary["invalid_citation_count"], 1)

    def test_detects_contradiction(self):
        result = verify_cited_answer(
            "The opposite is true [S1].", self.chunks, verifier=FakeVerifier()
        )
        self.assertEqual(result["summary"]["contradicted_claim_count"], 1)


if __name__ == "__main__":
    unittest.main()
