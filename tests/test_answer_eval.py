import unittest

from evaluate_answers import citation_diagnostic, term_group_recall


class AnswerDiagnosticTests(unittest.TestCase):
    def test_term_group_recall_accepts_synonyms(self):
        groups = [["copy-on-write", "copy on write"], ["shared"], ["physical block"]]
        self.assertEqual(
            term_group_recall("Copy on write uses a shared physical block.", groups), 1.0
        )

    def test_citation_ids_must_exist_in_context(self):
        result = citation_diagnostic("Claim [S1]. Other claim [S4].", source_count=3)
        self.assertEqual(result["citation_ids"], [1, 4])
        self.assertFalse(result["all_citation_ids_valid"])

    def test_no_citation_is_not_valid(self):
        result = citation_diagnostic("No source.", source_count=3)
        self.assertFalse(result["has_citation"])
        self.assertFalse(result["all_citation_ids_valid"])


if __name__ == "__main__":
    unittest.main()
