"""Format-only input normalization, and the input formats that are not handled.

Curly apostrophes and thousands separators are normalized before tokenizing
and number extraction. Number words, hyphenated ranges and "minus" are not,
and the second class below pins that behaviour so the README's limitation
note stays true.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hallucination_claim_grounding_eval import checker  # noqa: E402

CURLY = "’"
LEFT_CURLY = "‘"


class TestCurlyApostrophes(unittest.TestCase):
    def test_normalize_text_straightens_both_curly_forms(self):
        self.assertEqual(checker.normalize_text(f"doesn{CURLY}t and {LEFT_CURLY}x{CURLY}"), "doesn't and 'x'")

    def test_tokenize_gives_the_same_tokens_for_both_apostrophes(self):
        self.assertEqual(checker.tokenize(f"The company doesn{CURLY}t charge fees"), checker.tokenize("The company doesn't charge fees"))
        self.assertIn("doesn't", checker.tokenize(f"doesn{CURLY}t"))

    def test_has_negation_sees_a_curly_contraction(self):
        for word in ("doesn", "isn", "wasn", "won", "can", "didn", "hasn"):
            self.assertTrue(checker.has_negation(f"It {word}{CURLY}t happen"), word)
            self.assertEqual(checker.has_negation(f"It {word}{CURLY}t happen"), checker.has_negation(f"It {word}'t happen"))

    def test_content_words_are_identical_for_both_apostrophes(self):
        self.assertEqual(
            checker.content_words(f"The company doesn{CURLY}t charge fees"),
            checker.content_words("The company doesn't charge fees"),
        )

    def test_curly_negated_claim_contradicts_an_affirmative_source(self):
        straight = checker.classify_claim("The company doesn't charge fees", "The company charges fees.")
        curly = checker.classify_claim(f"The company doesn{CURLY}t charge fees", "The company charges fees.")
        self.assertEqual(curly.label, checker.CONTRADICTED)
        self.assertEqual(curly.reason, "negation_mismatch")
        self.assertEqual((curly.label, curly.reason, curly.overlap), (straight.label, straight.reason, straight.overlap))

    def test_curly_negated_source_contradicts_an_affirmative_claim(self):
        result = checker.classify_claim("The company charges fees", f"The company doesn{CURLY}t charge fees.")
        self.assertEqual(result.label, checker.CONTRADICTED)
        self.assertEqual(result.reason, "negation_mismatch")

    def test_matching_negations_in_mixed_apostrophes_are_supported(self):
        result = checker.classify_claim(f"The company doesn{CURLY}t charge fees", "The company doesn't charge fees.")
        self.assertEqual(result.label, checker.SUPPORTED)

    def test_a_possessive_with_a_curly_apostrophe_matches_the_straight_one(self):
        result = checker.classify_claim(f"Northfield{CURLY}s station recorded a record low", "Northfield's station recorded a record low.")
        self.assertEqual(result.label, checker.SUPPORTED)
        self.assertEqual(result.overlap, 1.0)


class TestThousandsSeparators(unittest.TestCase):
    def test_extract_numbers_removes_separators(self):
        self.assertEqual(checker.extract_numbers("shipped 4,200 units"), {"4200"})
        self.assertEqual(checker.extract_numbers("shipped 4200 units"), {"4200"})
        self.assertEqual(checker.extract_numbers("a population of 1,234,567"), {"1234567"})
        self.assertEqual(checker.extract_numbers("a cost of 4,200.50 dollars"), {"4200.50"})

    def test_extract_number_entities_removes_separators(self):
        self.assertEqual(checker.extract_number_entities("shipped 4,200 units"), {("4200", "unit")})
        self.assertEqual(checker.extract_number_entities("shipped 4,200 units"), checker.extract_number_entities("shipped 4200 units"))

    def test_separators_do_not_change_content_words(self):
        self.assertEqual(checker.content_words("shipped 4,200 units"), checker.content_words("shipped 4200 units"))

    def test_other_commas_between_numbers_are_left_alone(self):
        self.assertEqual(checker.extract_numbers("12, 200 and 7"), {"12", "200", "7"})
        self.assertEqual(checker.extract_numbers("values 1,23 and 1,2345"), {"1", "23", "1", "2345"})
        self.assertEqual(checker.extract_numbers("ids 3,4,5"), {"3", "4", "5"})

    def test_a_negative_number_keeps_its_sign_through_normalization(self):
        self.assertEqual(checker.extract_numbers("a change of negative 4,200 units"), {"-4200"})
        self.assertEqual(checker.extract_numbers("a change of -4,200 units"), {"-4200"})

    def test_separated_claim_is_supported_by_unseparated_source(self):
        result = checker.classify_claim("Meridian shipped 4,200 warehouse units", "Meridian shipped 4200 warehouse units.")
        self.assertEqual(result.label, checker.SUPPORTED)
        self.assertEqual(result.reason, "lexical_overlap")

    def test_unseparated_claim_is_supported_by_separated_source(self):
        result = checker.classify_claim("Meridian shipped 4200 warehouse units", "Meridian shipped 4,200 warehouse units.")
        self.assertEqual(result.label, checker.SUPPORTED)
        self.assertEqual(result.reason, "lexical_overlap")

    def test_both_separated_and_equal_is_supported(self):
        result = checker.classify_claim("Meridian shipped 4,200 warehouse units", "Meridian shipped 4,200 warehouse units.")
        self.assertEqual(result.label, checker.SUPPORTED)

    def test_a_real_difference_is_still_a_contradiction_in_every_format_mix(self):
        for claim_number, source_number in (("4,300", "4,200"), ("4300", "4,200"), ("4,300", "4200")):
            result = checker.classify_claim(
                f"Meridian shipped {claim_number} warehouse units",
                f"Meridian shipped {source_number} warehouse units.",
            )
            self.assertEqual(result.label, checker.CONTRADICTED, (claim_number, source_number))
            self.assertEqual(result.reason, "numeric_mismatch")

    def test_evidence_reports_the_normalized_numbers(self):
        result = checker.classify_claim("Meridian shipped 4,300 warehouse units", "Meridian shipped 4,200 warehouse units.")
        self.assertEqual(result.evidence["claim_numbers"], ["4300"])
        self.assertEqual(result.evidence["source_numbers"], ["4200"])


class TestUnhandledInputFormatsStayDocumented(unittest.TestCase):
    """Behaviour the README lists as a limitation. If one of these starts
    passing differently, update the limitation note with it."""

    def test_number_words_are_not_read_as_numbers(self):
        self.assertEqual(checker.extract_numbers("two study rooms"), set())
        result = checker.classify_claim("The renovation will add three study rooms", "The renovation will add two study rooms.")
        self.assertEqual(result.label, checker.SUPPORTED)

    def test_a_hyphenated_range_reads_its_second_number_as_negative(self):
        self.assertEqual(checker.extract_numbers("lasted 10-14 days"), {"10", "-14"})
        self.assertEqual(checker.extract_numbers("lasted 10 to 14 days"), {"10", "14"})

    def test_minus_is_not_read_as_a_sign(self):
        self.assertEqual(checker.extract_numbers("was minus 14 degrees"), {"14"})
        self.assertEqual(checker.extract_numbers("was negative 14 degrees"), {"-14"})


if __name__ == "__main__":
    unittest.main()
