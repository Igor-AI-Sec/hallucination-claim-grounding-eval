"""The two overlap thresholds, and behaviour just below, at and above each.

Overlap is the share of the claim's content words found in the best
matching source clause. The test words below are invented so that none of
them is a stopword and the overlap is exact.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hallucination_claim_grounding_eval import checker  # noqa: E402


class TestThresholdValues(unittest.TestCase):
    def test_support_threshold_is_point_six(self):
        self.assertEqual(checker.SUPPORT_OVERLAP_THRESHOLD, 0.6)

    def test_contradiction_threshold_is_point_five(self):
        self.assertEqual(checker.CONTRADICTION_OVERLAP_THRESHOLD, 0.5)

    def test_contradiction_gate_is_below_the_support_threshold(self):
        self.assertLess(checker.CONTRADICTION_OVERLAP_THRESHOLD, checker.SUPPORT_OVERLAP_THRESHOLD)


class TestSupportThreshold(unittest.TestCase):
    CLAIM = "alpha beta gamma delta epsilon"

    def test_just_below_is_unsupported(self):
        # 4 of 7 content words: 0.571
        result = checker.classify_claim("alpha beta gamma delta epsilon zeta eta", "alpha beta gamma delta theta iota kappa")
        self.assertLess(result.overlap, 0.6)
        self.assertEqual(result.label, checker.UNSUPPORTED)
        self.assertEqual(result.reason, "no_matching_clause")

    def test_exactly_at_the_threshold_is_supported(self):
        result = checker.classify_claim(self.CLAIM, "alpha beta gamma zeta eta")
        self.assertEqual(result.overlap, 0.6)
        self.assertEqual(result.label, checker.SUPPORTED)
        self.assertEqual(result.reason, "lexical_overlap")

    def test_above_the_threshold_is_supported(self):
        result = checker.classify_claim(self.CLAIM, "alpha beta gamma delta zeta")
        self.assertEqual(result.overlap, 0.8)
        self.assertEqual(result.label, checker.SUPPORTED)

    def test_well_below_the_threshold_is_unsupported(self):
        result = checker.classify_claim(self.CLAIM, "alpha zeta eta theta iota")
        self.assertEqual(result.overlap, 0.2)
        self.assertEqual(result.label, checker.UNSUPPORTED)

    def test_the_corpus_paraphrase_misses_sit_below_the_support_threshold(self):
        # c05 (0.4545) and c06 (0.3333) are the documented misses; a support
        # threshold at or below 0.45 would turn c05 into a hit.
        self.assertLess(0.4545, checker.SUPPORT_OVERLAP_THRESHOLD)
        self.assertGreater(0.4545, 0.45)


class TestContradictionGate(unittest.TestCase):
    """A negation or number mismatch only counts as a contradiction when the
    claim and the source clause share at least half of the claim's words."""

    def test_negation_mismatch_just_below_the_gate_is_not_a_contradiction(self):
        # 1 of 4 content words: 0.25 (nothing between 0.25 and 0.5 is reachable with 4 words)
        result = checker.classify_claim("alpha beta gamma delta not", "alpha zeta eta theta")
        self.assertLess(result.overlap, 0.5)
        self.assertEqual(result.label, checker.UNSUPPORTED)
        self.assertEqual(result.reason, "no_matching_clause")

    def test_negation_mismatch_just_below_the_gate_with_more_words(self):
        # 4 of 9 content words: 0.444
        result = checker.classify_claim(
            "alpha beta gamma delta epsilon zeta eta theta iota not",
            "alpha beta gamma delta kappa lambda mu nu xi",
        )
        self.assertAlmostEqual(result.overlap, 4 / 9)
        self.assertEqual(result.label, checker.UNSUPPORTED)

    def test_negation_mismatch_exactly_at_the_gate_is_a_contradiction(self):
        result = checker.classify_claim("alpha beta gamma delta not", "alpha beta zeta eta")
        self.assertEqual(result.overlap, 0.5)
        self.assertEqual(result.label, checker.CONTRADICTED)
        self.assertEqual(result.reason, "negation_mismatch")

    def test_negation_mismatch_above_the_gate_is_a_contradiction(self):
        result = checker.classify_claim("alpha beta gamma delta not", "alpha beta gamma eta")
        self.assertEqual(result.overlap, 0.75)
        self.assertEqual(result.label, checker.CONTRADICTED)

    def test_the_same_overlap_without_a_negation_mismatch_is_only_unsupported(self):
        result = checker.classify_claim("alpha beta gamma delta", "alpha beta zeta eta")
        self.assertEqual(result.overlap, 0.5)
        self.assertEqual(result.label, checker.UNSUPPORTED)

    def test_numeric_mismatch_exactly_at_the_gate_is_a_contradiction(self):
        # content words: alpha beta gamma widgets; source shares alpha and widgets
        result = checker.classify_claim("alpha beta gamma 12 widgets", "alpha 10 widgets zeta eta")
        self.assertEqual(result.overlap, 0.5)
        self.assertEqual(result.label, checker.CONTRADICTED)
        self.assertEqual(result.reason, "numeric_mismatch")

    def test_numeric_mismatch_below_the_gate_is_not_a_contradiction(self):
        result = checker.classify_claim("alpha beta gamma 12 widgets", "zeta 10 widgets eta theta")
        self.assertEqual(result.overlap, 0.25)
        self.assertEqual(result.label, checker.UNSUPPORTED)
        self.assertEqual(result.reason, "no_matching_clause")

    def test_the_gate_applies_to_the_corpus_contradictions_with_room_to_spare(self):
        # every corpus contradiction has overlap of at least 0.8
        for claim, source in (
            ("the Xantrol trial enrolled 300 patients across 12 clinics", "The Xantrol trial enrolled 340 patients across 12 clinics."),
            ("Cedar Valley Library will charge late fees after the renovation is complete", "Cedar Valley Library will not charge late fees after the renovation is complete."),
        ):
            result = checker.classify_claim(claim, source)
            self.assertEqual(result.label, checker.CONTRADICTED)
            self.assertGreaterEqual(result.overlap, 0.8)


if __name__ == "__main__":
    unittest.main()
