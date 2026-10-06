"""The README and report state the thresholds and the corpus limits that the
code and the corpus actually have."""

import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hallucination_claim_grounding_eval import checker  # noqa: E402

README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
REPORT = (PROJECT_ROOT / "reports" / "evaluation_report.md").read_text(encoding="utf-8")
EM_DASH = chr(0x2014)


class TestDocumentedThresholds(unittest.TestCase):
    def test_readme_names_both_thresholds_and_their_current_values(self):
        self.assertIn("SUPPORT_OVERLAP_THRESHOLD", README)
        self.assertIn("CONTRADICTION_OVERLAP_THRESHOLD", README)
        self.assertIn(f"({checker.SUPPORT_OVERLAP_THRESHOLD})", README)
        self.assertIn(f"({checker.CONTRADICTION_OVERLAP_THRESHOLD})", README)

    def test_readme_and_report_describe_the_contradiction_gate(self):
        self.assertRegex(README, r"only\s+fire\s+when\s+overlap\s+is\s+at\s+least\s+0\.5")
        self.assertRegex(REPORT, r"at\s+least\s+half\s+of\s+its\s+content\s+words\s+\(0\.5\)")


class TestCorpusLimitsAreStated(unittest.TestCase):
    def test_perfect_rows_are_not_presented_as_robustness(self):
        for text in (README, REPORT):
            flat = " ".join(text.split())
            self.assertRegex(flat, r"not evidence (?:that the checker is robust|of general robustness)")
            for kind in ("role swaps", "antonym", "single-content-word|one-content-word|one content word|one-word"):
                self.assertRegex(flat, kind)

    def test_the_false_supported_direction_is_not_called_the_costlier_one(self):
        self.assertNotIn("more costly", REPORT)
        self.assertNotIn("everything else the rules were built for works", README)

    def test_unhandled_input_formats_are_listed(self):
        for text in (README, REPORT):
            flat = " ".join(text.split())
            self.assertIn("two", flat)
            self.assertIn("10-14", flat)
            self.assertIn("minus 14", flat)


class TestCommaLimitsAreStated(unittest.TestCase):
    def test_comma_normalization_scope_and_its_limits_match_the_code(self):
        self.assertEqual(checker.extract_numbers("3,141"), {"3141"})
        self.assertEqual(checker.extract_numbers("1,234,56"), {"1234", "56"})
        for text in (README, REPORT):
            flat = " ".join(text.split()).lower()
            self.assertIn("decimal-comma", flat)
            self.assertIn("malformed grouping", flat)


class TestStaleWordingIsGone(unittest.TestCase):
    def test_report_does_not_point_at_a_date_that_results_json_lacks(self):
        self.assertNotIn("Run date: see", REPORT)

    def test_readme_does_not_claim_a_unit_test_module_for_every_module(self):
        self.assertNotIn("unit tests for each module", README)


class TestPublicProseHygiene(unittest.TestCase):
    def test_no_em_dash(self):
        for text in (README, REPORT):
            self.assertNotIn(EM_DASH, text)

    def test_readme_has_no_dangling_reference_to_a_missing_section(self):
        self.assertNotIn('"The two 1.000 rows"', README)

    def test_no_spaced_double_hyphen_used_as_a_dash(self):
        for text in (README, REPORT):
            self.assertIsNone(re.search(r"\S -- \S", text))

    def test_no_first_person_process_narration(self):
        for text in (README, REPORT):
            for phrase in ("I expected going in", "I didn't try", "What I'd add"):
                self.assertNotIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
