import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hallucination_claim_grounding_eval import segmenter  # noqa: E402


class TestSplitSentences(unittest.TestCase):
    def test_single_sentence(self):
        self.assertEqual(
            segmenter.split_sentences("Meridian shipped 4200 units."),
            ["Meridian shipped 4200 units."],
        )

    def test_multiple_sentences(self):
        text = "The trial ran for 26 weeks. It enrolled 340 patients. No serious side effects were reported."
        self.assertEqual(
            segmenter.split_sentences(text),
            [
                "The trial ran for 26 weeks.",
                "It enrolled 340 patients.",
                "No serious side effects were reported.",
            ],
        )

    def test_empty_text(self):
        self.assertEqual(segmenter.split_sentences(""), [])
        self.assertEqual(segmenter.split_sentences("   "), [])


class TestSegment(unittest.TestCase):
    def test_no_clause_connector_stays_one_claim(self):
        self.assertEqual(
            segmenter.segment("The trial ran for 26 weeks."),
            ["The trial ran for 26 weeks"],
        )

    def test_splits_on_comma_and(self):
        text = (
            "Meridian Robotics shipped 4200 warehouse units in the third quarter, "
            "and the company's revenue for the quarter was 25 million dollars."
        )
        self.assertEqual(
            segmenter.segment(text),
            [
                "Meridian Robotics shipped 4200 warehouse units in the third quarter",
                "the company's revenue for the quarter was 25 million dollars",
            ],
        )

    def test_splits_on_semicolon(self):
        text = "The Xantrol trial enrolled 340 patients across 12 clinics; the drug increased blood pressure in some participants."
        self.assertEqual(
            segmenter.segment(text),
            [
                "The Xantrol trial enrolled 340 patients across 12 clinics",
                "the drug increased blood pressure in some participants",
            ],
        )

    def test_splits_on_but(self):
        text = "Cedar Valley Library closed its main branch for renovation on March 3, but it also opened a temporary pop-up branch downtown."
        self.assertEqual(
            segmenter.segment(text),
            [
                "Cedar Valley Library closed its main branch for renovation on March 3",
                "it also opened a temporary pop-up branch downtown",
            ],
        )

    def test_multi_sentence_multi_clause(self):
        text = "It rained all day. The river rose two feet, and several roads flooded."
        self.assertEqual(
            segmenter.segment(text),
            [
                "It rained all day",
                "The river rose two feet",
                "several roads flooded",
            ],
        )

    def test_does_not_split_ordinary_and_within_a_clause(self):
        # A limitation stated in the README: "and" only triggers a split
        # when preceded by a comma, so a bare mid-clause "and" is not
        # treated as a clause boundary.
        self.assertEqual(
            segmenter.segment("Meridian shipped units and reported strong revenue."),
            ["Meridian shipped units and reported strong revenue"],
        )


if __name__ == "__main__":
    unittest.main()
