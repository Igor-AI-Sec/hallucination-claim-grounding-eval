"""End-to-end checks: the real evaluation over fixtures/corpus.json.

The expected numbers are written out here as plain literals and also
recomputed from the raw fixture file, so a change to the fixtures, the
classifier or the runner shows up as a failure instead of being absorbed by
a helper that produced the number it is checked against.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hallucination_claim_grounding_eval import checker, run_evaluation  # noqa: E402

CORPUS_PATH = PROJECT_ROOT / "fixtures" / "corpus.json"
RESULTS_PATH = PROJECT_ROOT / "results" / "results.json"

LABELS = ["SUPPORTED", "CONTRADICTED", "UNSUPPORTED"]


def raw_cases():
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))["cases"]


class TestHeadlineResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run_evaluation.evaluate(run_evaluation.load_corpus_store())
        cls.metrics = cls.results["metrics"]
        cls.rows = cls.results["per_claim_results"]

    def test_corpus_size_from_the_raw_fixture_file(self):
        cases = raw_cases()
        self.assertEqual(len(cases), 18)
        self.assertEqual(sum(len(c["claims"]) for c in cases), 21)
        self.assertEqual(Counter(c["category"] for c in cases), Counter({
            "directly_supported": 3, "paraphrased_supported": 3, "negation_contradicted": 3,
            "numeric_contradicted": 3, "unsupported": 3, "compound": 3,
        }))

    def test_ground_truth_label_totals_from_the_raw_fixture_file(self):
        labels = Counter(claim["label"] for c in raw_cases() for claim in c["claims"])
        self.assertEqual(labels, Counter({"SUPPORTED": 9, "CONTRADICTED": 8, "UNSUPPORTED": 4}))

    def test_total_and_correct_claims(self):
        self.assertEqual(self.metrics["total_claims"], 21)
        self.assertEqual(self.metrics["correct_claims"], 19)
        self.assertEqual(len(self.rows), 21)
        self.assertEqual(sum(1 for r in self.rows if r["correct"]), 19)

    def test_overall_accuracy(self):
        self.assertEqual(round(self.metrics["overall_accuracy"], 4), 0.9048)
        self.assertEqual(self.metrics["overall_accuracy"], 19 / 21)

    def test_per_class_results(self):
        per_class = self.metrics["per_category_accuracy"]
        self.assertEqual((per_class["SUPPORTED"]["correct"], per_class["SUPPORTED"]["total"]), (7, 9))
        self.assertEqual((per_class["CONTRADICTED"]["correct"], per_class["CONTRADICTED"]["total"]), (8, 8))
        self.assertEqual((per_class["UNSUPPORTED"]["correct"], per_class["UNSUPPORTED"]["total"]), (4, 4))

    def test_confusion_matrix_values_and_orientation(self):
        # rows = ground truth, columns = prediction
        self.assertEqual(self.metrics["confusion_matrix"], {
            "SUPPORTED": {"SUPPORTED": 7, "CONTRADICTED": 0, "UNSUPPORTED": 2},
            "CONTRADICTED": {"SUPPORTED": 0, "CONTRADICTED": 8, "UNSUPPORTED": 0},
            "UNSUPPORTED": {"SUPPORTED": 0, "CONTRADICTED": 0, "UNSUPPORTED": 4},
        })

    def test_confusion_matrix_matches_a_recount_of_the_per_claim_rows(self):
        recount = {a: Counter() for a in LABELS}
        for r in self.rows:
            recount[r["ground_truth"]][r["predicted"]] += 1
        for actual in LABELS:
            for predicted in LABELS:
                self.assertEqual(self.metrics["confusion_matrix"][actual][predicted], recount[actual][predicted])

    def test_ground_truth_comes_from_the_fixture_not_from_the_prediction(self):
        expected = [claim["label"] for c in raw_cases() for claim in c["claims"]]
        self.assertEqual([r["ground_truth"] for r in self.rows], expected)

    def test_the_two_known_misses_are_the_paraphrase_cases(self):
        misses = [r for r in self.rows if not r["correct"]]
        self.assertEqual([r["case_id"] for r in misses], ["c05_paraphrase_commute", "c06_paraphrase_solar"])
        for r in misses:
            self.assertEqual(r["category"], "paraphrased_supported")
            self.assertEqual(r["ground_truth"], "SUPPORTED")
            self.assertEqual(r["predicted"], "UNSUPPORTED")
            self.assertEqual(r["reason"], "no_matching_clause")
            self.assertLess(r["overlap"], checker.SUPPORT_OVERLAP_THRESHOLD)
        self.assertEqual(misses[0]["overlap"], 0.4545)
        self.assertEqual(misses[1]["overlap"], 0.3333)
        self.assertEqual(self.metrics["paraphrased_supported_misclassified_count"], 2)

    def test_the_correctly_classified_paraphrase_is_still_c04(self):
        row = next(r for r in self.rows if r["case_id"] == "c04_paraphrase_temperature")
        self.assertTrue(row["correct"])
        self.assertEqual(row["overlap"], 0.8889)


class TestCommittedResultsAreCurrent(unittest.TestCase):
    def test_a_fresh_evaluation_matches_the_committed_results_file(self):
        fresh = run_evaluation.evaluate(run_evaluation.load_corpus_store())
        committed = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(json.loads(json.dumps(fresh)), committed)

    def test_the_documented_command_reproduces_the_committed_file_ignoring_line_endings(self):
        # Text mode writes the platform's line endings, so a checkout with
        # LF and a run on Windows (CRLF) differ in newlines only. Everything
        # else must match byte for byte.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "results.json"
            original = run_evaluation.DEFAULT_RESULTS_PATH
            run_evaluation.DEFAULT_RESULTS_PATH = target
            try:
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    status = run_evaluation.main()
            finally:
                run_evaluation.DEFAULT_RESULTS_PATH = original
            self.assertEqual(status, 0)
            self.assertEqual(
                target.read_bytes().replace(b"\r\n", b"\n"),
                RESULTS_PATH.read_bytes().replace(b"\r\n", b"\n"),
            )
        self.assertIn("total claims: 21", out.getvalue())
        self.assertIn("overall accuracy: 0.905", out.getvalue())

    def test_the_committed_file_states_the_same_semantic_counts(self):
        committed = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        rows = committed["per_claim_results"]
        self.assertEqual(committed["metrics"]["total_claims"], len(rows))
        self.assertEqual(committed["metrics"]["correct_claims"], sum(1 for r in rows if r["predicted"] == r["ground_truth"]))
        self.assertEqual(sum(1 for r in rows if r["predicted"] == r["ground_truth"]), 19)


class TestRunnerScoring(unittest.TestCase):
    """The runner must score against the stored label, never the prediction."""

    def test_a_wrong_stored_label_counts_as_a_miss(self):
        from hallucination_claim_grounding_eval.corpus_store import CorpusStore
        store = CorpusStore([{
            "case_id": "x", "category": "directly_supported",
            "source_document": "The library opens at nine.",
            "claims": [{"text": "the library opens at nine", "label": "UNSUPPORTED"}],
        }])
        result = run_evaluation.evaluate(store)
        self.assertEqual(result["metrics"]["correct_claims"], 0)
        self.assertEqual(result["metrics"]["confusion_matrix"]["UNSUPPORTED"]["SUPPORTED"], 1)
        self.assertEqual(result["metrics"]["confusion_matrix"]["SUPPORTED"]["UNSUPPORTED"], 0)


if __name__ == "__main__":
    unittest.main()
