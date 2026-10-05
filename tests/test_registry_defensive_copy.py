import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hallucination_claim_grounding_eval.corpus_store import CorpusStore  # noqa: E402


class TestCorpusStoreDefensiveCopy(unittest.TestCase):
    """Regression tests for CorpusStore's defensive copying: mutating the
    object passed in, or the object handed back out, must never move the
    store's internal state. Without this, a caller holding a reference to
    either object could silently rewrite the ground truth an evaluation
    run is scored against.
    """

    def _sample_cases(self):
        return [
            {"case_id": "c1", "claims": [{"text": "a claim", "label": "SUPPORTED"}]},
            {"case_id": "c2", "claims": [{"text": "another claim", "label": "UNSUPPORTED"}]},
        ]

    def test_mutating_input_after_construction_does_not_affect_store(self):
        original = self._sample_cases()
        store = CorpusStore(original)

        original.append({"case_id": "c3", "claims": []})
        original[0]["claims"][0]["label"] = "CONTRADICTED"

        stored = store.cases()
        self.assertEqual(len(stored), 2)
        self.assertEqual(stored[0]["claims"][0]["label"], "SUPPORTED")

    def test_mutating_returned_cases_does_not_affect_store(self):
        store = CorpusStore(self._sample_cases())

        returned = store.cases()
        returned.append({"case_id": "intruder", "claims": []})
        returned[0]["claims"][0]["label"] = "CONTRADICTED"
        returned[0]["claims"].append({"text": "injected", "label": "UNSUPPORTED"})

        stored_again = store.cases()
        self.assertEqual(len(stored_again), 2)
        self.assertEqual(stored_again[0]["claims"][0]["label"], "SUPPORTED")
        self.assertEqual(len(stored_again[0]["claims"]), 1)

    def test_len(self):
        store = CorpusStore(self._sample_cases())
        self.assertEqual(len(store), 2)


if __name__ == "__main__":
    unittest.main()
