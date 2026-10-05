"""Run the claim-support classifier over the fixture corpus and score it against
the ground-truth label stored with each fixture.

Usage: python src/hallucination_claim_grounding_eval/run_evaluation.py
(run from the repository root)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hallucination_claim_grounding_eval import checker  # noqa: E402
from hallucination_claim_grounding_eval.corpus_store import CorpusStore  # noqa: E402

DEFAULT_CORPUS_PATH = PROJECT_ROOT / "fixtures" / "corpus.json"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results" / "results.json"

LABELS = [checker.SUPPORTED, checker.CONTRADICTED, checker.UNSUPPORTED]


def load_corpus_store(path: Path = DEFAULT_CORPUS_PATH) -> CorpusStore:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CorpusStore(data["cases"])


def evaluate(store: CorpusStore) -> dict:
    confusion = {actual: Counter() for actual in LABELS}
    per_claim_results = []

    for case in store.cases():
        for claim in case["claims"]:
            predicted = checker.classify_claim(claim["text"], case["source_document"])
            actual_label = claim["label"]
            confusion[actual_label][predicted.label] += 1
            per_claim_results.append({
                "case_id": case["case_id"],
                "category": case["category"],
                "claim": claim["text"],
                "ground_truth": actual_label,
                "predicted": predicted.label,
                "correct": predicted.label == actual_label,
                "reason": predicted.reason,
                "best_matching_source_clause": predicted.best_clause,
                "overlap": round(predicted.overlap, 4),
            })

    total = len(per_claim_results)
    correct = sum(1 for r in per_claim_results if r["correct"])

    per_category_totals: dict[str, Counter] = {}
    for r in per_claim_results:
        cat = r["ground_truth"]
        per_category_totals.setdefault(cat, Counter())
        per_category_totals[cat]["total"] += 1
        if r["correct"]:
            per_category_totals[cat]["correct"] += 1

    per_category_accuracy = {
        label: {
            "correct": per_category_totals.get(label, Counter())["correct"],
            "total": per_category_totals.get(label, Counter())["total"],
            "accuracy": (
                per_category_totals[label]["correct"] / per_category_totals[label]["total"]
                if per_category_totals.get(label, Counter())["total"] else None
            ),
        }
        for label in LABELS
    }

    confusion_matrix = {
        actual: {predicted: confusion[actual][predicted] for predicted in LABELS}
        for actual in LABELS
    }

    paraphrase_misses = [
        r for r in per_claim_results
        if r["category"] == "paraphrased_supported" and not r["correct"]
    ]

    return {
        "metrics": {
            "total_claims": total,
            "overall_accuracy": correct / total if total else None,
            "correct_claims": correct,
            "per_category_accuracy": per_category_accuracy,
            "confusion_matrix": confusion_matrix,
            "paraphrased_supported_misclassified_count": len(paraphrase_misses),
        },
        "per_claim_results": per_claim_results,
    }


def main() -> int:
    store = load_corpus_store()
    results = evaluate(store)

    DEFAULT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    m = results["metrics"]
    print(f"[run_evaluation] total claims: {m['total_claims']}")
    print(f"[run_evaluation] overall accuracy: {m['overall_accuracy']:.3f}")
    for label, stats in m["per_category_accuracy"].items():
        acc = f"{stats['accuracy']:.3f}" if stats["accuracy"] is not None else "n/a"
        print(f"  [{label}] {stats['correct']}/{stats['total']} ({acc})")
    print(f"[run_evaluation] wrote {DEFAULT_RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
