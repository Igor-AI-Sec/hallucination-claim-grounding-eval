# hallucination-claim-grounding-eval

A small, deterministic classifier that scores whether individual claims in
an answer are supported, contradicted, or unsupported by a source document,
and a benchmark run of that classifier against a synthetic fixture corpus
built specifically for this evaluation.

**All of the data here is synthetic.** The source documents, questions,
answers, and every claim label in `fixtures/corpus.json` were created for
this project (a fictional robotics company, a fictional drug trial, a
fictional library renovation, and so on) and are not real news stories,
trial results, or companies, and not production data of any kind. Each
claim's ground-truth label is stored directly in the fixture and is not
derived from the classifier's prediction.

## Why I built this

`sourcecheck-verify-ai-sources`, an earlier project in this portfolio,
checks whether a link an AI answer cites actually exists and whether the
linked page's content overlaps with what the answer says near it. That's
useful, but it only looks at citations. It never asks whether the answer's
own sentences are actually true relative to a document the model was
given. I wanted to try the more direct version of that question: given a
source document and a candidate answer, which individual claims in the
answer does the source actually support, and which does it not?

## Methodology

The pipeline has three pieces, all pure Python with no dependencies:

1. **Segmenter** (`segmenter.py`) splits text into sentences, then further
   splits each sentence on a fixed set of clause connectors (`, and `,
   `; `, ` but `) so that a single sentence carrying two facts becomes two
   claims. This is a punctuation rule, not a parser. It won't correctly
   split every possible compound sentence, and I didn't try to make it.
2. **Checker** (`checker.py`) takes one claim and a source document and
   returns `SUPPORTED`, `CONTRADICTED`, or `UNSUPPORTED`, using three rules
   applied in order:
   - If the claim states a number and the best-matching part of the source
     also states a number for the same fact, and they differ, that's a
     numeric contradiction. If the source's best-matching part never
     states any number at all for that fact, the claim's number is
     unsupported, not contradicted, even when the surrounding sentence
     otherwise overlaps heavily (see the numeric-mismatch bullet under
     "Limitations" below).
   - If the claim's negation (does it contain "not", "never", "doesn't",
     etc.) disagrees with the negation of the best-matching part of the
     source, on a topic they otherwise clearly share, that's a negation
     contradiction.
   - Otherwise, if enough of the claim's content words show up in some
     part of the source, it's supported; if not, it's unsupported.
   "Best-matching part of the source" means the checker splits the source
   document with the same segmenter and picks whichever resulting clause
   has the highest word overlap with the claim. This matters for sources
   that state more than one fact, so a negation in an unrelated clause
   doesn't wrongly flag every claim in the document (see
   `tests/test_checker.py::test_negation_localized_to_matching_clause_not_whole_source`
   for the regression test covering this).
3. **Evaluation runner** (`run_evaluation.py`) loads every fixture case,
   classifies every claim, and compares the prediction against the
   ground-truth label stored with the fixture, not against whatever the
   classifier happens to output. It writes a confusion matrix and
   per-category accuracy to `results/results.json`.

The fixture corpus is held by a small `CorpusStore` class
(`corpus_store.py`) that copies the case list on the way in and on the way
out of its `cases()` accessor, so nothing external can mutate the trusted
ground truth through a returned reference. `tests/test_registry_defensive_copy.py`
mutates both the object passed in and the object handed back, and checks
neither move the store's internal state -- a plain list assignment would
look safe under casual testing but still leaves the internal state open
to exactly this kind of accidental mutation, so it's worth testing
directly rather than assuming.

## The corpus

`fixtures/corpus.json` has 18 cases and 21 individually labeled claims,
split across six categories, three or so cases each: directly supported
claims, paraphrased-but-supported claims (same fact, different wording),
negation-flipped claims, numeric-mismatch claims, entirely unsupported
claims, and compound answers that mix two claims of different categories
in one answer.

## Results

Running `run_evaluation.py` against the fixture corpus gives:

| Category | Correct / Total | Accuracy |
|---|---|---|
| SUPPORTED | 7 / 9 | 0.778 |
| CONTRADICTED | 8 / 8 | 1.000 |
| UNSUPPORTED | 4 / 4 | 1.000 |
| **Overall** | **19 / 21** | **0.905** |

Every negation-contradiction and numeric-contradiction case in the corpus
was caught correctly, and every genuinely unsupported claim was correctly
flagged as unsupported. Both misses were in the paraphrase category, and
they're not surprising once you look at them.

### The two misses

**`c05_paraphrase_commute`**: the source says the new rail line "cut the
average commute between Elmwood and downtown in half." The answer says it
"dramatically reduced how long it takes Elmwood residents to reach
downtown." Those are the same claim, but "cut ... in half" and
"dramatically reduced" share no words, and "average commute" vs. "how
long it takes ... to reach" share none either. The overlap between the
claim's content words and the source came out to 0.45, below the 0.6
threshold the checker uses to call something supported, so it was marked
`UNSUPPORTED`.

**`c06_paraphrase_solar`**: the source says the assembly line "runs
entirely on solar power collected from panels on the factory roof." The
answer says it "draws all of its electricity from rooftop solar cells
above the plant." "Power" vs. "electricity," "panels" vs. "cells," "roof"
vs. "above," "factory" vs. "plant." The only words that survive exact
match are "solar," "assembly," and "line." Overlap came out to 0.33, well
below threshold, so this was also marked `UNSUPPORTED`.

This is exactly the limitation I expected going in: the checker has no
notion of synonymy. It doesn't stem words ("increase" and "increased"
don't match each other either; that one didn't happen to flip a verdict
in this corpus, but it's the same gap), and it has no concept of two
different phrases meaning the same thing. A third paraphrase case in the
corpus (`c04_paraphrase_temperature`) was classified correctly, but only
because it happened to keep enough of the original wording ("coldest
morning," "temperature," "negative 14 degrees") to clear the threshold,
not because the checker understood the paraphrase.

## What this measures, and what it doesn't

This is a measurement of one specific, fully deterministic classifier
against a small, purpose-built synthetic corpus. It tells you: given this
exact lexical/negation/numeric rule set, how often does it agree with the
ground-truth labels stored with these 21 claims. It does not tell you:

- How this would perform on real model output, on longer documents, or on
  claims phrased in ways I didn't think to include.
- Anything about entailment or paraphrase detection in general. The
  checker doesn't attempt either; it does keyword overlap plus two
  narrow, explicit rules.
- A false positive/negative rate you could treat as a population
  estimate. 21 claims is enough to see a pattern (paraphrase is the weak
  spot, everything else the rules were built for works), not enough to
  say "this happens X% of the time" about hallucination detection in
  general.

## Limitations

- **No general stemming or synonym handling.** This is the direct cause
  of both misclassifications above, and it's a design choice, not an
  oversight. The implementation does have one very narrow, internal
  exception: a naive trailing-s strip used only to line up singular and
  plural forms of the same noun when localizing negation scope and
  associating a number with its entity ("patient" vs. "patients"). It
  isn't a general stemmer, doesn't touch the main lexical-overlap check,
  and won't help with irregular forms or other suffixes. Adding a real
  synonym list would mean curating which words count as equivalent,
  which would make the corpus results reflect those curation choices
  more than the technique itself.
- **The corpus is small and purpose-built for this project.** 18 cases is enough to
  exercise every category at least three times, not enough to be a
  general benchmark. There's no case with more than two claims per
  answer, no multi-paragraph source document, and no case testing a unit
  mismatch (e.g. a claim stating "3.5 feet" against a source stating "42
  inches," the same fact, but the checker's numeric check only compares
  digit strings, so it would treat these as unrelated numbers rather than
  as a match or even a mismatch worth flagging).
- **Negation detection is a fixed word list, not real negation scope
  analysis.** "Not only did it increase" would be read as negated even
  though the actual claim being made is affirmative. The corpus doesn't
  have a case like this, so this is a known gap the numbers above don't
  cover.
- **Claims are pre-segmented, stored data, not the segmenter's output.**
  `run_evaluation.py` classifies the stored claim list from each fixture
  directly; it does not first run the segmenter on the answer text and
  hope the split lines up with those claims. The segmenter is tested
  separately in `tests/test_segmenter.py`. This keeps the two questions
  ("does the classifier work" and "does the segmenter split correctly")
  from being tangled together in one number, but it also means the
  reported accuracy doesn't include any segmentation errors that would
  occur in a fully automated pipeline.
- **The support threshold (0.6) was picked by hand, not calibrated on a
  held-out set.** The reported accuracy describes how this classifier
  performs on this exact fixture set at this exact threshold. It isn't an
  estimate of how well a different threshold, or this same threshold on
  different data, would generalize.
- **Numeric mismatch detection pairs each number with the word right
  after it** ("12 clinics" -> the number 12 is about clinics), singularized
  so "1 patient" and "340 patients" count as the same entity, and skipping
  a stopword or negation word right after the number (so "January 9 was"
  doesn't pair 9 with the copula "was" as if that were the fact in
  question). This is enough to tell "12 patients" apart from an unrelated
  "12 clinics" in the same sentence, and it's a genuine tri-state
  comparison: same entity with a different number is `CONTRADICTED`;
  same entity with the same number falls through to the rules below; a
  claim number for an entity the source's best-matching part never states
  any number for at all is `UNSUPPORTED`, and this holds even when the
  rest of the sentence overlaps heavily enough to otherwise clear the
  support threshold ("The large city park has 12 benches" against "The
  large city park has 10 tables" shares most of its words but the source
  never gives a number for benches, so it's correctly `UNSUPPORTED`, not
  `SUPPORTED`). It's still just adjacency, not real entity resolution: a
  number separated from its noun by punctuation, or stated before its
  noun ("clinics: 12"), won't be paired and falls back to the plain
  digit-set comparison, which can miss the same kind of mismatch this
  rule was added to catch.
- **A negative number is recognized ("negative 14" or "-14"), but only as
  a sign on the number immediately following it.** This is enough to
  keep the fixture corpus's own "negative 14 degrees" case from matching
  a claim of plain "14 degrees," but the checker still has no concept of
  units, ranges, or approximate values.
- **No semantic role or entity-relation matching.** The checker compares
  words, not who is doing what to whom. A claim that swaps one specific
  argument for another in an otherwise near-identical sentence ("The park
  allows cats" against a source that says "The park allows dogs") can
  still clear the lexical-overlap threshold and come back `SUPPORTED`,
  because "park," "allows," and the sentence shape all match; only the
  one word that actually matters differs. Fixing this reliably would mean
  parsing which noun fills which role in the sentence, which is exactly
  the kind of semantic parsing this project deliberately doesn't attempt
  (see "What this measures, and what it doesn't" above). The failure is
  disclosed here rather than patched around.

## Reproduction

From the repository root:

```bash
python -m unittest discover -s tests -v
python src/hallucination_claim_grounding_eval/run_evaluation.py
```

The second command overwrites `results/results.json`. Since everything
here is deterministic (no randomness, no model calls), re-running it
should reproduce the exact numbers in this README.

## Layout

```
src/hallucination_claim_grounding_eval/
  segmenter.py       sentence/clause splitting
  checker.py          the three-rule claim classifier
  corpus_store.py     defensive-copy wrapper around the fixture corpus
  run_evaluation.py   loads fixtures, classifies, writes results.json
fixtures/corpus.json  18 synthetic fixture cases, 21 labeled claims
tests/                unit tests for each module above
results/results.json  output of the last evaluation run
reports/evaluation_report.md  narrative write-up of the results
```
