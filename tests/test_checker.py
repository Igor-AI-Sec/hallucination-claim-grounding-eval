import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hallucination_claim_grounding_eval import checker  # noqa: E402


class TestTokenHelpers(unittest.TestCase):
    def test_tokenize_lowercases_and_strips_punctuation(self):
        self.assertEqual(checker.tokenize("The Park's benches, 10!"), ["the", "park's", "benches", "10"])

    def test_extract_numbers(self):
        self.assertEqual(checker.extract_numbers("It cost 12.5 dollars for 3 tickets"), {"12.5", "3"})
        self.assertEqual(checker.extract_numbers("no numbers here"), set())

    def test_content_words_drops_stopwords_and_negation(self):
        self.assertEqual(
            checker.content_words("The park does not allow dogs"),
            {"park", "allow", "dogs"},
        )

    def test_has_negation(self):
        self.assertTrue(checker.has_negation("The park does not allow dogs"))
        self.assertFalse(checker.has_negation("The park allows dogs"))
        self.assertTrue(checker.has_negation("The park doesn't allow dogs"))

    def test_overlap_ratio_empty_claim_words(self):
        self.assertEqual(checker.overlap_ratio(set(), {"a", "b"}), 0.0)

    def test_overlap_ratio_full_match(self):
        self.assertEqual(checker.overlap_ratio({"a", "b"}, {"a", "b", "c"}), 1.0)

    def test_extract_number_entities_pairs_each_number_with_the_next_word(self):
        # Entity nouns come back singularized (see _strip_plural_s) so
        # "1 patient" and "340 patients" are recognized as the same
        # entity rather than bypassing numeric-mismatch detection.
        self.assertEqual(
            checker.extract_number_entities("The trial enrolled 340 patients across 12 clinics"),
            {("340", "patient"), ("12", "clinic")},
        )

    def test_extract_number_entities_empty_for_no_numbers(self):
        self.assertEqual(checker.extract_number_entities("no numbers here"), set())

    def test_extract_number_entities_singular_and_plural_forms_match(self):
        self.assertEqual(
            checker.extract_number_entities("1 patient"),
            checker.extract_number_entities("1 patients"),
        )

    def test_extract_numbers_distinguishes_negative_from_positive(self):
        self.assertEqual(checker.extract_numbers("The temperature was negative 14 degrees."), {"-14"})
        self.assertEqual(checker.extract_numbers("The temperature was 14 degrees."), {"14"})
        self.assertEqual(checker.extract_numbers("The temperature was -14 degrees."), {"-14"})

    def test_extract_number_entities_captures_negative_sign(self):
        self.assertEqual(
            checker.extract_number_entities("The temperature was negative 14 degrees."),
            {("-14", "degree")},
        )
        self.assertEqual(
            checker.extract_number_entities("The temperature was 14 degrees."),
            {("14", "degree")},
        )

    def test_negation_scope_isolates_the_matching_sub_part(self):
        clause = "The park has 10 benches and does not allow dogs"
        scope = checker._negation_scope({"park", "benches"}, clause)
        self.assertFalse(checker.has_negation(scope))
        self.assertIn("benches", scope)

    def test_negation_scope_returns_whole_clause_when_not_splittable(self):
        clause = "The park does not allow dogs"
        scope = checker._negation_scope({"park", "dogs"}, clause)
        self.assertEqual(scope, clause)

    def test_negation_scope_breaks_ties_in_favor_of_the_specific_part(self):
        # "park" is shared with both sub-parts; "tables" only with the
        # second. Plain overlap_ratio against the claim ties at 0.5/0.5
        # (see the regression tests in TestClassifyClaim for the
        # end-to-end symptom this caused); Jaccard breaks the tie toward
        # the sub-part that is entirely about the claim's fact.
        clause = "The park has 10 benches and does not have 10 tables"
        scope = checker._negation_scope({"park", "tables"}, clause)
        self.assertIn("tables", scope)
        self.assertTrue(checker.has_negation(scope))

    def test_strip_plural_s_drops_trailing_s_on_longer_words(self):
        self.assertEqual(checker._strip_plural_s("allows"), "allow")
        self.assertEqual(checker._strip_plural_s("tables"), "table")

    def test_strip_plural_s_leaves_short_or_double_s_words_alone(self):
        self.assertEqual(checker._strip_plural_s("gas"), "gas")
        self.assertEqual(checker._strip_plural_s("is"), "is")
        self.assertEqual(checker._strip_plural_s("class"), "class")


class TestClassifyClaim(unittest.TestCase):
    def test_numeric_mismatch_is_contradicted(self):
        result = checker.classify_claim("The park has 12 benches", "The park has 10 benches.")
        self.assertEqual(result.label, checker.CONTRADICTED)
        self.assertEqual(result.reason, "numeric_mismatch")
        self.assertEqual(result.evidence["claim_numbers"], ["12"])
        self.assertEqual(result.evidence["source_numbers"], ["10"])

    def test_negation_mismatch_is_contradicted(self):
        result = checker.classify_claim("The park allows dogs", "The park does not allow dogs.")
        self.assertEqual(result.label, checker.CONTRADICTED)
        self.assertEqual(result.reason, "negation_mismatch")

    def test_negation_mismatch_other_direction(self):
        result = checker.classify_claim("The park does not allow dogs", "The park allows dogs.")
        self.assertEqual(result.label, checker.CONTRADICTED)
        self.assertEqual(result.reason, "negation_mismatch")

    def test_direct_lexical_match_is_supported(self):
        result = checker.classify_claim(
            "The museum opens at 9 am on weekdays",
            "The museum opens at 9 am on weekdays.",
        )
        self.assertEqual(result.label, checker.SUPPORTED)
        self.assertEqual(result.reason, "lexical_overlap")

    def test_unrelated_claim_is_unsupported(self):
        result = checker.classify_claim(
            "The museum has a gift shop on the third floor",
            "The museum opens at 9 am on weekdays.",
        )
        self.assertEqual(result.label, checker.UNSUPPORTED)

    def test_known_limitation_heavy_paraphrase_misclassified(self):
        """Documents an intentional, disclosed limitation: a claim that
        paraphrases a supported fact with almost entirely different
        wording falls below the lexical-overlap threshold and is
        misclassified as UNSUPPORTED even though it is in fact supported.
        See README.md 'Limitations'.
        """
        result = checker.classify_claim(
            "The company dramatically reduced its overall costs",
            "The company cut its expenses in half.",
        )
        self.assertEqual(result.label, checker.UNSUPPORTED)

    def test_picks_best_matching_clause_from_multi_clause_source(self):
        source = "The park has 10 benches. The library has 40 chairs."
        result = checker.classify_claim("The library has 40 chairs", source)
        self.assertEqual(result.label, checker.SUPPORTED)
        self.assertEqual(result.best_clause, "The library has 40 chairs")

    def test_negation_localized_to_matching_clause_not_whole_source(self):
        # A compound source where only one clause is negated: the
        # unrelated claim about the other clause must not be flagged
        # just because the source contains a negation word somewhere.
        source = "The trial enrolled 340 patients across 12 clinics; the drug did not increase blood pressure."
        result = checker.classify_claim("The trial enrolled 340 patients across 12 clinics", source)
        self.assertEqual(result.label, checker.SUPPORTED)

    def test_number_reused_for_a_different_entity_is_not_supported(self):
        # The claim's number (12) does appear in the source, but attached
        # to "clinics", not "patients" -- must not be treated as if the
        # source actually said "12 patients".
        result = checker.classify_claim(
            "The trial enrolled 12 patients",
            "The trial enrolled 340 patients across 12 clinics.",
        )
        self.assertNotEqual(result.label, checker.SUPPORTED)
        self.assertEqual(result.label, checker.CONTRADICTED)
        self.assertEqual(result.reason, "numeric_mismatch")

    def test_number_matching_a_different_noun_in_the_source_is_not_supported(self):
        result = checker.classify_claim(
            "The park has 12 benches",
            "The park has 10 benches and 12 tables.",
        )
        self.assertNotEqual(result.label, checker.SUPPORTED)
        self.assertEqual(result.label, checker.CONTRADICTED)
        self.assertEqual(result.reason, "numeric_mismatch")

    def test_negation_of_a_different_fact_in_the_same_clause_does_not_contradict(self):
        # "does not allow dogs" is about dogs, not benches -- the bench
        # count claim is unaffected by that unrelated negation.
        result = checker.classify_claim(
            "The park has 10 benches",
            "The park has 10 benches and does not allow dogs.",
        )
        self.assertEqual(result.label, checker.SUPPORTED)

    def test_negated_claim_not_supported_by_an_unrelated_negation_elsewhere(self):
        # The claim denies the bench count; the source actually affirms
        # it (the source's only negation is about dogs, unrelated). The
        # negated claim must not be marked SUPPORTED just because the
        # source contains some negation word.
        result = checker.classify_claim(
            "The park does not have 10 benches",
            "The park has 10 benches and does not allow dogs.",
        )
        self.assertNotEqual(result.label, checker.SUPPORTED)
        self.assertEqual(result.label, checker.CONTRADICTED)
        self.assertEqual(result.reason, "negation_mismatch")

    def test_negation_scope_tie_prefers_the_claims_specific_fact(self):
        # Both sub-parts share exactly one word with the claim ("park" in
        # one, "tables" in the other) if you count by plain overlap
        # ratio alone -- that tie used to resolve to whichever sub-part
        # came first in the source, which is the unrelated one here. The
        # claim is affirmative about tables; the source negates tables.
        result = checker.classify_claim(
            "The park has 10 tables",
            "The park has 10 benches and does not have 10 tables.",
        )
        self.assertNotEqual(result.label, checker.SUPPORTED)
        self.assertEqual(result.label, checker.CONTRADICTED)
        self.assertEqual(result.reason, "negation_mismatch")

    def test_negation_scope_tie_other_direction_is_supported(self):
        # Same source, negated claim this time -- the claim's negation
        # now agrees with what the source actually says about tables.
        result = checker.classify_claim(
            "The park does not have 10 tables",
            "The park has 10 benches and does not have 10 tables.",
        )
        self.assertEqual(result.label, checker.SUPPORTED)

    def test_negation_scope_tie_broken_by_plural_normalization(self):
        # "allow" (source) vs "allows" (claim) would tie the two
        # sub-parts even after the Jaccard fix, without also comparing
        # words after a naive trailing-s strip -- the source explicitly
        # negates allowing dogs, so this must not come back SUPPORTED.
        result = checker.classify_claim(
            "The park allows dogs",
            "The park has benches and does not allow dogs.",
        )
        self.assertNotEqual(result.label, checker.SUPPORTED)
        self.assertEqual(result.label, checker.CONTRADICTED)
        self.assertEqual(result.reason, "negation_mismatch")

    def test_number_for_a_different_entity_entirely_is_unsupported_not_contradicted(self):
        # The source gives a number for tables, not benches -- the claim's
        # fact (benches) is simply never addressed with a number at all.
        # That's UNSUPPORTED, not CONTRADICTED: a contradiction requires
        # the source to actually state a conflicting number for the SAME
        # thing the claim is about.
        result = checker.classify_claim(
            "The park has 12 benches",
            "The park has 10 tables.",
        )
        self.assertEqual(result.label, checker.UNSUPPORTED)

    def test_numeric_mismatch_survives_singular_plural_entity_mismatch(self):
        # The source's "1" belongs to "clinic", not "patient" -- the real
        # patient count (340) contradicts the claim's "1 patient". Before
        # entity nouns were singularized, "patient" (claim) vs "patients"
        # (source) never matched as the same entity, so this fell through
        # to an unrelated "1" the source does mention (for clinics) and
        # came back SUPPORTED.
        result = checker.classify_claim(
            "The trial enrolled 1 patient",
            "The trial enrolled 340 patients across 1 clinic.",
        )
        self.assertNotEqual(result.label, checker.SUPPORTED)
        self.assertEqual(result.label, checker.CONTRADICTED)
        self.assertEqual(result.reason, "numeric_mismatch")

    def test_negative_number_contradicts_the_same_positive_number(self):
        result = checker.classify_claim(
            "The temperature was 14 degrees",
            "The temperature was negative 14 degrees.",
        )
        self.assertEqual(result.label, checker.CONTRADICTED)
        self.assertEqual(result.reason, "numeric_mismatch")

    def test_negative_number_is_supported_by_the_same_negative_number(self):
        result = checker.classify_claim(
            "The temperature was negative 14 degrees",
            "The temperature was negative 14 degrees.",
        )
        self.assertEqual(result.label, checker.SUPPORTED)

    def test_numeric_entity_unaddressed_by_source_is_unsupported_despite_high_overlap(self):
        # "large city park" and "has" are shared, so lexical overlap alone
        # would clear the 0.6 support threshold, but the source never
        # states any number for "benches" -- only for "tables". A claim
        # number for an entity the source is silent on must not become
        # SUPPORTED just because the rest of the sentence overlaps.
        result = checker.classify_claim(
            "The large city park has 12 benches",
            "The large city park has 10 tables.",
        )
        self.assertEqual(result.label, checker.UNSUPPORTED)
        self.assertEqual(result.reason, "numeric_entity_unaddressed")

    def test_numeric_entity_unaddressed_generic_not_hardcoded_to_one_pair(self):
        # Same shape as the benches/tables case above but with an
        # unrelated pair of nouns, to confirm the fix is generic rather
        # than special-cased to "benches"/"tables".
        result = checker.classify_claim(
            "The company shipped 12 robots this quarter",
            "The company shipped 10 drones this quarter.",
        )
        self.assertEqual(result.label, checker.UNSUPPORTED)
        self.assertEqual(result.reason, "numeric_entity_unaddressed")


if __name__ == "__main__":
    unittest.main()
