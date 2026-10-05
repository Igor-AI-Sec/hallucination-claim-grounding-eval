"""Deterministic claim-support classifier.

Classifies a single claim as SUPPORTED, CONTRADICTED, or UNSUPPORTED
relative to a source document, using three explicit, inspectable rules
applied in order:

1. Numeric mismatch -- the claim states a number for a fact the source's
   best-matching clause also states a number for, and the numbers differ.
   A number is compared against the source's number for the *same*
   nearby noun where possible (see extract_number_entities below), not
   just "does this digit string appear anywhere in the clause" -- a
   clause can state several numbers for different things. Three
   outcomes: same entity + different number is a contradiction; same
   entity + same number continues to the rules below; a claim number
   whose noun the clause never states any number for at all is
   UNSUPPORTED (the fact is unaddressed, not confirmed, so it cannot
   become SUPPORTED just from unrelated lexical overlap). Also
   distinguishes "negative 14" / "-14" from a plain "14".
2. Negation mismatch -- the claim's negation status (does it contain a
   negation word?) disagrees with the negation status of whichever part
   of the best-matching source clause actually overlaps the claim's
   content words, on a topic the two otherwise clearly share.
3. Lexical overlap -- otherwise, the claim is SUPPORTED if enough of its
   content words appear in some source clause, else UNSUPPORTED.

This is a lexical/rule-based approach, not a general natural-language
entailment model. It is expected to (and, per the evaluation report, does)
miss claims that paraphrase a supported fact using largely different
words -- see the README's "Limitations" section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import segmenter

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
UNSUPPORTED = "UNSUPPORTED"

SUPPORT_OVERLAP_THRESHOLD = 0.6
CONTRADICTION_OVERLAP_THRESHOLD = 0.5

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "it", "its",
    "this", "that", "these", "those", "with", "as", "by", "from", "has",
    "have", "had", "do", "does", "did", "than", "then", "so", "such",
    "also", "which", "who", "whom", "what", "when", "where", "why", "how",
    "there", "their", "they", "he", "she", "his", "her", "we", "our",
    "you", "your", "i", "my", "me", "will", "would", "can", "could",
    "should", "shall", "into", "about", "over", "under", "more", "most",
    "very", "just", "only", "both", "each", "other", "some", "any", "all",
}

NEGATION_TOKENS = {
    "not", "no", "never", "cannot", "isn't", "wasn't", "doesn't", "didn't",
    "won't", "aren't", "weren't", "can't", "couldn't", "shouldn't",
    "wouldn't", "hasn't", "haven't", "hadn't",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
# Optional sign (a literal "-" or the word "negative" immediately before
# the digits) captured separately so "negative 14"/"-14" can be told
# apart from a plain "14" -- without this, the fixture corpus's own
# "negative 14 degrees" case and its positive-sign contradiction look
# identical to the numeric checks below.
_NUMBER_RE = re.compile(r"(-|negative\s+)?(\d+(?:\.\d+)?)", re.IGNORECASE)
_NUMBER_ENTITY_RE = re.compile(r"(-|negative\s+)?(\d+(?:\.\d+)?)\s+([a-zA-Z]+)", re.IGNORECASE)


def _signed_number(sign: str, digits: str) -> str:
    return ("-" + digits) if sign else digits

# Splits a clause further than segmenter.segment() does, for negation
# scoping only. segmenter.segment() deliberately leaves a bare " and "
# unsplit when extracting claims (a single fact can legitimately contain
# "and"), but that same bare "and" can also join two unrelated facts
# ("has 10 benches and does not allow dogs"), and a negation word must
# not be allowed to leak from one onto the other.
_NEGATION_SCOPE_SPLIT_RE = re.compile(r";\s*|\s+but\s+|\s+and\s+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def extract_numbers(text: str) -> set[str]:
    return {_signed_number(sign, digits) for sign, digits in _NUMBER_RE.findall(text or "")}


def extract_number_entities(text: str) -> set[tuple[str, str]]:
    """Pair each number with the word right after it, e.g. "12 clinics"
    -> ("12", "clinics"), "negative 14 degrees" -> ("-14", "degree"). A
    crude stand-in for real entity/unit resolution: it tells "12
    patients" apart from "12 clinics" in the same sentence, but a number
    not immediately followed by its noun (reordered phrasing, a number
    separated from its noun by punctuation) won't be paired and falls
    back to the plain digit-set comparison. The noun is singularized with
    the same narrow trailing-s strip used for negation scoping (see
    _strip_plural_s), so "1 patient" and "340 patients" are recognized as
    the same entity rather than bypassing the numeric-mismatch check.

    A stopword or negation word right after a number is never treated as
    its entity noun: "January 9 was Northfield's coldest morning" would
    otherwise pair 9 with the copula "was", which isn't a real fact the
    source separately needs to confirm, and one side having a trailing
    comma after the number ("on January 9, with ...") while the other
    doesn't is enough to make only one side extract that bogus pairing --
    which produced a false numeric_entity_unaddressed result on an
    otherwise-matching temperature claim. A number followed only by a
    stopword is left unpaired here, the same as a number with no adjacent
    noun at all."""
    return {
        (_signed_number(sign, digits), _strip_plural_s(word.lower()))
        for sign, digits, word in _NUMBER_ENTITY_RE.findall(text or "")
        if word.lower() not in STOPWORDS and word.lower() not in NEGATION_TOKENS
    }


def _strip_plural_s(word: str) -> str:
    """Naive singularization: drop a trailing "s" when the word is long
    enough that this is unlikely to mangle it ("gas", "is" stay as-is;
    words ending "ss" like "class" are also left alone). Not a stemmer --
    it exists only so _negation_scope doesn't treat "allow"/"allows" as
    unrelated words when deciding which sub-part of a clause a claim is
    actually about. Deliberately this narrow: it won't help with
    irregular forms ("mice"/"mouse") or other suffixes ("ran"/"run")."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _negation_scope(claim_words: set[str], clause: str) -> str:
    """The part of `clause` to check for negation against `claim_words`.

    Splits `clause` on "and"/"but"/";" (more aggressively than
    segmenter.segment(), see _NEGATION_SCOPE_SPLIT_RE above) and returns
    whichever piece the claim is actually about.

    Picking the part with the highest plain overlap_ratio (shared words
    over the claim's word count) breaks down when two sub-parts tie: a
    generic word the claim shares with BOTH parts (e.g. "park", the
    subject of the whole sentence) can tie against a part that shares
    only the claim's one truly specific word (e.g. "tables"), and Python's
    max() then silently picks whichever part happened to come first --
    which is exactly backwards when the first part is the unrelated one.
    Jaccard similarity (shared words over ALL words either side
    mentions, not just the claim's) breaks that tie correctly: a part
    that's entirely about the claim's fact scores higher than a part
    that only shares one word out of several unrelated ones. Comparing
    after a naive trailing-s strip additionally keeps "allow" vs "allows"
    from counting as a mismatch and reintroducing the same tie.
    """
    parts = [p.strip() for p in _NEGATION_SCOPE_SPLIT_RE.split(clause) if p.strip()]
    if len(parts) <= 1:
        return clause

    normalized_claim = {_strip_plural_s(w) for w in claim_words}

    def specificity(part: str) -> float:
        normalized_part = {_strip_plural_s(w) for w in content_words(part)}
        union = normalized_claim | normalized_part
        if not union:
            return 0.0
        return len(normalized_claim & normalized_part) / len(union)

    return max(parts, key=specificity)


def content_words(text: str) -> set[str]:
    return {
        tok for tok in tokenize(text)
        if tok not in STOPWORDS and tok not in NEGATION_TOKENS and not tok.isdigit()
    }


def has_negation(text: str) -> bool:
    return any(tok in NEGATION_TOKENS for tok in tokenize(text))


def overlap_ratio(claim_words: set[str], clause_words: set[str]) -> float:
    if not claim_words:
        return 0.0
    return len(claim_words & clause_words) / len(claim_words)


@dataclass
class ClassificationResult:
    label: str
    reason: str
    best_clause: str
    overlap: float
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "reason": self.reason,
            "best_clause": self.best_clause,
            "overlap": round(self.overlap, 4),
            "evidence": self.evidence,
        }


def _best_matching_clause(claim_words: set[str], source_document: str) -> tuple[str, float]:
    clauses = segmenter.segment(source_document) or [source_document]
    best_clause = ""
    best_score = 0.0
    for clause in clauses:
        score = overlap_ratio(claim_words, content_words(clause))
        if score > best_score:
            best_score = score
            best_clause = clause
    return best_clause, best_score


def classify_claim(claim: str, source_document: str) -> ClassificationResult:
    claim_words = content_words(claim)
    best_clause, overlap = _best_matching_clause(claim_words, source_document)

    claim_numbers = extract_numbers(claim)
    clause_numbers = extract_numbers(best_clause)
    claim_number_entities = extract_number_entities(claim)
    clause_number_entities = extract_number_entities(best_clause)

    if claim_numbers and clause_numbers:
        if claim_number_entities and clause_number_entities:
            # Entity-aware comparison, checked first: a numeric
            # contradiction requires the clause to state a number for the
            # SAME nearby noun as the claim's number ("12 benches" vs. a
            # clause about "10 tables" is simply a different fact, not a
            # contradiction, even though 12 and 10 are both "numbers in
            # the clause"). If the clause never states a number for that
            # noun at all, the claim's number is unaddressed, not
            # contradicted -- that's decided below, by the
            # numeric-entity-unaddressed check, not here.
            numeric_conflict = False
            for num, entity in claim_number_entities:
                clause_numbers_for_entity = {c_num for c_num, c_entity in clause_number_entities if c_entity == entity}
                if clause_numbers_for_entity and num not in clause_numbers_for_entity:
                    numeric_conflict = True
                    break
        else:
            # Entity extraction found nothing usable on one side (e.g. a
            # number not immediately followed by its noun) -- fall back
            # to the coarser "does this exact digit string appear
            # anywhere in the clause" check.
            numeric_conflict = bool(claim_numbers - clause_numbers)
        if numeric_conflict and overlap >= CONTRADICTION_OVERLAP_THRESHOLD:
            return ClassificationResult(
                label=CONTRADICTED,
                reason="numeric_mismatch",
                best_clause=best_clause,
                overlap=overlap,
                evidence={"claim_numbers": sorted(claim_numbers), "source_numbers": sorted(clause_numbers)},
            )

    if claim_number_entities:
        # Tri-state numeric handling, continued: same entity + different
        # number was CONTRADICTED above; same entity + same number falls
        # through normally below. This covers the third state -- the
        # claim states a number for an entity/noun the source-clause
        # never states ANY number for at all. That claim is unverified,
        # not merely "not contradicted" -- it must not be allowed to
        # become SUPPORTED just because the surrounding sentence shares
        # enough other words (e.g. "The park has 12 benches" against "The
        # park has 10 tables" shares "park"/"has", but the source never
        # gives a number for benches, so the bench count itself is
        # unaddressed evidence, not confirmed evidence).
        clause_entities_present = {entity for _, entity in clause_number_entities}
        unaddressed_entities = sorted({
            entity for _, entity in claim_number_entities
            if entity not in clause_entities_present
        })
        if unaddressed_entities:
            return ClassificationResult(
                label=UNSUPPORTED,
                reason="numeric_entity_unaddressed",
                best_clause=best_clause,
                overlap=overlap,
                evidence={"unaddressed_entities": unaddressed_entities},
            )

    claim_negated = has_negation(claim)
    clause_negated = has_negation(_negation_scope(claim_words, best_clause))
    if claim_negated != clause_negated and overlap >= CONTRADICTION_OVERLAP_THRESHOLD:
        return ClassificationResult(
            label=CONTRADICTED,
            reason="negation_mismatch",
            best_clause=best_clause,
            overlap=overlap,
            evidence={"claim_negated": claim_negated, "source_clause_negated": clause_negated},
        )

    if overlap >= SUPPORT_OVERLAP_THRESHOLD:
        return ClassificationResult(
            label=SUPPORTED,
            reason="lexical_overlap",
            best_clause=best_clause,
            overlap=overlap,
            evidence={},
        )

    return ClassificationResult(
        label=UNSUPPORTED,
        reason="no_matching_clause",
        best_clause=best_clause,
        overlap=overlap,
        evidence={},
    )
