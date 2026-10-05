"""Deterministic claim segmentation.

Splits a free-text answer (or source document) into individual claim
strings using fixed punctuation rules only -- no NLP model, no parsing of
grammatical structure. This is a known simplification: it will not
correctly split every possible compound sentence (e.g. clauses joined by
words other than "and"/"but", or joined without a comma/semicolon at all),
and that limitation is intentional and documented in the project README
rather than hidden.
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT_RE = re.compile(r",\s+and\s+|;\s+|\s+but\s+")


def split_sentences(text: str) -> list[str]:
    """Split `text` into sentences on '.', '!', or '?' followed by whitespace."""
    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def segment(text: str) -> list[str]:
    """Split `text` into individual claims.

    First splits into sentences, then further splits each sentence on a
    fixed set of clause connectors (", and ", "; ", " but ") so a single
    sentence carrying two independent facts becomes two claims. Trailing
    sentence-ending punctuation is stripped from each resulting claim.
    """
    claims: list[str] = []
    for sentence in split_sentences(text):
        for part in _CLAUSE_SPLIT_RE.split(sentence):
            part = part.strip().rstrip(".!?,").strip()
            if part:
                claims.append(part)
    return claims
