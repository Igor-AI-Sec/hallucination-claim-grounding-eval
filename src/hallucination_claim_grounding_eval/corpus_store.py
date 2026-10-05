"""Holds the fixture corpus as trusted, evaluation-time state.

Anything that holds a "trusted" snapshot a caller could mutate through a
returned reference should defensively copy both on the way in and the way
out -- otherwise a caller holding a reference to the constructor argument,
or to whatever `cases()` returns, can silently rewrite the ground truth
the evaluation is scored against. `CorpusStore` is the one piece of state
in this project that fits that description -- the ground truth the
evaluation run is scored against -- so it copies the case list
on construction and copies it again on every `cases()` call.
"""

from __future__ import annotations

import copy


class CorpusStore:
    def __init__(self, cases: list[dict]):
        self._cases = copy.deepcopy(cases)

    def cases(self) -> list[dict]:
        return copy.deepcopy(self._cases)

    def __len__(self) -> int:
        return len(self._cases)
