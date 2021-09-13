from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Protocol

import attr


Examples = frozenset[Any]


@attr.frozen
class LabeledExamples:
    positive: Examples = attr.ib(converter=frozenset, factory=frozenset)
    negative: Examples = attr.ib(converter=frozenset, factory=frozenset)

    def __contains__(self, val: Any) -> Optional[bool]:
        if val in self.positive:
            return True
        elif val in self.negative:
            return False

    @property
    def size(self):
        return self.dist(LabeledExamples())

    def __matmul__(self, other: LabeledExamples) -> LabeledExamples:
        return LabeledExamples(
            positive=(self.positive - other.negative) | other.positive,
            negative=(self.negative - other.positive) | other.negative,
        )

    def dist(self, other: LabeledExamples) -> int:
        pos_delta = self.positive ^ other.positive
        neg_delta = self.negative ^ other.negative
        return len(pos_delta) + len(neg_delta) - len(pos_delta & neg_delta)


class Concept(Protocol):
    size: int

    def __contains__(self, val: Any) -> bool:
        ...
