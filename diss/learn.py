from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Protocol

import attr

from diss import AnnotatedMarkovChain as MarkovChain
from diss import Demos, Path
from diss.prefix_tree import DemoPrefixTree as PrefixTree


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


###############################################################################
#                              Guided Search 
###############################################################################

Advice = tuple[
    float,               # Probability to deviate.
    dict[Path, float],   # Distribution over deviate prefixes.
    dict[Path, float],   # Dittribution over conform prefixes.
]
Concept2MC = Callable[[Concept, PrefixTree], MarkovChain]
Concepts = Iterable[Concept]
Identify = Callable[[LabeledExamples], Concept]


def search(
    demos: Demos, 
    to_chain: Concept2MC, 
    to_concept: Identify,
) -> Concepts:
    """Perform demonstration informed gradiented guided search."""
    tree = PrefixTree.from_demos(demos)
    example_state = LabeledExamples()

    while True:
        concept = to_concept(example_state)
        yield concept
        
        chain = to_markov_chain(concept)

        # TODO: Calculate gradient based advice.
        # TODO: Sample whether to deviate or conform.
        # TODO: If deviate, use prefix tree to change prefix to deviation.
        # TODO: Update example state.
