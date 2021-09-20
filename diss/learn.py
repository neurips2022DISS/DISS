from __future__ import annotations

import random
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

import attr
import numpy as np

from diss import AnnotatedMarkovChain as MarkovChain
from diss import Node, Demos, Path
from diss import DemoPrefixTree as PrefixTree


Examples = frozenset[Any]


@attr.frozen
class LabeledExamples:
    positive: Examples = attr.ib(converter=frozenset, factory=frozenset)
    negative: Examples = attr.ib(converter=frozenset, factory=frozenset)

    @property
    def size(self) -> int:
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

Identify = Callable[[LabeledExamples], Concept]
MarkovChainFact = Callable[[Concept, PrefixTree, int], MarkovChain]
ExampleSamplerFact = Callable[
    [Demos],  # Concept, PrefixTree, max_len
    Callable[[Concept], LabeledExamples]
]


def surprisal_grad(chain: MarkovChain, tree: PrefixTree) -> list[float]:
    edge_probs = chain.edge_probs 
    dS: list[float] = max(tree.nodes()) * [0.0]

    def compute_dS(node: Node) -> dict[Node, float]:
        kids = tree.tree.neighbors(node)
        if not kids:
            return {node: 1}

        reach_probs = {}
        for kid in kids:
            pkid = reach_probs[kid] = edge_probs[kid]
            dS[kid] -= pkid * tree.count(kid)  # Deviate contribution.

            for node2, reachp in compute_dS(kid).items():
                reachp = reach_probs[node2] = pkid * reachp
                dS[node2] += (1 / pkid - 1) * reachp

        return reach_probs
     
    # Zero out any exhausted nodes.
    for node in tree.nodes():
        if tree.is_leaf(node) or tree.unused_moves(node):
            continue
        dS[node] = 0

    return list(dS)


@attr.define
class GradientGuidedSampler:
    tree: PrefixTree
    to_chain: MarkovChainFact
    max_len: int
    prev: Optional[LabeledExamples] = None
    prev_prob: float = 1e-1  # Prob of outputting prev example.

    @staticmethod
    def from_demos(demos: Demos, to_chain: MarkovChainFact, max_len: int) -> GradientGuidedSampler:
        tree = PrefixTree.from_demos(demos)
        if max_len is None:
            max_len = tree.max_len
        return GradientGuidedSampler(tree, to_chain, max_len)

    def __call__(self, concept: Concept) -> LabeledExamples:
        if (self.prev is not None) and (random.random() < self.prev_prob):
            return self.prev
        tree, max_len = self.tree, self.max_len
        chain = self.to_chain(concept, tree, self.max_len)
        grad = surprisal_grad(chain, tree)
        while sum(grad) > 0:
            node = random.choices(range(len(grad)), grad)[0]  # Sample node.

            win = grad[node] > 0  # Target label.
            sample = chain.sample(pivot=node, max_len=max_len, win=not win)
            if sample is None:
                grad[node] = 0  # Don't try this node again.
                continue

            path, _ = sample  # Currently don't use sample probability.
            path = tuple(path) # Make immutable before sending out example.

            if win:
                return LabeledExamples(positive=[path])  # type: ignore
            else:
                return LabeledExamples(negative=[path])  # type: ignore
        raise RuntimeError("Gradient can't be use to guide search?!")


def search(
    demos: Demos, 
    to_concept: Identify,
    sample_fact: ExampleSamplerFact,
) -> Iterable[Concept]:
    """Perform demonstration informed gradiented guided search."""
    example_sampler = sample_fact(demos)

    examples = LabeledExamples()
    while True:
        concept = to_concept(examples)
        yield concept
        examples @= example_sampler(concept)
