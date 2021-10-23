from __future__ import annotations

import random
from pprint import pformat
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

import attr
import numpy as np

from diss import AnnotatedMarkovChain as MarkovChain
from diss import Node, Demos, Path
from diss import DemoPrefixTree as PrefixTree


__all__ = [
    'Concept', 
    'ConceptIdException',
    'ExampleSamplerFact', 
    'LabeledExamples', 
    'Identify', 
    'GradientGuidedSampler',
    'search',
]


Examples = frozenset[Any]


@attr.frozen
class LabeledExamples:
    positive: Examples = attr.ib(converter=frozenset, factory=frozenset)
    negative: Examples = attr.ib(converter=frozenset, factory=frozenset)

    @property
    def size(self) -> int:
        return self.dist(LabeledExamples())

    def __repr__(self) -> str:
        pos, neg = set(self.positive), set(self.negative)
        return f'+: {pformat(pos)}\n--------------\n-: {pformat(neg)}'

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
    @property
    def size(self) -> float: ...

    def __contains__(self, path: Path) -> bool: ...


###############################################################################
#                              Guided Search 
###############################################################################

Identify = Callable[[LabeledExamples], Concept]
MarkovChainFact = Callable[[Concept, PrefixTree], MarkovChain]
ExampleSamplerFact = Callable[
    [Demos],  # Concept, PrefixTree, max_len
    Callable[[Concept], tuple[LabeledExamples, float]]
]


def surprisal_grad(chain: MarkovChain, tree: PrefixTree) -> list[float]:
    conform_prob: float
    dS: list[float]
    # TODO: Remove recursion and base on numpy.

    dS = (max(tree.nodes()) + 1) * [0.0]
    edge_probs = chain.edge_probs 
    deviate_probs: dict[int, float] = {}
    for n in tree.nodes():
        kids = tree.tree.neighbors(n)
        conform_prob = sum(edge_probs[n, k] for k in kids)
        deviate_probs[n] = 1 - conform_prob 


    def compute_dS(node: Node) -> dict[int, float]:
        reach_probs: dict[int, float]
        kids = list(tree.tree.neighbors(node))

        # Compute recursive reach probabilities.
        reach_probs = {node: 1}
        for k in tree.tree.neighbors(node):
            reach_probs.update(compute_dS(k).items())

        parent = tree.parent(node)
        if parent is None:  # Root doesn't do anything.
            return reach_probs
 
        # Take into account traversing edge.
        edge_prob = edge_probs[parent, node]
        for node2 in reach_probs:
            reach_probs[node2] *= edge_prob

        if not tree.is_ego(parent):  # Ignore non-decision edges for dS.
            return reach_probs
      
        # Conform contribution.
        for node2, reach_prob in reach_probs.items():
            weight = tree.count(node) * (1 / edge_prob - 1) * reach_prob
            if not tree.is_leaf(node2):
                weight *= deviate_probs[node2]
            dS[node2] -= weight 

        # Deviate contribution.
        dS[parent] += tree.count(parent) * deviate_probs[parent]

        return reach_probs
    
    compute_dS(0)
     
    # Zero out any exhausted nodes.
    return list(dS)


def surprisal(chain: MarkovChain, tree: PrefixTree) -> float:
    edge_probs = chain.edge_probs
    surprise = 0
    for (node, move), edgep in edge_probs.items():
        if not tree.is_ego(node):
            continue
        surprise -= tree.count(move) * np.log(edgep)
    return surprise 


@attr.define
class GradientGuidedSampler:
    tree: PrefixTree
    to_chain: MarkovChainFact
    beta: float = 1.0

    @staticmethod
    def from_demos(demos: Demos, to_chain: MarkovChainFact, beta: float=1.0) -> GradientGuidedSampler:
        tree = PrefixTree.from_demos(demos)
        return GradientGuidedSampler(tree, to_chain, beta)

    def __call__(self, concept: Concept) -> tuple[LabeledExamples, Any]:
        tree = self.tree
        chain = self.to_chain(concept, tree)
        grad = surprisal_grad(chain, tree)
        surprisal_val = surprisal(chain, tree)

        examples = LabeledExamples()
        N = np.random.geometric(self.beta)
        while any(grad) > 0:
            weights = [abs(x) for x in grad]
            node = random.choices(range(len(grad)), weights)[0]  # Sample node.

            win = grad[node] < 0  # Target label.

            sample = chain.sample(pivot=node, win=not win)
            if sample is None:
                grad[node] = 0  # Don't try this node again.
                continue

            path, sample_prob = sample  
            path = tuple(path) # Make immutable before sending out example.

            if win:
                examples @= LabeledExamples(positive=[path])  # type: ignore
            else:
                examples @= LabeledExamples(negative=[path])  # type: ignore
            sample_prob *= weights[node] / sum(weights)

            if N <= 1:
                return examples, {"sample_prob": sample_prob, "surprisal": surprisal_val}
            else:
                N -= 1
        raise RuntimeError("Gradient can't be use to guide search?!")


class ConceptIdException(Exception):
    pass


def search(
    demos: Demos, 
    to_concept: Identify,
    sampler_fact: ExampleSamplerFact,
) -> Iterable[tuple[LabeledExamples, Optional[Concept]]]:
    """Perform demonstration informed gradiented guided search."""
    example_sampler = sampler_fact(demos)

    examples = LabeledExamples()
    example_path = []
    while True:
        try:
            concept = to_concept(examples)
            new_examples, metadata = example_sampler(concept)
            example_path.append((examples, concept, metadata))
            yield examples, concept, metadata
            examples @= new_examples

        except ConceptIdException:
            if example_path:
                examples, concept, metadata = example_path.pop()  # Roll back!
                yield examples, concept, metadata
