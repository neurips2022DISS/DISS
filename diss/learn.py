from __future__ import annotations

import random
from itertools import combinations
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
    'diss',
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

    def map(self, func: Callable[[Any], Any]) -> LabeledExamples:
        return LabeledExamples(
            positive=map(func, self.positive), 
            negative=map(func, self.negative),
        )


class Concept(Protocol):
    @property
    def size(self) -> float: ...

    def __contains__(self, path: Path) -> bool: ...

    def seperate(self, other: Concept) -> Path: ...


###############################################################################
#                              Guided Search 
###############################################################################

Identify = Callable[[LabeledExamples], Concept]
Competency = float
CompetencyEstimator = Callable[[Concept, PrefixTree], Competency]
MarkovChainFact = Callable[[Concept, PrefixTree, Competency], MarkovChain]
ExampleSampler = Callable[[Concept], tuple[LabeledExamples, dict[str, Any]]]
ExampleSamplerFact = Callable[[Demos], ExampleSampler]


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
    competency: CompetencyEstimator

    @staticmethod
    def from_demos(
            demos: Demos, 
            to_chain: MarkovChainFact, 
            competency: CompetencyEstimator,
    ) -> GradientGuidedSampler:
        tree = PrefixTree.from_demos(demos)
        return GradientGuidedSampler(tree, to_chain, competency)

    def __call__(self, concept: Concept) -> tuple[LabeledExamples, Any]:
        tree = self.tree
        chain = self.to_chain(concept, tree, self.competency(concept, tree))
        grad = surprisal_grad(chain, tree)
        surprisal_val = surprisal(chain, tree)
        meta_data = {'surprisal': surprisal_val, 'grad': grad}

        examples = LabeledExamples()
        while any(grad) > 0:
            weights = [x**2 for x in grad]
            node = random.choices(range(len(grad)), weights)[0]  # Sample node.

            win = grad[node] < 0  # Target label.

            sample = chain.sample(pivot=node, win=not win)
            if sample is None:
                grad[node] = 0  # Don't try this node again.
                continue

            path, sample_prob = sample  
            # Make immutable before sending out example.
            path = tuple(path)

            if win:
                examples @= LabeledExamples(positive=[path])  # type: ignore
            else:
                examples @= LabeledExamples(negative=[path])  # type: ignore
            return examples, meta_data
        raise RuntimeError("Gradient can't be use to guide search?!")


class ConceptIdException(Exception):
    pass


def search(
    demos: Demos, 
    to_concept: Identify,
    sampler_fact: ExampleSamplerFact,
    sensor: Callable[[Any], Any] = lambda x: x,
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
            examples @= new_examples.map(lambda x: tuple(map(sensor, x)))

        except ConceptIdException:
            if example_path:
                examples, concept, metadata = example_path.pop()  # Roll back!
                yield examples, concept, metadata


PathsOfInterest = set[Any]


def reset(
        temp: float,
        poi: set[Any],
        concept2energy: dict[Concept, float],
        cmf_threshold: float = 0.8,
) -> tuple[PathsOfInterest, LabeledExamples, float]:
    poi = set(poi)  # Decouple from input POI.

    # 1. Sort concepts by increasing energy and create energy vector.
    sorted_concepts = sorted(list(concept2energy), key=concept2energy.get)
    energies = np.array([concept2energy[c] for c in sorted_concepts])

    # 2. Turn energy vector into annealed probability mass function.
    pmf = np.exp(-energies / temp)
    pmf /= sum(pmf)  # Normalize.

    # 3. Compute distiguishing strings for top 80%.
    cmf = np.cumsum(pmf)                          # Cummalitive mass function.
    idx = (cmf <= cmf_threshold).sum()    # idx of cutoff.
    to_distinguish = combinations(sorted_concepts[:idx], 2)
    poi |= {c1.seperate(c2) for c1, c2 in to_distinguish}

    # 4. Compute current support's belief on unlabeled strings of interest.
    p_accept = {}
    for word in poi:
        votes = np.array([(word in c) for c in sorted_concepts])
        p_accept[word] = pmf @ votes

    # 5. Set examples based on marginalizing over current concept class.
    positive, negative = set(), set()
    for x, belief in p_accept.items():
        confidence = 2*(belief - 0.5 if belief > 0.5 else 0.5 - belief)
        if np.random.rand() > confidence:
            continue
        elif belief < 0.5:
            negative.add(x)
        elif belief > 0.5:
            positive.add(x)
    examples = LabeledExamples(positive, negative)  
    return poi, examples, float('inf')


def diss(
    demos: Demos, 
    to_concept: Identify,
    to_chain: MarkovChainFact,
    competency: CompetencyEstimator,
    lift_path: Callable[[Path], Path] = lambda x: x,
    n_iters: int = 25,
    reset_period: int = 5,
    cooling_schedule: Callable[[int], float] | None = None,
    size_weight: float = 1,
    surprise_weight: float = 1,
    cmf_threshold: float = 0.8,
) -> Iterable[tuple[LabeledExamples, Optional[Concept]]]:
    """Perform demonstration informed gradiented guided search."""
    if cooling_schedule is None:
        def cooling_schedule(t: int) -> float:
            return 30*(1 - t / n_iters) + 1

    sggs = GradientGuidedSampler.from_demos(
        demos=demos,
        to_chain=to_chain,
        competency=competency,
    )

    weights = np.array([size_weight, surprise_weight])

    poi = set()            # Paths of interest.
    concept2energy = {}    # Concepts seen so far + associated energies.
    new_data = LabeledExamples()
    for t in range(n_iters):
        temp = cooling_schedule(t)

        if t % reset_period == 0:
            poi, examples, energy = reset(
                temp, poi, concept2energy, cmf_threshold
            )

        # Sample from proposal distribution.
        proposed_examples = examples @ new_data
        try:
            concept = to_concept(proposed_examples)
        except ConceptIdException:
            new_data = LabeledExamples()  # Reject: New data caused problem. 
            continue

        new_data, metadata = sggs(concept)
        new_data = new_data.map(lift_path)
        new_energy = weights @ [concept.size, metadata['surprisal']]

        metadata |= {'energy': new_energy, 'conjecture': new_data, 'poi': poi}
        yield (proposed_examples, concept, metadata)

        # DISS Bookkeeping for resets.
        poi |= new_data.positive | new_data.negative
        concept2energy[concept] = new_energy

        # Accept/Reject proposal based on energy delta.
        dE = new_energy - energy
        if (dE < 0) or (np.exp(-dE / temp) < np.random.rand()): 
            energy = new_energy
            examples = proposed_examples   # Accept.
        else:
            new_data = LabeledExamples()   # Reject.
