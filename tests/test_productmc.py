from collections import Counter

import attr
import dfa

from diss import DemoPrefixTree, Edge, Node, SampledPath, Player
from diss.product_mc import ProductMC, Moves  
from diss.dfa_concept import DFAConcept
from diss import search, GradientGuidedSampler


@attr.frozen
class ExplicitDynamics:
    start: int
    graph: dict[int, Moves]

    def moves(self, state: int) -> Moves:
        return self.graph[state]

    def player(self, state: int) -> Player:
        is_ego = isinstance(self.graph[state], frozenset)
        return 'ego' if is_ego else 'env' 


def test_productmc():
    demos = [[
        (6, 'ego'),
        (5, 'env'),
        (3, 'ego'),
        (1, 'ego'),
    ], [
        (6, 'ego'),
        (5, 'env'),
        (4, 'ego'),
        (2, 'env'),
        (0, 'ego'),
    ]]
    tree = DemoPrefixTree.from_demos(demos)
    dyn = ExplicitDynamics(
        start=6,
        graph={
            6: frozenset((5, 4)),
            5: {4: 1/3, 3: 2/3},
            4: frozenset((2, 0)),
            3: frozenset((2, 1)),
            2: {1: 1/3, 0: 2/3},
            1: frozenset(),
            0: frozenset(),
        }
    )
    empty_lang = dfa.DFA(
        start=False,
        inputs=range(7),
        label=lambda _: False, 
        transition=lambda *_: False,
    )

    bot = DFAConcept.from_dfa(
        lang=empty_lang,
        sensor=lambda s: s,
    )

    top = DFAConcept.from_dfa(
        lang=~empty_lang,
        sensor=lambda s: s,
    )

    chain = ProductMC.construct(
        concept=top, 
        tree=tree,
        dyn=dyn,
        max_depth=None,
    )

    assert Counter(chain.edge_probs.values()) == {
        1/3: 1, 2/3: 2, 1/2: 3
    }
    assert Counter(chain.policy.prob(*e) for e in chain.policy.dag.edges) == {
        1/3: 3, 2/3: 3, 1/2: 8, 1: 5
    }
    sampler = GradientGuidedSampler.from_demos(
        demos=demos,
        to_chain=lambda c, t: ProductMC.construct(
            concept=c, tree=t, dyn=dyn, max_depth=None
        ),
    )

    example1 = sampler(top)
    assert example1.positive == set()
    assert len(example1.negative) == 1

    example2 = sampler(bot)
    assert example2.negative == set()
    assert len(example2.positive) == 1

    example12 = example1 @ example2
    assert len(example12.positive) == len(example12.negative) == 1
