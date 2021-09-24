from collections import Counter

import attr
import dfa

from diss import DemoPrefixTree, Edge, Node, SampledPath, Player
from diss.product_mc import ProductMC, Moves  
from diss.dfa_concept import DFAConcept


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
        sensor=lambda s: 0,
    )

    chain = ProductMC.construct(
        concept=bot, 
        tree=tree,
        dyn=dyn,
        max_depth=None,
    )

    assert Counter(chain.edge_probs.values()) == {
        1/3: 3, 2/3: 3, 1/2: 8, 1.0: 5
    }
