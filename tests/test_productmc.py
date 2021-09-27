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


def assert_consistent(data, concept):
    if concept is None:
        return
    assert all(x in concept for x in data.positive)
    assert not any(x in concept for x in data.negative)


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
        inputs={True, False},
        label=lambda _: False, 
        transition=lambda *_: False,
    )

    def sensor(s):
        return s == 1

    bot = DFAConcept.from_dfa(lang=empty_lang, sensor=sensor)
    top = DFAConcept.from_dfa(lang=~empty_lang, sensor=sensor)

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

    def sampler_factory(demos):
        return GradientGuidedSampler.from_demos(
            demos=demos,
            to_chain=lambda c, t: ProductMC.construct(
                concept=c, tree=t, dyn=dyn, max_depth=None
            ),
        )

    sampler = sampler_factory(demos)

    data1, prob = sampler(top)
    assert 0 <= prob <= 1
    assert data1.positive == set()
    assert len(data1.negative) == 1

    data2, prob = sampler(bot)
    assert 0 <= prob <= 1
    assert data2.negative == set()
    assert len(data2.positive) == 1

    data3 = data1 @ data2
    assert len(data3.positive) == len(data3.negative) == 1

    def to_concept(examples):
        if examples.size == 0:
            return bot
        return DFAConcept.from_examples(examples, sensor)
 
    dfa_search = search(demos, to_concept, sampler_factory)
    data1, concept1 = next(dfa_search)
    assert_consistent(data1, concept1)

    data2, concept2 = next(dfa_search)
    assert_consistent(data2, concept2)

    data3, concept3 = next(dfa_search)
    assert_consistent(data3, concept3)

    data4, concept4 = next(dfa_search)
    assert_consistent(data4, concept4)
