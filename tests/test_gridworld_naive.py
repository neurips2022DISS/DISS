from collections import Counter

import dfa
import funcy as fn

from diss.product_mc import ProductMC
from diss.dfa_concept import DFAConcept
from diss.domains.gridworld_naive import GridWorldNaive as World
from diss.domains.gridworld_naive import GridWorldState as State
from diss import search, LabeledExamples, GradientGuidedSampler


def test_gridworld_smoke():
    gw = World(
        dim=3,
        start=State(x=3, y=1),
        overlay={
          (1, 1): 'yellow',
          (1, 2): 'green',
          (1, 3): 'green',
          (2, 3): 'red',
          (3, 2): 'blue',
          (3, 3): 'blue',
        }
    )

    print()
    state = gw.start
    print(gw.to_string(state))
    assert gw.player(state) == 'ego'
    assert len(gw.moves(state)) == 2

    move_hist = Counter(len(gw.moves(m)) for m in gw.moves(state))
    assert move_hist == {1: 1, 2: 1}
    assert all(gw.player(m) == 'env' for m in gw.moves(state))

    state = State(x=2, y=2)
    assert gw.player(state) == 'ego'
    assert len(gw.moves(state)) == 4
 
    demos = [[
       (State(3, 1), 'ego'),
       (State(3, 1, '←'), 'env'),
       (State(3, 2), 'ego'),
       (State(3, 2, '←'), 'env'),
       (State(2, 2), 'ego'),
       (State(2, 2, '←'), 'env'),
       (State(1, 2), 'ego'),
       (State(1, 2, '↑'), 'env'),
       (State(1, 1), 'ego'),
    ]]
    print(gw.to_string(state))
    def sampler_factory(demos):
        return GradientGuidedSampler.from_demos(
            demos=demos,
            to_chain=lambda c, t: ProductMC.construct(
                concept=c, tree=t, dyn=gw, max_depth=9, psat=0.8
            ),
        )

    base_examples = LabeledExamples(
        positive=[
            ('yellow',),
            ('yellow', 'yellow'),
            ('blue', 'green', 'yellow'),
        ],
        negative=[
            (), ('red',), ('red', 'red'),
            ('red', 'yellow'), ('yellow', 'red'),
            ('yellow', 'yellow', 'red'),
        ]
    )

    def trace(path):
        return tuple(x for x in map(gw.sensor, path) if x != 'white')

    def to_concept(data):
        data = LabeledExamples(
            positive = [trace(x) for x in data.positive],
            negative = [trace(x) for x in data.negative],
        )
 
        data @= base_examples
        return DFAConcept.from_examples(data, gw.sensor)

    to_concept(LabeledExamples())
 
    dfa_search = search(demos, to_concept, sampler_factory)

    data1, concept1 = next(dfa_search)
    path1 = [x for x, _ in demos[0]]
    assert path1 in concept1

    data2, concept2 = next(dfa_search)
    data3, concept3 = next(dfa_search)
    data4, concept4 = next(dfa_search)
    data5, concept5 = next(dfa_search)
    breakpoint()
