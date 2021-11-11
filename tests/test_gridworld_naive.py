from collections import Counter

import dfa
import funcy as fn
from dfa.utils import find_subset_counterexample
from dfa_identify import find_dfa

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
            competency=lambda c, t: 0.8,
            to_chain=lambda c, t, psat: ProductMC.construct(
                concept=c, tree=t, dyn=gw, max_depth=9, psat=psat, sensor=gw.sensor
            ),
        )

    base_examples = LabeledExamples(
        positive=[
            ('yellow',),
            ('yellow', 'yellow'),
            #('blue', 'green', 'yellow'), # Demo
        ],
        negative=[
            (), ('red',), ('red', 'red'),
            ('red', 'yellow'), ('yellow', 'red'),
            ('yellow', 'red', 'yellow'),
            ('yellow', 'yellow', 'red'),
        ]
    )

    def partial_dfa(inputs):
        def transition(s, c):
            if c == 'red':
                return s | 0b01
            elif c == 'yellow':
                return s | 0b10
            return s

        return dfa.DFA(
            start=0b00,
            inputs=inputs,
            label=lambda s: s == 0b10,
            transition=transition
        )

    def trace(path):
        return tuple(x for x in map(gw.sensor, path) if x != 'white')

    def subset_check_wrapper(dfa_candidate):
        partial = partial_dfa(dfa_candidate.inputs)
        ce = find_subset_counterexample(dfa_candidate, partial)
        return ce is None

    def to_concept(data):
        data = data.map(trace)
        data @= base_examples

        # CEGIS for subset.
        for i in range(20):
            mydfa = find_dfa(data.positive, data.negative, order_by_stutter=True) 
            partial = partial_dfa(mydfa.inputs)
            ce = find_subset_counterexample(mydfa, partial)
            if ce is None:
                break
            data @= LabeledExamples(negative=[ce])

            partial = partial_dfa(mydfa.inputs)
            for k, lbl in enumerate(partial.transduce(ce)):
                prefix = ce[:k]
                if not lbl:
                    data @= LabeledExamples(negative=[prefix])

        return DFAConcept.from_examples(data, filter_pred=subset_check_wrapper) 

    dfa_search = search(demos, to_concept, sampler_factory)

    data1, concept1, metadata = next(dfa_search)
    path1 = [x for x, _ in demos[0]]
    assert trace(path1) in concept1

    data2, concept2, metadata = next(dfa_search)
    data3, concept3, metadata = next(dfa_search)

    # TODO: Try DISS
    # TODO: Remove sensor from to_concept.
