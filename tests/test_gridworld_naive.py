from collections import Counter

import attr
import dfa
import funcy as fn
from dfa.utils import find_subset_counterexample
from dfa_identify import find_dfa

from diss.planners.product_mc import ProductMC
from diss.concept_classes.dfa_concept import DFAConcept
from diss.domains.gridworld_naive import GridWorldNaive as World
from diss.domains.gridworld_naive import GridWorldState as State
from diss import diss, LabeledExamples, GradientGuidedSampler


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

    def ignore_white(path):
        return tuple(x for x in path if x != 'white')

    def subset_check_wrapper(dfa_candidate):
        partial = partial_dfa(dfa_candidate.inputs)
        ce = find_subset_counterexample(dfa_candidate, partial)
        return ce is None

    def to_concept(data):
        data = data.map(ignore_white) @ base_examples

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

        concept = DFAConcept.from_examples(data, filter_pred=subset_check_wrapper) 
        # Adjust description size due to subset knowledge.
        return attr.evolve(concept, size=concept.size / 100)

    dfa_search = diss(
        demos=demos,
        to_concept=to_concept,
        to_chain=lambda c, t, psat: ProductMC.construct(
            concept=c, tree=t, dyn=gw, max_depth=9, 
            psat=psat, sensor=gw.sensor,
        ),
        competency=lambda *_: 0.8,
        lift_path=lambda x: ignore_white(map(gw.sensor, x)),
    )

    data1, concept1, metadata = next(dfa_search)
    path1 = ignore_white(gw.sensor(x) for x, _ in demos[0])
    assert path1 in concept1

    data2, concept2, metadata = next(dfa_search)
