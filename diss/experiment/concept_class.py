from __future__ import annotations
from typing import Any, Optional, Sequence

import attr
import funcy as fn
import dfa
import numpy as np
from functools import lru_cache
from dfa import DFA
from dfa.utils import find_subset_counterexample, find_equiv_counterexample
from dfa_identify import find_dfa, find_dfas

from diss import LabeledExamples, ConceptIdException
from diss.concept_classes.dfa_concept import DFAConcept


__all__ = ['to_concept', 'ignore_white']


def transition(s, c):
    if c == 'red':
        return s | 0b01
    elif c == 'yellow':
        return s | 0b10
    return s


ALPHABET = frozenset({'red', 'yellow', 'blue', 'green'})


PARTIAL_DFA =  DFA(
    start=0b00,
    inputs=ALPHABET,
    label=lambda s: s == 0b10,
    transition=transition
)


def ignore_white(path):
    return tuple(x for x in path if x != 'white')


def subset_check_wrapper(dfa_candidate):
    partial = partial_dfa(dfa_candidate.inputs)
    return find_subset_counterexample(dfa_candidate, partial) is None



BASE_EXAMPLES = LabeledExamples(
    positive=[
        ('yellow',),
        ('yellow', 'yellow'),
    ],
    negative=[
        (),
        ('blue',),
        ('blue', 'blue'),
        ('blue', 'green'),
        ('blue', 'red'),
        ('blue', 'red', 'green'),
        ('blue', 'red', 'green', 'yellow'),
        ('blue', 'red', 'yellow'),
        ('red',),
        ('red', 'blue'),
        ('red', 'blue', 'yellow'),
        ('red', 'green'),
        ('red', 'green', 'green'),
        ('red', 'green', 'green', 'yellow'),
        ('red', 'green', 'yellow'),
        ('red', 'red'),
        ('red', 'red', 'green'),
        ('red', 'red', 'green', 'yellow'),
        ('red', 'red', 'yellow'),
        ('red', 'yellow'),
        ('red', 'yellow', 'green'),
        ('red', 'yellow', 'green', 'yellow'),
        ('yellow', 'red'),
        ('yellow', 'red', 'green'),
        ('yellow', 'red', 'green', 'yellow'),
        ('yellow', 'red', 'yellow'),
        ('yellow', 'yellow', 'red')
    ]
)


@lru_cache
def find_dfas2(accepting, rejecting, alphabet, order_by_stutter=False):
    dfas = find_dfas(
        accepting,
        rejecting,
        alphabet=alphabet,
        order_by_stutter=order_by_stutter,
    )
    return fn.take(5, dfas)


@lru_cache
def augment(self: PartialDFAIdentifier, data: LabeledExamples) -> LabeledExamples:
    data = data.map(ignore_white) @ self.base_examples

    for i in range(20):
        tests = find_dfas2(
            data.positive,
            data.negative,
            order_by_stutter=True,
            alphabet=self.partial.dfa.inputs,
        )
        new_data = LabeledExamples()
        for test in tests:
            assert test is not None
            ce = self.subset_ce(test)
            if ce is None:
                continue
            new_data @= LabeledExamples(negative=[ce])
            partial = self.partial_dfa(test.inputs)
            for k, lbl in enumerate(partial.transduce(ce)):
                prefix = ce[:k]
                if not lbl:
                    new_data @= LabeledExamples(negative=[prefix])
            data @= new_data
            if new_data.size == 0:
                break
    return data



@attr.frozen
class PartialDFAIdentifier:
    partial: DFAConcept = attr.ib(converter=DFAConcept.from_dfa)
    base_examples: LabeledExamples = LabeledExamples()

    def partial_dfa(self, inputs) -> DFA:
        assert inputs <= self.partial.dfa.inputs
        return attr.evolve(self.partial.dfa, inputs=inputs)

    def subset_ce(self, candidate: DFA) -> Optional[Sequence[Any]]:
        partial = self.partial_dfa(candidate.inputs)
        return find_subset_counterexample(candidate, partial)

    def is_subset(self, candidate: DFA) -> Optional[Sequence[Any]]:
        return self.subset_ce(candidate) is None

    def __call__(self, data: LabeledExamples) -> DFAConcept:
        data = augment(self, data)

        concept = DFAConcept.from_examples(
            data=data,
            filter_pred=self.is_subset,
            alphabet=self.partial.dfa.inputs,
            find_dfas=find_dfas2,
        ) 
        # Adjust size to account for subset information.
        return attr.evolve(concept, size=concept.size - self.partial.size)
