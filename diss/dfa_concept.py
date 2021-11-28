from __future__ import annotations

import random
from pprint import pformat
from typing import Callable 

import attr
import dfa
import funcy as fn
import numpy as np
from dfa.utils import find_equiv_counterexample
from pysat.solvers import Minicard
from dfa_identify import find_dfas
from scipy.special import softmax

from diss import State, Path, LabeledExamples, ConceptIdException
from diss.product_mc import MonitorState


__all__ = ['DFAConcept', 'Sensor']


DFA = dfa.DFA
Sensor = Callable[[dfa.State], dfa.Letter] 
ENUM_MAX = 10


def remove_stutter(graph: dfa.DFADict) -> None:
    for state, (_, kids) in graph.items():
        tokens = list(kids.keys())
        kids2 = {k: v for k, v in kids.items() if v != state}
        kids.clear()
        kids.update(kids2)


def count_edges(graph: dfa.DFADict) -> int:
    count = 0
    for _, (_, kids) in graph.items():
        count += sum(1 for k in kids.values()) 
    return count


@attr.frozen
class DFAConcept:
    dfa: dfa.DFA
    size: float
    monitor: MonitorState

    def __hash__(self) -> int:
        return hash(self.dfa)

    def __eq__(self, other) -> bool:
        return self.dfa == other.dfa

    def __repr__(self) -> str:
        graph, start = dfa.dfa2dict(self.dfa)
        remove_stutter(graph)
        return f'{start}\n{pformat(graph)}'

    def seperate(self, other: Concept) -> Path | None:
        if not isinstance(other, DFAConcept):
            raise NotImplementedError
        return find_equiv_counterexample(self.dfa, other.dfa)

    @staticmethod
    def from_examples(
            data: LabeledExamples, 
            filter_pred: Callable[[DFA], bool] = None,
            alphabet: frozenset = None,
            find_dfas=find_dfas,
            temp: float = 1) -> DFAConcept:
        langs = find_dfas(
            data.positive, data.negative, 
            alphabet=alphabet,
            order_by_stutter=True
        )  # type: ignore

        if filter_pred is not None:
            langs = filter(filter_pred, langs)
        langs = fn.take(ENUM_MAX, langs)
        if not langs:
            raise ConceptIdException

        concepts = [DFAConcept.from_dfa(lang) for lang in langs]
        weights = [np.exp(-c.size / temp) for c in concepts]
        return random.choices(concepts, weights)[0]  # type: ignore

    @staticmethod
    def from_dfa(lang: DFA) -> DFAConcept:
        # TODO: Support from graph.
        assert lang.inputs is not None
        assert lang.outputs <= {True, False}

        # Measure size by encoding number of nodes and 
        # number of non-stuttering labeled edges.
        graph, start = dfa.dfa2dict(lang)
        remove_stutter(graph)
        size = np.log(count_edges(graph))

        # Wrap dfa to conform to DFA Monitor API.
        @attr.frozen
        class DFAMonitor:
            state: dfa.State = start

            @property
            def accepts(self) -> bool:
                return lang._label(self.state)

            def update(self, sym: Any) -> DFAMonitor:
                """Assumes stuttering semantics for unknown symbols."""
                if sym not in lang.inputs:
                    return self
                return DFAMonitor(lang._transition(self.state, sym))

        return DFAConcept(lang, size, DFAMonitor())

    def __contains__(self, path: Path) -> bool:
        monitor = self.monitor
        for x in path:
            monitor = monitor.update(x)
        return monitor.accepts  # type: ignore
