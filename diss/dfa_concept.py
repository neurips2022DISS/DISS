from __future__ import annotations

import random
from pprint import pformat
from typing import Callable 

import attr
import dfa
import funcy as fn
import numpy as np
from pysat.solvers import Minicard
from dfa_identify import find_dfas
from scipy.special import softmax

from diss import State, Path, LabeledExamples, ConceptIdException
from diss.product_mc import MonitorState


__all__ = ['DFAConcept', 'Sensor']


DFA = dfa.DFA
Sensor = Callable[[dfa.State], dfa.Letter] 
TEMP = 20
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
    sensor: Sensor
    size: float
    monitor: MonitorState

    def __repr__(self) -> str:
        graph, start = dfa.dfa2dict(self.dfa)
        remove_stutter(graph)
        return f'{start}\n{pformat(graph)}'

    @staticmethod
    def from_examples(
            data: LabeledExamples, 
            sensor: Sensor, 
            filter_pred: Callable[[DFA], bool] = None,
            alphabet: frozenset = None,
            find_dfas=find_dfas) -> DFAConcept:
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

        concepts = [DFAConcept.from_dfa(lang, sensor) for lang in langs]
        weights = [np.exp(-c.size / TEMP) for c in concepts]
        return random.choices(concepts, weights)[0]  # type: ignore
  
    @staticmethod
    def from_dfa(lang: DFA, sensor: Sensor) -> DFAConcept:
        # TODO: Support from graph.
        assert lang.inputs is not None
        assert lang.outputs <= {True, False}

        # Measure size by encoding number of nodes and 
        # number of non-stuttering labeled edges.
        graph, start = dfa.dfa2dict(lang)
        remove_stutter(graph)
        state_bits = np.log2(len(graph))
        n_edges = count_edges(graph)
        size = state_bits * (1 + 2 * n_edges * np.log2(len(lang.inputs))) + 1

        # Wrap graph dfa to conform to DFA Monitor API.
        @attr.frozen
        class DFAMonitor:
            state: dfa.State = start

            @property
            def accepts(self) -> bool:
                return graph[self.state][0]

            def update(self, state: State) -> DFAMonitor:
                """Assumes stuttering semantics for unknown transitions."""
                symbol = sensor(state)
                transitions = graph[self.state][1]
                return DFAMonitor(transitions.get(symbol, self.state))

        return DFAConcept(lang, sensor, size, DFAMonitor())

    def __contains__(self, path: Path) -> bool:
        monitor = self.monitor
        for x in path:
            monitor = monitor.update(x)
        return monitor.accepts  # type: ignore
