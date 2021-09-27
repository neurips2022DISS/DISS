from __future__ import annotations

import random
from typing import Callable 

import attr
import dfa
import funcy as fn
import numpy as np
from dfa_identify import find_dfas
from scipy.special import softmax

from diss import State, Path, LabeledExamples, ConceptIdException
from diss.product_mc import MonitorState


__all__ = ['DFAConcept', 'Sensor']


DFA = dfa.DFA
Sensor = Callable[[dfa.State], dfa.Letter] 
TEMP = 3e-1
ENUM_MAX = 40


def count_nonstuttering(graph: dfa.DFADict) -> int:
    count = 0
    for state, (_, kids) in graph.items():
        count += sum(1 for k in kids.values() if k != state) 
    return count


@attr.frozen
class DFAConcept:
    dfa: dfa.DFA
    sensor: Sensor
    size: float
    monitor: MonitorState

    def __repr__(self) -> str:
        return str(dfa.dfa2dict(self.dfa))

    @staticmethod
    def from_examples(data: LabeledExamples, sensor: Sensor) -> DFAConcept:
        # Convert to correct alphabet.
        pos = [list(map(sensor, x)) for x in data.positive]
        neg = [list(map(sensor, x)) for x in data.negative]
        langs = fn.take(ENUM_MAX, find_dfas(pos, neg))
        if not langs:
            raise ConceptIdException

        concepts = [DFAConcept.from_dfa(lang, sensor) for lang in langs]
        weights = softmax([-c.size / TEMP for c in concepts])
        return random.choices(concepts, weights)[0]  # type: ignore
  
    @staticmethod
    def from_dfa(lang: DFA, sensor: Sensor) -> DFAConcept:
        # TODO: Support from graph.
        assert lang.inputs is not None
        assert lang.outputs <= {True, False}

        # Measure size by encoding number of nodes and 
        # number of non-stuttering labeled edges.
        graph, start = dfa.dfa2dict(lang)
        state_bits = np.log2(len(graph))
        n_edges = count_nonstuttering(graph)
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
        word = list(map(self.sensor, path))
        return self.dfa.label(word)
