from typing import Callable 

import attr
import dfa
import numpy as np

from diss import State, Path
from diss.product_mc import MonitorState


__all__ = ['DFAConcept', 'Sensor']


DFA = dfa.DFA
Sensor = Callable[[dfa.State], dfa.Letter] 


def count_nonstuttering(graph: dfa.DFADict) -> int:
    count = 0
    for state, kids in graph.items():
        count += sum(1 for k in kids if k != state) 
    return count


@attr.frozen
class DFAConcept:
    dfa: dfa.DFA
    sensor: Sensor
    size: float
    monitor: MonitorState
  
    @staticmethod
    def from_dfa(lang: DFA, sensor: Sensor) -> DFAConcept:
        # TODO: Support from graph.
        assert lang.inputs is not None
        assert lang.outputs == {True, False}

        # Measure size by encoding number of nodes and 
        # number of non-stuttering labeled edges.
        graph, start = dfa.dfa2dict(lang)
        state_bits = np.log2(len(graph))
        n_edges = count_nonstuttering(graph)
        size = state_bits * (1 + 2 * n_edges * np.log2(len(lang.inputs)))

        # Wrap graph dfa to conform to DFA Monitor API.
        @attr.frozen
        class DFAMonitor:
            state: dfa.State = start

            @property
            def accepts(self) -> bool:
                return graph[self.state][0]

            def update(self, state: State) -> dfa.State:
                symbol = sensor(state)
                return graph[self.state][1][symbol]

        return DFAConcept(lang, sensor, size, DFAMonitor())

    def __contains__(self, path: Path) -> bool:
        word = list(map(self.sensor, path))
        return self.dfa.label(self.sensor(path))
