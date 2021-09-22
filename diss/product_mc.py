"""Code for explicit (tabular) construction on product dynamics.""" 
from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol, Union, Optional, cast
from uuid import uuid1

import attr
import networkx as nx

from diss import Edge, Concept, Node, Player, SampledPath, State
from diss import DemoPrefixTree as PrefixTree


oo = float('inf')
EgoMoves = frozenset[State]
EnvMoves = Mapping[State, float]
Moves = Union[EgoMoves, EnvMoves]


class Dynamics(Protocol):
    start: State

    def moves(self, state: State) -> Moves: ...
    def player(self, state: State) -> Player: ...


class MonitorableConcept(Concept):
    monitor: MonitorState


class MonitorState(Protocol):
    state: Any
    accepts: bool

    def update(self, symbol: Any) -> MonitorState: ...


@attr.frozen
class ProductMC:
    dag: nx.DiGraph

    @property
    def edge_probs(self) -> dict[Edge, float]:
        ...

    def sample(self, pivot: Node, win: bool) -> SampledPath:
        ...
 
    @staticmethod
    def construct(
            concept: MonitorableConcept,
            tree: PrefixTree,
            dyn: Dynamics,
            max_depth: Optional[int],
        ) -> ProductMC:
        depth_budget: float = oo if max_depth is None else max_depth
        lose, win = map(str, (uuid1(), uuid1()))  # Unique names for win/lose.

        dag = nx.DiGraph()
        dag.add_node(lose, kind=False)
        dag.add_node(win, kind=True)

        stack = [(dyn.start, concept.monitor, 0)]
        while stack:
            state = stack.pop()
            dstate, mstate, depth = state
            dag.add_node(state, kind=dyn.player(dstate))
            if state in dag.nodes:
                continue

            moves = dyn.moves(dstate)

            if (not moves) or (depth >= depth_budget):
                leaf = win if mstate.accepts else lose
                dag.add_edge(state, leaf, prob=1.0)
                continue

            is_env = dyn.player(dstate) == 'env'
            for dstate2 in moves:
                mstate2 = mstate.update(dstate2)
                state2 = (dstate2, mstate2, depth + 1)
                stack.append((dstate2, mstate2, depth + 1))
                dag.add_edge(state, state2)

                if is_env:
                    moves = cast(EnvMoves, moves)
                    dag.edges[state, state2]['prob'] = moves[dstate2]

        return ProductMC(dag)
