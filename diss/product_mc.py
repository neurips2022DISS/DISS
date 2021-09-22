"""Code for explicit (tabular) construction on product dynamics.""" 
from __future__ import annotations

import random
from typing import Any, Iterable, Mapping, Protocol, Optional, Sequence, Union
from typing import cast
from uuid import uuid1

import attr
import networkx as nx
import numpy as np

from diss import Edge, Concept, Node, Player, SampledPath, State
from diss import DemoPrefixTree as PrefixTree
from diss.tabular import TabularPolicy 


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


def product_dag(
        concept: MonitorableConcept,
        tree: PrefixTree,
        dyn: Dynamics,
        max_depth: Optional[int],
    ) -> nx.DiGraph:
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

    return dag


def empirical_psat(tree: PrefixTree, concept: Concept) -> float:
    # TODO: Use monitor...
    leaves = (tree.is_leaf(n) for n in tree.nodes())
    accepted = total = 0
    for leaf in leaves:
        demo = tree.prefix(leaf)
        count = tree.count(leaf)
        total += count
        accepted = (demo in concept) * count
    return accepted / total


@attr.frozen
class ProductMC:
    tree: PrefixTree
    concept: MonitorableConcept
    policy: TabularPolicy 

    @property
    def edge_probs(self) -> dict[Edge, float]:
        edges = cast(Iterable[Edge], self.policy.dag.edges)
        return {e: self.policy.prob(*e) for e in edges}

    def sample(self, pivot: Node, win: bool) -> SampledPath:
        policy = self.policy
        if policy.psat(pivot) == 0:
            return None  # Impossible to realize is_sat label.

        path = list(self.tree.prefix(pivot))
        mstate = self.concept.monitor
        for dstate in path:
            mstate = mstate.update(dstate)
        state = (dstate, mstate, len(path))

        sample_prob: float = 1
        while (moves := list(policy.dag.neighbors(state))):
            # Apply bayes rule to get Pr(s' | is_sat, s).
            priors = np.array([policy.prob(state, m) for m in moves])
            likelihoods = np.array([policy.psat(m) for m in moves])
            normalizer = policy.psat(state)

            if not win:
                likelihoods = 1 - likelihoods
                normalizer = 1 - normalizer

            probs = cast(Sequence[float], priors * likelihoods / normalizer)
            prob, state = random.choices(list(zip(probs, moves)), probs)[0]
            sample_prob *= prob
            path.append(state)
        return path, sample_prob
 
    @staticmethod
    def construct(
            concept: MonitorableConcept,
            tree: PrefixTree,
            dyn: Dynamics,
            max_depth: Optional[int],
        ) -> ProductMC:
        """Constructs a tabular policy by unrolling of dynamics/concept."""
        dag = product_dag(concept, tree, dyn, max_depth)
        psat = empirical_psat(tree, concept)

        return ProductMC(
                tree=tree,
                concept=concept,
                policy=TabularPolicy.from_psat(dag, psat),
        )
