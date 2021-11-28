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


__all__ = [
    'EgoMoves', 'EnvMoves', 'Dynamics', 'Moves', 
    'MonitorableConcept', 'MonitorState', 'ProductMC'
]


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
    @property
    def state(self) -> Any: ...

    @property
    def accepts(self) -> Any: ...

    def update(self, symbol: Any) -> MonitorState: ...


def product_dag(
        concept: MonitorableConcept,
        tree: PrefixTree,
        dyn: Dynamics,
        max_depth: Optional[int],
        sensor: Callable[[State], Any],
    ) -> nx.DiGraph:
    depth_budget: float = oo if max_depth is None else max_depth
    lose, win = map(str, (uuid1(), uuid1()))  # Unique names for win/lose.

    dag = nx.DiGraph()
    stack = [(dyn.start, concept.monitor, 0)]
    while stack:
        state = stack.pop()
        dstate, mstate, depth = state
        dag.add_node(state, kind=dyn.player(dstate))
        moves = dyn.moves(dstate)

        if (not moves) or (depth >= depth_budget):
            leaf = win if mstate.accepts else lose
            dag.add_node(leaf, kind=mstate.accepts)
            dag.add_edge(state, leaf, prob=1.0)
            continue

        is_env = dyn.player(dstate) == 'env'
        for dstate2 in moves:
            mstate2 = mstate.update(sensor(dstate2))
            state2 = (dstate2, mstate2, depth + 1)
            if state2 not in dag.nodes:
                stack.append((dstate2, mstate2, depth + 1))

            dag.add_edge(state, state2)

            if is_env:
                moves = cast(EnvMoves, moves)
                dag.edges[state, state2]['prob'] = moves[dstate2]
    return dag


def empirical_psat(tree: PrefixTree, concept: Concept) -> float:
    # TODO: Use monitor...
    leaves = (n for n in tree.nodes() if tree.is_leaf(n))
    accepted = total = 0
    for leaf in leaves:
        demo = tree.prefix(leaf)
        count = tree.count(leaf)
        total += count
        accepted += (demo in concept) * count
    return accepted / total


@attr.frozen
class ProductMC:
    tree: PrefixTree
    concept: MonitorableConcept
    policy: TabularPolicy 
    tree2policy: dict[Node, State]

    @property
    def edge_probs(self) -> dict[Edge, float]:
        edge_probs = {}
        for tree_edge in self.tree.tree.edges:
            dag_edge = (self.tree2policy[s] for s in tree_edge)
            edge_probs[tree_edge] = self.policy.prob(*dag_edge)
        return edge_probs

    def sample(self, pivot: Node, win: bool) -> SampledPath:
        policy = self.policy
        state = self.tree2policy[pivot]

        if policy.psat(state) == float(not win):
            return None  # Impossible to realize is_sat label.

        sample_prob: float = 1
        path = list(self.tree.prefix(pivot))

        # Make sure to deviate from prefix tree at pivot.
        tmp = set(policy.dag.neighbors(state)) - \
              {self.tree2policy[s] for s in self.tree.tree.neighbors(pivot)}
        moves = list(m for m in tmp if policy.psat(m) != float(not win))
        if not moves:
            return None  # Couldn't deviate
        
        # Sample suffix to path conditioned on win.
        while moves:
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

            # Note: win/lose are strings so the below still works...
            path.append(state[0])    # Ignore policy state details.
            moves = list(policy.dag.neighbors(state))

        del path[-1]                 # Remove dummy win/lose state.
        return path, sample_prob
 
    @staticmethod
    def construct(
            concept: MonitorableConcept,
            tree: PrefixTree,
            dyn: Dynamics,
            max_depth: Optional[int],
            rationality: Optional[float] = None,
            psat: Optional[float] = None,
            sensor: Callable[[State], Any] = lambda x: x,
            rtol: Optional[float] = None,
            xtol: Optional[float] = None,
        ) -> ProductMC:
        """Constructs a tabular policy by unrolling of dynamics/concept."""
        dag = product_dag(concept, tree, dyn, max_depth, sensor)

        if rationality is None:
            if psat is None:
                psat = empirical_psat(tree, concept)
            policy = TabularPolicy.from_psat(dag, psat, xtol=xtol, rtol=rtol)
        else:
            policy = TabularPolicy.from_rationality(dag, rationality)

        # Need to associcate each tree stree with a policy state.
        stack = [(tree.root, policy.root)]
        tree2policy = {}
        while stack:
            tstate, pstate = stack.pop()
            tree2policy[tstate] = pstate

            # Compute local mapping from dynamics transition to next pstate.
            move = {s[0]: s for s in policy.dag.neighbors(pstate)}
            for tstate2 in tree.tree.neighbors(tstate):
                dstate = tree.state(tstate2)  # Dynamics state.
                pstate2 = move[dstate]
                stack.append((tstate2, pstate2))

        return ProductMC(
                tree=tree,
                concept=concept,
                policy=policy,
                tree2policy=tree2policy,
        )
