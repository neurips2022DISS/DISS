from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

import attr
import networkx as nx
import numpy as np
from scipy.optimize import bisect
from scipy.special import logsumexp, softmax
from scipy.stats import entropy


Node = Any
oo = float('inf')


def parse_unrolled_mdp(unrolled_mdp: nx.DiGraph):
    for node in nx.topological_sort(unrolled_mdp):
        data = unrolled_mdp.nodes[node]

        if data.get('kind') not in {'env', 'ego', True, False}:
            raise ValueError('All nodes must be labeled ego or env')

        if data['kind'] == 'ego':
            continue

        neighbors = unrolled_mdp.neighbors(node)
        edges = unrolled_mdp.edges
        if any('prob' not in edges[node, node2] for node2 in neighbors):
            raise ValueError('All env nodes must provide probs on edges.')


@attr.frozen
class TabularPolicy:
    root: Any
    dag: nx.DiGraph
    rationality: float

    def entropy(self, node: Optional[Node] = None) -> float:
        if node is None:
            node = self.root
        return self.dag.nodes[node]['entropy']

    def psat(self, node: Optional[Node] = None) -> float:
        if node is None:
            node = self.root
        return np.exp(self.dag.nodes[node]['lsat'])

    def value(self, node: Optional[Node] = None) -> float:
        if node is None:
            node = self.root
        return self.dag.nodes[node]['val']

    def prob(self, node: Node, move: Node) -> float:
        if (node, move) not in self.dag.edges:
            return 0
        if self.dag.nodes[node]['kind'] == 'ego':
            Q, V = self.dag.nodes[move]['val'], self.dag.nodes[node]['val']
            Q += self.dag.edges[node, move].get('entropy', 0)
            return np.exp(Q - V)
        return self.dag.edges[node, move]['prob']

    @staticmethod
    def from_psat(unrolled: nx.DiGraph, psat: float) -> TabularPolicy:
        @lru_cache(maxsize=3)
        def get_critic(rationality):
            return TabularPolicy.from_rationality(unrolled, rationality)

        def f(rationality):
            return get_critic(rationality).psat() - psat

        if f(0) > 0:
            return get_critic(0)

        # Doubling trick.
        for i in range(10):
            rat = 1 << i
            if f(rat) > 0:
                rat = bisect(f, 0, rat)
                break
        return get_critic(rat)

    @staticmethod
    def from_rationality(unrolled: nx.DiGraph, rationality: float) -> TabularPolicy:
        """Creates a critic from a given unrolled_mdp graph."""
        unrolled = nx.DiGraph(unrolled)  # Make a copy
        rat = rationality
        edges = unrolled.edges

        nodes = reversed(list(nx.topological_sort(unrolled)))
        roots = []
        for node in nodes:
            if unrolled.in_degree(node) == 0:
                roots.append(node)

            data = unrolled.nodes[node]
            kind = data['kind']

            if isinstance(kind, bool):
                data['val'] = rat * float(kind)
                data['lsat'] = 0 if kind else -oo
                data['entropy'] = 0
                continue

            moves = list(unrolled.neighbors(node))  # Fix order of moves.
            lsats = np.array([unrolled.nodes[m]['lsat'] for m in moves])
            node_entropies = np.array([unrolled.nodes[m]['entropy'] for m in moves])
            edge_entropies = np.array([
                edges[node, m].get('entropy', 0) for m in moves
            ])
            vals = np.array([unrolled.nodes[m]['val'] for m in moves])
            vals += edge_entropies

            if kind == 'ego':
                probs = softmax(vals)
                data['val'] = logsumexp(vals) # Compute V.
                data['entropy'] = entropy(probs)
            else:
                probs = np.array([edges[node, m]['prob'] for m in moves])
                data['val'] = probs @ vals    # Compute Q.
                data['entropy'] = 0  # Only ego can produce action entropy.

            data['entropy'] += probs @ (node_entropies + edge_entropies) 
            data['lsat'] = logsumexp(lsats, b=probs)

        if len(roots) != 1:
            raise ValueError('unrolled MDP must have a unique root node!')

        return TabularPolicy(root=roots[0], dag=unrolled, rationality=rationality)
