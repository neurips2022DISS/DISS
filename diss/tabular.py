from __future__ import annotations

from functools import lru_cache
from typing import Any

import attr
import networkx as nx
import numpy as np
from scipy.optimize import bisect
from scipy.special import logsumexp, softmax


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
    dag: nx.DiGraph
    rationality: float

    def psat(self, node: Node) -> float:
        return np.exp(self.dag.nodes[node]['lsat'])

    def value(self, node: Node) -> float:
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
            return get_critic(rationality) - psat

        if f(0) > 0:
            return get_critic(0)

        # Doubling trick.
        for i in range(10):
            rat = 1 << i
            if f(rat) > 0:
                rat = binary_search(f, bot, top)
                break
        return get_critic(rat)

    @staticmethod
    def from_rationality(unrolled: nx.DiGraph, rationality: float) -> TabularPolicy:
        """Creates a critic from a given unrolled_mdp graph."""
        unrolled = nx.DiGraph(unrolled)  # Make a copy
        rat = rationality
        edges = unrolled.edges

        nodes = reversed(list(nx.topological_sort(unrolled)))
        for node in nodes:
            data = unrolled.nodes[node]
            kind = data['kind']

            if isinstance(kind, bool):
                data['val'] = rat * float(kind)
                data['lsat'] = 0 if kind else -oo
                continue

            moves = list(unrolled.neighbors(node))  # Fix order of moves.
            vals = np.array([unrolled.nodes[m]['val'] for m in moves])
            lsats = np.array([unrolled.nodes[m]['lsat'] for m in moves])

            if kind == 'ego':
                vals += np.array([
                    edges[node, m].get('entropy', 0) for m in moves
                ]) # Entropy bump from traversing edge.

                data['val'] = logsumexp(vals) # Compute V.
                probs = softmax(vals)
            else:
                probs = np.array([edges[node, m]['prob'] for m in moves])
                data['val'] = probs @ vals    # Compute Q.

            data['lsat'] = logsumexp(lsats, b=probs)

        return TabularPolicy(dag=unrolled, rationality=rationality)
