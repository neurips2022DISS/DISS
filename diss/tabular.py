from __future__ import annotations

import random
from functools import lru_cache
from typing import Any, Optional, Sequence, cast

import attr
import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import bisect
from scipy.special import logsumexp, softmax
from scipy.stats import entropy

from diss import AnnotatedMarkovChain, Edge, Path, Node


oo = float('inf')


def parse_unrolled_mdp(unrolled_mdp: nx.DiGraph) -> None:
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

    def entropy(self, node: Node = None) -> float:
        if node is None:
            node = self.root
        return self.dag.nodes[node]['entropy']  # type: ignore

    def psat(self, node: Node = None) -> float:
        return np.exp(self.lsat(node))  # type: ignore

    def lsat(self, node: Node = None) -> float:
        if node is None:
            node = self.root
        return self.dag.nodes[node]['lsat']  # type: ignore

    def value(self, node: Node = None) -> float:
        if node is None:
            node = self.root
        return self.dag.nodes[node]['val']  # type: ignore

    def prob(self, node: Node, move: Node, log: bool = False) -> float:
        if (node, move) not in self.dag.edges:
            lprob = -oo
        elif self.dag.nodes[node]['kind'] == 'ego':
            Q, V = self.dag.nodes[move]['val'], self.dag.nodes[node]['val']
            Q += self.dag.edges[node, move].get('entropy', 0)
            lprob = Q - V
        else:
            prob = self.dag.edges[node, move]['prob']
            return np.log(prob) if log else prob  # type: ignore
        return lprob if log else np.exp(lprob)  # type: ignore

    def log_probs(self, path: Path) -> dict[Edge, float]:
        pairwise = zip(path, path[1:])
        return {(n, m): self.prob(n, m) for (n, _), (m, _) in pairwise}

    def extend(self, path: Path, max_len: int, is_sat: bool) -> Path:
        # TODO: handle case where impossible to sample.
        path = list(path)
        node = path[-1] if path else self.root 
        while len(path) < max_len:
            moves = list(self.dag.neighbors(node))
            if not moves:
                break
            # Apply bayes rule to get Pr(s' | is_sat, s).
            priors = np.array([self.prob(node, m) for m in moves])
            likelihoods = np.array([self.psat(m) for m in moves])
            normalizer = self.psat(node)

            if not is_sat:
                likelihoods = 1 - likelihoods
                normalizer = 1 - normalizer

            probs = cast(Sequence[float], priors * likelihoods / normalizer)
            node = random.choices(moves, probs)[0]
            path.append((node, frozenset(moves)))
        return path 

    @staticmethod
    def from_psat(unrolled: nx.DiGraph, psat: float) -> TabularPolicy:
        @lru_cache(maxsize=3)
        def get_critic(rationality: float) -> TabularPolicy:
            return TabularPolicy.from_rationality(unrolled, rationality)

        def f(rationality: float) -> float:
            return get_critic(rationality).psat() - psat

        if f(0) > 0:
            return get_critic(0)

        # Doubling trick.
        for i in range(10):
            rat: float = 1 << i
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

            entropies: ArrayLike = node_entropies + edge_entropies
            data['entropy'] += probs @ entropies  # type: ignore
            data['lsat'] = logsumexp(lsats, b=probs)

        if len(roots) != 1:
            raise ValueError('unrolled MDP must have a unique root node!')

        return TabularPolicy(root=roots[0], dag=unrolled, rationality=rationality)
