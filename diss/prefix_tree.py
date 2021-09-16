from __future__ import annotations

from typing import Any, Iterable, Sequence, cast

import attr
import networkx as nx

from diss import State, Demo, Demos, Path


Node = int  # nx.prefix tree used ints as Nodes.


def transition(tree: nx.DiGraph, src: State, dst: Any) -> State:
    for src in tree.neighbors(src):
        if tree.nodes[src]['source'] == dst:
            return src
    raise ValueError('{=src} is not connected to {=dst}.')


@attr.frozen
class DemoPrefixTree:
    """Data structure representing the prefix tree of the demonstrations."""
    tree: nx.DiGraph

    def state(self, node: int) -> State:
        """Returns which state node points to."""
        return self.tree.nodes[node]['source']

    def count(self, node: int) -> int:
        """Returns how many demonstrations pass through this node."""
        return cast(int, self.tree.nodes[node]['count'])

    def leaves(self) -> frozenset[State]:  # Corresponds to unique demos.
        return frozenset(self.tree.predecessors(-1))

    def prefix(self, node: int) -> Path:
        assert node > 0

        path = []
        while node != 0:
            data = self.tree.nodes[node]           
            path.append(data['source'])
            node, *_ = self.tree.predecessors(node)
        path.reverse() 
        return path

    def nodes(self, demo: Demo) -> Iterable[int]:
        """Yields nodes in prefix tree visited by demo."""
        node = 0
        for move, _ in demo:
            node = transition(self.tree, node, move)
            yield node

    @staticmethod
    def from_demos(demos: Demos) -> DemoPrefixTree:
        paths = [[x for x, _ in demo] for demo in demos]
        tree = nx.prefix_tree(paths)
        for path in paths:
            node = 0
            for state in path:
                node = transition(tree, node, state)
                data = tree.nodes[node]
                data.setdefault('count', 0)
                data['count'] += 1

        return DemoPrefixTree(tree=tree)

