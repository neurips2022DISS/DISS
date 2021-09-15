from __future__ import annotations

from typing import Any, Iterable, Sequence

import attr
import networkx as nx

from diss import Node, Path


def transition(tree: nx.DiGraph, src: Node, dst: Any) -> Node:
    for src in tree.neighbors(src):
        if tree.nodes[src]['source'] == dst:
            return src
    raise ValueError('{=src} is not connected to {=dst}.')


@attr.define
class DemoPrefixTree:
    tree: nx.DiGraph

    def _node(self, key: int) -> Node:
        ...

    def leaves(self) -> Iterable[Node]:  # Corresponds to unique demos.
        yield from map(self._node, self.tree.predecessors(-1))

    def nodes(self) -> Iterable[Node]:
        keys = (n for n in self.tree.nodes if n not in {0, -1})
        yield from map(self._node, keys)

    def prefix(self, key: int) -> Path:
        assert key > 0

        path = []
        while key != 0:
            data = self.tree.nodes[key]           
            path.append((data['source'], data['moves']))
            key, *_ = self.tree.predecessors(key)
        path.reverse() 
        return path

    def keys(self, path: Path) -> Iterable[int]:
        ...

    @staticmethod
    def from_demos(demos: Sequence[Path]) -> DemoPrefixTree:
        tree = nx.prefix_tree(demos)
        tree.nodes[0]['moves'] = frozenset()
        for demo in demos:
            state = 0
            for move, moves in demos:
                state = transition(tree, state, move)
                tree.nodes[state]['moves'] = moves
        return DemoPrefixTree(tree)  

