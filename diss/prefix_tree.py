from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, cast

import attr
import networkx as nx

from diss import Demo, Demos, Node, Path, Player, State


__all__ = ["DemoPrefixTree"]


def transition(tree: nx.DiGraph, src: Node, move: State) -> Node:
    for tgt in tree.neighbors(src):
        if tree.nodes[tgt]['source'] == move:
            return cast(int, tgt)
    raise ValueError(f'{src=} is not connected to {move=}.')


@attr.frozen
class DemoPrefixTree:
    """Data structure representing the prefix tree of the demonstrations."""
    tree: nx.DiGraph
    max_len: int
    root: int = 0

    def parent(self, node: int) -> Optional[Node]:
        if node == 0:
            return None
        node, *_ = self.tree.predecessors(node)
        return node 

    def state(self, node: int) -> State:
        """Returns which state node points to."""
        return self.tree.nodes[node]['source']

    def count(self, node: int) -> int:
        """Returns how many demonstrations pass through this node."""
        return cast(int, self.tree.nodes[node]['count'])

    def is_ego(self, node: int) -> bool:
        return 'ego' == cast(Player, self.tree.nodes[node]['player'])

    def is_leaf(self, node: int) -> bool:
        return self.tree.out_degree(node) == 0

    def prefix(self, start: int) -> Path:
        path: list[State] = []
        node: Optional[Node] = start 
        while node is not None:
            data = self.tree.nodes[node]           
            path.append(data['source'])
            node = self.parent(node)
        path.reverse() 
        return path

    def edges(self) -> Iterable[tuple[int, int]]:
        yield from self.tree.edges

    def nodes(self, demo: Optional[Demo] = None) -> Iterable[int]:
        """Yields nodes in prefix tree.

        Yields:
          - All nodes if demo is None.
          - Nodes visited in demo (in order) if demo is not None.
        """
        if demo is None:
            yield from self.tree.nodes
        else:
            yield (node := 0)
            for move, _ in demo[1:]:
                node = transition(self.tree, node, move)
                yield node

    @staticmethod
    def from_demos(demos: Demos) -> DemoPrefixTree:
        paths = [[x for x, _ in demo] for demo in demos]
        tree = nx.prefix_tree(paths)
        tree.remove_node(-1)  # Node added by networkx.
        assert set(tree.neighbors(0)) == {1}
        tree.remove_node(0)   # Node added by networkx.
        nx.relabel_nodes(
            G=tree,
            mapping={n: n-1 for n in tree.nodes}, 
            copy=False,
        )

        for demo in demos:
            for depth, (state, player) in enumerate(demo):
                node: int = 0 if depth == 0 else transition(tree, node, state)
                data = tree.nodes[node]
                data.update({'depth': depth, 'player': player})
                data.setdefault('count', 0)
                data['count'] += 1

        max_len = max(map(len, paths))
        return DemoPrefixTree(tree=tree, max_len=max_len)
