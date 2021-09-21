from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, cast

import attr
import networkx as nx

from diss import Demo, Demos, Moves, Node, Path, State


__all__ = ["DemoPrefixTree"]


def transition(tree: nx.DiGraph, src: Node, move: State) -> Node:
    for tgt in tree.neighbors(src):
        if tree.nodes[tgt]['source'] == move:
            return tgt
    raise ValueError(f'{src=} is not connected to {move=}.')


@attr.frozen
class DemoPrefixTree:
    """Data structure representing the prefix tree of the demonstrations."""
    tree: nx.DiGraph
    max_len: int

    def parent(self, node: int) -> Node:
        node, *_ = self.tree.predecessors(node)
        return node 

    def state(self, node: int) -> State:
        """Returns which state node points to."""
        return self.tree.nodes[node]['source']

    def count(self, node: int) -> int:
        """Returns how many demonstrations pass through this node."""
        return cast(int, self.tree.nodes[node]['count'])

    def is_ego(self, node: int) -> bool:
        return isinstance(self.moves, frozenset)

    def moves(self, node: int) -> Moves:
        return cast(Moves, self.tree.nodes[node]['moves'])

    def unused_moves(self, node: int) -> frozenset[State]:
        neighbors = map(self.state, self.tree.neighbors(node))
        return frozenset(self.moves(node)) - frozenset(neighbors)

    def is_leaf(self, node: int) -> bool:
        return self.tree.out_degree(node) == 0

    def prefix(self, node: int) -> Path:
        assert node > 0

        path = []
        while node != 0:
            data = self.tree.nodes[node]           
            path.append(data['source'])
            node = self.parent(node)
        path.reverse() 
        return path

    def nodes(self, demo: Optional[Demo] = None) -> Iterable[int]:
        """Yields nodes in prefix tree.

        Yields:
          - All nodes if demo is None.
          - Nodes visited in demo (in order) if demo is not None.
        """
        if demo is None:
            yield from self.tree.nodes
        else:
            node = 0
            for move, _ in demo[1:]:
                yield node
                node = transition(self.tree, node, move)

    @staticmethod
    def from_demos(demos: Demos) -> DemoPrefixTree:
        paths = [[x for x, _ in demo] for demo in demos]
        tree = nx.prefix_tree(paths)
        tree.remove_node(-1)  # Node added by networkx.
        assert set(tree.neighbors(0)) == {1}
        tree.remove_node(0)   # Node added by networkx.
        nx.relabel_nodes(
            G=tree,
            mapping={n: n-1 for n in range(len(tree.nodes))}, 
            copy=False,
        )

        for demo in demos:
            node = 0
            for depth, (state, moves) in enumerate(demo[1:]):
                node = transition(tree, node, state)
                data = tree.nodes[node]
                data.setdefault('count', 0)
                data.setdefault('depth', depth + 1)  # Skipped root.
                data.setdefault('moves', type(moves)())
                data['count'] += 1
                data['moves'] |= moves

        max_len = max(map(len, paths))
        return DemoPrefixTree(tree=tree, max_len=max_len)

