from __future__ import annotations

import attr

from diss import DemoPrefixTree, Edge, Node, SampledPath


@attr.frozen
class UniformMC:
    tree: DemoPrefixTree

    @property
    def edge_probs(self) -> dict[Edge, float]:
        return {e: 1/2 for e in self.tree.edges()}

    def sample(self, pivot: Node, max_len: int, win: bool) -> SampledPath:
        path = self.tree.prefix(pivot)
        assert len(path) + 2 < max_len
        unused_moves = self.tree.unused_moves(pivot) 
        if unused_moves:
            move, *_ = unused_moves
            path.append(move)
        padding = [win] * (max_len - len(path))
        return (path + padding, 0.8)

    @staticmethod
    def from_tree(tree: DemoPrefixTree) -> UniformMC:
        return UniformMC(tree)


def test_surprisal():
    demos = [[
        (6, frozenset({5, 4})),
        (5, {3: 2/3, 4: 1/3}),
        (3, frozenset({1, 2})),
        (1, frozenset()),
    ], [
        (6, frozenset({5, 4})),
        (5, {3: 2/3, 4: 1/3}),
        (4, frozenset({0, 2, 10})), # Adds an extra move.
        (2, {0: 2/3, 1: 1/3}),
        (0, frozenset()),
    ]]
    tree = DemoPrefixTree.from_demos(demos)
    chain = UniformMC.from_tree(tree)
    path, prob = chain.sample(pivot=0, max_len=10, win=True) 
    assert len(path) == 10
    assert path[-1] is True

    chain.edge_probs

def test_surprisal_grad():
    ...
