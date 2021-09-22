from __future__ import annotations

import attr
import numpy as np

from diss import DemoPrefixTree, Edge, Node, SampledPath
from diss.learn import surprisal, surprisal_grad


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
    loss = surprisal(chain, tree)
    assert loss == np.log(2) * 4

    expected = {
        6: -1,
        5: 0,  # Exhausted
    }
    grad = {
        tree.state(node): dS for node, dS in enumerate(surprisal_grad(chain, tree))
    }
    for node, dS in enumerate(surprisal_grad(chain, tree)):
        state = tree.state(node)
        if state in expected:
            assert dS == expected[state]
    breakpoint()
