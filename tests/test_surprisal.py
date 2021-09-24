from __future__ import annotations

from collections import defaultdict

import attr
import numpy as np
from pytest import approx

from diss import DemoPrefixTree, Edge, Node, SampledPath
from diss.learn import surprisal, surprisal_grad


@attr.frozen
class UniformMC:
    tree: DemoPrefixTree

    @property
    def edge_probs(self) -> dict[Edge, float]:
        state_edge_probs = defaultdict(lambda: 1/2, {
            (5, 3): 2/3, (5, 4): 1/3, (2, 0): 2/3, (2, 1): 1/3
        })
        edge_probs = {}
        for (parent, kid) in self.tree.edges():
            state_edge = self.tree.state(parent), self.tree.state(kid)
            edge_probs[parent, kid] = state_edge_probs[state_edge]
        return edge_probs

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
        (6, 'ego'),
        (5, 'env'),
        (3, 'ego'),
        (1, 'ego'),
    ], [
        (6, 'ego'),
        (5, 'env'),
        (4, 'ego'),
        (2, 'env'),
        (0, 'ego'),
    ]]
    tree = DemoPrefixTree.from_demos(demos)
    chain = UniformMC.from_tree(tree)
    loss = surprisal(chain, tree)
    assert loss == np.log(2) * 4

    expected = {
        6: 2*(-1/2),
        5: 2*(2 - 1) * 0 - 0,
        3: 2*(2 - 1) * 1/2 * 2/3 * 1/2 - 1/2,
        1: 2*(2 - 1) * 1/2 * 2/3 * 1/2 + 1/2,
        4: 2*(2 - 1) * 1/2 * 1/3 * 1/2 - 1/2,
        2: approx(2*(2 - 1) * 1/2 * 1/3 * 1/2 * 1/3 + 1/2 * 1/3),
        0: approx(2*(2 - 1) * 1/2 * 1/3 * 1/2 * 2/3 + 1/2 * 2/3),

    }
    grad = {
        tree.state(node): -dS 
        for node, dS in enumerate(surprisal_grad(chain, tree))
    }
    assert grad == expected
    # Want to make actions not taken worse.
    assert grad[6] < 0
    assert grad[3] < 0
    assert grad[4] < 0

    # Want to make actions taken less risky.
    assert grad[5] == 0  # Exhausted node.
    assert grad[2] > 0

    # Want to make where we landed more optimal.
    assert grad[1] > 0
    assert grad[0] > 0
