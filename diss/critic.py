from __future__ import annotations

import attr
import networkx as nx

from improvisers.game_graph import dfs_nodes, GameGraph, Node
from improvisers.critic import Critic, Distribution, DistLike
from improvisers.explicit import ExplicitDist as Dist


@attr.frozen
class TabularPolicy(Protocol):
    dag: nx.DiGraph
    root: Node
    rationality: float

    def psat(self, node: Node) -> float:
        ...

    def value(self, node: Node) -> float:
        ...

    def prob(self, node: Node, move: Node) -> float:
        ...

    @staticmethod
    def from_game_graph(game_graph: GameGraph, rationality: float) -> Critic:
        """Creates a critic from a given game graph."""
        dag = nx.DiGraph()
        sinks = []

        for node in dfs_nodes(game_graph):
            dag.add_node(node, kind=game_graph.label(node))

            moves = game_graph.moves(node)
            if not moves:
               sinks.append(node)
               continue

            for move in moves:
                entropy = game_graph.entropy(node, move)
                dag.add_edge(node, move, entropy=entropy)

        # TODO: annotate nodes with values
        # TODO: annotate nodes with lsats

        return Critic(dag=dag, root=game_graph.root, rationality=rationality)
