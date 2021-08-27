from pytest import approx

import numpy as np
import funcy as fn
import networkx as nx

from diss.tabular import TabularPolicy


def test_tabular_and():
    graph = nx.DiGraph()
    graph.add_node(0, kind=False)
    graph.add_node(1, kind=True)
    graph.add_node(2, kind="ego")
    graph.add_node(3, kind="ego")
    graph.add_edge(2, 1)
    graph.add_edge(2, 0)
    graph.add_edge(3, 2)
    graph.add_edge(3, 0)

    ctl = TabularPolicy.from_rationality(graph, rationality=np.log(2))

    assert ctl.value(0) == 0
    assert ctl.value(1) == np.log(2)
    assert ctl.value(2) == approx(np.log(3))
    assert ctl.value(3) == approx(np.log(4))

    assert ctl.prob(1, 2) == 0
    assert ctl.prob(2, 1) == approx(2 / 3)
    assert ctl.prob(2, 0) == approx(1 / 3)
    assert ctl.prob(3, 0) == approx(1 / 4)
    assert ctl.prob(3, 2) == approx(3 / 4)

    assert ctl.psat(3) == approx(1 / 2)

    # Add entropy to (3, 2) edge.
    graph.edges[3, 0]['entropy'] = np.log(2)
    ctl = TabularPolicy.from_rationality(graph, np.log(2))

    assert ctl.value(0) == 0
    assert ctl.value(2) == approx(np.log(3))
    assert ctl.value(3) == approx(np.log(5))
    assert ctl.prob(3, 2) == approx(3 / 5)
    assert ctl.prob(3, 0) == approx(2 / 5)


def test_tabular_simple_mdp():
    graph = nx.DiGraph()

    graph.add_node(0, kind=False)
    graph.add_node(1, kind=True)
    graph.add_node(2, kind="env")
    graph.add_node(3, kind="ego")
    graph.add_node(4, kind="ego")
    graph.add_node(5, kind="env")
    graph.add_node(6, kind="ego")

    graph.add_edge(2, 1, prob=1/3)
    graph.add_edge(2, 0, prob=2/3)
    graph.add_edge(3, 2)
    graph.add_edge(3, 1)
    graph.add_edge(4, 2)
    graph.add_edge(4, 0)
    graph.add_edge(5, 3, prob=2/3)
    graph.add_edge(5, 4, prob=1/3)
    graph.add_edge(6, 5)
    graph.add_edge(6, 4)

    ctl = TabularPolicy.from_rationality(graph, np.log(8))
    assert ctl.value(0) == 0
    assert ctl.value(1) == np.log(8)
    assert ctl.value(2) == approx(np.log(2))
    assert ctl.value(3) == approx(np.log(10))
    assert ctl.value(4) == approx(np.log(3))
    assert ctl.value(5) == approx(np.log(300)/3)
    assert ctl.value(6) == approx(np.log(3 + 300**(1/3)))
    assert ctl.prob(6, 5) == approx(300**(1/3) / (3 + 300**(1/3)))
    assert ctl.prob(6, 4) == approx(1 - ctl.prob(6, 5))
