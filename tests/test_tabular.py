from pytest import approx

import numpy as np
import funcy as fn
import networkx as nx

from diss.planners.tabular import TabularPolicy


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

    ctl = TabularPolicy.from_psat(graph, 1 / 2)
    assert ctl.psat() == approx(1 / 2)

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
    graph.add_edge(6, 5, entropy=2/3*np.log(300))
    graph.add_edge(6, 4)

    ctl = TabularPolicy.from_rationality(graph, np.log(8))
    assert ctl.value(0) == 0
    assert ctl.value(1) == np.log(8)
    assert ctl.value(2) == approx(np.log(2))
    assert ctl.value(3) == approx(np.log(10))
    assert ctl.value(4) == approx(np.log(3))
    assert ctl.value(5) == approx(np.log(300)/3)
    assert ctl.value(6) == approx(np.log(303))

    assert ctl.prob(3, 1) == approx(8 / 10)
    assert ctl.prob(3, 2) == approx(2 / 10)
    assert ctl.prob(4, 2) == approx(2 / 3)
    assert ctl.prob(4, 0) == approx(1 / 3)
    assert ctl.prob(6, 5) == approx(300 / 303)
    assert ctl.prob(6, 4) == approx(3 / 303)

    assert ctl.psat(2) == 1 / 3
    assert ctl.psat(3) == 13 / 15
    assert ctl.psat(4) == 2 / 9
    assert ctl.psat(5) == approx(88 / 135)
    assert ctl.psat(6) == approx(1766 / 2727)

    # Smoke test for MC api.
    # TODO
    #path = ctl.extend(path=(), max_len=3, is_sat=True)
    #assert len(path) == 3
    #log_probs = ctl.log_probs(path)
    #assert len(log_probs) == len(path) - 1
