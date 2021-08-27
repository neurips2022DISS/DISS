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
