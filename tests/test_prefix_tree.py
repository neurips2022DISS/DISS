from diss import DemoPrefixTree


def test_prefixtree():
    demo1 = [
        (6, frozenset({5, 4})),
        (5, {3: 2/3, 4: 1/3}),
        (3, frozenset({1, 2})),
        (1, frozenset()),
    ]
    tree = DemoPrefixTree.from_demos([demo1])

    assert tree.max_len == 4
    assert set(tree.nodes()) == set(tree.nodes(demo1))

    path1 = [x for x, _ in demo1]
    for i, node in enumerate(tree.nodes(demo1)):
        assert tree.prefix(node) == path1[:i+1]
