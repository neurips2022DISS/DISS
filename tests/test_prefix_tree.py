from diss import DemoPrefixTree


def test_prefixtree():
    tree = DemoPrefixTree.from_demos([[
        (6, frozenset({5, 4})),
        (5, {3: 2/3, 4: 1/3}),
        (3, frozenset({1, 2})),
        (1, frozenset()),
    ]])
