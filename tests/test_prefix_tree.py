from collections import Counter

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
        assert tree.count(node) == 1
        if i < 3:
            assert len(tree.unused_moves(node)) == 1
            assert len(tree.moves(node)) == 2
        else:
            assert tree.is_leaf(node)
            assert len(tree.unused_moves(node)) == 0

        if i != 1:
            assert tree.is_ego(node)

    demo2 = [
        (6, frozenset({5, 4})),
        (5, {3: 2/3, 4: 1/3}),
        (4, frozenset({0, 2, 10})), # Adds an extra move.
        (2, {0: 2/3, 1: 1/3}),
        (0, frozenset()),
    ]
    path2 = [x for x, _ in demo2]
    tree = DemoPrefixTree.from_demos([demo1, demo2])
    assert tree.max_len == 5
    assert len(list(tree.nodes())) == len(set(path1 + path2))

    visit_counts = Counter(tree.count(n) for n in tree.nodes())
    assert visit_counts == {1: 5, 2: 2}

