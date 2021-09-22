from collections import Counter

from diss import DemoPrefixTree


def test_prefixtree():
    demo1 = [
        (6, 'ego'),
        (5, 'env'),
        (3, 'ego'),
        (1, 'ego'),
    ]
    tree = DemoPrefixTree.from_demos([demo1])

    assert tree.max_len == 4
    assert set(tree.nodes()) == set(tree.nodes(demo1))

    path1 = [x for x, _ in demo1]
    for i, node in enumerate(tree.nodes(demo1)):
        assert tree.prefix(node) == path1[:i+1]
        assert tree.count(node) == 1
        if i >= 3:
            assert tree.is_leaf(node)

        if i != 1:
            assert tree.is_ego(node)

    demo2 = [
        (6, 'ego'),
        (5, 'env'),
        (4, 'ego'),
        (2, 'env'),
        (0, 'ego'),
    ]
    path2 = [x for x, _ in demo2]
    tree = DemoPrefixTree.from_demos([demo1, demo2])
    assert tree.max_len == 5
    assert len(list(tree.nodes())) == len(set(path1 + path2))

    visit_counts = Counter(tree.count(n) for n in tree.nodes())
    assert visit_counts == {1: 5, 2: 2}

