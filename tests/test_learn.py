from diss.learn import LabeledExamples


def test_labeled_examples():
    examples1 = LabeledExamples(positive=['x'], negative=['y'])
    examples2 = LabeledExamples(positive=[], negative=['x'])

    examples12 = LabeledExamples(positive=[], negative=['y', 'x'])
    assert examples12 == examples1 @ examples2
    assert examples1.dist(examples12) == 1
