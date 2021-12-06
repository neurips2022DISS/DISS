from typing import Any, Literal, Mapping, Sequence, Union, Protocol

State = Any
Player = Literal['ego', 'env']
Path = Sequence[State]
Demo = Sequence[tuple[State, Player]]
Demos = Sequence[Demo]
Node = int  # Node of prefix tree.
Edge = tuple[int, int]

from diss.prefix_tree import *
from diss.annotated_mc import *
from diss.learn import *
from diss.planners.product_mc import *

__all__ = [
    'AnnotatedMarkovChain',
    'Concept',
    'ConceptIdException',
    'Demo',
    'Demos',
    'DemoPrefixTree',
    'Edge',
    'LabeledExamples',
    'MonitorState',
    'Moves',
    'Path',
    'Player',
    'SampledPath',
    'State',
    'search',
]
