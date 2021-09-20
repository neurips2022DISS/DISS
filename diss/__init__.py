from typing import Any, Mapping, Sequence, Union


State = Any
EgoMoves = frozenset[State]
EnvMoves = Mapping[State, float]
Moves = Union[EgoMoves, EnvMoves]
Path = Sequence[State]
Demo = Sequence[tuple[State, Moves]]
Demos = Sequence[Demo]
Node = int  # Node of prefix tree.
Edge = tuple[int, int]

from diss.prefix_tree import *
from diss.annotated_mc import *

__all__ = [
    'AnnotatedMarkovChain',
    'Demo',
    'Demos',
    'DemoPrefixTree',
    'Edge',
    'EgoMoves',
    'EnvMoves',
    'Moves',
    'Path',
    'State',
]
