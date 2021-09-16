from typing import Any, Mapping, Sequence, Union


State = Any
EgoMoves = frozenset[State]
EnvMoves = Mapping[State, float]
Moves = Union[EgoMoves, EnvMoves]
Path = Sequence[State]
Demo = Sequence[tuple[State, Moves]]
Demos = Sequence[Demo]


from diss.annotated_mc import *

__all__ = [
    'AnnotatedMarkovChain',
    'State',
    'EgoMoves',
    'EnvMoves',
    'Moves',
    'Path',
    'Demo',
    'Demos',
]
