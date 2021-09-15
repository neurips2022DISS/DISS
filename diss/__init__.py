from typing import Any, Mapping, Sequence, Union


Node = Any
Edge = tuple[Node, Node]
EgoMoves = frozenset[Node]
EnvMoves = Mapping[Node, float]
Moves = Union[EgoMoves, EnvMoves]
Path = Sequence[tuple[Node, Moves]]
Demos = Sequence[Path]


from diss.annotated_mc import *

__all__ = [
    'AnnotatedMarkovChain',
    'Node',
    'Edge',
    'EgoMoves',
    'EnvMoves',
    'Moves',
    'Path',
    'Demos',
]
