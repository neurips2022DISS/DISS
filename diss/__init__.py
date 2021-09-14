from typing import Any, Sequence


Node = Any
Edge = tuple[Node, Node]
Moves = frozenset[Node]
Path = Sequence[tuple[Node, Moves]]
Demos = Sequence[Path]


from diss.annotated_mc import *
