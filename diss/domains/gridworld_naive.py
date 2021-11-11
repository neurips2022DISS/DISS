from __future__ import annotations

from itertools import product
from typing import Any, Literal, Optional, Union

import attr

from diss import State, Player
from diss.product_mc import Moves


Action = Literal['↑', '↓', '←', '→']
ACTION2VEC = {
    '→': (1, 0),
    '←': (-1, 0),
    '↑': (0, -1),
    '↓': (0, 1),
}


__all__ = [
    'Action',
    'GridWorldNaive',
    'GridWorldState',
]


@attr.frozen
class GridWorldState:
    x: int
    y: int
    action: Optional[Action] = None

    def __repr__(self) -> str:
        if self.action is not None:
            return self.action
        return f'({self.x}, {self.y})'

    @property
    def succeed(self) -> GridWorldState:
        assert self.action is not None
        dx, dy = ACTION2VEC[self.action]
        return attr.evolve(self, x=self.x + dx, y=self.y + dy, action=None)

    @property
    def slip(self) -> GridWorldState:
        return attr.evolve(self, action='↓').succeed

 
@attr.frozen
class GridWorldNaive:
    dim: int
    start: GridWorldState
    overlay: dict[tuple[int, int], str] = attr.ib(factory=dict)
    slip_prob: float = 1 / 32

    def sensor(self, state: Union[GridWorldState, tuple[int, int]]) -> Any:
        if isinstance(state, GridWorldState):
            if self.player(state) == 'env':
                return 'white'  # Ignore environment nodes.
            state = (state.x, state.y)
        return self.overlay.get(state, 'white')

    def clip(self, state: State) -> State:
        x = min(self.dim, max(1, state.x))
        y = min(self.dim, max(1, state.y))
        return attr.evolve(state, x=x, y=y)

    def moves(self, state: State) -> Moves:
        if self.player(state) == 'ego':
            moves = (attr.evolve(state, action=a) for a in ACTION2VEC)
            moves = (m for m in moves if self.clip(m.succeed) == m.succeed)
            return frozenset(moves)

        succeed, slip = self.clip(state.succeed), self.clip(state.slip)
        if state.action == '↓':
            return {succeed: 1}
        return {succeed: 1 - self.slip_prob, slip: self.slip_prob}

    def player(self, state: State) -> Player:
        return 'ego' if state.action is None else 'env'

    def to_string(self, state: GridWorldState) -> str:
        from blessings import Terminal  # type: ignore
        term = Terminal()
        buff = ''
        ego = 'x' if state.action is None else state.action

        def tile(point: tuple[int, int]) -> str:
            content = f' {ego} ' if point == (state.x, state.y) else '   '
            color = self.sensor(point)
            return getattr(term, f'on_{color}')(content)  # type: ignore

        for y in range(1, 1 + self.dim):
            row = ((x, y) for x in range(1, 1 + self.dim))
            buff += ''.join(map(tile, row)) + '\n'
        return buff
