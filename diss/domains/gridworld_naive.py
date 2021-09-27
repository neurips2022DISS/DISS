from __future__ import annotations

from itertools import product
from typing import Any, Literal, Optional, Union

import attr

from diss import State, Player
from diss.product_mc import ProductMC, Moves


Action = Literal['↑', '↓', '←', '→']
ACTION2VEC = {
    '→': (1, 0),
    '←': (-1, 0),
    '↑': (0, 1),
    '↓': (0, -1),
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

    @property
    def succeed(self) -> GridWorldState:
        assert self.action is not None
        dx, dy = ACTION2VEC[self.action]
        return GridWorldState(x=self.x + dx, y=self.y + dy)

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
            state = (state.x, state.y)
        return self.overlay.get(state, 'white')

    def moves(self, state: State) -> Moves:
        if self.player(state) == 'ego':
            return frozenset(attr.evolve(state, action=a) for a in ACTION2VEC)
        return {state.succeed: 1 - self.slip_prob, state.slip: self.slip_prob}

    def player(self, state: State) -> Player:
        return 'env' if state.action is None else 'ego'

    def to_string(self, state: GridWorldState) -> str:
        from blessings import Terminal  # type: ignore
        term = Terminal()
        buff = ''

        def tile(point: tuple[int, int]) -> str:
            content = 'x' if point == (state.x, state.y) else ' '
            color = self.sensor(point)
            return getattr(term, f'on_{color}')  # type: ignore

        for x in range(1, 1 + self.dim):
            row = ((x, y) for y in range(1, 1 + self.dim))
            buff += ''.join(map(tile, row)) + '\n'
        return buff
