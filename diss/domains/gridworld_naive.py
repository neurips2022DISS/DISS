from __future__ import annotations

from typing import Any, Literal, Optional

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
    overlay: dict[State, Any]
    slip_prob: float = 1 / 32

    def sensor(self, state: State) -> Any:
        return self.overlay.get(state)

    def moves(self, state: State) -> Moves:
        if self.player(state) == 'ego':
            return frozenset(attr.evolve(state, action=a) for a in ACTION2VEC)
        return {state.succeed: 1 - self.slip_prob, state.slip: self.slip_prob}

    def player(self, state: State) -> Player:
        return 'env' if state.action is None else 'ego'
