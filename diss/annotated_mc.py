from __future__ import annotations

from typing import Protocol, Optional, Sequence

from diss import State, Path


class AnnotatedMarkovChain(Protocol):
    def log_probs(self, path: Path) -> Sequence[float]:
        ...

    def extend(self, path: Path, max_size: int, is_sat: bool, moves: frozenset[State]) -> Optional[Path]:
        ...


__all__ = ['AnnotatedMarkovChain']
