from __future__ import annotations

from typing import Protocol

from diss import Edge, Path


class AnnotatedMarkovChain(Protocol):
    def log_probs(self, path: Path) -> dict[Edge, float]:
        ...

    def extend(self, path: Path, max_size: int, is_sat: bool) -> Path:
        ...


__all__ = ['AnnotatedMarkovChain']
