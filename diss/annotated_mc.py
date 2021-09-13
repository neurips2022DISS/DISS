from __future__ import annotations

from typing import Any, Sequence, Protocol


Path = Sequence[Any]
LogProbs = dict[Any, float]


class AnnotatedMarkovChain(Protocol):
    def log_probs(self, path: Path) -> LogProbs:
        ...

    def extend(self, path: Path, target_len: int, is_sat: bool) -> Path:
        ...


__all__ = ['AnnotatedMarkovChain', 'Path', 'LogProbs']
