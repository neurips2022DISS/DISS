from typing import Any, Optional, Iterable
import dfa

Examples = list[list[Any]]


def find_dfas(
        positive: Examples,
        negative: Examples,
        bounds: tuple[Optional[int], Optional[int]] = (None, None),
    ) -> Iterable[dfa.DFA]:
    ...
