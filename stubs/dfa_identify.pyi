from typing import Any, Optional
import dfa

Examples = list[list[Any]]


def find_dfa(
        positive: Examples,
        negative: Examples,
        bounds: tuple[Optional[int], Optional[int]] = (None, None),
    ) -> dfa.DFA:
    ...
