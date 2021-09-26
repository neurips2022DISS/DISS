from typing import Any
import dfa

Examples = list[list[Any]]


def find_dfa(
        positive: Examples,
        negative: Examples,
    ) -> dfa.DFA:
    ...
