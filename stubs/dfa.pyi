from typing import Protocol, Any, Sequence

State = Any
Letter = Any
Alphabet = frozenset[Letter]
DFADict = dict[State, tuple[bool, dict[Letter, State]]]

def dfa2dict(dfa: DFA) -> tuple[DFADict, State]: ...

class DFA(Protocol):
    inputs: Alphabet
    outputs: frozenset[bool]

    def label(self, word: Sequence[Letter]) -> bool: ...
    def states(self) -> set[State]: ...
