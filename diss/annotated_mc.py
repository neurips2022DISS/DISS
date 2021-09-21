from __future__ import annotations

from typing import Protocol, Optional, Sequence

from diss import Node, Edge, Path, State 
from diss import DemoPrefixTree as PrefixTree


__all__ = ['AnnotatedMarkovChain', 'SampledPath']


SampledPath = Optional[tuple[Path, float]]


class AnnotatedMarkovChain(Protocol):
    @property
    def edge_probs(self) -> dict[Edge, float]:
        """Returns the probablity of edges in the demo prefix tree."""
        ...

    def sample(self, pivot: Node, max_len: int, win: bool) -> SampledPath:
        """Sample a path conditioned on pivot, max_size, and is_sat.

        Arguments:
          - pivot: Last node in the prefix tree that the sampled path 
                   passes through.
          - max_size: Maximum length of the sampled path.
          - win: Determines if sampled path results in ego winning.

        Returns:
           A path and corresponding log probability of sample the path OR
           None, if sampling from the empty set, e.g., want to sample
           an ego winning path, but no ego winning paths exist that pass
           through the pivot.
        """
        ...

    @staticmethod
    def from_tree(tree: PrefixTree) -> AnnotatedMarkovChain:
        ...

