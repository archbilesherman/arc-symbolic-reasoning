"""Immutable, executable, bounded compositions of existing ARC primitives."""
from dataclasses import dataclass
import numpy as np


def _valid_intermediate(grid):
    if grid is None:
        return False
    arr = np.asarray(grid)
    return (arr.ndim == 2 and all(1 <= n <= 30 for n in arr.shape)
            and np.issubdtype(arr.dtype, np.integer)
            and bool(np.all((arr >= 0) & (arr <= 9))))


@dataclass(frozen=True)
class CompositeProgram:
    steps: tuple

    def __post_init__(self):
        if len(self.steps) != 2:
            raise ValueError('Only two-operation compositions are supported')
        if not all(callable(getattr(step, 'apply', None)) for step in self.steps):
            raise TypeError('Each step must be an executable transformation')

    def apply(self, grid):
        current = np.asarray(grid, dtype=np.int64)
        if not _valid_intermediate(current):
            return None
        for step in self.steps:
            try:
                current = step.apply(current)
            except (ValueError, IndexError, TypeError):
                return None
            if not _valid_intermediate(current):
                return None
        return np.asarray(current, dtype=np.int64).copy()