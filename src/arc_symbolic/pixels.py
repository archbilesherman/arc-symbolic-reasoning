from dataclasses import dataclass
from typing import Literal

import numpy as np


GEOMETRIC_OPS = (
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "flip_horizontal",
    "flip_vertical",
    "transpose",
    "anti_transpose",
)


@dataclass(frozen=True)
class PixelOperator:
    kind: Literal["geometry", "recolor"]
    operation: str | None = None
    source: int | None = None
    target: int | None = None

    def apply(self, grid) -> np.ndarray:
        a = np.asarray(grid, dtype=np.int64)

        if self.kind == "geometry":
            ops = {
                "identity": lambda x: x.copy(),
                "rotate_90": lambda x: np.rot90(x, 1),
                "rotate_180": lambda x: np.rot90(x, 2),
                "rotate_270": lambda x: np.rot90(x, 3),
                "flip_horizontal": np.fliplr,
                "flip_vertical": np.flipud,
                "transpose": lambda x: x.T.copy(),
                "anti_transpose": lambda x: np.rot90(x.T, 2),
            }

            if self.operation not in ops:
                raise ValueError(f"Unknown geometric operation: {self.operation}")

            return ops[self.operation](a)

        if self.kind == "recolor":
            if self.source is None or self.target is None:
                raise ValueError("Recolor requires source and target colors")

            result = a.copy()
            result[a == self.source] = self.target
            return result

        raise ValueError(f"Unknown pixel operator kind: {self.kind}")