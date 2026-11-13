from dataclasses import dataclass

import numpy as np

from .operators import ObjectOperator
from .pixels import PixelOperator


@dataclass(frozen=True)
class CandidateHypothesis:
    program: PixelOperator | ObjectOperator
    representation: str
    complexity: int
    label: str

    def apply(self, grid) -> np.ndarray | None:
        return self.program.apply(grid)


def make_hypothesis(program: PixelOperator | ObjectOperator) -> CandidateHypothesis:
    if isinstance(program, PixelOperator):
        representation = "pixel"
        if program.kind == "geometry":
            cost = 0 if program.operation == "identity" else 1
        else:
            cost = 2
    elif isinstance(program, ObjectOperator):
        representation = "object"
        cost = {
            "keep": 3,
            "erase": 3,
            "crop": 3,
            "recolor": 4,
            "translate": 5,
            "duplicate": 5,
        }[program.action]
        cost += int(program.attributes is not None)
        cost += int(program.rank is not None)
        cost += 2 * len(program.relations)
        if program.concept is not None:
            cost += 2 * len(program.concept.literals)
    else:
        raise TypeError(f"Unsupported program type: {type(program)}")

    return CandidateHypothesis(
        program=program,
        representation=representation,
        complexity=cost,
        label=f"{representation}:{program!r}",
    )