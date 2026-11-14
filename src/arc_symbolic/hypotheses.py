from dataclasses import dataclass

import numpy as np

from .operators import ObjectOperator, MultiColorOperator
from .programs import CompositeProgram
from .pixels import PixelOperator


@dataclass(frozen=True)
class CandidateHypothesis:
    program: PixelOperator | ObjectOperator | MultiColorOperator | CompositeProgram
    representation: str
    complexity: int
    label: str
    parent_label: str | None = None

    def apply(self, grid) -> np.ndarray | None:
        return self.program.apply(grid)


def make_hypothesis(
    program: PixelOperator | ObjectOperator | MultiColorOperator | CompositeProgram,
) -> CandidateHypothesis:
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
    elif isinstance(program, MultiColorOperator):
        representation = "multicolor"
        cost = {"keep":4, "erase":4, "crop":4, "recolor":5,
                "translate":6, "duplicate":6}[program.action]
        pred = program.predicate
        if pred is not None:
            cost += sum(getattr(pred, name, None) is not None for name in
                        ("color_set", "includes_color", "shape", "pattern",
                         "min_area", "max_area"))
        cost += int(program.rank is not None)
    elif isinstance(program, CompositeProgram):
        representation = "composite"
        # Complexity penalizes length while preserving the atomic scoring.
        cost = 2 + sum(make_hypothesis(step).complexity for step in program.steps)
    else:
        raise TypeError(f"Unsupported program type: {type(program)}")

    return CandidateHypothesis(
        program=program,
        representation=representation,
        complexity=cost,
        label=f"{representation}:{program!r}",
    )