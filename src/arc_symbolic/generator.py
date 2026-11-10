
from collections import Counter

import numpy as np

from .hypotheses import CandidateHypothesis, make_hypothesis
from .objects import extract_objects
from .operators import ObjectOperator
from .pixels import GEOMETRIC_OPS, PixelOperator
from .selectors import AttributePredicate, RelationPredicate


RELATIONS = (
    "left_of",
    "right_of",
    "above",
    "below",
    "touching",
    "same_shape",
    "same_color",
    "same_size",
    "row_overlap",
    "column_overlap",
)


def _motion_offsets(examples, connectivity: int, limit: int = 12):
    """Infer plausible offsets from matching object shapes."""

    counts = Counter()

    for example in examples:
        before = np.asarray(example["input"])
        after = np.asarray(example["output"])

        if before.shape != after.shape:
            continue

        before_objs = extract_objects(before, connectivity)
        after_objs = extract_objects(after, connectivity)

        for a in before_objs:
            for b in after_objs:
                if a.normalized_shape != b.normalized_shape:
                    continue

                dr = b.bounding_box[0] - a.bounding_box[0]
                dc = b.bounding_box[1] - a.bounding_box[1]

                if (dr, dc) != (0, 0):
                    counts[(dr, dc)] += 1

    return [
        pair for pair, _ in sorted(
            counts.items(),
            key=lambda item: (
                -item[1],
                abs(item[0][0]) + abs(item[0][1]),
                item[0],
            ),
        )[:limit]
    ]


def generate_hypotheses(
    task: dict,
    *,
    include_objects: bool = True,
    include_relations: bool = True,
    include_motion: bool = True,
) -> list[CandidateHypothesis]:
    """Enumerate single-step hypotheses from training pairs."""

    examples = task["train"]

    input_colors = sorted({
        int(c)
        for e in examples
        for c in np.unique(e["input"])
    })

    output_colors = sorted({
        int(c)
        for e in examples
        for c in np.unique(e["output"])
    })

    results = []
    seen = set()

    def add(program):
        if program not in seen:
            seen.add(program)
            results.append(make_hypothesis(program))

    # Tier 0: Original pixel operators
    for operation in GEOMETRIC_OPS:
        add(PixelOperator("geometry", operation=operation))

    for source in input_colors:
        for target in output_colors:
            if source != target:
                add(PixelOperator(
                    "recolor",
                    source=source,
                    target=target,
                ))

    if not include_objects:
        return sorted(
            results,
            key=lambda h: (h.complexity, h.label),
        )

    # Object representations: 4- and 8-connectivity
    for connectivity in (4, 8):
        objects = [
            obj
            for example in examples
            for obj in extract_objects(
                example["input"], connectivity
            )
        ]

        areas = sorted({obj.area for obj in objects})[:12]

        shapes = sorted({obj.normalized_shape for obj in objects})[:12]

        # Tiers 1-2: Attribute and ranked selectors
        selections = [(None, (), None)]

        selections += [(AttributePredicate(color=c), (), None)for c in input_colors if c != 0]

        selections += [(AttributePredicate(min_area=area, max_area=area), (), None,) for area in areas]

        selections += [(AttributePredicate(shape=shape), (), None) for shape in shapes]

        selections += [(None, (), rank) for rank in ("largest", "smallest")]

        # Tiers 3-4: Relational selectors
        if include_relations:
            for relation in RELATIONS:
                for color in input_colors:
                    if color == 0:
                        continue

                    constraint = (RelationPredicate(relation, AttributePredicate(color=color)),)

                    selections.append((None, constraint, None))

                    # Combine candidate color with a relation.
                    for candidate_color in input_colors:
                        if candidate_color != 0:
                            selections.append((AttributePredicate(color=candidate_color), constraint, None,))

        # Generate executable object transformations
        for attributes, relations, rank in selections:
            common = dict(attributes=attributes, relations=relations, rank=rank, connectivity=connectivity)

            for action in ("keep", "erase", "crop"):
                add(ObjectOperator(action=action, **common))

            for target_color in output_colors:
                add(ObjectOperator(action="recolor", target_color=target_color, **common))

        # Evidence-guided movement and duplication
        if include_motion:
            offsets = _motion_offsets(examples, connectivity)

            motion_selections = selections[:1 + len([c for c in input_colors if c != 0])]

            motion_selections += [(None, (), "largest"), (None, (), "smallest")]

            for attributes, relations, rank in motion_selections:
                common = dict(attributes=attributes, relations=relations, rank=rank, connectivity=connectivity)

                for offset in offsets:
                    for action in ("translate", "duplicate"):
                        add(ObjectOperator(action=action, offset=offset, **common))

    return sorted(results, key=lambda h: (h.complexity, h.label))