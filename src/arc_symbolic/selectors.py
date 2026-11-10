from dataclasses import dataclass
from typing import Literal

from .objects import ARCObject
from .relations import build_relation_graph


RELATION_NAMES = frozenset({
    "left_of",
    "right_of",
    "above",
    "below",
    "row_overlap",
    "column_overlap",
    "touching",
    "same_shape",
    "same_color",
    "same_size",
})


@dataclass(frozen=True)
class AttributePredicate:
    """Constraints on an individual ARC object."""

    color: int | None = None
    min_area: int | None = None
    max_area: int | None = None
    height: int | None = None
    width: int | None = None
    shape: tuple[tuple[int, int], ...] | None = None

    def matches(self, obj: ARCObject) -> bool:
        if self.color is not None and obj.color != self.color:
            return False

        if self.min_area is not None and obj.area < self.min_area:
            return False

        if self.max_area is not None and obj.area > self.max_area:
            return False

        if self.height is not None and obj.height != self.height:
            return False

        if self.width is not None and obj.width != self.width:
            return False

        if self.shape is not None and obj.normalized_shape != self.shape:
            return False

        return True


@dataclass(frozen=True)
class RelationPredicate:
    """
    A relationship between a candidate object and
    at least one reference object.
    """

    relation: str
    reference: AttributePredicate

    def __post_init__(self):
        if self.relation not in RELATION_NAMES:
            raise ValueError(
                f"Unsupported relationship: {self.relation}"
            )


def select_objects(
    objects: list[ARCObject],
    attributes: AttributePredicate | None = None,
    relations: tuple[RelationPredicate, ...] = (),
    rank: Literal["largest", "smallest"] | None = None,
    connectivity: int = 4,
    relation_graph: dict[tuple[int, int], set[str]] | None = None,
) -> list[ARCObject]:
    """
    Select objects satisfying attribute and relational constraints.

    Attribute conditions are combined with AND.
    Every relation must hold with at least one distinct reference object.

    Ranking is applied after filtering.
    """

    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8")

    if rank not in (None, "largest", "smallest"):
        raise ValueError("Rank must be largest, smallest, or None")

    ids = [obj.id for obj in objects]
    if len(ids) != len(set(ids)):
        raise ValueError("Object IDs must be unique")

    if attributes is None:
        attributes = AttributePredicate()

    candidates = [
        obj for obj in objects
        if attributes.matches(obj)
    ]

    if relations:
        if relation_graph is None:
            relation_graph = build_relation_graph(
                objects, connectivity
            )

        filtered = []

        for candidate in candidates:
            valid = True

            for constraint in relations:
                matching_reference = any(
                    reference.id != candidate.id
                    and constraint.reference.matches(reference)
                    and constraint.relation in relation_graph.get(
                        (candidate.id, reference.id), set()
                    )
                    for reference in objects
                )

                if not matching_reference:
                    valid = False
                    break

            if valid:
                filtered.append(candidate)

        candidates = filtered

    # Deterministic spatial ordering
    candidates.sort(key=lambda obj: (obj.bounding_box[0], obj.bounding_box[1], obj.id))

    if not candidates:
        return []

    if rank == "largest":
        return [min(candidates, key=lambda obj: -obj.area)]

    if rank == "smallest":
        return [min(candidates, key=lambda obj: obj.area)]

    return candidates