"""Executable, task-local concepts built from symbolic object properties.

A learned concept is a conjunction of generalizable literals.  It never
stores example-specific object IDs or pixel coordinates.
"""

from dataclasses import dataclass

from .objects import ARCObject
from .relations import describe_pair


STRUCTURAL_FEATURES = (
    "touches_border",
    "solid_rectangle",
    "square_bbox",
    "largest",
    "smallest",
)


def object_facts(
    obj: ARCObject,
    objects: tuple[ARCObject, ...] | list[ARCObject],
    grid_shape: tuple[int, int],
    connectivity: int = 4,
    *,
    include_relations: bool = True,
) -> frozenset[str]:
    """Describe an object using reusable, grid-relative properties."""
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must contain height and width")
    height, width = grid_shape
    if height < 1 or width < 1:
        raise ValueError("grid_shape dimensions must be positive")

    top, left, bottom, right = obj.bounding_box
    features = {f"color:{obj.color}"}

    properties = {
        "touches_border": (
            top == 0 or left == 0
            or bottom == height - 1 or right == width - 1
        ),
        "solid_rectangle": obj.area == obj.height * obj.width,
        "square_bbox": obj.height == obj.width,
        "largest": all(obj.area >= other.area for other in objects),
        "smallest": all(obj.area <= other.area for other in objects),
    }
    for name, holds in properties.items():
        features.add(name if holds else f"not:{name}")

    if include_relations:
        for other in objects:
            if other.id == obj.id:
                continue
            for relation in describe_pair(obj, other, connectivity):
                features.add(f"relation:{relation}:color:{other.color}")

    return frozenset(features)


@dataclass(frozen=True)
class LearnedConcept:
    """A reusable conjunction of symbolic property literals."""

    literals: tuple[str, ...]

    def __post_init__(self):
        if not self.literals:
            raise ValueError("A concept requires at least one literal")
        normalized = tuple(sorted(set(self.literals)))
        if len(normalized) != len(self.literals):
            raise ValueError("Duplicate concept literals")
        object.__setattr__(self, "literals", normalized)

    @property
    def needs_relations(self) -> bool:
        return any(x.startswith("relation:") for x in self.literals)

    def matches(
        self,
        obj: ARCObject,
        objects: tuple[ARCObject, ...] | list[ARCObject],
        grid_shape: tuple[int, int],
        connectivity: int = 4,
    ) -> bool:
        facts = object_facts(
            obj, objects, grid_shape, connectivity,
            include_relations=self.needs_relations,
        )
        return all(literal in facts for literal in self.literals)