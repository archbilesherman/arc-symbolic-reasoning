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

# Stage 4 additions; enabled only in v4_full.
EXPANDED_FEATURES = (
    "touches_corner", "touches_two_edges", "single_cell",
    "horizontal_line", "vertical_line", "symmetric_horizontal",
    "symmetric_vertical", "has_hole", "center_row", "center_col",
)


def _encloses_hole(obj: ARCObject) -> bool:
    """A background pocket enclosed by this object's monochrome cells."""
    from collections import deque
    top, left, bottom, right = obj.bounding_box
    height, width = bottom - top + 1, right - left + 1
    if height < 3 or width < 3:
        return False
    occupied = {(r - top, c - left) for r, c in obj.cells}
    vacant = {(r, c) for r in range(height) for c in range(width)
              if (r, c) not in occupied}
    if not vacant:
        return False
    seeds = [p for p in vacant if p[0] in (0, height - 1)
             or p[1] in (0, width - 1)]
    outside = set(seeds)
    queue = deque(seeds)
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_cell = (r + dr, c + dc)
            if next_cell in vacant and next_cell not in outside:
                outside.add(next_cell)
                queue.append(next_cell)
    return bool(vacant - outside)


def _expanded_facts(obj: ARCObject, grid_shape):
    height, width = grid_shape
    top, left, bottom, right = obj.bounding_box
    relative = {(r - top, c - left) for r, c in obj.cells}
    edge_count = sum((top == 0, left == 0,
                      bottom == height - 1, right == width - 1))
    properties = {
        "touches_corner": any((r, c) in obj.cells for r in (0, height - 1)
                               for c in (0, width - 1)),
        "touches_two_edges": edge_count >= 2,
        "single_cell": obj.area == 1,
        "horizontal_line": obj.height == 1 and obj.width > 1
                           and obj.area == obj.width,
        "vertical_line": obj.width == 1 and obj.height > 1
                         and obj.area == obj.height,
        "symmetric_horizontal": all((r, obj.width - 1 - c) in relative
                                    for r, c in relative),
        "symmetric_vertical": all((obj.height - 1 - r, c) in relative
                                  for r, c in relative),
        "has_hole": _encloses_hole(obj),
        "center_row": top + bottom == height - 1,
        "center_col": left + right == width - 1,
    }
    return {name if holds else f"not:{name}" for name, holds in properties.items()}


def object_facts(
    obj: ARCObject,
    objects: tuple[ARCObject, ...] | list[ARCObject],
    grid_shape: tuple[int, int],
    connectivity: int = 4,
    *,
    include_relations: bool = True,
    expanded_vocabulary: bool = False,
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

    if expanded_vocabulary:
        features.update(_expanded_facts(obj, grid_shape))

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
            expanded_vocabulary=any(
                lit.removeprefix("not:") in EXPANDED_FEATURES
                for lit in self.literals
            ),
        )
        return all(literal in facts for literal in self.literals)