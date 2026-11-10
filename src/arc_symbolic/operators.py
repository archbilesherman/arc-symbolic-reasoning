from dataclasses import dataclass
from typing import Literal

import numpy as np

from .objects import extract_objects
from .selectors import (
    AttributePredicate,
    RelationPredicate,
    select_objects,
)

from functools import lru_cache
from .relations import build_relation_graph


Action = Literal["recolor", "erase", "keep", "crop", "translate", "duplicate"]

Rank = Literal["largest", "smallest"]

@lru_cache(maxsize=128)
def _cached_relation_scene(shape, grid_bytes, connectivity):
    arr = np.frombuffer(
        grid_bytes, dtype=np.int64
    ).reshape(shape)

    objects = tuple(extract_objects(arr, connectivity=connectivity))

    graph = build_relation_graph(list(objects), connectivity=connectivity)

    return objects, graph


def apply_object_operator(
    grid,
    action: Action,
    *,
    attributes: AttributePredicate | None = None,
    relations: tuple[RelationPredicate, ...] = (),
    rank: Rank | None = None,
    connectivity: int = 4,
    target_color: int | None = None,
    offset: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """
    Apply an object-level transformation to an ARC grid.
    Objects are selected from the original grid using attribute and relational predicates.
    Returns None if no objects match or if a spatial transformation cannot be performed safely.
    """

    allowed = {"recolor", "erase", "keep", "crop", "translate", "duplicate"}

    if action not in allowed:
        raise ValueError(f"Unsupported action: {action}")

    if action == "recolor" and (
        not isinstance(target_color, (int, np.integer))
        or not 0 <= target_color <= 9
    ):
        raise ValueError("recolor requires target_color between 0 and 9")

    if action in ("translate", "duplicate"):
        if (
            offset is None
            or len(offset) != 2
            or not all(
                isinstance(v, (int, np.integer))
                for v in offset
            )
        ):
            raise ValueError(
                "translate/duplicate require an integer "
                "(dr, dc) offset"
            )

    # Extract objects from the original grid.
    arr = np.asarray(grid, dtype=np.int64)

    if relations:
        objects, graph = _cached_relation_scene(
            arr.shape, arr.tobytes(), connectivity
        )
    else:
        objects = extract_objects(
            arr, connectivity=connectivity
        )
        graph = None

    chosen = select_objects(
        objects,
        attributes=attributes,
        relations=relations,
        rank=rank,
        connectivity=connectivity,
        relation_graph=graph,
    )


    if not chosen:
        return None

    selected_cells = [
        cell
        for obj in chosen
        for cell in obj.cells
    ]

    # Recolor and Erase
    if action in ("recolor", "erase"):
        result = arr.copy()

        color = (
            int(target_color)
            if action == "recolor"
            else 0
        )

        for r, c in selected_cells:
            result[r, c] = color

        return result

    # Keep and Crop
    if action in ("keep", "crop"):
        result = np.zeros_like(arr)

        for r, c in selected_cells:
            result[r, c] = arr[r, c]

        if action == "crop":
            top = min(r for r, _ in selected_cells)
            bottom = max(r for r, _ in selected_cells)
            left = min(c for _, c in selected_cells)
            right = max(c for _, c in selected_cells)

            return result[top:bottom + 1, left:right + 1].copy()
        return result

    # Translate and Duplicate
    dr, dc = offset
    height, width = arr.shape

    destinations = []

    for r, c in selected_cells:
        nr = r + dr
        nc = c + dc

        # Reject movement outside the grid.
        if not (0 <= nr < height and 0 <= nc < width):
            return None

        destinations.append((nr, nc, int(arr[r, c])))

    result = arr.copy()

    if action == "translate":
        # Remove all selected objects before placing them.
        # This allows simultaneous movement over positions previously occupied by other selected objects.
        for r, c in selected_cells:
            result[r, c] = 0

    # Prevent overwriting unselected occupied cells.
    if any(
        result[r, c] != 0
        for r, c, _ in destinations
    ):
        return None

    for r, c, color in destinations:
        result[r, c] = color

    return result


# Declarative Operator Representation
@dataclass(frozen=True)
class ObjectOperator:
    """
    An executable symbolic object transformation.
    This representation can later be used as a component
    of a composed transformation program.
    """

    action: Action
    attributes: AttributePredicate | None = None
    relations: tuple[RelationPredicate, ...] = ()
    rank: Rank | None = None
    connectivity: int = 4
    target_color: int | None = None
    offset: tuple[int, int] | None = None

    def apply(self, grid) -> np.ndarray | None:
        return apply_object_operator(
            grid,
            self.action,
            attributes=self.attributes,
            relations=self.relations,
            rank=self.rank,
            connectivity=self.connectivity,
            target_color=self.target_color,
            offset=self.offset,
        )