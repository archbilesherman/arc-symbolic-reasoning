from .objects import ARCObject

# Spatial Relationships

def left_of(a: ARCObject, b: ARCObject) -> bool:
    """A is entirely to the left of B."""
    return a.bounding_box[3] < b.bounding_box[1]


def right_of(a: ARCObject, b: ARCObject) -> bool:
    """A is entirely to the right of B."""
    return left_of(b, a)


def above(a: ARCObject, b: ARCObject) -> bool:
    """A is entirely above B."""
    return a.bounding_box[2] < b.bounding_box[0]


def below(a: ARCObject, b: ARCObject) -> bool:
    """A is entirely below B."""
    return above(b, a)


def row_overlap(a: ARCObject, b: ARCObject) -> bool:
    """Bounding boxes share at least one row."""
    a_top, _, a_bottom, _ = a.bounding_box
    b_top, _, b_bottom, _ = b.bounding_box

    return max(a_top, b_top) <= min(a_bottom, b_bottom)


def column_overlap(a: ARCObject, b: ARCObject) -> bool:
    """Bounding boxes share at least one column."""
    _, a_left, _, a_right = a.bounding_box
    _, b_left, _, b_right = b.bounding_box

    return max(a_left, b_left) <= min(a_right, b_right)


def touching(a: ARCObject, b: ARCObject, connectivity: int = 4,) -> bool:
    """Whether any cells from A and B are adjacent."""

    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8")

    b_cells = set(b.cells)

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    if connectivity == 8:
        neighbors += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for row, col in a.cells:
        for dr, dc in neighbors:
            if (row + dr, col + dc) in b_cells:
                return True

    return False


# Attribute Relationships

def same_shape(a: ARCObject, b: ARCObject) -> bool:
    """Shapes match after translation normalization."""
    return a.normalized_shape == b.normalized_shape


def same_color(a: ARCObject, b: ARCObject) -> bool:
    return a.color == b.color


def same_size(a: ARCObject, b: ARCObject) -> bool:
    """Objects have equal cell counts."""
    return a.area == b.area


# Pairwise Relation Description

def describe_pair(
    a: ARCObject,
    b: ARCObject,
    connectivity: int = 4,
) -> set[str]:
    """Return all recognized relations from A to B."""

    relations = set()

    predicates = {
        "left_of": left_of,
        "right_of": right_of,
        "above": above,
        "below": below,
        "row_overlap": row_overlap,
        "column_overlap": column_overlap,
        "same_shape": same_shape,
        "same_color": same_color,
        "same_size": same_size,
    }

    for name, predicate in predicates.items():
        if predicate(a, b):
            relations.add(name)

    if touching(a, b, connectivity):
        relations.add("touching")

    return relations


# Relational Scene Graph

def build_relation_graph(
    objects: list[ARCObject],
    connectivity: int = 4,
) -> dict[tuple[int, int], set[str]]:
    """
    Construct directed relationships between every
    pair of distinct objects.

    Graph keys are (source_id, target_id).
    """

    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8")

    ids = [obj.id for obj in objects]
    if len(ids) != len(set(ids)):
        raise ValueError("Object IDs must be unique")

    graph = {}

    for a in objects:
        for b in objects:
            if a.id == b.id:
                continue

            graph[(a.id, b.id)] = describe_pair(
                a, b, connectivity
            )

    return graph