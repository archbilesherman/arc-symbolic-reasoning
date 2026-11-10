import pytest

from src.arc_symbolic.objects import extract_objects
from src.arc_symbolic.relations import (
    left_of,
    right_of,
    above,
    below,
    row_overlap,
    column_overlap,
    touching,
    same_shape,
    same_color,
    same_size,
    describe_pair,
    build_relation_graph,
)


def test_directionality():
    grid = [
        [1, 0, 0, 2],
        [1, 0, 0, 2],
    ]

    a, b = extract_objects(grid)

    assert left_of(a, b)
    assert right_of(b, a)
    assert not left_of(b, a)
    assert not right_of(a, b)

    assert row_overlap(a, b)
    assert not column_overlap(a, b)


def test_vertical_relationships():
    grid = [
        [1, 0],
        [0, 0],
        [0, 2],
    ]

    a, b = extract_objects(grid)

    assert above(a, b)
    assert below(b, a)
    assert not above(b, a)


def test_same_shape_different_color():
    grid = [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 2, 2, 0],
    ]

    a, b = extract_objects(grid)

    assert same_shape(a, b)
    assert same_size(a, b)
    assert not same_color(a, b)


def test_diagonal_touching():
    grid = [
        [1, 0],
        [0, 2],
    ]

    a, b = extract_objects(grid)

    assert not touching(a, b, connectivity=4)
    assert touching(a, b, connectivity=8)


def test_edge_touching():
    grid = [[1, 2]]

    a, b = extract_objects(grid)

    assert touching(a, b, connectivity=4)
    assert touching(b, a, connectivity=4)


def test_pair_description():
    grid = [
        [1, 0, 0, 2],
        [1, 0, 0, 2],
    ]

    a, b = extract_objects(grid)
    relations = describe_pair(a, b)

    assert "left_of" in relations
    assert "row_overlap" in relations
    assert "same_shape" in relations
    assert "same_size" in relations
    assert "touching" not in relations
    assert "same_color" not in relations


def test_relation_graph():
    grid = [[1, 0, 2, 0, 3]]

    objects = extract_objects(grid)
    graph = build_relation_graph(objects)

    assert len(graph) == 6

    assert (0, 1) in graph
    assert (1, 0) in graph

    assert "left_of" in graph[(0, 1)]
    assert "right_of" in graph[(1, 0)]

    assert (0, 0) not in graph


def test_empty_scene():
    assert build_relation_graph([]) == [] or build_relation_graph([]) == {}


def test_invalid_connectivity():
    objects = extract_objects([[1, 2]])

    with pytest.raises(ValueError):
        build_relation_graph(objects, connectivity=6)