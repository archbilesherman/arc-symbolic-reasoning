import numpy as np

from src.arc_symbolic.objects import extract_objects


def test_four_vs_eight_connectivity():
    grid = [
        [1, 0, 1, 0],
        [0, 1, 0, 2],
        [0, 0, 0, 2],
    ]

    objects_4 = extract_objects(grid, connectivity=4)
    objects_8 = extract_objects(grid, connectivity=8)

    assert len(objects_4) == 4
    assert len(objects_8) == 2


def test_object_attributes():
    grid = [
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ]

    obj = extract_objects(grid)[0]

    assert obj.id == 0
    assert obj.color == 1
    assert obj.area == 3
    assert obj.bounding_box == (0, 1, 1, 2)
    assert obj.height == 2
    assert obj.width == 2
    assert obj.centroid == (1 / 3, 4 / 3)
    assert obj.normalized_shape == (
        (0, 0), (0, 1), (1, 0)
    )


def test_different_positions_same_shape():
    grid = [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 2, 2, 0],
    ]

    objects = extract_objects(grid)

    assert len(objects) == 2
    assert objects[0].normalized_shape == objects[1].normalized_shape


def test_input_unchanged():
    grid = np.array([[1, 0], [1, 2]])
    original = grid.copy()

    extract_objects(grid)

    assert np.array_equal(grid, original)


def test_empty_scene():
    assert extract_objects([[0, 0], [0, 0]]) == []