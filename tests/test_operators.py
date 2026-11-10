import numpy as np
import pytest

from src.arc_symbolic.operators import (
    ObjectOperator,
    apply_object_operator,
)

from src.arc_symbolic.selectors import (
    AttributePredicate,
    RelationPredicate,
)


def test_recolor_and_erase():
    grid = [
        [1, 1, 0, 2],
        [0, 0, 0, 2],
    ]

    original = np.array(grid)

    recolored = apply_object_operator(
        grid,
        "recolor",
        attributes=AttributePredicate(color=1),
        target_color=3,
    )

    assert recolored.tolist() == [
        [3, 3, 0, 2],
        [0, 0, 0, 2],
    ]

    assert np.array_equal(original, grid)

    erased = apply_object_operator(
        grid,
        "erase",
        attributes=AttributePredicate(color=2),
    )

    assert erased.tolist() == [
        [1, 1, 0, 0],
        [0, 0, 0, 0],
    ]


def test_keep_and_crop():
    grid = [
        [1, 1, 0, 2],
        [1, 0, 0, 2],
        [0, 0, 0, 0],
    ]

    attrs = AttributePredicate(color=1)

    kept = apply_object_operator(
        grid, "keep", attributes=attrs
    )

    assert kept.tolist() == [
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    cropped = apply_object_operator(
        grid, "crop", attributes=attrs
    )

    assert cropped.tolist() == [
        [1, 1],
        [1, 0],
    ]


def test_translate_and_duplicate():
    grid = [
        [1, 0, 0, 2, 0],
        [1, 0, 0, 2, 0],
    ]

    attrs = AttributePredicate(color=1)

    moved = apply_object_operator(
        grid,
        "translate",
        attributes=attrs,
        offset=(0, 1),
    )

    assert moved.tolist() == [
        [0, 1, 0, 2, 0],
        [0, 1, 0, 2, 0],
    ]

    copied = apply_object_operator(
        grid,
        "duplicate",
        attributes=attrs,
        offset=(0, 1),
    )

    assert copied.tolist() == [
        [1, 1, 0, 2, 0],
        [1, 1, 0, 2, 0],
    ]


def test_unsafe_moves_rejected():
    grid = [
        [1, 0, 2],
        [0, 0, 0],
    ]

    attrs = AttributePredicate(color=1)

    # Collision with an existing object.
    assert apply_object_operator(
        grid, "translate",
        attributes=attrs,
        offset=(0, 2),
    ) is None

    # Out of bounds.
    assert apply_object_operator(
        grid, "translate",
        attributes=attrs,
        offset=(-1, 0),
    ) is None

    # Cannot duplicate into the same occupied position.
    assert apply_object_operator(
        grid, "duplicate",
        attributes=attrs,
        offset=(0, 0),
    ) is None


def test_relation_driven_operator():
    grid = [[1, 0, 2, 0, 3]]

    operator = ObjectOperator(
        action="recolor",
        relations=(
            RelationPredicate(
                "left_of",
                AttributePredicate(color=2),
            ),
        ),
        target_color=4,
    )

    result = operator.apply(grid)

    assert result.tolist() == [
        [4, 0, 2, 0, 3]
    ]


def test_no_matching_object():
    result = apply_object_operator(
        [[1]],
        "erase",
        attributes=AttributePredicate(color=9),
    )

    assert result is None


def test_invalid_parameters():
    with pytest.raises(ValueError):
        apply_object_operator(
            [[1]], "recolor", target_color=12
        )

    with pytest.raises(ValueError):
        apply_object_operator(
            [[1]], "translate", offset=None
        )

    with pytest.raises(ValueError):
        apply_object_operator(
            [[1]], "duplicate", offset=(1, 0.5)
        )

    with pytest.raises(ValueError):
        apply_object_operator(
            [[1]], "not_a_real_action"
        )


def test_simultaneous_translation():
    grid = [[1, 0, 2, 0, 0]]

    result = apply_object_operator(
        grid,
        "translate",
        offset=(0, 2),
    )

    assert result.tolist() == [
        [0, 0, 1, 0, 2]
    ]