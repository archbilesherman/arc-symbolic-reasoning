import pytest

from src.arc_symbolic.objects import extract_objects
from src.arc_symbolic.selectors import (
    AttributePredicate,
    RelationPredicate,
    select_objects,
)


@pytest.fixture
def scene():
    return extract_objects([
        [1, 0, 0, 2, 0, 3],
        [1, 0, 0, 2, 0, 3],
        [0, 0, 0, 0, 0, 0],
    ])


def test_select_by_color(scene):
    selected = select_objects(
        scene,
        attributes=AttributePredicate(color=2)
    )

    assert len(selected) == 1
    assert selected[0].color == 2


def test_select_by_area(scene):
    selected = select_objects(
        scene,
        attributes=AttributePredicate(
            min_area=2,
            max_area=2,
        )
    )

    assert len(selected) == 3


def test_select_by_shape(scene):
    shape = scene[0].normalized_shape

    selected = select_objects(
        scene,
        attributes=AttributePredicate(shape=shape)
    )

    assert len(selected) == 3


def test_select_left_of_reference(scene):
    selected = select_objects(
        scene,
        relations=(
            RelationPredicate(
                "left_of",
                AttributePredicate(color=3),
            ),
        ),
    )

    assert {obj.color for obj in selected} == {1, 2}


def test_compound_predicate(scene):
    selected = select_objects(
        scene,
        attributes=AttributePredicate(color=1),
        relations=(
            RelationPredicate(
                "left_of",
                AttributePredicate(color=3),
            ),
            RelationPredicate(
                "same_shape",
                AttributePredicate(color=2),
            ),
        ),
    )

    assert len(selected) == 1
    assert selected[0].color == 1


def test_ranking():
    objects = extract_objects([
        [1, 1, 0, 2],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
    ])

    largest = select_objects(objects, rank="largest")
    smallest = select_objects(objects, rank="smallest")

    assert largest[0].area == 4
    assert smallest[0].area == 1


def test_no_match(scene):
    selected = select_objects(
        scene,
        attributes=AttributePredicate(color=9)
    )

    assert selected == []


def test_invalid_relation():
    with pytest.raises(ValueError):
        RelationPredicate(
            "inside",
            AttributePredicate(color=2),
        )


def test_empty_scene():
    assert select_objects([]) == []