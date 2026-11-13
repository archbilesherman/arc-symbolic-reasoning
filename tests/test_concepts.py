import pytest

from src.arc_symbolic.concepts import LearnedConcept, object_facts
from src.arc_symbolic.objects import extract_objects
from src.arc_symbolic.selectors import select_objects


def test_object_facts_boundary_and_interior():
    scene = [
        [1, 0, 0, 0, 2],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    objects = extract_objects(scene)
    boundary = objects[0]
    interior = objects[2]
    assert "touches_border" in object_facts(boundary, objects, (5, 5))
    assert "not:touches_border" in object_facts(interior, objects, (5, 5))


def test_learned_conjunction_filters_objects():
    scene = [
        [1, 0, 0, 0, 2],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    objects = extract_objects(scene)
    concept = LearnedConcept(("touches_border", "color:1"))
    selected = select_objects(objects, concept=concept, grid_shape=(5, 5))
    assert len(selected) == 1
    assert selected[0].color == 1
    assert selected[0].bounding_box[:2] == (0, 0)


def test_concepts_are_hashable_and_canonical():
    one = LearnedConcept(("touches_border", "color:1"))
    two = LearnedConcept(("color:1", "touches_border"))
    assert one == two
    assert len({one, two}) == 1
    with pytest.raises(ValueError):
        LearnedConcept(())
    with pytest.raises(ValueError):
        LearnedConcept(("color:1", "color:1"))