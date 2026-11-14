"""Alternative connected-object extraction and color-preserving operations."""
import numpy as np
import pytest

from src.arc_symbolic.multicolor import extract_multicolor_objects, propose_multicolor_hypotheses
from src.arc_symbolic.objects import extract_objects
from src.arc_symbolic.operators import MultiColorOperator
from src.arc_symbolic.selectors import MultiColorPredicate


def test_multicolor_connectivity_and_color_histogram():
    grid = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    assert len(extract_objects(grid, connectivity=4)) == 3
    assert len(extract_multicolor_objects(grid, connectivity=4)) == 3
    connected = extract_multicolor_objects(grid, connectivity=8)
    assert len(connected) == 1
    obj = connected[0]
    assert obj.area == 3
    assert obj.color_set == frozenset({1, 2, 3})
    assert obj.color_histogram == ((1, 1), (2, 1), (3, 1))
    assert obj.normalized_pattern == ((0, 0, 1), (1, 1, 2), (2, 2, 3))


def test_color_preserving_crop_and_erase():
    grid = [[0, 0, 0, 0, 0],
            [0, 1, 2, 0, 3],
            [0, 1, 2, 0, 0],
            [0, 0, 0, 0, 0]]
    chosen = MultiColorPredicate(color_set=frozenset({1, 2}))
    crop = MultiColorOperator('crop', chosen).apply(grid)
    erase = MultiColorOperator('erase', chosen).apply(grid)
    assert crop.tolist() == [[1, 2], [1, 2]]
    assert erase.tolist() == [[0,0,0,0,0],[0,0,0,0,3],
                              [0,0,0,0,0],[0,0,0,0,0]]


def test_multicolor_translation_preserves_each_color():
    grid = [[1,2,0],[0,0,0],[0,0,0]]
    result = MultiColorOperator('translate', offset=(1,0)).apply(grid)
    assert result.tolist() == [[0,0,0],[1,2,0],[0,0,0]]
    assert MultiColorOperator('translate', offset=(0,-1)).apply(grid) is None


def test_multicolor_validation_and_candidate_budget():
    with pytest.raises(ValueError):
        extract_multicolor_objects([[10]])
    with pytest.raises(ValueError):
        extract_multicolor_objects([[1]], connectivity=6)
    examples = [{'input':[[1,2,0],[0,0,3]], 'output':[[1,2]]}]
    hypotheses = propose_multicolor_hypotheses(examples,max_candidates=8)
    assert 0 < len(hypotheses) <= 8
    assert all(h.representation == 'multicolor' for h in hypotheses)