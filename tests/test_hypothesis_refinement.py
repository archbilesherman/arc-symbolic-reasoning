"""Counterexample-guided refinement and exact verification tests."""

import numpy as np

from src.arc_symbolic.concepts import object_facts
from src.arc_symbolic.objects import extract_objects
from src.arc_symbolic.solver import solve_task


def _make_border_grid():
    return [
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ]


def _recolor_outer(grid, feature):
    arr = [row.copy() for row in grid]
    objects = extract_objects(grid)
    for obj in objects:
        kwargs = {'expanded_vocabulary': True} if feature == 'touches_corner' else {}
        if feature in object_facts(obj, objects, (len(grid), len(grid[0])), **kwargs):
            for r, c in obj.cells:
                arr[r][c] = 2
    return arr


def _border_task():
    first = _make_border_grid()
    second = [
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ]
    test = [
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
    return {
        'train': [
            {'input': first, 'output': _recolor_outer(first, 'touches_border')},
            {'input': second, 'output': _recolor_outer(second, 'touches_border')},
        ],
        'test': [{'input': test}],
        'expected': _recolor_outer(test, 'touches_border'),
    }


def test_refinement_tracks_parent_and_verifies_all_examples():
    task = _border_task()
    answer, survivors, history = solve_task(
        task, enable_concept_learning=True,
        enable_revision_memory=True,
        enable_counterexample_revision=True,
    )
    assert answer[0]['attempt_1'] == task['expected'] or answer[0]['attempt_2'] == task['expected']
    assert history[0]['revision_recovered']
    assert any(h.parent_label is not None for h in survivors)
    for hypothesis in survivors:
        for example in task['train']:
            assert np.array_equal(hypothesis.apply(example['input']), example['output'])