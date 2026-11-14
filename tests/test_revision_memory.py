"""Revision-memory persistence and exact-prediction invariance tests."""

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


def test_revision_memory_is_persistent_and_bounded():
    task = _border_task()
    _, _, hist = solve_task(task, enable_concept_learning=True,
                            enable_revision_memory=True)
    assert hist[0]['revision_memory_size'] > 0
    assert hist[1]['revision_memory_evaluations'] > 0
    assert hist[-1]['revision_memory_size'] <= 24


def test_revision_memory_does_not_change_exact_predictions():
    task = _border_task()
    baseline, old, _ = solve_task(task, enable_concept_learning=True)
    memory, new, _ = solve_task(task, enable_concept_learning=True,
                                enable_revision_memory=True)
    assert baseline == memory
    assert [h.label for h in old] == [h.label for h in new]