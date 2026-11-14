"""Failure classification and evidence diagnostics regression tests."""

from src.arc_symbolic.failure_analysis import diagnose_prediction, evidence_rejection_counts
from src.arc_symbolic.concepts import object_facts
from src.arc_symbolic.objects import extract_objects
from src.arc_symbolic.solver import solve_task
from src.arc_symbolic.evaluate import evaluate_arc


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


def test_diagnostics_classify_counterexamples():
    before = [[1, 0, 1]]
    desired = [[2, 0, 1]]
    assert diagnose_prediction(before, desired, [[2, 0, 2]]).kind == 'overselection'
    assert diagnose_prediction(before, desired, [[1, 0, 1]]).kind == 'underselection'
    assert diagnose_prediction(before, desired, [[3, 0, 1]]).kind == 'wrong_values'
    assert diagnose_prediction(before, desired, [[2]]).kind == 'wrong_dimensions'
    assert diagnose_prediction(before, desired, None).kind == 'invalid_operator'


def test_evidence_rejection_reasons_are_recorded():
    ex = {'input': [[1, 0]], 'output': [[1, 2]]}
    counts = evidence_rejection_counts([ex])
    assert counts['new_colored_cells'] > 0


def test_failure_diagnostics_preserve_baseline_accuracy():
    task = _border_task()
    inputs = {'task': {'train': task['train'], 'test': task['test']}}
    solutions = {'task': [task['expected']]}
    old = evaluate_arc(inputs, solutions, configuration='concepts')
    new = evaluate_arc(inputs, solutions, configuration='v4_diagnostics')
    assert old['summary']['tasks_solved'] == new['summary']['tasks_solved']
    assert old['summary']['tasks_with_consistent_hypothesis'] == new['summary']['tasks_with_consistent_hypothesis']
    assert new['summary']['evidence_rejection_counts']