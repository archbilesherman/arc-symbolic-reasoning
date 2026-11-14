"""Structural concept induction and out-of-sample prediction tests."""

from src.arc_symbolic.concepts import LearnedConcept, object_facts
from src.arc_symbolic.objects import extract_objects
from src.arc_symbolic.solver import solve_task
from src.arc_symbolic.concept_learner import extract_selection_evidence, induce_concepts


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


def _corner_task():
    first = [
        [1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1],
    ]
    second = [
        [0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0],
    ]
    test = [
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1],
    ]
    return {
        'train': [
            {'input': first, 'output': _recolor_outer(first, 'touches_corner')},
            {'input': second, 'output': _recolor_outer(second, 'touches_corner')},
        ],
        'test': [{'input': test}],
        'expected': _recolor_outer(test, 'touches_corner'),
    }


def test_corner_concept_requires_expanded_vocabulary():
    task = _corner_task()
    ordinary = extract_selection_evidence(task['train'], 'recolor', 2)
    assert ordinary is not None
    assert not induce_concepts(*ordinary)
    expanded = extract_selection_evidence(task['train'], 'recolor', 2,
                                           expanded_vocabulary=True)
    concepts = induce_concepts(*expanded)
    assert LearnedConcept(('touches_corner',)) in concepts


def test_expanded_vocabulary_recovers_corner_task():
    task = _corner_task()
    a3, s3, _ = solve_task(task, enable_concept_learning=True,
                            enable_revision_memory=True,
                            enable_counterexample_revision=True)
    a4, s4, h4 = solve_task(task, enable_concept_learning=True,
                            enable_revision_memory=True,
                            enable_counterexample_revision=True,
                            expanded_concept_vocabulary=True)
    assert not s3 or task['expected'] not in (a3[0]['attempt_1'], a3[0]['attempt_2'])
    assert s4
    assert task['expected'] in (a4[0]['attempt_1'], a4[0]['attempt_2'])
    assert any(x['version_space_recovered'] for x in h4)