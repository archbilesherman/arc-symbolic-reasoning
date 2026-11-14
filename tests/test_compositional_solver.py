"""V5 solver integration, task filtering and backward-compatibility checks."""
import numpy as np
from src.arc_symbolic.evaluate import evaluate_arc, CONFIGURATIONS
from src.arc_symbolic.solver import solve_task


def test_composition_recovers_task_unavailable_to_single_step_solver():
    first = [[0,0,0,0,0],[0,1,1,0,3],[0,1,1,0,0],[0,0,0,0,0]]
    second = [[0,3,0,0,0],[0,1,1,0,0],[0,1,1,0,0],[0,0,0,0,0]]
    task = {'train':[{'input':first,'output':[[2,2],[2,2]]},
                     {'input':second,'output':[[2,2],[2,2]]}],
            'test':[{'input':first}]}
    base, base_survivors, _ = solve_task(task)
    answers, survivors, history = solve_task(task, enable_composition=True)
    assert not base_survivors
    assert survivors
    assert [[2,2],[2,2]] in (answers[0]['attempt_1'],answers[0]['attempt_2'])
    assert any(h.representation == 'composite' for h in survivors)
    assert sum(x['composition_pair_evaluations'] for x in history) > 0


def test_existing_solution_is_preserved_and_composition_skipped():
    task = {
        'train':[{'input':[[1,0],[2,3]],'output':[[0,3],[1,2]]},
                 {'input':[[4,5],[0,6]],'output':[[5,6],[4,0]]}],
        'test':[{'input':[[7,0],[8,9]]}],
    }
    old, old_survivors, _ = solve_task(task)
    new, new_survivors, history = solve_task(
        task, enable_multicolor_objects=True, enable_composition=True)
    assert old == new
    assert [h.label for h in old_survivors] == [h.label for h in new_survivors]
    assert sum(row['composition_pair_evaluations'] for row in history) == 0


def test_evaluator_supports_targeted_tasks_and_v5_configurations():
    assert all(name in CONFIGURATIONS for name in
               ('v4_full','v5_multicolor','v5_composition','v5_full'))
    challenges = {
        'rot': {
            'train':[{'input':[[1,0],[2,3]],'output':[[0,3],[1,2]]}],
            'test':[{'input':[[4,5],[0,6]]}]},
        'other': {
            'train':[{'input':[[1]],'output':[[1]]}],
            'test':[{'input':[[2]]}]},
    }
    solutions = {'rot':[[[5,6],[4,0]]], 'other':[[[2]]]}
    report = evaluate_arc(challenges,solutions,configuration='v5_full',task_ids=['rot'])
    assert report['summary']['tasks'] == 1
    assert list(report['tasks']) == ['rot']
    assert report['summary']['tasks_solved'] == 1
    assert 'composition_pair_evaluations_total' in report['summary']


def test_multicolor_recovers_targeted_grouping_without_relational_selectors():
    first = [[0,0,0,0,0],[0,1,2,0,0],[0,0,0,0,0],[0,0,0,0,3]]
    second = [[3,0,0,0,0],[0,0,0,0,0],[0,1,2,0,0],[0,0,0,0,0]]
    task = {'train':[{'input':first,'output':[[1,2]]},
                     {'input':second,'output':[[1,2]]}],
            'test':[{'input':second}]}
    _, original, _ = solve_task(task,include_relations=False)
    predictions, enriched, _ = solve_task(
        task,include_relations=False,enable_multicolor_objects=True)
    assert not original
    assert enriched and any(h.representation == 'multicolor' for h in enriched)
    assert [[1,2]] in (predictions[0]['attempt_1'], predictions[0]['attempt_2'])