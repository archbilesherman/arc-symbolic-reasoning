from src.arc_symbolic.concept_learner import (
    extract_selection_evidence,
    induce_concepts,
    propose_concept_hypotheses,
    verify_hypotheses,
)
from src.arc_symbolic.solver import solve_task


def _recolor_border(grid, source=1, target=2):
    result = [row.copy() for row in grid]
    height, width = len(grid), len(grid[0])
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == source and (
                r == 0 or c == 0 or r == height - 1 or c == width - 1
            ):
                result[r][c] = target
    return result


def border_task():
    first = [
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ]
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
        "train": [
            {"input": first, "output": _recolor_border(first)},
            {"input": second, "output": _recolor_border(second)},
        ],
        "test": [{"input": test}],
        "expected": _recolor_border(test),
    }


def test_induces_border_concept_from_evidence():
    task = border_task()
    evidence = extract_selection_evidence(task["train"], "recolor", 2)
    assert evidence is not None
    concepts = induce_concepts(*evidence)
    assert any(c.literals == ("touches_border",) for c in concepts)


def test_v3_recovers_empty_version_space():
    task = border_task()
    baseline, baseline_hypotheses, _ = solve_task(
        task, enable_concept_learning=False
    )
    assert not baseline_hypotheses
    assert baseline[0]["attempt_1"] != task["expected"]

    answers, learned_hypotheses, history = solve_task(
        task, enable_concept_learning=True
    )
    assert learned_hypotheses
    assert any(step["version_space_recovered"] for step in history)
    assert task["expected"] in (
        answers[0]["attempt_1"], answers[0]["attempt_2"]
    )
    assert any(h.program.concept is not None for h in learned_hypotheses)


def test_verify_rejects_incorrect_concepts():
    task = border_task()
    proposals = propose_concept_hypotheses(task["train"])
    assert proposals
    survivors, calls = verify_hypotheses(proposals, task["train"])
    assert survivors
    assert calls >= len(proposals)


def test_learning_does_not_trigger_when_v2_solves():
    task = {
        "train": [
            {"input": [[1, 0], [2, 3]], "output": [[0, 3], [1, 2]]},
            {"input": [[4, 5], [0, 6]], "output": [[5, 6], [4, 0]]},
        ],
        "test": [{"input": [[7, 0], [8, 9]]}],
    }
    _, survivors, history = solve_task(
        task, include_objects=False, enable_concept_learning=True
    )
    assert survivors
    assert all(not x["concept_learning_triggered"] for x in history)


def test_induces_two_literal_conjunction():
    # Border alone includes the blue distractor; red alone includes interior.
    # Both facts are required to distinguish the changed object.
    example = {
        "input": [
            [1, 0, 0, 0, 2],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "output": [
            [3, 0, 0, 0, 2],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    }
    evidence = extract_selection_evidence([example], "recolor", 3)
    assert evidence is not None
    concepts = induce_concepts(*evidence)
    assert any(
        c.literals == ("color:1", "touches_border") for c in concepts
    )
    assert not any(c.literals == ("touches_border",) for c in concepts)