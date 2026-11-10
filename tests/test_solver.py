from src.arc_symbolic.solver import solve_task


def solved(task, **kwargs):
    answers, survivors, history = solve_task(
        task, **kwargs
    )

    correct = any(
        answers[0][attempt] == task["expected"]
        for attempt in ("attempt_1", "attempt_2")
    )

    return correct, survivors, history


def test_pixel_rotation():
    task = {
        "train": [
            {
                "input": [[1, 0], [2, 3]],
                "output": [[0, 3], [1, 2]],
            },
            {
                "input": [[4, 5], [0, 6]],
                "output": [[5, 6], [4, 0]],
            },
        ],
        "test": [
            {"input": [[7, 0], [8, 9]]}
        ],
        "expected": [[0, 9], [7, 8]],
    }

    correct, survivors, history = solved(
        task, include_objects=False
    )

    assert correct
    assert history[-1]["total_survivors"] > 0


def test_object_largest_recolor():
    task = {
        "train": [
            {
                "input": [[1, 1, 0, 2], [1, 0, 0, 2]],
                "output": [[3, 3, 0, 2], [3, 0, 0, 2]],
            },
            {
                "input": [[2, 2, 0, 1], [2, 0, 0, 1]],
                "output": [[3, 3, 0, 1], [3, 0, 0, 1]],
            },
        ],
        "test": [
            {"input": [[4, 4, 0, 5], [4, 0, 0, 5]]}
        ],
        "expected": [[3, 3, 0, 5], [3, 0, 0, 5]],
    }

    correct, survivors, _ = solved(
        task,
        include_relations=False,
        include_motion=False,
    )

    assert correct
    assert any(
        h.representation == "object"
        for h in survivors
    )


def test_relational_recolor():
    task = {
        "train": [
            {
                "input": [[2, 0, 1, 0, 3]],
                "output": [[2, 0, 4, 0, 4]],
            },
            {
                "input": [[1, 0, 2, 0, 3]],
                "output": [[1, 0, 2, 0, 4]],
            },
        ],
        "test": [
            {"input": [[2, 0, 3, 0, 1]]}
        ],
        "expected": [[2, 0, 4, 0, 4]],
    }

    correct, survivors, _ = solved(
        task,
        include_relations=True,
        include_motion=False,
    )

    assert correct
    assert any(
        h.program.relations
        for h in survivors
        if h.representation == "object"
    )

    baseline_correct, _, _ = solved(
        task,
        include_relations=False,
        include_motion=False,
    )

    assert not baseline_correct


def test_motion_from_demonstrations():
    task = {
        "train": [
            {
                "input": [[1, 0, 0, 0]],
                "output": [[0, 1, 0, 0]],
            },
            {
                "input": [[0, 1, 0, 0]],
                "output": [[0, 0, 1, 0]],
            },
        ],
        "test": [
            {"input": [[0, 0, 1, 0]]}
        ],
        "expected": [[0, 0, 0, 1]],
    }

    correct, survivors, _ = solved(
        task,
        include_relations=False,
        include_motion=True,
    )

    assert correct
    assert any(
        h.program.action == "translate"
        for h in survivors
        if h.representation == "object"
    )