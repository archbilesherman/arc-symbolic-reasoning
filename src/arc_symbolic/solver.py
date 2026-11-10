import numpy as np

from .generator import generate_hypotheses


def compute_version_space(task,*,include_objects=True,include_relations=True,include_motion=True):
    candidates = generate_hypotheses(
        task,
        include_objects=include_objects,
        include_relations=include_relations,
        include_motion=include_motion,
    )

    initial_count = len(candidates)
    history = []

    for index, example in enumerate(task["train"], start=1):
        truth = np.asarray(example["output"], dtype=np.int64)

        surviving = []

        for candidate in candidates:
            prediction = candidate.apply(example["input"])

            if (
                prediction is not None
                and np.array_equal(prediction, truth)
            ):
                surviving.append(candidate)

        history.append({
            "examples_seen": index,
            "evaluated": len(candidates),
            "total_survivors": len(surviving),
            "pixel_survivors": sum(
                h.representation == "pixel"
                for h in surviving
            ),
            "object_survivors": sum(
                h.representation == "object"
                for h in surviving
            ),
            "initial_candidates": initial_count,
        })

        candidates = surviving

    return candidates, history


def valid_arc_prediction(grid):
    if grid is None:
        return False

    a = np.asarray(grid)

    return (
        a.ndim == 2
        and 1 <= a.shape[0] <= 30
        and 1 <= a.shape[1] <= 30
        and np.issubdtype(a.dtype, np.integer)
        and np.all((a >= 0) & (a <= 9))
    )


def predict_two(test_input, version_space):
    predictions = []

    for candidate in version_space:
        guess = candidate.apply(test_input)

        if (
            valid_arc_prediction(guess)
            and not any(
                np.array_equal(guess, previous)
                for previous in predictions
            )
        ):
            predictions.append(guess)

            if len(predictions) == 2:
                break

    if not predictions:
        predictions = [np.asarray(test_input, dtype=np.int64).copy()]

    if len(predictions) == 1:
        predictions.append(predictions[0].copy())

    return {"attempt_1": predictions[0].tolist(), "attempt_2": predictions[1].tolist()}


def solve_task(task, *, include_objects=True, include_relations=True, include_motion=True):

    version_space, history = compute_version_space(
        task,
        include_objects=include_objects,
        include_relations=include_relations,
        include_motion=include_motion,
    )

    answers = [predict_two(item["input"], version_space)for item in task["test"]]

    return answers, version_space, history