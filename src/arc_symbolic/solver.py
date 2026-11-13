import numpy as np

from .concept_learner import propose_concept_hypotheses, verify_hypotheses
from .generator import generate_hypotheses
from .operators import ObjectOperator


def compute_version_space(
    task,
    *,
    include_objects=True,
    include_relations=True,
    include_motion=True,
    enable_concept_learning=False,
    max_concept_literals=2,
    max_learned_candidates=128,
):
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
        base_evaluated = len(candidates)

        for candidate in candidates:
            prediction = candidate.apply(example["input"])
            if prediction is not None and np.array_equal(prediction, truth):
                surviving.append(candidate)

        triggered = False
        learned_proposed = 0
        learned_evaluated = 0
        recovered = False

        if not surviving and enable_concept_learning and include_objects:
            triggered = True
            # Reconstruct the vocabulary from *all* examples seen so far.
            observed = task["train"][:index]
            proposals = propose_concept_hypotheses(
                observed,
                max_literals=max_concept_literals,
                max_candidates=max_learned_candidates,
            )
            learned_proposed = len(proposals)
            surviving, learned_evaluated = verify_hypotheses(
                proposals, observed,
            )
            recovered = bool(surviving)

        history.append({
            "examples_seen": index,
            "evaluated": base_evaluated + learned_evaluated,
            "base_evaluated": base_evaluated,
            "total_survivors": len(surviving),
            "pixel_survivors": sum(
                h.representation == "pixel" for h in surviving
            ),
            "object_survivors": sum(
                h.representation == "object" for h in surviving
            ),
            "initial_candidates": initial_count,
            "concept_learning_triggered": triggered,
            "learned_candidates_proposed": learned_proposed,
            "learned_candidates_evaluated": learned_evaluated,
            "concept_survivors": sum(
                isinstance(h.program, ObjectOperator)
                and h.program.concept is not None for h in surviving
            ),
            "version_space_recovered": recovered,
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
            and not any(np.array_equal(guess, p) for p in predictions)
        ):
            predictions.append(guess)
            if len(predictions) == 2:
                break
    if not predictions:
        predictions = [np.asarray(test_input, dtype=np.int64).copy()]
    if len(predictions) == 1:
        predictions.append(predictions[0].copy())
    return {
        "attempt_1": predictions[0].tolist(),
        "attempt_2": predictions[1].tolist(),
    }


def solve_task(
    task,
    *,
    include_objects=True,
    include_relations=True,
    include_motion=True,
    enable_concept_learning=False,
    max_concept_literals=2,
    max_learned_candidates=128,
):
    version_space, history = compute_version_space(
        task,
        include_objects=include_objects,
        include_relations=include_relations,
        include_motion=include_motion,
        enable_concept_learning=enable_concept_learning,
        max_concept_literals=max_concept_literals,
        max_learned_candidates=max_learned_candidates,
    )
    answers = [
        predict_two(item["input"], version_space)
        for item in task["test"]
    ]
    return answers, version_space, history