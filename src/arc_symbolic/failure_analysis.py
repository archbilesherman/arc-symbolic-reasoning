"""V4 Stage 1: structured counterexamples and concept-learning rejection reasons.

Only training demonstrations are inspected. Nothing in this module reads task
solutions or held-out test outputs.
"""
from collections import Counter
from dataclasses import dataclass
import numpy as np
from .objects import extract_objects


@dataclass(frozen=True)
class Failure:
    kind: str
    error_fraction: float
    incorrect_cells: int
    expected_cells: int


def diagnose_prediction(input_grid, expected_grid, prediction):
    """Return a normalized, actionable failure category for one candidate.

    The score is a ranking heuristic, NOT a replacement for exact consistency.
    """
    before = np.asarray(input_grid, dtype=np.int64)
    expected = np.asarray(expected_grid, dtype=np.int64)
    if prediction is None:
        return Failure('invalid_operator', 2.0, expected.size, expected.size)
    predicted = np.asarray(prediction)
    if predicted.shape != expected.shape:
        return Failure('wrong_dimensions', 2.0, expected.size, expected.size)
    wrong = int(np.count_nonzero(predicted != expected))
    if wrong == 0:
        return Failure('exact', 0.0, 0, expected.size)
    if before.shape != expected.shape:
        return Failure('wrong_values', wrong / expected.size, wrong, expected.size)
    desired_changed = expected != before
    actual_changed = predicted != before
    extra = int(np.count_nonzero(actual_changed & ~desired_changed))
    missing = int(np.count_nonzero(desired_changed & ~actual_changed))
    if extra and missing:
        kind = 'mixed_selection'
    elif extra:
        kind = 'overselection'
    elif missing:
        kind = 'underselection'
    else:
        kind = 'wrong_values'
    return Failure(kind, wrong / expected.size, wrong, expected.size)


def _reason_for_evidence(examples, action, target, connectivity):
    """Describe why V3 cannot infer a whole-object selection for one action."""
    if not examples:
        return 'no_examples'
    positives = negatives = 0
    for example in examples:
        before = np.asarray(example['input'], dtype=np.int64)
        after = np.asarray(example['output'], dtype=np.int64)
        if before.shape != after.shape:
            return 'resized_output'
        if np.any((before == 0) & (after != 0)):
            return 'new_colored_cells'
        objects = extract_objects(before, connectivity)
        if not objects:
            return 'empty_scene'
        selected_this_example = 0
        for obj in objects:
            rows, cols = zip(*obj.cells)
            observed = after[rows, cols]
            original = before[rows, cols]
            unchanged = np.array_equal(observed, original)
            if action == 'recolor':
                if obj.color == target:
                    if not unchanged:
                        return 'partially_changed_object'
                    continue
                if np.all(observed == target):
                    positives += 1
                    selected_this_example += 1
                elif unchanged:
                    negatives += 1
                else:
                    return 'partially_changed_object'
            elif action == 'erase':
                if np.all(observed == 0):
                    positives += 1
                    selected_this_example += 1
                elif unchanged:
                    negatives += 1
                else:
                    return 'partially_changed_object'
            elif action == 'keep':
                if unchanged:
                    positives += 1
                    selected_this_example += 1
                elif np.all(observed == 0):
                    negatives += 1
                else:
                    return 'partially_changed_object'
        if not selected_this_example:
            return 'no_positive_selection'
    if not negatives:
        return 'no_negative_selection'
    return 'usable_evidence'


def evidence_rejection_counts(examples):
    """Count explanatory outcomes across the restricted V3 action families."""
    targets = sorted({int(c) for example in examples
        for c in np.unique(example['output']) if int(c) != 0})
    counts = Counter()
    for connectivity in (4, 8):
        for action in ('recolor', 'erase', 'keep'):
            for target in (targets if action == 'recolor' else (None,)):
                counts[_reason_for_evidence(examples, action, target, connectivity)] += 1
    return dict(sorted(counts.items()))