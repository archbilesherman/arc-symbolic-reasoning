"""ARC version spaces with optional, bounded V4 counterexample revision.

All final survivors must pass every observed example exactly. Failure scores
are used only to choose candidates for repair, never to justify predictions.
"""
from collections import Counter
import numpy as np

from .concept_learner import propose_concept_hypotheses, verify_hypotheses
from .failure_analysis import diagnose_prediction, evidence_rejection_counts
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
    enable_failure_diagnostics=False,
    enable_revision_memory=False,
    enable_counterexample_revision=False,
    expanded_concept_vocabulary=False,
    max_revision_memory=24,
    max_revision_candidates=96,
):
    candidates = generate_hypotheses(
        task,
        include_objects=include_objects,
        include_relations=include_relations,
        include_motion=include_motion,
    )
    initial_count = len(candidates)
    history = []
    memory = []
    memory_enabled = include_objects and (enable_revision_memory or enable_counterexample_revision)
    diagnostics_enabled = enable_failure_diagnostics or enable_counterexample_revision
    if memory_enabled:
        from .revision_memory import MemoryEntry, remember_failed, retain_best

    for index, example in enumerate(task['train'], start=1):
        truth = np.asarray(example['output'], dtype=np.int64)
        surviving = []
        diagnostic_counts = Counter()
        seeds = []
        base_evaluated = len(candidates)
        memory_evaluated = 0

        # Persistent incompatible seeds must be checked against each new
        # demonstration without entering the exact version space.
        if memory_enabled:
            for entry in memory:
                prediction = entry.hypothesis.apply(example['input'])
                memory_evaluated += 1
                seeds.append(remember_failed(
                    entry.hypothesis, example, prediction, previous=entry
                ))

        for candidate in candidates:
            prediction = candidate.apply(example['input'])
            if prediction is not None and np.array_equal(prediction, truth):
                surviving.append(candidate)
                continue
            if diagnostics_enabled:
                result = diagnose_prediction(
                    example['input'], truth, prediction
                )
                diagnostic_counts[result.kind] += 1
            if memory_enabled:
                earlier = MemoryEntry(
                    candidate, 0.0, 0, index - 1, 'exact'
                )
                seeds.append(remember_failed(
                    candidate, example, prediction, previous=earlier
                ))

        if memory_enabled:
            memory = retain_best(seeds, max_items=max_revision_memory)

        triggered = False
        learned_proposed = 0
        learned_evaluated = 0
        revision_proposed = 0
        revision_evaluated = 0
        revision_recovered = False
        recovery = False
        revision_diagnostics = {}
        evidence_rejections = {}

        if not surviving and enable_concept_learning and include_objects:
            triggered = True
            observed = task['train'][:index]
            if enable_failure_diagnostics:
                evidence_rejections = evidence_rejection_counts(observed)

            # Revise failed parents FIRST; the V3 independent learner remains
            # a fallback so V4 can still solve its earlier concept tasks.
            if enable_counterexample_revision and memory:
                from .hypothesis_revision import propose_revisions
                proposed, revision_diagnostics = propose_revisions(
                    memory, observed,
                    max_literals=max_concept_literals,
                    max_candidates=max_revision_candidates,
                    expanded_vocabulary=expanded_concept_vocabulary,
                )
                revision_proposed = len(proposed)
                surviving, revision_evaluated = verify_hypotheses(
                    proposed, observed
                )
                revision_recovered = bool(surviving)

            if not surviving:
                kwargs = dict(
                    max_literals=max_concept_literals,
                    max_candidates=max_learned_candidates,
                )
                if expanded_concept_vocabulary:
                    kwargs['expanded_vocabulary'] = True
                proposals = propose_concept_hypotheses(observed, **kwargs)
                learned_proposed = len(proposals)
                surviving, learned_evaluated = verify_hypotheses(
                    proposals, observed
                )
            recovery = bool(surviving)

        history.append({
            'examples_seen': index,
            'evaluated': (base_evaluated + memory_evaluated
                          + revision_evaluated + learned_evaluated),
            'base_evaluated': base_evaluated,
            'total_survivors': len(surviving),
            'pixel_survivors': sum(h.representation == 'pixel' for h in surviving),
            'object_survivors': sum(h.representation == 'object' for h in surviving),
            'initial_candidates': initial_count,
            'concept_learning_triggered': triggered,
            'learned_candidates_proposed': learned_proposed,
            'learned_candidates_evaluated': learned_evaluated,
            'concept_survivors': sum(
                isinstance(h.program, ObjectOperator)
                and h.program.concept is not None for h in surviving
            ),
            'version_space_recovered': recovery,
            'failure_diagnostics': dict(sorted(diagnostic_counts.items())),
            'evidence_rejections': evidence_rejections,
            'revision_memory_size': len(memory),
            'revision_memory_evaluations': memory_evaluated,
            'revision_candidates_proposed': revision_proposed,
            'revision_candidates_evaluated': revision_evaluated,
            'revision_recovered': revision_recovered,
            'revision_diagnostics': revision_diagnostics,
            'revised_survivors': sum(
                getattr(h, 'parent_label', None) is not None for h in surviving
            ),
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
        'attempt_1': predictions[0].tolist(),
        'attempt_2': predictions[1].tolist(),
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
    enable_failure_diagnostics=False,
    enable_revision_memory=False,
    enable_counterexample_revision=False,
    expanded_concept_vocabulary=False,
    max_revision_memory=24,
    max_revision_candidates=96,
):
    survivors, history = compute_version_space(
        task,
        include_objects=include_objects,
        include_relations=include_relations,
        include_motion=include_motion,
        enable_concept_learning=enable_concept_learning,
        max_concept_literals=max_concept_literals,
        max_learned_candidates=max_learned_candidates,
        enable_failure_diagnostics=enable_failure_diagnostics,
        enable_revision_memory=enable_revision_memory,
        enable_counterexample_revision=enable_counterexample_revision,
        expanded_concept_vocabulary=expanded_concept_vocabulary,
        max_revision_memory=max_revision_memory,
        max_revision_candidates=max_revision_candidates,
    )
    answers = [predict_two(item['input'], survivors) for item in task['test']]
    return answers, survivors, history