"""V4 Stage 3: repair promising failed hypotheses with counterexamples.

Refinement modifies an existing object's selector, keeping its action and
parameters. Each proposal is verified by the ordinary exact replay stage.
"""
from dataclasses import replace
from collections import Counter
import numpy as np
from .concept_learner import extract_selection_evidence, induce_concepts
from .hypotheses import make_hypothesis
from .objects import extract_objects
from .selectors import select_objects


def _selection_errors(parent, examples, positive, negative):
    """Classify a parent's selection mistakes relative to supervised objects."""
    program = parent.program
    false_positive = false_negative = 0
    # Object observations are stored across examples in the same order as
    # extract_selection_evidence iterates objects. Neutral recolor objects are
    # excluded, so recompute positive/negative object positions per example.
    for example in examples:
        arr = np.asarray(example['input'], dtype=np.int64)
        truth = np.asarray(example['output'], dtype=np.int64)
        objects = extract_objects(arr, program.connectivity)
        selected = select_objects(
            objects, attributes=program.attributes, relations=program.relations,
            rank=program.rank, connectivity=program.connectivity,
            concept=program.concept, grid_shape=arr.shape,
        )
        selected_ids = {o.id for o in selected}
        for obj in objects:
            rows, cols = zip(*obj.cells)
            original = arr[rows, cols]
            observed = truth[rows, cols]
            if program.action == 'recolor':
                if obj.color == program.target_color:
                    continue
                desired = bool(np.all(observed == program.target_color))
            elif program.action == 'erase':
                desired = bool(np.all(observed == 0))
            else:  # keep
                desired = bool(np.array_equal(observed, original))
            if obj.id in selected_ids and not desired:
                false_positive += 1
            elif obj.id not in selected_ids and desired:
                false_negative += 1
    return false_positive, false_negative


def propose_revisions(
    memories,
    examples,
    *,
    max_literals=2,
    max_candidates=128,
    expanded_vocabulary=False,
):
    """Induce child concepts from failed parents; preserve parent traceability.

    Return proposals and diagnostic counters. Only action families supported by
    V3 object evidence inference can be refined in this version.
    """
    if max_candidates < 1:
        raise ValueError('max_candidates must be >= 1')
    evidence_cache = {}
    seen = set()
    proposed = []
    failure_types = Counter()
    attempted_parents = 0

    for entry in memories:
        parent = entry.hypothesis
        op = parent.program

        if op.action not in {"recolor", "erase", "keep"}:
            failure_types["unsupported_action"] += 1
            continue

        if op.action == "recolor" and (
            op.target_color is None
            or not 1 <= op.target_color <= 9
        ):
            failure_types["invalid_recolor_target"] += 1
            continue

        key = (op.action, op.target_color, op.connectivity)
        
        if key not in evidence_cache:
            kwargs = {'expanded_vocabulary': True} if expanded_vocabulary else {}
            evidence_cache[key] = extract_selection_evidence(
                examples, op.action, op.target_color, op.connectivity,
                **kwargs,
            )
        evidence = evidence_cache[key]
        if evidence is None:
            failure_types['unusable_selection_evidence'] += 1
            continue
        positives, negatives = evidence
        false_positive, false_negative = _selection_errors(
            parent, examples, positives, negatives
        )
        if false_positive == 0 and false_negative == 0:
            failure_types['no_selection_error'] += 1
            continue
        attempted_parents += 1
        if false_positive and false_negative:
            failure_types['mixed_selection'] += 1
        elif false_positive:
            failure_types['overselection'] += 1
        else:
            failure_types['underselection'] += 1
        concepts = induce_concepts(positives, negatives, max_literals)
        for concept in concepts:
            # Specialize only if the old selector was not missing any targets.
            if false_negative == 0:
                choices = [dict(
                    attributes=op.attributes, relations=op.relations,
                    rank=op.rank,
                )]
            else:
                # Generalize a restrictive original rule, then constrain with
                # a discriminating learned concept.
                choices = [dict(attributes=None, relations=(), rank=None)]
            for original in choices:
                new_op = replace(op, concept=concept, **original)
                if new_op == op or new_op in seen:
                    continue
                seen.add(new_op)
                child = make_hypothesis(new_op)
                child = replace(child, parent_label=parent.label)
                proposed.append((entry.mean_error, child))
                if len(proposed) >= max_candidates * 4:
                    break
            if len(proposed) >= max_candidates * 4:
                break
        if len(proposed) >= max_candidates * 4:
            break
    proposed.sort(key=lambda p: (p[0], p[1].complexity, p[1].label))
    return [p[1] for p in proposed[:max_candidates]], {
        'parents_examined': len(memories),
        'parents_revisable': attempted_parents,
        'revision_failure_types': dict(sorted(failure_types.items())),
    }