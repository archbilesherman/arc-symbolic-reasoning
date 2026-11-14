"""Induce generalizable object selectors after version-space failure.

This learner infers object-selection concepts for recolor, erase, and keep.
All proposals are rechecked with the normal executable operators.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .concepts import LearnedConcept, object_facts, EXPANDED_FEATURES
from .hypotheses import CandidateHypothesis, make_hypothesis
from .objects import ARCObject, extract_objects
from .operators import ObjectOperator


@dataclass(frozen=True)
class ObjectObservation:
    """An observed object and its scene-derived, reusable facts."""

    obj: ARCObject
    facts: frozenset[str]


def extract_selection_evidence(
    examples: list[dict],
    action: str,
    target_color: int | None = None,
    connectivity: int = 4,
    *,
    expanded_vocabulary: bool = False,
) -> tuple[list[ObjectObservation], list[ObjectObservation]] | None:
    """Infer selected/unselected objects for a *single* proposed action.

    Return None when any training pair cannot be explained by whole-object
    application of that action.  Neutral target-colored objects are excluded
    from recolor supervision because selection cannot be observed for them.
    """
    if action not in {"recolor", "erase", "keep"}:
        raise ValueError(f"Unsupported concept-learning action: {action}")
    if action == "recolor" and (
        target_color is None or not 1 <= target_color <= 9
    ):
        raise ValueError("recolor requires a target color from 1 through 9")
    if not examples:
        return None

    positives: list[ObjectObservation] = []
    negatives: list[ObjectObservation] = []

    for example in examples:
        before = np.asarray(example["input"], dtype=np.int64)
        after = np.asarray(example["output"], dtype=np.int64)
        if before.shape != after.shape:
            return None

        # None of these actions can create new colored cells from background.
        if np.any((before == 0) & (after != 0)):
            return None

        objects = extract_objects(before, connectivity)
        if not objects:
            return None
        scene = tuple(objects)
        positive_count = 0

        for obj in objects:
            cells = tuple(zip(*obj.cells))
            original = before[cells]
            observed = after[cells]

            if action == "recolor":
                if obj.color == target_color:
                    # Already target-colored: whether it was selected is unknowable.
                    if not np.array_equal(original, observed):
                        return None
                    continue
                if np.all(observed == target_color):
                    selected = True
                elif np.array_equal(original, observed):
                    selected = False
                else:
                    return None
            elif action == "erase":
                if np.all(observed == 0):
                    selected = True
                elif np.array_equal(original, observed):
                    selected = False
                else:
                    return None
            else:  # keep: selected objects remain, others disappear.
                if np.array_equal(original, observed):
                    selected = True
                elif np.all(observed == 0):
                    selected = False
                else:
                    return None

            facts = object_facts(
                obj, scene, before.shape, connectivity,
                expanded_vocabulary=expanded_vocabulary,
            )
            item = ObjectObservation(obj=obj, facts=facts)
            if selected:
                positives.append(item)
                positive_count += 1
            else:
                negatives.append(item)

        # ObjectOperator.apply currently rejects empty selections.
        if not positive_count:
            return None

    if not positives or not negatives:
        return None
    return positives, negatives


def _literal_priority(literal: str) -> tuple[int, str]:
    """Favor structural generalization over example-specific colors."""
    base = literal.removeprefix("not:")
    if base in {"touches_border", "solid_rectangle", "square_bbox", *EXPANDED_FEATURES}:
        return (0, literal)
    if base in {"largest", "smallest"}:
        return (1, literal)
    if base.startswith("relation:"):
        return (2, literal)
    return (3, literal)


def induce_concepts(
    positive_observations: list[ObjectObservation],
    negative_observations: list[ObjectObservation],
    max_literals: int = 2,
) -> list[LearnedConcept]:
    """Find minimal conjunctions covering all positives and no negatives."""
    if max_literals < 1:
        raise ValueError("max_literals must be >= 1")
    if not positive_observations or not negative_observations:
        return []

    shared = set.intersection(*(set(item.facts) for item in positive_observations))
    # A literal that holds for every negative observation cannot distinguish.
    shared = {
        literal for literal in shared
        if any(literal not in neg.facts for neg in negative_observations)
    }
    literal_pool = sorted(shared, key=_literal_priority)
    concepts = []

    for size in range(1, min(max_literals, len(literal_pool)) + 1):
        for combo in combinations(literal_pool, size):
            # These one-literal concepts are already available in V2.
            if size == 1 and (
                combo[0].startswith("color:")
                or combo[0] in {"largest", "smallest"}
            ):
                continue
            if all(
                not all(literal in neg.facts for literal in combo)
                for neg in negative_observations
            ):
                concepts.append(LearnedConcept(combo))

    # Favor shorter hypotheses, and non-color structural distinctions.
    concepts.sort(key=lambda c: (
        len(c.literals),
        tuple(sorted(_literal_priority(x) for x in c.literals)),
        c.literals,
    ))
    return concepts


def propose_concept_hypotheses(
    examples: list[dict],
    max_literals: int = 2,
    max_candidates: int = 128,
    *,
    expanded_vocabulary: bool = False,
) -> list[CandidateHypothesis]:
    """Generate evidence-guided new single-step hypotheses."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    if not examples:
        return []

    output_colors = sorted({
        int(color)
        for ex in examples
        for color in np.unique(ex["output"])
        if int(color) != 0
    })
    seen = set()
    candidates: list[CandidateHypothesis] = []

    for connectivity in (4, 8):
        for action in ("recolor", "erase", "keep"):
            targets = output_colors if action == "recolor" else [None]
            for target_color in targets:
                evidence = extract_selection_evidence(
                    examples, action, target_color, connectivity,
                    expanded_vocabulary=expanded_vocabulary,
                )
                if evidence is None:
                    continue
                positives, negatives = evidence
                for concept in induce_concepts(
                    positives, negatives, max_literals
                ):
                    program = ObjectOperator(
                        action=action,
                        connectivity=connectivity,
                        target_color=target_color,
                        concept=concept,
                    )
                    if program not in seen:
                        seen.add(program)
                        candidates.append(make_hypothesis(program))

    candidates.sort(key=lambda h: (h.complexity, h.label))
    return candidates[:max_candidates]


def verify_hypotheses(
    hypotheses: list[CandidateHypothesis],
    examples: list[dict],
) -> tuple[list[CandidateHypothesis], int]:
    """Exact replay against *all* observed training pairs, with work count."""
    survivors = []
    evaluations = 0
    for candidate in hypotheses:
        for example in examples:
            evaluations += 1
            prediction = candidate.apply(example["input"])
            if prediction is None or not np.array_equal(
                prediction, np.asarray(example["output"], dtype=np.int64)
            ):
                break
        else:
            survivors.append(candidate)
    return survivors, evaluations