"""V4 Stage 2: a bounded persistent memory of failed object hypotheses.

The version space remains exact; inconsistent programs are never returned as
solutions merely because they are in revision memory.
"""
from dataclasses import dataclass
from heapq import nsmallest
from .hypotheses import CandidateHypothesis
from .operators import ObjectOperator
from .failure_analysis import diagnose_prediction

SUPPORTED = frozenset({'recolor', 'erase', 'keep'})


@dataclass(frozen=True)
class MemoryEntry:
    hypothesis: CandidateHypothesis
    cumulative_error: float
    failures: int
    examples_seen: int
    last_failure: str

    @property
    def mean_error(self):
        return self.cumulative_error / max(1, self.examples_seen)


def remember_failed(hypothesis, example, prediction, previous=None):
    """Update a candidate's failure history using one new demonstration.

    Caller must ensure a hypothesis is never added as an exact survivor.
    """
    if not isinstance(hypothesis.program, ObjectOperator):
        return None
    if hypothesis.program.action not in SUPPORTED:
        return None
    failure = diagnose_prediction(example['input'], example['output'], prediction)
    if failure.kind == 'exact':
        # A previously failed program remains inconsistent, but can be ranked.
        pass
    old_error = previous.cumulative_error if previous else 0.0
    old_failures = previous.failures if previous else 0
    old_count = previous.examples_seen if previous else 0
    return MemoryEntry(
        hypothesis=hypothesis,
        cumulative_error=old_error + failure.error_fraction,
        failures=old_failures + int(failure.kind != 'exact'),
        examples_seen=old_count + 1,
        last_failure=failure.kind,
    )


def retain_best(entries, max_items=24):
    """Deduplicate and retain lowest-error revision seeds deterministically."""
    if max_items < 1:
        raise ValueError('max_items must be >= 1')
    by_program = {}
    for entry in entries:
        if entry is None:
            continue
        key = entry.hypothesis.program
        previous = by_program.get(key)
        if previous is None or _key(entry) < _key(previous):
            by_program[key] = entry
    return nsmallest(max_items, by_program.values(), key=_key)


def _key(entry):
    return (
        entry.mean_error,
        entry.failures,
        entry.hypothesis.complexity,
        entry.hypothesis.label,
    )