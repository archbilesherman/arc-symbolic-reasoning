import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm.auto import tqdm

from .operators import ObjectOperator
from .solver import solve_task


CONFIGURATIONS = {
    "pixel": dict(
        include_objects=False,
        include_relations=False,
        include_motion=False,
    ),
    "objects": dict(
        include_objects=True,
        include_relations=False,
        include_motion=False,
    ),
    "relations": dict(
        include_objects=True,
        include_relations=True,
        include_motion=False,
    ),
    "full": dict(
        include_objects=True,
        include_relations=True,
        include_motion=True,
    ),
    "concepts": dict(
        include_objects=True,
        include_relations=True,
        include_motion=True,
        enable_concept_learning=True,
    ),
    "v4_diagnostics": dict(
        include_objects=True, include_relations=True, include_motion=True,
        enable_concept_learning=True, enable_failure_diagnostics=True,
    ),
    "v4_memory": dict(
        include_objects=True, include_relations=True, include_motion=True,
        enable_concept_learning=True, enable_failure_diagnostics=True,
        enable_revision_memory=True,
    ),
    "v4_revision": dict(
        include_objects=True, include_relations=True, include_motion=True,
        enable_concept_learning=True, enable_failure_diagnostics=True,
        enable_revision_memory=True, enable_counterexample_revision=True,
    ),
    "v4_full": dict(
        include_objects=True, include_relations=True, include_motion=True,
        enable_concept_learning=True, enable_failure_diagnostics=True,
        enable_revision_memory=True, enable_counterexample_revision=True,
        expanded_concept_vocabulary=True,
    ),
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def evaluate_arc(
    challenges: dict,
    solutions: dict,
    *,
    configuration: str = "full",
    max_tasks: int | None = None,
) -> dict:
    """Evaluate without accessing held-out outputs during inference."""
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"Unknown configuration: {configuration}")
    if max_tasks is not None and max_tasks < 1:
        raise ValueError("max_tasks must be positive or None")

    task_ids = list(challenges)[:max_tasks]
    if not task_ids:
        raise ValueError("No tasks to evaluate")

    settings = CONFIGURATIONS[configuration]
    records = {}
    total_items = 0
    solved_items = 0
    attempt_1_items = 0
    attempt_2_items = 0
    solved_tasks = 0
    covered_tasks = 0
    fallback_solved_tasks = 0
    evaluated_candidates = 0
    initial_candidates = 0
    total_runtime = 0.0

    # Additional learning diagnostics (zero for pre-existing configurations).
    learning_triggers = 0
    recovered_version_spaces = 0
    proposed_learned_candidates = 0
    evaluated_learned_candidates = 0
    tasks_with_learned_survivors = 0
    tasks_solved_with_learned_survivors = 0
    revision_proposals_total = 0
    revision_evaluations_total = 0
    revision_recoveries_total = 0
    revision_memory_evaluations_total = 0
    tasks_with_revised_survivors = 0
    tasks_solved_with_revised_survivors = 0
    failure_category_totals = {}
    evidence_rejection_totals = {}

    for task_id in tqdm(task_ids, desc=f"ARC {configuration}"):
        if task_id not in solutions:
            raise KeyError(f"Missing solutions for task {task_id}")
        start = perf_counter()
        predictions, survivors, history = solve_task(
            challenges[task_id], **settings
        )
        runtime = perf_counter() - start
        total_runtime += runtime

        truths = solutions[task_id]
        if len(predictions) != len(truths):
            raise ValueError(f"Test-item count mismatch: {task_id}")
        items = []
        for prediction, truth in zip(predictions, truths):
            a1 = np.array_equal(
                np.asarray(prediction["attempt_1"]), np.asarray(truth)
            )
            a2 = np.array_equal(
                np.asarray(prediction["attempt_2"]), np.asarray(truth)
            )
            correct = a1 or a2
            attempt_1_items += int(a1)
            attempt_2_items += int(a2)
            solved_items += int(correct)
            total_items += 1
            items.append({
                "attempt_1_correct": bool(a1),
                "attempt_2_correct": bool(a2),
                "correct": bool(correct),
            })

        task_solved = bool(items) and all(item["correct"] for item in items)
        has_rule = bool(survivors)
        learned_survivors = [
            h for h in survivors
            if isinstance(h.program, ObjectOperator)
            and h.program.concept is not None
        ]
        has_learned_survivor = bool(learned_survivors)
        revised_survivors = [h for h in survivors if getattr(h, 'parent_label', None) is not None]
        task_revision_proposals = sum(x['revision_candidates_proposed'] for x in history)
        task_revision_evaluations = sum(x['revision_candidates_evaluated'] for x in history)
        task_revision_recoveries = sum(int(x['revision_recovered']) for x in history)
        task_memory_evaluations = sum(x['revision_memory_evaluations'] for x in history)
        for stage in history:
            for name, count in stage['failure_diagnostics'].items():
                failure_category_totals[name] = failure_category_totals.get(name, 0) + count
            for name, count in stage['evidence_rejections'].items():
                evidence_rejection_totals[name] = evidence_rejection_totals.get(name, 0) + count
        revision_proposals_total += task_revision_proposals
        revision_evaluations_total += task_revision_evaluations
        revision_recoveries_total += task_revision_recoveries
        revision_memory_evaluations_total += task_memory_evaluations
        tasks_with_revised_survivors += int(bool(revised_survivors))
        tasks_solved_with_revised_survivors += int(task_solved and bool(revised_survivors))
        task_triggers = sum(int(x["concept_learning_triggered"]) for x in history)
        task_recoveries = sum(int(x["version_space_recovered"]) for x in history)
        task_proposals = sum(x["learned_candidates_proposed"] for x in history)
        task_learned_eval = sum(x["learned_candidates_evaluated"] for x in history)

        solved_tasks += int(task_solved)
        covered_tasks += int(has_rule)
        fallback_solved_tasks += int(task_solved and not has_rule)
        tasks_with_learned_survivors += int(has_learned_survivor)
        tasks_solved_with_learned_survivors += int(
            task_solved and has_learned_survivor
        )
        learning_triggers += task_triggers
        recovered_version_spaces += task_recoveries
        proposed_learned_candidates += task_proposals
        evaluated_learned_candidates += task_learned_eval

        count_initial = history[0]["initial_candidates"] if history else 0
        count_evaluated = sum(stage["evaluated"] for stage in history)
        initial_candidates += count_initial
        evaluated_candidates += count_evaluated

        records[task_id] = {
            "solved": task_solved,
            "has_consistent_hypothesis": has_rule,
            "surviving_hypotheses": len(survivors),
            "initial_candidates": count_initial,
            "hypothesis_evaluations": count_evaluated,
            "runtime_seconds": round(runtime, 4),
            "item_results": items,
            "top_hypotheses": [h.label for h in survivors[:5]],
            "version_space_history": history,
            "predictions": predictions,
            "has_learned_survivor": has_learned_survivor,
            "learned_surviving_hypotheses": len(learned_survivors),
            "concept_learning_triggers": task_triggers,
            "version_space_recoveries": task_recoveries,
            "learned_candidates_proposed": task_proposals,
            "learned_candidates_evaluated": task_learned_eval,
            "revision_candidates_proposed": task_revision_proposals,
            "revision_candidates_evaluated": task_revision_evaluations,
            "revision_recoveries": task_revision_recoveries,
            "revision_memory_evaluations": task_memory_evaluations,
            "has_revised_survivor": bool(revised_survivors),
            "revised_survivors": len(revised_survivors),
        }

    tasks_count = len(task_ids)
    return {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "configuration": configuration,
            "solver_options": settings,
            "max_tasks": max_tasks,
        },
        "summary": {
            "tasks": tasks_count,
            "test_items": total_items,
            "tasks_solved": solved_tasks,
            "task_accuracy": solved_tasks / tasks_count,
            "items_solved": solved_items,
            "item_accuracy": solved_items / total_items if total_items else 0.0,
            "attempt_1_correct_items": attempt_1_items,
            "attempt_2_correct_items": attempt_2_items,
            "tasks_with_consistent_hypothesis": covered_tasks,
            "coverage_rate": covered_tasks / tasks_count,
            "tasks_with_empty_version_space": tasks_count - covered_tasks,
            "solved_by_fallback": fallback_solved_tasks,
            "unsolved_despite_consistent_hypothesis": (
                covered_tasks - (solved_tasks - fallback_solved_tasks)
            ),
            "initial_candidates_total": initial_candidates,
            "hypothesis_evaluations_total": evaluated_candidates,
            "average_hypothesis_evaluations_per_task": (
                evaluated_candidates / tasks_count
            ),
            "total_solver_runtime_seconds": round(total_runtime, 3),
            "average_solver_runtime_seconds": round(total_runtime / tasks_count, 4),
            "concept_learning_triggers_total": learning_triggers,
            "version_space_recoveries_total": recovered_version_spaces,
            "learned_candidates_proposed_total": proposed_learned_candidates,
            "learned_candidates_evaluated_total": evaluated_learned_candidates,
            "tasks_with_learned_survivors": tasks_with_learned_survivors,
            "tasks_solved_with_learned_survivors": tasks_solved_with_learned_survivors,
            "revision_candidates_proposed_total": revision_proposals_total,
            "revision_candidates_evaluated_total": revision_evaluations_total,
            "revision_recoveries_total": revision_recoveries_total,
            "revision_memory_evaluations_total": revision_memory_evaluations_total,
            "tasks_with_revised_survivors": tasks_with_revised_survivors,
            "tasks_solved_with_revised_survivors": tasks_solved_with_revised_survivors,
            "failure_category_counts": dict(sorted(failure_category_totals.items())),
            "evidence_rejection_counts": dict(sorted(evidence_rejection_totals.items())),
        },
        "tasks": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate symbolic ARC solver")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("input/ARC-AGI-1")
    )
    parser.add_argument(
        "--split", choices=("training", "evaluation"), default="training"
    )
    parser.add_argument(
        "--configuration", choices=CONFIGURATIONS, default="full"
    )
    parser.add_argument(
        "--max-tasks", type=int, default=25, help="0 means all tasks"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.max_tasks < 0:
        parser.error("--max-tasks must be nonnegative")
    if args.output.exists():
        parser.error(
            f"Output already exists: {args.output}. "
            "Choose another filename to preserve the run."
        )

    challenges = load_json(
        args.data_dir / f"arc-agi_{args.split}_challenges.json"
    )
    solutions = load_json(
        args.data_dir / f"arc-agi_{args.split}_solutions.json"
    )
    report = evaluate_arc(
        challenges,
        solutions,
        configuration=args.configuration,
        max_tasks=args.max_tasks or None,
    )
    report["metadata"].update({
        "split": args.split,
        "data_dir": str(args.data_dir),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()