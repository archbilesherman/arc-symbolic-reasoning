"""End-to-end checks for revision-enabled solver configurations."""

from src.arc_symbolic.evaluate import CONFIGURATIONS, evaluate_arc


def test_revision_ablation_configurations_are_available():
    expected = (
        "concepts",
        "v4_diagnostics",
        "v4_memory",
        "v4_revision",
        "v4_full",
    )
    assert all(name in CONFIGURATIONS for name in expected)

    assert CONFIGURATIONS["v4_diagnostics"]["enable_failure_diagnostics"]
    assert CONFIGURATIONS["v4_memory"]["enable_revision_memory"]
    assert not CONFIGURATIONS["v4_memory"].get(
        "enable_counterexample_revision", False
    )
    assert CONFIGURATIONS["v4_revision"]["enable_counterexample_revision"]
    assert CONFIGURATIONS["v4_full"]["expanded_concept_vocabulary"]


def test_revision_configurations_preserve_existing_pixel_solution():
    # The added learning machinery must not break an already-solvable task.
    task = {
        "train": [
            {"input": [[1, 0], [2, 3]], "output": [[0, 3], [1, 2]]},
            {"input": [[4, 5], [0, 6]], "output": [[5, 6], [4, 0]]},
        ],
        "test": [{"input": [[7, 0], [8, 9]]}],
    }
    challenges = {"rotation": task}
    solutions = {"rotation": [[[0, 9], [7, 8]]]}

    for name in (
        "concepts",
        "v4_diagnostics",
        "v4_memory",
        "v4_revision",
        "v4_full",
    ):
        report = evaluate_arc(challenges, solutions, configuration=name)
        assert report["summary"]["tasks_solved"] == 1, name
        assert report["summary"]["tasks_with_consistent_hypothesis"] == 1, name