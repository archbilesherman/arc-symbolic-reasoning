from src.arc_symbolic.evaluate import CONFIGURATIONS, evaluate_arc
from tests.test_concept_learner import border_task


def test_concept_configuration_reports_learning():
    task = border_task()
    challenges = {
        "synthetic": {k: v for k, v in task.items() if k != "expected"}
    }
    solutions = {"synthetic": [task["expected"]]}

    baseline = evaluate_arc(challenges, solutions, configuration="full")
    learned = evaluate_arc(challenges, solutions, configuration="concepts")

    assert not CONFIGURATIONS["full"].get("enable_concept_learning", False)
    assert CONFIGURATIONS["concepts"]["enable_concept_learning"]
    assert baseline["summary"]["tasks_with_empty_version_space"] == 1
    assert learned["summary"]["tasks_solved"] == 1
    assert learned["summary"]["tasks_with_learned_survivors"] == 1
    assert learned["summary"]["version_space_recoveries_total"] >= 1
    assert learned["summary"]["learned_candidates_evaluated_total"] > 0