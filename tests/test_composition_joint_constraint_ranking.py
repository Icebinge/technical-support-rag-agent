from __future__ import annotations

from ts_rag_agent.application import composition_joint_constraint_ranking as ranking
from ts_rag_agent.application.composition_action_audit import (
    ActionAuditRow,
    CompositionAction,
)
from ts_rag_agent.application.composition_joint_constraint_ranking import (
    JointConstraintPolicySpec,
    JointConstraintPrediction,
    evaluate_selected_actions,
    select_actions,
    stage186_policy_specs,
)


def test_stage186_policy_grid_is_complete_and_unique() -> None:
    specs = stage186_policy_specs()

    assert len(specs) == 72
    assert len({spec.name for spec in specs}) == 72
    assert {spec.feature_representation for spec in specs} == {
        "raw_runtime",
        "question_relative_runtime",
    }
    assert {spec.estimator_family for spec in specs} == {
        "class_balanced_logistic",
        "histogram_gradient_boosting",
    }
    assert {spec.ranking_rule for spec in specs} == {
        "max_safety_risk_lexicographic",
        "citation_first_lexicographic",
        "pareto_constraint_dominance",
    }


def test_lexicographic_selection_prefers_joint_safety_then_gain() -> None:
    risky = _row("q1", "fold_1", "risky", citation_delta=1, f1_delta=-0.1)
    safe = _row("q1", "fold_1", "safe", citation_delta=0, f1_delta=0.1)
    predictions = (
        JointConstraintPrediction(risky, 0.05, 0.90, 0.80),
        JointConstraintPrediction(safe, 0.10, 0.10, 0.70),
    )
    spec = _spec("max_safety_risk_lexicographic")

    selected = select_actions(predictions, spec)

    assert selected == (safe,)


def test_pareto_selection_uses_canonical_order_for_exact_ties() -> None:
    first = _row("q1", "fold_1", "a_action", citation_delta=0, f1_delta=0.0)
    second = _row("q1", "fold_1", "z_reference", citation_delta=0, f1_delta=0.0)
    predictions = (
        JointConstraintPrediction(second, 0.1, 0.1, 0.5),
        JointConstraintPrediction(first, 0.1, 0.1, 0.5),
    )
    spec = _spec("pareto_constraint_dominance")

    selected = select_actions(predictions, spec)

    assert selected == (first,)


def test_evaluation_reports_repairs_and_new_regressions() -> None:
    reference_regressed = _row(
        "q1",
        "fold_1",
        "reference_q1",
        citation_delta=0,
        f1_delta=-0.1,
    )
    reference_safe = _row(
        "q2",
        "fold_2",
        "reference_q2",
        citation_delta=0,
        f1_delta=0.0,
    )
    repaired = _row(
        "q1",
        "fold_1",
        "repaired",
        citation_delta=0,
        f1_delta=0.2,
        strict_expected=True,
    )
    new_regression = _row(
        "q2",
        "fold_2",
        "new_regression",
        citation_delta=-1,
        f1_delta=-0.2,
    )

    report = evaluate_selected_actions(
        selected_rows=(repaired, new_regression),
        references={"q1": reference_regressed, "q2": reference_safe},
        expected_fold_ids=("fold_1", "fold_2", "fold_3"),
    )

    assert report["changed_question_count"] == 2
    assert report["strict_success_precision"] == 0.5
    assert report["repaired_reference_regression_count"] == 1
    assert report["stage182_regression_repair_rate"] == 1.0
    assert report["new_f1_regression_count"] == 1
    assert report["new_f1_regression_rate"] == 1.0
    assert report["citation_nonregressing_fold_count"] == 1
    assert report["f1_nonregressing_fold_count"] == 1


def test_shared_feature_encoder_fits_all_four_three_head_bundles() -> None:
    rows = (
        _row("q1", "fold_1", "a", citation_delta=-1, f1_delta=-0.1),
        _row("q1", "fold_1", "b", citation_delta=0, f1_delta=0.1, strict_expected=True),
        _row("q2", "fold_2", "a", citation_delta=0, f1_delta=-0.2),
        _row("q2", "fold_2", "b", citation_delta=1, f1_delta=0.2, strict_expected=True),
    )
    indices = ranking.build_composition_feature_indices(rows)
    feature_indices = {
        "raw_runtime": indices["raw"],
        "question_relative_runtime": indices["question_relative"],
    }

    bundles = ranking._fit_all_bundles(rows, feature_indices)

    assert len(bundles) == 4
    for bundle in bundles.values():
        predictions = bundle.predict(rows, feature_indices[bundle.feature_representation])
        assert len(predictions) == len(rows)
        assert all(0.0 <= row.citation_loss_probability <= 1.0 for row in predictions)
        assert all(0.0 <= row.f1_loss_probability <= 1.0 for row in predictions)
        assert all(0.0 <= row.strict_gain_probability <= 1.0 for row in predictions)


def _spec(ranking_rule: str) -> JointConstraintPolicySpec:
    return JointConstraintPolicySpec(
        name=f"test_{ranking_rule}",
        feature_representation="raw_runtime",
        estimator_family="class_balanced_logistic",
        ranking_rule=ranking_rule,
        safety_dominance_margin=0.0,
        strict_gain_margin=0.0,
    )


def _row(
    question_key: str,
    fold_id: str,
    action_id: str,
    *,
    citation_delta: int,
    f1_delta: float,
    strict_expected: bool = False,
) -> ActionAuditRow:
    return ActionAuditRow(
        question_key=question_key,
        fold_id=fold_id,
        route="other",
        action=CompositionAction(
            action_id=action_id,
            family="test",
            aliases=(),
            selected_indices=(0,),
            matches_stage180=False,
        ),
        runtime_features={"score": 1.0},
        outcome_class="test",
        strict_expected=strict_expected,
        citation_delta=citation_delta,
        f1_delta=f1_delta,
    )
