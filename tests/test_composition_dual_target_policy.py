from __future__ import annotations

import pytest

from ts_rag_agent.application.composition_action_audit import (
    ActionAuditRow,
    CompositionAction,
)
from ts_rag_agent.application.composition_dual_target_policy import (
    DualTargetPolicySpec,
    DualTargetPrediction,
    ScoreMode,
    evaluate_dual_target_policy,
    run_nested_dual_target_selection,
    select_dual_target_actions,
    stage182_policy_specs,
)


def test_stage182_policy_grid_is_complete_and_unique() -> None:
    specs = stage182_policy_specs()

    assert len(specs) == 32
    assert len({spec.name for spec in specs}) == 32
    assert {spec.model_family for spec in specs} == {
        "logistic",
        "hist_gradient_boosting",
    }
    assert {spec.target_coverage for spec in specs} == {0.10, 0.25, 0.50, 1.00}


@pytest.mark.parametrize("coverage", [0.0, -0.1, 1.01])
def test_policy_spec_rejects_invalid_coverage(coverage: float) -> None:
    with pytest.raises(ValueError, match="target coverage"):
        DualTargetPolicySpec(
            name="invalid",
            model_family="logistic",
            score_mode="safe_product",
            target_coverage=coverage,
        )


def test_risk_aware_selection_prefers_safe_action() -> None:
    risky = _row(
        question="q1",
        fold="fold_1",
        action_id="risky",
        feature=0.0,
        citation_delta=1,
        f1_delta=-0.2,
    )
    safe = _row(
        question="q1",
        fold="fold_1",
        action_id="safe",
        feature=1.0,
        citation_delta=1,
        f1_delta=0.1,
    )
    predictions = (
        DualTargetPrediction(risky, 0.90, 0.80),
        DualTargetPrediction(safe, 0.75, 0.05),
    )
    citation_only = _spec("citation_only")
    risk_aware = _spec("citation_minus_risk")

    assert (
        select_dual_target_actions(
            predictions,
            spec=citation_only,
            utility_threshold=None,
        )[0].row.action.action_id
        == "risky"
    )
    assert (
        select_dual_target_actions(
            predictions,
            spec=risk_aware,
            utility_threshold=None,
        )[0].row.action.action_id
        == "safe"
    )


def test_policy_evaluation_counts_abstention_as_zero_fold_delta() -> None:
    safe = _row(
        question="q1",
        fold="fold_1",
        action_id="safe",
        feature=1.0,
        citation_delta=1,
        f1_delta=0.2,
    )
    abstained = _row(
        question="q2",
        fold="fold_1",
        action_id="abstain",
        feature=-1.0,
        citation_delta=-1,
        f1_delta=-0.4,
    )
    evaluation = evaluate_dual_target_policy(
        (
            DualTargetPrediction(safe, 0.9, 0.1),
            DualTargetPrediction(abstained, 0.2, 0.8),
        ),
        spec=_spec("citation_minus_risk"),
        utility_threshold=0.0,
        total_question_count=2,
        expected_fold_ids=("fold_1",),
    )

    assert evaluation["selected_question_count"] == 1
    assert evaluation["mean_f1_delta_all_questions"] == 0.1
    assert evaluation["folds"]["fold_1"]["mean_f1_delta_all_fold_questions"] == 0.1


def test_nested_selection_uses_inner_folds_and_returns_public_aggregates() -> None:
    rows = []
    for fold_index in range(5):
        fold = f"fold_{fold_index + 1}"
        for question_index in range(2):
            question = f"{fold}_q{question_index}"
            rows.extend(
                (
                    _row(
                        question=question,
                        fold=fold,
                        action_id="good",
                        feature=2.0,
                        citation_delta=1,
                        f1_delta=0.2,
                    ),
                    _row(
                        question=question,
                        fold=fold,
                        action_id="neutral",
                        feature=0.0,
                        citation_delta=0,
                        f1_delta=0.0,
                    ),
                    _row(
                        question=question,
                        fold=fold,
                        action_id="risky",
                        feature=-2.0,
                        citation_delta=0,
                        f1_delta=-0.2,
                    ),
                )
            )
    result = run_nested_dual_target_selection(
        rows,
        specs=(_spec("citation_minus_risk"),),
        total_question_count=10,
    )

    assert result["protocol"]["outer_fold_count"] == 5
    assert result["model_head_fit_count"] == 50
    assert result["aggregate"]["gold_citation_delta"] == 10
    assert result["aggregate"]["mean_f1_delta_all_questions"] == 0.2
    assert result["aggregate"]["strict_aggregate_pass"] is True
    assert "selected_actions" in result
    assert len(result["outer_predictions"]) == 30
    assert all(report["selected_spec"] for report in result["outer_folds"].values())


def _spec(score_mode: ScoreMode) -> DualTargetPolicySpec:
    return DualTargetPolicySpec(
        name=f"logistic_{score_mode}_c100",
        model_family="logistic",
        score_mode=score_mode,
        target_coverage=1.0,
    )


def _row(
    *,
    question: str,
    fold: str,
    action_id: str,
    feature: float,
    citation_delta: int,
    f1_delta: float,
) -> ActionAuditRow:
    return ActionAuditRow(
        question_key=question,
        fold_id=fold,
        route="other",
        action=CompositionAction(
            action_id=action_id,
            family="replace_slot_1",
            aliases=("replace_slot_1",),
            selected_indices=(0,),
            matches_stage180=False,
        ),
        runtime_features={"signal": feature, "action_family": action_id},
        outcome_class="synthetic",
        strict_expected=citation_delta >= 0
        and f1_delta >= 0
        and (citation_delta > 0 or f1_delta > 0),
        citation_delta=citation_delta,
        f1_delta=f1_delta,
    )
