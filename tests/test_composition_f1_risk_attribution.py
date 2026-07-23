from __future__ import annotations

from ts_rag_agent.application.composition_action_audit import (
    ActionAuditRow,
    CompositionAction,
)
from ts_rag_agent.application.composition_dual_target_policy import (
    DualTargetPrediction,
    SelectedAction,
)
from ts_rag_agent.application.composition_f1_risk_attribution import (
    run_f1_risk_attribution,
)


def test_f1_risk_attribution_finds_safe_alternative_and_weak_head() -> None:
    regressed = _row("q1", "selected", citation_delta=1, f1_delta=-0.2, signal=1.0)
    safe = _row("q1", "safe", citation_delta=1, f1_delta=0.1, signal=0.0)
    positive = _row("q2", "positive", citation_delta=0, f1_delta=0.2, signal=0.0)
    predictions = (
        DualTargetPrediction(regressed, 0.8, 0.2),
        DualTargetPrediction(safe, 0.7, 0.1),
        DualTargetPrediction(positive, 0.3, 0.1),
    )
    selected = (
        SelectedAction(
            regressed, utility=0.6, citation_gain_probability=0.8, f1_regression_probability=0.2
        ),
        SelectedAction(
            positive, utility=0.2, citation_gain_probability=0.3, f1_regression_probability=0.1
        ),
    )
    result = run_f1_risk_attribution(
        action_rows=(regressed, safe, positive),
        selected_actions=selected,
        outer_predictions=predictions,
        outer_fold_reports={
            "fold_1": _fold_report(
                selected_spec="logistic_citation_minus_risk_c100",
                risk_auc=0.58,
            )
        },
    )

    assert result["selected_action_summary"]["f1_regression_count"] == 1
    assert "selected_actions" not in result["risk_calibration"]
    assert (
        result["risk_calibration"]["selected_action_population"]["observed_regression_count"] == 1
    )
    assert (
        result["safe_alternative_headroom"][
            "questions_with_same_or_better_citation_safe_alternative"
        ]
        == 1
    )
    assert result["safe_alternative_headroom"]["same_or_better_safe_alternative_in_model_top3"] == 1
    assert result["diagnostic_findings"]["weak_f1_risk_head_separability"] is True
    assert result["diagnostic_findings"]["primary_bottleneck"] == (
        "f1_risk_separability_and_ranking"
    )


def test_no_eligible_fold_attributes_each_failed_constraint() -> None:
    row = _row("q1", "selected", citation_delta=0, f1_delta=0.1, signal=0.0)
    prediction = DualTargetPrediction(row, 0.2, 0.1)
    report = _fold_report(selected_spec=None, risk_auc=0.60)
    report["candidate_leaderboard"] = [
        {
            "name": "logistic_citation_minus_risk_c050",
            "evaluation": {
                "strict_aggregate_pass": True,
                "gold_citation_delta": 3,
                "mean_f1_delta_all_questions": 0.01,
                "citation_nonregressing_fold_count": 4,
                "f1_nonregressing_fold_count": 3,
                "folds": {f"fold_{index}": {} for index in range(1, 5)},
            },
        },
        {
            "name": "logistic_safe_product_c100",
            "evaluation": {
                "strict_aggregate_pass": False,
                "gold_citation_delta": 4,
                "mean_f1_delta_all_questions": -0.01,
                "citation_nonregressing_fold_count": 3,
                "f1_nonregressing_fold_count": 2,
                "folds": {f"fold_{index}": {} for index in range(1, 5)},
            },
        },
    ]
    result = run_f1_risk_attribution(
        action_rows=(row,),
        selected_actions=(),
        outer_predictions=(prediction,),
        outer_fold_reports={"fold_1": report},
    )

    attribution = result["no_inner_eligible_fold_attribution"]
    assert attribution["fold_count"] == 1
    assert attribution["failure_reason_counts"] == {
        "aggregate_strict_a_failed": 1,
        "citation_fold_nonregression_failed": 1,
        "f1_fold_nonregression_failed": 2,
    }
    assert attribution["folds"]["fold_1"]["closest_candidates"][0]["failure_reasons"] == [
        "f1_fold_nonregression_failed"
    ]


def _fold_report(*, selected_spec: str | None, risk_auc: float) -> dict:
    return {
        "selected_spec": selected_spec,
        "inner_head_metrics": {
            "logistic": {
                "aggregate": {
                    "citation_gain": {"roc_auc": 0.72},
                    "f1_regression": {"roc_auc": risk_auc},
                }
            }
        },
        "candidate_leaderboard": [],
    }


def _row(
    question: str,
    action_id: str,
    *,
    citation_delta: int,
    f1_delta: float,
    signal: float,
) -> ActionAuditRow:
    return ActionAuditRow(
        question_key=question,
        fold_id="fold_1",
        route="other",
        action=CompositionAction(
            action_id=action_id,
            family="replace_slot_1",
            aliases=("replace_slot_1",),
            selected_indices=(0,),
            matches_stage180=False,
        ),
        runtime_features={"signal": signal, "is_replace": True},
        outcome_class="synthetic",
        strict_expected=citation_delta >= 0
        and f1_delta >= 0
        and (citation_delta > 0 or f1_delta > 0),
        citation_delta=citation_delta,
        f1_delta=f1_delta,
    )
