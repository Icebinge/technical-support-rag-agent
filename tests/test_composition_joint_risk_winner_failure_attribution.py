from __future__ import annotations

from ts_rag_agent.application import (
    composition_joint_risk_winner_failure_attribution as attribution,
)
from ts_rag_agent.application.composition_action_audit import ActionAuditRow, CompositionAction
from ts_rag_agent.application.composition_joint_risk_winner_cv import (
    JointRiskWinnerCandidateSnapshot,
    JointRiskWinnerDiagnosticSnapshot,
)
from ts_rag_agent.application.composition_safety_first_frontier import (
    FrontierActionPrediction,
    SafetyFirstFrontierDecision,
)
from ts_rag_agent.application.primeqa_hybrid_joint_risk_winner_failure_attribution_protocol import (
    _eligibility_constraints,
)


def test_stage201_attributor_preserves_populations_and_exact_partitions() -> None:
    attributor = attribution.JointRiskWinnerFailureAttributor(
        constraints=_eligibility_constraints()
    )

    attributor(_snapshot())
    report = attributor.report()

    assert report["population"] == {
        "outer_context_count": 1,
        "outer_cell_context_count": 28,
        "fold_cell_context_count": 112,
        "question_cell_context_count": 112,
    }
    question = report["question_context_attribution"]["aggregate"]
    assert question["selected_outcome_partition_exact"] is True
    assert question["strict_opportunity_partition_exact"] is True
    assert question["selected_outcome_counts"]["strict_success"] == 112
    assert question["strict_opportunity_mechanism_counts"]["strict_selected"] == 112
    assert report["constraint_attribution"]["cofailure"]["pair_count"] == 78
    assert report["privacy_contract"]["question_level_rows_persisted"] is False


def test_stage201_user_confirmed_near_boundary_a_is_applied() -> None:
    attributor = attribution.JointRiskWinnerFailureAttributor(
        constraints=_eligibility_constraints()
    )

    attributor(_snapshot())
    constraints = attributor.report()["constraint_attribution"]["constraints"]

    assert constraints["citation_delta"]["near_boundary_count"] == 28
    assert constraints["mean_f1_delta"]["near_boundary_count"] == 28
    assert constraints["strict_opportunity_pool_recall"]["near_boundary_count"] == 28
    assert constraints["strict_success_precision"]["near_boundary_count"] == 0
    assert all(row["failure_count"] == 28 for row in constraints.values())


def test_stage201_finding_separates_success_partition_from_failure_mechanism() -> None:
    constraints = {
        name: {"failure_count": 1}
        for names in (
            (
                "strict_opportunity_pool_recall",
                "folds_meeting_pool_recall_minimum",
                "conditional_ranker_strict_capture",
                "folds_meeting_conditional_capture_minimum",
            ),
            (
                "strict_success_precision",
                "unsafe_selection_rate",
                "folds_meeting_unsafe_rate_maximum",
            ),
            (
                "citation_delta",
                "mean_f1_delta",
                "citation_nonregressing_fold_count",
                "f1_nonregressing_fold_count",
            ),
        )
        for name in names
    }
    questions = {
        "strict_opportunity_mechanism_counts": {
            "strict_selected": 100,
            "winner_selection_miss": 40,
            "risk_frontier_exclusion": 10,
            "safety_pool_exclusion": 5,
            "no_strict_opportunity": 2,
        }
    }

    finding = attribution._research_recommendation(constraints, questions)

    assert finding["dominant_question_partition"] == "strict_selected"
    assert finding["dominant_failure_mechanism"] == "winner_selection_miss"


def _snapshot() -> JointRiskWinnerDiagnosticSnapshot:
    fold_ids = tuple(f"fold_{index}" for index in range(1, 5))
    decisions = tuple(_decision(fold_id) for fold_id in fold_ids)
    risks = (
        "source_weighted_classifier",
        "decomposed_loss_risk",
        "pairwise_safety_ranker",
        "decomposed_pairwise_rank_fusion",
    )
    winners = (
        "gain_only",
        "rank_utility_0.25",
        "rank_utility_0.50",
        "rank_utility_1.00",
        "rank_utility_2.00",
        "gain_shortlist_2_then_risk",
        "gain_shortlist_4_then_risk",
    )
    candidates = tuple(
        _candidate(risk, winner, decisions, fold_ids) for risk in risks for winner in winners
    )
    return JointRiskWinnerDiagnosticSnapshot(
        outer_fold_id="fold_5",
        inner_fold_ids=fold_ids,
        inner_question_count=4,
        candidates=candidates,
    )


def _candidate(
    risk: str,
    winner: str,
    decisions: tuple[SafetyFirstFrontierDecision, ...],
    fold_ids: tuple[str, ...],
) -> JointRiskWinnerCandidateSnapshot:
    evaluation_folds = {
        fold_id: {
            "gold_citation_delta": 1 if index < 2 else -1,
            "mean_f1_delta": 0.01 if index < 2 else -0.01,
        }
        for index, fold_id in enumerate(fold_ids)
    }
    diagnostic_folds = {
        fold_id: {
            "strict_opportunity_pool_recall": 1.0 if index < 2 else 0.8,
            "conditional_ranker_strict_capture": 0.7 if index < 2 else 0.5,
            "unsafe_selection_rate": 0.2 if index < 2 else 0.5,
        }
        for index, fold_id in enumerate(fold_ids)
    }
    name = f"risk_{risk}__winner_{winner}"
    return JointRiskWinnerCandidateSnapshot(
        spec={"name": name, "risk_signal": risk, "winner_rule": winner},
        eligible=False,
        evaluation={
            "gold_citation_delta": -1,
            "mean_f1_delta": -0.0005,
            "citation_nonregressing_fold_count": 2,
            "f1_nonregressing_fold_count": 2,
            "changed_question_count": 0,
            "strict_success_count": 0,
            "strict_success_precision": 0.0,
            "folds": evaluation_folds,
        },
        diagnostics={
            "strict_opportunity_pool_recall": 0.945,
            "folds_meeting_pool_recall_minimum": 2,
            "conditional_ranker_strict_capture": 0.675,
            "folds_meeting_conditional_capture_minimum": 2,
            "unsafe_selection_rate": 0.255,
            "folds_meeting_unsafe_rate_maximum": 2,
            "folds": diagnostic_folds,
        },
        paired_vs_control={},
        decisions=decisions,
    )


def _decision(fold_id: str) -> SafetyFirstFrontierDecision:
    baseline = FrontierActionPrediction(_row(fold_id, "baseline"), 0.0, 0.0, 0.0, 0.0)
    strict = FrontierActionPrediction(
        _row(fold_id, "strict", strict=True, citation_delta=1, f1_delta=0.1),
        0.0,
        0.0,
        1.0,
        0.1,
    )
    return SafetyFirstFrontierDecision(
        question_key=f"question_{fold_id}",
        baseline=baseline,
        complete_pool=(baseline, strict),
        frontier=(baseline, strict),
        winner=strict,
        strict_opportunity=True,
        action_count=2,
        strict_action_count=1,
    )


def _row(
    fold_id: str,
    action_id: str,
    *,
    strict: bool = False,
    citation_delta: int = 0,
    f1_delta: float = 0.0,
) -> ActionAuditRow:
    return ActionAuditRow(
        question_key=f"question_{fold_id}",
        fold_id=fold_id,
        route="other",
        action=CompositionAction(
            action_id=action_id,
            family=action_id,
            aliases=(),
            selected_indices=(0,),
            matches_stage180=False,
        ),
        runtime_features={},
        outcome_class="test",
        strict_expected=strict,
        citation_delta=citation_delta,
        f1_delta=f1_delta,
    )
