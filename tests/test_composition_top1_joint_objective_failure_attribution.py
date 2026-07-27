from __future__ import annotations

from dataclasses import replace

import pytest

from ts_rag_agent.application import (
    composition_top1_joint_objective_failure_attribution as attribution,
)
from ts_rag_agent.application.composition_action_audit import ActionAuditRow, CompositionAction
from ts_rag_agent.application.composition_safety_first_frontier import (
    FrontierActionPrediction,
    SafetyFirstFrontierDecision,
)
from ts_rag_agent.application.composition_top1_joint_objective_cv import (
    Top1ObjectiveCandidateSnapshot,
    Top1ObjectiveDiagnosticSnapshot,
)


def test_stage204_attributor_preserves_frozen_populations_and_partitions() -> None:
    attributor = attribution.Top1JointObjectiveFailureAttributor()

    attributor(_snapshot())
    report = attributor.report()

    assert report["population"] == {
        "outer_context_count": 1,
        "outer_cell_context_count": 17,
        "custom_outer_cell_context_count": 16,
        "question_context_count": 2,
        "control_custom_question_comparison_count": 32,
        "precision_adjacent_question_comparison_count": 24,
        "safety_adjacent_question_comparison_count": 24,
    }
    aggregate = report["control_to_custom"]["aggregate"]
    assert aggregate["left_partition_exact"] is True
    assert aggregate["right_partition_exact"] is True
    assert aggregate["transition_partition_exact"] is True
    assert len(report["control_to_custom"]["by_candidate"]) == 16
    assert len(report["precision_adjacent_attribution"]["by_pair"]) == 12
    assert len(report["safety_adjacent_attribution"]["by_pair"]) == 12
    assert report["target_mechanics"]["precision_component_mass_sum_exact"] is True
    assert report["privacy_contract"]["question_level_rows_persisted"] is False


def test_stage204_attributor_exposes_precision_strict_displacement() -> None:
    attributor = attribution.Top1JointObjectiveFailureAttributor()

    attributor(_snapshot())
    report = attributor.report()
    precision = report["precision_adjacent_attribution"]["aggregate"]

    assert precision["strict_loss_count"] == 6
    assert precision["strict_gain_count"] == 0
    assert precision["baseline_addition_count"] == 8
    assert precision["cell_comparison_count"] == 12
    assert precision["strict_precision_nondecreasing_cell_count"] == 0
    assert report["diagnostic_finding"]["precision_displaces_more_strict_than_it_recovers"] is True
    assert (
        report["diagnostic_finding"]["dominant_precision_outcome_change"]
        == "strict_success__to__baseline"
    )


def test_stage204_rejects_candidate_pool_drift() -> None:
    snapshot = _snapshot()
    candidate = snapshot.candidates[1]
    decision = candidate.decisions[0]
    drifted = replace(decision, complete_pool=decision.complete_pool[:-1])
    candidates = list(snapshot.candidates)
    candidates[1] = replace(candidate, decisions=(drifted, candidate.decisions[1]))
    attributor = attribution.Top1JointObjectiveFailureAttributor()

    with pytest.raises(ValueError, match="identical candidate pool"):
        attributor(replace(snapshot, candidates=tuple(candidates)))


def _snapshot() -> Top1ObjectiveDiagnosticSnapshot:
    control_decisions = tuple(_decision(f"question_{index}", "strict") for index in range(2))
    candidates = [_candidate("stage196_exact_control", None, None, control_decisions)]
    for safety_weight in (0.0, 0.5, 1.0, 2.0):
        for precision_weight in (0.0, 0.5, 1.0, 2.0):
            winner = (
                "baseline"
                if precision_weight > 0.0
                else "unsafe"
                if safety_weight == 0.0
                else "strict"
            )
            decisions = tuple(_decision(f"question_{index}", winner) for index in range(2))
            candidates.append(
                _candidate(
                    f"top1_safety_{safety_weight:.2f}__precision_{precision_weight:.2f}",
                    safety_weight,
                    precision_weight,
                    decisions,
                )
            )
    return Top1ObjectiveDiagnosticSnapshot(
        outer_fold_id="fold_1",
        inner_fold_ids=("fold_2", "fold_3", "fold_4", "fold_5"),
        inner_question_count=2,
        candidates=tuple(candidates),
    )


def _candidate(
    name: str,
    safety_weight: float | None,
    precision_weight: float | None,
    decisions: tuple[SafetyFirstFrontierDecision, ...],
) -> Top1ObjectiveCandidateSnapshot:
    precision = 0.7 if precision_weight in (None, 0.0) else 0.6 - precision_weight / 10.0
    unsafe = 0.4 if safety_weight in (None, 0.0) else 0.3 - safety_weight / 20.0
    capture = 0.7 if precision_weight in (None, 0.0) else 0.6 - precision_weight / 10.0
    return Top1ObjectiveCandidateSnapshot(
        spec={
            "name": name,
            "safety_weight": safety_weight,
            "precision_weight": precision_weight,
            "ablation_family": "exact_control" if safety_weight is None else "test",
        },
        eligible=False,
        evaluation={"strict_success_precision": precision},
        diagnostics={
            "unsafe_selection_rate": unsafe,
            "conditional_ranker_strict_capture": capture,
        },
        paired_vs_control={},
        paired_vs_strict_only={},
        decisions=decisions,
    )


def _decision(question_key: str, winner_name: str) -> SafetyFirstFrontierDecision:
    rows = {
        "baseline": FrontierActionPrediction(_row(question_key, "baseline"), 0.0, 0.0, 0.0, 0.0),
        "strict": FrontierActionPrediction(
            _row(question_key, "strict", strict=True, citation_delta=1, f1_delta=0.1),
            0.0,
            0.0,
            1.0,
            0.0,
        ),
        "safe_zero": FrontierActionPrediction(_row(question_key, "safe_zero"), 0.0, 0.0, 0.5, 0.0),
        "unsafe": FrontierActionPrediction(
            _row(question_key, "unsafe", citation_delta=-1, f1_delta=-0.1),
            1.0,
            1.0,
            0.2,
            1.0,
        ),
    }
    pool = tuple(rows.values())
    return SafetyFirstFrontierDecision(
        question_key=question_key,
        baseline=rows["baseline"],
        complete_pool=pool,
        frontier=pool,
        winner=rows[winner_name],
        strict_opportunity=True,
        action_count=len(pool),
        strict_action_count=1,
    )


def _row(
    question_key: str,
    action_id: str,
    *,
    strict: bool = False,
    citation_delta: int = 0,
    f1_delta: float = 0.0,
) -> ActionAuditRow:
    return ActionAuditRow(
        question_key=question_key,
        fold_id="fold_2",
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
