from __future__ import annotations

import inspect
import xml.etree.ElementTree as ET
from pathlib import Path

from typer.testing import CliRunner

from scripts.validate_primeqa_hybrid_hierarchical_router import app, main
from ts_rag_agent.application import primeqa_hybrid_hierarchical_router_validation as stage171
from ts_rag_agent.application import primeqa_hybrid_iterative_router_calibration as stage169
from ts_rag_agent.application.primeqa_hybrid_hierarchical_decision_router import (
    EvidenceDisposition,
    HierarchicalLayerInvocationMetrics,
    HierarchicalRouterTrace,
    RequestDisposition,
)
from ts_rag_agent.application.primeqa_hybrid_iterative_decision_router import (
    IterativeDecisionAction,
    IterativeDecisionPhase,
)


def test_final_synthetic_refusal_is_complete_request_with_insufficient_evidence() -> None:
    outcome = _phase(
        action=IterativeDecisionAction.REFUSE.value,
        request=RequestDisposition.COMPLETE.value,
        evidence=EvidenceDisposition.INSUFFICIENT.value,
    )

    row = stage171._synthetic_row(
        case_id="inspect-then-refuse",
        phase=IterativeDecisionPhase.FINAL_AFTER_INSPECTION,
        expected_action=IterativeDecisionAction.REFUSE.value,
        expected_clarification_kind=None,
        outcome=outcome,
    )

    assert row["expected_request_disposition"] == RequestDisposition.COMPLETE.value
    assert row["expected_evidence_disposition"] == EvidenceDisposition.INSUFFICIENT.value


def test_train_aggregation_and_folds_preserve_path_and_absent_strata() -> None:
    outcomes = (
        _train(
            "initial_gold_visible",
            "fold_1",
            IterativeDecisionAction.COMPOSE.value,
            IterativeDecisionAction.COMPOSE.value,
        ),
        _train(
            "alternate_only_gold_visible",
            "fold_1",
            IterativeDecisionAction.INSPECT.value,
            IterativeDecisionAction.COMPOSE.value,
        ),
        _train(
            "unanswerable",
            "fold_2",
            IterativeDecisionAction.INSPECT.value,
            IterativeDecisionAction.REFUSE.value,
        ),
    )

    aggregate = stage171._aggregate_train_outcomes(outcomes)
    folds = stage171._fold_stability(outcomes)

    assert aggregate["alternate_only_gold_visible"]["inspect_then_compose_count"] == 1
    assert folds["folds"]["fold_1"]["alternate_only_path_success_rate"] == 1.0
    assert folds["folds"]["fold_2"]["alternate_only_path_success_rate"] is None
    assert folds["folds"]["fold_2"]["insufficient_final_compose_rate"] == 0.0


def test_hierarchy_gates_enforce_all_five_frozen_thresholds() -> None:
    metrics = {
        "synthetic_request_disposition_accuracy": 0.95,
        "synthetic_evidence_disposition_accuracy": 0.9,
        "train_request_complete_rate": 0.94,
        "request_layer_schema_valid_rate": 1.0,
        "evidence_layer_schema_valid_rate": 0.99,
    }

    gates = stage171._hierarchy_gates(metrics)

    assert [gate["passed"] for gate in gates] == [True, True, False, True, False]


def test_visualizations_write_six_parseable_svgs(tmp_path: Path) -> None:
    visualizations = stage171.write_stage171_visualizations(
        report=_visual_report(), output_dir=tmp_path
    )

    assert len(visualizations) == 6
    for visualization in visualizations:
        ET.parse(visualization.path)


def test_cli_exposes_no_development_test_retry_or_fallback_paths() -> None:
    result = CliRunner().invoke(app, ["--help"])
    parameters = set(inspect.signature(main).parameters)

    assert result.exit_code == 0
    assert "model_snapshot" in parameters
    assert not ({"dev", "development", "dev_split", "test", "test_split"} & parameters)
    assert not ({"retry", "fallback"} & parameters)


def _train(
    stratum: str,
    fold_id: str,
    initial_action: str,
    final_action: str,
) -> stage171.HierarchicalTrainOutcome:
    return stage171.HierarchicalTrainOutcome(
        stratum=stratum,
        fold_id=fold_id,
        initial=_phase(
            action=initial_action,
            request=RequestDisposition.COMPLETE.value,
            evidence=(
                EvidenceDisposition.SUFFICIENT.value
                if initial_action == IterativeDecisionAction.COMPOSE.value
                else EvidenceDisposition.INSUFFICIENT.value
            ),
        ),
        final=_phase(
            action=final_action,
            request=RequestDisposition.COMPLETE.value,
            evidence=(
                EvidenceDisposition.SUFFICIENT.value
                if final_action == IterativeDecisionAction.COMPOSE.value
                else EvidenceDisposition.INSUFFICIENT.value
            ),
        ),
    )


def _phase(action: str, request: str, evidence: str) -> stage171.HierarchicalPhaseOutcome:
    layer_metrics = (
        HierarchicalLayerInvocationMetrics("request", 10, 2, 1.0, True, request, None),
        HierarchicalLayerInvocationMetrics("evidence", 20, 2, 2.0, True, evidence, None),
    )
    trace = HierarchicalRouterTrace(
        phase=IterativeDecisionPhase.INITIAL.value,
        request_disposition=request,
        clarification_kind=None,
        evidence_disposition=evidence,
        selected_action=action,
        schema_valid=True,
        layer_metrics=layer_metrics,
    )
    decision = stage169.RouterCallObservation(
        action=action,
        clarification_kind=None,
        schema_valid=True,
        input_token_count=30,
        output_token_count=4,
        generation_latency_ms=3.0,
        process_working_set_bytes=0,
        process_private_usage_bytes=0,
        system_available_memory_bytes=0,
        gpu_peak_allocated_bytes=0,
        gpu_peak_reserved_bytes=0,
    )
    return stage171.HierarchicalPhaseOutcome(decision=decision, trace=trace)


def _visual_report() -> dict:
    fold = {
        "alternate_only_path_success_rate": 0.8,
        "insufficient_final_compose_rate": 0.1,
    }
    return {
        "legacy_quality_gate_pass_count": 7,
        "quality_metrics": {
            "synthetic_request_disposition_accuracy": 0.95,
            "synthetic_evidence_disposition_accuracy": 0.9,
            "synthetic_phase_action_accuracy": 0.9,
            "synthetic_clarification_kind_accuracy": 1.0,
            "real_initial_visible_compose_rate": 0.8,
            "real_alternate_only_inspect_rate": 0.8,
            "real_alternate_only_final_compose_rate": 0.8,
            "real_alternate_only_path_success_rate": 0.7,
            "real_insufficient_final_compose_rate": 0.1,
            "latency_ms_by_layer": {
                "request": {"p95": 400.0},
                "evidence": {"p95": 900.0},
            },
        },
        "fold_stability": {"folds": {f"fold_{index}": fold for index in range(1, 6)}},
        "resource_consumption": {
            "process_peak_working_set_bytes": 7 * 1024**3,
            "process_peak_private_usage_bytes": 13 * 1024**3,
            "gpu_peak_allocated_bytes": 5 * 1024**3,
            "gpu_peak_reserved_bytes": 7 * 1024**3,
        },
    }
