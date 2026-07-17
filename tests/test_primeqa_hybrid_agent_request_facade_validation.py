import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_agent_request_facade_validation import (
    run_primeqa_hybrid_agent_request_facade_validation,
    write_primeqa_hybrid_agent_request_facade_validation_visualizations,
)


def test_stage148_validates_complete_facade_contract_without_opening_network(
    tmp_path: Path,
) -> None:
    source_path = _write_stage147_source(tmp_path)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

    report = run_primeqa_hybrid_agent_request_facade_validation(
        stage147_protocol_path=source_path,
        user_confirmed_validation=True,
        confirmation_note="User requested the next large Stage148 implementation step.",
    )

    assert len(report["guard_checks"]) == 37
    assert all(check["passed"] for check in report["guard_checks"])
    assert report["source_gate_passed"] is True
    assert report["synthetic_validation_executed"] is True
    assert report["source_unchanged_after_validation"] is True
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
    decision = report["decision"]
    assert decision["transport_neutral_facade_implemented"] is True
    assert decision["facade_synthetic_validation_passed"] is True
    assert decision["network_transport_protocol_allowed_next"] is True
    assert decision["network_service_implemented"] is False
    assert decision["runtime_registered_as_default"] is False
    assert decision["test_gate_opened"] is False
    assert decision["queue_actions_enabled"] is False
    assert decision["retry_actions_enabled"] is False
    assert decision["fallback_strategies_enabled"] is False
    assert report["public_safe_contract"]["forbidden_keys_found"] == []


def test_stage148_reports_label_free_error_and_lifecycle_evidence(tmp_path: Path) -> None:
    report = run_primeqa_hybrid_agent_request_facade_validation(
        stage147_protocol_path=_write_stage147_source(tmp_path),
        user_confirmed_validation=True,
        confirmation_note="User requested the next large Stage148 implementation step.",
    )
    synthetic = report["synthetic_validation"]

    query = synthetic["runtime_query_boundary"]
    assert query["received_type"] == "PrimeQARuntimeQuery"
    assert query["received_fields"] == ["id", "text", "title"]
    assert query["forbidden_attributes_present"] == []
    assert query["arrival_pattern"] == "application_request"
    assert synthetic["invalid_request_validation"]["runtime_call_count"] == 0
    assert synthetic["predispatch_cancellation_validation"]["runtime_call_count"] == 0
    capacity = synthetic["capacity_mapping_validation"]
    assert capacity["facade_error_code"] == "capacity_exceeded"
    assert capacity["runtime_admission_state"] == "rejected_capacity"
    assert capacity["queue_action_count"] == 0
    assert capacity["retry_action_count"] == 0
    assert capacity["fallback_action_count"] == 0
    assert synthetic["downstream_error_validation"]["same_error_object_propagated"] is True
    lifecycle = synthetic["lifecycle_validation"]
    assert lifecycle["in_flight_before_release"] == 1
    assert lifecycle["in_flight_completed_naturally"] is True
    assert lifecycle["closed_after_in_flight_zero"] is True
    assert lifecycle["implicit_timeout_used"] is False
    assert lifecycle["force_cancel_used"] is False


def test_stage148_fail_closed_source_gate_skips_synthetic_validation(tmp_path: Path) -> None:
    source = _stage147_source()
    source["guard_checks"][0]["passed"] = False
    source["decision"]["facade_implementation_allowed_next"] = False
    source_path = tmp_path / "stage147.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    report = run_primeqa_hybrid_agent_request_facade_validation(
        stage147_protocol_path=source_path,
        user_confirmed_validation=False,
        confirmation_note="",
    )

    assert report["source_gate_passed"] is False
    assert report["synthetic_validation_executed"] is False
    assert report["synthetic_validation"] == {}
    failed = report["decision"]["failed_checks"]
    assert "stage148_user_confirmed" in failed
    assert "stage148_confirmation_note_present" in failed
    assert "stage147_all_34_guards_passed" in failed
    assert "stage147_facade_implementation_authorized" in failed
    assert report["decision"]["transport_neutral_facade_implemented"] is False


def test_stage148_writes_and_parses_all_visualizations(tmp_path: Path) -> None:
    report = run_primeqa_hybrid_agent_request_facade_validation(
        stage147_protocol_path=_write_stage147_source(tmp_path),
        user_confirmed_validation=True,
        confirmation_note="User requested the next large Stage148 implementation step.",
    )

    visualizations = write_primeqa_hybrid_agent_request_facade_validation_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    assert {visualization.name for visualization in visualizations} == {
        "stage148_source_gate.svg",
        "stage148_runtime_query_boundary.svg",
        "stage148_private_response_mapping.svg",
        "stage148_public_telemetry_fields.svg",
        "stage148_error_outcomes.svg",
        "stage148_dispatch_counts.svg",
        "stage148_lifecycle.svg",
        "stage148_closed_boundaries.svg",
        "stage148_decision_flags.svg",
        "stage148_guard_check_status.svg",
    }
    for visualization in visualizations:
        ET.parse(visualization.path)


def _write_stage147_source(tmp_path: Path) -> Path:
    path = tmp_path / "stage147.json"
    path.write_text(json.dumps(_stage147_source()), encoding="utf-8")
    return path


def _stage147_source() -> dict:
    return {
        "stage": "Stage 147",
        "protocol_id": "primeqa_hybrid_agent_request_facade_protocol_v1",
        "guard_checks": [
            {"name": f"stage147_guard_{index}", "passed": True} for index in range(34)
        ],
        "decision": {
            "status": "primeqa_hybrid_agent_request_facade_protocol_frozen",
            "agent_request_facade_protocol_frozen": True,
            "facade_protocol_policy_executable": True,
            "facade_implementation_allowed_next": True,
            "facade_implemented_now": False,
            "network_service_implemented": False,
            "runtime_registered_as_default": False,
            "runtime_defaultization_allowed_now": False,
            "test_gate_opened": False,
            "test_metrics_run": False,
            "queue_actions_enabled": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
        },
        "public_safe_contract": {
            "train_split_loaded": False,
            "dev_split_loaded": False,
            "test_split_loaded": False,
            "test_metrics_run": False,
            "questions_loaded": False,
            "documents_loaded": False,
            "models_loaded": False,
            "indexes_loaded": False,
            "candidate_pools_built": False,
            "forbidden_keys_found": [],
        },
    }
