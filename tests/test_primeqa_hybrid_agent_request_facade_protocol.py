import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_agent_request_facade_protocol import (
    AgentRequestFacadeProtocolEvidence,
    AgentRequestFacadeProtocolState,
    StrictAgentRequestFacadeProtocolPolicy,
    _forbidden_keys_found,
    freeze_primeqa_hybrid_agent_request_facade_protocol,
    write_primeqa_hybrid_agent_request_facade_protocol_visualizations,
)


def test_facade_policy_accepts_exact_contract_without_implementing() -> None:
    evaluation = StrictAgentRequestFacadeProtocolPolicy().evaluate(_compliant_evidence())

    assert evaluation.state is AgentRequestFacadeProtocolState.ELIGIBLE
    assert evaluation.rejection_reasons == ()
    assert evaluation.facade_implemented is False
    assert evaluation.network_service_implemented is False


def test_facade_policy_rejects_content_and_label_leaks() -> None:
    evaluation = StrictAgentRequestFacadeProtocolPolicy().evaluate(
        _compliant_evidence(
            runtime_request_is_label_free=False,
            public_telemetry_contains_request_content=True,
            public_telemetry_contains_response_content=True,
        )
    )

    assert evaluation.state is AgentRequestFacadeProtocolState.REJECTED
    assert evaluation.rejection_reasons == (
        "runtime_request_reads_labels",
        "request_content_exposed_in_public_telemetry",
        "response_content_exposed_in_public_telemetry",
    )


def test_facade_policy_rejects_queue_retry_fallback_and_answer_conversion() -> None:
    evaluation = StrictAgentRequestFacadeProtocolPolicy().evaluate(
        _compliant_evidence(
            capacity_error_mapping_exact=False,
            capacity_rejected_before_downstream=False,
            errors_converted_to_answers=True,
            queue_action_count=1,
            retry_action_count=1,
            fallback_action_count=1,
        )
    )

    assert set(evaluation.rejection_reasons) == {
        "capacity_mapping_not_exact",
        "capacity_rejection_reached_downstream",
        "errors_converted_to_answers",
        "queue_action_detected",
        "retry_action_detected",
        "fallback_action_detected",
    }


def test_facade_policy_rejects_hard_cancel_and_implicit_shutdown_timeout() -> None:
    evaluation = StrictAgentRequestFacadeProtocolPolicy().evaluate(
        _compliant_evidence(
            in_flight_hard_cancellation_allowed=True,
            shutdown_waits_for_in_flight=False,
            implicit_shutdown_timeout_allowed=True,
            force_cancel_allowed=True,
        )
    )

    assert evaluation.rejection_reasons == (
        "in_flight_hard_cancellation_enabled",
        "shutdown_does_not_wait_for_in_flight",
        "implicit_shutdown_timeout_enabled",
        "force_cancel_enabled",
    )


def test_stage147_freezes_all_facade_boundaries_from_stage146(tmp_path: Path) -> None:
    report = freeze_primeqa_hybrid_agent_request_facade_protocol(
        stage146_validation_path=_write_stage146_source(tmp_path),
        user_confirmed_protocol=True,
        confirmation_note="User requested the next large step after Stage146.",
    )

    assert len(report["guard_checks"]) == 34
    assert all(check["passed"] for check in report["guard_checks"])
    decision = report["decision"]
    assert decision["agent_request_facade_protocol_frozen"] is True
    assert decision["facade_protocol_policy_executable"] is True
    assert decision["facade_implementation_allowed_next"] is True
    assert decision["facade_implemented_now"] is False
    assert decision["network_service_implemented"] is False
    assert decision["runtime_registered_as_default"] is False
    assert decision["test_gate_opened"] is False
    assert decision["queue_actions_enabled"] is False
    assert decision["retry_actions_enabled"] is False
    assert decision["fallback_strategies_enabled"] is False
    assert report["public_safe_contract"]["forbidden_keys_found"] == []


def test_stage147_rejects_unconfirmed_or_invalid_stage146(tmp_path: Path) -> None:
    source = _stage146_source()
    source["decision"]["network_service_implemented"] = True
    source_path = tmp_path / "stage146.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    report = freeze_primeqa_hybrid_agent_request_facade_protocol(
        stage146_validation_path=source_path,
        user_confirmed_protocol=False,
        confirmation_note="",
    )

    failed = report["decision"]["failed_checks"]
    assert "stage147_user_confirmed" in failed
    assert "stage147_confirmation_note_present" in failed
    assert "stage146_closed_boundaries_preserved" in failed
    assert report["decision"]["facade_implementation_allowed_next"] is False


def test_stage147_public_safety_detects_nested_content_keys() -> None:
    assert _forbidden_keys_found({"request_count": 1, "nested": {"state": "ok"}}) == set()
    assert _forbidden_keys_found({"nested": {"question_text": "private"}}) == {"question_text"}
    assert _forbidden_keys_found({"nested": {"request_handle": "private"}}) == {"request_handle"}


def test_stage147_writes_and_parses_all_visualizations(tmp_path: Path) -> None:
    report = freeze_primeqa_hybrid_agent_request_facade_protocol(
        stage146_validation_path=_write_stage146_source(tmp_path),
        user_confirmed_protocol=True,
        confirmation_note="User requested the next large step after Stage146.",
    )

    visualizations = write_primeqa_hybrid_agent_request_facade_protocol_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    assert {visualization.name for visualization in visualizations} == {
        "stage147_source_activation_boundary.svg",
        "stage147_private_call_contract.svg",
        "stage147_public_telemetry_allowlist.svg",
        "stage147_error_mapping.svg",
        "stage147_cancellation_boundary.svg",
        "stage147_lifecycle.svg",
        "stage147_shutdown_contract.svg",
        "stage147_policy_cases.svg",
        "stage147_decision_flags.svg",
        "stage147_guard_check_status.svg",
    }
    for visualization in visualizations:
        ET.parse(visualization.path)


def _compliant_evidence(**overrides: object) -> AgentRequestFacadeProtocolEvidence:
    values = {
        "source_stage146_validated": True,
        "explicit_nondefault_concurrent_runtime_available": True,
        "active_runtime_required": True,
        "facade_owns_runtime_resources": False,
        "private_request_payload": True,
        "runtime_request_is_label_free": True,
        "private_response_payload": True,
        "public_telemetry_allowlist_only": True,
        "public_telemetry_contains_request_content": False,
        "public_telemetry_contains_response_content": False,
        "capacity_error_mapping_exact": True,
        "capacity_rejected_before_downstream": True,
        "invalid_request_rejected_before_downstream": True,
        "pre_dispatch_cancellation_only": True,
        "in_flight_hard_cancellation_allowed": False,
        "downstream_errors_propagate": True,
        "errors_converted_to_answers": False,
        "lifecycle_sequence_exact": True,
        "draining_rejects_new_requests": True,
        "shutdown_waits_for_in_flight": True,
        "implicit_shutdown_timeout_allowed": False,
        "force_cancel_allowed": False,
        "queue_action_count": 0,
        "retry_action_count": 0,
        "fallback_action_count": 0,
        "network_transport_deferred": True,
        "test_split_locked": True,
        "runtime_default_unchanged": True,
    }
    values.update(overrides)
    return AgentRequestFacadeProtocolEvidence(**values)  # type: ignore[arg-type]


def _write_stage146_source(tmp_path: Path) -> Path:
    path = tmp_path / "stage146.json"
    path.write_text(json.dumps(_stage146_source()), encoding="utf-8")
    return path


def _stage146_source() -> dict:
    return {
        "stage": "Stage 146",
        "analysis_id": ("primeqa_hybrid_concurrent_runtime_application_activation_validation_v1"),
        "guard_checks": [
            {"name": f"stage146_guard_{index}", "passed": True} for index in range(43)
        ],
        "decision": {
            "status": (
                "primeqa_hybrid_concurrent_runtime_application_activation_validation_passed"
            ),
            "application_activation_bootstrap_implemented": True,
            "eligible_runtime_full_workload_validation_passed": True,
            "explicit_nondefault_concurrent_activation_available": True,
            "runtime_registered_as_default": False,
            "runtime_defaultization_allowed_now": False,
            "network_service_implemented": False,
            "test_gate_opened": False,
            "test_metrics_run": False,
            "queue_actions_enabled": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
        },
        "public_safe_contract": {
            "test_split_loaded": False,
            "test_metrics_run": False,
            "forbidden_keys_found": [],
        },
    }
