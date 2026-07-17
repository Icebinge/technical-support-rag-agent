from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ts_rag_agent.application.primeqa_hybrid_agent_network_transport_protocol import (
    AgentNetworkTransportProtocolState,
    StrictAgentNetworkTransportProtocolPolicy,
    _compliant_evidence,
    _forbidden_keys_found,
    freeze_primeqa_hybrid_agent_network_transport_protocol,
    write_primeqa_hybrid_agent_network_transport_protocol_visualizations,
)


def test_strict_transport_policy_accepts_only_exact_compliant_evidence() -> None:
    result = StrictAgentNetworkTransportProtocolPolicy().evaluate(_compliant_evidence())

    assert result.state is AgentNetworkTransportProtocolState.ELIGIBLE
    assert result.rejection_reasons == ()
    assert result.network_service_implemented is False


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"source_stage148_validated": False}, "stage148_source_not_validated"),
        ({"local_loopback_only": False}, "loopback_binding_not_required"),
        ({"remote_exposure_enabled": True}, "remote_exposure_enabled"),
        ({"strict_json_content_type": False}, "json_content_type_not_strict"),
        ({"unknown_request_fields_rejected": False}, "unknown_request_fields_accepted"),
        ({"raw_body_limit_enforced_before_parse": False}, "raw_body_limit_not_preparse"),
        ({"domain_refusal_is_success": False}, "domain_refusal_mapped_as_error"),
        ({"in_flight_hard_cancellation_claimed": True}, "in_flight_hard_cancellation_claimed"),
        ({"lifespan_owns_runtime_resources": True}, "transport_lifespan_owns_runtime_resources"),
        ({"default_access_log_enabled": True}, "default_access_log_enabled"),
        ({"network_service_implemented": True}, "network_service_preimplemented"),
        ({"test_split_locked": False}, "test_split_not_locked"),
    ],
)
def test_strict_transport_policy_rejects_boundary_drift(
    overrides: dict[str, object], expected_reason: str
) -> None:
    result = StrictAgentNetworkTransportProtocolPolicy().evaluate(_compliant_evidence(**overrides))

    assert result.state is AgentNetworkTransportProtocolState.REJECTED
    assert expected_reason in result.rejection_reasons


def test_strict_transport_policy_rejects_queue_retry_and_fallback_together() -> None:
    result = StrictAgentNetworkTransportProtocolPolicy().evaluate(
        _compliant_evidence(
            queue_action_count=1,
            retry_action_count=1,
            fallback_action_count=1,
        )
    )

    assert result.rejection_reasons == (
        "queue_action_detected",
        "retry_action_detected",
        "fallback_action_detected",
    )


def test_stage149_freeze_uses_only_saved_stage148_public_aggregate(tmp_path: Path) -> None:
    source_path = _write_stage148_source(tmp_path)
    before = source_path.read_bytes()

    report = freeze_primeqa_hybrid_agent_network_transport_protocol(
        stage148_validation_path=source_path,
        user_confirmed_protocol=True,
        confirmation_note="User requested the next large Stage149 protocol-freeze step.",
    )

    assert report["decision"]["status"] == "primeqa_hybrid_agent_http_transport_protocol_frozen"
    assert report["decision"]["agent_http_transport_protocol_frozen"] is True
    assert report["decision"]["local_fastapi_implementation_allowed_next"] is True
    assert report["decision"]["network_service_implemented"] is False
    assert report["decision"]["remote_deployment_authorized"] is False
    assert report["decision"]["test_gate_opened"] is False
    assert report["decision"]["queue_actions_enabled"] is False
    assert len(report["guard_checks"]) >= 35
    assert all(check["passed"] for check in report["guard_checks"])
    assert source_path.read_bytes() == before
    assert (
        report["source_files"]["stage148_validation"]["sha256"]
        == hashlib.sha256(before).hexdigest()
    )
    assert report["public_safe_contract"] == {
        "source_kind": "saved_public_stage148_aggregate_only",
        "train_split_loaded": False,
        "dev_split_loaded": False,
        "test_split_loaded": False,
        "test_metrics_run": False,
        "questions_loaded": False,
        "documents_loaded": False,
        "models_loaded": False,
        "indexes_loaded": False,
        "candidate_pools_built": False,
        "synthetic_policy_cases_only": True,
        "network_service_started": False,
        "network_port_bound": False,
        "forbidden_keys_found": [],
    }


def test_stage149_freezes_exact_routes_limits_and_status_mapping(tmp_path: Path) -> None:
    report = _freeze(tmp_path)
    protocol = report["frozen_protocol"]

    assert protocol["transport_identity"]["binding_host"] == "127.0.0.1"
    assert protocol["transport_identity"]["protocol"] == "HTTP/1.1"
    assert protocol["route_contract"]["routes"] == [
        {"route_id": "agent_answer", "method": "POST", "path": "/v1/agent/answers"},
        {"route_id": "liveness", "method": "GET", "path": "/health/live"},
        {"route_id": "readiness", "method": "GET", "path": "/health/ready"},
    ]
    request = protocol["request_contract"]
    assert request["raw_body_max_bytes"] == 32768
    assert request["request_handle_max_chars"] == 128
    assert request["title_max_chars"] == 512
    assert request["text_max_chars"] == 24576
    assert request["unknown_fields_rejected"] is True
    assert request["coercion_enabled"] is False
    errors = protocol["http_error_mapping"]
    assert {key: value["status"] for key, value in errors.items()} == {
        "malformed_json": 400,
        "request_body_too_large": 413,
        "unsupported_media_type": 415,
        "schema_validation_failed": 422,
        "facade_invalid_request": 422,
        "facade_not_active": 503,
        "capacity_exceeded": 503,
        "facade_draining": 503,
        "facade_closed": 503,
        "unexpected_downstream_error": 500,
        "cancelled_before_dispatch": None,
    }


def test_stage149_freezes_honest_disconnect_and_natural_shutdown(tmp_path: Path) -> None:
    protocol = _freeze(tmp_path)["frozen_protocol"]

    disconnect = protocol["disconnect_contract"]
    assert disconnect["predispatch_disconnect_reaches_runtime"] is False
    assert disconnect["in_flight_hard_cancellation_supported"] is False
    assert disconnect["disconnect_race_eliminated"] is False
    assert disconnect["in_flight_work_completes_or_raises_naturally"] is True
    lifespan = protocol["lifespan_contract"]
    assert lifespan["shutdown_calls_facade_shutdown"] is True
    assert lifespan["shutdown_waits_for_in_flight_naturally"] is True
    assert lifespan["implicit_shutdown_timeout_seconds"] is None
    assert lifespan["force_cancel"] is False
    assert lifespan["transport_owns_runtime_resources"] is False


def test_stage149_freezes_health_execution_and_logging_boundaries(tmp_path: Path) -> None:
    protocol = _freeze(tmp_path)["frozen_protocol"]

    health = protocol["health_contract"]
    assert health["liveness_loads_or_probes_models"] is False
    assert health["readiness_requires_facade_accepting"] is True
    assert health["readiness_nonaccepting_status"] == 503
    execution = protocol["execution_contract"]
    assert execution["sync_facade_runs_off_event_loop"] is True
    assert execution["application_owned_waiting_queue"] is False
    assert execution["request_timeout_seconds"] is None
    logging = protocol["logging_contract"]
    assert logging["default_uvicorn_access_log_enabled"] is False
    assert len(logging["allowed_fields"]) == 18
    assert logging["request_or_response_content_logged"] is False
    assert logging["headers_cookies_or_client_address_logged"] is False


def test_stage149_unconfirmed_run_is_rejected_without_opening_boundaries(tmp_path: Path) -> None:
    source_path = _write_stage148_source(tmp_path)

    report = freeze_primeqa_hybrid_agent_network_transport_protocol(
        stage148_validation_path=source_path,
        user_confirmed_protocol=False,
        confirmation_note="not confirmed",
    )

    assert report["decision"]["status"] == "primeqa_hybrid_agent_http_transport_protocol_rejected"
    assert report["decision"]["failed_checks"] == ["stage149_user_confirmed"]
    assert report["decision"]["network_service_implemented"] is False
    assert report["public_safe_contract"]["network_service_started"] is False


def test_stage149_rejects_invalid_stage148_source(tmp_path: Path) -> None:
    source_path = _write_stage148_source(tmp_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["decision"]["status"] = "fabricated_status"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    report = freeze_primeqa_hybrid_agent_network_transport_protocol(
        stage148_validation_path=source_path,
        user_confirmed_protocol=True,
        confirmation_note="confirmed",
    )

    assert report["decision"]["agent_http_transport_protocol_frozen"] is False
    assert "stage148_source_identity_valid" in report["decision"]["failed_checks"]


def test_stage149_visualizations_are_ten_parseable_svg_files(tmp_path: Path) -> None:
    report = _freeze(tmp_path)

    visualizations = write_primeqa_hybrid_agent_network_transport_protocol_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    assert len(visualizations) == 10
    assert {item.name for item in visualizations} == {
        "stage149_source_gate.svg",
        "stage149_http_surface.svg",
        "stage149_request_limits.svg",
        "stage149_status_mapping.svg",
        "stage149_disconnect_boundary.svg",
        "stage149_lifespan_boundary.svg",
        "stage149_health_semantics.svg",
        "stage149_logging_boundary.svg",
        "stage149_policy_cases.svg",
        "stage149_guard_check_status.svg",
    }
    for visualization in visualizations:
        assert ET.parse(visualization.path).getroot().tag.endswith("svg")


def test_stage149_public_report_scanner_detects_private_keys() -> None:
    assert _forbidden_keys_found({"nested": {"request_handle": "private"}}) == {"request_handle"}


def _freeze(tmp_path: Path) -> dict[str, object]:
    return freeze_primeqa_hybrid_agent_network_transport_protocol(
        stage148_validation_path=_write_stage148_source(tmp_path),
        user_confirmed_protocol=True,
        confirmation_note="User requested the next large Stage149 protocol-freeze step.",
    )


def _write_stage148_source(tmp_path: Path) -> Path:
    path = tmp_path / "stage148.json"
    source = {
        "stage": "Stage 148",
        "analysis_id": "primeqa_hybrid_transport_neutral_agent_request_facade_validation_v1",
        "guard_checks": [{"name": f"guard_{index}", "passed": True} for index in range(37)],
        "decision": {
            "status": "primeqa_hybrid_transport_neutral_agent_request_facade_validation_passed",
            "transport_neutral_facade_implemented": True,
            "facade_synthetic_validation_passed": True,
            "label_free_runtime_query_validated": True,
            "public_telemetry_allowlists_validated": True,
            "lifecycle_and_natural_shutdown_validated": True,
            "network_transport_protocol_allowed_next": True,
            "network_service_implemented": False,
            "runtime_registered_as_default": False,
            "test_gate_opened": False,
            "queue_actions_enabled": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
        },
        "implementation_contract": {"network_service_implemented": False},
        "public_safe_contract": {
            "network_service_started": False,
            "test_split_loaded": False,
            "test_metrics_run": False,
            "forbidden_keys_found": [],
        },
    }
    path.write_text(json.dumps(source), encoding="utf-8")
    return path
