from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ts_rag_agent.application.primeqa_hybrid_agent_service_entrypoint_protocol import (
    AgentServiceEntrypointProtocolState,
    StrictAgentServiceEntrypointProtocolPolicy,
    _compliant_evidence,
    _forbidden_keys_found,
    freeze_primeqa_hybrid_agent_service_entrypoint_protocol,
    write_primeqa_hybrid_agent_service_entrypoint_protocol_visualizations,
)


def test_strict_entrypoint_policy_accepts_only_exact_compliant_evidence() -> None:
    result = StrictAgentServiceEntrypointProtocolPolicy().evaluate(_compliant_evidence())

    assert result.state is AgentServiceEntrypointProtocolState.ELIGIBLE
    assert result.rejection_reasons == ()
    assert result.service_entrypoint_implemented is False


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"source_stage150_validated": False}, "stage150_source_not_validated"),
        ({"local_loopback_only": False}, "loopback_binding_not_required"),
        ({"explicit_port_required": False}, "explicit_port_not_required"),
        ({"port_zero_allowed": True}, "ephemeral_port_allowed"),
        ({"host_override_allowed": True}, "host_override_allowed"),
        (
            {"source_path_cli_overrides_allowed": True},
            "source_path_cli_override_allowed",
        ),
        ({"stage150_checked_before_other_sources": False}, "stage150_not_checked_first"),
        (
            {"split_question_rows_loaded_for_warmup": True},
            "split_question_loaded_for_warmup",
        ),
        ({"socket_bound_after_warmup": False}, "socket_bound_before_warmup"),
        ({"reload_enabled": True}, "reload_enabled"),
        ({"server_runs_on_main_thread": False}, "server_not_on_main_thread"),
        ({"custom_signal_handlers_installed": True}, "custom_signal_handler_installed"),
        ({"implicit_shutdown_timeout_enabled": True}, "implicit_shutdown_timeout_enabled"),
        ({"force_cancel_enabled": True}, "force_cancel_enabled"),
        ({"public_exception_message_enabled": True}, "public_exception_message_enabled"),
        ({"service_entrypoint_implemented": True}, "service_entrypoint_preimplemented"),
        ({"test_split_locked": False}, "test_split_not_locked"),
    ],
)
def test_strict_entrypoint_policy_rejects_boundary_drift(
    overrides: dict[str, object], expected_reason: str
) -> None:
    result = StrictAgentServiceEntrypointProtocolPolicy().evaluate(_compliant_evidence(**overrides))

    assert result.state is AgentServiceEntrypointProtocolState.REJECTED
    assert expected_reason in result.rejection_reasons


def test_strict_entrypoint_policy_rejects_bind_and_hidden_recovery_actions() -> None:
    result = StrictAgentServiceEntrypointProtocolPolicy().evaluate(
        _compliant_evidence(
            bind_retry_count=1,
            queue_action_count=1,
            retry_action_count=1,
            fallback_action_count=1,
        )
    )

    assert result.rejection_reasons == (
        "bind_retry_detected",
        "queue_action_detected",
        "retry_action_detected",
        "fallback_action_detected",
    )


def test_stage151_freeze_uses_only_saved_stage150_public_aggregate(tmp_path: Path) -> None:
    source_path = _write_stage150_source(tmp_path)
    before = source_path.read_bytes()

    report = freeze_primeqa_hybrid_agent_service_entrypoint_protocol(
        stage150_validation_path=source_path,
        user_confirmed_protocol=True,
        confirmation_note="User requested the next large Stage151 protocol-freeze step.",
    )

    assert report["decision"]["status"] == (
        "primeqa_hybrid_local_agent_service_entrypoint_protocol_frozen"
    )
    assert report["decision"]["agent_service_entrypoint_protocol_frozen"] is True
    assert report["decision"]["local_service_entrypoint_implementation_allowed_next"] is True
    assert report["decision"]["bootstrap_warmup_type_refactor_required"] is True
    assert report["decision"]["service_entrypoint_implemented"] is False
    assert report["decision"]["network_service_started"] is False
    assert report["decision"]["network_port_bound"] is False
    assert report["decision"]["test_gate_opened"] is False
    assert len(report["guard_checks"]) >= 30
    assert all(check["passed"] for check in report["guard_checks"])
    assert source_path.read_bytes() == before
    assert (
        report["source_files"]["stage150_validation"]["sha256"]
        == hashlib.sha256(before).hexdigest()
    )
    assert report["public_safe_contract"] == {
        "source_kind": "saved_public_stage150_aggregate_only",
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
        "service_entrypoint_implemented": False,
        "network_service_started": False,
        "network_port_bound": False,
        "signal_handlers_installed": False,
        "forbidden_keys_found": [],
    }


def test_stage151_freezes_exact_invocation_port_and_source_order(tmp_path: Path) -> None:
    protocol = _freeze(tmp_path)["frozen_protocol"]
    invocation = protocol["invocation_contract"]

    assert invocation["command"] == ("python -m ts_rag_agent.local_agent_service --port <PORT>")
    assert invocation["required_options"] == ["--port"]
    assert invocation["optional_options"] == []
    assert invocation["port_has_default"] is False
    assert invocation["minimum_port"] == 1024
    assert invocation["maximum_port"] == 65535
    assert invocation["port_zero_allowed"] is False
    assert invocation["host_override_allowed"] is False
    assert invocation["source_path_cli_overrides_allowed"] is False
    sources = protocol["source_authorization_contract"]
    assert sources["canonical_source_keys"] == [
        "stage150_http_transport_validation",
        "stage145_concurrent_runtime_validation",
        "stage128_agent_retrieval_protocol",
        "stage125_recall_expansion_protocol",
        "stage80_dense_sparse_report",
        "primeqa_technote_documents",
    ]
    assert sources["startup_order"][0] == "parse_exact_cli"
    assert sources["startup_order"][1] == "load_and_validate_stage150_public_report"
    assert sources["startup_order"][-1] == "release_process_references"
    assert sources["startup_order"][-4:-1] == [
        "uvicorn_stops_accepting_and_waits_http_tasks",
        "lifespan_drains_transport",
        "entrypoint_finally_confirms_listener_closed",
    ]


def test_stage151_freezes_label_free_warmup_and_resource_ownership(tmp_path: Path) -> None:
    protocol = _freeze(tmp_path)["frozen_protocol"]
    warmup = protocol["warmup_contract"]
    composition = protocol["runtime_composition_contract"]

    assert warmup["input_model"] == "PrimeQARuntimeQuery"
    assert warmup["source"] == "built_in_synthetic_label_free"
    assert warmup["warmup_request_count"] == 1
    assert warmup["split_question_rows_loaded"] is False
    assert warmup["gold_fields_present"] is False
    assert warmup["bootstrap_signature_refactor_required"] is True
    assert composition["resource_build_count"] == 1
    assert composition["transport_owns_runtime_resources"] is False
    assert composition["resource_close_interface_claimed"] is False
    assert composition["process_releases_references_after_server_return"] is True


def test_stage151_freezes_socket_signal_shutdown_and_exit_behavior(tmp_path: Path) -> None:
    protocol = _freeze(tmp_path)["frozen_protocol"]
    socket_contract = protocol["socket_contract"]
    process = protocol["process_signal_contract"]
    shutdown = protocol["shutdown_contract"]
    exits = protocol["exit_status_contract"]

    assert socket_contract["binding_host"] == "127.0.0.1"
    assert socket_contract["listener_prebound_before_server_run"] is True
    assert socket_contract["listener_bind_attempt_count"] == 1
    assert socket_contract["bind_retry_count"] == 0
    assert socket_contract["alternate_port_fallback"] is False
    assert process["process_count"] == 1
    assert process["worker_count"] == 1
    assert process["reload_enabled"] is False
    assert process["server_runs_on_main_thread"] is True
    assert process["uvicorn_owns_supported_signal_handlers"] is True
    assert process["custom_signal_handlers_installed"] is False
    assert process["signal_exit_code_normalized"] is False
    assert shutdown["uvicorn_closes_listener_before_lifespan_shutdown"] is True
    assert shutdown["uvicorn_waits_http_tasks_before_lifespan_shutdown"] is True
    assert shutdown["transport_lifespan_shutdown_after_http_tasks"] is True
    assert shutdown["transport_closed_before_process_reference_release"] is True
    assert shutdown["entrypoint_listener_close_is_idempotent_finally"] is True
    assert shutdown["implicit_shutdown_timeout_seconds"] is None
    assert shutdown["force_cancel"] is False
    assert exits["clean_server_return"] == 0
    assert exits["server_or_lifespan_failure"] == 8
    assert exits["startup_failure_retry_count"] == 0


def test_stage151_freezes_exact_content_free_startup_event(tmp_path: Path) -> None:
    observability = _freeze(tmp_path)["frozen_protocol"]["observability_contract"]

    assert observability["public_startup_event_field_count"] == 18
    assert len(observability["public_startup_event_fields"]) == 18
    assert observability["exactly_one_terminal_event"] is True
    assert observability["request_access_log_enabled"] is False
    assert observability["request_or_response_content_in_public_event"] is False
    assert observability["warmup_content_in_public_event"] is False
    assert observability["source_paths_in_public_event"] is False
    assert observability["exception_message_in_public_event"] is False
    assert observability["uvicorn_framework_error_logging_is_separate"] is True


def test_stage151_unconfirmed_run_is_rejected_without_opening_boundaries(
    tmp_path: Path,
) -> None:
    report = freeze_primeqa_hybrid_agent_service_entrypoint_protocol(
        stage150_validation_path=_write_stage150_source(tmp_path),
        user_confirmed_protocol=False,
        confirmation_note="not confirmed",
    )

    assert report["decision"]["status"] == "primeqa_hybrid_agent_service_protocol_rejected"
    assert report["decision"]["failed_checks"] == ["stage151_user_confirmed"]
    assert report["decision"]["service_entrypoint_implemented"] is False
    assert report["public_safe_contract"]["network_service_started"] is False
    assert report["public_safe_contract"]["network_port_bound"] is False


def test_stage151_rejects_invalid_stage150_source(tmp_path: Path) -> None:
    source_path = _write_stage150_source(tmp_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["decision"]["status"] = "fabricated_status"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    report = freeze_primeqa_hybrid_agent_service_entrypoint_protocol(
        stage150_validation_path=source_path,
        user_confirmed_protocol=True,
        confirmation_note="confirmed",
    )

    assert report["decision"]["agent_service_entrypoint_protocol_frozen"] is False
    assert "stage150_source_identity_valid" in report["decision"]["failed_checks"]
    assert report["public_safe_contract"]["network_port_bound"] is False


def test_stage151_visualizations_are_ten_parseable_svg_files(tmp_path: Path) -> None:
    visualizations = write_primeqa_hybrid_agent_service_entrypoint_protocol_visualizations(
        report=_freeze(tmp_path),
        output_dir=tmp_path / "visuals",
    )

    assert len(visualizations) == 10
    assert {item.name for item in visualizations} == {
        "stage151_source_gate.svg",
        "stage151_invocation_surface.svg",
        "stage151_port_policy.svg",
        "stage151_source_order.svg",
        "stage151_activation_gate.svg",
        "stage151_resource_warmup.svg",
        "stage151_socket_process.svg",
        "stage151_shutdown_signal.svg",
        "stage151_policy_cases.svg",
        "stage151_guard_check_status.svg",
    }
    for visualization in visualizations:
        assert ET.parse(visualization.path).getroot().tag.endswith("svg")


def test_stage151_public_report_scanner_detects_private_keys() -> None:
    assert _forbidden_keys_found({"nested": {"warmup_text": "private"}}) == {"warmup_text"}


def _freeze(tmp_path: Path) -> dict[str, object]:
    return freeze_primeqa_hybrid_agent_service_entrypoint_protocol(
        stage150_validation_path=_write_stage150_source(tmp_path),
        user_confirmed_protocol=True,
        confirmation_note="User requested the next large Stage151 protocol-freeze step.",
    )


def _write_stage150_source(tmp_path: Path) -> Path:
    path = tmp_path / "stage150.json"
    source = {
        "stage": "Stage 150",
        "analysis_id": "primeqa_hybrid_local_fastapi_agent_transport_validation_v1",
        "guard_checks": [{"name": f"guard_{index}", "passed": True} for index in range(37)],
        "implementation_contract": {
            "binding_host": "127.0.0.1",
            "http_protocol": "HTTP/1.1",
            "max_in_flight": 4,
            "application_waiting_queue": False,
            "request_timeout_seconds": None,
            "implicit_shutdown_timeout_seconds": None,
        },
        "decision": {
            "status": "primeqa_hybrid_local_fastapi_agent_transport_validation_passed",
            "local_fastapi_transport_implemented": True,
            "in_process_asgi_validation_passed": True,
            "real_loopback_socket_validation_passed": True,
            "disabled_by_default": True,
            "local_loopback_only": True,
            "network_service_persistently_running": False,
            "runtime_registered_as_default": False,
            "remote_exposure_authorized": False,
            "test_gate_opened": False,
            "queue_actions_enabled": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
        },
        "public_safe_contract": {
            "network_service_persistently_running": False,
            "test_split_loaded": False,
            "test_metrics_run": False,
            "private_keys_found": [],
        },
    }
    path.write_text(json.dumps(source), encoding="utf-8")
    return path
