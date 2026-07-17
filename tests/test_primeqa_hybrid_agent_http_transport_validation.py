from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_agent_http_transport_validation import (
    _private_keys_found,
    run_primeqa_hybrid_agent_http_transport_validation,
    write_primeqa_hybrid_agent_http_transport_validation_visualizations,
)


def test_stage150_unconfirmed_run_does_not_open_transport_boundaries(
    tmp_path: Path,
) -> None:
    report = run_primeqa_hybrid_agent_http_transport_validation(
        stage149_protocol_path=_write_stage149_source(tmp_path),
        user_confirmed_validation=False,
        confirmation_note="not confirmed",
    )

    assert report["in_process_validation_executed"] is False
    assert report["loopback_socket_validation_executed"] is False
    assert report["decision"]["failed_checks"] == ["stage150_user_confirmed"]
    assert report["decision"]["local_fastapi_transport_implemented"] is False
    assert report["public_safe_contract"]["loopback_socket_opened_temporarily"] is False


def test_stage150_rejects_stage149_source_identity_drift(tmp_path: Path) -> None:
    source_path = _write_stage149_source(tmp_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["protocol_id"] = "drifted_protocol"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    report = run_primeqa_hybrid_agent_http_transport_validation(
        stage149_protocol_path=source_path,
        user_confirmed_validation=True,
        confirmation_note="confirmed",
    )

    assert report["in_process_validation_executed"] is False
    assert report["loopback_socket_validation_executed"] is False
    assert "stage149_source_identity_valid" in report["decision"]["failed_checks"]


def test_stage150_confirmed_run_passes_asgi_and_real_loopback_socket(
    tmp_path: Path,
) -> None:
    source_path = _write_stage149_source(tmp_path)
    before = source_path.read_bytes()

    report = run_primeqa_hybrid_agent_http_transport_validation(
        stage149_protocol_path=source_path,
        user_confirmed_validation=True,
        confirmation_note="User requested the next large Stage150 implementation step.",
    )

    assert report["decision"]["status"] == (
        "primeqa_hybrid_local_fastapi_agent_transport_validation_passed"
    )
    assert report["decision"]["local_fastapi_transport_implemented"] is True
    assert report["decision"]["real_loopback_socket_validation_passed"] is True
    assert report["decision"]["network_service_persistently_running"] is False
    assert report["loopback_socket_validation"]["server_stopped"] is True
    assert report["loopback_socket_validation"]["port_rebind_succeeded"] is True
    assert len(report["guard_checks"]) >= 30
    assert all(check["passed"] for check in report["guard_checks"])
    assert source_path.read_bytes() == before


def test_stage150_visualizations_are_ten_parseable_svg_files(tmp_path: Path) -> None:
    report = run_primeqa_hybrid_agent_http_transport_validation(
        stage149_protocol_path=_write_stage149_source(tmp_path),
        user_confirmed_validation=False,
        confirmation_note="not confirmed",
    )

    visualizations = write_primeqa_hybrid_agent_http_transport_validation_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    assert len(visualizations) == 10
    assert {item.name for item in visualizations} == {
        "stage150_source_gate.svg",
        "stage150_route_surface.svg",
        "stage150_status_mapping.svg",
        "stage150_body_limit.svg",
        "stage150_overload.svg",
        "stage150_disconnect.svg",
        "stage150_shutdown.svg",
        "stage150_loopback_socket.svg",
        "stage150_closed_boundaries.svg",
        "stage150_guard_check_status.svg",
    }
    for visualization in visualizations:
        assert ET.parse(visualization.path).getroot().tag.endswith("svg")


def test_stage150_private_report_scanner_detects_private_keys() -> None:
    assert _private_keys_found({"nested": {"request_handle": "private"}}) == {"request_handle"}


def _write_stage149_source(tmp_path: Path) -> Path:
    path = tmp_path / "stage149.json"
    source = {
        "stage": "Stage 149",
        "protocol_id": "primeqa_hybrid_agent_http_transport_protocol_v1",
        "guard_checks": [{"name": f"guard_{index}", "passed": True} for index in range(39)],
        "frozen_protocol": {
            "transport_identity": {
                "binding_host": "127.0.0.1",
                "protocol": "HTTP/1.1",
            }
        },
        "decision": {
            "status": "primeqa_hybrid_agent_http_transport_protocol_frozen",
            "agent_http_transport_protocol_frozen": True,
            "local_fastapi_implementation_allowed_next": True,
            "local_loopback_only": True,
            "network_service_implemented": False,
            "runtime_registered_as_default": False,
            "remote_deployment_authorized": False,
            "test_gate_opened": False,
            "queue_actions_enabled": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
        },
        "public_safe_contract": {
            "network_service_started": False,
            "network_port_bound": False,
            "test_split_loaded": False,
            "test_metrics_run": False,
            "forbidden_keys_found": [],
        },
    }
    path.write_text(json.dumps(source), encoding="utf-8")
    return path
