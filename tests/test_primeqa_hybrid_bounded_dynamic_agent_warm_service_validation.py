from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application import (
    primeqa_hybrid_bounded_dynamic_agent_warm_service_validation as validation,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_warm_service_validation import (
    _authorize_stage158_artifact,
    _guard_checks,
    write_stage159_visualizations,
)


def test_stage158_authorization_requires_exact_artifact_and_eight_source_fingerprints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    sources = _create_stage158_sources(project_root)
    report = {
        "stage": "Stage 158",
        "guard_checks": [{"name": f"guard-{index}", "passed": True} for index in range(51)],
        "decision": {
            "status": (
                "primeqa_hybrid_bounded_dynamic_agent_local_service_implemented_and_validated"
            ),
            "all_guards_passed": True,
            "runtime_registered_as_default": False,
            "test_gate_opened": False,
            "test_metrics_run": False,
        },
        "source_files": {key: _fingerprint(path) for key, path in sources.items()},
    }
    artifact = tmp_path / "stage158.json"
    artifact.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        validation,
        "STAGE159_EXPECTED_STAGE158_ARTIFACT_SHA256",
        _sha256(artifact),
    )

    authorized = _authorize_stage158_artifact(
        artifact_path=artifact,
        project_root=project_root,
    )

    assert authorized["artifact_identity_exact"] is True
    assert authorized["guard_count"] == 51
    assert authorized["source_fingerprint_match_count"] == 8

    sources["transport"].write_text("changed", encoding="utf-8")
    try:
        _authorize_stage158_artifact(
            artifact_path=artifact,
            project_root=project_root,
        )
    except RuntimeError as error:
        assert "transport" in str(error)
    else:
        raise AssertionError("changed Stage158 source was accepted")


def test_stage159_guard_set_accepts_exact_full_dev_and_capacity_shape() -> None:
    checks = _guard_checks(_synthetic_report())

    assert len(checks) >= 65
    assert all(check["passed"] for check in checks)


def test_stage159_guard_set_rejects_test_access_and_capacity_queueing() -> None:
    report = _synthetic_report()
    report["closed_boundaries"]["test_split_loaded"] = True
    report["closed_boundaries"]["queue_action_count"] = 1
    report["real_service"]["capacity_probe"]["second_request_http_status"] = 200

    failed = {check["name"] for check in _guard_checks(report) if not check["passed"]}

    assert "dev_only_split_boundary" in failed
    assert "no_queue_retry_fallback" in failed
    assert "capacity_second_request_rejected" in failed


def test_stage159_visualizations_are_ten_parseable_svgs(tmp_path: Path) -> None:
    report = _synthetic_report()
    report["guard_checks"] = _guard_checks(report)

    visualizations = write_stage159_visualizations(report=report, output_dir=tmp_path)

    assert len(visualizations) == 10
    for visualization in visualizations:
        path = Path(visualization.path)
        assert path.stat().st_size > 100
        assert ET.parse(path).getroot().tag.endswith("svg")


def _synthetic_report() -> dict:
    by_position = {
        str(position): {
            "turn_count": 31 if position == 1 else 30,
            "end_to_end_latency_ms": _distribution(1500 + position),
            "router_generation_latency_ms": _distribution(1000 + position),
            "router_input_token_count": _distribution(2000 + position * 100),
            "router_output_token_count": _distribution(10),
            "retained_state_bytes": _distribution(position * 200),
        }
        for position in range(1, 5)
    }
    counters = {
        "opened_thread_count": 0,
        "current_in_flight_turns": 0,
        "max_observed_in_flight_turns": 1,
        "admitted_turn_count": 122,
        "capacity_rejected_turn_count": 1,
        "completed_turn_count": 122,
        "failed_turn_count": 0,
        "queue_action_count": 0,
        "retry_action_count": 0,
        "fallback_action_count": 0,
    }
    return {
        "user_confirmation": {
            "full_dev_protocol_confirmed": True,
            "synthetic_conversation_grouping_disclosed": True,
        },
        "stage158_authorization": {
            "artifact_identity_exact": True,
            "guard_count": 51,
            "all_guards_passed": True,
            "source_fingerprint_match_count": 8,
            "runtime_registered_as_default": False,
            "test_gate_opened": False,
        },
        "source_unchanged_after_validation": True,
        "stage158_artifact_unchanged_after_validation": True,
        "dev_source_unchanged_after_validation": True,
        "dev_query_protocol": {
            "source_sha256": validation.STAGE159_EXPECTED_DEV_SHA256,
            "dev_query_count": 121,
            "assigned_split": "dev",
            "label_fields_used_for_selection": False,
            "label_fields_projected_into_runtime": False,
            "label_fields_used_for_metrics": False,
        },
        "workload_plan": {
            "thread_count": 31,
            "full_four_turn_thread_count": 30,
            "trailing_thread_turn_count": 1,
            "turn_count": 121,
            "turn_position_counts": {"1": 31, "2": 30, "3": 30, "4": 30},
        },
        "startup": {
            "resource_factory_build_count": 1,
            "model_generation_call_count": 123,
            "peak_gpu_memory_bytes": 5_000_000_000,
            "warmup": {"thread_opened_after_close": False},
            "timing_seconds": {
                "source_authorization": 3.0,
                "retrieval_resource_build": 40.0,
                "model_load": 8.0,
                "warmup": 2.0,
                "app_composition": 0.01,
            },
        },
        "real_service": {
            "server_started": True,
            "server_thread_alive_after_shutdown": False,
            "port_rebind_after_shutdown": True,
            "health_status": {"live": 200, "ready": 200},
            "health_state": {"live": "live", "ready": "ready"},
            "dev_http": {
                "open_request_count": 31,
                "turn_request_count": 121,
                "close_request_count": 31,
                "open_http_status_counts": {"201": 31},
                "turn_http_status_counts": {"200": 121},
                "close_http_status_counts": {"200": 31},
            },
            "dev_turn_summary": {
                "turn_count": 121,
                "thread_count": 31,
                "selected_action_counts": {
                    "compose_grounded_answer": 80,
                    "refuse_insufficient_evidence": 41,
                },
                "answer_count": 75,
                "refusal_count": 46,
                "retrieval_call_count": 121,
                "model_decision_count": 121,
                "composition_call_count": 80,
                "verification_call_count": 80,
                "diagnostic_observation_count": 80,
                "branch_protocol_valid_count": 121,
                "state_growth_monotonic_thread_count": 31,
                "by_turn_position": by_position,
            },
            "capacity_probe": {
                "observation_gate_wait_timeout_seconds": None,
                "first_thread_open_http_status": 201,
                "second_thread_open_http_status": 201,
                "first_request_http_status": 200,
                "first_request_completed_turn_count": 1,
                "second_request_http_status": 503,
                "second_request_error_code": "gpu_capacity_exceeded",
                "second_request_downstream_dispatched": False,
                "second_request_gpu_admitted": False,
                "first_thread_close_http_status": 200,
                "second_thread_close_http_status": 200,
                "first_thread_opened_after_close": False,
                "second_thread_opened_after_close": False,
                "capacity_rejected_turn_count": 1,
                "max_observed_in_flight_turns": 1,
            },
            "coordinator_counters_after_shutdown": counters,
        },
        "closed_boundaries": {
            "train_split_loaded": False,
            "dev_split_loaded": True,
            "test_split_loaded": False,
            "test_metrics_run": False,
            "dev_gold_quality_metrics_run": False,
            "queue_action_count": 0,
            "retry_action_count": 0,
            "fallback_action_count": 0,
            "runtime_registered_as_default": False,
            "remote_exposure_authorized": False,
            "persistent_state_enabled": False,
            "query_rewrite_enabled": False,
            "second_retrieval_enabled": False,
            "raw_question_saved": False,
            "raw_answer_saved": False,
            "raw_document_saved": False,
            "raw_model_output_saved": False,
            "private_sample_identity_saved": False,
        },
    }


def _distribution(value: float) -> dict:
    return {
        "count": 30,
        "min": value,
        "median": value,
        "p95": value,
        "max": value,
        "average": value,
    }


def _create_stage158_sources(project_root: Path) -> dict[str, Path]:
    paths = validation._stage158_source_paths(project_root)
    for index, path in enumerate(paths.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source-{index}", encoding="utf-8")
    return paths


def _fingerprint(path: Path) -> dict:
    return {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
