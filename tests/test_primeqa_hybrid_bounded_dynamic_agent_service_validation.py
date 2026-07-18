from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_http_transport import (
    bounded_dynamic_agent_http_transport_contract,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_service_entrypoint import (
    bounded_dynamic_agent_service_entrypoint_contract,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_service_validation import (
    _guard_checks,
    _synthetic_service_cases,
    write_stage158_visualizations,
)


def test_synthetic_service_cases_cover_strict_lifecycle_and_admission() -> None:
    cases = _synthetic_service_cases()

    assert set(cases) == {
        "explicit_open_close",
        "duplicate_open_rejected",
        "missing_thread_rejected",
        "second_thread_capacity_rejected",
        "same_thread_parallel_rejected",
        "close_while_busy_rejected",
        "shutdown_clears_threads",
    }
    assert all(case["passed"] for case in cases.values())


def test_stage158_guard_set_accepts_exact_synthetic_formal_shape() -> None:
    report = _synthetic_formal_report()

    checks = _guard_checks(report)

    assert len(checks) >= 50
    assert all(check["passed"] for check in checks)


def test_stage158_visualizations_are_ten_parseable_svgs(tmp_path: Path) -> None:
    report = _synthetic_formal_report()
    report["guard_checks"] = _guard_checks(report)

    visualizations = write_stage158_visualizations(report=report, output_dir=tmp_path)

    assert len(visualizations) == 10
    for visualization in visualizations:
        path = Path(visualization.path)
        assert path.stat().st_size > 100
        assert ET.parse(path).getroot().tag.endswith("svg")


def _synthetic_formal_report() -> dict:
    event = {
        "route_id": "thread_turn",
        "method": "POST",
        "http_status": 200,
        "outcome_code": "refuse",
        "coordinator_state": "accepting",
        "downstream_dispatched": True,
        "gpu_admitted": True,
        "current_in_flight_turns": 0,
        "max_observed_in_flight_turns": 1,
        "opened_thread_count": 1,
        "completed_turn_count": 1,
        "retained_state_bytes": 128,
        "terminal_state": "refuse",
        "selected_action": "refuse_insufficient_evidence",
        "model_decision_count": 1,
        "retrieval_call_count": 1,
        "composition_call_count": 0,
        "verification_call_count": 0,
        "diagnostic_observation_count": 0,
        "router_input_token_count": 2000,
        "router_output_token_count": 10,
        "router_generation_latency_ms": 1000.0,
        "queue_action_count": 0,
        "retry_action_count": 0,
        "fallback_action_count": 0,
    }
    boundaries = {
        "train_split_loaded": False,
        "dev_split_loaded": False,
        "test_split_loaded": False,
        "test_metrics_run": False,
        "gold_labels_read": False,
        "existing_answer_route_changed": False,
        "runtime_registered_as_default": False,
        "remote_exposure_authorized": False,
        "persistent_state_enabled": False,
        "implicit_thread_creation_enabled": False,
        "query_rewrite_enabled": False,
        "second_retrieval_enabled": False,
        "queue_action_count": 0,
        "retry_action_count": 0,
        "fallback_action_count": 0,
        "raw_question_saved": False,
        "raw_answer_saved": False,
        "raw_document_saved": False,
        "raw_model_output_saved": False,
    }
    counters = {
        "opened_thread_count": 0,
        "current_in_flight_turns": 0,
        "max_observed_in_flight_turns": 1,
        "admitted_turn_count": 1,
        "capacity_rejected_turn_count": 0,
        "completed_turn_count": 1,
        "failed_turn_count": 0,
        "queue_action_count": 0,
        "retry_action_count": 0,
        "fallback_action_count": 0,
    }
    return {
        "user_confirmation": {"protocol_a_confirmed": True},
        "stage157_authorization": {
            "artifact_identity_exact": True,
            "source_unchanged": True,
        },
        "source_unchanged_after_validation": True,
        "environment": {
            "torch_version": "2.11.0+cu128",
            "cuda_available": True,
            "cuda_version": "12.8",
            "gpu_capability": [12, 0],
        },
        "entrypoint_contract": bounded_dynamic_agent_service_entrypoint_contract(),
        "transport_contract": bounded_dynamic_agent_http_transport_contract(),
        "synthetic_service_cases": _synthetic_service_cases(),
        "real_service": {
            "server_started": True,
            "server_thread_alive_after_shutdown": False,
            "port_rebind_after_shutdown": True,
            "http_status": {
                "live": 200,
                "ready": 200,
                "open": 201,
                "turn": 200,
                "close": 200,
            },
            "health_state": {"live": "live", "ready": "ready"},
            "open_summary": {
                "opened": True,
                "completed_turn_count": 0,
                "retained_state_bytes": 0,
            },
            "turn_summary": {
                "refused": True,
                "citation_count": 0,
                "terminal_state": "refuse",
                "completed_turn_count": 1,
                "retained_state_bytes": 128,
            },
            "close_summary": {
                "opened": False,
                "completed_turn_count": 1,
                "retained_state_bytes": 128,
            },
            "successful_turn_public_event": event,
            "public_event_count": 5,
            "public_event_route_ids": [
                "liveness",
                "readiness",
                "thread_open",
                "thread_turn",
                "thread_close",
            ],
            "coordinator_counters_after_shutdown": counters,
        },
        "startup": {
            "resource_factory_build_count": 1,
            "retrieval_encoder_device": "cpu",
            "warmup": {
                "thread_opened_after_close": False,
                "retrieval_call_count": 1,
                "model_decision_count": 1,
                "composition_call_count": 0,
                "verification_call_count": 0,
                "diagnostic_observation_count": 0,
            },
            "model_generation_call_count": 2,
            "peak_gpu_memory_bytes": 5_000_000_000,
            "timing_seconds": {
                "source_authorization": 1.0,
                "retrieval_resource_build": 20.0,
                "model_load": 12.0,
                "warmup": 2.0,
                "app_composition": 0.01,
                "total_prepare": 35.01,
            },
        },
        "closed_boundaries": boundaries,
        "timing_seconds": {
            "source_and_synthetic": 1.0,
            "service_prepare": 35.0,
            "server_start": 0.1,
            "real_http_sequence_and_shutdown": 2.0,
            "final_audit": 0.1,
            "total": 38.2,
        },
    }
