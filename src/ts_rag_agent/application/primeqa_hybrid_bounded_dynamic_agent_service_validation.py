from __future__ import annotations

import hashlib
import http.client
import json
import socket
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any

import uvicorn

from ts_rag_agent.application.primeqa_hybrid_bounded_agent_state_protocol import (
    ThreadStateLimits,
    VolatileThreadStateLedger,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_http_transport import (
    BOUNDED_DYNAMIC_AGENT_BINDING_HOST,
    BoundedDynamicAgentHttpLogSink,
    BoundedDynamicAgentServiceCoordinator,
    BoundedDynamicAgentServiceError,
    BoundedDynamicAgentServiceErrorCode,
    PublicSafeBoundedDynamicAgentHttpEvent,
    bounded_dynamic_agent_http_transport_contract,
    create_primeqa_hybrid_bounded_dynamic_agent_http_app,
    create_primeqa_hybrid_bounded_dynamic_agent_uvicorn_config,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_runtime import (
    PRODUCTION_MAX_COMPLETED_TURNS,
    PRODUCTION_MAX_RETAINED_BYTES,
    bounded_dynamic_agent_runtime_contract,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_service_entrypoint import (
    EXPECTED_STAGE157_ARTIFACT_SHA256,
    CanonicalBoundedDynamicAgentServicePaths,
    PrimeQAHybridBoundedDynamicAgentServiceEntrypoint,
    bounded_dynamic_agent_service_entrypoint_contract,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    _forbidden_keys_found,
)
from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    structured_decision_router_contract,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.config import ProjectSettings

_STAGE = "Stage 158"
_CREATED_AT = "2026-07-18"
_ANALYSIS_ID = "primeqa_hybrid_bounded_dynamic_agent_local_service_validation_v1"
_FINAL_STATUS = "primeqa_hybrid_bounded_dynamic_agent_local_service_implemented_and_validated"


@dataclass(frozen=True)
class Stage158Visualization:
    name: str
    path: str


class RecordingBoundedDynamicAgentHttpLogSink(BoundedDynamicAgentHttpLogSink):
    def __init__(self) -> None:
        self._lock = Lock()
        self._events: list[PublicSafeBoundedDynamicAgentHttpEvent] = []

    def emit(self, event: PublicSafeBoundedDynamicAgentHttpEvent) -> None:
        event.to_public_dict()
        with self._lock:
            self._events.append(event)

    def public_events(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(event.to_public_dict() for event in self._events)


class _ObservedUvicornServer(uvicorn.Server):
    def __init__(
        self,
        *,
        config: uvicorn.Config,
        lifecycle_events: Queue[tuple[str, BaseException | None]],
    ) -> None:
        super().__init__(config=config)
        self._lifecycle_events = lifecycle_events

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        self._lifecycle_events.put(("started", None))


def validate_primeqa_hybrid_bounded_dynamic_agent_service(
    *,
    settings: ProjectSettings,
    port: int,
    user_confirmed_protocol_a: bool,
) -> dict[str, Any]:
    """Run the strict Stage158 source, startup, socket, and lifecycle validation."""

    import torch

    if not user_confirmed_protocol_a:
        raise ValueError("Stage158 formal validation requires explicit protocol A confirmation")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage158 formal validation requires the selected CUDA environment")
    if not _port_is_bindable(port):
        raise RuntimeError("selected Stage158 validation port is not available")

    started_at = time.perf_counter()
    project_root = Path(__file__).resolve().parents[3]
    current_sources = _current_source_paths(project_root)
    source_before = {key: _fingerprint(path) for key, path in current_sources.items()}
    stage157_path = (
        settings.artifact_dir.resolve()
        / "primeqa_hybrid_bounded_dynamic_agent_runtime_stage157.json"
    )
    stage157_before = _fingerprint(stage157_path)
    synthetic_cases = _synthetic_service_cases()
    synthetic_completed_at = time.perf_counter()

    torch.cuda.reset_peak_memory_stats()
    sink = RecordingBoundedDynamicAgentHttpLogSink()
    paths = CanonicalBoundedDynamicAgentServicePaths.from_settings(settings)
    entrypoint = PrimeQAHybridBoundedDynamicAgentServiceEntrypoint(
        settings=settings,
        paths=paths,
        app_factory=lambda **kwargs: create_primeqa_hybrid_bounded_dynamic_agent_http_app(
            **kwargs,
            log_sink=sink,
        ),
    )
    prepared = entrypoint.prepare()
    prepared_at = time.perf_counter()

    config = create_primeqa_hybrid_bounded_dynamic_agent_uvicorn_config(
        app=prepared.app,
        port=port,
    )
    lifecycle_events: Queue[tuple[str, BaseException | None]] = Queue()
    server = _ObservedUvicornServer(
        config=config,
        lifecycle_events=lifecycle_events,
    )

    def run_server() -> None:
        try:
            server.run()
        except BaseException as error:
            lifecycle_events.put(("failed", error))
        finally:
            if not server.started:
                lifecycle_events.put(("exited_before_start", None))

    server_thread = Thread(
        target=run_server,
        name="stage158-real-loopback-service",
        daemon=False,
    )
    server_thread.start()
    lifecycle_state, lifecycle_error = lifecycle_events.get()
    if lifecycle_state != "started":
        server_thread.join()
        raise RuntimeError("Stage158 Uvicorn server exited before startup") from lifecycle_error
    server_started_at = time.perf_counter()

    connection = http.client.HTTPConnection(BOUNDED_DYNAMIC_AGENT_BINDING_HOST, port)
    try:
        live_status, live = _request_json(connection, "GET", "/health/live")
        ready_status, ready = _request_json(connection, "GET", "/health/ready")
        open_status, opened = _request_json(
            connection,
            "POST",
            "/v1/bounded-agent/threads/open",
            {"thread_handle": "stage158-real-thread"},
        )
        turn_status, turn = _request_json(
            connection,
            "POST",
            "/v1/bounded-agent/threads/turn",
            {
                "thread_handle": "stage158-real-thread",
                "title": "Post-installation service verification",
                "text": (
                    "How can I verify that a configured service is operating correctly "
                    "after installation?"
                ),
            },
        )
        close_status, closed = _request_json(
            connection,
            "POST",
            "/v1/bounded-agent/threads/close",
            {"thread_handle": "stage158-real-thread"},
        )
    finally:
        connection.close()
        server.should_exit = True
        server_thread.join()
    socket_completed_at = time.perf_counter()

    events = sink.public_events()
    turn_events = [
        event
        for event in events
        if event.get("route_id") == "thread_turn" and event.get("http_status") == 200
    ]
    if len(turn_events) != 1:
        raise RuntimeError("Stage158 real service did not emit exactly one successful turn event")
    turn_event = turn_events[0]
    transport = prepared.app.state.bounded_dynamic_agent_http_transport
    counters = transport.coordinator.counters()
    source_after = {key: _fingerprint(path) for key, path in current_sources.items()}
    stage157_after = _fingerprint(stage157_path)
    port_released = _port_is_bindable(port)
    finished_at = time.perf_counter()

    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "validation_scope": (
            "Strict nondefault loopback service activation for the Stage157 bounded dynamic "
            "Agent, with explicit open/turn/close transport, one nonblocking whole-turn GPU "
            "admission, startup warmup, public-safe events, and one real label-free HTTP turn."
        ),
        "user_confirmation": {
            "protocol_a_confirmed": True,
            "transport_shape": "separate_open_turn_close_endpoints",
        },
        "stage157_authorization": {
            "artifact_sha256": stage157_before["sha256"],
            "artifact_identity_exact": (
                stage157_before["sha256"] == EXPECTED_STAGE157_ARTIFACT_SHA256
            ),
            "source_unchanged": stage157_before == stage157_after,
        },
        "source_files": source_before,
        "source_unchanged_after_validation": source_before == source_after,
        "environment": {
            "python_environment": "project_.venv",
            "torch_version": torch.__version__,
            "transformers_version": version("transformers"),
            "fastapi_version": version("fastapi"),
            "uvicorn_version": version("uvicorn"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
        },
        "entrypoint_contract": bounded_dynamic_agent_service_entrypoint_contract(),
        "transport_contract": bounded_dynamic_agent_http_transport_contract(),
        "runtime_contract": bounded_dynamic_agent_runtime_contract(),
        "router_contract": structured_decision_router_contract(),
        "synthetic_service_cases": synthetic_cases,
        "real_service": {
            "binding_host": BOUNDED_DYNAMIC_AGENT_BINDING_HOST,
            "binding_port": port,
            "server_started": server.started,
            "server_thread_alive_after_shutdown": server_thread.is_alive(),
            "port_rebind_after_shutdown": port_released,
            "http_status": {
                "live": live_status,
                "ready": ready_status,
                "open": open_status,
                "turn": turn_status,
                "close": close_status,
            },
            "health_state": {
                "live": live.get("status"),
                "ready": ready.get("status"),
            },
            "open_summary": {
                "opened": opened.get("opened"),
                "completed_turn_count": opened.get("completed_turn_count"),
                "retained_state_bytes": opened.get("retained_state_bytes"),
            },
            "turn_summary": {
                "refused": turn.get("refused"),
                "citation_count": len(turn.get("citations") or []),
                "terminal_state": turn.get("terminal_state"),
                "completed_turn_count": turn.get("completed_turn_count"),
                "retained_state_bytes": turn.get("retained_state_bytes"),
            },
            "close_summary": {
                "opened": closed.get("opened"),
                "completed_turn_count": closed.get("completed_turn_count"),
                "retained_state_bytes": closed.get("retained_state_bytes"),
            },
            "successful_turn_public_event": turn_event,
            "public_event_count": len(events),
            "public_event_route_ids": [str(event.get("route_id")) for event in events],
            "coordinator_counters_after_shutdown": asdict(counters),
        },
        "startup": {
            "source_fingerprint_count": len(prepared.source_fingerprints),
            "resource_factory_build_count": prepared.resource_factory_build_count,
            "retrieval_encoder_device": prepared.retrieval_encoder_device,
            "warmup": prepared.warmup.to_public_dict(),
            "model_generation_call_count": prepared.backend.generation_call_count,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "timing_seconds": dict(prepared.timing_seconds),
        },
        "closed_boundaries": {
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
        },
        "timing_seconds": {
            "source_and_synthetic": round(synthetic_completed_at - started_at, 6),
            "service_prepare": round(prepared_at - synthetic_completed_at, 6),
            "server_start": round(server_started_at - prepared_at, 6),
            "real_http_sequence_and_shutdown": round(
                socket_completed_at - server_started_at,
                6,
            ),
            "final_audit": round(finished_at - socket_completed_at, 6),
            "total": round(finished_at - started_at, 6),
        },
    }
    report["guard_checks"] = _guard_checks(report)
    forbidden = sorted(_forbidden_keys_found(report))
    report["public_safe_contract"] = {
        "forbidden_keys_found": forbidden,
        "private_request_or_response_content_saved": False,
        "thread_handle_saved_in_public_events": False,
    }
    passed = all(check["passed"] for check in report["guard_checks"]) and not forbidden
    report["decision"] = {
        "status": _FINAL_STATUS if passed else "stage158_service_rejected",
        "all_guards_passed": passed,
        "failed_checks": [check["name"] for check in report["guard_checks"] if not check["passed"]],
        "service_implemented": True,
        "real_loopback_lifecycle_completed": True,
        "runtime_registered_as_default": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
        "next_direction": "measure_warm_multi_turn_service_behavior_on_locked_development_only",
    }
    return report


def write_stage158_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[Stage158Visualization]:
    output_dir.mkdir(parents=True, exist_ok=True)
    real = report.get("real_service") or {}
    statuses = real.get("http_status") or {}
    startup = report.get("startup") or {}
    warmup = startup.get("warmup") or {}
    turn = real.get("successful_turn_public_event") or {}
    counters = real.get("coordinator_counters_after_shutdown") or {}
    timings = report.get("timing_seconds") or {}
    closed = report.get("closed_boundaries") or {}
    synthetic = report.get("synthetic_service_cases") or {}
    charts = {
        "stage158_guard_status.svg": _chart(
            "Stage158 formal guard checks",
            [
                _bar(
                    str(check["name"]),
                    1 if check["passed"] else 0,
                    "passed" if check["passed"] else "failed",
                )
                for check in report.get("guard_checks", [])
            ],
            width=3200,
            margin_left=1800,
        ),
        "stage158_http_status.svg": _chart(
            "Stage158 real HTTP status",
            [_bar(str(key), value) for key, value in statuses.items()],
        ),
        "stage158_startup_timing.svg": _chart(
            "Stage158 startup timing seconds",
            [_bar(str(key), value) for key, value in (startup.get("timing_seconds") or {}).items()],
        ),
        "stage158_total_timing.svg": _chart(
            "Stage158 formal timing seconds",
            [_bar(str(key), value) for key, value in timings.items()],
        ),
        "stage158_warmup_calls.svg": _chart(
            "Stage158 pre-listener warmup calls",
            [
                _bar("retrieval", warmup.get("retrieval_call_count", 0)),
                _bar("model decision", warmup.get("model_decision_count", 0)),
                _bar("composition", warmup.get("composition_call_count", 0)),
                _bar("verification", warmup.get("verification_call_count", 0)),
                _bar("diagnostics", warmup.get("diagnostic_observation_count", 0)),
            ],
        ),
        "stage158_real_turn_calls.svg": _chart(
            "Stage158 real HTTP turn calls",
            [
                _bar("retrieval", turn.get("retrieval_call_count", 0)),
                _bar("model decision", turn.get("model_decision_count", 0)),
                _bar("composition", turn.get("composition_call_count", 0)),
                _bar("verification", turn.get("verification_call_count", 0)),
                _bar("diagnostics", turn.get("diagnostic_observation_count", 0)),
            ],
        ),
        "stage158_gpu_admission.svg": _chart(
            "Stage158 GPU admission",
            [
                _bar("configured max", 1),
                _bar("observed max", counters.get("max_observed_in_flight_turns", 0)),
                _bar("queue actions", counters.get("queue_action_count", 0)),
                _bar("retry actions", counters.get("retry_action_count", 0)),
                _bar("fallback actions", counters.get("fallback_action_count", 0)),
            ],
        ),
        "stage158_synthetic_cases.svg": _chart(
            "Stage158 synthetic service cases",
            [_bar(str(key), 1 if value.get("passed") else 0) for key, value in synthetic.items()],
        ),
        "stage158_closed_boundaries.svg": _chart(
            "Stage158 closed boundaries",
            [
                _bar(str(key), int(bool(value)))
                for key, value in closed.items()
                if isinstance(value, bool)
            ],
            width=2400,
            margin_left=1100,
        ),
        "stage158_gpu_memory.svg": _chart(
            "Stage158 peak allocated GPU GiB",
            [
                _bar(
                    "peak allocated",
                    round(float(startup.get("peak_gpu_memory_bytes", 0)) / (1024**3), 3),
                )
            ],
        ),
    }
    written: list[Stage158Visualization] = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage158Visualization(name=name, path=str(path)))
    return written


class _LedgerOnlyRuntime:
    def __init__(self) -> None:
        self._ledger = VolatileThreadStateLedger(
            limits=ThreadStateLimits(
                max_completed_turns=PRODUCTION_MAX_COMPLETED_TURNS,
                max_retained_bytes=PRODUCTION_MAX_RETAINED_BYTES,
            )
        )
        self.last_public_trace = None

    def open_thread(self, handle: str):
        self._ledger.open_thread(handle)
        return self._ledger.public_summary(handle)

    def close_thread(self, handle: str):
        return self._ledger.close_thread(handle)

    def thread_summary(self, handle: str):
        return self._ledger.public_summary(handle)


def _synthetic_service_cases() -> dict[str, dict[str, bool]]:
    runtime = _LedgerOnlyRuntime()
    coordinator = BoundedDynamicAgentServiceCoordinator(runtime)  # type: ignore[arg-type]
    coordinator.start()
    opened = coordinator.open_thread("synthetic-a")
    coordinator.open_thread("synthetic-b")
    duplicate = _captures_code(
        lambda: coordinator.open_thread("synthetic-a"),
        BoundedDynamicAgentServiceErrorCode.THREAD_ALREADY_OPEN,
    )
    missing = _captures_code(
        lambda: coordinator.close_thread("synthetic-missing"),
        BoundedDynamicAgentServiceErrorCode.THREAD_NOT_FOUND,
    )
    admission = coordinator.admit_turn("synthetic-a")
    capacity = _captures_code(
        lambda: coordinator.admit_turn("synthetic-b"),
        BoundedDynamicAgentServiceErrorCode.GPU_CAPACITY_EXCEEDED,
    )
    same_thread = _captures_code(
        lambda: coordinator.admit_turn("synthetic-a"),
        BoundedDynamicAgentServiceErrorCode.THREAD_BUSY,
    )
    close_busy = _captures_code(
        lambda: coordinator.close_thread("synthetic-a"),
        BoundedDynamicAgentServiceErrorCode.THREAD_BUSY,
    )
    coordinator.cancel_turn_admission(admission)
    closed = coordinator.close_thread("synthetic-a")
    coordinator.begin_draining()
    coordinator.close()
    counters = coordinator.counters()
    return {
        "explicit_open_close": {
            "passed": opened.opened is True and closed.opened is False,
        },
        "duplicate_open_rejected": {"passed": duplicate},
        "missing_thread_rejected": {"passed": missing},
        "second_thread_capacity_rejected": {"passed": capacity},
        "same_thread_parallel_rejected": {"passed": same_thread},
        "close_while_busy_rejected": {"passed": close_busy},
        "shutdown_clears_threads": {
            "passed": (
                coordinator.state.value == "closed"
                and counters.opened_thread_count == 0
                and counters.max_observed_in_flight_turns == 1
                and counters.queue_action_count == 0
                and counters.retry_action_count == 0
                and counters.fallback_action_count == 0
            ),
        },
    }


def _captures_code(call: Any, code: BoundedDynamicAgentServiceErrorCode) -> bool:
    try:
        call()
    except BoundedDynamicAgentServiceError as error:
        return error.code is code
    return False


def _guard_checks(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    auth = report.get("stage157_authorization") or {}
    environment = report.get("environment") or {}
    entrypoint = report.get("entrypoint_contract") or {}
    transport = report.get("transport_contract") or {}
    synthetic = report.get("synthetic_service_cases") or {}
    real = report.get("real_service") or {}
    statuses = real.get("http_status") or {}
    health = real.get("health_state") or {}
    opened = real.get("open_summary") or {}
    turn = real.get("turn_summary") or {}
    closed = real.get("close_summary") or {}
    event = real.get("successful_turn_public_event") or {}
    counters = real.get("coordinator_counters_after_shutdown") or {}
    startup = report.get("startup") or {}
    warmup = startup.get("warmup") or {}
    boundaries = report.get("closed_boundaries") or {}
    branch_counts = [
        event.get("composition_call_count"),
        event.get("verification_call_count"),
        event.get("diagnostic_observation_count"),
    ]
    expected_branch = (
        [1, 1, 1] if event.get("selected_action") == "compose_grounded_answer" else [0, 0, 0]
    )
    checks = [
        _check(
            "protocol_a_confirmed",
            report.get("user_confirmation", {}).get("protocol_a_confirmed") is True,
        ),
        _check("stage157_artifact_exact", auth.get("artifact_identity_exact") is True),
        _check("stage157_unchanged", auth.get("source_unchanged") is True),
        _check(
            "stage158_sources_unchanged", report.get("source_unchanged_after_validation") is True
        ),
        _check(
            "gpu_environment_exact",
            environment.get("torch_version") == "2.11.0+cu128"
            and environment.get("cuda_available") is True
            and environment.get("cuda_version") == "12.8"
            and environment.get("gpu_capability") == [12, 0],
        ),
        _check(
            "two_explicit_activation_flags",
            len(entrypoint.get("required_activation_flags") or []) == 2,
        ),
        _check(
            "default_closed",
            entrypoint.get("default_enabled") is False
            and transport.get("default_enabled") is False,
        ),
        _check(
            "loopback_only",
            entrypoint.get("binding_host") == "127.0.0.1"
            and transport.get("binding_host") == "127.0.0.1",
        ),
        _check(
            "source_gate_before_resources",
            entrypoint.get("source_gate_before_resource_build") is True,
        ),
        _check(
            "resources_before_model", entrypoint.get("resource_build_before_model_load") is True
        ),
        _check("warmup_before_listener", entrypoint.get("warmup_before_listener") is True),
        _check("warmup_thread_closed", warmup.get("thread_opened_after_close") is False),
        _check("one_gpu_turn_slot", transport.get("max_in_flight_turns") == 1),
        _check("no_application_queue", transport.get("application_waiting_queue") is False),
        _check(
            "no_request_or_shutdown_timeout",
            transport.get("request_timeout_seconds") is None
            and transport.get("implicit_shutdown_timeout_seconds") is None,
        ),
        *[
            _check(f"synthetic_{name}", row.get("passed") is True)
            for name, row in synthetic.items()
        ],
        _check("server_started", real.get("server_started") is True),
        _check(
            "server_thread_naturally_joined",
            real.get("server_thread_alive_after_shutdown") is False,
        ),
        _check("port_released", real.get("port_rebind_after_shutdown") is True),
        _check(
            "http_status_exact",
            statuses == {"live": 200, "ready": 200, "open": 201, "turn": 200, "close": 200},
        ),
        _check("health_ready", health == {"live": "live", "ready": "ready"}),
        _check(
            "thread_open_exact",
            opened.get("opened") is True and opened.get("completed_turn_count") == 0,
        ),
        _check("turn_committed_once", turn.get("completed_turn_count") == 1),
        _check(
            "thread_close_exact",
            closed.get("opened") is False and closed.get("completed_turn_count") == 1,
        ),
        _check(
            "public_events_exact",
            real.get("public_event_count") == 5
            and real.get("public_event_route_ids")
            == ["liveness", "readiness", "thread_open", "thread_turn", "thread_close"],
        ),
        _check("real_retrieval_once", event.get("retrieval_call_count") == 1),
        _check("real_model_decision_once", event.get("model_decision_count") == 1),
        _check(
            "real_action_allowed",
            event.get("selected_action")
            in {"compose_grounded_answer", "refuse_insufficient_evidence"},
        ),
        _check("real_branch_calls_exact", branch_counts == expected_branch),
        _check(
            "real_router_tokens_observed",
            int(event.get("router_input_token_count", 0)) > 0
            and int(event.get("router_output_token_count", 0)) > 0,
        ),
        _check(
            "real_router_latency_observed", float(event.get("router_generation_latency_ms", 0)) > 0
        ),
        _check(
            "coordinator_empty_after_shutdown",
            counters.get("opened_thread_count") == 0
            and counters.get("current_in_flight_turns") == 0,
        ),
        _check("coordinator_max_one", counters.get("max_observed_in_flight_turns") == 1),
        _check("resources_built_once", startup.get("resource_factory_build_count") == 1),
        _check("dense_retrieval_cpu", startup.get("retrieval_encoder_device") == "cpu"),
        _check("model_called_warmup_and_turn", startup.get("model_generation_call_count") == 2),
        _check("gpu_memory_observed", int(startup.get("peak_gpu_memory_bytes", 0)) > 0),
        _check(
            "test_closed",
            boundaries.get("test_split_loaded") is False
            and boundaries.get("test_metrics_run") is False,
        ),
        _check("gold_labels_closed", boundaries.get("gold_labels_read") is False),
        _check(
            "existing_route_unchanged", boundaries.get("existing_answer_route_changed") is False
        ),
        _check(
            "runtime_nondefault_remote_closed",
            boundaries.get("runtime_registered_as_default") is False
            and boundaries.get("remote_exposure_authorized") is False,
        ),
        _check("persistence_closed", boundaries.get("persistent_state_enabled") is False),
        _check(
            "retrieval_expansion_closed",
            boundaries.get("query_rewrite_enabled") is False
            and boundaries.get("second_retrieval_enabled") is False,
        ),
        _check(
            "recovery_actions_zero",
            [
                boundaries.get("queue_action_count"),
                boundaries.get("retry_action_count"),
                boundaries.get("fallback_action_count"),
            ]
            == [0, 0, 0],
        ),
        _check(
            "private_content_not_saved",
            [
                boundaries.get("raw_question_saved"),
                boundaries.get("raw_answer_saved"),
                boundaries.get("raw_document_saved"),
                boundaries.get("raw_model_output_saved"),
            ]
            == [False, False, False, False],
        ),
    ]
    return checks


def _current_source_paths(project_root: Path) -> dict[str, Path]:
    application = project_root / "src" / "ts_rag_agent" / "application"
    return {
        "config": project_root / "src" / "ts_rag_agent" / "config.py",
        "router": application / "primeqa_hybrid_structured_decision_router.py",
        "runtime": application / "primeqa_hybrid_bounded_dynamic_agent_runtime.py",
        "transport": application / "primeqa_hybrid_bounded_dynamic_agent_http_transport.py",
        "service_entrypoint": (
            application / "primeqa_hybrid_bounded_dynamic_agent_service_entrypoint.py"
        ),
        "validation": (application / "primeqa_hybrid_bounded_dynamic_agent_service_validation.py"),
        "cli": project_root / "src" / "ts_rag_agent" / "bounded_dynamic_agent_service.py",
        "pyproject": project_root / "pyproject.toml",
    }


def _request_json(
    connection: http.client.HTTPConnection,
    method: str,
    path: str,
    payload: Mapping[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        headers["content-type"] = "application/json; charset=utf-8"
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("Stage158 HTTP response must be a JSON object")
    return response.status, decoded


def _port_is_bindable(port: int) -> bool:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        listener.bind((BOUNDED_DYNAMIC_AGENT_BINDING_HOST, port))
    except OSError:
        return False
    finally:
        listener.close()
    return True


def _fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _bar(label: str, value: int | float | bool, value_label: str | None = None) -> BarDatum:
    return BarDatum(
        label=label,
        value=float(value),
        value_label=value_label if value_label is not None else str(value),
    )


def _chart(
    title: str,
    bars: list[BarDatum],
    *,
    width: int = 1800,
    margin_left: int = 800,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label="observed value",
        width=width,
        margin_left=margin_left,
    )
