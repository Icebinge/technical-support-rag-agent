from __future__ import annotations

import hashlib
import http.client
import json
import socket
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any

import uvicorn

from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_http_transport import (
    BOUNDED_DYNAMIC_AGENT_BINDING_HOST,
    BoundedDynamicAgentHttpLogSink,
    PublicSafeBoundedDynamicAgentHttpEvent,
    create_primeqa_hybrid_bounded_dynamic_agent_http_app,
    create_primeqa_hybrid_bounded_dynamic_agent_uvicorn_config,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_service_entrypoint import (
    CanonicalBoundedDynamicAgentServicePaths,
    PrimeQAHybridBoundedDynamicAgentServiceEntrypoint,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_warm_service_protocol import (
    STAGE159_DEV_SPLIT_FILENAME,
    STAGE159_EXPECTED_DEV_ROW_COUNT,
    STAGE159_EXPECTED_DEV_SHA256,
    STAGE159_MAX_TURNS_PER_THREAD,
    Stage159DevRuntimeQuery,
    Stage159RuntimeObservationGate,
    Stage159TurnObservation,
    Stage159WorkloadPlan,
    build_stage159_workload_plan,
    load_stage159_dev_runtime_queries,
    summarize_stage159_turn_observations,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    _forbidden_keys_found,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.config import ProjectSettings

STAGE159_EXPECTED_STAGE158_ARTIFACT_SHA256 = (
    "12649c087c3140feeb4121837152b41ef4005922eb73931f3770a5fac83889b0"
)
STAGE159_EXPECTED_STAGE158_GUARD_COUNT = 51
STAGE159_EXPECTED_STAGE158_STATUS = (
    "primeqa_hybrid_bounded_dynamic_agent_local_service_implemented_and_validated"
)
_STAGE = "Stage 159"
_CREATED_AT = "2026-07-18"
_ANALYSIS_ID = "primeqa_hybrid_bounded_dynamic_agent_warm_service_validation_v1"
_FINAL_STATUS = "primeqa_hybrid_bounded_dynamic_agent_warm_service_validated"


@dataclass(frozen=True)
class Stage159Visualization:
    name: str
    path: str


class RecordingStage159HttpLogSink(BoundedDynamicAgentHttpLogSink):
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

    def successful_turn_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            event
            for event in self.public_events()
            if event.get("route_id") == "thread_turn" and event.get("http_status") == 200
        )


class _ObservedStage159UvicornServer(uvicorn.Server):
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


def validate_primeqa_hybrid_bounded_dynamic_agent_warm_service(
    *,
    settings: ProjectSettings,
    port: int,
    user_confirmed_full_dev_protocol: bool,
    progress_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the Stage159 full-dev warm service and real admission validation."""

    import torch

    if not user_confirmed_full_dev_protocol:
        raise ValueError("Stage159 formal validation requires full-dev protocol confirmation")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage159 formal validation requires the selected CUDA environment")
    if not _port_is_bindable(port):
        raise RuntimeError("selected Stage159 validation port is not available")

    started_at = time.perf_counter()
    project_root = Path(__file__).resolve().parents[3]
    artifact_dir = settings.artifact_dir.resolve()
    stage158_path = artifact_dir / "primeqa_hybrid_bounded_dynamic_agent_service_stage158.json"
    dev_path = artifact_dir / "primeqa_hybrid_split_stage68_splits" / STAGE159_DEV_SPLIT_FILENAME
    current_sources = _current_source_paths(project_root)
    source_before = {key: _fingerprint(path) for key, path in current_sources.items()}
    stage158_before = _authorize_stage158_artifact(
        artifact_path=stage158_path,
        project_root=project_root,
    )
    authorized_at = time.perf_counter()
    _emit_progress(progress_sink, phase="source_authorized")

    query_set = load_stage159_dev_runtime_queries(dev_path)
    workload_plan = build_stage159_workload_plan(query_set)
    dev_loaded_at = time.perf_counter()
    _emit_progress(
        progress_sink,
        phase="dev_workload_loaded",
        completed_turn_count=0,
        total_turn_count=len(query_set.queries),
        completed_thread_count=0,
        total_thread_count=len(workload_plan.threads),
    )

    torch.cuda.reset_peak_memory_stats()
    sink = RecordingStage159HttpLogSink()
    gate_holder: dict[str, Stage159RuntimeObservationGate] = {}

    def app_factory(**kwargs: Any):
        runtime = kwargs.pop("runtime")
        gate = Stage159RuntimeObservationGate(runtime)
        gate_holder["gate"] = gate
        return create_primeqa_hybrid_bounded_dynamic_agent_http_app(
            **kwargs,
            runtime=gate,
            log_sink=sink,
        )

    paths = CanonicalBoundedDynamicAgentServicePaths.from_settings(settings)
    entrypoint = PrimeQAHybridBoundedDynamicAgentServiceEntrypoint(
        settings=settings,
        paths=paths,
        app_factory=app_factory,
    )
    prepared = entrypoint.prepare()
    gate = gate_holder.get("gate")
    if gate is None:
        raise RuntimeError("Stage159 app composition did not expose the observation gate")
    prepared_at = time.perf_counter()
    _emit_progress(progress_sink, phase="service_prepared")

    config = create_primeqa_hybrid_bounded_dynamic_agent_uvicorn_config(
        app=prepared.app,
        port=port,
    )
    lifecycle_events: Queue[tuple[str, BaseException | None]] = Queue()
    server = _ObservedStage159UvicornServer(
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
        name="stage159-real-loopback-service",
        daemon=False,
    )
    server_thread.start()
    lifecycle_state, lifecycle_error = lifecycle_events.get()
    if lifecycle_state != "started":
        server_thread.join()
        raise RuntimeError("Stage159 Uvicorn server exited before startup") from lifecycle_error
    server_started_at = time.perf_counter()
    _emit_progress(progress_sink, phase="server_started")

    connection = http.client.HTTPConnection(BOUNDED_DYNAMIC_AGENT_BINDING_HOST, port)
    try:
        live_status, live = _request_json(connection, "GET", "/health/live")
        ready_status, ready = _request_json(connection, "GET", "/health/ready")
        observations, dev_http = _run_full_dev_workload(
            connection=connection,
            workload_plan=workload_plan,
            sink=sink,
            progress_sink=progress_sink,
        )
        dev_completed_at = time.perf_counter()
        capacity_probe = _run_real_capacity_probe(
            connection=connection,
            gate=gate,
            first_query=query_set.queries[0],
            second_query=query_set.queries[1],
            sink=sink,
            port=port,
            progress_sink=progress_sink,
        )
        capacity_counters = (
            prepared.app.state.bounded_dynamic_agent_http_transport.coordinator.counters()
        )
        capacity_probe = {
            **capacity_probe,
            "capacity_rejected_turn_count": capacity_counters.capacity_rejected_turn_count,
            "max_observed_in_flight_turns": (capacity_counters.max_observed_in_flight_turns),
        }
        capacity_completed_at = time.perf_counter()
    finally:
        connection.close()
        server.should_exit = True
        server_thread.join()
    shutdown_completed_at = time.perf_counter()
    _emit_progress(progress_sink, phase="server_stopped")

    turn_summary = summarize_stage159_turn_observations(observations)
    events = sink.public_events()
    transport = prepared.app.state.bounded_dynamic_agent_http_transport
    counters = transport.coordinator.counters()
    source_after = {key: _fingerprint(path) for key, path in current_sources.items()}
    stage158_after = _fingerprint(stage158_path)
    dev_after = _fingerprint(dev_path)
    port_released = _port_is_bindable(port)
    finished_at = time.perf_counter()

    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "validation_scope": (
            "Full frozen-development warm multi-turn behavior over the nondefault Stage158 "
            "loopback service, plus one deterministic real two-request admission rejection."
        ),
        "user_confirmation": {
            "full_dev_protocol_confirmed": True,
            "dev_query_count": STAGE159_EXPECTED_DEV_ROW_COUNT,
            "grouping_policy": "stable_hash_order_consecutive_groups_of_four",
            "synthetic_conversation_grouping_disclosed": True,
        },
        "stage158_authorization": stage158_before,
        "source_files": source_before,
        "source_unchanged_after_validation": source_before == source_after,
        "stage158_artifact_unchanged_after_validation": (
            stage158_before["artifact"] == stage158_after
        ),
        "dev_source_unchanged_after_validation": (
            query_set.source_sha256 == dev_after["sha256"]
            and query_set.source_size_bytes == dev_after["size_bytes"]
        ),
        "dev_query_protocol": query_set.public_summary(),
        "workload_plan": workload_plan.public_summary(),
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
        "startup": {
            "resource_factory_build_count": prepared.resource_factory_build_count,
            "retrieval_encoder_device": prepared.retrieval_encoder_device,
            "warmup": prepared.warmup.to_public_dict(),
            "model_generation_call_count": prepared.backend.generation_call_count,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "timing_seconds": dict(prepared.timing_seconds),
        },
        "real_service": {
            "binding_host": BOUNDED_DYNAMIC_AGENT_BINDING_HOST,
            "binding_port": port,
            "server_started": server.started,
            "server_thread_alive_after_shutdown": server_thread.is_alive(),
            "port_rebind_after_shutdown": port_released,
            "health_status": {"live": live_status, "ready": ready_status},
            "health_state": {
                "live": live.get("status"),
                "ready": ready.get("status"),
            },
            "dev_http": dev_http,
            "dev_turn_summary": turn_summary,
            "capacity_probe": capacity_probe,
            "public_event_count": len(events),
            "public_event_route_counts": dict(
                sorted(Counter(str(event.get("route_id")) for event in events).items())
            ),
            "public_event_http_status_counts": dict(
                sorted(Counter(str(event.get("http_status")) for event in events).items())
            ),
            "coordinator_counters_after_shutdown": asdict(counters),
        },
        "closed_boundaries": {
            "train_split_loaded": False,
            "dev_split_loaded": True,
            "test_split_loaded": False,
            "test_metrics_run": False,
            "dev_gold_quality_metrics_run": False,
            "dev_label_fields_used_for_selection": False,
            "dev_label_fields_projected_into_runtime": False,
            "existing_answer_route_changed": False,
            "runtime_registered_as_default": False,
            "remote_exposure_authorized": False,
            "persistent_state_enabled": False,
            "implicit_thread_creation_enabled": False,
            "query_rewrite_enabled": False,
            "second_retrieval_enabled": False,
            "queue_action_count": counters.queue_action_count,
            "retry_action_count": counters.retry_action_count,
            "fallback_action_count": counters.fallback_action_count,
            "raw_question_saved": False,
            "raw_answer_saved": False,
            "raw_document_saved": False,
            "raw_model_output_saved": False,
            "private_sample_identity_saved": False,
        },
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "dev_query_load_and_plan": round(dev_loaded_at - authorized_at, 6),
            "service_prepare": round(prepared_at - dev_loaded_at, 6),
            "server_start": round(server_started_at - prepared_at, 6),
            "full_dev_workload": round(dev_completed_at - server_started_at, 6),
            "capacity_probe": round(capacity_completed_at - dev_completed_at, 6),
            "shutdown": round(shutdown_completed_at - capacity_completed_at, 6),
            "final_audit": round(finished_at - shutdown_completed_at, 6),
            "total": round(finished_at - started_at, 6),
        },
    }
    report["guard_checks"] = _guard_checks(report)
    forbidden = sorted(_forbidden_keys_found(report))
    report["public_safe_contract"] = {
        "forbidden_keys_found": forbidden,
        "private_request_or_response_content_saved": False,
        "private_sample_identity_saved": False,
        "individual_turn_rows_saved": False,
    }
    passed = all(check["passed"] for check in report["guard_checks"]) and not forbidden
    report["decision"] = {
        "status": _FINAL_STATUS if passed else "stage159_warm_service_rejected",
        "all_guards_passed": passed,
        "failed_checks": [check["name"] for check in report["guard_checks"] if not check["passed"]],
        "warm_multi_turn_service_validated": passed,
        "real_capacity_rejection_validated": passed,
        "runtime_registered_as_default": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
        "next_direction": "analyze_dev_runtime_failures_or_freeze_agent_runtime_behavior",
    }
    _emit_progress(
        progress_sink,
        phase="validation_complete",
        guard_count=len(report["guard_checks"]),
        failed_guard_count=sum(not check["passed"] for check in report["guard_checks"]),
    )
    return report


def write_stage159_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[Stage159Visualization]:
    output_dir.mkdir(parents=True, exist_ok=True)
    turn_summary = report["real_service"]["dev_turn_summary"]
    by_position = turn_summary["by_turn_position"]
    capacity = report["real_service"]["capacity_probe"]
    startup = report["startup"]["timing_seconds"]
    counters = report["real_service"]["coordinator_counters_after_shutdown"]
    charts = {
        "stage159_turn_position_end_to_end_latency.svg": _position_chart(
            title="Stage159 end-to-end latency by turn position",
            by_position=by_position,
            metric="end_to_end_latency_ms",
            statistic="average",
            x_label="average milliseconds",
        ),
        "stage159_turn_position_generation_latency.svg": _position_chart(
            title="Stage159 model generation latency by turn position",
            by_position=by_position,
            metric="router_generation_latency_ms",
            statistic="average",
            x_label="average milliseconds",
        ),
        "stage159_turn_position_input_tokens.svg": _position_chart(
            title="Stage159 router input tokens by turn position",
            by_position=by_position,
            metric="router_input_token_count",
            statistic="average",
            x_label="average tokens",
        ),
        "stage159_turn_position_retained_state.svg": _position_chart(
            title="Stage159 retained state by turn position",
            by_position=by_position,
            metric="retained_state_bytes",
            statistic="average",
            x_label="average bytes",
        ),
        "stage159_answer_refusal_distribution.svg": _chart(
            "Stage159 answer and refusal distribution",
            [
                _bar("answer", turn_summary["answer_count"]),
                _bar("refusal", turn_summary["refusal_count"]),
            ],
            "turn count",
        ),
        "stage159_selected_action_distribution.svg": _chart(
            "Stage159 selected action distribution",
            [_bar(label, value) for label, value in turn_summary["selected_action_counts"].items()],
            "turn count",
        ),
        "stage159_branch_call_totals.svg": _chart(
            "Stage159 bounded branch call totals",
            [
                _bar("retrieval", turn_summary["retrieval_call_count"]),
                _bar("model decision", turn_summary["model_decision_count"]),
                _bar("composition", turn_summary["composition_call_count"]),
                _bar("verification", turn_summary["verification_call_count"]),
                _bar("diagnostics", turn_summary["diagnostic_observation_count"]),
            ],
            "call count",
        ),
        "stage159_real_capacity_probe.svg": _chart(
            "Stage159 real two-request admission probe",
            [
                _bar("first request status", capacity["first_request_http_status"]),
                _bar("second request status", capacity["second_request_http_status"]),
                _bar("capacity rejection count", capacity["capacity_rejected_turn_count"]),
                _bar("maximum in-flight", capacity["max_observed_in_flight_turns"]),
            ],
            "observed value",
        ),
        "stage159_startup_timing.svg": _chart(
            "Stage159 startup timing",
            [
                _bar("source authorization", startup["source_authorization"]),
                _bar("retrieval resource build", startup["retrieval_resource_build"]),
                _bar("model load", startup["model_load"]),
                _bar("warmup", startup["warmup"]),
                _bar("app composition", startup["app_composition"]),
            ],
            "seconds",
        ),
        "stage159_final_service_counters.svg": _chart(
            "Stage159 final service counters",
            [
                _bar("admitted turns", counters["admitted_turn_count"]),
                _bar("completed turns", counters["completed_turn_count"]),
                _bar("capacity rejected", counters["capacity_rejected_turn_count"]),
                _bar("failed turns", counters["failed_turn_count"]),
                _bar("open threads", counters["opened_thread_count"]),
            ],
            "count",
        ),
    }
    visualizations = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        visualizations.append(Stage159Visualization(name=filename, path=str(path)))
    return visualizations


def _run_full_dev_workload(
    *,
    connection: http.client.HTTPConnection,
    workload_plan: Stage159WorkloadPlan,
    sink: RecordingStage159HttpLogSink,
    progress_sink: Callable[[Mapping[str, Any]], None] | None,
) -> tuple[tuple[Stage159TurnObservation, ...], dict[str, Any]]:
    observations: list[Stage159TurnObservation] = []
    open_statuses: list[int] = []
    turn_statuses: list[int] = []
    close_statuses: list[int] = []
    for thread in workload_plan.threads:
        handle = f"stage159-dev-thread-{thread.ordinal:03d}"
        open_status, _ = _request_json(
            connection,
            "POST",
            "/v1/bounded-agent/threads/open",
            {"thread_handle": handle},
        )
        open_statuses.append(open_status)
        if open_status != 201:
            raise RuntimeError("Stage159 dev thread open did not return 201")
        for turn_position, query in enumerate(thread.queries, start=1):
            event_cursor = len(sink.successful_turn_events())
            turn_started = time.perf_counter()
            status, response = _request_json(
                connection,
                "POST",
                "/v1/bounded-agent/threads/turn",
                {
                    "thread_handle": handle,
                    "title": query.runtime_query.title,
                    "text": query.runtime_query.text,
                },
            )
            latency_ms = round((time.perf_counter() - turn_started) * 1000, 3)
            turn_statuses.append(status)
            if status != 200:
                raise RuntimeError("Stage159 dev turn did not return 200")
            successful_events = sink.successful_turn_events()
            if len(successful_events) != event_cursor + 1:
                raise RuntimeError("Stage159 dev turn did not emit one successful public event")
            event = successful_events[-1]
            observations.append(
                Stage159TurnObservation(
                    thread_ordinal=thread.ordinal,
                    turn_position=turn_position,
                    http_status=status,
                    refused=bool(response.get("refused")),
                    citation_count=len(response.get("citations") or []),
                    terminal_state=str(response.get("terminal_state")),
                    selected_action=str(event.get("selected_action")),
                    completed_turn_count=int(response.get("completed_turn_count") or 0),
                    retained_state_bytes=int(response.get("retained_state_bytes") or 0),
                    router_input_token_count=int(event.get("router_input_token_count") or 0),
                    router_output_token_count=int(event.get("router_output_token_count") or 0),
                    router_generation_latency_ms=float(
                        event.get("router_generation_latency_ms") or 0.0
                    ),
                    end_to_end_latency_ms=latency_ms,
                    retrieval_call_count=int(event.get("retrieval_call_count") or 0),
                    model_decision_count=int(event.get("model_decision_count") or 0),
                    composition_call_count=int(event.get("composition_call_count") or 0),
                    verification_call_count=int(event.get("verification_call_count") or 0),
                    diagnostic_observation_count=int(
                        event.get("diagnostic_observation_count") or 0
                    ),
                )
            )
        close_status, _ = _request_json(
            connection,
            "POST",
            "/v1/bounded-agent/threads/close",
            {"thread_handle": handle},
        )
        close_statuses.append(close_status)
        if close_status != 200:
            raise RuntimeError("Stage159 dev thread close did not return 200")
        _emit_progress(
            progress_sink,
            phase="dev_thread_completed",
            completed_turn_count=len(observations),
            total_turn_count=sum(len(item.queries) for item in workload_plan.threads),
            completed_thread_count=thread.ordinal,
            total_thread_count=len(workload_plan.threads),
        )
    return tuple(observations), {
        "open_request_count": len(open_statuses),
        "turn_request_count": len(turn_statuses),
        "close_request_count": len(close_statuses),
        "open_http_status_counts": _status_counts(open_statuses),
        "turn_http_status_counts": _status_counts(turn_statuses),
        "close_http_status_counts": _status_counts(close_statuses),
    }


def _run_real_capacity_probe(
    *,
    connection: http.client.HTTPConnection,
    gate: Stage159RuntimeObservationGate,
    first_query: Stage159DevRuntimeQuery,
    second_query: Stage159DevRuntimeQuery,
    sink: RecordingStage159HttpLogSink,
    port: int,
    progress_sink: Callable[[Mapping[str, Any]], None] | None,
) -> dict[str, Any]:
    first_handle = "stage159-capacity-first"
    second_handle = "stage159-capacity-second"
    first_open, _ = _request_json(
        connection,
        "POST",
        "/v1/bounded-agent/threads/open",
        {"thread_handle": first_handle},
    )
    second_open, _ = _request_json(
        connection,
        "POST",
        "/v1/bounded-agent/threads/open",
        {"thread_handle": second_handle},
    )
    if first_open != 201 or second_open != 201:
        raise RuntimeError("Stage159 capacity threads did not both open")
    first_results: Queue[tuple[str, Any]] = Queue()
    successful_cursor = len(sink.successful_turn_events())

    def run_first_request() -> None:
        first_connection = http.client.HTTPConnection(BOUNDED_DYNAMIC_AGENT_BINDING_HOST, port)
        try:
            started = time.perf_counter()
            status, response = _request_json(
                first_connection,
                "POST",
                "/v1/bounded-agent/threads/turn",
                {
                    "thread_handle": first_handle,
                    "title": first_query.runtime_query.title,
                    "text": first_query.runtime_query.text,
                },
            )
            first_results.put(
                (
                    "completed",
                    {
                        "status": status,
                        "response": response,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    },
                )
            )
        except BaseException as error:
            first_results.put(("failed", error))
        finally:
            first_connection.close()

    gate.arm()
    first_thread = Thread(
        target=run_first_request,
        name="stage159-capacity-first-request",
        daemon=False,
    )
    first_thread.start()
    gate.wait_until_entered()
    _emit_progress(progress_sink, phase="capacity_first_request_admitted")
    try:
        second_started = time.perf_counter()
        second_status, second_response = _request_json(
            connection,
            "POST",
            "/v1/bounded-agent/threads/turn",
            {
                "thread_handle": second_handle,
                "title": second_query.runtime_query.title,
                "text": second_query.runtime_query.text,
            },
        )
        second_latency_ms = round((time.perf_counter() - second_started) * 1000, 3)
    finally:
        gate.release()
    first_state, first_value = first_results.get()
    first_thread.join()
    if first_state != "completed":
        raise RuntimeError("Stage159 admitted first request failed") from first_value
    first_status = int(first_value["status"])
    first_response = first_value["response"]
    successful_events = sink.successful_turn_events()
    if len(successful_events) != successful_cursor + 1:
        raise RuntimeError("Stage159 capacity first request emitted unexpected events")
    all_events = sink.public_events()
    rejected_events = [
        event
        for event in all_events
        if event.get("route_id") == "thread_turn"
        and event.get("http_status") == 503
        and event.get("outcome_code") == "gpu_capacity_exceeded"
    ]
    if not rejected_events:
        raise RuntimeError("Stage159 capacity second request emitted no rejection event")
    rejected_event = rejected_events[-1]
    first_close, first_closed = _request_json(
        connection,
        "POST",
        "/v1/bounded-agent/threads/close",
        {"thread_handle": first_handle},
    )
    second_close, second_closed = _request_json(
        connection,
        "POST",
        "/v1/bounded-agent/threads/close",
        {"thread_handle": second_handle},
    )
    error = second_response.get("error") or {}
    _emit_progress(progress_sink, phase="capacity_probe_completed")
    return {
        "observation_gate": "validation_only_before_real_runtime_execution",
        "observation_gate_wait_timeout_seconds": None,
        "first_thread_open_http_status": first_open,
        "second_thread_open_http_status": second_open,
        "first_request_http_status": first_status,
        "first_request_end_to_end_latency_ms": first_value["latency_ms"],
        "first_request_terminal_state": first_response.get("terminal_state"),
        "first_request_completed_turn_count": first_response.get("completed_turn_count"),
        "second_request_http_status": second_status,
        "second_request_rejection_latency_ms": second_latency_ms,
        "second_request_error_code": error.get("code"),
        "second_request_downstream_dispatched": rejected_event.get("downstream_dispatched"),
        "second_request_gpu_admitted": rejected_event.get("gpu_admitted"),
        "first_thread_close_http_status": first_close,
        "second_thread_close_http_status": second_close,
        "first_thread_opened_after_close": first_closed.get("opened"),
        "second_thread_opened_after_close": second_closed.get("opened"),
    }


def _authorize_stage158_artifact(
    *,
    artifact_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    artifact = _fingerprint(artifact_path)
    if artifact["sha256"] != STAGE159_EXPECTED_STAGE158_ARTIFACT_SHA256:
        raise RuntimeError("Stage159 requires the exact corrected Stage158 artifact")
    report = _load_json_object(artifact_path)
    checks = report.get("guard_checks") or []
    decision = report.get("decision") or {}
    if not (
        report.get("stage") == "Stage 158"
        and len(checks) == STAGE159_EXPECTED_STAGE158_GUARD_COUNT
        and all(check.get("passed") is True for check in checks)
        and decision.get("status") == STAGE159_EXPECTED_STAGE158_STATUS
        and decision.get("all_guards_passed") is True
        and decision.get("runtime_registered_as_default") is False
        and decision.get("test_gate_opened") is False
        and decision.get("test_metrics_run") is False
    ):
        raise RuntimeError("Stage159 rejected the Stage158 decision boundary")
    expected_sources = report.get("source_files") or {}
    current_paths = _stage158_source_paths(project_root)
    matched = 0
    for key, path in current_paths.items():
        expected = expected_sources.get(key)
        current = _fingerprint(path)
        if not isinstance(expected, Mapping) or (
            current["size_bytes"] != expected.get("size_bytes")
            or current["sha256"] != expected.get("sha256")
        ):
            raise RuntimeError(f"Stage159 rejected changed Stage158 source: {key}")
        matched += 1
    return {
        "artifact": artifact,
        "artifact_identity_exact": True,
        "guard_count": len(checks),
        "all_guards_passed": True,
        "decision_status": decision.get("status"),
        "source_fingerprint_match_count": matched,
        "runtime_registered_as_default": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
    }


def _guard_checks(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    confirmation = report.get("user_confirmation") or {}
    authorization = report.get("stage158_authorization") or {}
    dev = report.get("dev_query_protocol") or {}
    plan = report.get("workload_plan") or {}
    startup = report.get("startup") or {}
    service = report.get("real_service") or {}
    dev_http = service.get("dev_http") or {}
    summary = service.get("dev_turn_summary") or {}
    capacity = service.get("capacity_probe") or {}
    counters = service.get("coordinator_counters_after_shutdown") or {}
    boundaries = report.get("closed_boundaries") or {}
    position_counts = plan.get("turn_position_counts") or {}
    action_counts = summary.get("selected_action_counts") or {}
    checks = [
        _check(
            "full_dev_protocol_confirmed", confirmation.get("full_dev_protocol_confirmed") is True
        ),
        _check(
            "synthetic_grouping_disclosed",
            confirmation.get("synthetic_conversation_grouping_disclosed") is True,
        ),
        _check("stage158_artifact_exact", authorization.get("artifact_identity_exact") is True),
        _check("stage158_guard_count_exact", authorization.get("guard_count") == 51),
        _check("stage158_guards_passed", authorization.get("all_guards_passed") is True),
        _check("stage158_sources_exact", authorization.get("source_fingerprint_match_count") == 8),
        _check(
            "stage158_runtime_nondefault",
            authorization.get("runtime_registered_as_default") is False,
        ),
        _check("stage158_test_closed", authorization.get("test_gate_opened") is False),
        _check(
            "stage159_sources_unchanged", report.get("source_unchanged_after_validation") is True
        ),
        _check(
            "stage158_artifact_unchanged",
            report.get("stage158_artifact_unchanged_after_validation") is True,
        ),
        _check("dev_source_unchanged", report.get("dev_source_unchanged_after_validation") is True),
        _check("dev_source_hash_exact", dev.get("source_sha256") == STAGE159_EXPECTED_DEV_SHA256),
        _check("dev_query_count_exact", dev.get("dev_query_count") == 121),
        _check("dev_split_exact", dev.get("assigned_split") == "dev"),
        _check("dev_label_selection_closed", dev.get("label_fields_used_for_selection") is False),
        _check(
            "dev_label_runtime_projection_closed",
            dev.get("label_fields_projected_into_runtime") is False,
        ),
        _check("dev_gold_metrics_closed", dev.get("label_fields_used_for_metrics") is False),
        _check("workload_thread_count_exact", plan.get("thread_count") == 31),
        _check("workload_full_thread_count_exact", plan.get("full_four_turn_thread_count") == 30),
        _check("workload_trailing_turn_count_exact", plan.get("trailing_thread_turn_count") == 1),
        _check("workload_turn_count_exact", plan.get("turn_count") == 121),
        _check(
            "turn_position_counts_exact", position_counts == {"1": 31, "2": 30, "3": 30, "4": 30}
        ),
        _check("server_started", service.get("server_started") is True),
        _check("server_thread_joined", service.get("server_thread_alive_after_shutdown") is False),
        _check("port_rebind_after_shutdown", service.get("port_rebind_after_shutdown") is True),
        _check("health_http_exact", service.get("health_status") == {"live": 200, "ready": 200}),
        _check(
            "health_state_exact", service.get("health_state") == {"live": "live", "ready": "ready"}
        ),
        _check("dev_open_requests_exact", dev_http.get("open_request_count") == 31),
        _check("dev_turn_requests_exact", dev_http.get("turn_request_count") == 121),
        _check("dev_close_requests_exact", dev_http.get("close_request_count") == 31),
        _check("dev_open_http_all_201", dev_http.get("open_http_status_counts") == {"201": 31}),
        _check("dev_turn_http_all_200", dev_http.get("turn_http_status_counts") == {"200": 121}),
        _check("dev_close_http_all_200", dev_http.get("close_http_status_counts") == {"200": 31}),
        _check("dev_summary_turn_count_exact", summary.get("turn_count") == 121),
        _check("dev_summary_thread_count_exact", summary.get("thread_count") == 31),
        _check("dev_branch_protocol_exact", summary.get("branch_protocol_valid_count") == 121),
        _check("dev_retrieval_calls_exact", summary.get("retrieval_call_count") == 121),
        _check("dev_model_decisions_exact", summary.get("model_decision_count") == 121),
        _check(
            "dev_state_growth_monotonic", summary.get("state_growth_monotonic_thread_count") == 31
        ),
        _check(
            "dev_answer_refusal_total_exact",
            summary.get("answer_count", 0) + summary.get("refusal_count", 0) == 121,
        ),
        _check("dev_action_total_exact", sum(action_counts.values()) == 121),
        _check(
            "capacity_gate_has_no_timeout",
            capacity.get("observation_gate_wait_timeout_seconds") is None,
        ),
        _check(
            "capacity_threads_opened",
            capacity.get("first_thread_open_http_status") == 201
            and capacity.get("second_thread_open_http_status") == 201,
        ),
        _check(
            "capacity_first_request_completed",
            capacity.get("first_request_http_status") == 200
            and capacity.get("first_request_completed_turn_count") == 1,
        ),
        _check(
            "capacity_second_request_rejected",
            capacity.get("second_request_http_status") == 503
            and capacity.get("second_request_error_code") == "gpu_capacity_exceeded",
        ),
        _check(
            "capacity_rejected_before_dispatch",
            capacity.get("second_request_downstream_dispatched") is False
            and capacity.get("second_request_gpu_admitted") is False,
        ),
        _check(
            "capacity_threads_closed",
            capacity.get("first_thread_close_http_status") == 200
            and capacity.get("second_thread_close_http_status") == 200
            and capacity.get("first_thread_opened_after_close") is False
            and capacity.get("second_thread_opened_after_close") is False,
        ),
        _check("one_global_in_flight_turn", counters.get("max_observed_in_flight_turns") == 1),
        _check("capacity_rejection_count_exact", counters.get("capacity_rejected_turn_count") == 1),
        _check("admitted_turn_count_exact", counters.get("admitted_turn_count") == 122),
        _check("completed_turn_count_exact", counters.get("completed_turn_count") == 122),
        _check("failed_turn_count_zero", counters.get("failed_turn_count") == 0),
        _check("opened_threads_zero_after_shutdown", counters.get("opened_thread_count") == 0),
        _check("resource_factory_built_once", startup.get("resource_factory_build_count") == 1),
        _check("model_generation_count_exact", startup.get("model_generation_call_count") == 123),
        _check(
            "warmup_thread_closed",
            (startup.get("warmup") or {}).get("thread_opened_after_close") is False,
        ),
        _check("gpu_memory_observed", startup.get("peak_gpu_memory_bytes", 0) > 0),
        _check(
            "dev_only_split_boundary",
            boundaries.get("dev_split_loaded") is True
            and boundaries.get("train_split_loaded") is False
            and boundaries.get("test_split_loaded") is False,
        ),
        _check("test_metrics_closed", boundaries.get("test_metrics_run") is False),
        _check(
            "dev_quality_metrics_closed", boundaries.get("dev_gold_quality_metrics_run") is False
        ),
        _check(
            "no_queue_retry_fallback",
            [
                boundaries.get("queue_action_count"),
                boundaries.get("retry_action_count"),
                boundaries.get("fallback_action_count"),
            ]
            == [0, 0, 0],
        ),
        _check("runtime_nondefault", boundaries.get("runtime_registered_as_default") is False),
        _check(
            "remote_and_persistence_closed",
            boundaries.get("remote_exposure_authorized") is False
            and boundaries.get("persistent_state_enabled") is False,
        ),
        _check(
            "rewrite_and_second_retrieval_closed",
            boundaries.get("query_rewrite_enabled") is False
            and boundaries.get("second_retrieval_enabled") is False,
        ),
        _check(
            "raw_and_private_content_not_saved",
            all(
                boundaries.get(key) is False
                for key in (
                    "raw_question_saved",
                    "raw_answer_saved",
                    "raw_document_saved",
                    "raw_model_output_saved",
                    "private_sample_identity_saved",
                )
            ),
        ),
    ]
    return checks


def _stage158_source_paths(project_root: Path) -> dict[str, Path]:
    application = project_root / "src" / "ts_rag_agent" / "application"
    return {
        "config": project_root / "src" / "ts_rag_agent" / "config.py",
        "router": application / "primeqa_hybrid_structured_decision_router.py",
        "runtime": application / "primeqa_hybrid_bounded_dynamic_agent_runtime.py",
        "transport": application / "primeqa_hybrid_bounded_dynamic_agent_http_transport.py",
        "service_entrypoint": application
        / "primeqa_hybrid_bounded_dynamic_agent_service_entrypoint.py",
        "validation": application / "primeqa_hybrid_bounded_dynamic_agent_service_validation.py",
        "cli": project_root / "src" / "ts_rag_agent" / "bounded_dynamic_agent_service.py",
        "pyproject": project_root / "pyproject.toml",
    }


def _current_source_paths(project_root: Path) -> dict[str, Path]:
    application = project_root / "src" / "ts_rag_agent" / "application"
    return {
        "warm_service_protocol": application
        / "primeqa_hybrid_bounded_dynamic_agent_warm_service_protocol.py",
        "warm_service_validation": application
        / "primeqa_hybrid_bounded_dynamic_agent_warm_service_validation.py",
        "validation_cli": project_root
        / "scripts"
        / "validate_primeqa_hybrid_bounded_dynamic_agent_warm_service.py",
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
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    decoded = json.loads(raw.decode("utf-8")) if raw else {}
    if not isinstance(decoded, dict):
        raise RuntimeError("Stage159 loopback response is not a JSON object")
    return response.status, decoded


def _status_counts(statuses: Sequence[int]) -> dict[str, int]:
    return dict(sorted(Counter(str(status) for status in statuses).items()))


def _port_is_bindable(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((BOUNDED_DYNAMIC_AGENT_BINDING_HOST, port))
        except OSError:
            return False
    return True


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(f"Unable to read Stage159 source artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Stage159 source artifact is not an object: {path.name}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size_bytes": resolved.stat().st_size, "sha256": digest.hexdigest()}


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit_progress(
    sink: Callable[[Mapping[str, Any]], None] | None,
    *,
    phase: str,
    **values: Any,
) -> None:
    if sink is not None:
        sink({"stage": _STAGE, "phase": phase, **values})


def _bar(label: str, value: int | float | bool) -> BarDatum:
    number = float(value)
    value_label = str(value) if isinstance(value, int | bool) else f"{number:.3f}"
    return BarDatum(label=label, value=number, value_label=value_label)


def _chart(title: str, bars: Sequence[BarDatum], x_label: str) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        margin_left=260,
    )


def _position_chart(
    *,
    title: str,
    by_position: Mapping[str, Any],
    metric: str,
    statistic: str,
    x_label: str,
) -> str:
    return _chart(
        title,
        [
            _bar(
                f"turn {position}",
                float((by_position.get(str(position)) or {}).get(metric, {}).get(statistic, 0.0)),
            )
            for position in range(1, STAGE159_MAX_TURNS_PER_THREAD + 1)
        ],
        x_label,
    )
