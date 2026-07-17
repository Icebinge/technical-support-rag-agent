from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import httpx
import uvicorn
from fastapi.testclient import TestClient

from ts_rag_agent.application.primeqa_hybrid_agent_http_transport import (
    AgentHttpTransportState,
    PublicSafeAgentHttpTransportEvent,
    agent_http_transport_contract,
    create_primeqa_hybrid_agent_http_app,
    create_primeqa_hybrid_agent_uvicorn_config,
)
from ts_rag_agent.application.primeqa_hybrid_agent_request_facade import (
    AgentRequestFacadeState,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_activation import (
    PrimeQAHybridConcurrentRuntimeBootstrapResult,
    PublicSafeConcurrentRuntimeStartupTrace,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_sidecar_agent_runtime import (
    ConcurrentArrivalPattern,
    PrimeQAHybridConcurrentCapacityExceededError,
    PublicSafeConcurrentRuntimeRequestTrace,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.domain.answer import AnswerCitation, GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQARuntimeQuery

_STAGE = "Stage 150"
_CREATED_AT = "2026-07-18"
_ANALYSIS_ID = "primeqa_hybrid_local_fastapi_agent_transport_validation_v1"
_SOURCE_PROTOCOL_ID = "primeqa_hybrid_agent_http_transport_protocol_v1"
_SOURCE_STATUS = "primeqa_hybrid_agent_http_transport_protocol_frozen"
_EXPECTED_SOURCE_GUARDS = 39
_FINAL_STATUS = "primeqa_hybrid_local_fastapi_agent_transport_validation_passed"
_NEXT_DIRECTION = "freeze_local_agent_service_entrypoint_protocol"
_PUBLIC_LOG_FIELD_COUNT = 18


@dataclass(frozen=True)
class PrimeQAHybridAgentHttpTransportValidationVisualization:
    """One Stage150 aggregate/synthetic/loopback visualization."""

    name: str
    path: str


@dataclass
class _RecordingLogSink:
    events: list[PublicSafeAgentHttpTransportEvent] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)

    def emit(self, event: PublicSafeAgentHttpTransportEvent) -> None:
        event.to_public_dict()
        with self.lock:
            self.events.append(event)


@dataclass(frozen=True)
class _SyntheticRuntimeRun:
    verified_answer: GeneratedAnswer
    public_safe_trace: PublicSafeConcurrentRuntimeRequestTrace


class _SyntheticRuntime:
    def __init__(self, *, refused: bool = False, error: RuntimeError | None = None) -> None:
        self.refused = refused
        self.error = error
        self.call_count = 0
        self.lock = Lock()

    def run(
        self,
        question: PrimeQARuntimeQuery,
        *,
        arrival_pattern: ConcurrentArrivalPattern,
    ) -> _SyntheticRuntimeRun:
        if arrival_pattern is not ConcurrentArrivalPattern.APPLICATION:
            raise AssertionError("HTTP transport must use application_request arrival")
        with self.lock:
            self.call_count += 1
        if self.error is not None:
            raise self.error
        return _synthetic_run(question.id, refused=self.refused)


class _CapacityRuntime(_SyntheticRuntime):
    def run(
        self,
        question: PrimeQARuntimeQuery,
        *,
        arrival_pattern: ConcurrentArrivalPattern,
    ) -> _SyntheticRuntimeRun:
        _ = question, arrival_pattern
        with self.lock:
            self.call_count += 1
        raise PrimeQAHybridConcurrentCapacityExceededError(
            _runtime_trace(
                admission_state="rejected_capacity",
                terminal_state="capacity_rejected",
                candidate_pool_depth=0,
            )
        )


class _BlockingRuntime(_SyntheticRuntime):
    def __init__(self, *, target_count: int = 1) -> None:
        super().__init__()
        self.target_count = target_count
        self.target_entered = Event()
        self.release = Event()

    def run(
        self,
        question: PrimeQARuntimeQuery,
        *,
        arrival_pattern: ConcurrentArrivalPattern,
    ) -> _SyntheticRuntimeRun:
        if arrival_pattern is not ConcurrentArrivalPattern.APPLICATION:
            raise AssertionError("HTTP transport must use application_request arrival")
        with self.lock:
            self.call_count += 1
            if self.call_count == self.target_count:
                self.target_entered.set()
        self.release.wait()
        return _synthetic_run(question.id)


def run_primeqa_hybrid_agent_http_transport_validation(
    *,
    stage149_protocol_path: Path,
    user_confirmed_validation: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Validate Stage150 against Stage149 using synthetic ASGI and loopback HTTP only."""

    started_at = time.perf_counter()
    source = _load_json_object(stage149_protocol_path)
    loaded_at = time.perf_counter()
    source_summary = _stage149_summary(source)
    source_checks = _source_gate_checks(
        source_summary=source_summary,
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
    )
    source_gate_passed = all(check["passed"] for check in source_checks)
    preliminary = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "validation_scope": (
            "Validate the disabled-by-default, loopback-only FastAPI adapter against the "
            "saved Stage149 public protocol. Confirmed validation uses synthetic runtimes, "
            "in-process ASGI calls, and one temporary real 127.0.0.1 HTTP/1.1 socket. It "
            "does not load train, dev, test, questions, documents, models, indexes, or "
            "candidate pools; change runtime defaults; expose a remote interface; or add "
            "queues, retries, fallback, request timeouts, forced shutdown, or hard cancellation."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_validation),
            "confirmation_note": confirmation_note,
        },
        "source_files": {"stage149_protocol": _fingerprint(stage149_protocol_path)},
        "stage149_summary": source_summary,
        "source_gate_checks": source_checks,
        "source_gate_passed": source_gate_passed,
        "implementation_contract": agent_http_transport_contract(),
    }
    if not source_gate_passed:
        checked_at = time.perf_counter()
        report = {
            **preliminary,
            "in_process_validation_executed": False,
            "loopback_socket_validation_executed": False,
            "in_process_validation": {},
            "loopback_socket_validation": {},
            "guard_checks": source_checks,
            "decision": _decision(source_checks, validations_executed=False),
            "timing_seconds": {
                "load_public_stage149_aggregate": round(loaded_at - started_at, 6),
                "in_process_asgi_validation": 0.0,
                "loopback_socket_validation": 0.0,
                "total": round(checked_at - started_at, 6),
            },
        }
        return {**report, "public_safe_contract": _public_safe_contract(report)}

    in_process_started = time.perf_counter()
    in_process = _run_in_process_validation()
    socket_started = time.perf_counter()
    loopback = _run_loopback_socket_validation()
    validated_at = time.perf_counter()
    source_unchanged = _load_json_object(stage149_protocol_path) == source
    guards = _guard_checks(
        report=preliminary,
        source_checks=source_checks,
        source_summary=source_summary,
        contract=preliminary["implementation_contract"],
        in_process=in_process,
        loopback=loopback,
        source_unchanged=source_unchanged,
    )
    report = {
        **preliminary,
        "in_process_validation_executed": True,
        "loopback_socket_validation_executed": True,
        "source_unchanged_after_validation": source_unchanged,
        "in_process_validation": in_process,
        "loopback_socket_validation": loopback,
        "guard_checks": guards,
        "decision": _decision(guards, validations_executed=True),
        "timing_seconds": {
            "load_public_stage149_aggregate": round(loaded_at - started_at, 6),
            "in_process_asgi_validation": round(socket_started - in_process_started, 6),
            "loopback_socket_validation": round(validated_at - socket_started, 6),
            "total": round(validated_at - started_at, 6),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_agent_http_transport_validation_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridAgentHttpTransportValidationVisualization]:
    """Write Stage150 public-safe SVG charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    in_process = report.get("in_process_validation") or {}
    loopback = report.get("loopback_socket_validation") or {}
    charts = {
        "stage150_source_gate.svg": _chart(
            "Stage150 Stage149 source gate",
            _flag_bars(
                report.get("stage149_summary") or {},
                (
                    "source_identity_valid",
                    "all_source_guards_passed",
                    "transport_protocol_frozen",
                    "local_fastapi_implementation_allowed_next",
                    "network_service_implemented",
                ),
            ),
        ),
        "stage150_route_surface.svg": _chart(
            "Stage150 exact route surface",
            [
                _bar("implemented routes", in_process.get("route_count", 0)),
                _bar("unexpected routes", in_process.get("unexpected_route_count", 0)),
            ],
            x_label="route count",
        ),
        "stage150_status_mapping.svg": _chart(
            "Stage150 HTTP status mapping",
            [
                _bar(str(name), status)
                for name, status in (in_process.get("status_mapping") or {}).items()
            ],
            x_label="HTTP status",
        ),
        "stage150_body_limit.svg": _chart(
            "Stage150 body cap boundary",
            [
                _bar("exact cap status", in_process.get("exact_body_cap_status", 0)),
                _bar("declared overflow status", in_process.get("declared_overflow_status", 0)),
                _bar("streamed overflow status", in_process.get("streamed_overflow_status", 0)),
            ],
            x_label="HTTP status",
        ),
        "stage150_overload.svg": _chart(
            "Stage150 nonblocking overload",
            [
                _bar("attempted", in_process.get("overload_attempt_count", 0)),
                _bar("completed", in_process.get("overload_completed_count", 0)),
                _bar("capacity rejected", in_process.get("overload_rejected_count", 0)),
                _bar("waiting", in_process.get("application_waiting_count", 0)),
            ],
            x_label="request count",
        ),
        "stage150_disconnect.svg": _chart(
            "Stage150 predispatch disconnect",
            [
                _bar("ASGI response frames", in_process.get("disconnect_response_frame_count", 0)),
                _bar("runtime calls", in_process.get("disconnect_runtime_call_count", 0)),
                _bar("facade cancellations", in_process.get("disconnect_cancel_count", 0)),
            ],
            x_label="count",
        ),
        "stage150_shutdown.svg": _chart(
            "Stage150 natural shutdown",
            _flag_bars(
                in_process,
                (
                    "shutdown_observed_draining",
                    "shutdown_rejected_new_request",
                    "shutdown_waited_for_in_flight",
                    "shutdown_natural_completion",
                    "shutdown_closed",
                ),
            ),
        ),
        "stage150_loopback_socket.svg": _chart(
            "Stage150 real loopback socket smoke",
            _flag_bars(
                loopback,
                (
                    "bound_to_loopback",
                    "server_started",
                    "http_1_1_observed",
                    "server_stopped",
                    "port_rebind_succeeded",
                    "access_log_disabled",
                ),
            ),
        ),
        "stage150_closed_boundaries.svg": _chart(
            "Stage150 closed boundaries",
            _flag_bars(
                report.get("decision") or {},
                (
                    "runtime_registered_as_default",
                    "remote_exposure_authorized",
                    "test_gate_opened",
                    "queue_actions_enabled",
                    "retry_actions_enabled",
                    "fallback_strategies_enabled",
                ),
            ),
        ),
        "stage150_guard_check_status.svg": _chart(
            "Stage150 validation guard checks",
            [
                BarDatum(
                    label=str(check["name"]),
                    value=1.0 if check["passed"] else 0.0,
                    value_label="passed" if check["passed"] else "failed",
                )
                for check in report.get("guard_checks", [])
            ],
            width=2600,
            margin_left=1480,
        ),
    }
    artifacts: list[PrimeQAHybridAgentHttpTransportValidationVisualization] = []
    for name, content in charts.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridAgentHttpTransportValidationVisualization(name=name, path=str(path))
        )
    return artifacts


def _run_in_process_validation() -> dict[str, Any]:
    disabled_runtime = _SyntheticRuntime()
    disabled_app = create_primeqa_hybrid_agent_http_app(
        settings=ProjectSettings(_env_file=None),
        bootstrap_result=_bootstrap_result(runtime=disabled_runtime, active=True),
        log_sink=_RecordingLogSink(),
    )
    with TestClient(disabled_app) as client:
        disabled_live = client.get("/health/live")
        disabled_ready = client.get("/health/ready")
        disabled_answer = client.post(
            "/v1/agent/answers",
            json={"request_handle": "private-disabled", "text": "Private question"},
        )

    complete_runtime = _SyntheticRuntime()
    complete_sink = _RecordingLogSink()
    complete_app = _active_app(complete_runtime, sink=complete_sink)
    valid = json.dumps(
        {"request_handle": "private-complete", "text": "Private question"},
        separators=(",", ":"),
    ).encode("utf-8")
    exact_body = valid + (b" " * (32768 - len(valid)))
    over_body = exact_body + b" "
    with TestClient(complete_app, raise_server_exceptions=False) as client:
        ready = client.get("/health/ready")
        complete = client.post(
            "/v1/agent/answers",
            json={"request_handle": "private-complete", "text": "Private question"},
        )
        malformed = client.post(
            "/v1/agent/answers",
            content=b"not-json",
            headers={"content-type": "application/json"},
        )
        unsupported = client.post(
            "/v1/agent/answers",
            content=b"{}",
            headers={"content-type": "text/plain"},
        )
        invalid = client.post(
            "/v1/agent/answers",
            json={"request_handle": 1, "text": "Private question"},
        )
        exact_cap = client.post(
            "/v1/agent/answers",
            content=exact_body,
            headers={"content-type": "application/json"},
        )
        declared_over = client.post(
            "/v1/agent/answers",
            content=over_body,
            headers={"content-type": "application/json"},
        )
        streamed_over = client.post(
            "/v1/agent/answers",
            content=over_body,
            headers={"content-type": "application/json", "content-length": "1"},
        )
        missing = client.get("/private-missing-path")
        wrong_method = client.get("/v1/agent/answers")

    refusal_runtime = _SyntheticRuntime(refused=True)
    refusal_app = _active_app(refusal_runtime)
    with TestClient(refusal_app) as client:
        refusal = client.post(
            "/v1/agent/answers",
            json={"request_handle": "private-refusal", "text": "Private question"},
        )

    capacity_runtime = _CapacityRuntime()
    capacity_app = _active_app(capacity_runtime)
    with TestClient(capacity_app) as client:
        capacity = client.post(
            "/v1/agent/answers",
            json={"request_handle": "private-capacity", "text": "Private question"},
        )

    secret = "private downstream exception detail"
    error_runtime = _SyntheticRuntime(error=RuntimeError(secret))
    error_app = _active_app(error_runtime)
    with TestClient(error_app, raise_server_exceptions=False) as client:
        failure = client.post(
            "/v1/agent/answers",
            json={"request_handle": "private-failure", "text": secret},
        )

    overload = asyncio.run(_run_overload_validation())
    disconnect = _run_disconnect_validation()
    shutdown = asyncio.run(_run_shutdown_validation())
    log_payloads = [event.to_public_dict() for event in complete_sink.events]
    status_mapping = {
        "complete": complete.status_code,
        "refusal": refusal.status_code,
        "malformed_json": malformed.status_code,
        "unsupported_media_type": unsupported.status_code,
        "invalid_request": invalid.status_code,
        "capacity_exceeded": capacity.status_code,
        "unexpected_error": failure.status_code,
        "not_found": missing.status_code,
        "method_not_allowed": wrong_method.status_code,
    }
    return {
        "route_count": len(complete_app.routes),
        "unexpected_route_count": max(0, len(complete_app.routes) - 3),
        "route_signature": [
            {"method": next(iter(route.methods)), "path": route.path}
            for route in complete_app.routes
        ],
        "disabled_liveness_status": disabled_live.status_code,
        "disabled_readiness_status": disabled_ready.status_code,
        "disabled_answer_status": disabled_answer.status_code,
        "disabled_runtime_call_count": disabled_runtime.call_count,
        "disabled_facade_created": disabled_app.state.agent_http_transport.facade is not None,
        "active_readiness_status": ready.status_code,
        "complete_response_schema_exact": set(complete.json())
        == {"request_handle", "text", "refused", "citations"},
        "complete_citation_schema_exact": set(complete.json()["citations"][0])
        == {"document_reference", "title", "rank", "evidence_score"},
        "refusal_is_http_200": refusal.status_code == 200,
        "refusal_has_zero_citations": refusal.json()["citations"] == [],
        "status_mapping": status_mapping,
        "all_error_envelopes_exact": all(
            set(response.json()) == {"error"}
            and set(response.json()["error"]) == {"code", "message"}
            for response in (
                malformed,
                unsupported,
                invalid,
                capacity,
                failure,
                missing,
                wrong_method,
            )
        ),
        "unknown_error_content_hidden": secret not in failure.text,
        "exact_body_cap_status": exact_cap.status_code,
        "declared_overflow_status": declared_over.status_code,
        "streamed_overflow_status": streamed_over.status_code,
        "complete_runtime_call_count": complete_runtime.call_count,
        **overload,
        **disconnect,
        **shutdown,
        "public_log_event_count": len(log_payloads),
        "public_log_field_count": len(log_payloads[0]) if log_payloads else 0,
        "public_log_fields_exact": all(
            set(payload) == set(agent_http_transport_contract()["public_log_fields"])
            for payload in log_payloads
        ),
        "public_log_contains_private_values": any(
            private in json.dumps(log_payloads)
            for private in ("private-complete", "Private question", secret)
        ),
        "private_payload_written_to_report": False,
    }


async def _run_overload_validation() -> dict[str, Any]:
    runtime = _BlockingRuntime(target_count=4)
    app = _active_app(runtime)
    transport = app.state.agent_http_transport
    transport.start()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
        timeout=None,
    ) as client:
        tasks = [
            asyncio.create_task(
                client.post(
                    "/v1/agent/answers",
                    json={"request_handle": f"private-{index}", "text": "Private question"},
                )
            )
            for index in range(4)
        ]
        await asyncio.to_thread(runtime.target_entered.wait)
        rejected = await client.post(
            "/v1/agent/answers",
            json={"request_handle": "private-fifth", "text": "Private question"},
        )
        blocked_counters = transport.counters()
        runtime.release.set()
        completed = list(await asyncio.gather(*tasks))
    transport.shutdown()
    final_counters = transport.counters()
    return {
        "overload_attempt_count": 5,
        "overload_completed_count": sum(response.status_code == 200 for response in completed),
        "overload_rejected_count": int(rejected.status_code == 503),
        "overload_rejection_code": rejected.json()["error"]["code"],
        "overload_runtime_call_count": runtime.call_count,
        "overload_current_in_flight_at_rejection": blocked_counters.current_in_flight,
        "overload_max_observed_in_flight": final_counters.max_observed_in_flight,
        "application_waiting_count": final_counters.application_waiting_request_count,
        "queue_action_count": final_counters.queue_action_count,
        "retry_action_count": final_counters.retry_action_count,
        "fallback_action_count": final_counters.fallback_action_count,
    }


def _run_disconnect_validation() -> dict[str, Any]:
    runtime = _SyntheticRuntime()
    sink = _RecordingLogSink()
    app = _active_app(runtime, sink=sink)
    transport = app.state.agent_http_transport
    transport.start()
    body = b'{"request_handle":"private-disconnect","text":"Private question"}'
    received = [
        {"type": "http.request", "body": body, "more_body": False},
        {"type": "http.disconnect"},
    ]
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return received.pop(0) if received else {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(app(_http_scope(body), receive, send))
    facade = transport.facade
    transport.shutdown()
    return {
        "disconnect_response_frame_count": len(sent),
        "disconnect_runtime_call_count": runtime.call_count,
        "disconnect_cancel_count": (
            facade.counters().cancelled_before_dispatch_count if facade is not None else 0
        ),
        "disconnect_public_event_count": sum(
            event.transport_outcome_code == "client_disconnected" for event in sink.events
        ),
    }


async def _run_shutdown_validation() -> dict[str, Any]:
    runtime = _BlockingRuntime()
    app = _active_app(runtime)
    transport = app.state.agent_http_transport
    transport.start()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
        timeout=None,
    ) as client:
        request_task = asyncio.create_task(
            client.post(
                "/v1/agent/answers",
                json={"request_handle": "private-in-flight", "text": "Private question"},
            )
        )
        await asyncio.to_thread(runtime.target_entered.wait)
        shutdown_task = asyncio.create_task(asyncio.to_thread(transport.shutdown))
        if transport.facade is None:
            raise AssertionError("eligible transport must create a facade")
        await asyncio.to_thread(
            transport.facade.wait_until_state,
            AgentRequestFacadeState.DRAINING,
        )
        readiness = await client.get("/health/ready")
        rejected = await client.post(
            "/v1/agent/answers",
            json={"request_handle": "private-new", "text": "Private question"},
        )
        waited = not shutdown_task.done()
        runtime.release.set()
        completed = await request_task
        await shutdown_task
        closed_rejection = await client.post(
            "/v1/agent/answers",
            json={"request_handle": "private-closed", "text": "Private question"},
        )
    return {
        "shutdown_observed_draining": readiness.json().get("facade_state") == "draining",
        "shutdown_rejected_new_request": (
            rejected.status_code == 503 and rejected.json()["error"]["code"] == "facade_draining"
        ),
        "shutdown_waited_for_in_flight": waited,
        "shutdown_natural_completion": completed.status_code == 200,
        "shutdown_closed": transport.state is AgentHttpTransportState.CLOSED,
        "shutdown_closed_request_status": closed_rejection.status_code,
        "shutdown_closed_request_code": closed_rejection.json()["error"]["code"],
        "shutdown_implicit_timeout_seconds": None,
        "shutdown_force_cancel": False,
    }


def _run_loopback_socket_validation() -> dict[str, Any]:
    runtime = _SyntheticRuntime()
    sink = _RecordingLogSink()
    app = _active_app(runtime, sink=sink)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    host, port = listener.getsockname()
    config = create_primeqa_hybrid_agent_uvicorn_config(app=app, port=port)
    server = uvicorn.Server(config)
    server_errors: list[str] = []

    def run_server() -> None:
        try:
            server.run(sockets=[listener])
        except BaseException as error:
            server_errors.append(type(error).__name__)

    server_thread = Thread(target=run_server, name="stage150-loopback-uvicorn")
    server_thread.start()
    while not server.started:
        if not server_thread.is_alive():
            raise RuntimeError(f"loopback Uvicorn stopped during startup: {server_errors}")
        time.sleep(0.001)

    try:
        with httpx.Client(base_url=f"http://{host}:{port}", timeout=None) as client:
            live = client.get("/health/live")
            ready = client.get("/health/ready")
            answer = client.post(
                "/v1/agent/answers",
                json={"request_handle": "private-socket", "text": "Private question"},
            )
            unsupported = client.post(
                "/v1/agent/answers",
                content=b"{}",
                headers={"content-type": "text/plain"},
            )
    finally:
        server.should_exit = True
        server_thread.join()
        listener.close()

    rebind = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    rebind.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        rebind.bind(("127.0.0.1", port))
        port_rebind_succeeded = True
    finally:
        rebind.close()
    public_logs = [event.to_public_dict() for event in sink.events]
    return {
        "bound_to_loopback": host == "127.0.0.1" and config.host == "127.0.0.1",
        "ephemeral_port_assigned": isinstance(port, int) and port > 0,
        "port_value_persisted": False,
        "server_started": server.started,
        "server_error_types": server_errors,
        "http_1_1_observed": all(
            response.http_version == "HTTP/1.1" for response in (live, ready, answer, unsupported)
        ),
        "liveness_status": live.status_code,
        "readiness_status": ready.status_code,
        "answer_status": answer.status_code,
        "answer_schema_exact": set(answer.json())
        == {"request_handle", "text", "refused", "citations"},
        "unsupported_media_status": unsupported.status_code,
        "runtime_call_count": runtime.call_count,
        "server_stopped": not server_thread.is_alive(),
        "transport_closed": (
            app.state.agent_http_transport.state is AgentHttpTransportState.CLOSED
        ),
        "port_rebind_succeeded": port_rebind_succeeded,
        "access_log_disabled": config.access_log is False,
        "server_header_disabled": config.server_header is False,
        "proxy_headers_disabled": config.proxy_headers is False,
        "single_worker": config.workers == 1,
        "implicit_shutdown_timeout_seconds": config.timeout_graceful_shutdown,
        "public_log_event_count": len(public_logs),
        "public_log_field_count": len(public_logs[0]) if public_logs else 0,
        "public_log_contains_private_values": any(
            private in json.dumps(public_logs) for private in ("private-socket", "Private question")
        ),
    }


def _active_app(
    runtime: _SyntheticRuntime,
    *,
    sink: _RecordingLogSink | None = None,
):
    return create_primeqa_hybrid_agent_http_app(
        settings=ProjectSettings(
            enable_concurrent_sidecar_agent=True,
            enable_local_agent_http_transport=True,
            _env_file=None,
        ),
        bootstrap_result=_bootstrap_result(runtime=runtime, active=True),
        log_sink=sink or _RecordingLogSink(),
    )


def _bootstrap_result(
    *,
    runtime: _SyntheticRuntime,
    active: bool,
) -> PrimeQAHybridConcurrentRuntimeBootstrapResult:
    return PrimeQAHybridConcurrentRuntimeBootstrapResult(
        runtime=runtime,
        startup_trace=PublicSafeConcurrentRuntimeStartupTrace(
            runtime_mode="optional_sidecar_agent_concurrent_four_request",
            settings_field="enable_concurrent_sidecar_agent",
            environment_flag="TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT",
            activation_requested=active,
            activation_state="eligible" if active else "disabled",
            source_validation_state="eligible" if active else "not_evaluated_disabled",
            slo_profile_id="strict_practical_b_concurrency4_v1",
            max_in_flight=4,
            warm_resources_ready=active,
            resources_initialized=active,
            runtime_activated=active,
            resource_factory_build_count=1 if active else 0,
            warmup_request_count=1 if active else 0,
            warmup_arrival_pattern="warmup_single_request" if active else "",
            warmup_candidate_pool_depth=400 if active else 0,
            warmup_retrieval_latency_ms=1.0 if active else 0.0,
            warmup_end_to_end_latency_ms=2.0 if active else 0.0,
            rejection_reasons=() if active else ("not_active",),
        ),
        resource_summary=None,
        source_evaluation=None,
    )


def _synthetic_run(request_handle: str, *, refused: bool = False) -> _SyntheticRuntimeRun:
    return _SyntheticRuntimeRun(
        verified_answer=GeneratedAnswer(
            question_id=request_handle,
            answer=(
                "I do not have enough retrieved evidence to answer this question."
                if refused
                else "Configure the adapter. [doc-1]"
            ),
            citations=(
                []
                if refused
                else [
                    AnswerCitation(
                        document_id="doc-1",
                        title="Adapter configuration",
                        retrieval_rank=1,
                        evidence_score=9.5,
                    )
                ]
            ),
            refused=refused,
        ),
        public_safe_trace=_runtime_trace(
            admission_state="admitted",
            terminal_state="refuse" if refused else "complete",
            candidate_pool_depth=400,
        ),
    )


def _runtime_trace(
    *,
    admission_state: str,
    terminal_state: str,
    candidate_pool_depth: int,
) -> PublicSafeConcurrentRuntimeRequestTrace:
    return PublicSafeConcurrentRuntimeRequestTrace(
        runtime_mode="optional_sidecar_agent_concurrent_four_request",
        activation_requested=True,
        activation_state="eligible",
        slo_profile_id="strict_practical_b_concurrency4_v1",
        warm_resources_ready=True,
        concurrency_limit=4,
        in_flight_at_admission=4 if admission_state == "rejected_capacity" else 1,
        admission_state=admission_state,
        arrival_pattern=ConcurrentArrivalPattern.APPLICATION.value,
        candidate_pool_depth=candidate_pool_depth,
        retrieval_latency_ms=1.0 if candidate_pool_depth else 0.0,
        end_to_end_latency_ms=2.0,
        latency_budget_passed=True,
        terminal_state=terminal_state,
    )


def _http_scope(body: bytes) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/agent/answers",
        "raw_path": b"/v1/agent/answers",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"127.0.0.1"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }


def _stage149_summary(source: Mapping[str, Any]) -> dict[str, Any]:
    decision = source.get("decision") or {}
    public = source.get("public_safe_contract") or {}
    protocol = source.get("frozen_protocol") or {}
    identity = protocol.get("transport_identity") or {}
    checks = source.get("guard_checks") or []
    passed_count = sum(check.get("passed") is True for check in checks)
    identity_valid = (
        source.get("stage") == "Stage 149"
        and source.get("protocol_id") == _SOURCE_PROTOCOL_ID
        and decision.get("status") == _SOURCE_STATUS
    )
    closed = (
        decision.get("network_service_implemented") is False
        and decision.get("runtime_registered_as_default") is False
        and decision.get("remote_deployment_authorized") is False
        and decision.get("test_gate_opened") is False
        and decision.get("queue_actions_enabled") is False
        and decision.get("retry_actions_enabled") is False
        and decision.get("fallback_strategies_enabled") is False
        and public.get("network_service_started") is False
        and public.get("network_port_bound") is False
        and public.get("test_split_loaded") is False
        and public.get("forbidden_keys_found") == []
    )
    return {
        "source_identity_valid": identity_valid,
        "source_guard_count": len(checks),
        "source_passed_guard_count": passed_count,
        "all_source_guards_passed": (
            len(checks) == _EXPECTED_SOURCE_GUARDS and passed_count == _EXPECTED_SOURCE_GUARDS
        ),
        "transport_protocol_frozen": decision.get("agent_http_transport_protocol_frozen"),
        "local_fastapi_implementation_allowed_next": decision.get(
            "local_fastapi_implementation_allowed_next"
        ),
        "local_loopback_only": decision.get("local_loopback_only"),
        "binding_host": identity.get("binding_host"),
        "http_protocol": identity.get("protocol"),
        "network_service_implemented": decision.get("network_service_implemented"),
        "runtime_registered_as_default": decision.get("runtime_registered_as_default"),
        "closed_boundaries_preserved": closed,
        "test_split_loaded": public.get("test_split_loaded"),
        "test_metrics_run": public.get("test_metrics_run"),
        "forbidden_keys_found": list(public.get("forbidden_keys_found") or []),
    }


def _source_gate_checks(
    *,
    source_summary: Mapping[str, Any],
    user_confirmed_validation: bool,
    confirmation_note: str,
) -> list[dict[str, Any]]:
    return [
        _check(
            "stage150_user_confirmed", user_confirmed_validation, user_confirmed_validation, True
        ),
        _check(
            "stage150_confirmation_note_present",
            bool(confirmation_note.strip()),
            bool(confirmation_note.strip()),
            True,
        ),
        _check(
            "stage149_source_identity_valid",
            source_summary.get("source_identity_valid") is True,
            source_summary.get("source_identity_valid"),
            True,
        ),
        _check(
            "stage149_all_39_guards_passed",
            source_summary.get("all_source_guards_passed") is True,
            source_summary.get("source_passed_guard_count"),
            _EXPECTED_SOURCE_GUARDS,
        ),
        _check(
            "stage149_protocol_frozen_and_implementation_authorized",
            source_summary.get("transport_protocol_frozen") is True
            and source_summary.get("local_fastapi_implementation_allowed_next") is True,
            [
                source_summary.get("transport_protocol_frozen"),
                source_summary.get("local_fastapi_implementation_allowed_next"),
            ],
            [True, True],
        ),
        _check(
            "stage149_loopback_http11_exact",
            source_summary.get("local_loopback_only") is True
            and source_summary.get("binding_host") == "127.0.0.1"
            and source_summary.get("http_protocol") == "HTTP/1.1",
            [
                source_summary.get("local_loopback_only"),
                source_summary.get("binding_host"),
                source_summary.get("http_protocol"),
            ],
            [True, "127.0.0.1", "HTTP/1.1"],
        ),
        _check(
            "stage149_closed_boundaries_preserved",
            source_summary.get("closed_boundaries_preserved") is True,
            source_summary.get("closed_boundaries_preserved"),
            True,
        ),
    ]


def _guard_checks(
    *,
    report: Mapping[str, Any],
    source_checks: list[dict[str, Any]],
    source_summary: Mapping[str, Any],
    contract: Mapping[str, Any],
    in_process: Mapping[str, Any],
    loopback: Mapping[str, Any],
    source_unchanged: bool,
) -> list[dict[str, Any]]:
    checks = [*source_checks]
    checks.extend(
        [
            _check(
                "implementation_framework_versions_present",
                all(
                    bool(contract.get(key))
                    for key in ("framework_version", "starlette_version", "uvicorn_version")
                ),
                [
                    contract.get("framework_version"),
                    contract.get("starlette_version"),
                    contract.get("uvicorn_version"),
                ],
                "three installed versions",
            ),
            _check(
                "transport_flag_default_false_and_requires_concurrent",
                contract.get("default_enabled") is False
                and contract.get("requires_concurrent_runtime") is True,
                [contract.get("default_enabled"), contract.get("requires_concurrent_runtime")],
                [False, True],
            ),
            _check(
                "implementation_loopback_http11_exact",
                contract.get("binding_host") == "127.0.0.1"
                and contract.get("http_protocol") == "HTTP/1.1",
                [contract.get("binding_host"), contract.get("http_protocol")],
                ["127.0.0.1", "HTTP/1.1"],
            ),
            _check(
                "implementation_routes_exact",
                contract.get("routes")
                == [
                    {
                        "route_id": "agent_answer",
                        "method": "POST",
                        "path": "/v1/agent/answers",
                    },
                    {"route_id": "liveness", "method": "GET", "path": "/health/live"},
                    {"route_id": "readiness", "method": "GET", "path": "/health/ready"},
                ],
                contract.get("routes"),
                "Stage149 exact routes",
            ),
            _check(
                "implementation_limits_exact",
                [
                    contract.get("raw_body_max_bytes"),
                    contract.get("request_handle_max_chars"),
                    contract.get("title_max_chars"),
                    contract.get("text_max_chars"),
                ]
                == [32768, 128, 512, 24576],
                [
                    contract.get("raw_body_max_bytes"),
                    contract.get("request_handle_max_chars"),
                    contract.get("title_max_chars"),
                    contract.get("text_max_chars"),
                ],
                [32768, 128, 512, 24576],
            ),
            _check(
                "implementation_has_no_wait_queue_or_timeouts",
                contract.get("application_waiting_queue") is False
                and contract.get("request_timeout_seconds") is None
                and contract.get("implicit_shutdown_timeout_seconds") is None,
                [
                    contract.get("application_waiting_queue"),
                    contract.get("request_timeout_seconds"),
                    contract.get("implicit_shutdown_timeout_seconds"),
                ],
                [False, None, None],
            ),
            _check(
                "implementation_access_log_disabled_and_public_fields_exact",
                contract.get("access_log_enabled") is False
                and len(contract.get("public_log_fields") or []) == _PUBLIC_LOG_FIELD_COUNT,
                [
                    contract.get("access_log_enabled"),
                    len(contract.get("public_log_fields") or []),
                ],
                [False, _PUBLIC_LOG_FIELD_COUNT],
            ),
            _check(
                "disabled_app_live_not_ready_and_no_runtime",
                in_process.get("disabled_liveness_status") == 200
                and in_process.get("disabled_readiness_status") == 503
                and in_process.get("disabled_answer_status") == 503
                and in_process.get("disabled_runtime_call_count") == 0
                and in_process.get("disabled_facade_created") is False,
                [
                    in_process.get("disabled_liveness_status"),
                    in_process.get("disabled_readiness_status"),
                    in_process.get("disabled_answer_status"),
                    in_process.get("disabled_runtime_call_count"),
                    in_process.get("disabled_facade_created"),
                ],
                [200, 503, 503, 0, False],
            ),
            _check(
                "active_app_has_exact_three_routes",
                in_process.get("route_count") == 3
                and in_process.get("unexpected_route_count") == 0,
                [in_process.get("route_count"), in_process.get("unexpected_route_count")],
                [3, 0],
            ),
            _check(
                "active_readiness_and_response_schema_pass",
                in_process.get("active_readiness_status") == 200
                and in_process.get("complete_response_schema_exact") is True
                and in_process.get("complete_citation_schema_exact") is True,
                [
                    in_process.get("active_readiness_status"),
                    in_process.get("complete_response_schema_exact"),
                    in_process.get("complete_citation_schema_exact"),
                ],
                [200, True, True],
            ),
            _check(
                "domain_refusal_is_http_200_with_zero_citations",
                in_process.get("refusal_is_http_200") is True
                and in_process.get("refusal_has_zero_citations") is True,
                [
                    in_process.get("refusal_is_http_200"),
                    in_process.get("refusal_has_zero_citations"),
                ],
                [True, True],
            ),
            _check(
                "input_and_framework_status_mapping_exact",
                in_process.get("status_mapping")
                == {
                    "complete": 200,
                    "refusal": 200,
                    "malformed_json": 400,
                    "unsupported_media_type": 415,
                    "invalid_request": 422,
                    "capacity_exceeded": 503,
                    "unexpected_error": 500,
                    "not_found": 404,
                    "method_not_allowed": 405,
                },
                in_process.get("status_mapping"),
                "exact status map",
            ),
            _check(
                "error_envelopes_exact_and_unknown_content_hidden",
                in_process.get("all_error_envelopes_exact") is True
                and in_process.get("unknown_error_content_hidden") is True,
                [
                    in_process.get("all_error_envelopes_exact"),
                    in_process.get("unknown_error_content_hidden"),
                ],
                [True, True],
            ),
            _check(
                "body_cap_enforced_declared_and_streamed",
                [
                    in_process.get("exact_body_cap_status"),
                    in_process.get("declared_overflow_status"),
                    in_process.get("streamed_overflow_status"),
                ]
                == [200, 413, 413],
                [
                    in_process.get("exact_body_cap_status"),
                    in_process.get("declared_overflow_status"),
                    in_process.get("streamed_overflow_status"),
                ],
                [200, 413, 413],
            ),
            _check(
                "overload_admits_four_rejects_fifth",
                in_process.get("overload_attempt_count") == 5
                and in_process.get("overload_completed_count") == 4
                and in_process.get("overload_rejected_count") == 1
                and in_process.get("overload_rejection_code") == "capacity_exceeded"
                and in_process.get("overload_runtime_call_count") == 4,
                [
                    in_process.get("overload_attempt_count"),
                    in_process.get("overload_completed_count"),
                    in_process.get("overload_rejected_count"),
                    in_process.get("overload_rejection_code"),
                    in_process.get("overload_runtime_call_count"),
                ],
                [5, 4, 1, "capacity_exceeded", 4],
            ),
            _check(
                "overload_has_no_application_wait_queue_retry_or_fallback",
                in_process.get("overload_current_in_flight_at_rejection") == 4
                and in_process.get("overload_max_observed_in_flight") == 4
                and in_process.get("application_waiting_count") == 0
                and in_process.get("queue_action_count") == 0
                and in_process.get("retry_action_count") == 0
                and in_process.get("fallback_action_count") == 0,
                [
                    in_process.get("overload_current_in_flight_at_rejection"),
                    in_process.get("overload_max_observed_in_flight"),
                    in_process.get("application_waiting_count"),
                    in_process.get("queue_action_count"),
                    in_process.get("retry_action_count"),
                    in_process.get("fallback_action_count"),
                ],
                [4, 4, 0, 0, 0, 0],
            ),
            _check(
                "predispatch_disconnect_sends_nothing_and_skips_runtime",
                in_process.get("disconnect_response_frame_count") == 0
                and in_process.get("disconnect_runtime_call_count") == 0
                and in_process.get("disconnect_cancel_count") == 1
                and in_process.get("disconnect_public_event_count") == 1,
                [
                    in_process.get("disconnect_response_frame_count"),
                    in_process.get("disconnect_runtime_call_count"),
                    in_process.get("disconnect_cancel_count"),
                    in_process.get("disconnect_public_event_count"),
                ],
                [0, 0, 1, 1],
            ),
            _check(
                "shutdown_drains_rejects_waits_completes_and_closes",
                all(
                    in_process.get(key) is True
                    for key in (
                        "shutdown_observed_draining",
                        "shutdown_rejected_new_request",
                        "shutdown_waited_for_in_flight",
                        "shutdown_natural_completion",
                        "shutdown_closed",
                    )
                ),
                [
                    in_process.get(key)
                    for key in (
                        "shutdown_observed_draining",
                        "shutdown_rejected_new_request",
                        "shutdown_waited_for_in_flight",
                        "shutdown_natural_completion",
                        "shutdown_closed",
                    )
                ],
                [True] * 5,
            ),
            _check(
                "shutdown_uses_no_timeout_or_force_cancel",
                in_process.get("shutdown_implicit_timeout_seconds") is None
                and in_process.get("shutdown_force_cancel") is False,
                [
                    in_process.get("shutdown_implicit_timeout_seconds"),
                    in_process.get("shutdown_force_cancel"),
                ],
                [None, False],
            ),
            _check(
                "closed_transport_rejects_without_executor_error",
                in_process.get("shutdown_closed_request_status") == 503
                and in_process.get("shutdown_closed_request_code") == "facade_closed",
                [
                    in_process.get("shutdown_closed_request_status"),
                    in_process.get("shutdown_closed_request_code"),
                ],
                [503, "facade_closed"],
            ),
            _check(
                "public_logs_have_exact_allowlist_and_no_private_values",
                in_process.get("public_log_event_count", 0) > 0
                and in_process.get("public_log_field_count") == _PUBLIC_LOG_FIELD_COUNT
                and in_process.get("public_log_fields_exact") is True
                and in_process.get("public_log_contains_private_values") is False,
                [
                    in_process.get("public_log_event_count"),
                    in_process.get("public_log_field_count"),
                    in_process.get("public_log_fields_exact"),
                    in_process.get("public_log_contains_private_values"),
                ],
                [">0", _PUBLIC_LOG_FIELD_COUNT, True, False],
            ),
            _check(
                "loopback_server_bound_started_and_http11",
                loopback.get("bound_to_loopback") is True
                and loopback.get("ephemeral_port_assigned") is True
                and loopback.get("port_value_persisted") is False
                and loopback.get("server_started") is True
                and loopback.get("http_1_1_observed") is True
                and loopback.get("server_error_types") == [],
                [
                    loopback.get("bound_to_loopback"),
                    loopback.get("ephemeral_port_assigned"),
                    loopback.get("port_value_persisted"),
                    loopback.get("server_started"),
                    loopback.get("http_1_1_observed"),
                    loopback.get("server_error_types"),
                ],
                [True, True, False, True, True, []],
            ),
            _check(
                "loopback_endpoints_and_runtime_pass",
                [
                    loopback.get("liveness_status"),
                    loopback.get("readiness_status"),
                    loopback.get("answer_status"),
                    loopback.get("unsupported_media_status"),
                    loopback.get("runtime_call_count"),
                ]
                == [200, 200, 200, 415, 1]
                and loopback.get("answer_schema_exact") is True,
                [
                    loopback.get("liveness_status"),
                    loopback.get("readiness_status"),
                    loopback.get("answer_status"),
                    loopback.get("unsupported_media_status"),
                    loopback.get("runtime_call_count"),
                    loopback.get("answer_schema_exact"),
                ],
                [200, 200, 200, 415, 1, True],
            ),
            _check(
                "loopback_server_stopped_transport_closed_and_port_rebound",
                loopback.get("server_stopped") is True
                and loopback.get("transport_closed") is True
                and loopback.get("port_rebind_succeeded") is True,
                [
                    loopback.get("server_stopped"),
                    loopback.get("transport_closed"),
                    loopback.get("port_rebind_succeeded"),
                ],
                [True, True, True],
            ),
            _check(
                "loopback_server_security_and_shutdown_config_exact",
                loopback.get("access_log_disabled") is True
                and loopback.get("server_header_disabled") is True
                and loopback.get("proxy_headers_disabled") is True
                and loopback.get("single_worker") is True
                and loopback.get("implicit_shutdown_timeout_seconds") is None,
                [
                    loopback.get("access_log_disabled"),
                    loopback.get("server_header_disabled"),
                    loopback.get("proxy_headers_disabled"),
                    loopback.get("single_worker"),
                    loopback.get("implicit_shutdown_timeout_seconds"),
                ],
                [True, True, True, True, None],
            ),
            _check(
                "loopback_public_logs_have_no_private_values",
                loopback.get("public_log_event_count", 0) > 0
                and loopback.get("public_log_field_count") == _PUBLIC_LOG_FIELD_COUNT
                and loopback.get("public_log_contains_private_values") is False,
                [
                    loopback.get("public_log_event_count"),
                    loopback.get("public_log_field_count"),
                    loopback.get("public_log_contains_private_values"),
                ],
                [">0", _PUBLIC_LOG_FIELD_COUNT, False],
            ),
            _check(
                "private_payload_not_written_to_report",
                in_process.get("private_payload_written_to_report") is False,
                in_process.get("private_payload_written_to_report"),
                False,
            ),
            _check(
                "source_stage149_file_unchanged_after_validation",
                source_unchanged,
                source_unchanged,
                True,
            ),
            _check(
                "network_default_test_and_recovery_boundaries_closed",
                contract.get("runtime_registered_as_default") is False
                and contract.get("remote_exposure_authorized") is False
                and contract.get("test_access_allowed") is False
                and contract.get("queue_actions_allowed") is False
                and contract.get("retry_actions_allowed") is False
                and contract.get("fallback_strategies_allowed") is False
                and source_summary.get("test_split_loaded") is False
                and source_summary.get("test_metrics_run") is False,
                [
                    contract.get("runtime_registered_as_default"),
                    contract.get("remote_exposure_authorized"),
                    contract.get("test_access_allowed"),
                    contract.get("queue_actions_allowed"),
                    contract.get("retry_actions_allowed"),
                    contract.get("fallback_strategies_allowed"),
                    source_summary.get("test_split_loaded"),
                    source_summary.get("test_metrics_run"),
                ],
                [False] * 8,
            ),
            _check(
                "public_report_contains_no_private_keys",
                _private_keys_found(report) == set(),
                sorted(_private_keys_found(report)),
                [],
            ),
        ]
    )
    return checks


def _decision(
    guards: Sequence[Mapping[str, Any]],
    *,
    validations_executed: bool,
) -> dict[str, Any]:
    failed = [str(check.get("name")) for check in guards if check.get("passed") is not True]
    passed = validations_executed and not failed
    return {
        "status": _FINAL_STATUS if passed else "primeqa_hybrid_local_fastapi_transport_rejected",
        "failed_checks": failed,
        "local_fastapi_transport_implemented": passed,
        "in_process_asgi_validation_passed": passed,
        "real_loopback_socket_validation_passed": passed,
        "disabled_by_default": True,
        "local_loopback_only": True,
        "network_service_persistently_running": False,
        "runtime_registered_as_default": False,
        "runtime_defaultization_allowed_now": False,
        "remote_exposure_authorized": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
        "queue_actions_enabled": False,
        "retry_actions_enabled": False,
        "fallback_strategies_enabled": False,
        "hard_cancellation_enabled": False,
        "implicit_shutdown_timeout_enabled": False,
        "next_direction": _NEXT_DIRECTION if passed else "repair_failed_stage150_guards",
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_kind": "saved_public_stage149_aggregate_plus_synthetic_http_only",
        "train_split_loaded": False,
        "dev_split_loaded": False,
        "test_split_loaded": False,
        "test_metrics_run": False,
        "questions_loaded": False,
        "documents_loaded": False,
        "models_loaded": False,
        "indexes_loaded": False,
        "candidate_pools_built": False,
        "synthetic_runtime_only": True,
        "loopback_socket_opened_temporarily": report.get("loopback_socket_validation_executed")
        is True,
        "loopback_socket_closed_after_validation": (
            (report.get("loopback_socket_validation") or {}).get("server_stopped") is True
            if report.get("loopback_socket_validation_executed") is True
            else False
        ),
        "network_service_persistently_running": False,
        "port_value_persisted": False,
        "private_keys_found": sorted(_private_keys_found(report)),
    }


def _private_keys_found(value: Any) -> set[str]:
    forbidden = {
        "answer",
        "answer_text",
        "citations",
        "document_id",
        "document_reference",
        "question_text",
        "request_handle",
        "response_text",
        "text",
        "title",
    }
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in forbidden:
                found.add(key_text)
            found.update(_private_keys_found(child))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for child in value:
            found.update(_private_keys_found(child))
    return found


def _bar(label: str, value: Any) -> BarDatum:
    numeric = float(value or 0)
    return BarDatum(label=label, value=numeric, value_label=str(value))


def _flag_bars(source: Mapping[str, Any], keys: Sequence[str]) -> list[BarDatum]:
    return [
        BarDatum(
            label=key,
            value=1.0 if source.get(key) is True else 0.0,
            value_label="true" if source.get(key) is True else "false",
        )
        for key in keys
    ]


def _chart(
    title: str,
    bars: Sequence[BarDatum],
    *,
    x_label: str = "1 means true",
    width: int = 1900,
    margin_left: int = 980,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        width=width,
        margin_left=margin_left,
    )


def _check(name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "expected": expected}


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
