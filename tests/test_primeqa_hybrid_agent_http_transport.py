from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

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
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.domain.answer import AnswerCitation, GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQARuntimeQuery


@dataclass
class RecordingLogSink:
    events: list[PublicSafeAgentHttpTransportEvent] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)

    def emit(self, event: PublicSafeAgentHttpTransportEvent) -> None:
        event.to_public_dict()
        with self.lock:
            self.events.append(event)


@dataclass(frozen=True)
class SyntheticRuntimeRun:
    verified_answer: GeneratedAnswer
    public_safe_trace: PublicSafeConcurrentRuntimeRequestTrace


class SyntheticRuntime:
    def __init__(self, *, refused: bool = False, error: RuntimeError | None = None) -> None:
        self.refused = refused
        self.error = error
        self.call_count = 0
        self.received: list[PrimeQARuntimeQuery] = []
        self.lock = Lock()

    def run(
        self,
        question: PrimeQARuntimeQuery,
        *,
        arrival_pattern: ConcurrentArrivalPattern,
    ) -> SyntheticRuntimeRun:
        assert arrival_pattern is ConcurrentArrivalPattern.APPLICATION
        with self.lock:
            self.call_count += 1
            self.received.append(question)
        if self.error is not None:
            raise self.error
        return _runtime_run(question.id, refused=self.refused)


class CapacityRuntime(SyntheticRuntime):
    def run(
        self,
        question: PrimeQARuntimeQuery,
        *,
        arrival_pattern: ConcurrentArrivalPattern,
    ) -> SyntheticRuntimeRun:
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


class BlockingRuntime(SyntheticRuntime):
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
    ) -> SyntheticRuntimeRun:
        assert arrival_pattern is ConcurrentArrivalPattern.APPLICATION
        with self.lock:
            self.call_count += 1
            self.received.append(question)
            if self.call_count == self.target_count:
                self.target_entered.set()
        self.release.wait()
        return _runtime_run(question.id)


def test_local_http_transport_setting_defaults_false_and_requires_concurrent_runtime(
    monkeypatch,
) -> None:
    monkeypatch.delenv("TS_RAG_ENABLE_LOCAL_AGENT_HTTP_TRANSPORT", raising=False)

    assert ProjectSettings(_env_file=None).enable_local_agent_http_transport is False
    assert (
        ProjectSettings(
            enable_concurrent_sidecar_agent=True,
            enable_local_agent_http_transport=True,
        ).enable_local_agent_http_transport
        is True
    )
    with pytest.raises(ValidationError, match="requires the concurrent sidecar runtime"):
        ProjectSettings(enable_local_agent_http_transport=True)


@pytest.mark.parametrize("raw", ["1", "yes", "on", "enabled", ""])
def test_local_http_transport_setting_rejects_ambiguous_values(
    monkeypatch,
    raw: str,
) -> None:
    monkeypatch.setenv("TS_RAG_ENABLE_LOCAL_AGENT_HTTP_TRANSPORT", raw)

    with pytest.raises(ValidationError, match="must be explicit true or false"):
        ProjectSettings(_env_file=None)


def test_disabled_transport_exposes_live_but_not_ready_and_builds_no_facade() -> None:
    runtime = SyntheticRuntime()
    sink = RecordingLogSink()
    app = create_primeqa_hybrid_agent_http_app(
        settings=ProjectSettings(_env_file=None),
        bootstrap_result=_bootstrap_result(runtime=runtime, active=True),
        log_sink=sink,
    )

    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "live"}
        readiness = client.get("/health/ready")
        answer = client.post(
            "/v1/agent/answers",
            json={"request_handle": "disabled", "text": "Question"},
        )

        assert readiness.status_code == 503
        assert readiness.json() == {"status": "not_ready", "facade_state": "disabled"}
        assert answer.status_code == 503
        assert answer.json()["error"]["code"] == "facade_not_active"
        assert app.state.agent_http_transport.facade is None
        assert runtime.call_count == 0

    assert app.state.agent_http_transport.state is AgentHttpTransportState.CLOSED


def test_enabled_transport_maps_complete_answer_and_refusal_as_http_200() -> None:
    complete_runtime = SyntheticRuntime()
    complete_app = _active_app(complete_runtime)

    with TestClient(complete_app) as client:
        assert client.get("/health/ready").json() == {
            "status": "ready",
            "facade_state": "accepting",
        }
        response = client.post(
            "/v1/agent/answers",
            json={
                "request_handle": "complete",
                "title": "Adapter",
                "text": "How do I configure it?",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "request_handle": "complete",
        "text": "Configure the adapter. [doc-1]",
        "refused": False,
        "citations": [
            {
                "document_reference": "doc-1",
                "title": "Adapter configuration",
                "rank": 1,
                "evidence_score": 9.5,
            }
        ],
    }
    assert complete_runtime.received[0].model_dump() == {
        "id": "complete",
        "title": "Adapter",
        "text": "How do I configure it?",
    }

    refusal_app = _active_app(SyntheticRuntime(refused=True))
    with TestClient(refusal_app) as client:
        refusal = client.post(
            "/v1/agent/answers",
            json={"request_handle": "refusal", "text": "Unknown issue"},
        )
    assert refusal.status_code == 200
    assert refusal.json()["refused"] is True
    assert refusal.json()["citations"] == []


@pytest.mark.parametrize(
    ("content", "headers", "status", "code"),
    [
        (b"not-json", {"content-type": "application/json"}, 400, "malformed_json"),
        (
            b'{"request_handle":"a","request_handle":"b","text":"q"}',
            {"content-type": "application/json"},
            400,
            "malformed_json",
        ),
        (b"{}", {"content-type": "text/plain"}, 415, "unsupported_media_type"),
        (
            b"{}",
            {"content-type": "application/json; charset=utf-16"},
            415,
            "unsupported_media_type",
        ),
        (b"{}", {"content-type": "application/json"}, 422, "invalid_request"),
        (
            b'{"request_handle":1,"text":"q"}',
            {"content-type": "application/json"},
            422,
            "invalid_request",
        ),
        (
            b'{"request_handle":"h","text":" ","unknown":true}',
            {"content-type": "application/json"},
            422,
            "invalid_request",
        ),
    ],
)
def test_transport_rejects_malformed_media_and_schema_inputs_before_runtime(
    content: bytes,
    headers: dict[str, str],
    status: int,
    code: str,
) -> None:
    runtime = SyntheticRuntime()
    app = _active_app(runtime)

    with TestClient(app) as client:
        response = client.post("/v1/agent/answers", content=content, headers=headers)

    assert response.status_code == status
    assert response.json() == {
        "error": {
            "code": code,
            "message": response.json()["error"]["message"],
        }
    }
    assert set(response.json()["error"]) == {"code", "message"}
    assert runtime.call_count == 0


def test_body_limit_accepts_exact_cap_and_rejects_declared_and_streamed_overflow() -> None:
    runtime = SyntheticRuntime()
    app = _active_app(runtime)
    valid = json.dumps(
        {"request_handle": "exact", "text": "Question"},
        separators=(",", ":"),
    ).encode("utf-8")
    exact = valid + (b" " * (32768 - len(valid)))
    over = exact + b" "

    with TestClient(app) as client:
        exact_response = client.post(
            "/v1/agent/answers",
            content=exact,
            headers={"content-type": "application/json"},
        )
        declared_over = client.post(
            "/v1/agent/answers",
            content=over,
            headers={"content-type": "application/json"},
        )
        streamed_over = client.post(
            "/v1/agent/answers",
            content=over,
            headers={"content-type": "application/json", "content-length": "1"},
        )

    assert exact_response.status_code == 200
    assert declared_over.status_code == 413
    assert declared_over.json()["error"]["code"] == "request_body_too_large"
    assert streamed_over.status_code == 413
    assert streamed_over.json()["error"]["code"] == "request_body_too_large"
    assert runtime.call_count == 1


def test_facade_capacity_and_unknown_error_map_without_answer_or_exception_leakage() -> None:
    capacity_runtime = CapacityRuntime()
    capacity_app = _active_app(capacity_runtime)
    with TestClient(capacity_app) as client:
        capacity = client.post(
            "/v1/agent/answers",
            json={"request_handle": "capacity", "text": "Question"},
        )
    assert capacity.status_code == 503
    assert capacity.json()["error"]["code"] == "capacity_exceeded"

    secret = "private downstream exception content"
    error_app = _active_app(SyntheticRuntime(error=RuntimeError(secret)))
    with TestClient(error_app, raise_server_exceptions=False) as client:
        failure = client.post(
            "/v1/agent/answers",
            json={"request_handle": "failure", "text": secret},
        )
    serialized = failure.text
    assert failure.status_code == 500
    assert failure.json()["error"]["code"] == "internal_error"
    assert secret not in serialized
    assert "RuntimeError" not in serialized


def test_four_admitted_requests_block_fifth_without_application_waiting_queue() -> None:
    runtime = BlockingRuntime(target_count=4)
    app = _active_app(runtime)
    responses, rejected, counters_while_blocked = asyncio.run(
        _run_four_request_overload_scenario(app=app, runtime=runtime)
    )

    assert rejected.status_code == 503
    assert rejected.json()["error"]["code"] == "capacity_exceeded"
    assert len(responses) == 4
    assert all(response.status_code == 200 for response in responses)
    assert runtime.call_count == 4
    assert counters_while_blocked.current_in_flight == 4
    assert counters_while_blocked.max_observed_in_flight == 4
    assert counters_while_blocked.application_waiting_request_count == 0
    assert counters_while_blocked.queue_action_count == 0
    assert counters_while_blocked.retry_action_count == 0
    assert counters_while_blocked.fallback_action_count == 0


def test_shutdown_enters_draining_rejects_new_and_waits_for_natural_completion() -> None:
    runtime = BlockingRuntime()
    app = _active_app(runtime)
    response, readiness, rejected, shutdown_was_waiting = asyncio.run(
        _run_shutdown_scenario(app=app, runtime=runtime)
    )

    assert readiness.status_code == 503
    assert readiness.json()["facade_state"] == "draining"
    assert rejected.status_code == 503
    assert rejected.json()["error"]["code"] == "facade_draining"
    assert shutdown_was_waiting is True
    assert response.status_code == 200
    assert app.state.agent_http_transport.state is AgentHttpTransportState.CLOSED


def test_request_after_transport_shutdown_is_lifecycle_503_not_executor_500() -> None:
    runtime = SyntheticRuntime()
    app = _active_app(runtime)
    transport = app.state.agent_http_transport
    transport.start()
    transport.shutdown()

    async def request_closed_transport() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1",
            timeout=None,
        ) as client:
            return await client.post(
                "/v1/agent/answers",
                json={"request_handle": "closed", "text": "Question"},
            )

    response = asyncio.run(request_closed_transport())

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "facade_closed"
    assert runtime.call_count == 0


def test_known_predispatch_disconnect_sends_no_asgi_response_and_skips_runtime() -> None:
    runtime = SyntheticRuntime()
    sink = RecordingLogSink()
    app = _active_app(runtime, sink=sink)
    transport = app.state.agent_http_transport
    transport.start()
    body = b'{"request_handle":"disconnect","text":"Question"}'
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
    transport.shutdown()

    assert sent == []
    assert runtime.call_count == 0
    assert transport.counters().known_disconnect_count == 1
    assert transport.facade is not None
    assert transport.facade.counters().cancelled_before_dispatch_count == 1
    assert sink.events[-1].transport_outcome_code == "client_disconnected"
    assert sink.events[-1].downstream_dispatched is False


def test_exact_routes_framework_errors_uvicorn_config_and_public_logs() -> None:
    runtime = SyntheticRuntime()
    sink = RecordingLogSink()
    app = _active_app(runtime, sink=sink)

    assert [(next(iter(route.methods)), route.path) for route in app.routes] == [
        ("POST", "/v1/agent/answers"),
        ("GET", "/health/live"),
        ("GET", "/health/ready"),
    ]
    with TestClient(app) as client:
        missing = client.get("/missing")
        wrong_method = client.get("/v1/agent/answers")
        client.post(
            "/v1/agent/answers",
            json={"request_handle": "private-handle", "text": "private question text"},
        )

    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"
    assert wrong_method.status_code == 405
    assert wrong_method.json()["error"]["code"] == "method_not_allowed"
    expected_fields = set(agent_http_transport_contract()["public_log_fields"])
    assert len(expected_fields) == 18
    for event in sink.events:
        public = event.to_public_dict()
        assert set(public) == expected_fields
        serialized = json.dumps(public)
        assert "private-handle" not in serialized
        assert "private question text" not in serialized

    config = create_primeqa_hybrid_agent_uvicorn_config(app=app, port=0)
    assert config.host == "127.0.0.1"
    assert config.port == 0
    assert config.http == "h11"
    assert config.ws == "none"
    assert config.lifespan == "on"
    assert config.access_log is False
    assert config.server_header is False
    assert config.proxy_headers is False
    assert config.workers == 1
    assert config.timeout_graceful_shutdown is None


def _active_app(
    runtime: SyntheticRuntime,
    *,
    sink: RecordingLogSink | None = None,
):
    return create_primeqa_hybrid_agent_http_app(
        settings=ProjectSettings(
            enable_concurrent_sidecar_agent=True,
            enable_local_agent_http_transport=True,
            _env_file=None,
        ),
        bootstrap_result=_bootstrap_result(runtime=runtime, active=True),
        log_sink=sink or RecordingLogSink(),
    )


def _bootstrap_result(
    *,
    runtime: SyntheticRuntime,
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


def _runtime_run(request_handle: str, *, refused: bool = False) -> SyntheticRuntimeRun:
    return SyntheticRuntimeRun(
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


async def _run_four_request_overload_scenario(
    *,
    app,
    runtime: BlockingRuntime,
) -> tuple[list[httpx.Response], httpx.Response, Any]:
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
                    json={"request_handle": f"request-{index}", "text": "Question"},
                )
            )
            for index in range(4)
        ]
        await asyncio.to_thread(runtime.target_entered.wait)
        rejected = await client.post(
            "/v1/agent/answers",
            json={"request_handle": "request-5", "text": "Question"},
        )
        counters = transport.counters()
        runtime.release.set()
        responses = list(await asyncio.gather(*tasks))
    transport.shutdown()
    return responses, rejected, counters


async def _run_shutdown_scenario(
    *,
    app,
    runtime: BlockingRuntime,
) -> tuple[httpx.Response, httpx.Response, httpx.Response, bool]:
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
                json={"request_handle": "in-flight", "text": "Question"},
            )
        )
        await asyncio.to_thread(runtime.target_entered.wait)
        shutdown_task = asyncio.create_task(asyncio.to_thread(transport.shutdown))
        assert transport.facade is not None
        await asyncio.to_thread(
            transport.facade.wait_until_state,
            AgentRequestFacadeState.DRAINING,
        )
        readiness = await client.get("/health/ready")
        rejected = await client.post(
            "/v1/agent/answers",
            json={"request_handle": "new", "text": "Question"},
        )
        shutdown_was_waiting = not shutdown_task.done()
        runtime.release.set()
        response = await request_task
        await shutdown_task
    return response, readiness, rejected, shutdown_was_waiting


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
