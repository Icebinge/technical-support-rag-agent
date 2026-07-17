from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from threading import BoundedSemaphore, Lock
from typing import Any, Protocol

import fastapi
import starlette
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect

from ts_rag_agent.application.primeqa_hybrid_agent_request_facade import (
    AgentFacadeRequest,
    AgentRequestCancellationSignal,
    AgentRequestFacadeCancelledError,
    AgentRequestFacadeCapacityExceededError,
    AgentRequestFacadeClosedError,
    AgentRequestFacadeDrainingError,
    AgentRequestFacadeError,
    AgentRequestFacadeInvalidRequestError,
    AgentRequestFacadeNotActiveError,
    AgentRequestFacadeRun,
    AgentRequestFacadeState,
    PrimeQAHybridAgentRequestFacade,
    PublicSafeAgentRequestFacadeEvent,
    create_primeqa_hybrid_agent_request_facade,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_activation import (
    PrimeQAHybridConcurrentRuntimeBootstrapResult,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_sidecar_agent_runtime import (
    PublicSafeConcurrentRuntimeRequestTrace,
)
from ts_rag_agent.config import ProjectSettings

_TRANSPORT_ID = "primeqa_hybrid_local_fastapi_agent_transport_v1"
_HOST = "127.0.0.1"
_MAX_BODY_BYTES = 32 * 1024
_MAX_REQUEST_HANDLE_CHARS = 128
_MAX_TITLE_CHARS = 512
_MAX_TEXT_CHARS = 24 * 1024
_MAX_IN_FLIGHT = 4
_ALLOWED_LOG_FIELDS = frozenset(
    {
        "route_id",
        "method",
        "http_status",
        "transport_outcome_code",
        "facade_state",
        "downstream_dispatched",
        "queue_action_count",
        "retry_action_count",
        "fallback_action_count",
        "runtime_mode",
        "activation_state",
        "admission_state",
        "arrival_pattern",
        "candidate_pool_depth",
        "retrieval_latency_ms",
        "end_to_end_latency_ms",
        "latency_budget_passed",
        "terminal_state",
    }
)
_FORBIDDEN_LOG_KEYS = frozenset(
    {
        "answer",
        "authorization",
        "citations",
        "client_ip",
        "cookie",
        "document_id",
        "document_reference",
        "headers",
        "question_text",
        "request_handle",
        "response_text",
        "text",
        "title",
        "user_agent",
    }
)
_ERROR_MESSAGES = {
    "malformed_json": "Request body must be valid UTF-8 JSON.",
    "request_body_too_large": "Request body exceeds the allowed size.",
    "unsupported_media_type": "Content-Type must be application/json with UTF-8.",
    "invalid_request": "Request does not match the required schema.",
    "facade_not_active": "Agent service is not active.",
    "capacity_exceeded": "Agent service is at capacity.",
    "facade_draining": "Agent service is draining.",
    "facade_closed": "Agent service is closed.",
    "internal_error": "Agent service encountered an internal error.",
    "not_found": "Route not found.",
    "method_not_allowed": "Method not allowed.",
}


class AgentHttpTransportState(str, Enum):
    """Observable lifecycle state of the local HTTP adapter."""

    CREATED = "created"
    DISABLED = "disabled"
    NOT_ACTIVE = "not_active"
    ACCEPTING = "accepting"
    DRAINING = "draining"
    CLOSED = "closed"


class AgentHttpRequestPayload(BaseModel):
    """Strict external request schema frozen by Stage149."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    request_handle: str = Field(min_length=1, max_length=_MAX_REQUEST_HANDLE_CHARS)
    title: str | None = Field(default=None, max_length=_MAX_TITLE_CHARS)
    text: str = Field(min_length=1, max_length=_MAX_TEXT_CHARS)

    @field_validator("request_handle", "text")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must contain non-whitespace characters")
        return value


class AgentHttpCitationPayload(BaseModel):
    """Stable private citation payload returned to the caller."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    document_reference: str
    title: str
    rank: int
    evidence_score: float


class AgentHttpSuccessPayload(BaseModel):
    """Stable private success/refusal payload returned to the caller."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    request_handle: str
    text: str
    refused: bool
    citations: tuple[AgentHttpCitationPayload, ...]


@dataclass(frozen=True)
class PublicSafeAgentHttpTransportEvent:
    """Exact eighteen-field public log event frozen by Stage149."""

    route_id: str
    method: str
    http_status: int
    transport_outcome_code: str
    facade_state: str
    downstream_dispatched: bool
    queue_action_count: int = 0
    retry_action_count: int = 0
    fallback_action_count: int = 0
    runtime_mode: str = ""
    activation_state: str = ""
    admission_state: str = ""
    arrival_pattern: str = ""
    candidate_pool_depth: int = 0
    retrieval_latency_ms: float = 0.0
    end_to_end_latency_ms: float = 0.0
    latency_budget_passed: bool = False
    terminal_state: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if set(payload) != _ALLOWED_LOG_FIELDS:
            raise ValueError("HTTP transport event fields do not match the Stage149 allowlist")
        forbidden = sorted(_forbidden_log_keys_found(payload))
        if forbidden:
            raise ValueError(f"HTTP transport event contains forbidden keys: {forbidden}")
        return payload


class AgentHttpTransportLogSink(Protocol):
    """Port for allowlisted structured transport events."""

    def emit(self, event: PublicSafeAgentHttpTransportEvent) -> None: ...


class PythonStructuredAgentHttpLogSink:
    """Default JSON logger that receives only validated public events."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("ts_rag_agent.agent_http")

    def emit(self, event: PublicSafeAgentHttpTransportEvent) -> None:
        self._logger.info(
            json.dumps(event.to_public_dict(), ensure_ascii=True, separators=(",", ":"))
        )


@dataclass(frozen=True)
class AgentHttpTransportCounters:
    """Aggregate counters with no request content or identifiers."""

    request_attempt_count: int
    admitted_request_count: int
    capacity_rejected_count: int
    completed_response_count: int
    error_response_count: int
    known_disconnect_count: int
    current_in_flight: int
    max_observed_in_flight: int
    application_waiting_request_count: int = 0
    queue_action_count: int = 0
    retry_action_count: int = 0
    fallback_action_count: int = 0


class _KnownClientDisconnect(RuntimeError):
    pass


class _AgentHttpNoResponse(Response):
    """ASGI response that intentionally sends nothing to a known-disconnected peer."""

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        _ = scope, receive, send


class PrimeQAHybridAgentHttpTransport:
    """Strict local HTTP adapter over one optional Stage148 facade."""

    def __init__(
        self,
        *,
        settings: ProjectSettings,
        bootstrap_result: PrimeQAHybridConcurrentRuntimeBootstrapResult,
        log_sink: AgentHttpTransportLogSink | None = None,
    ) -> None:
        self._settings = settings
        self._bootstrap_result = bootstrap_result
        self._log_sink = log_sink or PythonStructuredAgentHttpLogSink()
        self._lock = Lock()
        self._admission = BoundedSemaphore(_MAX_IN_FLIGHT)
        self._executor: ThreadPoolExecutor | None = None
        self._facade: PrimeQAHybridAgentRequestFacade | None = None
        self._state = AgentHttpTransportState.CREATED
        self._started = False
        self._request_attempt_count = 0
        self._admitted_request_count = 0
        self._capacity_rejected_count = 0
        self._completed_response_count = 0
        self._error_response_count = 0
        self._known_disconnect_count = 0
        self._current_in_flight = 0
        self._max_observed_in_flight = 0

    @property
    def state(self) -> AgentHttpTransportState:
        with self._lock:
            transport_state = self._state
            facade = self._facade
        if transport_state is not AgentHttpTransportState.ACCEPTING:
            return transport_state
        if facade is not None:
            facade_state = facade.state
            if facade_state is AgentRequestFacadeState.ACCEPTING:
                return AgentHttpTransportState.ACCEPTING
            if facade_state is AgentRequestFacadeState.DRAINING:
                return AgentHttpTransportState.DRAINING
            return AgentHttpTransportState.CLOSED
        return transport_state

    @property
    def facade(self) -> PrimeQAHybridAgentRequestFacade | None:
        return self._facade

    def counters(self) -> AgentHttpTransportCounters:
        with self._lock:
            return AgentHttpTransportCounters(
                request_attempt_count=self._request_attempt_count,
                admitted_request_count=self._admitted_request_count,
                capacity_rejected_count=self._capacity_rejected_count,
                completed_response_count=self._completed_response_count,
                error_response_count=self._error_response_count,
                known_disconnect_count=self._known_disconnect_count,
                current_in_flight=self._current_in_flight,
                max_observed_in_flight=self._max_observed_in_flight,
            )

    def start(self) -> None:
        """Create at most one facade and executor during application lifespan startup."""

        with self._lock:
            if self._started:
                raise RuntimeError("local Agent HTTP transport lifespan may start only once")
            self._started = True
            if not self._settings.enable_local_agent_http_transport:
                self._state = AgentHttpTransportState.DISABLED
                return
        try:
            facade = create_primeqa_hybrid_agent_request_facade(
                bootstrap_result=self._bootstrap_result
            )
        except AgentRequestFacadeNotActiveError:
            with self._lock:
                self._state = AgentHttpTransportState.NOT_ACTIVE
            return
        executor = ThreadPoolExecutor(
            max_workers=_MAX_IN_FLIGHT,
            thread_name_prefix="ts-rag-agent-http",
        )
        with self._lock:
            self._facade = facade
            self._executor = executor
            self._state = AgentHttpTransportState.ACCEPTING

    def shutdown(self) -> None:
        """Drain the facade naturally, then close the transport-owned executor."""

        with self._lock:
            facade = self._facade
            executor = self._executor
            if self._state is AgentHttpTransportState.CLOSED:
                return
            if facade is None:
                self._state = AgentHttpTransportState.CLOSED
                return
            self._state = AgentHttpTransportState.DRAINING
        facade.shutdown()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        with self._lock:
            self._state = AgentHttpTransportState.CLOSED

    async def answer(self, request: Request) -> Response:
        """Validate one HTTP request and invoke the synchronous facade off-loop."""

        with self._lock:
            self._request_attempt_count += 1
        facade = self._facade
        if facade is None:
            return self._error_response(
                route_id="agent_answer",
                method="POST",
                status=503,
                code="facade_not_active",
                facade_state=self.state.value,
            )
        initial_state = self.state
        if initial_state is not AgentHttpTransportState.ACCEPTING:
            return self._lifecycle_error_response(initial_state)

        try:
            payload = await _read_request_payload(request)
        except _KnownClientDisconnect:
            self._record_known_disconnect(facade_state=self.state.value)
            return _AgentHttpNoResponse()
        except _TransportInputError as error:
            return self._error_response(
                route_id="agent_answer",
                method="POST",
                status=error.status,
                code=error.code,
                facade_state=self.state.value,
            )

        cancellation = AgentRequestCancellationSignal()
        if await request.is_disconnected():
            cancellation.cancel()
        facade_request = AgentFacadeRequest(
            request_handle=payload.request_handle,
            title=payload.title,
            text=payload.text,
            cancellation_signal=cancellation,
        )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[AgentRequestFacadeRun] | None = None
        rejection_state: AgentHttpTransportState | None = None
        capacity_rejected = False
        submission_failed = False
        with self._lock:
            if self._state is not AgentHttpTransportState.ACCEPTING:
                rejection_state = self._state
            elif not self._admission.acquire(blocking=False):
                if cancellation.is_cancelled():
                    rejection_state = self._state
                else:
                    self._capacity_rejected_count += 1
                    capacity_rejected = True
            else:
                executor = self._executor
                if executor is None:
                    self._admission.release()
                    rejection_state = AgentHttpTransportState.CLOSED
                else:
                    self._admitted_request_count += 1
                    self._current_in_flight += 1
                    self._max_observed_in_flight = max(
                        self._max_observed_in_flight,
                        self._current_in_flight,
                    )
                    try:
                        # Submission is atomic with the transport lifecycle transition.
                        future = loop.run_in_executor(executor, facade.invoke, facade_request)
                    except RuntimeError:
                        self._current_in_flight -= 1
                        self._admission.release()
                        submission_failed = True

        if rejection_state is not None:
            if cancellation.is_cancelled():
                self._record_known_disconnect(facade_state=rejection_state.value)
                return _AgentHttpNoResponse()
            return self._lifecycle_error_response(rejection_state)
        if capacity_rejected:
            return self._error_response(
                route_id="agent_answer",
                method="POST",
                status=503,
                code="capacity_exceeded",
                facade_state=self.state.value,
            )
        if submission_failed or future is None:
            return self._error_response(
                route_id="agent_answer",
                method="POST",
                status=500,
                code="internal_error",
                facade_state=self.state.value,
            )

        try:
            try:
                run = await future
            except AgentRequestFacadeCancelledError as error:
                self._record_known_disconnect(
                    facade_state=error.public_safe_event.facade_state,
                    event=error.public_safe_event,
                )
                return _AgentHttpNoResponse()
            except AgentRequestFacadeError as error:
                return self._facade_error_response(error)
            except Exception:
                return self._error_response(
                    route_id="agent_answer",
                    method="POST",
                    status=500,
                    code="internal_error",
                    facade_state=self.state.value,
                    downstream_dispatched=True,
                )
            response = _success_payload(run)
            self._emit_from_facade(
                status=200,
                code="refuse" if response.refused else "complete",
                event=run.public_safe_event,
                trace=run.public_safe_runtime_trace,
            )
            with self._lock:
                self._completed_response_count += 1
            return JSONResponse(response.model_dump(mode="json"), status_code=200)
        finally:
            with self._lock:
                self._current_in_flight -= 1
            self._admission.release()

    def _lifecycle_error_response(self, state: AgentHttpTransportState) -> JSONResponse:
        if state is AgentHttpTransportState.DRAINING:
            code = "facade_draining"
        elif state is AgentHttpTransportState.CLOSED:
            code = "facade_closed"
        else:
            code = "facade_not_active"
        return self._error_response(
            route_id="agent_answer",
            method="POST",
            status=503,
            code=code,
            facade_state=state.value,
        )

    def liveness_response(self) -> JSONResponse:
        event = PublicSafeAgentHttpTransportEvent(
            route_id="liveness",
            method="GET",
            http_status=200,
            transport_outcome_code="live",
            facade_state=self.state.value,
            downstream_dispatched=False,
        )
        self._log_sink.emit(event)
        return JSONResponse({"status": "live"}, status_code=200)

    def readiness_response(self) -> JSONResponse:
        state = self.state
        ready = state is AgentHttpTransportState.ACCEPTING
        status = 200 if ready else 503
        event = PublicSafeAgentHttpTransportEvent(
            route_id="readiness",
            method="GET",
            http_status=status,
            transport_outcome_code="ready" if ready else "not_ready",
            facade_state=state.value,
            downstream_dispatched=False,
        )
        self._log_sink.emit(event)
        return JSONResponse(
            {
                "status": "ready" if ready else "not_ready",
                "facade_state": state.value,
            },
            status_code=status,
        )

    def framework_error_response(self, *, method: str, status: int) -> JSONResponse:
        code = "not_found" if status == 404 else "method_not_allowed"
        return self._error_response(
            route_id="unmatched_route",
            method=method,
            status=status,
            code=code,
            facade_state=self.state.value,
        )

    def _facade_error_response(self, error: AgentRequestFacadeError) -> JSONResponse:
        mapping: tuple[int, str]
        if isinstance(error, AgentRequestFacadeInvalidRequestError):
            mapping = (422, "invalid_request")
        elif isinstance(error, AgentRequestFacadeCapacityExceededError):
            mapping = (503, "capacity_exceeded")
        elif isinstance(error, AgentRequestFacadeDrainingError):
            mapping = (503, "facade_draining")
        elif isinstance(error, AgentRequestFacadeClosedError):
            mapping = (503, "facade_closed")
        elif isinstance(error, AgentRequestFacadeNotActiveError):
            mapping = (503, "facade_not_active")
        else:
            mapping = (500, "internal_error")
        status, code = mapping
        self._emit_from_facade(
            status=status,
            code=code,
            event=error.public_safe_event,
            trace=error.public_safe_runtime_trace,
        )
        with self._lock:
            self._error_response_count += 1
            if isinstance(error, AgentRequestFacadeCapacityExceededError):
                self._capacity_rejected_count += 1
        return _json_error(status=status, code=code)

    def _error_response(
        self,
        *,
        route_id: str,
        method: str,
        status: int,
        code: str,
        facade_state: str,
        downstream_dispatched: bool = False,
    ) -> JSONResponse:
        event = PublicSafeAgentHttpTransportEvent(
            route_id=route_id,
            method=method,
            http_status=status,
            transport_outcome_code=code,
            facade_state=facade_state,
            downstream_dispatched=downstream_dispatched,
        )
        self._log_sink.emit(event)
        with self._lock:
            self._error_response_count += 1
        return _json_error(status=status, code=code)

    def _emit_from_facade(
        self,
        *,
        status: int,
        code: str,
        event: PublicSafeAgentRequestFacadeEvent,
        trace: PublicSafeConcurrentRuntimeRequestTrace | None,
    ) -> None:
        self._log_sink.emit(
            _transport_event_from_facade(status=status, code=code, event=event, trace=trace)
        )

    def _record_known_disconnect(
        self,
        *,
        facade_state: str,
        event: PublicSafeAgentRequestFacadeEvent | None = None,
    ) -> None:
        with self._lock:
            self._known_disconnect_count += 1
        self._log_sink.emit(
            PublicSafeAgentHttpTransportEvent(
                route_id="agent_answer",
                method="POST",
                http_status=0,
                transport_outcome_code="client_disconnected",
                facade_state=facade_state,
                downstream_dispatched=(event.downstream_dispatched if event else False),
            )
        )


@dataclass(frozen=True)
class _TransportInputError(Exception):
    status: int
    code: str


async def _read_request_payload(request: Request) -> AgentHttpRequestPayload:
    if not _supported_json_content_type(request.headers.get("content-type")):
        raise _TransportInputError(status=415, code="unsupported_media_type")
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError as error:
            raise _TransportInputError(status=400, code="malformed_json") from error
        if declared_size < 0:
            raise _TransportInputError(status=400, code="malformed_json")
        if declared_size > _MAX_BODY_BYTES:
            raise _TransportInputError(status=413, code="request_body_too_large")

    body = bytearray()
    try:
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > _MAX_BODY_BYTES:
                raise _TransportInputError(status=413, code="request_body_too_large")
    except ClientDisconnect as error:
        raise _KnownClientDisconnect from error
    try:
        decoded = bytes(body).decode("utf-8", errors="strict")
        value = json.loads(decoded, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError) as error:
        raise _TransportInputError(status=400, code="malformed_json") from error
    try:
        return AgentHttpRequestPayload.model_validate(value)
    except ValidationError as error:
        raise _TransportInputError(status=422, code="invalid_request") from error


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _supported_json_content_type(value: str | None) -> bool:
    if value is None:
        return False
    segments = [segment.strip() for segment in value.split(";")]
    if not segments or segments[0].lower() != "application/json":
        return False
    parameters = segments[1:]
    if not parameters:
        return True
    return len(parameters) == 1 and parameters[0].lower().replace(" ", "") == "charset=utf-8"


def _success_payload(run: AgentRequestFacadeRun) -> AgentHttpSuccessPayload:
    response = run.response
    return AgentHttpSuccessPayload(
        request_handle=response.request_handle,
        text=response.text,
        refused=response.refused,
        citations=tuple(
            AgentHttpCitationPayload(
                document_reference=citation.document_reference,
                title=citation.title,
                rank=citation.rank,
                evidence_score=citation.evidence_score,
            )
            for citation in response.citations
        ),
    )


def _transport_event_from_facade(
    *,
    status: int,
    code: str,
    event: PublicSafeAgentRequestFacadeEvent,
    trace: PublicSafeConcurrentRuntimeRequestTrace | None,
) -> PublicSafeAgentHttpTransportEvent:
    runtime = trace.to_public_dict() if trace is not None else {}
    return PublicSafeAgentHttpTransportEvent(
        route_id="agent_answer",
        method="POST",
        http_status=status,
        transport_outcome_code=code,
        facade_state=event.facade_state,
        downstream_dispatched=event.downstream_dispatched,
        queue_action_count=event.queue_action_count,
        retry_action_count=event.retry_action_count,
        fallback_action_count=event.fallback_action_count,
        runtime_mode=str(runtime.get("runtime_mode") or ""),
        activation_state=str(runtime.get("activation_state") or ""),
        admission_state=str(runtime.get("admission_state") or ""),
        arrival_pattern=str(runtime.get("arrival_pattern") or ""),
        candidate_pool_depth=int(runtime.get("candidate_pool_depth") or 0),
        retrieval_latency_ms=float(runtime.get("retrieval_latency_ms") or 0.0),
        end_to_end_latency_ms=float(runtime.get("end_to_end_latency_ms") or 0.0),
        latency_budget_passed=runtime.get("latency_budget_passed") is True,
        terminal_state=str(runtime.get("terminal_state") or ""),
    )


def _json_error(*, status: int, code: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": _ERROR_MESSAGES[code]}},
        status_code=status,
    )


def create_primeqa_hybrid_agent_http_app(
    *,
    settings: ProjectSettings,
    bootstrap_result: PrimeQAHybridConcurrentRuntimeBootstrapResult,
    log_sink: AgentHttpTransportLogSink | None = None,
) -> FastAPI:
    """Create the exact three-route local FastAPI adapter without starting it."""

    transport = PrimeQAHybridAgentHttpTransport(
        settings=settings,
        bootstrap_result=bootstrap_result,
        log_sink=log_sink,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _ = app
        transport.start()
        try:
            yield
        finally:
            await asyncio.to_thread(transport.shutdown)

    app = FastAPI(
        title="Technical Support RAG Agent",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.agent_http_transport = transport

    @app.post("/v1/agent/answers", include_in_schema=False, name="agent_answer")
    async def answer(request: Request) -> Response:
        return await transport.answer(request)

    @app.get("/health/live", include_in_schema=False, name="liveness")
    async def liveness() -> Response:
        return transport.liveness_response()

    @app.get("/health/ready", include_in_schema=False, name="readiness")
    async def readiness() -> Response:
        return transport.readiness_response()

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, error: StarletteHTTPException) -> Response:
        if error.status_code in {404, 405}:
            return transport.framework_error_response(
                method=request.method,
                status=error.status_code,
            )
        return _json_error(status=500, code="internal_error")

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        error: RequestValidationError,
    ) -> Response:
        _ = error
        return transport._error_response(
            route_id="agent_answer",
            method=request.method,
            status=422,
            code="invalid_request",
            facade_state=transport.state.value,
        )

    return app


def create_primeqa_hybrid_agent_uvicorn_config(
    *,
    app: FastAPI,
    port: int,
    log_level: str = "warning",
) -> uvicorn.Config:
    """Return a single-worker loopback configuration with no access log or timeout."""

    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    return uvicorn.Config(
        app=app,
        host=_HOST,
        port=port,
        loop="asyncio",
        http="h11",
        ws="none",
        lifespan="on",
        log_level=log_level,
        access_log=False,
        server_header=False,
        proxy_headers=False,
        workers=1,
        timeout_graceful_shutdown=None,
    )


def agent_http_transport_contract() -> dict[str, Any]:
    """Return the implemented Stage150 contract without starting a server."""

    return {
        "transport_id": _TRANSPORT_ID,
        "framework": "FastAPI",
        "framework_version": fastapi.__version__,
        "starlette_version": starlette.__version__,
        "uvicorn_version": uvicorn.__version__,
        "settings_field": "enable_local_agent_http_transport",
        "environment_flag": "TS_RAG_ENABLE_LOCAL_AGENT_HTTP_TRANSPORT",
        "default_enabled": False,
        "requires_concurrent_runtime": True,
        "binding_host": _HOST,
        "http_protocol": "HTTP/1.1",
        "routes": [
            {"route_id": "agent_answer", "method": "POST", "path": "/v1/agent/answers"},
            {"route_id": "liveness", "method": "GET", "path": "/health/live"},
            {"route_id": "readiness", "method": "GET", "path": "/health/ready"},
        ],
        "raw_body_max_bytes": _MAX_BODY_BYTES,
        "request_handle_max_chars": _MAX_REQUEST_HANDLE_CHARS,
        "title_max_chars": _MAX_TITLE_CHARS,
        "text_max_chars": _MAX_TEXT_CHARS,
        "max_in_flight": _MAX_IN_FLIGHT,
        "application_waiting_queue": False,
        "request_timeout_seconds": None,
        "implicit_shutdown_timeout_seconds": None,
        "access_log_enabled": False,
        "public_log_fields": sorted(_ALLOWED_LOG_FIELDS),
        "runtime_registered_as_default": False,
        "remote_exposure_authorized": False,
        "test_access_allowed": False,
        "queue_actions_allowed": False,
        "retry_actions_allowed": False,
        "fallback_strategies_allowed": False,
    }


def _forbidden_log_keys_found(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in _FORBIDDEN_LOG_KEYS:
                found.add(key_text)
            found.update(_forbidden_log_keys_found(child))
    elif isinstance(value, list | tuple):
        for child in value:
            found.update(_forbidden_log_keys_found(child))
    return found
