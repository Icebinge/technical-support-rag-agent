from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from threading import Lock
from typing import Any, Protocol, TypeVar

import fastapi
import starlette
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect

from ts_rag_agent.application.primeqa_hybrid_bounded_agent_state_protocol import (
    ThreadStatePolicyViolationError,
    ThreadStateSummary,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_runtime import (
    BoundedDynamicAgentRuntimeRun,
    PrimeQAHybridBoundedDynamicAgentRuntime,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.domain.dataset import PrimeQARuntimeQuery

BOUNDED_DYNAMIC_AGENT_HTTP_TRANSPORT_ID = "primeqa_hybrid_bounded_dynamic_agent_http_transport_v1"
BOUNDED_DYNAMIC_AGENT_BINDING_HOST = "127.0.0.1"
BOUNDED_DYNAMIC_AGENT_MAX_IN_FLIGHT_TURNS = 1
_MAX_BODY_BYTES = 32 * 1024
_MAX_THREAD_HANDLE_CHARS = 128
_MAX_TITLE_CHARS = 512
_MAX_TEXT_CHARS = 24 * 1024
_THREAD_HANDLE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_PUBLIC_EVENT_FIELDS = frozenset(
    {
        "route_id",
        "method",
        "http_status",
        "outcome_code",
        "coordinator_state",
        "downstream_dispatched",
        "gpu_admitted",
        "current_in_flight_turns",
        "max_observed_in_flight_turns",
        "opened_thread_count",
        "completed_turn_count",
        "retained_state_bytes",
        "terminal_state",
        "selected_action",
        "model_decision_count",
        "retrieval_call_count",
        "composition_call_count",
        "verification_call_count",
        "diagnostic_observation_count",
        "router_input_token_count",
        "router_output_token_count",
        "router_generation_latency_ms",
        "queue_action_count",
        "retry_action_count",
        "fallback_action_count",
    }
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "authorization",
        "citations",
        "client_ip",
        "cookie",
        "document_id",
        "document_reference",
        "headers",
        "model_output",
        "question",
        "request_handle",
        "response_text",
        "text",
        "thread_handle",
        "title",
        "user_agent",
    }
)
_ERROR_MESSAGES = {
    "malformed_json": "Request body must be valid UTF-8 JSON.",
    "request_body_too_large": "Request body exceeds the allowed size.",
    "unsupported_media_type": "Content-Type must be application/json with UTF-8.",
    "invalid_request": "Request does not match the required schema.",
    "service_not_active": "Bounded dynamic Agent service is not active.",
    "service_draining": "Bounded dynamic Agent service is draining.",
    "service_closed": "Bounded dynamic Agent service is closed.",
    "thread_already_open": "Thread is already open.",
    "thread_not_found": "Thread is not open.",
    "thread_busy": "Thread already has an admitted turn.",
    "gpu_capacity_exceeded": "The bounded dynamic Agent GPU slot is in use.",
    "thread_state_limit_exceeded": "Thread state limit was exceeded without mutation.",
    "turn_execution_failed": "Bounded dynamic Agent turn execution failed.",
    "internal_error": "Bounded dynamic Agent service encountered an internal error.",
    "not_found": "Route not found.",
    "method_not_allowed": "Method not allowed.",
}


class BoundedDynamicAgentCoordinatorState(str, Enum):
    CREATED = "created"
    ACCEPTING = "accepting"
    DRAINING = "draining"
    CLOSED = "closed"


class BoundedDynamicAgentServiceErrorCode(str, Enum):
    SERVICE_NOT_ACTIVE = "service_not_active"
    SERVICE_DRAINING = "service_draining"
    SERVICE_CLOSED = "service_closed"
    THREAD_ALREADY_OPEN = "thread_already_open"
    THREAD_NOT_FOUND = "thread_not_found"
    THREAD_BUSY = "thread_busy"
    GPU_CAPACITY_EXCEEDED = "gpu_capacity_exceeded"
    THREAD_STATE_LIMIT_EXCEEDED = "thread_state_limit_exceeded"
    TURN_EXECUTION_FAILED = "turn_execution_failed"


class BoundedDynamicAgentServiceError(RuntimeError):
    def __init__(self, code: BoundedDynamicAgentServiceErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


class ThreadHandlePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    thread_handle: str = Field(min_length=1, max_length=_MAX_THREAD_HANDLE_CHARS)

    @field_validator("thread_handle")
    @classmethod
    def validate_opaque_handle(cls, value: str) -> str:
        if _THREAD_HANDLE_PATTERN.fullmatch(value) is None:
            raise ValueError("thread_handle must use the exact opaque-handle character set")
        return value


class ThreadTurnPayload(ThreadHandlePayload):
    title: str | None = Field(default=None, max_length=_MAX_TITLE_CHARS)
    text: str = Field(min_length=1, max_length=_MAX_TEXT_CHARS)

    @field_validator("text")
    @classmethod
    def validate_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value


class BoundedDynamicAgentCitationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    document_reference: str
    title: str
    rank: int
    evidence_score: float


class BoundedDynamicAgentThreadPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    thread_handle: str
    opened: bool
    completed_turn_count: int
    retained_state_bytes: int


class BoundedDynamicAgentTurnPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    thread_handle: str
    text: str
    refused: bool
    citations: tuple[BoundedDynamicAgentCitationPayload, ...]
    terminal_state: str
    completed_turn_count: int
    retained_state_bytes: int


@dataclass(frozen=True)
class PublicSafeBoundedDynamicAgentHttpEvent:
    route_id: str
    method: str
    http_status: int
    outcome_code: str
    coordinator_state: str
    downstream_dispatched: bool
    gpu_admitted: bool
    current_in_flight_turns: int
    max_observed_in_flight_turns: int
    opened_thread_count: int
    completed_turn_count: int = 0
    retained_state_bytes: int = 0
    terminal_state: str = ""
    selected_action: str = ""
    model_decision_count: int = 0
    retrieval_call_count: int = 0
    composition_call_count: int = 0
    verification_call_count: int = 0
    diagnostic_observation_count: int = 0
    router_input_token_count: int = 0
    router_output_token_count: int = 0
    router_generation_latency_ms: float = 0.0
    queue_action_count: int = 0
    retry_action_count: int = 0
    fallback_action_count: int = 0

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if set(payload) != _PUBLIC_EVENT_FIELDS:
            raise ValueError("bounded dynamic HTTP event fields do not match the allowlist")
        forbidden = sorted(_find_forbidden_public_keys(payload))
        if forbidden:
            raise ValueError(f"bounded dynamic HTTP event contains forbidden keys: {forbidden}")
        return payload


class BoundedDynamicAgentHttpLogSink(Protocol):
    def emit(self, event: PublicSafeBoundedDynamicAgentHttpEvent) -> None: ...


class PythonBoundedDynamicAgentHttpLogSink:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("ts_rag_agent.bounded_dynamic_agent_http")

    def emit(self, event: PublicSafeBoundedDynamicAgentHttpEvent) -> None:
        self._logger.info(
            json.dumps(event.to_public_dict(), ensure_ascii=True, separators=(",", ":"))
        )


@dataclass(frozen=True)
class BoundedDynamicAgentServiceCounters:
    opened_thread_count: int
    current_in_flight_turns: int
    max_observed_in_flight_turns: int
    admitted_turn_count: int
    capacity_rejected_turn_count: int
    completed_turn_count: int
    failed_turn_count: int
    queue_action_count: int = 0
    retry_action_count: int = 0
    fallback_action_count: int = 0


@dataclass(frozen=True)
class BoundedDynamicAgentTurnAdmission:
    token: object
    opaque_thread_handle: str
    sequence_number: int


@dataclass(frozen=True)
class BoundedDynamicAgentTurnRun:
    runtime_run: BoundedDynamicAgentRuntimeRun
    thread_summary: ThreadStateSummary


class BoundedDynamicAgentServiceCoordinator:
    """Own explicit thread lifecycle and one nonblocking whole-turn GPU slot."""

    def __init__(self, runtime: PrimeQAHybridBoundedDynamicAgentRuntime) -> None:
        self._runtime = runtime
        self._lock = Lock()
        self._state = BoundedDynamicAgentCoordinatorState.CREATED
        self._open_handles: set[str] = set()
        self._active_turns: dict[object, str] = {}
        self._max_observed_in_flight_turns = 0
        self._admitted_turn_count = 0
        self._capacity_rejected_turn_count = 0
        self._completed_turn_count = 0
        self._failed_turn_count = 0

    @property
    def state(self) -> BoundedDynamicAgentCoordinatorState:
        with self._lock:
            return self._state

    def counters(self) -> BoundedDynamicAgentServiceCounters:
        with self._lock:
            return self._counters_locked()

    def start(self) -> None:
        with self._lock:
            if self._state is not BoundedDynamicAgentCoordinatorState.CREATED:
                raise RuntimeError("bounded dynamic coordinator may start only once")
            self._state = BoundedDynamicAgentCoordinatorState.ACCEPTING

    def open_thread(self, opaque_thread_handle: str) -> ThreadStateSummary:
        with self._lock:
            self._require_accepting_locked()
            if opaque_thread_handle in self._open_handles:
                raise BoundedDynamicAgentServiceError(
                    BoundedDynamicAgentServiceErrorCode.THREAD_ALREADY_OPEN
                )
            try:
                summary = self._runtime.open_thread(opaque_thread_handle)
            except ThreadStatePolicyViolationError as error:
                raise BoundedDynamicAgentServiceError(
                    BoundedDynamicAgentServiceErrorCode.THREAD_ALREADY_OPEN
                ) from error
            self._open_handles.add(opaque_thread_handle)
            return summary

    def close_thread(self, opaque_thread_handle: str) -> ThreadStateSummary:
        with self._lock:
            self._require_accepting_locked()
            self._require_open_locked(opaque_thread_handle)
            if opaque_thread_handle in self._active_turns.values():
                raise BoundedDynamicAgentServiceError(
                    BoundedDynamicAgentServiceErrorCode.THREAD_BUSY
                )
            try:
                summary = self._runtime.close_thread(opaque_thread_handle)
            except ThreadStatePolicyViolationError as error:
                raise BoundedDynamicAgentServiceError(
                    BoundedDynamicAgentServiceErrorCode.THREAD_NOT_FOUND
                ) from error
            self._open_handles.remove(opaque_thread_handle)
            return summary

    def admit_turn(self, opaque_thread_handle: str) -> BoundedDynamicAgentTurnAdmission:
        with self._lock:
            self._require_accepting_locked()
            self._require_open_locked(opaque_thread_handle)
            if opaque_thread_handle in self._active_turns.values():
                raise BoundedDynamicAgentServiceError(
                    BoundedDynamicAgentServiceErrorCode.THREAD_BUSY
                )
            if self._active_turns:
                self._capacity_rejected_turn_count += 1
                raise BoundedDynamicAgentServiceError(
                    BoundedDynamicAgentServiceErrorCode.GPU_CAPACITY_EXCEEDED
                )
            summary = self._runtime.thread_summary(opaque_thread_handle)
            token = object()
            self._active_turns[token] = opaque_thread_handle
            self._admitted_turn_count += 1
            self._max_observed_in_flight_turns = max(
                self._max_observed_in_flight_turns,
                len(self._active_turns),
            )
            return BoundedDynamicAgentTurnAdmission(
                token=token,
                opaque_thread_handle=opaque_thread_handle,
                sequence_number=summary.completed_turn_count + 1,
            )

    def cancel_turn_admission(self, admission: BoundedDynamicAgentTurnAdmission) -> None:
        with self._lock:
            self._release_admission_locked(admission)

    def run_admitted_turn(
        self,
        *,
        admission: BoundedDynamicAgentTurnAdmission,
        question: PrimeQARuntimeQuery,
    ) -> BoundedDynamicAgentTurnRun:
        with self._lock:
            handle = self._active_turns.get(admission.token)
            if handle != admission.opaque_thread_handle:
                raise RuntimeError("turn admission is not active")
        try:
            runtime_run = self._runtime.run_turn(
                opaque_thread_handle=admission.opaque_thread_handle,
                question=question,
            )
            summary = self._runtime.thread_summary(admission.opaque_thread_handle)
        except ThreadStatePolicyViolationError as error:
            with self._lock:
                self._failed_turn_count += 1
            trace = self._runtime.last_public_trace
            code = (
                BoundedDynamicAgentServiceErrorCode.THREAD_STATE_LIMIT_EXCEEDED
                if trace is not None and trace.state_limit_rejected
                else BoundedDynamicAgentServiceErrorCode.TURN_EXECUTION_FAILED
            )
            raise BoundedDynamicAgentServiceError(code) from error
        except Exception as error:
            with self._lock:
                self._failed_turn_count += 1
            raise BoundedDynamicAgentServiceError(
                BoundedDynamicAgentServiceErrorCode.TURN_EXECUTION_FAILED
            ) from error
        else:
            with self._lock:
                self._completed_turn_count += 1
            return BoundedDynamicAgentTurnRun(
                runtime_run=runtime_run,
                thread_summary=summary,
            )
        finally:
            with self._lock:
                self._release_admission_locked(admission)

    def begin_draining(self) -> None:
        with self._lock:
            if self._state is BoundedDynamicAgentCoordinatorState.CLOSED:
                return
            if self._state is BoundedDynamicAgentCoordinatorState.CREATED:
                self._state = BoundedDynamicAgentCoordinatorState.CLOSED
                return
            self._state = BoundedDynamicAgentCoordinatorState.DRAINING

    def close(self) -> None:
        with self._lock:
            if self._state is BoundedDynamicAgentCoordinatorState.CLOSED:
                return
            if self._active_turns:
                raise RuntimeError("cannot close coordinator with an admitted turn")
            handles = tuple(sorted(self._open_handles))
            for handle in handles:
                self._runtime.close_thread(handle)
                self._open_handles.remove(handle)
            self._state = BoundedDynamicAgentCoordinatorState.CLOSED

    def _require_accepting_locked(self) -> None:
        if self._state is BoundedDynamicAgentCoordinatorState.ACCEPTING:
            return
        mapping = {
            BoundedDynamicAgentCoordinatorState.CREATED: (
                BoundedDynamicAgentServiceErrorCode.SERVICE_NOT_ACTIVE
            ),
            BoundedDynamicAgentCoordinatorState.DRAINING: (
                BoundedDynamicAgentServiceErrorCode.SERVICE_DRAINING
            ),
            BoundedDynamicAgentCoordinatorState.CLOSED: (
                BoundedDynamicAgentServiceErrorCode.SERVICE_CLOSED
            ),
        }
        raise BoundedDynamicAgentServiceError(mapping[self._state])

    def _require_open_locked(self, opaque_thread_handle: str) -> None:
        if opaque_thread_handle not in self._open_handles:
            raise BoundedDynamicAgentServiceError(
                BoundedDynamicAgentServiceErrorCode.THREAD_NOT_FOUND
            )

    def _release_admission_locked(self, admission: BoundedDynamicAgentTurnAdmission) -> None:
        handle = self._active_turns.get(admission.token)
        if handle != admission.opaque_thread_handle:
            raise RuntimeError("turn admission has already been released")
        del self._active_turns[admission.token]

    def _counters_locked(self) -> BoundedDynamicAgentServiceCounters:
        return BoundedDynamicAgentServiceCounters(
            opened_thread_count=len(self._open_handles),
            current_in_flight_turns=len(self._active_turns),
            max_observed_in_flight_turns=self._max_observed_in_flight_turns,
            admitted_turn_count=self._admitted_turn_count,
            capacity_rejected_turn_count=self._capacity_rejected_turn_count,
            completed_turn_count=self._completed_turn_count,
            failed_turn_count=self._failed_turn_count,
        )


class PrimeQAHybridBoundedDynamicAgentHttpTransport:
    """Five-route local adapter with no application waiting queue."""

    def __init__(
        self,
        *,
        settings: ProjectSettings,
        runtime: PrimeQAHybridBoundedDynamicAgentRuntime,
        log_sink: BoundedDynamicAgentHttpLogSink | None = None,
    ) -> None:
        self._settings = settings
        self._coordinator = BoundedDynamicAgentServiceCoordinator(runtime)
        self._log_sink = log_sink or PythonBoundedDynamicAgentHttpLogSink()
        self._lock = Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._started = False

    @property
    def coordinator(self) -> BoundedDynamicAgentServiceCoordinator:
        return self._coordinator

    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("bounded dynamic HTTP transport may start only once")
            self._started = True
            if not (
                self._settings.enable_bounded_dynamic_agent_runtime
                and self._settings.enable_bounded_dynamic_agent_http_transport
            ):
                return
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="ts-rag-bounded-agent-gpu",
            )
            self._coordinator.start()

    def shutdown(self) -> None:
        with self._lock:
            executor = self._executor
            self._coordinator.begin_draining()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        self._coordinator.close()

    async def open_thread(self, request: Request) -> Response:
        payload = await self._read_payload_or_response(
            request=request,
            route_id="thread_open",
            model=ThreadHandlePayload,
        )
        if isinstance(payload, Response):
            return payload
        try:
            summary = self._coordinator.open_thread(payload.thread_handle)
        except BoundedDynamicAgentServiceError as error:
            return self._service_error_response("thread_open", "POST", error)
        self._emit_summary(
            route_id="thread_open",
            method="POST",
            status=201,
            outcome="thread_opened",
            summary=summary,
        )
        return JSONResponse(
            BoundedDynamicAgentThreadPayload(
                thread_handle=payload.thread_handle,
                **summary.to_public_dict(),
            ).model_dump(mode="json"),
            status_code=201,
        )

    async def close_thread(self, request: Request) -> Response:
        payload = await self._read_payload_or_response(
            request=request,
            route_id="thread_close",
            model=ThreadHandlePayload,
        )
        if isinstance(payload, Response):
            return payload
        try:
            summary = self._coordinator.close_thread(payload.thread_handle)
        except BoundedDynamicAgentServiceError as error:
            return self._service_error_response("thread_close", "POST", error)
        self._emit_summary(
            route_id="thread_close",
            method="POST",
            status=200,
            outcome="thread_closed",
            summary=summary,
        )
        return JSONResponse(
            BoundedDynamicAgentThreadPayload(
                thread_handle=payload.thread_handle,
                **summary.to_public_dict(),
            ).model_dump(mode="json"),
            status_code=200,
        )

    async def run_turn(self, request: Request) -> Response:
        payload = await self._read_payload_or_response(
            request=request,
            route_id="thread_turn",
            model=ThreadTurnPayload,
        )
        if isinstance(payload, Response):
            return payload
        with self._lock:
            executor = self._executor
            try:
                admission = self._coordinator.admit_turn(payload.thread_handle)
            except BoundedDynamicAgentServiceError as error:
                return self._service_error_response("thread_turn", "POST", error)
            if executor is None:
                self._coordinator.cancel_turn_admission(admission)
                return self._error_response(
                    route_id="thread_turn",
                    method="POST",
                    status=503,
                    code="service_not_active",
                )
            question = PrimeQARuntimeQuery(
                id=f"bounded-dynamic-turn-{admission.sequence_number}",
                title=payload.title,
                text=payload.text,
            )
            loop = asyncio.get_running_loop()
            try:
                future = loop.run_in_executor(
                    executor,
                    lambda: self._coordinator.run_admitted_turn(
                        admission=admission,
                        question=question,
                    ),
                )
            except RuntimeError:
                self._coordinator.cancel_turn_admission(admission)
                return self._error_response(
                    route_id="thread_turn",
                    method="POST",
                    status=500,
                    code="internal_error",
                )
        try:
            turn_run = await future
        except BoundedDynamicAgentServiceError as error:
            return self._service_error_response(
                "thread_turn",
                "POST",
                error,
                downstream_dispatched=True,
                gpu_admitted=True,
            )
        except Exception:
            return self._error_response(
                route_id="thread_turn",
                method="POST",
                status=500,
                code="internal_error",
                downstream_dispatched=True,
                gpu_admitted=True,
            )
        return self._turn_success_response(payload.thread_handle, turn_run)

    def liveness_response(self) -> JSONResponse:
        self._emit_health(
            route_id="liveness",
            status=200,
            outcome="live",
        )
        return JSONResponse({"status": "live"}, status_code=200)

    def readiness_response(self) -> JSONResponse:
        state = self._coordinator.state
        ready = state is BoundedDynamicAgentCoordinatorState.ACCEPTING
        self._emit_health(
            route_id="readiness",
            status=200 if ready else 503,
            outcome="ready" if ready else "not_ready",
        )
        return JSONResponse(
            {"status": "ready" if ready else "not_ready", "coordinator_state": state.value},
            status_code=200 if ready else 503,
        )

    def framework_error_response(self, *, method: str, status: int) -> JSONResponse:
        code = "not_found" if status == 404 else "method_not_allowed"
        return self._error_response(
            route_id="unmatched_route",
            method=method,
            status=status,
            code=code,
        )

    async def _read_payload_or_response(
        self,
        *,
        request: Request,
        route_id: str,
        model: type[_PayloadT],
    ) -> _PayloadT | Response:
        try:
            return await _read_json_payload(request, model)
        except _KnownClientDisconnect:
            return Response(status_code=499)
        except _TransportInputError as error:
            return self._error_response(
                route_id=route_id,
                method="POST",
                status=error.status,
                code=error.code,
            )

    def _turn_success_response(
        self,
        thread_handle: str,
        turn_run: BoundedDynamicAgentTurnRun,
    ) -> JSONResponse:
        answer = turn_run.runtime_run.verified_answer
        trace = turn_run.runtime_run.public_safe_trace
        metrics = turn_run.runtime_run.workflow_run.router_metrics
        counters = self._coordinator.counters()
        event = PublicSafeBoundedDynamicAgentHttpEvent(
            route_id="thread_turn",
            method="POST",
            http_status=200,
            outcome_code=trace.terminal_state,
            coordinator_state=self._coordinator.state.value,
            downstream_dispatched=True,
            gpu_admitted=True,
            current_in_flight_turns=counters.current_in_flight_turns,
            max_observed_in_flight_turns=counters.max_observed_in_flight_turns,
            opened_thread_count=counters.opened_thread_count,
            completed_turn_count=turn_run.thread_summary.completed_turn_count,
            retained_state_bytes=turn_run.thread_summary.retained_state_bytes,
            terminal_state=trace.terminal_state,
            selected_action=trace.selected_action,
            model_decision_count=trace.model_decision_count,
            retrieval_call_count=trace.retrieval_call_count,
            composition_call_count=trace.composition_call_count,
            verification_call_count=trace.verification_call_count,
            diagnostic_observation_count=trace.diagnostic_observation_count,
            router_input_token_count=(metrics.input_token_count if metrics else 0),
            router_output_token_count=(metrics.output_token_count if metrics else 0),
            router_generation_latency_ms=(metrics.generation_latency_ms if metrics else 0.0),
        )
        self._log_sink.emit(event)
        payload = BoundedDynamicAgentTurnPayload(
            thread_handle=thread_handle,
            text=answer.answer,
            refused=answer.refused,
            citations=tuple(
                BoundedDynamicAgentCitationPayload(
                    document_reference=citation.document_id,
                    title=citation.title,
                    rank=citation.retrieval_rank,
                    evidence_score=citation.evidence_score,
                )
                for citation in answer.citations
            ),
            terminal_state=trace.terminal_state,
            completed_turn_count=turn_run.thread_summary.completed_turn_count,
            retained_state_bytes=turn_run.thread_summary.retained_state_bytes,
        )
        return JSONResponse(payload.model_dump(mode="json"), status_code=200)

    def _emit_health(self, *, route_id: str, status: int, outcome: str) -> None:
        counters = self._coordinator.counters()
        self._log_sink.emit(
            PublicSafeBoundedDynamicAgentHttpEvent(
                route_id=route_id,
                method="GET",
                http_status=status,
                outcome_code=outcome,
                coordinator_state=self._coordinator.state.value,
                downstream_dispatched=False,
                gpu_admitted=False,
                current_in_flight_turns=counters.current_in_flight_turns,
                max_observed_in_flight_turns=counters.max_observed_in_flight_turns,
                opened_thread_count=counters.opened_thread_count,
            )
        )

    def _emit_summary(
        self,
        *,
        route_id: str,
        method: str,
        status: int,
        outcome: str,
        summary: ThreadStateSummary,
    ) -> None:
        counters = self._coordinator.counters()
        self._log_sink.emit(
            PublicSafeBoundedDynamicAgentHttpEvent(
                route_id=route_id,
                method=method,
                http_status=status,
                outcome_code=outcome,
                coordinator_state=self._coordinator.state.value,
                downstream_dispatched=True,
                gpu_admitted=False,
                current_in_flight_turns=counters.current_in_flight_turns,
                max_observed_in_flight_turns=counters.max_observed_in_flight_turns,
                opened_thread_count=counters.opened_thread_count,
                completed_turn_count=summary.completed_turn_count,
                retained_state_bytes=summary.retained_state_bytes,
            )
        )

    def _service_error_response(
        self,
        route_id: str,
        method: str,
        error: BoundedDynamicAgentServiceError,
        *,
        downstream_dispatched: bool = False,
        gpu_admitted: bool = False,
    ) -> JSONResponse:
        status = {
            BoundedDynamicAgentServiceErrorCode.THREAD_NOT_FOUND: 404,
            BoundedDynamicAgentServiceErrorCode.THREAD_ALREADY_OPEN: 409,
            BoundedDynamicAgentServiceErrorCode.THREAD_BUSY: 409,
            BoundedDynamicAgentServiceErrorCode.THREAD_STATE_LIMIT_EXCEEDED: 409,
            BoundedDynamicAgentServiceErrorCode.SERVICE_NOT_ACTIVE: 503,
            BoundedDynamicAgentServiceErrorCode.SERVICE_DRAINING: 503,
            BoundedDynamicAgentServiceErrorCode.SERVICE_CLOSED: 503,
            BoundedDynamicAgentServiceErrorCode.GPU_CAPACITY_EXCEEDED: 503,
            BoundedDynamicAgentServiceErrorCode.TURN_EXECUTION_FAILED: 500,
        }[error.code]
        return self._error_response(
            route_id=route_id,
            method=method,
            status=status,
            code=error.code.value,
            downstream_dispatched=downstream_dispatched,
            gpu_admitted=gpu_admitted,
        )

    def _error_response(
        self,
        *,
        route_id: str,
        method: str,
        status: int,
        code: str,
        downstream_dispatched: bool = False,
        gpu_admitted: bool = False,
    ) -> JSONResponse:
        counters = self._coordinator.counters()
        self._log_sink.emit(
            PublicSafeBoundedDynamicAgentHttpEvent(
                route_id=route_id,
                method=method,
                http_status=status,
                outcome_code=code,
                coordinator_state=self._coordinator.state.value,
                downstream_dispatched=downstream_dispatched,
                gpu_admitted=gpu_admitted,
                current_in_flight_turns=counters.current_in_flight_turns,
                max_observed_in_flight_turns=counters.max_observed_in_flight_turns,
                opened_thread_count=counters.opened_thread_count,
            )
        )
        return _json_error(status=status, code=code)


@dataclass(frozen=True)
class _TransportInputError(Exception):
    status: int
    code: str


class _KnownClientDisconnect(RuntimeError):
    pass


class _DuplicateJsonKeyError(ValueError):
    pass


_PayloadT = TypeVar("_PayloadT", bound=BaseModel)


async def _read_json_payload(request: Request, model: type[_PayloadT]) -> _PayloadT:
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
        return model.model_validate(value)
    except ValidationError as error:
        raise _TransportInputError(status=422, code="invalid_request") from error


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


def _json_error(*, status: int, code: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": _ERROR_MESSAGES[code]}},
        status_code=status,
    )


def create_primeqa_hybrid_bounded_dynamic_agent_http_app(
    *,
    settings: ProjectSettings,
    runtime: PrimeQAHybridBoundedDynamicAgentRuntime,
    log_sink: BoundedDynamicAgentHttpLogSink | None = None,
) -> FastAPI:
    transport = PrimeQAHybridBoundedDynamicAgentHttpTransport(
        settings=settings,
        runtime=runtime,
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
        title="Technical Support Bounded Dynamic Agent",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.bounded_dynamic_agent_http_transport = transport

    @app.post(
        "/v1/bounded-agent/threads/open",
        include_in_schema=False,
        name="bounded_thread_open",
    )
    async def open_thread(request: Request) -> Response:
        return await transport.open_thread(request)

    @app.post(
        "/v1/bounded-agent/threads/turn",
        include_in_schema=False,
        name="bounded_thread_turn",
    )
    async def run_turn(request: Request) -> Response:
        return await transport.run_turn(request)

    @app.post(
        "/v1/bounded-agent/threads/close",
        include_in_schema=False,
        name="bounded_thread_close",
    )
    async def close_thread(request: Request) -> Response:
        return await transport.close_thread(request)

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
                method=request.method, status=error.status_code
            )
        return _json_error(status=500, code="internal_error")

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        error: RequestValidationError,
    ) -> Response:
        _ = error
        return transport._error_response(
            route_id="framework_validation",
            method=request.method,
            status=422,
            code="invalid_request",
        )

    return app


def create_primeqa_hybrid_bounded_dynamic_agent_uvicorn_config(
    *,
    app: FastAPI,
    port: int,
    log_level: str = "warning",
) -> uvicorn.Config:
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    return uvicorn.Config(
        app=app,
        host=BOUNDED_DYNAMIC_AGENT_BINDING_HOST,
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


def bounded_dynamic_agent_http_transport_contract() -> dict[str, Any]:
    return {
        "transport_id": BOUNDED_DYNAMIC_AGENT_HTTP_TRANSPORT_ID,
        "framework": "FastAPI",
        "framework_version": fastapi.__version__,
        "starlette_version": starlette.__version__,
        "uvicorn_version": uvicorn.__version__,
        "runtime_flag": "TS_RAG_ENABLE_BOUNDED_DYNAMIC_AGENT_RUNTIME",
        "transport_flag": "TS_RAG_ENABLE_BOUNDED_DYNAMIC_AGENT_HTTP_TRANSPORT",
        "model_snapshot_setting": "TS_RAG_BOUNDED_DYNAMIC_AGENT_MODEL_SNAPSHOT",
        "default_enabled": False,
        "binding_host": BOUNDED_DYNAMIC_AGENT_BINDING_HOST,
        "http_protocol": "HTTP/1.1",
        "routes": [
            {
                "route_id": "thread_open",
                "method": "POST",
                "path": "/v1/bounded-agent/threads/open",
            },
            {
                "route_id": "thread_turn",
                "method": "POST",
                "path": "/v1/bounded-agent/threads/turn",
            },
            {
                "route_id": "thread_close",
                "method": "POST",
                "path": "/v1/bounded-agent/threads/close",
            },
            {"route_id": "liveness", "method": "GET", "path": "/health/live"},
            {"route_id": "readiness", "method": "GET", "path": "/health/ready"},
        ],
        "max_in_flight_turns": BOUNDED_DYNAMIC_AGENT_MAX_IN_FLIGHT_TURNS,
        "open_close_consume_gpu_slot": False,
        "same_thread_parallel_turns_allowed": False,
        "close_while_turn_active_allowed": False,
        "application_waiting_queue": False,
        "request_timeout_seconds": None,
        "shutdown_waits_for_admitted_turn": True,
        "implicit_shutdown_timeout_seconds": None,
        "raw_body_max_bytes": _MAX_BODY_BYTES,
        "thread_handle_max_chars": _MAX_THREAD_HANDLE_CHARS,
        "title_max_chars": _MAX_TITLE_CHARS,
        "text_max_chars": _MAX_TEXT_CHARS,
        "access_log_enabled": False,
        "public_log_fields": sorted(_PUBLIC_EVENT_FIELDS),
        "runtime_registered_as_default": False,
        "existing_answer_route_changed": False,
        "remote_exposure_authorized": False,
        "persistent_state_enabled": False,
        "test_access_allowed": False,
        "queue_actions_allowed": False,
        "retry_actions_allowed": False,
        "fallback_strategies_allowed": False,
    }


def _find_forbidden_public_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in _FORBIDDEN_PUBLIC_KEYS:
                found.add(key_text)
            found.update(_find_forbidden_public_keys(child))
    elif isinstance(value, list | tuple):
        for child in value:
            found.update(_find_forbidden_public_keys(child))
    return found
