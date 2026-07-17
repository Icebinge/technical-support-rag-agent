from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, dataclass
from enum import Enum
from threading import Condition, Event
from typing import Any, Protocol

from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_activation import (
    PrimeQAHybridConcurrentRuntimeBootstrapResult,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_sidecar_agent_runtime import (
    ConcurrentArrivalPattern,
    PrimeQAHybridConcurrentCapacityExceededError,
    PublicSafeConcurrentRuntimeRequestTrace,
    concurrent_sidecar_runtime_contract,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    _forbidden_keys_found,
)
from ts_rag_agent.domain.answer import GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQAQuery, PrimeQARuntimeQuery

_FACADE_ID = "primeqa_hybrid_transport_neutral_agent_request_facade_v1"
_ACTIVE_RUNTIME_BINDING = object()
_ALLOWED_EVENT_FIELDS = frozenset(
    {
        "facade_state",
        "outcome_code",
        "downstream_dispatched",
        "queue_action_count",
        "retry_action_count",
        "fallback_action_count",
    }
)


class AgentRequestFacadeState(str, Enum):
    """Monotonic lifecycle states frozen by Stage147."""

    ACCEPTING = "accepting"
    DRAINING = "draining"
    CLOSED = "closed"


class CancellationSignalPort(Protocol):
    """Cooperative signal checked only before runtime dispatch."""

    def is_cancelled(self) -> bool: ...


class ConcurrentAgentRuntimeRunPort(Protocol):
    """Subset of one concurrent runtime result consumed by the facade."""

    public_safe_trace: PublicSafeConcurrentRuntimeRequestTrace

    @property
    def verified_answer(self) -> GeneratedAnswer: ...


class ConcurrentAgentRuntimePort(Protocol):
    """Active Stage146 runtime operation consumed by the facade."""

    def run(
        self,
        question: PrimeQAQuery,
        *,
        arrival_pattern: ConcurrentArrivalPattern,
    ) -> ConcurrentAgentRuntimeRunPort: ...


class AgentRequestCancellationSignal:
    """Thread-safe caller-owned cooperative pre-dispatch cancellation signal."""

    def __init__(self) -> None:
        self._cancelled = Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()


@dataclass(frozen=True)
class AgentFacadeRequest:
    """Private transport-neutral request; validation occurs at the facade boundary."""

    request_handle: str
    text: str
    title: str | None = None
    cancellation_signal: CancellationSignalPort | None = None


@dataclass(frozen=True)
class AgentFacadeCitation:
    """Private citation mapped exactly from the verified answer."""

    document_reference: str
    title: str
    rank: int
    evidence_score: float


@dataclass(frozen=True)
class AgentFacadeResponse:
    """Private facade response; it is deliberately not public-serializable."""

    request_handle: str
    text: str
    refused: bool
    citations: tuple[AgentFacadeCitation, ...]


@dataclass(frozen=True)
class PublicSafeAgentRequestFacadeEvent:
    """Six-field Stage147 allowlisted public event for one facade outcome."""

    facade_state: str
    outcome_code: str
    downstream_dispatched: bool
    queue_action_count: int = 0
    retry_action_count: int = 0
    fallback_action_count: int = 0

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if set(payload) != _ALLOWED_EVENT_FIELDS:
            raise ValueError("facade event fields do not match the Stage147 allowlist")
        forbidden = sorted(_forbidden_keys_found(payload))
        if forbidden:
            raise ValueError(f"facade event contains forbidden keys: {forbidden}")
        return payload


@dataclass(frozen=True)
class AgentRequestFacadeRun:
    """Private response paired with separately serializable public telemetry."""

    response: AgentFacadeResponse
    public_safe_event: PublicSafeAgentRequestFacadeEvent
    public_safe_runtime_trace: PublicSafeConcurrentRuntimeRequestTrace

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "facade_event": self.public_safe_event.to_public_dict(),
            "runtime_request": self.public_safe_runtime_trace.to_public_dict(),
        }


@dataclass(frozen=True)
class AgentRequestFacadeCounters:
    """Aggregate counters read atomically without request content or identifiers."""

    invocation_attempt_count: int
    accepted_call_count: int
    runtime_dispatch_count: int
    completed_response_count: int
    refused_response_count: int
    invalid_rejected_count: int
    lifecycle_rejected_count: int
    cancelled_before_dispatch_count: int
    capacity_rejected_count: int
    downstream_error_count: int
    current_in_flight: int
    max_observed_in_flight: int
    queue_action_count: int = 0
    retry_action_count: int = 0
    fallback_action_count: int = 0


class AgentRequestFacadeError(RuntimeError):
    """Base for typed facade-owned failures carrying only public-safe telemetry."""

    code: str

    def __init__(
        self,
        *,
        message: str,
        public_safe_event: PublicSafeAgentRequestFacadeEvent,
        public_safe_runtime_trace: PublicSafeConcurrentRuntimeRequestTrace | None = None,
    ) -> None:
        super().__init__(message)
        self.public_safe_event = public_safe_event
        self.public_safe_runtime_trace = public_safe_runtime_trace


class AgentRequestFacadeNotActiveError(AgentRequestFacadeError):
    code = "facade_not_active"


class AgentRequestFacadeInvalidRequestError(AgentRequestFacadeError):
    code = "invalid_request"

    def __init__(
        self,
        *,
        reasons: tuple[str, ...],
        public_safe_event: PublicSafeAgentRequestFacadeEvent,
    ) -> None:
        super().__init__(
            message="agent facade request is invalid",
            public_safe_event=public_safe_event,
        )
        self.reasons = reasons


class AgentRequestFacadeCancelledError(AgentRequestFacadeError):
    code = "cancelled_before_dispatch"


class AgentRequestFacadeCapacityExceededError(AgentRequestFacadeError):
    code = "capacity_exceeded"


class AgentRequestFacadeDrainingError(AgentRequestFacadeError):
    code = "facade_draining"


class AgentRequestFacadeClosedError(AgentRequestFacadeError):
    code = "facade_closed"


class PrimeQAHybridAgentRequestFacade:
    """Transport-neutral synchronous facade over one active Stage146 runtime."""

    def __init__(
        self,
        *,
        runtime: ConcurrentAgentRuntimePort,
        _active_runtime_binding: object,
    ) -> None:
        if _active_runtime_binding is not _ACTIVE_RUNTIME_BINDING:
            raise ValueError("facade construction requires an active bootstrap binding")
        self._runtime = runtime
        self._condition = Condition()
        self._state = AgentRequestFacadeState.ACCEPTING
        self._last_public_event: ContextVar[PublicSafeAgentRequestFacadeEvent | None] = ContextVar(
            f"primeqa_hybrid_facade_event_{id(self)}", default=None
        )
        self._invocation_attempt_count = 0
        self._accepted_call_count = 0
        self._runtime_dispatch_count = 0
        self._completed_response_count = 0
        self._refused_response_count = 0
        self._invalid_rejected_count = 0
        self._lifecycle_rejected_count = 0
        self._cancelled_before_dispatch_count = 0
        self._capacity_rejected_count = 0
        self._downstream_error_count = 0
        self._current_in_flight = 0
        self._max_observed_in_flight = 0

    @property
    def state(self) -> AgentRequestFacadeState:
        with self._condition:
            return self._state

    @property
    def last_public_event(self) -> PublicSafeAgentRequestFacadeEvent | None:
        """Return the event for the current request context, including failures."""

        return self._last_public_event.get()

    def counters(self) -> AgentRequestFacadeCounters:
        with self._condition:
            return AgentRequestFacadeCounters(
                invocation_attempt_count=self._invocation_attempt_count,
                accepted_call_count=self._accepted_call_count,
                runtime_dispatch_count=self._runtime_dispatch_count,
                completed_response_count=self._completed_response_count,
                refused_response_count=self._refused_response_count,
                invalid_rejected_count=self._invalid_rejected_count,
                lifecycle_rejected_count=self._lifecycle_rejected_count,
                cancelled_before_dispatch_count=self._cancelled_before_dispatch_count,
                capacity_rejected_count=self._capacity_rejected_count,
                downstream_error_count=self._downstream_error_count,
                current_in_flight=self._current_in_flight,
                max_observed_in_flight=self._max_observed_in_flight,
            )

    def wait_until_state(self, target: AgentRequestFacadeState) -> None:
        """Wait naturally for a monotonic lifecycle state without a timeout."""

        order = {
            AgentRequestFacadeState.ACCEPTING: 0,
            AgentRequestFacadeState.DRAINING: 1,
            AgentRequestFacadeState.CLOSED: 2,
        }
        with self._condition:
            while self._state is not target:
                if order[self._state] > order[target]:
                    raise RuntimeError("facade lifecycle has already passed the target state")
                self._condition.wait()

    def invoke(self, request: AgentFacadeRequest) -> AgentRequestFacadeRun:
        """Execute one private request while preserving the frozen public boundary."""

        self._last_public_event.set(None)
        with self._condition:
            self._invocation_attempt_count += 1

        invalid_reasons = _invalid_request_reasons(request)
        if invalid_reasons:
            with self._condition:
                self._invalid_rejected_count += 1
                state = self._state
            event = _event(state=state, outcome_code="invalid_request", dispatched=False)
            self._record_event(event)
            raise AgentRequestFacadeInvalidRequestError(
                reasons=invalid_reasons,
                public_safe_event=event,
            )

        with self._condition:
            if self._state is AgentRequestFacadeState.DRAINING:
                self._lifecycle_rejected_count += 1
                event = _event(
                    state=self._state,
                    outcome_code="facade_draining",
                    dispatched=False,
                )
                self._record_event(event)
                raise AgentRequestFacadeDrainingError(
                    message="agent request facade is draining",
                    public_safe_event=event,
                )
            if self._state is AgentRequestFacadeState.CLOSED:
                self._lifecycle_rejected_count += 1
                event = _event(
                    state=self._state,
                    outcome_code="facade_closed",
                    dispatched=False,
                )
                self._record_event(event)
                raise AgentRequestFacadeClosedError(
                    message="agent request facade is closed",
                    public_safe_event=event,
                )
            self._accepted_call_count += 1
            self._current_in_flight += 1
            self._max_observed_in_flight = max(
                self._max_observed_in_flight,
                self._current_in_flight,
            )

        try:
            signal = request.cancellation_signal
            if signal is not None and signal.is_cancelled():
                with self._condition:
                    self._cancelled_before_dispatch_count += 1
                    state = self._state
                event = _event(
                    state=state,
                    outcome_code="cancelled_before_dispatch",
                    dispatched=False,
                )
                self._record_event(event)
                raise AgentRequestFacadeCancelledError(
                    message="agent facade request was cancelled before dispatch",
                    public_safe_event=event,
                )

            runtime_query = PrimeQARuntimeQuery(
                id=request.request_handle,
                title=request.title or "",
                text=request.text,
            )
            with self._condition:
                self._runtime_dispatch_count += 1
            try:
                runtime_run = self._runtime.run(
                    runtime_query,
                    arrival_pattern=ConcurrentArrivalPattern.APPLICATION,
                )
                runtime_run.public_safe_trace.to_public_dict()
                response = _private_response(
                    request_handle=request.request_handle,
                    answer=runtime_run.verified_answer,
                )
            except PrimeQAHybridConcurrentCapacityExceededError as error:
                with self._condition:
                    self._capacity_rejected_count += 1
                    state = self._state
                event = _event(
                    state=state,
                    outcome_code="capacity_exceeded",
                    dispatched=False,
                )
                self._record_event(event)
                raise AgentRequestFacadeCapacityExceededError(
                    message="agent runtime capacity is exhausted",
                    public_safe_event=event,
                    public_safe_runtime_trace=error.public_safe_trace,
                ) from error
            except Exception:
                with self._condition:
                    self._downstream_error_count += 1
                    state = self._state
                self._record_event(
                    _event(
                        state=state,
                        outcome_code="downstream_error",
                        dispatched=True,
                    )
                )
                raise

            with self._condition:
                self._completed_response_count += 1
                if response.refused:
                    self._refused_response_count += 1
                state = self._state
            event = _event(
                state=state,
                outcome_code="refuse" if response.refused else "complete",
                dispatched=True,
            )
            self._record_event(event)
            return AgentRequestFacadeRun(
                response=response,
                public_safe_event=event,
                public_safe_runtime_trace=runtime_run.public_safe_trace,
            )
        finally:
            with self._condition:
                self._current_in_flight -= 1
                self._condition.notify_all()

    def shutdown(self) -> None:
        """Reject new work, wait naturally for in-flight calls, then close."""

        with self._condition:
            if self._state is AgentRequestFacadeState.CLOSED:
                return
            if self._state is AgentRequestFacadeState.ACCEPTING:
                self._state = AgentRequestFacadeState.DRAINING
                self._condition.notify_all()
            while self._current_in_flight:
                self._condition.wait()
            self._state = AgentRequestFacadeState.CLOSED
            self._condition.notify_all()

    def _record_event(self, event: PublicSafeAgentRequestFacadeEvent) -> None:
        event.to_public_dict()
        self._last_public_event.set(event)


def create_primeqa_hybrid_agent_request_facade(
    *,
    bootstrap_result: PrimeQAHybridConcurrentRuntimeBootstrapResult,
) -> PrimeQAHybridAgentRequestFacade:
    """Create a facade only from an eligible active Stage146 bootstrap result."""

    startup = bootstrap_result.startup_trace
    runtime = bootstrap_result.runtime
    if not (
        runtime is not None
        and startup.activation_state == "eligible"
        and startup.runtime_activated
        and startup.warm_resources_ready
    ):
        event = PublicSafeAgentRequestFacadeEvent(
            facade_state="not_active",
            outcome_code="facade_not_active",
            downstream_dispatched=False,
        )
        event.to_public_dict()
        raise AgentRequestFacadeNotActiveError(
            message="an eligible active concurrent runtime is required",
            public_safe_event=event,
        )
    return PrimeQAHybridAgentRequestFacade(
        runtime=runtime,
        _active_runtime_binding=_ACTIVE_RUNTIME_BINDING,
    )


def agent_request_facade_contract() -> dict[str, Any]:
    """Return the Stage148 implementation contract without network registration."""

    runtime_contract = concurrent_sidecar_runtime_contract()
    return {
        "facade_id": _FACADE_ID,
        "runtime_dependency": runtime_contract["runtime_mode"],
        "application_arrival_pattern": ConcurrentArrivalPattern.APPLICATION.value,
        "runtime_query_type": "PrimeQARuntimeQuery",
        "runtime_query_fields": ["id", "title", "text"],
        "runtime_query_contains_gold_labels": False,
        "private_request_fields": [
            "request_handle",
            "title",
            "text",
            "cancellation_signal",
        ],
        "private_response_fields": ["request_handle", "text", "refused", "citations"],
        "private_citation_fields": [
            "document_reference",
            "title",
            "rank",
            "evidence_score",
        ],
        "public_event_allowed_fields": sorted(_ALLOWED_EVENT_FIELDS),
        "public_runtime_trace_allowed_fields": runtime_contract["request_trace_allowed_fields"],
        "capacity_source_error": "PrimeQAHybridConcurrentCapacityExceededError",
        "capacity_facade_error": "AgentRequestFacadeCapacityExceededError",
        "lifecycle_states": [state.value for state in AgentRequestFacadeState],
        "shutdown_waits_without_implicit_timeout": True,
        "facade_owns_runtime_resources": False,
        "registered_as_runtime_default": False,
        "network_service_implemented": False,
        "test_access_allowed": False,
        "queue_actions_allowed": False,
        "retry_actions_allowed": False,
        "fallback_strategies_allowed": False,
        "downstream_errors_propagate_unchanged": True,
    }


def _invalid_request_reasons(request: Any) -> tuple[str, ...]:
    if not isinstance(request, AgentFacadeRequest):
        return ("request_type_invalid",)
    reasons: list[str] = []
    if not isinstance(request.request_handle, str) or not request.request_handle.strip():
        reasons.append("request_handle_required")
    if not isinstance(request.text, str) or not request.text.strip():
        reasons.append("request_text_required")
    if request.title is not None and not isinstance(request.title, str):
        reasons.append("request_title_type_invalid")
    signal = request.cancellation_signal
    if signal is not None and not callable(getattr(signal, "is_cancelled", None)):
        reasons.append("cancellation_signal_type_invalid")
    return tuple(reasons)


def _private_response(*, request_handle: str, answer: GeneratedAnswer) -> AgentFacadeResponse:
    if answer.question_id != request_handle:
        raise RuntimeError("verified answer does not match the facade request")
    return AgentFacadeResponse(
        request_handle=request_handle,
        text=answer.answer,
        refused=answer.refused,
        citations=tuple(
            AgentFacadeCitation(
                document_reference=citation.document_id,
                title=citation.title,
                rank=citation.retrieval_rank,
                evidence_score=citation.evidence_score,
            )
            for citation in answer.citations
        ),
    )


def _event(
    *,
    state: AgentRequestFacadeState,
    outcome_code: str,
    dispatched: bool,
) -> PublicSafeAgentRequestFacadeEvent:
    return PublicSafeAgentRequestFacadeEvent(
        facade_state=state.value,
        outcome_code=outcome_code,
        downstream_dispatched=dispatched,
    )
