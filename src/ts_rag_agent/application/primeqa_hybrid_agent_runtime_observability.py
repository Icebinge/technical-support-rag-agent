from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any, Protocol, TextIO

from ts_rag_agent.application.primeqa_hybrid_agent_tool_orchestration_protocol import (
    AGENT_TOOL_ORCHESTRATION_PROTOCOL_ID,
    AGENT_TOOL_WORKFLOW_GRAPH_ID,
    agent_tool_workflow_state_contract,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    _forbidden_keys_found,
)

AGENT_RUNTIME_OBSERVABILITY_PROTOCOL_ID = "primeqa_hybrid_agent_runtime_activation_observability_v1"
AGENT_RUNTIME_OBSERVABILITY_SCHEMA_VERSION = "1.0"

_WORKFLOW_NODE_ID = "workflow"
_EVENT_TYPES = (
    "workflow_started",
    "node_completed",
    "node_failed",
    "workflow_completed",
    "workflow_failed",
)
_OUTCOMES = ("started", "completed", "failed")
_NODE_IDS = tuple(agent_tool_workflow_state_contract()["nodes"])
_ALLOWED_EVENT_FIELDS = frozenset(
    {
        "observability_protocol_id",
        "schema_version",
        "workflow_protocol_id",
        "graph_id",
        "invocation_sequence",
        "event_sequence",
        "event_type",
        "node_id",
        "outcome",
        "node_latency_ms",
        "workflow_elapsed_ms",
        "current_state",
        "transition_count",
        "tool_call_count",
        "candidate_pool_depth",
        "generation_context_count",
        "verification_context_count",
        "failure_stage",
        "current_in_flight",
        "queue_action_count",
        "retry_action_count",
        "fallback_action_count",
    }
)


@dataclass(frozen=True)
class PublicSafeAgentWorkflowObservationEvent:
    """One content-free operational event for the local Agent workflow."""

    observability_protocol_id: str
    schema_version: str
    workflow_protocol_id: str
    graph_id: str
    invocation_sequence: int
    event_sequence: int
    event_type: str
    node_id: str
    outcome: str
    node_latency_ms: float
    workflow_elapsed_ms: float
    current_state: str
    transition_count: int
    tool_call_count: int
    candidate_pool_depth: int
    generation_context_count: int
    verification_context_count: int
    failure_stage: str | None
    current_in_flight: int
    queue_action_count: int = 0
    retry_action_count: int = 0
    fallback_action_count: int = 0

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if set(payload) != _ALLOWED_EVENT_FIELDS:
            raise ValueError("workflow observation fields do not match the Stage155 allowlist")
        if self.event_type not in _EVENT_TYPES:
            raise ValueError("workflow observation event type is not allowed")
        if self.outcome not in _OUTCOMES:
            raise ValueError("workflow observation outcome is not allowed")
        if self.node_id not in {*_NODE_IDS, _WORKFLOW_NODE_ID}:
            raise ValueError("workflow observation node id is not allowed")
        forbidden = sorted(_forbidden_keys_found(payload))
        if forbidden:
            raise ValueError(f"workflow observation contains forbidden keys: {forbidden}")
        return payload


class AgentWorkflowObservationSink(Protocol):
    """Synchronous delivery port for validated public-safe events."""

    def emit(self, event: PublicSafeAgentWorkflowObservationEvent) -> None: ...


class JsonLineAgentWorkflowObservationSink:
    """Write each validated event as one locked and flushed JSON line."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stderr
        self._lock = Lock()

    def emit(self, event: PublicSafeAgentWorkflowObservationEvent) -> None:
        line = json.dumps(
            event.to_public_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()


@dataclass(frozen=True)
class AgentWorkflowObservationCounters:
    invocation_started_count: int
    invocation_completed_count: int
    invocation_failed_count: int
    node_completed_count: int
    node_failed_count: int
    emitted_event_count: int
    delivery_failure_count: int
    current_in_flight: int
    max_observed_in_flight: int


@dataclass
class _InvocationObservationState:
    invocation_sequence: int
    started_at: float
    current_in_flight: int
    next_event_sequence: int = 1


class PrimeQAHybridAgentWorkflowObserver:
    """Track request-isolated node timings without retaining request content."""

    def __init__(
        self,
        *,
        sink: AgentWorkflowObservationSink,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._sink = sink
        self._clock = clock
        self._lock = Lock()
        self._token: ContextVar[object | None] = ContextVar(
            f"primeqa_hybrid_agent_observation_{id(self)}",
            default=None,
        )
        self._invocations: dict[object, _InvocationObservationState] = {}
        self._invocation_started_count = 0
        self._invocation_completed_count = 0
        self._invocation_failed_count = 0
        self._node_completed_count = 0
        self._node_failed_count = 0
        self._emitted_event_count = 0
        self._delivery_failure_count = 0
        self._current_in_flight = 0
        self._max_observed_in_flight = 0

    def begin(self, state: Mapping[str, Any]) -> None:
        if self._token.get() is not None:
            raise RuntimeError("workflow observation invocation is already active")
        token = object()
        started_at = self._clock()
        with self._lock:
            self._invocation_started_count += 1
            self._current_in_flight += 1
            self._max_observed_in_flight = max(
                self._max_observed_in_flight,
                self._current_in_flight,
            )
            invocation = _InvocationObservationState(
                invocation_sequence=self._invocation_started_count,
                started_at=started_at,
                current_in_flight=self._current_in_flight,
            )
            self._invocations[token] = invocation
        self._token.set(token)
        try:
            self._emit(
                event_type="workflow_started",
                node_id=_WORKFLOW_NODE_ID,
                outcome="started",
                state=state,
                node_latency_ms=0.0,
            )
        except BaseException:
            self.end()
            raise

    def node_started(self) -> float:
        self._active_invocation()
        return self._clock()

    def node_completed(
        self,
        *,
        node_id: str,
        state: Mapping[str, Any],
        started_at: float,
    ) -> None:
        if node_id not in _NODE_IDS:
            raise ValueError("completed observation node is not in the frozen graph")
        with self._lock:
            self._node_completed_count += 1
        self._emit(
            event_type="node_completed",
            node_id=node_id,
            outcome="completed",
            state=state,
            node_latency_ms=_milliseconds(self._clock() - started_at),
        )

    def node_failed(
        self,
        *,
        node_id: str,
        state: Mapping[str, Any],
        started_at: float,
    ) -> None:
        if node_id not in _NODE_IDS:
            raise ValueError("failed observation node is not in the frozen graph")
        with self._lock:
            self._node_failed_count += 1
        self._emit(
            event_type="node_failed",
            node_id=node_id,
            outcome="failed",
            state=state,
            node_latency_ms=_milliseconds(self._clock() - started_at),
        )

    def complete(self, state: Mapping[str, Any]) -> None:
        self._emit(
            event_type="workflow_completed",
            node_id=_WORKFLOW_NODE_ID,
            outcome="completed",
            state=state,
            node_latency_ms=0.0,
        )
        with self._lock:
            self._invocation_completed_count += 1

    def fail(self, state: Mapping[str, Any]) -> None:
        self._emit(
            event_type="workflow_failed",
            node_id=_WORKFLOW_NODE_ID,
            outcome="failed",
            state=state,
            node_latency_ms=0.0,
        )
        with self._lock:
            self._invocation_failed_count += 1

    def end(self) -> None:
        token = self._token.get()
        if token is None:
            return
        with self._lock:
            removed = self._invocations.pop(token, None)
            if removed is not None:
                self._current_in_flight -= 1
        self._token.set(None)

    def counters(self) -> AgentWorkflowObservationCounters:
        with self._lock:
            return AgentWorkflowObservationCounters(
                invocation_started_count=self._invocation_started_count,
                invocation_completed_count=self._invocation_completed_count,
                invocation_failed_count=self._invocation_failed_count,
                node_completed_count=self._node_completed_count,
                node_failed_count=self._node_failed_count,
                emitted_event_count=self._emitted_event_count,
                delivery_failure_count=self._delivery_failure_count,
                current_in_flight=self._current_in_flight,
                max_observed_in_flight=self._max_observed_in_flight,
            )

    def _emit(
        self,
        *,
        event_type: str,
        node_id: str,
        outcome: str,
        state: Mapping[str, Any],
        node_latency_ms: float,
    ) -> None:
        token, invocation = self._active_invocation()
        now = self._clock()
        with self._lock:
            active = self._invocations.get(token)
            if active is not invocation:
                raise RuntimeError("workflow observation invocation state changed unexpectedly")
            event_sequence = invocation.next_event_sequence
            invocation.next_event_sequence += 1
        event = _observation_event(
            invocation=invocation,
            event_sequence=event_sequence,
            event_type=event_type,
            node_id=node_id,
            outcome=outcome,
            node_latency_ms=node_latency_ms,
            workflow_elapsed_ms=_milliseconds(now - invocation.started_at),
            state=state,
        )
        event.to_public_dict()
        try:
            self._sink.emit(event)
        except BaseException:
            with self._lock:
                self._delivery_failure_count += 1
            raise
        with self._lock:
            self._emitted_event_count += 1

    def _active_invocation(self) -> tuple[object, _InvocationObservationState]:
        token = self._token.get()
        if token is None:
            raise RuntimeError("workflow observation requires an active invocation")
        with self._lock:
            invocation = self._invocations.get(token)
        if invocation is None:
            raise RuntimeError("workflow observation invocation state is missing")
        return token, invocation


def agent_runtime_activation_observability_contract() -> dict[str, Any]:
    """Return the strict Stage155 activation and observability contract."""

    return {
        "protocol_id": AGENT_RUNTIME_OBSERVABILITY_PROTOCOL_ID,
        "schema_version": AGENT_RUNTIME_OBSERVABILITY_SCHEMA_VERSION,
        "workflow_protocol_id": AGENT_TOOL_ORCHESTRATION_PROTOCOL_ID,
        "graph_id": AGENT_TOOL_WORKFLOW_GRAPH_ID,
        "required_explicit_environment_flags": [
            "TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT",
            "TS_RAG_ENABLE_LOCAL_AGENT_HTTP_TRANSPORT",
        ],
        "stage154_formal_evidence_required": True,
        "stage154_current_source_fingerprints_required": True,
        "binding_host": "127.0.0.1",
        "runtime_registered_as_default": False,
        "remote_exposure_authorized": False,
        "test_access_allowed": False,
        "observability_disable_flag": None,
        "delivery_mode": "synchronous_validated_json_line",
        "default_sink": "JsonLineAgentWorkflowObservationSink",
        "event_types": list(_EVENT_TYPES),
        "node_ids": list(_NODE_IDS),
        "workflow_event_node_id": _WORKFLOW_NODE_ID,
        "public_event_fields": sorted(_ALLOWED_EVENT_FIELDS),
        "public_event_field_count": len(_ALLOWED_EVENT_FIELDS),
        "wall_clock_timestamp_recorded": False,
        "request_content_recorded": False,
        "request_identifiers_recorded": False,
        "document_content_recorded": False,
        "document_identifiers_recorded": False,
        "sampling_enabled": False,
        "batching_enabled": False,
        "remote_export_enabled": False,
        "queue_actions_allowed": False,
        "retry_actions_allowed": False,
        "fallback_strategies_allowed": False,
    }


def _observation_event(
    *,
    invocation: _InvocationObservationState,
    event_sequence: int,
    event_type: str,
    node_id: str,
    outcome: str,
    node_latency_ms: float,
    workflow_elapsed_ms: float,
    state: Mapping[str, Any],
) -> PublicSafeAgentWorkflowObservationEvent:
    tool_counts = state.get("tool_call_counts") or {}
    current_state = state.get("current_state")
    current_state_value = getattr(current_state, "value", str(current_state or ""))
    return PublicSafeAgentWorkflowObservationEvent(
        observability_protocol_id=AGENT_RUNTIME_OBSERVABILITY_PROTOCOL_ID,
        schema_version=AGENT_RUNTIME_OBSERVABILITY_SCHEMA_VERSION,
        workflow_protocol_id=AGENT_TOOL_ORCHESTRATION_PROTOCOL_ID,
        graph_id=AGENT_TOOL_WORKFLOW_GRAPH_ID,
        invocation_sequence=invocation.invocation_sequence,
        event_sequence=event_sequence,
        event_type=event_type,
        node_id=node_id,
        outcome=outcome,
        node_latency_ms=node_latency_ms,
        workflow_elapsed_ms=workflow_elapsed_ms,
        current_state=current_state_value,
        transition_count=max(len(state.get("visited_states") or ()) - 1, 0),
        tool_call_count=sum(int(value) for value in tool_counts.values()),
        candidate_pool_depth=len(state.get("candidate_pool_results") or ()),
        generation_context_count=len(state.get("generation_context_results") or ()),
        verification_context_count=len(state.get("verification_context_results") or ()),
        failure_stage=state.get("failure_stage"),
        current_in_flight=invocation.current_in_flight,
    )


def _milliseconds(seconds: float) -> float:
    return round(max(seconds, 0.0) * 1000, 3)
