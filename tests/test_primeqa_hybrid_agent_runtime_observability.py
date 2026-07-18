from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest

from ts_rag_agent.application.primeqa_hybrid_agent_runtime_observability import (
    AGENT_RUNTIME_OBSERVABILITY_PROTOCOL_ID,
    JsonLineAgentWorkflowObservationSink,
    PublicSafeAgentWorkflowObservationEvent,
    agent_runtime_activation_observability_contract,
)
from ts_rag_agent.application.primeqa_hybrid_agent_tool_workflow import (
    create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset,
)
from ts_rag_agent.application.primeqa_hybrid_agent_tool_workflow_validation import (
    _FourRequestRetriever,
    _question,
    _StaticRetriever,
    _toolset,
)


class RecordingObservationSink:
    def __init__(self) -> None:
        self._lock = Lock()
        self.events: list[PublicSafeAgentWorkflowObservationEvent] = []

    def emit(self, event: PublicSafeAgentWorkflowObservationEvent) -> None:
        event.to_public_dict()
        with self._lock:
            self.events.append(event)


@pytest.mark.parametrize(
    ("refuse", "terminal_state"),
    [(False, "complete"), (True, "refuse")],
)
def test_complete_and_refuse_runs_emit_exact_node_timeline(
    refuse: bool,
    terminal_state: str,
) -> None:
    sink = RecordingObservationSink()
    workflow = create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
        toolset=_toolset(_StaticRetriever(prefix="observed"), refuse=refuse),
        observation_sink=sink,
    )

    run = workflow.run(_question("private-observation-handle"))

    assert run.public_safe_trace.terminal_state == terminal_state
    assert [event.event_sequence for event in sink.events] == list(range(1, 10))
    assert [event.event_type for event in sink.events] == [
        "workflow_started",
        *(["node_completed"] * 7),
        "workflow_completed",
    ]
    assert [event.node_id for event in sink.events[1:-1]] == [
        "validate_request",
        "retrieve_candidate_pool",
        "prepare_context",
        "compose_grounded_answer",
        "verify_grounded_answer",
        "observe_diagnostics",
        "finalize_response",
    ]
    assert sink.events[-1].current_state == terminal_state
    assert sink.events[-1].candidate_pool_depth == 400
    assert sink.events[-1].generation_context_count == 10
    assert sink.events[-1].verification_context_count == 200
    assert sink.events[-1].tool_call_count == 3
    assert all(event.node_latency_ms >= 0 for event in sink.events)
    assert all(event.workflow_elapsed_ms >= 0 for event in sink.events)
    assert "private-observation-handle" not in str(
        [event.to_public_dict() for event in sink.events]
    )
    counters = workflow.observation_counters()
    assert counters.invocation_started_count == 1
    assert counters.invocation_completed_count == 1
    assert counters.invocation_failed_count == 0
    assert counters.node_completed_count == 7
    assert counters.node_failed_count == 0
    assert counters.emitted_event_count == 9
    assert counters.delivery_failure_count == 0
    assert counters.current_in_flight == 0


def test_tool_failure_emits_failed_node_and_workflow_without_error_content() -> None:
    error = RuntimeError("private retrieval failure text")
    sink = RecordingObservationSink()
    workflow = create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
        toolset=_toolset(_StaticRetriever(prefix="failure", error=error), refuse=False),
        observation_sink=sink,
    )

    with pytest.raises(RuntimeError) as captured:
        workflow.run(_question("private-failure-handle"))

    assert captured.value is error
    assert [event.event_type for event in sink.events] == [
        "workflow_started",
        "node_completed",
        "node_failed",
        "workflow_failed",
    ]
    assert sink.events[2].node_id == "retrieve_candidate_pool"
    assert sink.events[2].failure_stage == "retrieve_candidate_pool"
    assert sink.events[-1].failure_stage == "retrieve_candidate_pool"
    public = [event.to_public_dict() for event in sink.events]
    assert "private retrieval failure text" not in str(public)
    assert "private-failure-handle" not in str(public)
    counters = workflow.observation_counters()
    assert counters.invocation_failed_count == 1
    assert counters.node_completed_count == 1
    assert counters.node_failed_count == 1
    assert counters.emitted_event_count == 4


def test_four_concurrent_invocations_have_isolated_sequences() -> None:
    sink = RecordingObservationSink()
    workflow = create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
        toolset=_toolset(_FourRequestRetriever(), refuse=False),
        observation_sink=sink,
    )

    handles = ("alpha", "beta", "gamma", "delta")
    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(lambda handle: workflow.run(_question(handle)), handles))

    assert len(runs) == 4
    grouped = {
        sequence: [event for event in sink.events if event.invocation_sequence == sequence]
        for sequence in {event.invocation_sequence for event in sink.events}
    }
    assert len(sink.events) == 36
    assert set(grouped) == {1, 2, 3, 4}
    for events in grouped.values():
        assert [event.event_sequence for event in events] == list(range(1, 10))
        assert events[0].event_type == "workflow_started"
        assert events[-1].event_type == "workflow_completed"
        assert {event.current_in_flight for event in events} <= {1, 2, 3, 4}
    counters = workflow.observation_counters()
    assert counters.invocation_completed_count == 4
    assert counters.node_completed_count == 28
    assert counters.emitted_event_count == 36
    assert counters.current_in_flight == 0
    assert counters.max_observed_in_flight == 4


def test_json_line_sink_writes_only_validated_public_event() -> None:
    recording = RecordingObservationSink()
    workflow = create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
        toolset=_toolset(_StaticRetriever(prefix="json"), refuse=False),
        observation_sink=recording,
    )
    workflow.run(_question("private-json-handle"))
    stream = io.StringIO()
    sink = JsonLineAgentWorkflowObservationSink(stream=stream)

    sink.emit(recording.events[-1])

    payload = json.loads(stream.getvalue())
    assert len(payload) == 22
    assert payload["observability_protocol_id"] == AGENT_RUNTIME_OBSERVABILITY_PROTOCOL_ID
    assert payload["event_type"] == "workflow_completed"
    assert "private-json-handle" not in stream.getvalue()


def test_activation_observability_contract_keeps_all_closed_boundaries() -> None:
    contract = agent_runtime_activation_observability_contract()

    assert contract["required_explicit_environment_flags"] == [
        "TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT",
        "TS_RAG_ENABLE_LOCAL_AGENT_HTTP_TRANSPORT",
    ]
    assert contract["stage154_formal_evidence_required"] is True
    assert contract["stage154_current_source_fingerprints_required"] is True
    assert contract["binding_host"] == "127.0.0.1"
    assert contract["observability_disable_flag"] is None
    assert contract["public_event_field_count"] == 22
    assert contract["runtime_registered_as_default"] is False
    assert contract["remote_exposure_authorized"] is False
    assert contract["test_access_allowed"] is False
    assert contract["sampling_enabled"] is False
    assert contract["batching_enabled"] is False
    assert contract["remote_export_enabled"] is False
    assert contract["queue_actions_allowed"] is False
    assert contract["retry_actions_allowed"] is False
    assert contract["fallback_strategies_allowed"] is False
