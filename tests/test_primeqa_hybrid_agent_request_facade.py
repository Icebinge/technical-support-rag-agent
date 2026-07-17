from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from threading import Event, Thread

import pytest

from ts_rag_agent.application.primeqa_hybrid_agent_request_facade import (
    AgentFacadeRequest,
    AgentRequestCancellationSignal,
    AgentRequestFacadeCancelledError,
    AgentRequestFacadeCapacityExceededError,
    AgentRequestFacadeClosedError,
    AgentRequestFacadeDrainingError,
    AgentRequestFacadeInvalidRequestError,
    AgentRequestFacadeNotActiveError,
    AgentRequestFacadeState,
    PrimeQAHybridAgentRequestFacade,
    agent_request_facade_contract,
    create_primeqa_hybrid_agent_request_facade,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_activation import (
    PrimeQAHybridConcurrentRuntimeBootstrapResult,
    PublicSafeConcurrentRuntimeStartupTrace,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_sidecar_agent_runtime import (
    ConcurrentArrivalPattern,
    PrimeQAHybridConcurrentCapacityExceededError,
    PublicSafeConcurrentRuntimeRequestTrace,
    create_primeqa_hybrid_concurrent_sidecar_agent_runtime,
)
from ts_rag_agent.application.primeqa_hybrid_online_candidate_pool_retriever import (
    CandidatePoolRetrievalConfig,
    IndependentCandidatePoolSearchChannel,
    PrimeQAHybridOnlineCandidatePoolRetriever,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    PrimeQAHybridRuntimeResourceSummary,
    PrimeQAHybridSharedRuntimeResources,
)
from ts_rag_agent.domain.answer import AnswerCitation, GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQARuntimeQuery
from ts_rag_agent.domain.retrieval import RetrievalResult


@dataclass(frozen=True)
class SyntheticRuntimeRun:
    verified_answer: GeneratedAnswer
    public_safe_trace: PublicSafeConcurrentRuntimeRequestTrace


class SyntheticRuntime:
    def __init__(self, *, refused: bool = False) -> None:
        self.refused = refused
        self.received_queries: list[PrimeQARuntimeQuery] = []
        self.arrival_patterns: list[ConcurrentArrivalPattern] = []

    def run(
        self,
        question: PrimeQARuntimeQuery,
        *,
        arrival_pattern: ConcurrentArrivalPattern,
    ) -> SyntheticRuntimeRun:
        self.received_queries.append(question)
        self.arrival_patterns.append(arrival_pattern)
        return _runtime_run(question.id, refused=self.refused)


class CapacityRejectingRuntime(SyntheticRuntime):
    def run(
        self,
        question: PrimeQARuntimeQuery,
        *,
        arrival_pattern: ConcurrentArrivalPattern,
    ) -> SyntheticRuntimeRun:
        self.received_queries.append(question)
        self.arrival_patterns.append(arrival_pattern)
        raise PrimeQAHybridConcurrentCapacityExceededError(
            _runtime_trace(
                admission_state="rejected_capacity",
                terminal_state="capacity_rejected",
                candidate_pool_depth=0,
            )
        )


class FailingRuntime(SyntheticRuntime):
    def __init__(self, error: RuntimeError) -> None:
        super().__init__()
        self.error = error

    def run(
        self,
        question: PrimeQARuntimeQuery,
        *,
        arrival_pattern: ConcurrentArrivalPattern,
    ) -> SyntheticRuntimeRun:
        self.received_queries.append(question)
        self.arrival_patterns.append(arrival_pattern)
        raise self.error


class BlockingRuntime(SyntheticRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.entered = Event()
        self.release = Event()

    def run(
        self,
        question: PrimeQARuntimeQuery,
        *,
        arrival_pattern: ConcurrentArrivalPattern,
    ) -> SyntheticRuntimeRun:
        self.received_queries.append(question)
        self.arrival_patterns.append(arrival_pattern)
        self.entered.set()
        self.release.wait()
        return _runtime_run(question.id)


def test_runtime_query_has_no_gold_or_test_fields() -> None:
    query = PrimeQARuntimeQuery(id="private-handle", title="Install", text="How?")

    assert set(type(query).model_fields) == {"id", "title", "text"}
    assert query.full_question == "Install\n\nHow?"
    for forbidden in (
        "answer",
        "answerable",
        "answer_doc_id",
        "doc_ids",
        "start_offset",
        "end_offset",
        "test_membership",
    ):
        assert not hasattr(query, forbidden)


def test_facade_maps_complete_private_response_and_public_traces() -> None:
    runtime = SyntheticRuntime()
    facade = _active_facade(runtime)

    run = facade.invoke(
        AgentFacadeRequest(
            request_handle="private-handle",
            title="Install adapter",
            text="How do I configure it?",
        )
    )

    assert len(runtime.received_queries) == 1
    query = runtime.received_queries[0]
    assert isinstance(query, PrimeQARuntimeQuery)
    assert query.model_dump() == {
        "id": "private-handle",
        "title": "Install adapter",
        "text": "How do I configure it?",
    }
    assert runtime.arrival_patterns == [ConcurrentArrivalPattern.APPLICATION]
    assert run.response.request_handle == "private-handle"
    assert run.response.text == "Configure the adapter. [doc-1]"
    assert run.response.refused is False
    assert run.response.citations[0].document_reference == "doc-1"
    assert run.response.citations[0].rank == 1
    assert run.public_safe_event.outcome_code == "complete"
    assert run.public_safe_event.downstream_dispatched is True
    assert set(run.to_public_dict()) == {"facade_event", "runtime_request"}
    assert "private-handle" not in str(run.to_public_dict())
    assert facade.last_public_event == run.public_safe_event
    counters = facade.counters()
    assert counters.invocation_attempt_count == 1
    assert counters.accepted_call_count == 1
    assert counters.runtime_dispatch_count == 1
    assert counters.completed_response_count == 1
    assert counters.refused_response_count == 0
    assert counters.current_in_flight == 0


def test_facade_maps_verified_refusal_without_citations() -> None:
    facade = _active_facade(SyntheticRuntime(refused=True))

    run = facade.invoke(AgentFacadeRequest(request_handle="private", text="Unknown issue"))

    assert run.response.refused is True
    assert run.response.citations == ()
    assert run.public_safe_event.outcome_code == "refuse"
    assert facade.counters().refused_response_count == 1


def test_label_free_query_runs_through_real_concurrent_online_pipeline() -> None:
    candidate_results = tuple(
        RetrievalResult(
            document=PrimeQADocument(
                id=f"doc-{index:03d}",
                title=f"Adapter repair procedure {index}",
                text=("Apply the documented adapter repair procedure and restart the service."),
            ),
            score=float(1 / (index + 1)),
            rank=index + 1,
        )
        for index in range(400)
    )

    def searcher(query: str, top_k: int) -> Sequence[RetrievalResult]:
        assert query == "Adapter installation\n\nHow do I repair the adapter?"
        return candidate_results[:top_k]

    retriever = PrimeQAHybridOnlineCandidatePoolRetriever(
        channels=(
            IndependentCandidatePoolSearchChannel(
                channel_id="synthetic",
                family="synthetic",
                weight=1.0,
                searcher=searcher,
            ),
        ),
        config=CandidatePoolRetrievalConfig(
            channel_top_k=400,
            prefix_depth=200,
            target_pool_depth=400,
            rrf_k=60,
        ),
    )
    runtime = create_primeqa_hybrid_concurrent_sidecar_agent_runtime(
        shared_resources=PrimeQAHybridSharedRuntimeResources(
            candidate_pool_retriever=retriever,
            summary=PrimeQAHybridRuntimeResourceSummary(
                dense_model_count=2,
                dense_embedding_cache_count=2,
                lexical_index_count=4,
                derived_route_count=1,
                candidate_pool_retriever_instance_count=1,
                optional_entrypoint_instance_count=1,
            ),
        )
    )
    facade = _active_facade(runtime)

    run = facade.invoke(
        AgentFacadeRequest(
            request_handle="real-online-chain",
            title="Adapter installation",
            text="How do I repair the adapter?",
        )
    )

    assert run.response.refused is True
    assert run.response.citations == ()
    assert run.public_safe_runtime_trace.candidate_pool_depth == 400
    assert run.public_safe_runtime_trace.arrival_pattern == "application_request"
    assert run.public_safe_runtime_trace.terminal_state == "refuse"
    assert run.public_safe_event.outcome_code == "refuse"


def test_invalid_request_and_predispatch_cancellation_never_call_runtime() -> None:
    runtime = SyntheticRuntime()
    facade = _active_facade(runtime)

    with pytest.raises(AgentRequestFacadeInvalidRequestError) as invalid:
        facade.invoke(AgentFacadeRequest(request_handle=" ", text=""))
    assert invalid.value.reasons == ("request_handle_required", "request_text_required")
    assert invalid.value.public_safe_event.downstream_dispatched is False

    signal = AgentRequestCancellationSignal()
    signal.cancel()
    with pytest.raises(AgentRequestFacadeCancelledError) as cancelled:
        facade.invoke(
            AgentFacadeRequest(
                request_handle="cancelled",
                text="Do not dispatch",
                cancellation_signal=signal,
            )
        )
    assert cancelled.value.code == "cancelled_before_dispatch"
    assert cancelled.value.public_safe_event.downstream_dispatched is False
    assert runtime.received_queries == []
    counters = facade.counters()
    assert counters.invalid_rejected_count == 1
    assert counters.cancelled_before_dispatch_count == 1
    assert counters.runtime_dispatch_count == 0


def test_capacity_error_is_mapped_exactly_without_queue_retry_or_fallback() -> None:
    runtime = CapacityRejectingRuntime()
    facade = _active_facade(runtime)

    with pytest.raises(AgentRequestFacadeCapacityExceededError) as caught:
        facade.invoke(AgentFacadeRequest(request_handle="capacity", text="Run request"))

    error = caught.value
    assert error.code == "capacity_exceeded"
    assert error.public_safe_event.outcome_code == "capacity_exceeded"
    assert error.public_safe_event.downstream_dispatched is False
    assert error.public_safe_runtime_trace is not None
    assert error.public_safe_runtime_trace.admission_state == "rejected_capacity"
    assert isinstance(error.__cause__, PrimeQAHybridConcurrentCapacityExceededError)
    counters = facade.counters()
    assert counters.runtime_dispatch_count == 1
    assert counters.capacity_rejected_count == 1
    assert counters.queue_action_count == 0
    assert counters.retry_action_count == 0
    assert counters.fallback_action_count == 0


def test_downstream_error_object_propagates_unchanged_with_request_local_event() -> None:
    original = RuntimeError("synthetic downstream failure")
    facade = _active_facade(FailingRuntime(original))

    with pytest.raises(RuntimeError) as caught:
        facade.invoke(AgentFacadeRequest(request_handle="failure", text="Run request"))

    assert caught.value is original
    assert facade.last_public_event is not None
    assert facade.last_public_event.outcome_code == "downstream_error"
    assert facade.last_public_event.downstream_dispatched is True
    assert facade.counters().downstream_error_count == 1


def test_shutdown_drains_naturally_rejects_new_calls_and_closes() -> None:
    runtime = BlockingRuntime()
    facade = _active_facade(runtime)
    completed_runs = []
    invocation_errors = []

    def invoke_blocking() -> None:
        try:
            completed_runs.append(
                facade.invoke(AgentFacadeRequest(request_handle="in-flight", text="Wait"))
            )
        except Exception as error:  # pragma: no cover - assertion captures unexpected failure
            invocation_errors.append(error)

    invoke_thread = Thread(target=invoke_blocking)
    invoke_thread.start()
    runtime.entered.wait()

    shutdown_thread = Thread(target=facade.shutdown)
    shutdown_thread.start()
    facade.wait_until_state(AgentRequestFacadeState.DRAINING)

    with pytest.raises(AgentRequestFacadeDrainingError) as draining:
        facade.invoke(AgentFacadeRequest(request_handle="new", text="Reject while draining"))
    assert draining.value.public_safe_event.downstream_dispatched is False
    assert facade.counters().current_in_flight == 1

    runtime.release.set()
    invoke_thread.join()
    shutdown_thread.join()

    assert invocation_errors == []
    assert len(completed_runs) == 1
    assert completed_runs[0].public_safe_event.facade_state == "draining"
    assert facade.state is AgentRequestFacadeState.CLOSED
    assert facade.counters().current_in_flight == 0
    with pytest.raises(AgentRequestFacadeClosedError):
        facade.invoke(AgentFacadeRequest(request_handle="closed", text="Reject"))

    facade.shutdown()
    assert facade.state is AgentRequestFacadeState.CLOSED


def test_factory_rejects_inactive_bootstrap_and_contract_keeps_boundaries_closed() -> None:
    with pytest.raises(AgentRequestFacadeNotActiveError) as caught:
        create_primeqa_hybrid_agent_request_facade(
            bootstrap_result=_bootstrap_result(runtime=None, active=False)
        )

    assert caught.value.code == "facade_not_active"
    assert caught.value.public_safe_event.facade_state == "not_active"
    contract = agent_request_facade_contract()
    assert contract["runtime_query_fields"] == ["id", "title", "text"]
    assert contract["runtime_query_contains_gold_labels"] is False
    assert contract["shutdown_waits_without_implicit_timeout"] is True
    assert contract["facade_owns_runtime_resources"] is False
    assert contract["registered_as_runtime_default"] is False
    assert contract["network_service_implemented"] is False
    assert contract["test_access_allowed"] is False
    assert contract["queue_actions_allowed"] is False
    assert contract["retry_actions_allowed"] is False
    assert contract["fallback_strategies_allowed"] is False


def test_direct_facade_construction_without_active_binding_is_rejected() -> None:
    with pytest.raises(ValueError, match="active bootstrap binding"):
        PrimeQAHybridAgentRequestFacade(
            runtime=SyntheticRuntime(),
            _active_runtime_binding=object(),
        )


def _active_facade(runtime) -> PrimeQAHybridAgentRequestFacade:
    return create_primeqa_hybrid_agent_request_facade(
        bootstrap_result=_bootstrap_result(runtime=runtime, active=True)
    )


def _bootstrap_result(*, runtime, active: bool) -> PrimeQAHybridConcurrentRuntimeBootstrapResult:
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
            rejection_reasons=() if active else ("explicit_concurrent_activation_not_requested",),
        ),
        resource_summary=None,
        source_evaluation=None,
    )


def _runtime_run(request_handle: str, *, refused: bool = False) -> SyntheticRuntimeRun:
    answer = GeneratedAnswer(
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
    )
    return SyntheticRuntimeRun(
        verified_answer=answer,
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
