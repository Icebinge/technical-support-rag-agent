from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from threading import Barrier, Lock
from typing import cast

import pytest

from ts_rag_agent.application.primeqa_hybrid_agent_tool_orchestration_protocol import (
    AgentToolWorkflowState,
    InvalidAgentToolWorkflowTransitionError,
    agent_tool_workflow_state_contract,
)
from ts_rag_agent.application.primeqa_hybrid_agent_tool_workflow import (
    AgentToolWorkflowNodeExecutor,
    AgentToolWorkflowPrivateState,
    PrimeQAHybridAgentToolset,
    create_primeqa_hybrid_langgraph_agent_tool_workflow,
    create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset,
    create_primeqa_hybrid_reference_agent_tool_workflow,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_entrypoint import (
    create_primeqa_hybrid_optional_sidecar_agent_entrypoint,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    PrimeQAHybridSidecarObservationAdapter,
)
from ts_rag_agent.domain.answer import (
    AnswerCitation,
    AnswerVerificationResult,
    GeneratedAnswer,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQARuntimeQuery
from ts_rag_agent.domain.retrieval import RetrievalResult


class StaticCandidatePoolRetriever:
    def __init__(
        self,
        *,
        prefix: str,
        error: Exception | None = None,
    ) -> None:
        self._prefix = prefix
        self._error = error
        self._lock = Lock()
        self.call_count = 0

    def retrieve(self, question: PrimeQARuntimeQuery) -> tuple[RetrievalResult, ...]:
        with self._lock:
            self.call_count += 1
        if self._error is not None:
            raise self._error
        return _candidate_results(self._prefix)


class FourRequestCandidatePoolRetriever:
    def __init__(self) -> None:
        self._barrier = Barrier(4)

    def retrieve(self, question: PrimeQARuntimeQuery) -> tuple[RetrievalResult, ...]:
        self._barrier.wait()
        return _candidate_results(question.id)


class SyntheticAnswerGenerator:
    def generate(
        self,
        question: PrimeQARuntimeQuery,
        retrieval_results: tuple[RetrievalResult, ...],
    ) -> GeneratedAnswer:
        first = retrieval_results[0]
        return GeneratedAnswer(
            question_id=question.id,
            answer=f"Apply the verified procedure for {question.id}.",
            citations=[
                AnswerCitation(
                    document_id=first.document.id,
                    title=first.document.title,
                    retrieval_rank=first.rank,
                    evidence_score=first.score,
                )
            ],
            refused=False,
        )


class SyntheticAnswerGeneratorFactory:
    def create(self) -> SyntheticAnswerGenerator:
        return SyntheticAnswerGenerator()


class SyntheticAnswerVerifier:
    def __init__(self, *, refuse: bool) -> None:
        self._refuse = refuse

    def verify(
        self,
        answer: GeneratedAnswer,
        retrieval_results: tuple[RetrievalResult, ...],
    ) -> AnswerVerificationResult:
        assert len(retrieval_results) == 200
        verified = (
            GeneratedAnswer(
                question_id=answer.question_id,
                answer="I cannot verify an answer from the retrieved evidence.",
                citations=[],
                refused=True,
            )
            if self._refuse
            else answer
        )
        return AnswerVerificationResult(
            original_answer=answer,
            verified_answer=verified,
            citation_context_valid=not self._refuse,
            reasons=["synthetic_refusal"] if self._refuse else [],
        )


class SyntheticAnswerVerifierFactory:
    def __init__(self, *, refuse: bool) -> None:
        self._refuse = refuse

    def create(self) -> SyntheticAnswerVerifier:
        return SyntheticAnswerVerifier(refuse=self._refuse)


@pytest.mark.parametrize("refuse", [False, True])
def test_reference_and_langgraph_engines_are_output_equivalent(refuse: bool) -> None:
    reference_retriever = StaticCandidatePoolRetriever(prefix="reference")
    langgraph_retriever = StaticCandidatePoolRetriever(prefix="reference")
    reference = create_primeqa_hybrid_reference_agent_tool_workflow(
        toolset=_toolset(reference_retriever, refuse=refuse)
    )
    langgraph = create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
        toolset=_toolset(langgraph_retriever, refuse=refuse)
    )
    question = _question("equivalence")

    reference_run = reference.run(question)
    langgraph_run = langgraph.run(question)

    assert langgraph_run.verified_answer == reference_run.verified_answer
    assert langgraph_run.original_answer == reference_run.original_answer
    assert langgraph_run.verification_result == reference_run.verification_result
    assert langgraph_run.candidate_pool_results == reference_run.candidate_pool_results
    assert langgraph_run.generation_context_results == reference_run.generation_context_results
    assert langgraph_run.verification_context_results == reference_run.verification_context_results
    assert langgraph_run.public_safe_trace == reference_run.public_safe_trace
    assert langgraph_run.public_safe_trace.terminal_state == ("refuse" if refuse else "complete")
    assert langgraph_run.public_safe_trace.transition_count == 7
    assert langgraph_run.public_safe_trace.tool_call_count == 3
    assert langgraph_run.public_safe_trace.candidate_pool_depth == 400
    assert langgraph_run.public_safe_trace.generation_context_count == 10
    assert langgraph_run.public_safe_trace.verification_context_count == 200
    assert langgraph_run.public_safe_trace.sidecar_observation_count == 3
    assert reference_retriever.call_count == 1
    assert langgraph_retriever.call_count == 1


def test_langgraph_topology_matches_frozen_nodes_and_compiles_once() -> None:
    workflow = create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
        toolset=_toolset(StaticCandidatePoolRetriever(prefix="topology"), refuse=False)
    )

    first = workflow.run(_question("topology-one"))
    second = workflow.run(_question("topology-two"))
    topology = workflow.topology()

    assert set(topology["node_ids"]) == set(agent_tool_workflow_state_contract()["nodes"])
    assert topology["node_count"] == 7
    assert topology["conditional_edge_count"] == 1
    assert topology["checkpointer_attached"] is False
    assert topology["cache_attached"] is False
    assert first.public_safe_trace.terminal_state == "complete"
    assert second.public_safe_trace.terminal_state == "complete"
    counters = workflow.counters()
    assert counters.graph_compile_count == 1
    assert counters.invocation_count == 2
    assert counters.completed_count == 2
    assert counters.failed_count == 0


def test_invalid_node_transition_rejects_before_tool_call_or_state_mutation() -> None:
    retriever = StaticCandidatePoolRetriever(prefix="invalid")
    executor = AgentToolWorkflowNodeExecutor(toolset=_toolset(retriever, refuse=False))
    state = _private_received_state(_question("invalid"))
    before = _copy_private_state(state)

    with pytest.raises(InvalidAgentToolWorkflowTransitionError):
        executor.retrieve_candidate_pool(state)

    assert retriever.call_count == 0
    assert state == before
    assert executor.last_snapshot is None


def test_langgraph_propagates_same_tool_error_and_records_public_failure_stage() -> None:
    error = RuntimeError("synthetic retrieval failure")
    retriever = StaticCandidatePoolRetriever(prefix="failure", error=error)
    workflow = create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
        toolset=_toolset(retriever, refuse=False)
    )

    with pytest.raises(RuntimeError) as captured:
        workflow.run(_question("failure-private-handle"))

    assert captured.value is error
    assert retriever.call_count == 1
    trace = workflow.last_public_trace
    assert trace is not None
    assert trace.failure_stage == "retrieve_candidate_pool"
    assert trace.terminal_state == "validated"
    assert trace.transition_count == 1
    assert trace.retrieval_tool_call_count == 1
    assert trace.answer_tool_call_count == 0
    assert trace.retry_action_count == 0
    assert trace.fallback_action_count == 0
    assert "synthetic retrieval failure" not in str(trace.to_public_dict())
    assert "failure-private-handle" not in str(trace.to_public_dict())
    assert workflow.counters().failed_count == 1


def test_four_langgraph_invocations_keep_request_state_isolated() -> None:
    workflow = create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
        toolset=_toolset(FourRequestCandidatePoolRetriever(), refuse=False)
    )
    handles = ("alpha", "beta", "gamma", "delta")

    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(lambda handle: workflow.run(_question(handle)), handles))

    for handle, run in zip(handles, runs, strict=True):
        assert run.verified_answer.question_id == handle
        assert run.verified_answer.answer.endswith(f"{handle}.")
        assert all(
            result.document.id.startswith(f"{handle}-") for result in run.candidate_pool_results
        )
        assert run.public_safe_trace.terminal_state == "complete"
        assert run.public_safe_trace.failure_stage is None
    counters = workflow.counters()
    assert counters.invocation_count == 4
    assert counters.completed_count == 4
    assert counters.current_in_flight == 0
    assert counters.max_observed_in_flight == 4
    assert counters.graph_compile_count == 1


def test_real_answer_path_matches_stage139_entrypoint_for_label_free_query() -> None:
    old_entrypoint = create_primeqa_hybrid_optional_sidecar_agent_entrypoint(
        candidate_pool_retriever=StaticCandidatePoolRetriever(prefix="parity")
    )
    workflow = create_primeqa_hybrid_langgraph_agent_tool_workflow(
        candidate_pool_retriever=StaticCandidatePoolRetriever(prefix="parity")
    )
    question = _question("real-parity")

    old_run = old_entrypoint.run(question)
    workflow_run = workflow.run(question)

    assert workflow_run.original_answer == old_run.original_answer
    assert workflow_run.verification_result == old_run.verification_result
    assert workflow_run.verified_answer == old_run.verified_answer
    assert workflow_run.generation_context_results == old_run.generation_context_results
    assert workflow_run.verification_context_results == old_run.verification_context_results
    assert workflow_run.agent_run.public_safe_trace == old_run.agent_run.public_safe_trace


def test_private_and_public_state_contracts_remain_exact() -> None:
    private_fields = set(AgentToolWorkflowPrivateState.__annotations__)
    frozen = agent_tool_workflow_state_contract()

    assert private_fields == set(frozen["private_state_fields"])
    workflow = create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
        toolset=_toolset(StaticCandidatePoolRetriever(prefix="public"), refuse=False)
    )
    run = workflow.run(_question("private-content-handle"))
    public = run.public_safe_trace.to_public_dict()

    assert set(public) == set(frozen["public_trace_fields"])
    assert "private-content-handle" not in str(public)
    assert "Apply the verified procedure" not in str(public)
    assert public["queue_action_count"] == 0
    assert public["retry_action_count"] == 0
    assert public["fallback_action_count"] == 0


def test_installed_langgraph_version_is_exact() -> None:
    assert version("langgraph") == "1.2.9"


def _toolset(
    retriever: StaticCandidatePoolRetriever | FourRequestCandidatePoolRetriever,
    *,
    refuse: bool,
) -> PrimeQAHybridAgentToolset:
    return PrimeQAHybridAgentToolset(
        candidate_pool_retriever=retriever,
        observation_adapter=PrimeQAHybridSidecarObservationAdapter(),
        answer_generator_factory=SyntheticAnswerGeneratorFactory(),
        answer_verifier_factory=SyntheticAnswerVerifierFactory(refuse=refuse),
    )


def _question(handle: str) -> PrimeQARuntimeQuery:
    return PrimeQARuntimeQuery(
        id=handle,
        title="Adapter installation",
        text="How do I apply the verified adapter procedure?",
    )


def _candidate_results(prefix: str) -> tuple[RetrievalResult, ...]:
    return tuple(
        RetrievalResult(
            document=PrimeQADocument(
                id=f"{prefix}-{rank:03d}",
                title=f"Adapter procedure {rank}",
                text="Apply the verified adapter procedure and restart the service.",
            ),
            score=1.0 / rank,
            rank=rank,
        )
        for rank in range(1, 401)
    )


def _private_received_state(question: PrimeQARuntimeQuery) -> AgentToolWorkflowPrivateState:
    return {
        "request_handle": question.id,
        "runtime_query": question,
        "candidate_pool_results": (),
        "generation_context_results": (),
        "verification_context_results": (),
        "sidecar_observation_bundle": None,
        "original_answer": None,
        "verification_result": None,
        "terminal_response": None,
        "current_state": AgentToolWorkflowState.RECEIVED,
        "visited_states": ("received",),
        "tool_call_counts": {
            "retrieve_candidate_pool": 0,
            "compose_grounded_answer": 0,
            "verify_grounded_answer": 0,
        },
        "failure_stage": None,
    }


def _copy_private_state(
    state: AgentToolWorkflowPrivateState,
) -> AgentToolWorkflowPrivateState:
    copied = dict(state)
    copied["tool_call_counts"] = dict(state["tool_call_counts"])
    return cast(AgentToolWorkflowPrivateState, copied)
