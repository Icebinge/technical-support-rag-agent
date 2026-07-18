from __future__ import annotations

import json

import pytest

from ts_rag_agent.application.primeqa_hybrid_agent_tool_workflow import (
    PrimeQAHybridAgentToolset,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_agent_state_protocol import (
    ThreadStateLimits,
    ThreadStatePolicyViolationError,
    VolatileThreadStateLedger,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_runtime import (
    FIXED_INSUFFICIENT_EVIDENCE_RESPONSE,
    PRODUCTION_MAX_COMPLETED_TURNS,
    PRODUCTION_MAX_RETAINED_BYTES,
    bounded_dynamic_agent_runtime_contract,
    create_primeqa_hybrid_bounded_dynamic_agent_runtime_from_toolset,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    PrimeQAHybridSidecarObservationAdapter,
)
from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    BoundedAnswerDecision,
    StructuredDecisionSchemaError,
)
from ts_rag_agent.domain.answer import (
    AnswerCitation,
    AnswerVerificationResult,
    GeneratedAnswer,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQARuntimeQuery
from ts_rag_agent.domain.retrieval import RetrievalResult


class StaticCandidatePoolRetriever:
    def __init__(self) -> None:
        self.call_count = 0

    def retrieve(self, question: PrimeQARuntimeQuery) -> tuple[RetrievalResult, ...]:
        self.call_count += 1
        return _candidate_results(question.id)


class StaticDecisionRouter:
    def __init__(self, action: str, *, error: Exception | None = None) -> None:
        self.action = action
        self.error = error
        self.call_count = 0
        self.history_sizes: list[int] = []

    @property
    def last_metrics(self):
        return None

    def decide(
        self,
        *,
        question: PrimeQARuntimeQuery,
        generation_context_results: tuple[RetrievalResult, ...],
        completed_turns: tuple,
    ) -> BoundedAnswerDecision:
        self.call_count += 1
        self.history_sizes.append(len(completed_turns))
        assert len(generation_context_results) == 10
        if self.error is not None:
            raise self.error
        return BoundedAnswerDecision(action=self.action)


class CountingAnswerGenerator:
    def __init__(self, counters: dict[str, int]) -> None:
        self._counters = counters

    def generate(
        self,
        question: PrimeQARuntimeQuery,
        retrieval_results: tuple[RetrievalResult, ...],
    ) -> GeneratedAnswer:
        self._counters["compose"] += 1
        first = retrieval_results[0]
        return GeneratedAnswer(
            question_id=question.id,
            answer="Apply the documented adapter procedure.",
            citations=[
                AnswerCitation(
                    document_id=first.document.id,
                    title=first.document.title,
                    retrieval_rank=first.rank,
                    evidence_score=10.0,
                )
            ],
            refused=False,
        )


class CountingAnswerGeneratorFactory:
    def __init__(self, counters: dict[str, int]) -> None:
        self._counters = counters

    def create(self) -> CountingAnswerGenerator:
        return CountingAnswerGenerator(self._counters)


class CountingAnswerVerifier:
    def __init__(self, counters: dict[str, int], *, refuse: bool) -> None:
        self._counters = counters
        self._refuse = refuse

    def verify(
        self,
        answer: GeneratedAnswer,
        retrieval_results: tuple[RetrievalResult, ...],
    ) -> AnswerVerificationResult:
        self._counters["verify"] += 1
        assert len(retrieval_results) == 200
        verified = (
            GeneratedAnswer(
                question_id=answer.question_id,
                answer=FIXED_INSUFFICIENT_EVIDENCE_RESPONSE,
                citations=[],
                refused=True,
            )
            if self._refuse
            else answer
        )
        return AnswerVerificationResult(
            original_answer=answer,
            verified_answer=verified,
            citation_context_valid=True,
            reasons=["synthetic_refusal"] if self._refuse else ["verified"],
        )


class CountingAnswerVerifierFactory:
    def __init__(self, counters: dict[str, int], *, refuse: bool) -> None:
        self._counters = counters
        self._refuse = refuse

    def create(self) -> CountingAnswerVerifier:
        return CountingAnswerVerifier(self._counters, refuse=self._refuse)


def test_compose_branch_runs_each_required_operation_once_and_commits_turn() -> None:
    runtime, retriever, router, counters = _runtime(action="compose_grounded_answer")
    runtime.open_thread("thread-compose")

    run = runtime.run_turn(
        opaque_thread_handle="thread-compose",
        question=_question("compose"),
    )

    trace = run.public_safe_trace
    assert run.verified_answer.refused is False
    assert retriever.call_count == router.call_count == 1
    assert counters == {"compose": 1, "verify": 1}
    assert trace.terminal_state == "complete"
    assert trace.completed_turn_count == 1
    assert trace.model_decision_count == 1
    assert trace.selected_action == "compose_grounded_answer"
    assert trace.retrieval_call_count == 1
    assert trace.composition_call_count == 1
    assert trace.verification_call_count == 1
    assert trace.diagnostic_observation_count == 1
    assert trace.failure_stage is None
    assert trace.retry_action_count == trace.fallback_action_count == 0


def test_early_refuse_skips_composition_verification_and_diagnostics() -> None:
    runtime, retriever, router, counters = _runtime(action="refuse_insufficient_evidence")
    runtime.open_thread("thread-refuse")

    run = runtime.run_turn(
        opaque_thread_handle="thread-refuse",
        question=_question("refuse"),
    )

    trace = run.public_safe_trace
    assert run.verified_answer.answer == FIXED_INSUFFICIENT_EVIDENCE_RESPONSE
    assert run.verified_answer.refused is True
    assert run.verified_answer.citations == []
    assert retriever.call_count == router.call_count == 1
    assert counters == {"compose": 0, "verify": 0}
    assert trace.terminal_state == "refuse"
    assert trace.selected_action == "refuse_insufficient_evidence"
    assert trace.retrieval_call_count == 1
    assert trace.composition_call_count == 0
    assert trace.verification_call_count == 0
    assert trace.diagnostic_observation_count == 0


def test_verifier_refusal_still_runs_compose_branch_diagnostics_once() -> None:
    runtime, _, _, counters = _runtime(action="compose_grounded_answer", verifier_refuses=True)
    runtime.open_thread("thread-verified-refuse")

    run = runtime.run_turn(
        opaque_thread_handle="thread-verified-refuse",
        question=_question("verified-refuse"),
    )

    assert run.verified_answer.refused is True
    assert counters == {"compose": 1, "verify": 1}
    assert run.public_safe_trace.terminal_state == "refuse"
    assert run.public_safe_trace.diagnostic_observation_count == 1


def test_second_turn_receives_only_completed_private_history() -> None:
    runtime, _, router, _ = _runtime(action="compose_grounded_answer")
    runtime.open_thread("thread-history")

    runtime.run_turn(opaque_thread_handle="thread-history", question=_question("one"))
    runtime.run_turn(opaque_thread_handle="thread-history", question=_question("two"))

    assert router.history_sizes == [0, 1]
    assert runtime.thread_summary("thread-history").completed_turn_count == 2


def test_threads_are_isolated_and_close_clears_private_state() -> None:
    runtime, _, router, _ = _runtime(action="compose_grounded_answer")
    runtime.open_thread("thread-a")
    runtime.open_thread("thread-b")
    runtime.run_turn(opaque_thread_handle="thread-a", question=_question("a"))
    runtime.run_turn(opaque_thread_handle="thread-b", question=_question("b"))

    assert router.history_sizes == [0, 0]
    closed = runtime.close_thread("thread-a")
    assert closed.opened is False
    with pytest.raises(ThreadStatePolicyViolationError, match="not open"):
        runtime.thread_summary("thread-a")
    assert runtime.thread_summary("thread-b").completed_turn_count == 1


def test_missing_thread_rejects_before_retrieval_or_model() -> None:
    runtime, retriever, router, counters = _runtime(action="compose_grounded_answer")

    with pytest.raises(ThreadStatePolicyViolationError, match="not open"):
        runtime.run_turn(opaque_thread_handle="missing", question=_question("missing"))

    assert retriever.call_count == router.call_count == 0
    assert counters == {"compose": 0, "verify": 0}
    assert runtime.last_public_trace is not None
    assert runtime.last_public_trace.thread_state_opened is False
    assert runtime.last_public_trace.failure_stage == "load_thread_state"


def test_schema_error_propagates_after_one_decision_without_answer_or_retry() -> None:
    error = StructuredDecisionSchemaError("synthetic schema error")
    runtime, retriever, router, counters = _runtime(
        action="compose_grounded_answer",
        router_error=error,
    )
    runtime.open_thread("thread-schema")

    with pytest.raises(StructuredDecisionSchemaError) as captured:
        runtime.run_turn(opaque_thread_handle="thread-schema", question=_question("schema"))

    assert captured.value is error
    assert retriever.call_count == router.call_count == 1
    assert counters == {"compose": 0, "verify": 0}
    assert runtime.thread_summary("thread-schema").completed_turn_count == 0
    trace = runtime.last_public_trace
    assert trace is not None
    assert trace.failure_stage == "select_action"
    assert trace.model_decision_count == 1
    assert trace.retrieval_call_count == 1
    assert trace.retry_action_count == trace.fallback_action_count == 0


def test_turn_limit_rejects_fifth_commit_without_mutating_four_turn_history() -> None:
    runtime, retriever, router, counters = _runtime(action="refuse_insufficient_evidence")
    runtime.open_thread("thread-limit")
    for index in range(PRODUCTION_MAX_COMPLETED_TURNS):
        runtime.run_turn(
            opaque_thread_handle="thread-limit",
            question=_question(f"accepted-{index}"),
        )
    before = runtime.thread_summary("thread-limit")

    with pytest.raises(ThreadStatePolicyViolationError, match="turn limit"):
        runtime.run_turn(
            opaque_thread_handle="thread-limit",
            question=_question("rejected-five"),
        )

    assert runtime.thread_summary("thread-limit") == before
    assert retriever.call_count == router.call_count == 5
    assert counters == {"compose": 0, "verify": 0}
    assert runtime.last_public_trace is not None
    assert runtime.last_public_trace.state_limit_rejected is True
    assert runtime.last_public_trace.failure_stage == "commit_thread_state"


def test_public_trace_contains_no_private_content_or_identifiers() -> None:
    runtime, _, _, _ = _runtime(action="compose_grounded_answer")
    sentinel = "private-stage157-sentinel"
    runtime.open_thread("private-thread-handle")
    run = runtime.run_turn(
        opaque_thread_handle="private-thread-handle",
        question=PrimeQARuntimeQuery(id="private-id", text=sentinel),
    )

    serialized = json.dumps(run.public_safe_trace.to_public_dict())
    assert sentinel not in serialized
    assert "private-thread-handle" not in serialized
    assert "private-id" not in serialized


def test_graph_and_runtime_contract_match_confirmed_bounded_configuration() -> None:
    runtime, _, _, _ = _runtime(action="compose_grounded_answer")
    topology = runtime.topology()
    contract = bounded_dynamic_agent_runtime_contract()

    assert topology["node_count"] == 9
    assert topology["conditional_edge_count"] == 1
    assert topology["checkpointer_attached"] is False
    assert topology["cache_attached"] is False
    assert topology["compile_count"] == 1
    assert contract["thread_limits"] == {
        "max_completed_turns": PRODUCTION_MAX_COMPLETED_TURNS,
        "max_retained_bytes": PRODUCTION_MAX_RETAINED_BYTES,
    }
    assert contract["refuse_branch"] == ["finalize_insufficient_evidence_refusal"]
    assert contract["runtime_registered_as_default"] is False
    assert contract["http_service_integrated"] is False
    assert contract["test_gate_opened"] is False
    assert contract["retry_actions_enabled"] is False
    assert contract["fallback_actions_enabled"] is False


def _runtime(
    *,
    action: str,
    verifier_refuses: bool = False,
    router_error: Exception | None = None,
):
    retriever = StaticCandidatePoolRetriever()
    router = StaticDecisionRouter(action, error=router_error)
    counters = {"compose": 0, "verify": 0}
    toolset = PrimeQAHybridAgentToolset(
        candidate_pool_retriever=retriever,
        observation_adapter=PrimeQAHybridSidecarObservationAdapter(),
        answer_generator_factory=CountingAnswerGeneratorFactory(counters),
        answer_verifier_factory=CountingAnswerVerifierFactory(
            counters,
            refuse=verifier_refuses,
        ),
    )
    runtime = create_primeqa_hybrid_bounded_dynamic_agent_runtime_from_toolset(
        toolset=toolset,
        decision_router=router,
        thread_ledger=VolatileThreadStateLedger(
            limits=ThreadStateLimits(
                max_completed_turns=PRODUCTION_MAX_COMPLETED_TURNS,
                max_retained_bytes=PRODUCTION_MAX_RETAINED_BYTES,
            )
        ),
    )
    return runtime, retriever, router, counters


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
            score=10.0 / rank,
            rank=rank,
        )
        for rank in range(1, 401)
    )
