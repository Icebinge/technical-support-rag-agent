from __future__ import annotations

import json

from ts_rag_agent.application.primeqa_hybrid_agent_tool_workflow import (
    PrimeQAHybridAgentToolset,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_agent_state_protocol import (
    CompletedThreadTurn,
    ThreadStateLimits,
    VolatileThreadStateLedger,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_iterative_agent_runtime import (
    ALTERNATE_EVIDENCE_DEPTH,
    ITERATIVE_ALLOWED_TERMINAL_STATES,
    SYSTEM_CLARIFICATION_RESPONSES,
    OriginalRrfAlternateEvidenceInspector,
    bounded_iterative_agent_runtime_contract,
    create_bounded_iterative_agent_runtime_from_toolset,
)
from ts_rag_agent.application.primeqa_hybrid_iterative_decision_router import (
    ClarificationKind,
    IterativeAgentDecision,
    IterativeDecisionPhase,
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


class StaticRetriever:
    def __init__(self) -> None:
        self.call_count = 0

    def retrieve(self, question: PrimeQARuntimeQuery) -> tuple[RetrievalResult, ...]:
        self.call_count += 1
        return _candidate_results(question.id)


class SequencedRouter:
    def __init__(self, decisions: list[IterativeAgentDecision]) -> None:
        self._decisions = list(decisions)
        self.phases: list[str] = []
        self.history_sizes: list[int] = []

    @property
    def last_metrics(self):
        return None

    def decide(
        self,
        *,
        phase: IterativeDecisionPhase,
        question: PrimeQARuntimeQuery,
        initial_evidence_results: tuple[RetrievalResult, ...],
        alternate_evidence_results: tuple[RetrievalResult, ...],
        completed_turns: tuple[CompletedThreadTurn, ...],
    ) -> IterativeAgentDecision:
        _ = question
        self.phases.append(phase.value)
        self.history_sizes.append(len(completed_turns))
        assert len(initial_evidence_results) == 10
        if phase is IterativeDecisionPhase.INITIAL:
            assert not alternate_evidence_results
        else:
            assert len(alternate_evidence_results) == ALTERNATE_EVIDENCE_DEPTH
        return self._decisions.pop(0)


class CountingGenerator:
    def __init__(self, counters: dict[str, int]) -> None:
        self._counters = counters

    def generate(
        self,
        question: PrimeQARuntimeQuery,
        retrieval_results: tuple[RetrievalResult, ...],
    ) -> GeneratedAnswer:
        self._counters["compose"] += 1
        self._counters["composition_context_count"] = len(retrieval_results)
        first = retrieval_results[0]
        return GeneratedAnswer(
            question_id=question.id,
            answer="Apply the documented procedure.",
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


class CountingGeneratorFactory:
    def __init__(self, counters: dict[str, int]) -> None:
        self._counters = counters

    def create(self) -> CountingGenerator:
        return CountingGenerator(self._counters)


class CountingVerifier:
    def __init__(self, counters: dict[str, int]) -> None:
        self._counters = counters

    def verify(
        self,
        answer: GeneratedAnswer,
        retrieval_results: tuple[RetrievalResult, ...],
    ) -> AnswerVerificationResult:
        self._counters["verify"] += 1
        assert len(retrieval_results) == 200
        return AnswerVerificationResult(
            original_answer=answer,
            verified_answer=answer,
            citation_context_valid=True,
            reasons=["verified"],
        )


class CountingVerifierFactory:
    def __init__(self, counters: dict[str, int]) -> None:
        self._counters = counters

    def create(self) -> CountingVerifier:
        return CountingVerifier(self._counters)


def test_original_rrf_inspector_uses_existing_contiguous_top10() -> None:
    pool = _candidate_results("inspect")
    initial = pool[10:20]

    alternate = OriginalRrfAlternateEvidenceInspector().inspect(
        candidate_pool_results=pool,
        initial_evidence_results=initial,
    )

    assert [result.rank for result in alternate] == list(range(1, 11))


def test_initial_compose_runs_one_decision_without_inspection() -> None:
    runtime, retriever, router, counters = _runtime([_decision("compose_grounded_answer")])
    runtime.open_thread("compose")

    run = runtime.run_turn(opaque_thread_handle="compose", question=_question("compose"))

    assert run.verified_answer.refused is False
    assert retriever.call_count == 1
    assert router.phases == ["initial"]
    assert counters["compose"] == counters["verify"] == 1
    assert run.public_safe_trace.evidence_inspection_count == 0
    assert run.public_safe_trace.model_decision_count == 1
    assert run.public_safe_trace.terminal_state == "complete"


def test_inspect_then_compose_uses_one_retrieval_two_decisions_and_union_context() -> None:
    runtime, retriever, router, counters = _runtime(
        [_decision("inspect_alternate_evidence"), _decision("compose_grounded_answer")]
    )
    runtime.open_thread("inspect-compose")

    run = runtime.run_turn(
        opaque_thread_handle="inspect-compose",
        question=_question("inspect-compose"),
    )

    trace = run.public_safe_trace
    assert retriever.call_count == 1
    assert router.phases == ["initial", "final_after_inspection"]
    assert trace.retrieval_call_count == 1
    assert trace.evidence_inspection_count == 1
    assert trace.model_decision_count == 2
    assert trace.selected_actions == (
        "inspect_alternate_evidence",
        "compose_grounded_answer",
    )
    assert trace.alternate_evidence_count == 10
    assert trace.composition_context_count == counters["composition_context_count"] == 20
    assert trace.retry_action_count == trace.fallback_action_count == 0


def test_direct_clarification_uses_system_text_and_commits_clarify_terminal() -> None:
    kind = ClarificationKind.VERSION_OR_BUILD.value
    runtime, retriever, router, counters = _runtime(
        [_decision("request_clarification", clarification_kind=kind)]
    )
    runtime.open_thread("clarify")

    run = runtime.run_turn(opaque_thread_handle="clarify", question=_question("clarify"))

    trace = run.public_safe_trace
    assert run.verified_answer.answer == SYSTEM_CLARIFICATION_RESPONSES[kind]
    assert run.verified_answer.refused is True
    assert run.verified_answer.citations == []
    assert retriever.call_count == 1
    assert router.phases == ["initial"]
    assert counters["compose"] == counters["verify"] == 0
    assert trace.terminal_state == "clarify"
    assert trace.clarification_kind == kind
    assert trace.clarification_fallback_count == trace.fallback_action_count == 1
    assert runtime.thread_summary("clarify").completed_turn_count == 1


def test_inspect_then_clarify_is_bounded_to_two_decisions() -> None:
    kind = ClarificationKind.ERROR_CODE_OR_LOG.value
    runtime, retriever, router, counters = _runtime(
        [
            _decision("inspect_alternate_evidence"),
            _decision("request_clarification", clarification_kind=kind),
        ]
    )
    runtime.open_thread("inspect-clarify")

    run = runtime.run_turn(
        opaque_thread_handle="inspect-clarify",
        question=_question("inspect-clarify"),
    )

    assert retriever.call_count == 1
    assert router.phases == ["initial", "final_after_inspection"]
    assert counters["compose"] == counters["verify"] == 0
    assert run.public_safe_trace.evidence_inspection_count == 1
    assert run.public_safe_trace.model_decision_count == 2
    assert run.public_safe_trace.terminal_state == "clarify"


def test_refusal_remains_distinct_from_clarification_fallback() -> None:
    runtime, _, _, counters = _runtime([_decision("refuse_insufficient_evidence")])
    runtime.open_thread("refuse")

    run = runtime.run_turn(opaque_thread_handle="refuse", question=_question("refuse"))

    assert run.verified_answer.refused is True
    assert counters["compose"] == counters["verify"] == 0
    assert run.public_safe_trace.terminal_state == "refuse"
    assert run.public_safe_trace.clarification_fallback_count == 0
    assert run.public_safe_trace.fallback_action_count == 0


def test_clarification_history_is_visible_on_next_turn_and_public_trace_is_safe() -> None:
    kind = ClarificationKind.PRODUCT_OR_COMPONENT.value
    runtime, _, router, _ = _runtime(
        [
            _decision("request_clarification", clarification_kind=kind),
            _decision("compose_grounded_answer"),
        ]
    )
    runtime.open_thread("private-thread")
    runtime.run_turn(opaque_thread_handle="private-thread", question=_question("one"))
    second = runtime.run_turn(opaque_thread_handle="private-thread", question=_question("two"))

    assert router.history_sizes == [0, 1]
    serialized = json.dumps(second.public_safe_trace.to_public_dict())
    assert "private-thread" not in serialized
    assert "How do I configure" not in serialized


def test_runtime_contract_freezes_ac_plus_c_without_second_retrieval_or_loop() -> None:
    contract = bounded_iterative_agent_runtime_contract()

    assert contract["maximum_model_decisions_per_turn"] == 2
    assert contract["retrieval_call_count_per_turn"] == 1
    assert contract["maximum_evidence_inspection_count_per_turn"] == 1
    assert contract["second_retrieval_enabled"] is False
    assert contract["decision_loop_enabled"] is False
    assert contract["clarification_fallback_user_authorized"] is True
    assert contract["clarification_text_system_owned"] is True
    assert contract["runtime_registered_as_default"] is False
    assert contract["test_gate_opened"] is False


def test_ledger_can_explicitly_authorize_clarify_without_changing_v1_default() -> None:
    default = ThreadStateLimits(max_completed_turns=4, max_retained_bytes=1024)
    iterative = ThreadStateLimits(
        max_completed_turns=4,
        max_retained_bytes=1024,
        allowed_terminal_states=ITERATIVE_ALLOWED_TERMINAL_STATES,
    )
    ledger = VolatileThreadStateLedger(limits=iterative)
    ledger.open_thread("thread")
    summary = ledger.append_completed_turn(
        "thread",
        CompletedThreadTurn(1, "question", "clarification", "clarify"),
    )

    assert default.allowed_terminal_states == ("complete", "refuse")
    assert summary.completed_turn_count == 1


def _runtime(decisions: list[IterativeAgentDecision]):
    retriever = StaticRetriever()
    router = SequencedRouter(decisions)
    counters = {"compose": 0, "verify": 0, "composition_context_count": 0}
    toolset = PrimeQAHybridAgentToolset(
        candidate_pool_retriever=retriever,
        observation_adapter=PrimeQAHybridSidecarObservationAdapter(),
        answer_generator_factory=CountingGeneratorFactory(counters),
        answer_verifier_factory=CountingVerifierFactory(counters),
    )
    runtime = create_bounded_iterative_agent_runtime_from_toolset(
        toolset=toolset,
        decision_router=router,
    )
    return runtime, retriever, router, counters


def _decision(action: str, *, clarification_kind: str | None = None) -> IterativeAgentDecision:
    return IterativeAgentDecision(action=action, clarification_kind=clarification_kind)


def _question(identifier: str) -> PrimeQARuntimeQuery:
    return PrimeQARuntimeQuery(
        id=identifier,
        title="Adapter setup",
        text="How do I configure the verified adapter procedure?",
    )


def _candidate_results(prefix: str) -> tuple[RetrievalResult, ...]:
    results = []
    for rank in range(1, 401):
        relevant = rank > 10
        results.append(
            RetrievalResult(
                document=PrimeQADocument(
                    id=f"{prefix}-{rank:03d}",
                    title=("Verified adapter procedure" if relevant else f"Reference {rank}"),
                    text=(
                        "Configure the verified adapter procedure and restart the service."
                        if relevant
                        else "General product reference information."
                    ),
                ),
                score=1.0 / rank,
                rank=rank,
            )
        )
    return tuple(results)
