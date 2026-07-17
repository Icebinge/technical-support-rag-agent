from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from ts_rag_agent.application.answer_verification import AnswerVerifier
from ts_rag_agent.application.primeqa_hybrid_agent_retrieval_integration_validation import (
    _BASELINE_PREFIX_DEPTH,
    _DEFAULT_COMPOSITION_POLICY,
    _DEFAULT_EVIDENCE_SELECTOR,
    _DEFAULT_MAX_CANDIDATES_PER_DOCUMENT,
    _DEFAULT_MAX_SENTENCES,
    _DEFAULT_MIN_EVIDENCE_SCORE,
    _DEFAULT_MIN_SENTENCE_SCORE,
    _answer_generator,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_entrypoint_protocol import (
    SidecarAgentAction,
    SidecarAgentActionStateMachine,
    SidecarAgentState,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_orchestrator import (
    AnswerGeneratorPort,
    AnswerVerifierPort,
    PrimeQAHybridSidecarAgentOrchestrator,
    PrimeQAHybridSidecarAgentRun,
    SidecarAgentConsumerPolicy,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    PrimeQAHybridSidecarObservationAdapter,
)
from ts_rag_agent.domain.answer import AnswerVerificationResult, GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQAQuery
from ts_rag_agent.domain.retrieval import RetrievalResult

_ENTRYPOINT_ID = "stage138_optional_sidecar_agent_entrypoint_v1"
_ACTION_STATE_PROTOCOL_ID = "primeqa_hybrid_optional_sidecar_agent_entrypoint_protocol_v1"
_ORCHESTRATOR_ID = "stage116_primary_plus_stage128_sidecar_agent_orchestrator_v1"
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "answer_doc_id",
        "answer_text",
        "candidate_doc_ids",
        "cited_doc_ids",
        "document_body",
        "document_id",
        "document_text",
        "document_title",
        "gold_answer",
        "matched_token_strings",
        "question_id",
        "question_text",
        "question_title",
        "raw_answer_text",
        "raw_document_text",
        "raw_question_text",
        "retrieved_doc_ids",
        "runtime_content_handle",
        "source_doc_ids",
    }
)


class CandidatePoolRetrieverPort(Protocol):
    """Build or retrieve the frozen candidate pool for one entrypoint request."""

    def retrieve(self, question: PrimeQAQuery) -> Sequence[RetrievalResult]: ...


class SidecarAgentOrchestratorExecutionFactoryPort(Protocol):
    """Create one state-bound orchestrator execution bundle."""

    def create(
        self,
        state_machine: SidecarAgentActionStateMachine,
    ) -> InstrumentedSidecarAgentOrchestrator: ...


@dataclass(frozen=True)
class PublicSafeOptionalSidecarAgentEntrypointTrace:
    """Public-safe trace for one optional entrypoint execution."""

    entrypoint_id: str
    action_state_protocol_id: str
    orchestrator_id: str
    action_count: int
    terminal_state: str
    terminal: bool
    verified_refused: bool
    retriever_call_count: int
    orchestrator_call_count: int
    answer_generator_call_count: int
    answer_verifier_call_count: int
    sidecar_observation_count: int
    sidecar_used_for_answer_generation: bool
    sidecar_used_for_answer_verification: bool
    sidecar_replaced_primary_context: bool
    runtime_gold_labels_read: bool
    test_membership_read: bool
    retry_action_count: int
    fallback_action_count: int
    action_trace: tuple[dict[str, Any], ...]

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        forbidden = sorted(_forbidden_keys_found(payload))
        if forbidden:
            raise ValueError(f"Entrypoint trace contains forbidden keys: {forbidden}")
        return payload


@dataclass(frozen=True)
class OptionalSidecarAgentEntrypointRun:
    """Private entrypoint result plus its separately serializable public trace."""

    candidate_pool_results: tuple[RetrievalResult, ...]
    agent_run: PrimeQAHybridSidecarAgentRun
    generation_context_results: tuple[RetrievalResult, ...]
    verification_context_results: tuple[RetrievalResult, ...]
    public_safe_trace: PublicSafeOptionalSidecarAgentEntrypointTrace

    @property
    def original_answer(self) -> GeneratedAnswer:
        return self.agent_run.original_answer

    @property
    def verification_result(self) -> AnswerVerificationResult:
        return self.agent_run.verification_result

    @property
    def verified_answer(self) -> GeneratedAnswer:
        return self.agent_run.verified_answer


class StateAdvancingAnswerGenerator:
    """Advance from answer to verify only after real generation succeeds."""

    def __init__(
        self,
        *,
        delegate: AnswerGeneratorPort,
        state_machine: SidecarAgentActionStateMachine,
    ) -> None:
        self._delegate = delegate
        self._state_machine = state_machine
        self.call_count = 0
        self.last_context: tuple[RetrievalResult, ...] = ()

    def generate(
        self,
        question: PrimeQAQuery,
        retrieval_results: Sequence[RetrievalResult],
    ) -> GeneratedAnswer:
        if self._state_machine.state is not SidecarAgentState.ANSWER:
            raise RuntimeError("answer generator called outside the answer state")
        self.call_count += 1
        self.last_context = tuple(retrieval_results)
        generated = self._delegate.generate(question, retrieval_results)
        self._state_machine.advance(SidecarAgentAction.VERIFY)
        return generated


class StateAdvancingAnswerVerifier:
    """Advance from verify to observe only after real verification succeeds."""

    def __init__(
        self,
        *,
        delegate: AnswerVerifierPort,
        state_machine: SidecarAgentActionStateMachine,
    ) -> None:
        self._delegate = delegate
        self._state_machine = state_machine
        self.call_count = 0
        self.last_context: tuple[RetrievalResult, ...] = ()

    def verify(
        self,
        answer: GeneratedAnswer,
        retrieval_results: Sequence[RetrievalResult],
    ) -> AnswerVerificationResult:
        if self._state_machine.state is not SidecarAgentState.VERIFY:
            raise RuntimeError("answer verifier called outside the verify state")
        self.call_count += 1
        self.last_context = tuple(retrieval_results)
        verification = self._delegate.verify(answer, retrieval_results)
        self._state_machine.advance(SidecarAgentAction.OBSERVE)
        return verification


@dataclass(frozen=True)
class InstrumentedSidecarAgentOrchestrator:
    """One orchestrator plus the state-aware dependency wrappers it owns."""

    orchestrator: PrimeQAHybridSidecarAgentOrchestrator
    answer_generator: StateAdvancingAnswerGenerator
    answer_verifier: StateAdvancingAnswerVerifier


class FrozenSidecarAgentOrchestratorExecutionFactory:
    """Build one frozen Stage137-equivalent orchestrator per request."""

    def __init__(
        self,
        *,
        evidence_selector_name: str = _DEFAULT_EVIDENCE_SELECTOR,
        max_candidates_per_document: int = _DEFAULT_MAX_CANDIDATES_PER_DOCUMENT,
        composition_policy_name: str = _DEFAULT_COMPOSITION_POLICY,
        max_sentences: int = _DEFAULT_MAX_SENTENCES,
        min_sentence_score: float = _DEFAULT_MIN_SENTENCE_SCORE,
        min_evidence_score: float = _DEFAULT_MIN_EVIDENCE_SCORE,
    ) -> None:
        self._evidence_selector_name = evidence_selector_name
        self._max_candidates_per_document = max_candidates_per_document
        self._composition_policy_name = composition_policy_name
        self._max_sentences = max_sentences
        self._min_sentence_score = min_sentence_score
        self._min_evidence_score = min_evidence_score

    def create(
        self,
        state_machine: SidecarAgentActionStateMachine,
    ) -> InstrumentedSidecarAgentOrchestrator:
        generator = StateAdvancingAnswerGenerator(
            delegate=_answer_generator(
                evidence_selector_name=self._evidence_selector_name,
                max_candidates_per_document=self._max_candidates_per_document,
                composition_policy_name=self._composition_policy_name,
                max_sentences=self._max_sentences,
                min_sentence_score=self._min_sentence_score,
            ),
            state_machine=state_machine,
        )
        verifier = StateAdvancingAnswerVerifier(
            delegate=AnswerVerifier(
                min_citations=1,
                min_evidence_score=self._min_evidence_score,
                max_citation_rank=_BASELINE_PREFIX_DEPTH,
            ),
            state_machine=state_machine,
        )
        orchestrator = PrimeQAHybridSidecarAgentOrchestrator(
            observation_adapter=PrimeQAHybridSidecarObservationAdapter(),
            answer_generator=generator,
            answer_verifier=verifier,
            consumer_policy=SidecarAgentConsumerPolicy(),
        )
        return InstrumentedSidecarAgentOrchestrator(
            orchestrator=orchestrator,
            answer_generator=generator,
            answer_verifier=verifier,
        )


class PrimeQAHybridOptionalSidecarAgentEntrypoint:
    """Optional Stage139 entrypoint; it is not registered as the runtime default."""

    def __init__(
        self,
        *,
        candidate_pool_retriever: CandidatePoolRetrieverPort,
        orchestrator_factory: SidecarAgentOrchestratorExecutionFactoryPort | None = None,
    ) -> None:
        self._candidate_pool_retriever = candidate_pool_retriever
        self._orchestrator_factory = (
            orchestrator_factory or FrozenSidecarAgentOrchestratorExecutionFactory()
        )

    def run(self, question: PrimeQAQuery) -> OptionalSidecarAgentEntrypointRun:
        state_machine = SidecarAgentActionStateMachine()
        state_machine.advance(SidecarAgentAction.RETRIEVE)
        candidate_pool = tuple(self._candidate_pool_retriever.retrieve(question))
        state_machine.advance(SidecarAgentAction.ANSWER)

        execution = self._orchestrator_factory.create(state_machine)
        agent_run = execution.orchestrator.run(
            question=question,
            candidate_pool_results=candidate_pool,
        )
        if state_machine.state is not SidecarAgentState.OBSERVE:
            raise RuntimeError("orchestrator returned before the observe state")

        terminal_action = (
            SidecarAgentAction.REFUSE
            if agent_run.verified_answer.refused
            else SidecarAgentAction.COMPLETE
        )
        state_machine.advance(terminal_action)
        public_trace = _build_public_safe_entrypoint_trace(
            state_machine=state_machine,
            agent_run=agent_run,
            execution=execution,
        )
        return OptionalSidecarAgentEntrypointRun(
            candidate_pool_results=candidate_pool,
            agent_run=agent_run,
            generation_context_results=execution.answer_generator.last_context,
            verification_context_results=execution.answer_verifier.last_context,
            public_safe_trace=public_trace,
        )


def create_primeqa_hybrid_optional_sidecar_agent_entrypoint(
    *,
    candidate_pool_retriever: CandidatePoolRetrieverPort,
    evidence_selector_name: str = _DEFAULT_EVIDENCE_SELECTOR,
    max_candidates_per_document: int = _DEFAULT_MAX_CANDIDATES_PER_DOCUMENT,
    composition_policy_name: str = _DEFAULT_COMPOSITION_POLICY,
    max_sentences: int = _DEFAULT_MAX_SENTENCES,
    min_sentence_score: float = _DEFAULT_MIN_SENTENCE_SCORE,
    min_evidence_score: float = _DEFAULT_MIN_EVIDENCE_SCORE,
) -> PrimeQAHybridOptionalSidecarAgentEntrypoint:
    """Build the optional entrypoint without registering it as a runtime default."""

    factory = FrozenSidecarAgentOrchestratorExecutionFactory(
        evidence_selector_name=evidence_selector_name,
        max_candidates_per_document=max_candidates_per_document,
        composition_policy_name=composition_policy_name,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        min_evidence_score=min_evidence_score,
    )
    return PrimeQAHybridOptionalSidecarAgentEntrypoint(
        candidate_pool_retriever=candidate_pool_retriever,
        orchestrator_factory=factory,
    )


def optional_sidecar_agent_entrypoint_contract() -> dict[str, Any]:
    """Return the public-safe Stage139 optional entrypoint implementation contract."""

    return {
        "entrypoint_id": _ENTRYPOINT_ID,
        "implementation_module": (
            "ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_entrypoint"
        ),
        "factory": "create_primeqa_hybrid_optional_sidecar_agent_entrypoint",
        "action_state_protocol_id": _ACTION_STATE_PROTOCOL_ID,
        "orchestrator_id": _ORCHESTRATOR_ID,
        "dependency_ports": {
            "candidate_pool_retriever": "CandidatePoolRetrieverPort",
            "orchestrator_execution_factory": ("SidecarAgentOrchestratorExecutionFactoryPort"),
        },
        "execution_order": (
            "retrieve",
            "answer",
            "verify",
            "observe",
            "complete_or_refuse",
        ),
        "observation_semantics": (
            "sidecar_data_is_prepared_inside_the_orchestrator_but_published_only_in_"
            "the_post_verification_observe_state"
        ),
        "request_isolation": {
            "new_state_machine_per_request": True,
            "new_instrumented_orchestrator_per_request": True,
        },
        "permissions": {
            "optional_entrypoint": True,
            "registered_as_runtime_default": False,
            "sidecar_answer_generation_allowed": False,
            "sidecar_verification_context_allowed": False,
            "sidecar_primary_context_replacement_allowed": False,
            "retry_actions_allowed": False,
            "fallback_strategies_allowed": False,
            "test_access_allowed": False,
        },
        "error_behavior": {
            "retrieval_error": "propagate_without_retry_or_fallback",
            "generation_error": "propagate_without_retry_or_fallback",
            "verification_error": "propagate_without_retry_or_fallback",
            "invalid_transition": "raise_without_state_change",
        },
        "public_trace": {
            "contains_raw_question_text": False,
            "contains_raw_answer_text": False,
            "contains_raw_document_text": False,
            "contains_document_identifiers": False,
            "contains_runtime_content_handles": False,
            "contains_gold_labels": False,
            "contains_test_membership": False,
            "action_trace_available": True,
            "dependency_call_counts_available": True,
        },
    }


def _build_public_safe_entrypoint_trace(
    *,
    state_machine: SidecarAgentActionStateMachine,
    agent_run: PrimeQAHybridSidecarAgentRun,
    execution: InstrumentedSidecarAgentOrchestrator,
) -> PublicSafeOptionalSidecarAgentEntrypointTrace:
    orchestrator_trace = agent_run.public_safe_trace
    trace = PublicSafeOptionalSidecarAgentEntrypointTrace(
        entrypoint_id=_ENTRYPOINT_ID,
        action_state_protocol_id=_ACTION_STATE_PROTOCOL_ID,
        orchestrator_id=_ORCHESTRATOR_ID,
        action_count=len(state_machine.trace),
        terminal_state=state_machine.state.value,
        terminal=state_machine.terminal,
        verified_refused=agent_run.verified_answer.refused,
        retriever_call_count=1,
        orchestrator_call_count=1,
        answer_generator_call_count=execution.answer_generator.call_count,
        answer_verifier_call_count=execution.answer_verifier.call_count,
        sidecar_observation_count=orchestrator_trace.sidecar_observation_count,
        sidecar_used_for_answer_generation=(orchestrator_trace.sidecar_used_for_answer_generation),
        sidecar_used_for_answer_verification=(
            orchestrator_trace.sidecar_used_for_answer_verification
        ),
        sidecar_replaced_primary_context=orchestrator_trace.sidecar_replaced_primary_context,
        runtime_gold_labels_read=orchestrator_trace.runtime_gold_labels_read,
        test_membership_read=orchestrator_trace.test_membership_read,
        retry_action_count=0,
        fallback_action_count=0,
        action_trace=tuple(state_machine.public_trace()),
    )
    trace.to_public_dict()
    return trace


def _forbidden_keys_found(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in _FORBIDDEN_PUBLIC_KEYS:
                found.add(key_text)
            found.update(_forbidden_keys_found(child))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for child in value:
            found.update(_forbidden_keys_found(child))
    return found
