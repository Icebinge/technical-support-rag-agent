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
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    PrimeQAHybridSidecarObservationAdapter,
    SidecarObservationBundle,
    SidecarObservationRecord,
)
from ts_rag_agent.domain.answer import AnswerVerificationResult, GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQAQuery
from ts_rag_agent.domain.retrieval import RetrievalResult

_ORCHESTRATOR_ID = "stage116_primary_plus_stage128_sidecar_agent_orchestrator_v1"
_PRIMARY_CHANNEL_ID = "stage116_primary_answer_context"
_VERIFICATION_CHANNEL_ID = "stage116_prefix_verification_context"
_SIDECAR_CHANNEL_ID = "stage128_stage132_sidecar_observation"
_SIDECAR_EFFECTIVENESS_STATUS = "diagnostic_only_unproven"
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


class AnswerGeneratorPort(Protocol):
    """Answer-generation dependency accepted by the orchestrator."""

    def generate(
        self,
        question: PrimeQAQuery,
        retrieval_results: Sequence[RetrievalResult],
    ) -> GeneratedAnswer: ...


class AnswerVerifierPort(Protocol):
    """Answer-verification dependency accepted by the orchestrator."""

    def verify(
        self,
        answer: GeneratedAnswer,
        retrieval_results: Sequence[RetrievalResult],
    ) -> AnswerVerificationResult: ...


@dataclass(frozen=True)
class SidecarAgentConsumerPolicy:
    """Frozen Stage136 sidecar consumer permissions."""

    observation_rendering_allowed: bool = True
    evidence_gap_trace_allowed: bool = True
    citation_verification_probe_mode: str = _SIDECAR_EFFECTIVENESS_STATUS
    answer_generation_allowed: bool = False
    answer_verification_context_allowed: bool = False
    primary_context_replacement_allowed: bool = False
    runtime_defaultization_allowed: bool = False
    fallback_strategy_allowed: bool = False

    def __post_init__(self) -> None:
        if self.answer_generation_allowed:
            raise ValueError("sidecar answer generation is blocked")
        if self.answer_verification_context_allowed:
            raise ValueError("sidecar answer verification context is blocked")
        if self.primary_context_replacement_allowed:
            raise ValueError("sidecar primary-context replacement is blocked")
        if self.runtime_defaultization_allowed:
            raise ValueError("Stage136 runtime defaultization is blocked")
        if self.fallback_strategy_allowed:
            raise ValueError("Stage136 fallback strategies are blocked")
        if self.citation_verification_probe_mode != _SIDECAR_EFFECTIVENESS_STATUS:
            raise ValueError("citation verification must remain diagnostic_only_unproven")


@dataclass(frozen=True)
class PublicSafeSidecarSelectionTrace:
    """Public-safe metadata for one selected sidecar observation."""

    sidecar_observation_rank: int
    candidate_pool_rank: int
    source_region: str
    route_family: str
    query_overlap_count: int
    query_overlap_ratio: float
    retrieval_prior: float
    combined_score: float
    query_overlap_present: bool
    novel_query_term_count: int
    novel_query_term_ratio: float
    extends_primary_query_coverage: bool
    duplicate_of_primary_context: bool
    selected_for_answer_generation: bool = False
    selected_for_answer_verification: bool = False


@dataclass(frozen=True)
class PublicSafeSidecarAgentTrace:
    """One public-safe Stage136 agent execution trace."""

    orchestrator_id: str
    primary_channel_id: str
    verification_channel_id: str
    sidecar_channel_id: str
    primary_context_count: int
    verification_context_count: int
    sidecar_observation_count: int
    sidecar_query_overlap_signal_count: int
    sidecar_novel_query_coverage_signal_count: int
    sidecar_used_for_answer_generation: bool
    sidecar_used_for_answer_verification: bool
    sidecar_replaced_primary_context: bool
    original_refused: bool
    verified_refused: bool
    verified_citation_count: int
    citation_context_valid: bool
    verification_reasons: tuple[str, ...]
    evidence_gap_trace_status: str
    sidecar_effectiveness_status: str
    runtime_gold_labels_read: bool
    test_membership_read: bool
    miss_analysis_status: str
    sidecar_selection_trace: tuple[PublicSafeSidecarSelectionTrace, ...]

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize the trace after enforcing the public-safe field contract."""

        payload = asdict(self)
        forbidden = sorted(_forbidden_keys_found(payload))
        if forbidden:
            raise ValueError(f"Public trace contains forbidden keys: {forbidden}")
        return payload


@dataclass(frozen=True)
class PrimeQAHybridSidecarAgentRun:
    """Private runtime result plus its separately serializable public trace."""

    original_answer: GeneratedAnswer
    verification_result: AnswerVerificationResult
    observation_bundle: SidecarObservationBundle
    public_safe_trace: PublicSafeSidecarAgentTrace

    @property
    def verified_answer(self) -> GeneratedAnswer:
        return self.verification_result.verified_answer


class PrimeQAHybridSidecarAgentOrchestrator:
    """Orchestrate Stage116 answering with an isolated Stage128 sidecar."""

    def __init__(
        self,
        *,
        observation_adapter: PrimeQAHybridSidecarObservationAdapter,
        answer_generator: AnswerGeneratorPort,
        answer_verifier: AnswerVerifierPort,
        consumer_policy: SidecarAgentConsumerPolicy | None = None,
        prefix_depth: int = _BASELINE_PREFIX_DEPTH,
    ) -> None:
        if prefix_depth <= 0:
            raise ValueError("prefix_depth must be positive")
        self._observation_adapter = observation_adapter
        self._answer_generator = answer_generator
        self._answer_verifier = answer_verifier
        self._consumer_policy = consumer_policy or SidecarAgentConsumerPolicy()
        self._prefix_depth = prefix_depth

    @property
    def consumer_policy(self) -> SidecarAgentConsumerPolicy:
        return self._consumer_policy

    def run(
        self,
        *,
        question: PrimeQAQuery,
        candidate_pool_results: Sequence[RetrievalResult],
    ) -> PrimeQAHybridSidecarAgentRun:
        """Run one answer while keeping sidecar candidates out of answer paths."""

        _validate_candidate_pool(candidate_pool_results)
        bundle = self._observation_adapter.observe(
            question=question,
            candidate_pool_results=candidate_pool_results,
        )
        primary_answer_context = bundle.answer_context_for_generation()
        verification_context = tuple(
            sorted(
                (result for result in candidate_pool_results if result.rank <= self._prefix_depth),
                key=lambda result: result.rank,
            )
        )
        original_answer = self._answer_generator.generate(
            question,
            primary_answer_context,
        )
        verification = self._answer_verifier.verify(
            original_answer,
            verification_context,
        )
        trace = _build_public_safe_trace(
            bundle=bundle,
            verification_context=verification_context,
            original_answer=original_answer,
            verification=verification,
        )
        return PrimeQAHybridSidecarAgentRun(
            original_answer=original_answer,
            verification_result=verification,
            observation_bundle=bundle,
            public_safe_trace=trace,
        )


def create_primeqa_hybrid_sidecar_agent_orchestrator(
    *,
    evidence_selector_name: str = _DEFAULT_EVIDENCE_SELECTOR,
    max_candidates_per_document: int = _DEFAULT_MAX_CANDIDATES_PER_DOCUMENT,
    composition_policy_name: str = _DEFAULT_COMPOSITION_POLICY,
    max_sentences: int = _DEFAULT_MAX_SENTENCES,
    min_sentence_score: float = _DEFAULT_MIN_SENTENCE_SCORE,
    min_evidence_score: float = _DEFAULT_MIN_EVIDENCE_SCORE,
) -> PrimeQAHybridSidecarAgentOrchestrator:
    """Build the Stage136 orchestrator with the frozen Stage116 answer path."""

    generator = _answer_generator(
        evidence_selector_name=evidence_selector_name,
        max_candidates_per_document=max_candidates_per_document,
        composition_policy_name=composition_policy_name,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
    )
    verifier = AnswerVerifier(
        min_citations=1,
        min_evidence_score=min_evidence_score,
        max_citation_rank=_BASELINE_PREFIX_DEPTH,
    )
    return PrimeQAHybridSidecarAgentOrchestrator(
        observation_adapter=PrimeQAHybridSidecarObservationAdapter(),
        answer_generator=generator,
        answer_verifier=verifier,
    )


def sidecar_agent_orchestrator_contract() -> dict[str, Any]:
    """Return the public-safe Stage136 implementation contract."""

    policy = SidecarAgentConsumerPolicy()
    return {
        "orchestrator_id": _ORCHESTRATOR_ID,
        "implementation_module": (
            "ts_rag_agent.application.primeqa_hybrid_sidecar_agent_orchestrator"
        ),
        "factory": "create_primeqa_hybrid_sidecar_agent_orchestrator",
        "channel_routing": {
            "answer_generation": _PRIMARY_CHANNEL_ID,
            "answer_verification": _VERIFICATION_CHANNEL_ID,
            "sidecar_observation": _SIDECAR_CHANNEL_ID,
        },
        "depths": {
            "primary_answer_context": 10,
            "stage116_verification_context_max": _BASELINE_PREFIX_DEPTH,
            "stage128_candidate_pool_max": 400,
            "sidecar_observation_slots": 3,
        },
        "consumer_policy": asdict(policy),
        "public_trace": {
            "contains_raw_question_text": False,
            "contains_raw_answer_text": False,
            "contains_raw_document_text": False,
            "contains_document_identifiers": False,
            "contains_runtime_content_handles": False,
            "contains_gold_labels": False,
            "contains_test_membership": False,
            "sidecar_selection_metadata_available": True,
            "verification_reason_metadata_available": True,
            "miss_analysis_status": ("runtime_gold_free_trace_cannot_label_answer_document_miss"),
        },
        "effectiveness_boundary": {
            "status": _SIDECAR_EFFECTIVENESS_STATUS,
            "citation_verification_effectiveness_claim_allowed": False,
            "answer_quality_improvement_claim_allowed": False,
            "retrieval_improvement_claim_allowed": False,
        },
    }


def _build_public_safe_trace(
    *,
    bundle: SidecarObservationBundle,
    verification_context: Sequence[RetrievalResult],
    original_answer: GeneratedAnswer,
    verification: AnswerVerificationResult,
) -> PublicSafeSidecarAgentTrace:
    sidecar_rows = tuple(
        _public_sidecar_selection_trace(record) for record in bundle.sidecar_observations
    )
    novel_count = sum(row.extends_primary_query_coverage for row in sidecar_rows)
    evidence_gap_status = _evidence_gap_trace_status(
        verified_refused=verification.verified_answer.refused,
        sidecar_observation_count=len(sidecar_rows),
        novel_signal_count=novel_count,
    )
    trace = PublicSafeSidecarAgentTrace(
        orchestrator_id=_ORCHESTRATOR_ID,
        primary_channel_id=_PRIMARY_CHANNEL_ID,
        verification_channel_id=_VERIFICATION_CHANNEL_ID,
        sidecar_channel_id=_SIDECAR_CHANNEL_ID,
        primary_context_count=len(bundle.answer_context_results),
        verification_context_count=len(verification_context),
        sidecar_observation_count=len(sidecar_rows),
        sidecar_query_overlap_signal_count=sum(row.query_overlap_present for row in sidecar_rows),
        sidecar_novel_query_coverage_signal_count=novel_count,
        sidecar_used_for_answer_generation=False,
        sidecar_used_for_answer_verification=False,
        sidecar_replaced_primary_context=False,
        original_refused=original_answer.refused,
        verified_refused=verification.verified_answer.refused,
        verified_citation_count=len(verification.verified_answer.citations),
        citation_context_valid=verification.citation_context_valid,
        verification_reasons=tuple(verification.reasons),
        evidence_gap_trace_status=evidence_gap_status,
        sidecar_effectiveness_status=_SIDECAR_EFFECTIVENESS_STATUS,
        runtime_gold_labels_read=False,
        test_membership_read=False,
        miss_analysis_status=("runtime_gold_free_trace_cannot_label_answer_document_miss"),
        sidecar_selection_trace=sidecar_rows,
    )
    trace.to_public_dict()
    return trace


def _public_sidecar_selection_trace(
    record: SidecarObservationRecord,
) -> PublicSafeSidecarSelectionTrace:
    score = record.sidecar_score_summary
    signal = record.citation_verification_signal
    return PublicSafeSidecarSelectionTrace(
        sidecar_observation_rank=record.sidecar_observation_rank,
        candidate_pool_rank=score.retrieval_rank,
        source_region=record.sidecar_source_region,
        route_family=record.sidecar_route_family,
        query_overlap_count=score.query_overlap_count,
        query_overlap_ratio=score.query_overlap_ratio,
        retrieval_prior=score.retrieval_prior,
        combined_score=score.combined_score,
        query_overlap_present=signal.query_overlap_present,
        novel_query_term_count=signal.novel_query_term_count,
        novel_query_term_ratio=signal.novel_query_term_ratio,
        extends_primary_query_coverage=signal.extends_primary_query_coverage,
        duplicate_of_primary_context=signal.duplicate_of_primary_context,
    )


def _evidence_gap_trace_status(
    *,
    verified_refused: bool,
    sidecar_observation_count: int,
    novel_signal_count: int,
) -> str:
    if verified_refused:
        return "verified_answer_refused_sidecar_diagnostic_only"
    if sidecar_observation_count == 0:
        return "no_sidecar_observation_available"
    if novel_signal_count > 0:
        return "novel_query_coverage_observed_not_answer_evidence"
    return "sidecar_observed_without_novel_query_coverage"


def _validate_candidate_pool(
    candidate_pool_results: Sequence[RetrievalResult],
) -> None:
    ranks = [result.rank for result in candidate_pool_results]
    if any(rank <= 0 for rank in ranks):
        raise ValueError("candidate ranks must be positive")
    if len(ranks) != len(set(ranks)):
        raise ValueError("candidate ranks must be unique")
    if any(rank > 400 for rank in ranks):
        raise ValueError("candidate ranks must not exceed the Stage128 depth 400")


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
