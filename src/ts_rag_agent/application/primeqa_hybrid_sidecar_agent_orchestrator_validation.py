from __future__ import annotations

import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.answer_verification import AnswerVerifier
from ts_rag_agent.application.primeqa_hybrid_agent_retrieval_integration_validation import (
    _BASELINE_PREFIX_DEPTH,
    _DEFAULT_BM25_B,
    _DEFAULT_BM25_K1,
    _DEFAULT_COMPOSITION_POLICY,
    _DEFAULT_ENCODER_BATCH_SIZE,
    _DEFAULT_EVIDENCE_SELECTOR,
    _DEFAULT_MAX_CANDIDATES_PER_DOCUMENT,
    _DEFAULT_MAX_SENTENCES,
    _DEFAULT_MIN_EVIDENCE_SCORE,
    _DEFAULT_MIN_SENTENCE_SCORE,
    _DEV_SPLIT,
    _SELECTED_CONFIG_ID,
    _TARGET_POOL_DEPTH,
    _TRAIN_SPLIT,
    _answer_generator,
    _answer_signature,
    _candidate_pool_summary,
    _candidate_pools_by_split,
    _DocumentEvidenceShortlister,
    _evaluation_channels,
    _public_channel_catalog,
    _selected_append_config,
    _selected_channel_top_k,
)
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    EncoderFactory,
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
    _fingerprint,
    _load_json_object,
    _rounded_mean,
    _rounded_ratio,
    _section_summary,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_orchestrator import (
    AnswerGeneratorPort,
    AnswerVerifierPort,
    PrimeQAHybridSidecarAgentOrchestrator,
    PrimeQAHybridSidecarAgentRun,
    SidecarAgentConsumerPolicy,
    sidecar_agent_orchestrator_contract,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    PrimeQAHybridSidecarObservationAdapter,
    _pool_results,
    _result_signature,
)
from ts_rag_agent.application.rag_answering import evaluate_answers
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.answer import AnswerVerificationResult, GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
    summarize_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 137"
_CREATED_AT = "2026-07-16"
_ANALYSIS_ID = "primeqa_hybrid_sidecar_agent_orchestrator_train_cv_dev_validation_v1"
_SOURCE_STAGE136_STATUS = "primeqa_hybrid_sidecar_agent_orchestrator_protocol_frozen"
_SOURCE_STAGE136_PROTOCOL_ID = "primeqa_hybrid_sidecar_agent_orchestrator_protocol_v1"
_SOURCE_STAGE136_NEXT = (
    "run_stage116_primary_plus_sidecar_agent_orchestrator_train_cv_dev_validation"
)
_SOURCE_STAGE135_STATUS = "primeqa_hybrid_sidecar_observation_validation_passed"
_SOURCE_STAGE135_ANALYSIS_ID = (
    "primeqa_hybrid_stage116_answer_context_stage128_sidecar_observation_validation_v1"
)
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_MINIMUM_TRAIN_FOLDS = 5
_PRIMARY_CONTEXT_DEPTH = 10
_SIDECAR_APPEND_START_RANK = 201
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


@dataclass(frozen=True)
class Stage116ControlRun:
    """Private Stage116 control execution used only inside Stage137."""

    answer_context_results: tuple[RetrievalResult, ...]
    verification_context_results: tuple[RetrievalResult, ...]
    original_answer: GeneratedAnswer
    verification_result: AnswerVerificationResult

    @property
    def verified_answer(self) -> GeneratedAnswer:
        return self.verification_result.verified_answer


class Stage116ControlRunner:
    """Execute the frozen Stage116 answer and verification path."""

    def __init__(
        self,
        *,
        answer_generator: AnswerGeneratorPort,
        answer_verifier: AnswerVerifierPort,
        shortlister: _DocumentEvidenceShortlister | None = None,
    ) -> None:
        self._answer_generator = answer_generator
        self._answer_verifier = answer_verifier
        self._shortlister = shortlister or _DocumentEvidenceShortlister()

    def run(
        self,
        *,
        question: PrimeQAQuestion,
        prefix_results: Sequence[RetrievalResult],
    ) -> Stage116ControlRun:
        answer_context = tuple(
            self._shortlister.shortlist(
                question=question,
                candidates=prefix_results,
                top_k=_PRIMARY_CONTEXT_DEPTH,
            )
        )
        verification_context = tuple(
            sorted(
                (result for result in prefix_results if result.rank <= _BASELINE_PREFIX_DEPTH),
                key=lambda result: result.rank,
            )
        )
        original_answer = self._answer_generator.generate(question, answer_context)
        verification = self._answer_verifier.verify(original_answer, verification_context)
        return Stage116ControlRun(
            answer_context_results=answer_context,
            verification_context_results=verification_context,
            original_answer=original_answer,
            verification_result=verification,
        )


class RecordingAnswerGenerator:
    """Record the exact generation context before delegating."""

    def __init__(self, delegate: AnswerGeneratorPort) -> None:
        self._delegate = delegate
        self.last_context: tuple[RetrievalResult, ...] = ()

    def generate(
        self,
        question: PrimeQAQuestion,
        retrieval_results: Sequence[RetrievalResult],
    ) -> GeneratedAnswer:
        self.last_context = tuple(retrieval_results)
        return self._delegate.generate(question, retrieval_results)


class RecordingAnswerVerifier:
    """Record the exact verification context before delegating."""

    def __init__(self, delegate: AnswerVerifierPort) -> None:
        self._delegate = delegate
        self.last_context: tuple[RetrievalResult, ...] = ()

    def verify(
        self,
        answer: GeneratedAnswer,
        retrieval_results: Sequence[RetrievalResult],
    ) -> AnswerVerificationResult:
        self.last_context = tuple(retrieval_results)
        return self._delegate.verify(answer, retrieval_results)


@dataclass(frozen=True)
class AgentHarnessRun:
    """Agent result plus the exact contexts observed by validation wrappers."""

    agent_run: PrimeQAHybridSidecarAgentRun
    generation_context_results: tuple[RetrievalResult, ...]
    verification_context_results: tuple[RetrievalResult, ...]


class PrimeQAHybridSidecarAgentValidationHarness:
    """Run the Stage136 orchestrator while recording dependency inputs."""

    def __init__(
        self,
        *,
        orchestrator: PrimeQAHybridSidecarAgentOrchestrator,
        recording_generator: RecordingAnswerGenerator,
        recording_verifier: RecordingAnswerVerifier,
    ) -> None:
        self._orchestrator = orchestrator
        self._recording_generator = recording_generator
        self._recording_verifier = recording_verifier

    def run(
        self,
        *,
        question: PrimeQAQuestion,
        candidate_pool_results: Sequence[RetrievalResult],
    ) -> AgentHarnessRun:
        self._recording_generator.last_context = ()
        self._recording_verifier.last_context = ()
        agent_run = self._orchestrator.run(
            question=question,
            candidate_pool_results=candidate_pool_results,
        )
        return AgentHarnessRun(
            agent_run=agent_run,
            generation_context_results=self._recording_generator.last_context,
            verification_context_results=self._recording_verifier.last_context,
        )


@dataclass(frozen=True)
class PrimeQAHybridSidecarAgentOrchestratorValidationVisualization:
    """One generated Stage137 public-safe chart."""

    name: str
    path: str


@dataclass(frozen=True)
class _AgentValidationTrace:
    sample: PrimeQAHybridSplitSample
    question: PrimeQAQuestion
    control_original_answer: GeneratedAnswer
    control_verified_answer: GeneratedAnswer
    agent_original_answer: GeneratedAnswer
    agent_verified_answer: GeneratedAnswer
    generation_context_identity_violation: bool
    verification_context_identity_violation: bool
    bundle_generation_context_identity_violation: bool
    original_answer_identity_violation: bool
    verified_answer_identity_violation: bool
    verification_reason_identity_violation: bool
    sidecar_generation_leak_count: int
    sidecar_verification_leak_count: int
    sidecar_primary_overlap_count: int
    public_trace_serialization_violation: bool
    public_trace_forbidden_key_count: int
    public_trace_contract_violation_count: int
    sidecar_observation_count: int
    sidecar_query_overlap_signal_count: int
    sidecar_novel_query_coverage_signal_count: int
    evidence_gap_trace_status: str
    primary_context_gold_hit: bool
    append_pool_gold_hit: bool
    sidecar_gold_hit: bool


def run_primeqa_hybrid_sidecar_agent_orchestrator_validation(
    *,
    stage136_protocol_path: Path,
    stage135_validation_path: Path,
    stage128_protocol_path: Path,
    stage125_protocol_path: Path,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    user_confirmed_validation: bool,
    confirmation_note: str,
    stage80_report_path: Path | None = None,
    include_dense_channels: bool = True,
    bm25_k1: float = _DEFAULT_BM25_K1,
    bm25_b: float = _DEFAULT_BM25_B,
    train_fold_count: int = _MINIMUM_TRAIN_FOLDS,
    encoder_batch_size: int = _DEFAULT_ENCODER_BATCH_SIZE,
    encoder_device: str | None = None,
    encoder_factory: EncoderFactory | None = None,
    evidence_selector_name: str = _DEFAULT_EVIDENCE_SELECTOR,
    max_candidates_per_document: int = _DEFAULT_MAX_CANDIDATES_PER_DOCUMENT,
    composition_policy_name: str = _DEFAULT_COMPOSITION_POLICY,
    max_sentences: int = _DEFAULT_MAX_SENTENCES,
    min_sentence_score: float = _DEFAULT_MIN_SENTENCE_SCORE,
    min_evidence_score: float = _DEFAULT_MIN_EVIDENCE_SCORE,
) -> dict[str, Any]:
    """Run real Stage137 train grouped-CV/dev agent validation."""

    _validate_options(
        train_fold_count=train_fold_count,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        encoder_batch_size=encoder_batch_size,
        max_candidates_per_document=max_candidates_per_document,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        min_evidence_score=min_evidence_score,
    )
    started_at = time.perf_counter()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    stage136_protocol = _load_json_object(stage136_protocol_path)
    stage135_validation = _load_json_object(stage135_validation_path)
    stage128_protocol = _load_json_object(stage128_protocol_path)
    stage125_protocol = _load_json_object(stage125_protocol_path)
    stage136_summary = _stage136_summary(stage136_protocol)
    stage135_summary = _stage135_summary(stage135_validation)
    stage128_summary = _stage128_summary(stage128_protocol)
    selected_config = _selected_append_config(
        stage125_protocol=stage125_protocol,
        stage128_summary=stage128_summary,
    )
    loaded_protocols_at = time.perf_counter()

    split_samples = {
        _TRAIN_SPLIT: load_primeqa_hybrid_split_samples(train_split_path),
        _DEV_SPLIT: load_primeqa_hybrid_split_samples(dev_split_path),
    }
    train_fold_assignments = _build_train_fold_assignments(
        split_samples[_TRAIN_SPLIT],
        fold_count=train_fold_count,
    )
    loaded_splits_at = time.perf_counter()

    documents_by_id = load_primeqa_documents(documents_path)
    sections_by_document = load_primeqa_document_sections(documents_path)
    documents = list(documents_by_id.values())
    document_ids = tuple(document.id for document in documents)
    stage80_report = _load_json_object(stage80_report_path) if stage80_report_path else None
    loaded_documents_at = time.perf_counter()

    dense_channels, dense_summary = _build_dense_channels(
        include_dense_channels=include_dense_channels,
        stage80_report=stage80_report,
        stage80_report_path=stage80_report_path,
        documents=documents,
        document_ids=document_ids,
        encoder_batch_size=encoder_batch_size,
        encoder_device=encoder_device,
        encoder_factory=encoder_factory,
    )
    dense_preflight_at = time.perf_counter()

    pre_checks = _pre_validation_guard_checks(
        stage136_summary=stage136_summary,
        stage135_summary=stage135_summary,
        selected_config=selected_config,
        split_samples=split_samples,
        train_fold_count=train_fold_count,
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
        include_dense_channels=include_dense_channels,
        dense_summary=dense_summary,
    )
    if not all(check["passed"] for check in pre_checks):
        checked_at = time.perf_counter()
        report = _blocked_report(
            stage136_protocol_path=stage136_protocol_path,
            stage135_validation_path=stage135_validation_path,
            stage128_protocol_path=stage128_protocol_path,
            stage125_protocol_path=stage125_protocol_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
            stage136_summary=stage136_summary,
            stage135_summary=stage135_summary,
            stage128_summary=stage128_summary,
            split_samples=split_samples,
            documents=documents,
            sections_by_document=sections_by_document,
            dense_summary=dense_summary,
            guard_checks=pre_checks,
            timing_seconds={
                "load_protocols": round(loaded_protocols_at - started_at, 3),
                "load_splits_and_build_train_folds": round(
                    loaded_splits_at - loaded_protocols_at,
                    3,
                ),
                "load_documents_sections": round(
                    loaded_documents_at - loaded_splits_at,
                    3,
                ),
                "dense_preflight": round(dense_preflight_at - loaded_documents_at, 3),
                "guard_checks": round(checked_at - dense_preflight_at, 3),
                "total": round(checked_at - started_at, 3),
            },
        )
        return {**report, "public_safe_contract": _public_safe_contract(report)}

    assert selected_config is not None
    lexical_channels = _build_lexical_channels(
        documents=documents,
        sections_by_document=sections_by_document,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        component_depth=_selected_channel_top_k(selected_config),
    )
    channels = _evaluation_channels(
        lexical_channels=lexical_channels,
        dense_channels=dense_channels,
    )
    channel_catalog = _public_channel_catalog(channels)
    indexed_at = time.perf_counter()

    candidate_pools_by_split = _candidate_pools_by_split(
        split_samples=split_samples,
        selected_config=selected_config,
        channels=channels,
    )
    pools_built_at = time.perf_counter()

    control_runner = create_stage116_control_runner(
        evidence_selector_name=evidence_selector_name,
        max_candidates_per_document=max_candidates_per_document,
        composition_policy_name=composition_policy_name,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        min_evidence_score=min_evidence_score,
    )
    agent_harness = create_sidecar_agent_validation_harness(
        evidence_selector_name=evidence_selector_name,
        max_candidates_per_document=max_candidates_per_document,
        composition_policy_name=composition_policy_name,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        min_evidence_score=min_evidence_score,
    )
    traces_by_split = {
        split: [
            _trace_agent_validation(
                sample=sample,
                pool=candidate_pools_by_split[split][sample.sample_id],
                documents_by_id=documents_by_id,
                control_runner=control_runner,
                agent_harness=agent_harness,
            )
            for sample in samples
        ]
        for split, samples in split_samples.items()
    }
    evaluated_at = time.perf_counter()

    split_reports = {
        split: _summarize_agent_traces(traces) for split, traces in traces_by_split.items()
    }
    train_fold_reports = _train_fold_reports(
        traces=traces_by_split[_TRAIN_SPLIT],
        fold_assignments=train_fold_assignments,
    )
    train_cv_validation = _train_cv_validation(
        train_summary=split_reports[_TRAIN_SPLIT],
        fold_reports=train_fold_reports,
    )
    dev_report = _dev_report(split_reports[_DEV_SPLIT])
    pool_summary = _candidate_pool_summary(candidate_pools_by_split)
    report_payload = {
        "candidate_pool_summary": pool_summary,
        "split_agent_reports": split_reports,
        "train_fold_reports": train_fold_reports,
        "train_cv_validation": train_cv_validation,
        "dev_report": dev_report,
    }
    guard_checks = pre_checks + _post_validation_guard_checks(
        stage136_summary=stage136_summary,
        stage135_summary=stage135_summary,
        pool_summary=pool_summary,
        report_payload=report_payload,
    )
    checked_at = time.perf_counter()
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Real train five-fold grouped-CV and dev report-only validation of the "
            "fixed Stage136 Stage116-primary plus Stage128-sidecar agent orchestrator. "
            "It rebuilds train/dev candidate pools and runs private per-row control and "
            "agent traces in memory. Public output contains aggregate statistics only. "
            "It does not load test, tune on dev, run final metrics, change runtime "
            "defaults, or enable fallback strategies."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_validation),
            "confirmation_note": confirmation_note,
        },
        "split_contract": _split_contract(),
        "source_files": _source_files(
            stage136_protocol_path=stage136_protocol_path,
            stage135_validation_path=stage135_validation_path,
            stage128_protocol_path=stage128_protocol_path,
            stage125_protocol_path=stage125_protocol_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
        ),
        "stage136_summary": stage136_summary,
        "stage135_summary": stage135_summary,
        "stage128_summary": stage128_summary,
        "orchestrator_contract": sidecar_agent_orchestrator_contract(),
        "validation_harness_contract": _validation_harness_contract(),
        "evaluation_options": {
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "train_fold_count": train_fold_count,
            "include_dense_channels": include_dense_channels,
            "encoder_batch_size": encoder_batch_size,
            "encoder_device": encoder_device,
            "evidence_selector_name": evidence_selector_name,
            "max_candidates_per_document": max_candidates_per_document,
            "composition_policy_name": composition_policy_name,
            "max_sentences": max_sentences,
            "min_sentence_score": min_sentence_score,
            "min_evidence_score": min_evidence_score,
            "test_split_loaded": False,
        },
        "loaded_data_summary": {
            "split_samples": summarize_primeqa_hybrid_split_samples(split_samples),
            "document_count": len(documents),
            **_section_summary(sections_by_document),
            "test_split_loaded": False,
        },
        "dense_channel_preflight": dense_summary,
        "channel_catalog": channel_catalog,
        **report_payload,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks, split_reports=split_reports),
        "timing_seconds": {
            "load_protocols": round(loaded_protocols_at - started_at, 3),
            "load_splits_and_build_train_folds": round(
                loaded_splits_at - loaded_protocols_at,
                3,
            ),
            "load_documents_sections": round(
                loaded_documents_at - loaded_splits_at,
                3,
            ),
            "dense_preflight": round(dense_preflight_at - loaded_documents_at, 3),
            "build_indexes": round(indexed_at - dense_preflight_at, 3),
            "build_candidate_pools": round(pools_built_at - indexed_at, 3),
            "run_control_and_agent_traces": round(evaluated_at - pools_built_at, 3),
            "summarize_and_guard": round(checked_at - evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def create_stage116_control_runner(
    *,
    evidence_selector_name: str = _DEFAULT_EVIDENCE_SELECTOR,
    max_candidates_per_document: int = _DEFAULT_MAX_CANDIDATES_PER_DOCUMENT,
    composition_policy_name: str = _DEFAULT_COMPOSITION_POLICY,
    max_sentences: int = _DEFAULT_MAX_SENTENCES,
    min_sentence_score: float = _DEFAULT_MIN_SENTENCE_SCORE,
    min_evidence_score: float = _DEFAULT_MIN_EVIDENCE_SCORE,
) -> Stage116ControlRunner:
    """Build the Stage116 control with the frozen answer defaults."""

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
    return Stage116ControlRunner(answer_generator=generator, answer_verifier=verifier)


def create_sidecar_agent_validation_harness(
    *,
    evidence_selector_name: str = _DEFAULT_EVIDENCE_SELECTOR,
    max_candidates_per_document: int = _DEFAULT_MAX_CANDIDATES_PER_DOCUMENT,
    composition_policy_name: str = _DEFAULT_COMPOSITION_POLICY,
    max_sentences: int = _DEFAULT_MAX_SENTENCES,
    min_sentence_score: float = _DEFAULT_MIN_SENTENCE_SCORE,
    min_evidence_score: float = _DEFAULT_MIN_EVIDENCE_SCORE,
) -> PrimeQAHybridSidecarAgentValidationHarness:
    """Build the fixed Stage136 orchestrator with validation recorders."""

    generator = RecordingAnswerGenerator(
        _answer_generator(
            evidence_selector_name=evidence_selector_name,
            max_candidates_per_document=max_candidates_per_document,
            composition_policy_name=composition_policy_name,
            max_sentences=max_sentences,
            min_sentence_score=min_sentence_score,
        )
    )
    verifier = RecordingAnswerVerifier(
        AnswerVerifier(
            min_citations=1,
            min_evidence_score=min_evidence_score,
            max_citation_rank=_BASELINE_PREFIX_DEPTH,
        )
    )
    orchestrator = PrimeQAHybridSidecarAgentOrchestrator(
        observation_adapter=PrimeQAHybridSidecarObservationAdapter(),
        answer_generator=generator,
        answer_verifier=verifier,
        consumer_policy=SidecarAgentConsumerPolicy(),
    )
    return PrimeQAHybridSidecarAgentValidationHarness(
        orchestrator=orchestrator,
        recording_generator=generator,
        recording_verifier=verifier,
    )


def _trace_agent_validation(
    *,
    sample: PrimeQAHybridSplitSample,
    pool: Mapping[str, Any],
    documents_by_id: Mapping[str, PrimeQADocument],
    control_runner: Stage116ControlRunner,
    agent_harness: PrimeQAHybridSidecarAgentValidationHarness,
) -> _AgentValidationTrace:
    prefix_results = _pool_results(pool["prefix_pool"], documents_by_id)
    stage128_results = _pool_results(pool["stage128_pool"], documents_by_id)
    question = sample.to_primeqa_question()
    control = control_runner.run(question=question, prefix_results=prefix_results)
    harness_run = agent_harness.run(
        question=question,
        candidate_pool_results=stage128_results,
    )
    agent = harness_run.agent_run
    try:
        public_trace = agent.public_safe_trace.to_public_dict()
        trace_serialization_violation = False
    except ValueError:
        public_trace = {}
        trace_serialization_violation = True

    sidecar_handles = {
        record.runtime_content_handle for record in agent.observation_bundle.sidecar_observations
    }
    primary_handles = {
        result.document.id for result in agent.observation_bundle.answer_context_results
    }
    generation_handles = {result.document.id for result in harness_run.generation_context_results}
    verification_handles = {
        result.document.id for result in harness_run.verification_context_results
    }
    append_handles = {
        result.document.id
        for result in stage128_results
        if _SIDECAR_APPEND_START_RANK <= result.rank <= _TARGET_POOL_DEPTH
    }
    gold_handle = sample.answer_doc_id if sample.answerable else None
    public_contract_violations = _public_trace_contract_violation_count(
        public_trace=public_trace,
        generation_context=harness_run.generation_context_results,
        verification_context=harness_run.verification_context_results,
        sidecar_observation_count=len(agent.observation_bundle.sidecar_observations),
    )
    return _AgentValidationTrace(
        sample=sample,
        question=question,
        control_original_answer=control.original_answer,
        control_verified_answer=control.verified_answer,
        agent_original_answer=agent.original_answer,
        agent_verified_answer=agent.verified_answer,
        generation_context_identity_violation=(
            _result_signature(control.answer_context_results)
            != _result_signature(harness_run.generation_context_results)
        ),
        verification_context_identity_violation=(
            _result_signature(control.verification_context_results)
            != _result_signature(harness_run.verification_context_results)
        ),
        bundle_generation_context_identity_violation=(
            _result_signature(agent.observation_bundle.answer_context_results)
            != _result_signature(harness_run.generation_context_results)
        ),
        original_answer_identity_violation=(
            _answer_signature(control.original_answer) != _answer_signature(agent.original_answer)
        ),
        verified_answer_identity_violation=(
            _answer_signature(control.verified_answer) != _answer_signature(agent.verified_answer)
        ),
        verification_reason_identity_violation=(
            tuple(control.verification_result.reasons) != tuple(agent.verification_result.reasons)
        ),
        sidecar_generation_leak_count=len(sidecar_handles & generation_handles),
        sidecar_verification_leak_count=len(sidecar_handles & verification_handles),
        sidecar_primary_overlap_count=len(sidecar_handles & primary_handles),
        public_trace_serialization_violation=trace_serialization_violation,
        public_trace_forbidden_key_count=len(_forbidden_keys_found(public_trace)),
        public_trace_contract_violation_count=public_contract_violations,
        sidecar_observation_count=len(agent.observation_bundle.sidecar_observations),
        sidecar_query_overlap_signal_count=(
            agent.public_safe_trace.sidecar_query_overlap_signal_count
        ),
        sidecar_novel_query_coverage_signal_count=(
            agent.public_safe_trace.sidecar_novel_query_coverage_signal_count
        ),
        evidence_gap_trace_status=agent.public_safe_trace.evidence_gap_trace_status,
        primary_context_gold_hit=bool(gold_handle and gold_handle in primary_handles),
        append_pool_gold_hit=bool(gold_handle and gold_handle in append_handles),
        sidecar_gold_hit=bool(gold_handle and gold_handle in sidecar_handles),
    )


def _public_trace_contract_violation_count(
    *,
    public_trace: Mapping[str, Any],
    generation_context: Sequence[RetrievalResult],
    verification_context: Sequence[RetrievalResult],
    sidecar_observation_count: int,
) -> int:
    checks = (
        public_trace.get("primary_context_count") == len(generation_context),
        public_trace.get("verification_context_count") == len(verification_context),
        public_trace.get("sidecar_observation_count") == sidecar_observation_count,
        public_trace.get("sidecar_used_for_answer_generation") is False,
        public_trace.get("sidecar_used_for_answer_verification") is False,
        public_trace.get("sidecar_replaced_primary_context") is False,
        public_trace.get("runtime_gold_labels_read") is False,
        public_trace.get("test_membership_read") is False,
    )
    return sum(not passed for passed in checks)


def _summarize_agent_traces(traces: Sequence[_AgentValidationTrace]) -> dict[str, Any]:
    questions = [trace.question for trace in traces]
    answerable = [trace for trace in traces if trace.sample.answerable]
    control_original = evaluate_answers(
        questions,
        [trace.control_original_answer for trace in traces],
    )
    control_verified = evaluate_answers(
        questions,
        [trace.control_verified_answer for trace in traces],
    )
    agent_original = evaluate_answers(
        questions,
        [trace.agent_original_answer for trace in traces],
    )
    agent_verified = evaluate_answers(
        questions,
        [trace.agent_verified_answer for trace in traces],
    )
    observation_count = sum(trace.sidecar_observation_count for trace in traces)
    rows_with_observations = sum(trace.sidecar_observation_count > 0 for trace in traces)
    rows_with_query_overlap = sum(trace.sidecar_query_overlap_signal_count > 0 for trace in traces)
    rows_with_novel_coverage = sum(
        trace.sidecar_novel_query_coverage_signal_count > 0 for trace in traces
    )
    primary_hits = sum(trace.primary_context_gold_hit for trace in answerable)
    append_hits = sum(trace.append_pool_gold_hit for trace in answerable)
    sidecar_hits = sum(trace.sidecar_gold_hit for trace in answerable)
    append_incremental = sum(
        trace.append_pool_gold_hit and not trace.primary_context_gold_hit for trace in answerable
    )
    sidecar_incremental = sum(
        trace.sidecar_gold_hit and not trace.primary_context_gold_hit for trace in answerable
    )
    return {
        "row_count": len(traces),
        "answerable_count": len(answerable),
        "control_original_metrics": asdict(control_original),
        "agent_original_metrics": asdict(agent_original),
        "original_metric_deltas_vs_stage116": _metric_deltas(
            agent=asdict(agent_original),
            control=asdict(control_original),
        ),
        "control_verified_metrics": asdict(control_verified),
        "agent_verified_metrics": asdict(agent_verified),
        "verified_metric_deltas_vs_stage116": _metric_deltas(
            agent=asdict(agent_verified),
            control=asdict(control_verified),
        ),
        "control_verified_gold_citation_count": _gold_citation_count(
            traces,
            answer_attr="control_verified_answer",
        ),
        "agent_verified_gold_citation_count": _gold_citation_count(
            traces,
            answer_attr="agent_verified_answer",
        ),
        "generation_context_identity_violation_count": sum(
            trace.generation_context_identity_violation for trace in traces
        ),
        "verification_context_identity_violation_count": sum(
            trace.verification_context_identity_violation for trace in traces
        ),
        "bundle_generation_context_identity_violation_count": sum(
            trace.bundle_generation_context_identity_violation for trace in traces
        ),
        "original_answer_identity_violation_count": sum(
            trace.original_answer_identity_violation for trace in traces
        ),
        "verified_answer_identity_violation_count": sum(
            trace.verified_answer_identity_violation for trace in traces
        ),
        "verification_reason_identity_violation_count": sum(
            trace.verification_reason_identity_violation for trace in traces
        ),
        "sidecar_generation_leak_count": sum(
            trace.sidecar_generation_leak_count for trace in traces
        ),
        "sidecar_verification_leak_count": sum(
            trace.sidecar_verification_leak_count for trace in traces
        ),
        "sidecar_primary_overlap_count": sum(
            trace.sidecar_primary_overlap_count for trace in traces
        ),
        "public_trace_serialization_violation_count": sum(
            trace.public_trace_serialization_violation for trace in traces
        ),
        "public_trace_forbidden_key_count": sum(
            trace.public_trace_forbidden_key_count for trace in traces
        ),
        "public_trace_contract_violation_count": sum(
            trace.public_trace_contract_violation_count for trace in traces
        ),
        "rows_with_sidecar_observations": rows_with_observations,
        "sidecar_observation_count": observation_count,
        "sidecar_observation_availability_rate": _rounded_ratio(
            rows_with_observations,
            len(traces),
        ),
        "sidecar_query_overlap_signal_count": sum(
            trace.sidecar_query_overlap_signal_count for trace in traces
        ),
        "row_query_overlap_signal_coverage": _rounded_ratio(
            rows_with_query_overlap,
            len(traces),
        ),
        "sidecar_novel_query_coverage_signal_count": sum(
            trace.sidecar_novel_query_coverage_signal_count for trace in traces
        ),
        "row_novel_query_coverage_signal_coverage": _rounded_ratio(
            rows_with_novel_coverage,
            len(traces),
        ),
        "evidence_gap_trace_status_counts": dict(
            sorted(Counter(trace.evidence_gap_trace_status for trace in traces).items())
        ),
        "primary_context_gold_hit_count": primary_hits,
        "append_pool_gold_hit_count": append_hits,
        "append_pool_incremental_gold_hit_count": append_incremental,
        "sidecar_gold_hit_count": sidecar_hits,
        "sidecar_incremental_gold_hit_count": sidecar_incremental,
        "sidecar_capture_rate_of_append_gold_opportunities": _rounded_ratio(
            sidecar_incremental,
            append_incremental,
        ),
    }


def _metric_deltas(
    *,
    agent: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: round(float(agent[key]) - float(control[key]), 4)
        for key in (
            "gold_doc_citation_rate",
            "answerable_refusal_rate",
            "unanswerable_refusal_rate",
            "average_token_f1",
        )
    }


def _gold_citation_count(
    traces: Sequence[_AgentValidationTrace],
    *,
    answer_attr: str,
) -> int:
    count = 0
    for trace in traces:
        if not trace.question.answerable:
            continue
        answer = getattr(trace, answer_attr)
        cited = {citation.document_id for citation in answer.citations}
        count += trace.question.answer_doc_id in cited
    return count


def _train_fold_reports(
    *,
    traces: Sequence[_AgentValidationTrace],
    fold_assignments: Mapping[str, str],
) -> list[dict[str, Any]]:
    reports = []
    for fold_id in sorted(set(fold_assignments.values())):
        fold_traces = [
            trace for trace in traces if fold_assignments[trace.sample.sample_id] == fold_id
        ]
        summary = _summarize_agent_traces(fold_traces)
        reports.append(
            {
                "fold_id": fold_id,
                "row_count": summary["row_count"],
                "generation_context_identity_violation_count": summary[
                    "generation_context_identity_violation_count"
                ],
                "verification_context_identity_violation_count": summary[
                    "verification_context_identity_violation_count"
                ],
                "original_answer_identity_violation_count": summary[
                    "original_answer_identity_violation_count"
                ],
                "verified_answer_identity_violation_count": summary[
                    "verified_answer_identity_violation_count"
                ],
                "sidecar_answer_path_leak_count": int(summary["sidecar_generation_leak_count"])
                + int(summary["sidecar_verification_leak_count"]),
                "public_trace_violation_count": int(
                    summary["public_trace_serialization_violation_count"]
                )
                + int(summary["public_trace_forbidden_key_count"])
                + int(summary["public_trace_contract_violation_count"]),
                "sidecar_observation_availability_rate": summary[
                    "sidecar_observation_availability_rate"
                ],
                "row_query_overlap_signal_coverage": summary["row_query_overlap_signal_coverage"],
                "row_novel_query_coverage_signal_coverage": summary[
                    "row_novel_query_coverage_signal_coverage"
                ],
                "append_pool_incremental_gold_hit_count": summary[
                    "append_pool_incremental_gold_hit_count"
                ],
                "sidecar_incremental_gold_hit_count": summary["sidecar_incremental_gold_hit_count"],
                "verified_average_token_f1_delta_vs_stage116": summary[
                    "verified_metric_deltas_vs_stage116"
                ]["average_token_f1"],
                "verified_gold_citation_count_delta_vs_stage116": int(
                    summary["agent_verified_gold_citation_count"]
                )
                - int(summary["control_verified_gold_citation_count"]),
            }
        )
    return reports


def _train_cv_validation(
    *,
    train_summary: Mapping[str, Any],
    fold_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checks = {
        "all_folds_generation_context_identity_preserved": all(
            int(fold["generation_context_identity_violation_count"]) == 0 for fold in fold_reports
        ),
        "all_folds_verification_context_identity_preserved": all(
            int(fold["verification_context_identity_violation_count"]) == 0 for fold in fold_reports
        ),
        "all_folds_original_answers_identical": all(
            int(fold["original_answer_identity_violation_count"]) == 0 for fold in fold_reports
        ),
        "all_folds_verified_answers_identical": all(
            int(fold["verified_answer_identity_violation_count"]) == 0 for fold in fold_reports
        ),
        "all_folds_sidecar_isolated": all(
            int(fold["sidecar_answer_path_leak_count"]) == 0 for fold in fold_reports
        ),
        "all_folds_public_trace_valid": all(
            int(fold["public_trace_violation_count"]) == 0 for fold in fold_reports
        ),
        "all_folds_sidecar_observations_available": all(
            float(fold["sidecar_observation_availability_rate"]) == 1.0 for fold in fold_reports
        ),
        "aggregate_verified_f1_delta_is_zero": float(
            train_summary["verified_metric_deltas_vs_stage116"]["average_token_f1"]
        )
        == 0.0,
        "aggregate_gold_citation_delta_is_zero": int(
            train_summary["agent_verified_gold_citation_count"]
        )
        - int(train_summary["control_verified_gold_citation_count"])
        == 0,
    }
    return {
        "selection_split": _TRAIN_SPLIT,
        "selection_mode": "fixed_orchestrator_train_grouped_cross_validation_integrity",
        "candidate_selection_performed": False,
        "threshold_tuning_performed": False,
        "fold_count": len(fold_reports),
        "fold_row_counts": [int(fold["row_count"]) for fold in fold_reports],
        "minimum_fold_sidecar_observation_availability_rate": min(
            (float(fold["sidecar_observation_availability_rate"]) for fold in fold_reports),
            default=0.0,
        ),
        "mean_fold_query_overlap_signal_coverage": _rounded_mean(
            [float(fold["row_query_overlap_signal_coverage"]) for fold in fold_reports]
        ),
        "mean_fold_novel_query_coverage_signal_coverage": _rounded_mean(
            [float(fold["row_novel_query_coverage_signal_coverage"]) for fold in fold_reports]
        ),
        "aggregate_train_summary": dict(train_summary),
        "checks": checks,
        "passed": all(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "train_cv_group_values_written": False,
    }


def _dev_report(dev_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "validation_split": _DEV_SPLIT,
        "status": "reported_not_used_for_selection_or_retuning",
        "dev_used_for_selection": False,
        "dev_used_for_retuning": False,
        "agent_summary": dict(dev_summary),
        "dev_gate_status": "report_only_no_runtime_default_or_test_gate",
    }


def _stage136_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision") or {}
    guards = report.get("guard_checks") or []
    frozen = report.get("frozen_protocol") or {}
    stage137 = frozen.get("stage137_validation_plan") or {}
    public = report.get("public_safe_contract") or {}
    return {
        "stage": report.get("stage"),
        "protocol_id": report.get("protocol_id"),
        "status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "can_run_stage137_train_dev_validation_now": decision.get(
            "can_run_stage137_train_dev_validation_now"
        ),
        "guard_check_count": len(guards),
        "guard_check_passed_count": sum(bool(check.get("passed")) for check in guards),
        "minimum_train_folds": stage137.get("minimum_train_folds"),
        "dev_mode": stage137.get("dev_mode"),
        "sidecar_effectiveness_status": decision.get("sidecar_effectiveness_status"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "runtime_defaultization_allowed_now": decision.get("runtime_defaultization_allowed_now"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "public_safe_forbidden_keys_found": public.get("forbidden_keys_found") or [],
    }


def _stage135_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision") or {}
    guards = report.get("guard_checks") or []
    splits = report.get("split_observation_reports") or {}
    public = report.get("public_safe_contract") or {}
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "status": decision.get("status"),
        "guard_check_count": len(guards),
        "guard_check_passed_count": sum(bool(check.get("passed")) for check in guards),
        "train": _stage135_split_summary(splits.get(_TRAIN_SPLIT) or {}),
        "dev": _stage135_split_summary(splits.get(_DEV_SPLIT) or {}),
        "public_safe_forbidden_keys_found": public.get("forbidden_keys_found") or [],
    }


def _stage135_split_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "row_count": summary.get("row_count"),
        "append_pool_incremental_gold_hit_count": summary.get(
            "append_pool_incremental_gold_hit_count"
        ),
        "sidecar_incremental_gold_hit_count": summary.get("sidecar_incremental_gold_hit_count"),
        "primary_context_identity_violation_count": summary.get(
            "primary_context_identity_violation_count"
        ),
        "sidecar_answer_context_leak_count": summary.get("sidecar_answer_context_leak_count"),
    }


def _stage128_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision") or {}
    frozen = report.get("frozen_protocol") or {}
    selected = frozen.get("selected_retrieval_config") or {}
    return {
        "stage": report.get("stage"),
        "selected_config_id": decision.get("selected_config_id") or selected.get("config_id"),
        "selected_family_id": decision.get("selected_family_id") or selected.get("family_id"),
    }


def _pre_validation_guard_checks(
    *,
    stage136_summary: Mapping[str, Any],
    stage135_summary: Mapping[str, Any],
    selected_config: Mapping[str, Any] | None,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    train_fold_count: int,
    user_confirmed_validation: bool,
    confirmation_note: str,
    include_dense_channels: bool,
    dense_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _check(
            name="user_confirmed_stage137_validation",
            passed=user_confirmed_validation and "Stage137" in confirmation_note,
            observed=confirmation_note,
            expected="user confirmed Stage137 validation",
        ),
        _check(
            name="stage136_protocol_frozen",
            passed=stage136_summary.get("status") == _SOURCE_STAGE136_STATUS,
            observed=stage136_summary.get("status"),
            expected=_SOURCE_STAGE136_STATUS,
        ),
        _check(
            name="stage136_protocol_id_matches",
            passed=stage136_summary.get("protocol_id") == _SOURCE_STAGE136_PROTOCOL_ID,
            observed=stage136_summary.get("protocol_id"),
            expected=_SOURCE_STAGE136_PROTOCOL_ID,
        ),
        _check(
            name="stage136_recommends_stage137_validation",
            passed=stage136_summary.get("recommended_next_direction") == _SOURCE_STAGE136_NEXT,
            observed=stage136_summary.get("recommended_next_direction"),
            expected=_SOURCE_STAGE136_NEXT,
        ),
        _check(
            name="stage136_allows_stage137_validation",
            passed=stage136_summary.get("can_run_stage137_train_dev_validation_now") is True,
            observed=stage136_summary.get("can_run_stage137_train_dev_validation_now"),
            expected=True,
        ),
        _check(
            name="stage136_all_guard_checks_passed",
            passed=stage136_summary.get("guard_check_count") == 21
            and stage136_summary.get("guard_check_passed_count") == 21,
            observed={
                "passed": stage136_summary.get("guard_check_passed_count"),
                "total": stage136_summary.get("guard_check_count"),
            },
            expected={"passed": 21, "total": 21},
        ),
        _check(
            name="stage135_validation_source_available",
            passed=stage135_summary.get("status") == _SOURCE_STAGE135_STATUS
            and stage135_summary.get("analysis_id") == _SOURCE_STAGE135_ANALYSIS_ID
            and stage135_summary.get("guard_check_count") == 30
            and stage135_summary.get("guard_check_passed_count") == 30,
            observed={
                "status": stage135_summary.get("status"),
                "analysis_id": stage135_summary.get("analysis_id"),
                "guards": {
                    "passed": stage135_summary.get("guard_check_passed_count"),
                    "total": stage135_summary.get("guard_check_count"),
                },
            },
            expected="Stage135 validation passed with 30/30 guards",
        ),
        _check(
            name="stage135_negative_effectiveness_boundary_available",
            passed=_source_boundary_matches(stage135_summary, train=(9, 0), dev=(1, 0)),
            observed={
                "train": stage135_summary.get("train"),
                "dev": stage135_summary.get("dev"),
            },
            expected={
                "train": {"opportunities": 9, "captures": 0},
                "dev": {"opportunities": 1, "captures": 0},
            },
        ),
        _check(
            name="stage128_candidate_pool_config_available",
            passed=selected_config is not None
            and selected_config.get("config_id") == _SELECTED_CONFIG_ID,
            observed=None if selected_config is None else selected_config.get("config_id"),
            expected=_SELECTED_CONFIG_ID,
        ),
        _check(
            name="only_train_dev_splits_loaded",
            passed=tuple(split_samples) == _ALLOWED_DEVELOPMENT_SPLITS,
            observed=list(split_samples),
            expected=list(_ALLOWED_DEVELOPMENT_SPLITS),
        ),
        _check(
            name="loaded_samples_are_train_dev_only",
            passed=all(
                sample.assigned_split == split
                for split, samples in split_samples.items()
                for sample in samples
            ),
            observed={
                split: dict(Counter(sample.assigned_split for sample in samples))
                for split, samples in split_samples.items()
            },
            expected={split: split for split in _ALLOWED_DEVELOPMENT_SPLITS},
        ),
        _check(
            name="split_row_counts_match_stage135_source",
            passed=all(
                len(split_samples[split])
                == int((stage135_summary.get(split) or {}).get("row_count") or -1)
                for split in _ALLOWED_DEVELOPMENT_SPLITS
            ),
            observed={split: len(split_samples[split]) for split in split_samples},
            expected={
                split: (stage135_summary.get(split) or {}).get("row_count")
                for split in _ALLOWED_DEVELOPMENT_SPLITS
            },
        ),
        _check(
            name="train_fold_count_matches_frozen_protocol",
            passed=train_fold_count >= _MINIMUM_TRAIN_FOLDS
            and train_fold_count >= int(stage136_summary.get("minimum_train_folds") or 0),
            observed=train_fold_count,
            expected=f">= {_MINIMUM_TRAIN_FOLDS}",
        ),
        _check(
            name="dense_channels_use_existing_cache_only",
            passed=(not include_dense_channels)
            or (
                bool(dense_summary.get("can_run_without_download"))
                and bool(dense_summary.get("no_model_download_attempted"))
            ),
            observed=dense_summary,
            expected="existing local cache only; no model download",
        ),
        _check(
            name="source_public_contracts_have_no_forbidden_keys",
            passed=stage136_summary.get("public_safe_forbidden_keys_found") == []
            and stage135_summary.get("public_safe_forbidden_keys_found") == [],
            observed={
                "stage136": stage136_summary.get("public_safe_forbidden_keys_found"),
                "stage135": stage135_summary.get("public_safe_forbidden_keys_found"),
            },
            expected=[],
        ),
    ]


def _post_validation_guard_checks(
    *,
    stage136_summary: Mapping[str, Any],
    stage135_summary: Mapping[str, Any],
    pool_summary: Mapping[str, Any],
    report_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    train = report_payload["split_agent_reports"][_TRAIN_SPLIT]
    dev = report_payload["split_agent_reports"][_DEV_SPLIT]
    train_cv = report_payload["train_cv_validation"]
    dev_report = report_payload["dev_report"]
    public_payload = {
        "candidate_pool_summary": pool_summary,
        "split_agent_reports": report_payload["split_agent_reports"],
        "train_fold_reports": report_payload["train_fold_reports"],
        "train_cv_validation": train_cv,
        "dev_report": dev_report,
    }

    def total(key: str) -> int:
        return int(train[key]) + int(dev[key])

    return [
        _check(
            name="stage137_candidate_pool_prefix_identity_preserved",
            passed=pool_summary["all_splits_prefix_identity_violation_count"] == 0,
            observed=pool_summary["all_splits_prefix_identity_violation_count"],
            expected=0,
        ),
        _check(
            name="stage137_candidate_pool_append_budget_preserved",
            passed=pool_summary["all_splits_append_budget_exceeded_count"] == 0,
            observed=pool_summary["all_splits_append_budget_exceeded_count"],
            expected=0,
        ),
        _check(
            name="stage137_generation_context_identity_preserved",
            passed=total("generation_context_identity_violation_count") == 0
            and total("bundle_generation_context_identity_violation_count") == 0,
            observed={
                "control_vs_dependency": total("generation_context_identity_violation_count"),
                "bundle_vs_dependency": total("bundle_generation_context_identity_violation_count"),
            },
            expected={"control_vs_dependency": 0, "bundle_vs_dependency": 0},
        ),
        _check(
            name="stage137_verification_context_identity_preserved",
            passed=total("verification_context_identity_violation_count") == 0,
            observed=total("verification_context_identity_violation_count"),
            expected=0,
        ),
        _check(
            name="stage137_original_answers_identical_to_stage116",
            passed=total("original_answer_identity_violation_count") == 0,
            observed=total("original_answer_identity_violation_count"),
            expected=0,
        ),
        _check(
            name="stage137_verified_answers_identical_to_stage116",
            passed=total("verified_answer_identity_violation_count") == 0,
            observed=total("verified_answer_identity_violation_count"),
            expected=0,
        ),
        _check(
            name="stage137_verification_reasons_identical_to_stage116",
            passed=total("verification_reason_identity_violation_count") == 0,
            observed=total("verification_reason_identity_violation_count"),
            expected=0,
        ),
        _check(
            name="stage137_sidecar_isolated_from_generation_and_verification",
            passed=total("sidecar_generation_leak_count") == 0
            and total("sidecar_verification_leak_count") == 0
            and total("sidecar_primary_overlap_count") == 0,
            observed={
                "generation": total("sidecar_generation_leak_count"),
                "verification": total("sidecar_verification_leak_count"),
                "primary_overlap": total("sidecar_primary_overlap_count"),
            },
            expected={"generation": 0, "verification": 0, "primary_overlap": 0},
        ),
        _check(
            name="stage137_public_traces_serialize_safely",
            passed=total("public_trace_serialization_violation_count") == 0
            and total("public_trace_forbidden_key_count") == 0
            and total("public_trace_contract_violation_count") == 0,
            observed={
                "serialization": total("public_trace_serialization_violation_count"),
                "forbidden_keys": total("public_trace_forbidden_key_count"),
                "contract": total("public_trace_contract_violation_count"),
            },
            expected={"serialization": 0, "forbidden_keys": 0, "contract": 0},
        ),
        _check(
            name="stage137_sidecar_observations_available_on_all_rows",
            passed=float(train["sidecar_observation_availability_rate"]) == 1.0
            and float(dev["sidecar_observation_availability_rate"]) == 1.0,
            observed={
                "train": train["sidecar_observation_availability_rate"],
                "dev": dev["sidecar_observation_availability_rate"],
            },
            expected={"train": 1.0, "dev": 1.0},
        ),
        _check(
            name="stage137_answer_metric_deltas_are_zero",
            passed=all(
                float(delta) == 0.0
                for split in (train, dev)
                for metric_group in (
                    split["original_metric_deltas_vs_stage116"],
                    split["verified_metric_deltas_vs_stage116"],
                )
                for delta in metric_group.values()
            ),
            observed={
                "train": train["verified_metric_deltas_vs_stage116"],
                "dev": dev["verified_metric_deltas_vs_stage116"],
            },
            expected="all zero",
        ),
        _check(
            name="stage137_gold_citation_count_deltas_are_zero",
            passed=all(
                int(split["agent_verified_gold_citation_count"])
                == int(split["control_verified_gold_citation_count"])
                for split in (train, dev)
            ),
            observed={
                name: int(split["agent_verified_gold_citation_count"])
                - int(split["control_verified_gold_citation_count"])
                for name, split in ((_TRAIN_SPLIT, train), (_DEV_SPLIT, dev))
            },
            expected={_TRAIN_SPLIT: 0, _DEV_SPLIT: 0},
        ),
        _check(
            name="stage137_train_grouped_cv_integrity_passed",
            passed=bool(train_cv.get("passed")),
            observed=train_cv.get("checks"),
            expected="all train grouped-CV checks pass",
        ),
        _check(
            name="stage137_dev_report_only",
            passed=dev_report.get("dev_used_for_selection") is False
            and dev_report.get("dev_used_for_retuning") is False,
            observed=dev_report.get("status"),
            expected="dev report only",
        ),
        _check(
            name="stage137_train_sidecar_boundary_matches_stage135",
            passed=_split_boundary_matches_source(train, stage135_summary.get("train") or {}),
            observed={
                "opportunities": train["append_pool_incremental_gold_hit_count"],
                "captures": train["sidecar_incremental_gold_hit_count"],
            },
            expected=stage135_summary.get("train"),
        ),
        _check(
            name="stage137_dev_sidecar_boundary_matches_stage135",
            passed=_split_boundary_matches_source(dev, stage135_summary.get("dev") or {}),
            observed={
                "opportunities": dev["append_pool_incremental_gold_hit_count"],
                "captures": dev["sidecar_incremental_gold_hit_count"],
            },
            expected=stage135_summary.get("dev"),
        ),
        _check(
            name="stage137_test_locked",
            passed=stage136_summary.get("can_run_final_test_metrics_now") is False
            and stage136_summary.get("can_use_test_for_tuning") is False,
            observed={
                "metrics": stage136_summary.get("can_run_final_test_metrics_now"),
                "tuning": stage136_summary.get("can_use_test_for_tuning"),
            },
            expected="test locked",
        ),
        _check(
            name="stage137_runtime_defaults_unchanged",
            passed=stage136_summary.get("runtime_defaultization_allowed_now") is False
            and stage136_summary.get("default_runtime_policy") == "unchanged",
            observed={
                "allowed": stage136_summary.get("runtime_defaultization_allowed_now"),
                "policy": stage136_summary.get("default_runtime_policy"),
            },
            expected="unchanged",
        ),
        _check(
            name="stage137_no_fallback_strategies",
            passed=stage136_summary.get("fallback_strategies_enabled") is False,
            observed=stage136_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="stage137_public_outputs_have_no_forbidden_keys",
            passed=not _forbidden_keys_found(public_payload),
            observed=sorted(_forbidden_keys_found(public_payload)),
            expected=[],
        ),
        _check(
            name="stage137_train_cv_group_values_not_written",
            passed=train_cv.get("train_cv_group_values_written") is False,
            observed=train_cv.get("train_cv_group_values_written"),
            expected=False,
        ),
    ]


def _source_boundary_matches(
    summary: Mapping[str, Any],
    *,
    train: tuple[int, int],
    dev: tuple[int, int],
) -> bool:
    return (
        _numeric_boundary(summary.get("train") or {}) == train
        and _numeric_boundary(summary.get("dev") or {}) == dev
    )


def _split_boundary_matches_source(
    actual: Mapping[str, Any],
    source: Mapping[str, Any],
) -> bool:
    return _numeric_boundary(actual) == _numeric_boundary(source)


def _numeric_boundary(summary: Mapping[str, Any]) -> tuple[int, int] | None:
    opportunity = summary.get("append_pool_incremental_gold_hit_count")
    capture = summary.get("sidecar_incremental_gold_hit_count")
    if type(opportunity) not in (int, float) or type(capture) not in (int, float):
        return None
    return int(opportunity), int(capture)


def _decision(
    guard_checks: Sequence[Mapping[str, Any]],
    *,
    split_reports: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    train = (split_reports or {}).get(_TRAIN_SPLIT) or {}
    dev = (split_reports or {}).get(_DEV_SPLIT) or {}
    captures = int(train.get("sidecar_incremental_gold_hit_count") or 0) + int(
        dev.get("sidecar_incremental_gold_hit_count") or 0
    )
    base = {
        "analysis_id": _ANALYSIS_ID,
        "validated_protocol_id": _SOURCE_STAGE136_PROTOCOL_ID,
        "sidecar_effectiveness_status": (
            "diagnostic_capture_observed" if captures > 0 else "safe_but_neutral"
        ),
        "sidecar_citation_verification_effectiveness_demonstrated": captures > 0,
        "can_claim_answer_quality_improvement": False,
        "can_claim_retrieval_improvement": False,
        "sidecar_can_generate_answer_text": False,
        "sidecar_can_enter_answer_verification_context": False,
        "sidecar_can_replace_primary_context": False,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "runtime_defaultization_allowed_now": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }
    if failed_checks:
        return {
            **base,
            "status": "primeqa_hybrid_sidecar_agent_orchestrator_validation_blocked_or_failed",
            "failed_checks": failed_checks,
            "agent_orchestrator_integration_validated": False,
            "can_freeze_optional_agent_entrypoint_protocol_now": False,
            "recommended_next_direction": "review_stage137_agent_orchestrator_failures",
        }
    return {
        **base,
        "status": "primeqa_hybrid_sidecar_agent_orchestrator_train_cv_dev_validation_passed",
        "failed_checks": [],
        "agent_orchestrator_integration_validated": True,
        "can_freeze_optional_agent_entrypoint_protocol_now": True,
        "recommended_next_direction": "freeze_optional_sidecar_agent_entrypoint_protocol",
    }


def _validation_harness_contract() -> dict[str, Any]:
    return {
        "harness_id": "stage137_recording_control_vs_agent_harness_v1",
        "stage116_control_executed_per_row": True,
        "agent_orchestrator_executed_per_row": True,
        "generation_dependency_input_recorded_in_memory": True,
        "verification_dependency_input_recorded_in_memory": True,
        "exact_context_identity_compared": True,
        "original_answer_identity_compared": True,
        "verified_answer_identity_compared": True,
        "verification_reason_identity_compared": True,
        "public_trace_serialized_per_row_in_memory": True,
        "private_per_row_traces_written": False,
        "gold_used_only_for_offline_train_dev_aggregate_diagnostics": True,
    }


def _split_contract() -> dict[str, Any]:
    return {
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
        "selection_split": _TRAIN_SPLIT,
        "selection_mode": "fixed_orchestrator_grouped_cross_validation_integrity",
        "candidate_selection_performed": False,
        "threshold_tuning_performed": False,
        "validation_split": _DEV_SPLIT,
        "dev_selection_used": False,
        "dev_retuning_used": False,
        "forbidden_final_splits": ["test"],
    }


def _source_files(
    *,
    stage136_protocol_path: Path,
    stage135_validation_path: Path,
    stage128_protocol_path: Path,
    stage125_protocol_path: Path,
    stage80_report_path: Path | None,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
) -> dict[str, Any]:
    files = {
        "stage136_protocol": _fingerprint(stage136_protocol_path),
        "stage135_validation": _fingerprint(stage135_validation_path),
        "stage128_protocol": _fingerprint(stage128_protocol_path),
        "stage125_protocol": _fingerprint(stage125_protocol_path),
        "train_split": _fingerprint(train_split_path),
        "dev_split": _fingerprint(dev_split_path),
        "corpus_documents": _fingerprint(documents_path),
    }
    if stage80_report_path is not None:
        files["stage80_dense_cache_report"] = _fingerprint(stage80_report_path)
    return files


def _blocked_report(
    *,
    stage136_protocol_path: Path,
    stage135_validation_path: Path,
    stage128_protocol_path: Path,
    stage125_protocol_path: Path,
    stage80_report_path: Path | None,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    stage136_summary: Mapping[str, Any],
    stage135_summary: Mapping[str, Any],
    stage128_summary: Mapping[str, Any],
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    documents: Sequence[PrimeQADocument],
    sections_by_document: Mapping[str, Sequence[Any]],
    dense_summary: Mapping[str, Any],
    guard_checks: Sequence[Mapping[str, Any]],
    timing_seconds: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": "Stage137 blocked before candidate-pool and agent execution.",
        "split_contract": _split_contract(),
        "source_files": _source_files(
            stage136_protocol_path=stage136_protocol_path,
            stage135_validation_path=stage135_validation_path,
            stage128_protocol_path=stage128_protocol_path,
            stage125_protocol_path=stage125_protocol_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
        ),
        "stage136_summary": dict(stage136_summary),
        "stage135_summary": dict(stage135_summary),
        "stage128_summary": dict(stage128_summary),
        "orchestrator_contract": sidecar_agent_orchestrator_contract(),
        "validation_harness_contract": _validation_harness_contract(),
        "loaded_data_summary": {
            "split_samples": summarize_primeqa_hybrid_split_samples(split_samples),
            "document_count": len(documents),
            **_section_summary(sections_by_document),
            "test_split_loaded": False,
        },
        "dense_channel_preflight": dict(dense_summary),
        "channel_catalog": [],
        "candidate_pool_summary": {},
        "split_agent_reports": {},
        "train_fold_reports": [],
        "train_cv_validation": {
            "passed": False,
            "failed_checks": [str(check["name"]) for check in guard_checks if not check["passed"]],
            "train_cv_group_values_written": False,
        },
        "dev_report": {
            "status": "not_run_due_to_precheck_failure",
            "dev_used_for_selection": False,
            "dev_used_for_retuning": False,
        },
        "guard_checks": list(guard_checks),
        "decision": _decision(guard_checks),
        "timing_seconds": dict(timing_seconds),
    }


def _validate_options(
    *,
    train_fold_count: int,
    bm25_k1: float,
    bm25_b: float,
    encoder_batch_size: int,
    max_candidates_per_document: int,
    max_sentences: int,
    min_sentence_score: float,
    min_evidence_score: float,
) -> None:
    if train_fold_count <= 0:
        raise ValueError("train_fold_count must be positive")
    if bm25_k1 <= 0:
        raise ValueError("bm25_k1 must be positive")
    if not 0 <= bm25_b <= 1:
        raise ValueError("bm25_b must be between 0 and 1")
    if encoder_batch_size <= 0:
        raise ValueError("encoder_batch_size must be positive")
    if max_candidates_per_document <= 0:
        raise ValueError("max_candidates_per_document must be positive")
    if max_sentences <= 0:
        raise ValueError("max_sentences must be positive")
    if min_sentence_score < 0:
        raise ValueError("min_sentence_score must be non-negative")
    if min_evidence_score < 0:
        raise ValueError("min_evidence_score must be non-negative")


def _check(*, name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    runtime_traces_created = bool(report.get("split_agent_reports"))
    return {
        "public_safe_summary_only": True,
        "runtime_control_and_agent_traces_created_in_memory": runtime_traces_created,
        "runtime_control_and_agent_traces_written": False,
        "raw_question_text_written": False,
        "raw_answer_text_written": False,
        "raw_document_text_written": False,
        "raw_document_ids_written": False,
        "raw_runtime_content_handles_written": False,
        "raw_candidate_rows_written": False,
        "raw_sample_ids_written": False,
        "train_cv_group_values_written": False,
        "test_split_loaded": False,
        "final_test_metrics_run": False,
        "forbidden_keys_found": sorted(_forbidden_keys_found(report)),
    }


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


def write_primeqa_hybrid_sidecar_agent_orchestrator_validation_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridSidecarAgentOrchestratorValidationVisualization]:
    """Write public-safe SVG charts for Stage137."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage137_train_fold_identity_violations.svg": render_horizontal_bar_chart_svg(
            title="Stage137 train fold identity violations",
            bars=_train_fold_identity_bars(report),
            x_label="violation count",
            width=1900,
            margin_left=980,
        ),
        "stage137_split_answer_metric_deltas.svg": render_horizontal_bar_chart_svg(
            title="Stage137 agent answer metric deltas vs Stage116",
            bars=_answer_delta_bars(report),
            x_label="delta",
            width=1760,
            margin_left=900,
        ),
        "stage137_sidecar_isolation_violations.svg": render_horizontal_bar_chart_svg(
            title="Stage137 sidecar isolation violations",
            bars=_isolation_bars(report),
            x_label="violation count",
            width=1860,
            margin_left=960,
        ),
        "stage137_sidecar_opportunity_capture.svg": render_horizontal_bar_chart_svg(
            title="Stage137 sidecar opportunity boundary",
            bars=_opportunity_bars(report),
            x_label="answerable row count",
            width=1660,
            margin_left=860,
        ),
        "stage137_sidecar_signal_coverage.svg": render_horizontal_bar_chart_svg(
            title="Stage137 sidecar signal coverage",
            bars=_signal_bars(report),
            x_label="rate",
            width=1660,
            margin_left=860,
        ),
        "stage137_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage137 decision flags",
            bars=_decision_bars(report),
            x_label="1 means true",
            width=1900,
            margin_left=1020,
        ),
        "stage137_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage137 guard checks",
            bars=_guard_bars(report),
            x_label="1 means passed",
            width=2260,
            margin_left=1300,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridSidecarAgentOrchestratorValidationVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _train_fold_identity_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for fold in report.get("train_fold_reports") or []:
        for key, label in (
            ("generation_context_identity_violation_count", "generation context"),
            ("verification_context_identity_violation_count", "verification context"),
            ("original_answer_identity_violation_count", "original answer"),
            ("verified_answer_identity_violation_count", "verified answer"),
        ):
            bars.append(
                BarDatum(
                    label=f"{fold['fold_id']} {label}",
                    value=float(fold[key]),
                    value_label=str(fold[key]),
                )
            )
    return bars


def _answer_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for split, summary in (report.get("split_agent_reports") or {}).items():
        deltas = summary.get("verified_metric_deltas_vs_stage116") or {}
        for key, label in (
            ("average_token_f1", "verified F1"),
            ("gold_doc_citation_rate", "gold citation rate"),
            ("answerable_refusal_rate", "answerable refusal"),
            ("unanswerable_refusal_rate", "unanswerable refusal"),
        ):
            value = float(deltas.get(key) or 0.0)
            bars.append(
                BarDatum(
                    label=f"{split} {label}",
                    value=value,
                    value_label=f"{value:+.4f}",
                )
            )
    return bars


def _isolation_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for split, summary in (report.get("split_agent_reports") or {}).items():
        for key, label in (
            ("sidecar_generation_leak_count", "generation leak"),
            ("sidecar_verification_leak_count", "verification leak"),
            ("sidecar_primary_overlap_count", "primary overlap"),
            ("public_trace_contract_violation_count", "trace contract"),
        ):
            value = int(summary[key])
            bars.append(
                BarDatum(
                    label=f"{split} {label}",
                    value=float(value),
                    value_label=str(value),
                )
            )
    return bars


def _opportunity_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for split, summary in (report.get("split_agent_reports") or {}).items():
        for key, label in (
            ("append_pool_incremental_gold_hit_count", "append opportunities"),
            ("sidecar_incremental_gold_hit_count", "sidecar captures"),
        ):
            value = int(summary[key])
            bars.append(
                BarDatum(
                    label=f"{split} {label}",
                    value=float(value),
                    value_label=str(value),
                )
            )
    return bars


def _signal_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for split, summary in (report.get("split_agent_reports") or {}).items():
        for key, label in (
            ("sidecar_observation_availability_rate", "observation availability"),
            ("row_query_overlap_signal_coverage", "query overlap"),
            ("row_novel_query_coverage_signal_coverage", "novel query coverage"),
        ):
            value = float(summary[key])
            bars.append(
                BarDatum(
                    label=f"{split} {label}",
                    value=value,
                    value_label=f"{value:.2%}",
                )
            )
    return bars


def _decision_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    keys = (
        "agent_orchestrator_integration_validated",
        "sidecar_citation_verification_effectiveness_demonstrated",
        "can_freeze_optional_agent_entrypoint_protocol_now",
        "can_open_final_test_gate_now",
        "runtime_defaultization_allowed_now",
        "fallback_strategies_enabled",
    )
    return [
        BarDatum(
            label=key,
            value=1.0 if decision.get(key) else 0.0,
            value_label="true" if decision.get(key) else "false",
        )
        for key in keys
    ]


def _guard_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(check["name"]),
            value=1.0 if check["passed"] else 0.0,
            value_label="passed" if check["passed"] else "failed",
        )
        for check in report.get("guard_checks") or []
    ]
