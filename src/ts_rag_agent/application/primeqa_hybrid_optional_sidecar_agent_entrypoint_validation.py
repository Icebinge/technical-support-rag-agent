from __future__ import annotations

import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_agent_retrieval_integration_validation import (
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
    _answer_signature,
    _candidate_pool_summary,
    _candidate_pools_by_split,
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
    _rounded_ratio,
    _section_summary,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_entrypoint import (
    CandidatePoolRetrieverPort,
    OptionalSidecarAgentEntrypointRun,
    PrimeQAHybridOptionalSidecarAgentEntrypoint,
    create_primeqa_hybrid_optional_sidecar_agent_entrypoint,
    optional_sidecar_agent_entrypoint_contract,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_orchestrator_validation import (
    Stage116ControlRunner,
    _AgentValidationTrace,
    _dev_report,
    _forbidden_keys_found,
    _public_trace_contract_violation_count,
    _stage128_summary,
    _summarize_agent_traces,
    _train_cv_validation,
    _train_fold_reports,
    _validate_options,
    create_stage116_control_runner,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    _pool_results,
    _result_signature,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
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

_STAGE = "Stage 139"
_CREATED_AT = "2026-07-17"
_ANALYSIS_ID = "primeqa_hybrid_optional_sidecar_agent_entrypoint_train_cv_dev_validation_v1"
_SOURCE_STAGE138_STATUS = "primeqa_hybrid_optional_sidecar_agent_entrypoint_protocol_frozen"
_SOURCE_STAGE138_PROTOCOL_ID = "primeqa_hybrid_optional_sidecar_agent_entrypoint_protocol_v1"
_SOURCE_STAGE138_NEXT = (
    "implement_optional_sidecar_agent_entrypoint_and_train_dev_action_trace_validation"
)
_SOURCE_STAGE137_STATUS = "primeqa_hybrid_sidecar_agent_orchestrator_train_cv_dev_validation_passed"
_SOURCE_STAGE137_ANALYSIS_ID = (
    "primeqa_hybrid_sidecar_agent_orchestrator_train_cv_dev_validation_v1"
)
_ENTRYPOINT_ID = "stage138_optional_sidecar_agent_entrypoint_v1"
_ORCHESTRATOR_ID = "stage116_primary_plus_stage128_sidecar_agent_orchestrator_v1"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_MINIMUM_TRAIN_FOLDS = 5
_SIDECAR_APPEND_START_RANK = 201
_EXPECTED_SOURCE_GUARDS = 36
_EXPECTED_PROTOCOL_GUARDS = 31
_SOURCE_PARITY_KEYS = (
    "row_count",
    "answerable_count",
    "control_original_metrics",
    "agent_original_metrics",
    "original_metric_deltas_vs_stage116",
    "control_verified_metrics",
    "agent_verified_metrics",
    "verified_metric_deltas_vs_stage116",
    "control_verified_gold_citation_count",
    "agent_verified_gold_citation_count",
    "generation_context_identity_violation_count",
    "verification_context_identity_violation_count",
    "bundle_generation_context_identity_violation_count",
    "original_answer_identity_violation_count",
    "verified_answer_identity_violation_count",
    "verification_reason_identity_violation_count",
    "sidecar_generation_leak_count",
    "sidecar_verification_leak_count",
    "sidecar_primary_overlap_count",
    "public_trace_serialization_violation_count",
    "public_trace_forbidden_key_count",
    "public_trace_contract_violation_count",
    "rows_with_sidecar_observations",
    "sidecar_observation_count",
    "sidecar_observation_availability_rate",
    "sidecar_query_overlap_signal_count",
    "row_query_overlap_signal_coverage",
    "sidecar_novel_query_coverage_signal_count",
    "row_novel_query_coverage_signal_coverage",
    "evidence_gap_trace_status_counts",
    "primary_context_gold_hit_count",
    "append_pool_gold_hit_count",
    "append_pool_incremental_gold_hit_count",
    "sidecar_gold_hit_count",
    "sidecar_incremental_gold_hit_count",
    "sidecar_capture_rate_of_append_gold_opportunities",
)


class InMemoryFrozenCandidatePoolRetriever(CandidatePoolRetrieverPort):
    """Private validation retriever backed by already-built candidate pools."""

    def __init__(self, pools_by_question_id: Mapping[str, Sequence[RetrievalResult]]) -> None:
        self._pools_by_question_id = {
            question_id: tuple(results) for question_id, results in pools_by_question_id.items()
        }
        self.call_count = 0
        self.missing_question_count = 0

    def retrieve(self, question: PrimeQAQuestion) -> Sequence[RetrievalResult]:
        self.call_count += 1
        try:
            return self._pools_by_question_id[question.id]
        except KeyError:
            self.missing_question_count += 1
            raise KeyError("question is not present in the frozen validation pool") from None


@dataclass(frozen=True)
class _EntrypointValidationTrace:
    sample: PrimeQAHybridSplitSample
    agent_trace: _AgentValidationTrace
    candidate_pool_identity_violation: bool
    entrypoint_trace_serialization_violation: bool
    entrypoint_trace_forbidden_key_count: int
    entrypoint_trace_contract_violation_count: int
    dependency_call_count_violation: bool
    terminal_state_mismatch: bool
    retry_action_count: int
    fallback_action_count: int
    terminal_state: str


@dataclass(frozen=True)
class PrimeQAHybridOptionalSidecarAgentEntrypointValidationVisualization:
    """One generated Stage139 public-safe chart."""

    name: str
    path: str


def run_primeqa_hybrid_optional_sidecar_agent_entrypoint_validation(
    *,
    stage138_protocol_path: Path,
    stage137_validation_path: Path,
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
    """Run the real Stage139 train grouped-CV/dev entrypoint validation."""

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

    stage138_protocol = _load_json_object(stage138_protocol_path)
    stage137_validation = _load_json_object(stage137_validation_path)
    stage128_protocol = _load_json_object(stage128_protocol_path)
    stage125_protocol = _load_json_object(stage125_protocol_path)
    stage138_summary = _stage138_summary(stage138_protocol)
    stage137_summary = _stage137_summary(stage137_validation)
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

    current_sources = _source_files(
        stage138_protocol_path=stage138_protocol_path,
        stage137_validation_path=stage137_validation_path,
        stage128_protocol_path=stage128_protocol_path,
        stage125_protocol_path=stage125_protocol_path,
        stage80_report_path=stage80_report_path,
        train_split_path=train_split_path,
        dev_split_path=dev_split_path,
        documents_path=documents_path,
    )
    pre_checks = _pre_validation_guard_checks(
        stage138_summary=stage138_summary,
        stage137_summary=stage137_summary,
        selected_config=selected_config,
        split_samples=split_samples,
        train_fold_count=train_fold_count,
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
        include_dense_channels=include_dense_channels,
        dense_summary=dense_summary,
        current_sources=current_sources,
    )
    if not all(check["passed"] for check in pre_checks):
        checked_at = time.perf_counter()
        report = _blocked_report(
            source_files=current_sources,
            stage138_summary=stage138_summary,
            stage137_summary=stage137_summary,
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
    traces_by_split: dict[str, list[_EntrypointValidationTrace]] = {}
    retriever_summaries: dict[str, dict[str, int]] = {}
    for split, samples in split_samples.items():
        retriever = _build_validation_retriever(
            samples=samples,
            pools=candidate_pools_by_split[split],
            documents_by_id=documents_by_id,
        )
        entrypoint = create_primeqa_hybrid_optional_sidecar_agent_entrypoint(
            candidate_pool_retriever=retriever,
            evidence_selector_name=evidence_selector_name,
            max_candidates_per_document=max_candidates_per_document,
            composition_policy_name=composition_policy_name,
            max_sentences=max_sentences,
            min_sentence_score=min_sentence_score,
            min_evidence_score=min_evidence_score,
        )
        traces_by_split[split] = [
            _trace_entrypoint_validation(
                sample=sample,
                pool=candidate_pools_by_split[split][sample.sample_id],
                documents_by_id=documents_by_id,
                control_runner=control_runner,
                entrypoint=entrypoint,
            )
            for sample in samples
        ]
        retriever_summaries[split] = {
            "call_count": retriever.call_count,
            "missing_question_count": retriever.missing_question_count,
        }
    evaluated_at = time.perf_counter()

    split_reports = {
        split: _summarize_entrypoint_traces(traces) for split, traces in traces_by_split.items()
    }
    train_fold_reports = _entrypoint_train_fold_reports(
        traces=traces_by_split[_TRAIN_SPLIT],
        fold_assignments=train_fold_assignments,
    )
    train_cv_validation = _entrypoint_train_cv_validation(
        train_summary=split_reports[_TRAIN_SPLIT],
        fold_reports=train_fold_reports,
    )
    dev_report = _entrypoint_dev_report(split_reports[_DEV_SPLIT])
    pool_summary = _candidate_pool_summary(candidate_pools_by_split)
    source_parity = _stage137_source_parity(
        pool_summary=pool_summary,
        split_reports=split_reports,
        stage137_summary=stage137_summary,
    )
    report_payload = {
        "candidate_pool_summary": pool_summary,
        "retriever_summaries": retriever_summaries,
        "split_entrypoint_reports": split_reports,
        "train_fold_reports": train_fold_reports,
        "train_cv_validation": train_cv_validation,
        "dev_report": dev_report,
        "stage137_source_parity": source_parity,
    }
    guard_checks = pre_checks + _post_validation_guard_checks(
        stage138_summary=stage138_summary,
        stage137_summary=stage137_summary,
        report_payload=report_payload,
    )
    checked_at = time.perf_counter()
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Real train five-fold grouped-CV and dev report-only validation of the "
            "optional Stage139 entrypoint connected to the frozen Stage138 state "
            "machine and Stage137-validated orchestrator. It rebuilds train/dev "
            "candidate pools and runs private per-row control and entrypoint traces "
            "in memory. Public output contains aggregate statistics only. It does not "
            "load test, tune on dev, run final metrics, register a runtime default, "
            "retry failed actions, or enable fallback strategies."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_validation),
            "confirmation_note": confirmation_note,
        },
        "split_contract": _split_contract(),
        "source_files": current_sources,
        "stage138_summary": stage138_summary,
        "stage137_summary": stage137_summary,
        "stage128_summary": stage128_summary,
        "entrypoint_contract": optional_sidecar_agent_entrypoint_contract(),
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
            "run_control_and_entrypoint_traces": round(evaluated_at - pools_built_at, 3),
            "summarize_and_guard": round(checked_at - evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def _build_validation_retriever(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    pools: Mapping[str, Mapping[str, Any]],
    documents_by_id: Mapping[str, PrimeQADocument],
) -> InMemoryFrozenCandidatePoolRetriever:
    by_question_id = {
        sample.to_primeqa_question().id: _pool_results(
            pools[sample.sample_id]["stage128_pool"],
            documents_by_id,
        )
        for sample in samples
    }
    if len(by_question_id) != len(samples):
        raise ValueError("validation question ids must be unique within each split")
    return InMemoryFrozenCandidatePoolRetriever(by_question_id)


def _trace_entrypoint_validation(
    *,
    sample: PrimeQAHybridSplitSample,
    pool: Mapping[str, Any],
    documents_by_id: Mapping[str, PrimeQADocument],
    control_runner: Stage116ControlRunner,
    entrypoint: PrimeQAHybridOptionalSidecarAgentEntrypoint,
) -> _EntrypointValidationTrace:
    prefix_results = _pool_results(pool["prefix_pool"], documents_by_id)
    stage128_results = _pool_results(pool["stage128_pool"], documents_by_id)
    question = sample.to_primeqa_question()
    control = control_runner.run(question=question, prefix_results=prefix_results)
    entrypoint_run = entrypoint.run(question)
    agent = entrypoint_run.agent_run
    orchestrator_trace, orchestrator_serialization_violation = _serialize_orchestrator_trace(
        entrypoint_run
    )
    entrypoint_trace, entrypoint_serialization_violation = _serialize_entrypoint_trace(
        entrypoint_run
    )

    sidecar_handles = {
        record.runtime_content_handle for record in agent.observation_bundle.sidecar_observations
    }
    primary_handles = {
        result.document.id for result in agent.observation_bundle.answer_context_results
    }
    generation_handles = {
        result.document.id for result in entrypoint_run.generation_context_results
    }
    verification_handles = {
        result.document.id for result in entrypoint_run.verification_context_results
    }
    append_handles = {
        result.document.id
        for result in stage128_results
        if _SIDECAR_APPEND_START_RANK <= result.rank <= _TARGET_POOL_DEPTH
    }
    gold_handle = sample.answer_doc_id if sample.answerable else None
    orchestrator_contract_violations = _public_trace_contract_violation_count(
        public_trace=orchestrator_trace,
        generation_context=entrypoint_run.generation_context_results,
        verification_context=entrypoint_run.verification_context_results,
        sidecar_observation_count=len(agent.observation_bundle.sidecar_observations),
    )
    entrypoint_contract_violations = _entrypoint_trace_contract_violation_count(
        public_trace=entrypoint_trace,
        verified_refused=agent.verified_answer.refused,
    )
    base_trace = _AgentValidationTrace(
        sample=sample,
        question=question,
        control_original_answer=control.original_answer,
        control_verified_answer=control.verified_answer,
        agent_original_answer=agent.original_answer,
        agent_verified_answer=agent.verified_answer,
        generation_context_identity_violation=(
            _result_signature(control.answer_context_results)
            != _result_signature(entrypoint_run.generation_context_results)
        ),
        verification_context_identity_violation=(
            _result_signature(control.verification_context_results)
            != _result_signature(entrypoint_run.verification_context_results)
        ),
        bundle_generation_context_identity_violation=(
            _result_signature(agent.observation_bundle.answer_context_results)
            != _result_signature(entrypoint_run.generation_context_results)
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
        public_trace_serialization_violation=orchestrator_serialization_violation,
        public_trace_forbidden_key_count=len(_forbidden_keys_found(orchestrator_trace)),
        public_trace_contract_violation_count=orchestrator_contract_violations,
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
    expected_terminal = "refuse" if agent.verified_answer.refused else "complete"
    return _EntrypointValidationTrace(
        sample=sample,
        agent_trace=base_trace,
        candidate_pool_identity_violation=(
            _result_signature(stage128_results)
            != _result_signature(entrypoint_run.candidate_pool_results)
        ),
        entrypoint_trace_serialization_violation=entrypoint_serialization_violation,
        entrypoint_trace_forbidden_key_count=len(_forbidden_keys_found(entrypoint_trace)),
        entrypoint_trace_contract_violation_count=entrypoint_contract_violations,
        dependency_call_count_violation=not _dependency_calls_are_exact(entrypoint_trace),
        terminal_state_mismatch=entrypoint_trace.get("terminal_state") != expected_terminal,
        retry_action_count=int(entrypoint_trace.get("retry_action_count") or 0),
        fallback_action_count=int(entrypoint_trace.get("fallback_action_count") or 0),
        terminal_state=str(entrypoint_trace.get("terminal_state") or "missing"),
    )


def _serialize_orchestrator_trace(
    run: OptionalSidecarAgentEntrypointRun,
) -> tuple[dict[str, Any], bool]:
    try:
        return run.agent_run.public_safe_trace.to_public_dict(), False
    except ValueError:
        return {}, True


def _serialize_entrypoint_trace(
    run: OptionalSidecarAgentEntrypointRun,
) -> tuple[dict[str, Any], bool]:
    try:
        return run.public_safe_trace.to_public_dict(), False
    except ValueError:
        return {}, True


def _entrypoint_trace_contract_violation_count(
    *,
    public_trace: Mapping[str, Any],
    verified_refused: bool,
) -> int:
    terminal = "refuse" if verified_refused else "complete"
    expected_actions = ["retrieve", "answer", "verify", "observe", terminal]
    expected_states = [
        ("ready", "retrieve"),
        ("retrieve", "answer"),
        ("answer", "verify"),
        ("verify", "observe"),
        ("observe", terminal),
    ]
    action_trace = public_trace.get("action_trace") or []
    checks = [
        public_trace.get("entrypoint_id") == _ENTRYPOINT_ID,
        public_trace.get("action_state_protocol_id") == _SOURCE_STAGE138_PROTOCOL_ID,
        public_trace.get("orchestrator_id") == _ORCHESTRATOR_ID,
        public_trace.get("action_count") == 5,
        public_trace.get("terminal_state") == terminal,
        public_trace.get("terminal") is True,
        public_trace.get("verified_refused") is verified_refused,
        _dependency_calls_are_exact(public_trace),
        public_trace.get("sidecar_used_for_answer_generation") is False,
        public_trace.get("sidecar_used_for_answer_verification") is False,
        public_trace.get("sidecar_replaced_primary_context") is False,
        public_trace.get("runtime_gold_labels_read") is False,
        public_trace.get("test_membership_read") is False,
        public_trace.get("retry_action_count") == 0,
        public_trace.get("fallback_action_count") == 0,
        len(action_trace) == 5,
        [row.get("sequence_number") for row in action_trace] == [1, 2, 3, 4, 5],
        [row.get("action") for row in action_trace] == expected_actions,
        [(row.get("previous_state"), row.get("next_state")) for row in action_trace]
        == expected_states,
    ]
    return sum(not passed for passed in checks)


def _dependency_calls_are_exact(public_trace: Mapping[str, Any]) -> bool:
    return all(
        public_trace.get(key) == 1
        for key in (
            "retriever_call_count",
            "orchestrator_call_count",
            "answer_generator_call_count",
            "answer_verifier_call_count",
        )
    )


def _summarize_entrypoint_traces(
    traces: Sequence[_EntrypointValidationTrace],
) -> dict[str, Any]:
    base = _summarize_agent_traces([trace.agent_trace for trace in traces])
    complete_count = sum(trace.terminal_state == "complete" for trace in traces)
    refuse_count = sum(trace.terminal_state == "refuse" for trace in traces)
    exact_trace_count = sum(
        not trace.entrypoint_trace_serialization_violation
        and trace.entrypoint_trace_forbidden_key_count == 0
        and trace.entrypoint_trace_contract_violation_count == 0
        for trace in traces
    )
    return {
        **base,
        "candidate_pool_identity_violation_count": sum(
            trace.candidate_pool_identity_violation for trace in traces
        ),
        "entrypoint_trace_serialization_violation_count": sum(
            trace.entrypoint_trace_serialization_violation for trace in traces
        ),
        "entrypoint_trace_forbidden_key_count": sum(
            trace.entrypoint_trace_forbidden_key_count for trace in traces
        ),
        "entrypoint_trace_contract_violation_count": sum(
            trace.entrypoint_trace_contract_violation_count for trace in traces
        ),
        "dependency_call_count_violation_count": sum(
            trace.dependency_call_count_violation for trace in traces
        ),
        "terminal_state_mismatch_count": sum(trace.terminal_state_mismatch for trace in traces),
        "retry_action_count": sum(trace.retry_action_count for trace in traces),
        "fallback_action_count": sum(trace.fallback_action_count for trace in traces),
        "complete_terminal_count": complete_count,
        "refuse_terminal_count": refuse_count,
        "exact_five_transition_trace_count": exact_trace_count,
        "exact_five_transition_trace_rate": _rounded_ratio(exact_trace_count, len(traces)),
    }


def _entrypoint_train_fold_reports(
    *,
    traces: Sequence[_EntrypointValidationTrace],
    fold_assignments: Mapping[str, str],
) -> list[dict[str, Any]]:
    base_reports = _train_fold_reports(
        traces=[trace.agent_trace for trace in traces],
        fold_assignments=fold_assignments,
    )
    reports = []
    for base in base_reports:
        fold_id = str(base["fold_id"])
        fold_traces = [
            trace for trace in traces if fold_assignments[trace.sample.sample_id] == fold_id
        ]
        summary = _summarize_entrypoint_traces(fold_traces)
        reports.append(
            {
                **base,
                "candidate_pool_identity_violation_count": summary[
                    "candidate_pool_identity_violation_count"
                ],
                "entrypoint_trace_violation_count": int(
                    summary["entrypoint_trace_serialization_violation_count"]
                )
                + int(summary["entrypoint_trace_forbidden_key_count"])
                + int(summary["entrypoint_trace_contract_violation_count"]),
                "dependency_call_count_violation_count": summary[
                    "dependency_call_count_violation_count"
                ],
                "terminal_state_mismatch_count": summary["terminal_state_mismatch_count"],
                "retry_action_count": summary["retry_action_count"],
                "fallback_action_count": summary["fallback_action_count"],
                "complete_terminal_count": summary["complete_terminal_count"],
                "refuse_terminal_count": summary["refuse_terminal_count"],
                "exact_five_transition_trace_rate": summary["exact_five_transition_trace_rate"],
            }
        )
    return reports


def _entrypoint_train_cv_validation(
    *,
    train_summary: Mapping[str, Any],
    fold_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base = _train_cv_validation(
        train_summary=train_summary,
        fold_reports=fold_reports,
    )
    action_checks = {
        "all_folds_candidate_pool_identity_preserved": all(
            int(fold["candidate_pool_identity_violation_count"]) == 0 for fold in fold_reports
        ),
        "all_folds_entrypoint_traces_valid": all(
            int(fold["entrypoint_trace_violation_count"]) == 0 for fold in fold_reports
        ),
        "all_folds_dependency_calls_exactly_once": all(
            int(fold["dependency_call_count_violation_count"]) == 0 for fold in fold_reports
        ),
        "all_folds_terminal_states_match_verification": all(
            int(fold["terminal_state_mismatch_count"]) == 0 for fold in fold_reports
        ),
        "all_folds_have_exact_five_transition_traces": all(
            float(fold["exact_five_transition_trace_rate"]) == 1.0 for fold in fold_reports
        ),
        "all_folds_have_no_retry_or_fallback_actions": all(
            int(fold["retry_action_count"]) == 0 and int(fold["fallback_action_count"]) == 0
            for fold in fold_reports
        ),
    }
    checks = {**base["checks"], **action_checks}
    return {
        **base,
        "selection_mode": "fixed_optional_entrypoint_grouped_cross_validation_integrity",
        "checks": checks,
        "passed": all(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def _entrypoint_dev_report(dev_summary: Mapping[str, Any]) -> dict[str, Any]:
    base = _dev_report(dev_summary)
    return {
        **base,
        "entrypoint_action_trace_status": "report_only_exact_transition_integrity",
        "exact_five_transition_trace_rate": dev_summary.get("exact_five_transition_trace_rate"),
    }


def _stage137_source_parity(
    *,
    pool_summary: Mapping[str, Any],
    split_reports: Mapping[str, Mapping[str, Any]],
    stage137_summary: Mapping[str, Any],
) -> dict[str, Any]:
    source_splits = stage137_summary.get("split_agent_reports") or {}
    split_checks = {}
    for split in _ALLOWED_DEVELOPMENT_SPLITS:
        actual = split_reports.get(split) or {}
        source = source_splits.get(split) or {}
        mismatched = [key for key in _SOURCE_PARITY_KEYS if actual.get(key) != source.get(key)]
        split_checks[split] = {
            "passed": not mismatched,
            "mismatched_keys": mismatched,
            "verified_average_token_f1": (actual.get("agent_verified_metrics") or {}).get(
                "average_token_f1"
            ),
            "source_verified_average_token_f1": (source.get("agent_verified_metrics") or {}).get(
                "average_token_f1"
            ),
            "verified_gold_citation_count": actual.get("agent_verified_gold_citation_count"),
            "source_verified_gold_citation_count": source.get("agent_verified_gold_citation_count"),
        }
    source_pool = stage137_summary.get("candidate_pool_summary") or {}
    pool_matches = dict(pool_summary) == source_pool
    return {
        "candidate_pool_summary_matches_stage137": pool_matches,
        "split_checks": split_checks,
        "passed": pool_matches and all(check["passed"] for check in split_checks.values()),
    }


def _stage138_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision") or {}
    guards = report.get("guard_checks") or []
    public = report.get("public_safe_contract") or {}
    return {
        "stage": report.get("stage"),
        "protocol_id": report.get("protocol_id"),
        "status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "optional_agent_entrypoint_protocol_frozen": decision.get(
            "optional_agent_entrypoint_protocol_frozen"
        ),
        "can_implement_optional_agent_entrypoint_now": decision.get(
            "can_implement_optional_agent_entrypoint_now"
        ),
        "entrypoint_registered_as_runtime_default": decision.get(
            "entrypoint_registered_as_runtime_default"
        ),
        "runtime_defaultization_allowed_now": decision.get("runtime_defaultization_allowed_now"),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "retry_actions_enabled": decision.get("retry_actions_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "guard_check_count": len(guards),
        "guard_check_passed_count": sum(bool(check.get("passed")) for check in guards),
        "public_safe_forbidden_keys_found": public.get("forbidden_keys_found") or [],
    }


def _stage137_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision") or {}
    guards = report.get("guard_checks") or []
    public = report.get("public_safe_contract") or {}
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "status": decision.get("status"),
        "agent_orchestrator_integration_validated": decision.get(
            "agent_orchestrator_integration_validated"
        ),
        "sidecar_effectiveness_status": decision.get("sidecar_effectiveness_status"),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "runtime_defaultization_allowed_now": decision.get("runtime_defaultization_allowed_now"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "guard_check_count": len(guards),
        "guard_check_passed_count": sum(bool(check.get("passed")) for check in guards),
        "source_files": report.get("source_files") or {},
        "candidate_pool_summary": report.get("candidate_pool_summary") or {},
        "split_agent_reports": report.get("split_agent_reports") or {},
        "train_fold_reports": report.get("train_fold_reports") or [],
        "public_safe_forbidden_keys_found": public.get("forbidden_keys_found") or [],
        "test_split_loaded": public.get("test_split_loaded"),
        "final_test_metrics_run": public.get("final_test_metrics_run"),
    }


def _pre_validation_guard_checks(
    *,
    stage138_summary: Mapping[str, Any],
    stage137_summary: Mapping[str, Any],
    selected_config: Mapping[str, Any] | None,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    train_fold_count: int,
    user_confirmed_validation: bool,
    confirmation_note: str,
    include_dense_channels: bool,
    dense_summary: Mapping[str, Any],
    current_sources: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _check(
            name="user_confirmed_stage139_validation",
            passed=user_confirmed_validation and "Stage139" in confirmation_note,
            observed=confirmation_note,
            expected="user confirmed Stage139 validation",
        ),
        _check(
            name="stage138_protocol_frozen",
            passed=stage138_summary.get("status") == _SOURCE_STAGE138_STATUS,
            observed=stage138_summary.get("status"),
            expected=_SOURCE_STAGE138_STATUS,
        ),
        _check(
            name="stage138_protocol_id_matches",
            passed=stage138_summary.get("protocol_id") == _SOURCE_STAGE138_PROTOCOL_ID,
            observed=stage138_summary.get("protocol_id"),
            expected=_SOURCE_STAGE138_PROTOCOL_ID,
        ),
        _check(
            name="stage138_recommends_stage139_validation",
            passed=stage138_summary.get("recommended_next_direction") == _SOURCE_STAGE138_NEXT,
            observed=stage138_summary.get("recommended_next_direction"),
            expected=_SOURCE_STAGE138_NEXT,
        ),
        _check(
            name="stage138_allows_optional_entrypoint_implementation",
            passed=stage138_summary.get("optional_agent_entrypoint_protocol_frozen") is True
            and stage138_summary.get("can_implement_optional_agent_entrypoint_now") is True,
            observed={
                "frozen": stage138_summary.get("optional_agent_entrypoint_protocol_frozen"),
                "allowed": stage138_summary.get("can_implement_optional_agent_entrypoint_now"),
            },
            expected=True,
        ),
        _check(
            name="stage138_all_guard_checks_passed",
            passed=stage138_summary.get("guard_check_count") == _EXPECTED_PROTOCOL_GUARDS
            and stage138_summary.get("guard_check_passed_count") == _EXPECTED_PROTOCOL_GUARDS,
            observed={
                "passed": stage138_summary.get("guard_check_passed_count"),
                "total": stage138_summary.get("guard_check_count"),
            },
            expected={"passed": _EXPECTED_PROTOCOL_GUARDS, "total": _EXPECTED_PROTOCOL_GUARDS},
        ),
        _check(
            name="stage138_runtime_test_retry_and_fallback_remain_closed",
            passed=stage138_summary.get("entrypoint_registered_as_runtime_default") is False
            and stage138_summary.get("runtime_defaultization_allowed_now") is False
            and stage138_summary.get("can_open_final_test_gate_now") is False
            and stage138_summary.get("can_run_final_test_metrics_now") is False
            and stage138_summary.get("can_use_test_for_tuning") is False
            and stage138_summary.get("fallback_strategies_enabled") is False
            and stage138_summary.get("retry_actions_enabled") is False,
            observed=stage138_summary,
            expected="runtime default, test, retry, and fallback closed",
        ),
        _check(
            name="stage137_validation_source_available",
            passed=stage137_summary.get("status") == _SOURCE_STAGE137_STATUS
            and stage137_summary.get("analysis_id") == _SOURCE_STAGE137_ANALYSIS_ID
            and stage137_summary.get("agent_orchestrator_integration_validated") is True,
            observed={
                "status": stage137_summary.get("status"),
                "analysis_id": stage137_summary.get("analysis_id"),
                "validated": stage137_summary.get("agent_orchestrator_integration_validated"),
            },
            expected="Stage137 orchestrator validation passed",
        ),
        _check(
            name="stage137_all_guard_checks_passed",
            passed=stage137_summary.get("guard_check_count") == _EXPECTED_SOURCE_GUARDS
            and stage137_summary.get("guard_check_passed_count") == _EXPECTED_SOURCE_GUARDS,
            observed={
                "passed": stage137_summary.get("guard_check_passed_count"),
                "total": stage137_summary.get("guard_check_count"),
            },
            expected={"passed": _EXPECTED_SOURCE_GUARDS, "total": _EXPECTED_SOURCE_GUARDS},
        ),
        _check(
            name="stage137_sidecar_boundary_remains_safe_but_neutral",
            passed=stage137_summary.get("sidecar_effectiveness_status") == "safe_but_neutral",
            observed=stage137_summary.get("sidecar_effectiveness_status"),
            expected="safe_but_neutral",
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
            name="split_row_counts_match_stage137_source",
            passed=all(
                len(split_samples[split])
                == int(
                    ((stage137_summary.get("split_agent_reports") or {}).get(split) or {}).get(
                        "row_count"
                    )
                    or -1
                )
                for split in _ALLOWED_DEVELOPMENT_SPLITS
            ),
            observed={split: len(split_samples[split]) for split in split_samples},
            expected={
                split: ((stage137_summary.get("split_agent_reports") or {}).get(split) or {}).get(
                    "row_count"
                )
                for split in _ALLOWED_DEVELOPMENT_SPLITS
            },
        ),
        _check(
            name="train_fold_count_matches_frozen_protocol",
            passed=train_fold_count >= _MINIMUM_TRAIN_FOLDS,
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
            name="validation_inputs_match_stage137_source_fingerprints",
            passed=_source_fingerprints_match_stage137(
                current_sources=current_sources,
                stage137_sources=stage137_summary.get("source_files") or {},
            ),
            observed=_source_fingerprint_comparison(
                current_sources=current_sources,
                stage137_sources=stage137_summary.get("source_files") or {},
            ),
            expected="Stage128/125/80, train/dev, and corpus SHA256 match Stage137",
        ),
        _check(
            name="source_public_contracts_have_no_forbidden_keys",
            passed=stage138_summary.get("public_safe_forbidden_keys_found") == []
            and stage137_summary.get("public_safe_forbidden_keys_found") == [],
            observed={
                "stage138": stage138_summary.get("public_safe_forbidden_keys_found"),
                "stage137": stage137_summary.get("public_safe_forbidden_keys_found"),
            },
            expected=[],
        ),
    ]


def _post_validation_guard_checks(
    *,
    stage138_summary: Mapping[str, Any],
    stage137_summary: Mapping[str, Any],
    report_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    train = report_payload["split_entrypoint_reports"][_TRAIN_SPLIT]
    dev = report_payload["split_entrypoint_reports"][_DEV_SPLIT]
    pool_summary = report_payload["candidate_pool_summary"]
    retrievers = report_payload["retriever_summaries"]
    train_cv = report_payload["train_cv_validation"]
    dev_report = report_payload["dev_report"]
    source_parity = report_payload["stage137_source_parity"]
    public_payload = dict(report_payload)

    def total(key: str) -> int:
        return int(train[key]) + int(dev[key])

    return [
        _check(
            name="stage139_candidate_pool_prefix_identity_preserved",
            passed=pool_summary["all_splits_prefix_identity_violation_count"] == 0,
            observed=pool_summary["all_splits_prefix_identity_violation_count"],
            expected=0,
        ),
        _check(
            name="stage139_candidate_pool_append_budget_preserved",
            passed=pool_summary["all_splits_append_budget_exceeded_count"] == 0,
            observed=pool_summary["all_splits_append_budget_exceeded_count"],
            expected=0,
        ),
        _check(
            name="stage139_entrypoint_received_exact_candidate_pools",
            passed=total("candidate_pool_identity_violation_count") == 0,
            observed=total("candidate_pool_identity_violation_count"),
            expected=0,
        ),
        _check(
            name="stage139_retrieval_port_called_once_per_row",
            passed=all(
                int(retrievers[split]["call_count"]) == int(report["row_count"])
                and int(retrievers[split]["missing_question_count"]) == 0
                for split, report in ((_TRAIN_SPLIT, train), (_DEV_SPLIT, dev))
            ),
            observed=retrievers,
            expected="one successful retrieval-port call per row",
        ),
        _check(
            name="stage139_generation_context_identity_preserved",
            passed=total("generation_context_identity_violation_count") == 0
            and total("bundle_generation_context_identity_violation_count") == 0,
            observed={
                "control_vs_dependency": total("generation_context_identity_violation_count"),
                "bundle_vs_dependency": total("bundle_generation_context_identity_violation_count"),
            },
            expected={"control_vs_dependency": 0, "bundle_vs_dependency": 0},
        ),
        _check(
            name="stage139_verification_context_identity_preserved",
            passed=total("verification_context_identity_violation_count") == 0,
            observed=total("verification_context_identity_violation_count"),
            expected=0,
        ),
        _check(
            name="stage139_original_and_verified_answers_identical_to_stage116",
            passed=total("original_answer_identity_violation_count") == 0
            and total("verified_answer_identity_violation_count") == 0,
            observed={
                "original": total("original_answer_identity_violation_count"),
                "verified": total("verified_answer_identity_violation_count"),
            },
            expected={"original": 0, "verified": 0},
        ),
        _check(
            name="stage139_verification_reasons_identical_to_stage116",
            passed=total("verification_reason_identity_violation_count") == 0,
            observed=total("verification_reason_identity_violation_count"),
            expected=0,
        ),
        _check(
            name="stage139_sidecar_isolated_from_answer_paths",
            passed=total("sidecar_generation_leak_count") == 0
            and total("sidecar_verification_leak_count") == 0
            and total("sidecar_primary_overlap_count") == 0,
            observed={
                "generation": total("sidecar_generation_leak_count"),
                "verification": total("sidecar_verification_leak_count"),
                "primary_overlap": total("sidecar_primary_overlap_count"),
            },
            expected=0,
        ),
        _check(
            name="stage139_orchestrator_public_traces_valid",
            passed=total("public_trace_serialization_violation_count") == 0
            and total("public_trace_forbidden_key_count") == 0
            and total("public_trace_contract_violation_count") == 0,
            observed={
                "serialization": total("public_trace_serialization_violation_count"),
                "forbidden": total("public_trace_forbidden_key_count"),
                "contract": total("public_trace_contract_violation_count"),
            },
            expected=0,
        ),
        _check(
            name="stage139_entrypoint_public_action_traces_valid",
            passed=total("entrypoint_trace_serialization_violation_count") == 0
            and total("entrypoint_trace_forbidden_key_count") == 0
            and total("entrypoint_trace_contract_violation_count") == 0,
            observed={
                "serialization": total("entrypoint_trace_serialization_violation_count"),
                "forbidden": total("entrypoint_trace_forbidden_key_count"),
                "contract": total("entrypoint_trace_contract_violation_count"),
            },
            expected=0,
        ),
        _check(
            name="stage139_dependencies_called_exactly_once_per_row",
            passed=total("dependency_call_count_violation_count") == 0,
            observed=total("dependency_call_count_violation_count"),
            expected=0,
        ),
        _check(
            name="stage139_terminal_states_match_verified_refusal",
            passed=total("terminal_state_mismatch_count") == 0,
            observed=total("terminal_state_mismatch_count"),
            expected=0,
        ),
        _check(
            name="stage139_all_rows_have_exact_five_transition_trace",
            passed=float(train["exact_five_transition_trace_rate"]) == 1.0
            and float(dev["exact_five_transition_trace_rate"]) == 1.0,
            observed={
                "train": train["exact_five_transition_trace_rate"],
                "dev": dev["exact_five_transition_trace_rate"],
            },
            expected={"train": 1.0, "dev": 1.0},
        ),
        _check(
            name="stage139_no_retry_or_fallback_actions",
            passed=total("retry_action_count") == 0 and total("fallback_action_count") == 0,
            observed={
                "retry": total("retry_action_count"),
                "fallback": total("fallback_action_count"),
            },
            expected=0,
        ),
        _check(
            name="stage139_terminal_counts_cover_every_row",
            passed=all(
                int(report["complete_terminal_count"]) + int(report["refuse_terminal_count"])
                == int(report["row_count"])
                for report in (train, dev)
            ),
            observed={
                split: {
                    "complete": report["complete_terminal_count"],
                    "refuse": report["refuse_terminal_count"],
                    "rows": report["row_count"],
                }
                for split, report in ((_TRAIN_SPLIT, train), (_DEV_SPLIT, dev))
            },
            expected="complete plus refuse equals rows",
        ),
        _check(
            name="stage139_sidecar_observations_available_on_all_rows",
            passed=float(train["sidecar_observation_availability_rate"]) == 1.0
            and float(dev["sidecar_observation_availability_rate"]) == 1.0,
            observed={
                "train": train["sidecar_observation_availability_rate"],
                "dev": dev["sidecar_observation_availability_rate"],
            },
            expected=1.0,
        ),
        _check(
            name="stage139_answer_metric_deltas_are_zero",
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
            name="stage139_gold_citation_count_deltas_are_zero",
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
            expected=0,
        ),
        _check(
            name="stage139_exactly_matches_saved_stage137_aggregates",
            passed=bool(source_parity.get("passed")),
            observed=source_parity,
            expected="candidate pools and all Stage137 split aggregate keys identical",
        ),
        _check(
            name="stage139_train_grouped_cv_integrity_passed",
            passed=bool(train_cv.get("passed")),
            observed=train_cv.get("checks"),
            expected="all train grouped-CV checks pass",
        ),
        _check(
            name="stage139_dev_report_only",
            passed=dev_report.get("dev_used_for_selection") is False
            and dev_report.get("dev_used_for_retuning") is False,
            observed=dev_report.get("status"),
            expected="dev report only",
        ),
        _check(
            name="stage139_sidecar_effectiveness_boundary_preserved",
            passed=(
                int(train["append_pool_incremental_gold_hit_count"]),
                int(train["sidecar_incremental_gold_hit_count"]),
            )
            == (9, 0)
            and (
                int(dev["append_pool_incremental_gold_hit_count"]),
                int(dev["sidecar_incremental_gold_hit_count"]),
            )
            == (1, 0)
            and stage137_summary.get("sidecar_effectiveness_status") == "safe_but_neutral",
            observed={
                "train": (
                    train["append_pool_incremental_gold_hit_count"],
                    train["sidecar_incremental_gold_hit_count"],
                ),
                "dev": (
                    dev["append_pool_incremental_gold_hit_count"],
                    dev["sidecar_incremental_gold_hit_count"],
                ),
            },
            expected={"train": (9, 0), "dev": (1, 0)},
        ),
        _check(
            name="stage139_test_locked",
            passed=stage138_summary.get("can_open_final_test_gate_now") is False
            and stage138_summary.get("can_run_final_test_metrics_now") is False
            and stage138_summary.get("can_use_test_for_tuning") is False,
            observed=stage138_summary,
            expected="test locked",
        ),
        _check(
            name="stage139_runtime_defaults_unchanged",
            passed=stage138_summary.get("entrypoint_registered_as_runtime_default") is False
            and stage138_summary.get("runtime_defaultization_allowed_now") is False
            and stage138_summary.get("default_runtime_policy") == "unchanged",
            observed={
                "registered": stage138_summary.get("entrypoint_registered_as_runtime_default"),
                "allowed": stage138_summary.get("runtime_defaultization_allowed_now"),
                "policy": stage138_summary.get("default_runtime_policy"),
            },
            expected="unchanged",
        ),
        _check(
            name="stage139_public_outputs_have_no_forbidden_keys",
            passed=not _forbidden_keys_found(public_payload),
            observed=sorted(_forbidden_keys_found(public_payload)),
            expected=[],
        ),
        _check(
            name="stage139_train_cv_group_values_not_written",
            passed=train_cv.get("train_cv_group_values_written") is False,
            observed=train_cv.get("train_cv_group_values_written"),
            expected=False,
        ),
    ]


def _source_fingerprints_match_stage137(
    *,
    current_sources: Mapping[str, Any],
    stage137_sources: Mapping[str, Any],
) -> bool:
    comparison = _source_fingerprint_comparison(
        current_sources=current_sources,
        stage137_sources=stage137_sources,
    )
    return all(row["matches"] for row in comparison.values())


def _source_fingerprint_comparison(
    *,
    current_sources: Mapping[str, Any],
    stage137_sources: Mapping[str, Any],
) -> dict[str, Any]:
    keys = (
        "stage128_protocol",
        "stage125_protocol",
        "train_split",
        "dev_split",
        "corpus_documents",
        "stage80_dense_cache_report",
    )
    return {
        key: {
            "current_sha256": (current_sources.get(key) or {}).get("sha256"),
            "stage137_sha256": (stage137_sources.get(key) or {}).get("sha256"),
            "matches": (current_sources.get(key) or {}).get("sha256")
            == (stage137_sources.get(key) or {}).get("sha256"),
        }
        for key in keys
    }


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
        "validated_protocol_id": _SOURCE_STAGE138_PROTOCOL_ID,
        "validated_entrypoint_id": _ENTRYPOINT_ID,
        "entrypoint_optional": True,
        "entrypoint_registered_as_runtime_default": False,
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
        "retry_actions_enabled": False,
        "default_runtime_policy": "unchanged",
    }
    if failed_checks:
        return {
            **base,
            "status": "primeqa_hybrid_optional_sidecar_agent_entrypoint_validation_failed",
            "failed_checks": failed_checks,
            "optional_entrypoint_implementation_validated": False,
            "runtime_action_order_validated": False,
            "answer_path_invariance_validated": False,
            "can_expose_optional_runtime_entrypoint_now": False,
            "recommended_next_direction": "review_stage139_optional_entrypoint_failures",
        }
    return {
        **base,
        "status": "primeqa_hybrid_optional_sidecar_agent_entrypoint_train_cv_dev_validation_passed",
        "failed_checks": [],
        "optional_entrypoint_implementation_validated": True,
        "runtime_action_order_validated": True,
        "answer_path_invariance_validated": True,
        "can_expose_optional_runtime_entrypoint_now": True,
        "recommended_next_direction": (
            "freeze_explicit_nondefault_agent_runtime_activation_protocol"
        ),
    }


def _split_contract() -> dict[str, Any]:
    return {
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
        "selection_split": _TRAIN_SPLIT,
        "selection_mode": "fixed_optional_entrypoint_grouped_cross_validation_integrity",
        "candidate_selection_performed": False,
        "threshold_tuning_performed": False,
        "validation_split": _DEV_SPLIT,
        "dev_selection_used": False,
        "dev_retuning_used": False,
        "forbidden_final_splits": ["test"],
    }


def _validation_harness_contract() -> dict[str, Any]:
    return {
        "harness_id": "stage139_control_vs_optional_entrypoint_action_trace_harness_v1",
        "stage116_control_executed_per_row": True,
        "optional_entrypoint_executed_per_row": True,
        "candidate_pool_retrieval_port_called_per_row": True,
        "generation_dependency_input_recorded_in_memory": True,
        "verification_dependency_input_recorded_in_memory": True,
        "exact_action_trace_validated_per_row": True,
        "dependency_call_counts_validated_per_row": True,
        "terminal_state_matches_verified_refusal": True,
        "stage137_aggregate_parity_checked": True,
        "private_per_row_traces_written": False,
        "gold_used_only_for_offline_train_dev_aggregate_diagnostics": True,
    }


def _source_files(
    *,
    stage138_protocol_path: Path,
    stage137_validation_path: Path,
    stage128_protocol_path: Path,
    stage125_protocol_path: Path,
    stage80_report_path: Path | None,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
) -> dict[str, Any]:
    files = {
        "stage138_protocol": _fingerprint(stage138_protocol_path),
        "stage137_validation": _fingerprint(stage137_validation_path),
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
    source_files: Mapping[str, Any],
    stage138_summary: Mapping[str, Any],
    stage137_summary: Mapping[str, Any],
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
        "analysis_scope": "Stage139 blocked before candidate-pool and entrypoint execution.",
        "split_contract": _split_contract(),
        "source_files": dict(source_files),
        "stage138_summary": dict(stage138_summary),
        "stage137_summary": dict(stage137_summary),
        "stage128_summary": dict(stage128_summary),
        "entrypoint_contract": optional_sidecar_agent_entrypoint_contract(),
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
        "retriever_summaries": {},
        "split_entrypoint_reports": {},
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
        "stage137_source_parity": {"passed": False},
        "guard_checks": list(guard_checks),
        "decision": _decision(guard_checks),
        "timing_seconds": dict(timing_seconds),
    }


def _check(*, name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    runtime_traces_created = bool(report.get("split_entrypoint_reports"))
    return {
        "public_safe_summary_only": True,
        "runtime_control_and_entrypoint_traces_created_in_memory": runtime_traces_created,
        "runtime_control_and_entrypoint_traces_written": False,
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


def write_primeqa_hybrid_optional_sidecar_agent_entrypoint_validation_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridOptionalSidecarAgentEntrypointValidationVisualization]:
    """Write public-safe Stage139 SVG charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage139_train_fold_action_trace_violations.svg": render_horizontal_bar_chart_svg(
            title="Stage139 train-fold action trace violations",
            bars=_train_fold_action_bars(report),
            x_label="violation count",
            width=1840,
            margin_left=1000,
        ),
        "stage139_split_terminal_paths.svg": render_horizontal_bar_chart_svg(
            title="Stage139 split terminal paths",
            bars=_terminal_path_bars(report),
            x_label="row count",
            width=1500,
            margin_left=760,
        ),
        "stage139_split_answer_metric_deltas.svg": render_horizontal_bar_chart_svg(
            title="Stage139 entrypoint answer deltas versus Stage116",
            bars=_answer_delta_bars(report),
            x_label="delta",
            width=1700,
            margin_left=900,
        ),
        "stage139_stage137_source_parity.svg": render_horizontal_bar_chart_svg(
            title="Stage139 saved Stage137 source parity",
            bars=_source_parity_bars(report),
            x_label="1 means exact",
            width=1700,
            margin_left=900,
        ),
        "stage139_sidecar_isolation_violations.svg": render_horizontal_bar_chart_svg(
            title="Stage139 sidecar isolation violations",
            bars=_isolation_bars(report),
            x_label="violation count",
            width=1780,
            margin_left=960,
        ),
        "stage139_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage139 decision flags",
            bars=_decision_bars(report),
            x_label="1 means true",
            width=1940,
            margin_left=1080,
        ),
        "stage139_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage139 guard checks",
            bars=_guard_bars(report),
            x_label="1 means passed",
            width=2320,
            margin_left=1340,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridOptionalSidecarAgentEntrypointValidationVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _train_fold_action_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    keys = (
        "candidate_pool_identity_violation_count",
        "entrypoint_trace_violation_count",
        "dependency_call_count_violation_count",
        "terminal_state_mismatch_count",
        "retry_action_count",
        "fallback_action_count",
    )
    return [
        BarDatum(
            label=f"{fold.get('fold_id')} {key}",
            value=float(fold.get(key) or 0),
            value_label=str(int(fold.get(key) or 0)),
        )
        for fold in report.get("train_fold_reports") or []
        for key in keys
    ]


def _terminal_path_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    reports = report.get("split_entrypoint_reports") or {}
    return [
        BarDatum(
            label=f"{split} {label}",
            value=float((reports.get(split) or {}).get(key) or 0),
            value_label=str(int((reports.get(split) or {}).get(key) or 0)),
        )
        for split in _ALLOWED_DEVELOPMENT_SPLITS
        for key, label in (
            ("complete_terminal_count", "complete"),
            ("refuse_terminal_count", "refuse"),
        )
    ]


def _answer_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    reports = report.get("split_entrypoint_reports") or {}
    bars = []
    for split in _ALLOWED_DEVELOPMENT_SPLITS:
        summary = reports.get(split) or {}
        deltas = summary.get("verified_metric_deltas_vs_stage116") or {}
        for key in ("average_token_f1", "gold_doc_citation_rate"):
            value = float(deltas.get(key) or 0.0)
            bars.append(
                BarDatum(
                    label=f"{split} verified {key}",
                    value=value,
                    value_label=f"{value:+.4f}",
                )
            )
        citation_delta = int(summary.get("agent_verified_gold_citation_count") or 0) - int(
            summary.get("control_verified_gold_citation_count") or 0
        )
        bars.append(
            BarDatum(
                label=f"{split} verified gold citation count",
                value=float(citation_delta),
                value_label=f"{citation_delta:+d}",
            )
        )
    return bars


def _source_parity_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    parity = report.get("stage137_source_parity") or {}
    split_checks = parity.get("split_checks") or {}
    rows = [
        ("candidate pool summary", parity.get("candidate_pool_summary_matches_stage137")),
        *[
            (f"{split} split aggregate", (split_checks.get(split) or {}).get("passed"))
            for split in _ALLOWED_DEVELOPMENT_SPLITS
        ],
        ("overall parity", parity.get("passed")),
    ]
    return [
        BarDatum(
            label=label,
            value=1.0 if passed else 0.0,
            value_label="exact" if passed else "mismatch",
        )
        for label, passed in rows
    ]


def _isolation_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    reports = report.get("split_entrypoint_reports") or {}
    keys = (
        "sidecar_generation_leak_count",
        "sidecar_verification_leak_count",
        "sidecar_primary_overlap_count",
        "entrypoint_trace_contract_violation_count",
    )
    return [
        BarDatum(
            label=f"{split} {key}",
            value=float((reports.get(split) or {}).get(key) or 0),
            value_label=str(int((reports.get(split) or {}).get(key) or 0)),
        )
        for split in _ALLOWED_DEVELOPMENT_SPLITS
        for key in keys
    ]


def _decision_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    keys = (
        "optional_entrypoint_implementation_validated",
        "runtime_action_order_validated",
        "answer_path_invariance_validated",
        "can_expose_optional_runtime_entrypoint_now",
        "entrypoint_registered_as_runtime_default",
        "can_claim_answer_quality_improvement",
        "can_open_final_test_gate_now",
        "runtime_defaultization_allowed_now",
        "retry_actions_enabled",
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
            label=str(check.get("name")),
            value=1.0 if check.get("passed") else 0.0,
            value_label="passed" if check.get("passed") else "failed",
        )
        for check in report.get("guard_checks") or []
    ]
