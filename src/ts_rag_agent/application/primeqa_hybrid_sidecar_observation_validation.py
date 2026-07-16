from __future__ import annotations

import math
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ts_rag_agent.application.primeqa_hybrid_agent_retrieval_integration_validation import (
    _BASELINE_PREFIX_DEPTH,
    _DEFAULT_BM25_B,
    _DEFAULT_BM25_K1,
    _DEFAULT_ENCODER_BATCH_SIZE,
    _DEV_SPLIT,
    _SELECTED_CONFIG_ID,
    _TARGET_POOL_DEPTH,
    _TRAIN_SPLIT,
    _candidate_pool_summary,
    _candidate_pools_by_split,
    _DocumentEvidenceShortlister,
    _evaluation_channels,
    _public_channel_catalog,
    _rank_score,
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
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.bm25_retriever import tokenize_text
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
    summarize_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 135"
_CREATED_AT = "2026-07-16"
_ANALYSIS_ID = "primeqa_hybrid_stage116_answer_context_stage128_sidecar_observation_validation_v1"
_SOURCE_STAGE134_STATUS = (
    "primeqa_hybrid_stage116_answer_context_stage128_sidecar_agent_protocol_frozen"
)
_SOURCE_STAGE134_PROTOCOL_ID = (
    "primeqa_hybrid_stage116_answer_context_stage128_sidecar_agent_protocol_v1"
)
_SOURCE_STAGE134_NEXT = (
    "run_stage116_answer_context_stage128_sidecar_observation_train_cv_dev_validation"
)
_SOURCE_STAGE132_STATUS = "primeqa_hybrid_append_candidate_evidence_shortlist_validation_completed"
_SOURCE_STAGE132_SELECTED_CONFIG = "prefix10_append_sidecar_probe_v1"
_SOURCE_STAGE132_SELECTED_PROFILE = "stage132_prefix10_append_sidecar_probe_v1"
_SOURCE_STAGE116_PROFILE = "stage116_top200_agent_pool_control"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_MINIMUM_TRAIN_FOLDS = 5
_PRIMARY_CONTEXT_DEPTH = 10
_SIDECAR_OBSERVATION_SLOTS = 3
_SIDECAR_APPEND_START_RANK = 201
_MAX_TEXT_CHARS = 5000
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
        "source_doc_ids",
    }
)


@dataclass(frozen=True)
class CandidateScoreSummary:
    """Runtime-visible score components without private text."""

    retrieval_rank: int
    retrieval_score: float
    query_overlap_count: int
    query_overlap_ratio: float
    retrieval_prior: float
    combined_score: float


@dataclass(frozen=True)
class CitationVerificationSignal:
    """Untuned runtime signal for a future citation-verification consumer."""

    query_overlap_present: bool
    novel_query_term_count: int
    novel_query_term_ratio: float
    extends_primary_query_coverage: bool
    duplicate_of_primary_context: bool


@dataclass(frozen=True)
class PrimaryContextRecord:
    """One Stage116 primary answer-context record."""

    runtime_content_handle: str
    primary_context_rank: int
    primary_context_source_region: str
    retrieval_score_summary: CandidateScoreSummary


@dataclass(frozen=True)
class SidecarObservationRecord:
    """One Stage128 append observation that cannot enter answer generation."""

    runtime_content_handle: str
    sidecar_observation_rank: int
    sidecar_source_region: str
    sidecar_route_family: str
    sidecar_score_summary: CandidateScoreSummary
    citation_verification_signal: CitationVerificationSignal


@dataclass(frozen=True)
class SidecarObservationBundle:
    """Isolated primary answer context and metadata-only sidecar observations."""

    answer_context_results: tuple[RetrievalResult, ...]
    primary_context_records: tuple[PrimaryContextRecord, ...]
    sidecar_observations: tuple[SidecarObservationRecord, ...]

    def answer_context_for_generation(self) -> tuple[RetrievalResult, ...]:
        """Return only the Stage116 primary channel to an answer generator."""

        return self.answer_context_results


class CandidateScoringPolicy(Protocol):
    """Polymorphic runtime-visible scoring policy used by the adapter."""

    def query_terms(self, question: PrimeQAQuestion) -> set[str]: ...

    def document_terms(self, document: PrimeQADocument) -> set[str]: ...

    def score_summary(
        self,
        *,
        query_terms: set[str],
        result: RetrievalResult,
    ) -> CandidateScoreSummary: ...

    def rank(
        self,
        *,
        query_terms: set[str],
        candidates: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]: ...


class QueryOverlapCandidateScoringPolicy:
    """Stage129-compatible query-overlap scorer using runtime-visible inputs."""

    def __init__(self, *, max_text_chars: int = _MAX_TEXT_CHARS) -> None:
        if max_text_chars <= 0:
            raise ValueError("max_text_chars must be positive")
        self._max_text_chars = max_text_chars
        self._term_cache: dict[str, set[str]] = {}

    def query_terms(self, question: PrimeQAQuestion) -> set[str]:
        return set(tokenize_text(question.full_question))

    def document_terms(self, document: PrimeQADocument) -> set[str]:
        if document.id not in self._term_cache:
            text = f"{document.title}\n{document.text[: self._max_text_chars]}"
            self._term_cache[document.id] = set(tokenize_text(text))
        return self._term_cache[document.id]

    def score_summary(
        self,
        *,
        query_terms: set[str],
        result: RetrievalResult,
    ) -> CandidateScoreSummary:
        overlap_count = len(query_terms & self.document_terms(result.document))
        overlap_ratio = overlap_count / max(1, len(query_terms))
        retrieval_prior = 1.0 / math.log2(result.rank + 1)
        combined_score = overlap_count + overlap_ratio + 0.35 * retrieval_prior
        return CandidateScoreSummary(
            retrieval_rank=result.rank,
            retrieval_score=round(float(result.score), 8),
            query_overlap_count=overlap_count,
            query_overlap_ratio=round(overlap_ratio, 4),
            retrieval_prior=round(retrieval_prior, 8),
            combined_score=round(combined_score, 8),
        )

    def rank(
        self,
        *,
        query_terms: set[str],
        candidates: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]:
        if not query_terms:
            return list(candidates)
        return sorted(
            candidates,
            key=lambda result: (
                -self.score_summary(
                    query_terms=query_terms,
                    result=result,
                ).combined_score,
                result.rank,
                result.document.id,
            ),
        )


class PrimeQAHybridSidecarObservationAdapter:
    """Build the frozen Stage116 primary plus Stage128 sidecar interface."""

    def __init__(
        self,
        *,
        scoring_policy: CandidateScoringPolicy | None = None,
        primary_context_depth: int = _PRIMARY_CONTEXT_DEPTH,
        sidecar_observation_slots: int = _SIDECAR_OBSERVATION_SLOTS,
        prefix_depth: int = _BASELINE_PREFIX_DEPTH,
        candidate_pool_depth: int = _TARGET_POOL_DEPTH,
    ) -> None:
        if primary_context_depth <= 0:
            raise ValueError("primary_context_depth must be positive")
        if sidecar_observation_slots <= 0:
            raise ValueError("sidecar_observation_slots must be positive")
        if prefix_depth < primary_context_depth:
            raise ValueError("prefix_depth must cover the primary context")
        if candidate_pool_depth <= prefix_depth:
            raise ValueError("candidate_pool_depth must exceed prefix_depth")
        self._scoring_policy = scoring_policy or QueryOverlapCandidateScoringPolicy()
        self._primary_context_depth = primary_context_depth
        self._sidecar_observation_slots = sidecar_observation_slots
        self._prefix_depth = prefix_depth
        self._candidate_pool_depth = candidate_pool_depth

    def observe(
        self,
        *,
        question: PrimeQAQuestion,
        candidate_pool_results: Sequence[RetrievalResult],
    ) -> SidecarObservationBundle:
        query_terms = self._scoring_policy.query_terms(question)
        prefix = [result for result in candidate_pool_results if result.rank <= self._prefix_depth]
        append = [
            result
            for result in candidate_pool_results
            if self._prefix_depth < result.rank <= self._candidate_pool_depth
        ]
        primary = self._scoring_policy.rank(
            query_terms=query_terms,
            candidates=prefix,
        )[: self._primary_context_depth]
        sidecar = self._scoring_policy.rank(
            query_terms=query_terms,
            candidates=append,
        )[: self._sidecar_observation_slots]

        primary_handles = {result.document.id for result in primary}
        primary_covered_terms = set().union(
            *(
                self._scoring_policy.document_terms(result.document) & query_terms
                for result in primary
            ),
            set(),
        )
        primary_records = tuple(
            PrimaryContextRecord(
                runtime_content_handle=result.document.id,
                primary_context_rank=index,
                primary_context_source_region=_primary_source_region(result.rank),
                retrieval_score_summary=self._scoring_policy.score_summary(
                    query_terms=query_terms,
                    result=result,
                ),
            )
            for index, result in enumerate(primary, start=1)
        )
        sidecar_records = tuple(
            self._sidecar_record(
                result=result,
                observation_rank=index,
                query_terms=query_terms,
                primary_handles=primary_handles,
                primary_covered_terms=primary_covered_terms,
            )
            for index, result in enumerate(sidecar, start=1)
        )
        return SidecarObservationBundle(
            answer_context_results=tuple(primary),
            primary_context_records=primary_records,
            sidecar_observations=sidecar_records,
        )

    def _sidecar_record(
        self,
        *,
        result: RetrievalResult,
        observation_rank: int,
        query_terms: set[str],
        primary_handles: set[str],
        primary_covered_terms: set[str],
    ) -> SidecarObservationRecord:
        document_query_terms = self._scoring_policy.document_terms(result.document) & query_terms
        novel_terms = document_query_terms - primary_covered_terms
        signal = CitationVerificationSignal(
            query_overlap_present=bool(document_query_terms),
            novel_query_term_count=len(novel_terms),
            novel_query_term_ratio=round(
                len(novel_terms) / max(1, len(query_terms)),
                4,
            ),
            extends_primary_query_coverage=bool(novel_terms),
            duplicate_of_primary_context=result.document.id in primary_handles,
        )
        return SidecarObservationRecord(
            runtime_content_handle=result.document.id,
            sidecar_observation_rank=observation_rank,
            sidecar_source_region="stage128_append_expansion_201_400",
            sidecar_route_family="stage128_append_fusion_candidate",
            sidecar_score_summary=self._scoring_policy.score_summary(
                query_terms=query_terms,
                result=result,
            ),
            citation_verification_signal=signal,
        )


@dataclass(frozen=True)
class PrimeQAHybridSidecarObservationValidationVisualization:
    """One generated Stage135 validation chart."""

    name: str
    path: str


@dataclass(frozen=True)
class _ObservationTrace:
    sample: PrimeQAHybridSplitSample
    primary_identity_violation: bool
    generation_identity_violation: bool
    primary_record_field_violation_count: int
    sidecar_record_field_violation_count: int
    sidecar_region_violation_count: int
    sidecar_answer_context_leak_count: int
    sidecar_primary_overlap_count: int
    sidecar_slot_overflow: bool
    sidecar_observation_count: int
    sidecar_query_overlap_count: int
    sidecar_novel_coverage_count: int
    primary_context_gold_hit: bool
    append_pool_gold_hit: bool
    sidecar_gold_hit: bool


def run_primeqa_hybrid_sidecar_observation_validation(
    *,
    stage134_protocol_path: Path,
    stage132_validation_path: Path,
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
) -> dict[str, Any]:
    """Run real train grouped-CV/dev sidecar observation integrity validation."""

    _validate_options(
        train_fold_count=train_fold_count,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        encoder_batch_size=encoder_batch_size,
    )
    started_at = time.perf_counter()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    stage134_protocol = _load_json_object(stage134_protocol_path)
    stage132_validation = _load_json_object(stage132_validation_path)
    stage128_protocol = _load_json_object(stage128_protocol_path)
    stage125_protocol = _load_json_object(stage125_protocol_path)
    stage134_summary = _stage134_summary(stage134_protocol)
    stage132_summary = _stage132_summary(stage132_validation)
    stage128_summary = _stage128_summary(stage128_protocol)
    source_answer_invariance = _stage132_answer_invariance(stage132_validation)
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
        stage134_summary=stage134_summary,
        stage132_summary=stage132_summary,
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
            stage134_protocol_path=stage134_protocol_path,
            stage132_validation_path=stage132_validation_path,
            stage128_protocol_path=stage128_protocol_path,
            stage125_protocol_path=stage125_protocol_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
            stage134_summary=stage134_summary,
            stage132_summary=stage132_summary,
            stage128_summary=stage128_summary,
            source_answer_invariance=source_answer_invariance,
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

    adapter = PrimeQAHybridSidecarObservationAdapter()
    stage116_shortlister = _DocumentEvidenceShortlister()
    traces_by_split = {
        split: [
            _trace_observation(
                sample=sample,
                pool=candidate_pools_by_split[split][sample.sample_id],
                documents_by_id=documents_by_id,
                adapter=adapter,
                stage116_shortlister=stage116_shortlister,
            )
            for sample in samples
        ]
        for split, samples in split_samples.items()
    }
    observed_at = time.perf_counter()

    split_reports = {
        split: _summarize_observation_traces(traces) for split, traces in traces_by_split.items()
    }
    train_fold_reports = _train_fold_reports(
        traces=traces_by_split[_TRAIN_SPLIT],
        fold_assignments=train_fold_assignments,
    )
    train_cv_validation = _train_cv_validation(
        train_summary=split_reports[_TRAIN_SPLIT],
        fold_reports=train_fold_reports,
        source_answer_invariance=source_answer_invariance[_TRAIN_SPLIT],
    )
    dev_report = _dev_report(
        dev_summary=split_reports[_DEV_SPLIT],
        source_answer_invariance=source_answer_invariance[_DEV_SPLIT],
    )
    pool_summary = _candidate_pool_summary(candidate_pools_by_split)
    report_payload = {
        "candidate_pool_summary": pool_summary,
        "split_observation_reports": split_reports,
        "train_fold_reports": train_fold_reports,
        "train_cv_validation": train_cv_validation,
        "dev_report_observations": dev_report,
        "source_answer_invariance": source_answer_invariance,
    }
    guard_checks = pre_checks + _post_validation_guard_checks(
        stage134_summary=stage134_summary,
        pool_summary=pool_summary,
        report_payload=report_payload,
    )
    checked_at = time.perf_counter()
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Real train grouped-CV and dev report-only validation of the frozen "
            "Stage116 primary answer-context plus Stage128/Stage132 sidecar-observation "
            "protocol. It rebuilds train/dev candidate pools and creates runtime "
            "observation records in memory, while public output contains aggregate "
            "statistics only. It does not load test, run final metrics, expose private "
            "rows, allow sidecar answer generation or prefix replacement, add fallback "
            "strategies, or change runtime defaults."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_validation),
            "confirmation_note": confirmation_note,
        },
        "split_contract": _split_contract(),
        "source_files": _source_files(
            stage134_protocol_path=stage134_protocol_path,
            stage132_validation_path=stage132_validation_path,
            stage128_protocol_path=stage128_protocol_path,
            stage125_protocol_path=stage125_protocol_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
        ),
        "stage134_summary": stage134_summary,
        "stage132_summary": stage132_summary,
        "stage128_summary": stage128_summary,
        "adapter_contract": _adapter_contract(),
        "evaluation_options": {
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "train_fold_count": train_fold_count,
            "include_dense_channels": include_dense_channels,
            "encoder_batch_size": encoder_batch_size,
            "encoder_device": encoder_device,
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
        "decision": _decision(guard_checks=guard_checks),
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
            "build_and_validate_observations": round(observed_at - pools_built_at, 3),
            "summarize_and_guard": round(checked_at - observed_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_sidecar_observation_validation_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridSidecarObservationValidationVisualization]:
    """Write public-safe SVG charts for Stage135."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage135_split_sidecar_signal_coverage.svg": render_horizontal_bar_chart_svg(
            title="Stage135 sidecar signal coverage",
            bars=_split_signal_coverage_bars(report),
            x_label="rate",
            width=1600,
            margin_left=820,
        ),
        "stage135_train_fold_signal_coverage.svg": render_horizontal_bar_chart_svg(
            title="Stage135 train grouped-CV signal coverage",
            bars=_train_fold_signal_coverage_bars(report),
            x_label="rate",
            width=1540,
            margin_left=760,
        ),
        "stage135_gold_observation_opportunities.svg": render_horizontal_bar_chart_svg(
            title="Stage135 aggregate gold observation opportunities",
            bars=_gold_opportunity_bars(report),
            x_label="answerable row count",
            width=1660,
            margin_left=860,
        ),
        "stage135_isolation_violation_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage135 isolation violation counts",
            bars=_isolation_violation_bars(report),
            x_label="violation count",
            width=1740,
            margin_left=900,
        ),
        "stage135_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage135 decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1740,
            margin_left=900,
        ),
        "stage135_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage135 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=2100,
            margin_left=1180,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridSidecarObservationValidationVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _trace_observation(
    *,
    sample: PrimeQAHybridSplitSample,
    pool: Mapping[str, Any],
    documents_by_id: Mapping[str, PrimeQADocument],
    adapter: PrimeQAHybridSidecarObservationAdapter,
    stage116_shortlister: _DocumentEvidenceShortlister,
) -> _ObservationTrace:
    prefix_results = _pool_results(pool["prefix_pool"], documents_by_id)
    stage128_results = _pool_results(pool["stage128_pool"], documents_by_id)
    question = sample.to_primeqa_question()
    control_context = stage116_shortlister.shortlist(
        question=question,
        candidates=prefix_results,
        top_k=_PRIMARY_CONTEXT_DEPTH,
    )
    bundle = adapter.observe(
        question=question,
        candidate_pool_results=stage128_results,
    )
    generation_context = bundle.answer_context_for_generation()
    control_signature = _result_signature(control_context)
    primary_signature = _result_signature(bundle.answer_context_results)
    primary_record_signature = tuple(
        (record.runtime_content_handle, record.retrieval_score_summary.retrieval_rank)
        for record in bundle.primary_context_records
    )
    sidecar_handles = {record.runtime_content_handle for record in bundle.sidecar_observations}
    generation_handles = {result.document.id for result in generation_context}
    primary_handles = {record.runtime_content_handle for record in bundle.primary_context_records}
    gold_handle = sample.answer_doc_id if sample.answerable else None
    append_handles = {
        result.document.id
        for result in stage128_results
        if _SIDECAR_APPEND_START_RANK <= result.rank <= _TARGET_POOL_DEPTH
    }
    return _ObservationTrace(
        sample=sample,
        primary_identity_violation=control_signature != primary_signature,
        generation_identity_violation=(
            primary_signature != _result_signature(generation_context)
            or primary_record_signature
            != tuple((result.document.id, result.rank) for result in generation_context)
        ),
        primary_record_field_violation_count=sum(
            not _primary_record_complete(record) for record in bundle.primary_context_records
        ),
        sidecar_record_field_violation_count=sum(
            not _sidecar_record_complete(record) for record in bundle.sidecar_observations
        ),
        sidecar_region_violation_count=sum(
            not (
                _SIDECAR_APPEND_START_RANK
                <= record.sidecar_score_summary.retrieval_rank
                <= _TARGET_POOL_DEPTH
            )
            for record in bundle.sidecar_observations
        ),
        sidecar_answer_context_leak_count=len(sidecar_handles & generation_handles),
        sidecar_primary_overlap_count=len(sidecar_handles & primary_handles),
        sidecar_slot_overflow=(len(bundle.sidecar_observations) > _SIDECAR_OBSERVATION_SLOTS),
        sidecar_observation_count=len(bundle.sidecar_observations),
        sidecar_query_overlap_count=sum(
            record.citation_verification_signal.query_overlap_present
            for record in bundle.sidecar_observations
        ),
        sidecar_novel_coverage_count=sum(
            record.citation_verification_signal.extends_primary_query_coverage
            for record in bundle.sidecar_observations
        ),
        primary_context_gold_hit=bool(gold_handle and gold_handle in primary_handles),
        append_pool_gold_hit=bool(gold_handle and gold_handle in append_handles),
        sidecar_gold_hit=bool(gold_handle and gold_handle in sidecar_handles),
    )


def _pool_results(
    handles: Sequence[str],
    documents_by_id: Mapping[str, PrimeQADocument],
) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            document=documents_by_id[handle],
            score=_rank_score(rank),
            rank=rank,
        )
        for rank, handle in enumerate(handles, start=1)
        if handle in documents_by_id
    ]


def _result_signature(results: Sequence[RetrievalResult]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (result.document.id, result.rank, round(float(result.score), 8)) for result in results
    )


def _primary_record_complete(record: PrimaryContextRecord) -> bool:
    return bool(
        record.runtime_content_handle
        and record.primary_context_rank > 0
        and record.primary_context_source_region
        and record.retrieval_score_summary.retrieval_rank > 0
    )


def _sidecar_record_complete(record: SidecarObservationRecord) -> bool:
    return bool(
        record.runtime_content_handle
        and record.sidecar_observation_rank > 0
        and record.sidecar_source_region
        and record.sidecar_route_family
        and record.sidecar_score_summary.retrieval_rank > 0
        and isinstance(record.citation_verification_signal.query_overlap_present, bool)
        and isinstance(
            record.citation_verification_signal.extends_primary_query_coverage,
            bool,
        )
    )


def _summarize_observation_traces(
    traces: Sequence[_ObservationTrace],
) -> dict[str, Any]:
    answerable = [trace for trace in traces if trace.sample.answerable]
    observation_count = sum(trace.sidecar_observation_count for trace in traces)
    query_overlap_count = sum(trace.sidecar_query_overlap_count for trace in traces)
    novel_coverage_count = sum(trace.sidecar_novel_coverage_count for trace in traces)
    rows_with_observations = sum(trace.sidecar_observation_count > 0 for trace in traces)
    rows_with_full_observations = sum(
        trace.sidecar_observation_count == _SIDECAR_OBSERVATION_SLOTS for trace in traces
    )
    rows_with_query_overlap = sum(trace.sidecar_query_overlap_count > 0 for trace in traces)
    rows_with_novel_coverage = sum(trace.sidecar_novel_coverage_count > 0 for trace in traces)
    primary_gold_hits = sum(trace.primary_context_gold_hit for trace in answerable)
    append_gold_hits = sum(trace.append_pool_gold_hit for trace in answerable)
    sidecar_gold_hits = sum(trace.sidecar_gold_hit for trace in answerable)
    append_incremental = sum(
        trace.append_pool_gold_hit and not trace.primary_context_gold_hit for trace in answerable
    )
    sidecar_incremental = sum(
        trace.sidecar_gold_hit and not trace.primary_context_gold_hit for trace in answerable
    )
    return {
        "row_count": len(traces),
        "answerable_count": len(answerable),
        "primary_context_identity_violation_count": sum(
            trace.primary_identity_violation for trace in traces
        ),
        "answer_generation_context_identity_violation_count": sum(
            trace.generation_identity_violation for trace in traces
        ),
        "primary_record_field_violation_count": sum(
            trace.primary_record_field_violation_count for trace in traces
        ),
        "sidecar_record_field_violation_count": sum(
            trace.sidecar_record_field_violation_count for trace in traces
        ),
        "sidecar_region_violation_count": sum(
            trace.sidecar_region_violation_count for trace in traces
        ),
        "sidecar_answer_context_leak_count": sum(
            trace.sidecar_answer_context_leak_count for trace in traces
        ),
        "sidecar_primary_overlap_count": sum(
            trace.sidecar_primary_overlap_count for trace in traces
        ),
        "sidecar_slot_overflow_count": sum(trace.sidecar_slot_overflow for trace in traces),
        "rows_with_sidecar_observations": rows_with_observations,
        "rows_with_full_sidecar_observations": rows_with_full_observations,
        "sidecar_observation_count": observation_count,
        "sidecar_observation_availability_rate": _rounded_ratio(
            rows_with_observations,
            len(traces),
        ),
        "full_sidecar_observation_rate": _rounded_ratio(
            rows_with_full_observations,
            len(traces),
        ),
        "sidecar_query_overlap_signal_count": query_overlap_count,
        "sidecar_query_overlap_signal_rate": _rounded_ratio(
            query_overlap_count,
            observation_count,
        ),
        "rows_with_query_overlap_signal": rows_with_query_overlap,
        "row_query_overlap_signal_coverage": _rounded_ratio(
            rows_with_query_overlap,
            len(traces),
        ),
        "sidecar_novel_query_coverage_signal_count": novel_coverage_count,
        "sidecar_novel_query_coverage_signal_rate": _rounded_ratio(
            novel_coverage_count,
            observation_count,
        ),
        "rows_with_novel_query_coverage_signal": rows_with_novel_coverage,
        "row_novel_query_coverage_signal_coverage": _rounded_ratio(
            rows_with_novel_coverage,
            len(traces),
        ),
        "primary_context_gold_hit_count": primary_gold_hits,
        "append_pool_gold_hit_count": append_gold_hits,
        "append_pool_incremental_gold_hit_count": append_incremental,
        "sidecar_gold_hit_count": sidecar_gold_hits,
        "sidecar_incremental_gold_hit_count": sidecar_incremental,
        "sidecar_capture_rate_of_append_gold_opportunities": _rounded_ratio(
            sidecar_incremental,
            append_incremental,
        ),
    }


def _train_fold_reports(
    *,
    traces: Sequence[_ObservationTrace],
    fold_assignments: Mapping[str, str],
) -> list[dict[str, Any]]:
    reports = []
    for fold_id in sorted(set(fold_assignments.values())):
        fold_traces = [
            trace for trace in traces if fold_assignments[trace.sample.sample_id] == fold_id
        ]
        summary = _summarize_observation_traces(fold_traces)
        reports.append(
            {
                "fold_id": fold_id,
                "row_count": summary["row_count"],
                "primary_context_identity_violation_count": summary[
                    "primary_context_identity_violation_count"
                ],
                "sidecar_answer_context_leak_count": summary["sidecar_answer_context_leak_count"],
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
            }
        )
    return reports


def _train_cv_validation(
    *,
    train_summary: Mapping[str, Any],
    fold_reports: Sequence[Mapping[str, Any]],
    source_answer_invariance: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "all_folds_primary_context_identity_preserved": all(
            int(fold["primary_context_identity_violation_count"]) == 0 for fold in fold_reports
        ),
        "all_folds_sidecar_isolated_from_answer_context": all(
            int(fold["sidecar_answer_context_leak_count"]) == 0 for fold in fold_reports
        ),
        "all_folds_have_sidecar_observations": all(
            float(fold["sidecar_observation_availability_rate"]) > 0.0 for fold in fold_reports
        ),
        "all_folds_have_query_overlap_signals": all(
            float(fold["row_query_overlap_signal_coverage"]) > 0.0 for fold in fold_reports
        ),
        "source_stage132_answer_f1_delta_is_zero": float(
            source_answer_invariance["verified_average_token_f1_delta"]
        )
        == 0.0,
        "source_stage132_gold_citation_delta_is_zero": int(
            source_answer_invariance["verified_gold_citation_count_delta"]
        )
        == 0,
        "source_stage132_changed_answer_count_is_zero": int(
            source_answer_invariance["changed_verified_answer_count"]
        )
        == 0,
    }
    return {
        "selection_split": _TRAIN_SPLIT,
        "selection_mode": "train_grouped_cross_validation_sidecar_observation_integrity",
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
        "source_answer_invariance": dict(source_answer_invariance),
        "checks": checks,
        "passed": all(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "train_cv_group_values_written": False,
    }


def _dev_report(
    *,
    dev_summary: Mapping[str, Any],
    source_answer_invariance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "validation_split": _DEV_SPLIT,
        "status": "reported_not_used_for_selection_or_retuning",
        "dev_used_for_selection": False,
        "dev_used_for_retuning": False,
        "observation_summary": dict(dev_summary),
        "source_answer_invariance": dict(source_answer_invariance),
        "dev_gate_status": "report_only_no_runtime_default_or_test_gate",
    }


def _stage134_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision") or {}
    frozen = report.get("frozen_protocol") or {}
    channels = frozen.get("agent_channel_contract") or {}
    primary = channels.get("primary_answer_context_channel") or {}
    sidecar = channels.get("sidecar_observation_channel") or {}
    public_safe = report.get("public_safe_contract") or {}
    return {
        "stage": report.get("stage"),
        "protocol_id": report.get("protocol_id"),
        "status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "primary_channel_id": primary.get("channel_id"),
        "primary_answer_context_depth": primary.get("answer_context_depth"),
        "primary_allowed_to_generate_answer_text": primary.get("allowed_to_generate_answer_text"),
        "primary_sidecar_candidates_included": primary.get("sidecar_candidates_included"),
        "sidecar_channel_id": sidecar.get("channel_id"),
        "sidecar_candidate_pool_depth": sidecar.get("candidate_pool_depth"),
        "sidecar_selected_config": sidecar.get("selected_sidecar_config"),
        "sidecar_observation_slots": sidecar.get("observation_slots"),
        "sidecar_allowed_to_generate_answer_text": sidecar.get("allowed_to_generate_answer_text"),
        "sidecar_allowed_to_replace_primary_context": sidecar.get(
            "allowed_to_replace_primary_context"
        ),
        "direct_stage128_all400_answer_context_remains_blocked": decision.get(
            "direct_stage128_all400_answer_context_remains_blocked"
        ),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "runtime_defaultization_allowed_now": decision.get("runtime_defaultization_allowed_now"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
    }


def _stage132_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision") or {}
    public_safe = report.get("public_safe_contract") or {}
    guard_checks = report.get("guard_checks") or []
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "status": decision.get("status"),
        "selected_config_id": decision.get("selected_config_id"),
        "selected_profile_id": decision.get("selected_profile_id"),
        "guard_check_count": len(guard_checks),
        "guard_check_passed_count": sum(bool(check.get("passed")) for check in guard_checks),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "runtime_defaultization_allowed_now": decision.get("runtime_defaultization_allowed_now"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
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


def _stage132_answer_invariance(report: Mapping[str, Any]) -> dict[str, Any]:
    profiles = report.get("profile_reports") or {}
    stage116 = profiles.get(_SOURCE_STAGE116_PROFILE) or {}
    sidecar = profiles.get(_SOURCE_STAGE132_SELECTED_PROFILE) or {}
    output = {}
    for split, source_key in ((_TRAIN_SPLIT, "train_cv"), (_DEV_SPLIT, "dev")):
        control_split = (stage116.get("split_reports") or {}).get(source_key) or {}
        sidecar_split = (sidecar.get("split_reports") or {}).get(source_key) or {}
        control_verified = control_split.get("verified_metrics") or {}
        sidecar_verified = sidecar_split.get("verified_metrics") or {}
        control_evidence = control_split.get("selected_evidence_summary") or {}
        sidecar_evidence = sidecar_split.get("selected_evidence_summary") or {}
        output[split] = {
            "source": "Stage132 measured selected sidecar profile vs Stage116 control",
            "row_count": sidecar_split.get("row_count"),
            "verified_average_token_f1_delta": round(
                float(sidecar_verified.get("average_token_f1") or 0.0)
                - float(control_verified.get("average_token_f1") or 0.0),
                4,
            ),
            "verified_gold_citation_count_delta": int(
                sidecar_evidence.get("gold_citation_count") or 0
            )
            - int(control_evidence.get("gold_citation_count") or 0),
            "changed_verified_answer_count": int(
                sidecar_split.get("changed_verified_answers_vs_stage116_control") or 0
            ),
        }
    return output


def _pre_validation_guard_checks(
    *,
    stage134_summary: Mapping[str, Any],
    stage132_summary: Mapping[str, Any],
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
            name="user_confirmed_stage135_validation",
            passed=user_confirmed_validation and "Stage135" in confirmation_note,
            observed=confirmation_note,
            expected="user confirmed Stage135 validation",
        ),
        _check(
            name="stage134_protocol_frozen",
            passed=stage134_summary.get("status") == _SOURCE_STAGE134_STATUS,
            observed=stage134_summary.get("status"),
            expected=_SOURCE_STAGE134_STATUS,
        ),
        _check(
            name="stage134_protocol_id_matches",
            passed=stage134_summary.get("protocol_id") == _SOURCE_STAGE134_PROTOCOL_ID,
            observed=stage134_summary.get("protocol_id"),
            expected=_SOURCE_STAGE134_PROTOCOL_ID,
        ),
        _check(
            name="stage134_recommends_stage135_validation",
            passed=stage134_summary.get("recommended_next_direction") == _SOURCE_STAGE134_NEXT,
            observed=stage134_summary.get("recommended_next_direction"),
            expected=_SOURCE_STAGE134_NEXT,
        ),
        _check(
            name="stage132_selected_sidecar_validation_available",
            passed=stage132_summary.get("status") == _SOURCE_STAGE132_STATUS
            and stage132_summary.get("selected_config_id") == _SOURCE_STAGE132_SELECTED_CONFIG
            and stage132_summary.get("selected_profile_id") == _SOURCE_STAGE132_SELECTED_PROFILE,
            observed={
                "status": stage132_summary.get("status"),
                "selected_config_id": stage132_summary.get("selected_config_id"),
                "selected_profile_id": stage132_summary.get("selected_profile_id"),
            },
            expected=_SOURCE_STAGE132_SELECTED_CONFIG,
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
            name="train_fold_count_matches_protocol_minimum",
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
    ]


def _post_validation_guard_checks(
    *,
    stage134_summary: Mapping[str, Any],
    pool_summary: Mapping[str, Any],
    report_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    train = report_payload["split_observation_reports"][_TRAIN_SPLIT]
    dev = report_payload["split_observation_reports"][_DEV_SPLIT]
    train_cv = report_payload["train_cv_validation"]
    source_invariance = report_payload["source_answer_invariance"]
    public_payload = {
        "candidate_pool_summary": pool_summary,
        "split_observation_reports": report_payload["split_observation_reports"],
        "train_fold_reports": report_payload["train_fold_reports"],
        "train_cv_validation": train_cv,
        "dev_report_observations": report_payload["dev_report_observations"],
        "source_answer_invariance": source_invariance,
    }

    def total(key: str) -> int:
        return int(train[key]) + int(dev[key])

    return [
        _check(
            name="stage135_candidate_pool_prefix_identity_preserved",
            passed=pool_summary["all_splits_prefix_identity_violation_count"] == 0,
            observed=pool_summary["all_splits_prefix_identity_violation_count"],
            expected=0,
        ),
        _check(
            name="stage135_candidate_pool_append_budget_preserved",
            passed=pool_summary["all_splits_append_budget_exceeded_count"] == 0,
            observed=pool_summary["all_splits_append_budget_exceeded_count"],
            expected=0,
        ),
        _check(
            name="stage135_primary_context_byte_identity_preserved",
            passed=total("primary_context_identity_violation_count") == 0,
            observed=total("primary_context_identity_violation_count"),
            expected=0,
        ),
        _check(
            name="stage135_answer_generation_context_identity_preserved",
            passed=total("answer_generation_context_identity_violation_count") == 0,
            observed=total("answer_generation_context_identity_violation_count"),
            expected=0,
        ),
        _check(
            name="stage135_required_runtime_record_fields_complete",
            passed=total("primary_record_field_violation_count") == 0
            and total("sidecar_record_field_violation_count") == 0,
            observed={
                "primary": total("primary_record_field_violation_count"),
                "sidecar": total("sidecar_record_field_violation_count"),
            },
            expected={"primary": 0, "sidecar": 0},
        ),
        _check(
            name="stage135_sidecar_uses_append_region_only",
            passed=total("sidecar_region_violation_count") == 0,
            observed=total("sidecar_region_violation_count"),
            expected=0,
        ),
        _check(
            name="stage135_sidecar_isolated_from_answer_generation",
            passed=total("sidecar_answer_context_leak_count") == 0,
            observed=total("sidecar_answer_context_leak_count"),
            expected=0,
        ),
        _check(
            name="stage135_sidecar_isolated_from_primary_context",
            passed=total("sidecar_primary_overlap_count") == 0,
            observed=total("sidecar_primary_overlap_count"),
            expected=0,
        ),
        _check(
            name="stage135_sidecar_slot_budget_preserved",
            passed=total("sidecar_slot_overflow_count") == 0,
            observed=total("sidecar_slot_overflow_count"),
            expected=0,
        ),
        _check(
            name="stage135_sidecar_observations_available_on_all_rows",
            passed=float(train["sidecar_observation_availability_rate"]) == 1.0
            and float(dev["sidecar_observation_availability_rate"]) == 1.0,
            observed={
                "train": train["sidecar_observation_availability_rate"],
                "dev": dev["sidecar_observation_availability_rate"],
            },
            expected={"train": 1.0, "dev": 1.0},
        ),
        _check(
            name="stage135_citation_verification_signal_observed",
            passed=int(train["sidecar_query_overlap_signal_count"]) > 0
            and int(dev["sidecar_query_overlap_signal_count"]) > 0,
            observed={
                "train": train["sidecar_query_overlap_signal_count"],
                "dev": dev["sidecar_query_overlap_signal_count"],
            },
            expected="positive on train and dev",
        ),
        _check(
            name="stage135_train_grouped_cv_integrity_passed",
            passed=bool(train_cv.get("passed")),
            observed=train_cv.get("checks"),
            expected="all train grouped-CV integrity checks pass",
        ),
        _check(
            name="stage135_source_answer_metrics_remain_invariant",
            passed=all(
                float(summary["verified_average_token_f1_delta"]) == 0.0
                and int(summary["verified_gold_citation_count_delta"]) == 0
                and int(summary["changed_verified_answer_count"]) == 0
                for summary in source_invariance.values()
            ),
            observed=source_invariance,
            expected="zero F1, gold citation, and changed-answer deltas",
        ),
        _check(
            name="stage135_dev_report_only",
            passed=report_payload["dev_report_observations"].get("dev_used_for_selection") is False
            and report_payload["dev_report_observations"].get("dev_used_for_retuning") is False,
            observed=report_payload["dev_report_observations"].get("status"),
            expected="dev report only",
        ),
        _check(
            name="stage135_direct_stage128_all400_answer_context_remains_blocked",
            passed=stage134_summary.get("direct_stage128_all400_answer_context_remains_blocked")
            is True,
            observed=stage134_summary.get("direct_stage128_all400_answer_context_remains_blocked"),
            expected=True,
        ),
        _check(
            name="stage135_test_locked",
            passed=stage134_summary.get("can_run_final_test_metrics_now") is False
            and stage134_summary.get("can_use_test_for_tuning") is False,
            observed={
                "can_run_final_test_metrics_now": stage134_summary.get(
                    "can_run_final_test_metrics_now"
                ),
                "can_use_test_for_tuning": stage134_summary.get("can_use_test_for_tuning"),
            },
            expected="test locked",
        ),
        _check(
            name="stage135_runtime_defaults_unchanged",
            passed=stage134_summary.get("default_runtime_policy") == "unchanged"
            and stage134_summary.get("runtime_defaultization_allowed_now") is False,
            observed={
                "default_runtime_policy": stage134_summary.get("default_runtime_policy"),
                "runtime_defaultization_allowed_now": stage134_summary.get(
                    "runtime_defaultization_allowed_now"
                ),
            },
            expected="unchanged",
        ),
        _check(
            name="stage135_no_fallback_strategies",
            passed=stage134_summary.get("fallback_strategies_enabled") is False,
            observed=stage134_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="stage135_public_outputs_have_no_forbidden_keys",
            passed=not _forbidden_keys_found(public_payload),
            observed=sorted(_forbidden_keys_found(public_payload)),
            expected=[],
        ),
        _check(
            name="stage135_train_cv_group_values_not_written",
            passed=train_cv.get("train_cv_group_values_written") is False,
            observed=train_cv.get("train_cv_group_values_written"),
            expected=False,
        ),
    ]


def _adapter_contract() -> dict[str, Any]:
    return {
        "adapter_id": "stage116_primary_plus_stage128_sidecar_observation_adapter_v1",
        "scoring_policy": "runtime_visible_query_overlap_plus_retrieval_prior_v1",
        "scoring_policy_tuned_on_dev": False,
        "primary_context_depth": _PRIMARY_CONTEXT_DEPTH,
        "primary_source_region": "Stage116 immutable prefix ranks 1-200",
        "sidecar_observation_slots": _SIDECAR_OBSERVATION_SLOTS,
        "sidecar_source_region": "Stage128 append ranks 201-400",
        "answer_generator_receives_primary_channel_only": True,
        "sidecar_contains_document_text": False,
        "sidecar_contains_gold_labels": False,
        "sidecar_can_generate_answer_text": False,
        "sidecar_can_replace_primary_context": False,
        "citation_verification_signal_thresholded": False,
        "primary_context_record_fields": [
            "runtime_content_handle",
            "primary_context_rank",
            "primary_context_source_region",
            "retrieval_score_summary",
        ],
        "sidecar_observation_fields": [
            "runtime_content_handle",
            "sidecar_observation_rank",
            "sidecar_source_region",
            "sidecar_route_family",
            "sidecar_score_summary",
            "citation_verification_signal",
        ],
    }


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    base = {
        "analysis_id": _ANALYSIS_ID,
        "validated_protocol_id": _SOURCE_STAGE134_PROTOCOL_ID,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "runtime_defaultization_allowed_now": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "direct_stage128_all400_answer_context_remains_blocked": True,
        "sidecar_can_generate_answer_text": False,
        "sidecar_can_replace_primary_context": False,
    }
    if failed_checks:
        return {
            **base,
            "status": "primeqa_hybrid_sidecar_observation_validation_blocked_or_failed",
            "failed_checks": failed_checks,
            "sidecar_observation_protocol_validated": False,
            "can_implement_train_dev_agent_orchestrator_now": False,
            "validated_train_dev_consumers": [],
            "recommended_next_direction": "review_stage135_sidecar_observation_failures",
        }
    return {
        **base,
        "status": "primeqa_hybrid_sidecar_observation_validation_passed",
        "failed_checks": [],
        "sidecar_observation_protocol_validated": True,
        "can_implement_train_dev_agent_orchestrator_now": True,
        "validated_train_dev_consumers": [
            "sidecar_observation_rendering",
            "citation_verification_probe",
            "evidence_gap_explanation",
        ],
        "recommended_next_direction": (
            "implement_stage116_primary_plus_sidecar_observation_agent_orchestrator_train_dev"
        ),
    }


def _blocked_report(
    *,
    stage134_protocol_path: Path,
    stage132_validation_path: Path,
    stage128_protocol_path: Path,
    stage125_protocol_path: Path,
    stage80_report_path: Path | None,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    stage134_summary: Mapping[str, Any],
    stage132_summary: Mapping[str, Any],
    stage128_summary: Mapping[str, Any],
    source_answer_invariance: Mapping[str, Any],
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
        "analysis_scope": "Stage135 blocked before candidate-pool observation validation.",
        "split_contract": _split_contract(),
        "source_files": _source_files(
            stage134_protocol_path=stage134_protocol_path,
            stage132_validation_path=stage132_validation_path,
            stage128_protocol_path=stage128_protocol_path,
            stage125_protocol_path=stage125_protocol_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
        ),
        "stage134_summary": dict(stage134_summary),
        "stage132_summary": dict(stage132_summary),
        "stage128_summary": dict(stage128_summary),
        "adapter_contract": _adapter_contract(),
        "loaded_data_summary": {
            "split_samples": summarize_primeqa_hybrid_split_samples(split_samples),
            "document_count": len(documents),
            **_section_summary(sections_by_document),
            "test_split_loaded": False,
        },
        "dense_channel_preflight": dict(dense_summary),
        "channel_catalog": [],
        "candidate_pool_summary": {},
        "split_observation_reports": {},
        "train_fold_reports": [],
        "train_cv_validation": {
            "passed": False,
            "failed_checks": [str(check["name"]) for check in guard_checks if not check["passed"]],
            "train_cv_group_values_written": False,
        },
        "dev_report_observations": {
            "status": "not_run_due_to_precheck_failure",
            "dev_used_for_selection": False,
            "dev_used_for_retuning": False,
        },
        "source_answer_invariance": dict(source_answer_invariance),
        "guard_checks": list(guard_checks),
        "decision": _decision(guard_checks),
        "timing_seconds": dict(timing_seconds),
    }


def _split_contract() -> dict[str, Any]:
    return {
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
        "selection_split": _TRAIN_SPLIT,
        "selection_mode": "grouped_cross_validation_integrity_only",
        "candidate_selection_performed": False,
        "threshold_tuning_performed": False,
        "validation_split": _DEV_SPLIT,
        "dev_selection_used": False,
        "dev_retuning_used": False,
        "forbidden_final_splits": ["test"],
    }


def _source_files(
    *,
    stage134_protocol_path: Path,
    stage132_validation_path: Path,
    stage128_protocol_path: Path,
    stage125_protocol_path: Path,
    stage80_report_path: Path | None,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
) -> dict[str, Any]:
    files = {
        "stage134_protocol": _fingerprint(stage134_protocol_path),
        "stage132_validation": _fingerprint(stage132_validation_path),
        "stage128_protocol": _fingerprint(stage128_protocol_path),
        "stage125_protocol": _fingerprint(stage125_protocol_path),
        "train_split": _fingerprint(train_split_path),
        "dev_split": _fingerprint(dev_split_path),
        "corpus_documents": _fingerprint(documents_path),
    }
    if stage80_report_path is not None:
        files["stage80_dense_cache_report"] = _fingerprint(stage80_report_path)
    return files


def _validate_options(
    *,
    train_fold_count: int,
    bm25_k1: float,
    bm25_b: float,
    encoder_batch_size: int,
) -> None:
    if train_fold_count <= 0:
        raise ValueError("train_fold_count must be positive")
    if bm25_k1 <= 0:
        raise ValueError("bm25_k1 must be positive")
    if not 0 <= bm25_b <= 1:
        raise ValueError("bm25_b must be between 0 and 1")
    if encoder_batch_size <= 0:
        raise ValueError("encoder_batch_size must be positive")


def _primary_source_region(rank: int) -> str:
    if rank <= _PRIMARY_CONTEXT_DEPTH:
        return "rank_001_010"
    return "stage116_immutable_prefix_011_200"


def _check(*, name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    runtime_records_created = bool(report.get("split_observation_reports"))
    return {
        "public_safe_summary_only": True,
        "runtime_observation_records_created_in_memory": runtime_records_created,
        "runtime_observation_records_written": False,
        "raw_question_text_written": False,
        "raw_answer_text_written": False,
        "raw_document_text_written": False,
        "raw_document_ids_written": False,
        "raw_candidate_rows_written": False,
        "raw_sample_ids_written": False,
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


def _split_signal_coverage_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for split, summary in (report.get("split_observation_reports") or {}).items():
        for key, label in (
            ("sidecar_observation_availability_rate", "observation availability"),
            ("row_query_overlap_signal_coverage", "query overlap signal"),
            ("row_novel_query_coverage_signal_coverage", "novel query coverage"),
        ):
            value = float(summary.get(key) or 0.0)
            bars.append(
                BarDatum(
                    label=f"{split} {label}",
                    value=value,
                    value_label=f"{value:.4f}",
                )
            )
    return bars


def _train_fold_signal_coverage_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=f"{fold['fold_id']} query overlap signal",
            value=float(fold["row_query_overlap_signal_coverage"]),
            value_label=f"{float(fold['row_query_overlap_signal_coverage']):.4f}",
        )
        for fold in report.get("train_fold_reports") or []
    ]


def _gold_opportunity_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for split, summary in (report.get("split_observation_reports") or {}).items():
        for key, label in (
            ("primary_context_gold_hit_count", "primary-context gold hits"),
            ("append_pool_incremental_gold_hit_count", "append incremental opportunities"),
            ("sidecar_incremental_gold_hit_count", "sidecar captured opportunities"),
        ):
            value = int(summary.get(key) or 0)
            bars.append(
                BarDatum(
                    label=f"{split} {label}",
                    value=float(value),
                    value_label=str(value),
                )
            )
    return bars


def _isolation_violation_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    keys = (
        "primary_context_identity_violation_count",
        "answer_generation_context_identity_violation_count",
        "sidecar_record_field_violation_count",
        "sidecar_region_violation_count",
        "sidecar_answer_context_leak_count",
        "sidecar_primary_overlap_count",
        "sidecar_slot_overflow_count",
    )
    return [
        BarDatum(
            label=f"{split} {key}",
            value=float(summary.get(key) or 0),
            value_label=str(int(summary.get(key) or 0)),
        )
        for split, summary in (report.get("split_observation_reports") or {}).items()
        for key in keys
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    keys = (
        "sidecar_observation_protocol_validated",
        "can_implement_train_dev_agent_orchestrator_now",
        "can_open_final_test_gate_now",
        "can_run_final_test_metrics_now",
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


def _guard_check_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(check.get("name")),
            value=1.0 if check.get("passed") else 0.0,
            value_label="passed" if check.get("passed") else "failed",
        )
        for check in report.get("guard_checks") or []
    ]
