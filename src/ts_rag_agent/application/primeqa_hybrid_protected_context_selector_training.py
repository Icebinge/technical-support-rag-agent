from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.answer_verification import AnswerVerifier
from ts_rag_agent.application.primeqa_hybrid_agent_retrieval_integration_validation import (
    _DEFAULT_COMPOSITION_POLICY,
    _DEFAULT_EVIDENCE_SELECTOR,
    _DEFAULT_MAX_CANDIDATES_PER_DOCUMENT,
    _DEFAULT_MAX_SENTENCES,
    _DEFAULT_MIN_EVIDENCE_SCORE,
    _DEFAULT_MIN_SENTENCE_SCORE,
    _answer_generator,
)
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    EncoderFactory,
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
    _load_json_object,
    _rank_union_pool,
    _special_tokens,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    CANDIDATE_POOL_DEPTH,
    CONTEXT_DEPTH,
    CONTEXT_SELECTOR_PROTOCOL_ID,
    PROTECTED_PREFIX_DEPTHS,
    RUNTIME_FEATURE_NAMES,
    ContextCandidateRecord,
    ContextSelection,
    ProtectedContextSelectorConfig,
    ProtectedPrefixContextSelector,
    ScorerFitSummary,
    create_candidate_scorer,
    frozen_stage161_selector_configs,
    records_by_sample,
    select_current_query_overlap_top10,
    select_original_rrf_top10,
)
from ts_rag_agent.application.primeqa_hybrid_second_stage_reranking_validation import (
    _rrf_scores,
    _safe_feature_name,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.application.text_metrics import token_f1
from ts_rag_agent.domain.answer import GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQADocumentSection
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.bm25_retriever import tokenize_text
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 161"
_CREATED_AT = "2026-07-19"
_ANALYSIS_ID = "primeqa_hybrid_protected_context_selector_train_cv_v1"
_TRAIN_SPLIT = "train"
_EXPECTED_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_EXPECTED_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_FOLD_COUNT = 5
_RRF_K = 60
_EXPECTED_TRAIN_ROWS = 562
_EXPECTED_ANSWERABLE_ROWS = 370
_EXPECTED_UNANSWERABLE_ROWS = 192
_EXPECTED_CANDIDATE_ROWS = _EXPECTED_TRAIN_ROWS * CANDIDATE_POOL_DEPTH
_EXPECTED_TRAIN_POOL_GOLD_HIT_COUNT = 345
_EXPECTED_TRAIN_RRF_TOP10_GOLD_HIT_COUNT = 255
_EXPECTED_CURRENT_COMPLETED_F1 = 0.1946
_EXPECTED_CURRENT_GOLD_CITATION_COUNT = 151
_QUERY_OVERLAP_CONTROL_ID = "stage160_query_overlap_top10_control"
_RRF_CONTROL_ID = "stage116_original_rrf_top10_control"
_SOURCE_STAGE119_STATUS = "primeqa_hybrid_second_stage_reranking_family_stopped"
_SOURCE_STAGE121_STATUS = (
    "primeqa_hybrid_fast_filter_screening_completed_train_cv_selected_dev_reported"
)
_SOURCE_STAGE121_SELECTED = "special_token_exact_window40_rule_selector_v1"
_SOURCE_STAGE160_STATUS = "primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_completed"
_EXPECTED_SOURCE_HASHES = {
    "stage119": "470c02a92a8c4dbc86284db3d5fd52fc22cdef987a452ae39f108fec03fb13c1",
    "stage121": "f0152eb8c9cf08d531261c7ae07cc55fce53d0896df82f6b1e81d95c4d6a766b",
    "stage160": "e17e5fe5bbc5fef4e25e41234e47b89daf19ea4ef18f3c7270601f0fee7d9377",
    "stage80": "2441bb1cb1e7888299d3f57962b18cd59df84e2086ac281105abcacfc144880f",
    "train": "cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155",
    "documents": "f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b",
}
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "answer_doc_id",
        "answer_text",
        "candidate_doc_ids",
        "cited_doc_ids",
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
        "sample_id",
        "source_doc_ids",
    }
)

ProgressSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class _DocumentFeatureProfile:
    title_tokens: frozenset[str]
    heading_tokens: frozenset[str]
    body_tokens: frozenset[str]
    short_document_tokens: frozenset[str]
    title_special_tokens: frozenset[str]
    heading_special_tokens: frozenset[str]
    body_special_tokens: frozenset[str]
    document_length_bucket: float


@dataclass(frozen=True)
class _CaseEvaluation:
    fold_id: str
    answerable: bool
    context_gold_hit: bool
    token_f1_all: float
    gold_cited: bool
    refused: bool
    tail_promotion_count: int
    protected_prefix_violation_count: int
    selected_rank_sum: int
    answer_signature: tuple[Any, ...]


@dataclass(frozen=True)
class _SelectionRun:
    selections: Mapping[str, ContextSelection]
    fit_summaries: tuple[ScorerFitSummary, ...]
    fit_seconds: float
    selection_latency_ms: tuple[float, ...]


@dataclass(frozen=True)
class PrimeQAHybridProtectedContextSelectorVisualization:
    """One generated Stage161 train-only context-selector chart."""

    name: str
    path: str


class RuntimeVisibleCandidateFeatureExtractor:
    """Cache document-side tokens and produce only runtime-visible features."""

    def __init__(
        self,
        *,
        documents_by_id: Mapping[str, PrimeQADocument],
        sections_by_document: Mapping[str, Sequence[PrimeQADocumentSection]],
    ) -> None:
        heading_text = {
            document_id: " ".join(section.section_id for section in sections)
            for document_id, sections in sections_by_document.items()
        }
        self._profiles = {
            document_id: _document_profile(document, heading_text.get(document_id, ""))
            for document_id, document in documents_by_id.items()
        }

    def extract(
        self,
        *,
        query: str,
        document_id: str,
        baseline_rank: int,
        rrf_score: float,
        route_rank_maps: Mapping[str, Mapping[str, int]],
        route_score_maps: Mapping[str, Mapping[str, float]],
    ) -> dict[str, float]:
        profile = self._profiles[document_id]
        query_tokens = set(tokenize_text(query))
        query_special_tokens = _special_tokens(query)
        route_ranks = {
            channel_id: rank_map.get(document_id)
            for channel_id, rank_map in route_rank_maps.items()
        }
        lexical_route_ids = [
            channel_id
            for channel_id in route_rank_maps
            if not channel_id.startswith("dense_cache__")
        ]
        dense_route_ids = [
            channel_id for channel_id in route_rank_maps if channel_id.startswith("dense_cache__")
        ]
        present_ranks = [rank for rank in route_ranks.values() if rank is not None]
        overlap_count = len(query_tokens & set(profile.short_document_tokens))
        overlap_ratio = overlap_count / max(1, len(query_tokens))
        retrieval_prior = 1.0 / math.log2(baseline_rank + 1)
        features = {
            "stage116_rrf_score": float(rrf_score),
            "baseline_rank_inverse": 1.0 / baseline_rank,
            "current_query_overlap_count": float(overlap_count),
            "current_query_overlap_ratio": overlap_ratio,
            "current_query_overlap_combined_score": (
                overlap_count + overlap_ratio + 0.35 * retrieval_prior
            ),
            "route_hit_count": float(len(present_ranks)),
            "lexical_route_hit_count": float(
                sum(route_ranks[channel_id] is not None for channel_id in lexical_route_ids)
            ),
            "dense_route_hit_count": float(
                sum(route_ranks[channel_id] is not None for channel_id in dense_route_ids)
            ),
            "best_route_inverse_rank": 1.0 / min(present_ranks) if present_ranks else 0.0,
            "query_title_token_overlap": float(len(query_tokens & set(profile.title_tokens))),
            "query_section_heading_overlap": float(len(query_tokens & set(profile.heading_tokens))),
            "query_token_coverage": _coverage(
                query_tokens,
                set(profile.title_tokens | profile.heading_tokens | profile.body_tokens),
            ),
            "query_body_token_coverage": _coverage(
                query_tokens,
                set(profile.body_tokens),
            ),
            "document_length_bucket": profile.document_length_bucket,
            "query_special_token_match_count": float(
                len(
                    query_special_tokens
                    & set(
                        profile.title_special_tokens
                        | profile.heading_special_tokens
                        | profile.body_special_tokens
                    )
                )
            ),
            "title_special_token_match_count": float(
                len(query_special_tokens & set(profile.title_special_tokens))
            ),
            "heading_special_token_match_count": float(
                len(query_special_tokens & set(profile.heading_special_tokens))
            ),
            "bm25_top10_indicator": float((route_ranks.get("full_document_bm25") or 999) <= 10),
        }
        for channel_id, rank in route_ranks.items():
            feature_prefix = _safe_feature_name(channel_id)
            features[f"{feature_prefix}_rank_inverse"] = 1.0 / rank if rank else 0.0
            features[f"{feature_prefix}_score"] = float(
                route_score_maps.get(channel_id, {}).get(document_id, 0.0)
            )
        for feature_name in RUNTIME_FEATURE_NAMES:
            features.setdefault(feature_name, 0.0)
        return features


class Stage161TrainCandidateDatasetBuilder:
    """Build the exact Stage116 top200 records for every frozen train row."""

    def __init__(
        self,
        *,
        documents_by_id: Mapping[str, PrimeQADocument],
        sections_by_document: Mapping[str, Sequence[PrimeQADocumentSection]],
        channels: Sequence[Any],
        fold_assignments: Mapping[str, str],
        progress_sink: ProgressSink | None = None,
        progress_stage: str = _STAGE,
    ) -> None:
        if not progress_stage.strip():
            raise ValueError("candidate builder progress stage must not be empty")
        self._documents_by_id = documents_by_id
        self._channels = tuple(channels)
        self._fold_assignments = dict(fold_assignments)
        self._progress_sink = progress_sink
        self._progress_stage = progress_stage
        self._feature_extractor = RuntimeVisibleCandidateFeatureExtractor(
            documents_by_id=documents_by_id,
            sections_by_document=sections_by_document,
        )

    def build(
        self,
        samples: Sequence[PrimeQAHybridSplitSample],
    ) -> tuple[ContextCandidateRecord, ...]:
        records: list[ContextCandidateRecord] = []
        total = len(samples)
        for index, sample in enumerate(samples, start=1):
            query = sample.to_primeqa_question().full_question
            results_by_channel = {
                channel.channel_id: channel.retriever.search(
                    query,
                    top_k=CANDIDATE_POOL_DEPTH,
                )
                for channel in self._channels
            }
            ranked_pool_doc_ids = _rank_union_pool(
                channels=self._channels,
                results_by_channel=results_by_channel,
                rrf_k=_RRF_K,
            )[:CANDIDATE_POOL_DEPTH]
            if len(ranked_pool_doc_ids) != CANDIDATE_POOL_DEPTH:
                raise ValueError("Stage161 candidate builder did not produce exact top200")
            route_rank_maps = {
                channel_id: {result.document.id: result.rank for result in results}
                for channel_id, results in results_by_channel.items()
            }
            route_score_maps = {
                channel_id: {result.document.id: result.score for result in results}
                for channel_id, results in results_by_channel.items()
            }
            rrf_scores = _rrf_scores(
                channels=self._channels,
                results_by_channel=results_by_channel,
                rrf_k=_RRF_K,
            )
            for baseline_rank, document_id in enumerate(ranked_pool_doc_ids, start=1):
                records.append(
                    ContextCandidateRecord(
                        sample_id=sample.sample_id,
                        fold_id=self._fold_assignments[sample.sample_id],
                        document_id=document_id,
                        baseline_rank=baseline_rank,
                        answerable=sample.answerable,
                        is_gold=(sample.answerable and document_id == sample.answer_doc_id),
                        features=self._feature_extractor.extract(
                            query=query,
                            document_id=document_id,
                            baseline_rank=baseline_rank,
                            rrf_score=rrf_scores.get(document_id, 0.0),
                            route_rank_maps=route_rank_maps,
                            route_score_maps=route_score_maps,
                        ),
                    )
                )
            if self._progress_sink is not None and (index % 25 == 0 or index == total):
                self._progress_sink(
                    {
                        "stage": self._progress_stage,
                        "phase": "train_candidate_pool_build",
                        "completed": index,
                        "total": total,
                    }
                )
        return tuple(records)


def run_primeqa_hybrid_protected_context_selector_training(
    *,
    stage119_report_path: Path,
    stage121_report_path: Path,
    stage160_report_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    documents_path: Path,
    user_confirmed_training: bool,
    confirmation_note: str,
    include_dense_channels: bool = True,
    encoder_batch_size: int = 64,
    encoder_device: str | None = None,
    encoder_factory: EncoderFactory | None = None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    progress_sink: ProgressSink | None = None,
) -> dict[str, Any]:
    """Run Stage161 train-only grouped-CV context-selector development."""

    started_at = time.perf_counter()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    source_authorization = _authorize_sources(
        stage119_report_path=stage119_report_path,
        stage121_report_path=stage121_report_path,
        stage160_report_path=stage160_report_path,
        stage80_report_path=stage80_report_path,
        train_split_path=train_split_path,
        documents_path=documents_path,
    )
    protocol = _frozen_protocol()
    protocol_sha256 = _canonical_json_sha256(protocol)
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="sources_authorized")

    samples = load_primeqa_hybrid_split_samples(train_split_path)
    fold_assignments = _build_train_fold_assignments(samples, fold_count=_FOLD_COUNT)
    documents_by_id = load_primeqa_documents(documents_path)
    sections_by_document = load_primeqa_document_sections(documents_path)
    documents = list(documents_by_id.values())
    loaded_at = time.perf_counter()
    _emit(progress_sink, phase="train_and_documents_loaded", train_rows=len(samples))

    stage80_report = _load_json_object(stage80_report_path)
    dense_channels, dense_summary = _build_dense_channels(
        include_dense_channels=include_dense_channels,
        stage80_report=stage80_report,
        stage80_report_path=stage80_report_path,
        documents=documents,
        document_ids=tuple(document.id for document in documents),
        encoder_batch_size=encoder_batch_size,
        encoder_device=encoder_device,
        encoder_factory=encoder_factory,
    )
    lexical_channels = _build_lexical_channels(
        documents=documents,
        sections_by_document=sections_by_document,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        component_depth=CANDIDATE_POOL_DEPTH,
    )
    channels = tuple([*lexical_channels, *dense_channels])
    channels_at = time.perf_counter()
    _emit(progress_sink, phase="retrieval_channels_ready", channel_count=len(channels))

    records = Stage161TrainCandidateDatasetBuilder(
        documents_by_id=documents_by_id,
        sections_by_document=sections_by_document,
        channels=channels,
        fold_assignments=fold_assignments,
        progress_sink=progress_sink,
    ).build(samples)
    records_at = time.perf_counter()

    grouped_records = records_by_sample(records)
    query_overlap_run = _control_selection_run(
        grouped_records=grouped_records,
        selector=select_current_query_overlap_top10,
    )
    rrf_run = _control_selection_run(
        grouped_records=grouped_records,
        selector=select_original_rrf_top10,
    )
    query_overlap_evaluation = _evaluate_selection_run(
        samples=samples,
        grouped_records=grouped_records,
        selection_run=query_overlap_run,
        documents_by_id=documents_by_id,
    )
    rrf_evaluation = _evaluate_selection_run(
        samples=samples,
        grouped_records=grouped_records,
        selection_run=rrf_run,
        documents_by_id=documents_by_id,
    )
    controls_at = time.perf_counter()
    _emit(progress_sink, phase="control_contexts_evaluated")

    config_results = []
    configs = frozen_stage161_selector_configs()
    for index, config in enumerate(configs, start=1):
        result = _evaluate_config(
            config=config,
            records=records,
            samples=samples,
            grouped_records=grouped_records,
            documents_by_id=documents_by_id,
            query_overlap_evaluation=query_overlap_evaluation,
            rrf_evaluation=rrf_evaluation,
        )
        config_results.append(result)
        _emit(
            progress_sink,
            phase="selector_config_evaluated",
            completed=index,
            total=len(configs),
            config_id=config.config_id,
        )
    evaluated_at = time.perf_counter()

    selection = _select_config(config_results)
    selected_full_train = _refit_selected_config(
        selected_config_id=selection.get("selected_config_id"),
        configs=configs,
        records=records,
        samples=samples,
        grouped_records=grouped_records,
        documents_by_id=documents_by_id,
    )
    candidate_summary = _candidate_pool_summary(records, samples)
    fold_summary = _fold_assignment_summary(samples, fold_assignments)
    guarded_at = time.perf_counter()

    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train-only grouped five-fold development for a protected original-RRF-prefix "
            "generation-context selector over the frozen Stage116 top200 candidate pool. "
            "The run compares the current query-overlap Top10 and untouched RRF Top10 "
            "controls, evaluates six user-authorized lightweight selectors, loads neither "
            "dev nor test, performs no runtime defaultization, and enables no fallback."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_training),
            "selected_route": "A",
            "confirmation_note": confirmation_note,
        },
        "source_authorization": source_authorization,
        "frozen_protocol": protocol,
        "frozen_protocol_sha256": protocol_sha256,
        "split_contract": {
            "split_name": _EXPECTED_SPLIT_NAME,
            "protocol_version": _EXPECTED_PROTOCOL_VERSION,
            "loaded_split": _TRAIN_SPLIT,
            "fit_split": _TRAIN_SPLIT,
            "selection_mode": "train_grouped_five_fold_out_of_fold_only",
            "group_key": "normalized_question_plus_answer_document_or_unanswerable",
            "dev_split_loaded": False,
            "test_split_loaded": False,
            "dev_metrics_run": False,
            "test_metrics_run": False,
        },
        "analysis_config": {
            "candidate_pool_depth": CANDIDATE_POOL_DEPTH,
            "context_depth": CONTEXT_DEPTH,
            "fold_count": _FOLD_COUNT,
            "rrf_k": _RRF_K,
            "include_dense_channels": include_dense_channels,
            "encoder_batch_size": encoder_batch_size,
            "encoder_device": encoder_device or "configured_default",
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "candidate_config_count": len(configs),
            "runtime_feature_count": len(RUNTIME_FEATURE_NAMES),
        },
        "loaded_data_summary": {
            "train_row_count": len(samples),
            "train_answerable_count": sum(sample.answerable for sample in samples),
            "train_unanswerable_count": sum(not sample.answerable for sample in samples),
            "document_count": len(documents_by_id),
            "section_count": sum(len(value) for value in sections_by_document.values()),
            "dev_rows_loaded": 0,
            "test_rows_loaded": 0,
            "raw_candidate_rows_written": False,
        },
        "grouped_fold_summary": fold_summary,
        "dense_channel_preflight": dense_summary,
        "candidate_pool_summary": candidate_summary,
        "control_results": {
            _QUERY_OVERLAP_CONTROL_ID: _public_evaluation(query_overlap_evaluation),
            _RRF_CONTROL_ID: _public_evaluation(rrf_evaluation),
        },
        "config_results": config_results,
        "train_cv_selection": selection,
        "selected_full_train_refit": selected_full_train,
        "closed_boundaries": {
            "dev_loaded": False,
            "test_loaded": False,
            "dev_used_for_fit_or_selection": False,
            "test_used_for_fit_selection_or_metrics": False,
            "runtime_registered_as_default": False,
            "runtime_integration_run": False,
            "model_artifact_written": False,
            "fallback_strategies_enabled": False,
            "query_rewrite_enabled": False,
            "second_retrieval_enabled": False,
        },
        "timing_seconds": {
            "source_authorization_and_protocol": round(authorized_at - started_at, 6),
            "load_train_and_documents": round(loaded_at - authorized_at, 6),
            "build_retrieval_channels": round(channels_at - loaded_at, 6),
            "build_train_candidate_records": round(records_at - channels_at, 6),
            "evaluate_controls": round(controls_at - records_at, 6),
            "evaluate_six_oof_configs": round(evaluated_at - controls_at, 6),
            "selection_and_full_train_refit": round(guarded_at - evaluated_at, 6),
            "total": round(guarded_at - started_at, 6),
        },
    }
    report["guard_checks"] = _guard_checks(report)
    report["public_safe_contract"] = _public_safe_contract(report)
    passed = all(check["passed"] for check in report["guard_checks"])
    report["decision"] = _decision(report=report, guards_passed=passed)
    return report


def write_stage161_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridProtectedContextSelectorVisualization]:
    """Write ten public-safe SVG views for Stage161."""

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_specs = {
        "stage161_context_gold_hit_rate.svg": (
            "Stage161 train OOF context gold hit rate",
            _metric_bars(report, "context_gold_hit_rate"),
        ),
        "stage161_verified_f1_all.svg": (
            "Stage161 train OOF verified token F1 over all answerable rows",
            _metric_bars(report, "average_token_f1_all_answerable"),
        ),
        "stage161_gold_citation_count.svg": (
            "Stage161 train OOF gold citation count",
            _metric_bars(report, "gold_citation_count"),
        ),
        "stage161_answerable_refusal_count.svg": (
            "Stage161 train OOF answerable refusal count",
            _metric_bars(report, "answerable_refusal_count"),
        ),
        "stage161_unanswerable_false_answer_count.svg": (
            "Stage161 train OOF unanswerable false-answer count",
            _metric_bars(report, "unanswerable_false_answer_count"),
        ),
        "stage161_tail_promotions.svg": (
            "Stage161 average tail promotions into generation Top10",
            _metric_bars(report, "average_tail_promotion_count"),
        ),
        "stage161_selection_latency.svg": (
            "Stage161 selector-only average latency",
            _metric_bars(report, "selection_latency_average_ms"),
        ),
        "stage161_fold_hit_delta.svg": (
            "Stage161 minimum fold hit-rate delta vs current control",
            _config_guard_value_bars(report, "minimum_fold_hit_rate_delta"),
        ),
        "stage161_config_guard_status.svg": (
            "Stage161 train-CV config guard status",
            _config_selectable_bars(report),
        ),
        "stage161_guard_check_status.svg": (
            "Stage161 process guard checks",
            [
                _bar(str(check["name"]), bool(check["passed"]))
                for check in report.get("guard_checks", [])
            ],
        ),
    }
    artifacts = []
    for filename, (title, bars) in chart_specs.items():
        path = output_dir / filename
        path.write_text(
            render_horizontal_bar_chart_svg(title=title, bars=bars, x_label="value"),
            encoding="utf-8",
        )
        artifacts.append(
            PrimeQAHybridProtectedContextSelectorVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _evaluate_config(
    *,
    config: ProtectedContextSelectorConfig,
    records: Sequence[ContextCandidateRecord],
    samples: Sequence[PrimeQAHybridSplitSample],
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
    documents_by_id: Mapping[str, PrimeQADocument],
    query_overlap_evaluation: Mapping[str, Any],
    rrf_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        selection_run = _out_of_fold_selection_run(config=config, records=records)
        evaluation = _evaluate_selection_run(
            samples=samples,
            grouped_records=grouped_records,
            selection_run=selection_run,
            documents_by_id=documents_by_id,
        )
        comparison = _comparison(
            evaluation=evaluation,
            current=query_overlap_evaluation,
            rrf=rrf_evaluation,
        )
        guard_results = _config_guard_results(
            evaluation=evaluation,
            current=query_overlap_evaluation,
            rrf=rrf_evaluation,
            comparison=comparison,
        )
        selectable = all(guard_results.values())
        return {
            "config": asdict(config),
            "training_status": "completed",
            "fit_summaries": [asdict(item) for item in selection_run.fit_summaries],
            "fit_seconds": round(selection_run.fit_seconds, 6),
            "train_oof_metrics": _public_evaluation(evaluation),
            "comparison": comparison,
            "guard_results": guard_results,
            "train_cv_selectable": selectable,
        }
    except Exception as error:
        return {
            "config": asdict(config),
            "training_status": "failed",
            "training_error": f"{type(error).__name__}: {error}",
            "fit_summaries": [],
            "fit_seconds": 0.0,
            "train_oof_metrics": None,
            "comparison": None,
            "guard_results": {"training_completed": False},
            "train_cv_selectable": False,
        }


def _out_of_fold_selection_run(
    *,
    config: ProtectedContextSelectorConfig,
    records: Sequence[ContextCandidateRecord],
) -> _SelectionRun:
    fold_ids = sorted({record.fold_id for record in records})
    selections: dict[str, ContextSelection] = {}
    summaries = []
    selection_latencies = []
    fit_started = time.perf_counter()
    for fold_id in fold_ids:
        fit_records = [record for record in records if record.fold_id != fold_id]
        validation_records = [record for record in records if record.fold_id == fold_id]
        scorer = create_candidate_scorer(config.model_family)
        summaries.append(
            scorer.fit(
                fit_records,
                protected_prefix_depth=config.protected_prefix_depth,
            )
        )
        selector = ProtectedPrefixContextSelector(config=config, scorer=scorer)
        for sample_id, sample_records in records_by_sample(validation_records).items():
            selected_at = time.perf_counter()
            selections[sample_id] = selector.select(sample_records)
            selection_latencies.append((time.perf_counter() - selected_at) * 1000.0)
    expected_ids = set(records_by_sample(records))
    if set(selections) != expected_ids:
        raise RuntimeError("Stage161 out-of-fold selection did not cover every train row")
    return _SelectionRun(
        selections=selections,
        fit_summaries=tuple(summaries),
        fit_seconds=time.perf_counter() - fit_started,
        selection_latency_ms=tuple(selection_latencies),
    )


def _control_selection_run(
    *,
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
    selector: Callable[[Sequence[ContextCandidateRecord]], ContextSelection],
) -> _SelectionRun:
    selections = {}
    latencies = []
    for sample_id, sample_records in grouped_records.items():
        started_at = time.perf_counter()
        selections[sample_id] = selector(sample_records)
        latencies.append((time.perf_counter() - started_at) * 1000.0)
    return _SelectionRun(
        selections=selections,
        fit_summaries=(),
        fit_seconds=0.0,
        selection_latency_ms=tuple(latencies),
    )


def _evaluate_selection_run(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
    selection_run: _SelectionRun,
    documents_by_id: Mapping[str, PrimeQADocument],
) -> dict[str, Any]:
    generator = _answer_generator(
        evidence_selector_name=_DEFAULT_EVIDENCE_SELECTOR,
        max_candidates_per_document=_DEFAULT_MAX_CANDIDATES_PER_DOCUMENT,
        composition_policy_name=_DEFAULT_COMPOSITION_POLICY,
        max_sentences=_DEFAULT_MAX_SENTENCES,
        min_sentence_score=_DEFAULT_MIN_SENTENCE_SCORE,
    )
    verifier = AnswerVerifier(
        min_citations=1,
        min_evidence_score=_DEFAULT_MIN_EVIDENCE_SCORE,
        max_citation_rank=CANDIDATE_POOL_DEPTH,
    )
    cases: dict[str, _CaseEvaluation] = {}
    for sample in samples:
        sample_records = grouped_records[sample.sample_id]
        selection = selection_run.selections[sample.sample_id]
        context = _retrieval_results(selection.selected, documents_by_id)
        verification_context = _retrieval_results(sample_records, documents_by_id)
        question = sample.to_primeqa_question()
        original = generator.generate(question, context)
        verified = verifier.verify(original, verification_context).verified_answer
        gold_cited = bool(
            sample.answerable
            and sample.answer_doc_id in {citation.document_id for citation in verified.citations}
        )
        cases[sample.sample_id] = _CaseEvaluation(
            fold_id=sample_records[0].fold_id,
            answerable=sample.answerable,
            context_gold_hit=bool(
                sample.answerable and any(record.is_gold for record in selection.selected)
            ),
            token_f1_all=(0.0 if verified.refused else token_f1(verified.answer, sample.answer))
            if sample.answerable
            else 0.0,
            gold_cited=gold_cited,
            refused=verified.refused,
            tail_promotion_count=selection.tail_promotion_count,
            protected_prefix_violation_count=(selection.protected_prefix_violation_count),
            selected_rank_sum=sum(record.baseline_rank for record in selection.selected),
            answer_signature=_answer_signature(verified),
        )
    aggregate = _aggregate_cases(tuple(cases.values()))
    folds = {
        fold_id: _aggregate_cases(tuple(case for case in cases.values() if case.fold_id == fold_id))
        for fold_id in sorted({case.fold_id for case in cases.values()})
    }
    return {
        "aggregate": {
            **aggregate,
            "selection_latency_average_ms": _mean(selection_run.selection_latency_ms),
            "selection_latency_p95_ms": _percentile(
                selection_run.selection_latency_ms,
                0.95,
            ),
        },
        "folds": folds,
        "private_cases": cases,
    }


def _aggregate_cases(cases: Sequence[_CaseEvaluation]) -> dict[str, Any]:
    answerable = [case for case in cases if case.answerable]
    unanswerable = [case for case in cases if not case.answerable]
    completed_answerable = [case for case in answerable if not case.refused]
    return {
        "case_count": len(cases),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "context_gold_hit_count": sum(case.context_gold_hit for case in answerable),
        "context_gold_hit_rate": _ratio(
            sum(case.context_gold_hit for case in answerable),
            len(answerable),
        ),
        "average_token_f1_all_answerable": _mean([case.token_f1_all for case in answerable]),
        "average_token_f1_completed_answerable": _mean(
            [case.token_f1_all for case in completed_answerable]
        ),
        "gold_citation_count": sum(case.gold_cited for case in answerable),
        "gold_citation_rate_all_answerable": _ratio(
            sum(case.gold_cited for case in answerable),
            len(answerable),
        ),
        "answerable_refusal_count": sum(case.refused for case in answerable),
        "answerable_refusal_rate": _ratio(
            sum(case.refused for case in answerable),
            len(answerable),
        ),
        "unanswerable_false_answer_count": sum(not case.refused for case in unanswerable),
        "unanswerable_false_answer_rate": _ratio(
            sum(not case.refused for case in unanswerable),
            len(unanswerable),
        ),
        "protected_prefix_violation_count": sum(
            case.protected_prefix_violation_count for case in cases
        ),
        "average_tail_promotion_count": _mean([case.tail_promotion_count for case in cases]),
        "average_selected_baseline_rank": _ratio(
            sum(case.selected_rank_sum for case in cases),
            len(cases) * CONTEXT_DEPTH,
        ),
    }


def _comparison(
    *,
    evaluation: Mapping[str, Any],
    current: Mapping[str, Any],
    rrf: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = evaluation["aggregate"]
    current_metrics = current["aggregate"]
    rrf_metrics = rrf["aggregate"]
    candidate_cases = evaluation["private_cases"]
    current_cases = current["private_cases"]
    improved = 0
    regressed = 0
    tied = 0
    changed_answers = 0
    for sample_id, candidate in candidate_cases.items():
        baseline = current_cases[sample_id]
        if candidate.answerable:
            if candidate.token_f1_all > baseline.token_f1_all + 1e-12:
                improved += 1
            elif candidate.token_f1_all + 1e-12 < baseline.token_f1_all:
                regressed += 1
            else:
                tied += 1
        changed_answers += candidate.answer_signature != baseline.answer_signature
    fold_deltas = {
        fold_id: {
            "context_gold_hit_rate_delta": round(
                float(fold_metrics["context_gold_hit_rate"])
                - float(current["folds"][fold_id]["context_gold_hit_rate"]),
                6,
            ),
            "average_token_f1_all_answerable_delta": round(
                float(fold_metrics["average_token_f1_all_answerable"])
                - float(current["folds"][fold_id]["average_token_f1_all_answerable"]),
                6,
            ),
        }
        for fold_id, fold_metrics in evaluation["folds"].items()
    }
    return {
        "vs_current_query_overlap": {
            "context_gold_hit_count_delta": int(metrics["context_gold_hit_count"])
            - int(current_metrics["context_gold_hit_count"]),
            "context_gold_hit_rate_delta": round(
                float(metrics["context_gold_hit_rate"])
                - float(current_metrics["context_gold_hit_rate"]),
                6,
            ),
            "average_token_f1_all_answerable_delta": round(
                float(metrics["average_token_f1_all_answerable"])
                - float(current_metrics["average_token_f1_all_answerable"]),
                6,
            ),
            "gold_citation_count_delta": int(metrics["gold_citation_count"])
            - int(current_metrics["gold_citation_count"]),
            "answerable_refusal_count_delta": int(metrics["answerable_refusal_count"])
            - int(current_metrics["answerable_refusal_count"]),
            "unanswerable_false_answer_count_delta": int(metrics["unanswerable_false_answer_count"])
            - int(current_metrics["unanswerable_false_answer_count"]),
            "answerable_f1_improved_count": improved,
            "answerable_f1_regressed_count": regressed,
            "answerable_f1_tied_count": tied,
            "changed_verified_answer_count": changed_answers,
        },
        "vs_original_rrf_top10": {
            "context_gold_hit_count_delta": int(metrics["context_gold_hit_count"])
            - int(rrf_metrics["context_gold_hit_count"]),
            "context_gold_hit_rate_delta": round(
                float(metrics["context_gold_hit_rate"])
                - float(rrf_metrics["context_gold_hit_rate"]),
                6,
            ),
        },
        "fold_deltas_vs_current": fold_deltas,
        "minimum_fold_hit_rate_delta": min(
            value["context_gold_hit_rate_delta"] for value in fold_deltas.values()
        ),
        "minimum_fold_f1_delta": min(
            value["average_token_f1_all_answerable_delta"] for value in fold_deltas.values()
        ),
    }


def _config_guard_results(
    *,
    evaluation: Mapping[str, Any],
    current: Mapping[str, Any],
    rrf: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, bool]:
    metrics = evaluation["aggregate"]
    current_metrics = current["aggregate"]
    rrf_metrics = rrf["aggregate"]
    current_delta = comparison["vs_current_query_overlap"]
    return {
        "training_completed": True,
        "context_hit_strictly_improves_current": int(current_delta["context_gold_hit_count_delta"])
        > 0,
        "context_hit_not_below_original_rrf_top10": int(metrics["context_gold_hit_count"])
        >= int(rrf_metrics["context_gold_hit_count"]),
        "verified_f1_all_not_below_current": float(metrics["average_token_f1_all_answerable"])
        + 1e-12
        >= float(current_metrics["average_token_f1_all_answerable"]),
        "gold_citations_not_below_current": int(metrics["gold_citation_count"])
        >= int(current_metrics["gold_citation_count"]),
        "answerable_refusals_not_above_current": int(metrics["answerable_refusal_count"])
        <= int(current_metrics["answerable_refusal_count"]),
        "unanswerable_false_answers_not_above_current": int(
            metrics["unanswerable_false_answer_count"]
        )
        <= int(current_metrics["unanswerable_false_answer_count"]),
        "protected_prefix_identity_exact": int(metrics["protected_prefix_violation_count"]) == 0,
        "every_fold_hit_not_below_current": float(comparison["minimum_fold_hit_rate_delta"]) >= 0.0,
        "every_fold_f1_not_below_current": float(comparison["minimum_fold_f1_delta"]) >= 0.0,
    }


def _select_config(config_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selectable = [result for result in config_results if result["train_cv_selectable"]]
    if not selectable:
        return {
            "status": "no_train_cv_safe_config",
            "selected_config_id": None,
            "selectable_config_count": 0,
            "selection_evidence": "train_grouped_five_fold_out_of_fold_only",
            "dev_used": False,
            "test_used": False,
        }
    selected = sorted(
        selectable,
        key=lambda result: (
            -int(result["train_oof_metrics"]["aggregate"]["context_gold_hit_count"]),
            -float(result["train_oof_metrics"]["aggregate"]["average_token_f1_all_answerable"]),
            -int(result["train_oof_metrics"]["aggregate"]["gold_citation_count"]),
            int(result["train_oof_metrics"]["aggregate"]["unanswerable_false_answer_count"]),
            float(result["train_oof_metrics"]["aggregate"]["average_tail_promotion_count"]),
            float(result["train_oof_metrics"]["aggregate"]["selection_latency_average_ms"]),
            str(result["config"]["config_id"]),
        ),
    )[0]
    return {
        "status": "train_cv_safe_config_selected",
        "selected_config_id": selected["config"]["config_id"],
        "selectable_config_count": len(selectable),
        "selection_evidence": "train_grouped_five_fold_out_of_fold_only",
        "selection_order": [
            "context_gold_hit_count_desc",
            "verified_f1_all_desc",
            "gold_citation_count_desc",
            "unanswerable_false_answer_count_asc",
            "tail_promotions_asc",
            "selection_latency_asc",
            "config_id_asc",
        ],
        "dev_used": False,
        "test_used": False,
    }


def _refit_selected_config(
    *,
    selected_config_id: str | None,
    configs: Sequence[ProtectedContextSelectorConfig],
    records: Sequence[ContextCandidateRecord],
    samples: Sequence[PrimeQAHybridSplitSample],
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
    documents_by_id: Mapping[str, PrimeQADocument],
) -> dict[str, Any]:
    if selected_config_id is None:
        return {
            "status": "not_run_no_train_cv_safe_config",
            "selected_config_id": None,
            "used_for_selection": False,
            "model_artifact_written": False,
        }
    config = next(config for config in configs if config.config_id == selected_config_id)
    scorer = create_candidate_scorer(config.model_family)
    fit_started = time.perf_counter()
    fit_summary = scorer.fit(records, protected_prefix_depth=config.protected_prefix_depth)
    selector = ProtectedPrefixContextSelector(config=config, scorer=scorer)
    selections = {}
    latencies = []
    for sample_id, sample_records in grouped_records.items():
        selected_at = time.perf_counter()
        selections[sample_id] = selector.select(sample_records)
        latencies.append((time.perf_counter() - selected_at) * 1000.0)
    evaluation = _evaluate_selection_run(
        samples=samples,
        grouped_records=grouped_records,
        selection_run=_SelectionRun(
            selections=selections,
            fit_summaries=(fit_summary,),
            fit_seconds=time.perf_counter() - fit_started,
            selection_latency_ms=tuple(latencies),
        ),
        documents_by_id=documents_by_id,
    )
    return {
        "status": "full_train_refit_completed_diagnostic_only",
        "selected_config_id": selected_config_id,
        "fit_summary": asdict(fit_summary),
        "metrics": _public_evaluation(evaluation),
        "used_for_selection": False,
        "model_artifact_written": False,
    }


def _candidate_pool_summary(
    records: Sequence[ContextCandidateRecord],
    samples: Sequence[PrimeQAHybridSplitSample],
) -> dict[str, Any]:
    grouped = records_by_sample(records)
    answerable_ids = {sample.sample_id for sample in samples if sample.answerable}
    pool_hit = sum(
        any(record.is_gold for record in grouped[sample_id]) for sample_id in answerable_ids
    )
    return {
        "candidate_record_count_in_memory": len(records),
        "sample_pool_count": len(grouped),
        "minimum_pool_depth": min(len(value) for value in grouped.values()),
        "maximum_pool_depth": max(len(value) for value in grouped.values()),
        "answerable_gold_pool_hit_count": pool_hit,
        "answerable_gold_pool_hit_rate": _ratio(pool_hit, len(answerable_ids)),
        "raw_candidate_rows_written": False,
    }


def _fold_assignment_summary(
    samples: Sequence[PrimeQAHybridSplitSample],
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    row_counts = Counter(assignments.values())
    answerable_counts = Counter(
        assignments[sample.sample_id] for sample in samples if sample.answerable
    )
    group_to_fold: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        group_to_fold[_group_key(sample)].add(assignments[sample.sample_id])
    return {
        "fold_count": len(row_counts),
        "row_counts": dict(sorted(row_counts.items())),
        "answerable_counts": dict(sorted(answerable_counts.items())),
        "group_count": len(group_to_fold),
        "cross_fold_group_violation_count": sum(
            len(folds) != 1 for folds in group_to_fold.values()
        ),
        "assignment_sha256": hashlib.sha256(
            "\n".join(
                f"{_sha256_text(sample.sample_id)}:{assignments[sample.sample_id]}"
                for sample in sorted(samples, key=lambda item: item.sample_id)
            ).encode("ascii")
        ).hexdigest(),
        "raw_group_values_written": False,
    }


def _frozen_protocol() -> dict[str, Any]:
    configs = frozen_stage161_selector_configs()
    return {
        "protocol_id": CONTEXT_SELECTOR_PROTOCOL_ID,
        "selection_split": "train",
        "selection_mode": "grouped_five_fold_out_of_fold_only",
        "candidate_pool_source": "stage116_original_rrf_top200",
        "candidate_pool_depth": CANDIDATE_POOL_DEPTH,
        "generation_context_depth": CONTEXT_DEPTH,
        "controls": [_QUERY_OVERLAP_CONTROL_ID, _RRF_CONTROL_ID],
        "candidate_configs": [asdict(config) for config in configs],
        "runtime_features": list(RUNTIME_FEATURE_NAMES),
        "training_labels": "train_gold_document_membership_only",
        "hard_negative_contract": {
            "baseline_rank_hard_negatives": 20,
            "query_overlap_hard_negatives": 20,
            "deduplicated_by_document": True,
        },
        "strict_selection_guards": [
            "context_hit_strictly_improves_current",
            "context_hit_not_below_original_rrf_top10",
            "verified_f1_all_not_below_current",
            "gold_citations_not_below_current",
            "answerable_refusals_not_above_current",
            "unanswerable_false_answers_not_above_current",
            "protected_prefix_identity_exact",
            "every_fold_hit_not_below_current",
            "every_fold_f1_not_below_current",
        ],
        "blocked": {
            "dev_load": True,
            "test_load": True,
            "dev_selection": True,
            "test_metrics": True,
            "full_top200_rerank": True,
            "fallback": True,
            "runtime_defaultization": True,
        },
    }


def _authorize_sources(
    *,
    stage119_report_path: Path,
    stage121_report_path: Path,
    stage160_report_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    documents_path: Path,
) -> dict[str, Any]:
    paths = {
        "stage119": stage119_report_path,
        "stage121": stage121_report_path,
        "stage160": stage160_report_path,
        "stage80": stage80_report_path,
        "train": train_split_path,
        "documents": documents_path,
    }
    fingerprints = {name: _fingerprint(path) for name, path in paths.items()}
    mismatches = {
        name: fingerprint["sha256"]
        for name, fingerprint in fingerprints.items()
        if fingerprint["sha256"] != _EXPECTED_SOURCE_HASHES[name]
    }
    if mismatches:
        raise ValueError(f"Stage161 source fingerprint mismatch: {mismatches}")
    stage119 = _load_json_object(stage119_report_path)
    stage121 = _load_json_object(stage121_report_path)
    stage160 = _load_json_object(stage160_report_path)
    if (stage119.get("decision") or {}).get("status") != _SOURCE_STAGE119_STATUS:
        raise ValueError("Stage161 requires the frozen Stage119 stop decision")
    if (stage121.get("decision") or {}).get("status") != _SOURCE_STAGE121_STATUS:
        raise ValueError("Stage161 requires the completed Stage121 screening report")
    if (stage121.get("decision") or {}).get("selected_config_id") != _SOURCE_STAGE121_SELECTED:
        raise ValueError("Stage161 requires the Stage121 no-promotion selected control")
    if (stage160.get("decision") or {}).get("status") != _SOURCE_STAGE160_STATUS:
        raise ValueError("Stage161 requires the completed Stage160 failure diagnosis")
    if (stage160.get("decision") or {}).get("dominant_answerable_refusal_mechanism") != (
        "generation_top10_loss"
    ):
        raise ValueError("Stage161 requires Stage160 generation Top10 loss evidence")
    return {
        "fingerprints": fingerprints,
        "stage119_status": _SOURCE_STAGE119_STATUS,
        "stage121_status": _SOURCE_STAGE121_STATUS,
        "stage121_selected_config_id": _SOURCE_STAGE121_SELECTED,
        "stage160_status": _SOURCE_STAGE160_STATUS,
        "stage160_dominant_mechanism": "generation_top10_loss",
    }


def _guard_checks(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    loaded = report["loaded_data_summary"]
    folds = report["grouped_fold_summary"]
    pool = report["candidate_pool_summary"]
    controls = report["control_results"]
    current = controls[_QUERY_OVERLAP_CONTROL_ID]["aggregate"]
    rrf = controls[_RRF_CONTROL_ID]["aggregate"]
    protocol = report["frozen_protocol"]
    selection = report["train_cv_selection"]
    boundaries = report["closed_boundaries"]
    config_results = report["config_results"]
    checks = [
        _check("user_confirmed_route_a", report["user_confirmation"]["confirmed"] is True),
        _check(
            "frozen_protocol_identity_exact",
            protocol.get("protocol_id") == CONTEXT_SELECTOR_PROTOCOL_ID
            and report.get("frozen_protocol_sha256") == _canonical_json_sha256(protocol),
        ),
        _check(
            "train_source_exact",
            report["source_authorization"]["fingerprints"]["train"]["sha256"]
            == _EXPECTED_SOURCE_HASHES["train"],
        ),
        _check(
            "only_train_loaded",
            loaded["train_row_count"] == _EXPECTED_TRAIN_ROWS
            and loaded["dev_rows_loaded"] == 0
            and loaded["test_rows_loaded"] == 0,
        ),
        _check(
            "train_answerability_exact",
            loaded["train_answerable_count"] == _EXPECTED_ANSWERABLE_ROWS
            and loaded["train_unanswerable_count"] == _EXPECTED_UNANSWERABLE_ROWS,
        ),
        _check(
            "grouped_five_fold_isolation_exact",
            folds["fold_count"] == _FOLD_COUNT and folds["cross_fold_group_violation_count"] == 0,
        ),
        _check(
            "candidate_pool_depth_exact",
            pool["candidate_record_count_in_memory"] == _EXPECTED_CANDIDATE_ROWS
            and pool["minimum_pool_depth"] == CANDIDATE_POOL_DEPTH
            and pool["maximum_pool_depth"] == CANDIDATE_POOL_DEPTH,
        ),
        _check(
            "stage116_train_pool_reproduced",
            pool["answerable_gold_pool_hit_count"] == _EXPECTED_TRAIN_POOL_GOLD_HIT_COUNT,
        ),
        _check(
            "stage116_rrf_top10_reproduced",
            rrf["context_gold_hit_count"] == _EXPECTED_TRAIN_RRF_TOP10_GOLD_HIT_COUNT,
        ),
        _check(
            "current_query_overlap_answer_pipeline_reproduced",
            abs(
                float(current["average_token_f1_completed_answerable"])
                - _EXPECTED_CURRENT_COMPLETED_F1
            )
            <= 0.00005
            and current["gold_citation_count"] == _EXPECTED_CURRENT_GOLD_CITATION_COUNT,
        ),
        _check(
            "six_authorized_configs_exact",
            len(config_results) == 6
            and {result["config"]["protected_prefix_depth"] for result in config_results}
            == set(PROTECTED_PREFIX_DEPTHS)
            and {result["config"]["model_family"] for result in config_results}
            == {"pairwise_logistic", "pointwise_histogram_gbdt"},
        ),
        _check(
            "runtime_feature_contract_has_no_gold",
            all("gold" not in feature.lower() for feature in protocol["runtime_features"]),
        ),
        _check(
            "train_oof_is_only_selection_evidence",
            selection["selection_evidence"] == "train_grouped_five_fold_out_of_fold_only"
            and selection["dev_used"] is False
            and selection["test_used"] is False,
        ),
        _check(
            "selected_config_is_train_cv_safe_or_none",
            selection["selected_config_id"] is None
            or any(
                result["config"]["config_id"] == selection["selected_config_id"]
                and result["train_cv_selectable"] is True
                for result in config_results
            ),
        ),
        _check("raw_candidate_rows_not_written", loaded["raw_candidate_rows_written"] is False),
        _check(
            "dev_test_metrics_closed",
            boundaries["dev_loaded"] is False
            and boundaries["test_loaded"] is False
            and boundaries["dev_used_for_fit_or_selection"] is False
            and boundaries["test_used_for_fit_selection_or_metrics"] is False,
        ),
        _check(
            "runtime_and_fallback_closed",
            boundaries["runtime_registered_as_default"] is False
            and boundaries["runtime_integration_run"] is False
            and boundaries["fallback_strategies_enabled"] is False,
        ),
        _check(
            "rewrite_and_second_retrieval_closed",
            boundaries["query_rewrite_enabled"] is False
            and boundaries["second_retrieval_enabled"] is False,
        ),
    ]
    return checks


def _decision(*, report: Mapping[str, Any], guards_passed: bool) -> dict[str, Any]:
    selection = report["train_cv_selection"]
    selected = selection.get("selected_config_id")
    if not guards_passed:
        status = "primeqa_hybrid_protected_context_selector_training_invalid"
        next_direction = "repair_stage161_process_guards_before_any_further_evaluation"
    elif selected is None:
        status = "primeqa_hybrid_protected_context_selector_no_train_cv_safe_config"
        next_direction = "analyze_train_oof_selector_failures_without_opening_dev_or_test"
    else:
        status = "primeqa_hybrid_protected_context_selector_train_cv_selected"
        next_direction = "freeze_selected_selector_then_run_one_shot_dev_validation"
    return {
        "status": status,
        "all_process_guards_passed": guards_passed,
        "failed_process_guards": [
            check["name"] for check in report["guard_checks"] if not check["passed"]
        ],
        "selected_config_id": selected,
        "selectable_config_count": selection["selectable_config_count"],
        "dev_gate_opened": False,
        "test_gate_opened": False,
        "runtime_registered_as_default": False,
        "fallback_strategies_enabled": False,
        "next_direction": next_direction,
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = sorted(_find_forbidden_keys(report))
    return {
        "forbidden_keys_found": forbidden,
        "contains_case_rows": False,
        "contains_raw_question": False,
        "contains_raw_answer": False,
        "contains_raw_document": False,
        "raw_candidate_rows_written": False,
        "public_safe": not forbidden,
    }


def _public_evaluation(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "aggregate": dict(evaluation["aggregate"]),
        "folds": {fold_id: dict(metrics) for fold_id, metrics in evaluation["folds"].items()},
    }


def _retrieval_results(
    records: Sequence[ContextCandidateRecord],
    documents_by_id: Mapping[str, PrimeQADocument],
) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            document=documents_by_id[record.document_id],
            score=float(record.features.get("stage116_rrf_score", 0.0)),
            rank=record.baseline_rank,
        )
        for record in records
    ]


def _answer_signature(answer: GeneratedAnswer) -> tuple[Any, ...]:
    return (
        answer.refused,
        answer.answer,
        tuple(
            (citation.document_id, citation.retrieval_rank, citation.evidence_score)
            for citation in answer.citations
        ),
    )


def _document_profile(document: PrimeQADocument, heading_text: str) -> _DocumentFeatureProfile:
    title_tokens = frozenset(tokenize_text(document.title))
    heading_tokens = frozenset(tokenize_text(heading_text))
    body_tokens = frozenset(tokenize_text(document.text))
    short_document_tokens = frozenset(tokenize_text(f"{document.title}\n{document.text[:5000]}"))
    return _DocumentFeatureProfile(
        title_tokens=title_tokens,
        heading_tokens=heading_tokens,
        body_tokens=body_tokens,
        short_document_tokens=short_document_tokens,
        title_special_tokens=frozenset(_special_tokens(document.title)),
        heading_special_tokens=frozenset(_special_tokens(heading_text)),
        body_special_tokens=frozenset(_special_tokens(document.text)),
        document_length_bucket=min(len(body_tokens) / 1000.0, 5.0),
    )


def _group_key(sample: PrimeQAHybridSplitSample) -> str:
    normalized_question = " ".join(
        f"{sample.question_title} {sample.question_text}".lower().split()
    )
    document_marker = sample.answer_doc_id if sample.answerable else "UNANSWERABLE"
    return f"{normalized_question}::{document_marker}"


def _coverage(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left) if left else 0.0


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _mean(values: Sequence[int | float]) -> float:
    return round(sum(float(value) for value in values) / len(values), 6) if values else 0.0


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return round(float(ordered[index]), 6)


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _find_forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in _FORBIDDEN_PUBLIC_KEYS:
                found.add(str(key))
            found.update(_find_forbidden_keys(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            found.update(_find_forbidden_keys(nested))
    return found


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(progress_sink: ProgressSink | None, *, phase: str, **values: Any) -> None:
    if progress_sink is not None:
        progress_sink({"stage": _STAGE, "phase": phase, **values})


def _metric_bars(report: Mapping[str, Any], metric: str) -> list[BarDatum]:
    bars = []
    for control_id, result in report["control_results"].items():
        value = result["aggregate"][metric]
        bars.append(_bar(control_id, value))
    for result in report["config_results"]:
        if result["train_oof_metrics"] is None:
            continue
        value = result["train_oof_metrics"]["aggregate"][metric]
        bars.append(_bar(result["config"]["config_id"], value))
    return bars


def _config_guard_value_bars(report: Mapping[str, Any], metric: str) -> list[BarDatum]:
    return [
        _bar(result["config"]["config_id"], result["comparison"][metric])
        for result in report["config_results"]
        if result["comparison"] is not None
    ]


def _config_selectable_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        _bar(result["config"]["config_id"], result["train_cv_selectable"])
        for result in report["config_results"]
    ]


def _bar(label: str, value: int | float | bool) -> BarDatum:
    numeric = float(value)
    if isinstance(value, bool):
        value_label = "pass" if value else "fail"
    elif isinstance(value, int):
        value_label = str(value)
    else:
        value_label = f"{numeric:.6f}"
    return BarDatum(label=label, value=numeric, value_label=value_label)
