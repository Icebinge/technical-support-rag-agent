from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ts_rag_agent.application.primeqa_hybrid_dense_sparse_rrf_comparison import (
    EncoderFactory,
    _default_encoder_factory,
    _dense_cache_configs_from_stage80,
    _load_cache_embeddings,
    _preflight_dense_caches,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQADocumentSection
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever
from ts_rag_agent.infrastructure.dense_retriever import DenseRetriever
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
    summarize_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)
from ts_rag_agent.infrastructure.section_bm25_retriever import SectionBM25Retriever

_STAGE = "Stage 116"
_CREATED_AT = "2026-07-16"
_ANALYSIS_ID = "primeqa_hybrid_high_recall_union_candidate_pool_v1"
_SOURCE_STAGE115_STATUS = "primeqa_hybrid_retrieval_index_redesign_family_stopped"
_SOURCE_STOPPED_FAMILY_ID = "retrieval_index_redesign_candidate_family"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})
_CONFIRMED_DENSE_CACHE_PROTOCOL = "compare_existing_cached_dense_models"
_BASELINE_CHANNEL_ID = "full_document_bm25"
_UNION_POOL_ID = "stage116_multi_route_union_candidate_pool"
_DEFAULT_CHANNEL_TOP_K = 100
_DEFAULT_POOL_TOP_K_VALUES = (10, 20, 50, 100, 200)
_DEFAULT_RRF_K = 60
_DEFAULT_TRAIN_FOLD_COUNT = 5
_SPECIAL_TOKEN_RE = re.compile(
    r"\b(?:CVE-\d{4}-\d{4,7}|[A-Z]{1,8}\d{2,}[A-Z0-9-]*|"
    r"\d+(?:\.\d+){1,}|[A-Z0-9]+(?:[-_][A-Z0-9]+)+)\b",
    re.IGNORECASE,
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "answer_doc_id",
        "candidate_doc_ids",
        "cited_doc_ids",
        "document_text",
        "document_title",
        "gold_answer",
        "question_text",
        "question_title",
        "raw_answer_text",
        "raw_question_text",
        "retrieved_doc_ids",
        "source_doc_ids",
    }
)


class _Retriever(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Return ranked retrieval results."""


@dataclass(frozen=True)
class _Channel:
    channel_id: str
    family: str
    retriever: _Retriever
    weight: float
    description: str


@dataclass(frozen=True)
class PrimeQAHybridHighRecallUnionVisualization:
    """One generated Stage116 high-recall union comparison visualization."""

    name: str
    path: str


class _MappedBM25Retriever:
    """BM25 over transformed documents that returns the original documents."""

    def __init__(
        self,
        *,
        indexed_documents: Iterable[PrimeQADocument],
        original_documents_by_id: Mapping[str, PrimeQADocument],
        k1: float,
        b: float,
    ) -> None:
        self._original_documents_by_id = dict(original_documents_by_id)
        self._retriever = BM25Retriever(k1=k1, b=b)
        self._retriever.fit(indexed_documents)

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        results = self._retriever.search(query, top_k=top_k)
        return self._map_results(results)

    def search_full_sort_reference(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Run the wrapped BM25 historical full-sort reference path."""

        results = self._retriever.search_full_sort_reference(query, top_k=top_k)
        return self._map_results(results)

    def _map_results(
        self,
        results: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                document=self._original_documents_by_id[result.document.id],
                score=result.score,
                rank=rank,
            )
            for rank, result in enumerate(results, start=1)
        ]


class _SpecialTokenBoostRetriever:
    """A deterministic exact-token route over runtime-visible document text."""

    def __init__(
        self,
        *,
        base_retriever: _Retriever,
        documents: Sequence[PrimeQADocument],
        sections_by_document: Mapping[str, Sequence[PrimeQADocumentSection]],
        boost: float,
        component_depth: int,
    ) -> None:
        if boost < 0:
            raise ValueError("boost must be non-negative")
        if component_depth <= 0:
            raise ValueError("component_depth must be positive")
        self._base_retriever = base_retriever
        self._documents_by_id = {document.id: document for document in documents}
        self._boost = boost
        self._component_depth = component_depth
        self._token_to_doc_ids = _special_token_index(
            documents=documents,
            sections_by_document=sections_by_document,
        )

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        base_results = self._base_retriever.search(
            query,
            top_k=max(top_k, self._component_depth),
        )
        return self.search_from_base_results(
            query,
            base_results=base_results,
            top_k=top_k,
        )

    def search_from_base_results(
        self,
        query: str,
        *,
        base_results: Sequence[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Apply the exact-token boost to an already-resolved baseline search."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores: dict[str, float] = {
            result.document.id: result.score for result in base_results
        }
        for token in _special_tokens(query):
            for doc_id in self._token_to_doc_ids.get(token, ()):
                scores[doc_id] = scores.get(doc_id, 0.0) + self._boost

        ranked_doc_ids = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
        return [
            RetrievalResult(
                document=self._documents_by_id[doc_id],
                score=scores[doc_id],
                rank=rank,
            )
            for rank, doc_id in enumerate(ranked_doc_ids[:top_k], start=1)
        ]


def run_primeqa_hybrid_high_recall_union_comparison(
    *,
    stage115_report_path: Path,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    user_confirmed_direction: bool,
    confirmation_note: str,
    stage80_report_path: Path | None = None,
    include_dense_channels: bool = True,
    channel_top_k: int = _DEFAULT_CHANNEL_TOP_K,
    pool_top_k_values: tuple[int, ...] = _DEFAULT_POOL_TOP_K_VALUES,
    rrf_k: int = _DEFAULT_RRF_K,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    train_fold_count: int = _DEFAULT_TRAIN_FOLD_COUNT,
    encoder_batch_size: int = 64,
    encoder_device: str | None = None,
    encoder_factory: EncoderFactory | None = None,
) -> dict[str, Any]:
    """Run the Stage116 train/dev-only high-recall multi-route union experiment."""

    _validate_options(
        channel_top_k=channel_top_k,
        pool_top_k_values=pool_top_k_values,
        rrf_k=rrf_k,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        train_fold_count=train_fold_count,
    )
    started_at = time.perf_counter()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    stage115_report = _load_json_object(stage115_report_path)
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
    loaded_inputs_at = time.perf_counter()

    dense_report = _load_json_object(stage80_report_path) if stage80_report_path else None
    dense_channels, dense_summary = _build_dense_channels(
        include_dense_channels=include_dense_channels,
        stage80_report=dense_report,
        stage80_report_path=stage80_report_path,
        documents=documents,
        document_ids=document_ids,
        encoder_batch_size=encoder_batch_size,
        encoder_device=encoder_device,
        encoder_factory=encoder_factory,
    )
    dense_preflight_at = time.perf_counter()

    guard_checks = _guard_checks(
        stage115_report=stage115_report,
        user_confirmed_direction=user_confirmed_direction,
        confirmation_note=confirmation_note,
        split_samples=split_samples,
        include_dense_channels=include_dense_channels,
        dense_summary=dense_summary,
        channel_top_k=channel_top_k,
        pool_top_k_values=pool_top_k_values,
    )
    if not all(check["passed"] for check in guard_checks):
        checked_at = time.perf_counter()
        report = {
            "stage": _STAGE,
            "created_at": _CREATED_AT,
            "analysis_id": _ANALYSIS_ID,
            "analysis_scope": _analysis_scope(),
            "user_confirmation": {
                "user_confirmed_direction": user_confirmed_direction,
                "confirmation_note": confirmation_note,
            },
            "split_contract": _split_contract(),
            "source_files": _source_files(
                stage115_report_path=stage115_report_path,
                stage80_report_path=stage80_report_path,
                train_split_path=train_split_path,
                dev_split_path=dev_split_path,
                documents_path=documents_path,
            ),
            "analysis_config": _analysis_config(
                channel_top_k=channel_top_k,
                pool_top_k_values=pool_top_k_values,
                rrf_k=rrf_k,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
                train_fold_count=train_fold_count,
                include_dense_channels=include_dense_channels,
            ),
            "loaded_data_summary": {
                "split_samples": summarize_primeqa_hybrid_split_samples(split_samples),
                "document_count": len(documents),
                **_section_summary(sections_by_document),
                "test_split_loaded": False,
            },
            "dense_channel_preflight": dense_summary,
            "channel_catalog": [],
            "channel_metrics_by_split": {},
            "candidate_pool_metrics_by_split": {},
            "comparisons_to_baseline": {},
            "train_fold_stability": {},
            "guard_checks": guard_checks,
            "decision": _decision(
                guard_checks=guard_checks,
                candidate_pool_metrics_by_split={},
                comparisons_to_baseline={},
            ),
            "timing_seconds": {
                "load_stage115_splits_and_build_train_folds": round(
                    loaded_splits_at - started_at,
                    3,
                ),
                "load_documents_sections": round(loaded_inputs_at - loaded_splits_at, 3),
                "dense_preflight": round(dense_preflight_at - loaded_inputs_at, 3),
                "guard_checks": round(checked_at - dense_preflight_at, 3),
                "total": round(checked_at - started_at, 3),
            },
        }
        return {**report, "public_safe_contract": _public_safe_contract(report)}

    channels = _build_lexical_channels(
        documents=documents,
        sections_by_document=sections_by_document,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        component_depth=channel_top_k,
    ) + dense_channels
    indexed_at = time.perf_counter()

    split_evaluations = {
        split: _evaluate_split(
            split=split,
            samples=samples,
            channels=channels,
            channel_top_k=channel_top_k,
            pool_top_k_values=pool_top_k_values,
            rrf_k=rrf_k,
            fold_assignments=(
                train_fold_assignments if split == _TRAIN_SPLIT else None
            ),
        )
        for split, samples in split_samples.items()
    }
    evaluated_at = time.perf_counter()

    channel_metrics_by_split = {
        split: split_result["channel_metrics"]
        for split, split_result in split_evaluations.items()
    }
    candidate_pool_metrics_by_split = {
        split: split_result["candidate_pool_metrics"]
        for split, split_result in split_evaluations.items()
    }
    comparisons_to_baseline = {
        split: _compare_pool_to_baseline(
            channel_metrics=channel_metrics_by_split[split],
            candidate_pool_metrics=candidate_pool_metrics_by_split[split],
            pool_top_k_values=pool_top_k_values,
        )
        for split in _ALLOWED_DEVELOPMENT_SPLITS
    }
    train_fold_stability = _train_fold_stability(
        split_evaluation=split_evaluations[_TRAIN_SPLIT],
        pool_top_k_values=pool_top_k_values,
    )
    checked_at = time.perf_counter()

    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": _analysis_scope(),
        "user_confirmation": {
            "user_confirmed_direction": user_confirmed_direction,
            "confirmation_note": confirmation_note,
        },
        "split_contract": _split_contract(),
        "source_files": _source_files(
            stage115_report_path=stage115_report_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
        ),
        "source_stage115_summary": _stage115_summary(stage115_report),
        "analysis_config": _analysis_config(
            channel_top_k=channel_top_k,
            pool_top_k_values=pool_top_k_values,
            rrf_k=rrf_k,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            train_fold_count=train_fold_count,
            include_dense_channels=include_dense_channels,
        ),
        "loaded_data_summary": {
            "split_samples": summarize_primeqa_hybrid_split_samples(split_samples),
            "document_count": len(documents),
            **_section_summary(sections_by_document),
            "test_split_loaded": False,
        },
        "dense_channel_preflight": dense_summary,
        "channel_catalog": [_public_channel(channel) for channel in channels],
        "channel_metrics_by_split": channel_metrics_by_split,
        "candidate_pool_metrics_by_split": candidate_pool_metrics_by_split,
        "comparisons_to_baseline": comparisons_to_baseline,
        "train_fold_stability": train_fold_stability,
        "guard_checks": guard_checks,
        "decision": _decision(
            guard_checks=guard_checks,
            candidate_pool_metrics_by_split=candidate_pool_metrics_by_split,
            comparisons_to_baseline=comparisons_to_baseline,
        ),
        "timing_seconds": {
            "load_stage115_splits_and_build_train_folds": round(
                loaded_splits_at - started_at,
                3,
            ),
            "load_documents_sections": round(loaded_inputs_at - loaded_splits_at, 3),
            "dense_preflight": round(dense_preflight_at - loaded_inputs_at, 3),
            "build_indexes": round(indexed_at - dense_preflight_at, 3),
            "evaluate_candidate_pool": round(evaluated_at - indexed_at, 3),
            "comparisons_and_guard_checks": round(checked_at - evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_high_recall_union_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridHighRecallUnionVisualization]:
    """Write Stage116 SVG visualizations."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage116_dev_channel_hit_at_100.svg": render_horizontal_bar_chart_svg(
            title="Stage116 dev channel hit@100",
            bars=_channel_hit_bars(report, split=_DEV_SPLIT, top_k=100),
            x_label="hit@100",
            width=1180,
            margin_left=460,
        ),
        "stage116_dev_union_recall_by_pool_depth.svg": render_horizontal_bar_chart_svg(
            title="Stage116 dev union recall by pool depth",
            bars=_pool_hit_bars(report, split=_DEV_SPLIT),
            x_label="candidate-pool recall",
            width=1040,
            margin_left=320,
        ),
        "stage116_dev_union_delta_vs_baseline.svg": render_horizontal_bar_chart_svg(
            title="Stage116 dev union recall delta vs BM25",
            bars=_pool_delta_bars(report, split=_DEV_SPLIT),
            x_label="recall delta",
            width=1040,
            margin_left=320,
        ),
        "stage116_dev_marginal_hits_by_channel.svg": render_horizontal_bar_chart_svg(
            title="Stage116 dev first-new gold hits by channel",
            bars=_marginal_hit_bars(report, split=_DEV_SPLIT),
            x_label="first-new hits",
            width=1180,
            margin_left=460,
        ),
        "stage116_train_fold_union_hit_at_100.svg": render_horizontal_bar_chart_svg(
            title="Stage116 train-fold union hit@100",
            bars=_train_fold_bars(report, top_k=100),
            x_label="hit@100",
            width=1040,
            margin_left=280,
        ),
        "stage116_candidate_pool_size_summary.svg": render_horizontal_bar_chart_svg(
            title="Stage116 candidate pool size summary",
            bars=_candidate_pool_size_bars(report),
            x_label="documents",
            width=1040,
            margin_left=300,
        ),
        "stage116_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage116 guard check status",
            bars=[
                BarDatum(
                    label="passed",
                    value=sum(1 for check in report["guard_checks"] if check["passed"]),
                    value_label=str(
                        sum(1 for check in report["guard_checks"] if check["passed"])
                    ),
                ),
                BarDatum(
                    label="failed",
                    value=sum(
                        1 for check in report["guard_checks"] if not check["passed"]
                    ),
                    value_label=str(
                        sum(1 for check in report["guard_checks"] if not check["passed"])
                    ),
                ),
            ],
            x_label="guard checks",
            width=820,
            margin_left=180,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridHighRecallUnionVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _build_lexical_channels(
    *,
    documents: Sequence[PrimeQADocument],
    sections_by_document: Mapping[str, Sequence[PrimeQADocumentSection]],
    bm25_k1: float,
    bm25_b: float,
    component_depth: int,
) -> list[_Channel]:
    baseline = BM25Retriever(k1=bm25_k1, b=bm25_b)
    baseline.fit(documents)
    section = SectionBM25Retriever(k1=bm25_k1, b=bm25_b)
    section.fit(documents, dict(sections_by_document))
    weighted_body = _weighted_document_retriever(
        documents=documents,
        sections_by_document=sections_by_document,
        title_weight=2.0,
        section_heading_weight=2.0,
        body_weight=1.0,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )
    heading_only = _weighted_document_retriever(
        documents=documents,
        sections_by_document=sections_by_document,
        title_weight=2.0,
        section_heading_weight=2.0,
        body_weight=0.0,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )
    special_token = _SpecialTokenBoostRetriever(
        base_retriever=baseline,
        documents=documents,
        sections_by_document=sections_by_document,
        boost=1.5,
        component_depth=component_depth,
    )
    return [
        _Channel(
            channel_id=_BASELINE_CHANNEL_ID,
            family="lexical_bm25",
            retriever=baseline,
            weight=1.0,
            description="Full-document BM25 baseline channel.",
        ),
        _Channel(
            channel_id="section_bm25_max_section_rollup",
            family="lexical_section_rollup",
            retriever=section,
            weight=1.0,
            description="Section BM25 with max-section parent-document rollup.",
        ),
        _Channel(
            channel_id="title_heading_weighted_bm25",
            family="lexical_weighted_document",
            retriever=weighted_body,
            weight=1.0,
            description="Document BM25 over title and section headings weighted with body.",
        ),
        _Channel(
            channel_id="title_heading_only_bm25",
            family="lexical_weighted_document",
            retriever=heading_only,
            weight=1.0,
            description="Document BM25 over title and section-heading fields only.",
        ),
        _Channel(
            channel_id="special_token_boosted_bm25",
            family="lexical_exact_token",
            retriever=special_token,
            weight=1.0,
            description="Full-document BM25 plus exact runtime-visible special-token boost.",
        ),
    ]


def _build_dense_channels(
    *,
    include_dense_channels: bool,
    stage80_report: Mapping[str, Any] | None,
    stage80_report_path: Path | None,
    documents: Sequence[PrimeQADocument],
    document_ids: tuple[str, ...],
    encoder_batch_size: int,
    encoder_device: str | None,
    encoder_factory: EncoderFactory | None,
) -> tuple[list[_Channel], dict[str, Any]]:
    if not include_dense_channels:
        return [], {
            "include_dense_channels": False,
            "status": "dense_channels_explicitly_disabled_for_this_run",
            "can_run_without_download": True,
            "no_model_download_attempted": True,
            "dense_cache_configs": [],
            "cache_checks": [],
        }
    if stage80_report is None or stage80_report_path is None:
        return [], {
            "include_dense_channels": True,
            "status": "blocked_missing_stage80_report",
            "can_run_without_download": False,
            "no_model_download_attempted": True,
            "dense_cache_configs": [],
            "cache_checks": [],
        }

    dense_cache_configs = _dense_cache_configs_from_stage80(
        stage80_report=stage80_report,
        current_document_ids=document_ids,
        user_confirmed_protocol=_CONFIRMED_DENSE_CACHE_PROTOCOL,
    )
    cache_preflight = _preflight_dense_caches(
        dense_cache_configs=dense_cache_configs,
        current_document_ids=document_ids,
    )
    if not cache_preflight["can_run_evaluation_without_download"]:
        return [], {
            "include_dense_channels": True,
            "status": "blocked_dense_cache_preflight_failed",
            "can_run_without_download": False,
            "no_model_download_attempted": True,
            "dense_cache_configs": _public_dense_cache_configs(dense_cache_configs),
            "cache_checks": cache_preflight["cache_checks"],
        }

    factory = encoder_factory or _default_encoder_factory(
        batch_size=encoder_batch_size,
        device=encoder_device,
    )
    channels = []
    for config in dense_cache_configs:
        embeddings = _load_cache_embeddings(
            cache_path=Path(str(config["cache_path"])),
            expected_document_ids=document_ids,
        )
        retriever = DenseRetriever(
            encoder=factory(config),
            document_text_max_chars=int(config["document_text_max_chars"]),
            query_prefix=str(config["query_prefix"] or ""),
            document_prefix=str(config["document_prefix"] or ""),
        )
        retriever.fit_embeddings(documents, embeddings)
        channel_id = str(config["config_id"]).replace(
            "dense_sparse_rrf__",
            "dense_cache__",
            1,
        )
        channels.append(
            _Channel(
                channel_id=channel_id,
                family="dense_cache",
                retriever=retriever,
                weight=1.0,
                description=(
                    "Dense retrieval over an existing local Stage80-compatible "
                    "embedding cache."
                ),
            )
        )
    return channels, {
        "include_dense_channels": True,
        "status": "dense_channels_ready",
        "can_run_without_download": True,
        "no_model_download_attempted": True,
        "dense_cache_configs": _public_dense_cache_configs(dense_cache_configs),
        "cache_checks": cache_preflight["cache_checks"],
    }


def _evaluate_split(
    *,
    split: str,
    samples: Sequence[PrimeQAHybridSplitSample],
    channels: Sequence[_Channel],
    channel_top_k: int,
    pool_top_k_values: tuple[int, ...],
    rrf_k: int,
    fold_assignments: Mapping[str, str] | None,
) -> dict[str, Any]:
    answerable_samples = [
        sample
        for sample in samples
        if sample.answerable and sample.answer_doc_id is not None
    ]
    channel_accumulators = {
        channel.channel_id: _empty_channel_accumulator(
            split=split,
            channel_id=channel.channel_id,
            total_questions=len(samples),
            top_k_values=pool_top_k_values,
        )
        for channel in channels
    }
    pool_accumulator = _empty_pool_accumulator(
        split=split,
        total_questions=len(samples),
        top_k_values=pool_top_k_values,
    )
    fold_accumulators = (
        {
            fold_id: _empty_pool_accumulator(
                split=fold_id,
                total_questions=sum(
                    1
                    for sample in answerable_samples
                    if fold_assignments.get(sample.sample_id) == fold_id
                ),
                top_k_values=pool_top_k_values,
            )
            for fold_id in sorted(set(fold_assignments.values()))
        }
        if fold_assignments is not None
        else {}
    )

    for sample in answerable_samples:
        query = sample.to_primeqa_question().full_question
        answer_doc_id = str(sample.answer_doc_id)
        results_by_channel = {
            channel.channel_id: channel.retriever.search(query, top_k=channel_top_k)
            for channel in channels
        }
        for channel in channels:
            _record_channel_result(
                accumulator=channel_accumulators[channel.channel_id],
                results=results_by_channel[channel.channel_id],
                answer_doc_id=answer_doc_id,
                top_k_values=pool_top_k_values,
            )
        ranked_pool_doc_ids = _rank_union_pool(
            channels=channels,
            results_by_channel=results_by_channel,
            rrf_k=rrf_k,
        )
        _record_pool_result(
            accumulator=pool_accumulator,
            channels=channels,
            results_by_channel=results_by_channel,
            ranked_pool_doc_ids=ranked_pool_doc_ids,
            answer_doc_id=answer_doc_id,
            top_k_values=pool_top_k_values,
        )
        if fold_assignments is not None:
            fold_id = fold_assignments[sample.sample_id]
            _record_pool_result(
                accumulator=fold_accumulators[fold_id],
                channels=channels,
                results_by_channel=results_by_channel,
                ranked_pool_doc_ids=ranked_pool_doc_ids,
                answer_doc_id=answer_doc_id,
                top_k_values=pool_top_k_values,
            )

    return {
        "channel_metrics": {
            channel_id: _finalize_channel_accumulator(accumulator)
            for channel_id, accumulator in channel_accumulators.items()
        },
        "candidate_pool_metrics": _finalize_pool_accumulator(pool_accumulator),
        "fold_metrics": {
            fold_id: _finalize_pool_accumulator(accumulator)
            for fold_id, accumulator in fold_accumulators.items()
        },
    }


def _rank_union_pool(
    *,
    channels: Sequence[_Channel],
    results_by_channel: Mapping[str, Sequence[RetrievalResult]],
    rrf_k: int,
) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for channel in channels:
        for result in results_by_channel[channel.channel_id]:
            scores[result.document.id] += channel.weight / (rrf_k + result.rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def _empty_channel_accumulator(
    *,
    split: str,
    channel_id: str,
    total_questions: int,
    top_k_values: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "split": split,
        "channel_id": channel_id,
        "total_questions": total_questions,
        "evaluated_questions": 0,
        "hit_counts": {top_k: 0 for top_k in top_k_values},
        "not_found_count_at_channel_top_k": 0,
        "reciprocal_rank_sum": 0.0,
    }


def _empty_pool_accumulator(
    *,
    split: str,
    total_questions: int,
    top_k_values: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "split": split,
        "total_questions": total_questions,
        "evaluated_questions": 0,
        "hit_counts": {top_k: 0 for top_k in top_k_values},
        "uncapped_union_hit_count": 0,
        "candidate_pool_sizes": [],
        "marginal_hit_counts": Counter(),
    }


def _record_channel_result(
    *,
    accumulator: dict[str, Any],
    results: Sequence[RetrievalResult],
    answer_doc_id: str,
    top_k_values: tuple[int, ...],
) -> None:
    result_doc_ids = [result.document.id for result in results]
    rank = (
        result_doc_ids.index(answer_doc_id) + 1
        if answer_doc_id in result_doc_ids
        else None
    )
    accumulator["evaluated_questions"] += 1
    if rank is None:
        accumulator["not_found_count_at_channel_top_k"] += 1
        return
    accumulator["reciprocal_rank_sum"] += 1 / rank
    for top_k in top_k_values:
        if rank <= top_k:
            accumulator["hit_counts"][top_k] += 1


def _record_pool_result(
    *,
    accumulator: dict[str, Any],
    channels: Sequence[_Channel],
    results_by_channel: Mapping[str, Sequence[RetrievalResult]],
    ranked_pool_doc_ids: Sequence[str],
    answer_doc_id: str,
    top_k_values: tuple[int, ...],
) -> None:
    pool_doc_set = set(ranked_pool_doc_ids)
    accumulator["evaluated_questions"] += 1
    accumulator["candidate_pool_sizes"].append(len(ranked_pool_doc_ids))
    if answer_doc_id in pool_doc_set:
        accumulator["uncapped_union_hit_count"] += 1
    for top_k in top_k_values:
        if answer_doc_id in set(ranked_pool_doc_ids[:top_k]):
            accumulator["hit_counts"][top_k] += 1

    seen_doc_ids: set[str] = set()
    for channel in channels:
        channel_doc_ids = {
            result.document.id for result in results_by_channel[channel.channel_id]
        }
        if answer_doc_id in channel_doc_ids and answer_doc_id not in seen_doc_ids:
            accumulator["marginal_hit_counts"][channel.channel_id] += 1
            break
        seen_doc_ids.update(channel_doc_ids)


def _finalize_channel_accumulator(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    evaluated_count = int(accumulator["evaluated_questions"])
    hit_counts = {
        int(top_k): int(count) for top_k, count in accumulator["hit_counts"].items()
    }
    return {
        "split": accumulator["split"],
        "channel_id": accumulator["channel_id"],
        "total_questions": int(accumulator["total_questions"]),
        "evaluated_questions": evaluated_count,
        "hit_counts": hit_counts,
        "hit_at_k": {
            str(top_k): _rounded_ratio(count, evaluated_count)
            for top_k, count in hit_counts.items()
        },
        "not_found_count_at_channel_top_k": int(
            accumulator["not_found_count_at_channel_top_k"]
        ),
        "not_found_rate_at_channel_top_k": _rounded_ratio(
            int(accumulator["not_found_count_at_channel_top_k"]),
            evaluated_count,
        ),
        "mrr_at_channel_top_k": _rounded_ratio_float(
            float(accumulator["reciprocal_rank_sum"]),
            evaluated_count,
        ),
    }


def _finalize_pool_accumulator(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    evaluated_count = int(accumulator["evaluated_questions"])
    hit_counts = {
        int(top_k): int(count) for top_k, count in accumulator["hit_counts"].items()
    }
    pool_sizes = [int(value) for value in accumulator["candidate_pool_sizes"]]
    return {
        "split": accumulator["split"],
        "pool_id": _UNION_POOL_ID,
        "total_questions": int(accumulator["total_questions"]),
        "evaluated_questions": evaluated_count,
        "hit_counts": hit_counts,
        "hit_at_k": {
            str(top_k): _rounded_ratio(count, evaluated_count)
            for top_k, count in hit_counts.items()
        },
        "uncapped_union_hit_count": int(accumulator["uncapped_union_hit_count"]),
        "uncapped_union_hit_rate": _rounded_ratio(
            int(accumulator["uncapped_union_hit_count"]),
            evaluated_count,
        ),
        "uncapped_union_not_found_count": evaluated_count
        - int(accumulator["uncapped_union_hit_count"]),
        "candidate_pool_size": {
            "average": _rounded_mean(pool_sizes),
            "median": _rounded_percentile(pool_sizes, 50),
            "p95": _rounded_percentile(pool_sizes, 95),
            "max": max(pool_sizes, default=0),
        },
        "marginal_hit_counts_by_channel_order": dict(
            sorted(accumulator["marginal_hit_counts"].items())
        ),
    }


def _compare_pool_to_baseline(
    *,
    channel_metrics: Mapping[str, Mapping[str, Any]],
    candidate_pool_metrics: Mapping[str, Any],
    pool_top_k_values: tuple[int, ...],
) -> dict[str, Any]:
    baseline = channel_metrics[_BASELINE_CHANNEL_ID]
    evaluated = int(candidate_pool_metrics["evaluated_questions"])
    comparisons = {}
    for top_k in pool_top_k_values:
        pool_hit_count = int(candidate_pool_metrics["hit_counts"][top_k])
        baseline_hit_count = int(baseline["hit_counts"][top_k])
        comparisons[f"hit@{top_k}"] = {
            "baseline_hit_count": baseline_hit_count,
            "union_hit_count": pool_hit_count,
            "hit_count_delta": pool_hit_count - baseline_hit_count,
            "baseline_hit_rate": baseline["hit_at_k"][str(top_k)],
            "union_hit_rate": candidate_pool_metrics["hit_at_k"][str(top_k)],
            "hit_rate_delta": _rounded_ratio(pool_hit_count - baseline_hit_count, evaluated),
        }
    return comparisons


def _train_fold_stability(
    *,
    split_evaluation: Mapping[str, Any],
    pool_top_k_values: tuple[int, ...],
) -> dict[str, Any]:
    fold_metrics = split_evaluation["fold_metrics"]
    if not fold_metrics:
        return {}
    stability = {
        "fold_count": len(fold_metrics),
        "fold_metrics": fold_metrics,
        "raw_group_values_written": False,
    }
    for top_k in pool_top_k_values:
        rates = [
            float(metrics["hit_at_k"][str(top_k)]) for metrics in fold_metrics.values()
        ]
        stability[f"hit@{top_k}_summary"] = {
            "min": round(min(rates), 4) if rates else 0.0,
            "max": round(max(rates), 4) if rates else 0.0,
            "spread": round((max(rates) - min(rates)), 4) if rates else 0.0,
            "average": _rounded_mean(rates),
        }
    return stability


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    candidate_pool_metrics_by_split: Mapping[str, Any],
    comparisons_to_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    if not all(check["passed"] for check in guard_checks):
        return {
            "status": "primeqa_hybrid_high_recall_union_candidate_pool_blocked",
            "can_continue_train_dev_development": False,
            "can_continue_second_stage_precision_experiment": False,
            "recommended_next_direction": "fix_stage116_blockers_before_any_new_metrics",
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }

    dev_pool = candidate_pool_metrics_by_split[_DEV_SPLIT]
    dev_comparison = comparisons_to_baseline[_DEV_SPLIT]
    available_top_k_values = sorted(
        int(key.split("@", maxsplit=1)[1]) for key in dev_comparison
    )
    primary_top_k = 100 if 100 in available_top_k_values else available_top_k_values[-1]
    max_top_k = available_top_k_values[-1]
    primary_key = f"hit@{primary_top_k}"
    max_key = f"hit@{max_top_k}"
    dev_primary_delta = int(dev_comparison[primary_key]["hit_count_delta"])
    dev_max_delta = int(dev_comparison[max_key]["hit_count_delta"])
    second_stage_reasonable = dev_primary_delta > 0 or dev_max_delta > 0
    next_direction = (
        "design_second_stage_precision_reranking_protocol_over_stage116_pool"
        if second_stage_reasonable
        else "expand_first_stage_routes_before_second_stage_precision"
    )
    return {
        "status": "primeqa_hybrid_high_recall_union_candidate_pool_completed",
        "analysis_result": {
            "dev_primary_pool_depth": primary_top_k,
            "dev_max_pool_depth": max_top_k,
            "dev_union_hit_at_primary_depth": dev_pool["hit_at_k"][str(primary_top_k)],
            "dev_union_hit_at_max_depth": dev_pool["hit_at_k"][str(max_top_k)],
            "dev_hit_count_delta_at_primary_depth_vs_bm25": dev_primary_delta,
            "dev_hit_count_delta_at_max_depth_vs_bm25": dev_max_delta,
            "dev_uncapped_union_not_found_count": dev_pool[
                "uncapped_union_not_found_count"
            ],
        },
        "can_continue_train_dev_development": True,
        "can_continue_second_stage_precision_experiment": second_stage_reasonable,
        "recommended_next_direction": next_direction,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage117: keep test locked and either design a second-stage precision "
            "reranking protocol over the fixed Stage116 candidate pool if the "
            "dev recall lift is positive, or add new first-stage retrieval routes "
            "if Stage116 still leaves too many gold documents unrecalled."
        ),
    }


def _guard_checks(
    *,
    stage115_report: Mapping[str, Any],
    user_confirmed_direction: bool,
    confirmation_note: str,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    include_dense_channels: bool,
    dense_summary: Mapping[str, Any],
    channel_top_k: int,
    pool_top_k_values: tuple[int, ...],
) -> list[dict[str, Any]]:
    stage115_decision = stage115_report.get("decision") or {}
    return [
        _check(
            name="user_confirmed_stage116_high_recall_union_direction",
            passed=user_confirmed_direction,
            observed=confirmation_note,
            expected=(
                "user confirmed first-stage multi-route union recall experiment"
            ),
        ),
        _check(
            name="stage115_retrieval_index_redesign_family_stopped",
            passed=stage115_decision.get("status") == _SOURCE_STAGE115_STATUS,
            observed=stage115_decision.get("status"),
            expected=_SOURCE_STAGE115_STATUS,
        ),
        _check(
            name="stage115_stopped_expected_family",
            passed=stage115_decision.get("stopped_family_id")
            == _SOURCE_STOPPED_FAMILY_ID,
            observed=stage115_decision.get("stopped_family_id"),
            expected=_SOURCE_STOPPED_FAMILY_ID,
        ),
        _check(
            name="stage116_uses_only_train_dev_splits",
            passed=set(split_samples) == set(_ALLOWED_DEVELOPMENT_SPLITS),
            observed=sorted(split_samples),
            expected=list(_ALLOWED_DEVELOPMENT_SPLITS),
        ),
        _check(
            name="stage116_final_test_metrics_not_run",
            passed=True,
            observed="test split path is not accepted by Stage116 runner",
            expected="no final test metrics",
        ),
        _check(
            name="stage116_runtime_defaults_unchanged",
            passed=True,
            observed="offline analysis only",
            expected="runtime defaults unchanged",
        ),
        _check(
            name="stage116_fallback_strategies_not_added",
            passed=True,
            observed="candidate-pool union experiment only",
            expected="no fallback strategies",
        ),
        _check(
            name="stage116_does_not_use_source_doc_ids_as_runtime_retrieval",
            passed=True,
            observed="only question text and corpus document text indexes are used",
            expected="no source DOC_IDS oracle route",
        ),
        _check(
            name="stage116_channel_top_k_covers_pool_depths",
            passed=channel_top_k >= max(pool_top_k_values),
            observed={"channel_top_k": channel_top_k, "pool_top_k_values": pool_top_k_values},
            expected="channel_top_k >= max(pool_top_k_values)",
        ),
        _check(
            name="stage116_dense_channels_ready_or_explicitly_disabled",
            passed=(not include_dense_channels)
            or bool(dense_summary.get("can_run_without_download")),
            observed={
                "include_dense_channels": include_dense_channels,
                "dense_status": dense_summary.get("status"),
                "can_run_without_download": dense_summary.get("can_run_without_download"),
            },
            expected="dense caches local-ready when dense channels are enabled",
        ),
        _check(
            name="stage116_no_model_download_attempted",
            passed=bool(dense_summary.get("no_model_download_attempted")),
            observed=dense_summary.get("no_model_download_attempted"),
            expected=True,
        ),
    ]


def _weighted_document_retriever(
    *,
    documents: Sequence[PrimeQADocument],
    sections_by_document: Mapping[str, Sequence[PrimeQADocumentSection]],
    title_weight: float,
    section_heading_weight: float,
    body_weight: float,
    bm25_k1: float,
    bm25_b: float,
) -> _MappedBM25Retriever:
    synthetic_documents = [
        PrimeQADocument(
            id=document.id,
            title="",
            text=_weighted_document_search_text(
                document=document,
                sections=sections_by_document.get(document.id, ()),
                title_weight=title_weight,
                section_heading_weight=section_heading_weight,
                body_weight=body_weight,
            ),
        )
        for document in documents
    ]
    return _MappedBM25Retriever(
        indexed_documents=synthetic_documents,
        original_documents_by_id={document.id: document for document in documents},
        k1=bm25_k1,
        b=bm25_b,
    )


def _weighted_document_search_text(
    *,
    document: PrimeQADocument,
    sections: Sequence[PrimeQADocumentSection],
    title_weight: float,
    section_heading_weight: float,
    body_weight: float,
) -> str:
    headings = "\n".join(section.section_id for section in sections)
    return "\n".join(
        part
        for part in (
            _repeat_text(document.title, title_weight),
            _repeat_text(headings, section_heading_weight),
            _repeat_text(document.text, body_weight),
        )
        if part
    )


def _repeat_text(text: str, weight: float) -> str:
    count = max(0, int(round(weight)))
    return "\n".join(text for _ in range(count) if text.strip())


def _special_token_index(
    *,
    documents: Sequence[PrimeQADocument],
    sections_by_document: Mapping[str, Sequence[PrimeQADocumentSection]],
) -> dict[str, set[str]]:
    token_to_doc_ids: dict[str, set[str]] = defaultdict(set)
    for document in documents:
        sections = sections_by_document.get(document.id, ())
        text = "\n".join(
            [document.title, document.text]
            + [section.section_id for section in sections]
            + [section.text for section in sections]
        )
        for token in _special_tokens(text):
            token_to_doc_ids[token].add(document.id)
    return dict(token_to_doc_ids)


def _special_tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _SPECIAL_TOKEN_RE.finditer(text)}


def _build_train_fold_assignments(
    samples: Sequence[PrimeQAHybridSplitSample],
    *,
    fold_count: int,
) -> dict[str, str]:
    groups: dict[str, list[PrimeQAHybridSplitSample]] = defaultdict(list)
    for sample in samples:
        groups[_group_key(sample)].append(sample)
    fold_rows: list[list[PrimeQAHybridSplitSample]] = [[] for _ in range(fold_count)]
    for group_key, group_samples in sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), _stable_hash(item[0])),
    ):
        _ = group_key
        target_index = min(
            range(fold_count),
            key=lambda index: (len(fold_rows[index]), index),
        )
        fold_rows[target_index].extend(group_samples)
    return {
        sample.sample_id: f"fold_{fold_index + 1}"
        for fold_index, fold_samples in enumerate(fold_rows)
        for sample in fold_samples
    }


def _group_key(sample: PrimeQAHybridSplitSample) -> str:
    normalized_question = " ".join(
        f"{sample.question_title} {sample.question_text}".lower().split()
    )
    doc_marker = sample.answer_doc_id if sample.answerable else "UNANSWERABLE"
    return f"{normalized_question}::{doc_marker}"


def _analysis_scope() -> str:
    return (
        "Train/dev-only first-stage high-recall candidate-pool experiment. "
        "Stage116 unions several simple retrieval routes, ranks the deduplicated "
        "candidate pool with fixed reciprocal-rank scoring, reports recall-only "
        "gold-document coverage, keeps the frozen test split locked, does not run "
        "answer generation or final metrics, does not use source DOC_IDS as a "
        "runtime retrieval route, does not add fallback strategies, and does not "
        "change runtime defaults."
    )


def _split_contract() -> dict[str, Any]:
    return {
        "split_name": _SPLIT_NAME,
        "protocol_version": _PROTOCOL_VERSION,
        "development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
        "train_fold_stability_split": _TRAIN_SPLIT,
        "validation_split": _DEV_SPLIT,
        "dev_selection_used": False,
        "dev_retuning_used": False,
        "forbidden_final_splits": sorted(_FORBIDDEN_FINAL_SPLITS),
    }


def _analysis_config(
    *,
    channel_top_k: int,
    pool_top_k_values: tuple[int, ...],
    rrf_k: int,
    bm25_k1: float,
    bm25_b: float,
    train_fold_count: int,
    include_dense_channels: bool,
) -> dict[str, Any]:
    return {
        "candidate_pool_id": _UNION_POOL_ID,
        "channel_top_k": channel_top_k,
        "pool_top_k_values": list(pool_top_k_values),
        "pool_ranking": "fixed_weight_reciprocal_rank_score_after_route_union",
        "rrf_k": rrf_k,
        "channel_weights": "all_channels_weight_1",
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "include_dense_channels": include_dense_channels,
        "dense_cache_protocol": _CONFIRMED_DENSE_CACHE_PROTOCOL,
        "train_fold_count": train_fold_count,
        "selection_rule": (
            "No model is selected from dev. Stage116 reports one fixed first-stage "
            "candidate-pool design requested by the user."
        ),
    }


def _source_files(
    *,
    stage115_report_path: Path,
    stage80_report_path: Path | None,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
) -> dict[str, Any]:
    files = {
        "stage115_report": _fingerprint(stage115_report_path),
        "train_split": _fingerprint(train_split_path),
        "dev_split": _fingerprint(dev_split_path),
        "documents": _fingerprint(documents_path),
    }
    if stage80_report_path is not None:
        files["stage80_report"] = _fingerprint(stage80_report_path)
    return files


def _stage115_summary(stage115_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage115_report.get("decision") or {}
    stopped_family = stage115_report.get("stopped_family") or {}
    return {
        "stage": stage115_report.get("stage"),
        "status": decision.get("status"),
        "stopped_family_id": decision.get("stopped_family_id"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "stage114_selectable_config_count": (
            (stopped_family.get("stage114_summary") or {}).get(
                "selectable_config_count"
            )
        ),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _public_channel(channel: _Channel) -> dict[str, Any]:
    return {
        "channel_id": channel.channel_id,
        "family": channel.family,
        "weight": channel.weight,
        "description": channel.description,
        "uses_gold_labels": False,
        "uses_source_doc_ids": False,
    }


def _public_dense_cache_configs(
    dense_cache_configs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    public_configs = []
    for config in dense_cache_configs:
        public_configs.append(
            {
                "config_id": config.get("config_id"),
                "model_name": config.get("model_name"),
                "document_text_max_chars": config.get("document_text_max_chars"),
                "document_prefix": config.get("document_prefix"),
                "query_prefix_source": config.get("query_prefix_source"),
                "embedding_shape": config.get("embedding_shape"),
                "document_id_count": config.get("document_id_count"),
                "cache_path": config.get("cache_path"),
                "snapshot_path": config.get("snapshot_path"),
            }
        )
    return public_configs


def _section_summary(
    sections_by_document: Mapping[str, Sequence[PrimeQADocumentSection]],
) -> dict[str, Any]:
    section_counts = [len(sections) for sections in sections_by_document.values()]
    return {
        "documents_with_sections": sum(count > 0 for count in section_counts),
        "section_count": sum(section_counts),
        "average_sections_per_document": _rounded_mean(section_counts),
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    forbidden_keys = sorted(_find_forbidden_public_keys(report))
    return {
        "public_safe_summary_only": True,
        "raw_question_text_written": False,
        "raw_answer_text_written": False,
        "raw_document_text_written": False,
        "raw_document_ids_written": False,
        "forbidden_keys_found": forbidden_keys,
    }


def _find_forbidden_public_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_string = str(key)
            if key_string in _FORBIDDEN_PUBLIC_KEYS:
                found.add(key_string)
            found.update(_find_forbidden_public_keys(child))
    elif isinstance(value, list | tuple):
        for child in value:
            found.update(_find_forbidden_public_keys(child))
    return found


def _validate_options(
    *,
    channel_top_k: int,
    pool_top_k_values: tuple[int, ...],
    rrf_k: int,
    bm25_k1: float,
    bm25_b: float,
    train_fold_count: int,
) -> None:
    if channel_top_k <= 0:
        raise ValueError("channel_top_k must be positive")
    if not pool_top_k_values or any(top_k <= 0 for top_k in pool_top_k_values):
        raise ValueError("pool_top_k_values must be non-empty positive integers")
    if tuple(sorted(set(pool_top_k_values))) != pool_top_k_values:
        raise ValueError("pool_top_k_values must be unique and sorted")
    if channel_top_k < max(pool_top_k_values):
        raise ValueError("channel_top_k must cover the largest pool_top_k value")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if bm25_k1 <= 0:
        raise ValueError("bm25_k1 must be positive")
    if not 0 <= bm25_b <= 1:
        raise ValueError("bm25_b must be between 0 and 1")
    if train_fold_count < 2:
        raise ValueError("train_fold_count must be at least 2")


def _check(
    *,
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _channel_hit_bars(
    report: Mapping[str, Any],
    *,
    split: str,
    top_k: int,
) -> list[BarDatum]:
    metrics = report.get("channel_metrics_by_split", {}).get(split, {})
    bars = []
    for channel_id, channel_metrics in metrics.items():
        value = float(channel_metrics["hit_at_k"].get(str(top_k), 0.0))
        bars.append(
            BarDatum(
                label=str(channel_id),
                value=value,
                value_label=f"{value:.4f}",
            )
        )
    return sorted(bars, key=lambda bar: (-bar.value, bar.label))


def _pool_hit_bars(report: Mapping[str, Any], *, split: str) -> list[BarDatum]:
    metrics = report.get("candidate_pool_metrics_by_split", {}).get(split, {})
    return [
        BarDatum(label=f"union hit@{top_k}", value=float(value), value_label=f"{value:.4f}")
        for top_k, value in sorted(
            ((int(top_k), hit_at) for top_k, hit_at in metrics.get("hit_at_k", {}).items()),
            key=lambda item: item[0],
        )
    ]


def _pool_delta_bars(report: Mapping[str, Any], *, split: str) -> list[BarDatum]:
    comparisons = report.get("comparisons_to_baseline", {}).get(split, {})
    bars = []
    for metric_id, metric in sorted(
        comparisons.items(),
        key=lambda item: int(item[0].split("@", maxsplit=1)[1]),
    ):
        value = float(metric["hit_rate_delta"])
        bars.append(
            BarDatum(
                label=metric_id,
                value=value,
                value_label=f"{value:+.4f}",
            )
        )
    return bars


def _marginal_hit_bars(report: Mapping[str, Any], *, split: str) -> list[BarDatum]:
    metrics = report.get("candidate_pool_metrics_by_split", {}).get(split, {})
    counts = metrics.get("marginal_hit_counts_by_channel_order", {})
    return sorted(
        [
            BarDatum(label=str(channel_id), value=float(count), value_label=str(count))
            for channel_id, count in counts.items()
        ],
        key=lambda bar: (-bar.value, bar.label),
    )


def _train_fold_bars(report: Mapping[str, Any], *, top_k: int) -> list[BarDatum]:
    fold_metrics = report.get("train_fold_stability", {}).get("fold_metrics", {})
    return [
        BarDatum(
            label=str(fold_id),
            value=float(metrics["hit_at_k"].get(str(top_k), 0.0)),
            value_label=f"{float(metrics['hit_at_k'].get(str(top_k), 0.0)):.4f}",
        )
        for fold_id, metrics in sorted(fold_metrics.items())
    ]


def _candidate_pool_size_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for split in _ALLOWED_DEVELOPMENT_SPLITS:
        metrics = report.get("candidate_pool_metrics_by_split", {}).get(split, {})
        summary = metrics.get("candidate_pool_size", {})
        for key in ("average", "median", "p95", "max"):
            value = float(summary.get(key, 0.0))
            bars.append(
                BarDatum(
                    label=f"{split} {key}",
                    value=value,
                    value_label=f"{value:.1f}",
                )
            )
    return bars


def _load_json_object(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return loaded


def _fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": _sha256(path) if path.exists() and path.is_file() else None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rounded_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _rounded_ratio_float(numerator: float, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _rounded_mean(values: Sequence[float | int]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _rounded_percentile(values: Sequence[int], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((percentile / 100) * (len(ordered) - 1))
    return round(float(ordered[index]), 4)
