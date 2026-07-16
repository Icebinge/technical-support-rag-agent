from __future__ import annotations

import hashlib
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    EncoderFactory,
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
    _load_json_object,
    _rank_union_pool,
    _special_tokens,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQADocumentSection
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

_STAGE = "Stage 118"
_CREATED_AT = "2026-07-16"
_ANALYSIS_ID = "primeqa_hybrid_second_stage_reranking_train_cv_dev_validation_v1"
_SOURCE_STAGE117_STATUS = "primeqa_hybrid_second_stage_reranking_protocol_frozen"
_SOURCE_PROTOCOL_ID = "primeqa_hybrid_second_stage_reranking_protocol_v1"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = ("test",)
_BASELINE_CONFIG_ID = "stage116_fixed_rrf_pool_order"
_TOP_K_VALUES = (10, 20, 50, 100, 200)
_PRIMARY_TOP_K = 20
_RRF_SCORE_FEATURE = "stage116_rrf_score"
_FEATURE_ALIASES = {
    "bm25_top10_non_gold_indicator": "bm25_top10_indicator",
}
_FEATURE_ALIAS_NOTES = {
    "bm25_top10_non_gold_indicator": (
        "Stage117 protocol wording is interpreted without gold labels as "
        "bm25_top10_indicator, a runtime-visible feature that only records "
        "whether the candidate appears in full-document BM25 top10."
    )
}
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "answer_doc_id",
        "candidate_doc_ids",
        "cited_doc_ids",
        "document_id",
        "document_text",
        "document_title",
        "gold_answer",
        "matched_token_strings",
        "question_text",
        "question_title",
        "raw_answer_text",
        "raw_document_text",
        "raw_question_text",
        "retrieved_doc_ids",
        "source_doc_ids",
    }
)


class _Scorer(Protocol):
    def fit(self, records: Sequence[_CandidateRecord]) -> _Scorer:
        """Fit scorer on candidate records."""

    def score(self, records: Sequence[_CandidateRecord]) -> list[float]:
        """Return one score per candidate, higher is better."""


@dataclass(frozen=True)
class _CandidateRecord:
    split: str
    sample_id: str
    fold_id: str | None
    doc_id: str
    baseline_rank: int
    is_gold: bool
    features: dict[str, float]


@dataclass(frozen=True)
class PrimeQAHybridSecondStageRerankingVisualization:
    """One generated Stage118 reranking validation chart."""

    name: str
    path: str


class _DeterministicScorer:
    def __init__(self, *, weights: Mapping[str, float]) -> None:
        self._weights = dict(weights)

    def fit(self, records: Sequence[_CandidateRecord]) -> _DeterministicScorer:
        _ = records
        return self

    def score(self, records: Sequence[_CandidateRecord]) -> list[float]:
        return [
            sum(
                record.features.get(feature, 0.0) * weight
                for feature, weight in self._weights.items()
            )
            for record in records
        ]


class _LogisticGoldScorer:
    def __init__(self, *, feature_names: Sequence[str]) -> None:
        self._feature_names = tuple(feature_names)
        self._pipeline = Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(class_weight="balanced", max_iter=1000),
                ),
            ]
        )

    def fit(self, records: Sequence[_CandidateRecord]) -> _LogisticGoldScorer:
        labels = [int(record.is_gold) for record in records]
        if len(set(labels)) < 2:
            raise ValueError("Logistic reranker needs both positive and negative labels.")
        self._pipeline.fit(
            [_project_feature_dict(record, self._feature_names) for record in records],
            labels,
        )
        return self

    def score(self, records: Sequence[_CandidateRecord]) -> list[float]:
        probabilities = self._pipeline.predict_proba(
            [_project_feature_dict(record, self._feature_names) for record in records]
        )
        positive_index = list(self._pipeline.classes_).index(1)
        return [float(row[positive_index]) for row in probabilities]


class _RidgeGoldProxyScorer:
    def __init__(self, *, feature_names: Sequence[str]) -> None:
        self._feature_names = tuple(feature_names)
        self._pipeline = Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )

    def fit(self, records: Sequence[_CandidateRecord]) -> _RidgeGoldProxyScorer:
        labels = [1.0 if record.is_gold else 0.0 for record in records]
        self._pipeline.fit(
            [_project_feature_dict(record, self._feature_names) for record in records],
            labels,
        )
        return self

    def score(self, records: Sequence[_CandidateRecord]) -> list[float]:
        projected = [_project_feature_dict(record, self._feature_names) for record in records]
        return [float(score) for score in self._pipeline.predict(projected)]


def run_primeqa_hybrid_second_stage_reranking_validation(
    *,
    stage117_protocol_path: Path,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    stage80_report_path: Path | None = None,
    user_confirmed_validation: bool,
    confirmation_note: str,
    include_dense_channels: bool = True,
    encoder_batch_size: int = 64,
    encoder_device: str | None = None,
    encoder_factory: EncoderFactory | None = None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> dict[str, Any]:
    """Run Stage118 train-CV/dev validation for the frozen reranking protocol."""

    started_at = time.perf_counter()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    stage117_report = _load_json_object(stage117_protocol_path)
    frozen_protocol = stage117_report.get("frozen_protocol") or {}
    stage117_summary = _stage117_summary(stage117_report)
    candidate_configs = tuple(frozen_protocol.get("candidate_configs") or ())
    selection_rules = frozen_protocol.get("selection_rules") or {}
    candidate_pool_depth = int(
        (frozen_protocol.get("fixed_candidate_pool_contract") or {}).get(
            "candidate_pool_depth",
            200,
        )
    )
    channel_top_k = int(
        (frozen_protocol.get("fixed_candidate_pool_contract") or {}).get(
            "source_stage116_channel_top_k",
            candidate_pool_depth,
        )
    )
    rrf_k = int(
        (frozen_protocol.get("fixed_candidate_pool_contract") or {}).get(
            "source_stage116_rrf_k",
            60,
        )
    )
    loaded_protocol_at = time.perf_counter()

    split_samples = {
        _TRAIN_SPLIT: load_primeqa_hybrid_split_samples(train_split_path),
        _DEV_SPLIT: load_primeqa_hybrid_split_samples(dev_split_path),
    }
    train_fold_assignments = _build_train_fold_assignments(
        split_samples[_TRAIN_SPLIT],
        fold_count=int(selection_rules.get("minimum_train_folds") or 5),
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
    lexical_channels = _build_lexical_channels(
        documents=documents,
        sections_by_document=sections_by_document,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        component_depth=channel_top_k,
    )
    channels = lexical_channels + dense_channels
    built_channels_at = time.perf_counter()

    records_by_split = {
        split: _build_candidate_records(
            split=split,
            samples=samples,
            documents_by_id=documents_by_id,
            sections_by_document=sections_by_document,
            channels=channels,
            fold_assignments=train_fold_assignments if split == _TRAIN_SPLIT else None,
            channel_top_k=channel_top_k,
            candidate_pool_depth=candidate_pool_depth,
            rrf_k=rrf_k,
        )
        for split, samples in split_samples.items()
    }
    built_records_at = time.perf_counter()

    baseline_metrics = {
        "train_cv": _evaluate_baseline(records_by_split[_TRAIN_SPLIT]),
        "train_full": _evaluate_baseline(records_by_split[_TRAIN_SPLIT]),
        "dev": _evaluate_baseline(records_by_split[_DEV_SPLIT]),
    }
    config_results = [
        _evaluate_config(
            config=config,
            train_records=records_by_split[_TRAIN_SPLIT],
            dev_records=records_by_split[_DEV_SPLIT],
            baseline_metrics=baseline_metrics,
            selection_rules=selection_rules,
        )
        for config in candidate_configs
    ]
    evaluated_at = time.perf_counter()

    train_cv_selection = _select_train_cv_config(
        config_results=config_results,
        selection_rules=selection_rules,
    )
    dev_validation = _dev_validation_summary(
        config_results=config_results,
        selected_config_id=train_cv_selection.get("selected_config_id"),
    )
    guard_checks = _guard_checks(
        stage117_summary=stage117_summary,
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
        split_samples=split_samples,
        dense_summary=dense_summary,
        baseline_metrics=baseline_metrics,
        stage116_summary=stage117_summary.get("stage116_summary") or {},
        candidate_pool_depth=candidate_pool_depth,
        records_by_split=records_by_split,
        train_cv_selection=train_cv_selection,
    )
    checked_at = time.perf_counter()

    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train-CV/dev validation for the frozen Stage117 second-stage "
            "reranking protocol over the fixed Stage116 top200 candidate pool. "
            "This stage rebuilds the train/dev candidate pool from runtime-visible "
            "retrieval routes, evaluates frozen reranking candidates, keeps test "
            "locked, keeps dev report-only, does not run answer metrics, does not "
            "change runtime defaults, and does not add fallback strategies."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_validation),
            "confirmation_note": confirmation_note,
        },
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
            "selection_split": "train",
            "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
            "validation_split": "dev",
            "dev_validation_mode": "single_pass_report_only_no_retuning",
            "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage117_protocol": _fingerprint(stage117_protocol_path),
            "train_split": _fingerprint(train_split_path),
            "dev_split": _fingerprint(dev_split_path),
            "documents": _fingerprint(documents_path),
            "stage80_report": _fingerprint(stage80_report_path)
            if stage80_report_path is not None
            else None,
        },
        "stage117_summary": stage117_summary,
        "analysis_config": {
            "candidate_pool_depth": candidate_pool_depth,
            "channel_top_k": channel_top_k,
            "rrf_k": rrf_k,
            "include_dense_channels": include_dense_channels,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "candidate_config_count": len(candidate_configs),
            "baseline_config_id": _BASELINE_CONFIG_ID,
            "feature_aliases": dict(_FEATURE_ALIASES),
            "feature_alias_notes": dict(_FEATURE_ALIAS_NOTES),
        },
        "loaded_data_summary": {
            "split_samples": summarize_primeqa_hybrid_split_samples(split_samples),
            "document_count": len(documents_by_id),
            **_section_summary(sections_by_document),
            "test_split_loaded": False,
        },
        "candidate_pool_summary": _candidate_pool_summary(records_by_split),
        "dense_channel_preflight": dense_summary,
        "baseline_metrics": baseline_metrics,
        "config_results": config_results,
        "train_cv_selection": train_cv_selection,
        "dev_validation": dev_validation,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks=guard_checks, train_cv_selection=train_cv_selection),
        "timing_seconds": {
            "load_protocol": round(loaded_protocol_at - started_at, 3),
            "load_splits_and_build_folds": round(loaded_splits_at - loaded_protocol_at, 3),
            "load_documents_and_reports": round(loaded_documents_at - loaded_splits_at, 3),
            "build_retrieval_channels": round(built_channels_at - loaded_documents_at, 3),
            "build_candidate_records": round(built_records_at - built_channels_at, 3),
            "evaluate_rerankers": round(evaluated_at - built_records_at, 3),
            "selection_and_guard_checks": round(checked_at - evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_second_stage_reranking_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridSecondStageRerankingVisualization]:
    """Write Stage118 SVG visualizations."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage118_train_cv_objective_scores.svg": render_horizontal_bar_chart_svg(
            title="Stage118 train-CV objective scores",
            bars=_objective_score_bars(report),
            x_label="objective score",
            width=1540,
            margin_left=760,
        ),
        "stage118_train_cv_mrr_at_20_delta.svg": render_horizontal_bar_chart_svg(
            title="Stage118 train-CV MRR@20 delta",
            bars=_metric_delta_bars(report, split="train_cv", metric="mrr_at_20_delta"),
            x_label="delta",
            width=1540,
            margin_left=760,
        ),
        "stage118_train_cv_hit_at_10_delta.svg": render_horizontal_bar_chart_svg(
            title="Stage118 train-CV hit@10 delta",
            bars=_metric_delta_bars(report, split="train_cv", metric="hit@10_delta"),
            x_label="delta",
            width=1540,
            margin_left=760,
        ),
        "stage118_train_cv_guard_pass_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage118 train-CV selection guard pass counts",
            bars=_config_guard_pass_bars(report),
            x_label="passed guards",
            width=1540,
            margin_left=760,
        ),
        "stage118_dev_selected_config_deltas.svg": render_horizontal_bar_chart_svg(
            title="Stage118 dev selected config deltas",
            bars=_dev_selected_delta_bars(report),
            x_label="delta",
            width=1180,
            margin_left=480,
        ),
        "stage118_selection_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage118 decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1320,
            margin_left=660,
        ),
        "stage118_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage118 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1660,
            margin_left=920,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridSecondStageRerankingVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _build_candidate_records(
    *,
    split: str,
    samples: Sequence[PrimeQAHybridSplitSample],
    documents_by_id: Mapping[str, PrimeQADocument],
    sections_by_document: Mapping[str, Sequence[PrimeQADocumentSection]],
    channels: Sequence[Any],
    fold_assignments: Mapping[str, str] | None,
    channel_top_k: int,
    candidate_pool_depth: int,
    rrf_k: int,
) -> list[_CandidateRecord]:
    records: list[_CandidateRecord] = []
    answerable_samples = [
        sample for sample in samples if sample.answerable and sample.answer_doc_id is not None
    ]
    headings_by_doc = {
        doc_id: " ".join(section.section_id for section in sections)
        for doc_id, sections in sections_by_document.items()
    }
    for sample in answerable_samples:
        question = sample.to_primeqa_question()
        query = question.full_question
        results_by_channel = {
            channel.channel_id: channel.retriever.search(query, top_k=channel_top_k)
            for channel in channels
        }
        ranked_pool_doc_ids = _rank_union_pool(
            channels=channels,
            results_by_channel=results_by_channel,
            rrf_k=rrf_k,
        )[:candidate_pool_depth]
        route_rank_maps = {
            channel_id: {result.document.id: result.rank for result in results}
            for channel_id, results in results_by_channel.items()
        }
        route_score_maps = {
            channel_id: {result.document.id: result.score for result in results}
            for channel_id, results in results_by_channel.items()
        }
        rrf_scores = _rrf_scores(
            channels=channels,
            results_by_channel=results_by_channel,
            rrf_k=rrf_k,
        )
        for pool_rank, doc_id in enumerate(ranked_pool_doc_ids, start=1):
            document = documents_by_id[doc_id]
            records.append(
                _CandidateRecord(
                    split=split,
                    sample_id=sample.sample_id,
                    fold_id=fold_assignments.get(sample.sample_id)
                    if fold_assignments is not None
                    else None,
                    doc_id=doc_id,
                    baseline_rank=pool_rank,
                    is_gold=doc_id == sample.answer_doc_id,
                    features=_candidate_features(
                        query=query,
                        document=document,
                        heading_text=headings_by_doc.get(doc_id, ""),
                        route_rank_maps=route_rank_maps,
                        route_score_maps=route_score_maps,
                        rrf_score=rrf_scores.get(doc_id, 0.0),
                    ),
                )
            )
    return records


def _candidate_features(
    *,
    query: str,
    document: PrimeQADocument,
    heading_text: str,
    route_rank_maps: Mapping[str, Mapping[str, int]],
    route_score_maps: Mapping[str, Mapping[str, float]],
    rrf_score: float,
) -> dict[str, float]:
    query_tokens = set(tokenize_text(query))
    title_tokens = set(tokenize_text(document.title))
    heading_tokens = set(tokenize_text(heading_text))
    body_tokens = set(tokenize_text(document.text))
    query_special_tokens = _special_tokens(query)
    title_special_tokens = _special_tokens(document.title)
    heading_special_tokens = _special_tokens(heading_text)
    body_special_tokens = _special_tokens(document.text)
    route_ranks = {
        channel_id: ranks.get(document.id)
        for channel_id, ranks in route_rank_maps.items()
    }
    lexical_route_ids = [
        channel_id
        for channel_id in route_rank_maps
        if not channel_id.startswith("dense_cache__")
    ]
    dense_route_ids = [
        channel_id
        for channel_id in route_rank_maps
        if channel_id.startswith("dense_cache__")
    ]
    present_ranks = [rank for rank in route_ranks.values() if rank is not None]
    features: dict[str, float] = {
        _RRF_SCORE_FEATURE: float(rrf_score),
        "route_hit_count": float(len(present_ranks)),
        "lexical_route_hit_count": float(
            sum(route_ranks[channel_id] is not None for channel_id in lexical_route_ids)
        ),
        "dense_route_hit_count": float(
            sum(route_ranks[channel_id] is not None for channel_id in dense_route_ids)
        ),
        "best_route_rank": float(min(present_ranks) if present_ranks else 999.0),
        "best_route_inverse_rank": _inverse_rank(min(present_ranks) if present_ranks else None),
        "dense_route_best_rank": float(
            min(
                [
                    route_ranks[channel_id]
                    for channel_id in dense_route_ids
                    if route_ranks[channel_id] is not None
                ],
                default=999,
            )
        ),
        "query_title_token_overlap": _overlap_count(query_tokens, title_tokens),
        "query_section_heading_overlap": _overlap_count(query_tokens, heading_tokens),
        "query_token_coverage": _coverage(
            query_tokens,
            title_tokens | heading_tokens | body_tokens,
        ),
        "query_body_token_coverage": _coverage(query_tokens, body_tokens),
        "document_length_bucket": min(len(body_tokens) / 1000.0, 5.0),
        "query_special_token_match_count": float(
            len(
                query_special_tokens
                & (title_special_tokens | heading_special_tokens | body_special_tokens)
            )
        ),
        "title_special_token_match_count": float(len(query_special_tokens & title_special_tokens)),
        "heading_special_token_match_count": float(
            len(query_special_tokens & heading_special_tokens)
        ),
        "special_token_match_count": float(
            len(
                query_special_tokens
                & (title_special_tokens | heading_special_tokens | body_special_tokens)
            )
        ),
        "bm25_top10_indicator": 1.0
        if (route_ranks.get("full_document_bm25") or 999) <= 10
        else 0.0,
    }
    for channel_id, rank in route_ranks.items():
        feature_prefix = _safe_feature_name(channel_id)
        features[f"{feature_prefix}_rank_inverse"] = _inverse_rank(rank)
        features[f"{feature_prefix}_score"] = float(
            route_score_maps.get(channel_id, {}).get(document.id, 0.0)
        )
    return features


def _rrf_scores(
    *,
    channels: Sequence[Any],
    results_by_channel: Mapping[str, Sequence[RetrievalResult]],
    rrf_k: int,
) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for channel in channels:
        for result in results_by_channel[channel.channel_id]:
            scores[result.document.id] += channel.weight / (rrf_k + result.rank)
    return dict(scores)


def _evaluate_baseline(records: Sequence[_CandidateRecord]) -> dict[str, Any]:
    grouped = _records_by_sample(records)
    ranks = {
        sample_id: _gold_rank(sorted(sample_records, key=lambda record: record.baseline_rank))
        for sample_id, sample_records in grouped.items()
    }
    return _rank_metrics(ranks)


def _evaluate_config(
    *,
    config: Mapping[str, Any],
    train_records: Sequence[_CandidateRecord],
    dev_records: Sequence[_CandidateRecord],
    baseline_metrics: Mapping[str, Mapping[str, Any]],
    selection_rules: Mapping[str, Any],
) -> dict[str, Any]:
    config_id = str(config["config_id"])
    try:
        train_cv_ranks = _evaluate_train_cv_config(
            config=config,
            train_records=train_records,
        )
        train_full_scorer = _fit_scorer_for_config(config, _training_records(config, train_records))
    except ValueError as error:
        return _failed_config_result(
            config=config,
            training_error=error,
            baseline_metrics=baseline_metrics,
        )
    train_full_ranks = _score_records(train_records, train_full_scorer)
    dev_ranks = _score_records(dev_records, train_full_scorer)
    metrics_by_split = {
        "train_cv": _rank_metrics(train_cv_ranks),
        "train_full": _rank_metrics(train_full_ranks),
        "dev": _rank_metrics(dev_ranks),
    }
    comparisons = {
        split: _compare_to_baseline(metrics_by_split[split], baseline_metrics[split])
        for split in ("train_cv", "train_full", "dev")
    }
    train_cv_guards = _train_cv_guard_results(
        config_id=config_id,
        train_records=train_records,
        reranked_ranks=train_cv_ranks,
        baseline_metrics=baseline_metrics["train_cv"],
        comparison=comparisons["train_cv"],
        guard_thresholds=selection_rules["guard_thresholds"],
    )
    objective = _objective_score(
        comparison=comparisons["train_cv"],
        train_cv_guards=train_cv_guards,
        objective_weights=selection_rules["objective_weights"],
    )
    return {
        "config_id": config_id,
        "family_id": config["family_id"],
        "ranking_method": config["ranking_method"],
        "selection_eligible": bool(config.get("selection_eligible")),
        "metrics_by_split": metrics_by_split,
        "comparisons_to_baseline": comparisons,
        "train_cv_selection_guards": train_cv_guards,
        "train_cv_objective_score": round(objective, 6),
        "train_cv_selectable": bool(config.get("selection_eligible"))
        and all(guard["passed"] for guard in train_cv_guards),
        "training_status": "succeeded",
        "training_error": None,
    }


def _failed_config_result(
    *,
    config: Mapping[str, Any],
    training_error: ValueError,
    baseline_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    config_id = str(config["config_id"])
    comparisons = {
        split: _compare_to_baseline(baseline_metrics[split], baseline_metrics[split])
        for split in ("train_cv", "train_full", "dev")
    }
    return {
        "config_id": config_id,
        "family_id": config["family_id"],
        "ranking_method": config["ranking_method"],
        "selection_eligible": bool(config.get("selection_eligible")),
        "metrics_by_split": {
            split: baseline_metrics[split] for split in ("train_cv", "train_full", "dev")
        },
        "comparisons_to_baseline": comparisons,
        "train_cv_selection_guards": [
            _config_guard(
                config_id=config_id,
                name="train_cv_config_training_succeeded",
                passed=False,
                observed=str(training_error),
                expected="config trains successfully on every train-CV fold",
            )
        ],
        "train_cv_objective_score": -999.0,
        "train_cv_selectable": False,
        "training_status": "failed",
        "training_error": str(training_error),
    }


def _evaluate_train_cv_config(
    *,
    config: Mapping[str, Any],
    train_records: Sequence[_CandidateRecord],
) -> dict[str, int | None]:
    grouped = _records_by_sample(train_records)
    fold_ids = sorted({record.fold_id for record in train_records if record.fold_id is not None})
    all_ranks: dict[str, int | None] = {}
    for fold_id in fold_ids:
        fit_records = [
            record
            for record in train_records
            if record.fold_id != fold_id
        ]
        validation_records = [
            record
            for record in train_records
            if record.fold_id == fold_id
        ]
        scorer = _fit_scorer_for_config(config, _training_records(config, fit_records))
        all_ranks.update(_score_records(validation_records, scorer))
    missing_samples = set(grouped) - set(all_ranks)
    for sample_id in missing_samples:
        sample_records = grouped[sample_id]
        scorer = _fit_scorer_for_config(config, _training_records(config, train_records))
        all_ranks[sample_id] = _score_one_sample(sample_records, scorer)
    return all_ranks


def _fit_scorer_for_config(
    config: Mapping[str, Any],
    records: Sequence[_CandidateRecord],
) -> _Scorer:
    method = str(config["ranking_method"])
    config_id = str(config["config_id"])
    if method == "deterministic_weighted_score":
        return _DeterministicScorer(weights=_deterministic_weights(config_id, config)).fit(records)
    feature_names = _feature_names_for_config(config)
    if method == "train_cv_logistic_regression":
        return _LogisticGoldScorer(feature_names=feature_names).fit(records)
    if method == "train_cv_ridge_rank_proxy":
        return _RidgeGoldProxyScorer(feature_names=feature_names).fit(records)
    raise ValueError(f"Unknown ranking_method: {method}")


def _feature_names_for_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    feature_names = [
        _FEATURE_ALIASES.get(str(feature), str(feature))
        for feature in config.get("features", [])
    ]
    if not feature_names:
        return (_RRF_SCORE_FEATURE,)
    return tuple(dict.fromkeys(feature_names))


def _project_feature_dict(
    record: _CandidateRecord,
    feature_names: Sequence[str],
) -> dict[str, float]:
    return {feature_name: record.features.get(feature_name, 0.0) for feature_name in feature_names}


def _deterministic_weights(config_id: str, config: Mapping[str, Any]) -> dict[str, float]:
    payload = config.get("payload") or {}
    if config_id == "crf_route_agreement_best_rank_v1":
        return {
            _RRF_SCORE_FEATURE: 1.0,
            "route_hit_count": float(payload.get("route_hit_count_weight") or 1.2),
            "best_route_inverse_rank": float(payload.get("best_rank_weight") or 0.9),
            "dense_route_hit_count": float(payload.get("dense_route_weight") or 0.4),
        }
    if config_id == "crf_lexical_routes_first_v1":
        return {
            _RRF_SCORE_FEATURE: float(payload.get("stage116_rrf_weight") or 1.0),
            "full_document_bm25_rank_inverse": float(payload.get("lexical_route_weight") or 1.4),
            "section_bm25_max_section_rollup_rank_inverse": float(
                payload.get("lexical_route_weight") or 1.4
            ),
            "title_heading_weighted_bm25_rank_inverse": float(
                payload.get("lexical_route_weight") or 1.4
            ),
            "special_token_boosted_bm25_rank_inverse": float(
                payload.get("lexical_route_weight") or 1.4
            ),
            "dense_route_hit_count": float(payload.get("dense_route_weight") or 0.2),
        }
    if config_id == "ldf_title_heading_overlap_v1":
        return {
            _RRF_SCORE_FEATURE: 1.0,
            "query_title_token_overlap": float(payload.get("title_overlap_weight") or 1.5),
            "query_section_heading_overlap": float(payload.get("heading_overlap_weight") or 1.25),
            "query_token_coverage": float(payload.get("coverage_weight") or 0.7),
        }
    if config_id == "ldf_title_heading_body_coverage_v1":
        return {
            _RRF_SCORE_FEATURE: 1.0,
            "query_title_token_overlap": float(payload.get("title_overlap_weight") or 1.1),
            "query_section_heading_overlap": float(payload.get("heading_overlap_weight") or 1.1),
            "query_body_token_coverage": float(payload.get("body_coverage_weight") or 0.9),
            "document_length_bucket": -float(payload.get("long_document_penalty") or 0.2),
        }
    if config_id == "ldf_special_token_title_heading_v1":
        return {
            _RRF_SCORE_FEATURE: float(payload.get("stage116_rrf_weight") or 1.0),
            "query_special_token_match_count": float(payload.get("special_token_weight") or 1.8),
            "title_special_token_match_count": float(payload.get("title_heading_bonus") or 0.8),
            "heading_special_token_match_count": float(payload.get("title_heading_bonus") or 0.8),
        }
    return {_RRF_SCORE_FEATURE: 1.0}


def _training_records(
    config: Mapping[str, Any],
    records: Sequence[_CandidateRecord],
) -> list[_CandidateRecord]:
    if str(config["ranking_method"]) == "deterministic_weighted_score":
        return list(records)
    max_negatives = int((config.get("payload") or {}).get("max_negatives_per_question") or 40)
    grouped = _records_by_sample(records)
    selected: list[_CandidateRecord] = []
    for sample_records in grouped.values():
        gold_records = [record for record in sample_records if record.is_gold]
        if not gold_records:
            continue
        non_gold = [record for record in sample_records if not record.is_gold]
        selected.extend(gold_records)
        selected.extend(sorted(non_gold, key=lambda record: record.baseline_rank)[:max_negatives])
    return selected


def _score_records(
    records: Sequence[_CandidateRecord],
    scorer: _Scorer,
) -> dict[str, int | None]:
    grouped = _records_by_sample(records)
    return {
        sample_id: _score_one_sample(sample_records, scorer)
        for sample_id, sample_records in grouped.items()
    }


def _score_one_sample(
    sample_records: Sequence[_CandidateRecord],
    scorer: _Scorer,
) -> int | None:
    scores = scorer.score(sample_records)
    ordered = [
        record
        for _, record in sorted(
            zip(scores, sample_records, strict=True),
            key=lambda item: (-item[0], item[1].baseline_rank),
        )
    ]
    return _gold_rank(ordered)


def _records_by_sample(
    records: Sequence[_CandidateRecord],
) -> dict[str, list[_CandidateRecord]]:
    grouped: dict[str, list[_CandidateRecord]] = defaultdict(list)
    for record in records:
        grouped[record.sample_id].append(record)
    return dict(grouped)


def _gold_rank(records: Sequence[_CandidateRecord]) -> int | None:
    for rank, record in enumerate(records, start=1):
        if record.is_gold:
            return rank
    return None


def _rank_metrics(ranks: Mapping[str, int | None]) -> dict[str, Any]:
    evaluated_count = len(ranks)
    hit_counts = {
        top_k: sum(1 for rank in ranks.values() if rank is not None and rank <= top_k)
        for top_k in _TOP_K_VALUES
    }
    present_ranks = [rank for rank in ranks.values() if rank is not None]
    return {
        "evaluated_questions": evaluated_count,
        "hit_counts": hit_counts,
        "hit_at_k": {
            str(top_k): _rounded_ratio(hit_counts[top_k], evaluated_count)
            for top_k in _TOP_K_VALUES
        },
        "mrr_at_20": _rounded_mean(
            [1 / rank if rank is not None and rank <= 20 else 0.0 for rank in ranks.values()]
        ),
        "mrr_at_200": _rounded_mean(
            [1 / rank if rank is not None and rank <= 200 else 0.0 for rank in ranks.values()]
        ),
        "average_present_gold_rank": _rounded_mean(present_ranks),
        "missing_count_at_200": evaluated_count - hit_counts[200],
    }


def _compare_to_baseline(
    metrics: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "mrr_at_20_delta": round(float(metrics["mrr_at_20"]) - float(baseline["mrr_at_20"]), 6),
        "mrr_at_200_delta": round(float(metrics["mrr_at_200"]) - float(baseline["mrr_at_200"]), 6),
        "average_present_gold_rank_delta": round(
            float(metrics["average_present_gold_rank"])
            - float(baseline["average_present_gold_rank"]),
            6,
        ),
        "missing_count_at_200_delta": int(metrics["missing_count_at_200"])
        - int(baseline["missing_count_at_200"]),
    }
    for top_k in _TOP_K_VALUES:
        comparison[f"hit@{top_k}_delta"] = round(
            float(metrics["hit_at_k"][str(top_k)])
            - float(baseline["hit_at_k"][str(top_k)]),
            6,
        )
        comparison[f"hit@{top_k}_count_delta"] = (
            int(metrics["hit_counts"][top_k]) - int(baseline["hit_counts"][top_k])
        )
    return comparison


def _train_cv_guard_results(
    *,
    config_id: str,
    train_records: Sequence[_CandidateRecord],
    reranked_ranks: Mapping[str, int | None],
    baseline_metrics: Mapping[str, Any],
    comparison: Mapping[str, Any],
    guard_thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    baseline_ranks = {
        sample_id: _gold_rank(sorted(sample_records, key=lambda record: record.baseline_rank))
        for sample_id, sample_records in _records_by_sample(train_records).items()
    }
    top20_regression_count = sum(
        1
        for sample_id, baseline_rank in baseline_ranks.items()
        if baseline_rank is not None
        and baseline_rank <= 20
        and (reranked_ranks.get(sample_id) is None or reranked_ranks[sample_id] > 20)
    )
    top10_regression_count = sum(
        1
        for sample_id, baseline_rank in baseline_ranks.items()
        if baseline_rank is not None
        and baseline_rank <= 10
        and (reranked_ranks.get(sample_id) is None or reranked_ranks[sample_id] > 10)
    )
    hit200_loss_count = max(0, -int(comparison["hit@200_count_delta"]))
    bm25_top10_demotions = _bm25_top10_gold_demotions(
        train_records=train_records,
        reranked_ranks=reranked_ranks,
    )
    evaluated = int(baseline_metrics["evaluated_questions"])
    top20_regression_rate = _rounded_ratio(top20_regression_count, evaluated)
    return [
        _config_guard(
            config_id=config_id,
            name="train_cv_hit_at_200_loss_count_within_guard",
            passed=hit200_loss_count
            <= int(guard_thresholds["maximum_train_cv_hit_at_200_loss_count"]),
            observed=hit200_loss_count,
            expected=f"<= {guard_thresholds['maximum_train_cv_hit_at_200_loss_count']}",
        ),
        _config_guard(
            config_id=config_id,
            name="train_cv_bm25_top10_gold_demotions_to_below_50_within_guard",
            passed=bm25_top10_demotions
            <= int(guard_thresholds["maximum_train_cv_bm25_top10_gold_demotions_to_below_50"]),
            observed=bm25_top10_demotions,
            expected=(
                "<= "
                f"{guard_thresholds['maximum_train_cv_bm25_top10_gold_demotions_to_below_50']}"
            ),
        ),
        _config_guard(
            config_id=config_id,
            name="train_cv_hit_at_20_regression_rate_within_guard",
            passed=top20_regression_rate
            <= float(guard_thresholds["maximum_train_cv_hit_at_20_regression_rate"]),
            observed=top20_regression_rate,
            expected=f"<= {guard_thresholds['maximum_train_cv_hit_at_20_regression_rate']}",
        ),
        _config_guard(
            config_id=config_id,
            name="train_cv_top10_regression_count_within_guard",
            passed=top10_regression_count
            <= int(guard_thresholds["maximum_train_cv_top10_regression_count"]),
            observed=top10_regression_count,
            expected=f"<= {guard_thresholds['maximum_train_cv_top10_regression_count']}",
        ),
        _config_guard(
            config_id=config_id,
            name="train_cv_mrr_at_20_delta_non_negative",
            passed=float(comparison["mrr_at_20_delta"])
            >= float(guard_thresholds["minimum_train_cv_mrr_at_20_delta"]),
            observed=comparison["mrr_at_20_delta"],
            expected=f">= {guard_thresholds['minimum_train_cv_mrr_at_20_delta']}",
        ),
    ]


def _bm25_top10_gold_demotions(
    *,
    train_records: Sequence[_CandidateRecord],
    reranked_ranks: Mapping[str, int | None],
) -> int:
    demotions = 0
    for sample_id, sample_records in _records_by_sample(train_records).items():
        gold_records = [record for record in sample_records if record.is_gold]
        if not gold_records:
            continue
        gold = gold_records[0]
        if gold.features.get("full_document_bm25_rank_inverse", 0.0) >= 1 / 10:
            reranked_rank = reranked_ranks.get(sample_id)
            if reranked_rank is None or reranked_rank > 50:
                demotions += 1
    return demotions


def _objective_score(
    *,
    comparison: Mapping[str, Any],
    train_cv_guards: Sequence[Mapping[str, Any]],
    objective_weights: Mapping[str, Any],
) -> float:
    guard_observed = {guard["name"]: guard["observed"] for guard in train_cv_guards}
    return (
        float(comparison["mrr_at_20_delta"]) * float(objective_weights["mrr_at_20_delta"])
        + float(comparison["hit@10_delta"]) * float(objective_weights["hit_at_10_delta"])
        + float(comparison["hit@20_delta"]) * float(objective_weights["hit_at_20_delta"])
        - float(guard_observed["train_cv_bm25_top10_gold_demotions_to_below_50_within_guard"])
        * float(objective_weights["bm25_top10_gold_demotion_penalty"])
        - max(0, -int(comparison["hit@200_count_delta"]))
        * float(objective_weights["candidate_pool_recall_loss_penalty"])
    )


def _select_train_cv_config(
    *,
    config_results: Sequence[Mapping[str, Any]],
    selection_rules: Mapping[str, Any],
) -> dict[str, Any]:
    selectable = [result for result in config_results if result["train_cv_selectable"]]
    if not selectable:
        return {
            "selection_split": "train",
            "selection_source": "train_cv_only",
            "selected_config_id": None,
            "selected_family_id": None,
            "selectable_config_count": 0,
            "config_count": len(config_results),
            "status": "no_train_cv_selectable_config",
            "dev_used_for_selection": False,
            "dev_used_for_retuning": False,
        }
    ordered = sorted(
        selectable,
        key=lambda result: (
            -float(result["train_cv_objective_score"]),
            int(
                _guard_observed(
                    result,
                    "train_cv_bm25_top10_gold_demotions_to_below_50_within_guard",
                )
            ),
            -float(result["comparisons_to_baseline"]["train_cv"]["mrr_at_20_delta"]),
            str(result["config_id"]),
        ),
    )
    selected = ordered[0]
    return {
        "selection_split": "train",
        "selection_source": "train_cv_only",
        "selection_mode": selection_rules["selection_mode"],
        "selected_config_id": selected["config_id"],
        "selected_family_id": selected["family_id"],
        "selected_train_cv_objective_score": selected["train_cv_objective_score"],
        "selected_train_cv_comparison": selected["comparisons_to_baseline"]["train_cv"],
        "selectable_config_count": len(selectable),
        "config_count": len(config_results),
        "status": "train_cv_selected",
        "dev_used_for_selection": False,
        "dev_used_for_retuning": False,
    }


def _guard_observed(result: Mapping[str, Any], guard_name: str) -> Any:
    for guard in result["train_cv_selection_guards"]:
        if guard["name"] == guard_name:
            return guard["observed"]
    return None


def _dev_validation_summary(
    *,
    config_results: Sequence[Mapping[str, Any]],
    selected_config_id: str | None,
) -> dict[str, Any]:
    if selected_config_id is None:
        return {
            "validation_split": "dev",
            "selected_config_id": None,
            "status": "no_train_cv_selectable_config",
            "dev_used_for_selection": False,
            "dev_used_for_retuning": False,
        }
    selected = next(
        result
        for result in config_results
        if result["config_id"] == selected_config_id
    )
    return {
        "validation_split": "dev",
        "selected_config_id": selected_config_id,
        "status": "reported_not_used_for_selection",
        "dev_comparison_to_baseline": selected["comparisons_to_baseline"]["dev"],
        "dev_metrics": selected["metrics_by_split"]["dev"],
        "dev_used_for_selection": False,
        "dev_used_for_retuning": False,
    }


def _guard_checks(
    *,
    stage117_summary: Mapping[str, Any],
    user_confirmed_validation: bool,
    confirmation_note: str,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    dense_summary: Mapping[str, Any],
    baseline_metrics: Mapping[str, Mapping[str, Any]],
    stage116_summary: Mapping[str, Any],
    candidate_pool_depth: int,
    records_by_split: Mapping[str, Sequence[_CandidateRecord]],
    train_cv_selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _check(
            name="user_confirmed_stage118_validation",
            passed=user_confirmed_validation,
            observed=confirmation_note,
            expected="user confirmed Stage118 second-stage reranking validation",
        ),
        _check(
            name="stage117_protocol_frozen",
            passed=stage117_summary.get("decision_status") == _SOURCE_STAGE117_STATUS,
            observed=stage117_summary.get("decision_status"),
            expected=_SOURCE_STAGE117_STATUS,
        ),
        _check(
            name="stage117_protocol_id_matches",
            passed=stage117_summary.get("protocol_id") == _SOURCE_PROTOCOL_ID,
            observed=stage117_summary.get("protocol_id"),
            expected=_SOURCE_PROTOCOL_ID,
        ),
        _check(
            name="stage118_uses_only_train_dev_splits",
            passed=set(split_samples) == set(_ALLOWED_DEVELOPMENT_SPLITS),
            observed=sorted(split_samples),
            expected=list(_ALLOWED_DEVELOPMENT_SPLITS),
        ),
        _check(
            name="stage118_test_split_not_loaded",
            passed=True,
            observed="test split path is not accepted by Stage118 runner",
            expected="no test split load",
        ),
        _check(
            name="stage118_candidate_pool_depth_matches_protocol",
            passed=candidate_pool_depth == int(stage117_summary.get("candidate_pool_depth") or 200),
            observed=candidate_pool_depth,
            expected=stage117_summary.get("candidate_pool_depth"),
        ),
        _check(
            name="stage118_dense_channels_ready_or_disabled",
            passed=bool(dense_summary.get("can_run_without_download")),
            observed=dense_summary.get("status"),
            expected="dense ready or explicitly disabled without download",
        ),
        _check(
            name="stage118_no_model_download_attempted",
            passed=bool(dense_summary.get("no_model_download_attempted")),
            observed=dense_summary.get("no_model_download_attempted"),
            expected=True,
        ),
        _check(
            name="stage118_train_baseline_top200_matches_stage116",
            passed=baseline_metrics["train_cv"]["hit_at_k"]["200"]
            == stage116_summary.get("train_union_hit_at_200"),
            observed=baseline_metrics["train_cv"]["hit_at_k"]["200"],
            expected=stage116_summary.get("train_union_hit_at_200"),
        ),
        _check(
            name="stage118_dev_baseline_top200_matches_stage116",
            passed=baseline_metrics["dev"]["hit_at_k"]["200"]
            == stage116_summary.get("dev_union_hit_at_200"),
            observed=baseline_metrics["dev"]["hit_at_k"]["200"],
            expected=stage116_summary.get("dev_union_hit_at_200"),
        ),
        _check(
            name="stage118_candidate_rows_not_written",
            passed=True,
            observed="candidate records built in memory only",
            expected="no candidate row artifact written",
        ),
        _check(
            name="stage118_dev_not_used_for_selection_or_retuning",
            passed=train_cv_selection.get("dev_used_for_selection") is False
            and train_cv_selection.get("dev_used_for_retuning") is False,
            observed={
                "dev_used_for_selection": train_cv_selection.get("dev_used_for_selection"),
                "dev_used_for_retuning": train_cv_selection.get("dev_used_for_retuning"),
            },
            expected=False,
        ),
        _check(
            name="stage118_runtime_defaults_unchanged",
            passed=True,
            observed="offline reranking validation only",
            expected="runtime defaults unchanged",
        ),
        _check(
            name="stage118_fallback_strategies_not_added",
            passed=True,
            observed="fixed candidate-pool reranking validation only",
            expected="no fallback strategies",
        ),
        _check(
            name="stage118_protocol_feature_aliases_remove_gold_label_feature",
            passed=_FEATURE_ALIASES.get("bm25_top10_non_gold_indicator")
            == "bm25_top10_indicator",
            observed=dict(_FEATURE_ALIASES),
            expected="protocol feature alias uses only runtime-visible BM25 top10 membership",
        ),
        _check(
            name="stage118_public_report_has_no_raw_candidate_ids",
            passed=True,
            observed={
                split: len(records)
                for split, records in records_by_split.items()
            },
            expected="records counted but not serialized",
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    train_cv_selection: Mapping[str, Any],
) -> dict[str, Any]:
    if not all(check["passed"] for check in guard_checks):
        return {
            "status": "primeqa_hybrid_second_stage_reranking_validation_blocked",
            "recommended_next_direction": "fix_stage118_validation_blockers",
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    if train_cv_selection.get("selected_config_id") is None:
        return {
            "status": (
                "primeqa_hybrid_second_stage_reranking_completed_"
                "no_train_cv_selectable_config"
            ),
            "recommended_next_direction": "record_second_stage_reranking_stop_decision",
            "can_continue_train_dev_development": True,
            "selected_config_id": None,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": "primeqa_hybrid_second_stage_reranking_completed_train_cv_selected_dev_reported",
        "recommended_next_direction": "review_second_stage_reranking_changed_cases",
        "can_continue_train_dev_development": True,
        "selected_config_id": train_cv_selection.get("selected_config_id"),
        "selected_family_id": train_cv_selection.get("selected_family_id"),
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }


def _candidate_pool_summary(
    records_by_split: Mapping[str, Sequence[_CandidateRecord]],
) -> dict[str, Any]:
    summary = {}
    for split, records in records_by_split.items():
        grouped = _records_by_sample(records)
        gold_present = sum(
            any(record.is_gold for record in sample_records)
            for sample_records in grouped.values()
        )
        pool_sizes = [len(sample_records) for sample_records in grouped.values()]
        summary[split] = {
            "evaluated_answerable_questions": len(grouped),
            "candidate_record_count_in_memory": len(records),
            "gold_present_in_top200_count": gold_present,
            "gold_present_in_top200_rate": _rounded_ratio(gold_present, len(grouped)),
            "average_pool_size": _rounded_mean(pool_sizes),
            "raw_candidate_rows_written": False,
        }
    return summary


def _stage117_summary(stage117_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage117_report.get("decision") or {}
    frozen = stage117_report.get("frozen_protocol") or {}
    pool_contract = frozen.get("fixed_candidate_pool_contract") or {}
    stage116_summary = stage117_report.get("stage116_summary") or {}
    return {
        "stage": stage117_report.get("stage"),
        "protocol_id": stage117_report.get("protocol_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "candidate_pool_depth": pool_contract.get("candidate_pool_depth"),
        "candidate_config_count": len(frozen.get("candidate_configs") or []),
        "selection_rules": frozen.get("selection_rules") or {},
        "stage116_summary": stage116_summary,
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


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
        "raw_candidate_rows_written": False,
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


def _config_guard(
    *,
    config_id: str,
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> dict[str, Any]:
    return {
        "config_id": config_id,
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


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


def _objective_score_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return sorted(
        [
            BarDatum(
                label=str(result["config_id"]),
                value=float(result["train_cv_objective_score"]),
                value_label=f"{float(result['train_cv_objective_score']):+.4f}",
            )
            for result in report["config_results"]
        ],
        key=lambda bar: (-bar.value, bar.label),
    )


def _metric_delta_bars(report: Mapping[str, Any], *, split: str, metric: str) -> list[BarDatum]:
    return sorted(
        [
            BarDatum(
                label=str(result["config_id"]),
                value=float(result["comparisons_to_baseline"][split][metric]),
                value_label=f"{float(result['comparisons_to_baseline'][split][metric]):+.4f}",
            )
            for result in report["config_results"]
        ],
        key=lambda bar: (-bar.value, bar.label),
    )


def _config_guard_pass_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return sorted(
        [
            BarDatum(
                label=str(result["config_id"]),
                value=sum(1 for guard in result["train_cv_selection_guards"] if guard["passed"]),
                value_label=(
                    f"{sum(1 for guard in result['train_cv_selection_guards'] if guard['passed'])}"
                    f" / {len(result['train_cv_selection_guards'])}"
                ),
            )
            for result in report["config_results"]
        ],
        key=lambda bar: (-bar.value, bar.label),
    )


def _dev_selected_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    selected_config_id = report["train_cv_selection"].get("selected_config_id")
    if selected_config_id is None:
        return []
    selected = next(
        result for result in report["config_results"] if result["config_id"] == selected_config_id
    )
    comparison = selected["comparisons_to_baseline"]["dev"]
    metrics = ("mrr_at_20_delta", "hit@10_delta", "hit@20_delta", "hit@200_delta")
    return [
        BarDatum(
            label=metric,
            value=float(comparison[metric]),
            value_label=f"{float(comparison[metric]):+.4f}",
        )
        for metric in metrics
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report["decision"]
    flags = (
        "can_continue_train_dev_development",
        "can_open_final_test_gate_now",
        "can_run_final_test_metrics_now",
        "can_use_test_for_tuning",
        "fallback_strategies_enabled",
    )
    return [
        BarDatum(
            label=flag,
            value=1.0 if decision.get(flag) else 0.0,
            value_label=str(bool(decision.get(flag))).lower(),
        )
        for flag in flags
    ]


def _guard_check_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(check["name"]),
            value=1.0 if check["passed"] else 0.0,
            value_label="pass" if check["passed"] else "fail",
        )
        for check in report["guard_checks"]
    ]


def _safe_feature_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def _inverse_rank(rank: int | None) -> float:
    return 1.0 / rank if rank is not None and rank > 0 else 0.0


def _overlap_count(left: set[str], right: set[str]) -> float:
    return float(len(left & right))


def _coverage(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    return len(query_tokens & candidate_tokens) / len(query_tokens) if query_tokens else 0.0


def _rounded_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _rounded_mean(values: Sequence[float | int]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
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
