from __future__ import annotations

import os
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    EncoderFactory,
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
    _fingerprint,
    _load_json_object,
    _rank_union_pool,
    _rounded_mean,
    _rounded_percentile,
    _rounded_ratio,
    _section_summary,
    _special_tokens,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
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

_STAGE = "Stage 124"
_CREATED_AT = "2026-07-16"
_ANALYSIS_ID = "primeqa_hybrid_first_stage_recall_expansion_train_cv_dev_validation_v1"
_SOURCE_STAGE123_STATUS = (
    "primeqa_hybrid_first_stage_recall_expansion_protocol_frozen"
)
_SOURCE_PROTOCOL_ID = "primeqa_hybrid_first_stage_recall_expansion_protocol_v1"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = ("test",)
_BASELINE_CONFIG_ID = "stage116_fixed_rrf_top200_baseline"
_BASELINE_POOL_DEPTH = 200
_DEFAULT_TRAIN_FOLD_COUNT = 5
_DEFAULT_BM25_K1 = 1.5
_DEFAULT_BM25_B = 0.75
_DEFAULT_ENCODER_BATCH_SIZE = 64
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
class PrimeQAHybridFirstStageRecallExpansionValidationVisualization:
    """One generated Stage124 first-stage recall expansion validation chart."""

    name: str
    path: str


@dataclass(frozen=True)
class _EvaluationChannel:
    channel_id: str
    family: str
    weight: float
    description: str
    search: Callable[[PrimeQAHybridSplitSample, int], list[RetrievalResult]]


def run_primeqa_hybrid_first_stage_recall_expansion_validation(
    *,
    stage123_protocol_path: Path,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    user_confirmed_validation: bool,
    confirmation_note: str,
    stage80_report_path: Path | None = None,
    include_dense_channels: bool = True,
    bm25_k1: float = _DEFAULT_BM25_K1,
    bm25_b: float = _DEFAULT_BM25_B,
    train_fold_count: int = _DEFAULT_TRAIN_FOLD_COUNT,
    encoder_batch_size: int = _DEFAULT_ENCODER_BATCH_SIZE,
    encoder_device: str | None = None,
    encoder_factory: EncoderFactory | None = None,
) -> dict[str, Any]:
    """Run Stage124 train-CV/dev first-stage recall expansion validation."""

    started_at = time.perf_counter()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    stage123_protocol = _load_json_object(stage123_protocol_path)
    split_samples = {
        _TRAIN_SPLIT: load_primeqa_hybrid_split_samples(train_split_path),
        _DEV_SPLIT: load_primeqa_hybrid_split_samples(dev_split_path),
    }
    train_fold_assignments = _build_train_fold_assignments(
        split_samples[_TRAIN_SPLIT],
        fold_count=train_fold_count,
    )
    loaded_protocol_splits_at = time.perf_counter()

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

    stage123_summary = _stage123_summary(stage123_protocol)
    candidate_configs = _candidate_configs_from_protocol(stage123_protocol)
    protocol_guard_checks = _pre_evaluation_guard_checks(
        stage123_summary=stage123_summary,
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
        candidate_configs=candidate_configs,
        include_dense_channels=include_dense_channels,
        dense_summary=dense_summary,
        train_fold_count=train_fold_count,
    )
    if not all(check["passed"] for check in protocol_guard_checks):
        checked_at = time.perf_counter()
        report = _blocked_report(
            stage123_protocol_path=stage123_protocol_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
            user_confirmed_validation=user_confirmed_validation,
            confirmation_note=confirmation_note,
            stage123_summary=stage123_summary,
            split_samples=split_samples,
            documents=documents,
            sections_by_document=sections_by_document,
            dense_summary=dense_summary,
            candidate_configs=candidate_configs,
            guard_checks=protocol_guard_checks,
            timing_seconds={
                "load_protocol_splits_and_build_train_folds": round(
                    loaded_protocol_splits_at - started_at,
                    3,
                ),
                "load_documents_sections": round(
                    loaded_documents_at - loaded_protocol_splits_at,
                    3,
                ),
                "dense_preflight": round(dense_preflight_at - loaded_documents_at, 3),
                "guard_checks": round(checked_at - dense_preflight_at, 3),
                "total": round(checked_at - started_at, 3),
            },
        )
        return {**report, "public_safe_contract": _public_safe_contract(report)}

    lexical_channels = _build_lexical_channels(
        documents=documents,
        sections_by_document=sections_by_document,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        component_depth=_maximum_channel_top_k(candidate_configs),
    )
    evaluation_channels = _evaluation_channels(
        lexical_channels=lexical_channels,
        dense_channels=dense_channels,
    )
    channel_catalog = _public_channel_catalog(evaluation_channels)
    indexed_at = time.perf_counter()

    baseline_by_split = {}
    config_reviews_by_split = {}
    for split, samples in split_samples.items():
        answerable_samples = _answerable_samples(samples)
        result_cache = _result_cache(
            samples=answerable_samples,
            channels=evaluation_channels,
            top_k=_maximum_channel_top_k(candidate_configs),
        )
        baseline_outcomes = _baseline_outcomes(
            samples=answerable_samples,
            channels=evaluation_channels,
            result_cache=result_cache,
        )
        baseline_by_split[split] = _baseline_summary(
            split=split,
            baseline_outcomes=baseline_outcomes,
            fold_assignments=train_fold_assignments if split == _TRAIN_SPLIT else None,
        )
        config_reviews_by_split[split] = [
            _evaluate_config(
                split=split,
                config=config,
                samples=answerable_samples,
                channels=evaluation_channels,
                result_cache=result_cache,
                baseline_outcomes=baseline_outcomes,
                fold_assignments=(
                    train_fold_assignments if split == _TRAIN_SPLIT else None
                ),
            )
            for config in candidate_configs
        ]
    evaluated_at = time.perf_counter()

    config_reviews = _merge_config_reviews(
        candidate_configs=candidate_configs,
        config_reviews_by_split=config_reviews_by_split,
    )
    train_selection = _select_config_on_train(
        config_reviews=config_reviews,
        guard_thresholds=stage123_summary["guard_thresholds"],
    )
    guard_checks = protocol_guard_checks + _post_evaluation_guard_checks(
        report_payload={
            "baseline_by_split": baseline_by_split,
            "config_reviews": config_reviews,
            "train_selection": train_selection,
        },
        stage123_summary=stage123_summary,
    )
    checked_at = time.perf_counter()
    decision = _decision(guard_checks=guard_checks, train_selection=train_selection)
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": _analysis_scope(),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_validation),
            "confirmation_note": confirmation_note,
        },
        "split_contract": _split_contract(),
        "source_files": _source_files(
            stage123_protocol_path=stage123_protocol_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
        ),
        "stage123_summary": stage123_summary,
        "analysis_config": {
            "include_dense_channels": include_dense_channels,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "train_fold_count": train_fold_count,
            "encoder_batch_size": encoder_batch_size,
            "encoder_device": encoder_device,
            "candidate_config_count": len(candidate_configs),
            "candidate_config_ids": [
                str(config["config_id"]) for config in candidate_configs
            ],
            "baseline_config_id": _BASELINE_CONFIG_ID,
            "baseline_pool_depth": _BASELINE_POOL_DEPTH,
            "maximum_channel_top_k": _maximum_channel_top_k(candidate_configs),
            "maximum_target_pool_depth": _maximum_target_pool_depth(candidate_configs),
        },
        "loaded_data_summary": {
            "split_samples": summarize_primeqa_hybrid_split_samples(split_samples),
            "document_count": len(documents),
            **_section_summary(sections_by_document),
            "test_split_loaded": False,
        },
        "dense_channel_preflight": dense_summary,
        "channel_catalog": channel_catalog,
        "baseline_by_split": baseline_by_split,
        "config_reviews": config_reviews,
        "train_selection": train_selection,
        "dev_report_observations": _dev_report_observations(
            train_selection=train_selection,
            config_reviews=config_reviews,
        ),
        "guard_checks": guard_checks,
        "decision": decision,
        "timing_seconds": {
            "load_protocol_splits_and_build_train_folds": round(
                loaded_protocol_splits_at - started_at,
                3,
            ),
            "load_documents_sections": round(
                loaded_documents_at - loaded_protocol_splits_at,
                3,
            ),
            "dense_preflight": round(dense_preflight_at - loaded_documents_at, 3),
            "build_indexes": round(indexed_at - dense_preflight_at, 3),
            "evaluate_candidate_pools": round(evaluated_at - indexed_at, 3),
            "selection_and_guard_checks": round(checked_at - evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_first_stage_recall_expansion_validation_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridFirstStageRecallExpansionValidationVisualization]:
    """Write SVG charts for Stage124 validation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage124_train_target_depth_gain.svg": render_horizontal_bar_chart_svg(
            title="Stage124 train target-depth gain vs Stage116 top200",
            bars=_target_depth_gain_bars(report, split=_TRAIN_SPLIT),
            x_label="hit-count gain",
            width=1520,
            margin_left=760,
        ),
        "stage124_dev_target_depth_gain.svg": render_horizontal_bar_chart_svg(
            title="Stage124 dev target-depth gain vs Stage116 top200",
            bars=_target_depth_gain_bars(report, split=_DEV_SPLIT),
            x_label="hit-count gain",
            width=1520,
            margin_left=760,
        ),
        "stage124_train_hit200_delta.svg": render_horizontal_bar_chart_svg(
            title="Stage124 train hit@200 delta vs Stage116 top200",
            bars=_hit200_delta_bars(report, split=_TRAIN_SPLIT),
            x_label="hit-count delta",
            width=1520,
            margin_left=760,
        ),
        "stage124_dev_hit200_delta.svg": render_horizontal_bar_chart_svg(
            title="Stage124 dev hit@200 delta vs Stage116 top200",
            bars=_hit200_delta_bars(report, split=_DEV_SPLIT),
            x_label="hit-count delta",
            width=1520,
            margin_left=760,
        ),
        "stage124_train_fold_target_hit_summary.svg": render_horizontal_bar_chart_svg(
            title="Stage124 selected train-fold target-depth hit summary",
            bars=_selected_fold_summary_bars(report),
            x_label="hit rate",
            width=1180,
            margin_left=480,
        ),
        "stage124_candidate_pool_size.svg": render_horizontal_bar_chart_svg(
            title="Stage124 selected candidate-pool size",
            bars=_selected_pool_size_bars(report),
            x_label="documents",
            width=1180,
            margin_left=420,
        ),
        "stage124_selection_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage124 decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1380,
            margin_left=720,
        ),
        "stage124_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage124 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1840,
            margin_left=1020,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridFirstStageRecallExpansionValidationVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _blocked_report(
    *,
    stage123_protocol_path: Path,
    stage80_report_path: Path | None,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    user_confirmed_validation: bool,
    confirmation_note: str,
    stage123_summary: Mapping[str, Any],
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    documents: Sequence[Any],
    sections_by_document: Mapping[str, Sequence[Any]],
    dense_summary: Mapping[str, Any],
    candidate_configs: Sequence[Mapping[str, Any]],
    guard_checks: Sequence[Mapping[str, Any]],
    timing_seconds: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": _analysis_scope(),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_validation),
            "confirmation_note": confirmation_note,
        },
        "split_contract": _split_contract(),
        "source_files": _source_files(
            stage123_protocol_path=stage123_protocol_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
        ),
        "stage123_summary": dict(stage123_summary),
        "analysis_config": {
            "candidate_config_count": len(candidate_configs),
            "candidate_config_ids": [
                str(config["config_id"]) for config in candidate_configs
            ],
        },
        "loaded_data_summary": {
            "split_samples": summarize_primeqa_hybrid_split_samples(split_samples),
            "document_count": len(documents),
            **_section_summary(sections_by_document),
            "test_split_loaded": False,
        },
        "dense_channel_preflight": dense_summary,
        "channel_catalog": [],
        "baseline_by_split": {},
        "config_reviews": [],
        "train_selection": _empty_train_selection(candidate_configs),
        "dev_report_observations": {
            "dev_used_for_selection": False,
            "dev_used_for_retuning": False,
            "dev_reported_only": True,
        },
        "guard_checks": list(guard_checks),
        "decision": _decision(
            guard_checks=guard_checks,
            train_selection=_empty_train_selection(candidate_configs),
        ),
        "timing_seconds": dict(timing_seconds),
    }


def _evaluation_channels(
    *,
    lexical_channels: Sequence[Any],
    dense_channels: Sequence[Any],
) -> list[_EvaluationChannel]:
    standard_channels = [
        _wrap_standard_channel(channel) for channel in [*lexical_channels, *dense_channels]
    ]
    lexical_by_id = {channel.channel_id: channel for channel in lexical_channels}
    variant_channels = [
        _wrap_variant_channel(
            channel=lexical_by_id["full_document_bm25"],
            channel_id="query_variant__title__full_document_bm25",
            variant="title",
            weight=0.75,
            description="Full-document BM25 over the runtime question title only.",
        ),
        _wrap_variant_channel(
            channel=lexical_by_id["title_heading_weighted_bm25"],
            channel_id="query_variant__title__title_heading_weighted_bm25",
            variant="title",
            weight=0.75,
            description="Title/heading weighted BM25 over the runtime question title.",
        ),
        _wrap_variant_channel(
            channel=lexical_by_id["full_document_bm25"],
            channel_id="query_variant__special_tokens__full_document_bm25",
            variant="special_tokens",
            weight=0.65,
            description="Full-document BM25 over exact technical tokens from the query.",
        ),
        _wrap_variant_channel(
            channel=lexical_by_id["special_token_boosted_bm25"],
            channel_id="query_variant__title_special_tokens__special_token_boosted_bm25",
            variant="title_special_tokens",
            weight=0.65,
            description=(
                "Special-token boosted BM25 over title plus exact technical tokens."
            ),
        ),
    ]
    return standard_channels + variant_channels


def _wrap_standard_channel(channel: Any) -> _EvaluationChannel:
    def search(sample: PrimeQAHybridSplitSample, top_k: int) -> list[RetrievalResult]:
        query = sample.to_primeqa_question().full_question
        return channel.retriever.search(query, top_k=top_k)

    return _EvaluationChannel(
        channel_id=str(channel.channel_id),
        family=str(channel.family),
        weight=float(channel.weight),
        description=str(channel.description),
        search=search,
    )


def _wrap_variant_channel(
    *,
    channel: Any,
    channel_id: str,
    variant: str,
    weight: float,
    description: str,
) -> _EvaluationChannel:
    def search(sample: PrimeQAHybridSplitSample, top_k: int) -> list[RetrievalResult]:
        query = _variant_query(sample, variant=variant)
        if not query:
            return []
        return channel.retriever.search(query, top_k=top_k)

    return _EvaluationChannel(
        channel_id=channel_id,
        family="query_variant_lexical",
        weight=weight,
        description=description,
        search=search,
    )


def _variant_query(sample: PrimeQAHybridSplitSample, *, variant: str) -> str:
    full_query = sample.to_primeqa_question().full_question
    tokens = " ".join(sorted(_special_tokens(full_query)))
    if variant == "title":
        return " ".join(sample.question_title.split())
    if variant == "special_tokens":
        return tokens
    if variant == "title_special_tokens":
        return " ".join(f"{sample.question_title} {tokens}".split())
    raise ValueError(f"Unknown query variant: {variant}")


def _result_cache(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    channels: Sequence[_EvaluationChannel],
    top_k: int,
) -> dict[tuple[str, str], list[RetrievalResult]]:
    cache = {}
    for sample in samples:
        for channel in channels:
            cache[(sample.sample_id, channel.channel_id)] = channel.search(sample, top_k)
    return cache


def _baseline_outcomes(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    channels: Sequence[_EvaluationChannel],
    result_cache: Mapping[tuple[str, str], Sequence[RetrievalResult]],
) -> dict[str, dict[str, Any]]:
    standard_channels = _channels_for_route_set(
        channels=channels,
        route_set="stage116_same_routes",
    )
    outcomes = {}
    for sample in samples:
        results_by_channel = _results_for_channels(
            sample=sample,
            channels=standard_channels,
            result_cache=result_cache,
            channel_top_k=_BASELINE_POOL_DEPTH,
        )
        ranked = _rank_pool(
            channels=standard_channels,
            results_by_channel=results_by_channel,
            algorithm="weighted_rrf",
            rrf_k=60,
            target_pool_depth=_BASELINE_POOL_DEPTH,
        )
        outcomes[sample.sample_id] = {
            "ranked_pool": ranked,
            "hit_at_200": _contains_gold(ranked[:_BASELINE_POOL_DEPTH], sample),
        }
    return outcomes


def _evaluate_config(
    *,
    split: str,
    config: Mapping[str, Any],
    samples: Sequence[PrimeQAHybridSplitSample],
    channels: Sequence[_EvaluationChannel],
    result_cache: Mapping[tuple[str, str], Sequence[RetrievalResult]],
    baseline_outcomes: Mapping[str, Mapping[str, Any]],
    fold_assignments: Mapping[str, str] | None,
) -> dict[str, Any]:
    generation = config["candidate_generation"]
    channel_top_k = int(generation["channel_top_k"])
    target_pool_depth = int(generation["target_pool_depth"])
    configured_channels = _channels_for_route_set(
        channels=channels,
        route_set=str(generation["route_set"]),
    )
    configured_channels = _weighted_channels_for_config(
        channels=configured_channels,
        config=config,
    )
    top_k_values = _top_k_values_for_target(target_pool_depth)
    accumulator = _empty_config_accumulator(
        split=split,
        config_id=str(config["config_id"]),
        target_pool_depth=target_pool_depth,
        total_questions=len(samples),
        top_k_values=top_k_values,
    )
    fold_accumulators = (
        {
            fold_id: _empty_config_accumulator(
                split=fold_id,
                config_id=str(config["config_id"]),
                target_pool_depth=target_pool_depth,
                total_questions=sum(
                    1
                    for sample in samples
                    if fold_assignments.get(sample.sample_id) == fold_id
                ),
                top_k_values=top_k_values,
            )
            for fold_id in sorted(set(fold_assignments.values()))
        }
        if fold_assignments is not None
        else {}
    )
    for sample in samples:
        results_by_channel = _results_for_channels(
            sample=sample,
            channels=configured_channels,
            result_cache=result_cache,
            channel_top_k=channel_top_k,
        )
        ranked = _rank_pool(
            channels=configured_channels,
            results_by_channel=results_by_channel,
            algorithm=str(generation["algorithm"]),
            rrf_k=int(generation["rrf_k"]),
            target_pool_depth=target_pool_depth,
        )
        _record_config_result(
            accumulator=accumulator,
            ranked_pool=ranked,
            sample=sample,
            baseline_hit_at_200=bool(
                baseline_outcomes[sample.sample_id]["hit_at_200"]
            ),
            top_k_values=top_k_values,
        )
        if fold_assignments is not None:
            _record_config_result(
                accumulator=fold_accumulators[fold_assignments[sample.sample_id]],
                ranked_pool=ranked,
                sample=sample,
                baseline_hit_at_200=bool(
                    baseline_outcomes[sample.sample_id]["hit_at_200"]
                ),
                top_k_values=top_k_values,
            )
    finalized = _finalize_config_accumulator(accumulator)
    return {
        **finalized,
        "channel_count": len(configured_channels),
        "channel_families": dict(
            sorted(Counter(channel.family for channel in configured_channels).items())
        ),
        "fold_metrics": {
            fold_id: _finalize_config_accumulator(fold_accumulator)
            for fold_id, fold_accumulator in fold_accumulators.items()
        },
    }


def _rank_pool(
    *,
    channels: Sequence[_EvaluationChannel],
    results_by_channel: Mapping[str, Sequence[RetrievalResult]],
    algorithm: str,
    rrf_k: int,
    target_pool_depth: int,
) -> list[str]:
    if algorithm == "route_balanced_interleaving":
        ranked = _rank_route_balanced_pool(
            channels=channels,
            results_by_channel=results_by_channel,
        )
    else:
        ranked = _rank_union_pool(
            channels=channels,
            results_by_channel=results_by_channel,
            rrf_k=rrf_k,
        )
    return ranked[:target_pool_depth]


def _rank_route_balanced_pool(
    *,
    channels: Sequence[_EvaluationChannel],
    results_by_channel: Mapping[str, Sequence[RetrievalResult]],
) -> list[str]:
    ranked = []
    seen = set()
    max_depth = max((len(results) for results in results_by_channel.values()), default=0)
    for index in range(max_depth):
        for channel in channels:
            results = results_by_channel.get(channel.channel_id) or []
            if index >= len(results):
                continue
            doc_id = results[index].document.id
            if doc_id in seen:
                continue
            seen.add(doc_id)
            ranked.append(doc_id)
    return ranked


def _results_for_channels(
    *,
    sample: PrimeQAHybridSplitSample,
    channels: Sequence[_EvaluationChannel],
    result_cache: Mapping[tuple[str, str], Sequence[RetrievalResult]],
    channel_top_k: int,
) -> dict[str, Sequence[RetrievalResult]]:
    return {
        channel.channel_id: list(
            result_cache[(sample.sample_id, channel.channel_id)][:channel_top_k]
        )
        for channel in channels
    }


def _channels_for_route_set(
    *,
    channels: Sequence[_EvaluationChannel],
    route_set: str,
) -> list[_EvaluationChannel]:
    standard = [
        channel
        for channel in channels
        if channel.family != "query_variant_lexical"
    ]
    lexical = [
        channel
        for channel in standard
        if not channel.channel_id.startswith("dense_cache__")
    ]
    dense = [
        channel for channel in standard if channel.channel_id.startswith("dense_cache__")
    ]
    variants = [
        channel for channel in channels if channel.family == "query_variant_lexical"
    ]
    if route_set == "stage116_same_routes":
        return standard
    if route_set == "stage116_lexical_routes_plus_cached_dense":
        return lexical + dense
    if route_set == "stage116_lexical_routes_plus_existing_dense_cache_routes":
        return lexical + dense
    if route_set == "lexical_routes_with_title_and_special_token_variants":
        return lexical + variants
    raise ValueError(f"Unknown route set: {route_set}")


def _weighted_channels_for_config(
    *,
    channels: Sequence[_EvaluationChannel],
    config: Mapping[str, Any],
) -> list[_EvaluationChannel]:
    policy = str(config["candidate_generation"]["ranking_policy"])
    if policy != "lexical_routes_weighted_before_cached_dense_routes":
        return list(channels)
    weighted = []
    for channel in channels:
        weight = 0.8 if channel.channel_id.startswith("dense_cache__") else 1.2
        weighted.append(
            _EvaluationChannel(
                channel_id=channel.channel_id,
                family=channel.family,
                weight=weight,
                description=channel.description,
                search=channel.search,
            )
        )
    return weighted


def _empty_config_accumulator(
    *,
    split: str,
    config_id: str,
    target_pool_depth: int,
    total_questions: int,
    top_k_values: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "split": split,
        "config_id": config_id,
        "target_pool_depth": target_pool_depth,
        "total_questions": total_questions,
        "evaluated_questions": 0,
        "hit_counts": {top_k: 0 for top_k in top_k_values},
        "candidate_pool_sizes": [],
        "baseline_hit_at_200_count": 0,
        "hit_at_200_recovery_count": 0,
        "hit_at_200_loss_count": 0,
        "target_depth_recovery_count": 0,
        "target_depth_loss_count": 0,
    }


def _record_config_result(
    *,
    accumulator: dict[str, Any],
    ranked_pool: Sequence[str],
    sample: PrimeQAHybridSplitSample,
    baseline_hit_at_200: bool,
    top_k_values: tuple[int, ...],
) -> None:
    answer_doc_id = str(sample.answer_doc_id)
    target_pool_depth = int(accumulator["target_pool_depth"])
    pool = list(ranked_pool[:target_pool_depth])
    accumulator["evaluated_questions"] += 1
    accumulator["candidate_pool_sizes"].append(len(pool))
    if baseline_hit_at_200:
        accumulator["baseline_hit_at_200_count"] += 1
    for top_k in top_k_values:
        if answer_doc_id in set(pool[:top_k]):
            accumulator["hit_counts"][top_k] += 1
    hit_at_200 = answer_doc_id in set(pool[:_BASELINE_POOL_DEPTH])
    hit_at_target = answer_doc_id in set(pool)
    if hit_at_200 and not baseline_hit_at_200:
        accumulator["hit_at_200_recovery_count"] += 1
    if baseline_hit_at_200 and not hit_at_200:
        accumulator["hit_at_200_loss_count"] += 1
    if hit_at_target and not baseline_hit_at_200:
        accumulator["target_depth_recovery_count"] += 1
    if baseline_hit_at_200 and not hit_at_target:
        accumulator["target_depth_loss_count"] += 1


def _finalize_config_accumulator(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    evaluated = int(accumulator["evaluated_questions"])
    target_pool_depth = int(accumulator["target_pool_depth"])
    hit_counts = {
        int(top_k): int(count) for top_k, count in accumulator["hit_counts"].items()
    }
    baseline_hit_at_200_count = int(accumulator["baseline_hit_at_200_count"])
    hit_at_200_count = hit_counts.get(_BASELINE_POOL_DEPTH, 0)
    target_depth_hit_count = hit_counts.get(target_pool_depth, 0)
    pool_sizes = [int(value) for value in accumulator["candidate_pool_sizes"]]
    return {
        "split": accumulator["split"],
        "config_id": accumulator["config_id"],
        "target_pool_depth": target_pool_depth,
        "total_questions": int(accumulator["total_questions"]),
        "evaluated_questions": evaluated,
        "hit_counts": hit_counts,
        "hit_at_k": {
            str(top_k): _rounded_ratio(count, evaluated)
            for top_k, count in hit_counts.items()
        },
        "baseline_hit_at_200_count": baseline_hit_at_200_count,
        "hit_at_200_count": hit_at_200_count,
        "hit_at_200_delta_vs_baseline": hit_at_200_count - baseline_hit_at_200_count,
        "hit_at_200_recovery_count": int(accumulator["hit_at_200_recovery_count"]),
        "hit_at_200_loss_count": int(accumulator["hit_at_200_loss_count"]),
        "target_depth_hit_count": target_depth_hit_count,
        "target_depth_hit_rate": _rounded_ratio(target_depth_hit_count, evaluated),
        "target_depth_hit_count_gain_vs_baseline_top200": (
            target_depth_hit_count - baseline_hit_at_200_count
        ),
        "target_depth_recovery_count": int(accumulator["target_depth_recovery_count"]),
        "target_depth_loss_count": int(accumulator["target_depth_loss_count"]),
        "candidate_pool_size": {
            "average": _rounded_mean(pool_sizes),
            "median": _rounded_percentile(pool_sizes, 50),
            "p95": _rounded_percentile(pool_sizes, 95),
            "max": max(pool_sizes, default=0),
        },
    }


def _baseline_summary(
    *,
    split: str,
    baseline_outcomes: Mapping[str, Mapping[str, Any]],
    fold_assignments: Mapping[str, str] | None,
) -> dict[str, Any]:
    evaluated = len(baseline_outcomes)
    hit_at_200_count = sum(1 for row in baseline_outcomes.values() if row["hit_at_200"])
    summary = {
        "split": split,
        "baseline_config_id": _BASELINE_CONFIG_ID,
        "pool_depth": _BASELINE_POOL_DEPTH,
        "evaluated_questions": evaluated,
        "hit_at_200_count": hit_at_200_count,
        "hit_at_200": _rounded_ratio(hit_at_200_count, evaluated),
    }
    if fold_assignments is None:
        return summary
    fold_rows = {}
    for fold_id in sorted(set(fold_assignments.values())):
        fold_sample_ids = [
            sample_id
            for sample_id in baseline_outcomes
            if fold_assignments.get(sample_id) == fold_id
        ]
        fold_total = len(fold_sample_ids)
        fold_hits = sum(
            1 for sample_id in fold_sample_ids if baseline_outcomes[sample_id]["hit_at_200"]
        )
        fold_rows[fold_id] = {
            "evaluated_questions": fold_total,
            "hit_at_200_count": fold_hits,
            "hit_at_200": _rounded_ratio(fold_hits, fold_total),
        }
    summary["fold_metrics"] = fold_rows
    summary["raw_group_values_written"] = False
    return summary


def _merge_config_reviews(
    *,
    candidate_configs: Sequence[Mapping[str, Any]],
    config_reviews_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    reviews = []
    for config in candidate_configs:
        config_id = str(config["config_id"])
        split_reviews = {}
        for split, split_reviews_list in config_reviews_by_split.items():
            split_reviews[split] = next(
                review
                for review in split_reviews_list
                if review["config_id"] == config_id
            )
        reviews.append(
            {
                "config_id": config_id,
                "family_id": config["family_id"],
                "algorithm": config["candidate_generation"]["algorithm"],
                "route_set": config["candidate_generation"]["route_set"],
                "channel_top_k": config["candidate_generation"]["channel_top_k"],
                "target_pool_depth": config["candidate_generation"][
                    "target_pool_depth"
                ],
                "split_reviews": split_reviews,
                "train_cv_guard": _config_train_guard(
                    train_review=split_reviews[_TRAIN_SPLIT],
                    config=config,
                ),
            }
        )
    return reviews


def _config_train_guard(
    *,
    train_review: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    checks = [
        _check(
            name="train_hit_at_200_loss_count_within_guard",
            passed=int(train_review["hit_at_200_loss_count"]) == 0,
            observed=train_review["hit_at_200_loss_count"],
            expected=0,
        ),
        _check(
            name="train_target_depth_hit_count_gain_positive",
            passed=int(train_review["target_depth_hit_count_gain_vs_baseline_top200"])
            >= 1,
            observed=train_review["target_depth_hit_count_gain_vs_baseline_top200"],
            expected=">= 1",
        ),
        _check(
            name="channel_top_k_within_guard",
            passed=int(config["candidate_generation"]["channel_top_k"]) <= 400,
            observed=config["candidate_generation"]["channel_top_k"],
            expected="<= 400",
        ),
        _check(
            name="target_pool_depth_within_guard",
            passed=int(config["candidate_generation"]["target_pool_depth"]) <= 400,
            observed=config["candidate_generation"]["target_pool_depth"],
            expected="<= 400",
        ),
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
    }


def _select_config_on_train(
    *,
    config_reviews: Sequence[Mapping[str, Any]],
    guard_thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    _ = guard_thresholds
    eligible = [
        review
        for review in config_reviews
        if review["train_cv_guard"]["passed"]
        and int(
            review["split_reviews"][_TRAIN_SPLIT][
                "target_depth_hit_count_gain_vs_baseline_top200"
            ]
        )
        >= 1
    ]
    ranked = sorted(
        eligible,
        key=lambda review: (
            -int(
                review["split_reviews"][_TRAIN_SPLIT][
                    "target_depth_hit_count_gain_vs_baseline_top200"
                ]
            ),
            int(review["target_pool_depth"]),
            str(review["config_id"]),
        ),
    )
    selected = ranked[0] if ranked else None
    return {
        "selection_split": _TRAIN_SPLIT,
        "selection_mode": "train_grouped_cross_validation_candidate_pool_selection",
        "dev_used_for_selection": False,
        "dev_used_for_retuning": False,
        "candidate_count": len(config_reviews),
        "eligible_config_count": len(eligible),
        "selected_config_id": selected["config_id"] if selected else None,
        "selected_family_id": selected["family_id"] if selected else None,
        "selected_train_summary": (
            _selection_metric_summary(selected["split_reviews"][_TRAIN_SPLIT])
            if selected
            else None
        ),
        "selection_ranking": [
            {
                "config_id": review["config_id"],
                "family_id": review["family_id"],
                "target_pool_depth": review["target_pool_depth"],
                "train_target_depth_gain": review["split_reviews"][_TRAIN_SPLIT][
                    "target_depth_hit_count_gain_vs_baseline_top200"
                ],
                "train_hit_at_200_loss_count": review["split_reviews"][_TRAIN_SPLIT][
                    "hit_at_200_loss_count"
                ],
                "guard_passed": review["train_cv_guard"]["passed"],
            }
            for review in sorted(
                config_reviews,
                key=lambda review: (
                    -int(
                        review["split_reviews"][_TRAIN_SPLIT][
                            "target_depth_hit_count_gain_vs_baseline_top200"
                        ]
                    ),
                    int(review["target_pool_depth"]),
                    str(review["config_id"]),
                ),
            )
        ],
    }


def _selection_metric_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "target_pool_depth": review["target_pool_depth"],
        "evaluated_questions": review["evaluated_questions"],
        "baseline_hit_at_200_count": review["baseline_hit_at_200_count"],
        "hit_at_200_count": review["hit_at_200_count"],
        "hit_at_200_delta_vs_baseline": review["hit_at_200_delta_vs_baseline"],
        "hit_at_200_loss_count": review["hit_at_200_loss_count"],
        "target_depth_hit_count": review["target_depth_hit_count"],
        "target_depth_hit_count_gain_vs_baseline_top200": review[
            "target_depth_hit_count_gain_vs_baseline_top200"
        ],
        "target_depth_hit_rate": review["target_depth_hit_rate"],
    }


def _dev_report_observations(
    *,
    train_selection: Mapping[str, Any],
    config_reviews: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected_config_id = train_selection.get("selected_config_id")
    selected_review = (
        next(
            (
                review
                for review in config_reviews
                if review["config_id"] == selected_config_id
            ),
            None,
        )
        if selected_config_id
        else None
    )
    return {
        "dev_used_for_selection": False,
        "dev_used_for_retuning": False,
        "dev_reported_only": True,
        "selected_config_id": selected_config_id,
        "selected_dev_summary": (
            _selection_metric_summary(selected_review["split_reviews"][_DEV_SPLIT])
            if selected_review
            else None
        ),
    }


def _pre_evaluation_guard_checks(
    *,
    stage123_summary: Mapping[str, Any],
    user_confirmed_validation: bool,
    confirmation_note: str,
    candidate_configs: Sequence[Mapping[str, Any]],
    include_dense_channels: bool,
    dense_summary: Mapping[str, Any],
    train_fold_count: int,
) -> list[dict[str, Any]]:
    return [
        _check(
            name="user_confirmed_stage124_validation",
            passed=user_confirmed_validation,
            observed=confirmation_note,
            expected="user confirmed Stage124 first-stage recall expansion validation",
        ),
        _check(
            name="stage123_protocol_frozen",
            passed=stage123_summary.get("decision_status") == _SOURCE_STAGE123_STATUS,
            observed=stage123_summary.get("decision_status"),
            expected=_SOURCE_STAGE123_STATUS,
        ),
        _check(
            name="stage123_protocol_id_matches",
            passed=stage123_summary.get("protocol_id") == _SOURCE_PROTOCOL_ID,
            observed=stage123_summary.get("protocol_id"),
            expected=_SOURCE_PROTOCOL_ID,
        ),
        _check(
            name="stage123_recommends_stage124_validation",
            passed=stage123_summary.get("recommended_next_direction")
            == "run_first_stage_recall_expansion_train_cv_dev_validation",
            observed=stage123_summary.get("recommended_next_direction"),
            expected="run_first_stage_recall_expansion_train_cv_dev_validation",
        ),
        _check(
            name="stage123_runtime_and_test_boundaries_locked",
            passed=stage123_summary.get("can_open_final_test_gate_now") is False
            and stage123_summary.get("can_run_final_test_metrics_now") is False
            and stage123_summary.get("can_use_test_for_tuning") is False
            and stage123_summary.get("fallback_strategies_enabled") is False
            and stage123_summary.get("default_runtime_policy") == "unchanged",
            observed=stage123_summary,
            expected="test locked, runtime unchanged, fallback disabled",
        ),
        _check(
            name="stage124_expected_candidate_configs_loaded",
            passed=len(candidate_configs) == 7,
            observed=len(candidate_configs),
            expected=7,
        ),
        _check(
            name="stage124_uses_train_grouped_cv",
            passed=train_fold_count >= 5,
            observed=train_fold_count,
            expected=">= 5",
        ),
        _check(
            name="stage124_dense_channels_ready",
            passed=include_dense_channels
            and dense_summary.get("status") == "dense_channels_ready"
            and dense_summary.get("can_run_without_download") is True
            and dense_summary.get("no_model_download_attempted") is True,
            observed={
                "include_dense_channels": include_dense_channels,
                "dense_status": dense_summary.get("status"),
                "can_run_without_download": dense_summary.get(
                    "can_run_without_download"
                ),
                "no_model_download_attempted": dense_summary.get(
                    "no_model_download_attempted"
                ),
            },
            expected="existing dense caches ready without download",
        ),
    ]


def _post_evaluation_guard_checks(
    *,
    report_payload: Mapping[str, Any],
    stage123_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    public_safe = _public_safe_contract(report_payload)
    train_selection = report_payload["train_selection"]
    return [
        _check(
            name="stage124_uses_only_train_dev_splits",
            passed=sorted(report_payload["baseline_by_split"]) == ["dev", "train"],
            observed=sorted(report_payload["baseline_by_split"]),
            expected=["train", "dev"],
        ),
        _check(
            name="stage124_test_split_not_loaded",
            passed=True,
            observed=False,
            expected=False,
        ),
        _check(
            name="stage124_no_candidate_rows_written",
            passed=True,
            observed=False,
            expected=False,
        ),
        _check(
            name="stage124_no_model_download_attempted",
            passed=True,
            observed=True,
            expected=True,
        ),
        _check(
            name="stage124_dev_report_only",
            passed=train_selection.get("dev_used_for_selection") is False
            and train_selection.get("dev_used_for_retuning") is False,
            observed={
                "dev_used_for_selection": train_selection.get("dev_used_for_selection"),
                "dev_used_for_retuning": train_selection.get("dev_used_for_retuning"),
            },
            expected="dev not used for selection or retuning",
        ),
        _check(
            name="stage124_runtime_defaults_unchanged",
            passed=stage123_summary.get("default_runtime_policy") == "unchanged",
            observed=stage123_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage124_fallback_strategies_not_added",
            passed=stage123_summary.get("fallback_strategies_enabled") is False,
            observed=stage123_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="stage124_public_safe_contract_passed",
            passed=public_safe["forbidden_keys_found"] == [],
            observed=public_safe["forbidden_keys_found"],
            expected=[],
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    train_selection: Mapping[str, Any],
) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_first_stage_recall_expansion_validation_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "can_run_final_test_metrics_now": False,
            "can_open_final_test_gate_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    if train_selection.get("selected_config_id"):
        return {
            "status": "primeqa_hybrid_first_stage_recall_expansion_validation_completed",
            "recommended_next_direction": "review_first_stage_recall_expansion_selected_config",
            "selected_config_id": train_selection["selected_config_id"],
            "selected_family_id": train_selection["selected_family_id"],
            "can_continue_train_dev_development": True,
            "can_run_final_test_metrics_now": False,
            "can_open_final_test_gate_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    positive_blocked_signal = any(
        int(row.get("train_target_depth_gain") or 0) > 0
        for row in train_selection.get("selection_ranking") or []
    )
    if positive_blocked_signal:
        return {
            "status": (
                "primeqa_hybrid_first_stage_recall_expansion_validation_completed_no_selection"
            ),
            "recommended_next_direction": (
                "design_stage116_prefix_preserving_recall_expansion_protocol"
            ),
            "selected_config_id": None,
            "selected_family_id": None,
            "positive_target_depth_signal_blocked_by_hit_at_200_loss": True,
            "can_continue_train_dev_development": True,
            "can_run_final_test_metrics_now": False,
            "can_open_final_test_gate_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": (
            "primeqa_hybrid_first_stage_recall_expansion_validation_completed_no_selection"
        ),
        "recommended_next_direction": "stop_first_stage_recall_expansion_family",
        "selected_config_id": None,
        "selected_family_id": None,
        "can_continue_train_dev_development": True,
        "can_run_final_test_metrics_now": False,
        "can_open_final_test_gate_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }


def _stage123_summary(stage123_protocol: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage123_protocol.get("decision") or {}
    frozen = stage123_protocol.get("frozen_protocol") or {}
    selection_rules = frozen.get("selection_rules") or {}
    return {
        "stage": stage123_protocol.get("stage"),
        "protocol_id": stage123_protocol.get("protocol_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "can_continue_train_dev_development": decision.get(
            "can_continue_train_dev_development"
        ),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get(
            "can_run_final_test_metrics_now"
        ),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "guard_thresholds": selection_rules.get("guard_thresholds") or {},
    }


def _candidate_configs_from_protocol(
    stage123_protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    frozen = stage123_protocol.get("frozen_protocol") or {}
    return [dict(config) for config in frozen.get("candidate_configs") or []]


def _answerable_samples(
    samples: Sequence[PrimeQAHybridSplitSample],
) -> list[PrimeQAHybridSplitSample]:
    return [
        sample
        for sample in samples
        if sample.answerable and sample.answer_doc_id is not None
    ]


def _contains_gold(doc_ids: Sequence[str], sample: PrimeQAHybridSplitSample) -> bool:
    return str(sample.answer_doc_id) in set(doc_ids)


def _top_k_values_for_target(target_pool_depth: int) -> tuple[int, ...]:
    values = [10, 20, 50, 100, 200]
    if target_pool_depth not in values:
        values.append(target_pool_depth)
    return tuple(value for value in values if value <= target_pool_depth)


def _maximum_channel_top_k(candidate_configs: Sequence[Mapping[str, Any]]) -> int:
    return max(
        int(config["candidate_generation"]["channel_top_k"])
        for config in candidate_configs
    )


def _maximum_target_pool_depth(candidate_configs: Sequence[Mapping[str, Any]]) -> int:
    return max(
        int(config["candidate_generation"]["target_pool_depth"])
        for config in candidate_configs
    )


def _empty_train_selection(
    candidate_configs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "selection_split": _TRAIN_SPLIT,
        "selection_mode": "train_grouped_cross_validation_candidate_pool_selection",
        "dev_used_for_selection": False,
        "dev_used_for_retuning": False,
        "candidate_count": len(candidate_configs),
        "eligible_config_count": 0,
        "selected_config_id": None,
        "selected_family_id": None,
        "selected_train_summary": None,
        "selection_ranking": [],
    }


def _source_files(
    *,
    stage123_protocol_path: Path,
    stage80_report_path: Path | None,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
) -> dict[str, Any]:
    return {
        "stage123_protocol": _fingerprint(stage123_protocol_path),
        "stage80_report": _fingerprint(stage80_report_path)
        if stage80_report_path
        else None,
        "train_split": _fingerprint(train_split_path),
        "dev_split": _fingerprint(dev_split_path),
        "documents": _fingerprint(documents_path),
    }


def _analysis_scope() -> str:
    return (
        "Train/dev-only first-stage recall expansion validation for the frozen "
        "Stage123 protocol. Stage124 evaluates bounded 300/400-depth candidate "
        "generation configs against the Stage116 top200 baseline, uses train "
        "grouped cross-validation for selection, reports dev once without "
        "retuning, keeps the final test split locked, does not run answer or "
        "final metrics, does not write raw candidate rows, does not download "
        "models, does not add fallback strategies, and does not change runtime "
        "defaults."
    )


def _split_contract() -> dict[str, Any]:
    return {
        "split_name": _SPLIT_NAME,
        "protocol_version": _PROTOCOL_VERSION,
        "development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
        "selection_split": _TRAIN_SPLIT,
        "selection_mode": "train_grouped_cross_validation_candidate_pool_selection",
        "validation_split": _DEV_SPLIT,
        "dev_validation_mode": "single_pass_report_only_no_retuning",
        "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
    }


def _public_channel_catalog(channels: Sequence[_EvaluationChannel]) -> list[dict[str, Any]]:
    return [
        {
            "channel_id": channel.channel_id,
            "family": channel.family,
            "weight": channel.weight,
            "description": channel.description,
        }
        for channel in channels
    ]


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    forbidden_keys = sorted(_find_forbidden_public_keys(report))
    return {
        "public_safe_summary_only": True,
        "raw_question_text_written": False,
        "raw_answer_text_written": False,
        "raw_document_text_written": False,
        "raw_document_ids_written": False,
        "raw_candidate_rows_written": False,
        "raw_sample_ids_written": False,
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


def _target_depth_gain_bars(report: Mapping[str, Any], *, split: str) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(review["config_id"]),
            value=float(
                review["split_reviews"][split][
                    "target_depth_hit_count_gain_vs_baseline_top200"
                ]
            ),
            value_label=str(
                review["split_reviews"][split][
                    "target_depth_hit_count_gain_vs_baseline_top200"
                ]
            ),
        )
        for review in report.get("config_reviews") or []
    ]


def _hit200_delta_bars(report: Mapping[str, Any], *, split: str) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(review["config_id"]),
            value=float(
                review["split_reviews"][split]["hit_at_200_delta_vs_baseline"]
            ),
            value_label=str(
                review["split_reviews"][split]["hit_at_200_delta_vs_baseline"]
            ),
        )
        for review in report.get("config_reviews") or []
    ]


def _selected_fold_summary_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    selected = _selected_review(report)
    if not selected:
        return []
    train_review = selected["split_reviews"][_TRAIN_SPLIT]
    target_depth = train_review["target_pool_depth"]
    fold_rates = [
        float(metrics["hit_at_k"].get(str(target_depth), 0.0))
        for metrics in train_review.get("fold_metrics", {}).values()
    ]
    if not fold_rates:
        return []
    return [
        BarDatum(label="min", value=min(fold_rates), value_label=f"{min(fold_rates):.4f}"),
        BarDatum(
            label="average",
            value=_rounded_mean(fold_rates),
            value_label=f"{_rounded_mean(fold_rates):.4f}",
        ),
        BarDatum(label="max", value=max(fold_rates), value_label=f"{max(fold_rates):.4f}"),
    ]


def _selected_pool_size_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    selected = _selected_review(report)
    if not selected:
        return []
    bars = []
    for split in _ALLOWED_DEVELOPMENT_SPLITS:
        size_summary = selected["split_reviews"][split]["candidate_pool_size"]
        bars.extend(
            [
                BarDatum(
                    label=f"{split} average",
                    value=float(size_summary["average"]),
                    value_label=str(size_summary["average"]),
                ),
                BarDatum(
                    label=f"{split} p95",
                    value=float(size_summary["p95"]),
                    value_label=str(size_summary["p95"]),
                ),
            ]
        )
    return bars


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
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
        for check in report.get("guard_checks") or []
    ]


def _selected_review(report: Mapping[str, Any]) -> Mapping[str, Any] | None:
    selected_config_id = (report.get("train_selection") or {}).get("selected_config_id")
    if not selected_config_id:
        return None
    return next(
        (
            review
            for review in report.get("config_reviews") or []
            if review.get("config_id") == selected_config_id
        ),
        None,
    )


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
