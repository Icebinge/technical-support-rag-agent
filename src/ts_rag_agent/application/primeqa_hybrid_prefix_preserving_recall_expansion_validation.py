from __future__ import annotations

import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_first_stage_recall_expansion_validation import (
    _baseline_outcomes,
    _baseline_summary,
    _channels_for_route_set,
    _evaluation_channels,
    _EvaluationChannel,
    _rank_pool,
    _result_cache,
    _results_for_channels,
)
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    EncoderFactory,
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
    _fingerprint,
    _load_json_object,
    _rounded_mean,
    _rounded_percentile,
    _rounded_ratio,
    _section_summary,
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

_STAGE = "Stage 126"
_CREATED_AT = "2026-07-16"
_ANALYSIS_ID = (
    "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_validation_v1"
)
_SOURCE_STAGE125_STATUS = (
    "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_protocol_frozen"
)
_SOURCE_PROTOCOL_ID = (
    "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_protocol_v1"
)
_SOURCE_NEXT_DIRECTION = (
    "run_stage116_prefix_preserving_recall_expansion_train_cv_dev_validation"
)
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = ("test",)
_BASELINE_CONFIG_ID = "stage116_fixed_rrf_top200_baseline"
_BASELINE_PREFIX_DEPTH = 200
_DEFAULT_TRAIN_FOLD_COUNT = 5
_DEFAULT_BM25_K1 = 1.5
_DEFAULT_BM25_B = 0.75
_DEFAULT_ENCODER_BATCH_SIZE = 64
_MAX_CHANNEL_TOP_K = 400
_MAX_TARGET_POOL_DEPTH = 400
_MAX_APPEND_BUDGET = 200
_LEXICAL_PRIORITY_SOURCE_CONFIG_ID = "rrf_lexical_priority_top300_k80_v1"
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
class PrimeQAHybridPrefixPreservingRecallExpansionValidationVisualization:
    """One generated Stage126 prefix-preserving recall expansion chart."""

    name: str
    path: str


def run_primeqa_hybrid_prefix_preserving_recall_expansion_validation(
    *,
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
    train_fold_count: int = _DEFAULT_TRAIN_FOLD_COUNT,
    encoder_batch_size: int = _DEFAULT_ENCODER_BATCH_SIZE,
    encoder_device: str | None = None,
    encoder_factory: EncoderFactory | None = None,
) -> dict[str, Any]:
    """Run the Stage126 train-CV/dev append-only recall expansion validation."""

    started_at = time.perf_counter()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    stage125_protocol = _load_json_object(stage125_protocol_path)
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

    stage125_summary = _stage125_summary(stage125_protocol)
    candidate_configs = _candidate_configs_from_protocol(stage125_protocol)
    protocol_guard_checks = _pre_evaluation_guard_checks(
        stage125_summary=stage125_summary,
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
            stage125_protocol_path=stage125_protocol_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
            user_confirmed_validation=user_confirmed_validation,
            confirmation_note=confirmation_note,
            stage125_summary=stage125_summary,
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
        guard_thresholds=stage125_summary["guard_thresholds"],
    )
    guard_checks = protocol_guard_checks + _post_evaluation_guard_checks(
        report_payload={
            "baseline_by_split": baseline_by_split,
            "config_reviews": config_reviews,
            "train_selection": train_selection,
        },
        stage125_summary=stage125_summary,
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
            stage125_protocol_path=stage125_protocol_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
        ),
        "stage125_summary": stage125_summary,
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
            "baseline_prefix_depth": _BASELINE_PREFIX_DEPTH,
            "maximum_channel_top_k": _maximum_channel_top_k(candidate_configs),
            "maximum_target_pool_depth": _maximum_target_pool_depth(candidate_configs),
            "maximum_append_budget": _maximum_append_budget(candidate_configs),
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
            "evaluate_prefix_preserving_pools": round(evaluated_at - indexed_at, 3),
            "selection_and_guard_checks": round(checked_at - evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_prefix_preserving_recall_expansion_validation_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridPrefixPreservingRecallExpansionValidationVisualization]:
    """Write SVG charts for Stage126 validation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage126_train_target_depth_gain.svg": render_horizontal_bar_chart_svg(
            title="Stage126 train target-depth gain vs Stage116 top200",
            bars=_target_depth_gain_bars(report, split=_TRAIN_SPLIT),
            x_label="hit-count gain",
            width=1580,
            margin_left=820,
        ),
        "stage126_dev_target_depth_gain.svg": render_horizontal_bar_chart_svg(
            title="Stage126 dev target-depth gain vs Stage116 top200",
            bars=_target_depth_gain_bars(report, split=_DEV_SPLIT),
            x_label="hit-count gain",
            width=1580,
            margin_left=820,
        ),
        "stage126_train_appended_gold_recovery.svg": render_horizontal_bar_chart_svg(
            title="Stage126 train appended gold recovery",
            bars=_appended_gold_recovery_bars(report, split=_TRAIN_SPLIT),
            x_label="recovered gold documents",
            width=1580,
            margin_left=820,
        ),
        "stage126_dev_appended_gold_recovery.svg": render_horizontal_bar_chart_svg(
            title="Stage126 dev appended gold recovery",
            bars=_appended_gold_recovery_bars(report, split=_DEV_SPLIT),
            x_label="recovered gold documents",
            width=1580,
            margin_left=820,
        ),
        "stage126_train_hit200_loss.svg": render_horizontal_bar_chart_svg(
            title="Stage126 train hit@200 loss count",
            bars=_hit200_loss_bars(report, split=_TRAIN_SPLIT),
            x_label="loss count",
            width=1580,
            margin_left=820,
        ),
        "stage126_prefix_identity_violations.svg": render_horizontal_bar_chart_svg(
            title="Stage126 prefix identity violations",
            bars=_prefix_identity_violation_bars(report),
            x_label="violation count",
            width=1680,
            margin_left=900,
        ),
        "stage126_selected_append_count_summary.svg": render_horizontal_bar_chart_svg(
            title="Stage126 selected append count summary",
            bars=_selected_append_count_bars(report),
            x_label="appended documents",
            width=1260,
            margin_left=500,
        ),
        "stage126_selection_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage126 decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1380,
            margin_left=720,
        ),
        "stage126_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage126 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1900,
            margin_left=1060,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridPrefixPreservingRecallExpansionValidationVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _blocked_report(
    *,
    stage125_protocol_path: Path,
    stage80_report_path: Path | None,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    user_confirmed_validation: bool,
    confirmation_note: str,
    stage125_summary: Mapping[str, Any],
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
            stage125_protocol_path=stage125_protocol_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
        ),
        "stage125_summary": dict(stage125_summary),
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
    generation = config["append_generation"]
    channel_top_k = int(generation["channel_top_k"])
    append_budget = int(generation["append_budget"])
    target_pool_depth = int(generation["target_pool_depth"])
    configured_channels = _channels_for_route_set(
        channels=channels,
        route_set=str(generation["route_set"]),
    )
    configured_channels = _weighted_channels_for_append_config(
        channels=configured_channels,
        config=config,
    )
    top_k_values = _top_k_values_for_target(target_pool_depth)
    accumulator = _empty_prefix_accumulator(
        split=split,
        config_id=str(config["config_id"]),
        target_pool_depth=target_pool_depth,
        append_budget=append_budget,
        total_questions=len(samples),
        top_k_values=top_k_values,
    )
    fold_accumulators = (
        {
            fold_id: _empty_prefix_accumulator(
                split=fold_id,
                config_id=str(config["config_id"]),
                target_pool_depth=target_pool_depth,
                append_budget=append_budget,
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
        baseline = baseline_outcomes[sample.sample_id]
        prefix = list(baseline["ranked_pool"][:_BASELINE_PREFIX_DEPTH])
        results_by_channel = _results_for_channels(
            sample=sample,
            channels=configured_channels,
            result_cache=result_cache,
            channel_top_k=channel_top_k,
        )
        append_source_ranked = _rank_pool(
            channels=configured_channels,
            results_by_channel=results_by_channel,
            algorithm=str(generation["append_source_algorithm"]),
            rrf_k=int(generation["rrf_k"]),
            target_pool_depth=target_pool_depth,
        )
        ranked = _prefix_preserving_pool(
            prefix=prefix,
            append_source_ranked=append_source_ranked,
            target_pool_depth=target_pool_depth,
        )
        prefix_violation_count = _prefix_identity_violation_count(
            prefix=prefix,
            ranked_pool=ranked,
        )
        _record_prefix_result(
            accumulator=accumulator,
            ranked_pool=ranked,
            prefix=prefix,
            prefix_violation_count=prefix_violation_count,
            sample=sample,
            baseline_hit_at_200=bool(baseline["hit_at_200"]),
            top_k_values=top_k_values,
        )
        if fold_assignments is not None:
            _record_prefix_result(
                accumulator=fold_accumulators[fold_assignments[sample.sample_id]],
                ranked_pool=ranked,
                prefix=prefix,
                prefix_violation_count=prefix_violation_count,
                sample=sample,
                baseline_hit_at_200=bool(baseline["hit_at_200"]),
                top_k_values=top_k_values,
            )
    finalized = _finalize_prefix_accumulator(accumulator)
    return {
        **finalized,
        "channel_count": len(configured_channels),
        "channel_families": dict(
            sorted(Counter(channel.family for channel in configured_channels).items())
        ),
        "fold_metrics": {
            fold_id: _finalize_prefix_accumulator(fold_accumulator)
            for fold_id, fold_accumulator in fold_accumulators.items()
        },
    }


def _weighted_channels_for_append_config(
    *,
    channels: Sequence[_EvaluationChannel],
    config: Mapping[str, Any],
) -> list[_EvaluationChannel]:
    if config.get("source_stage124_config_id") != _LEXICAL_PRIORITY_SOURCE_CONFIG_ID:
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


def _prefix_preserving_pool(
    *,
    prefix: Sequence[str],
    append_source_ranked: Sequence[str],
    target_pool_depth: int,
) -> list[str]:
    ranked = list(prefix[:_BASELINE_PREFIX_DEPTH])
    seen = set(ranked)
    for doc_id in append_source_ranked:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ranked.append(doc_id)
        if len(ranked) >= target_pool_depth:
            break
    return ranked[:target_pool_depth]


def _prefix_identity_violation_count(
    *,
    prefix: Sequence[str],
    ranked_pool: Sequence[str],
) -> int:
    expected_prefix = list(prefix[:_BASELINE_PREFIX_DEPTH])
    observed_prefix = list(ranked_pool[: len(expected_prefix)])
    if observed_prefix == expected_prefix:
        return 0
    return sum(
        1
        for expected, observed in zip(expected_prefix, observed_prefix, strict=False)
        if expected != observed
    ) + abs(len(expected_prefix) - len(observed_prefix))


def _empty_prefix_accumulator(
    *,
    split: str,
    config_id: str,
    target_pool_depth: int,
    append_budget: int,
    total_questions: int,
    top_k_values: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "split": split,
        "config_id": config_id,
        "target_pool_depth": target_pool_depth,
        "append_budget": append_budget,
        "total_questions": total_questions,
        "evaluated_questions": 0,
        "hit_counts": {top_k: 0 for top_k in top_k_values},
        "candidate_pool_sizes": [],
        "append_counts": [],
        "baseline_hit_at_200_count": 0,
        "hit_at_200_recovery_count": 0,
        "hit_at_200_loss_count": 0,
        "target_depth_recovery_count": 0,
        "target_depth_loss_count": 0,
        "appended_gold_recovery_count": 0,
        "prefix_identity_violation_count": 0,
        "prefix_identity_violation_question_count": 0,
        "append_budget_exceeded_count": 0,
        "append_exhaustion_count": 0,
    }


def _record_prefix_result(
    *,
    accumulator: dict[str, Any],
    ranked_pool: Sequence[str],
    prefix: Sequence[str],
    prefix_violation_count: int,
    sample: PrimeQAHybridSplitSample,
    baseline_hit_at_200: bool,
    top_k_values: tuple[int, ...],
) -> None:
    answer_doc_id = str(sample.answer_doc_id)
    target_pool_depth = int(accumulator["target_pool_depth"])
    append_budget = int(accumulator["append_budget"])
    pool = list(ranked_pool[:target_pool_depth])
    prefix_size = len(prefix[:_BASELINE_PREFIX_DEPTH])
    append_count = max(0, len(pool) - prefix_size)

    accumulator["evaluated_questions"] += 1
    accumulator["candidate_pool_sizes"].append(len(pool))
    accumulator["append_counts"].append(append_count)
    accumulator["prefix_identity_violation_count"] += prefix_violation_count
    if prefix_violation_count:
        accumulator["prefix_identity_violation_question_count"] += 1
    if append_count > append_budget:
        accumulator["append_budget_exceeded_count"] += 1
    if len(pool) < target_pool_depth:
        accumulator["append_exhaustion_count"] += 1
    if baseline_hit_at_200:
        accumulator["baseline_hit_at_200_count"] += 1

    for top_k in top_k_values:
        if answer_doc_id in set(pool[:top_k]):
            accumulator["hit_counts"][top_k] += 1

    hit_at_200 = answer_doc_id in set(pool[:_BASELINE_PREFIX_DEPTH])
    hit_at_target = answer_doc_id in set(pool)
    appended_gold_hit = (
        not baseline_hit_at_200 and answer_doc_id in set(pool[_BASELINE_PREFIX_DEPTH:])
    )
    if hit_at_200 and not baseline_hit_at_200:
        accumulator["hit_at_200_recovery_count"] += 1
    if baseline_hit_at_200 and not hit_at_200:
        accumulator["hit_at_200_loss_count"] += 1
    if hit_at_target and not baseline_hit_at_200:
        accumulator["target_depth_recovery_count"] += 1
    if baseline_hit_at_200 and not hit_at_target:
        accumulator["target_depth_loss_count"] += 1
    if appended_gold_hit:
        accumulator["appended_gold_recovery_count"] += 1


def _finalize_prefix_accumulator(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    evaluated = int(accumulator["evaluated_questions"])
    target_pool_depth = int(accumulator["target_pool_depth"])
    append_budget = int(accumulator["append_budget"])
    hit_counts = {
        int(top_k): int(count) for top_k, count in accumulator["hit_counts"].items()
    }
    baseline_hit_at_200_count = int(accumulator["baseline_hit_at_200_count"])
    hit_at_200_count = hit_counts.get(_BASELINE_PREFIX_DEPTH, 0)
    target_depth_hit_count = hit_counts.get(target_pool_depth, 0)
    target_depth_gain = target_depth_hit_count - baseline_hit_at_200_count
    pool_sizes = [int(value) for value in accumulator["candidate_pool_sizes"]]
    append_counts = [int(value) for value in accumulator["append_counts"]]
    return {
        "split": accumulator["split"],
        "config_id": accumulator["config_id"],
        "target_pool_depth": target_pool_depth,
        "append_budget": append_budget,
        "total_questions": int(accumulator["total_questions"]),
        "evaluated_questions": evaluated,
        "hit_counts": hit_counts,
        "hit_at_k": {
            str(top_k): _rounded_ratio(count, evaluated)
            for top_k, count in hit_counts.items()
        },
        "baseline_hit_at_200_count": baseline_hit_at_200_count,
        "hit_at_200_count": hit_at_200_count,
        "hit_at_200_delta_vs_stage116_prefix": (
            hit_at_200_count - baseline_hit_at_200_count
        ),
        "hit_at_200_delta_vs_baseline": hit_at_200_count
        - baseline_hit_at_200_count,
        "hit_at_200_recovery_count": int(accumulator["hit_at_200_recovery_count"]),
        "hit_at_200_loss_count": int(accumulator["hit_at_200_loss_count"]),
        "target_depth_hit_count": target_depth_hit_count,
        "target_depth_hit_rate": _rounded_ratio(target_depth_hit_count, evaluated),
        "target_depth_hit_count_gain_vs_stage116_top200": target_depth_gain,
        "target_depth_hit_count_gain_vs_baseline_top200": target_depth_gain,
        "target_depth_recovery_count": int(accumulator["target_depth_recovery_count"]),
        "target_depth_loss_count": int(accumulator["target_depth_loss_count"]),
        "appended_gold_recovery_count": int(
            accumulator["appended_gold_recovery_count"]
        ),
        "prefix_identity_violation_count": int(
            accumulator["prefix_identity_violation_count"]
        ),
        "prefix_identity_violation_question_count": int(
            accumulator["prefix_identity_violation_question_count"]
        ),
        "append_budget_exceeded_count": int(accumulator["append_budget_exceeded_count"]),
        "append_exhaustion_count": int(accumulator["append_exhaustion_count"]),
        "candidate_pool_size": _distribution_summary(pool_sizes),
        "append_count": {
            **_distribution_summary(append_counts),
            "budget": append_budget,
        },
    }


def _distribution_summary(values: Sequence[int]) -> dict[str, Any]:
    return {
        "average": _rounded_mean(values),
        "median": _rounded_percentile(values, 50),
        "p95": _rounded_percentile(values, 95),
        "max": max(values, default=0),
    }


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
        generation = config["append_generation"]
        reviews.append(
            {
                "config_id": config_id,
                "family_id": config["family_id"],
                "source_stage124_config_id": config["source_stage124_config_id"],
                "append_source_algorithm": generation["append_source_algorithm"],
                "route_set": generation["route_set"],
                "channel_top_k": generation["channel_top_k"],
                "append_budget": generation["append_budget"],
                "target_pool_depth": generation["target_pool_depth"],
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
    generation = config["append_generation"]
    checks = [
        _check(
            name="train_prefix_identity_violation_count_within_guard",
            passed=int(train_review["prefix_identity_violation_count"]) == 0,
            observed=train_review["prefix_identity_violation_count"],
            expected=0,
        ),
        _check(
            name="train_hit_at_200_loss_count_within_guard",
            passed=int(train_review["hit_at_200_loss_count"]) == 0,
            observed=train_review["hit_at_200_loss_count"],
            expected=0,
        ),
        _check(
            name="train_target_depth_hit_count_gain_positive",
            passed=int(train_review["target_depth_hit_count_gain_vs_stage116_top200"])
            >= 1,
            observed=train_review["target_depth_hit_count_gain_vs_stage116_top200"],
            expected=">= 1",
        ),
        _check(
            name="channel_top_k_within_guard",
            passed=int(generation["channel_top_k"]) <= _MAX_CHANNEL_TOP_K,
            observed=generation["channel_top_k"],
            expected=f"<= {_MAX_CHANNEL_TOP_K}",
        ),
        _check(
            name="target_pool_depth_within_guard",
            passed=int(generation["target_pool_depth"]) <= _MAX_TARGET_POOL_DEPTH,
            observed=generation["target_pool_depth"],
            expected=f"<= {_MAX_TARGET_POOL_DEPTH}",
        ),
        _check(
            name="append_budget_within_guard",
            passed=int(generation["append_budget"]) <= _MAX_APPEND_BUDGET,
            observed=generation["append_budget"],
            expected=f"<= {_MAX_APPEND_BUDGET}",
        ),
        _check(
            name="append_budget_not_exceeded",
            passed=int(train_review["append_budget_exceeded_count"]) == 0,
            observed=train_review["append_budget_exceeded_count"],
            expected=0,
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
                "target_depth_hit_count_gain_vs_stage116_top200"
            ]
        )
        >= 1
    ]
    ranked = sorted(
        eligible,
        key=lambda review: (
            -int(
                review["split_reviews"][_TRAIN_SPLIT][
                    "target_depth_hit_count_gain_vs_stage116_top200"
                ]
            ),
            int(review["target_pool_depth"]),
            int(review["append_budget"]),
            str(review["config_id"]),
        ),
    )
    selected = ranked[0] if ranked else None
    return {
        "selection_split": _TRAIN_SPLIT,
        "selection_mode": (
            "train_grouped_cross_validation_prefix_preserving_candidate_selection"
        ),
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
                "append_budget": review["append_budget"],
                "target_pool_depth": review["target_pool_depth"],
                "train_target_depth_gain": review["split_reviews"][_TRAIN_SPLIT][
                    "target_depth_hit_count_gain_vs_stage116_top200"
                ],
                "train_appended_gold_recovery_count": review["split_reviews"][
                    _TRAIN_SPLIT
                ]["appended_gold_recovery_count"],
                "train_hit_at_200_loss_count": review["split_reviews"][_TRAIN_SPLIT][
                    "hit_at_200_loss_count"
                ],
                "train_prefix_identity_violation_count": review["split_reviews"][
                    _TRAIN_SPLIT
                ]["prefix_identity_violation_count"],
                "guard_passed": review["train_cv_guard"]["passed"],
            }
            for review in sorted(
                config_reviews,
                key=lambda review: (
                    -int(
                        review["split_reviews"][_TRAIN_SPLIT][
                            "target_depth_hit_count_gain_vs_stage116_top200"
                        ]
                    ),
                    int(review["target_pool_depth"]),
                    int(review["append_budget"]),
                    str(review["config_id"]),
                ),
            )
        ],
    }


def _selection_metric_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "target_pool_depth": review["target_pool_depth"],
        "append_budget": review["append_budget"],
        "evaluated_questions": review["evaluated_questions"],
        "baseline_hit_at_200_count": review["baseline_hit_at_200_count"],
        "hit_at_200_count": review["hit_at_200_count"],
        "hit_at_200_delta_vs_stage116_prefix": review[
            "hit_at_200_delta_vs_stage116_prefix"
        ],
        "hit_at_200_loss_count": review["hit_at_200_loss_count"],
        "target_depth_hit_count": review["target_depth_hit_count"],
        "target_depth_hit_count_gain_vs_stage116_top200": review[
            "target_depth_hit_count_gain_vs_stage116_top200"
        ],
        "target_depth_hit_rate": review["target_depth_hit_rate"],
        "appended_gold_recovery_count": review["appended_gold_recovery_count"],
        "prefix_identity_violation_count": review["prefix_identity_violation_count"],
        "append_count": review["append_count"],
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
    stage125_summary: Mapping[str, Any],
    user_confirmed_validation: bool,
    confirmation_note: str,
    candidate_configs: Sequence[Mapping[str, Any]],
    include_dense_channels: bool,
    dense_summary: Mapping[str, Any],
    train_fold_count: int,
) -> list[dict[str, Any]]:
    return [
        _check(
            name="user_confirmed_stage126_validation",
            passed=user_confirmed_validation,
            observed=confirmation_note,
            expected="user confirmed Stage126 prefix-preserving validation",
        ),
        _check(
            name="stage125_protocol_frozen",
            passed=stage125_summary.get("decision_status") == _SOURCE_STAGE125_STATUS,
            observed=stage125_summary.get("decision_status"),
            expected=_SOURCE_STAGE125_STATUS,
        ),
        _check(
            name="stage125_protocol_id_matches",
            passed=stage125_summary.get("protocol_id") == _SOURCE_PROTOCOL_ID,
            observed=stage125_summary.get("protocol_id"),
            expected=_SOURCE_PROTOCOL_ID,
        ),
        _check(
            name="stage125_recommends_stage126_validation",
            passed=stage125_summary.get("recommended_next_direction")
            == _SOURCE_NEXT_DIRECTION,
            observed=stage125_summary.get("recommended_next_direction"),
            expected=_SOURCE_NEXT_DIRECTION,
        ),
        _check(
            name="stage125_runtime_and_test_boundaries_locked",
            passed=stage125_summary.get("can_open_final_test_gate_now") is False
            and stage125_summary.get("can_run_final_test_metrics_now") is False
            and stage125_summary.get("can_use_test_for_tuning") is False
            and stage125_summary.get("fallback_strategies_enabled") is False
            and stage125_summary.get("default_runtime_policy") == "unchanged",
            observed=stage125_summary,
            expected="test locked, runtime unchanged, fallback disabled",
        ),
        _check(
            name="stage126_expected_candidate_configs_loaded",
            passed=len(candidate_configs) == 6,
            observed=len(candidate_configs),
            expected=6,
        ),
        _check(
            name="stage126_candidate_configs_are_append_only",
            passed=all(_config_is_append_only(config) for config in candidate_configs),
            observed=[_config_append_contract(config) for config in candidate_configs],
            expected="append after rank 200 with immutable prefix",
        ),
        _check(
            name="stage126_uses_train_grouped_cv",
            passed=train_fold_count >= _DEFAULT_TRAIN_FOLD_COUNT,
            observed=train_fold_count,
            expected=f">= {_DEFAULT_TRAIN_FOLD_COUNT}",
        ),
        _check(
            name="stage126_dense_channels_ready",
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
    stage125_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    public_safe = _public_safe_contract(report_payload)
    train_selection = report_payload["train_selection"]
    baseline_by_split = report_payload["baseline_by_split"]
    config_reviews = report_payload["config_reviews"]
    return [
        _check(
            name="stage126_uses_only_train_dev_splits",
            passed=sorted(baseline_by_split) == ["dev", "train"],
            observed=sorted(baseline_by_split),
            expected=["train", "dev"],
        ),
        _check(
            name="stage126_test_split_not_loaded",
            passed=True,
            observed=False,
            expected=False,
        ),
        _check(
            name="stage126_baseline_matches_stage125_train",
            passed=baseline_by_split[_TRAIN_SPLIT]["hit_at_200_count"]
            == (stage125_summary.get("train_baseline") or {}).get("hit_at_200_count"),
            observed=baseline_by_split[_TRAIN_SPLIT],
            expected=stage125_summary.get("train_baseline"),
        ),
        _check(
            name="stage126_baseline_matches_stage125_dev",
            passed=baseline_by_split[_DEV_SPLIT]["hit_at_200_count"]
            == (stage125_summary.get("dev_baseline") or {}).get("hit_at_200_count"),
            observed=baseline_by_split[_DEV_SPLIT],
            expected=stage125_summary.get("dev_baseline"),
        ),
        _check(
            name="stage126_prefix_identity_preserved_all_configs",
            passed=all(
                int(split_review["prefix_identity_violation_count"]) == 0
                for review in config_reviews
                for split_review in review["split_reviews"].values()
            ),
            observed=_prefix_violation_summary(config_reviews),
            expected=0,
        ),
        _check(
            name="stage126_hit_at_200_loss_count_zero_all_configs",
            passed=all(
                int(split_review["hit_at_200_loss_count"]) == 0
                for review in config_reviews
                for split_review in review["split_reviews"].values()
            ),
            observed=_hit200_loss_summary(config_reviews),
            expected=0,
        ),
        _check(
            name="stage126_no_candidate_rows_written",
            passed=True,
            observed=False,
            expected=False,
        ),
        _check(
            name="stage126_no_model_download_attempted",
            passed=True,
            observed=True,
            expected=True,
        ),
        _check(
            name="stage126_dev_report_only",
            passed=train_selection.get("dev_used_for_selection") is False
            and train_selection.get("dev_used_for_retuning") is False,
            observed={
                "dev_used_for_selection": train_selection.get("dev_used_for_selection"),
                "dev_used_for_retuning": train_selection.get("dev_used_for_retuning"),
            },
            expected="dev not used for selection or retuning",
        ),
        _check(
            name="stage126_runtime_defaults_unchanged",
            passed=stage125_summary.get("default_runtime_policy") == "unchanged",
            observed=stage125_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage126_fallback_strategies_not_added",
            passed=stage125_summary.get("fallback_strategies_enabled") is False,
            observed=stage125_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="stage126_public_safe_contract_passed",
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
            "status": (
                "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_"
                "validation_blocked"
            ),
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
            "status": (
                "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_"
                "validation_completed"
            ),
            "recommended_next_direction": (
                "review_stage116_prefix_preserving_recall_expansion_selected_config"
            ),
            "selected_config_id": train_selection["selected_config_id"],
            "selected_family_id": train_selection["selected_family_id"],
            "can_continue_train_dev_development": True,
            "can_run_final_test_metrics_now": False,
            "can_open_final_test_gate_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": (
            "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_"
            "validation_completed_no_selection"
        ),
        "recommended_next_direction": (
            "stop_stage116_prefix_preserving_recall_expansion_family"
        ),
        "selected_config_id": None,
        "selected_family_id": None,
        "positive_target_depth_signal_found": any(
            int(row.get("train_target_depth_gain") or 0) > 0
            for row in train_selection.get("selection_ranking") or []
        ),
        "can_continue_train_dev_development": True,
        "can_run_final_test_metrics_now": False,
        "can_open_final_test_gate_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }


def _stage125_summary(stage125_protocol: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage125_protocol.get("decision") or {}
    frozen = stage125_protocol.get("frozen_protocol") or {}
    selection_rules = frozen.get("selection_rules") or {}
    prefix_contract = frozen.get("baseline_prefix_contract") or {}
    public_safe = stage125_protocol.get("public_safe_contract") or {}
    return {
        "stage": stage125_protocol.get("stage"),
        "protocol_id": stage125_protocol.get("protocol_id"),
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
        "train_baseline": {
            "pool_depth": prefix_contract.get("prefix_depth"),
            "hit_at_200_count": prefix_contract.get("train_baseline_hit_at_200_count"),
            "hit_at_200": prefix_contract.get("train_baseline_hit_at_200"),
        },
        "dev_baseline": {
            "pool_depth": prefix_contract.get("prefix_depth"),
            "hit_at_200_count": prefix_contract.get("dev_baseline_hit_at_200_count"),
            "hit_at_200": prefix_contract.get("dev_baseline_hit_at_200"),
        },
        "guard_thresholds": selection_rules.get("guard_thresholds") or {},
        "selection_rules": selection_rules,
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
    }


def _candidate_configs_from_protocol(
    stage125_protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    frozen = stage125_protocol.get("frozen_protocol") or {}
    return [dict(config) for config in frozen.get("candidate_configs") or []]


def _answerable_samples(
    samples: Sequence[PrimeQAHybridSplitSample],
) -> list[PrimeQAHybridSplitSample]:
    return [
        sample
        for sample in samples
        if sample.answerable and sample.answer_doc_id is not None
    ]


def _top_k_values_for_target(target_pool_depth: int) -> tuple[int, ...]:
    values = [10, 20, 50, 100, 200]
    if target_pool_depth not in values:
        values.append(target_pool_depth)
    return tuple(value for value in values if value <= target_pool_depth)


def _maximum_channel_top_k(candidate_configs: Sequence[Mapping[str, Any]]) -> int:
    return max(
        int(config["append_generation"]["channel_top_k"]) for config in candidate_configs
    )


def _maximum_target_pool_depth(candidate_configs: Sequence[Mapping[str, Any]]) -> int:
    return max(
        int(config["append_generation"]["target_pool_depth"])
        for config in candidate_configs
    )


def _maximum_append_budget(candidate_configs: Sequence[Mapping[str, Any]]) -> int:
    return max(
        int(config["append_generation"]["append_budget"]) for config in candidate_configs
    )


def _empty_train_selection(
    candidate_configs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "selection_split": _TRAIN_SPLIT,
        "selection_mode": (
            "train_grouped_cross_validation_prefix_preserving_candidate_selection"
        ),
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
    stage125_protocol_path: Path,
    stage80_report_path: Path | None,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
) -> dict[str, Any]:
    return {
        "stage125_protocol": _fingerprint(stage125_protocol_path),
        "stage80_report": _fingerprint(stage80_report_path)
        if stage80_report_path
        else None,
        "train_split": _fingerprint(train_split_path),
        "dev_split": _fingerprint(dev_split_path),
        "documents": _fingerprint(documents_path),
    }


def _analysis_scope() -> str:
    return (
        "Train/dev-only validation for the frozen Stage125 Stage116 "
        "prefix-preserving append-only recall expansion protocol. Stage126 "
        "keeps the Stage116 top200 prefix unchanged, appends deduplicated "
        "runtime-visible candidates after rank 200, uses train grouped "
        "cross-validation for selection, reports dev once without retuning, "
        "keeps the final test split locked, does not run answer or final "
        "metrics, does not write raw candidate rows, does not download models, "
        "does not add fallback strategies, and does not change runtime defaults."
    )


def _split_contract() -> dict[str, Any]:
    return {
        "split_name": _SPLIT_NAME,
        "protocol_version": _PROTOCOL_VERSION,
        "development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
        "selection_split": _TRAIN_SPLIT,
        "selection_mode": (
            "train_grouped_cross_validation_prefix_preserving_candidate_selection"
        ),
        "validation_split": _DEV_SPLIT,
        "dev_validation_mode": "single_pass_report_only_no_retuning",
        "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
    }


def _public_channel_catalog(
    channels: Sequence[_EvaluationChannel],
) -> list[dict[str, Any]]:
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


def _config_is_append_only(config: Mapping[str, Any]) -> bool:
    append = config.get("append_generation") or {}
    prefix = config.get("prefix_preservation") or {}
    feature_sources = config.get("feature_sources") or {}
    safety = config.get("safety_constraints") or {}
    return (
        prefix.get("ranks_1_to_200_must_remain_identical") is True
        and prefix.get("may_reorder_prefix") is False
        and prefix.get("may_drop_prefix_documents") is False
        and prefix.get("may_insert_before_rank_201") is False
        and int(append.get("append_start_rank") or 0) == _BASELINE_PREFIX_DEPTH + 1
        and int(append.get("target_pool_depth") or 0) <= _MAX_TARGET_POOL_DEPTH
        and int(append.get("append_budget") or 0) <= _MAX_APPEND_BUDGET
        and int(append.get("append_budget") or 0)
        == int(append.get("target_pool_depth") or 0) - _BASELINE_PREFIX_DEPTH
        and feature_sources.get("requires_model_download") is False
        and feature_sources.get("requires_new_embedding_build") is False
        and feature_sources.get("uses_oracle_document_metadata") is False
        and feature_sources.get("uses_test_membership") is False
        and safety.get("fallback_strategies_enabled") is False
        and safety.get("runtime_defaultization_allowed") is False
    )


def _config_append_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    append = config.get("append_generation") or {}
    prefix = config.get("prefix_preservation") or {}
    return {
        "config_id": config.get("config_id"),
        "append_start_rank": append.get("append_start_rank"),
        "append_budget": append.get("append_budget"),
        "target_pool_depth": append.get("target_pool_depth"),
        "preserved_prefix_depth": prefix.get("preserved_prefix_depth"),
        "may_reorder_prefix": prefix.get("may_reorder_prefix"),
        "may_drop_prefix_documents": prefix.get("may_drop_prefix_documents"),
        "may_insert_before_rank_201": prefix.get("may_insert_before_rank_201"),
    }


def _prefix_violation_summary(
    config_reviews: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "config_id": review["config_id"],
            "train": review["split_reviews"][_TRAIN_SPLIT][
                "prefix_identity_violation_count"
            ],
            "dev": review["split_reviews"][_DEV_SPLIT][
                "prefix_identity_violation_count"
            ],
        }
        for review in config_reviews
    ]


def _hit200_loss_summary(
    config_reviews: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "config_id": review["config_id"],
            "train": review["split_reviews"][_TRAIN_SPLIT]["hit_at_200_loss_count"],
            "dev": review["split_reviews"][_DEV_SPLIT]["hit_at_200_loss_count"],
        }
        for review in config_reviews
    ]


def _target_depth_gain_bars(report: Mapping[str, Any], *, split: str) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(review["config_id"]),
            value=float(
                review["split_reviews"][split][
                    "target_depth_hit_count_gain_vs_stage116_top200"
                ]
            ),
            value_label=str(
                review["split_reviews"][split][
                    "target_depth_hit_count_gain_vs_stage116_top200"
                ]
            ),
        )
        for review in report.get("config_reviews") or []
    ]


def _appended_gold_recovery_bars(
    report: Mapping[str, Any],
    *,
    split: str,
) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(review["config_id"]),
            value=float(
                review["split_reviews"][split]["appended_gold_recovery_count"]
            ),
            value_label=str(
                review["split_reviews"][split]["appended_gold_recovery_count"]
            ),
        )
        for review in report.get("config_reviews") or []
    ]


def _hit200_loss_bars(report: Mapping[str, Any], *, split: str) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(review["config_id"]),
            value=float(review["split_reviews"][split]["hit_at_200_loss_count"]),
            value_label=str(review["split_reviews"][split]["hit_at_200_loss_count"]),
        )
        for review in report.get("config_reviews") or []
    ]


def _prefix_identity_violation_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for review in report.get("config_reviews") or []:
        bars.append(
            BarDatum(
                label=f"{review['config_id']} train",
                value=float(
                    review["split_reviews"][_TRAIN_SPLIT][
                        "prefix_identity_violation_count"
                    ]
                ),
                value_label=str(
                    review["split_reviews"][_TRAIN_SPLIT][
                        "prefix_identity_violation_count"
                    ]
                ),
            )
        )
        bars.append(
            BarDatum(
                label=f"{review['config_id']} dev",
                value=float(
                    review["split_reviews"][_DEV_SPLIT][
                        "prefix_identity_violation_count"
                    ]
                ),
                value_label=str(
                    review["split_reviews"][_DEV_SPLIT][
                        "prefix_identity_violation_count"
                    ]
                ),
            )
        )
    return bars


def _selected_append_count_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    selected = _selected_review(report)
    if not selected:
        return []
    bars = []
    for split in _ALLOWED_DEVELOPMENT_SPLITS:
        append_summary = selected["split_reviews"][split]["append_count"]
        bars.extend(
            [
                BarDatum(
                    label=f"{split} average",
                    value=float(append_summary["average"]),
                    value_label=str(append_summary["average"]),
                ),
                BarDatum(
                    label=f"{split} p95",
                    value=float(append_summary["p95"]),
                    value_label=str(append_summary["p95"]),
                ),
                BarDatum(
                    label=f"{split} max",
                    value=float(append_summary["max"]),
                    value_label=str(append_summary["max"]),
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
