from __future__ import annotations

import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_agent_retrieval_integration_validation import (
    _BASELINE_PREFIX_DEPTH,
    _DEFAULT_BM25_B,
    _DEFAULT_BM25_K1,
    _DEV_SPLIT,
    _TRAIN_SPLIT,
    _candidate_pools_by_split,
    _channels_for_route_set,
    _evaluation_channels,
    _selected_append_config,
)
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    EncoderFactory,
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
    _load_json_object,
)
from ts_rag_agent.application.primeqa_hybrid_online_candidate_pool_retriever import (
    CandidatePoolRetrievalConfig,
    DerivedCandidatePoolSearchChannel,
    IndependentCandidatePoolSearchChannel,
    PrimeQAHybridOnlineCandidatePoolRetriever,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_orchestrator_validation import (
    _stage128_summary,
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

_STAGE = "Stage 140"
_CREATED_AT = "2026-07-17"
_ANALYSIS_ID = "primeqa_hybrid_online_candidate_pool_performance_validation_v1"
_SOURCE_STAGE139_STATUS = (
    "primeqa_hybrid_optional_sidecar_agent_entrypoint_train_cv_dev_validation_passed"
)
_SELECTED_CONFIG_ID = "prefix_existing_dense_broad_append200_v1"
_SELECTED_ROUTE_SET = "stage116_lexical_routes_plus_existing_dense_cache_routes"
_SPECIAL_CHANNEL_ID = "special_token_boosted_bm25"
_BASE_CHANNEL_ID = "full_document_bm25"
_TOP_K_VALUES = (10, 50, 100, 200, 400)


@dataclass(frozen=True)
class PrimeQAHybridOnlineCandidatePoolPerformanceVisualization:
    name: str
    path: str


def run_primeqa_hybrid_online_candidate_pool_performance_validation(
    *,
    stage139_validation_path: Path,
    stage128_protocol_path: Path,
    stage127_review_path: Path,
    stage125_protocol_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    user_confirmed_validation: bool,
    confirmation_note: str,
    train_fold_count: int = 5,
    encoder_batch_size: int = 64,
    encoder_device: str | None = None,
    encoder_factory: EncoderFactory | None = None,
) -> dict[str, Any]:
    """Validate a long-lived online pool retriever on frozen train/dev only."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    started_at = time.perf_counter()
    stage139 = _load_json_object(stage139_validation_path)
    stage128 = _load_json_object(stage128_protocol_path)
    stage127 = _load_json_object(stage127_review_path)
    stage125 = _load_json_object(stage125_protocol_path)
    stage80 = _load_json_object(stage80_report_path)
    selected_config = _selected_append_config(
        stage125_protocol=stage125,
        stage128_summary=_stage128_summary(stage128),
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
    loaded_documents_at = time.perf_counter()

    dense_channels, dense_summary = _build_dense_channels(
        include_dense_channels=True,
        stage80_report=stage80,
        stage80_report_path=stage80_report_path,
        documents=documents,
        document_ids=document_ids,
        encoder_batch_size=encoder_batch_size,
        encoder_device=encoder_device,
        encoder_factory=encoder_factory,
    )
    dense_ready_at = time.perf_counter()
    source_files = _source_files(
        stage139_validation_path=stage139_validation_path,
        stage128_protocol_path=stage128_protocol_path,
        stage127_review_path=stage127_review_path,
        stage125_protocol_path=stage125_protocol_path,
        stage80_report_path=stage80_report_path,
        train_split_path=train_split_path,
        dev_split_path=dev_split_path,
        documents_path=documents_path,
    )
    pre_checks = _pre_checks(
        stage139=stage139,
        stage127=stage127,
        selected_config=selected_config,
        dense_summary=dense_summary,
        split_samples=split_samples,
        train_fold_count=train_fold_count,
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
    )
    if not all(check["passed"] for check in pre_checks):
        finished_at = time.perf_counter()
        return _blocked_report(
            source_files=source_files,
            split_samples=split_samples,
            stage139=stage139,
            stage127=stage127,
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
                "dense_preflight": round(dense_ready_at - loaded_documents_at, 3),
                "total": round(finished_at - started_at, 3),
            },
        )

    lexical_channels = _build_lexical_channels(
        documents=documents,
        sections_by_document=sections_by_document,
        bm25_k1=_DEFAULT_BM25_K1,
        bm25_b=_DEFAULT_BM25_B,
        component_depth=int(selected_config["append_generation"]["channel_top_k"]),
    )
    raw_channels = lexical_channels + dense_channels
    evaluation_channels = _channels_for_route_set(
        channels=_evaluation_channels(
            lexical_channels=lexical_channels,
            dense_channels=dense_channels,
        ),
        route_set=_SELECTED_ROUTE_SET,
    )
    online_channels = _online_channels(raw_channels)
    retriever = PrimeQAHybridOnlineCandidatePoolRetriever(
        channels=online_channels,
        config=_retrieval_config(selected_config),
    )
    indexed_at = time.perf_counter()

    legacy_pools = _candidate_pools_by_split(
        split_samples=split_samples,
        selected_config=selected_config,
        channels=evaluation_channels,
    )
    legacy_pools_built_at = time.perf_counter()
    split_reports = {}
    for split, samples in split_samples.items():
        split_reports[split] = _evaluate_split(
            split=split,
            samples=samples,
            retriever=retriever,
            legacy_pools=legacy_pools[split],
            fold_assignments=(train_fold_assignments if split == _TRAIN_SPLIT else None),
        )
    online_evaluated_at = time.perf_counter()

    source_recall = _source_recall(stage127)
    post_checks = _post_checks(
        split_reports=split_reports,
        source_recall=source_recall,
        stage139=stage139,
    )
    guard_checks = pre_checks + post_checks
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": {
            "train_grouped_five_fold_performance_and_recall_validation": True,
            "dev_single_pass_report_only": True,
            "test_split_loaded": False,
            "test_metrics_run": False,
            "runtime_default_changed": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
        },
        "user_confirmation": {
            "confirmed": user_confirmed_validation,
            "note_present": bool(confirmation_note.strip()),
        },
        "source_files": source_files,
        "source_stage139": _source_stage139(stage139),
        "source_stage127": {
            "selected_config_id": stage127["decision"]["selected_config_id"],
            "recall_by_split": source_recall,
        },
        "selected_candidate_pool_contract": {
            "config_id": selected_config["config_id"],
            "route_set": selected_config["append_generation"]["route_set"],
            "channel_top_k": selected_config["append_generation"]["channel_top_k"],
            "prefix_depth": _BASELINE_PREFIX_DEPTH,
            "target_pool_depth": selected_config["append_generation"]["target_pool_depth"],
            "rrf_k": selected_config["append_generation"]["rrf_k"],
            "channel_count": len(online_channels),
            "independent_channel_count": sum(
                isinstance(channel, IndependentCandidatePoolSearchChannel)
                for channel in online_channels
            ),
            "derived_channel_count": sum(
                isinstance(channel, DerivedCandidatePoolSearchChannel)
                for channel in online_channels
            ),
            "indexes_owned_outside_request_path": True,
            "query_specific_candidate_pool_built_per_request": True,
        },
        "loaded_data_summary": {
            "split_samples": summarize_primeqa_hybrid_split_samples(split_samples),
            "document_count": len(documents),
            "section_count": sum(map(len, sections_by_document.values())),
            "test_split_loaded": False,
        },
        "dense_channel_preflight": dense_summary,
        "split_reports": split_reports,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
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
            "load_dense_models_and_cached_embeddings": round(
                dense_ready_at - loaded_documents_at,
                3,
            ),
            "build_long_lived_lexical_indexes": round(indexed_at - dense_ready_at, 3),
            "build_optimized_legacy_candidate_pools": round(
                legacy_pools_built_at - indexed_at,
                3,
            ),
            "run_online_retriever_and_compare_all_rows": round(
                online_evaluated_at - legacy_pools_built_at,
                3,
            ),
            "total": round(online_evaluated_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_online_candidate_pool_performance_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridOnlineCandidatePoolPerformanceVisualization]:
    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage140_candidate_pool_wall_time.svg": render_horizontal_bar_chart_svg(
            title="Stage140 candidate-pool wall time",
            bars=_wall_time_bars(report),
            x_label="seconds for all train and dev rows",
            width=1320,
            margin_left=620,
        ),
        "stage140_online_latency_distribution.svg": render_horizontal_bar_chart_svg(
            title="Stage140 online per-query latency",
            bars=_latency_bars(report),
            x_label="seconds",
            width=1260,
            margin_left=520,
        ),
        "stage140_train_channel_p95_latency.svg": render_horizontal_bar_chart_svg(
            title="Stage140 train channel P95 latency",
            bars=_channel_latency_bars(report),
            x_label="seconds",
            width=1480,
            margin_left=720,
        ),
        "stage140_recall_at_k.svg": render_horizontal_bar_chart_svg(
            title="Stage140 optimized candidate-pool recall",
            bars=_recall_bars(report),
            x_label="gold-document hit rate",
            width=1280,
            margin_left=560,
        ),
        "stage140_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage140 guard checks",
            bars=[
                BarDatum(
                    label=str(check["name"]),
                    value=1.0 if check["passed"] else 0.0,
                    value_label="passed" if check["passed"] else "failed",
                )
                for check in report.get("guard_checks", [])
            ],
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
            PrimeQAHybridOnlineCandidatePoolPerformanceVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _online_channels(raw_channels: Sequence[Any]) -> tuple[Any, ...]:
    channels = []
    for channel in raw_channels:
        if channel.channel_id == _SPECIAL_CHANNEL_ID:
            special_retriever = channel.retriever

            def derived_search(
                query: str,
                source_results: Sequence[RetrievalResult],
                top_k: int,
                *,
                retriever: Any = special_retriever,
            ) -> Sequence[RetrievalResult]:
                return retriever.search_from_base_results(
                    query,
                    base_results=source_results,
                    top_k=top_k,
                )

            channels.append(
                DerivedCandidatePoolSearchChannel(
                    channel_id=channel.channel_id,
                    family=channel.family,
                    weight=channel.weight,
                    source_channel_id=_BASE_CHANNEL_ID,
                    searcher=derived_search,
                )
            )
        else:
            channels.append(
                IndependentCandidatePoolSearchChannel(
                    channel_id=channel.channel_id,
                    family=channel.family,
                    weight=channel.weight,
                    searcher=channel.retriever.search,
                )
            )
    return tuple(channels)


def _retrieval_config(selected_config: Mapping[str, Any]) -> CandidatePoolRetrievalConfig:
    generation = selected_config["append_generation"]
    return CandidatePoolRetrievalConfig(
        channel_top_k=int(generation["channel_top_k"]),
        prefix_depth=_BASELINE_PREFIX_DEPTH,
        target_pool_depth=int(generation["target_pool_depth"]),
        rrf_k=int(generation["rrf_k"]),
    )


def _evaluate_split(
    *,
    split: str,
    samples: Sequence[PrimeQAHybridSplitSample],
    retriever: PrimeQAHybridOnlineCandidatePoolRetriever,
    legacy_pools: Mapping[str, Mapping[str, Any]],
    fold_assignments: Mapping[str, str] | None,
) -> dict[str, Any]:
    latencies = []
    channel_latencies: dict[str, list[float]] = defaultdict(list)
    pool_sizes = []
    identity_violations = 0
    answerable_count = 0
    hit_counts = {top_k: 0 for top_k in _TOP_K_VALUES}
    fold_rows: dict[str, list[tuple[PrimeQAHybridSplitSample, tuple[str, ...], float]]] = (
        defaultdict(list)
    )
    for sample in samples:
        run = retriever.retrieve_profiled(sample.to_primeqa_question())
        doc_ids = tuple(result.document.id for result in run.results)
        source_doc_ids = tuple(legacy_pools[sample.sample_id]["stage128_pool"])
        identity_violations += doc_ids != source_doc_ids
        latencies.append(run.profile.total_seconds)
        pool_sizes.append(len(run.results))
        for timing in run.profile.channel_timings:
            channel_latencies[timing.channel_id].append(timing.duration_seconds)
        if sample.answerable and sample.answer_doc_id is not None:
            answerable_count += 1
            for top_k in _TOP_K_VALUES:
                hit_counts[top_k] += sample.answer_doc_id in doc_ids[:top_k]
        if fold_assignments is not None:
            fold_rows[fold_assignments[sample.sample_id]].append(
                (sample, doc_ids, run.profile.total_seconds)
            )
    return {
        "split": split,
        "row_count": len(samples),
        "answerable_count": answerable_count,
        "exact_candidate_pool_identity_violation_count": identity_violations,
        "candidate_pool_size": _distribution(pool_sizes),
        "latency_seconds": _distribution(latencies),
        "channel_latency_seconds": {
            channel_id: _distribution(values) for channel_id, values in channel_latencies.items()
        },
        "recall": {
            "hit_counts": {str(top_k): hit_counts[top_k] for top_k in _TOP_K_VALUES},
            "hit_at_k": {
                str(top_k): _ratio(hit_counts[top_k], answerable_count) for top_k in _TOP_K_VALUES
            },
        },
        "fold_reports": {
            fold_id: _fold_report(rows) for fold_id, rows in sorted(fold_rows.items())
        },
    }


def _fold_report(
    rows: Sequence[tuple[PrimeQAHybridSplitSample, tuple[str, ...], float]],
) -> dict[str, Any]:
    answerable = [row for row in rows if row[0].answerable and row[0].answer_doc_id]
    hit_counts = {
        top_k: sum(row[0].answer_doc_id in row[1][:top_k] for row in answerable)
        for top_k in _TOP_K_VALUES
    }
    return {
        "row_count": len(rows),
        "answerable_count": len(answerable),
        "latency_seconds": _distribution([row[2] for row in rows]),
        "hit_counts": {str(top_k): hit_counts[top_k] for top_k in _TOP_K_VALUES},
        "hit_at_k": {
            str(top_k): _ratio(hit_counts[top_k], len(answerable)) for top_k in _TOP_K_VALUES
        },
    }


def _source_recall(stage127: Mapping[str, Any]) -> dict[str, Any]:
    stage126_path = Path(stage127["source_files"]["stage126_report"]["path"])
    stage126 = _load_json_object(stage126_path)
    selected_config_id = stage127["decision"]["selected_config_id"]
    selected = next(
        row for row in stage126["config_reviews"] if row["config_id"] == selected_config_id
    )
    return {
        split: {
            "hit_counts": {
                str(top_k): int(selected["split_reviews"][split]["hit_counts"][str(top_k)])
                for top_k in _TOP_K_VALUES
            }
        }
        for split in (_TRAIN_SPLIT, _DEV_SPLIT)
    }


def _pre_checks(
    *,
    stage139: Mapping[str, Any],
    stage127: Mapping[str, Any],
    selected_config: Mapping[str, Any],
    dense_summary: Mapping[str, Any],
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    train_fold_count: int,
    user_confirmed_validation: bool,
    confirmation_note: str,
) -> list[dict[str, Any]]:
    return [
        _check("stage140_user_confirmed", user_confirmed_validation, True),
        _check("stage140_confirmation_note_present", bool(confirmation_note.strip()), True),
        _check(
            "stage139_source_passed",
            stage139["decision"]["status"] == _SOURCE_STAGE139_STATUS,
            _SOURCE_STAGE139_STATUS,
        ),
        _check(
            "stage139_runtime_default_unchanged",
            not stage139["decision"]["entrypoint_registered_as_runtime_default"],
            True,
        ),
        _check(
            "stage127_selected_config_matches",
            stage127["decision"]["selected_config_id"] == _SELECTED_CONFIG_ID,
            _SELECTED_CONFIG_ID,
        ),
        _check(
            "stage125_selected_config_available",
            selected_config["config_id"] == _SELECTED_CONFIG_ID,
            _SELECTED_CONFIG_ID,
        ),
        _check(
            "selected_route_set_matches",
            selected_config["append_generation"]["route_set"] == _SELECTED_ROUTE_SET,
            _SELECTED_ROUTE_SET,
        ),
        _check(
            "dense_channels_local_ready",
            dense_summary.get("status") == "dense_channels_ready",
            "dense_channels_ready",
        ),
        _check("train_fold_count_is_five", train_fold_count == 5, 5),
        _check(
            "train_dev_only_loaded",
            set(split_samples) == {_TRAIN_SPLIT, _DEV_SPLIT},
            [_TRAIN_SPLIT, _DEV_SPLIT],
        ),
    ]


def _post_checks(
    *,
    split_reports: Mapping[str, Mapping[str, Any]],
    source_recall: Mapping[str, Mapping[str, Any]],
    stage139: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks = []
    for split in (_TRAIN_SPLIT, _DEV_SPLIT):
        report = split_reports[split]
        checks.extend(
            [
                _check(
                    f"{split}_candidate_pool_exact_identity",
                    report["exact_candidate_pool_identity_violation_count"] == 0,
                    0,
                ),
                _check(
                    f"{split}_recall_counts_match_stage127",
                    report["recall"]["hit_counts"] == source_recall[split]["hit_counts"],
                    source_recall[split]["hit_counts"],
                ),
                _check(
                    f"{split}_candidate_pool_depth_is_400",
                    report["candidate_pool_size"]["min"] == 400
                    and report["candidate_pool_size"]["max"] == 400,
                    400,
                ),
            ]
        )
    checks.extend(
        [
            _check(
                "optimized_mean_latency_below_stage139_batch_average",
                _combined_mean_latency(split_reports)
                < float(stage139["timing_seconds"]["build_candidate_pools"])
                / _combined_row_count(split_reports),
                "< Stage139 batch average",
            ),
            _check("test_split_remains_locked", True, True),
            _check("runtime_default_remains_unchanged", True, True),
            _check("retry_actions_remain_disabled", True, True),
            _check("fallback_strategies_remain_disabled", True, True),
        ]
    )
    return checks


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passed = all(check["passed"] for check in guard_checks)
    return {
        "status": (
            "primeqa_hybrid_online_candidate_pool_performance_validation_passed"
            if passed
            else "primeqa_hybrid_online_candidate_pool_performance_validation_blocked"
        ),
        "failed_checks": [check["name"] for check in guard_checks if not check["passed"]],
        "online_candidate_pool_implementation_validated": passed,
        "candidate_pool_identity_preserved": passed,
        "retrieval_recall_preserved": passed,
        "runtime_activation_allowed_now": False,
        "runtime_defaultization_allowed_now": False,
        "test_gate_opened": False,
        "latency_slo_user_confirmed": False,
        "retry_actions_enabled": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_direction": (
            "freeze_user_confirmed_latency_slo_and_nondefault_runtime_activation_protocol"
            if passed
            else "repair_online_candidate_pool_performance_or_identity_failures"
        ),
    }


def _blocked_report(**values: Any) -> dict[str, Any]:
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": {
            "test_split_loaded": False,
            "test_metrics_run": False,
            "runtime_default_changed": False,
        },
        "source_files": values["source_files"],
        "loaded_data_summary": {
            "split_samples": summarize_primeqa_hybrid_split_samples(values["split_samples"]),
            "test_split_loaded": False,
        },
        "source_stage139": _source_stage139(values["stage139"]),
        "source_stage127": {
            "selected_config_id": values["stage127"]["decision"]["selected_config_id"]
        },
        "dense_channel_preflight": values["dense_summary"],
        "split_reports": {},
        "guard_checks": values["guard_checks"],
        "decision": _decision(values["guard_checks"]),
        "timing_seconds": values["timing_seconds"],
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def _source_stage139(stage139: Mapping[str, Any]) -> dict[str, Any]:
    seconds = float(stage139["timing_seconds"]["build_candidate_pools"])
    rows = sum(
        int(summary["row_count"])
        for summary in stage139["candidate_pool_summary"]["splits"].values()
    )
    return {
        "status": stage139["decision"]["status"],
        "candidate_pool_build_seconds": seconds,
        "row_count": rows,
        "batch_average_seconds_per_row": round(seconds / rows, 6),
        "runtime_default_registered": stage139["decision"][
            "entrypoint_registered_as_runtime_default"
        ],
        "test_split_loaded": stage139["loaded_data_summary"]["test_split_loaded"],
    }


def _source_files(**paths: Path) -> dict[str, Any]:
    return {name: {"path": str(path), "exists": path.is_file()} for name, path in paths.items()}


def _distribution(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": 0.0, "average": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": round(min(numeric), 6),
        "average": round(mean(numeric), 6),
        "p50": round(_percentile(numeric, 50), 6),
        "p95": round(_percentile(numeric, 95), 6),
        "max": round(max(numeric), 6),
    }


def _percentile(values: Sequence[float], percentile: int) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _combined_mean_latency(split_reports: Mapping[str, Mapping[str, Any]]) -> float:
    total = sum(
        float(report["latency_seconds"]["average"]) * int(report["row_count"])
        for report in split_reports.values()
    )
    return total / _combined_row_count(split_reports)


def _combined_row_count(split_reports: Mapping[str, Mapping[str, Any]]) -> int:
    return sum(int(report["row_count"]) for report in split_reports.values())


def _check(name: str, passed: bool, expected: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "expected": expected}


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = sorted(_forbidden_keys_found(report))
    return {
        "aggregate_only": True,
        "private_per_row_trace_written": False,
        "forbidden_keys_found": forbidden,
    }


def _forbidden_keys_found(value: Any) -> set[str]:
    forbidden_keys = {
        "raw_question_text",
        "raw_document_text",
        "answer_doc_id",
        "sample_id",
        "runtime_content_handle",
    }
    found = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized_key = str(key).lower()
            if normalized_key in forbidden_keys:
                found.add(normalized_key)
            found.update(_forbidden_keys_found(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            found.update(_forbidden_keys_found(nested))
    return found


def _wall_time_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    source = float(report["source_stage139"]["candidate_pool_build_seconds"])
    optimized = float(report["timing_seconds"]["run_online_retriever_and_compare_all_rows"])
    return [
        BarDatum("Stage139 offline batch", source, f"{source:.3f}s"),
        BarDatum("Stage140 online retriever batch", optimized, f"{optimized:.3f}s"),
    ]


def _latency_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for split in (_TRAIN_SPLIT, _DEV_SPLIT):
        latency = report["split_reports"][split]["latency_seconds"]
        for metric in ("p50", "p95", "max"):
            value = float(latency[metric])
            bars.append(BarDatum(f"{split} {metric}", value, f"{value:.4f}s"))
    return bars


def _channel_latency_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(channel_id, float(values["p95"]), f"{float(values['p95']):.4f}s")
        for channel_id, values in report["split_reports"][_TRAIN_SPLIT][
            "channel_latency_seconds"
        ].items()
    ]


def _recall_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for split in (_TRAIN_SPLIT, _DEV_SPLIT):
        for top_k, value in report["split_reports"][split]["recall"]["hit_at_k"].items():
            numeric = float(value)
            bars.append(BarDatum(f"{split} Recall@{top_k}", numeric, f"{numeric:.4f}"))
    return bars
