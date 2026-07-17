from __future__ import annotations

import hashlib
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
    _selected_append_config,
)
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    EncoderFactory,
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
    _load_json_object,
)
from ts_rag_agent.application.primeqa_hybrid_online_candidate_pool_performance_validation import (
    _online_channels,
    _retrieval_config,
)
from ts_rag_agent.application.primeqa_hybrid_online_candidate_pool_retriever import (
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

_STAGE = "Stage 142"
_CREATED_AT = "2026-07-17"
_ANALYSIS_ID = "primeqa_hybrid_strict_warm_single_request_latency_validation_v1"
_SOURCE_STAGE141_STATUS = "primeqa_hybrid_nondefault_runtime_activation_protocol_frozen"
_SOURCE_STAGE140_STATUS = "primeqa_hybrid_online_candidate_pool_performance_validation_passed"
_SLO_PROFILE_ID = "strict_c_warm_single_request_v1"
_SELECTED_CONFIG_ID = "prefix_existing_dense_broad_append200_v1"
_SELECTED_ROUTE_SET = "stage116_lexical_routes_plus_existing_dense_cache_routes"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_TRAIN_REPETITIONS = 3
_TRAIN_FOLD_COUNT = 5
_P95_LIMIT_SECONDS = 0.3
_P99_LIMIT_SECONDS = 1.0
_TOP_K_VALUES = (10, 50, 100, 200, 400)
_SPECIAL_CHANNEL_ID = "special_token_boosted_bm25"
_BASE_CHANNEL_ID = "full_document_bm25"
_NEXT_DIRECTION = "implement_nondefault_single_request_runtime_wiring_and_activation_validation"
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
        "question_id",
        "question_text",
        "question_title",
        "raw_answer_text",
        "raw_document_text",
        "raw_question_text",
        "retrieved_doc_ids",
        "runtime_content_handle",
        "sample_id",
        "source_doc_ids",
    }
)


@dataclass(frozen=True)
class PrimeQAHybridStrictLatencyVisualization:
    name: str
    path: str


@dataclass(frozen=True)
class _PassObservation:
    report: dict[str, Any]
    latencies: tuple[float, ...]
    channel_latencies: Mapping[str, tuple[float, ...]]
    fusion_latencies: tuple[float, ...]
    materialization_latencies: tuple[float, ...]
    fold_latencies: Mapping[str, tuple[float, ...]]


def run_primeqa_hybrid_strict_latency_validation(
    *,
    stage141_protocol_path: Path,
    stage140_validation_path: Path,
    stage128_protocol_path: Path,
    stage127_review_path: Path,
    stage125_protocol_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    user_confirmed_validation: bool,
    confirmation_note: str,
    encoder_batch_size: int = 64,
    encoder_device: str | None = None,
    encoder_factory: EncoderFactory | None = None,
) -> dict[str, Any]:
    """Run the frozen strict-C train-first warm latency validation."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    started_at = time.perf_counter()
    stage141 = _load_json_object(stage141_protocol_path)
    stage140 = _load_json_object(stage140_validation_path)
    stage128 = _load_json_object(stage128_protocol_path)
    stage127 = _load_json_object(stage127_review_path)
    stage125 = _load_json_object(stage125_protocol_path)
    stage80 = _load_json_object(stage80_report_path)
    selected_config = _selected_append_config(
        stage125_protocol=stage125,
        stage128_summary=_stage128_summary(stage128),
    )
    loaded_protocols_at = time.perf_counter()
    source_files = _source_files(
        stage141_protocol=stage141_protocol_path,
        stage140_validation=stage140_validation_path,
        stage128_protocol=stage128_protocol_path,
        stage127_review=stage127_review_path,
        stage125_protocol=stage125_protocol_path,
        stage80_report=stage80_report_path,
        train_split=train_split_path,
        dev_split=dev_split_path,
        documents=documents_path,
    )
    source_checks = _source_checks(
        stage141=stage141,
        stage140=stage140,
        stage127=stage127,
        selected_config=selected_config,
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
    )
    if not all(check["passed"] for check in source_checks):
        return _source_blocked_report(
            source_files=source_files,
            stage141=stage141,
            stage140=stage140,
            guard_checks=source_checks,
            timing_seconds={
                "load_public_protocols": round(loaded_protocols_at - started_at, 3),
                "total": round(time.perf_counter() - started_at, 3),
            },
        )

    train_samples = load_primeqa_hybrid_split_samples(train_split_path)
    train_fold_assignments = _build_train_fold_assignments(
        train_samples,
        fold_count=_TRAIN_FOLD_COUNT,
    )
    train_loaded_at = time.perf_counter()
    documents_by_id = load_primeqa_documents(documents_path)
    sections_by_document = load_primeqa_document_sections(documents_path)
    documents = list(documents_by_id.values())
    document_ids = tuple(document.id for document in documents)
    documents_loaded_at = time.perf_counter()
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
    lexical_channels = _build_lexical_channels(
        documents=documents,
        sections_by_document=sections_by_document,
        bm25_k1=_DEFAULT_BM25_K1,
        bm25_b=_DEFAULT_BM25_B,
        component_depth=int(selected_config["append_generation"]["channel_top_k"]),
    )
    raw_channels = lexical_channels + dense_channels
    online_channels = _online_channels(raw_channels)
    retriever = PrimeQAHybridOnlineCandidatePoolRetriever(
        channels=online_channels,
        config=_retrieval_config(selected_config),
    )
    reference_retriever = PrimeQAHybridOnlineCandidatePoolRetriever(
        channels=_full_sort_reference_channels(raw_channels),
        config=_retrieval_config(selected_config),
    )
    indexes_ready_at = time.perf_counter()
    train_reference_pools = _build_full_sort_reference_pools(
        samples=train_samples,
        reference_retriever=reference_retriever,
    )
    train_reference_ready_at = time.perf_counter()
    warmup_report = _warmup(
        samples=train_samples,
        retriever=retriever,
        reference_pools=train_reference_pools,
    )
    warmed_at = time.perf_counter()

    train_observations = [
        _evaluate_pass(
            split=_TRAIN_SPLIT,
            repetition=repetition,
            samples=train_samples,
            retriever=retriever,
            reference_pools=train_reference_pools,
            fold_assignments=train_fold_assignments,
        )
        for repetition in range(1, _TRAIN_REPETITIONS + 1)
    ]
    train_evaluated_at = time.perf_counter()
    train_report = _aggregate_train_observations(train_observations)
    train_gate_passed = _train_gate_passed(train_report, warmup_report)

    dev_samples: Sequence[PrimeQAHybridSplitSample] = ()
    dev_observation: _PassObservation | None = None
    dev_reference_ready_at = train_evaluated_at
    dev_evaluated_at = train_evaluated_at
    if train_gate_passed:
        dev_samples = load_primeqa_hybrid_split_samples(dev_split_path)
        dev_reference_pools = _build_full_sort_reference_pools(
            samples=dev_samples,
            reference_retriever=reference_retriever,
        )
        dev_reference_ready_at = time.perf_counter()
        dev_observation = _evaluate_pass(
            split=_DEV_SPLIT,
            repetition=1,
            samples=dev_samples,
            retriever=retriever,
            reference_pools=dev_reference_pools,
            fold_assignments=None,
        )
        dev_evaluated_at = time.perf_counter()

    dev_report = dev_observation.report if dev_observation is not None else None
    report_without_guards = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": {
            "train_loaded_before_dev": True,
            "train_repetitions": _TRAIN_REPETITIONS,
            "train_grouped_fold_count": _TRAIN_FOLD_COUNT,
            "dev_loaded_only_after_train_gate_passed": train_gate_passed,
            "dev_single_pass_report_only": True,
            "test_split_loaded": False,
            "test_metrics_run": False,
            "runtime_flag_implemented": False,
            "runtime_entrypoint_registered": False,
            "runtime_default_changed": False,
            "concurrent_request_validation_run": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
        },
        "user_confirmation": {
            "confirmed": user_confirmed_validation,
            "note_present": bool(confirmation_note.strip()),
        },
        "source_files": source_files,
        "source_stage141": _stage141_summary(stage141),
        "source_stage140": _stage140_summary(stage140),
        "optimization_contract": {
            "strategy_id": "exact_boundary_top_k_sequential_channels_v1",
            "score_formulas_changed": False,
            "query_tokenization_changed": False,
            "tie_break_semantics_changed": False,
            "candidate_pool_fusion_changed": False,
            "full_eligible_sort_replaced": True,
            "boundary_ties_retained_before_final_sort": True,
            "length_normalizers_precomputed_at_index_build": True,
            "historical_full_sort_reference_executed": True,
            "identity_reference": "same_scores_historical_full_eligible_sort",
            "request_internal_channel_parallelism_enabled": False,
            "request_execution": "sequential_seven_channel_graph",
            "channel_count": len(online_channels),
            "independent_channel_count": sum(
                isinstance(channel, IndependentCandidatePoolSearchChannel)
                for channel in online_channels
            ),
            "derived_channel_count": sum(
                isinstance(channel, DerivedCandidatePoolSearchChannel)
                for channel in online_channels
            ),
        },
        "strict_latency_slo": {
            "profile_id": _SLO_PROFILE_ID,
            "scope": "warm_single_request_end_to_end_candidate_pool_retrieval",
            "p95_limit_seconds": _P95_LIMIT_SECONDS,
            "p99_limit_seconds": _P99_LIMIT_SECONDS,
            "percentile_method": "linear_interpolation_at_(n_minus_1)_times_p",
        },
        "candidate_pool_contract": {
            "config_id": selected_config["config_id"],
            "route_set": selected_config["append_generation"]["route_set"],
            "channel_top_k": selected_config["append_generation"]["channel_top_k"],
            "prefix_depth": _BASELINE_PREFIX_DEPTH,
            "target_pool_depth": selected_config["append_generation"]["target_pool_depth"],
            "rrf_k": selected_config["append_generation"]["rrf_k"],
        },
        "loaded_data_summary": {
            "train": summarize_primeqa_hybrid_split_samples({_TRAIN_SPLIT: train_samples})[
                _TRAIN_SPLIT
            ],
            "dev": (
                summarize_primeqa_hybrid_split_samples({_DEV_SPLIT: dev_samples})[_DEV_SPLIT]
                if dev_samples
                else None
            ),
            "document_count": len(documents),
            "section_count": sum(map(len, sections_by_document.values())),
            "test_split_loaded": False,
        },
        "dense_channel_preflight": dense_summary,
        "warmup": warmup_report,
        "train_validation": train_report,
        "dev_report_only_validation": dev_report,
        "timing_seconds": {
            "load_public_protocols": round(loaded_protocols_at - started_at, 3),
            "load_train_and_build_folds": round(train_loaded_at - loaded_protocols_at, 3),
            "load_documents_sections": round(documents_loaded_at - train_loaded_at, 3),
            "load_dense_models_and_cached_embeddings": round(
                dense_ready_at - documents_loaded_at,
                3,
            ),
            "build_long_lived_lexical_indexes": round(indexes_ready_at - dense_ready_at, 3),
            "build_train_full_sort_reference_pools": round(
                train_reference_ready_at - indexes_ready_at,
                3,
            ),
            "run_explicit_warmup": round(warmed_at - train_reference_ready_at, 6),
            "run_three_train_measurement_passes": round(
                train_evaluated_at - warmed_at,
                3,
            ),
            "build_dev_full_sort_reference_pools_after_train_gate": round(
                dev_reference_ready_at - train_evaluated_at,
                3,
            ),
            "run_dev_report_only_after_train_gate": round(
                dev_evaluated_at - dev_reference_ready_at,
                3,
            ),
            "total": round(dev_evaluated_at - started_at, 3),
        },
    }
    guards = source_checks + _validation_checks(report_without_guards)
    report = {
        **report_without_guards,
        "guard_checks": guards,
        "decision": _decision(guards, train_gate_passed=train_gate_passed),
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_strict_latency_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridStrictLatencyVisualization]:
    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage142_train_pass_latency_vs_slo.svg": render_horizontal_bar_chart_svg(
            title="Stage142 train pass latency versus strict-C SLO",
            bars=_train_pass_latency_bars(report),
            x_label="seconds",
            width=1540,
            margin_left=760,
        ),
        "stage142_train_fold_worst_latency.svg": render_horizontal_bar_chart_svg(
            title="Stage142 worst train-fold latency across passes",
            bars=_train_fold_latency_bars(report),
            x_label="seconds",
            width=1500,
            margin_left=720,
        ),
        "stage142_stage140_latency_comparison.svg": render_horizontal_bar_chart_svg(
            title="Stage142 versus Stage140 warm latency",
            bars=_source_comparison_bars(report),
            x_label="seconds",
            width=1480,
            margin_left=720,
        ),
        "stage142_train_channel_p95_latency.svg": render_horizontal_bar_chart_svg(
            title="Stage142 pooled train channel P95 latency",
            bars=_channel_latency_bars(report),
            x_label="seconds",
            width=1640,
            margin_left=840,
        ),
        "stage142_dev_latency_vs_slo.svg": render_horizontal_bar_chart_svg(
            title="Stage142 dev report-only latency versus strict-C SLO",
            bars=_dev_latency_bars(report),
            x_label="seconds",
            width=1420,
            margin_left=700,
        ),
        "stage142_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage142 strict latency decision flags",
            bars=_decision_bars(report),
            x_label="1 means true",
            width=1800,
            margin_left=960,
        ),
        "stage142_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage142 strict latency guard checks",
            bars=[
                BarDatum(
                    label=str(check["name"]),
                    value=1.0 if check["passed"] else 0.0,
                    value_label="passed" if check["passed"] else "failed",
                )
                for check in report.get("guard_checks", [])
            ],
            x_label="1 means passed",
            width=2320,
            margin_left=1320,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(PrimeQAHybridStrictLatencyVisualization(filename, str(path)))
    return artifacts


def _full_sort_reference_channels(raw_channels: Sequence[Any]) -> tuple[Any, ...]:
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
            continue
        reference_searcher = getattr(channel.retriever, "search_full_sort_reference", None)
        if not callable(reference_searcher):
            raise TypeError(
                "Every independent Stage142 channel must expose "
                f"search_full_sort_reference: {channel.channel_id}"
            )
        channels.append(
            IndependentCandidatePoolSearchChannel(
                channel_id=channel.channel_id,
                family=channel.family,
                weight=channel.weight,
                searcher=reference_searcher,
            )
        )
    return tuple(channels)


def _build_full_sort_reference_pools(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    reference_retriever: PrimeQAHybridOnlineCandidatePoolRetriever,
) -> dict[str, tuple[str, ...]]:
    return {
        sample.sample_id: tuple(
            result.document.id
            for result in reference_retriever.retrieve(sample.to_primeqa_question())
        )
        for sample in samples
    }


def _warmup(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    retriever: PrimeQAHybridOnlineCandidatePoolRetriever,
    reference_pools: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    sample = min(
        samples,
        key=lambda row: hashlib.sha256(f"{_ANALYSIS_ID}:{row.sample_id}".encode()).hexdigest(),
    )
    run = retriever.retrieve_profiled(sample.to_primeqa_question())
    doc_ids = tuple(result.document.id for result in run.results)
    reference = tuple(reference_pools[sample.sample_id])
    return {
        "request_count": 1,
        "source": "deterministic_train_only_row_selected_without_labels",
        "excluded_from_measurement": True,
        "same_row_retained_in_every_complete_train_pass": True,
        "candidate_pool_exact_identity_violation_count": int(doc_ids != reference),
        "candidate_pool_depth": len(doc_ids),
        "latency_seconds": round(run.profile.total_seconds, 6),
    }


def _evaluate_pass(
    *,
    split: str,
    repetition: int,
    samples: Sequence[PrimeQAHybridSplitSample],
    retriever: PrimeQAHybridOnlineCandidatePoolRetriever,
    reference_pools: Mapping[str, Sequence[str]],
    fold_assignments: Mapping[str, str] | None,
) -> _PassObservation:
    latencies = []
    channel_latencies: dict[str, list[float]] = defaultdict(list)
    fusion_latencies = []
    materialization_latencies = []
    fold_latencies: dict[str, list[float]] = defaultdict(list)
    pool_sizes = []
    identity_violations = 0
    answerable_count = 0
    hit_counts = {top_k: 0 for top_k in _TOP_K_VALUES}
    fold_hit_counts: dict[str, dict[int, int]] = defaultdict(
        lambda: {top_k: 0 for top_k in _TOP_K_VALUES}
    )
    fold_answerable_counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        run = retriever.retrieve_profiled(sample.to_primeqa_question())
        doc_ids = tuple(result.document.id for result in run.results)
        reference = tuple(reference_pools[sample.sample_id])
        identity_violations += doc_ids != reference
        latency = run.profile.total_seconds
        latencies.append(latency)
        fusion_latencies.append(run.profile.fusion_seconds)
        materialization_latencies.append(run.profile.materialization_seconds)
        pool_sizes.append(len(doc_ids))
        for timing in run.profile.channel_timings:
            channel_latencies[timing.channel_id].append(timing.duration_seconds)
        fold_id = fold_assignments.get(sample.sample_id) if fold_assignments else None
        if fold_id is not None:
            fold_latencies[fold_id].append(latency)
        if sample.answerable and sample.answer_doc_id is not None:
            answerable_count += 1
            if fold_id is not None:
                fold_answerable_counts[fold_id] += 1
            for top_k in _TOP_K_VALUES:
                hit = sample.answer_doc_id in doc_ids[:top_k]
                hit_counts[top_k] += hit
                if fold_id is not None:
                    fold_hit_counts[fold_id][top_k] += hit
    latency_distribution = _distribution(latencies)
    fold_reports = {
        fold_id: {
            "row_count": len(values),
            "answerable_count": fold_answerable_counts[fold_id],
            "latency_seconds": _distribution(values),
            "strict_slo_passed": _strict_latency_pass(_distribution(values)),
            "hit_counts": {str(top_k): fold_hit_counts[fold_id][top_k] for top_k in _TOP_K_VALUES},
        }
        for fold_id, values in sorted(fold_latencies.items())
    }
    report = {
        "split": split,
        "repetition": repetition,
        "row_count": len(samples),
        "answerable_count": answerable_count,
        "exact_candidate_pool_identity_violation_count": identity_violations,
        "candidate_pool_size": _distribution(pool_sizes),
        "latency_seconds": latency_distribution,
        "strict_slo_passed": _strict_latency_pass(latency_distribution),
        "channel_latency_seconds": {
            channel_id: _distribution(values) for channel_id, values in channel_latencies.items()
        },
        "fusion_latency_seconds": _distribution(fusion_latencies),
        "materialization_latency_seconds": _distribution(materialization_latencies),
        "recall": {
            "hit_counts": {str(top_k): hit_counts[top_k] for top_k in _TOP_K_VALUES},
            "hit_at_k": {
                str(top_k): _ratio(hit_counts[top_k], answerable_count) for top_k in _TOP_K_VALUES
            },
        },
        "fold_reports": fold_reports,
        "all_fold_strict_slo_passed": all(
            row["strict_slo_passed"] for row in fold_reports.values()
        ),
    }
    return _PassObservation(
        report=report,
        latencies=tuple(latencies),
        channel_latencies={key: tuple(values) for key, values in channel_latencies.items()},
        fusion_latencies=tuple(fusion_latencies),
        materialization_latencies=tuple(materialization_latencies),
        fold_latencies={key: tuple(values) for key, values in fold_latencies.items()},
    )


def _aggregate_train_observations(
    observations: Sequence[_PassObservation],
) -> dict[str, Any]:
    latencies = [value for observation in observations for value in observation.latencies]
    channels: dict[str, list[float]] = defaultdict(list)
    folds: dict[str, list[float]] = defaultdict(list)
    for observation in observations:
        for channel_id, values in observation.channel_latencies.items():
            channels[channel_id].extend(values)
        for fold_id, values in observation.fold_latencies.items():
            folds[fold_id].extend(values)
    combined_distribution = _distribution(latencies)
    combined_folds = {
        fold_id: {
            "request_count": len(values),
            "latency_seconds": _distribution(values),
            "strict_slo_passed": _strict_latency_pass(_distribution(values)),
        }
        for fold_id, values in sorted(folds.items())
    }
    return {
        "measurement_repetitions": len(observations),
        "row_count_per_pass": observations[0].report["row_count"] if observations else 0,
        "measured_request_count": len(latencies),
        "pass_reports": [observation.report for observation in observations],
        "combined_latency_seconds": combined_distribution,
        "combined_strict_slo_passed": _strict_latency_pass(combined_distribution),
        "combined_channel_latency_seconds": {
            channel_id: _distribution(values) for channel_id, values in channels.items()
        },
        "combined_fusion_latency_seconds": _distribution(
            [value for observation in observations for value in observation.fusion_latencies]
        ),
        "combined_materialization_latency_seconds": _distribution(
            [
                value
                for observation in observations
                for value in observation.materialization_latencies
            ]
        ),
        "combined_fold_reports": combined_folds,
        "all_combined_folds_strict_slo_passed": all(
            row["strict_slo_passed"] for row in combined_folds.values()
        ),
        "all_passes_strict_slo_passed": all(
            observation.report["strict_slo_passed"] for observation in observations
        ),
        "all_pass_folds_strict_slo_passed": all(
            observation.report["all_fold_strict_slo_passed"] for observation in observations
        ),
        "total_exact_candidate_pool_identity_violation_count": sum(
            observation.report["exact_candidate_pool_identity_violation_count"]
            for observation in observations
        ),
    }


def _train_gate_passed(
    train_report: Mapping[str, Any],
    warmup_report: Mapping[str, Any],
) -> bool:
    return (
        warmup_report.get("candidate_pool_exact_identity_violation_count") == 0
        and train_report.get("measurement_repetitions") == _TRAIN_REPETITIONS
        and train_report.get("all_passes_strict_slo_passed") is True
        and train_report.get("all_pass_folds_strict_slo_passed") is True
        and train_report.get("combined_strict_slo_passed") is True
        and train_report.get("all_combined_folds_strict_slo_passed") is True
        and train_report.get("total_exact_candidate_pool_identity_violation_count") == 0
    )


def _strict_latency_pass(distribution: Mapping[str, Any]) -> bool:
    return (
        float(distribution.get("p95", float("inf"))) <= _P95_LIMIT_SECONDS
        and float(distribution.get("p99", float("inf"))) <= _P99_LIMIT_SECONDS
    )


def _distribution(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "average": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": round(min(numeric), 6),
        "average": round(mean(numeric), 6),
        "p50": round(_percentile(numeric, 50), 6),
        "p95": round(_percentile(numeric, 95), 6),
        "p99": round(_percentile(numeric, 99), 6),
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


def _source_checks(
    *,
    stage141: Mapping[str, Any],
    stage140: Mapping[str, Any],
    stage127: Mapping[str, Any],
    selected_config: Mapping[str, Any],
    user_confirmed_validation: bool,
    confirmation_note: str,
) -> list[dict[str, Any]]:
    stage141_decision = stage141.get("decision") or {}
    stage141_guards = stage141.get("guard_checks") or []
    stage141_slo = (stage141.get("frozen_protocol") or {}).get("latency_slo") or {}
    stage140_decision = stage140.get("decision") or {}
    stage140_guards = stage140.get("guard_checks") or []
    stage140_contract = stage140.get("selected_candidate_pool_contract") or {}
    return [
        _check(
            "stage142_user_confirmed",
            user_confirmed_validation and bool(confirmation_note.strip()),
            {
                "confirmed": user_confirmed_validation,
                "note_present": bool(confirmation_note.strip()),
            },
            True,
        ),
        _check(
            "stage141_source_protocol_frozen",
            stage141.get("stage") == "Stage 141"
            and stage141_decision.get("status") == _SOURCE_STAGE141_STATUS
            and stage141_decision.get("runtime_activation_protocol_frozen") is True,
            {
                "stage": stage141.get("stage"),
                "status": stage141_decision.get("status"),
                "frozen": stage141_decision.get("runtime_activation_protocol_frozen"),
            },
            _SOURCE_STAGE141_STATUS,
        ),
        _check(
            "stage141_all_guards_passed",
            len(stage141_guards) == 19
            and sum(bool(row.get("passed")) for row in stage141_guards) == 19,
            {
                "passed": sum(bool(row.get("passed")) for row in stage141_guards),
                "total": len(stage141_guards),
            },
            {"passed": 19, "total": 19},
        ),
        _check(
            "strict_c_slo_matches_frozen_protocol",
            stage141_slo.get("profile_id") == _SLO_PROFILE_ID
            and stage141_slo.get("p95_seconds") == _P95_LIMIT_SECONDS
            and stage141_slo.get("p99_seconds") == _P99_LIMIT_SECONDS
            and stage141_slo.get("measurement_repetitions") == _TRAIN_REPETITIONS,
            {
                key: stage141_slo.get(key)
                for key in (
                    "profile_id",
                    "p95_seconds",
                    "p99_seconds",
                    "measurement_repetitions",
                )
            },
            {
                "profile_id": _SLO_PROFILE_ID,
                "p95_seconds": _P95_LIMIT_SECONDS,
                "p99_seconds": _P99_LIMIT_SECONDS,
                "measurement_repetitions": _TRAIN_REPETITIONS,
            },
        ),
        _check(
            "stage141_runtime_and_test_boundaries_remain_closed",
            stage141_decision.get("runtime_settings_flag_implemented") is False
            and stage141_decision.get("runtime_entrypoint_registered") is False
            and stage141_decision.get("runtime_activation_allowed_now") is False
            and stage141_decision.get("runtime_defaultization_allowed_now") is False
            and stage141_decision.get("test_gate_opened") is False
            and stage141_decision.get("retry_actions_enabled") is False
            and stage141_decision.get("fallback_strategies_enabled") is False,
            stage141_decision,
            "runtime/test/default/retry/fallback closed",
        ),
        _check(
            "stage140_source_validation_passed",
            stage140.get("stage") == "Stage 140"
            and stage140_decision.get("status") == _SOURCE_STAGE140_STATUS
            and len(stage140_guards) == 21
            and sum(bool(row.get("passed")) for row in stage140_guards) == 21,
            {
                "stage": stage140.get("stage"),
                "status": stage140_decision.get("status"),
                "passed": sum(bool(row.get("passed")) for row in stage140_guards),
                "total": len(stage140_guards),
            },
            {"status": _SOURCE_STAGE140_STATUS, "passed": 21, "total": 21},
        ),
        _check(
            "frozen_candidate_pool_contract_matches",
            stage140_contract.get("config_id") == _SELECTED_CONFIG_ID
            and stage140_contract.get("route_set") == _SELECTED_ROUTE_SET
            and stage140_contract.get("channel_top_k") == 400
            and stage140_contract.get("prefix_depth") == 200
            and stage140_contract.get("target_pool_depth") == 400
            and stage140_contract.get("rrf_k") == 60
            and stage140_contract.get("channel_count") == 7
            and selected_config.get("config_id") == _SELECTED_CONFIG_ID
            and (selected_config.get("append_generation") or {}).get("route_set")
            == _SELECTED_ROUTE_SET,
            {
                "stage140": stage140_contract,
                "selected_config_id": selected_config.get("config_id"),
            },
            "frozen Stage128 Top200/Top400 seven-channel contract",
        ),
        _check(
            "stage127_selected_config_matches",
            (stage127.get("decision") or {}).get("selected_config_id") == _SELECTED_CONFIG_ID,
            (stage127.get("decision") or {}).get("selected_config_id"),
            _SELECTED_CONFIG_ID,
        ),
        _check(
            "public_sources_are_safe",
            (stage141.get("public_safe_contract") or {}).get("forbidden_keys_found") == []
            and (stage140.get("public_safe_contract") or {}).get("forbidden_keys_found") == [],
            {
                "stage141": (stage141.get("public_safe_contract") or {}).get(
                    "forbidden_keys_found"
                ),
                "stage140": (stage140.get("public_safe_contract") or {}).get(
                    "forbidden_keys_found"
                ),
            },
            [],
        ),
    ]


def _validation_checks(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    optimization = report.get("optimization_contract") or {}
    scope = report.get("analysis_scope") or {}
    warmup = report.get("warmup") or {}
    train = report.get("train_validation") or {}
    dev = report.get("dev_report_only_validation") or {}
    source_recall = (report.get("source_stage140") or {}).get("recall_by_split") or {}
    passes = train.get("pass_reports") or []
    checks = [
        _check(
            "exact_boundary_top_k_strategy_is_frozen",
            optimization.get("strategy_id") == "exact_boundary_top_k_sequential_channels_v1"
            and optimization.get("score_formulas_changed") is False
            and optimization.get("query_tokenization_changed") is False
            and optimization.get("tie_break_semantics_changed") is False
            and optimization.get("candidate_pool_fusion_changed") is False
            and optimization.get("boundary_ties_retained_before_final_sort") is True
            and optimization.get("historical_full_sort_reference_executed") is True
            and optimization.get("identity_reference")
            == "same_scores_historical_full_eligible_sort",
            optimization,
            "exact boundary Top-K with unchanged scoring, ties, and fusion",
        ),
        _check(
            "sequential_seven_channel_graph_preserved",
            optimization.get("request_internal_channel_parallelism_enabled") is False
            and optimization.get("request_execution") == "sequential_seven_channel_graph"
            and optimization.get("channel_count") == 7
            and optimization.get("independent_channel_count") == 6
            and optimization.get("derived_channel_count") == 1,
            optimization,
            "sequential 7=6+1 channel graph",
        ),
        _check(
            "warmup_is_train_only_excluded_and_exact",
            warmup.get("request_count") == 1
            and warmup.get("source") == "deterministic_train_only_row_selected_without_labels"
            and warmup.get("excluded_from_measurement") is True
            and warmup.get("same_row_retained_in_every_complete_train_pass") is True
            and warmup.get("candidate_pool_exact_identity_violation_count") == 0
            and warmup.get("candidate_pool_depth") == 400,
            warmup,
            "one exact train-only warmup excluded from timing",
        ),
        _check(
            "three_complete_train_passes_measured",
            train.get("measurement_repetitions") == _TRAIN_REPETITIONS
            and train.get("row_count_per_pass") == 562
            and train.get("measured_request_count") == 562 * _TRAIN_REPETITIONS
            and len(passes) == _TRAIN_REPETITIONS,
            {
                "repetitions": train.get("measurement_repetitions"),
                "rows_per_pass": train.get("row_count_per_pass"),
                "requests": train.get("measured_request_count"),
            },
            {"repetitions": 3, "rows_per_pass": 562, "requests": 1686},
        ),
        _check(
            "all_train_candidate_pools_exact",
            train.get("total_exact_candidate_pool_identity_violation_count") == 0
            and all(
                row.get("exact_candidate_pool_identity_violation_count") == 0 for row in passes
            ),
            train.get("total_exact_candidate_pool_identity_violation_count"),
            0,
        ),
        _check(
            "all_train_candidate_pool_depths_are_400",
            all(
                (row.get("candidate_pool_size") or {}).get("min") == 400
                and (row.get("candidate_pool_size") or {}).get("max") == 400
                for row in passes
            ),
            [row.get("candidate_pool_size") for row in passes],
            400,
        ),
        _check(
            "all_train_recall_counts_match_stage140",
            len(passes) == _TRAIN_REPETITIONS
            and all(
                (row.get("recall") or {}).get("hit_counts")
                == (source_recall.get(_TRAIN_SPLIT) or {}).get("hit_counts")
                for row in passes
            ),
            [(row.get("recall") or {}).get("hit_counts") for row in passes],
            (source_recall.get(_TRAIN_SPLIT) or {}).get("hit_counts"),
        ),
        _check(
            "all_train_passes_meet_strict_slo",
            train.get("all_passes_strict_slo_passed") is True,
            [
                {
                    "p95": (row.get("latency_seconds") or {}).get("p95"),
                    "p99": (row.get("latency_seconds") or {}).get("p99"),
                    "passed": row.get("strict_slo_passed"),
                }
                for row in passes
            ],
            {"p95": "<=0.3", "p99": "<=1.0"},
        ),
        _check(
            "all_train_pass_folds_meet_strict_slo",
            train.get("all_pass_folds_strict_slo_passed") is True
            and all(len(row.get("fold_reports") or {}) == _TRAIN_FOLD_COUNT for row in passes),
            [
                {
                    fold_id: fold.get("strict_slo_passed")
                    for fold_id, fold in (row.get("fold_reports") or {}).items()
                }
                for row in passes
            ],
            "five folds pass in every repetition",
        ),
        _check(
            "combined_train_and_folds_meet_strict_slo",
            train.get("combined_strict_slo_passed") is True
            and train.get("all_combined_folds_strict_slo_passed") is True
            and len(train.get("combined_fold_reports") or {}) == _TRAIN_FOLD_COUNT,
            {
                "combined": train.get("combined_latency_seconds"),
                "folds": train.get("combined_fold_reports"),
            },
            {"p95": "<=0.3", "p99": "<=1.0"},
        ),
        _check(
            "dev_loaded_only_after_train_gate",
            scope.get("train_loaded_before_dev") is True
            and scope.get("dev_loaded_only_after_train_gate_passed") is True
            and _train_gate_passed(train, warmup),
            {
                "train_gate": _train_gate_passed(train, warmup),
                "dev_after_gate": scope.get("dev_loaded_only_after_train_gate_passed"),
            },
            True,
        ),
        _check(
            "dev_single_pass_is_exact_and_depth_400",
            dev.get("split") == _DEV_SPLIT
            and dev.get("repetition") == 1
            and dev.get("row_count") == 121
            and dev.get("exact_candidate_pool_identity_violation_count") == 0
            and (dev.get("candidate_pool_size") or {}).get("min") == 400
            and (dev.get("candidate_pool_size") or {}).get("max") == 400,
            {
                "split": dev.get("split"),
                "repetition": dev.get("repetition"),
                "rows": dev.get("row_count"),
                "identity": dev.get("exact_candidate_pool_identity_violation_count"),
                "pool": dev.get("candidate_pool_size"),
            },
            "one dev pass, 121 exact pools of depth 400",
        ),
        _check(
            "dev_recall_matches_stage140",
            (dev.get("recall") or {}).get("hit_counts")
            == (source_recall.get(_DEV_SPLIT) or {}).get("hit_counts"),
            (dev.get("recall") or {}).get("hit_counts"),
            (source_recall.get(_DEV_SPLIT) or {}).get("hit_counts"),
        ),
        _check(
            "dev_report_only_meets_strict_slo",
            dev.get("strict_slo_passed") is True
            and _strict_latency_pass(dev.get("latency_seconds") or {}),
            dev.get("latency_seconds"),
            {"p95": "<=0.3", "p99": "<=1.0"},
        ),
        _check(
            "test_runtime_default_concurrency_retry_fallback_remain_closed",
            scope.get("test_split_loaded") is False
            and scope.get("test_metrics_run") is False
            and scope.get("runtime_flag_implemented") is False
            and scope.get("runtime_entrypoint_registered") is False
            and scope.get("runtime_default_changed") is False
            and scope.get("concurrent_request_validation_run") is False
            and scope.get("retry_actions_enabled") is False
            and scope.get("fallback_strategies_enabled") is False,
            scope,
            False,
        ),
        _check(
            "stage142_report_is_public_safe",
            not _forbidden_keys_found(report),
            sorted(_forbidden_keys_found(report)),
            [],
        ),
    ]
    return checks


def _decision(
    guard_checks: Sequence[Mapping[str, Any]],
    *,
    train_gate_passed: bool,
) -> dict[str, Any]:
    failed = [str(row["name"]) for row in guard_checks if not row["passed"]]
    base = {
        "slo_profile_id": _SLO_PROFILE_ID,
        "strict_warm_p95_limit_seconds": _P95_LIMIT_SECONDS,
        "strict_warm_p99_limit_seconds": _P99_LIMIT_SECONDS,
        "train_gate_passed_before_dev": train_gate_passed,
        "runtime_settings_flag_implemented": False,
        "runtime_entrypoint_registered": False,
        "runtime_activation_allowed_now": False,
        "runtime_activated_now": False,
        "concurrent_runtime_activation_allowed": False,
        "runtime_defaultization_allowed_now": False,
        "test_gate_opened": False,
        "retry_actions_enabled": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }
    if failed:
        return {
            **base,
            "status": "primeqa_hybrid_strict_warm_latency_validation_blocked",
            "failed_checks": failed,
            "strict_slo_validation_passed": False,
            "strict_slo_evidence_state": "rejected",
            "can_implement_nondefault_runtime_wiring_now": False,
            "recommended_next_direction": "repair_stage142_strict_latency_or_identity_failures",
        }
    return {
        **base,
        "status": "primeqa_hybrid_strict_warm_latency_validation_passed",
        "failed_checks": [],
        "strict_slo_validation_passed": True,
        "strict_slo_evidence_state": "eligible",
        "can_implement_nondefault_runtime_wiring_now": True,
        "recommended_next_direction": _NEXT_DIRECTION,
    }


def _source_blocked_report(
    *,
    source_files: Mapping[str, Any],
    stage141: Mapping[str, Any],
    stage140: Mapping[str, Any],
    guard_checks: Sequence[Mapping[str, Any]],
    timing_seconds: Mapping[str, Any],
) -> dict[str, Any]:
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": {
            "split_files_loaded": False,
            "test_split_loaded": False,
            "test_metrics_run": False,
            "runtime_default_changed": False,
        },
        "source_files": dict(source_files),
        "source_stage141": _stage141_summary(stage141),
        "source_stage140": _stage140_summary(stage140),
        "guard_checks": list(guard_checks),
        "decision": _decision(guard_checks, train_gate_passed=False),
        "timing_seconds": dict(timing_seconds),
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def _stage141_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision") or {}
    guards = report.get("guard_checks") or []
    slo = (report.get("frozen_protocol") or {}).get("latency_slo") or {}
    return {
        "stage": report.get("stage"),
        "protocol_id": report.get("protocol_id"),
        "status": decision.get("status"),
        "guard_check_count": len(guards),
        "guard_check_passed_count": sum(bool(row.get("passed")) for row in guards),
        "runtime_activation_protocol_frozen": decision.get("runtime_activation_protocol_frozen"),
        "strict_slo_currently_satisfied": decision.get("strict_slo_currently_satisfied"),
        "latency_slo": {
            key: slo.get(key)
            for key in (
                "profile_id",
                "p95_seconds",
                "p99_seconds",
                "measurement_repetitions",
                "train_protocol",
                "dev_protocol",
                "test_protocol",
            )
        },
        "runtime_settings_flag_implemented": decision.get("runtime_settings_flag_implemented"),
        "runtime_entrypoint_registered": decision.get("runtime_entrypoint_registered"),
        "runtime_activation_allowed_now": decision.get("runtime_activation_allowed_now"),
        "test_gate_opened": decision.get("test_gate_opened"),
    }


def _stage140_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision") or {}
    guards = report.get("guard_checks") or []
    splits = report.get("split_reports") or {}
    contract = report.get("selected_candidate_pool_contract") or {}
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "status": decision.get("status"),
        "guard_check_count": len(guards),
        "guard_check_passed_count": sum(bool(row.get("passed")) for row in guards),
        "candidate_pool_identity_preserved": decision.get("candidate_pool_identity_preserved"),
        "retrieval_recall_preserved": decision.get("retrieval_recall_preserved"),
        "candidate_pool_contract": {
            key: contract.get(key)
            for key in (
                "config_id",
                "route_set",
                "channel_top_k",
                "prefix_depth",
                "target_pool_depth",
                "rrf_k",
                "channel_count",
            )
        },
        "latency_by_split": {
            split: {
                key: ((splits.get(split) or {}).get("latency_seconds") or {}).get(key)
                for key in ("average", "p50", "p95", "p99", "max")
            }
            for split in (_TRAIN_SPLIT, _DEV_SPLIT)
        },
        "recall_by_split": {
            split: {"hit_counts": ((splits.get(split) or {}).get("recall") or {}).get("hit_counts")}
            for split in (_TRAIN_SPLIT, _DEV_SPLIT)
        },
    }


def _source_files(**paths: Path) -> dict[str, Any]:
    return {
        name: {"path": str(path), "exists": path.is_file(), "size_bytes": path.stat().st_size}
        for name, path in paths.items()
    }


def _check(name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "aggregate_only": True,
        "private_per_request_trace_written": False,
        "raw_questions_written": False,
        "raw_answers_written": False,
        "raw_documents_written": False,
        "raw_candidate_rows_written": False,
        "raw_sample_or_document_ids_written": False,
        "test_split_loaded": False,
        "test_metrics_run": False,
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


def _train_pass_latency_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    passes = (report.get("train_validation") or {}).get("pass_reports") or []
    bars = []
    for row in passes:
        latency = row.get("latency_seconds") or {}
        repetition = row.get("repetition")
        bars.extend(
            [
                BarDatum(
                    f"train pass {repetition} P95",
                    float(latency.get("p95") or 0),
                    f"{float(latency.get('p95') or 0):.6f}s",
                ),
                BarDatum(
                    f"train pass {repetition} P99",
                    float(latency.get("p99") or 0),
                    f"{float(latency.get('p99') or 0):.6f}s",
                ),
            ]
        )
    bars.extend(
        [
            BarDatum("strict P95 limit", _P95_LIMIT_SECONDS, "0.300000s"),
            BarDatum("strict P99 limit", _P99_LIMIT_SECONDS, "1.000000s"),
        ]
    )
    return bars


def _train_fold_latency_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    passes = (report.get("train_validation") or {}).get("pass_reports") or []
    fold_values: dict[str, dict[str, float]] = defaultdict(lambda: {"p95": 0.0, "p99": 0.0})
    for row in passes:
        for fold_id, fold in (row.get("fold_reports") or {}).items():
            latency = fold.get("latency_seconds") or {}
            fold_values[fold_id]["p95"] = max(
                fold_values[fold_id]["p95"],
                float(latency.get("p95") or 0),
            )
            fold_values[fold_id]["p99"] = max(
                fold_values[fold_id]["p99"],
                float(latency.get("p99") or 0),
            )
    return [
        BarDatum(
            f"{fold_id} worst {percentile.upper()}",
            values[percentile],
            f"{values[percentile]:.6f}s",
        )
        for fold_id, values in sorted(fold_values.items())
        for percentile in ("p95", "p99")
    ]


def _source_comparison_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    source = (report.get("source_stage140") or {}).get("latency_by_split") or {}
    train = (report.get("train_validation") or {}).get("combined_latency_seconds") or {}
    dev = (report.get("dev_report_only_validation") or {}).get("latency_seconds") or {}
    rows = [
        ("Stage140 train P95", (source.get(_TRAIN_SPLIT) or {}).get("p95")),
        ("Stage142 train combined P95", train.get("p95")),
        ("Stage142 train combined P99", train.get("p99")),
        ("Stage140 dev P95", (source.get(_DEV_SPLIT) or {}).get("p95")),
        ("Stage142 dev P95", dev.get("p95")),
        ("Stage142 dev P99", dev.get("p99")),
    ]
    return [
        BarDatum(label, float(value or 0), f"{float(value or 0):.6f}s") for label, value in rows
    ]


def _channel_latency_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    channels = (report.get("train_validation") or {}).get("combined_channel_latency_seconds") or {}
    return [
        BarDatum(
            str(channel_id),
            float((latency or {}).get("p95") or 0),
            f"{float((latency or {}).get('p95') or 0):.6f}s",
        )
        for channel_id, latency in channels.items()
    ]


def _dev_latency_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    latency = (report.get("dev_report_only_validation") or {}).get("latency_seconds") or {}
    return [
        BarDatum(
            "dev P95",
            float(latency.get("p95") or 0),
            f"{float(latency.get('p95') or 0):.6f}s",
        ),
        BarDatum(
            "dev P99",
            float(latency.get("p99") or 0),
            f"{float(latency.get('p99') or 0):.6f}s",
        ),
        BarDatum("strict P95 limit", _P95_LIMIT_SECONDS, "0.300000s"),
        BarDatum("strict P99 limit", _P99_LIMIT_SECONDS, "1.000000s"),
    ]


def _decision_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    keys = (
        "strict_slo_validation_passed",
        "can_implement_nondefault_runtime_wiring_now",
        "runtime_settings_flag_implemented",
        "runtime_entrypoint_registered",
        "runtime_activation_allowed_now",
        "runtime_activated_now",
        "concurrent_runtime_activation_allowed",
        "runtime_defaultization_allowed_now",
        "test_gate_opened",
        "retry_actions_enabled",
        "fallback_strategies_enabled",
    )
    return [
        BarDatum(
            key,
            1.0 if decision.get(key) else 0.0,
            "true" if decision.get(key) else "false",
        )
        for key in keys
    ]
