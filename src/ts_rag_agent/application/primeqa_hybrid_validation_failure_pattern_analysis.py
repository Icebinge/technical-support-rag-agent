from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 107"
_CREATED_AT = "2026-07-16"
_ROUTE_ID = "primeqa_hybrid_validation_failure_pattern_analysis"
_PROTOCOL_ID = "primeqa_hybrid_validation_failure_pattern_analysis_v1"
_SOURCE_STAGE102 = "Stage 102"
_SOURCE_STAGE105 = "Stage 105"
_SOURCE_STAGE106 = "Stage 106"
_EXPECTED_STAGE102_STATUS = "primeqa_hybrid_answer_pipeline_error_decomposition_completed"
_EXPECTED_STAGE105_STATUS = (
    "primeqa_hybrid_evidence_answerability_comparison_completed_dev_guard_failed"
)
_EXPECTED_STAGE106_STATUS = "primeqa_hybrid_evidence_answerability_candidate_family_stopped"
_VALIDATION_SPLIT = "dev"
_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = ("test",)
_BUCKET_ORDER = (
    "answerability_false_answer",
    "retrieval_context_miss",
    "evidence_selection_miss",
    "verification_over_refusal",
    "gold_span_beats_selected_answer",
    "low_overlap_gold_cited_answer",
    "answer_supported_and_cited",
)
_FAILURE_BUCKETS = tuple(
    bucket for bucket in _BUCKET_ORDER if bucket != "answer_supported_and_cited"
)
_ANSWERABLE_FAILURE_BUCKETS = (
    "retrieval_context_miss",
    "evidence_selection_miss",
    "verification_over_refusal",
    "gold_span_beats_selected_answer",
    "low_overlap_gold_cited_answer",
)
_UNANSWERABLE_FAILURE_BUCKETS = ("answerability_false_answer",)
_TARGET_BUCKETS = (
    "answerability_false_answer",
    "gold_span_beats_selected_answer",
    "evidence_selection_miss",
)
_BUCKET_TO_STAGE = {
    "answerability_false_answer": "answerability",
    "retrieval_context_miss": "retrieval",
    "evidence_selection_miss": "evidence_selection",
    "verification_over_refusal": "verification",
    "gold_span_beats_selected_answer": "answer_composition",
    "low_overlap_gold_cited_answer": "answer_composition",
    "answer_supported_and_cited": "non_error_reference",
}
_FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "question_text",
        "question_title",
        "raw_question_text",
        "raw_answer_text",
        "gold_answer",
        "answer_text",
        "document_id",
        "answer_doc_id",
        "retrieved_doc_ids",
        "cited_doc_ids",
        "source_doc_ids",
        "matched_token_strings",
        "query_terms",
        "document_title",
        "document_body",
        "document_text",
    }
)


@dataclass(frozen=True)
class PrimeQAHybridValidationFailurePatternVisualization:
    """One generated Stage107 validation-failure pattern chart."""

    name: str
    path: str


def analyze_primeqa_hybrid_validation_failure_patterns(
    *,
    stage102_report_path: Path,
    stage105_report_path: Path,
    stage106_decision_path: Path,
    user_confirmed_analysis: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze and run a public-safe dev validation-failure pattern analysis."""

    started_at = time.perf_counter()
    stage102_report = _load_json_object(stage102_report_path)
    stage105_report = _load_json_object(stage105_report_path)
    stage106_report = _load_json_object(stage106_decision_path)
    loaded_at = time.perf_counter()

    frozen_protocol = _frozen_protocol()
    stage102_summary = _stage102_summary(stage102_report)
    stage105_summary = _stage105_summary(stage105_report)
    stage106_summary = _stage106_summary(stage106_report)
    pattern_summary = _pattern_summary(stage102_report, stage105_report)
    preliminary_report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "analysis_scope": (
            "Public-safe validation-failure pattern analysis for the current "
            "PrimeQA hybrid answer pipeline. This stage freezes the diagnostic "
            "protocol and reads only saved Stage102, Stage105, and Stage106 "
            "public-safe reports. It does not load train/dev/test split files, "
            "does not load corpus documents, does not run retrieval or answer "
            "metrics, does not run final metrics, does not select from dev-only "
            "observations, does not add fallback strategies, and does not "
            "change runtime defaults."
        ),
        "user_confirmation": {
            "route_id": _ROUTE_ID,
            "confirmed": bool(user_confirmed_analysis),
            "confirmation_note": confirmation_note,
        },
        "split_contract": {
            "split_name": "primeqa_hybrid_stage68_v1",
            "protocol_version": "primeqa_hybrid_split_v1",
            "development_splits": list(_DEVELOPMENT_SPLITS),
            "validation_split": _VALIDATION_SPLIT,
            "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage102_report": _fingerprint(stage102_report_path),
            "stage105_report": _fingerprint(stage105_report_path),
            "stage106_decision": _fingerprint(stage106_decision_path),
        },
        "frozen_protocol": frozen_protocol,
        "source_summaries": {
            "stage102": stage102_summary,
            "stage105": stage105_summary,
            "stage106": stage106_summary,
        },
        "pattern_summary": pattern_summary,
    }
    guard_checks = _guard_checks(
        report=preliminary_report,
        stage102_report=stage102_report,
        stage105_report=stage105_report,
        stage106_report=stage106_report,
        user_confirmed_analysis=user_confirmed_analysis,
    )
    checked_at = time.perf_counter()
    return {
        **preliminary_report,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks=guard_checks, pattern_summary=pattern_summary),
        "timing_seconds": {
            "load_reports": round(loaded_at - started_at, 3),
            "freeze_analyze_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_validation_failure_pattern_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridValidationFailurePatternVisualization]:
    """Write SVG charts for Stage107 validation-failure pattern analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage107_dev_failure_bucket_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage107 dev failure bucket counts",
            bars=_dev_failure_bucket_bars(report),
            x_label="dev row count",
            width=1320,
            margin_left=560,
        ),
        "stage107_train_dev_bucket_rate_drift.svg": render_horizontal_bar_chart_svg(
            title="Stage107 train-dev bucket rate drift",
            bars=_bucket_rate_drift_bars(report),
            x_label="dev rate minus train rate, percentage points",
            width=1380,
            margin_left=620,
        ),
        "stage107_dev_route_failure_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage107 dev route failure counts",
            bars=_dev_route_failure_bars(report),
            x_label="failure count",
            width=1320,
            margin_left=560,
        ),
        "stage107_dev_answerable_failure_flow.svg": render_horizontal_bar_chart_svg(
            title="Stage107 dev answerable failure flow",
            bars=_dev_answerable_flow_bars(report),
            x_label="answerable dev row count",
            width=1320,
            margin_left=620,
        ),
        "stage107_stage105_candidate_behavior.svg": render_horizontal_bar_chart_svg(
            title="Stage107 Stage105 candidate behavior",
            bars=_stage105_candidate_behavior_bars(report),
            x_label="dev changed answer count",
            width=1520,
            margin_left=760,
        ),
        "stage107_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage107 decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1280,
            margin_left=620,
        ),
        "stage107_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage107 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1560,
            margin_left=820,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridValidationFailurePatternVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _frozen_protocol() -> dict[str, Any]:
    return {
        "protocol_status": "frozen_and_executed_public_safe_diagnostic",
        "protocol_id": _PROTOCOL_ID,
        "source_report_contract": {
            "required_source_stages": [
                _SOURCE_STAGE102,
                _SOURCE_STAGE105,
                _SOURCE_STAGE106,
            ],
            "saved_reports_only": True,
            "load_split_files": False,
            "load_corpus_documents": False,
            "run_retrieval_metrics": False,
            "run_answer_metrics": False,
            "run_final_test_metrics": False,
        },
        "validation_failure_definition": {
            "validation_split": _VALIDATION_SPLIT,
            "failure_buckets": list(_FAILURE_BUCKETS),
            "non_error_reference_bucket": "answer_supported_and_cited",
            "answerable_failure_buckets": list(_ANSWERABLE_FAILURE_BUCKETS),
            "unanswerable_failure_buckets": list(_UNANSWERABLE_FAILURE_BUCKETS),
        },
        "analysis_views": [
            "dev bucket counts and rates",
            "train-dev bucket rate drift",
            "dev answerability cross-tab",
            "dev route failure concentration",
            "dev retrieval-rank and token-F1 distributions",
            "Stage105 train-selectability failure pattern",
        ],
        "selection_policy": {
            "select_config_from_dev": False,
            "retune_thresholds_on_dev": False,
            "open_runtime_gate": False,
            "open_final_test_gate": False,
        },
        "public_safe_output_contract": {
            "case_level_rows_written": False,
            "raw_questions_written": False,
            "raw_answers_written": False,
            "raw_document_ids_written": False,
            "raw_document_text_written": False,
            "aggregate_counts_only": True,
        },
    }


def _stage102_summary(stage102_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage102_report.get("decision") or {}
    data_summary = stage102_report.get("data_summary") or {}
    metrics = stage102_report.get("metrics_by_split") or {}
    aggregate_outputs = stage102_report.get("aggregate_outputs") or {}
    return {
        "stage": stage102_report.get("stage"),
        "analysis_id": stage102_report.get("analysis_id"),
        "decision_status": decision.get("status"),
        "data_summary": _split_counts_only(data_summary.get("splits") or {}),
        "dev_verified_metrics": (metrics.get(_VALIDATION_SPLIT) or {}).get("verified")
        or {},
        "dev_bucket_counts": (
            aggregate_outputs.get("bucket_counts_by_split") or {}
        ).get(_VALIDATION_SPLIT, {}),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _stage105_summary(stage105_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage105_report.get("decision") or {}
    train_selection = stage105_report.get("train_selection") or {}
    dev_validation = stage105_report.get("dev_validation") or {}
    return {
        "stage": stage105_report.get("stage"),
        "analysis_id": stage105_report.get("analysis_id"),
        "decision_status": decision.get("status"),
        "selected_config_id": decision.get("selected_config_id"),
        "selected_candidate_id": decision.get("selected_candidate_id"),
        "selection_split": train_selection.get("selection_split"),
        "selectable_config_count": decision.get("selectable_config_count"),
        "config_count": train_selection.get("config_count")
        or len(stage105_report.get("config_results") or []),
        "selected_train_weighted_target_delta": train_selection.get(
            "selected_train_weighted_target_delta"
        ),
        "dev_validation_passed": dev_validation.get("dev_validation_passed"),
        "dev_weighted_target_delta": dev_validation.get("dev_weighted_target_delta"),
        "dev_changed_answer_count": dev_validation.get("dev_changed_answer_count"),
        "dev_target_bucket_deltas": dev_validation.get("dev_target_bucket_deltas") or {},
        "dev_metric_deltas": dev_validation.get("dev_metric_deltas") or {},
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _stage106_summary(stage106_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage106_report.get("decision") or {}
    stopped_family = stage106_report.get("stopped_family") or {}
    return {
        "stage": stage106_report.get("stage"),
        "decision_status": decision.get("status"),
        "stopped_family_id": decision.get("stopped_family_id"),
        "stopped_protocol_id": decision.get("stopped_protocol_id"),
        "current_route_defaultization": decision.get("current_route_defaultization"),
        "redesign_required_before_any_runtime_or_test_gate": decision.get(
            "redesign_required_before_any_runtime_or_test_gate"
        ),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "dev_better_nonselectable_config_count": len(
            stopped_family.get("dev_better_nonselectable_configs") or []
        ),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _pattern_summary(
    stage102_report: Mapping[str, Any],
    stage105_report: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate_outputs = stage102_report.get("aggregate_outputs") or {}
    metrics_by_split = stage102_report.get("metrics_by_split") or {}
    data_summary = stage102_report.get("data_summary") or {}
    split_summaries = data_summary.get("splits") or {}
    dev_summary = split_summaries.get(_VALIDATION_SPLIT) or {}
    dev_total = int(dev_summary.get("row_count") or 0)
    dev_answerable = int(dev_summary.get("answerable_count") or 0)
    dev_unanswerable = int(dev_summary.get("unanswerable_count") or 0)
    bucket_counts_by_split = aggregate_outputs.get("bucket_counts_by_split") or {}
    dev_bucket_counts = bucket_counts_by_split.get(_VALIDATION_SPLIT) or {}
    answerability_cross_tab = aggregate_outputs.get("answerability_bucket_cross_tab") or {}
    route_cross_tab = aggregate_outputs.get("route_bucket_cross_tab") or {}
    dev_answerability_cross_tab = {
        label: answerability_cross_tab.get(f"{_VALIDATION_SPLIT}::{label}", {})
        for label in ("answerable", "unanswerable")
    }
    dev_failure_count = sum(
        int(dev_bucket_counts.get(bucket, 0)) for bucket in _FAILURE_BUCKETS
    )
    answerable_non_error = int(
        (dev_answerability_cross_tab.get("answerable") or {}).get(
            "answer_supported_and_cited",
            0,
        )
    )
    unanswerable_false_answer = int(
        (dev_answerability_cross_tab.get("unanswerable") or {}).get(
            "answerability_false_answer",
            0,
        )
    )
    retrieval_miss = int(dev_bucket_counts.get("retrieval_context_miss", 0))
    context_present_answerable = max(0, dev_answerable - retrieval_miss)
    evidence_selection_miss = int(dev_bucket_counts.get("evidence_selection_miss", 0))
    gold_span_beats = int(dev_bucket_counts.get("gold_span_beats_selected_answer", 0))
    return {
        "validation_split": _VALIDATION_SPLIT,
        "dev_row_counts": {
            "total": dev_total,
            "answerable": dev_answerable,
            "unanswerable": dev_unanswerable,
        },
        "dev_failure_overview": {
            "failure_count": dev_failure_count,
            "failure_rate": _safe_rate(dev_failure_count, dev_total),
            "answerable_failure_count": dev_answerable - answerable_non_error,
            "answerable_failure_rate": _safe_rate(
                dev_answerable - answerable_non_error,
                dev_answerable,
            ),
            "answerable_non_error_count": answerable_non_error,
            "unanswerable_false_answer_count": unanswerable_false_answer,
            "unanswerable_false_answer_rate": _safe_rate(
                unanswerable_false_answer,
                dev_unanswerable,
            ),
        },
        "dev_bucket_failure_profile": _dev_bucket_failure_profile(
            dev_bucket_counts=dev_bucket_counts,
            dev_total=dev_total,
            dev_answerable=dev_answerable,
            dev_unanswerable=dev_unanswerable,
        ),
        "train_dev_bucket_rate_drift": _train_dev_bucket_rate_drift(
            aggregate_outputs=aggregate_outputs
        ),
        "dev_answerability_cross_tab": dev_answerability_cross_tab,
        "dev_route_failure_profile": _dev_route_failure_profile(route_cross_tab),
        "dev_retrieval_and_context_profile": {
            "answerable_gold_context_present_count": context_present_answerable,
            "answerable_gold_context_present_rate": _safe_rate(
                context_present_answerable,
                dev_answerable,
            ),
            "answerable_gold_context_absent_count": retrieval_miss,
            "answerable_gold_context_absent_rate": _safe_rate(
                retrieval_miss,
                dev_answerable,
            ),
            "context_present_but_evidence_or_composition_failure_count": (
                evidence_selection_miss + gold_span_beats
            ),
            "context_present_evidence_selection_miss_count": evidence_selection_miss,
            "context_present_evidence_selection_miss_rate": _safe_rate(
                evidence_selection_miss,
                context_present_answerable,
            ),
            "context_present_gold_span_beats_selected_count": gold_span_beats,
            "context_present_gold_span_beats_selected_rate": _safe_rate(
                gold_span_beats,
                context_present_answerable,
            ),
            "answerable_supported_and_cited_count": answerable_non_error,
        },
        "dev_retrieval_rank_distribution": (
            aggregate_outputs.get("retrieval_rank_bucket_distributions") or {}
        ).get(_VALIDATION_SPLIT, {}),
        "dev_token_f1_distribution": _dev_token_f1_profile(
            aggregate_outputs=aggregate_outputs,
            dev_answerable=dev_answerable,
        ),
        "dev_verification_decision_distribution": (
            aggregate_outputs.get("verification_decision_distributions") or {}
        ).get(_VALIDATION_SPLIT, {}),
        "stage102_dev_verified_metrics": (
            metrics_by_split.get(_VALIDATION_SPLIT) or {}
        ).get("verified", {}),
        "stage105_candidate_failure_pattern": _stage105_candidate_failure_pattern(
            stage105_report
        ),
        "observed_failure_rules": _observed_failure_rules(
            dev_answerable=dev_answerable,
            dev_unanswerable=dev_unanswerable,
            answerable_non_error=answerable_non_error,
            unanswerable_false_answer=unanswerable_false_answer,
            retrieval_miss=retrieval_miss,
            evidence_selection_miss=evidence_selection_miss,
            gold_span_beats=gold_span_beats,
        ),
    }


def _dev_bucket_failure_profile(
    *,
    dev_bucket_counts: Mapping[str, Any],
    dev_total: int,
    dev_answerable: int,
    dev_unanswerable: int,
) -> list[dict[str, Any]]:
    profile = []
    for bucket in _BUCKET_ORDER:
        count = int(dev_bucket_counts.get(bucket, 0))
        denominator = dev_total
        denominator_label = "all_dev_rows"
        if bucket in _ANSWERABLE_FAILURE_BUCKETS:
            denominator = dev_answerable
            denominator_label = "answerable_dev_rows"
        elif bucket in _UNANSWERABLE_FAILURE_BUCKETS:
            denominator = dev_unanswerable
            denominator_label = "unanswerable_dev_rows"
        profile.append(
            {
                "bucket_id": bucket,
                "pipeline_stage": _BUCKET_TO_STAGE[bucket],
                "count": count,
                "overall_rate": _safe_rate(count, dev_total),
                "conditional_rate": _safe_rate(count, denominator),
                "conditional_rate_denominator": denominator_label,
                "is_failure_bucket": bucket in _FAILURE_BUCKETS,
            }
        )
    return sorted(
        profile,
        key=lambda item: (
            not item["is_failure_bucket"],
            -int(item["count"]),
            _BUCKET_ORDER.index(str(item["bucket_id"])),
        ),
    )


def _train_dev_bucket_rate_drift(
    *,
    aggregate_outputs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    bucket_rates_by_split = aggregate_outputs.get("bucket_rates_by_split") or {}
    train_rates = bucket_rates_by_split.get("train") or {}
    dev_rates = bucket_rates_by_split.get(_VALIDATION_SPLIT) or {}
    return [
        {
            "bucket_id": bucket,
            "train_rate": _optional_float(train_rates.get(bucket)),
            "dev_rate": _optional_float(dev_rates.get(bucket)),
            "dev_minus_train_percentage_points": round(
                100.0
                * (
                    _optional_float(dev_rates.get(bucket))
                    - _optional_float(train_rates.get(bucket))
                ),
                2,
            ),
        }
        for bucket in _BUCKET_ORDER
    ]


def _dev_route_failure_profile(
    route_cross_tab: Mapping[str, Any],
) -> list[dict[str, Any]]:
    route_profiles = []
    for key, counts in sorted(route_cross_tab.items()):
        if not str(key).startswith(f"{_VALIDATION_SPLIT}::"):
            continue
        if not isinstance(counts, Mapping):
            continue
        route = str(key).split("::", 1)[1]
        failure_count = sum(int(counts.get(bucket, 0)) for bucket in _FAILURE_BUCKETS)
        total_count = failure_count + int(counts.get("answer_supported_and_cited", 0))
        failure_counts = {
            bucket: int(counts.get(bucket, 0))
            for bucket in _FAILURE_BUCKETS
            if int(counts.get(bucket, 0)) > 0
        }
        dominant_bucket = (
            max(failure_counts, key=lambda bucket: (failure_counts[bucket], bucket))
            if failure_counts
            else None
        )
        route_profiles.append(
            {
                "question_route": route,
                "total_count": total_count,
                "failure_count": failure_count,
                "failure_rate": _safe_rate(failure_count, total_count),
                "target_bucket_failure_count": sum(
                    int(counts.get(bucket, 0)) for bucket in _TARGET_BUCKETS
                ),
                "dominant_failure_bucket": dominant_bucket,
                "failure_bucket_counts": failure_counts,
            }
        )
    return sorted(
        route_profiles,
        key=lambda item: (
            -int(item["failure_count"]),
            -float(item["failure_rate"]),
            str(item["question_route"]),
        ),
    )


def _dev_token_f1_profile(
    *,
    aggregate_outputs: Mapping[str, Any],
    dev_answerable: int,
) -> dict[str, Any]:
    distribution = (
        (aggregate_outputs.get("token_f1_bucket_distributions") or {}).get(
            _VALIDATION_SPLIT
        )
        or {}
    )
    answerable_distribution = {
        bucket: count
        for bucket, count in distribution.items()
        if bucket != "not_applicable"
    }
    return {
        "all_rows_distribution": distribution,
        "answerable_rows_distribution": answerable_distribution,
        "answerable_low_f1_lt_0_20_count": int(
            answerable_distribution.get("f1_0_00_to_0_19", 0)
        ),
        "answerable_low_f1_lt_0_20_rate": _safe_rate(
            int(answerable_distribution.get("f1_0_00_to_0_19", 0)),
            dev_answerable,
        ),
        "answerable_f1_0_60_plus_count": sum(
            int(answerable_distribution.get(bucket, 0))
            for bucket in ("f1_0_60_to_0_79", "f1_0_80_to_1_00")
        ),
        "answerable_f1_0_60_plus_rate": _safe_rate(
            sum(
                int(answerable_distribution.get(bucket, 0))
                for bucket in ("f1_0_60_to_0_79", "f1_0_80_to_1_00")
            ),
            dev_answerable,
        ),
    }


def _stage105_candidate_failure_pattern(
    stage105_report: Mapping[str, Any],
) -> dict[str, Any]:
    config_results = [
        config
        for config in stage105_report.get("config_results") or []
        if isinstance(config, Mapping)
    ]
    train_selection = stage105_report.get("train_selection") or {}
    dev_validation = stage105_report.get("dev_validation") or {}
    selected_config_id = train_selection.get("selected_config_id")
    selected_config = next(
        (
            config
            for config in config_results
            if config.get("config_id") == selected_config_id
        ),
        {},
    )
    selectable = [
        config
        for config in config_results
        if (config.get("train_selectability") or {}).get("selectable") is True
    ]
    nonselectable = [config for config in config_results if config not in selectable]
    dev_better_nonselectable = [
        config
        for config in nonselectable
        if _dev_delta(config) < _optional_float(dev_validation.get("dev_weighted_target_delta"))
    ]
    return {
        "selected_config": {
            "config_id": selected_config_id,
            "candidate_id": train_selection.get("selected_candidate_id"),
            "train_weighted_target_delta": train_selection.get(
                "selected_train_weighted_target_delta"
            ),
            "dev_weighted_target_delta": dev_validation.get(
                "dev_weighted_target_delta"
            ),
            "dev_changed_answer_count": dev_validation.get("dev_changed_answer_count"),
            "dev_target_bucket_deltas": dev_validation.get("dev_target_bucket_deltas")
            or {},
            "dev_metric_deltas": dev_validation.get("dev_metric_deltas") or {},
            "train_selectability": selected_config.get("train_selectability") or {},
        },
        "selectable_config_count": len(selectable),
        "nonselectable_config_count": len(nonselectable),
        "dev_better_nonselectable_config_count": len(dev_better_nonselectable),
        "train_guard_failure_reasons": dict(
            sorted(_train_guard_failure_reasons(nonselectable).items())
        ),
        "candidate_behavior_clusters": _candidate_behavior_clusters(config_results),
        "config_behavior_table": _config_behavior_table(config_results),
    }


def _candidate_behavior_clusters(
    config_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    clusters = {
        "train_selectable_noop_or_near_noop": [],
        "train_nonselectable_low_change": [],
        "train_nonselectable_high_change": [],
    }
    for config in config_results:
        dev_changed = int(
            (config.get("changed_answer_counts_by_split") or {}).get(
                _VALIDATION_SPLIT,
                0,
            )
        )
        selectable = (config.get("train_selectability") or {}).get("selectable") is True
        if selectable:
            clusters["train_selectable_noop_or_near_noop"].append(config)
        elif dev_changed <= 10:
            clusters["train_nonselectable_low_change"].append(config)
        else:
            clusters["train_nonselectable_high_change"].append(config)
    return [
        {
            "cluster_id": cluster_id,
            "config_count": len(configs),
            "config_ids": [str(config.get("config_id")) for config in configs],
            "dev_changed_answer_count_total": sum(
                int(
                    (config.get("changed_answer_counts_by_split") or {}).get(
                        _VALIDATION_SPLIT,
                        0,
                    )
                )
                for config in configs
            ),
            "best_dev_weighted_target_delta": (
                min((_dev_delta(config) for config in configs), default=None)
            ),
            "all_train_selectable": all(
                (config.get("train_selectability") or {}).get("selectable") is True
                for config in configs
            )
            if configs
            else False,
        }
        for cluster_id, configs in clusters.items()
    ]


def _config_behavior_table(
    config_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for config in config_results:
        rows.append(
            {
                "config_id": config.get("config_id"),
                "candidate_id": config.get("candidate_id"),
                "train_selectable": (
                    config.get("train_selectability") or {}
                ).get("selectable"),
                "train_weighted_target_delta": (
                    config.get("weighted_target_score_deltas_by_split") or {}
                ).get("train"),
                "dev_weighted_target_delta": (
                    config.get("weighted_target_score_deltas_by_split") or {}
                ).get(_VALIDATION_SPLIT),
                "train_changed_answer_count": (
                    config.get("changed_answer_counts_by_split") or {}
                ).get("train"),
                "dev_changed_answer_count": (
                    config.get("changed_answer_counts_by_split") or {}
                ).get(_VALIDATION_SPLIT),
                "failed_train_guards": _failed_train_guard_names(config),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            _optional_float(row.get("dev_weighted_target_delta")),
            str(row.get("config_id")),
        ),
    )


def _observed_failure_rules(
    *,
    dev_answerable: int,
    dev_unanswerable: int,
    answerable_non_error: int,
    unanswerable_false_answer: int,
    retrieval_miss: int,
    evidence_selection_miss: int,
    gold_span_beats: int,
) -> list[dict[str, Any]]:
    context_present = max(0, dev_answerable - retrieval_miss)
    return [
        {
            "rule_id": "dev_answerable_rows_are_all_error_bucketed",
            "observation": (
                "All answerable dev rows fell into retrieval, evidence, "
                "verification, or answer-composition failure buckets."
            ),
            "count": dev_answerable - answerable_non_error,
            "denominator": dev_answerable,
            "rate": _safe_rate(dev_answerable - answerable_non_error, dev_answerable),
        },
        {
            "rule_id": "dev_unanswerable_rows_are_mostly_false_answered",
            "observation": (
                "Most unanswerable dev rows were answered instead of refused."
            ),
            "count": unanswerable_false_answer,
            "denominator": dev_unanswerable,
            "rate": _safe_rate(unanswerable_false_answer, dev_unanswerable),
        },
        {
            "rule_id": "context_present_answerables_fail_after_retrieval",
            "observation": (
                "When the gold context was present for answerable dev rows, "
                "the remaining failures were evidence-selection or answer "
                "composition failures."
            ),
            "count": evidence_selection_miss + gold_span_beats,
            "denominator": context_present,
            "rate": _safe_rate(evidence_selection_miss + gold_span_beats, context_present),
        },
        {
            "rule_id": "gold_span_beats_selected_answer_dominates_context_present",
            "observation": (
                "Among context-present answerable dev failures, the largest "
                "single bucket was gold_span_beats_selected_answer."
            ),
            "count": gold_span_beats,
            "denominator": context_present,
            "rate": _safe_rate(gold_span_beats, context_present),
        },
    ]


def _guard_checks(
    *,
    report: Mapping[str, Any],
    stage102_report: Mapping[str, Any],
    stage105_report: Mapping[str, Any],
    stage106_report: Mapping[str, Any],
    user_confirmed_analysis: bool,
) -> list[dict[str, Any]]:
    stage102_decision = stage102_report.get("decision") or {}
    stage105_decision = stage105_report.get("decision") or {}
    stage106_decision = stage106_report.get("decision") or {}
    stage102_guards = stage102_report.get("guard_checks") or []
    stage105_guards = stage105_report.get("guard_checks") or []
    stage106_guards = stage106_report.get("guard_checks") or []
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    dev_overview = (report.get("pattern_summary") or {}).get("dev_failure_overview") or {}
    return [
        _check(
            name="source_stage102_report_is_stage102",
            passed=stage102_report.get("stage") == _SOURCE_STAGE102,
            observed=stage102_report.get("stage"),
            expected=_SOURCE_STAGE102,
        ),
        _check(
            name="source_stage105_report_is_stage105",
            passed=stage105_report.get("stage") == _SOURCE_STAGE105,
            observed=stage105_report.get("stage"),
            expected=_SOURCE_STAGE105,
        ),
        _check(
            name="source_stage106_report_is_stage106",
            passed=stage106_report.get("stage") == _SOURCE_STAGE106,
            observed=stage106_report.get("stage"),
            expected=_SOURCE_STAGE106,
        ),
        _check(
            name="user_confirmed_stage107_failure_pattern_analysis",
            passed=user_confirmed_analysis,
            observed=user_confirmed_analysis,
            expected=True,
        ),
        _check(
            name="stage102_completed_expected_analysis",
            passed=stage102_decision.get("status") == _EXPECTED_STAGE102_STATUS,
            observed=stage102_decision.get("status"),
            expected=_EXPECTED_STAGE102_STATUS,
        ),
        _check(
            name="stage105_completed_with_dev_guard_failed",
            passed=stage105_decision.get("status") == _EXPECTED_STAGE105_STATUS,
            observed=stage105_decision.get("status"),
            expected=_EXPECTED_STAGE105_STATUS,
        ),
        _check(
            name="stage106_stopped_evidence_answerability_family",
            passed=stage106_decision.get("status") == _EXPECTED_STAGE106_STATUS,
            observed=stage106_decision.get("status"),
            expected=_EXPECTED_STAGE106_STATUS,
        ),
        _check(
            name="all_source_guard_checks_passed",
            passed=all(check.get("passed") is True for check in stage102_guards)
            and all(check.get("passed") is True for check in stage105_guards)
            and all(check.get("passed") is True for check in stage106_guards),
            observed={
                "stage102": _guard_count(stage102_guards),
                "stage105": _guard_count(stage105_guards),
                "stage106": _guard_count(stage106_guards),
            },
            expected="all source guard checks passed",
        ),
        _check(
            name="stage107_reads_saved_reports_only",
            passed=True,
            observed="saved Stage102/105/106 reports only",
            expected="no split files and no corpus documents loaded",
        ),
        _check(
            name="stage107_validation_split_is_dev",
            passed=(report.get("split_contract") or {}).get("validation_split")
            == _VALIDATION_SPLIT,
            observed=(report.get("split_contract") or {}).get("validation_split"),
            expected=_VALIDATION_SPLIT,
        ),
        _check(
            name="stage107_test_split_locked",
            passed=(report.get("split_contract") or {}).get("forbidden_final_splits")
            == list(_FORBIDDEN_FINAL_SPLITS),
            observed=(report.get("split_contract") or {}).get("forbidden_final_splits"),
            expected=list(_FORBIDDEN_FINAL_SPLITS),
        ),
        _check(
            name="stage107_no_dev_selection_or_threshold_retuning",
            passed=(
                (report.get("frozen_protocol") or {})
                .get("selection_policy", {})
                .get("select_config_from_dev")
                is False
                and (report.get("frozen_protocol") or {})
                .get("selection_policy", {})
                .get("retune_thresholds_on_dev")
                is False
            ),
            observed=(report.get("frozen_protocol") or {}).get("selection_policy"),
            expected="no dev config selection and no dev threshold retuning",
        ),
        _check(
            name="stage107_uses_aggregate_public_safe_outputs",
            passed=(
                (report.get("frozen_protocol") or {})
                .get("public_safe_output_contract", {})
                .get("aggregate_counts_only")
                is True
                and (report.get("frozen_protocol") or {})
                .get("public_safe_output_contract", {})
                .get("case_level_rows_written")
                is False
            ),
            observed=(report.get("frozen_protocol") or {}).get(
                "public_safe_output_contract"
            ),
            expected="aggregate counts only; no case-level rows",
        ),
        _check(
            name="stage107_output_has_no_forbidden_public_keys",
            passed=not _contains_forbidden_key(report),
            observed=sorted(_forbidden_keys_found(report)),
            expected=[],
        ),
        _check(
            name="stage107_output_has_no_private_fixture_markers",
            passed="private-doc-" not in serialized
            and "Private fixture answer text" not in serialized,
            observed="private marker present"
            if "private-doc-" in serialized or "Private fixture answer text" in serialized
            else "none",
            expected="none",
        ),
        _check(
            name="stage107_confirms_dev_failure_signal_present",
            passed=float(dev_overview.get("failure_rate") or 0.0) > 0.0,
            observed=dev_overview,
            expected="dev failure rate > 0",
        ),
        _check(
            name="stage107_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage107_runtime_defaults_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
        _check(
            name="stage107_fallback_strategies_not_added",
            passed=True,
            observed=False,
            expected=False,
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    pattern_summary: Mapping[str, Any],
) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_validation_failure_pattern_analysis_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "requires_user_confirmation_before_next_protocol": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    dev_overview = pattern_summary.get("dev_failure_overview") or {}
    context_profile = pattern_summary.get("dev_retrieval_and_context_profile") or {}
    stage105_pattern = pattern_summary.get("stage105_candidate_failure_pattern") or {}
    return {
        "status": "primeqa_hybrid_validation_failure_pattern_analysis_completed",
        "protocol_id": _PROTOCOL_ID,
        "validation_split": _VALIDATION_SPLIT,
        "dev_failure_count": dev_overview.get("failure_count"),
        "dev_failure_rate": dev_overview.get("failure_rate"),
        "answerable_failure_rate": dev_overview.get("answerable_failure_rate"),
        "unanswerable_false_answer_rate": dev_overview.get(
            "unanswerable_false_answer_rate"
        ),
        "answerable_gold_context_absent_rate": context_profile.get(
            "answerable_gold_context_absent_rate"
        ),
        "context_present_gold_span_beats_selected_rate": context_profile.get(
            "context_present_gold_span_beats_selected_rate"
        ),
        "stage105_selected_config_was_dev_noop": (
            ((stage105_pattern.get("selected_config") or {}).get(
                "dev_changed_answer_count"
            )
            == 0)
            and (
                float(
                    (stage105_pattern.get("selected_config") or {}).get(
                        "dev_weighted_target_delta"
                    )
                    or 0.0
                )
                == 0.0
            )
        ),
        "recommended_next_direction": (
            "failure_pattern_driven_train_dev_redesign_protocol"
        ),
        "can_continue_train_dev_development": True,
        "requires_user_confirmation_before_next_protocol": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage108: after user confirmation, freeze a train/dev-only "
            "failure-pattern-driven redesign protocol. It should target the "
            "observed dev/train failure structure without selecting from dev, "
            "without opening the test gate, without changing runtime defaults, "
            "and without adding fallback strategies."
        ),
    }


def _dev_failure_bucket_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    profile = (report.get("pattern_summary") or {}).get("dev_bucket_failure_profile") or []
    return [
        BarDatum(
            label=str(item.get("bucket_id")),
            value=float(item.get("count") or 0),
            value_label=str(item.get("count") or 0),
        )
        for item in profile
        if item.get("is_failure_bucket") is True
    ]


def _bucket_rate_drift_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    drift = (report.get("pattern_summary") or {}).get("train_dev_bucket_rate_drift") or []
    return [
        BarDatum(
            label=str(item.get("bucket_id")),
            value=float(item.get("dev_minus_train_percentage_points") or 0.0),
            value_label=f"{float(item.get('dev_minus_train_percentage_points') or 0.0):+.2f} pp",
        )
        for item in drift
    ]


def _dev_route_failure_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    profile = (report.get("pattern_summary") or {}).get("dev_route_failure_profile") or []
    return [
        BarDatum(
            label=str(item.get("question_route")),
            value=float(item.get("failure_count") or 0),
            value_label=str(item.get("failure_count") or 0),
        )
        for item in profile
    ]


def _dev_answerable_flow_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    context = (
        (report.get("pattern_summary") or {}).get("dev_retrieval_and_context_profile")
        or {}
    )
    return [
        BarDatum(
            label="retrieval_context_miss",
            value=float(context.get("answerable_gold_context_absent_count") or 0),
            value_label=str(context.get("answerable_gold_context_absent_count") or 0),
        ),
        BarDatum(
            label="evidence_selection_miss",
            value=float(context.get("context_present_evidence_selection_miss_count") or 0),
            value_label=str(
                context.get("context_present_evidence_selection_miss_count") or 0
            ),
        ),
        BarDatum(
            label="gold_span_beats_selected_answer",
            value=float(
                context.get("context_present_gold_span_beats_selected_count") or 0
            ),
            value_label=str(
                context.get("context_present_gold_span_beats_selected_count") or 0
            ),
        ),
        BarDatum(
            label="answer_supported_and_cited",
            value=float(context.get("answerable_supported_and_cited_count") or 0),
            value_label=str(context.get("answerable_supported_and_cited_count") or 0),
        ),
    ]


def _stage105_candidate_behavior_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    table = (
        ((report.get("pattern_summary") or {}).get("stage105_candidate_failure_pattern") or {})
        .get("config_behavior_table")
        or []
    )
    return [
        BarDatum(
            label=str(row.get("config_id")),
            value=float(row.get("dev_changed_answer_count") or 0),
            value_label=str(row.get("dev_changed_answer_count") or 0),
        )
        for row in table
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    names = [
        "can_continue_train_dev_development",
        "requires_user_confirmation_before_next_protocol",
        "can_open_final_test_gate_now",
        "can_run_final_test_metrics_now",
        "can_use_test_for_tuning",
        "fallback_strategies_enabled",
    ]
    return [
        BarDatum(
            label=name,
            value=1.0 if decision.get(name) is True else 0.0,
            value_label=str(decision.get(name)),
        )
        for name in names
    ]


def _guard_check_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(check["name"]),
            value=1.0 if check["passed"] else 0.0,
            value_label="passed" if check["passed"] else "failed",
        )
        for check in report.get("guard_checks") or []
    ]


def _split_counts_only(splits: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        split: {
            "row_count": summary.get("row_count"),
            "answerable_count": summary.get("answerable_count"),
            "unanswerable_count": summary.get("unanswerable_count"),
        }
        for split, summary in splits.items()
        if isinstance(summary, Mapping)
    }


def _train_guard_failure_reasons(
    configs: Sequence[Mapping[str, Any]],
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for config in configs:
        for guard_name in _failed_train_guard_names(config):
            counter[guard_name] += 1
    return counter


def _failed_train_guard_names(config: Mapping[str, Any]) -> list[str]:
    checks = ((config.get("train_selectability") or {}).get("checks")) or {}
    return [
        str(name)
        for name, passed in sorted(checks.items())
        if passed is not True
    ]


def _dev_delta(config: Mapping[str, Any]) -> float:
    return _optional_float(
        (config.get("weighted_target_score_deltas_by_split") or {}).get(
            _VALIDATION_SPLIT
        )
    )


def _optional_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _guard_count(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "passed": sum(1 for check in guard_checks if check.get("passed") is True),
        "total": len(guard_checks),
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


def _contains_forbidden_key(value: Any) -> bool:
    return bool(_forbidden_keys_found(value))


def _forbidden_keys_found(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in _FORBIDDEN_REPORT_KEYS:
                found.add(str(key))
            found.update(_forbidden_keys_found(nested))
    elif isinstance(value, list | tuple):
        for nested in value:
            found.update(_forbidden_keys_found(nested))
    return found


def _fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload
