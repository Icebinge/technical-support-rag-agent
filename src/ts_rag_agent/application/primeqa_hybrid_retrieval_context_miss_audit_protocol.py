from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 111"
_CREATED_AT = "2026-07-16"
_PROTOCOL_ID = "primeqa_hybrid_retrieval_context_miss_audit_protocol_v1"
_ROUTE_ID = "retrieval_context_miss_root_cause_audit_protocol"
_NEXT_DIRECTION = "run_retrieval_context_miss_root_cause_audit_train_dev"
_SOURCE_STAGE102 = "Stage 102"
_SOURCE_STAGE107 = "Stage 107"
_SOURCE_STAGE110 = "Stage 110"
_STAGE102_ANALYSIS_ID = "answer_pipeline_error_decomposition_train_dev_analysis_v1"
_STAGE107_PROTOCOL_ID = "primeqa_hybrid_validation_failure_pattern_analysis_v1"
_STAGE110_STOP_STATUS = "primeqa_hybrid_failure_pattern_redesign_family_stopped"
_STAGE110_NEXT_DIRECTION = "user_confirmed_next_research_direction_required"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = ("test",)
_AUDIT_DIMENSIONS = (
    "query_expression_gap",
    "title_heading_mismatch",
    "section_boundary_or_span_locality",
    "long_document_score_dilution",
    "entity_version_error_code_mismatch",
    "bm25_field_weighting_or_index_structure",
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
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
class PrimeQAHybridRetrievalContextMissAuditProtocolVisualization:
    """One generated Stage111 retrieval-context-miss audit protocol chart."""

    name: str
    path: str


def freeze_primeqa_hybrid_retrieval_context_miss_audit_protocol(
    *,
    stage102_report_path: Path,
    stage107_report_path: Path,
    stage110_report_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage111 train/dev-only retrieval miss root-cause audit protocol."""

    started_at = time.perf_counter()
    stage102_report = _load_json_object(stage102_report_path)
    stage107_report = _load_json_object(stage107_report_path)
    stage110_report = _load_json_object(stage110_report_path)
    loaded_at = time.perf_counter()

    stage102_summary = _stage102_summary(stage102_report)
    stage107_summary = _stage107_summary(stage107_report)
    stage110_summary = _stage110_summary(stage110_report)
    frozen_protocol = _frozen_protocol(
        stage102_summary=stage102_summary,
        stage107_summary=stage107_summary,
        stage110_summary=stage110_summary,
    )
    preliminary_report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "route_id": _ROUTE_ID,
        "protocol_scope": (
            "Train/dev-only protocol freeze for a retrieval-context-miss "
            "root-cause audit after the Stage108/109/110 answer-pipeline "
            "redesign family stopped. This stage reads only saved public-safe "
            "Stage102, Stage107, and Stage110 reports; does not load split "
            "files; does not load corpus documents; does not run retrieval or "
            "answer metrics; does not run final metrics; does not select from "
            "dev-only observations; does not add fallback strategies; and does "
            "not change runtime defaults."
        ),
        "user_confirmation": {
            "route_id": _ROUTE_ID,
            "confirmed": bool(user_confirmed_protocol),
            "confirmation_note": confirmation_note,
        },
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "development_splits": list(_DEVELOPMENT_SPLITS),
            "analysis_split_policy": "train_and_dev_reported_separately",
            "selection_split": None,
            "validation_split": None,
            "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage102_report": _fingerprint(stage102_report_path),
            "stage107_report": _fingerprint(stage107_report_path),
            "stage110_report": _fingerprint(stage110_report_path),
        },
        "stage102_summary": stage102_summary,
        "stage107_summary": stage107_summary,
        "stage110_summary": stage110_summary,
        "frozen_protocol": frozen_protocol,
    }
    guard_checks = _guard_checks(
        report=preliminary_report,
        stage102_summary=stage102_summary,
        stage107_summary=stage107_summary,
        stage110_summary=stage110_summary,
        user_confirmed_protocol=user_confirmed_protocol,
    )
    checked_at = time.perf_counter()
    return {
        **preliminary_report,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
        "timing_seconds": {
            "load_reports": round(loaded_at - started_at, 3),
            "freeze_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_retrieval_context_miss_audit_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridRetrievalContextMissAuditProtocolVisualization]:
    """Write SVG charts for Stage111 retrieval miss audit protocol."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage111_retrieval_context_miss_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage111 retrieval-context-miss counts",
            bars=_retrieval_miss_count_bars(report),
            x_label="retrieval_context_miss count",
            width=1180,
            margin_left=520,
        ),
        "stage111_audit_dimension_priorities.svg": render_horizontal_bar_chart_svg(
            title="Stage111 audit dimension priorities",
            bars=_audit_dimension_priority_bars(report),
            x_label="priority score",
            width=1440,
            margin_left=740,
        ),
        "stage111_stage112_data_access_contract.svg": render_horizontal_bar_chart_svg(
            title="Stage111 Stage112 data access contract",
            bars=_stage112_access_contract_bars(report),
            x_label="1 means allowed",
            width=1320,
            margin_left=660,
        ),
        "stage111_protocol_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage111 protocol decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1320,
            margin_left=660,
        ),
        "stage111_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage111 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1640,
            margin_left=860,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridRetrievalContextMissAuditProtocolVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage102_summary(stage102_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage102_report.get("decision") or {}
    aggregate = stage102_report.get("aggregate_outputs") or {}
    bucket_counts = aggregate.get("bucket_counts_by_split") or {}
    metrics = stage102_report.get("metrics_by_split") or {}
    train_counts = bucket_counts.get("train") or {}
    dev_counts = bucket_counts.get("dev") or {}
    return {
        "stage": stage102_report.get("stage"),
        "analysis_id": stage102_report.get("analysis_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "train_retrieval_context_miss_count": train_counts.get(
            "retrieval_context_miss"
        ),
        "dev_retrieval_context_miss_count": dev_counts.get("retrieval_context_miss"),
        "train_answerability_false_answer_count": train_counts.get(
            "answerability_false_answer"
        ),
        "dev_answerability_false_answer_count": dev_counts.get(
            "answerability_false_answer"
        ),
        "train_gold_span_beats_selected_count": train_counts.get(
            "gold_span_beats_selected_answer"
        ),
        "dev_gold_span_beats_selected_count": dev_counts.get(
            "gold_span_beats_selected_answer"
        ),
        "train_verified_average_token_f1": (
            ((metrics.get("train") or {}).get("verified") or {}).get(
                "average_token_f1"
            )
        ),
        "dev_verified_average_token_f1": (
            ((metrics.get("dev") or {}).get("verified") or {}).get("average_token_f1")
        ),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get(
            "can_run_final_test_metrics_now"
        ),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _stage107_summary(stage107_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage107_report.get("decision") or {}
    pattern = stage107_report.get("pattern_summary") or {}
    context = pattern.get("dev_retrieval_and_context_profile") or {}
    overview = pattern.get("dev_failure_overview") or {}
    return {
        "stage": stage107_report.get("stage"),
        "protocol_id": stage107_report.get("protocol_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "dev_failure_count": overview.get("failure_count"),
        "dev_failure_rate": overview.get("failure_rate"),
        "answerable_failure_rate": overview.get("answerable_failure_rate"),
        "answerable_gold_context_absent_count": context.get(
            "answerable_gold_context_absent_count"
        ),
        "answerable_gold_context_absent_rate": context.get(
            "answerable_gold_context_absent_rate"
        ),
        "answerable_gold_context_present_count": context.get(
            "answerable_gold_context_present_count"
        ),
        "context_present_failure_count": context.get(
            "context_present_but_evidence_or_composition_failure_count"
        ),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get(
            "can_run_final_test_metrics_now"
        ),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _stage110_summary(stage110_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage110_report.get("decision") or {}
    stopped_family = stage110_report.get("stopped_family") or {}
    return {
        "stage": stage110_report.get("stage"),
        "decision_status": decision.get("status"),
        "stopped_family_id": decision.get("stopped_family_id"),
        "stopped_protocol_id": decision.get("stopped_protocol_id"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "requires_user_confirmation_before_next_protocol": decision.get(
            "requires_user_confirmation_before_next_protocol"
        ),
        "stage109_selectable_config_count": (
            (stopped_family.get("stage109_summary") or {}).get(
                "selectable_config_count"
            )
        ),
        "stage109_config_count": (
            (stopped_family.get("stage109_summary") or {}).get("config_count")
        ),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get(
            "can_run_final_test_metrics_now"
        ),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _frozen_protocol(
    *,
    stage102_summary: Mapping[str, Any],
    stage107_summary: Mapping[str, Any],
    stage110_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "protocol_status": "frozen_requires_user_confirmation_before_train_dev_audit",
        "source_stages": [_SOURCE_STAGE102, _SOURCE_STAGE107, _SOURCE_STAGE110],
        "route_id": _ROUTE_ID,
        "audit_mode": "protocol_freeze_only_no_metrics",
        "objective": (
            "Diagnose why answerable PrimeQA hybrid rows become "
            "retrieval_context_miss under the Stage102 verified BM25 top10 "
            "baseline before proposing another retrieval or index redesign."
        ),
        "motivation": {
            "stage102_train_retrieval_context_miss_count": stage102_summary.get(
                "train_retrieval_context_miss_count"
            ),
            "stage102_dev_retrieval_context_miss_count": stage102_summary.get(
                "dev_retrieval_context_miss_count"
            ),
            "stage107_answerable_gold_context_absent_count": stage107_summary.get(
                "answerable_gold_context_absent_count"
            ),
            "stage107_answerable_gold_context_absent_rate": stage107_summary.get(
                "answerable_gold_context_absent_rate"
            ),
            "stage110_stopped_family_id": stage110_summary.get("stopped_family_id"),
        },
        "audit_dimensions": _audit_dimensions(),
        "stage112_run_contract": _stage112_run_contract(),
        "public_safe_output_contract": _public_safe_output_contract(),
        "explicit_exclusions": [
            "no_split_loading_in_stage111",
            "no_corpus_document_loading_in_stage111",
            "no_metric_run_in_stage111",
            "no_test_split_loading",
            "no_final_test_metrics",
            "no_dev_selection",
            "no_dev_threshold_tuning",
            "no_runtime_default_change",
            "no_fallback_strategy",
            "no_raw_question_answer_or_document_text_in_outputs",
            "no_runtime_use_of_gold_document_identifiers",
        ],
        "fallback_strategy_policy": {
            "fallback_strategies_enabled": False,
            "requires_user_confirmation_before_any_fallback": True,
        },
        "next_stage_contract": {
            "stage": "Stage 112",
            "recommended_direction": _NEXT_DIRECTION,
            "requires_user_confirmation_before_train_dev_audit": True,
            "source_protocol_id": _PROTOCOL_ID,
            "audit_train_and_dev_separately": True,
            "must_not_load_or_score_test": True,
            "must_not_change_runtime_defaults": True,
            "must_not_add_fallback_strategies": True,
        },
    }


def _audit_dimensions() -> list[dict[str, Any]]:
    return [
        {
            "dimension_id": "query_expression_gap",
            "priority_score": 0.95,
            "question": (
                "Do missed questions use terms that poorly overlap the gold "
                "document section text under lexical BM25 retrieval?"
            ),
            "audit_signals": [
                "question_to_gold_section_token_overlap_bucket",
                "rare_query_token_coverage_bucket",
                "query_length_bucket",
            ],
            "allowed_only_for_offline_audit": True,
        },
        {
            "dimension_id": "title_heading_mismatch",
            "priority_score": 0.85,
            "question": (
                "Does the gold document title or section heading fail to match "
                "the user's title/body vocabulary?"
            ),
            "audit_signals": [
                "question_to_gold_title_overlap_bucket",
                "question_to_gold_heading_overlap_bucket",
            ],
            "allowed_only_for_offline_audit": True,
        },
        {
            "dimension_id": "section_boundary_or_span_locality",
            "priority_score": 0.80,
            "question": (
                "Is the gold answer span hard to retrieve because it is isolated "
                "inside a section boundary or far from section-level lexical anchors?"
            ),
            "audit_signals": [
                "gold_span_sentence_position_bucket",
                "gold_section_length_bucket",
                "answer_span_anchor_density_bucket",
            ],
            "allowed_only_for_offline_audit": True,
        },
        {
            "dimension_id": "long_document_score_dilution",
            "priority_score": 0.75,
            "question": (
                "Are long documents or long sections diluting BM25 scores for "
                "the relevant gold context?"
            ),
            "audit_signals": [
                "gold_document_length_bucket",
                "gold_section_length_bucket",
                "retrieved_top10_length_skew_bucket",
            ],
            "allowed_only_for_offline_audit": True,
        },
        {
            "dimension_id": "entity_version_error_code_mismatch",
            "priority_score": 0.70,
            "question": (
                "Are product names, versions, APARs, error codes, or CVE-like "
                "tokens missing or mismatched between the query and gold context?"
            ),
            "audit_signals": [
                "entity_token_overlap_bucket",
                "version_token_overlap_bucket",
                "error_code_token_overlap_bucket",
            ],
            "allowed_only_for_offline_audit": True,
        },
        {
            "dimension_id": "bm25_field_weighting_or_index_structure",
            "priority_score": 0.65,
            "question": (
                "Does the current BM25 field/index structure underweight titles, "
                "headings, or local sections relative to full document text?"
            ),
            "audit_signals": [
                "gold_doc_rank_at_50_bucket",
                "title_only_rank_bucket",
                "section_only_rank_bucket",
                "doc_rollup_vs_section_rank_gap_bucket",
            ],
            "allowed_only_for_offline_audit": True,
        },
    ]


def _stage112_run_contract() -> dict[str, Any]:
    return {
        "allowed_inputs_after_user_confirmation": [
            "Stage111 frozen protocol",
            "Stage68 train split",
            "Stage68 dev split",
            "PrimeQA training/dev corpus sections",
            "Stage102 public-safe report for baseline bucket targets",
        ],
        "forbidden_inputs": [
            "test split",
            "final-test labels",
            "runtime oracle document identifiers",
            "raw question, answer, or document text in public outputs",
        ],
        "audit_population": (
            "answerable train/dev rows classified as retrieval_context_miss "
            "under the Stage102 verified BM25 top10 baseline"
        ),
        "retrieval_depth_for_diagnostic_only": 50,
        "stage112_may_use_gold_doc_id_for_offline_labeling": True,
        "gold_doc_id_allowed_as_runtime_feature": False,
        "reported_splits": ["train", "dev"],
        "selection_or_threshold_tuning_allowed": False,
        "candidate_defaultization_allowed": False,
        "final_test_metrics_allowed": False,
    }


def _public_safe_output_contract() -> dict[str, Any]:
    return {
        "aggregate_only_by_default": True,
        "allowed_case_fields": [
            "sample_id",
            "split",
            "retrieval_context_miss_root_cause_bucket",
            "question_route",
            "gold_doc_rank_bucket",
            "query_expression_gap_bucket",
            "title_heading_overlap_bucket",
            "section_locality_bucket",
            "document_length_bucket",
            "entity_version_error_code_bucket",
            "index_structure_signal_bucket",
            "confidence_band",
        ],
        "forbidden_fields": sorted(_FORBIDDEN_PUBLIC_KEYS),
        "raw_group_values_written": False,
    }


def _guard_checks(
    *,
    report: Mapping[str, Any],
    stage102_summary: Mapping[str, Any],
    stage107_summary: Mapping[str, Any],
    stage110_summary: Mapping[str, Any],
    user_confirmed_protocol: bool,
) -> list[dict[str, Any]]:
    frozen = report.get("frozen_protocol") or {}
    stage112_contract = frozen.get("stage112_run_contract") or {}
    public_contract = frozen.get("public_safe_output_contract") or {}
    forbidden_keys_found = _forbidden_keys_found(report)
    return [
        _check(
            name="source_stage102_is_expected",
            passed=stage102_summary.get("stage") == _SOURCE_STAGE102,
            observed=stage102_summary.get("stage"),
            expected=_SOURCE_STAGE102,
        ),
        _check(
            name="source_stage102_analysis_id_matches",
            passed=stage102_summary.get("analysis_id") == _STAGE102_ANALYSIS_ID,
            observed=stage102_summary.get("analysis_id"),
            expected=_STAGE102_ANALYSIS_ID,
        ),
        _check(
            name="stage102_has_train_dev_retrieval_context_misses",
            passed=int(stage102_summary.get("train_retrieval_context_miss_count") or 0)
            > 0
            and int(stage102_summary.get("dev_retrieval_context_miss_count") or 0)
            > 0,
            observed={
                "train": stage102_summary.get("train_retrieval_context_miss_count"),
                "dev": stage102_summary.get("dev_retrieval_context_miss_count"),
            },
            expected="positive train and dev retrieval_context_miss counts",
        ),
        _check(
            name="source_stage107_is_expected",
            passed=stage107_summary.get("stage") == _SOURCE_STAGE107,
            observed=stage107_summary.get("stage"),
            expected=_SOURCE_STAGE107,
        ),
        _check(
            name="source_stage107_protocol_id_matches",
            passed=stage107_summary.get("protocol_id") == _STAGE107_PROTOCOL_ID,
            observed=stage107_summary.get("protocol_id"),
            expected=_STAGE107_PROTOCOL_ID,
        ),
        _check(
            name="stage107_confirms_dev_gold_context_absent",
            passed=int(stage107_summary.get("answerable_gold_context_absent_count") or 0)
            > 0,
            observed={
                "count": stage107_summary.get("answerable_gold_context_absent_count"),
                "rate": stage107_summary.get("answerable_gold_context_absent_rate"),
            },
            expected="positive dev answerable gold-context-absent count",
        ),
        _check(
            name="source_stage110_is_expected",
            passed=stage110_summary.get("stage") == _SOURCE_STAGE110,
            observed=stage110_summary.get("stage"),
            expected=_SOURCE_STAGE110,
        ),
        _check(
            name="stage110_stopped_failure_pattern_redesign_family",
            passed=stage110_summary.get("decision_status") == _STAGE110_STOP_STATUS,
            observed=stage110_summary.get("decision_status"),
            expected=_STAGE110_STOP_STATUS,
        ),
        _check(
            name="stage110_requires_user_confirmed_next_direction",
            passed=stage110_summary.get("recommended_next_direction")
            == _STAGE110_NEXT_DIRECTION
            and stage110_summary.get("requires_user_confirmation_before_next_protocol")
            is True,
            observed={
                "recommended_next_direction": stage110_summary.get(
                    "recommended_next_direction"
                ),
                "requires_confirmation": stage110_summary.get(
                    "requires_user_confirmation_before_next_protocol"
                ),
            },
            expected=_STAGE110_NEXT_DIRECTION,
        ),
        _check(
            name="user_confirmed_stage111_protocol",
            passed=user_confirmed_protocol,
            observed=user_confirmed_protocol,
            expected=True,
        ),
        _check(
            name="protocol_id_matches",
            passed=frozen.get("protocol_id") == _PROTOCOL_ID,
            observed=frozen.get("protocol_id"),
            expected=_PROTOCOL_ID,
        ),
        _check(
            name="audit_dimensions_cover_offered_route_a",
            passed=tuple(item.get("dimension_id") for item in frozen.get("audit_dimensions") or [])
            == _AUDIT_DIMENSIONS,
            observed=[item.get("dimension_id") for item in frozen.get("audit_dimensions") or []],
            expected=list(_AUDIT_DIMENSIONS),
        ),
        _check(
            name="stage112_contract_is_train_dev_only",
            passed=stage112_contract.get("reported_splits") == ["train", "dev"]
            and "test split" in stage112_contract.get("forbidden_inputs", [])
            and stage112_contract.get("final_test_metrics_allowed") is False,
            observed=stage112_contract,
            expected="train/dev audit only; test and final metrics forbidden",
        ),
        _check(
            name="gold_doc_ids_are_offline_audit_only",
            passed=stage112_contract.get("stage112_may_use_gold_doc_id_for_offline_labeling")
            is True
            and stage112_contract.get("gold_doc_id_allowed_as_runtime_feature") is False,
            observed=stage112_contract,
            expected="gold doc IDs audit-only, never runtime feature",
        ),
        _check(
            name="stage112_selection_and_defaultization_forbidden",
            passed=stage112_contract.get("selection_or_threshold_tuning_allowed") is False
            and stage112_contract.get("candidate_defaultization_allowed") is False,
            observed=stage112_contract,
            expected="no selection, threshold tuning, or defaultization",
        ),
        _check(
            name="public_safe_contract_has_forbidden_fields",
            passed=set(public_contract.get("forbidden_fields") or [])
            == _FORBIDDEN_PUBLIC_KEYS,
            observed=public_contract.get("forbidden_fields"),
            expected=sorted(_FORBIDDEN_PUBLIC_KEYS),
        ),
        _check(
            name="stage111_does_not_load_splits_or_corpus",
            passed=True,
            observed="not_loaded",
            expected="not_loaded",
        ),
        _check(
            name="stage111_does_not_run_retrieval_or_answer_metrics",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage111_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="all_source_reports_keep_test_locked",
            passed=all(
                summary.get("can_open_final_test_gate_now") is False
                and summary.get("can_run_final_test_metrics_now") is False
                for summary in (stage102_summary, stage107_summary, stage110_summary)
            ),
            observed={
                "stage102": {
                    "can_open_final_test_gate_now": stage102_summary.get(
                        "can_open_final_test_gate_now"
                    ),
                    "can_run_final_test_metrics_now": stage102_summary.get(
                        "can_run_final_test_metrics_now"
                    ),
                },
                "stage107": {
                    "can_open_final_test_gate_now": stage107_summary.get(
                        "can_open_final_test_gate_now"
                    ),
                    "can_run_final_test_metrics_now": stage107_summary.get(
                        "can_run_final_test_metrics_now"
                    ),
                },
                "stage110": {
                    "can_open_final_test_gate_now": stage110_summary.get(
                        "can_open_final_test_gate_now"
                    ),
                    "can_run_final_test_metrics_now": stage110_summary.get(
                        "can_run_final_test_metrics_now"
                    ),
                },
            },
            expected=False,
        ),
        _check(
            name="all_source_reports_forbid_test_tuning",
            passed=all(
                summary.get("can_use_test_for_tuning") is False
                for summary in (stage102_summary, stage107_summary, stage110_summary)
            ),
            observed={
                "stage102": stage102_summary.get("can_use_test_for_tuning"),
                "stage107": stage107_summary.get("can_use_test_for_tuning"),
                "stage110": stage110_summary.get("can_use_test_for_tuning"),
            },
            expected=False,
        ),
        _check(
            name="runtime_defaults_remain_unchanged",
            passed=all(
                summary.get("default_runtime_policy") == "unchanged"
                for summary in (stage102_summary, stage107_summary, stage110_summary)
            ),
            observed={
                "stage102": stage102_summary.get("default_runtime_policy"),
                "stage107": stage107_summary.get("default_runtime_policy"),
                "stage110": stage110_summary.get("default_runtime_policy"),
            },
            expected="unchanged",
        ),
        _check(
            name="fallback_strategies_remain_disabled",
            passed=all(
                summary.get("fallback_strategies_enabled") is False
                for summary in (stage102_summary, stage107_summary, stage110_summary)
            )
            and (frozen.get("fallback_strategy_policy") or {}).get(
                "fallback_strategies_enabled"
            )
            is False,
            observed={
                "stage102": stage102_summary.get("fallback_strategies_enabled"),
                "stage107": stage107_summary.get("fallback_strategies_enabled"),
                "stage110": stage110_summary.get("fallback_strategies_enabled"),
                "protocol": frozen.get("fallback_strategy_policy") or {},
            },
            expected=False,
        ),
        _check(
            name="public_outputs_have_no_forbidden_keys",
            passed=not forbidden_keys_found,
            observed=sorted(forbidden_keys_found),
            expected=[],
        ),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_retrieval_context_miss_audit_protocol_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "requires_user_confirmation_before_train_dev_audit": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": "primeqa_hybrid_retrieval_context_miss_audit_protocol_frozen",
        "protocol_id": _PROTOCOL_ID,
        "route_id": _ROUTE_ID,
        "recommended_next_direction": _NEXT_DIRECTION,
        "can_continue_train_dev_development": True,
        "can_run_train_dev_audit_after_user_confirmation": True,
        "requires_user_confirmation_before_train_dev_audit": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage112: after user confirmation, run the frozen train/dev-only "
            "retrieval-context-miss root-cause audit. Keep test locked, report "
            "train/dev separately, do not choose a candidate from dev, keep "
            "runtime defaults unchanged, and do not add fallback strategies."
        ),
    }


def _retrieval_miss_count_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = report.get("stage102_summary") or {}
    return [
        BarDatum(
            label="train retrieval_context_miss",
            value=float(summary.get("train_retrieval_context_miss_count") or 0),
            value_label=str(summary.get("train_retrieval_context_miss_count") or 0),
        ),
        BarDatum(
            label="dev retrieval_context_miss",
            value=float(summary.get("dev_retrieval_context_miss_count") or 0),
            value_label=str(summary.get("dev_retrieval_context_miss_count") or 0),
        ),
    ]


def _audit_dimension_priority_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(item["dimension_id"]),
            value=float(item["priority_score"]),
            value_label=f"{float(item['priority_score']):.2f}",
        )
        for item in (report.get("frozen_protocol") or {}).get("audit_dimensions") or []
    ]


def _stage112_access_contract_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    contract = (report.get("frozen_protocol") or {}).get("stage112_run_contract") or {}
    values = {
        "train/dev splits allowed": "Stage68 train split"
        in contract.get("allowed_inputs_after_user_confirmation", [])
        and "Stage68 dev split"
        in contract.get("allowed_inputs_after_user_confirmation", []),
        "corpus allowed": "PrimeQA training/dev corpus sections"
        in contract.get("allowed_inputs_after_user_confirmation", []),
        "test split allowed": "test split"
        not in contract.get("forbidden_inputs", []),
        "gold doc runtime feature": contract.get(
            "gold_doc_id_allowed_as_runtime_feature"
        )
        is True,
        "selection/tuning allowed": contract.get(
            "selection_or_threshold_tuning_allowed"
        )
        is True,
        "candidate defaultization allowed": contract.get(
            "candidate_defaultization_allowed"
        )
        is True,
        "final test metrics allowed": contract.get("final_test_metrics_allowed") is True,
    }
    return [
        BarDatum(
            label=name,
            value=1.0 if enabled else 0.0,
            value_label=str(enabled).lower(),
        )
        for name, enabled in values.items()
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    names = [
        "can_continue_train_dev_development",
        "can_run_train_dev_audit_after_user_confirmation",
        "requires_user_confirmation_before_train_dev_audit",
        "can_open_final_test_gate_now",
        "can_run_final_test_metrics_now",
        "can_use_test_for_tuning",
        "fallback_strategies_enabled",
    ]
    return [
        BarDatum(
            label=name,
            value=1.0 if decision.get(name) is True else 0.0,
            value_label=str(decision.get(name)).lower(),
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


def _load_json_object(path: Path) -> dict[str, Any]:
    _ensure_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    _ensure_file(path)
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")


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


def _forbidden_keys_found(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in _FORBIDDEN_PUBLIC_KEYS:
                found.add(str(key))
            found.update(_forbidden_keys_found(nested))
    elif isinstance(value, list | tuple):
        for nested in value:
            found.update(_forbidden_keys_found(nested))
    return found
