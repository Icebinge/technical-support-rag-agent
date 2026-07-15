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

_STAGE = "Stage 113"
_CREATED_AT = "2026-07-16"
_SOURCE_STAGE = "Stage 112"
_SOURCE_ANALYSIS_ID = "primeqa_hybrid_retrieval_context_miss_root_cause_audit_v1"
_SOURCE_STATUS = "primeqa_hybrid_retrieval_context_miss_root_cause_audit_completed"
_PROTOCOL_ID = "primeqa_hybrid_retrieval_index_redesign_protocol_v1"
_NEXT_DIRECTION = "run_retrieval_index_redesign_train_cv_dev_validation"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = ("test",)
_CANDIDATE_FAMILY_IDS = (
    "title_heading_weighted_bm25_candidate_v1",
    "section_level_index_rollup_candidate_v1",
    "entity_version_error_code_handling_candidate_v1",
)
_REQUIRED_ROOT_CAUSES = (
    "title_heading_mismatch",
    "query_expression_gap",
)
_REQUIRED_HIGH_SIGNAL_DIMENSIONS = (
    "title_heading_mismatch",
    "bm25_field_weighting_or_index_structure",
    "entity_version_error_code_mismatch",
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
class PrimeQAHybridRetrievalIndexRedesignProtocolVisualization:
    """One generated Stage113 retrieval/index redesign protocol chart."""

    name: str
    path: str


def freeze_primeqa_hybrid_retrieval_index_redesign_protocol(
    *,
    stage112_report_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage113 train/dev-only retrieval/index redesign protocol."""

    started_at = time.perf_counter()
    stage112_report = _load_json_object(stage112_report_path)
    loaded_at = time.perf_counter()
    stage112_summary = _stage112_summary(stage112_report)
    frozen_protocol = _frozen_protocol(stage112_summary)
    preliminary_report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "protocol_scope": (
            "Train/dev-only protocol freeze for retrieval/index redesign after "
            "Stage112 diagnosed retrieval_context_miss root causes. This stage "
            "reads only the saved public-safe Stage112 report, freezes candidate "
            "families and train grouped-CV/dev validation rules, does not load "
            "split files, does not load corpus documents, does not run retrieval "
            "or answer metrics, does not run final metrics, does not select from "
            "dev-only observations, does not add fallback strategies, and does "
            "not change runtime defaults."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_protocol),
            "confirmation_note": confirmation_note,
        },
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "development_splits": list(_DEVELOPMENT_SPLITS),
            "selection_split": "train",
            "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
            "validation_split": "dev",
            "dev_validation_mode": "single_pass_no_retuning",
            "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage112_report": _fingerprint(stage112_report_path),
        },
        "stage112_summary": stage112_summary,
        "frozen_protocol": frozen_protocol,
    }
    guard_checks = _guard_checks(
        report=preliminary_report,
        stage112_report=stage112_report,
        stage112_summary=stage112_summary,
        user_confirmed_protocol=user_confirmed_protocol,
    )
    checked_at = time.perf_counter()
    return {
        **preliminary_report,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
        "timing_seconds": {
            "load_report": round(loaded_at - started_at, 3),
            "freeze_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_retrieval_index_redesign_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridRetrievalIndexRedesignProtocolVisualization]:
    """Write SVG charts for the Stage113 protocol freeze."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage113_stage112_primary_root_causes.svg": render_horizontal_bar_chart_svg(
            title="Stage113 Stage112 primary root causes",
            bars=_stage112_root_cause_bars(report),
            x_label="audit cases",
            width=1320,
            margin_left=620,
        ),
        "stage113_stage112_high_signal_dimensions.svg": render_horizontal_bar_chart_svg(
            title="Stage113 Stage112 high-signal dimensions",
            bars=_stage112_high_signal_bars(report),
            x_label="audit cases",
            width=1360,
            margin_left=680,
        ),
        "stage113_candidate_family_priorities.svg": render_horizontal_bar_chart_svg(
            title="Stage113 candidate family priorities",
            bars=_candidate_family_priority_bars(report),
            x_label="priority score",
            width=1500,
            margin_left=760,
        ),
        "stage113_candidate_config_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage113 candidate config counts",
            bars=_candidate_config_count_bars(report),
            x_label="config count",
            width=1480,
            margin_left=760,
        ),
        "stage113_selection_guard_thresholds.svg": render_horizontal_bar_chart_svg(
            title="Stage113 train-CV guard thresholds",
            bars=_selection_guard_threshold_bars(report),
            x_label="maximum allowed train-CV regression",
            width=1500,
            margin_left=760,
        ),
        "stage113_protocol_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage113 protocol decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1320,
            margin_left=660,
        ),
        "stage113_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage113 guard checks",
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
            PrimeQAHybridRetrievalIndexRedesignProtocolVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage112_summary(stage112_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage112_report.get("decision") or {}
    cross = stage112_report.get("cross_split_summary") or {}
    loaded = stage112_report.get("loaded_data_summary") or {}
    split_reports = stage112_report.get("split_reports") or {}
    return {
        "stage": stage112_report.get("stage"),
        "analysis_id": stage112_report.get("analysis_id"),
        "decision_status": decision.get("status"),
        "recommended_next_stage": decision.get("recommended_next_stage"),
        "document_count": loaded.get("document_count"),
        "section_count": loaded.get("section_count"),
        "test_split_loaded": loaded.get("test_split_loaded"),
        "answerable_rows": cross.get("answerable_rows"),
        "audit_case_count": cross.get("audit_case_count"),
        "audit_case_rate_among_answerable": cross.get(
            "audit_case_rate_among_answerable"
        ),
        "train_audit_case_count": (split_reports.get("train") or {}).get(
            "audit_case_count"
        ),
        "dev_audit_case_count": (split_reports.get("dev") or {}).get(
            "audit_case_count"
        ),
        "primary_root_cause_counts": cross.get("primary_root_cause_counts") or {},
        "dimension_high_signal_counts": cross.get("dimension_high_signal_counts") or {},
        "gold_doc_rank_bucket_counts": cross.get("gold_doc_rank_bucket_counts") or {},
        "question_route_counts": cross.get("question_route_counts") or {},
        "common_train_dev_root_causes": cross.get("common_train_dev_root_causes")
        or [],
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get(
            "can_run_final_test_metrics_now"
        ),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _frozen_protocol(stage112_summary: Mapping[str, Any]) -> dict[str, Any]:
    candidate_families = _candidate_families(stage112_summary)
    return {
        "protocol_id": _PROTOCOL_ID,
        "protocol_status": "frozen_requires_user_confirmation_before_train_dev_run",
        "source_stages": [_SOURCE_STAGE],
        "redesign_mode": "protocol_freeze_only_no_metrics",
        "objective": (
            "Compare retrieval/index candidates that target Stage112 "
            "retrieval_context_miss root causes using train grouped-CV for "
            "selection and one dev validation pass. The protocol focuses on "
            "title/heading alignment, section-level indexing, and entity, "
            "version, or error-code handling while guarding answer quality and "
            "citation coverage."
        ),
        "stage112_failure_basis": {
            "audit_case_count": stage112_summary.get("audit_case_count"),
            "train_audit_case_count": stage112_summary.get("train_audit_case_count"),
            "dev_audit_case_count": stage112_summary.get("dev_audit_case_count"),
            "audit_case_rate_among_answerable": stage112_summary.get(
                "audit_case_rate_among_answerable"
            ),
            "primary_root_cause_counts": stage112_summary.get(
                "primary_root_cause_counts"
            ),
            "dimension_high_signal_counts": stage112_summary.get(
                "dimension_high_signal_counts"
            ),
            "gold_doc_rank_bucket_counts": stage112_summary.get(
                "gold_doc_rank_bucket_counts"
            ),
        },
        "candidate_families": candidate_families,
        "candidate_configs": _candidate_configs(),
        "selection_rules": _selection_rules(),
        "output_contract": _output_contract(),
        "explicit_exclusions": [
            "no_split_loading_in_stage113",
            "no_corpus_document_loading_in_stage113",
            "no_metric_run_in_stage113",
            "no_test_split_loading",
            "no_final_test_metrics",
            "no_dev_selection",
            "no_dev_retuning",
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
            "stage": "Stage 114",
            "recommended_direction": _NEXT_DIRECTION,
            "requires_user_confirmation_before_train_dev_run": True,
            "source_protocol_id": _PROTOCOL_ID,
            "selection_split": "train",
            "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
            "validation_split": "dev",
            "dev_validation_mode": "single_pass_no_retuning",
            "must_not_load_or_score_test": True,
            "must_not_change_runtime_defaults": True,
            "must_not_add_fallback_strategies": True,
        },
    }


def _candidate_families(stage112_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    primary = stage112_summary.get("primary_root_cause_counts") or {}
    high_signal = stage112_summary.get("dimension_high_signal_counts") or {}
    return [
        {
            "family_id": "title_heading_weighted_bm25_candidate_v1",
            "priority_score": 0.95,
            "stage112_basis": {
                "primary_title_heading_mismatch_count": primary.get(
                    "title_heading_mismatch",
                    0,
                ),
                "high_signal_title_heading_mismatch_count": high_signal.get(
                    "title_heading_mismatch",
                    0,
                ),
                "primary_query_expression_gap_count": primary.get(
                    "query_expression_gap",
                    0,
                ),
            },
            "hypothesis": (
                "BM25 under-ranks gold documents when user wording aligns more "
                "with document titles or section headings than full body text."
            ),
            "allowed_runtime_features": [
                "question text",
                "document title",
                "section heading",
                "document body text",
            ],
            "forbidden_runtime_features": [
                "gold document identifier",
                "gold answer span",
                "test labels",
            ],
            "config_count": 3,
        },
        {
            "family_id": "section_level_index_rollup_candidate_v1",
            "priority_score": 0.90,
            "stage112_basis": {
                "high_signal_index_structure_count": high_signal.get(
                    "bm25_field_weighting_or_index_structure",
                    0,
                ),
                "not_found_top50_count": (
                    stage112_summary.get("gold_doc_rank_bucket_counts") or {}
                ).get("not_found_top50", 0),
            },
            "hypothesis": (
                "Document-level BM25 hides relevant local sections; section "
                "indexes and section-to-document rollup may recover gold "
                "context without answer-pipeline fallback behavior."
            ),
            "allowed_runtime_features": [
                "question text",
                "section heading",
                "section body text",
                "document title",
            ],
            "forbidden_runtime_features": [
                "gold document identifier",
                "gold answer span",
                "test labels",
            ],
            "config_count": 3,
        },
        {
            "family_id": "entity_version_error_code_handling_candidate_v1",
            "priority_score": 0.80,
            "stage112_basis": {
                "high_signal_entity_version_error_code_count": high_signal.get(
                    "entity_version_error_code_mismatch",
                    0,
                ),
                "primary_entity_version_error_code_count": primary.get(
                    "entity_version_error_code_mismatch",
                    0,
                ),
            },
            "hypothesis": (
                "Special tokens such as product versions, CVEs, APARs, and error "
                "codes need protected lexical handling before ranking."
            ),
            "allowed_runtime_features": [
                "question text",
                "document title",
                "section heading",
                "document body text",
                "runtime-visible special-token matches",
            ],
            "forbidden_runtime_features": [
                "gold document identifier",
                "gold answer span",
                "test labels",
            ],
            "config_count": 2,
        },
    ]


def _candidate_configs() -> list[dict[str, Any]]:
    return [
        {
            "config_id": "thw_title2_heading2_body1_doc_bm25_v1",
            "family_id": "title_heading_weighted_bm25_candidate_v1",
            "retrieval_mode": "weighted_document_bm25",
            "description": (
                "Boost document title and section-heading text while keeping "
                "body text in the document index."
            ),
            "weights": {"title": 2.0, "section_heading": 2.0, "body": 1.0},
            "selection_eligible": True,
        },
        {
            "config_id": "thw_title3_heading2_body1_doc_bm25_v1",
            "family_id": "title_heading_weighted_bm25_candidate_v1",
            "retrieval_mode": "weighted_document_bm25",
            "description": "Stronger title weighting for title-heading mismatch dominant cases.",
            "weights": {"title": 3.0, "section_heading": 2.0, "body": 1.0},
            "selection_eligible": True,
        },
        {
            "config_id": "thw_title_heading_query_view_rrf_v1",
            "family_id": "title_heading_weighted_bm25_candidate_v1",
            "retrieval_mode": "document_bm25_rrf",
            "description": (
                "Fuse body BM25 with a title-and-heading query view using "
                "reciprocal rank fusion."
            ),
            "rrf_k": 60,
            "selection_eligible": True,
        },
        {
            "config_id": "slr_section_top1_doc_rollup_v1",
            "family_id": "section_level_index_rollup_candidate_v1",
            "retrieval_mode": "section_bm25_document_rollup",
            "description": "Rank sections and roll each document up by its best matching section.",
            "section_rollup": "top1_section_score",
            "selection_eligible": True,
        },
        {
            "config_id": "slr_section_top3_rrf_doc_rollup_v1",
            "family_id": "section_level_index_rollup_candidate_v1",
            "retrieval_mode": "section_document_rrf",
            "description": "Fuse document BM25 with top3 section evidence per document.",
            "section_rollup": "top3_section_rrf",
            "rrf_k": 60,
            "selection_eligible": True,
        },
        {
            "config_id": "slr_heading_section_title_rollup_v1",
            "family_id": "section_level_index_rollup_candidate_v1",
            "retrieval_mode": "heading_section_title_rollup",
            "description": (
                "Use section heading plus section text and document title for "
                "local context recovery."
            ),
            "section_heading_weight": 2.0,
            "document_title_weight": 2.0,
            "selection_eligible": True,
        },
        {
            "config_id": "evc_special_token_exact_boost_v1",
            "family_id": "entity_version_error_code_handling_candidate_v1",
            "retrieval_mode": "bm25_with_runtime_special_token_boost",
            "description": (
                "Apply deterministic boosts for runtime-visible special-token "
                "exact matches."
            ),
            "special_token_boost": 1.5,
            "selection_eligible": True,
        },
        {
            "config_id": "evc_special_token_title_heading_boost_v1",
            "family_id": "entity_version_error_code_handling_candidate_v1",
            "retrieval_mode": "weighted_bm25_with_special_token_boost",
            "description": "Combine title/heading weighting with protected special-token matches.",
            "title_weight": 2.0,
            "heading_weight": 2.0,
            "special_token_boost": 1.5,
            "selection_eligible": True,
        },
    ]


def _selection_rules() -> dict[str, Any]:
    return {
        "selection_split": "train",
        "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
        "train_group_key": "normalized_question_plus_answer_document_or_technote",
        "minimum_train_folds": 5,
        "validation_split": "dev",
        "dev_validation_mode": "single_pass_no_retuning",
        "primary_objective": {
            "name": "reduce_retrieval_context_miss",
            "required_train_cv_delta": "negative",
            "weight": 2.0,
        },
        "secondary_objectives": [
            {
                "name": "improve_gold_doc_recall_at_10",
                "required_train_cv_delta": "positive",
                "weight": 1.5,
            },
            {
                "name": "avoid_average_token_f1_regression",
                "required_train_cv_drop_lte": 0.005,
                "weight": 1.0,
            },
            {
                "name": "avoid_gold_doc_citation_rate_regression",
                "required_train_cv_drop_lte": 0.015,
                "weight": 1.0,
            },
        ],
        "guard_thresholds": {
            "max_train_cv_average_token_f1_drop": 0.005,
            "max_train_cv_gold_doc_citation_rate_drop": 0.015,
            "max_train_cv_answerable_refusal_rate_delta": 0.02,
            "max_train_cv_answerability_false_answer_delta": 0,
            "max_train_cv_evidence_selection_miss_delta": 0,
            "max_train_cv_gold_span_beats_selected_delta": 0,
            "max_train_cv_changed_answer_rate": 0.25,
        },
        "selection_forbidden_if": [
            "train_cv_retrieval_context_miss_delta_nonnegative",
            "train_cv_average_token_f1_drop_exceeds_guard",
            "train_cv_gold_doc_citation_rate_drop_exceeds_guard",
            "train_cv_answerable_refusal_rate_delta_exceeds_guard",
            "train_cv_answerability_false_answer_delta_positive",
            "train_cv_evidence_selection_miss_delta_positive",
            "train_cv_gold_span_beats_selected_delta_positive",
            "train_cv_changed_answer_rate_exceeds_guard",
            "candidate_is_noop",
            "uses_gold_document_identifier_as_runtime_feature",
            "uses_test_data",
        ],
        "dev_rules": {
            "dev_selection_allowed": False,
            "dev_retuning_allowed": False,
            "dev_threshold_tuning_allowed": False,
            "dev_report_required": True,
        },
        "test_rules": {
            "test_access_allowed": False,
            "final_test_metrics_allowed": False,
            "test_tuning_allowed": False,
        },
        "runtime_rules": {
            "default_runtime_policy": "unchanged",
            "fallback_strategies_enabled": False,
        },
    }


def _output_contract() -> dict[str, Any]:
    return {
        "aggregate_only_by_default": True,
        "allowed_case_fields": [
            "sample_id",
            "split",
            "config_id",
            "family_id",
            "question_route",
            "baseline_retrieval_context_miss_bucket",
            "candidate_retrieval_context_miss_bucket",
            "gold_doc_rank_delta_bucket",
            "root_cause_bucket",
            "metric_delta_bucket",
            "selection_status",
            "guard_failure_reasons",
        ],
        "forbidden_fields": sorted(_FORBIDDEN_PUBLIC_KEYS),
        "raw_group_values_written": False,
    }


def _guard_checks(
    *,
    report: Mapping[str, Any],
    stage112_report: Mapping[str, Any],
    stage112_summary: Mapping[str, Any],
    user_confirmed_protocol: bool,
) -> list[dict[str, Any]]:
    frozen = report.get("frozen_protocol") or {}
    selection = frozen.get("selection_rules") or {}
    candidate_families = frozen.get("candidate_families") or []
    candidate_configs = frozen.get("candidate_configs") or []
    forbidden_keys_found = _forbidden_keys_found(report)
    stage112_forbidden_keys_found = _forbidden_keys_found(stage112_report)
    primary = Counter(stage112_summary.get("primary_root_cause_counts") or {})
    high_signal = Counter(stage112_summary.get("dimension_high_signal_counts") or {})
    return [
        _check(
            name="source_stage112_is_expected",
            passed=stage112_summary.get("stage") == _SOURCE_STAGE,
            observed=stage112_summary.get("stage"),
            expected=_SOURCE_STAGE,
        ),
        _check(
            name="source_stage112_analysis_id_matches",
            passed=stage112_summary.get("analysis_id") == _SOURCE_ANALYSIS_ID,
            observed=stage112_summary.get("analysis_id"),
            expected=_SOURCE_ANALYSIS_ID,
        ),
        _check(
            name="stage112_audit_completed",
            passed=stage112_summary.get("decision_status") == _SOURCE_STATUS,
            observed=stage112_summary.get("decision_status"),
            expected=_SOURCE_STATUS,
        ),
        _check(
            name="user_confirmed_stage113_protocol",
            passed=user_confirmed_protocol,
            observed=user_confirmed_protocol,
            expected=True,
        ),
        _check(
            name="stage112_test_split_was_not_loaded",
            passed=stage112_summary.get("test_split_loaded") is False,
            observed=stage112_summary.get("test_split_loaded"),
            expected=False,
        ),
        _check(
            name="stage112_audit_has_train_dev_cases",
            passed=int(stage112_summary.get("train_audit_case_count") or 0) > 0
            and int(stage112_summary.get("dev_audit_case_count") or 0) > 0,
            observed={
                "train": stage112_summary.get("train_audit_case_count"),
                "dev": stage112_summary.get("dev_audit_case_count"),
            },
            expected="positive train and dev audit case counts",
        ),
        _check(
            name="required_primary_root_causes_present",
            passed=all(primary.get(cause, 0) > 0 for cause in _REQUIRED_ROOT_CAUSES),
            observed=dict(primary),
            expected=list(_REQUIRED_ROOT_CAUSES),
        ),
        _check(
            name="required_high_signal_dimensions_present",
            passed=all(
                high_signal.get(dimension, 0) > 0
                for dimension in _REQUIRED_HIGH_SIGNAL_DIMENSIONS
            ),
            observed=dict(high_signal),
            expected=list(_REQUIRED_HIGH_SIGNAL_DIMENSIONS),
        ),
        _check(
            name="candidate_families_match_stage112_findings",
            passed=tuple(item.get("family_id") for item in candidate_families)
            == _CANDIDATE_FAMILY_IDS,
            observed=[item.get("family_id") for item in candidate_families],
            expected=list(_CANDIDATE_FAMILY_IDS),
        ),
        _check(
            name="candidate_configs_are_selection_eligible_and_nonempty",
            passed=len(candidate_configs) == 8
            and all(config.get("selection_eligible") is True for config in candidate_configs),
            observed={
                "config_count": len(candidate_configs),
                "selection_eligible_count": sum(
                    config.get("selection_eligible") is True
                    for config in candidate_configs
                ),
            },
            expected={"config_count": 8, "selection_eligible_count": 8},
        ),
        _check(
            name="selection_is_train_grouped_cv",
            passed=selection.get("selection_split") == "train"
            and selection.get("selection_mode")
            == "train_grouped_cross_validation_then_full_train_refit",
            observed=selection,
            expected="train grouped-CV selection",
        ),
        _check(
            name="dev_is_validation_only",
            passed=(selection.get("dev_rules") or {}).get("dev_selection_allowed")
            is False
            and (selection.get("dev_rules") or {}).get("dev_retuning_allowed") is False
            and (selection.get("dev_rules") or {}).get("dev_threshold_tuning_allowed")
            is False,
            observed=selection.get("dev_rules"),
            expected="dev validation only; no selection or retuning",
        ),
        _check(
            name="test_access_forbidden",
            passed=(selection.get("test_rules") or {}).get("test_access_allowed") is False
            and (selection.get("test_rules") or {}).get("final_test_metrics_allowed")
            is False
            and (selection.get("test_rules") or {}).get("test_tuning_allowed") is False,
            observed=selection.get("test_rules"),
            expected="test locked",
        ),
        _check(
            name="runtime_defaults_unchanged",
            passed=(selection.get("runtime_rules") or {}).get("default_runtime_policy")
            == "unchanged",
            observed=selection.get("runtime_rules"),
            expected="unchanged",
        ),
        _check(
            name="fallback_strategies_disabled",
            passed=(selection.get("runtime_rules") or {}).get(
                "fallback_strategies_enabled"
            )
            is False
            and (frozen.get("fallback_strategy_policy") or {}).get(
                "fallback_strategies_enabled"
            )
            is False,
            observed={
                "selection_runtime_rules": selection.get("runtime_rules"),
                "fallback_strategy_policy": frozen.get("fallback_strategy_policy"),
            },
            expected=False,
        ),
        _check(
            name="stage113_does_not_load_splits_or_corpus",
            passed=True,
            observed="not_loaded",
            expected="not_loaded",
        ),
        _check(
            name="stage113_does_not_run_retrieval_or_answer_metrics",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage113_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="source_stage112_report_is_public_safe",
            passed=not stage112_forbidden_keys_found,
            observed=sorted(stage112_forbidden_keys_found),
            expected=[],
        ),
        _check(
            name="stage113_public_outputs_have_no_forbidden_keys",
            passed=not forbidden_keys_found,
            observed=sorted(forbidden_keys_found),
            expected=[],
        ),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_retrieval_index_redesign_protocol_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "requires_user_confirmation_before_train_dev_run": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": "primeqa_hybrid_retrieval_index_redesign_protocol_frozen",
        "protocol_id": _PROTOCOL_ID,
        "recommended_next_direction": _NEXT_DIRECTION,
        "can_continue_train_dev_development": True,
        "requires_user_confirmation_before_train_dev_run": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage114: after user confirmation, run the frozen train grouped-CV "
            "retrieval/index redesign comparison, then one dev validation pass. "
            "Keep test locked, do not choose from dev-only observations, keep "
            "runtime defaults unchanged, and do not add fallback strategies."
        ),
    }


def _stage112_root_cause_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counter = Counter(
        (report.get("stage112_summary") or {}).get("primary_root_cause_counts") or {}
    )
    return [
        BarDatum(label, float(value), str(value))
        for label, value in _top_counter_items(counter, limit=8)
    ]


def _stage112_high_signal_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counter = Counter(
        (report.get("stage112_summary") or {}).get("dimension_high_signal_counts")
        or {}
    )
    return [
        BarDatum(label, float(value), str(value))
        for label, value in _top_counter_items(counter, limit=8)
    ]


def _candidate_family_priority_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    families = (report.get("frozen_protocol") or {}).get("candidate_families") or []
    return [
        BarDatum(
            str(family["family_id"]),
            float(family["priority_score"]),
            f"{float(family['priority_score']):.2f}",
        )
        for family in families
    ]


def _candidate_config_count_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    families = (report.get("frozen_protocol") or {}).get("candidate_families") or []
    return [
        BarDatum(
            str(family["family_id"]),
            float(family["config_count"]),
            str(family["config_count"]),
        )
        for family in families
    ]


def _selection_guard_threshold_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    thresholds = (
        ((report.get("frozen_protocol") or {}).get("selection_rules") or {})
        .get("guard_thresholds")
        or {}
    )
    return [
        BarDatum(label, float(value), str(value))
        for label, value in sorted(thresholds.items())
        if isinstance(value, int | float)
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    names = [
        "can_continue_train_dev_development",
        "requires_user_confirmation_before_train_dev_run",
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


def _top_counter_items(counter: Counter[str], *, limit: int) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]


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
