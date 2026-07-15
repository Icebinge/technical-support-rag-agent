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

_STAGE = "Stage 108"
_CREATED_AT = "2026-07-16"
_SOURCE_STAGE107 = "Stage 107"
_SOURCE_PROTOCOL_ID = "primeqa_hybrid_validation_failure_pattern_analysis_v1"
_SOURCE_DECISION_STATUS = "primeqa_hybrid_validation_failure_pattern_analysis_completed"
_PROTOCOL_ID = "primeqa_hybrid_failure_pattern_redesign_protocol_v1"
_NEXT_DIRECTION = "run_failure_pattern_redesign_train_cv_dev_validation"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = ("test",)
_CANDIDATE_FAMILY_IDS = (
    "support_aware_answerability_gate_candidate_v1",
    "context_present_span_composer_candidate_v1",
    "joint_support_gate_span_composer_candidate_v1",
)
_STOPPED_STAGE104_PREFIXES = ("amg_", "ewr_", "jgw_")
_TARGET_BUCKET_WEIGHTS = {
    "answerability_false_answer": 1.75,
    "gold_span_beats_selected_answer": 1.80,
    "evidence_selection_miss": 1.50,
}
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
class PrimeQAHybridFailurePatternRedesignProtocolVisualization:
    """One generated Stage108 failure-pattern redesign protocol chart."""

    name: str
    path: str


def freeze_primeqa_hybrid_failure_pattern_redesign_protocol(
    *,
    stage107_report_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage108 train/dev-only failure-pattern redesign protocol."""

    started_at = time.perf_counter()
    stage107_report = _load_json_object(stage107_report_path)
    loaded_at = time.perf_counter()
    stage107_summary = _stage107_summary(stage107_report)
    frozen_protocol = _frozen_protocol(stage107_summary)
    preliminary_report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "protocol_scope": (
            "Train/dev-only protocol freeze for a failure-pattern-driven "
            "answer-pipeline redesign after Stage107 showed broad validation "
            "failure. This stage reads only the saved public-safe Stage107 "
            "report, freezes candidate families, train grouped-CV selection, "
            "and dev validation rules, does not load split files, does not "
            "load corpus documents, does not run retrieval or answer metrics, "
            "does not run final metrics, does not select from dev-only "
            "observations, does not add fallback strategies, and does not "
            "change runtime defaults."
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
            "validation_split": "dev",
            "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage107_report": _fingerprint(stage107_report_path),
        },
        "stage107_summary": stage107_summary,
        "frozen_protocol": frozen_protocol,
    }
    guard_checks = _guard_checks(
        report=preliminary_report,
        stage107_report=stage107_report,
        stage107_summary=stage107_summary,
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


def write_primeqa_hybrid_failure_pattern_redesign_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridFailurePatternRedesignProtocolVisualization]:
    """Write SVG charts for Stage108 failure-pattern redesign protocol."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage108_candidate_family_priorities.svg": render_horizontal_bar_chart_svg(
            title="Stage108 candidate family priorities",
            bars=_family_priority_bars(report),
            x_label="protocol priority score",
            width=1400,
            margin_left=680,
        ),
        "stage108_candidate_config_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage108 candidate config counts",
            bars=_candidate_config_count_bars(report),
            x_label="config count",
            width=1380,
            margin_left=680,
        ),
        "stage108_target_bucket_weights.svg": render_horizontal_bar_chart_svg(
            title="Stage108 target bucket weights",
            bars=_target_weight_bars(report),
            x_label="train-CV objective weight",
            width=1300,
            margin_left=580,
        ),
        "stage108_train_cv_guard_thresholds.svg": render_horizontal_bar_chart_svg(
            title="Stage108 train-CV guard thresholds",
            bars=_train_guard_threshold_bars(report),
            x_label="maximum allowed train-CV regression",
            width=1420,
            margin_left=720,
        ),
        "stage108_protocol_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage108 protocol decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1280,
            margin_left=620,
        ),
        "stage108_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage108 guard checks",
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
            PrimeQAHybridFailurePatternRedesignProtocolVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage107_summary(stage107_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage107_report.get("decision") or {}
    pattern = stage107_report.get("pattern_summary") or {}
    overview = pattern.get("dev_failure_overview") or {}
    context = pattern.get("dev_retrieval_and_context_profile") or {}
    candidate_pattern = pattern.get("stage105_candidate_failure_pattern") or {}
    return {
        "stage": stage107_report.get("stage"),
        "protocol_id": stage107_report.get("protocol_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "dev_failure_count": overview.get("failure_count"),
        "dev_failure_rate": overview.get("failure_rate"),
        "answerable_failure_rate": overview.get("answerable_failure_rate"),
        "unanswerable_false_answer_rate": overview.get(
            "unanswerable_false_answer_rate"
        ),
        "answerable_gold_context_absent_rate": context.get(
            "answerable_gold_context_absent_rate"
        ),
        "context_present_gold_span_beats_selected_rate": context.get(
            "context_present_gold_span_beats_selected_rate"
        ),
        "context_present_evidence_selection_miss_rate": context.get(
            "context_present_evidence_selection_miss_rate"
        ),
        "answerable_supported_and_cited_count": context.get(
            "answerable_supported_and_cited_count"
        ),
        "stage105_selected_config_was_dev_noop": decision.get(
            "stage105_selected_config_was_dev_noop"
        ),
        "stage105_dev_better_nonselectable_config_count": candidate_pattern.get(
            "dev_better_nonselectable_config_count"
        ),
        "stage105_train_guard_failure_reasons": candidate_pattern.get(
            "train_guard_failure_reasons"
        )
        or {},
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _frozen_protocol(stage107_summary: Mapping[str, Any]) -> dict[str, Any]:
    candidate_families = _candidate_families(stage107_summary)
    return {
        "protocol_id": _PROTOCOL_ID,
        "protocol_status": "frozen_requires_user_confirmation_before_train_dev_run",
        "source_stages": [_SOURCE_STAGE107],
        "redesign_mode": "protocol_freeze_only_no_metrics",
        "objective": (
            "Compare redesigned answer-pipeline candidates against the Stage102 "
            "verified baseline using train grouped CV for selection and one "
            "dev validation pass. The redesign targets answerability false "
            "answers, context-present answer composition failures, and evidence "
            "selection misses while monitoring retrieval-context misses as a "
            "non-target boundary."
        ),
        "stage107_failure_basis": {
            "dev_failure_rate": stage107_summary.get("dev_failure_rate"),
            "answerable_failure_rate": stage107_summary.get("answerable_failure_rate"),
            "unanswerable_false_answer_rate": stage107_summary.get(
                "unanswerable_false_answer_rate"
            ),
            "answerable_gold_context_absent_rate": stage107_summary.get(
                "answerable_gold_context_absent_rate"
            ),
            "context_present_gold_span_beats_selected_rate": stage107_summary.get(
                "context_present_gold_span_beats_selected_rate"
            ),
            "context_present_evidence_selection_miss_rate": stage107_summary.get(
                "context_present_evidence_selection_miss_rate"
            ),
            "stage105_selected_config_was_dev_noop": stage107_summary.get(
                "stage105_selected_config_was_dev_noop"
            ),
        },
        "candidate_families": candidate_families,
        "candidate_config_grid": _candidate_config_grid(candidate_families),
        "train_selection_rule": _train_selection_rule(),
        "dev_validation_rule": _dev_validation_rule(),
        "metric_contract": _metric_contract(),
        "runtime_feature_contract": _runtime_feature_contract(),
        "public_safe_output_contract": _public_safe_output_contract(),
        "explicit_exclusions": [
            "no_split_loading_in_stage108",
            "no_corpus_document_loading_in_stage108",
            "no_metric_run_in_stage108",
            "no_test_split_loading",
            "no_final_test_metrics",
            "no_dev_threshold_tuning",
            "no_dev_config_selection",
            "no_runtime_default_change",
            "no_fallback_strategy",
            "no_oracle_document_identifier_runtime_feature",
            "no_gold_answer_runtime_feature",
            "no_raw_question_answer_or_document_text_in_outputs",
            "no_reuse_of_stopped_stage104_config_ids",
        ],
        "fallback_strategy_policy": {
            "fallback_strategies_enabled": False,
            "requires_user_confirmation_before_any_fallback": True,
        },
        "next_stage_contract": {
            "stage": "Stage 109",
            "recommended_direction": _NEXT_DIRECTION,
            "requires_user_confirmation_before_train_dev_metric_run": True,
            "source_protocol_id": _PROTOCOL_ID,
            "implement_candidate_runtime_components_before_comparison": True,
            "must_select_on_train_grouped_cv_only": True,
            "dev_is_single_validation_only": True,
            "must_not_load_or_score_test": True,
            "must_not_change_runtime_defaults": True,
            "must_not_add_fallback_strategies": True,
        },
    }


def _candidate_families(stage107_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "family_id": "support_aware_answerability_gate_candidate_v1",
            "status": "frozen_for_stage109_train_dev_comparison",
            "priority_score": _optional_float(
                stage107_summary.get("unanswerable_false_answer_rate")
            ),
            "target_buckets": ["answerability_false_answer"],
            "risk_to_guard": "answerable_over_refusal",
            "stage107_basis": {
                "unanswerable_false_answer_rate": stage107_summary.get(
                    "unanswerable_false_answer_rate"
                ),
                "stage105_train_guard_failure": (
                    "prior aggressive variants failed train answerable-refusal guard"
                ),
            },
            "runtime_signal_boundary": _runtime_signal_boundary(),
            "candidate_configs": [
                {
                    "config_id": "saag_support2_evidence7_rank3_v1",
                    "component_family": "support_aware_answerability_gate",
                    "min_supporting_evidence_count": 2,
                    "min_evidence_score": 7.0,
                    "max_citation_rank": 3,
                    "preserve_answer_when_support_present": True,
                },
                {
                    "config_id": "saag_support2_evidence6_rank5_v1",
                    "component_family": "support_aware_answerability_gate",
                    "min_supporting_evidence_count": 2,
                    "min_evidence_score": 6.0,
                    "max_citation_rank": 5,
                    "preserve_answer_when_support_present": True,
                },
            ],
        },
        {
            "family_id": "context_present_span_composer_candidate_v1",
            "status": "frozen_for_stage109_train_dev_comparison",
            "priority_score": _optional_float(
                stage107_summary.get("context_present_gold_span_beats_selected_rate")
            ),
            "target_buckets": [
                "gold_span_beats_selected_answer",
                "evidence_selection_miss",
            ],
            "risk_to_guard": "gold_citation_or_token_f1_regression",
            "stage107_basis": {
                "context_present_gold_span_beats_selected_rate": stage107_summary.get(
                    "context_present_gold_span_beats_selected_rate"
                ),
                "context_present_evidence_selection_miss_rate": stage107_summary.get(
                    "context_present_evidence_selection_miss_rate"
                ),
            },
            "runtime_signal_boundary": _runtime_signal_boundary(),
            "candidate_configs": [
                {
                    "config_id": "cpsc_anchor_top2_mcpd3_rank3_v1",
                    "component_family": "context_present_span_composer",
                    "anchor_strategy": "top_scoring_evidence_sentences",
                    "anchor_top_n": 2,
                    "max_candidates_per_document": 3,
                    "max_sentences": 2,
                    "min_evidence_score": 7.0,
                    "max_citation_rank": 3,
                },
                {
                    "config_id": "cpsc_anchor_top3_mcpd3_rank3_v1",
                    "component_family": "context_present_span_composer",
                    "anchor_strategy": "top_scoring_evidence_sentences",
                    "anchor_top_n": 3,
                    "max_candidates_per_document": 3,
                    "max_sentences": 3,
                    "min_evidence_score": 7.0,
                    "max_citation_rank": 3,
                },
                {
                    "config_id": "cpsc_title_query_anchor_top2_mcpd3_rank3_v1",
                    "component_family": "context_present_span_composer",
                    "anchor_strategy": "title_query_overlap_then_evidence_score",
                    "anchor_top_n": 2,
                    "max_candidates_per_document": 3,
                    "max_sentences": 2,
                    "min_evidence_score": 7.0,
                    "max_citation_rank": 3,
                },
            ],
        },
        {
            "family_id": "joint_support_gate_span_composer_candidate_v1",
            "status": "frozen_for_stage109_train_dev_comparison",
            "priority_score": 1.0,
            "target_buckets": [
                "answerability_false_answer",
                "gold_span_beats_selected_answer",
                "evidence_selection_miss",
            ],
            "risk_to_guard": "combined_refusal_and_citation_regression",
            "stage107_basis": {
                "answerable_failure_rate": stage107_summary.get(
                    "answerable_failure_rate"
                ),
                "unanswerable_false_answer_rate": stage107_summary.get(
                    "unanswerable_false_answer_rate"
                ),
                "stage105_selected_config_was_dev_noop": stage107_summary.get(
                    "stage105_selected_config_was_dev_noop"
                ),
            },
            "runtime_signal_boundary": _runtime_signal_boundary(),
            "candidate_configs": [
                {
                    "config_id": "jsgc_support2_evidence7_anchor_top2_v1",
                    "component_family": "joint_support_gate_span_composer",
                    "gate_min_supporting_evidence_count": 2,
                    "gate_min_evidence_score": 7.0,
                    "composer_anchor_strategy": "top_scoring_evidence_sentences",
                    "composer_anchor_top_n": 2,
                    "max_sentences": 2,
                    "max_citation_rank": 3,
                },
                {
                    "config_id": "jsgc_support2_evidence6_title_anchor_top2_v1",
                    "component_family": "joint_support_gate_span_composer",
                    "gate_min_supporting_evidence_count": 2,
                    "gate_min_evidence_score": 6.0,
                    "composer_anchor_strategy": "title_query_overlap_then_evidence_score",
                    "composer_anchor_top_n": 2,
                    "max_sentences": 2,
                    "max_citation_rank": 5,
                },
            ],
        },
    ]


def _candidate_config_grid(
    candidate_families: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grid = []
    for family in candidate_families:
        for config in family.get("candidate_configs") or []:
            row = {
                "candidate_family_id": family["family_id"],
                "target_buckets": family["target_buckets"],
                **config,
            }
            grid.append(row)
    return grid


def _train_selection_rule() -> dict[str, Any]:
    return {
        "selection_split": "train",
        "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
        "train_cv_fold_count": 5,
        "train_cv_grouping_policy": {
            "purpose": "keep related train rows in the same fold",
            "group_key_inputs": [
                "normalized_question_text_for_grouping_only",
                "gold_document_group_marker_for_grouping_only",
                "unanswerable_marker_for_grouping_only",
            ],
            "raw_group_values_written_to_reports": False,
            "group_keys_allowed_as_runtime_features": False,
            "dev_rows_used_for_grouping": False,
            "test_rows_used_for_grouping": False,
        },
        "objective": {
            "weighted_target_bucket_score": dict(_TARGET_BUCKET_WEIGHTS),
            "lower_is_better": True,
            "requires_negative_train_cv_weighted_delta": True,
            "no_op_candidate_selectable": False,
        },
        "selectability_guards": {
            "max_train_cv_answerable_refusal_rate_delta": 0.02,
            "max_train_cv_average_token_f1_drop": 0.005,
            "max_train_cv_gold_doc_citation_rate_drop": 0.015,
            "max_train_cv_retrieval_context_miss_delta": 0,
        },
        "tie_breakers": [
            "lower train-CV answerability_false_answer count",
            "lower train-CV gold_span_beats_selected_answer count",
            "lower train-CV evidence_selection_miss count",
            "higher train-CV verified average token F1",
            "higher train-CV gold document citation rate",
            "lower train-CV answerable refusal rate",
            "lower train-CV changed answer count",
            "lexicographic config_id",
        ],
        "dev_threshold_tuning_allowed": False,
        "dev_config_selection_allowed": False,
        "test_access_allowed": False,
    }


def _dev_validation_rule() -> dict[str, Any]:
    return {
        "validation_split": "dev",
        "validated_item": "single train-CV-selected config",
        "dev_selection_allowed": False,
        "dev_retuning_allowed": False,
        "dev_threshold_tuning_allowed": False,
        "test_access_allowed": False,
        "pass_conditions": {
            "dev_weighted_target_delta_must_be_negative": True,
            "dev_answerable_refusal_rate_delta_must_not_exceed": 0.02,
            "dev_average_token_f1_drop_must_not_exceed": 0.005,
            "dev_gold_doc_citation_rate_drop_must_not_exceed": 0.015,
        },
    }


def _metric_contract() -> dict[str, Any]:
    return {
        "baseline_reference": "Stage102 verified BM25 top10 answer pipeline",
        "primary_train_cv_metric": "weighted_target_bucket_score_delta",
        "target_bucket_weights": dict(_TARGET_BUCKET_WEIGHTS),
        "required_reported_splits": ["train_cv", "train_full", "dev"],
        "required_bucket_counts": [
            "answerability_false_answer",
            "retrieval_context_miss",
            "evidence_selection_miss",
            "verification_over_refusal",
            "gold_span_beats_selected_answer",
            "low_overlap_gold_cited_answer",
            "answer_supported_and_cited",
        ],
        "required_metric_deltas": [
            "answerable_refusal_rate",
            "unanswerable_refusal_rate",
            "gold_doc_citation_rate",
            "average_token_f1",
        ],
        "retrieval_context_miss_policy": {
            "direct_target_for_stage109": False,
            "reason": (
                "Stage107 showed retrieval misses remain, but this redesign "
                "targets answer-pipeline behavior after the Stage84 retrieval "
                "candidate queue was exhausted."
            ),
            "must_report_and_not_increase": True,
        },
    }


def _runtime_feature_contract() -> dict[str, Any]:
    return {
        "allowed_runtime_signals": [
            "question_route",
            "retrieved_document_rank",
            "retrieval_score",
            "evidence_sentence_score",
            "evidence_support_count",
            "question_title_overlap",
            "question_text_overlap",
            "citation_rank",
            "selected_evidence_window_position",
        ],
        "forbidden_runtime_signals": [
            "gold_answer_text",
            "gold_document_identifier",
            "dataset_split_membership",
            "validation_or_test_label",
            "source_candidate_document_identifier_list",
            "raw_private_document_text_as_reported_feature",
        ],
        "runtime_defaults_changed_in_stage108": False,
    }


def _runtime_signal_boundary() -> dict[str, Any]:
    return {
        "uses_gold_labels_at_runtime": False,
        "uses_oracle_document_identifiers_at_runtime": False,
        "uses_test_membership_at_runtime": False,
        "uses_source_candidate_document_lists_at_runtime": False,
        "allowed_signal_family": "retrieval_scores_routes_and_public_runtime_text_overlap",
    }


def _public_safe_output_contract() -> dict[str, Any]:
    return {
        "stage108_writes_case_level_rows": False,
        "stage109_may_write_public_safe_changed_case_samples": True,
        "raw_questions_written": False,
        "raw_answers_written": False,
        "raw_document_identifiers_written": False,
        "raw_document_text_written": False,
        "allowed_stage109_case_fields": [
            "sample_id",
            "split",
            "fold_id",
            "config_id",
            "candidate_family_id",
            "baseline_bucket_id",
            "candidate_bucket_id",
            "baseline_answer_token_f1_bucket",
            "candidate_answer_token_f1_bucket",
            "baseline_citation_status",
            "candidate_citation_status",
            "answerability_action",
            "composition_action",
            "changed_case_confidence_band",
        ],
    }


def _guard_checks(
    *,
    report: Mapping[str, Any],
    stage107_report: Mapping[str, Any],
    stage107_summary: Mapping[str, Any],
    user_confirmed_protocol: bool,
) -> list[dict[str, Any]]:
    frozen = report.get("frozen_protocol") or {}
    grid = frozen.get("candidate_config_grid") or []
    config_ids = [str(config.get("config_id")) for config in grid]
    family_ids = [str(family.get("family_id")) for family in frozen.get("candidate_families") or []]
    guard_checks = stage107_report.get("guard_checks") or []
    train_rule = frozen.get("train_selection_rule") or {}
    dev_rule = frozen.get("dev_validation_rule") or {}
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    return [
        _check(
            name="source_stage107_report_is_stage107",
            passed=stage107_summary.get("stage") == _SOURCE_STAGE107,
            observed=stage107_summary.get("stage"),
            expected=_SOURCE_STAGE107,
        ),
        _check(
            name="source_stage107_protocol_matches",
            passed=stage107_summary.get("protocol_id") == _SOURCE_PROTOCOL_ID,
            observed=stage107_summary.get("protocol_id"),
            expected=_SOURCE_PROTOCOL_ID,
        ),
        _check(
            name="user_confirmed_stage108_protocol",
            passed=user_confirmed_protocol,
            observed=user_confirmed_protocol,
            expected=True,
        ),
        _check(
            name="stage107_analysis_completed",
            passed=stage107_summary.get("decision_status") == _SOURCE_DECISION_STATUS,
            observed=stage107_summary.get("decision_status"),
            expected=_SOURCE_DECISION_STATUS,
        ),
        _check(
            name="stage107_recommends_failure_pattern_redesign",
            passed=stage107_summary.get("recommended_next_direction")
            == "failure_pattern_driven_train_dev_redesign_protocol",
            observed=stage107_summary.get("recommended_next_direction"),
            expected="failure_pattern_driven_train_dev_redesign_protocol",
        ),
        _check(
            name="stage107_all_guard_checks_passed",
            passed=all(check.get("passed") is True for check in guard_checks),
            observed={
                "passed": sum(1 for check in guard_checks if check.get("passed")),
                "total": len(guard_checks),
            },
            expected="all passed",
        ),
        _check(
            name="stage107_final_test_gate_locked",
            passed=stage107_summary.get("can_open_final_test_gate_now") is False
            and stage107_summary.get("can_run_final_test_metrics_now") is False,
            observed={
                "can_open_final_test_gate_now": stage107_summary.get(
                    "can_open_final_test_gate_now"
                ),
                "can_run_final_test_metrics_now": stage107_summary.get(
                    "can_run_final_test_metrics_now"
                ),
            },
            expected=False,
        ),
        _check(
            name="stage107_runtime_defaults_unchanged",
            passed=stage107_summary.get("default_runtime_policy") == "unchanged",
            observed=stage107_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage107_fallback_disabled",
            passed=stage107_summary.get("fallback_strategies_enabled") is False,
            observed=stage107_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="stage108_protocol_status_frozen",
            passed=frozen.get("protocol_status")
            == "frozen_requires_user_confirmation_before_train_dev_run",
            observed=frozen.get("protocol_status"),
            expected="frozen_requires_user_confirmation_before_train_dev_run",
        ),
        _check(
            name="stage108_candidate_families_expected",
            passed=tuple(family_ids) == _CANDIDATE_FAMILY_IDS,
            observed=family_ids,
            expected=list(_CANDIDATE_FAMILY_IDS),
        ),
        _check(
            name="stage108_candidate_grid_has_seven_configs",
            passed=len(grid) == 7,
            observed=len(grid),
            expected=7,
        ),
        _check(
            name="stage108_candidate_config_ids_unique",
            passed=len(config_ids) == len(set(config_ids)),
            observed=config_ids,
            expected="unique config ids",
        ),
        _check(
            name="stage108_does_not_reuse_stopped_stage104_config_ids",
            passed=not any(
                config_id.startswith(prefix)
                for config_id in config_ids
                for prefix in _STOPPED_STAGE104_PREFIXES
            ),
            observed=config_ids,
            expected="no amg_/ewr_/jgw_ stopped config ids",
        ),
        _check(
            name="stage108_train_selection_uses_grouped_cv",
            passed=train_rule.get("selection_split") == "train"
            and train_rule.get("selection_mode")
            == "train_grouped_cross_validation_then_full_train_refit"
            and int(train_rule.get("train_cv_fold_count") or 0) == 5,
            observed=train_rule,
            expected="train-only grouped CV with 5 folds",
        ),
        _check(
            name="stage108_requires_negative_train_cv_delta",
            passed=(train_rule.get("objective") or {}).get(
                "requires_negative_train_cv_weighted_delta"
            )
            is True
            and (train_rule.get("objective") or {}).get("no_op_candidate_selectable")
            is False,
            observed=train_rule.get("objective") or {},
            expected="negative train-CV delta required; no-op not selectable",
        ),
        _check(
            name="stage108_dev_validation_forbids_selection_and_retuning",
            passed=dev_rule.get("dev_selection_allowed") is False
            and dev_rule.get("dev_retuning_allowed") is False
            and dev_rule.get("dev_threshold_tuning_allowed") is False
            and dev_rule.get("test_access_allowed") is False,
            observed=dev_rule,
            expected="dev validation only; no dev selection or tuning; no test",
        ),
        _check(
            name="stage108_test_split_locked",
            passed=(report.get("split_contract") or {}).get("forbidden_final_splits")
            == list(_FORBIDDEN_FINAL_SPLITS),
            observed=(report.get("split_contract") or {}).get("forbidden_final_splits"),
            expected=list(_FORBIDDEN_FINAL_SPLITS),
        ),
        _check(
            name="stage108_runtime_defaults_unchanged",
            passed=(frozen.get("runtime_feature_contract") or {}).get(
                "runtime_defaults_changed_in_stage108"
            )
            is False,
            observed=(frozen.get("runtime_feature_contract") or {}).get(
                "runtime_defaults_changed_in_stage108"
            ),
            expected=False,
        ),
        _check(
            name="stage108_fallback_strategies_not_added",
            passed=(frozen.get("fallback_strategy_policy") or {}).get(
                "fallback_strategies_enabled"
            )
            is False,
            observed=frozen.get("fallback_strategy_policy") or {},
            expected=False,
        ),
        _check(
            name="stage108_output_has_no_forbidden_public_keys",
            passed=not _contains_forbidden_key(report),
            observed=sorted(_forbidden_keys_found(report)),
            expected=[],
        ),
        _check(
            name="stage108_output_has_no_private_fixture_markers",
            passed="private-doc-" not in serialized
            and "Private fixture answer text" not in serialized,
            observed="private marker present"
            if "private-doc-" in serialized or "Private fixture answer text" in serialized
            else "none",
            expected="none",
        ),
        _check(
            name="stage108_no_metric_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_failure_pattern_redesign_protocol_blocked",
            "failed_checks": failed_checks,
            "protocol_id": _PROTOCOL_ID,
            "can_run_train_dev_comparison_after_user_confirmation": False,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": "primeqa_hybrid_failure_pattern_redesign_protocol_frozen",
        "protocol_id": _PROTOCOL_ID,
        "recommended_next_direction": _NEXT_DIRECTION,
        "candidate_family_count": len(_CANDIDATE_FAMILY_IDS),
        "candidate_config_count": 7,
        "train_selection_mode": "train_grouped_cross_validation_then_full_train_refit",
        "can_run_train_dev_comparison_after_user_confirmation": True,
        "can_continue_train_dev_development": True,
        "requires_user_confirmation_before_train_dev_run": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage109: after user confirmation, implement the frozen candidate "
            "components and run the train grouped-CV plus dev validation "
            "comparison. Select only from train-CV, validate once on dev, keep "
            "test locked, keep runtime defaults unchanged, and add no fallback "
            "strategies."
        ),
    }


def _family_priority_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    families = (report.get("frozen_protocol") or {}).get("candidate_families") or []
    return [
        BarDatum(
            label=str(family.get("family_id")),
            value=float(family.get("priority_score") or 0.0),
            value_label=f"{float(family.get('priority_score') or 0.0):.4f}",
        )
        for family in families
    ]


def _candidate_config_count_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counts: Counter[str] = Counter(
        str(config.get("candidate_family_id"))
        for config in (report.get("frozen_protocol") or {}).get("candidate_config_grid")
        or []
    )
    return [
        BarDatum(label=family_id, value=float(count), value_label=str(count))
        for family_id, count in sorted(counts.items())
    ]


def _target_weight_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    weights = (
        ((report.get("frozen_protocol") or {}).get("train_selection_rule") or {})
        .get("objective", {})
        .get("weighted_target_bucket_score", {})
    )
    return [
        BarDatum(
            label=str(bucket),
            value=float(weight),
            value_label=f"{float(weight):.2f}",
        )
        for bucket, weight in weights.items()
    ]


def _train_guard_threshold_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    guards = (
        ((report.get("frozen_protocol") or {}).get("train_selection_rule") or {})
        .get("selectability_guards", {})
    )
    return [
        BarDatum(
            label=str(name),
            value=float(value),
            value_label=str(value),
        )
        for name, value in guards.items()
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    names = [
        "can_run_train_dev_comparison_after_user_confirmation",
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


def _optional_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


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
            if str(key) in _FORBIDDEN_PUBLIC_KEYS:
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
