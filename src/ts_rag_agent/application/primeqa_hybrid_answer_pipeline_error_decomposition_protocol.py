from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 101"
_CREATED_AT = "2026-07-15"
_SOURCE_STAGE100 = "Stage 100"
_PROTOCOL_ID = "answer_pipeline_error_decomposition_train_dev_v1"
_RECOMMENDED_DIRECTION = "answer_pipeline_error_decomposition"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = ("test",)
_EXPECTED_BUCKET_ORDER = (
    "answerability_false_answer",
    "retrieval_context_miss",
    "evidence_selection_miss",
    "verification_over_refusal",
    "gold_span_beats_selected_answer",
    "low_overlap_gold_cited_answer",
    "answer_supported_and_cited",
)
_PUBLIC_SAFE_CASE_FIELDS = (
    "sample_id",
    "split",
    "answerability_label",
    "pipeline_bucket_id",
    "pipeline_stage",
    "retrieval_rank_bucket",
    "retrieval_context_status",
    "citation_status",
    "evidence_selection_status",
    "answer_token_f1_bucket",
    "best_gold_span_f1_bucket",
    "answer_gold_span_gap_bucket",
    "verifier_decision",
    "refusal_reason_code",
    "question_route",
    "evidence_selector_name",
    "composition_policy_id",
    "bucket_confidence_band",
)
_FORBIDDEN_PUBLIC_CASE_FIELDS = frozenset(
    {
        "question_text",
        "question_title",
        "raw_question_text",
        "raw_answer_text",
        "gold_answer",
        "answer_text",
        "document_title",
        "document_body",
        "document_text",
        "document_id",
        "answer_doc_id",
        "retrieved_doc_ids",
        "cited_doc_ids",
        "source_doc_ids",
        "matched_token_strings",
        "query_terms",
    }
)


@dataclass(frozen=True)
class PrimeQAHybridAnswerPipelineProtocolVisualization:
    """One generated Stage101 answer-pipeline protocol visualization."""

    name: str
    path: str


def freeze_primeqa_hybrid_answer_pipeline_error_decomposition_protocol(
    *,
    stage100_report_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the train/dev protocol for answer-pipeline error decomposition."""

    started_at = time.perf_counter()
    stage100_report = _load_json_object(stage100_report_path)
    stage100_summary = _stage100_public_summary(stage100_report)
    frozen_protocol = _frozen_protocol()
    guard_checks = _guard_checks(
        stage100_summary=stage100_summary,
        frozen_protocol=frozen_protocol,
        user_confirmed_protocol=user_confirmed_protocol,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_scope": (
            "Train/dev-only protocol freeze for answer-pipeline error "
            "decomposition after Stage100 showed that first-wave and "
            "second-wave retrieval route families are exhausted. This stage "
            "reads only the public-safe Stage100 summary, defines the next "
            "analysis contract, does not load split files, does not run "
            "retrieval or answer metrics, does not run final metrics, does not "
            "use document identifiers as runtime evidence, does not add "
            "fallback strategies, and does not change runtime defaults."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_protocol),
            "confirmation_note": confirmation_note,
        },
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "development_splits": list(_DEVELOPMENT_SPLITS),
            "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage100_report": _fingerprint(stage100_report_path),
        },
        "stage100_summary": stage100_summary,
        "frozen_protocol": frozen_protocol,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
        "timing_seconds": {
            "load_and_freeze": round(checked_at - started_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_answer_pipeline_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridAnswerPipelineProtocolVisualization]:
    """Write SVG charts for the Stage101 answer-pipeline protocol."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage101_error_bucket_priority_weights.svg": render_horizontal_bar_chart_svg(
            title="Stage101 answer-pipeline bucket priority weights",
            bars=_bucket_priority_bars(report),
            x_label="priority weight",
            width=1280,
            margin_left=520,
        ),
        "stage101_pipeline_stage_order.svg": render_horizontal_bar_chart_svg(
            title="Stage101 answer-pipeline bucket order",
            bars=_pipeline_stage_order_bars(report),
            x_label="assignment precedence",
            width=1260,
            margin_left=520,
        ),
        "stage101_public_case_field_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage101 public-safe case field groups",
            bars=_public_case_field_bars(report),
            x_label="field count",
            width=1180,
            margin_left=460,
        ),
        "stage101_output_artifact_contract.svg": render_horizontal_bar_chart_svg(
            title="Stage101 output artifact contract",
            bars=_output_artifact_bars(report),
            x_label="planned field count",
            width=1240,
            margin_left=540,
        ),
        "stage101_protocol_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage101 protocol decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1180,
            margin_left=560,
        ),
        "stage101_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage101 answer-pipeline guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1460,
            margin_left=760,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridAnswerPipelineProtocolVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage100_public_summary(stage100_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage100_report.get("decision") or {}
    aggregate = stage100_report.get("aggregate_summary") or {}
    return {
        "stage": stage100_report.get("stage"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "first_wave_retrieval_candidates_exhausted": decision.get(
            "first_wave_retrieval_candidates_exhausted"
        ),
        "second_wave_retrieval_route_family_exhausted": decision.get(
            "second_wave_retrieval_route_family_exhausted"
        ),
        "second_wave_expected_candidate_count": aggregate.get(
            "second_wave_expected_candidate_count"
        ),
        "second_wave_stopped_candidate_count": aggregate.get(
            "second_wave_stopped_candidate_count"
        ),
        "runtime_advancing_second_wave_candidate_count": decision.get(
            "runtime_advancing_second_wave_candidate_count"
        ),
        "remaining_actionable_candidate_count": decision.get(
            "remaining_actionable_candidate_count"
        ),
        "best_second_wave_dev_hit10_delta": aggregate.get(
            "best_second_wave_dev_hit10_delta"
        ),
        "best_second_wave_top10_net": aggregate.get("best_second_wave_top10_net"),
        "blocked_source_document_diagnostic_status": aggregate.get(
            "blocked_source_doc_ids_diagnostic_status"
        ),
        "requires_user_confirmation_before_next_protocol": decision.get(
            "requires_user_confirmation_before_next_protocol"
        ),
        "can_continue_train_dev_development": decision.get(
            "can_continue_train_dev_development"
        ),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get(
            "can_run_final_test_metrics_now"
        ),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _frozen_protocol() -> dict[str, Any]:
    buckets = _decomposition_buckets()
    return {
        "protocol_id": _PROTOCOL_ID,
        "recommended_direction": _RECOMMENDED_DIRECTION,
        "protocol_status": "frozen_requires_user_confirmation_before_analysis_run",
        "source_stages": [_SOURCE_STAGE100],
        "objective": (
            "Classify remaining train/dev answer-pipeline failures after "
            "retrieval-route exhaustion into deterministic public-safe buckets "
            "before proposing another intervention."
        ),
        "analysis_mode": "protocol_freeze_only",
        "allowed_input_contract": {
            "stage101_inputs": [
                "Stage100 public-safe route exhaustion summary.",
            ],
            "stage102_inputs_after_confirmation": [
                "Train/dev-only answer-pipeline traces regenerated or read under "
                "this public-safe output contract.",
                "Gold labels used only for metric scoring after predictions are "
                "produced, never as runtime features.",
                "Existing local train/dev artifacts only when they can be "
                "sanitized to the public-safe field contract before reporting.",
            ],
            "forbidden_inputs": [
                "test split rows",
                "final-test metric files",
                "runtime features derived from label-only document identifiers",
                "raw question, answer, document, snippet, or matched-token text in reports",
            ],
        },
        "bucket_assignment_contract": {
            "assignment_precedence": list(_EXPECTED_BUCKET_ORDER),
            "buckets": buckets,
            "multi_label_allowed": False,
            "tie_break_rule": "first matching bucket by assignment_precedence",
        },
        "public_safe_output_contract": {
            "aggregate_outputs": [
                "bucket counts by split",
                "bucket rates by split",
                "pipeline-stage counts by split",
                "route and bucket cross-tab counts",
                "answerability and bucket cross-tab counts",
                "token-F1 bucket distributions",
                "retrieval-rank bucket distributions",
                "verification decision distributions",
            ],
            "case_sample_fields": list(_PUBLIC_SAFE_CASE_FIELDS),
            "case_sample_limit_per_bucket": 5,
            "case_sample_sort_key": [
                "pipeline_bucket_id",
                "answer_gold_span_gap_bucket",
                "answer_token_f1_bucket",
                "sample_id",
            ],
            "private_label_inputs_not_written": [
                "gold answer text used only to compute token-F1 buckets and not written",
                (
                    "gold answer document identifier used only to compute "
                    "retrieval and citation status and not written"
                ),
                (
                    "retrieved and cited document identifiers used only to "
                    "compute status buckets and not written"
                ),
            ],
        },
        "train_dev_execution_rule": {
            "allowed_splits": list(_DEVELOPMENT_SPLITS),
            "selection_split": "train",
            "validation_split": "dev",
            "dev_threshold_selection_forbidden": True,
            "test_selection_forbidden": True,
            "final_test_metrics_forbidden": True,
            "runtime_default_change_forbidden": True,
            "next_intervention_selection_rule": (
                "Use train bucket mass and severity to draft candidate fixes; "
                "use dev only to validate that the same bottleneck exists. Do "
                "not tune thresholds or defaults on dev."
            ),
        },
        "fallback_strategy_policy": {
            "fallback_strategies_enabled": False,
            "policy": (
                "If required train/dev artifacts are missing or unsafe, the "
                "Stage102 analysis blocks instead of switching to another path."
            ),
        },
        "explicit_exclusions": [
            "Do not read, score, or summarize the test split.",
            "Do not run final metrics.",
            "Do not use label-only document identifiers as runtime evidence.",
            "Do not write raw question, answer, document, snippet, or token-match text.",
            "Do not choose thresholds, policies, or runtime defaults from dev.",
            "Do not add fallback strategies in Stage101.",
            "Do not change runtime defaults.",
        ],
        "recommended_stage102_outputs": _recommended_stage102_outputs(),
    }


def _decomposition_buckets() -> list[dict[str, Any]]:
    return [
        {
            "bucket_id": "answerability_false_answer",
            "pipeline_stage": "answerability",
            "assignment_precedence": 1,
            "priority_weight": 1.55,
            "diagnostic_question": (
                "Did the system answer an unanswerable train/dev question?"
            ),
            "aggregate_signals": [
                "unanswerable_answered_count",
                "unanswerable_answered_rate",
                "refusal_reason_distribution",
            ],
            "candidate_intervention_family": "answerability_or_evidence_sufficiency_gate",
        },
        {
            "bucket_id": "retrieval_context_miss",
            "pipeline_stage": "retrieval",
            "assignment_precedence": 2,
            "priority_weight": 1.35,
            "diagnostic_question": (
                "For answerable questions, was the gold evidence absent from "
                "the retrieved context?"
            ),
            "aggregate_signals": [
                "gold_context_absent_count",
                "retrieval_rank_bucket_distribution",
                "not_found_context_rate",
            ],
            "candidate_intervention_family": "retrieval_only_if_new_signal_exists",
        },
        {
            "bucket_id": "evidence_selection_miss",
            "pipeline_stage": "evidence_selection",
            "assignment_precedence": 3,
            "priority_weight": 1.70,
            "diagnostic_question": (
                "Was gold evidence present in context but not selected or cited?"
            ),
            "aggregate_signals": [
                "gold_context_present_not_selected_count",
                "selector_route_distribution",
                "citation_status_distribution",
            ],
            "candidate_intervention_family": "evidence_selector_or_reranker",
        },
        {
            "bucket_id": "verification_over_refusal",
            "pipeline_stage": "verification",
            "assignment_precedence": 4,
            "priority_weight": 1.20,
            "diagnostic_question": (
                "Was an answerable item refused despite selected gold evidence?"
            ),
            "aggregate_signals": [
                "answerable_refused_gold_present_count",
                "near_threshold_refusal_bucket_count",
                "verifier_reason_distribution",
            ],
            "candidate_intervention_family": "verification_threshold_or_calibration",
        },
        {
            "bucket_id": "gold_span_beats_selected_answer",
            "pipeline_stage": "answer_composition",
            "assignment_precedence": 5,
            "priority_weight": 1.45,
            "diagnostic_question": (
                "Did a public-safe gold-span score bucket beat the selected "
                "answer score bucket?"
            ),
            "aggregate_signals": [
                "gold_span_gap_bucket_distribution",
                "composition_policy_distribution",
                "selected_answer_f1_bucket_distribution",
            ],
            "candidate_intervention_family": "answer_composition_or_span_selection",
        },
        {
            "bucket_id": "low_overlap_gold_cited_answer",
            "pipeline_stage": "answer_composition",
            "assignment_precedence": 6,
            "priority_weight": 1.10,
            "diagnostic_question": (
                "Was gold evidence cited but the final answer still low-overlap?"
            ),
            "aggregate_signals": [
                "gold_cited_low_answer_f1_count",
                "answer_length_bucket_distribution",
                "composition_policy_distribution",
            ],
            "candidate_intervention_family": "answer_synthesis_or_sentence_windowing",
        },
        {
            "bucket_id": "answer_supported_and_cited",
            "pipeline_stage": "non_error_reference",
            "assignment_precedence": 7,
            "priority_weight": 0.25,
            "diagnostic_question": (
                "Was the answer supported and cited enough to serve as a "
                "reference slice?"
            ),
            "aggregate_signals": [
                "supported_reference_count",
                "supported_reference_rate",
            ],
            "candidate_intervention_family": "no_fix_reference_slice",
        },
    ]


def _recommended_stage102_outputs() -> list[dict[str, Any]]:
    return [
        {
            "artifact_id": "aggregate_bucket_summary",
            "planned_field_count": 8,
            "contains_case_samples": False,
        },
        {
            "artifact_id": "split_bucket_summary",
            "planned_field_count": 8,
            "contains_case_samples": False,
        },
        {
            "artifact_id": "route_bucket_cross_tab",
            "planned_field_count": 9,
            "contains_case_samples": False,
        },
        {
            "artifact_id": "public_safe_case_samples",
            "planned_field_count": len(_PUBLIC_SAFE_CASE_FIELDS),
            "contains_case_samples": True,
        },
        {
            "artifact_id": "guard_and_safety_report",
            "planned_field_count": 6,
            "contains_case_samples": False,
        },
    ]


def _guard_checks(
    *,
    stage100_summary: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
    user_confirmed_protocol: bool,
) -> list[dict[str, Any]]:
    decision_inputs = {
        "stage": stage100_summary.get("stage"),
        "decision_status": stage100_summary.get("decision_status"),
        "recommended_next_direction": stage100_summary.get(
            "recommended_next_direction"
        ),
        "first_wave_retrieval_candidates_exhausted": stage100_summary.get(
            "first_wave_retrieval_candidates_exhausted"
        ),
        "second_wave_retrieval_route_family_exhausted": stage100_summary.get(
            "second_wave_retrieval_route_family_exhausted"
        ),
        "runtime_advancing_second_wave_candidate_count": stage100_summary.get(
            "runtime_advancing_second_wave_candidate_count"
        ),
        "remaining_actionable_candidate_count": stage100_summary.get(
            "remaining_actionable_candidate_count"
        ),
    }
    split_contract = frozen_protocol.get("train_dev_execution_rule") or {}
    output_contract = frozen_protocol.get("public_safe_output_contract") or {}
    case_fields = output_contract.get("case_sample_fields") or []
    bucket_contract = frozen_protocol.get("bucket_assignment_contract") or {}
    buckets = bucket_contract.get("buckets") or []
    bucket_ids = [str(bucket.get("bucket_id")) for bucket in buckets]
    precedence_values = [
        int(bucket.get("assignment_precedence") or 0) for bucket in buckets
    ]
    explicit_exclusions = frozen_protocol.get("explicit_exclusions") or []
    fallback_policy = frozen_protocol.get("fallback_strategy_policy") or {}
    return [
        _check(
            name="stage100_source_is_expected_stage",
            passed=stage100_summary.get("stage") == _SOURCE_STAGE100,
            observed=stage100_summary.get("stage"),
            expected=_SOURCE_STAGE100,
        ),
        _check(
            name="user_confirmed_stage101_protocol",
            passed=user_confirmed_protocol,
            observed=user_confirmed_protocol,
            expected=True,
        ),
        _check(
            name="stage100_completed",
            passed=stage100_summary.get("decision_status")
            == "primeqa_hybrid_second_wave_route_exhaustion_summary_completed",
            observed=stage100_summary.get("decision_status"),
            expected="primeqa_hybrid_second_wave_route_exhaustion_summary_completed",
        ),
        _check(
            name="stage100_recommends_answer_pipeline_decomposition",
            passed=stage100_summary.get("recommended_next_direction")
            == _RECOMMENDED_DIRECTION,
            observed=stage100_summary.get("recommended_next_direction"),
            expected=_RECOMMENDED_DIRECTION,
        ),
        _check(
            name="stage100_first_wave_exhausted",
            passed=stage100_summary.get("first_wave_retrieval_candidates_exhausted")
            is True,
            observed=stage100_summary.get("first_wave_retrieval_candidates_exhausted"),
            expected=True,
        ),
        _check(
            name="stage100_second_wave_exhausted",
            passed=stage100_summary.get("second_wave_retrieval_route_family_exhausted")
            is True,
            observed=stage100_summary.get(
                "second_wave_retrieval_route_family_exhausted"
            ),
            expected=True,
        ),
        _check(
            name="stage100_has_no_runtime_advancing_retrieval_candidate",
            passed=int(
                stage100_summary.get("runtime_advancing_second_wave_candidate_count")
                or 0
            )
            == 0,
            observed=decision_inputs,
            expected="0 runtime-advancing retrieval candidates",
        ),
        _check(
            name="stage100_has_no_remaining_actionable_retrieval_candidate",
            passed=int(stage100_summary.get("remaining_actionable_candidate_count") or 0)
            == 0,
            observed=decision_inputs,
            expected="0 remaining retrieval candidates",
        ),
        _check(
            name="stage100_final_test_gate_closed",
            passed=stage100_summary.get("can_open_final_test_gate_now") is False,
            observed=stage100_summary.get("can_open_final_test_gate_now"),
            expected=False,
        ),
        _check(
            name="stage100_final_metrics_locked",
            passed=stage100_summary.get("can_run_final_test_metrics_now") is False,
            observed=stage100_summary.get("can_run_final_test_metrics_now"),
            expected=False,
        ),
        _check(
            name="stage100_forbids_test_tuning",
            passed=stage100_summary.get("can_use_test_for_tuning") is False,
            observed=stage100_summary.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage100_runtime_default_unchanged",
            passed=stage100_summary.get("default_runtime_policy") == "unchanged",
            observed=stage100_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="protocol_id_is_fixed",
            passed=frozen_protocol.get("protocol_id") == _PROTOCOL_ID,
            observed=frozen_protocol.get("protocol_id"),
            expected=_PROTOCOL_ID,
        ),
        _check(
            name="protocol_requires_confirmation_before_analysis_run",
            passed=frozen_protocol.get("protocol_status")
            == "frozen_requires_user_confirmation_before_analysis_run",
            observed=frozen_protocol.get("protocol_status"),
            expected="frozen_requires_user_confirmation_before_analysis_run",
        ),
        _check(
            name="split_contract_is_train_dev_only",
            passed=tuple(split_contract.get("allowed_splits") or ())
            == _DEVELOPMENT_SPLITS,
            observed=split_contract.get("allowed_splits"),
            expected=list(_DEVELOPMENT_SPLITS),
        ),
        _check(
            name="test_split_is_forbidden",
            passed=split_contract.get("test_selection_forbidden") is True
            and split_contract.get("final_test_metrics_forbidden") is True,
            observed=split_contract,
            expected="test selection and final metrics forbidden",
        ),
        _check(
            name="expected_decomposition_buckets_are_frozen",
            passed=tuple(bucket_ids) == _EXPECTED_BUCKET_ORDER,
            observed=bucket_ids,
            expected=list(_EXPECTED_BUCKET_ORDER),
        ),
        _check(
            name="bucket_precedence_is_deterministic_and_unique",
            passed=precedence_values == list(range(1, len(_EXPECTED_BUCKET_ORDER) + 1))
            and len(set(precedence_values)) == len(precedence_values),
            observed=precedence_values,
            expected=list(range(1, len(_EXPECTED_BUCKET_ORDER) + 1)),
        ),
        _check(
            name="public_case_fields_are_whitelisted",
            passed=tuple(case_fields) == _PUBLIC_SAFE_CASE_FIELDS,
            observed=case_fields,
            expected=list(_PUBLIC_SAFE_CASE_FIELDS),
        ),
        _check(
            name="public_case_fields_exclude_private_text_and_document_ids",
            passed=_forbidden_public_fields_absent(case_fields),
            observed=case_fields,
            expected="no raw text, document identifier, or token string fields",
        ),
        _check(
            name="metric_labels_allowed_only_after_prediction",
            passed=_private_label_inputs_are_metric_only(output_contract),
            observed=output_contract.get("private_label_inputs_not_written"),
            expected="labels used only for scoring buckets and never written",
        ),
        _check(
            name="source_document_identifiers_forbidden_as_runtime_evidence",
            passed=any(
                "document identifiers as runtime evidence" in str(item)
                for item in explicit_exclusions
            ),
            observed=explicit_exclusions,
            expected="document identifiers forbidden as runtime evidence",
        ),
        _check(
            name="fallback_strategies_are_disabled",
            passed=fallback_policy.get("fallback_strategies_enabled") is False,
            observed=fallback_policy,
            expected=False,
        ),
        _check(
            name="stage101_freezes_protocol_without_analysis_metrics",
            passed=True,
            observed="protocol_freeze_only",
            expected="protocol_freeze_only",
        ),
        _check(
            name="stage101_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage101_default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_answer_pipeline_error_decomposition_protocol_blocked",
            "protocol_id": _PROTOCOL_ID,
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "requires_user_confirmation_before_train_dev_analysis": True,
            "can_run_train_dev_error_decomposition_after_user_confirmation": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": "primeqa_hybrid_answer_pipeline_error_decomposition_protocol_frozen",
        "protocol_id": _PROTOCOL_ID,
        "recommended_direction": _RECOMMENDED_DIRECTION,
        "can_continue_train_dev_development": True,
        "requires_user_confirmation_before_train_dev_analysis": True,
        "can_run_train_dev_error_decomposition_after_user_confirmation": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage102: after user confirmation, run the frozen train/dev-only "
            "answer-pipeline error decomposition analysis with public-safe "
            "aggregate and sanitized case outputs; keep test locked and runtime "
            "defaults unchanged."
        ),
    }


def _bucket_priority_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    buckets = report["frozen_protocol"]["bucket_assignment_contract"]["buckets"]
    return [
        BarDatum(
            label=str(bucket["bucket_id"]),
            value=float(bucket["priority_weight"]),
            value_label=f"{float(bucket['priority_weight']):.2f}",
        )
        for bucket in buckets
    ]


def _pipeline_stage_order_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    buckets = report["frozen_protocol"]["bucket_assignment_contract"]["buckets"]
    return [
        BarDatum(
            label=str(bucket["bucket_id"]),
            value=float(bucket["assignment_precedence"]),
            value_label=str(bucket["assignment_precedence"]),
        )
        for bucket in buckets
    ]


def _public_case_field_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    fields = report["frozen_protocol"]["public_safe_output_contract"][
        "case_sample_fields"
    ]
    groups = {
        "identity_and_split": [
            field for field in fields if field in {"sample_id", "split"}
        ],
        "bucket_and_stage": [
            field
            for field in fields
            if field in {"pipeline_bucket_id", "pipeline_stage", "bucket_confidence_band"}
        ],
        "retrieval_and_citation_status": [
            field
            for field in fields
            if field
            in {
                "retrieval_rank_bucket",
                "retrieval_context_status",
                "citation_status",
                "evidence_selection_status",
            }
        ],
        "answer_quality_buckets": [
            field
            for field in fields
            if field
            in {
                "answer_token_f1_bucket",
                "best_gold_span_f1_bucket",
                "answer_gold_span_gap_bucket",
                "answerability_label",
            }
        ],
        "policy_and_verifier_codes": [
            field
            for field in fields
            if field
            in {
                "verifier_decision",
                "refusal_reason_code",
                "question_route",
                "evidence_selector_name",
                "composition_policy_id",
            }
        ],
    }
    return [
        BarDatum(label=group, value=float(len(group_fields)), value_label=str(len(group_fields)))
        for group, group_fields in groups.items()
    ]


def _output_artifact_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    outputs = report["frozen_protocol"]["recommended_stage102_outputs"]
    return [
        BarDatum(
            label=str(output["artifact_id"]),
            value=float(output["planned_field_count"]),
            value_label=str(output["planned_field_count"]),
        )
        for output in outputs
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report["decision"]
    names = [
        "can_run_train_dev_error_decomposition_after_user_confirmation",
        "can_run_final_test_metrics_now",
        "can_use_test_for_tuning",
        "fallback_strategies_enabled",
    ]
    return [
        BarDatum(
            label=name,
            value=1.0 if decision[name] else 0.0,
            value_label="yes" if decision[name] else "no",
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
        for check in report["guard_checks"]
    ]


def _forbidden_public_fields_absent(case_fields: Sequence[Any]) -> bool:
    normalized_fields = {str(field).lower() for field in case_fields}
    return not (normalized_fields & _FORBIDDEN_PUBLIC_CASE_FIELDS)


def _private_label_inputs_are_metric_only(
    output_contract: Mapping[str, Any],
) -> bool:
    entries = output_contract.get("private_label_inputs_not_written") or []
    if len(entries) != 3:
        return False
    return all("only" in str(entry) and "not written" in str(entry) for entry in entries)


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
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
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
