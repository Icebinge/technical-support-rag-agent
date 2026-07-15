from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 103"
_CREATED_AT = "2026-07-15"
_SOURCE_STAGE102 = "Stage 102"
_DESIGN_ID = "evidence_selection_and_answerability_candidate_design_v1"
_SOURCE_ANALYSIS_ID = "answer_pipeline_error_decomposition_train_dev_analysis_v1"
_SOURCE_RECOMMENDED_DIRECTION = "evidence_selection_and_answerability_candidate_design"
_NEXT_DIRECTION = "evidence_answerability_train_dev_candidate_comparison"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = ("test",)
_PRIMARY_BUCKETS = (
    "answerability_false_answer",
    "gold_span_beats_selected_answer",
)
_SECONDARY_BUCKETS = (
    "evidence_selection_miss",
    "retrieval_context_miss",
)
_REPORT_BUCKET_ORDER = _PRIMARY_BUCKETS + _SECONDARY_BUCKETS
_FORBIDDEN_PUBLIC_FIELDS = frozenset(
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
_FORBIDDEN_RUNTIME_FEATURE_TOKENS = frozenset(
    {
        "gold",
        "answer_doc_id",
        "source_doc",
        "doc_id",
        "test",
        "label",
        "raw_",
        "oracle",
    }
)


@dataclass(frozen=True)
class PrimeQAHybridEvidenceAnswerabilityCandidateProtocolVisualization:
    """One generated Stage103 evidence/answerability protocol visualization."""

    name: str
    path: str


@dataclass(frozen=True)
class _CandidatePolicySpec:
    candidate_id: str
    name: str
    intervention_focus: str
    status: str
    risk_level: str
    implementation_readiness: float
    target_buckets: tuple[str, ...]
    rationale: str
    runtime_feature_groups: Mapping[str, tuple[str, ...]]
    stage104_protocol_outline: tuple[str, ...]
    target_metric_contract: tuple[str, ...]
    safety_contract: tuple[str, ...]


def freeze_primeqa_hybrid_evidence_answerability_candidate_protocol(
    *,
    stage102_report_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze Stage103 train/dev-only evidence and answerability design."""

    started_at = time.perf_counter()
    stage102_report = _load_json_object(stage102_report_path)
    stage102_summary = _stage102_public_summary(stage102_report)
    bottleneck_summary = _bottleneck_summary(stage102_summary)
    frozen_candidate_protocol = _frozen_candidate_protocol(bottleneck_summary)
    guard_checks = _guard_checks(
        stage102_summary=stage102_summary,
        bottleneck_summary=bottleneck_summary,
        frozen_candidate_protocol=frozen_candidate_protocol,
        user_confirmed_protocol=user_confirmed_protocol,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "design_id": _DESIGN_ID,
        "design_scope": (
            "Train/dev-only candidate protocol freeze for the shared "
            "answerability and evidence-selection bottlenecks found in "
            "Stage102. This stage reads only the saved public-safe Stage102 "
            "aggregate report, does not load split files, does not load corpus "
            "documents, does not run retrieval or answer metrics, does not run "
            "final metrics, does not use oracle document identifiers or gold "
            "answers as runtime evidence, does not add fallback strategies, and "
            "does not change runtime defaults."
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
            "stage102_report": _fingerprint(stage102_report_path),
        },
        "stage102_summary": stage102_summary,
        "bottleneck_summary": bottleneck_summary,
        "frozen_candidate_protocol": frozen_candidate_protocol,
        "guard_checks": guard_checks,
        "decision": _decision(
            guard_checks=guard_checks,
            frozen_candidate_protocol=frozen_candidate_protocol,
        ),
        "timing_seconds": {
            "load_and_design": round(checked_at - started_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_evidence_answerability_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridEvidenceAnswerabilityCandidateProtocolVisualization]:
    """Write SVG charts for the Stage103 candidate protocol."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage103_shared_bottleneck_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage103 shared bottleneck counts",
            bars=_bottleneck_count_bars(report),
            x_label="case count",
            width=1320,
            margin_left=560,
        ),
        "stage103_shared_bottleneck_rates.svg": render_horizontal_bar_chart_svg(
            title="Stage103 shared bottleneck rates",
            bars=_bottleneck_rate_bars(report),
            x_label="rate",
            width=1320,
            margin_left=560,
        ),
        "stage103_candidate_priority_scores.svg": render_horizontal_bar_chart_svg(
            title="Stage103 candidate priority scores",
            bars=_candidate_priority_bars(report),
            x_label="design priority score",
            width=1320,
            margin_left=560,
        ),
        "stage103_candidate_target_case_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage103 candidate target case counts",
            bars=_candidate_target_case_bars(report),
            x_label="train/dev target case count",
            width=1320,
            margin_left=560,
        ),
        "stage103_candidate_feature_group_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage103 candidate runtime feature groups",
            bars=_candidate_feature_group_bars(report),
            x_label="allowed runtime feature count",
            width=1320,
            margin_left=560,
        ),
        "stage103_protocol_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage103 protocol decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1320,
            margin_left=660,
        ),
        "stage103_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage103 guard checks",
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
            PrimeQAHybridEvidenceAnswerabilityCandidateProtocolVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage102_public_summary(stage102_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage102_report.get("decision") or {}
    data_summary = stage102_report.get("data_summary") or {}
    aggregate_outputs = stage102_report.get("aggregate_outputs") or {}
    metrics_by_split = stage102_report.get("metrics_by_split") or {}
    guard_checks = stage102_report.get("guard_checks") or []
    return {
        "stage": stage102_report.get("stage"),
        "analysis_id": stage102_report.get("analysis_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
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
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "train_top_bucket": decision.get("train_top_bucket"),
        "dev_top_bucket": decision.get("dev_top_bucket"),
        "train_answerability_false_answer": decision.get(
            "train_answerability_false_answer"
        ),
        "dev_answerability_false_answer": decision.get(
            "dev_answerability_false_answer"
        ),
        "train_evidence_selection_miss": decision.get(
            "train_evidence_selection_miss"
        ),
        "dev_evidence_selection_miss": decision.get("dev_evidence_selection_miss"),
        "train_verified_average_token_f1": decision.get(
            "train_verified_average_token_f1"
        ),
        "dev_verified_average_token_f1": decision.get(
            "dev_verified_average_token_f1"
        ),
        "data_summary": _public_data_summary(data_summary),
        "metrics_by_split": _public_metric_summary(metrics_by_split),
        "bucket_counts_by_split": _select_bucket_mapping(
            aggregate_outputs.get("bucket_counts_by_split") or {}
        ),
        "bucket_rates_by_split": _select_bucket_mapping(
            aggregate_outputs.get("bucket_rates_by_split") or {}
        ),
        "top_priority_buckets": _select_top_priority_buckets(
            aggregate_outputs.get("top_priority_buckets") or {}
        ),
        "stage102_guard_check_count": len(guard_checks),
        "stage102_guard_check_passed_count": sum(
            1 for check in guard_checks if check.get("passed") is True
        ),
    }


def _public_data_summary(data_summary: Mapping[str, Any]) -> dict[str, Any]:
    splits = data_summary.get("splits") or {}
    return {
        "documents": data_summary.get("documents"),
        "splits": {
            split: {
                "row_count": split_summary.get("row_count"),
                "answerable_count": split_summary.get("answerable_count"),
                "unanswerable_count": split_summary.get("unanswerable_count"),
            }
            for split, split_summary in splits.items()
            if split in _DEVELOPMENT_SPLITS
        },
    }


def _public_metric_summary(metrics_by_split: Mapping[str, Any]) -> dict[str, Any]:
    summary = {}
    for split in _DEVELOPMENT_SPLITS:
        split_metrics = metrics_by_split.get(split) or {}
        verified = split_metrics.get("verified") or {}
        original = split_metrics.get("original") or {}
        summary[split] = {
            "original_average_token_f1": original.get("average_token_f1"),
            "verified_average_token_f1": verified.get("average_token_f1"),
            "verified_gold_doc_citation_rate": verified.get(
                "gold_doc_citation_rate"
            ),
            "verified_answerable_refusal_rate": verified.get(
                "answerable_refusal_rate"
            ),
            "verified_unanswerable_refusal_rate": verified.get(
                "unanswerable_refusal_rate"
            ),
            "verified_refused_answerable_questions": verified.get(
                "refused_answerable_questions"
            ),
            "verified_refused_unanswerable_questions": verified.get(
                "refused_unanswerable_questions"
            ),
            "answerable_gold_context_count": split_metrics.get(
                "answerable_gold_context_count"
            ),
            "answerable_gold_context_rate": split_metrics.get(
                "answerable_gold_context_rate"
            ),
        }
    return summary


def _select_bucket_mapping(mapping_by_split: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        split: {
            bucket: (mapping_by_split.get(split) or {}).get(bucket, 0)
            for bucket in _REPORT_BUCKET_ORDER
        }
        for split in _DEVELOPMENT_SPLITS
    }


def _select_top_priority_buckets(
    priority_by_split: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    selected = {}
    for split in _DEVELOPMENT_SPLITS:
        entries = priority_by_split.get(split) or []
        selected[split] = [
            {
                "bucket_id": entry.get("bucket_id"),
                "case_count": entry.get("case_count"),
                "priority_weight": entry.get("priority_weight"),
                "priority_score": entry.get("priority_score"),
            }
            for entry in entries
            if entry.get("bucket_id") in set(_REPORT_BUCKET_ORDER)
        ]
    return selected


def _bottleneck_summary(stage102_summary: Mapping[str, Any]) -> dict[str, Any]:
    counts = stage102_summary.get("bucket_counts_by_split") or {}
    rates = stage102_summary.get("bucket_rates_by_split") or {}
    data_summary = stage102_summary.get("data_summary") or {}
    split_rows = {
        split: int(
            ((data_summary.get("splits") or {}).get(split) or {}).get("row_count")
            or 0
        )
        for split in _DEVELOPMENT_SPLITS
    }
    combined_rows = sum(split_rows.values())
    bucket_rows = []
    for bucket in _REPORT_BUCKET_ORDER:
        train_count = int((counts.get("train") or {}).get(bucket) or 0)
        dev_count = int((counts.get("dev") or {}).get(bucket) or 0)
        combined_count = train_count + dev_count
        bucket_rows.append(
            {
                "bucket_id": bucket,
                "bottleneck_role": _bucket_role(bucket),
                "train_count": train_count,
                "dev_count": dev_count,
                "combined_count": combined_count,
                "train_rate": float((rates.get("train") or {}).get(bucket) or 0.0),
                "dev_rate": float((rates.get("dev") or {}).get(bucket) or 0.0),
                "combined_rate": _safe_rate(combined_count, combined_rows),
                "combined_priority_score": _combined_priority_score(
                    stage102_summary,
                    bucket,
                ),
                "shared_train_dev": train_count > 0 and dev_count > 0,
            }
        )
    sorted_rows = sorted(
        bucket_rows,
        key=lambda item: (
            0 if item["bucket_id"] in _PRIMARY_BUCKETS else 1,
            -float(item["combined_priority_score"]),
            str(item["bucket_id"]),
        ),
    )
    return {
        "source_stage": stage102_summary.get("stage"),
        "primary_bottleneck_bucket_ids": list(_PRIMARY_BUCKETS),
        "secondary_context_bucket_ids": list(_SECONDARY_BUCKETS),
        "split_row_counts": split_rows,
        "combined_row_count": combined_rows,
        "bucket_rows": sorted_rows,
        "shared_primary_bottleneck_count": sum(
            1
            for row in sorted_rows
            if row["bucket_id"] in _PRIMARY_BUCKETS and row["shared_train_dev"]
        ),
        "retrieval_context_miss_policy": (
            "tracked_as_secondary_context_only_after_retrieval_route_exhaustion"
        ),
    }


def _bucket_role(bucket_id: str) -> str:
    if bucket_id in _PRIMARY_BUCKETS:
        return "primary_candidate_target"
    if bucket_id == "evidence_selection_miss":
        return "secondary_evidence_context"
    return "secondary_retrieval_context_not_a_new_route"


def _combined_priority_score(
    stage102_summary: Mapping[str, Any],
    bucket_id: str,
) -> float:
    priority_by_split = stage102_summary.get("top_priority_buckets") or {}
    total = 0.0
    for split in _DEVELOPMENT_SPLITS:
        for entry in priority_by_split.get(split) or []:
            if entry.get("bucket_id") == bucket_id:
                total += float(entry.get("priority_score") or 0.0)
    return round(total, 4)


def _frozen_candidate_protocol(
    bottleneck_summary: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_policies = [
        _candidate_policy(spec=spec, bottleneck_summary=bottleneck_summary)
        for spec in _candidate_policy_specs()
    ]
    return {
        "design_id": _DESIGN_ID,
        "protocol_status": "frozen_requires_user_confirmation_before_train_dev_run",
        "source_stages": [_SOURCE_STAGE102],
        "recommended_direction": _NEXT_DIRECTION,
        "design_mode": "protocol_freeze_only",
        "objective": (
            "Define a fixed, train/dev-only intervention family for the "
            "shared Stage102 bottlenecks: false answers on unanswerable "
            "questions and selected evidence that underperforms an available "
            "gold-quality span. Stage103 does not execute the candidates."
        ),
        "candidate_policies": candidate_policies,
        "recommended_execution_order": [
            candidate["candidate_id"]
            for candidate in sorted(
                candidate_policies,
                key=lambda candidate: (
                    -float(candidate["priority_score"]),
                    str(candidate["risk_level"]),
                    str(candidate["candidate_id"]),
                ),
            )
        ],
        "blocked_items": _blocked_items(),
        "runtime_feature_contract": _runtime_feature_contract(),
        "stage104_train_dev_comparison_contract": (
            _stage104_train_dev_comparison_contract()
        ),
        "public_safe_output_contract": _public_safe_output_contract(),
        "explicit_exclusions": [
            "no_test_split_loading",
            "no_final_test_metrics",
            "no_dev_threshold_tuning",
            "no_runtime_default_change",
            "no_fallback_strategy",
            "no_oracle_doc_id_or_gold_answer_runtime_feature",
            "no_raw_question_answer_or_document_text_in_outputs",
        ],
        "fallback_strategy_policy": {
            "fallback_strategies_enabled": False,
            "requires_user_confirmation_before_any_fallback": True,
        },
    }


def _candidate_policy_specs() -> tuple[_CandidatePolicySpec, ...]:
    return (
        _CandidatePolicySpec(
            candidate_id="answerability_margin_gate_candidate_v1",
            name="Answerability margin gate candidate",
            intervention_focus="answerability_pre_generation_gate",
            status="recommended_for_stage104_train_dev_protocol",
            risk_level="medium",
            implementation_readiness=0.88,
            target_buckets=("answerability_false_answer",),
            rationale=(
                "Stage102 shows false generated answers on unanswerable "
                "questions as the top train/dev bucket. This candidate designs "
                "a stricter pre-generation evidence sufficiency gate from "
                "runtime-observable retrieval, citation, and verifier signals."
            ),
            runtime_feature_groups={
                "question_observables": (
                    "question_route",
                    "normalized_question_token_count_bucket",
                    "action_or_error_term_presence_bucket",
                ),
                "retrieval_observables": (
                    "top_result_score_bucket",
                    "top_result_score_margin_bucket",
                    "retrieved_context_count_bucket",
                    "best_context_rank_bucket",
                ),
                "evidence_observables": (
                    "max_sentence_support_score_bucket",
                    "support_score_margin_bucket",
                    "citation_count_bucket",
                    "best_citation_rank_bucket",
                ),
                "answer_observables": (
                    "candidate_answer_length_bucket",
                    "verifier_reason_code_bucket",
                ),
            },
            stage104_protocol_outline=(
                "Freeze the candidate threshold grid before the run.",
                "Select all gate thresholds on train only.",
                "Apply the train-selected gate once to dev without retuning.",
                "Report bucket deltas against Stage102 verified baseline.",
            ),
            target_metric_contract=(
                "primary: reduce answerability_false_answer on train selection",
                "primary-dev-check: dev answerability_false_answer must be reported",
                "guard: answerable_refusal_rate regression must be reported",
                "guard: verified average token F1 must be reported",
            ),
            safety_contract=(
                "No oracle identifiers, gold answers, or source DOC_IDS.",
                "No test split access.",
                "No runtime default promotion in Stage104.",
            ),
        ),
        _CandidatePolicySpec(
            candidate_id="evidence_window_reselector_candidate_v1",
            name="Evidence window reselector candidate",
            intervention_focus="evidence_selection_and_composition_ordering",
            status="recommended_for_stage104_train_dev_protocol",
            risk_level="medium",
            implementation_readiness=0.78,
            target_buckets=(
                "gold_span_beats_selected_answer",
                "evidence_selection_miss",
            ),
            rationale=(
                "Stage102 shows many answerable cases where an available "
                "stronger span beats the selected answer. This candidate "
                "designs a deterministic evidence-window selector using only "
                "runtime-observable sentence, rank, and section signals."
            ),
            runtime_feature_groups={
                "question_observables": (
                    "question_route",
                    "normalized_question_token_count_bucket",
                    "question_term_coverage_bucket",
                ),
                "sentence_window_observables": (
                    "sentence_support_score_bucket",
                    "adjacent_sentence_support_score_bucket",
                    "window_support_score_margin_bucket",
                    "action_sentence_presence_bucket",
                ),
                "retrieval_observables": (
                    "source_result_rank_bucket",
                    "source_result_score_bucket",
                    "section_heading_signal_bucket",
                ),
                "composition_observables": (
                    "selected_window_count_bucket",
                    "selected_window_length_bucket",
                ),
            },
            stage104_protocol_outline=(
                "Freeze selector feature buckets before the run.",
                "Select reselector configuration on train only.",
                "Apply the train-selected reselector once to dev without retuning.",
                "Report answer-composition and evidence-selection bucket deltas.",
            ),
            target_metric_contract=(
                "primary: reduce gold_span_beats_selected_answer on train",
                "secondary: reduce evidence_selection_miss on train",
                "primary-dev-check: dev bucket deltas must be reported",
                "guard: gold citation rate and average token F1 must be reported",
            ),
            safety_contract=(
                "The selector may score runtime evidence windows only.",
                "Gold spans are evaluation labels only, never runtime features.",
                "No runtime default promotion in Stage104.",
            ),
        ),
        _CandidatePolicySpec(
            candidate_id="joint_gate_then_window_candidate_v1",
            name="Joint gate then window candidate",
            intervention_focus="answerability_gate_plus_evidence_window_ordering",
            status="recommended_for_stage104_train_dev_protocol",
            risk_level="medium_high",
            implementation_readiness=0.62,
            target_buckets=(
                "answerability_false_answer",
                "gold_span_beats_selected_answer",
                "evidence_selection_miss",
            ),
            rationale=(
                "The two largest Stage102 buckets are both shared across "
                "train/dev, so a joint candidate should be compared after the "
                "component policies are frozen. It gates weakly supported "
                "queries first, then applies the evidence-window ordering rule."
            ),
            runtime_feature_groups={
                "answerability_gate_observables": (
                    "top_result_score_bucket",
                    "support_score_margin_bucket",
                    "citation_count_bucket",
                    "verifier_reason_code_bucket",
                ),
                "evidence_window_observables": (
                    "sentence_support_score_bucket",
                    "window_support_score_margin_bucket",
                    "section_heading_signal_bucket",
                    "action_sentence_presence_bucket",
                ),
                "composition_observables": (
                    "selected_window_count_bucket",
                    "candidate_answer_length_bucket",
                ),
            },
            stage104_protocol_outline=(
                "Freeze the component order before the run: gate first, window second.",
                "Select joint thresholds on train only.",
                "Apply the train-selected joint policy once to dev without retuning.",
                "Report whether the joint policy improves over both components.",
            ),
            target_metric_contract=(
                "primary: reduce the train joint priority score",
                "primary-dev-check: dev joint priority score must be reported",
                "guard: answerable refusals, citation rate, and token F1 must be reported",
                "guard: changed-case summary must remain public-safe",
            ),
            safety_contract=(
                "The joint policy is a candidate intervention, not a fallback.",
                "No oracle identifiers, gold answers, or test access.",
                "No runtime default promotion in Stage104.",
            ),
        ),
    )


def _candidate_policy(
    *,
    spec: _CandidatePolicySpec,
    bottleneck_summary: Mapping[str, Any],
) -> dict[str, Any]:
    bucket_rows = {
        row["bucket_id"]: row
        for row in bottleneck_summary.get("bucket_rows") or []
        if row.get("bucket_id") in spec.target_buckets
    }
    target_case_count_by_split = {
        split: sum(int(row.get(f"{split}_count") or 0) for row in bucket_rows.values())
        for split in _DEVELOPMENT_SPLITS
    }
    target_priority_score = round(
        sum(float(row.get("combined_priority_score") or 0.0) for row in bucket_rows.values()),
        4,
    )
    priority_score = round(
        target_priority_score
        * spec.implementation_readiness
        * _risk_multiplier(spec.risk_level),
        4,
    )
    return {
        "candidate_id": spec.candidate_id,
        "name": spec.name,
        "intervention_focus": spec.intervention_focus,
        "status": spec.status,
        "risk_level": spec.risk_level,
        "implementation_readiness": spec.implementation_readiness,
        "target_buckets": list(spec.target_buckets),
        "target_case_count_by_split": target_case_count_by_split,
        "target_combined_case_count": sum(target_case_count_by_split.values()),
        "target_priority_score": target_priority_score,
        "priority_score": priority_score,
        "rationale": spec.rationale,
        "runtime_feature_groups": {
            group: list(features)
            for group, features in spec.runtime_feature_groups.items()
        },
        "stage104_protocol_outline": list(spec.stage104_protocol_outline),
        "target_metric_contract": list(spec.target_metric_contract),
        "safety_contract": list(spec.safety_contract),
    }


def _risk_multiplier(risk_level: str) -> float:
    return {
        "low": 1.0,
        "medium": 0.9,
        "medium_high": 0.75,
        "high": 0.6,
    }.get(risk_level, 0.7)


def _blocked_items() -> list[dict[str, Any]]:
    return [
        {
            "blocked_item_id": "source_doc_id_oracle_candidate_blocked",
            "blocked_reason": (
                "Would require oracle document identifiers that are labels, not "
                "runtime evidence."
            ),
            "status": "blocked_from_train_dev_experiment",
        },
        {
            "blocked_item_id": "gold_span_oracle_selector_blocked",
            "blocked_reason": (
                "Would select evidence from gold answer/span labels instead of "
                "runtime-observable evidence."
            ),
            "status": "blocked_from_train_dev_experiment",
        },
        {
            "blocked_item_id": "test_tuned_threshold_candidate_blocked",
            "blocked_reason": (
                "Would use the final test split for development or threshold "
                "selection."
            ),
            "status": "blocked_from_train_dev_experiment",
        },
    ]


def _runtime_feature_contract() -> dict[str, Any]:
    return {
        "allowed_runtime_feature_groups": {
            "question_observables": [
                "question_route",
                "normalized token count buckets",
                "deterministic action/error term buckets",
            ],
            "retrieval_observables": [
                "retrieval scores and score margins",
                "rank buckets",
                "retrieved context count buckets",
            ],
            "evidence_observables": [
                "sentence/window support scores",
                "citation count and rank buckets",
                "section heading signal buckets",
            ],
            "composition_observables": [
                "selected window count buckets",
                "candidate answer length buckets",
                "verifier reason code buckets",
            ],
        },
        "prohibited_runtime_inputs": [
            "gold answers",
            "gold spans",
            "answer document identifiers",
            "source DOC_IDS",
            "test split labels",
            "dev-selected thresholds",
            "raw private question or answer strings written to reports",
        ],
    }


def _stage104_train_dev_comparison_contract() -> dict[str, Any]:
    return {
        "comparison_id": "evidence_answerability_candidate_train_dev_comparison_v1",
        "run_mode": "train_dev_only_after_user_confirmation",
        "baseline_reference": "Stage102 verified BM25 top10 answer pipeline",
        "train_selection_rule": {
            "candidate_thresholds_selected_on": "train_only",
            "dev_threshold_tuning_allowed": False,
            "test_access_allowed": False,
            "selection_objective": (
                "Select the train candidate that most reduces the combined "
                "answerability/evidence priority score while reporting guard "
                "metric regressions."
            ),
        },
        "dev_validation_rule": {
            "dev_used_for": "single validation of train-selected candidate",
            "dev_retuning_allowed": False,
            "must_report_all_candidates_on_dev": True,
        },
        "metric_contract": {
            "primary_metrics": [
                "answerability_false_answer count/rate",
                "gold_span_beats_selected_answer count/rate",
                "combined target priority score",
            ],
            "secondary_metrics": [
                "evidence_selection_miss count/rate",
                "verified average token F1",
                "verified gold document citation rate",
            ],
            "guard_metrics": [
                "answerable refusal rate",
                "unanswerable refusal rate",
                "changed answer count",
                "public-safe changed-case bucket summary",
            ],
        },
        "promotion_rule": {
            "runtime_default_change_allowed_in_stage104": False,
            "requires_separate_user_confirmation_after_stage104": True,
            "final_test_gate_remains_closed": True,
        },
    }


def _public_safe_output_contract() -> dict[str, Any]:
    return {
        "stage103_outputs_are_aggregate_only": True,
        "stage104_allowed_aggregate_fields": [
            "split",
            "candidate_id",
            "bucket_counts_by_split",
            "bucket_rates_by_split",
            "metric_deltas_by_split",
            "train_selection_summary",
            "guard_checks",
        ],
        "stage104_allowed_case_fields": [
            "sample_id",
            "split",
            "candidate_id",
            "baseline_bucket_id",
            "candidate_bucket_id",
            "baseline_answer_token_f1_bucket",
            "candidate_answer_token_f1_bucket",
            "baseline_citation_status",
            "candidate_citation_status",
            "answerability_action",
            "evidence_selection_action",
            "changed_case_confidence_band",
        ],
        "forbidden_fields": sorted(_FORBIDDEN_PUBLIC_FIELDS),
        "private_label_inputs_not_written": [
            "gold answers are evaluation labels only and not written",
            "answer document identifiers are evaluation labels only and not written",
            "source DOC_IDS are never runtime evidence and not written",
        ],
    }


def _guard_checks(
    *,
    stage102_summary: Mapping[str, Any],
    bottleneck_summary: Mapping[str, Any],
    frozen_candidate_protocol: Mapping[str, Any],
    user_confirmed_protocol: bool,
) -> list[dict[str, Any]]:
    candidate_policies = frozen_candidate_protocol.get("candidate_policies") or []
    candidate_ids = [str(candidate.get("candidate_id")) for candidate in candidate_policies]
    execution_order = frozen_candidate_protocol.get("recommended_execution_order") or []
    stage104_contract = (
        frozen_candidate_protocol.get("stage104_train_dev_comparison_contract") or {}
    )
    train_selection_rule = stage104_contract.get("train_selection_rule") or {}
    dev_validation_rule = stage104_contract.get("dev_validation_rule") or {}
    output_contract = frozen_candidate_protocol.get("public_safe_output_contract") or {}
    explicit_exclusions = frozen_candidate_protocol.get("explicit_exclusions") or []
    fallback_policy = frozen_candidate_protocol.get("fallback_strategy_policy") or {}
    blocked_items = frozen_candidate_protocol.get("blocked_items") or []
    return [
        _check(
            name="stage102_source_is_expected_stage",
            passed=stage102_summary.get("stage") == _SOURCE_STAGE102,
            observed=stage102_summary.get("stage"),
            expected=_SOURCE_STAGE102,
        ),
        _check(
            name="user_confirmed_stage103_protocol",
            passed=user_confirmed_protocol,
            observed=user_confirmed_protocol,
            expected=True,
        ),
        _check(
            name="stage102_analysis_id_is_expected",
            passed=stage102_summary.get("analysis_id") == _SOURCE_ANALYSIS_ID,
            observed=stage102_summary.get("analysis_id"),
            expected=_SOURCE_ANALYSIS_ID,
        ),
        _check(
            name="stage102_analysis_completed",
            passed=stage102_summary.get("decision_status")
            == "primeqa_hybrid_answer_pipeline_error_decomposition_completed",
            observed=stage102_summary.get("decision_status"),
            expected="primeqa_hybrid_answer_pipeline_error_decomposition_completed",
        ),
        _check(
            name="stage102_recommends_evidence_answerability_design",
            passed=stage102_summary.get("recommended_next_direction")
            == _SOURCE_RECOMMENDED_DIRECTION,
            observed=stage102_summary.get("recommended_next_direction"),
            expected=_SOURCE_RECOMMENDED_DIRECTION,
        ),
        _check(
            name="stage102_train_top_bucket_is_answerability_false_answer",
            passed=stage102_summary.get("train_top_bucket")
            == "answerability_false_answer",
            observed=stage102_summary.get("train_top_bucket"),
            expected="answerability_false_answer",
        ),
        _check(
            name="stage102_dev_top_bucket_is_answerability_false_answer",
            passed=stage102_summary.get("dev_top_bucket")
            == "answerability_false_answer",
            observed=stage102_summary.get("dev_top_bucket"),
            expected="answerability_false_answer",
        ),
        _check(
            name="shared_answerability_false_answer_observed",
            passed=_bucket_shared(bottleneck_summary, "answerability_false_answer"),
            observed=_bucket_counts(bottleneck_summary, "answerability_false_answer"),
            expected="train_count > 0 and dev_count > 0",
        ),
        _check(
            name="shared_gold_span_gap_observed",
            passed=_bucket_shared(
                bottleneck_summary,
                "gold_span_beats_selected_answer",
            ),
            observed=_bucket_counts(
                bottleneck_summary,
                "gold_span_beats_selected_answer",
            ),
            expected="train_count > 0 and dev_count > 0",
        ),
        _check(
            name="shared_evidence_selection_miss_observed",
            passed=_bucket_shared(bottleneck_summary, "evidence_selection_miss"),
            observed=_bucket_counts(bottleneck_summary, "evidence_selection_miss"),
            expected="train_count > 0 and dev_count > 0",
        ),
        _check(
            name="retrieval_context_miss_is_secondary_context_only",
            passed=bottleneck_summary.get("retrieval_context_miss_policy")
            == "tracked_as_secondary_context_only_after_retrieval_route_exhaustion",
            observed=bottleneck_summary.get("retrieval_context_miss_policy"),
            expected="tracked_as_secondary_context_only_after_retrieval_route_exhaustion",
        ),
        _check(
            name="stage102_all_guard_checks_passed",
            passed=stage102_summary.get("stage102_guard_check_count")
            == stage102_summary.get("stage102_guard_check_passed_count"),
            observed={
                "count": stage102_summary.get("stage102_guard_check_count"),
                "passed": stage102_summary.get("stage102_guard_check_passed_count"),
            },
            expected="all Stage102 guard checks passed",
        ),
        _check(
            name="stage102_can_continue_train_dev",
            passed=stage102_summary.get("can_continue_train_dev_development") is True,
            observed=stage102_summary.get("can_continue_train_dev_development"),
            expected=True,
        ),
        _check(
            name="stage102_final_test_gate_locked",
            passed=stage102_summary.get("can_run_final_test_metrics_now") is False
            and stage102_summary.get("can_open_final_test_gate_now") is False,
            observed={
                "can_open_final_test_gate_now": stage102_summary.get(
                    "can_open_final_test_gate_now"
                ),
                "can_run_final_test_metrics_now": stage102_summary.get(
                    "can_run_final_test_metrics_now"
                ),
            },
            expected=False,
        ),
        _check(
            name="stage102_test_not_available_for_tuning",
            passed=stage102_summary.get("can_use_test_for_tuning") is False,
            observed=stage102_summary.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage102_fallback_disabled",
            passed=stage102_summary.get("fallback_strategies_enabled") is False,
            observed=stage102_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="stage102_runtime_defaults_unchanged",
            passed=stage102_summary.get("default_runtime_policy") == "unchanged",
            observed=stage102_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage103_protocol_status_frozen",
            passed=frozen_candidate_protocol.get("protocol_status")
            == "frozen_requires_user_confirmation_before_train_dev_run",
            observed=frozen_candidate_protocol.get("protocol_status"),
            expected="frozen_requires_user_confirmation_before_train_dev_run",
        ),
        _check(
            name="stage103_candidate_count_is_three",
            passed=len(candidate_policies) == 3,
            observed=len(candidate_policies),
            expected=3,
        ),
        _check(
            name="stage103_candidate_ids_are_unique",
            passed=len(candidate_ids) == len(set(candidate_ids)),
            observed=candidate_ids,
            expected="unique candidate IDs",
        ),
        _check(
            name="stage103_candidate_execution_order_complete",
            passed=set(execution_order) == set(candidate_ids),
            observed=execution_order,
            expected=candidate_ids,
        ),
        _check(
            name="stage103_candidates_are_train_dev_protocol_candidates",
            passed=all(
                candidate.get("status") == "recommended_for_stage104_train_dev_protocol"
                for candidate in candidate_policies
            ),
            observed=[
                candidate.get("status") for candidate in candidate_policies
            ],
            expected="all candidates recommended for Stage104 train/dev protocol",
        ),
        _check(
            name="stage103_candidate_runtime_features_are_declared",
            passed=all(
                bool(candidate.get("runtime_feature_groups"))
                for candidate in candidate_policies
            ),
            observed=[
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "feature_group_count": len(
                        candidate.get("runtime_feature_groups") or {}
                    ),
                }
                for candidate in candidate_policies
            ],
            expected="each candidate has runtime feature groups",
        ),
        _check(
            name="stage103_candidates_do_not_use_forbidden_runtime_inputs",
            passed=_candidate_allowed_features_are_runtime_only(candidate_policies),
            observed=_candidate_allowed_feature_tokens(candidate_policies),
            expected="no forbidden runtime feature tokens",
        ),
        _check(
            name="stage103_oracle_items_are_blocked",
            passed=_oracle_items_are_blocked(blocked_items),
            observed=blocked_items,
            expected="oracle/source/test-tuned items blocked",
        ),
        _check(
            name="stage104_train_selection_forbids_dev_threshold_tuning",
            passed=train_selection_rule.get("candidate_thresholds_selected_on")
            == "train_only"
            and train_selection_rule.get("dev_threshold_tuning_allowed") is False,
            observed=train_selection_rule,
            expected="train-only selection and no dev threshold tuning",
        ),
        _check(
            name="stage104_dev_validation_forbids_retuning",
            passed=dev_validation_rule.get("dev_retuning_allowed") is False,
            observed=dev_validation_rule,
            expected="dev validation without retuning",
        ),
        _check(
            name="stage104_test_access_forbidden",
            passed=train_selection_rule.get("test_access_allowed") is False,
            observed=train_selection_rule.get("test_access_allowed"),
            expected=False,
        ),
        _check(
            name="stage103_public_safe_output_contract_has_no_forbidden_fields",
            passed=_public_output_contract_is_safe(output_contract),
            observed=output_contract.get("stage104_allowed_case_fields"),
            expected="no raw/private/oracle output fields",
        ),
        _check(
            name="stage103_exclusions_lock_test_runtime_fallback",
            passed={
                "no_test_split_loading",
                "no_final_test_metrics",
                "no_runtime_default_change",
                "no_fallback_strategy",
            }.issubset(set(explicit_exclusions)),
            observed=explicit_exclusions,
            expected=[
                "no_test_split_loading",
                "no_final_test_metrics",
                "no_runtime_default_change",
                "no_fallback_strategy",
            ],
        ),
        _check(
            name="stage103_fallback_policy_disabled",
            passed=fallback_policy.get("fallback_strategies_enabled") is False,
            observed=fallback_policy,
            expected={"fallback_strategies_enabled": False},
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    frozen_candidate_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    base = {
        "design_id": _DESIGN_ID,
        "recommended_direction": _NEXT_DIRECTION,
        "recommended_execution_order": frozen_candidate_protocol.get(
            "recommended_execution_order"
        ),
        "requires_user_confirmation_before_train_dev_run": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }
    if failed_checks:
        return {
            **base,
            "status": "primeqa_hybrid_evidence_answerability_candidate_protocol_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "can_run_train_dev_candidate_comparison_after_user_confirmation": False,
        }
    return {
        **base,
        "status": "primeqa_hybrid_evidence_answerability_candidate_protocol_frozen",
        "can_continue_train_dev_development": True,
        "can_run_train_dev_candidate_comparison_after_user_confirmation": True,
        "recommended_next_stage": (
            "Stage104: after user confirmation, run the frozen train/dev-only "
            "evidence-answerability candidate comparison against the Stage102 "
            "verified baseline; keep test locked, do not retune on dev, and "
            "do not change runtime defaults."
        ),
    }


def _bottleneck_count_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for row in report["bottleneck_summary"]["bucket_rows"]:
        for split in _DEVELOPMENT_SPLITS:
            bars.append(
                BarDatum(
                    label=f"{split}:{row['bucket_id']}",
                    value=float(row[f"{split}_count"]),
                    value_label=str(row[f"{split}_count"]),
                )
            )
    return bars


def _bottleneck_rate_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for row in report["bottleneck_summary"]["bucket_rows"]:
        for split in _DEVELOPMENT_SPLITS:
            value = float(row[f"{split}_rate"])
            bars.append(
                BarDatum(
                    label=f"{split}:{row['bucket_id']}",
                    value=value,
                    value_label=f"{value:.4f}",
                )
            )
    return bars


def _candidate_priority_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    candidates = report["frozen_candidate_protocol"]["candidate_policies"]
    return [
        BarDatum(
            label=str(candidate["candidate_id"]),
            value=float(candidate["priority_score"]),
            value_label=f"{float(candidate['priority_score']):.2f}",
        )
        for candidate in candidates
    ]


def _candidate_target_case_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    candidates = report["frozen_candidate_protocol"]["candidate_policies"]
    return [
        BarDatum(
            label=str(candidate["candidate_id"]),
            value=float(candidate["target_combined_case_count"]),
            value_label=str(candidate["target_combined_case_count"]),
        )
        for candidate in candidates
    ]


def _candidate_feature_group_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    candidates = report["frozen_candidate_protocol"]["candidate_policies"]
    bars = []
    for candidate in candidates:
        feature_count = sum(
            len(features)
            for features in (candidate.get("runtime_feature_groups") or {}).values()
        )
        bars.append(
            BarDatum(
                label=str(candidate["candidate_id"]),
                value=float(feature_count),
                value_label=str(feature_count),
            )
        )
    return bars


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report["decision"]
    names = [
        "can_run_train_dev_candidate_comparison_after_user_confirmation",
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


def _bucket_shared(
    bottleneck_summary: Mapping[str, Any],
    bucket_id: str,
) -> bool:
    return bool(_bucket_row(bottleneck_summary, bucket_id).get("shared_train_dev"))


def _bucket_counts(
    bottleneck_summary: Mapping[str, Any],
    bucket_id: str,
) -> dict[str, int]:
    row = _bucket_row(bottleneck_summary, bucket_id)
    return {
        "train_count": int(row.get("train_count") or 0),
        "dev_count": int(row.get("dev_count") or 0),
    }


def _bucket_row(
    bottleneck_summary: Mapping[str, Any],
    bucket_id: str,
) -> Mapping[str, Any]:
    for row in bottleneck_summary.get("bucket_rows") or []:
        if row.get("bucket_id") == bucket_id:
            return row
    return {}


def _candidate_allowed_features_are_runtime_only(
    candidate_policies: Sequence[Mapping[str, Any]],
) -> bool:
    tokens = _candidate_allowed_feature_tokens(candidate_policies)
    return all(
        forbidden not in token
        for token in tokens
        for forbidden in _FORBIDDEN_RUNTIME_FEATURE_TOKENS
    )


def _candidate_allowed_feature_tokens(
    candidate_policies: Sequence[Mapping[str, Any]],
) -> list[str]:
    tokens = []
    for candidate in candidate_policies:
        for features in (candidate.get("runtime_feature_groups") or {}).values():
            tokens.extend(str(feature).lower() for feature in features)
    return tokens


def _oracle_items_are_blocked(blocked_items: Sequence[Mapping[str, Any]]) -> bool:
    blocked_ids = {str(item.get("blocked_item_id")) for item in blocked_items}
    expected = {
        "source_doc_id_oracle_candidate_blocked",
        "gold_span_oracle_selector_blocked",
        "test_tuned_threshold_candidate_blocked",
    }
    return expected.issubset(blocked_ids) and all(
        item.get("status") == "blocked_from_train_dev_experiment"
        for item in blocked_items
    )


def _public_output_contract_is_safe(output_contract: Mapping[str, Any]) -> bool:
    case_fields = {
        str(field).lower()
        for field in output_contract.get("stage104_allowed_case_fields") or []
    }
    aggregate_fields = {
        str(field).lower()
        for field in output_contract.get("stage104_allowed_aggregate_fields") or []
    }
    forbidden = {field.lower() for field in _FORBIDDEN_PUBLIC_FIELDS}
    return not ((case_fields | aggregate_fields) & forbidden)


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


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
