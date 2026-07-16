from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 131"
_CREATED_AT = "2026-07-16"
_PROTOCOL_ID = "primeqa_hybrid_append_candidate_evidence_shortlist_redesign_protocol_v1"
_SOURCE_STAGE = "Stage 130"
_SOURCE_REVIEW_ID = "primeqa_hybrid_stage129_agent_integration_failure_review_v1"
_SOURCE_STATUS = "primeqa_hybrid_stage129_agent_integration_failure_review_completed"
_SOURCE_NEXT = "freeze_append_candidate_evidence_shortlist_redesign_protocol"
_NEXT_DIRECTION = "run_append_candidate_evidence_shortlist_train_cv_dev_validation"
_BASELINE_PREFIX_DEPTH = 200
_APPEND_START_RANK = 201
_TARGET_POOL_DEPTH = 400
_ANSWER_CONTEXT_DEPTH = 10
_MAX_REPLACEMENT_APPEND_SLOTS = 2
_MINIMUM_TRAIN_FOLDS = 5
_REQUIRED_FAILURE_PATTERNS = frozenset(
    {
        "recall_gain_not_citation_safe",
        "append_region_displaces_prefix_evidence",
        "changed_answer_churn_too_high",
    }
)
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
class PrimeQAHybridAppendCandidateEvidenceShortlistProtocolVisualization:
    """One generated Stage131 append-candidate shortlist protocol chart."""

    name: str
    path: str


def freeze_primeqa_hybrid_append_candidate_evidence_shortlist_protocol(
    *,
    stage130_review_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage131 append-candidate evidence shortlist redesign protocol."""

    started_at = time.perf_counter()
    stage130_review = _load_json_object(stage130_review_path)
    loaded_at = time.perf_counter()

    stage130_summary = _stage130_summary(stage130_review)
    frozen_protocol = _frozen_protocol(stage130_summary)
    preliminary = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "protocol_scope": (
            "Train/dev-only protocol freeze for redesigning the evidence "
            "shortlist over the Stage116 prefix plus Stage128 append candidate "
            "pool. This stage reads only the public-safe Stage130 aggregate "
            "failure review, does not load split files, corpus documents, raw "
            "candidate rows, raw questions, raw answers, raw document "
            "identifiers, or test data, does not run retrieval, answering, "
            "validation metrics, final metrics, runtime defaultization, or "
            "fallback strategies."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_protocol),
            "confirmation_note": confirmation_note,
        },
        "source_files": {
            "stage130_review": _fingerprint(stage130_review_path),
        },
        "stage130_summary": stage130_summary,
        "frozen_protocol": frozen_protocol,
    }
    guard_checks = _guard_checks(
        report=preliminary,
        stage130_summary=stage130_summary,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    checked_at = time.perf_counter()
    report = {
        **preliminary,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
        "timing_seconds": {
            "load_stage130_review": round(loaded_at - started_at, 3),
            "freeze_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_append_candidate_evidence_shortlist_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridAppendCandidateEvidenceShortlistProtocolVisualization]:
    """Write SVG charts for the Stage131 protocol freeze."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage131_source_failure_pressure.svg": render_horizontal_bar_chart_svg(
            title="Stage131 source failure pressure",
            bars=_source_failure_pressure_bars(report),
            x_label="count or percent",
            width=1540,
            margin_left=820,
        ),
        "stage131_shortlist_candidate_budgets.svg": render_horizontal_bar_chart_svg(
            title="Stage131 shortlist candidate budgets",
            bars=_candidate_budget_bars(report),
            x_label="document slots",
            width=1500,
            margin_left=780,
        ),
        "stage131_validation_guard_thresholds.svg": render_horizontal_bar_chart_svg(
            title="Stage131 validation guard thresholds",
            bars=_validation_guard_threshold_bars(report),
            x_label="threshold",
            width=1580,
            margin_left=860,
        ),
        "stage131_protocol_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage131 protocol decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1460,
            margin_left=780,
        ),
        "stage131_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage131 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1860,
            margin_left=1040,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridAppendCandidateEvidenceShortlistProtocolVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage130_summary(stage130_review: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage130_review.get("decision") or {}
    action_boundary = stage130_review.get("action_boundary") or {}
    train_review = stage130_review.get("train_cv_failure_review") or {}
    dev_review = stage130_review.get("dev_report_only_review") or {}
    failure_patterns = stage130_review.get("failure_patterns") or []
    public_safe = stage130_review.get("public_safe_contract") or {}
    train_deltas = train_review.get("candidate_vs_control_deltas") or {}
    dev_deltas = dev_review.get("candidate_vs_control_deltas") or {}
    train_shift = train_review.get("selected_citation_region_shift") or {}
    dev_shift = dev_review.get("selected_citation_region_shift") or {}
    return {
        "stage": stage130_review.get("stage"),
        "review_id": stage130_review.get("review_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "can_continue_train_dev_development": decision.get(
            "can_continue_train_dev_development"
        ),
        "stage128_direct_agent_integration_path_blocked": decision.get(
            "stage128_direct_agent_integration_path_blocked"
        )
        or action_boundary.get("stage128_direct_agent_integration_path_blocked"),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "runtime_defaultization_allowed_now": decision.get(
            "runtime_defaultization_allowed_now"
        ),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "failure_pattern_ids": [pattern.get("pattern_id") for pattern in failure_patterns],
        "blocking_failure_pattern_ids": [
            pattern.get("pattern_id")
            for pattern in failure_patterns
            if pattern.get("severity") == "blocking"
        ],
        "train_gold_hit_delta": train_deltas.get(
            "gold_hit_count_at_profile_depth_delta"
        ),
        "train_gold_citation_delta": train_deltas.get(
            "verified_gold_citation_count_delta"
        ),
        "train_verified_f1_delta": train_deltas.get("verified_average_token_f1_delta"),
        "train_changed_answer_rate": train_review.get(
            "changed_verified_answer_rate_vs_control"
        ),
        "train_append_selected_citations": train_shift.get(
            "append_region_selected_citation_count"
        ),
        "train_prefix_like_selected_citation_delta": train_shift.get(
            "prefix_like_selected_citation_delta"
        ),
        "dev_gold_hit_delta": dev_deltas.get("gold_hit_count_at_profile_depth_delta"),
        "dev_gold_citation_delta": dev_deltas.get("verified_gold_citation_count_delta"),
        "dev_verified_f1_delta": dev_deltas.get("verified_average_token_f1_delta"),
        "dev_changed_answer_rate": dev_review.get(
            "changed_verified_answer_rate_vs_control"
        ),
        "dev_append_selected_citations": dev_shift.get(
            "append_region_selected_citation_count"
        ),
        "dev_prefix_like_selected_citation_delta": dev_shift.get(
            "prefix_like_selected_citation_delta"
        ),
        "guard_check_count": len(stage130_review.get("guard_checks") or []),
        "guard_check_passed_count": sum(
            1 for check in stage130_review.get("guard_checks") or [] if check.get("passed")
        ),
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
    }


def _frozen_protocol(stage130_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "route_name": "append_candidate_evidence_shortlist_redesign",
        "source_failure_review": {
            "stage": stage130_summary.get("stage"),
            "review_id": stage130_summary.get("review_id"),
            "status": stage130_summary.get("decision_status"),
            "recommended_next_direction": stage130_summary.get(
                "recommended_next_direction"
            ),
            "blocking_failure_patterns": stage130_summary.get(
                "blocking_failure_pattern_ids"
            ),
        },
        "source_candidate_pool_contract": {
            "stage116_prefix_depth": _BASELINE_PREFIX_DEPTH,
            "stage128_candidate_pool_depth": _TARGET_POOL_DEPTH,
            "append_start_rank": _APPEND_START_RANK,
            "candidate_pool_role": "recall_pool_not_unrestricted_answer_context",
            "stage116_prefix_must_remain_available": True,
            "append_candidates_are_supplemental": True,
        },
        "shortlist_redesign_principles": [
            {
                "principle_id": "protect_stage116_evidence_first",
                "requirement": (
                    "The Stage116 prefix-derived evidence set is the stability "
                    "anchor. Append candidates must not freely replace it."
                ),
            },
            {
                "principle_id": "append_candidates_enter_through_explicit_gate",
                "requirement": (
                    "Ranks 201-400 may enter the answer context only through a "
                    "predeclared append gate with runtime-visible signals."
                ),
            },
            {
                "principle_id": "citation_guard_is_primary",
                "requirement": (
                    "A config cannot pass if train-CV gold citation count is "
                    "lower than the Stage116 control, even when recall improves."
                ),
            },
            {
                "principle_id": "dev_is_report_only",
                "requirement": (
                    "Dev may confirm risk direction but cannot be used to tune "
                    "thresholds, budgets, or config selection."
                ),
            },
        ],
        "candidate_shortlist_configs": _candidate_shortlist_configs(),
        "append_gate_runtime_signals": {
            "allowed": [
                "query/document token overlap",
                "query/document phrase overlap",
                "section-title overlap",
                "special-token exact match",
                "Stage116 prefix rank prior",
                "Stage128 append region rank prior",
                "sentence evidence score from runtime text",
                "retrieval route family agreement count",
            ],
            "forbidden": [
                "gold labels",
                "test membership",
                "dev-selected thresholds",
                "source-provided labels",
                "private raw rows in public artifacts",
            ],
        },
        "selection_and_validation_plan": {
            "next_stage": "Stage132",
            "action": _NEXT_DIRECTION,
            "selection_split": "train",
            "selection_mode": (
                "train_grouped_cross_validation_append_shortlist_config_selection"
            ),
            "minimum_train_folds": _MINIMUM_TRAIN_FOLDS,
            "validation_split": "dev",
            "dev_mode": "single_pass_report_only_no_retuning",
            "profile_comparison_baseline": "stage116_top200_agent_pool_control",
            "comparison_candidate_pool": "stage128_prefix_append_top400_agent_pool",
            "primary_train_cv_guard": (
                "gold_citation_count_delta_vs_stage116_non_negative"
            ),
            "secondary_train_cv_guards": [
                "verified_f1_delta_vs_stage116_non_negative",
                "answerable_refusal_rate_delta_vs_stage116_non_positive",
                "unanswerable_refusal_rate_delta_vs_stage116_non_positive",
                "changed_verified_answer_rate_not_above_stage129_candidate",
                "append_selected_citations_do_not_displace_prefix_like_citations_without_gold_gain",
            ],
            "dev_reporting": [
                "verified F1 delta",
                "gold citation count delta",
                "gold hit count at profile depth delta",
                "changed verified answer rate",
                "selected citation rank-region mix",
            ],
            "test_rules": {
                "test_access_allowed": False,
                "final_test_metrics_allowed": False,
                "test_tuning_allowed": False,
            },
            "runtime_rules": {
                "default_runtime_policy": "unchanged",
                "runtime_defaultization_allowed_in_stage131": False,
                "fallback_strategies_enabled": False,
            },
        },
        "failure_controls": {
            "stage130_patterns_addressed": [
                {
                    "pattern_id": "recall_gain_not_citation_safe",
                    "control": "citation guard blocks recall-only wins",
                },
                {
                    "pattern_id": "append_region_displaces_prefix_evidence",
                    "control": "append budgets are capped and prefix protection is explicit",
                },
                {
                    "pattern_id": "changed_answer_churn_too_high",
                    "control": "changed-answer churn is a reported guard dimension",
                },
            ],
            "stage128_direct_integration_remains_blocked": True,
            "runtime_defaultization_requires_future_gate": True,
        },
    }


def _candidate_shortlist_configs() -> list[dict[str, Any]]:
    return [
        {
            "config_id": "prefix10_append_sidecar_probe_v1",
            "selection_role": "conservative_control",
            "answer_context_depth": _ANSWER_CONTEXT_DEPTH,
            "protected_prefix_slots": 10,
            "replacement_append_slots": 0,
            "append_sidecar_slots": 3,
            "append_sidecar_can_generate_answer_text": False,
            "append_sidecar_can_support_citation_verification": True,
            "append_acceptance_rule": (
                "Append candidates are evaluated as supplemental citation "
                "evidence only; they do not replace the primary answer context."
            ),
        },
        {
            "config_id": "prefix9_append1_high_precision_v1",
            "selection_role": "high_precision_single_append",
            "answer_context_depth": _ANSWER_CONTEXT_DEPTH,
            "protected_prefix_slots": 9,
            "replacement_append_slots": 1,
            "append_sidecar_slots": 2,
            "append_sidecar_can_generate_answer_text": False,
            "append_sidecar_can_support_citation_verification": True,
            "append_acceptance_rule": (
                "At most one append candidate may replace the weakest prefix "
                "slot, and only when runtime-visible evidence support exceeds "
                "the predeclared high-precision gate."
            ),
        },
        {
            "config_id": "prefix8_append2_balanced_probe_v1",
            "selection_role": "bounded_balanced_append",
            "answer_context_depth": _ANSWER_CONTEXT_DEPTH,
            "protected_prefix_slots": 8,
            "replacement_append_slots": 2,
            "append_sidecar_slots": 2,
            "append_sidecar_can_generate_answer_text": False,
            "append_sidecar_can_support_citation_verification": True,
            "append_acceptance_rule": (
                "At most two append candidates may enter the answer context; "
                "this config is blocked if citation preservation fails."
            ),
        },
    ]


def _guard_checks(
    *,
    report: Mapping[str, Any],
    stage130_summary: Mapping[str, Any],
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> list[dict[str, Any]]:
    frozen = report["frozen_protocol"]
    plan = frozen["selection_and_validation_plan"]
    configs = frozen["candidate_shortlist_configs"]
    public_safe = _public_safe_contract(report)
    return [
        _check(
            name="user_confirmed_stage131_protocol",
            passed=user_confirmed_protocol and "Stage131" in confirmation_note,
            observed=confirmation_note,
            expected="user confirmed Stage131 protocol freeze",
        ),
        _check(
            name="stage130_review_completed",
            passed=stage130_summary.get("decision_status") == _SOURCE_STATUS,
            observed=stage130_summary.get("decision_status"),
            expected=_SOURCE_STATUS,
        ),
        _check(
            name="stage130_review_id_matches",
            passed=stage130_summary.get("review_id") == _SOURCE_REVIEW_ID,
            observed=stage130_summary.get("review_id"),
            expected=_SOURCE_REVIEW_ID,
        ),
        _check(
            name="stage130_recommends_stage131_protocol",
            passed=stage130_summary.get("recommended_next_direction") == _SOURCE_NEXT,
            observed=stage130_summary.get("recommended_next_direction"),
            expected=_SOURCE_NEXT,
        ),
        _check(
            name="stage130_blocks_direct_stage128_integration",
            passed=stage130_summary.get("stage128_direct_agent_integration_path_blocked")
            is True,
            observed=stage130_summary.get(
                "stage128_direct_agent_integration_path_blocked"
            ),
            expected=True,
        ),
        _check(
            name="stage130_failure_patterns_are_present",
            passed=_REQUIRED_FAILURE_PATTERNS.issubset(
                set(stage130_summary.get("failure_pattern_ids") or [])
            ),
            observed=stage130_summary.get("failure_pattern_ids"),
            expected=sorted(_REQUIRED_FAILURE_PATTERNS),
        ),
        _check(
            name="stage131_configs_are_prefix_protected",
            passed=all(
                int(config["protected_prefix_slots"]) >= _ANSWER_CONTEXT_DEPTH
                - _MAX_REPLACEMENT_APPEND_SLOTS
                and int(config["replacement_append_slots"]) <= _MAX_REPLACEMENT_APPEND_SLOTS
                for config in configs
            ),
            observed=[
                {
                    "config_id": config["config_id"],
                    "protected_prefix_slots": config["protected_prefix_slots"],
                    "replacement_append_slots": config["replacement_append_slots"],
                }
                for config in configs
            ],
            expected=(
                "at least 8 protected prefix slots and at most 2 replacement "
                "append slots"
            ),
        ),
        _check(
            name="stage131_has_conservative_sidecar_control",
            passed=any(
                config["config_id"] == "prefix10_append_sidecar_probe_v1"
                and int(config["replacement_append_slots"]) == 0
                and config["append_sidecar_can_generate_answer_text"] is False
                for config in configs
            ),
            observed=[config["config_id"] for config in configs],
            expected="prefix10 append sidecar probe with zero replacement slots",
        ),
        _check(
            name="stage131_validation_uses_train_cv_and_dev_report_only",
            passed=plan["selection_split"] == "train"
            and int(plan["minimum_train_folds"]) >= _MINIMUM_TRAIN_FOLDS
            and plan["validation_split"] == "dev"
            and plan["dev_mode"] == "single_pass_report_only_no_retuning",
            observed=plan,
            expected="train grouped-CV selection and dev report-only",
        ),
        _check(
            name="stage131_citation_guard_is_primary",
            passed=plan["primary_train_cv_guard"]
            == "gold_citation_count_delta_vs_stage116_non_negative",
            observed=plan["primary_train_cv_guard"],
            expected="gold citation count must not regress",
        ),
        _check(
            name="stage131_test_locked",
            passed=plan["test_rules"]["test_access_allowed"] is False
            and plan["test_rules"]["final_test_metrics_allowed"] is False
            and plan["test_rules"]["test_tuning_allowed"] is False
            and stage130_summary.get("can_run_final_test_metrics_now") is False
            and stage130_summary.get("can_use_test_for_tuning") is False,
            observed={
                "stage130_can_run_final_test_metrics_now": stage130_summary.get(
                    "can_run_final_test_metrics_now"
                ),
                "stage130_can_use_test_for_tuning": stage130_summary.get(
                    "can_use_test_for_tuning"
                ),
                "stage131_test_rules": plan["test_rules"],
            },
            expected="test locked",
        ),
        _check(
            name="stage131_runtime_defaults_unchanged",
            passed=plan["runtime_rules"]["default_runtime_policy"] == "unchanged"
            and plan["runtime_rules"]["runtime_defaultization_allowed_in_stage131"]
            is False
            and stage130_summary.get("runtime_defaultization_allowed_now") is False
            and stage130_summary.get("default_runtime_policy") == "unchanged",
            observed={
                "stage130_runtime_defaultization_allowed_now": stage130_summary.get(
                    "runtime_defaultization_allowed_now"
                ),
                "stage130_default_runtime_policy": stage130_summary.get(
                    "default_runtime_policy"
                ),
                "stage131_runtime_rules": plan["runtime_rules"],
            },
            expected="runtime default unchanged",
        ),
        _check(
            name="stage131_no_fallback_strategies",
            passed=plan["runtime_rules"]["fallback_strategies_enabled"] is False
            and stage130_summary.get("fallback_strategies_enabled") is False,
            observed={
                "stage130_fallback": stage130_summary.get("fallback_strategies_enabled"),
                "stage131_fallback": plan["runtime_rules"][
                    "fallback_strategies_enabled"
                ],
            },
            expected=False,
        ),
        _check(
            name="stage131_public_safe_contract_passed",
            passed=public_safe["forbidden_keys_found"] == [],
            observed=public_safe["forbidden_keys_found"],
            expected=[],
        ),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    base = {
        "protocol_id": _PROTOCOL_ID,
        "recommended_next_direction": _NEXT_DIRECTION,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "runtime_defaultization_allowed_now": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }
    if failed_checks:
        return {
            **base,
            "status": (
                "primeqa_hybrid_append_candidate_evidence_shortlist_"
                "redesign_protocol_blocked"
            ),
            "failed_checks": failed_checks,
            "can_run_append_shortlist_validation_now": False,
            "can_continue_train_dev_development": False,
        }
    return {
        **base,
        "status": (
            "primeqa_hybrid_append_candidate_evidence_shortlist_"
            "redesign_protocol_frozen"
        ),
        "failed_checks": [],
        "can_run_append_shortlist_validation_now": True,
        "can_continue_train_dev_development": True,
        "stage128_direct_agent_integration_path_remains_blocked": True,
    }


def _source_failure_pressure_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = report["stage130_summary"]
    return [
        BarDatum(
            label="train gold hit delta",
            value=float(summary.get("train_gold_hit_delta") or 0),
            value_label=f"{int(summary.get('train_gold_hit_delta') or 0):+d}",
        ),
        BarDatum(
            label="train gold citation delta",
            value=float(summary.get("train_gold_citation_delta") or 0),
            value_label=f"{int(summary.get('train_gold_citation_delta') or 0):+d}",
        ),
        BarDatum(
            label="train append selected citations",
            value=float(summary.get("train_append_selected_citations") or 0),
            value_label=str(summary.get("train_append_selected_citations") or 0),
        ),
        BarDatum(
            label="train prefix-like citation delta",
            value=float(summary.get("train_prefix_like_selected_citation_delta") or 0),
            value_label=f"{int(summary.get('train_prefix_like_selected_citation_delta') or 0):+d}",
        ),
        BarDatum(
            label="train changed answer rate percent",
            value=float(summary.get("train_changed_answer_rate") or 0) * 100,
            value_label=f"{float(summary.get('train_changed_answer_rate') or 0):.2%}",
        ),
    ]


def _candidate_budget_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    configs = report["frozen_protocol"]["candidate_shortlist_configs"]
    bars: list[BarDatum] = []
    for config in configs:
        bars.append(
            BarDatum(
                label=f"{config['config_id']} protected prefix",
                value=float(config["protected_prefix_slots"]),
                value_label=str(config["protected_prefix_slots"]),
            )
        )
        bars.append(
            BarDatum(
                label=f"{config['config_id']} replacement append",
                value=float(config["replacement_append_slots"]),
                value_label=str(config["replacement_append_slots"]),
            )
        )
    return bars


def _validation_guard_threshold_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    plan = report["frozen_protocol"]["selection_and_validation_plan"]
    return [
        BarDatum(
            label="minimum train folds",
            value=float(plan["minimum_train_folds"]),
            value_label=str(plan["minimum_train_folds"]),
        ),
        BarDatum(
            label="minimum gold citation delta",
            value=0.0,
            value_label=">= 0",
        ),
        BarDatum(
            label="maximum replacement append slots",
            value=float(_MAX_REPLACEMENT_APPEND_SLOTS),
            value_label=str(_MAX_REPLACEMENT_APPEND_SLOTS),
        ),
        BarDatum(
            label="answer context depth",
            value=float(_ANSWER_CONTEXT_DEPTH),
            value_label=str(_ANSWER_CONTEXT_DEPTH),
        ),
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    flags = (
        "can_run_append_shortlist_validation_now",
        "can_continue_train_dev_development",
        "can_open_final_test_gate_now",
        "can_run_final_test_metrics_now",
        "can_use_test_for_tuning",
        "runtime_defaultization_allowed_now",
        "fallback_strategies_enabled",
        "stage128_direct_agent_integration_path_remains_blocked",
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


def _check(*, name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "public_safe_summary_only": True,
        "raw_question_text_written": False,
        "raw_answer_text_written": False,
        "raw_document_text_written": False,
        "raw_document_ids_written": False,
        "raw_candidate_rows_written": False,
        "raw_sample_ids_written": False,
        "test_split_loaded": False,
        "final_test_metrics_run": False,
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


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
