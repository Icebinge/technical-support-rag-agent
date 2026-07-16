from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    _fingerprint,
    _load_json_object,
    _rounded_ratio,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 127"
_CREATED_AT = "2026-07-16"
_REVIEW_ID = (
    "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_"
    "selected_config_review_v1"
)
_SOURCE_STAGE126_STATUS = (
    "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_validation_completed"
)
_SOURCE_STAGE126_ANALYSIS_ID = (
    "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_validation_v1"
)
_SOURCE_STAGE126_NEXT_DIRECTION = (
    "review_stage116_prefix_preserving_recall_expansion_selected_config"
)
_NEXT_DIRECTION = (
    "freeze_agent_retrieval_integration_protocol_for_selected_prefix_expansion"
)
_SELECTED_CONFIG_ID = "prefix_existing_dense_broad_append200_v1"
_SELECTED_FAMILY_ID = "stage116_prefix_existing_dense_append_family_v1"
_BASELINE_PREFIX_DEPTH = 200
_TARGET_POOL_DEPTH = 400
_APPEND_BUDGET = 200
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
class PrimeQAHybridPrefixPreservingRecallExpansionSelectedConfigReviewVisualization:
    """One generated Stage127 selected-config review chart."""

    name: str
    path: str


def review_primeqa_hybrid_prefix_preserving_recall_expansion_selected_config(
    *,
    stage126_report_path: Path,
    user_confirmed_review: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Review the Stage126 selected prefix-preserving expansion config."""

    started_at = time.perf_counter()
    stage126_report = _load_json_object(stage126_report_path)
    loaded_at = time.perf_counter()

    stage126_summary = _stage126_summary(stage126_report)
    selected_config_review = _selected_config_review(stage126_report)
    config_landscape = _config_landscape(stage126_report)
    agent_design_review = _agent_design_review(
        stage126_summary=stage126_summary,
        selected_config_review=selected_config_review,
        config_landscape=config_landscape,
    )
    preliminary_report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "review_id": _REVIEW_ID,
        "review_scope": (
            "Train/dev-only selected-config review after Stage126 validated the "
            "Stage116 prefix-preserving append-only recall expansion family. "
            "This stage reads only the public-safe Stage126 report, does not "
            "load split files, does not load corpus documents, does not build "
            "candidate rows, does not run retrieval, reranking, answering, or "
            "final metrics, does not select from dev-only observations, does "
            "not add fallback strategies, and does not change runtime defaults."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_review),
            "confirmation_note": confirmation_note,
        },
        "source_files": {
            "stage126_report": _fingerprint(stage126_report_path),
        },
        "stage126_summary": stage126_summary,
        "selected_config_review": selected_config_review,
        "config_landscape": config_landscape,
        "agent_design_review": agent_design_review,
    }
    guard_checks = _guard_checks(
        report=preliminary_report,
        user_confirmed_review=user_confirmed_review,
    )
    checked_at = time.perf_counter()
    report = {
        **preliminary_report,
        "guard_checks": guard_checks,
        "decision": _decision(
            guard_checks=guard_checks,
            selected_config_review=selected_config_review,
        ),
        "timing_seconds": {
            "load_report": round(loaded_at - started_at, 3),
            "review_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_prefix_preserving_recall_expansion_selected_config_review_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridPrefixPreservingRecallExpansionSelectedConfigReviewVisualization]:
    """Write SVG charts for Stage127 selected-config review."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage127_selected_incremental_recall.svg": render_horizontal_bar_chart_svg(
            title="Stage127 selected incremental recall",
            bars=_selected_incremental_recall_bars(report),
            x_label="hit-count gain",
            width=1320,
            margin_left=620,
        ),
        "stage127_config_train_dev_gain.svg": render_horizontal_bar_chart_svg(
            title="Stage127 config train/dev target-depth gain",
            bars=_config_train_dev_gain_bars(report),
            x_label="hit-count gain",
            width=1580,
            margin_left=820,
        ),
        "stage127_boundary_safety.svg": render_horizontal_bar_chart_svg(
            title="Stage127 selected boundary safety",
            bars=_boundary_safety_bars(report),
            x_label="count",
            width=1420,
            margin_left=700,
        ),
        "stage127_candidate_pool_shape.svg": render_horizontal_bar_chart_svg(
            title="Stage127 candidate pool shape",
            bars=_candidate_pool_shape_bars(report),
            x_label="documents",
            width=1320,
            margin_left=620,
        ),
        "stage127_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage127 decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1440,
            margin_left=760,
        ),
        "stage127_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage127 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1880,
            margin_left=1040,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridPrefixPreservingRecallExpansionSelectedConfigReviewVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage126_summary(stage126_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage126_report.get("decision") or {}
    train_selection = stage126_report.get("train_selection") or {}
    dev_report = stage126_report.get("dev_report_observations") or {}
    baseline = stage126_report.get("baseline_by_split") or {}
    public_safe = stage126_report.get("public_safe_contract") or {}
    guard_checks = stage126_report.get("guard_checks") or []
    return {
        "stage": stage126_report.get("stage"),
        "analysis_id": stage126_report.get("analysis_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "selected_config_id": decision.get("selected_config_id")
        or train_selection.get("selected_config_id"),
        "selected_family_id": decision.get("selected_family_id")
        or train_selection.get("selected_family_id"),
        "candidate_count": train_selection.get("candidate_count"),
        "eligible_config_count": train_selection.get("eligible_config_count"),
        "dev_used_for_selection": train_selection.get("dev_used_for_selection")
        or dev_report.get("dev_used_for_selection"),
        "dev_used_for_retuning": train_selection.get("dev_used_for_retuning")
        or dev_report.get("dev_used_for_retuning"),
        "train_baseline": _baseline_summary(baseline.get("train") or {}),
        "dev_baseline": _baseline_summary(baseline.get("dev") or {}),
        "guard_check_count": len(guard_checks),
        "guard_check_passed_count": sum(1 for check in guard_checks if check.get("passed")),
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
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
    }


def _baseline_summary(split_baseline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evaluated_questions": split_baseline.get("evaluated_questions"),
        "pool_depth": split_baseline.get("pool_depth"),
        "hit_at_200_count": split_baseline.get("hit_at_200_count"),
        "hit_at_200": split_baseline.get("hit_at_200"),
    }


def _selected_config_review(stage126_report: Mapping[str, Any]) -> dict[str, Any]:
    selected_id = (stage126_report.get("train_selection") or {}).get(
        "selected_config_id"
    )
    selected = next(
        (
            review
            for review in stage126_report.get("config_reviews") or []
            if review.get("config_id") == selected_id
        ),
        None,
    )
    if not selected:
        return {
            "selected_config_id": selected_id,
            "selected_config_found": False,
        }
    train = selected["split_reviews"]["train"]
    dev = selected["split_reviews"]["dev"]
    return {
        "selected_config_found": True,
        "config_id": selected.get("config_id"),
        "family_id": selected.get("family_id"),
        "source_stage124_config_id": selected.get("source_stage124_config_id"),
        "append_source_algorithm": selected.get("append_source_algorithm"),
        "route_set": selected.get("route_set"),
        "channel_top_k": selected.get("channel_top_k"),
        "append_budget": selected.get("append_budget"),
        "target_pool_depth": selected.get("target_pool_depth"),
        "train": _split_review_summary(train),
        "dev": _split_review_summary(dev),
        "train_fold_target_depth_gains": _fold_target_depth_gains(train),
        "train_guard_passed": (selected.get("train_cv_guard") or {}).get("passed"),
        "train_guard_failed_checks": (selected.get("train_cv_guard") or {}).get(
            "failed_checks"
        )
        or [],
    }


def _split_review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    evaluated = int(review.get("evaluated_questions") or 0)
    gain = int(review.get("target_depth_hit_count_gain_vs_stage116_top200") or 0)
    return {
        "evaluated_questions": evaluated,
        "baseline_hit_at_200_count": review.get("baseline_hit_at_200_count"),
        "hit_at_200_count": review.get("hit_at_200_count"),
        "hit_at_200_delta_vs_stage116_prefix": review.get(
            "hit_at_200_delta_vs_stage116_prefix"
        ),
        "hit_at_200_loss_count": review.get("hit_at_200_loss_count"),
        "target_depth_hit_count": review.get("target_depth_hit_count"),
        "target_depth_hit_rate": review.get("target_depth_hit_rate"),
        "target_depth_hit_count_gain_vs_stage116_top200": gain,
        "incremental_gain_rate": _rounded_ratio(gain, evaluated),
        "appended_gold_recovery_count": review.get("appended_gold_recovery_count"),
        "prefix_identity_violation_count": review.get(
            "prefix_identity_violation_count"
        ),
        "append_budget_exceeded_count": review.get("append_budget_exceeded_count"),
        "append_exhaustion_count": review.get("append_exhaustion_count"),
        "candidate_pool_size": review.get("candidate_pool_size") or {},
        "append_count": review.get("append_count") or {},
        "channel_count": review.get("channel_count"),
        "channel_families": review.get("channel_families") or {},
    }


def _fold_target_depth_gains(train_review: Mapping[str, Any]) -> dict[str, int]:
    fold_metrics = train_review.get("fold_metrics") or {}
    return {
        str(fold_id): int(
            metrics.get("target_depth_hit_count_gain_vs_stage116_top200") or 0
        )
        for fold_id, metrics in sorted(fold_metrics.items())
    }


def _config_landscape(stage126_report: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for review in stage126_report.get("config_reviews") or []:
        train = review["split_reviews"]["train"]
        dev = review["split_reviews"]["dev"]
        rows.append(
            {
                "config_id": review.get("config_id"),
                "family_id": review.get("family_id"),
                "append_budget": review.get("append_budget"),
                "target_pool_depth": review.get("target_pool_depth"),
                "train_target_depth_gain": train.get(
                    "target_depth_hit_count_gain_vs_stage116_top200"
                ),
                "dev_target_depth_gain": dev.get(
                    "target_depth_hit_count_gain_vs_stage116_top200"
                ),
                "train_hit_at_200_loss_count": train.get("hit_at_200_loss_count"),
                "dev_hit_at_200_loss_count": dev.get("hit_at_200_loss_count"),
                "train_prefix_identity_violation_count": train.get(
                    "prefix_identity_violation_count"
                ),
                "dev_prefix_identity_violation_count": dev.get(
                    "prefix_identity_violation_count"
                ),
                "train_guard_passed": (review.get("train_cv_guard") or {}).get(
                    "passed"
                ),
            }
        )
    ranked_by_dev = sorted(
        rows,
        key=lambda row: (
            -int(row.get("dev_target_depth_gain") or 0),
            int(row.get("target_pool_depth") or 0),
            str(row.get("config_id")),
        ),
    )
    return {
        "config_count": len(rows),
        "train_guard_passed_count": sum(1 for row in rows if row["train_guard_passed"]),
        "all_configs_prefix_safe": all(
            int(row.get("train_prefix_identity_violation_count") or 0) == 0
            and int(row.get("dev_prefix_identity_violation_count") or 0) == 0
            for row in rows
        ),
        "all_configs_hit200_safe": all(
            int(row.get("train_hit_at_200_loss_count") or 0) == 0
            and int(row.get("dev_hit_at_200_loss_count") or 0) == 0
            for row in rows
        ),
        "rows": sorted(
            rows,
            key=lambda row: (
                -int(row.get("train_target_depth_gain") or 0),
                int(row.get("target_pool_depth") or 0),
                str(row.get("config_id")),
            ),
        ),
        "best_dev_config": ranked_by_dev[0] if ranked_by_dev else None,
    }


def _agent_design_review(
    *,
    stage126_summary: Mapping[str, Any],
    selected_config_review: Mapping[str, Any],
    config_landscape: Mapping[str, Any],
) -> dict[str, Any]:
    selected_train = selected_config_review.get("train") or {}
    selected_dev = selected_config_review.get("dev") or {}
    best_dev = config_landscape.get("best_dev_config") or {}
    return {
        "review_status": "selected_config_supported_for_agent_protocol_design",
        "runtime_defaultization_allowed_now": False,
        "final_test_gate_allowed_now": False,
        "retrieval_contract": {
            "selected_config_id": selected_config_review.get("config_id"),
            "baseline_prefix_depth": _BASELINE_PREFIX_DEPTH,
            "append_start_rank": _BASELINE_PREFIX_DEPTH + 1,
            "append_budget": selected_config_review.get("append_budget"),
            "target_pool_depth": selected_config_review.get("target_pool_depth"),
            "rank_regions": [
                {
                    "region_id": "stage116_immutable_prefix",
                    "rank_start": 1,
                    "rank_end": _BASELINE_PREFIX_DEPTH,
                    "role": "preserve the validated Stage116 top200 boundary",
                    "may_reorder": False,
                    "may_drop": False,
                },
                {
                    "region_id": "stage126_append_expansion",
                    "rank_start": _BASELINE_PREFIX_DEPTH + 1,
                    "rank_end": selected_config_review.get("target_pool_depth"),
                    "role": "add recall candidates for downstream evidence selection",
                    "deduplicate_against_prefix": True,
                    "may_insert_before_rank_201": False,
                },
            ],
        },
        "observed_value": {
            "train_incremental_recall_gain_count": selected_train.get(
                "target_depth_hit_count_gain_vs_stage116_top200"
            ),
            "train_incremental_recall_gain_rate": selected_train.get(
                "incremental_gain_rate"
            ),
            "dev_incremental_recall_gain_count": selected_dev.get(
                "target_depth_hit_count_gain_vs_stage116_top200"
            ),
            "dev_incremental_recall_gain_rate": selected_dev.get(
                "incremental_gain_rate"
            ),
            "train_hit_at_200_loss_count": selected_train.get(
                "hit_at_200_loss_count"
            ),
            "dev_hit_at_200_loss_count": selected_dev.get("hit_at_200_loss_count"),
            "train_prefix_identity_violation_count": selected_train.get(
                "prefix_identity_violation_count"
            ),
            "dev_prefix_identity_violation_count": selected_dev.get(
                "prefix_identity_violation_count"
            ),
        },
        "cost_profile": {
            "baseline_candidate_depth": _BASELINE_PREFIX_DEPTH,
            "target_candidate_depth": selected_config_review.get("target_pool_depth"),
            "candidate_depth_multiplier_vs_stage116": _rounded_ratio(
                int(selected_config_review.get("target_pool_depth") or 0),
                _BASELINE_PREFIX_DEPTH,
            ),
            "additional_candidates_per_query": selected_config_review.get(
                "append_budget"
            ),
            "selected_train_average_append_count": (
                selected_train.get("append_count") or {}
            ).get("average"),
            "selected_dev_average_append_count": (
                selected_dev.get("append_count") or {}
            ).get("average"),
            "channel_count": selected_train.get("channel_count"),
            "channel_families": selected_train.get("channel_families") or {},
        },
        "risk_review": {
            "dev_gain_is_smaller_than_train_gain": (
                int(selected_dev.get("target_depth_hit_count_gain_vs_stage116_top200") or 0)
                < int(
                    selected_train.get(
                        "target_depth_hit_count_gain_vs_stage116_top200"
                    )
                    or 0
                )
            ),
            "best_dev_config_differs_from_train_selected": best_dev.get("config_id")
            != selected_config_review.get("config_id"),
            "best_dev_config_id": best_dev.get("config_id"),
            "best_dev_target_depth_gain": best_dev.get("dev_target_depth_gain"),
            "answer_quality_not_measured": True,
            "final_test_not_run": stage126_summary.get("can_run_final_test_metrics_now")
            is False,
            "runtime_default_unchanged": stage126_summary.get("default_runtime_policy")
            == "unchanged",
        },
        "agent_protocol_constraints": [
            "preserve Stage116 ranks 1-200 exactly",
            "append only deduplicated candidates after rank 200",
            "treat 400-depth output as a candidate pool, not as an automatic answer context",
            "keep dev report-only observations out of selection and threshold tuning",
            "do not use test membership, source DOC_IDS, answer document IDs, or gold labels",
            "do not add fallback strategies in the integration protocol",
            "do not change runtime defaults before a dedicated integration validation",
        ],
        "recommended_next_stage": {
            "stage": "Stage128",
            "action": _NEXT_DIRECTION,
            "requires_user_confirmation": True,
            "test_locked": True,
            "runtime_defaults_unchanged": True,
            "fallback_strategies_enabled": False,
        },
    }


def _guard_checks(
    *,
    report: Mapping[str, Any],
    user_confirmed_review: bool,
) -> list[dict[str, Any]]:
    stage126_summary = report["stage126_summary"]
    selected = report["selected_config_review"]
    selected_train = selected.get("train") or {}
    selected_dev = selected.get("dev") or {}
    landscape = report["config_landscape"]
    public_safe = _public_safe_contract(report)
    return [
        _check(
            name="user_confirmed_stage127_review",
            passed=user_confirmed_review,
            observed=report["user_confirmation"]["confirmation_note"],
            expected="user confirmed Stage127 selected-config review",
        ),
        _check(
            name="stage126_validation_completed",
            passed=stage126_summary.get("decision_status")
            == _SOURCE_STAGE126_STATUS,
            observed=stage126_summary.get("decision_status"),
            expected=_SOURCE_STAGE126_STATUS,
        ),
        _check(
            name="stage126_analysis_id_matches",
            passed=stage126_summary.get("analysis_id") == _SOURCE_STAGE126_ANALYSIS_ID,
            observed=stage126_summary.get("analysis_id"),
            expected=_SOURCE_STAGE126_ANALYSIS_ID,
        ),
        _check(
            name="stage126_recommends_selected_config_review",
            passed=stage126_summary.get("recommended_next_direction")
            == _SOURCE_STAGE126_NEXT_DIRECTION,
            observed=stage126_summary.get("recommended_next_direction"),
            expected=_SOURCE_STAGE126_NEXT_DIRECTION,
        ),
        _check(
            name="stage126_selected_config_present",
            passed=selected.get("selected_config_found") is True
            and selected.get("config_id") == _SELECTED_CONFIG_ID
            and selected.get("family_id") == _SELECTED_FAMILY_ID,
            observed={
                "selected_config_found": selected.get("selected_config_found"),
                "config_id": selected.get("config_id"),
                "family_id": selected.get("family_id"),
            },
            expected={
                "config_id": _SELECTED_CONFIG_ID,
                "family_id": _SELECTED_FAMILY_ID,
            },
        ),
        _check(
            name="stage126_all_configs_train_guard_passed",
            passed=landscape.get("train_guard_passed_count")
            == landscape.get("config_count")
            == 6,
            observed={
                "passed": landscape.get("train_guard_passed_count"),
                "total": landscape.get("config_count"),
            },
            expected="6 / 6",
        ),
        _check(
            name="selected_config_train_gain_positive",
            passed=int(
                selected_train.get("target_depth_hit_count_gain_vs_stage116_top200")
                or 0
            )
            > 0,
            observed=selected_train.get(
                "target_depth_hit_count_gain_vs_stage116_top200"
            ),
            expected="> 0",
        ),
        _check(
            name="selected_config_dev_gain_nonnegative",
            passed=int(
                selected_dev.get("target_depth_hit_count_gain_vs_stage116_top200") or 0
            )
            >= 0,
            observed=selected_dev.get("target_depth_hit_count_gain_vs_stage116_top200"),
            expected=">= 0",
        ),
        _check(
            name="selected_config_preserves_prefix_and_hit200",
            passed=int(selected_train.get("hit_at_200_loss_count") or 0) == 0
            and int(selected_dev.get("hit_at_200_loss_count") or 0) == 0
            and int(selected_train.get("prefix_identity_violation_count") or 0) == 0
            and int(selected_dev.get("prefix_identity_violation_count") or 0) == 0,
            observed={
                "train_hit_at_200_loss_count": selected_train.get(
                    "hit_at_200_loss_count"
                ),
                "dev_hit_at_200_loss_count": selected_dev.get(
                    "hit_at_200_loss_count"
                ),
                "train_prefix_identity_violation_count": selected_train.get(
                    "prefix_identity_violation_count"
                ),
                "dev_prefix_identity_violation_count": selected_dev.get(
                    "prefix_identity_violation_count"
                ),
            },
            expected="all zero",
        ),
        _check(
            name="stage126_all_configs_preserve_prefix_and_hit200",
            passed=landscape.get("all_configs_prefix_safe") is True
            and landscape.get("all_configs_hit200_safe") is True,
            observed={
                "all_configs_prefix_safe": landscape.get("all_configs_prefix_safe"),
                "all_configs_hit200_safe": landscape.get("all_configs_hit200_safe"),
            },
            expected=True,
        ),
        _check(
            name="stage126_baseline_reproduced",
            passed=(stage126_summary.get("train_baseline") or {}).get(
                "hit_at_200_count"
            )
            == 345
            and (stage126_summary.get("dev_baseline") or {}).get("hit_at_200_count")
            == 69,
            observed={
                "train": stage126_summary.get("train_baseline"),
                "dev": stage126_summary.get("dev_baseline"),
            },
            expected={"train_hit_at_200_count": 345, "dev_hit_at_200_count": 69},
        ),
        _check(
            name="stage126_dev_report_only",
            passed=stage126_summary.get("dev_used_for_selection") is False
            and stage126_summary.get("dev_used_for_retuning") is False,
            observed={
                "dev_used_for_selection": stage126_summary.get(
                    "dev_used_for_selection"
                ),
                "dev_used_for_retuning": stage126_summary.get(
                    "dev_used_for_retuning"
                ),
            },
            expected=False,
        ),
        _check(
            name="stage126_test_runtime_and_fallback_boundaries_locked",
            passed=stage126_summary.get("can_open_final_test_gate_now") is False
            and stage126_summary.get("can_run_final_test_metrics_now") is False
            and stage126_summary.get("can_use_test_for_tuning") is False
            and stage126_summary.get("fallback_strategies_enabled") is False
            and stage126_summary.get("default_runtime_policy") == "unchanged",
            observed=stage126_summary,
            expected="test locked, runtime unchanged, fallback disabled",
        ),
        _check(
            name="stage127_runtime_defaultization_not_allowed",
            passed=(
                report["agent_design_review"].get("runtime_defaultization_allowed_now")
                is False
            )
            and report["agent_design_review"].get("final_test_gate_allowed_now")
            is False,
            observed={
                "runtime_defaultization_allowed_now": report[
                    "agent_design_review"
                ].get("runtime_defaultization_allowed_now"),
                "final_test_gate_allowed_now": report["agent_design_review"].get(
                    "final_test_gate_allowed_now"
                ),
            },
            expected=False,
        ),
        _check(
            name="stage127_public_safe_contract_passed",
            passed=public_safe["forbidden_keys_found"] == [],
            observed=public_safe["forbidden_keys_found"],
            expected=[],
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    selected_config_review: Mapping[str, Any],
) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": (
                "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_"
                "selected_config_review_blocked"
            ),
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "runtime_defaultization_allowed_now": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": (
            "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_"
            "selected_config_review_completed"
        ),
        "recommended_next_direction": _NEXT_DIRECTION,
        "selected_config_id": selected_config_review.get("config_id"),
        "selected_family_id": selected_config_review.get("family_id"),
        "selected_config_supported_for_agent_protocol_design": True,
        "can_continue_train_dev_development": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "runtime_defaultization_allowed_now": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }


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
        "test_split_loaded": False,
        "final_test_metrics_run": False,
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


def _selected_incremental_recall_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    selected = report["selected_config_review"]
    return [
        BarDatum(
            label="train gain",
            value=float(
                selected["train"]["target_depth_hit_count_gain_vs_stage116_top200"]
            ),
            value_label=str(
                selected["train"]["target_depth_hit_count_gain_vs_stage116_top200"]
            ),
        ),
        BarDatum(
            label="dev gain",
            value=float(selected["dev"]["target_depth_hit_count_gain_vs_stage116_top200"]),
            value_label=str(
                selected["dev"]["target_depth_hit_count_gain_vs_stage116_top200"]
            ),
        ),
    ]


def _config_train_dev_gain_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for row in (report.get("config_landscape") or {}).get("rows") or []:
        bars.append(
            BarDatum(
                label=f"{row['config_id']} train",
                value=float(row["train_target_depth_gain"]),
                value_label=str(row["train_target_depth_gain"]),
            )
        )
        bars.append(
            BarDatum(
                label=f"{row['config_id']} dev",
                value=float(row["dev_target_depth_gain"]),
                value_label=str(row["dev_target_depth_gain"]),
            )
        )
    return bars


def _boundary_safety_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    selected = report["selected_config_review"]
    values = [
        (
            "train hit@200 loss",
            selected["train"]["hit_at_200_loss_count"],
        ),
        (
            "dev hit@200 loss",
            selected["dev"]["hit_at_200_loss_count"],
        ),
        (
            "train prefix violations",
            selected["train"]["prefix_identity_violation_count"],
        ),
        (
            "dev prefix violations",
            selected["dev"]["prefix_identity_violation_count"],
        ),
    ]
    return [
        BarDatum(label=label, value=float(value), value_label=str(value))
        for label, value in values
    ]


def _candidate_pool_shape_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    review = report["agent_design_review"]["retrieval_contract"]
    return [
        BarDatum(
            label="Stage116 prefix",
            value=float(review["baseline_prefix_depth"]),
            value_label=str(review["baseline_prefix_depth"]),
        ),
        BarDatum(
            label="append budget",
            value=float(review["append_budget"]),
            value_label=str(review["append_budget"]),
        ),
        BarDatum(
            label="target pool depth",
            value=float(review["target_pool_depth"]),
            value_label=str(review["target_pool_depth"]),
        ),
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    flags = (
        "selected_config_supported_for_agent_protocol_design",
        "can_continue_train_dev_development",
        "can_open_final_test_gate_now",
        "can_run_final_test_metrics_now",
        "can_use_test_for_tuning",
        "runtime_defaultization_allowed_now",
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
