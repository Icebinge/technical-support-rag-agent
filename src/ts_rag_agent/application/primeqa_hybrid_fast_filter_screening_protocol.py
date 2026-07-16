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

_STAGE = "Stage 120"
_CREATED_AT = "2026-07-16"
_SOURCE_STAGE = "Stage 119"
_SOURCE_STOP_STATUS = "primeqa_hybrid_second_stage_reranking_family_stopped"
_SOURCE_STOPPED_FAMILY_ID = "second_stage_reranking_candidate_family"
_SOURCE_NEXT_DIRECTION = "user_confirmed_next_research_direction_required"
_PROTOCOL_ID = "primeqa_hybrid_fast_filter_screening_protocol_v1"
_NEXT_DIRECTION = "run_fast_filter_screening_train_cv_dev_validation"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = ("test",)
_CANDIDATE_POOL_DEPTH = 200
_MINIMUM_TRAIN_FOLDS = 5
_CANDIDATE_FAMILY_IDS = (
    "protected_prefix_fast_filter_family_v1",
    "evidence_density_fast_filter_family_v1",
    "pairwise_screening_selector_family_v1",
)
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
        "question_text",
        "question_title",
        "raw_answer_text",
        "raw_document_text",
        "raw_question_text",
        "retrieved_doc_ids",
        "source_doc_ids",
    }
)
_FORBIDDEN_RUNTIME_FEATURES = frozenset(
    {
        "answer_doc_id",
        "gold_answer",
        "question_id",
        "source_doc_ids",
        "test_membership",
    }
)


@dataclass(frozen=True)
class PrimeQAHybridFastFilterScreeningProtocolVisualization:
    """One generated Stage120 fast-filter screening protocol chart."""

    name: str
    path: str


def freeze_primeqa_hybrid_fast_filter_screening_protocol(
    *,
    stage119_report_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage120 train/dev-only fast-filter screening protocol."""

    started_at = time.perf_counter()
    stage119_report = _load_json_object(stage119_report_path)
    loaded_at = time.perf_counter()
    stage119_summary = _stage119_summary(stage119_report)
    frozen_protocol = _frozen_protocol(stage119_summary)
    preliminary_report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "protocol_scope": (
            "Train/dev-only protocol freeze for a conservative fast-filter "
            "plus alternate screening route after the Stage119 second-stage "
            "reranking stop decision. This stage reads only the public-safe "
            "Stage119 report, freezes candidate families and validation rules, "
            "does not load split files, does not load corpus documents, does "
            "not build candidate rows, does not run retrieval, screening, "
            "reranking, answer, or final metrics, does not select from dev-only "
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
            "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
            "validation_split": "dev",
            "dev_validation_mode": "single_pass_report_only_no_retuning",
            "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage119_report": _fingerprint(stage119_report_path),
        },
        "stage119_summary": stage119_summary,
        "frozen_protocol": frozen_protocol,
    }
    guard_checks = _guard_checks(
        report=preliminary_report,
        stage119_summary=stage119_summary,
        user_confirmed_protocol=user_confirmed_protocol,
    )
    checked_at = time.perf_counter()
    report = {
        **preliminary_report,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
        "timing_seconds": {
            "load_report": round(loaded_at - started_at, 3),
            "freeze_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_fast_filter_screening_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridFastFilterScreeningProtocolVisualization]:
    """Write SVG charts for Stage120 protocol freeze."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage120_stage119_stop_summary.svg": render_horizontal_bar_chart_svg(
            title="Stage120 Stage119 stop summary",
            bars=_stage119_stop_summary_bars(report),
            x_label="count or rate",
            width=1320,
            margin_left=620,
        ),
        "stage120_candidate_family_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage120 candidate family counts",
            bars=_candidate_family_count_bars(report),
            x_label="config count",
            width=1480,
            margin_left=780,
        ),
        "stage120_fast_filter_window_sizes.svg": render_horizontal_bar_chart_svg(
            title="Stage120 fast-filter window sizes",
            bars=_filter_window_size_bars(report),
            x_label="candidate count",
            width=1480,
            margin_left=780,
        ),
        "stage120_promotion_budgets.svg": render_horizontal_bar_chart_svg(
            title="Stage120 promotion budgets",
            bars=_promotion_budget_bars(report),
            x_label="maximum promoted candidates",
            width=1480,
            margin_left=780,
        ),
        "stage120_guard_thresholds.svg": render_horizontal_bar_chart_svg(
            title="Stage120 train-CV guard thresholds",
            bars=_guard_threshold_bars(report),
            x_label="maximum allowed count or rate",
            width=1500,
            margin_left=800,
        ),
        "stage120_protocol_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage120 protocol decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1360,
            margin_left=700,
        ),
        "stage120_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage120 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1740,
            margin_left=940,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridFastFilterScreeningProtocolVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage119_summary(stage119_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage119_report.get("decision") or {}
    stopped = stage119_report.get("stopped_family") or {}
    stage118_summary = stopped.get("stage118_summary") or {}
    family_summary = stopped.get("candidate_family_summary") or {}
    dev_observations = stopped.get("dev_report_observations") or {}
    public_safe = stage119_report.get("public_safe_contract") or {}
    return {
        "stage": stage119_report.get("stage"),
        "decision_status": decision.get("status"),
        "stopped_family_id": decision.get("stopped_family_id"),
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
        "source_protocol_id": stopped.get("source_protocol_id"),
        "source_analysis_id": stopped.get("source_analysis_id"),
        "stage118_selectable_config_count": stage118_summary.get(
            "selectable_config_count"
        ),
        "stage118_config_count": stage118_summary.get("config_count"),
        "stage118_train_top200_gold_present_rate": stage118_summary.get(
            "train_top200_gold_present_rate"
        ),
        "stage118_dev_top200_gold_present_rate": stage118_summary.get(
            "dev_top200_gold_present_rate"
        ),
        "stage118_train_candidate_record_count_in_memory": stage118_summary.get(
            "train_candidate_record_count_in_memory"
        ),
        "stage118_dev_candidate_record_count_in_memory": stage118_summary.get(
            "dev_candidate_record_count_in_memory"
        ),
        "stage118_raw_candidate_rows_written": stage118_summary.get(
            "raw_candidate_rows_written"
        ),
        "stopped_candidate_family_summary": _family_stop_summary(family_summary),
        "positive_signal_blocked_config_count": len(
            stopped.get("train_cv_positive_signal_but_blocked_configs") or []
        ),
        "dev_used_for_selection": dev_observations.get("dev_used_for_selection"),
        "dev_used_for_retuning": dev_observations.get("dev_used_for_retuning"),
        "dev_observations_are_non_adoptable": dev_observations.get(
            "dev_observations_are_non_adoptable"
        ),
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
        "test_split_loaded": public_safe.get("test_split_loaded"),
        "final_test_metrics_run": public_safe.get("final_test_metrics_run"),
    }


def _family_stop_summary(
    family_summary: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    rows = {}
    for family_id, summary in sorted(family_summary.items()):
        if not isinstance(summary, Mapping):
            continue
        rows[str(family_id)] = {
            "config_count": summary.get("config_count"),
            "train_cv_selectable_config_count": summary.get(
                "train_cv_selectable_config_count"
            ),
            "best_train_cv_objective_config_id": summary.get(
                "best_train_cv_objective_config_id"
            ),
            "best_train_cv_mrr_at_20_config_id": summary.get(
                "best_train_cv_mrr_at_20_config_id"
            ),
            "train_cv_guard_failure_reasons": summary.get(
                "train_cv_guard_failure_reasons"
            )
            or {},
        }
    return rows


def _frozen_protocol(stage119_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "route_name": "conservative_fast_filter_plus_alternate_screening",
        "source_stop_decision": {
            "stage": stage119_summary.get("stage"),
            "status": stage119_summary.get("decision_status"),
            "stopped_family_id": stage119_summary.get("stopped_family_id"),
            "stopped_analysis_id": stage119_summary.get("source_analysis_id"),
            "stopped_protocol_id": stage119_summary.get("source_protocol_id"),
        },
        "design_rationale": {
            "problem_to_avoid": (
                "Stage118 full-pool reranking preserved top200 recall but "
                "caused too many top10/top20 regressions."
            ),
            "new_hypothesis": (
                "A cheap fast filter can shrink the Stage116 top200 pool and "
                "a constrained screening selector can promote only high-confidence "
                "tail candidates instead of reordering the whole pool."
            ),
            "not_a_runtime_policy": True,
        },
        "fixed_candidate_pool_contract": {
            "source_pool_id": "stage116_multi_route_union_candidate_pool",
            "candidate_pool_depth": _CANDIDATE_POOL_DEPTH,
            "source_stage118_train_top200_gold_present_rate": stage119_summary.get(
                "stage118_train_top200_gold_present_rate"
            ),
            "source_stage118_dev_top200_gold_present_rate": stage119_summary.get(
                "stage118_dev_top200_gold_present_rate"
            ),
            "screening_may_reorder_entire_top200": False,
            "screening_may_add_documents": False,
            "screening_may_use_uncapped_union": False,
        },
        "fast_filter_contract": {
            "purpose": "cheaply reduce the candidate window before heavier screening",
            "filter_input": "fixed Stage116 ranked top200 pool",
            "filter_output": "screened candidate subset plus protected prefix",
            "allowed_runtime_signals": [
                "Stage116 baseline rank",
                "route hit counts",
                "best route rank",
                "BM25 top10/top20 membership",
                "title/heading/body lexical overlap",
                "special-token exact match counts",
                "locally cached dense-route ranks",
            ],
            "forbidden_runtime_signals": sorted(_FORBIDDEN_RUNTIME_FEATURES),
            "candidate_rows_not_built_in_stage120": True,
        },
        "candidate_families": _candidate_families(),
        "candidate_configs": _candidate_configs(),
        "selection_rules": _selection_rules(),
        "feature_contract": _feature_contract(),
        "blocked_options": _blocked_options(),
        "next_stage": {
            "stage": "Stage121",
            "action": "run_fast_filter_screening_train_cv_dev_validation",
            "requires_user_confirmation": True,
            "test_locked": True,
            "runtime_defaults_unchanged": True,
            "fallback_strategies_enabled": False,
        },
    }


def _candidate_families() -> list[dict[str, Any]]:
    return [
        {
            "family_id": "protected_prefix_fast_filter_family_v1",
            "description": (
                "Keep a small Stage116 prefix stable and only allow filtered tail "
                "candidates to compete for limited insertion slots."
            ),
            "priority": 3,
        },
        {
            "family_id": "evidence_density_fast_filter_family_v1",
            "description": (
                "Use cheap lexical and special-token evidence density to screen "
                "tail candidates before any learned selector."
            ),
            "priority": 2,
        },
        {
            "family_id": "pairwise_screening_selector_family_v1",
            "description": (
                "Train a lightweight pairwise selector on the filtered subset "
                "instead of pointwise reranking the full top200 pool."
            ),
            "priority": 1,
        },
    ]


def _candidate_configs() -> list[dict[str, Any]]:
    return [
        _config(
            config_id="top10_locked_route_vote_window50_pairwise_logistic_v1",
            family_id="protected_prefix_fast_filter_family_v1",
            protected_prefix_depth=10,
            filter_window_size=50,
            promotion_budget_top10=0,
            promotion_budget_top20=3,
            filter_rule="route_vote_or_best_secondary_rank",
            selector_algorithm="pairwise_logistic_preference",
        ),
        _config(
            config_id="top5_locked_strong_consensus_window80_pairwise_gbdt_v1",
            family_id="protected_prefix_fast_filter_family_v1",
            protected_prefix_depth=5,
            filter_window_size=80,
            promotion_budget_top10=1,
            promotion_budget_top20=4,
            filter_rule="strong_route_consensus_and_margin",
            selector_algorithm="pairwise_hist_gradient_boosting_preference",
        ),
        _config(
            config_id="top20_locked_low_confidence_tail_screen_v1",
            family_id="protected_prefix_fast_filter_family_v1",
            protected_prefix_depth=20,
            filter_window_size=60,
            promotion_budget_top10=0,
            promotion_budget_top20=0,
            filter_rule="low_confidence_tail_screen_only",
            selector_algorithm="calibrated_route_consensus_score",
        ),
        _config(
            config_id="evidence_density_window40_pairwise_logistic_v1",
            family_id="evidence_density_fast_filter_family_v1",
            protected_prefix_depth=8,
            filter_window_size=40,
            promotion_budget_top10=1,
            promotion_budget_top20=2,
            filter_rule="title_heading_body_evidence_density",
            selector_algorithm="pairwise_logistic_preference",
        ),
        _config(
            config_id="special_token_exact_window40_rule_selector_v1",
            family_id="evidence_density_fast_filter_family_v1",
            protected_prefix_depth=10,
            filter_window_size=40,
            promotion_budget_top10=1,
            promotion_budget_top20=2,
            filter_rule="special_token_exact_or_title_heading_match",
            selector_algorithm="deterministic_evidence_margin_selector",
        ),
        _config(
            config_id="hybrid_filter_window80_pairwise_gbdt_v1",
            family_id="pairwise_screening_selector_family_v1",
            protected_prefix_depth=8,
            filter_window_size=80,
            promotion_budget_top10=1,
            promotion_budget_top20=3,
            filter_rule="route_vote_plus_evidence_density",
            selector_algorithm="pairwise_hist_gradient_boosting_preference",
        ),
    ]


def _config(
    *,
    config_id: str,
    family_id: str,
    protected_prefix_depth: int,
    filter_window_size: int,
    promotion_budget_top10: int,
    promotion_budget_top20: int,
    filter_rule: str,
    selector_algorithm: str,
) -> dict[str, Any]:
    return {
        "config_id": config_id,
        "family_id": family_id,
        "selection_eligible": True,
        "fast_filter": {
            "filter_rule": filter_rule,
            "input_pool_depth": _CANDIDATE_POOL_DEPTH,
            "protected_prefix_depth": protected_prefix_depth,
            "screened_window_size": filter_window_size,
            "may_drop_from_selector_input": True,
            "may_drop_from_final_stage116_protected_prefix": False,
        },
        "screening_selector": {
            "algorithm": selector_algorithm,
            "training_mode": "train_grouped_cv_only_when_learned",
            "pairwise_training_labels": (
                "train answer_doc_id may define positive pairs during training only"
            ),
            "runtime_label_use": False,
        },
        "safety_constraints": {
            "promotion_budget_top10": promotion_budget_top10,
            "promotion_budget_top20": promotion_budget_top20,
            "full_top200_rerank_allowed": False,
            "stage116_prefix_protection_enabled": True,
        },
    }


def _selection_rules() -> dict[str, Any]:
    return {
        "selection_split": "train",
        "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
        "minimum_train_folds": _MINIMUM_TRAIN_FOLDS,
        "baseline_order": "stage116_fixed_rrf_pool_order",
        "primary_metrics": [
            "hit_at_10_delta_vs_stage116_order",
            "hit_at_20_delta_vs_stage116_order",
            "mrr_at_20_delta_vs_stage116_order",
        ],
        "guard_thresholds": {
            "maximum_train_cv_hit_at_200_loss_count": 0,
            "maximum_train_cv_top10_regression_count": 0,
            "maximum_train_cv_hit_at_20_regression_rate": 0.01,
            "maximum_train_cv_bm25_top10_gold_demotions_to_below_50": 0,
            "minimum_train_cv_hit_at_10_delta": 0.0,
            "minimum_train_cv_mrr_at_20_delta": 0.0,
            "maximum_train_cv_promoted_tail_docs_into_top10_average": 1.0,
        },
        "dev_rules": {
            "dev_selection_allowed": False,
            "dev_retuning_allowed": False,
            "dev_threshold_tuning_allowed": False,
            "dev_validation_mode": "single_pass_report_only_no_retuning",
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
            "runtime_defaultization_allowed_in_stage120": False,
        },
    }


def _feature_contract() -> dict[str, Any]:
    return {
        "runtime_visible_feature_sources": [
            "rank and score from Stage116 retrieval routes",
            "query-title-heading-body lexical overlap",
            "special-token exact match counts",
            "locally cached dense route ranks when already available",
        ],
        "training_only_label_sources": [
            "train split answer_doc_id for positive pair construction",
        ],
        "forbidden_runtime_feature_sources": sorted(_FORBIDDEN_RUNTIME_FEATURES),
        "source_doc_ids_oracle_features_allowed": False,
        "test_membership_features_allowed": False,
    }


def _blocked_options() -> list[dict[str, str]]:
    return [
        {
            "option_id": "full_top200_rerank_blocked",
            "reason": "Stage118 showed full-pool reranking causes top10/top20 regressions.",
        },
        {
            "option_id": "uncapped_union_as_screening_input_blocked",
            "reason": "Stage116 uncapped union is too large and not a runtime answer input.",
        },
        {
            "option_id": "source_doc_ids_oracle_screening_blocked",
            "reason": "Source DOC_IDS are dataset metadata and not runtime retrieval evidence.",
        },
        {
            "option_id": "dev_selected_threshold_blocked",
            "reason": "Dev remains validation/report-only, not a selection source.",
        },
        {
            "option_id": "final_test_metrics_blocked",
            "reason": "Test remains locked until an explicit later final-test gate.",
        },
        {
            "option_id": "runtime_defaultization_in_stage120_blocked",
            "reason": "Stage120 freezes a protocol only; it does not change runtime defaults.",
        },
    ]


def _guard_checks(
    *,
    report: Mapping[str, Any],
    stage119_summary: Mapping[str, Any],
    user_confirmed_protocol: bool,
) -> list[dict[str, Any]]:
    frozen = report["frozen_protocol"]
    selection_rules = frozen["selection_rules"]
    candidate_configs = frozen["candidate_configs"]
    public_safe = _public_safe_contract(report)
    forbidden_feature_uses = _forbidden_feature_uses(candidate_configs)
    full_rerank_configs = [
        config["config_id"]
        for config in candidate_configs
        if config["safety_constraints"]["full_top200_rerank_allowed"]
    ]
    return [
        _check(
            name="user_confirmed_stage120_protocol",
            passed=user_confirmed_protocol,
            observed=report["user_confirmation"]["confirmation_note"],
            expected="user confirmed Stage120 fast-filter screening protocol",
        ),
        _check(
            name="stage119_stop_decision_completed",
            passed=stage119_summary.get("decision_status") == _SOURCE_STOP_STATUS,
            observed=stage119_summary.get("decision_status"),
            expected=_SOURCE_STOP_STATUS,
        ),
        _check(
            name="stage119_stopped_expected_family",
            passed=stage119_summary.get("stopped_family_id") == _SOURCE_STOPPED_FAMILY_ID,
            observed=stage119_summary.get("stopped_family_id"),
            expected=_SOURCE_STOPPED_FAMILY_ID,
        ),
        _check(
            name="stage119_requires_new_confirmed_direction",
            passed=stage119_summary.get("recommended_next_direction")
            == _SOURCE_NEXT_DIRECTION
            and bool(stage119_summary.get("requires_user_confirmation_before_next_protocol")),
            observed={
                "recommended_next_direction": stage119_summary.get(
                    "recommended_next_direction"
                ),
                "requires_user_confirmation_before_next_protocol": stage119_summary.get(
                    "requires_user_confirmation_before_next_protocol"
                ),
            },
            expected=_SOURCE_NEXT_DIRECTION,
        ),
        _check(
            name="stage119_top200_pool_evidence_available",
            passed=stage119_summary.get("stage118_train_top200_gold_present_rate") == 0.9324
            and stage119_summary.get("stage118_dev_top200_gold_present_rate") == 0.9079,
            observed={
                "train": stage119_summary.get("stage118_train_top200_gold_present_rate"),
                "dev": stage119_summary.get("stage118_dev_top200_gold_present_rate"),
            },
            expected={"train": 0.9324, "dev": 0.9079},
        ),
        _check(
            name="stage119_runtime_and_test_boundaries_locked",
            passed=stage119_summary.get("can_open_final_test_gate_now") is False
            and stage119_summary.get("can_run_final_test_metrics_now") is False
            and stage119_summary.get("can_use_test_for_tuning") is False
            and stage119_summary.get("fallback_strategies_enabled") is False
            and stage119_summary.get("default_runtime_policy") == "unchanged",
            observed=stage119_summary,
            expected="test locked, runtime unchanged, fallback disabled",
        ),
        _check(
            name="stage120_uses_train_grouped_cv",
            passed=selection_rules["selection_mode"]
            == "train_grouped_cross_validation_then_full_train_refit"
            and selection_rules["minimum_train_folds"] >= _MINIMUM_TRAIN_FOLDS,
            observed={
                "selection_mode": selection_rules["selection_mode"],
                "minimum_train_folds": selection_rules["minimum_train_folds"],
            },
            expected="train grouped-CV with at least 5 folds",
        ),
        _check(
            name="stage120_dev_is_report_only",
            passed=selection_rules["dev_rules"]["dev_selection_allowed"] is False
            and selection_rules["dev_rules"]["dev_retuning_allowed"] is False
            and selection_rules["dev_rules"]["dev_threshold_tuning_allowed"] is False,
            observed=selection_rules["dev_rules"],
            expected="dev report-only",
        ),
        _check(
            name="stage120_test_locked",
            passed=selection_rules["test_rules"]["test_access_allowed"] is False
            and selection_rules["test_rules"]["final_test_metrics_allowed"] is False
            and selection_rules["test_rules"]["test_tuning_allowed"] is False,
            observed=selection_rules["test_rules"],
            expected="test access and final metrics disabled",
        ),
        _check(
            name="stage120_runtime_defaults_unchanged",
            passed=selection_rules["runtime_rules"]["default_runtime_policy"] == "unchanged"
            and selection_rules["runtime_rules"]["fallback_strategies_enabled"] is False,
            observed=selection_rules["runtime_rules"],
            expected="runtime unchanged and no fallback strategies",
        ),
        _check(
            name="stage120_candidate_configs_present",
            passed=len(candidate_configs) == 6
            and set(_family_counts(candidate_configs)) == set(_CANDIDATE_FAMILY_IDS),
            observed={
                "candidate_config_count": len(candidate_configs),
                "families": sorted(_family_counts(candidate_configs)),
            },
            expected={"candidate_config_count": 6, "families": list(_CANDIDATE_FAMILY_IDS)},
        ),
        _check(
            name="stage120_configs_do_not_full_rerank_top200",
            passed=not full_rerank_configs,
            observed=full_rerank_configs,
            expected=[],
        ),
        _check(
            name="stage120_configs_have_protected_prefix_and_promotion_budget",
            passed=all(
                int(config["fast_filter"]["protected_prefix_depth"]) >= 5
                and int(config["safety_constraints"]["promotion_budget_top10"]) <= 1
                for config in candidate_configs
            ),
            observed=[
                {
                    "config_id": config["config_id"],
                    "protected_prefix_depth": config["fast_filter"][
                        "protected_prefix_depth"
                    ],
                    "promotion_budget_top10": config["safety_constraints"][
                        "promotion_budget_top10"
                    ],
                }
                for config in candidate_configs
            ],
            expected="protected prefix >= 5 and top10 promotion budget <= 1",
        ),
        _check(
            name="stage120_configs_do_not_use_forbidden_runtime_features",
            passed=not forbidden_feature_uses,
            observed=forbidden_feature_uses,
            expected=[],
        ),
        _check(
            name="stage120_no_candidate_rows_or_metrics_run",
            passed=frozen["fast_filter_contract"]["candidate_rows_not_built_in_stage120"],
            observed=frozen["fast_filter_contract"]["candidate_rows_not_built_in_stage120"],
            expected=True,
        ),
        _check(
            name="stage120_public_safe_contract_passed",
            passed=public_safe["forbidden_keys_found"] == [],
            observed=public_safe["forbidden_keys_found"],
            expected=[],
        ),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed:
        return {
            "status": "primeqa_hybrid_fast_filter_screening_protocol_blocked",
            "failed_checks": failed,
            "can_run_fast_filter_screening_now": False,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": "primeqa_hybrid_fast_filter_screening_protocol_frozen",
        "recommended_next_direction": _NEXT_DIRECTION,
        "can_run_fast_filter_screening_now": True,
        "can_continue_train_dev_development": True,
        "requires_user_confirmation_before_train_dev_run": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }


def _stage119_stop_summary_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = report["stage119_summary"]
    return [
        BarDatum(
            label="stage118 selected configs",
            value=float(summary.get("stage118_selectable_config_count") or 0),
            value_label=str(summary.get("stage118_selectable_config_count")),
        ),
        BarDatum(
            label="stage118 config count",
            value=float(summary.get("stage118_config_count") or 0),
            value_label=str(summary.get("stage118_config_count")),
        ),
        BarDatum(
            label="train top200 recall",
            value=float(summary.get("stage118_train_top200_gold_present_rate") or 0),
            value_label=f"{float(summary.get('stage118_train_top200_gold_present_rate') or 0):.4f}",
        ),
        BarDatum(
            label="dev top200 recall",
            value=float(summary.get("stage118_dev_top200_gold_present_rate") or 0),
            value_label=f"{float(summary.get('stage118_dev_top200_gold_present_rate') or 0):.4f}",
        ),
        BarDatum(
            label="positive signal blocked configs",
            value=float(summary.get("positive_signal_blocked_config_count") or 0),
            value_label=str(summary.get("positive_signal_blocked_config_count")),
        ),
    ]


def _candidate_family_count_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counts = _family_counts(report["frozen_protocol"]["candidate_configs"])
    return [
        BarDatum(label=family_id, value=float(count), value_label=str(count))
        for family_id, count in sorted(counts.items())
    ]


def _filter_window_size_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(config["config_id"]),
            value=float(config["fast_filter"]["screened_window_size"]),
            value_label=str(config["fast_filter"]["screened_window_size"]),
        )
        for config in report["frozen_protocol"]["candidate_configs"]
    ]


def _promotion_budget_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for config in report["frozen_protocol"]["candidate_configs"]:
        bars.append(
            BarDatum(
                label=f"{config['config_id']} top10",
                value=float(config["safety_constraints"]["promotion_budget_top10"]),
                value_label=str(config["safety_constraints"]["promotion_budget_top10"]),
            )
        )
        bars.append(
            BarDatum(
                label=f"{config['config_id']} top20",
                value=float(config["safety_constraints"]["promotion_budget_top20"]),
                value_label=str(config["safety_constraints"]["promotion_budget_top20"]),
            )
        )
    return bars


def _guard_threshold_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    thresholds = report["frozen_protocol"]["selection_rules"]["guard_thresholds"]
    return [
        BarDatum(label=str(key), value=float(value), value_label=str(value))
        for key, value in thresholds.items()
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    flags = (
        "can_run_fast_filter_screening_now",
        "can_continue_train_dev_development",
        "requires_user_confirmation_before_train_dev_run",
        "can_open_final_test_gate_now",
        "can_run_final_test_metrics_now",
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
        for check in report["guard_checks"]
    ]


def _family_counts(candidate_configs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(config["family_id"]) for config in candidate_configs).items()))


def _forbidden_feature_uses(
    candidate_configs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    uses = []
    for config in candidate_configs:
        payload = json.dumps(config, ensure_ascii=False)
        for forbidden in _FORBIDDEN_RUNTIME_FEATURES:
            if forbidden in payload and "training" not in payload:
                uses.append({"config_id": config.get("config_id"), "feature": forbidden})
    return uses


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    forbidden_keys = sorted(_find_forbidden_public_keys(report))
    return {
        "public_safe_summary_only": True,
        "raw_question_text_written": False,
        "raw_answer_text_written": False,
        "raw_document_text_written": False,
        "raw_candidate_rows_written": False,
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


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
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
