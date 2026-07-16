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

_STAGE = "Stage 125"
_CREATED_AT = "2026-07-16"
_SOURCE_STAGE = "Stage 124"
_SOURCE_STAGE124_STATUS = (
    "primeqa_hybrid_first_stage_recall_expansion_validation_completed_no_selection"
)
_SOURCE_NEXT_DIRECTION = "design_stage116_prefix_preserving_recall_expansion_protocol"
_PROTOCOL_ID = "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_protocol_v1"
_NEXT_DIRECTION = (
    "run_stage116_prefix_preserving_recall_expansion_train_cv_dev_validation"
)
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = ("test",)
_BASELINE_CONFIG_ID = "stage116_fixed_rrf_top200_baseline"
_BASELINE_PREFIX_DEPTH = 200
_MAX_TARGET_POOL_DEPTH = 400
_MAX_CHANNEL_TOP_K = 400
_MINIMUM_TRAIN_FOLDS = 5
_CANDIDATE_FAMILY_IDS = (
    "stage116_prefix_rrf_append_family_v1",
    "stage116_prefix_existing_dense_append_family_v1",
    "stage116_prefix_query_variant_append_family_v1",
    "stage116_prefix_route_balanced_append_family_v1",
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
class PrimeQAHybridPrefixPreservingRecallExpansionProtocolVisualization:
    """One generated Stage125 prefix-preserving recall expansion protocol chart."""

    name: str
    path: str


def freeze_primeqa_hybrid_prefix_preserving_recall_expansion_protocol(
    *,
    stage124_report_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage125 train/dev-only prefix-preserving expansion protocol."""

    started_at = time.perf_counter()
    stage124_report = _load_json_object(stage124_report_path)
    loaded_at = time.perf_counter()
    stage124_summary = _stage124_summary(stage124_report)
    frozen_protocol = _frozen_protocol(stage124_summary)
    preliminary_report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "protocol_scope": (
            "Train/dev-only protocol freeze for Stage116 prefix-preserving "
            "recall expansion. This stage reads only the public-safe Stage124 "
            "validation report, freezes append-only candidate generation rules, "
            "does not load split files, does not load corpus documents, does "
            "not build candidate rows, does not run retrieval or final metrics, "
            "does not select from dev-only observations, does not add fallback "
            "strategies, and does not change runtime defaults."
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
            "selection_mode": (
                "train_grouped_cross_validation_prefix_preserving_candidate_selection"
            ),
            "validation_split": "dev",
            "dev_validation_mode": "single_pass_report_only_no_retuning",
            "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage124_report": _fingerprint(stage124_report_path),
        },
        "stage124_summary": stage124_summary,
        "frozen_protocol": frozen_protocol,
    }
    guard_checks = _guard_checks(
        report=preliminary_report,
        stage124_summary=stage124_summary,
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


def write_primeqa_hybrid_prefix_preserving_recall_expansion_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridPrefixPreservingRecallExpansionProtocolVisualization]:
    """Write SVG charts for Stage125 protocol freeze."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage125_stage124_blocked_signal_summary.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage125 Stage124 blocked signal summary",
                bars=_stage124_signal_bars(report),
                x_label="count",
                width=1460,
                margin_left=760,
            )
        ),
        "stage125_candidate_family_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage125 candidate family counts",
            bars=_candidate_family_count_bars(report),
            x_label="config count",
            width=1480,
            margin_left=780,
        ),
        "stage125_append_budgets.svg": render_horizontal_bar_chart_svg(
            title="Stage125 append budgets",
            bars=_append_budget_bars(report),
            x_label="appended candidates",
            width=1500,
            margin_left=800,
        ),
        "stage125_target_pool_depths.svg": render_horizontal_bar_chart_svg(
            title="Stage125 target pool depths",
            bars=_target_pool_depth_bars(report),
            x_label="candidate depth",
            width=1500,
            margin_left=800,
        ),
        "stage125_guard_thresholds.svg": render_horizontal_bar_chart_svg(
            title="Stage125 train-CV guard thresholds",
            bars=_guard_threshold_bars(report),
            x_label="maximum or minimum threshold",
            width=1580,
            margin_left=880,
        ),
        "stage125_protocol_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage125 protocol decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1380,
            margin_left=720,
        ),
        "stage125_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage125 guard checks",
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
            PrimeQAHybridPrefixPreservingRecallExpansionProtocolVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage124_summary(stage124_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage124_report.get("decision") or {}
    train_selection = stage124_report.get("train_selection") or {}
    public_safe = stage124_report.get("public_safe_contract") or {}
    baseline = stage124_report.get("baseline_by_split") or {}
    config_reviews = stage124_report.get("config_reviews") or []
    ranked_signals = _ranked_stage124_signals(config_reviews)
    return {
        "stage": stage124_report.get("stage"),
        "analysis_id": stage124_report.get("analysis_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "selected_config_id": decision.get("selected_config_id"),
        "selected_family_id": decision.get("selected_family_id"),
        "positive_target_depth_signal_blocked_by_hit_at_200_loss": decision.get(
            "positive_target_depth_signal_blocked_by_hit_at_200_loss"
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
        "candidate_count": train_selection.get("candidate_count"),
        "eligible_config_count": train_selection.get("eligible_config_count"),
        "train_baseline": _baseline_summary(baseline.get("train") or {}),
        "dev_baseline": _baseline_summary(baseline.get("dev") or {}),
        "top_blocked_signals": ranked_signals[:4],
        "best_train_target_depth_gain": (
            ranked_signals[0]["train_target_depth_gain"] if ranked_signals else None
        ),
        "minimum_positive_train_hit_at_200_loss_count": (
            min(
                row["train_hit_at_200_loss_count"]
                for row in ranked_signals
                if row["train_target_depth_gain"] > 0
            )
            if any(row["train_target_depth_gain"] > 0 for row in ranked_signals)
            else None
        ),
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
    }


def _baseline_summary(baseline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evaluated_questions": baseline.get("evaluated_questions"),
        "hit_at_200_count": baseline.get("hit_at_200_count"),
        "hit_at_200": baseline.get("hit_at_200"),
        "pool_depth": baseline.get("pool_depth"),
    }


def _ranked_stage124_signals(
    config_reviews: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for review in config_reviews:
        train = (review.get("split_reviews") or {}).get("train") or {}
        dev = (review.get("split_reviews") or {}).get("dev") or {}
        row = {
            "config_id": review.get("config_id"),
            "family_id": review.get("family_id"),
            "target_pool_depth": review.get("target_pool_depth"),
            "train_target_depth_gain": int(
                train.get("target_depth_hit_count_gain_vs_baseline_top200") or 0
            ),
            "train_hit_at_200_delta": int(
                train.get("hit_at_200_delta_vs_baseline") or 0
            ),
            "train_hit_at_200_loss_count": int(
                train.get("hit_at_200_loss_count") or 0
            ),
            "dev_target_depth_gain": int(
                dev.get("target_depth_hit_count_gain_vs_baseline_top200") or 0
            ),
            "dev_hit_at_200_delta": int(dev.get("hit_at_200_delta_vs_baseline") or 0),
            "dev_hit_at_200_loss_count": int(dev.get("hit_at_200_loss_count") or 0),
            "train_guard_passed": bool(
                (review.get("train_cv_guard") or {}).get("passed")
            ),
        }
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            -row["train_target_depth_gain"],
            row["train_hit_at_200_loss_count"],
            str(row["config_id"]),
        ),
    )


def _frozen_protocol(stage124_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "route_name": "stage116_prefix_preserving_append_only_recall_expansion",
        "source_validation": {
            "stage": stage124_summary.get("stage"),
            "status": stage124_summary.get("decision_status"),
            "analysis_id": stage124_summary.get("analysis_id"),
            "recommended_next_direction": stage124_summary.get(
                "recommended_next_direction"
            ),
        },
        "design_rationale": {
            "stage124_observation": (
                "Bounded 300/400-depth pools produced positive target-depth "
                "recall signal, but every positive train signal also introduced "
                "at least one Stage116 hit@200 loss."
            ),
            "new_hypothesis": (
                "Preserving the Stage116 top200 prefix exactly and appending "
                "only deduplicated new candidates after rank 200 can keep the "
                "hit@200 boundary intact while testing the extra recall signal."
            ),
            "not_a_runtime_policy": True,
        },
        "baseline_prefix_contract": {
            "baseline_config_id": _BASELINE_CONFIG_ID,
            "prefix_depth": _BASELINE_PREFIX_DEPTH,
            "train_baseline_hit_at_200_count": (
                stage124_summary.get("train_baseline") or {}
            ).get("hit_at_200_count"),
            "train_baseline_hit_at_200": (
                stage124_summary.get("train_baseline") or {}
            ).get("hit_at_200"),
            "dev_baseline_hit_at_200_count": (
                stage124_summary.get("dev_baseline") or {}
            ).get("hit_at_200_count"),
            "dev_baseline_hit_at_200": (
                stage124_summary.get("dev_baseline") or {}
            ).get("hit_at_200"),
            "ranks_1_to_200_must_remain_identical": True,
            "prefix_documents_may_be_reordered": False,
            "prefix_documents_may_be_dropped": False,
            "prefix_duplicate_in_append_region_allowed": False,
            "hit_at_200_loss_count_must_be_zero_by_construction": True,
        },
        "append_generation_contract": {
            "purpose": (
                "append new runtime-visible candidates after the Stage116 top200 "
                "prefix without perturbing ranks 1-200"
            ),
            "append_start_rank": _BASELINE_PREFIX_DEPTH + 1,
            "allowed_target_pool_depths": [300, 400],
            "allowed_append_budgets": [100, 200],
            "candidate_ordering": (
                "rank expansion candidates by the configured broad route, skip "
                "any document already present in the Stage116 prefix, then append "
                "until the target depth or candidate exhaustion"
            ),
            "allowed_runtime_signals": [
                "Stage116 top200 ranked prefix",
                "query text visible at runtime",
                "document title/body/section text visible at runtime",
                "BM25 and section BM25 ranks",
                "exact special-token matches",
                "locally cached dense-route ranks when already available",
            ],
            "model_download_allowed": False,
            "oracle_document_metadata_allowed": False,
            "test_membership_allowed": False,
            "training_labels_used_at_runtime": False,
            "raw_candidate_rows_written_in_stage125": False,
        },
        "candidate_families": _candidate_families(),
        "candidate_configs": _candidate_configs(),
        "selection_rules": _selection_rules(),
        "blocked_options": _blocked_options(),
        "next_stage": {
            "stage": "Stage126",
            "action": _NEXT_DIRECTION,
            "requires_user_confirmation": True,
            "test_locked": True,
            "runtime_defaults_unchanged": True,
            "fallback_strategies_enabled": False,
        },
    }


def _candidate_families() -> list[dict[str, Any]]:
    return [
        {
            "family_id": "stage116_prefix_rrf_append_family_v1",
            "description": (
                "Reuse the strongest Stage124 RRF signal but append only documents "
                "not already in the immutable Stage116 top200 prefix."
            ),
            "priority": 1,
        },
        {
            "family_id": "stage116_prefix_existing_dense_append_family_v1",
            "description": (
                "Use existing local dense-cache broad routes as append-only "
                "candidate sources after the immutable Stage116 top200 prefix."
            ),
            "priority": 2,
        },
        {
            "family_id": "stage116_prefix_query_variant_append_family_v1",
            "description": (
                "Use deterministic title and special-token query variants only as "
                "append sources after the immutable prefix."
            ),
            "priority": 3,
        },
        {
            "family_id": "stage116_prefix_route_balanced_append_family_v1",
            "description": (
                "Test route-balanced append sources without allowing route-balanced "
                "interleaving to disturb ranks 1-200."
            ),
            "priority": 4,
        },
    ]


def _candidate_configs() -> list[dict[str, Any]]:
    return [
        _config(
            config_id="prefix_rrf_same_routes_append100_k60_v1",
            family_id="stage116_prefix_rrf_append_family_v1",
            source_stage124_config_id="rrf_same_routes_top300_k60_v1",
            append_source_algorithm="weighted_rrf",
            route_set="stage116_same_routes",
            channel_top_k=300,
            append_budget=100,
            target_pool_depth=300,
            rrf_k=60,
            priority=1,
        ),
        _config(
            config_id="prefix_rrf_same_routes_append200_k60_v1",
            family_id="stage116_prefix_rrf_append_family_v1",
            source_stage124_config_id="rrf_same_routes_top400_k60_v1",
            append_source_algorithm="weighted_rrf",
            route_set="stage116_same_routes",
            channel_top_k=400,
            append_budget=200,
            target_pool_depth=400,
            rrf_k=60,
            priority=2,
        ),
        _config(
            config_id="prefix_existing_dense_broad_append200_v1",
            family_id="stage116_prefix_existing_dense_append_family_v1",
            source_stage124_config_id="existing_dense_cache_broad_union_top400_v1",
            append_source_algorithm="cached_dense_plus_lexical_rrf",
            route_set="stage116_lexical_routes_plus_existing_dense_cache_routes",
            channel_top_k=400,
            append_budget=200,
            target_pool_depth=400,
            rrf_k=60,
            priority=3,
        ),
        _config(
            config_id="prefix_rrf_same_routes_append100_k80_v1",
            family_id="stage116_prefix_rrf_append_family_v1",
            source_stage124_config_id="rrf_lexical_priority_top300_k80_v1",
            append_source_algorithm="weighted_rrf",
            route_set="stage116_lexical_routes_plus_cached_dense",
            channel_top_k=300,
            append_budget=100,
            target_pool_depth=300,
            rrf_k=80,
            priority=4,
        ),
        _config(
            config_id="prefix_query_variant_append100_v1",
            family_id="stage116_prefix_query_variant_append_family_v1",
            source_stage124_config_id="query_variant_title_special_token_top300_v1",
            append_source_algorithm="deterministic_query_variant_union",
            route_set="lexical_routes_with_title_and_special_token_variants",
            channel_top_k=250,
            append_budget=100,
            target_pool_depth=300,
            rrf_k=60,
            priority=5,
        ),
        _config(
            config_id="prefix_route_balanced_append200_v1",
            family_id="stage116_prefix_route_balanced_append_family_v1",
            source_stage124_config_id="route_balanced_round_robin_top400_v1",
            append_source_algorithm="route_balanced_interleaving",
            route_set="stage116_same_routes",
            channel_top_k=400,
            append_budget=200,
            target_pool_depth=400,
            rrf_k=60,
            priority=6,
        ),
    ]


def _config(
    *,
    config_id: str,
    family_id: str,
    source_stage124_config_id: str,
    append_source_algorithm: str,
    route_set: str,
    channel_top_k: int,
    append_budget: int,
    target_pool_depth: int,
    rrf_k: int,
    priority: int,
) -> dict[str, Any]:
    return {
        "config_id": config_id,
        "family_id": family_id,
        "selection_eligible": True,
        "source_stage124_config_id": source_stage124_config_id,
        "append_generation": {
            "append_source_algorithm": append_source_algorithm,
            "route_set": route_set,
            "channel_top_k": channel_top_k,
            "rrf_k": rrf_k,
            "append_start_rank": _BASELINE_PREFIX_DEPTH + 1,
            "append_budget": append_budget,
            "target_pool_depth": target_pool_depth,
            "priority": priority,
            "deduplicate_against_prefix": True,
            "deduplicate_within_append_region": True,
        },
        "prefix_preservation": {
            "source_prefix_config_id": _BASELINE_CONFIG_ID,
            "preserved_prefix_depth": _BASELINE_PREFIX_DEPTH,
            "ranks_1_to_200_must_remain_identical": True,
            "may_reorder_prefix": False,
            "may_drop_prefix_documents": False,
            "may_insert_before_rank_201": False,
        },
        "feature_sources": {
            "uses_runtime_query_text": True,
            "uses_runtime_corpus_text": True,
            "uses_existing_dense_cache": "dense" in family_id
            or "same_routes" in route_set
            or "cached_dense" in route_set,
            "requires_new_embedding_build": False,
            "requires_model_download": False,
            "uses_oracle_document_metadata": False,
            "uses_test_membership": False,
        },
        "safety_constraints": {
            "maximum_channel_top_k": _MAX_CHANNEL_TOP_K,
            "maximum_output_pool_depth": _MAX_TARGET_POOL_DEPTH,
            "maximum_append_budget": _MAX_TARGET_POOL_DEPTH - _BASELINE_PREFIX_DEPTH,
            "training_labels_used_at_runtime": False,
            "runtime_defaultization_allowed": False,
            "fallback_strategies_enabled": False,
        },
    }


def _selection_rules() -> dict[str, Any]:
    return {
        "selection_split": "train",
        "selection_mode": (
            "train_grouped_cross_validation_prefix_preserving_candidate_selection"
        ),
        "minimum_train_folds": _MINIMUM_TRAIN_FOLDS,
        "baseline_order": "stage116_fixed_rrf_pool_order",
        "primary_metrics": [
            "prefix_identity_violation_count",
            "hit_at_200_delta_vs_stage116_prefix",
            "target_depth_hit_count_gain_vs_stage116_top200",
            "appended_gold_recovery_count",
            "train_fold_stability_at_target_depth",
        ],
        "selection_objective": (
            "maximize train-CV target-depth hit-count gain after all prefix "
            "identity and hit@200 no-loss guards pass; break ties by lower "
            "target depth, then lower append budget"
        ),
        "guard_thresholds": {
            "maximum_train_cv_prefix_identity_violation_count": 0,
            "maximum_train_cv_hit_at_200_loss_count": 0,
            "minimum_train_cv_hit_at_200_delta": 0,
            "minimum_train_cv_target_depth_hit_count_gain": 1,
            "maximum_channel_top_k": _MAX_CHANNEL_TOP_K,
            "maximum_output_pool_depth": _MAX_TARGET_POOL_DEPTH,
            "maximum_append_budget": _MAX_TARGET_POOL_DEPTH - _BASELINE_PREFIX_DEPTH,
            "maximum_model_download_attempts": 0,
            "maximum_raw_candidate_rows_written": 0,
            "minimum_train_fold_count": _MINIMUM_TRAIN_FOLDS,
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
            "runtime_defaultization_allowed_in_stage125": False,
        },
    }


def _blocked_options() -> list[dict[str, str]]:
    return [
        {
            "option_id": "rerank_entire_300_or_400_pool_blocked",
            "reason": (
                "Stage124 showed whole-pool 300/400 reranking can introduce "
                "Stage116 hit@200 losses."
            ),
        },
        {
            "option_id": "drop_or_reorder_stage116_top200_blocked",
            "reason": "The Stage116 top200 prefix must remain exactly unchanged.",
        },
        {
            "option_id": "uncapped_union_default_runtime_blocked",
            "reason": "Uncapped unions remain too large for default runtime use.",
        },
        {
            "option_id": "new_dense_model_download_blocked",
            "reason": "Only existing local dense caches may be used.",
        },
        {
            "option_id": "dev_selected_threshold_blocked",
            "reason": "Dev remains validation/report-only, not a selection source.",
        },
        {
            "option_id": "final_test_metrics_blocked",
            "reason": "Test remains locked until an explicit final-test gate.",
        },
    ]


def _guard_checks(
    *,
    report: Mapping[str, Any],
    stage124_summary: Mapping[str, Any],
    user_confirmed_protocol: bool,
) -> list[dict[str, Any]]:
    frozen = report["frozen_protocol"]
    candidate_configs = frozen["candidate_configs"]
    selection_rules = frozen["selection_rules"]
    public_safe = _public_safe_contract(report)
    return [
        _check(
            name="user_confirmed_stage125_protocol",
            passed=user_confirmed_protocol,
            observed=report["user_confirmation"]["confirmation_note"],
            expected="user confirmed Stage125 prefix-preserving protocol",
        ),
        _check(
            name="stage124_validation_completed_no_selection",
            passed=stage124_summary.get("decision_status") == _SOURCE_STAGE124_STATUS,
            observed=stage124_summary.get("decision_status"),
            expected=_SOURCE_STAGE124_STATUS,
        ),
        _check(
            name="stage124_recommends_prefix_preserving_protocol",
            passed=stage124_summary.get("recommended_next_direction")
            == _SOURCE_NEXT_DIRECTION,
            observed=stage124_summary.get("recommended_next_direction"),
            expected=_SOURCE_NEXT_DIRECTION,
        ),
        _check(
            name="stage124_positive_signal_blocked_by_hit200_loss",
            passed=stage124_summary.get(
                "positive_target_depth_signal_blocked_by_hit_at_200_loss"
            )
            is True
            and int(stage124_summary.get("best_train_target_depth_gain") or 0) > 0
            and int(
                stage124_summary.get(
                    "minimum_positive_train_hit_at_200_loss_count"
                )
                or 0
            )
            > 0,
            observed={
                "positive_signal": stage124_summary.get(
                    "positive_target_depth_signal_blocked_by_hit_at_200_loss"
                ),
                "best_train_target_depth_gain": stage124_summary.get(
                    "best_train_target_depth_gain"
                ),
                "minimum_positive_train_hit_at_200_loss_count": stage124_summary.get(
                    "minimum_positive_train_hit_at_200_loss_count"
                ),
            },
            expected="positive target-depth signal with at least one hit@200 loss",
        ),
        _check(
            name="stage124_no_config_selected",
            passed=stage124_summary.get("selected_config_id") is None
            and int(stage124_summary.get("eligible_config_count") or 0) == 0,
            observed={
                "selected_config_id": stage124_summary.get("selected_config_id"),
                "eligible_config_count": stage124_summary.get("eligible_config_count"),
            },
            expected="no selected config and zero eligible configs",
        ),
        _check(
            name="stage124_baseline_reproduced",
            passed=(stage124_summary.get("train_baseline") or {}).get(
                "hit_at_200_count"
            )
            == 345
            and (stage124_summary.get("dev_baseline") or {}).get("hit_at_200_count")
            == 69,
            observed={
                "train": stage124_summary.get("train_baseline"),
                "dev": stage124_summary.get("dev_baseline"),
            },
            expected={"train_hit_at_200_count": 345, "dev_hit_at_200_count": 69},
        ),
        _check(
            name="stage124_runtime_and_test_boundaries_locked",
            passed=stage124_summary.get("can_open_final_test_gate_now") is False
            and stage124_summary.get("can_run_final_test_metrics_now") is False
            and stage124_summary.get("can_use_test_for_tuning") is False
            and stage124_summary.get("fallback_strategies_enabled") is False
            and stage124_summary.get("default_runtime_policy") == "unchanged",
            observed=stage124_summary,
            expected="test locked, runtime unchanged, fallback disabled",
        ),
        _check(
            name="stage125_candidate_configs_present",
            passed=len(candidate_configs) == 6
            and set(_family_counts(candidate_configs)) == set(_CANDIDATE_FAMILY_IDS),
            observed={
                "candidate_config_count": len(candidate_configs),
                "families": sorted(_family_counts(candidate_configs)),
            },
            expected={"candidate_config_count": 6, "families": list(_CANDIDATE_FAMILY_IDS)},
        ),
        _check(
            name="stage125_configs_preserve_stage116_prefix",
            passed=all(
                config["prefix_preservation"]["ranks_1_to_200_must_remain_identical"]
                is True
                and config["prefix_preservation"]["may_reorder_prefix"] is False
                and config["prefix_preservation"]["may_drop_prefix_documents"] is False
                and config["prefix_preservation"]["may_insert_before_rank_201"] is False
                for config in candidate_configs
            ),
            observed=[
                {
                    "config_id": config["config_id"],
                    **config["prefix_preservation"],
                }
                for config in candidate_configs
            ],
            expected="immutable Stage116 top200 prefix",
        ),
        _check(
            name="stage125_configs_append_only_after_rank200",
            passed=all(
                int(config["append_generation"]["append_start_rank"])
                == _BASELINE_PREFIX_DEPTH + 1
                and int(config["append_generation"]["target_pool_depth"])
                > _BASELINE_PREFIX_DEPTH
                and int(config["append_generation"]["target_pool_depth"])
                <= _MAX_TARGET_POOL_DEPTH
                and int(config["append_generation"]["append_budget"])
                == int(config["append_generation"]["target_pool_depth"])
                - _BASELINE_PREFIX_DEPTH
                for config in candidate_configs
            ),
            observed=[
                {
                    "config_id": config["config_id"],
                    "append_start_rank": config["append_generation"][
                        "append_start_rank"
                    ],
                    "append_budget": config["append_generation"]["append_budget"],
                    "target_pool_depth": config["append_generation"][
                        "target_pool_depth"
                    ],
                }
                for config in candidate_configs
            ],
            expected="append starts at 201 and target depth <= 400",
        ),
        _check(
            name="stage125_configs_do_not_download_models",
            passed=all(
                config["feature_sources"]["requires_model_download"] is False
                and config["feature_sources"]["requires_new_embedding_build"] is False
                for config in candidate_configs
            ),
            observed=[
                {
                    "config_id": config["config_id"],
                    "requires_model_download": config["feature_sources"][
                        "requires_model_download"
                    ],
                    "requires_new_embedding_build": config["feature_sources"][
                        "requires_new_embedding_build"
                    ],
                }
                for config in candidate_configs
            ],
            expected="no model download and no new embedding build",
        ),
        _check(
            name="stage125_configs_do_not_use_oracle_features",
            passed=all(
                config["feature_sources"]["uses_oracle_document_metadata"] is False
                and config["feature_sources"]["uses_test_membership"] is False
                and config["safety_constraints"]["training_labels_used_at_runtime"]
                is False
                for config in candidate_configs
            ),
            observed=[
                {
                    "config_id": config["config_id"],
                    "uses_oracle_document_metadata": config["feature_sources"][
                        "uses_oracle_document_metadata"
                    ],
                    "uses_test_membership": config["feature_sources"][
                        "uses_test_membership"
                    ],
                    "training_labels_used_at_runtime": config["safety_constraints"][
                        "training_labels_used_at_runtime"
                    ],
                }
                for config in candidate_configs
            ],
            expected="runtime-visible retrieval features only",
        ),
        _check(
            name="stage125_uses_train_grouped_cv",
            passed=selection_rules["selection_mode"]
            == "train_grouped_cross_validation_prefix_preserving_candidate_selection"
            and int(selection_rules["minimum_train_folds"]) >= _MINIMUM_TRAIN_FOLDS,
            observed={
                "selection_mode": selection_rules["selection_mode"],
                "minimum_train_folds": selection_rules["minimum_train_folds"],
            },
            expected="train grouped-CV with at least 5 folds",
        ),
        _check(
            name="stage125_dev_is_report_only",
            passed=selection_rules["dev_rules"]["dev_selection_allowed"] is False
            and selection_rules["dev_rules"]["dev_retuning_allowed"] is False
            and selection_rules["dev_rules"]["dev_threshold_tuning_allowed"] is False,
            observed=selection_rules["dev_rules"],
            expected="dev report-only",
        ),
        _check(
            name="stage125_test_locked",
            passed=selection_rules["test_rules"]["test_access_allowed"] is False
            and selection_rules["test_rules"]["final_test_metrics_allowed"] is False
            and selection_rules["test_rules"]["test_tuning_allowed"] is False,
            observed=selection_rules["test_rules"],
            expected="test access and final metrics disabled",
        ),
        _check(
            name="stage125_runtime_defaults_unchanged",
            passed=selection_rules["runtime_rules"]["default_runtime_policy"]
            == "unchanged"
            and selection_rules["runtime_rules"]["fallback_strategies_enabled"] is False
            and selection_rules["runtime_rules"][
                "runtime_defaultization_allowed_in_stage125"
            ]
            is False,
            observed=selection_rules["runtime_rules"],
            expected="runtime unchanged and no fallback strategies",
        ),
        _check(
            name="stage125_no_candidate_rows_or_metrics_run",
            passed=frozen["append_generation_contract"][
                "raw_candidate_rows_written_in_stage125"
            ]
            is False,
            observed=frozen["append_generation_contract"][
                "raw_candidate_rows_written_in_stage125"
            ],
            expected=False,
        ),
        _check(
            name="stage125_public_safe_contract_passed",
            passed=public_safe["forbidden_keys_found"] == [],
            observed=public_safe["forbidden_keys_found"],
            expected=[],
        ),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed:
        return {
            "status": (
                "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_protocol_blocked"
            ),
            "failed_checks": failed,
            "can_run_prefix_preserving_recall_expansion_now": False,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": (
            "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_protocol_frozen"
        ),
        "recommended_next_direction": _NEXT_DIRECTION,
        "can_run_prefix_preserving_recall_expansion_now": True,
        "can_continue_train_dev_development": True,
        "requires_user_confirmation_before_train_dev_run": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }


def _stage124_signal_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = report["stage124_summary"]
    rows = list(summary.get("top_blocked_signals") or [])[:3]
    bars = [
        BarDatum(
            label="best train target-depth gain",
            value=float(summary.get("best_train_target_depth_gain") or 0),
            value_label=str(summary.get("best_train_target_depth_gain") or 0),
        ),
        BarDatum(
            label="min positive train hit200 losses",
            value=float(
                summary.get("minimum_positive_train_hit_at_200_loss_count") or 0
            ),
            value_label=str(
                summary.get("minimum_positive_train_hit_at_200_loss_count") or 0
            ),
        ),
    ]
    for row in rows:
        bars.append(
            BarDatum(
                label=f"{row['config_id']} train gain",
                value=float(row["train_target_depth_gain"]),
                value_label=str(row["train_target_depth_gain"]),
            )
        )
        bars.append(
            BarDatum(
                label=f"{row['config_id']} train hit200 losses",
                value=float(row["train_hit_at_200_loss_count"]),
                value_label=str(row["train_hit_at_200_loss_count"]),
            )
        )
    return bars


def _candidate_family_count_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counts = _family_counts(report["frozen_protocol"]["candidate_configs"])
    return [
        BarDatum(label=family_id, value=float(count), value_label=str(count))
        for family_id, count in sorted(counts.items())
    ]


def _append_budget_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(config["config_id"]),
            value=float(config["append_generation"]["append_budget"]),
            value_label=str(config["append_generation"]["append_budget"]),
        )
        for config in report["frozen_protocol"]["candidate_configs"]
    ]


def _target_pool_depth_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(config["config_id"]),
            value=float(config["append_generation"]["target_pool_depth"]),
            value_label=str(config["append_generation"]["target_pool_depth"]),
        )
        for config in report["frozen_protocol"]["candidate_configs"]
    ]


def _guard_threshold_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    thresholds = report["frozen_protocol"]["selection_rules"]["guard_thresholds"]
    return [
        BarDatum(label=str(key), value=float(value), value_label=str(value))
        for key, value in thresholds.items()
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    flags = (
        "can_run_prefix_preserving_recall_expansion_now",
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
    return dict(
        sorted(Counter(str(config["family_id"]) for config in candidate_configs).items())
    )


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
        "exists": path.exists(),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
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

