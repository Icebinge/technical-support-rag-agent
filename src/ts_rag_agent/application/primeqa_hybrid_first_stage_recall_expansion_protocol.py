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

_STAGE = "Stage 123"
_CREATED_AT = "2026-07-16"
_SOURCE_STAGE = "Stage 122"
_SOURCE_STAGE122_STATUS = (
    "primeqa_hybrid_fast_filter_screening_changed_case_review_completed"
)
_SOURCE_NEXT_DIRECTION = "design_first_stage_recall_expansion_protocol"
_PROTOCOL_ID = "primeqa_hybrid_first_stage_recall_expansion_protocol_v1"
_NEXT_DIRECTION = "run_first_stage_recall_expansion_train_cv_dev_validation"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = ("test",)
_BASELINE_POOL_ID = "stage116_multi_route_union_candidate_pool"
_BASELINE_POOL_DEPTH = 200
_MAX_CHANNEL_TOP_K = 400
_MAX_OUTPUT_POOL_DEPTH = 400
_MINIMUM_TRAIN_FOLDS = 5
_CANDIDATE_FAMILY_IDS = (
    "rrf_depth_expansion_family_v1",
    "route_balanced_union_family_v1",
    "query_variant_lexical_family_v1",
    "existing_dense_cache_union_family_v1",
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
class PrimeQAHybridFirstStageRecallExpansionProtocolVisualization:
    """One generated Stage123 first-stage recall expansion protocol chart."""

    name: str
    path: str


def freeze_primeqa_hybrid_first_stage_recall_expansion_protocol(
    *,
    stage122_report_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage123 train/dev-only first-stage recall expansion protocol."""

    started_at = time.perf_counter()
    stage122_report = _load_json_object(stage122_report_path)
    loaded_at = time.perf_counter()
    stage122_summary = _stage122_summary(stage122_report)
    frozen_protocol = _frozen_protocol(stage122_summary)
    preliminary_report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "protocol_scope": (
            "Train/dev-only protocol freeze for a broader first-stage recall "
            "expansion route after Stage122 showed second-stage screening has "
            "real but guard-risky hit@20 signal. This stage reads only the "
            "public-safe Stage122 report, freezes candidate generation families "
            "and validation rules, does not load split files, does not load "
            "corpus documents, does not build candidate rows, does not run "
            "retrieval or final metrics, does not select from dev-only "
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
            "selection_mode": "train_grouped_cross_validation_candidate_pool_selection",
            "validation_split": "dev",
            "dev_validation_mode": "single_pass_report_only_no_retuning",
            "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage122_report": _fingerprint(stage122_report_path),
        },
        "stage122_summary": stage122_summary,
        "frozen_protocol": frozen_protocol,
    }
    guard_checks = _guard_checks(
        report=preliminary_report,
        stage122_summary=stage122_summary,
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


def write_primeqa_hybrid_first_stage_recall_expansion_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridFirstStageRecallExpansionProtocolVisualization]:
    """Write SVG charts for Stage123 protocol freeze."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage123_stage122_signal_summary.svg": render_horizontal_bar_chart_svg(
            title="Stage123 Stage122 signal summary",
            bars=_stage122_signal_bars(report),
            x_label="changed-case count",
            width=1440,
            margin_left=760,
        ),
        "stage123_candidate_family_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage123 candidate family counts",
            bars=_candidate_family_count_bars(report),
            x_label="config count",
            width=1480,
            margin_left=780,
        ),
        "stage123_target_pool_depths.svg": render_horizontal_bar_chart_svg(
            title="Stage123 target pool depths",
            bars=_target_pool_depth_bars(report),
            x_label="candidate depth",
            width=1480,
            margin_left=780,
        ),
        "stage123_channel_top_k_budgets.svg": render_horizontal_bar_chart_svg(
            title="Stage123 channel top-k budgets",
            bars=_channel_top_k_bars(report),
            x_label="per-channel top-k",
            width=1480,
            margin_left=780,
        ),
        "stage123_guard_thresholds.svg": render_horizontal_bar_chart_svg(
            title="Stage123 train-CV guard thresholds",
            bars=_guard_threshold_bars(report),
            x_label="maximum or minimum threshold",
            width=1560,
            margin_left=860,
        ),
        "stage123_protocol_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage123 protocol decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1380,
            margin_left=720,
        ),
        "stage123_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage123 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1800,
            margin_left=980,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridFirstStageRecallExpansionProtocolVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage122_summary(stage122_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage122_report.get("decision") or {}
    cross = stage122_report.get("cross_config_findings") or {}
    public_safe = stage122_report.get("public_safe_contract") or {}
    config_reviews = _config_reviews_by_id(stage122_report.get("config_reviews") or [])
    selected_review = config_reviews.get("special_token_exact_window40_rule_selector_v1")
    blocked_review = config_reviews.get(
        "top10_locked_route_vote_window50_pairwise_logistic_v1"
    )
    return {
        "stage": stage122_report.get("stage"),
        "analysis_id": stage122_report.get("analysis_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
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
        "runtime_defaultization_supported": decision.get(
            "runtime_defaultization_supported"
        ),
        "blocked_signal_has_real_hit20_recoveries": cross.get(
            "blocked_signal_has_real_hit20_recoveries"
        ),
        "blocked_signal_has_guard_relevant_regressions": cross.get(
            "blocked_signal_has_guard_relevant_regressions"
        ),
        "selected_config_is_low_change": cross.get("selected_config_is_low_change"),
        "selected_config_review": _public_config_review_summary(selected_review),
        "blocked_signal_config_review": _public_config_review_summary(blocked_review),
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
        "raw_candidate_rows_written": public_safe.get("raw_candidate_rows_written"),
    }


def _config_reviews_by_id(
    config_reviews: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(review.get("config_id")): review
        for review in config_reviews
        if isinstance(review, Mapping)
    }


def _public_config_review_summary(review: Mapping[str, Any] | None) -> dict[str, Any]:
    if not review:
        return {}
    interpretation = review.get("interpretation") or {}
    split_reviews = review.get("split_reviews") or {}
    train_cv = review.get("train_cv") or split_reviews.get("train_cv") or {}
    dev = review.get("dev") or split_reviews.get("dev") or {}
    return {
        "config_id": review.get("config_id"),
        "status": interpretation.get("status"),
        "runtime_defaultization_supported": interpretation.get(
            "runtime_defaultization_supported"
        ),
        "train_changed_case_count": train_cv.get("changed_case_count"),
        "train_improved_count": train_cv.get("improved_count"),
        "train_regressed_count": train_cv.get("regressed_count"),
        "train_hit20_recovery_count": train_cv.get("hit20_recovery_count"),
        "train_hit20_regression_count": train_cv.get("hit20_regression_count"),
        "dev_changed_case_count": dev.get("changed_case_count"),
        "dev_improved_count": dev.get("improved_count"),
        "dev_regressed_count": dev.get("regressed_count"),
        "dev_hit20_recovery_count": dev.get("hit20_recovery_count"),
        "dev_hit20_regression_count": dev.get("hit20_regression_count"),
    }


def _frozen_protocol(stage122_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "route_name": "first_stage_recall_expansion_before_precision_screening",
        "source_changed_case_review": {
            "stage": stage122_summary.get("stage"),
            "status": stage122_summary.get("decision_status"),
            "analysis_id": stage122_summary.get("analysis_id"),
            "recommended_next_direction": stage122_summary.get(
                "recommended_next_direction"
            ),
        },
        "design_rationale": {
            "problem_to_solve": (
                "Stage116 top200 candidate-pool recall caps downstream recall "
                "at 0.9324 on train and 0.9079 on dev."
            ),
            "stage122_observation": (
                "Second-stage screening has real hit@20 recovery signal but also "
                "guard-relevant hit@20 regressions."
            ),
            "new_hypothesis": (
                "A broader but still simple first-stage union can recover more "
                "gold documents before precision screening, while preserving the "
                "Stage116 top200 boundary as a guard."
            ),
            "not_a_runtime_policy": True,
        },
        "baseline_candidate_pool_contract": {
            "baseline_pool_id": _BASELINE_POOL_ID,
            "baseline_pool_depth": _BASELINE_POOL_DEPTH,
            "baseline_train_hit_at_200": 0.9324,
            "baseline_dev_hit_at_200": 0.9079,
            "baseline_uncapped_train_hit": 0.9676,
            "baseline_uncapped_dev_hit": 0.9474,
            "baseline_uncapped_pool_too_large_for_default_runtime": True,
        },
        "candidate_generation_contract": {
            "purpose": (
                "increase first-stage candidate-pool recall before any learned "
                "or rule-based precision selector"
            ),
            "allowed_runtime_signals": [
                "query text visible at runtime",
                "document title and body text visible at runtime",
                "document section text visible at runtime",
                "BM25 and section BM25 ranks",
                "exact special-token matches",
                "locally cached dense-route ranks when already available",
            ],
            "model_download_allowed": False,
            "oracle_document_metadata_allowed": False,
            "test_membership_allowed": False,
            "training_labels_used_at_runtime": False,
            "raw_candidate_rows_written_in_stage123": False,
        },
        "candidate_families": _candidate_families(),
        "candidate_configs": _candidate_configs(),
        "selection_rules": _selection_rules(),
        "blocked_options": _blocked_options(),
        "next_stage": {
            "stage": "Stage124",
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
            "family_id": "rrf_depth_expansion_family_v1",
            "description": (
                "Keep the Stage116 route set and expand per-route depth plus final "
                "pool depth with fixed reciprocal-rank fusion."
            ),
            "priority": 1,
        },
        {
            "family_id": "route_balanced_union_family_v1",
            "description": (
                "Balance route contribution so a single strong route does not "
                "consume the broader pool before weaker routes add tail evidence."
            ),
            "priority": 2,
        },
        {
            "family_id": "query_variant_lexical_family_v1",
            "description": (
                "Generate deterministic lexical query variants from runtime-visible "
                "question text and exact technical tokens."
            ),
            "priority": 3,
        },
        {
            "family_id": "existing_dense_cache_union_family_v1",
            "description": (
                "Use only existing local dense caches as additional broad routes; "
                "no model download or new embedding build is allowed."
            ),
            "priority": 4,
        },
    ]


def _candidate_configs() -> list[dict[str, Any]]:
    return [
        _config(
            config_id="rrf_same_routes_top300_k60_v1",
            family_id="rrf_depth_expansion_family_v1",
            algorithm="weighted_rrf",
            route_set="stage116_same_routes",
            channel_top_k=300,
            target_pool_depth=300,
            rrf_k=60,
            ranking_policy="stage116_weights_preserved",
            includes_query_variants=False,
            includes_existing_dense_cache=True,
        ),
        _config(
            config_id="rrf_same_routes_top400_k60_v1",
            family_id="rrf_depth_expansion_family_v1",
            algorithm="weighted_rrf",
            route_set="stage116_same_routes",
            channel_top_k=400,
            target_pool_depth=400,
            rrf_k=60,
            ranking_policy="stage116_weights_preserved",
            includes_query_variants=False,
            includes_existing_dense_cache=True,
        ),
        _config(
            config_id="rrf_lexical_priority_top300_k80_v1",
            family_id="rrf_depth_expansion_family_v1",
            algorithm="weighted_rrf",
            route_set="stage116_lexical_routes_plus_cached_dense",
            channel_top_k=300,
            target_pool_depth=300,
            rrf_k=80,
            ranking_policy="lexical_routes_weighted_before_cached_dense_routes",
            includes_query_variants=False,
            includes_existing_dense_cache=True,
        ),
        _config(
            config_id="route_balanced_round_robin_top300_v1",
            family_id="route_balanced_union_family_v1",
            algorithm="route_balanced_interleaving",
            route_set="stage116_same_routes",
            channel_top_k=300,
            target_pool_depth=300,
            rrf_k=60,
            ranking_policy="deduplicate_after_round_robin_route_slots",
            includes_query_variants=False,
            includes_existing_dense_cache=True,
        ),
        _config(
            config_id="route_balanced_round_robin_top400_v1",
            family_id="route_balanced_union_family_v1",
            algorithm="route_balanced_interleaving",
            route_set="stage116_same_routes",
            channel_top_k=400,
            target_pool_depth=400,
            rrf_k=60,
            ranking_policy="deduplicate_after_round_robin_route_slots",
            includes_query_variants=False,
            includes_existing_dense_cache=True,
        ),
        _config(
            config_id="query_variant_title_special_token_top300_v1",
            family_id="query_variant_lexical_family_v1",
            algorithm="deterministic_query_variant_union",
            route_set="lexical_routes_with_title_and_special_token_variants",
            channel_top_k=250,
            target_pool_depth=300,
            rrf_k=60,
            ranking_policy="original_query_routes_before_variant_routes",
            includes_query_variants=True,
            includes_existing_dense_cache=False,
        ),
        _config(
            config_id="existing_dense_cache_broad_union_top400_v1",
            family_id="existing_dense_cache_union_family_v1",
            algorithm="cached_dense_plus_lexical_rrf",
            route_set="stage116_lexical_routes_plus_existing_dense_cache_routes",
            channel_top_k=400,
            target_pool_depth=400,
            rrf_k=60,
            ranking_policy="dense_and_lexical_routes_joint_rrf",
            includes_query_variants=False,
            includes_existing_dense_cache=True,
        ),
    ]


def _config(
    *,
    config_id: str,
    family_id: str,
    algorithm: str,
    route_set: str,
    channel_top_k: int,
    target_pool_depth: int,
    rrf_k: int,
    ranking_policy: str,
    includes_query_variants: bool,
    includes_existing_dense_cache: bool,
) -> dict[str, Any]:
    return {
        "config_id": config_id,
        "family_id": family_id,
        "selection_eligible": True,
        "candidate_generation": {
            "algorithm": algorithm,
            "route_set": route_set,
            "channel_top_k": channel_top_k,
            "target_pool_depth": target_pool_depth,
            "rrf_k": rrf_k,
            "ranking_policy": ranking_policy,
            "deduplication_unit": "runtime document identity",
            "may_expand_beyond_stage116_top200": target_pool_depth > _BASELINE_POOL_DEPTH,
            "maximum_output_candidates": target_pool_depth,
        },
        "feature_sources": {
            "uses_runtime_query_text": True,
            "uses_runtime_corpus_text": True,
            "uses_query_variants": includes_query_variants,
            "uses_existing_dense_cache": includes_existing_dense_cache,
            "requires_new_embedding_build": False,
            "requires_model_download": False,
            "uses_oracle_document_metadata": False,
            "uses_test_membership": False,
        },
        "safety_constraints": {
            "maximum_channel_top_k": _MAX_CHANNEL_TOP_K,
            "maximum_output_pool_depth": _MAX_OUTPUT_POOL_DEPTH,
            "preserve_stage116_top200_as_guard": True,
            "training_labels_used_at_runtime": False,
            "runtime_defaultization_allowed": False,
            "fallback_strategies_enabled": False,
        },
    }


def _selection_rules() -> dict[str, Any]:
    return {
        "selection_split": "train",
        "selection_mode": "train_grouped_cross_validation_candidate_pool_selection",
        "minimum_train_folds": _MINIMUM_TRAIN_FOLDS,
        "baseline_order": "stage116_fixed_rrf_pool_order",
        "primary_metrics": [
            "hit_at_200_delta_vs_stage116_order",
            "target_depth_hit_count_gain_vs_stage116_top200",
            "target_depth_missing_count_reduction_vs_stage116_top200",
            "train_fold_stability_at_target_depth",
        ],
        "selection_objective": (
            "maximize train-CV target-depth hit-count gain, break ties by zero "
            "hit@200 loss, then lower output depth"
        ),
        "guard_thresholds": {
            "maximum_train_cv_hit_at_200_loss_count": 0,
            "minimum_train_cv_target_depth_hit_count_gain": 1,
            "maximum_channel_top_k": _MAX_CHANNEL_TOP_K,
            "maximum_output_pool_depth": _MAX_OUTPUT_POOL_DEPTH,
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
            "runtime_defaultization_allowed_in_stage123": False,
        },
    }


def _blocked_options() -> list[dict[str, str]]:
    return [
        {
            "option_id": "uncapped_union_default_runtime_blocked",
            "reason": (
                "Stage116 uncapped union improves recall but has an average size "
                "above 600 documents, so it is not a default runtime candidate pool."
            ),
        },
        {
            "option_id": "new_dense_model_download_blocked",
            "reason": (
                "Stage123 protocol permits only existing local dense caches; no "
                "new model download or embedding build is allowed."
            ),
        },
        {
            "option_id": "oracle_document_metadata_route_blocked",
            "reason": (
                "Dataset-only oracle metadata must not become a runtime retrieval "
                "feature."
            ),
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
            "option_id": "runtime_defaultization_in_stage123_blocked",
            "reason": "Stage123 freezes a protocol only; it does not change defaults.",
        },
    ]


def _guard_checks(
    *,
    report: Mapping[str, Any],
    stage122_summary: Mapping[str, Any],
    user_confirmed_protocol: bool,
) -> list[dict[str, Any]]:
    frozen = report["frozen_protocol"]
    selection_rules = frozen["selection_rules"]
    candidate_configs = frozen["candidate_configs"]
    public_safe = _public_safe_contract(report)
    return [
        _check(
            name="user_confirmed_stage123_protocol",
            passed=user_confirmed_protocol,
            observed=report["user_confirmation"]["confirmation_note"],
            expected="user confirmed Stage123 first-stage recall expansion protocol",
        ),
        _check(
            name="stage122_changed_case_review_completed",
            passed=stage122_summary.get("decision_status") == _SOURCE_STAGE122_STATUS,
            observed=stage122_summary.get("decision_status"),
            expected=_SOURCE_STAGE122_STATUS,
        ),
        _check(
            name="stage122_recommends_first_stage_recall_expansion",
            passed=stage122_summary.get("recommended_next_direction")
            == _SOURCE_NEXT_DIRECTION,
            observed=stage122_summary.get("recommended_next_direction"),
            expected=_SOURCE_NEXT_DIRECTION,
        ),
        _check(
            name="stage122_signal_supports_first_stage_direction",
            passed=stage122_summary.get("blocked_signal_has_real_hit20_recoveries")
            is True
            and stage122_summary.get("blocked_signal_has_guard_relevant_regressions")
            is True
            and _review_status(stage122_summary, "selected_config_review")
            == "safe_but_weak"
            and _review_status(stage122_summary, "blocked_signal_config_review")
            == "positive_signal_but_guard_risky",
            observed={
                "blocked_signal_has_real_hit20_recoveries": stage122_summary.get(
                    "blocked_signal_has_real_hit20_recoveries"
                ),
                "blocked_signal_has_guard_relevant_regressions": stage122_summary.get(
                    "blocked_signal_has_guard_relevant_regressions"
                ),
                "selected_status": _review_status(
                    stage122_summary, "selected_config_review"
                ),
                "blocked_status": _review_status(
                    stage122_summary, "blocked_signal_config_review"
                ),
            },
            expected="safe selected config plus real but guard-risky blocked signal",
        ),
        _check(
            name="stage122_runtime_and_test_boundaries_locked",
            passed=stage122_summary.get("can_open_final_test_gate_now") is False
            and stage122_summary.get("can_run_final_test_metrics_now") is False
            and stage122_summary.get("can_use_test_for_tuning") is False
            and stage122_summary.get("fallback_strategies_enabled") is False
            and stage122_summary.get("default_runtime_policy") == "unchanged"
            and stage122_summary.get("runtime_defaultization_supported") is False,
            observed=stage122_summary,
            expected="test locked, runtime unchanged, fallback disabled",
        ),
        _check(
            name="stage123_uses_train_grouped_cv",
            passed=selection_rules["selection_mode"]
            == "train_grouped_cross_validation_candidate_pool_selection"
            and selection_rules["minimum_train_folds"] >= _MINIMUM_TRAIN_FOLDS,
            observed={
                "selection_mode": selection_rules["selection_mode"],
                "minimum_train_folds": selection_rules["minimum_train_folds"],
            },
            expected="train grouped-CV with at least 5 folds",
        ),
        _check(
            name="stage123_dev_is_report_only",
            passed=selection_rules["dev_rules"]["dev_selection_allowed"] is False
            and selection_rules["dev_rules"]["dev_retuning_allowed"] is False
            and selection_rules["dev_rules"]["dev_threshold_tuning_allowed"] is False,
            observed=selection_rules["dev_rules"],
            expected="dev report-only",
        ),
        _check(
            name="stage123_test_locked",
            passed=selection_rules["test_rules"]["test_access_allowed"] is False
            and selection_rules["test_rules"]["final_test_metrics_allowed"] is False
            and selection_rules["test_rules"]["test_tuning_allowed"] is False,
            observed=selection_rules["test_rules"],
            expected="test access and final metrics disabled",
        ),
        _check(
            name="stage123_runtime_defaults_unchanged",
            passed=selection_rules["runtime_rules"]["default_runtime_policy"]
            == "unchanged"
            and selection_rules["runtime_rules"]["fallback_strategies_enabled"] is False
            and selection_rules["runtime_rules"][
                "runtime_defaultization_allowed_in_stage123"
            ]
            is False,
            observed=selection_rules["runtime_rules"],
            expected="runtime unchanged and no fallback strategies",
        ),
        _check(
            name="stage123_candidate_configs_present",
            passed=len(candidate_configs) == 7
            and set(_family_counts(candidate_configs)) == set(_CANDIDATE_FAMILY_IDS),
            observed={
                "candidate_config_count": len(candidate_configs),
                "families": sorted(_family_counts(candidate_configs)),
            },
            expected={"candidate_config_count": 7, "families": list(_CANDIDATE_FAMILY_IDS)},
        ),
        _check(
            name="stage123_expands_beyond_stage116_top200",
            passed=all(
                int(config["candidate_generation"]["target_pool_depth"])
                > _BASELINE_POOL_DEPTH
                for config in candidate_configs
            ),
            observed=[
                {
                    "config_id": config["config_id"],
                    "target_pool_depth": config["candidate_generation"][
                        "target_pool_depth"
                    ],
                }
                for config in candidate_configs
            ],
            expected="all configs target a pool deeper than Stage116 top200",
        ),
        _check(
            name="stage123_stays_within_fast_filter_budget",
            passed=all(
                int(config["candidate_generation"]["channel_top_k"]) <= _MAX_CHANNEL_TOP_K
                and int(config["candidate_generation"]["target_pool_depth"])
                <= _MAX_OUTPUT_POOL_DEPTH
                for config in candidate_configs
            ),
            observed=[
                {
                    "config_id": config["config_id"],
                    "channel_top_k": config["candidate_generation"]["channel_top_k"],
                    "target_pool_depth": config["candidate_generation"][
                        "target_pool_depth"
                    ],
                }
                for config in candidate_configs
            ],
            expected={
                "maximum_channel_top_k": _MAX_CHANNEL_TOP_K,
                "maximum_output_pool_depth": _MAX_OUTPUT_POOL_DEPTH,
            },
        ),
        _check(
            name="stage123_configs_do_not_download_models",
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
            name="stage123_configs_do_not_use_oracle_features",
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
            name="stage123_no_candidate_rows_or_metrics_run",
            passed=frozen["candidate_generation_contract"][
                "raw_candidate_rows_written_in_stage123"
            ]
            is False,
            observed=frozen["candidate_generation_contract"][
                "raw_candidate_rows_written_in_stage123"
            ],
            expected=False,
        ),
        _check(
            name="stage123_public_safe_contract_passed",
            passed=public_safe["forbidden_keys_found"] == [],
            observed=public_safe["forbidden_keys_found"],
            expected=[],
        ),
    ]


def _review_status(stage122_summary: Mapping[str, Any], key: str) -> str | None:
    review = stage122_summary.get(key)
    if not isinstance(review, Mapping):
        return None
    status = review.get("status")
    return str(status) if status is not None else None


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed:
        return {
            "status": "primeqa_hybrid_first_stage_recall_expansion_protocol_blocked",
            "failed_checks": failed,
            "can_run_first_stage_recall_expansion_now": False,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": "primeqa_hybrid_first_stage_recall_expansion_protocol_frozen",
        "recommended_next_direction": _NEXT_DIRECTION,
        "can_run_first_stage_recall_expansion_now": True,
        "can_continue_train_dev_development": True,
        "requires_user_confirmation_before_train_dev_run": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }


def _stage122_signal_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = report["stage122_summary"]
    selected = summary.get("selected_config_review") or {}
    blocked = summary.get("blocked_signal_config_review") or {}
    return [
        BarDatum(
            label="selected train hit20 recoveries",
            value=float(selected.get("train_hit20_recovery_count") or 0),
            value_label=str(selected.get("train_hit20_recovery_count") or 0),
        ),
        BarDatum(
            label="selected train hit20 regressions",
            value=float(selected.get("train_hit20_regression_count") or 0),
            value_label=str(selected.get("train_hit20_regression_count") or 0),
        ),
        BarDatum(
            label="selected dev hit20 recoveries",
            value=float(selected.get("dev_hit20_recovery_count") or 0),
            value_label=str(selected.get("dev_hit20_recovery_count") or 0),
        ),
        BarDatum(
            label="blocked train hit20 recoveries",
            value=float(blocked.get("train_hit20_recovery_count") or 0),
            value_label=str(blocked.get("train_hit20_recovery_count") or 0),
        ),
        BarDatum(
            label="blocked train hit20 regressions",
            value=float(blocked.get("train_hit20_regression_count") or 0),
            value_label=str(blocked.get("train_hit20_regression_count") or 0),
        ),
        BarDatum(
            label="blocked dev hit20 recoveries",
            value=float(blocked.get("dev_hit20_recovery_count") or 0),
            value_label=str(blocked.get("dev_hit20_recovery_count") or 0),
        ),
        BarDatum(
            label="blocked dev hit20 regressions",
            value=float(blocked.get("dev_hit20_regression_count") or 0),
            value_label=str(blocked.get("dev_hit20_regression_count") or 0),
        ),
    ]


def _candidate_family_count_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counts = _family_counts(report["frozen_protocol"]["candidate_configs"])
    return [
        BarDatum(label=family_id, value=float(count), value_label=str(count))
        for family_id, count in sorted(counts.items())
    ]


def _target_pool_depth_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(config["config_id"]),
            value=float(config["candidate_generation"]["target_pool_depth"]),
            value_label=str(config["candidate_generation"]["target_pool_depth"]),
        )
        for config in report["frozen_protocol"]["candidate_configs"]
    ]


def _channel_top_k_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(config["config_id"]),
            value=float(config["candidate_generation"]["channel_top_k"]),
            value_label=str(config["candidate_generation"]["channel_top_k"]),
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
        "can_run_first_stage_recall_expansion_now",
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
