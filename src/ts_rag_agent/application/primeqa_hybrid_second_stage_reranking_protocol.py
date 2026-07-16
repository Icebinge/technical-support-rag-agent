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

_STAGE = "Stage 117"
_CREATED_AT = "2026-07-16"
_SOURCE_STAGE = "Stage 116"
_SOURCE_ANALYSIS_ID = "primeqa_hybrid_high_recall_union_candidate_pool_v1"
_SOURCE_STATUS = "primeqa_hybrid_high_recall_union_candidate_pool_completed"
_SOURCE_POOL_ID = "stage116_multi_route_union_candidate_pool"
_PROTOCOL_ID = "primeqa_hybrid_second_stage_reranking_protocol_v1"
_NEXT_DIRECTION = "run_second_stage_reranking_train_cv_dev_validation"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = ("test",)
_FIXED_CANDIDATE_POOL_DEPTH = 200
_MINIMUM_TRAIN_FOLDS = 5
_CANDIDATE_FAMILY_IDS = (
    "channel_rank_feature_reranker_family_v1",
    "lexical_document_feature_reranker_family_v1",
    "supervised_lightweight_reranker_family_v1",
)
_OBJECTIVE_WEIGHTS = {
    "mrr_at_20_delta": 2.0,
    "hit_at_10_delta": 1.5,
    "hit_at_20_delta": 1.0,
    "bm25_top10_gold_demotion_penalty": 2.0,
    "candidate_pool_recall_loss_penalty": 4.0,
}
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
class PrimeQAHybridSecondStageRerankingProtocolVisualization:
    """One generated Stage117 second-stage reranking protocol chart."""

    name: str
    path: str


def freeze_primeqa_hybrid_second_stage_reranking_protocol(
    *,
    stage116_report_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage117 train/dev-only second-stage reranking protocol."""

    started_at = time.perf_counter()
    stage116_report = _load_json_object(stage116_report_path)
    loaded_at = time.perf_counter()
    stage116_summary = _stage116_summary(stage116_report)
    frozen_protocol = _frozen_protocol(stage116_summary)
    preliminary_report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "protocol_scope": (
            "Train/dev-only protocol freeze for second-stage precision reranking "
            "over the fixed Stage116 ranked top200 candidate pool. This stage "
            "reads only the saved public-safe Stage116 report, freezes candidate "
            "families and train grouped-CV/dev reporting rules, does not load "
            "split files, does not load corpus documents, does not build candidate "
            "rows, does not run reranking or answer metrics, does not run final "
            "metrics, does not select from dev-only observations, does not add "
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
            "selection_split": "train",
            "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
            "validation_split": "dev",
            "dev_validation_mode": "single_pass_report_only_no_retuning",
            "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage116_report": _fingerprint(stage116_report_path),
        },
        "stage116_summary": stage116_summary,
        "frozen_protocol": frozen_protocol,
    }
    guard_checks = _guard_checks(
        report=preliminary_report,
        stage116_report=stage116_report,
        stage116_summary=stage116_summary,
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


def write_primeqa_hybrid_second_stage_reranking_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridSecondStageRerankingProtocolVisualization]:
    """Write SVG charts for Stage117 protocol freeze."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage117_stage116_candidate_pool_recall.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage117 Stage116 candidate-pool recall",
                bars=_stage116_recall_bars(report),
                x_label="gold-document recall",
                width=1280,
                margin_left=560,
            )
        ),
        "stage117_candidate_family_priorities.svg": render_horizontal_bar_chart_svg(
            title="Stage117 candidate family priorities",
            bars=_family_priority_bars(report),
            x_label="priority score",
            width=1480,
            margin_left=760,
        ),
        "stage117_candidate_config_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage117 candidate config counts",
            bars=_candidate_config_count_bars(report),
            x_label="config count",
            width=1480,
            margin_left=760,
        ),
        "stage117_objective_weights.svg": render_horizontal_bar_chart_svg(
            title="Stage117 train-CV objective weights",
            bars=_objective_weight_bars(report),
            x_label="objective weight",
            width=1380,
            margin_left=680,
        ),
        "stage117_guard_thresholds.svg": render_horizontal_bar_chart_svg(
            title="Stage117 train-CV guard thresholds",
            bars=_guard_threshold_bars(report),
            x_label="maximum allowed count or rate",
            width=1500,
            margin_left=800,
        ),
        "stage117_protocol_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage117 protocol decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1320,
            margin_left=660,
        ),
        "stage117_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage117 guard checks",
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
            PrimeQAHybridSecondStageRerankingProtocolVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage116_summary(stage116_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage116_report.get("decision") or {}
    analysis_result = decision.get("analysis_result") or {}
    split_contract = stage116_report.get("split_contract") or {}
    config = stage116_report.get("analysis_config") or {}
    loaded = stage116_report.get("loaded_data_summary") or {}
    split_samples = loaded.get("split_samples") or {}
    pool_metrics = stage116_report.get("candidate_pool_metrics_by_split") or {}
    comparisons = stage116_report.get("comparisons_to_baseline") or {}
    dense = stage116_report.get("dense_channel_preflight") or {}
    public_safe = stage116_report.get("public_safe_contract") or {}
    return {
        "stage": stage116_report.get("stage"),
        "analysis_id": stage116_report.get("analysis_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "can_continue_second_stage_precision_experiment": decision.get(
            "can_continue_second_stage_precision_experiment"
        ),
        "candidate_pool_id": config.get("candidate_pool_id"),
        "candidate_pool_depth": max(config.get("pool_top_k_values") or [0]),
        "channel_top_k": config.get("channel_top_k"),
        "rrf_k": config.get("rrf_k"),
        "channel_count": len(stage116_report.get("channel_catalog") or []),
        "dense_status": dense.get("status"),
        "dense_can_run_without_download": dense.get("can_run_without_download"),
        "split_name": split_contract.get("split_name"),
        "protocol_version": split_contract.get("protocol_version"),
        "development_splits": split_contract.get("development_splits") or [],
        "dev_selection_used": split_contract.get("dev_selection_used"),
        "dev_retuning_used": split_contract.get("dev_retuning_used"),
        "forbidden_final_splits": split_contract.get("forbidden_final_splits") or [],
        "train_answerable": (split_samples.get("train") or {}).get("answerable_count"),
        "dev_answerable": (split_samples.get("dev") or {}).get("answerable_count"),
        "train_union_hit_at_100": _split_hit(pool_metrics, "train", "100"),
        "train_union_hit_at_200": _split_hit(pool_metrics, "train", "200"),
        "train_uncapped_union_hit_rate": (
            (pool_metrics.get("train") or {}).get("uncapped_union_hit_rate")
        ),
        "dev_union_hit_at_100": _split_hit(pool_metrics, "dev", "100"),
        "dev_union_hit_at_200": _split_hit(pool_metrics, "dev", "200"),
        "dev_uncapped_union_hit_rate": (
            (pool_metrics.get("dev") or {}).get("uncapped_union_hit_rate")
        ),
        "dev_uncapped_union_not_found_count": analysis_result.get(
            "dev_uncapped_union_not_found_count"
        ),
        "dev_hit_count_delta_at_100_vs_bm25": (
            (comparisons.get("dev") or {}).get("hit@100") or {}
        ).get("hit_count_delta"),
        "dev_hit_count_delta_at_200_vs_bm25": (
            (comparisons.get("dev") or {}).get("hit@200") or {}
        ).get("hit_count_delta"),
        "dev_average_uncapped_pool_size": (
            ((pool_metrics.get("dev") or {}).get("candidate_pool_size") or {}).get(
                "average"
            )
        ),
        "dev_p95_uncapped_pool_size": (
            ((pool_metrics.get("dev") or {}).get("candidate_pool_size") or {}).get(
                "p95"
            )
        ),
        "guard_pass_count": sum(
            1 for check in stage116_report.get("guard_checks") or [] if check.get("passed")
        ),
        "guard_count": len(stage116_report.get("guard_checks") or []),
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _frozen_protocol(stage116_summary: Mapping[str, Any]) -> dict[str, Any]:
    candidate_families = _candidate_families(stage116_summary)
    candidate_configs = _candidate_configs()
    return {
        "protocol_id": _PROTOCOL_ID,
        "protocol_status": "frozen_requires_user_confirmation_before_train_dev_run",
        "source_stages": [_SOURCE_STAGE],
        "reranking_mode": "protocol_freeze_only_no_candidate_rows_no_metrics",
        "objective": (
            "Use the fixed Stage116 ranked top200 candidate pool as a high-recall "
            "first-stage input, then compare lightweight second-stage rerankers "
            "that improve gold-document rank within the pool without reducing "
            "candidate-pool recall or using dev/test for tuning."
        ),
        "fixed_candidate_pool_contract": {
            "source_pool_id": _SOURCE_POOL_ID,
            "candidate_pool_depth": _FIXED_CANDIDATE_POOL_DEPTH,
            "source_stage116_rrf_k": stage116_summary.get("rrf_k"),
            "source_stage116_channel_top_k": stage116_summary.get("channel_top_k"),
            "reranker_may_reorder_pool": True,
            "reranker_may_add_documents": False,
            "reranker_may_drop_documents_before_top200_metric": False,
            "uncapped_union_is_not_runtime_input": True,
        },
        "candidate_families": candidate_families,
        "candidate_configs": candidate_configs,
        "selection_rules": _selection_rules(),
        "feature_contract": _feature_contract(),
        "candidate_artifact_contract": _candidate_artifact_contract(),
        "blocked_options": _blocked_options(),
        "public_safe_summary_only": True,
    }


def _candidate_families(stage116_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    dev_delta_200 = stage116_summary.get("dev_hit_count_delta_at_200_vs_bm25") or 0
    return [
        {
            "family_id": "channel_rank_feature_reranker_family_v1",
            "name": "Channel-rank feature rerankers",
            "priority_score": 84 + int(dev_delta_200),
            "candidate_count": 2,
            "why_included": (
                "Stage116 already computes a route-union order. These candidates "
                "test whether route agreement, best route rank, and route-family "
                "signals can improve precision without new model downloads."
            ),
            "runtime_feature_boundary": "uses only runtime-visible route ranks and scores",
        },
        {
            "family_id": "lexical_document_feature_reranker_family_v1",
            "name": "Lexical document-feature rerankers",
            "priority_score": 81,
            "candidate_count": 3,
            "why_included": (
                "Stage116 still leaves a small dev tail unrecalled and the next "
                "precision problem is ranking the recalled gold document higher "
                "inside top200 using title, heading, body coverage, and exact "
                "special-token features."
            ),
            "runtime_feature_boundary": "uses only question text and candidate document text",
        },
        {
            "family_id": "supervised_lightweight_reranker_family_v1",
            "name": "Supervised lightweight rerankers",
            "priority_score": 78,
            "candidate_count": 3,
            "why_included": (
                "Train-only labels can teach a small linear model how to combine "
                "route-rank and lexical features; grouped CV must prove stability "
                "before any dev report or runtime discussion."
            ),
            "runtime_feature_boundary": (
                "gold labels are allowed only during train-CV fitting, never as "
                "runtime features"
            ),
        },
    ]


def _candidate_configs() -> list[dict[str, Any]]:
    return [
        _candidate_config(
            config_id="crf_route_agreement_best_rank_v1",
            family_id="channel_rank_feature_reranker_family_v1",
            ranking_method="deterministic_weighted_score",
            features=[
                "stage116_rrf_score",
                "route_hit_count",
                "best_route_rank",
                "lexical_route_hit_count",
                "dense_route_hit_count",
            ],
            payload={
                "route_hit_count_weight": 1.2,
                "best_rank_weight": 0.9,
                "dense_route_weight": 0.4,
            },
        ),
        _candidate_config(
            config_id="crf_lexical_routes_first_v1",
            family_id="channel_rank_feature_reranker_family_v1",
            ranking_method="deterministic_weighted_score",
            features=[
                "stage116_rrf_score",
                "full_document_bm25_rank",
                "section_bm25_rank",
                "title_heading_rank",
                "special_token_boosted_rank",
            ],
            payload={
                "lexical_route_weight": 1.4,
                "dense_route_weight": 0.2,
                "stage116_rrf_weight": 1.0,
            },
        ),
        _candidate_config(
            config_id="ldf_title_heading_overlap_v1",
            family_id="lexical_document_feature_reranker_family_v1",
            ranking_method="deterministic_weighted_score",
            features=[
                "query_title_token_overlap",
                "query_section_heading_overlap",
                "query_token_coverage",
                "stage116_rrf_score",
            ],
            payload={
                "title_overlap_weight": 1.5,
                "heading_overlap_weight": 1.25,
                "coverage_weight": 0.7,
            },
        ),
        _candidate_config(
            config_id="ldf_title_heading_body_coverage_v1",
            family_id="lexical_document_feature_reranker_family_v1",
            ranking_method="deterministic_weighted_score",
            features=[
                "query_title_token_overlap",
                "query_section_heading_overlap",
                "query_body_token_coverage",
                "document_length_bucket",
                "stage116_rrf_score",
            ],
            payload={
                "title_overlap_weight": 1.1,
                "heading_overlap_weight": 1.1,
                "body_coverage_weight": 0.9,
                "long_document_penalty": 0.2,
            },
        ),
        _candidate_config(
            config_id="ldf_special_token_title_heading_v1",
            family_id="lexical_document_feature_reranker_family_v1",
            ranking_method="deterministic_weighted_score",
            features=[
                "query_special_token_match_count",
                "title_special_token_match_count",
                "heading_special_token_match_count",
                "stage116_rrf_score",
            ],
            payload={
                "special_token_weight": 1.8,
                "title_heading_bonus": 0.8,
                "stage116_rrf_weight": 1.0,
            },
        ),
        _candidate_config(
            config_id="slr_logistic_balanced_v1",
            family_id="supervised_lightweight_reranker_family_v1",
            ranking_method="train_cv_logistic_regression",
            features=[
                "stage116_rrf_score",
                "route_hit_count",
                "best_route_rank",
                "query_title_token_overlap",
                "query_body_token_coverage",
                "special_token_match_count",
            ],
            payload={
                "class_weight": "balanced",
                "negative_sampling": "top40_ranked_pool_plus_route_disagreements",
                "max_negatives_per_question": 40,
            },
        ),
        _candidate_config(
            config_id="slr_logistic_hard_negative_v1",
            family_id="supervised_lightweight_reranker_family_v1",
            ranking_method="train_cv_logistic_regression",
            features=[
                "stage116_rrf_score",
                "route_hit_count",
                "best_route_rank",
                "query_title_token_overlap",
                "query_body_token_coverage",
                "special_token_match_count",
                "bm25_top10_non_gold_indicator",
            ],
            payload={
                "class_weight": "balanced",
                "negative_sampling": "bm25_top10_and_stage116_top50_hard_negatives",
                "max_negatives_per_question": 50,
            },
        ),
        _candidate_config(
            config_id="slr_ridge_rank_proxy_v1",
            family_id="supervised_lightweight_reranker_family_v1",
            ranking_method="train_cv_ridge_rank_proxy",
            features=[
                "stage116_rrf_score",
                "route_hit_count",
                "best_route_rank",
                "query_title_token_overlap",
                "query_body_token_coverage",
                "special_token_match_count",
                "dense_route_best_rank",
            ],
            payload={
                "target": "gold_doc_rank_proxy",
                "negative_sampling": "top60_ranked_pool",
                "max_negatives_per_question": 60,
            },
        ),
    ]


def _candidate_config(
    *,
    config_id: str,
    family_id: str,
    ranking_method: str,
    features: list[str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "config_id": config_id,
        "family_id": family_id,
        "ranking_method": ranking_method,
        "candidate_pool_depth": _FIXED_CANDIDATE_POOL_DEPTH,
        "features": features,
        "payload": payload,
        "uses_gold_labels_at_runtime": False,
        "uses_source_doc_ids": False,
        "requires_model_download": False,
        "selection_eligible": True,
    }


def _selection_rules() -> dict[str, Any]:
    return {
        "selection_split": "train",
        "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
        "minimum_train_folds": _MINIMUM_TRAIN_FOLDS,
        "train_group_key": "normalized_question_plus_answer_document",
        "eligible_training_rows": (
            "answerable train rows whose gold document is present in fixed "
            "Stage116 top200 pool"
        ),
        "baseline_order": "stage116_fixed_rrf_pool_order",
        "primary_metrics": [
            "mrr_at_20_delta_vs_stage116_order",
            "hit_at_10_delta_vs_stage116_order",
            "hit_at_20_delta_vs_stage116_order",
        ],
        "objective_weights": dict(_OBJECTIVE_WEIGHTS),
        "selection_tie_breakers": [
            "fewer_bm25_top10_gold_demotions",
            "lower_top20_regression_count",
            "lower_average_selected_rank",
            "config_id",
        ],
        "guard_thresholds": {
            "maximum_train_cv_hit_at_200_loss_count": 0,
            "maximum_train_cv_bm25_top10_gold_demotions_to_below_50": 0,
            "maximum_train_cv_hit_at_20_regression_rate": 0.02,
            "maximum_train_cv_top10_regression_count": 3,
            "minimum_train_cv_mrr_at_20_delta": 0.0,
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
            "runtime_defaultization_allowed_in_stage117": False,
        },
    }


def _feature_contract() -> dict[str, Any]:
    return {
        "allowed_runtime_feature_sources": [
            "question text",
            "candidate document title/text/section headings",
            "Stage116 route ranks and route scores regenerated from runtime-visible indexes",
            "existing local dense-cache route ranks when dense route is explicitly enabled",
        ],
        "training_only_label_sources": [
            "train split answer document labels",
        ],
        "forbidden_runtime_feature_sources": sorted(_FORBIDDEN_RUNTIME_FEATURES),
        "new_model_download_allowed": False,
        "llm_judging_allowed": False,
        "source_doc_ids_oracle_allowed": False,
    }


def _candidate_artifact_contract() -> dict[str, Any]:
    return {
        "candidate_rows_not_built_in_stage117": True,
        "next_stage_candidate_artifact_policy": "ignored_artifact_under_artifacts",
        "public_report_writes_raw_question_text": False,
        "public_report_writes_raw_answer_text": False,
        "public_report_writes_raw_document_text": False,
        "public_report_writes_document_ids": False,
        "candidate_pool_depth": _FIXED_CANDIDATE_POOL_DEPTH,
    }


def _blocked_options() -> list[dict[str, Any]]:
    return [
        {
            "option_id": "uncapped_union_as_answer_input_blocked",
            "reason": (
                "Stage116 uncapped union averages hundreds of documents; Stage117 "
                "must use fixed ranked top200 as the precision boundary."
            ),
        },
        {
            "option_id": "source_doc_ids_oracle_reranker_blocked",
            "reason": "source DOC_IDS are dataset metadata, not runtime retrieval evidence.",
        },
        {
            "option_id": "dev_selected_threshold_blocked",
            "reason": "dev is report-only and cannot tune thresholds or select configs.",
        },
        {
            "option_id": "final_test_reranking_metrics_blocked",
            "reason": "test remains locked until a later explicit final-test gate.",
        },
        {
            "option_id": "runtime_defaultization_in_stage117_blocked",
            "reason": "Stage117 is protocol freeze only; no runtime/default changes.",
        },
    ]


def _guard_checks(
    *,
    report: Mapping[str, Any],
    stage116_report: Mapping[str, Any],
    stage116_summary: Mapping[str, Any],
    user_confirmed_protocol: bool,
) -> list[dict[str, Any]]:
    frozen = report["frozen_protocol"]
    selection_rules = frozen["selection_rules"]
    candidate_configs = frozen["candidate_configs"]
    public_safe = _public_safe_contract(report)
    forbidden_feature_uses = _forbidden_feature_uses(candidate_configs)
    return [
        _check(
            name="user_confirmed_stage117_protocol",
            passed=user_confirmed_protocol,
            observed=report["user_confirmation"]["confirmation_note"],
            expected="user confirmed Stage117 second-stage reranking protocol freeze",
        ),
        _check(
            name="stage116_completed",
            passed=stage116_summary.get("decision_status") == _SOURCE_STATUS,
            observed=stage116_summary.get("decision_status"),
            expected=_SOURCE_STATUS,
        ),
        _check(
            name="stage116_analysis_id_matches",
            passed=stage116_summary.get("analysis_id") == _SOURCE_ANALYSIS_ID,
            observed=stage116_summary.get("analysis_id"),
            expected=_SOURCE_ANALYSIS_ID,
        ),
        _check(
            name="stage116_fixed_pool_is_top200",
            passed=stage116_summary.get("candidate_pool_depth")
            == _FIXED_CANDIDATE_POOL_DEPTH,
            observed=stage116_summary.get("candidate_pool_depth"),
            expected=_FIXED_CANDIDATE_POOL_DEPTH,
        ),
        _check(
            name="stage116_can_continue_second_stage_precision",
            passed=bool(
                stage116_summary.get("can_continue_second_stage_precision_experiment")
            ),
            observed=stage116_summary.get(
                "can_continue_second_stage_precision_experiment"
            ),
            expected=True,
        ),
        _check(
            name="stage116_dev_top200_lift_positive",
            passed=(stage116_summary.get("dev_hit_count_delta_at_200_vs_bm25") or 0) > 0,
            observed=stage116_summary.get("dev_hit_count_delta_at_200_vs_bm25"),
            expected="positive dev hit@200 lift over full-document BM25",
        ),
        _check(
            name="stage116_final_test_metrics_not_run",
            passed=stage116_summary.get("can_run_final_test_metrics_now") is False,
            observed=stage116_summary.get("can_run_final_test_metrics_now"),
            expected=False,
        ),
        _check(
            name="stage116_runtime_defaults_unchanged",
            passed=stage116_summary.get("default_runtime_policy") == "unchanged",
            observed=stage116_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage116_fallback_strategies_not_added",
            passed=stage116_summary.get("fallback_strategies_enabled") is False,
            observed=stage116_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="stage117_protocol_uses_train_grouped_cv",
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
            name="stage117_dev_is_report_only",
            passed=selection_rules["dev_rules"]["dev_selection_allowed"] is False
            and selection_rules["dev_rules"]["dev_threshold_tuning_allowed"] is False
            and selection_rules["dev_rules"]["dev_retuning_allowed"] is False,
            observed=selection_rules["dev_rules"],
            expected="dev report-only, no selection, no retuning",
        ),
        _check(
            name="stage117_test_locked",
            passed=selection_rules["test_rules"]["test_access_allowed"] is False
            and selection_rules["test_rules"]["final_test_metrics_allowed"] is False,
            observed=selection_rules["test_rules"],
            expected="test access and final metrics disabled",
        ),
        _check(
            name="stage117_runtime_defaults_unchanged",
            passed=selection_rules["runtime_rules"]["default_runtime_policy"]
            == "unchanged"
            and selection_rules["runtime_rules"]["fallback_strategies_enabled"] is False,
            observed=selection_rules["runtime_rules"],
            expected="runtime unchanged and no fallback strategies",
        ),
        _check(
            name="stage117_candidate_configs_present",
            passed=len(candidate_configs) == 8
            and set(_family_counts(candidate_configs)) == set(_CANDIDATE_FAMILY_IDS),
            observed={
                "candidate_config_count": len(candidate_configs),
                "families": sorted(_family_counts(candidate_configs)),
            },
            expected={"candidate_config_count": 8, "families": list(_CANDIDATE_FAMILY_IDS)},
        ),
        _check(
            name="stage117_candidate_configs_do_not_use_forbidden_runtime_features",
            passed=not forbidden_feature_uses,
            observed=forbidden_feature_uses,
            expected=[],
        ),
        _check(
            name="stage117_no_candidate_rows_or_metrics_run",
            passed=frozen["candidate_artifact_contract"][
                "candidate_rows_not_built_in_stage117"
            ]
            and frozen["reranking_mode"] == "protocol_freeze_only_no_candidate_rows_no_metrics",
            observed={
                "candidate_rows_not_built": frozen["candidate_artifact_contract"][
                    "candidate_rows_not_built_in_stage117"
                ],
                "reranking_mode": frozen["reranking_mode"],
            },
            expected="protocol freeze only, no candidate rows, no metrics",
        ),
        _check(
            name="stage117_public_safe_contract_passed",
            passed=public_safe["forbidden_keys_found"] == [],
            observed=public_safe["forbidden_keys_found"],
            expected=[],
        ),
        _check(
            name="stage116_public_safe_contract_was_clean",
            passed=stage116_summary.get("public_safe_forbidden_keys_found") == [],
            observed=stage116_summary.get("public_safe_forbidden_keys_found"),
            expected=[],
        ),
        _check(
            name="stage116_report_loaded_without_test_split",
            passed=not _stage116_loaded_test_split(stage116_report),
            observed=_stage116_loaded_test_split(stage116_report),
            expected=False,
        ),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passed = all(check["passed"] for check in guard_checks)
    if not passed:
        return {
            "status": "primeqa_hybrid_second_stage_reranking_protocol_blocked",
            "recommended_next_direction": "fix_stage117_protocol_blockers",
            "can_continue_train_dev_development": False,
            "can_run_second_stage_reranking_now": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": "primeqa_hybrid_second_stage_reranking_protocol_frozen",
        "recommended_next_direction": _NEXT_DIRECTION,
        "can_continue_train_dev_development": True,
        "can_run_second_stage_reranking_now": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage118: run the frozen second-stage reranking train-CV/dev "
            "validation over the fixed Stage116 top200 candidate pool. Keep test "
            "locked, do not tune on dev, and do not change runtime defaults."
        ),
    }


def _split_hit(
    pool_metrics: Mapping[str, Any],
    split: str,
    top_k: str,
) -> float | None:
    return ((pool_metrics.get(split) or {}).get("hit_at_k") or {}).get(top_k)


def _forbidden_feature_uses(
    candidate_configs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    violations = []
    for config in candidate_configs:
        features = set(str(feature) for feature in config.get("features") or [])
        forbidden = sorted(features & _FORBIDDEN_RUNTIME_FEATURES)
        if forbidden or config.get("uses_source_doc_ids") is True:
            violations.append(
                {
                    "config_id": config.get("config_id"),
                    "forbidden_features": forbidden,
                    "uses_source_doc_ids": config.get("uses_source_doc_ids"),
                }
            )
    return violations


def _family_counts(candidate_configs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(config["family_id"]) for config in candidate_configs).items()))


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    forbidden_keys = sorted(_find_forbidden_public_keys(report))
    return {
        "public_safe_summary_only": True,
        "raw_question_text_written": False,
        "raw_answer_text_written": False,
        "raw_document_text_written": False,
        "raw_document_ids_written": False,
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


def _stage116_loaded_test_split(stage116_report: Mapping[str, Any]) -> bool:
    loaded = stage116_report.get("loaded_data_summary") or {}
    return bool(loaded.get("test_split_loaded"))


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


def _stage116_recall_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = report["stage116_summary"]
    rows = [
        ("train union hit@100", summary.get("train_union_hit_at_100")),
        ("train union hit@200", summary.get("train_union_hit_at_200")),
        ("train uncapped union", summary.get("train_uncapped_union_hit_rate")),
        ("dev union hit@100", summary.get("dev_union_hit_at_100")),
        ("dev union hit@200", summary.get("dev_union_hit_at_200")),
        ("dev uncapped union", summary.get("dev_uncapped_union_hit_rate")),
    ]
    return [
        BarDatum(label=label, value=float(value or 0.0), value_label=f"{float(value or 0.0):.4f}")
        for label, value in rows
    ]


def _family_priority_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    families = report["frozen_protocol"]["candidate_families"]
    return sorted(
        [
            BarDatum(
                label=str(family["family_id"]),
                value=float(family["priority_score"]),
                value_label=str(family["priority_score"]),
            )
            for family in families
        ],
        key=lambda bar: (-bar.value, bar.label),
    )


def _candidate_config_count_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counts = _family_counts(report["frozen_protocol"]["candidate_configs"])
    return [
        BarDatum(label=family_id, value=float(count), value_label=str(count))
        for family_id, count in counts.items()
    ]


def _objective_weight_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    weights = report["frozen_protocol"]["selection_rules"]["objective_weights"]
    return [
        BarDatum(label=str(name), value=float(value), value_label=str(value))
        for name, value in sorted(weights.items())
    ]


def _guard_threshold_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    thresholds = report["frozen_protocol"]["selection_rules"]["guard_thresholds"]
    return [
        BarDatum(label=str(name), value=float(value), value_label=str(value))
        for name, value in sorted(thresholds.items())
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report["decision"]
    flags = (
        "can_continue_train_dev_development",
        "can_run_second_stage_reranking_now",
        "can_open_final_test_gate_now",
        "can_run_final_test_metrics_now",
        "can_use_test_for_tuning",
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


def _load_json_object(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return loaded


def _fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": _sha256(path) if path.exists() and path.is_file() else None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
