from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 128"
_CREATED_AT = "2026-07-16"
_SOURCE_STAGE127_STATUS = (
    "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_"
    "selected_config_review_completed"
)
_SOURCE_STAGE127_REVIEW_ID = (
    "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_"
    "selected_config_review_v1"
)
_SOURCE_NEXT_DIRECTION = (
    "freeze_agent_retrieval_integration_protocol_for_selected_prefix_expansion"
)
_PROTOCOL_ID = "primeqa_hybrid_agent_retrieval_integration_protocol_v1"
_NEXT_DIRECTION = "run_agent_retrieval_integration_train_cv_dev_validation"
_SELECTED_CONFIG_ID = "prefix_existing_dense_broad_append200_v1"
_SELECTED_FAMILY_ID = "stage116_prefix_existing_dense_append_family_v1"
_BEST_DEV_CONFIG_ID = "prefix_query_variant_append100_v1"
_BASELINE_PREFIX_DEPTH = 200
_APPEND_START_RANK = 201
_APPEND_BUDGET = 200
_TARGET_POOL_DEPTH = 400
_MINIMUM_TRAIN_FOLDS = 5
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
class PrimeQAHybridAgentRetrievalIntegrationProtocolVisualization:
    """One generated Stage128 agent retrieval integration protocol chart."""

    name: str
    path: str


def freeze_primeqa_hybrid_agent_retrieval_integration_protocol(
    *,
    stage127_review_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage128 train/dev-only agent retrieval integration protocol."""

    started_at = time.perf_counter()
    stage127_review = _load_json_object(stage127_review_path)
    loaded_at = time.perf_counter()

    stage127_summary = _stage127_summary(stage127_review)
    frozen_protocol = _frozen_protocol(stage127_summary)
    preliminary_report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "protocol_scope": (
            "Train/dev-only protocol freeze for integrating the Stage126 "
            "selected Stage116 prefix-preserving 400-depth candidate pool into "
            "the agent retrieval design. This stage reads only the public-safe "
            "Stage127 review report, does not load split files, does not load "
            "corpus documents, does not build candidate rows, does not run "
            "retrieval, reranking, answering, or final metrics, does not select "
            "from dev-only observations, does not add fallback strategies, and "
            "does not change runtime defaults."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_protocol),
            "confirmation_note": confirmation_note,
        },
        "source_files": {
            "stage127_review": _fingerprint(stage127_review_path),
        },
        "stage127_summary": stage127_summary,
        "frozen_protocol": frozen_protocol,
    }
    guard_checks = _guard_checks(
        report=preliminary_report,
        stage127_summary=stage127_summary,
        user_confirmed_protocol=user_confirmed_protocol,
    )
    checked_at = time.perf_counter()
    report = {
        **preliminary_report,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
        "timing_seconds": {
            "load_review": round(loaded_at - started_at, 3),
            "freeze_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_agent_retrieval_integration_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridAgentRetrievalIntegrationProtocolVisualization]:
    """Write SVG charts for Stage128 protocol freeze."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage128_selected_config_value.svg": render_horizontal_bar_chart_svg(
            title="Stage128 selected config value",
            bars=_selected_value_bars(report),
            x_label="count",
            width=1320,
            margin_left=640,
        ),
        "stage128_candidate_pool_contract.svg": render_horizontal_bar_chart_svg(
            title="Stage128 candidate pool contract",
            bars=_candidate_pool_contract_bars(report),
            x_label="documents",
            width=1320,
            margin_left=640,
        ),
        "stage128_agent_consumer_policy.svg": render_horizontal_bar_chart_svg(
            title="Stage128 agent consumer policy",
            bars=_consumer_policy_bars(report),
            x_label="1 means allowed",
            width=1480,
            margin_left=780,
        ),
        "stage128_risk_review_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage128 risk review flags",
            bars=_risk_review_flag_bars(report),
            x_label="1 means true",
            width=1520,
            margin_left=820,
        ),
        "stage128_protocol_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage128 protocol decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1460,
            margin_left=780,
        ),
        "stage128_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage128 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1900,
            margin_left=1060,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridAgentRetrievalIntegrationProtocolVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage127_summary(stage127_review: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage127_review.get("decision") or {}
    selected = stage127_review.get("selected_config_review") or {}
    agent_review = stage127_review.get("agent_design_review") or {}
    risk_review = agent_review.get("risk_review") or {}
    cost = agent_review.get("cost_profile") or {}
    contract = agent_review.get("retrieval_contract") or {}
    public_safe = stage127_review.get("public_safe_contract") or {}
    guard_checks = stage127_review.get("guard_checks") or []
    return {
        "stage": stage127_review.get("stage"),
        "review_id": stage127_review.get("review_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "selected_config_id": decision.get("selected_config_id")
        or selected.get("config_id"),
        "selected_family_id": decision.get("selected_family_id")
        or selected.get("family_id"),
        "selected_config_supported_for_agent_protocol_design": decision.get(
            "selected_config_supported_for_agent_protocol_design"
        ),
        "selected_train_incremental_gain_count": (
            (selected.get("train") or {}).get(
                "target_depth_hit_count_gain_vs_stage116_top200"
            )
        ),
        "selected_train_incremental_gain_rate": (
            (selected.get("train") or {}).get("incremental_gain_rate")
        ),
        "selected_dev_incremental_gain_count": (
            (selected.get("dev") or {}).get(
                "target_depth_hit_count_gain_vs_stage116_top200"
            )
        ),
        "selected_dev_incremental_gain_rate": (
            (selected.get("dev") or {}).get("incremental_gain_rate")
        ),
        "selected_train_hit200_loss_count": (
            (selected.get("train") or {}).get("hit_at_200_loss_count")
        ),
        "selected_dev_hit200_loss_count": (
            (selected.get("dev") or {}).get("hit_at_200_loss_count")
        ),
        "selected_train_prefix_violation_count": (
            (selected.get("train") or {}).get("prefix_identity_violation_count")
        ),
        "selected_dev_prefix_violation_count": (
            (selected.get("dev") or {}).get("prefix_identity_violation_count")
        ),
        "baseline_prefix_depth": contract.get("baseline_prefix_depth"),
        "append_start_rank": contract.get("append_start_rank"),
        "append_budget": contract.get("append_budget"),
        "target_pool_depth": contract.get("target_pool_depth"),
        "candidate_depth_multiplier_vs_stage116": cost.get(
            "candidate_depth_multiplier_vs_stage116"
        ),
        "additional_candidates_per_query": cost.get("additional_candidates_per_query"),
        "channel_count": cost.get("channel_count"),
        "channel_families": cost.get("channel_families") or {},
        "risk_review": {
            "dev_gain_is_smaller_than_train_gain": risk_review.get(
                "dev_gain_is_smaller_than_train_gain"
            ),
            "best_dev_config_differs_from_train_selected": risk_review.get(
                "best_dev_config_differs_from_train_selected"
            ),
            "best_dev_config_id": risk_review.get("best_dev_config_id"),
            "best_dev_target_depth_gain": risk_review.get(
                "best_dev_target_depth_gain"
            ),
            "answer_quality_not_measured": risk_review.get(
                "answer_quality_not_measured"
            ),
            "final_test_not_run": risk_review.get("final_test_not_run"),
            "runtime_default_unchanged": risk_review.get("runtime_default_unchanged"),
        },
        "runtime_defaultization_allowed_now": decision.get(
            "runtime_defaultization_allowed_now"
        ),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get(
            "can_run_final_test_metrics_now"
        ),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "guard_check_count": len(guard_checks),
        "guard_check_passed_count": sum(1 for check in guard_checks if check.get("passed")),
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
    }


def _frozen_protocol(stage127_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "route_name": "stage116_prefix_expansion_agent_candidate_pool_integration",
        "source_review": {
            "stage": stage127_summary.get("stage"),
            "review_id": stage127_summary.get("review_id"),
            "status": stage127_summary.get("decision_status"),
            "recommended_next_direction": stage127_summary.get(
                "recommended_next_direction"
            ),
        },
        "selected_retrieval_config": {
            "config_id": _SELECTED_CONFIG_ID,
            "family_id": _SELECTED_FAMILY_ID,
            "selection_source": "Stage126 train grouped cross-validation",
            "dev_role": "report_only_no_retuning",
            "append_source_algorithm": "cached_dense_plus_lexical_rrf",
            "route_set": "stage116_lexical_routes_plus_existing_dense_cache_routes",
            "channel_top_k": _TARGET_POOL_DEPTH,
            "channel_count": stage127_summary.get("channel_count"),
            "channel_families": stage127_summary.get("channel_families"),
            "observed_train_incremental_gain_count": stage127_summary.get(
                "selected_train_incremental_gain_count"
            ),
            "observed_train_incremental_gain_rate": stage127_summary.get(
                "selected_train_incremental_gain_rate"
            ),
            "observed_dev_incremental_gain_count": stage127_summary.get(
                "selected_dev_incremental_gain_count"
            ),
            "observed_dev_incremental_gain_rate": stage127_summary.get(
                "selected_dev_incremental_gain_rate"
            ),
        },
        "agent_retrieval_contract": {
            "candidate_pool_output_depth": _TARGET_POOL_DEPTH,
            "candidate_pool_is_not_automatic_answer_context": True,
            "answer_context_policy": "unchanged_until_stage129_validation",
            "rank_regions": [
                {
                    "region_id": "stage116_immutable_prefix",
                    "rank_start": 1,
                    "rank_end": _BASELINE_PREFIX_DEPTH,
                    "source": "Stage116 fixed top200 order",
                    "purpose": "preserve the validated hit@200 retrieval boundary",
                    "may_reorder": False,
                    "may_drop": False,
                    "may_insert_expansion_candidate": False,
                },
                {
                    "region_id": "stage128_append_expansion",
                    "rank_start": _APPEND_START_RANK,
                    "rank_end": _TARGET_POOL_DEPTH,
                    "source": _SELECTED_CONFIG_ID,
                    "purpose": "provide additional recall candidates for evidence selection",
                    "append_budget": _APPEND_BUDGET,
                    "deduplicate_against_prefix": True,
                    "deduplicate_within_region": True,
                    "may_insert_before_rank_201": False,
                },
            ],
        },
        "agent_candidate_interface": {
            "interface_id": "agent_retrieval_candidate_pool_v1",
            "public_report_contains_raw_candidate_rows": False,
            "candidate_record_fields_to_validate": [
                "runtime_document_key",
                "candidate_rank",
                "rank_region_id",
                "region_rank",
                "retrieval_route_family",
                "retrieval_score_summary",
                "runtime_content_handle",
            ],
            "allowed_runtime_signals": [
                "runtime query text",
                "runtime corpus title/body/section content",
                "Stage116 immutable prefix rank",
                "Stage128 append region rank",
                "BM25 and section BM25 route ranks",
                "exact special-token route rank",
                "existing local dense-cache route ranks",
            ],
            "forbidden_runtime_signals": [
                "test membership",
                "gold labels",
                "answer document labels",
                "source-provided candidate labels",
                "dev-selected thresholds",
                "raw private rows in public artifacts",
            ],
        },
        "agent_consumer_policy": {
            "allowed_consumers": [
                {
                    "consumer_id": "evidence_selection",
                    "allowed": True,
                    "requires_stage129_validation": True,
                },
                {
                    "consumer_id": "answerability_estimation",
                    "allowed": True,
                    "requires_stage129_validation": True,
                },
                {
                    "consumer_id": "citation_validation",
                    "allowed": True,
                    "requires_stage129_validation": True,
                },
            ],
            "blocked_consumers": [
                {
                    "consumer_id": "direct_answer_context_all_400",
                    "blocked": True,
                    "reason": (
                        "The extra 200 candidates are validated as recall "
                        "candidates, not as final answer context."
                    ),
                },
                {
                    "consumer_id": "runtime_default_retrieval_route",
                    "blocked": True,
                    "reason": (
                        "A dedicated Stage129 train/dev integration validation "
                        "is still required."
                    ),
                },
                {
                    "consumer_id": "fallback_strategy_route",
                    "blocked": True,
                    "reason": "Fallback strategies remain disabled.",
                },
            ],
        },
        "validation_plan": {
            "next_stage": "Stage129",
            "action": _NEXT_DIRECTION,
            "selection_split": "train",
            "selection_mode": "train_grouped_cross_validation_agent_integration_validation",
            "minimum_train_folds": _MINIMUM_TRAIN_FOLDS,
            "validation_split": "dev",
            "dev_mode": "single_pass_report_only_no_retuning",
            "primary_checks": [
                "Stage116 ranks 1-200 remain identical",
                "hit@200 loss count remains zero",
                "candidate-pool target-depth recall does not regress",
                "agent evidence selection does not reduce verified answer quality on train-CV",
                "dev remains report-only",
            ],
            "metrics_to_report": [
                "retrieval hit@200 preservation",
                "target-depth gold-document recall",
                "selected evidence count and route mix",
                "answer F1 delta",
                "gold citation preservation",
                "changed answer count",
            ],
            "test_rules": {
                "test_access_allowed": False,
                "final_test_metrics_allowed": False,
                "test_tuning_allowed": False,
            },
            "runtime_rules": {
                "default_runtime_policy": "unchanged",
                "runtime_defaultization_allowed_in_stage128": False,
                "fallback_strategies_enabled": False,
            },
        },
        "risk_controls": {
            "stage127_risks_carried_forward": stage127_summary.get("risk_review"),
            "best_dev_config_not_selected_for_protocol": _BEST_DEV_CONFIG_ID,
            "reason_best_dev_config_not_selected": (
                "Dev was report-only in Stage126 and cannot be used to retune "
                "or replace the train-selected config."
            ),
            "answer_quality_not_yet_measured_for_integration": True,
            "requires_separate_integration_validation_before_runtime_change": True,
        },
    }


def _guard_checks(
    *,
    report: Mapping[str, Any],
    stage127_summary: Mapping[str, Any],
    user_confirmed_protocol: bool,
) -> list[dict[str, Any]]:
    frozen = report["frozen_protocol"]
    retrieval_contract = frozen["agent_retrieval_contract"]
    interface = frozen["agent_candidate_interface"]
    consumer_policy = frozen["agent_consumer_policy"]
    validation = frozen["validation_plan"]
    public_safe = _public_safe_contract(report)
    return [
        _check(
            name="user_confirmed_stage128_protocol",
            passed=user_confirmed_protocol,
            observed=report["user_confirmation"]["confirmation_note"],
            expected="user confirmed Stage128 agent integration protocol",
        ),
        _check(
            name="stage127_review_completed",
            passed=stage127_summary.get("decision_status") == _SOURCE_STAGE127_STATUS,
            observed=stage127_summary.get("decision_status"),
            expected=_SOURCE_STAGE127_STATUS,
        ),
        _check(
            name="stage127_review_id_matches",
            passed=stage127_summary.get("review_id") == _SOURCE_STAGE127_REVIEW_ID,
            observed=stage127_summary.get("review_id"),
            expected=_SOURCE_STAGE127_REVIEW_ID,
        ),
        _check(
            name="stage127_recommends_stage128_protocol",
            passed=stage127_summary.get("recommended_next_direction")
            == _SOURCE_NEXT_DIRECTION,
            observed=stage127_summary.get("recommended_next_direction"),
            expected=_SOURCE_NEXT_DIRECTION,
        ),
        _check(
            name="stage127_selected_config_supported",
            passed=stage127_summary.get(
                "selected_config_supported_for_agent_protocol_design"
            )
            is True
            and stage127_summary.get("selected_config_id") == _SELECTED_CONFIG_ID
            and stage127_summary.get("selected_family_id") == _SELECTED_FAMILY_ID,
            observed={
                "supported": stage127_summary.get(
                    "selected_config_supported_for_agent_protocol_design"
                ),
                "config_id": stage127_summary.get("selected_config_id"),
                "family_id": stage127_summary.get("selected_family_id"),
            },
            expected={
                "config_id": _SELECTED_CONFIG_ID,
                "family_id": _SELECTED_FAMILY_ID,
            },
        ),
        _check(
            name="stage127_selected_config_has_positive_train_signal",
            passed=int(stage127_summary.get("selected_train_incremental_gain_count") or 0)
            > 0,
            observed=stage127_summary.get("selected_train_incremental_gain_count"),
            expected="> 0",
        ),
        _check(
            name="stage127_selected_config_has_nonnegative_dev_signal",
            passed=int(stage127_summary.get("selected_dev_incremental_gain_count") or 0)
            >= 0,
            observed=stage127_summary.get("selected_dev_incremental_gain_count"),
            expected=">= 0",
        ),
        _check(
            name="stage127_selected_config_preserves_prefix_and_hit200",
            passed=int(stage127_summary.get("selected_train_hit200_loss_count") or 0)
            == 0
            and int(stage127_summary.get("selected_dev_hit200_loss_count") or 0) == 0
            and int(stage127_summary.get("selected_train_prefix_violation_count") or 0)
            == 0
            and int(stage127_summary.get("selected_dev_prefix_violation_count") or 0)
            == 0,
            observed={
                "train_hit200_loss": stage127_summary.get(
                    "selected_train_hit200_loss_count"
                ),
                "dev_hit200_loss": stage127_summary.get(
                    "selected_dev_hit200_loss_count"
                ),
                "train_prefix_violation": stage127_summary.get(
                    "selected_train_prefix_violation_count"
                ),
                "dev_prefix_violation": stage127_summary.get(
                    "selected_dev_prefix_violation_count"
                ),
            },
            expected="all zero",
        ),
        _check(
            name="stage128_candidate_pool_contract_is_prefix_preserving",
            passed=retrieval_contract["candidate_pool_output_depth"]
            == _TARGET_POOL_DEPTH
            and retrieval_contract["candidate_pool_is_not_automatic_answer_context"]
            is True
            and retrieval_contract["rank_regions"][0]["rank_end"]
            == _BASELINE_PREFIX_DEPTH
            and retrieval_contract["rank_regions"][1]["rank_start"]
            == _APPEND_START_RANK
            and retrieval_contract["rank_regions"][1]["append_budget"]
            == _APPEND_BUDGET,
            observed=retrieval_contract,
            expected="top200 immutable prefix plus ranks 201-400 append region",
        ),
        _check(
            name="stage128_candidate_interface_uses_runtime_visible_fields",
            passed=all(
                forbidden not in interface["candidate_record_fields_to_validate"]
                for forbidden in _FORBIDDEN_PUBLIC_KEYS
            )
            and "runtime_document_key" in interface["candidate_record_fields_to_validate"],
            observed=interface["candidate_record_fields_to_validate"],
            expected="runtime-visible candidate fields only",
        ),
        _check(
            name="stage128_blocks_direct_all400_answer_context",
            passed=any(
                item["consumer_id"] == "direct_answer_context_all_400"
                and item["blocked"] is True
                for item in consumer_policy["blocked_consumers"]
            ),
            observed=consumer_policy["blocked_consumers"],
            expected="direct all-400 answer context blocked",
        ),
        _check(
            name="stage128_allows_only_validation_gated_agent_consumers",
            passed=all(
                item["allowed"] is True
                and item["requires_stage129_validation"] is True
                for item in consumer_policy["allowed_consumers"]
            ),
            observed=consumer_policy["allowed_consumers"],
            expected="allowed consumers all require Stage129 validation",
        ),
        _check(
            name="stage128_validation_plan_uses_train_cv_and_dev_report_only",
            passed=validation["selection_split"] == "train"
            and int(validation["minimum_train_folds"]) >= _MINIMUM_TRAIN_FOLDS
            and validation["validation_split"] == "dev"
            and validation["dev_mode"] == "single_pass_report_only_no_retuning",
            observed=validation,
            expected="train grouped-CV and dev report-only",
        ),
        _check(
            name="stage128_test_locked",
            passed=validation["test_rules"]["test_access_allowed"] is False
            and validation["test_rules"]["final_test_metrics_allowed"] is False
            and validation["test_rules"]["test_tuning_allowed"] is False,
            observed=validation["test_rules"],
            expected="test access and final metrics disabled",
        ),
        _check(
            name="stage128_runtime_defaults_unchanged",
            passed=validation["runtime_rules"]["default_runtime_policy"] == "unchanged"
            and validation["runtime_rules"][
                "runtime_defaultization_allowed_in_stage128"
            ]
            is False
            and validation["runtime_rules"]["fallback_strategies_enabled"] is False
            and stage127_summary.get("runtime_defaultization_allowed_now") is False
            and stage127_summary.get("default_runtime_policy") == "unchanged",
            observed={
                "stage127_runtime_defaultization_allowed_now": stage127_summary.get(
                    "runtime_defaultization_allowed_now"
                ),
                "stage127_default_runtime_policy": stage127_summary.get(
                    "default_runtime_policy"
                ),
                "stage128_runtime_rules": validation["runtime_rules"],
            },
            expected="runtime unchanged and no defaultization in Stage128",
        ),
        _check(
            name="stage128_no_fallback_strategies",
            passed=validation["runtime_rules"]["fallback_strategies_enabled"] is False
            and stage127_summary.get("fallback_strategies_enabled") is False,
            observed={
                "stage127_fallback": stage127_summary.get(
                    "fallback_strategies_enabled"
                ),
                "stage128_fallback": validation["runtime_rules"][
                    "fallback_strategies_enabled"
                ],
            },
            expected=False,
        ),
        _check(
            name="stage128_public_safe_contract_passed",
            passed=public_safe["forbidden_keys_found"] == [],
            observed=public_safe["forbidden_keys_found"],
            expected=[],
        ),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_agent_retrieval_integration_protocol_blocked",
            "failed_checks": failed_checks,
            "can_run_agent_retrieval_integration_validation_now": False,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "runtime_defaultization_allowed_now": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": "primeqa_hybrid_agent_retrieval_integration_protocol_frozen",
        "recommended_next_direction": _NEXT_DIRECTION,
        "selected_config_id": _SELECTED_CONFIG_ID,
        "selected_family_id": _SELECTED_FAMILY_ID,
        "can_run_agent_retrieval_integration_validation_now": True,
        "can_continue_train_dev_development": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "runtime_defaultization_allowed_now": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }


def _selected_value_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = report["stage127_summary"]
    return [
        BarDatum(
            label="train incremental gain",
            value=float(summary["selected_train_incremental_gain_count"]),
            value_label=str(summary["selected_train_incremental_gain_count"]),
        ),
        BarDatum(
            label="dev incremental gain",
            value=float(summary["selected_dev_incremental_gain_count"]),
            value_label=str(summary["selected_dev_incremental_gain_count"]),
        ),
        BarDatum(
            label="train hit@200 loss",
            value=float(summary["selected_train_hit200_loss_count"]),
            value_label=str(summary["selected_train_hit200_loss_count"]),
        ),
        BarDatum(
            label="dev hit@200 loss",
            value=float(summary["selected_dev_hit200_loss_count"]),
            value_label=str(summary["selected_dev_hit200_loss_count"]),
        ),
    ]


def _candidate_pool_contract_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    contract = report["frozen_protocol"]["agent_retrieval_contract"]
    return [
        BarDatum(
            label="immutable prefix depth",
            value=float(_BASELINE_PREFIX_DEPTH),
            value_label=str(_BASELINE_PREFIX_DEPTH),
        ),
        BarDatum(
            label="append budget",
            value=float(_APPEND_BUDGET),
            value_label=str(_APPEND_BUDGET),
        ),
        BarDatum(
            label="candidate pool output depth",
            value=float(contract["candidate_pool_output_depth"]),
            value_label=str(contract["candidate_pool_output_depth"]),
        ),
    ]


def _consumer_policy_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    policy = report["frozen_protocol"]["agent_consumer_policy"]
    bars = [
        BarDatum(
            label=f"allow {item['consumer_id']}",
            value=1.0 if item["allowed"] else 0.0,
            value_label=str(bool(item["allowed"])).lower(),
        )
        for item in policy["allowed_consumers"]
    ]
    bars.extend(
        BarDatum(
            label=f"block {item['consumer_id']}",
            value=0.0 if item["blocked"] else 1.0,
            value_label="blocked" if item["blocked"] else "allowed",
        )
        for item in policy["blocked_consumers"]
    )
    return bars


def _risk_review_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    risks = (
        report["frozen_protocol"]["risk_controls"]["stage127_risks_carried_forward"]
        or {}
    )
    flags = (
        "dev_gain_is_smaller_than_train_gain",
        "best_dev_config_differs_from_train_selected",
        "answer_quality_not_measured",
        "final_test_not_run",
        "runtime_default_unchanged",
    )
    return [
        BarDatum(
            label=flag,
            value=1.0 if risks.get(flag) else 0.0,
            value_label=str(bool(risks.get(flag))).lower(),
        )
        for flag in flags
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    flags = (
        "can_run_agent_retrieval_integration_validation_now",
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


def _fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


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
