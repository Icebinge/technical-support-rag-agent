from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

STAGE = "Stage 193"
CREATED_AT = "2026-07-27"
PROTOCOL_ID = "primeqa_hybrid_safety_constrained_lambdamart_protocol_v1"
NEXT_STAGE = "Stage 194"
STAGE192_SHA256 = "8f454c07b8889d7cbbb6e66f2a0ce1960c89f7197c88018334a587947133887f"
LIGHTGBM_VERSION = "4.7.0"
LIGHTGBM_WINDOWS_WHEEL_SHA256 = "f42d1e5b32b6f170e606d7c689c6165671da98d7bf37f1addec2623efc8740c9"
FEATURE_REPRESENTATIONS = ("raw_runtime", "question_relative_runtime")
POOL_SAFETY_ESTIMATORS = ("class_balanced_logistic", "histogram_gradient_boosting")
POOL_SAFETY_TARGETS = ("citation_loss", "f1_loss")
TREE_PROFILES = {
    "conservative": {
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 40,
        "reg_lambda": 2.0,
    },
    "moderate": {
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 25,
        "reg_lambda": 1.0,
    },
}
RISK_PENALTIES = (0.25, 0.5, 1.0, 2.0)
POOL_CAP = 16
OUTER_FOLD_COUNT = 5
INNER_FOLD_COUNT = 4
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "action_id",
        "answer",
        "answer_doc_id",
        "candidate_actions",
        "citation_delta",
        "document_id",
        "document_text",
        "f1_delta",
        "feature_rows",
        "gold_answer",
        "gold_document_id",
        "outcome_class",
        "predictions",
        "question_id",
        "question_key",
        "question_text",
        "strict_expected",
    }
)


@dataclass(frozen=True)
class Stage193Visualization:
    """One public-safe Stage 193 protocol visualization."""

    name: str
    path: str


def freeze_safety_constrained_lambdamart_protocol(
    *,
    stage192_report_path: Path,
    user_confirmed: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage 194 train-only LambdaMART reranking protocol."""

    started_at = time.perf_counter()
    stage192_report = _load_json_object(stage192_report_path)
    source_file = _fingerprint(stage192_report_path)
    loaded_at = time.perf_counter()
    evidence = _evidence_summary(stage192_report)
    protocol = _frozen_protocol()
    preliminary = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "protocol_id": PROTOCOL_ID,
        "protocol_scope": (
            "Aggregate-only freeze for a train-only grouped nested-CV experiment. "
            "Stage194 will retain the Stage191 cap-16 safety pool and replace the "
            "within-pool gain ranker with deterministic LightGBM LambdaMART plus an "
            "independent unsafe-risk head. Stage193 loads no split rows or documents, "
            "imports no LightGBM runtime, fits no model, evaluates no policy, opens no "
            "development or test data, adds no fallback, and changes no runtime default."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed),
            "selected_route": "A",
            "confirmation_note": confirmation_note,
        },
        "source_file": source_file,
        "official_dependency_evidence": {
            "package": "lightgbm",
            "version": LIGHTGBM_VERSION,
            "pypi_url": "https://pypi.org/project/lightgbm/4.7.0/",
            "python_api_url": (
                "https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html"
            ),
            "parameter_reference_url": (
                "https://lightgbm.readthedocs.io/en/latest/Parameters.html"
            ),
            "windows_x86_64_wheel_sha256": LIGHTGBM_WINDOWS_WHEEL_SHA256,
            "windows_x86_64_wheel_available": True,
            "dependency_installation_deferred_to_stage194": True,
        },
        "evidence_summary": evidence,
        "frozen_protocol": protocol,
        "execution_boundaries": {
            "loaded_public_reports_only": True,
            "train_rows_loaded": False,
            "development_loaded": False,
            "test_loaded": False,
            "lightgbm_imported": False,
            "dependency_installed_or_changed": False,
            "model_fit_count": 0,
            "ranking_groups_materialized": 0,
            "private_feature_rows_materialized": 0,
            "policy_evaluation_run": False,
            "replacement_policy_selected": False,
            "runtime_e2e_run": False,
            "runtime_registered_as_default": False,
            "stage178b_run": False,
            "retry_action_count": 0,
            "fallback_action_count": 0,
        },
    }
    guard_checks = _guard_checks(preliminary, stage192_report=stage192_report)
    checked_at = time.perf_counter()
    decision = _decision(guard_checks)
    report = {
        **preliminary,
        "guard_checks": guard_checks,
        "decision": decision,
        "timing_seconds": {
            "load_public_report": round(loaded_at - started_at, 6),
            "freeze_and_guard": round(checked_at - loaded_at, 6),
            "total": round(checked_at - started_at, 6),
        },
    }
    forbidden = sorted(_forbidden_keys_found(report))
    return {
        **report,
        "public_safe_contract": {
            "forbidden_public_keys": sorted(FORBIDDEN_PUBLIC_KEYS),
            "forbidden_keys_found": forbidden,
            "private_action_rows_persisted": False,
            "private_ranking_groups_persisted": False,
            "private_feature_rows_persisted": False,
            "private_predictions_persisted": False,
            "public_report_safe": not forbidden,
        },
    }


def _frozen_protocol() -> dict[str, Any]:
    pool_builder_count = len(FEATURE_REPRESENTATIONS) * len(POOL_SAFETY_ESTIMATORS)
    policy_config_count = (
        pool_builder_count * len(FEATURE_REPRESENTATIONS) * len(TREE_PROFILES) * len(RISK_PENALTIES)
    )
    pool_safety_fits = (
        len(FEATURE_REPRESENTATIONS) * len(POOL_SAFETY_ESTIMATORS) * len(POOL_SAFETY_TARGETS)
    )
    new_reranker_fits = len(FEATURE_REPRESENTATIONS) * len(TREE_PROFILES) * 2
    fits_per_partition = pool_safety_fits + new_reranker_fits
    inner_partition_count = OUTER_FOLD_COUNT * INNER_FOLD_COUNT
    outer_refit_count = OUTER_FOLD_COUNT
    common_tree_parameters = {
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "n_estimators": 300,
        "max_bin": 63,
        "min_split_gain": 0.0,
        "reg_alpha": 0.0,
        "subsample": 1.0,
        "subsample_freq": 0,
        "colsample_bytree": 1.0,
        "random_state": 193,
        "n_jobs": 8,
        "device_type": "cpu",
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    return {
        "experiment_name": "train_only_safety_constrained_lambdamart_nested_cv",
        "split_contract": {
            "selection_split": "train",
            "frozen_question_grouped_outer_folds": OUTER_FOLD_COUNT,
            "inner_folds_per_outer_fold": INNER_FOLD_COUNT,
            "development_opened": False,
            "test_opened": False,
            "all_actions_for_one_question_remain_in_one_fold": True,
            "all_query_group_sizes_sum_to_action_row_count": True,
        },
        "dependency_contract": {
            "package_requirement": f"lightgbm=={LIGHTGBM_VERSION}",
            "installation_stage": NEXT_STAGE,
            "wheel_sha256_verification_required": True,
            "import_and_version_preflight_required": True,
            "no_dependency_installation_during_protocol_freeze": True,
        },
        "action_contract": {
            "candidate_set": (
                "all unique runtime-generatable Stage181 actions including the unique "
                "original baseline action"
            ),
            "baseline_always_unioned_into_pool": True,
            "empty_pool_possible": False,
            "fallback_enabled": False,
            "deterministic_tie_break": "canonical runtime action generation order",
            "gold_outcomes_available_to_runtime": False,
        },
        "first_stage_pool": {
            "construction_reused_from_stage191": True,
            "feature_representations": list(FEATURE_REPRESENTATIONS),
            "safety_estimators": list(POOL_SAFETY_ESTIMATORS),
            "safety_targets": list(POOL_SAFETY_TARGETS),
            "joint_safety_risk": "max(p(citation_loss), p(f1_loss))",
            "ordering": [
                "ascending joint safety risk",
                "ascending p(citation_loss) + p(f1_loss)",
                "canonical runtime action order",
            ],
            "pool_cap": POOL_CAP,
            "pool_builder_count": pool_builder_count,
            "baseline_unioned_after_cap": True,
            "pool_expansion_search_removed": True,
        },
        "relevance_contract": {
            "labels": {"unsafe": 0, "safe_zero": 1, "strict_gain": 2},
            "label_gain": [0, 1, 4],
            "strict_gain_definition": (
                "citation delta >= 0 and F1 delta >= -1e-12 and at least one delta positive"
            ),
            "safe_zero_definition": ("citation delta == 0 and absolute F1 delta <= 1e-12"),
            "unsafe_definition": "citation delta < 0 or F1 delta < -1e-12",
            "question_balanced_sample_weight": "1 / action count for the question",
            "labels_used_only_for_training_and_offline_evaluation": True,
        },
        "lambdamart_contract": {
            "estimator": "lightgbm.LGBMRanker",
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [1],
            "lambdarank_truncation_level": 4,
            "lambdarank_norm": True,
            "feature_representations": list(FEATURE_REPRESENTATIONS),
            "tree_profiles": TREE_PROFILES,
            "common_parameters": common_tree_parameters,
            "training_actions": "all actions in each training question group",
            "heldout_group_labels_used_for_fit_or_early_stopping": False,
            "early_stopping_enabled": False,
            "row_or_group_sampling_enabled": False,
        },
        "unsafe_risk_contract": {
            "estimator": "lightgbm.LGBMClassifier",
            "objective": "binary",
            "metric": "binary_logloss",
            "positive_label": "unsafe",
            "feature_representations": list(FEATURE_REPRESENTATIONS),
            "tree_profiles_shared_with_lambdamart": True,
            "common_parameters": common_tree_parameters,
            "class_weight": None,
            "scale_pos_weight": 1.0,
            "question_balanced_sample_weight": "1 / action count for the question",
            "probability_calibration_required": False,
            "reason_calibration_not_required": (
                "selection uses only deterministic within-question risk rank, not an "
                "absolute probability threshold"
            ),
            "heldout_labels_used_for_fit_or_early_stopping": False,
            "early_stopping_enabled": False,
        },
        "within_pool_selection": {
            "gain_rank_fraction": (
                "zero-based rank under descending LambdaMART score divided by pool_size - 1; "
                "zero when pool_size is one"
            ),
            "unsafe_rank_fraction": (
                "zero-based rank under ascending unsafe probability divided by pool_size - 1; "
                "zero when pool_size is one"
            ),
            "combined_utility": ("1 - gain_rank_fraction - risk_penalty * unsafe_rank_fraction"),
            "risk_penalties": list(RISK_PENALTIES),
            "ordering": [
                "descending combined utility",
                "ascending unsafe probability",
                "descending LambdaMART score",
                "canonical runtime action order",
            ],
            "absolute_probability_threshold_used": False,
            "runtime_gold_filter_used": False,
            "fallback_branch_used": False,
        },
        "candidate_grid": {
            "pool_builders": pool_builder_count,
            "reranker_feature_representations": list(FEATURE_REPRESENTATIONS),
            "tree_profiles": list(TREE_PROFILES),
            "risk_penalties": list(RISK_PENALTIES),
            "policy_config_count": policy_config_count,
            "models_shared_across_pool_builders_and_risk_penalties": True,
        },
        "cross_validation": {
            "outer_fold_count": OUTER_FOLD_COUNT,
            "inner_fold_count": INNER_FOLD_COUNT,
            "inner_partition_count": inner_partition_count,
            "outer_refit_count": outer_refit_count,
            "pool_safety_model_fits_per_partition": pool_safety_fits,
            "new_reranker_model_fits_per_partition": new_reranker_fits,
            "model_fits_per_partition": fits_per_partition,
            "maximum_model_fit_count": (
                (inner_partition_count + outer_refit_count) * fits_per_partition
            ),
            "inner_selection_uses_only_inner_oof_predictions": True,
            "outer_fold_used_once_after_inner_selection": True,
            "no_inner_eligible_config_behavior": (
                "record no-eligible and do not evaluate a weaker outer configuration"
            ),
            "no_retry": True,
            "no_fallback": True,
        },
        "inner_selection": _inner_selection_contract(),
        "advancement_gates": _advancement_gates(),
        "resource_contract": {
            "scipy_sparse_features_preserved": True,
            "one_partition_and_representation_materialized_at_a_time": True,
            "released_models_and_matrices_collected_between_partitions": True,
            "minimum_preflight_system_available_memory_gib": 6.0,
            "cpu_device_frozen_for_reproducibility": True,
            "physical_cpu_threads": 8,
            "gpu_required": False,
            "resource_statistics_required": [
                "wall and CPU time",
                "peak working set and private usage",
                "minimum system available memory",
                "LightGBM fit count and tree count",
            ],
            "insufficient_memory_behavior": (
                "do not start; request resource clearance instead of reducing protocol"
            ),
            "process_monitoring": (
                "one PowerShell Wait-Process call for the formal PID until natural exit"
            ),
        },
        "authorization_boundary": {
            "stage194_dependency_provisioning_authorized": True,
            "stage194_train_only_experiment_may_run_if_preflight_passes": True,
            "development_evaluation_authorized": False,
            "test_evaluation_authorized": False,
            "runtime_e2e_authorized": False,
            "full_train_policy_selection_authorized": False,
            "replacement_policy_selection_authorized": False,
            "default_runtime_activation_authorized": False,
            "stage178b_authorized": False,
        },
    }


def _inner_selection_contract() -> dict[str, Any]:
    return {
        "eligibility_constraints": [
            "aggregate citation delta >= 0",
            "aggregate mean F1 delta >= 0",
            "citation nonregression in at least 3 of 4 inner folds",
            "F1 nonregression in at least 3 of 4 inner folds",
            "changed-question count >= 10% of inner questions",
            "strict-success count >= 8% of inner questions",
            "strict-success precision >= 0.65",
            "aggregate strict-opportunity pool recall >= 0.95",
            "pool recall >= 0.90 in at least 3 of 4 inner folds",
            "aggregate conditional ranker strict capture >= 0.68",
            "conditional capture >= 0.60 in at least 3 of 4 inner folds",
            "aggregate unsafe selection rate <= 0.25",
            "unsafe selection rate <= 0.35 in at least 3 of 4 inner folds",
        ],
        "thresholds": {
            "strict_success_precision_minimum": 0.65,
            "aggregate_pool_recall_minimum": 0.95,
            "per_fold_pool_recall_minimum": 0.90,
            "folds_meeting_pool_recall": 3,
            "aggregate_conditional_capture_minimum": 0.68,
            "per_fold_conditional_capture_minimum": 0.60,
            "folds_meeting_conditional_capture": 3,
            "aggregate_unsafe_selection_rate_maximum": 0.25,
            "per_fold_unsafe_selection_rate_maximum": 0.35,
            "folds_meeting_unsafe_rate": 3,
        },
        "lexicographic_objective": [
            "maximize strict-success count",
            "maximize conditional ranker strict capture",
            "maximize strict-success precision",
            "minimize unsafe selection count",
            "minimize F1-regression action count",
            "minimize citation-loss action count",
            "maximize gold-citation delta",
            "maximize mean F1 delta",
            "maximize repaired Stage182 regressions",
            "deterministic candidate name",
        ],
        "weaker_ineligible_candidate_substitution": False,
        "gold_used_for_offline_eligibility_only": True,
    }


def _advancement_gates() -> list[dict[str, Any]]:
    return [
        _threshold("outer_folds_with_inner_eligible_config", ">=", 5, "count"),
        _threshold("gold_citation_delta", ">=", 5, "count"),
        _threshold("mean_f1_delta", ">=", 0.005249, "rate"),
        _threshold("citation_bootstrap_ci95_lower", ">=", 0.0, "count"),
        _threshold("f1_bootstrap_ci95_lower", ">=", 0.0, "rate"),
        _threshold("citation_nonregressing_outer_folds", ">=", 4, "count"),
        _threshold("f1_nonregressing_outer_folds", ">=", 4, "count"),
        _threshold("strict_success_count", ">=", 37, "count"),
        _threshold("strict_success_precision", ">=", 0.65, "rate"),
        _threshold("citation_loss_action_count", "<=", 4, "count"),
        _threshold("f1_regression_action_count", "<=", 27, "count"),
        _threshold("stage182_regression_repair_rate", ">=", 0.50, "rate"),
        _threshold("new_f1_regression_rate", "<=", 0.02, "rate"),
        _threshold("changed_question_count", ">=", 37, "count"),
        _threshold("strict_opportunity_pool_recall", ">=", 0.95, "rate"),
        _threshold("conditional_ranker_strict_capture", ">=", 0.68, "rate"),
        _threshold("unsafe_selection_rate", "<=", 0.25, "rate"),
    ]


def _evidence_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    attribution = report["rank_capped_safety_pool_failure_attribution"]
    reference = attribution["reference_trajectory"]
    partition = reference["opportunity_partition"]
    miss = reference["ranker_miss_breakdown"]
    factors = attribution["factor_aggregates"]
    return {
        "primary_bottleneck": report["decision"]["primary_bottleneck"],
        "question_context_count": reference["question_context_count"],
        "strict_opportunity_context_count": reference["strict_opportunity_context_count"],
        "pool_exclusion_context_count": partition["pool_exclusion_context_count"],
        "ranker_miss_context_count": partition["ranker_miss_context_count"],
        "strict_selected_context_count": partition["strict_selected_context_count"],
        "opportunity_partition_exact": partition["partition_exact"],
        "strict_opportunity_pool_recall": reference["strict_opportunity_pool_recall"],
        "conditional_ranker_strict_capture": reference["conditional_ranker_strict_capture"],
        "actual_strict_opportunity_capture": reference["actual_strict_opportunity_capture"],
        "baseline_change_strict_precision": reference["baseline_change_strict_precision"],
        "unsafe_selection_rate": reference["unsafe_selection_rate"],
        "ranker_miss_safe_zero_context_count": miss["safe_zero_winner_context_count"],
        "ranker_miss_unsafe_context_count": miss["unsafe_winner_context_count"],
        "pool_cap_metrics": {
            name: {
                "pool_recall": row["strict_opportunity_pool_recall"],
                "conditional_capture": row["conditional_ranker_strict_capture"],
                "unsafe_rate": row["unsafe_selection_rate"],
            }
            for name, row in factors["pool_cap"].items()
        },
        "gain_ranker_metrics": {
            name: {
                "conditional_capture": row["conditional_ranker_strict_capture"],
                "unsafe_rate": row["unsafe_selection_rate"],
            }
            for name, row in factors["gain_ranker"].items()
        },
        "feature_representation_metrics": {
            name: {
                "conditional_capture": row["conditional_ranker_strict_capture"],
                "unsafe_rate": row["unsafe_selection_rate"],
            }
            for name, row in factors["feature_representation"].items()
        },
        "private_prediction_count_consumed": attribution["execution"][
            "private_bundle_prediction_count_consumed"
        ],
        "new_attribution_model_fit_count": attribution["execution"]["new_model_fit_count"],
    }


def _guard_checks(
    report: Mapping[str, Any],
    *,
    stage192_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evidence = report["evidence_summary"]
    protocol = report["frozen_protocol"]
    boundaries = report["execution_boundaries"]
    source_decision = stage192_report["decision"]
    source_boundaries = stage192_report["execution_boundaries"]
    ranker = evidence["gain_ranker_metrics"]
    representation = evidence["feature_representation_metrics"]
    return [
        _gate("user_confirmed_route_a", report["user_confirmation"]["confirmed"] is True),
        _gate("selected_route_is_a", report["user_confirmation"]["selected_route"] == "A"),
        _gate("stage192_hash_matches", report["source_file"]["sha256"] == STAGE192_SHA256),
        _gate("stage192_report_type_matches", stage192_report.get("stage") == "Stage 192"),
        _gate(
            "stage192_status_complete",
            source_decision["status"]
            == "stage192_rank_capped_safety_pool_failure_attribution_complete",
        ),
        _gate("stage192_diagnostic_complete", source_decision["diagnostic_complete"] is True),
        _gate(
            "stage192_bottleneck_is_within_pool_ranker",
            evidence["primary_bottleneck"] == "within_pool_ranker_miss",
        ),
        _gate(
            "stage192_process_guards_passed",
            len(stage192_report["process_guards"]) == 23
            and all(row["passed"] for row in stage192_report["process_guards"]),
        ),
        _gate("question_context_count_is_1480", evidence["question_context_count"] == 1480),
        _gate(
            "strict_opportunity_count_is_1456",
            evidence["strict_opportunity_context_count"] == 1456,
        ),
        _gate("pool_exclusion_count_is_58", evidence["pool_exclusion_context_count"] == 58),
        _gate("ranker_miss_count_is_500", evidence["ranker_miss_context_count"] == 500),
        _gate("strict_selected_count_is_898", evidence["strict_selected_context_count"] == 898),
        _gate("opportunity_partition_exact", evidence["opportunity_partition_exact"] is True),
        _gate(
            "reference_pool_recall_matches", evidence["strict_opportunity_pool_recall"] == 0.960165
        ),
        _gate(
            "reference_conditional_capture_matches",
            evidence["conditional_ranker_strict_capture"] == 0.642346,
        ),
        _gate("reference_unsafe_rate_matches", evidence["unsafe_selection_rate"] == 0.352703),
        _gate(
            "ranker_miss_dominates_pool_exclusion",
            evidence["ranker_miss_context_count"] > evidence["pool_exclusion_context_count"],
        ),
        _gate(
            "ranker_miss_is_mostly_unsafe_winners",
            evidence["ranker_miss_unsafe_context_count"] == 465
            and evidence["ranker_miss_safe_zero_context_count"] == 35,
        ),
        _gate(
            "cap16_pool_recall_is_high",
            evidence["pool_cap_metrics"]["16"]["pool_recall"] == 0.986607,
        ),
        _gate("cap_all_recall_is_one", evidence["pool_cap_metrics"]["all"]["pool_recall"] == 1.0),
        _gate(
            "existing_ranker_capture_tradeoff_present",
            ranker["pairwise_pareto_logistic"]["conditional_capture"]
            > ranker["linear_listnet_top_frontier"]["conditional_capture"],
        ),
        _gate(
            "existing_ranker_unsafe_tradeoff_present",
            ranker["pairwise_pareto_logistic"]["unsafe_rate"]
            > ranker["linear_listnet_top_frontier"]["unsafe_rate"],
        ),
        _gate(
            "representation_capture_tradeoff_present",
            representation["raw_runtime"]["conditional_capture"]
            > representation["question_relative_runtime"]["conditional_capture"],
        ),
        _gate(
            "representation_unsafe_tradeoff_present",
            representation["raw_runtime"]["unsafe_rate"]
            > representation["question_relative_runtime"]["unsafe_rate"],
        ),
        _gate(
            "lightgbm_version_is_4_7_0",
            report["official_dependency_evidence"]["version"] == "4.7.0",
        ),
        _gate(
            "windows_wheel_hash_frozen",
            report["official_dependency_evidence"]["windows_x86_64_wheel_sha256"]
            == LIGHTGBM_WINDOWS_WHEEL_SHA256,
        ),
        _gate("pool_cap_is_16", protocol["first_stage_pool"]["pool_cap"] == 16),
        _gate("pool_builder_count_is_4", protocol["first_stage_pool"]["pool_builder_count"] == 4),
        _gate(
            "tree_profile_count_is_2", len(protocol["lambdamart_contract"]["tree_profiles"]) == 2
        ),
        _gate(
            "risk_penalty_count_is_4", len(protocol["within_pool_selection"]["risk_penalties"]) == 4
        ),
        _gate("policy_config_count_is_64", protocol["candidate_grid"]["policy_config_count"] == 64),
        _gate(
            "relevance_labels_are_ordinal",
            protocol["relevance_contract"]["labels"]
            == {"unsafe": 0, "safe_zero": 1, "strict_gain": 2},
        ),
        _gate("label_gain_is_frozen", protocol["relevance_contract"]["label_gain"] == [0, 1, 4]),
        _gate(
            "query_grouping_is_frozen",
            protocol["split_contract"]["all_query_group_sizes_sum_to_action_row_count"] is True,
        ),
        _gate(
            "cpu_determinism_enabled",
            protocol["lambdamart_contract"]["common_parameters"]["deterministic"] is True,
        ),
        _gate(
            "force_col_wise_enabled",
            protocol["lambdamart_contract"]["common_parameters"]["force_col_wise"] is True,
        ),
        _gate(
            "heldout_early_stopping_forbidden",
            protocol["lambdamart_contract"]["early_stopping_enabled"] is False,
        ),
        _gate(
            "maximum_fit_count_is_400",
            protocol["cross_validation"]["maximum_model_fit_count"] == 400,
        ),
        _gate(
            "inner_pool_recall_gate_is_0_95",
            protocol["inner_selection"]["thresholds"]["aggregate_pool_recall_minimum"] == 0.95,
        ),
        _gate(
            "inner_capture_gate_is_0_68",
            protocol["inner_selection"]["thresholds"]["aggregate_conditional_capture_minimum"]
            == 0.68,
        ),
        _gate(
            "inner_unsafe_gate_is_0_25",
            protocol["inner_selection"]["thresholds"]["aggregate_unsafe_selection_rate_maximum"]
            == 0.25,
        ),
        _gate("advancement_gate_count_is_17", len(protocol["advancement_gates"]) == 17),
        _gate("source_development_was_closed", source_boundaries["development_loaded"] is False),
        _gate("source_test_was_closed", source_boundaries["test_loaded"] is False),
        _gate("public_reports_only", boundaries["loaded_public_reports_only"] is True),
        _gate("train_rows_not_loaded", boundaries["train_rows_loaded"] is False),
        _gate("development_closed", boundaries["development_loaded"] is False),
        _gate("test_closed", boundaries["test_loaded"] is False),
        _gate("lightgbm_not_imported", boundaries["lightgbm_imported"] is False),
        _gate("dependency_not_changed", boundaries["dependency_installed_or_changed"] is False),
        _gate("no_model_fit", boundaries["model_fit_count"] == 0),
        _gate("no_policy_evaluation", boundaries["policy_evaluation_run"] is False),
        _gate("no_runtime_e2e", boundaries["runtime_e2e_run"] is False),
        _gate("default_runtime_unchanged", boundaries["runtime_registered_as_default"] is False),
        _gate("stage178b_not_run", boundaries["stage178b_run"] is False),
        _gate("no_retry", boundaries["retry_action_count"] == 0),
        _gate("no_fallback", boundaries["fallback_action_count"] == 0),
        _gate("preliminary_report_public_safe", not _forbidden_keys_found(report)),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = all(row["passed"] for row in guard_checks)
    return {
        "status": (
            "stage193_safety_constrained_lambdamart_protocol_frozen"
            if valid
            else "stage193_safety_constrained_lambdamart_protocol_invalid"
        ),
        "protocol_valid": valid,
        "stage194_dependency_provisioning_authorized": valid,
        "stage194_train_only_experiment_authorized": valid,
        "development_opened": False,
        "test_opened": False,
        "runtime_e2e_authorized": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "default_runtime_activation": False,
    }


def write_stage193_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage193Visualization, ...]:
    """Write and XML-validate aggregate Stage 193 protocol charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = report["evidence_summary"]
    protocol = report["frozen_protocol"]
    partition = (
        evidence["pool_exclusion_context_count"],
        evidence["ranker_miss_context_count"],
        evidence["strict_selected_context_count"],
    )
    charts = {
        "stage193_source_partition.svg": _chart(
            "Stage 193 source opportunity partition",
            tuple(
                _count_bar(name, value)
                for name, value in zip(
                    ("pool exclusion", "within-pool ranker miss", "strict selected"),
                    partition,
                    strict=True,
                )
            ),
            "question-context count",
        ),
        "stage193_source_rates.svg": _chart(
            "Stage 193 source diagnostic rates",
            tuple(
                BarDatum(name, value, f"{value:.3f}")
                for name, value in (
                    ("pool recall", evidence["strict_opportunity_pool_recall"]),
                    ("conditional capture", evidence["conditional_ranker_strict_capture"]),
                    ("strict precision", evidence["baseline_change_strict_precision"]),
                    ("unsafe selection rate", evidence["unsafe_selection_rate"]),
                )
            ),
            "rate",
        ),
        "stage193_ranker_tradeoff.svg": _chart(
            "Stage 192 ranker capture and unsafe tradeoff",
            tuple(
                BarDatum(
                    f"{name} capture",
                    row["conditional_capture"],
                    f"{row['conditional_capture']:.3f}",
                )
                for name, row in evidence["gain_ranker_metrics"].items()
            )
            + tuple(
                BarDatum(f"{name} unsafe", row["unsafe_rate"], f"{row['unsafe_rate']:.3f}")
                for name, row in evidence["gain_ranker_metrics"].items()
            ),
            "rate",
            margin_left=620,
        ),
        "stage193_candidate_grid.svg": _chart(
            "Stage 194 frozen candidate grid",
            tuple(
                _count_bar(name, value)
                for name, value in (
                    ("pool builders", protocol["candidate_grid"]["pool_builders"]),
                    ("reranker representations", len(FEATURE_REPRESENTATIONS)),
                    ("tree profiles", len(TREE_PROFILES)),
                    ("risk penalties", len(RISK_PENALTIES)),
                    ("policy configurations", protocol["candidate_grid"]["policy_config_count"]),
                )
            ),
            "count",
        ),
        "stage193_fit_budget.svg": _chart(
            "Stage 194 frozen fit budget",
            tuple(
                _count_bar(name, value)
                for name, value in (
                    ("inner partitions", protocol["cross_validation"]["inner_partition_count"]),
                    ("outer refits", protocol["cross_validation"]["outer_refit_count"]),
                    (
                        "fits per partition",
                        protocol["cross_validation"]["model_fits_per_partition"],
                    ),
                    ("maximum model fits", protocol["cross_validation"]["maximum_model_fit_count"]),
                )
            ),
            "count",
        ),
        "stage193_risk_penalties.svg": _chart(
            "Stage 194 frozen unsafe-risk penalties",
            tuple(BarDatum(str(value), value, f"{value:.2f}") for value in RISK_PENALTIES),
            "risk penalty",
        ),
        "stage193_inner_thresholds.svg": _chart(
            "Stage 194 frozen inner eligibility thresholds",
            (
                BarDatum("pool recall minimum", 0.95, "0.95"),
                BarDatum("conditional capture minimum", 0.68, "0.68"),
                BarDatum("strict precision minimum", 0.65, "0.65"),
                BarDatum("unsafe rate maximum", 0.25, "0.25"),
            ),
            "rate",
            margin_left=430,
        ),
        "stage193_decision_flags.svg": _chart(
            "Stage 193 protocol decision flags",
            tuple(
                BarDatum(name, float(value), "true" if value else "false")
                for name, value in report["decision"].items()
                if isinstance(value, bool)
            ),
            "1 means true",
            margin_left=800,
        ),
        "stage193_guard_checks.svg": _chart(
            "Stage 193 protocol guard checks",
            tuple(
                BarDatum(row["name"], float(row["passed"]), "pass" if row["passed"] else "fail")
                for row in report["guard_checks"]
            ),
            "1 means passed",
            margin_left=920,
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage193Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _threshold(name: str, operator: str, threshold: int | float, unit: str) -> dict[str, Any]:
    return {"name": name, "operator": operator, "threshold": threshold, "unit": unit}


def _count_bar(name: str, value: int) -> BarDatum:
    return BarDatum(name, value, str(value))


def _chart(
    title: str,
    bars: Sequence[BarDatum],
    x_label: str,
    *,
    margin_left: int = 350,
) -> str:
    return render_horizontal_bar_chart_svg(
        title,
        bars,
        x_label,
        width=1680,
        margin_left=margin_left,
        margin_right=260,
    )


def _fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "byte_size": resolved.stat().st_size,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _load_json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def _forbidden_keys_found(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in FORBIDDEN_PUBLIC_KEYS:
                found.add(str(key))
            found.update(_forbidden_keys_found(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_forbidden_keys_found(child))
    return found


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}
