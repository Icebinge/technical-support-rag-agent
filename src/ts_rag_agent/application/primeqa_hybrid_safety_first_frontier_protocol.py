from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

STAGE = "Stage 195"
CREATED_AT = "2026-07-27"
PROTOCOL_ID = "primeqa_hybrid_safety_first_frontier_protocol_v1"
NEXT_STAGE = "Stage 196"
STAGE194_SHA256 = "c1208348e79fd404e7b360a49a3f4d4e9663a3e9bd61c3ef99cf3f9ac60ece57"
LIGHTGBM_VERSION = "4.7.0"
NARWHALS_VERSION = "2.24.0"
LIGHTGBM_WHEEL_SHA256 = "f42d1e5b32b6f170e606d7c689c6165671da98d7bf37f1addec2623efc8740c9"
NARWHALS_WHEEL_SHA256 = "42fdedf44e5b2ca7505630d45b4ac3058f38d8485cba9fe1652ca23152df7489"
REPRESENTATIONS = ("raw_runtime", "question_relative_runtime")
TREE_PROFILES = ("conservative", "moderate")
POOL_SAFETY_ESTIMATORS = ("class_balanced_logistic", "histogram_gradient_boosting")
SCALE_POS_WEIGHTS = (1.0, 2.0, 4.0)
SAFEST_PREFIX_SIZES = (2, 4, 8, 12, 16)
POOL_CAP = 16
OUTER_FOLDS = 5
INNER_FOLDS = 4
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
class Stage195Visualization:
    name: str
    path: str


def freeze_safety_first_frontier_protocol(
    *,
    stage194_report_path: Path,
    user_confirmed: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage 196 train-only safety-first frontier experiment."""

    started_at = time.perf_counter()
    source = _load_json(stage194_report_path)
    source_file = _fingerprint(stage194_report_path)
    loaded_at = time.perf_counter()
    evidence = _evidence_summary(source)
    protocol = _frozen_protocol()
    preliminary = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "protocol_id": PROTOCOL_ID,
        "protocol_scope": (
            "Aggregate-only freeze for a train-only safety-first constrained-selection "
            "experiment. Stage195 reads only the public Stage194 report, loads no split "
            "rows or documents, imports no LightGBM runtime, fits no model, evaluates no "
            "policy, opens no development or test data, adds no fallback, and changes no "
            "runtime default."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed),
            "selected_route": "recommended_stage195_safety_first_frontier",
            "confirmation_note": confirmation_note,
        },
        "source_file": source_file,
        "official_dependency_evidence": {
            "parameter_reference_url": (
                "https://lightgbm.readthedocs.io/en/latest/Parameters.html"
            ),
            "classifier_api_url": (
                "https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html"
            ),
            "scale_pos_weight_is_positive_class_weight": True,
            "scale_pos_weight_and_is_unbalance_mutually_exclusive": True,
            "weighted_binary_probabilities_may_be_poor": True,
            "protocol_uses_risk_rank_not_absolute_probability": True,
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
            "private_feature_rows_materialized": 0,
            "private_predictions_materialized": 0,
            "policy_evaluation_run": False,
            "full_train_policy_selected": False,
            "replacement_policy_selected": False,
            "runtime_e2e_run": False,
            "runtime_registered_as_default": False,
            "stage178b_run": False,
            "retry_action_count": 0,
            "fallback_action_count": 0,
        },
    }
    guards = _guard_checks(preliminary, source)
    checked_at = time.perf_counter()
    report = {
        **preliminary,
        "guard_checks": guards,
        "decision": _decision(guards),
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
            "private_feature_rows_persisted": False,
            "private_predictions_persisted": False,
            "public_report_safe": not forbidden,
        },
    }


def _frozen_protocol() -> dict[str, Any]:
    pool_builder_count = len(REPRESENTATIONS) * len(POOL_SAFETY_ESTIMATORS)
    gain_model_count = len(REPRESENTATIONS) * len(TREE_PROFILES)
    risk_model_count = gain_model_count * len(SCALE_POS_WEIGHTS)
    policy_count = (
        pool_builder_count * gain_model_count * risk_model_count * len(SAFEST_PREFIX_SIZES)
    )
    pool_fits = len(REPRESENTATIONS) * len(POOL_SAFETY_ESTIMATORS) * 2
    fits_per_partition = pool_fits + gain_model_count + risk_model_count
    partition_count = OUTER_FOLDS * INNER_FOLDS + OUTER_FOLDS
    return {
        "experiment_name": "train_only_cost_sensitive_safety_first_frontier_nested_cv",
        "split_contract": {
            "selection_split": "train",
            "outer_fold_count": OUTER_FOLDS,
            "inner_fold_count": INNER_FOLDS,
            "all_actions_for_one_question_remain_in_one_fold": True,
            "development_opened": False,
            "test_opened": False,
        },
        "dependency_contract": {
            "lightgbm_requirement": f"lightgbm=={LIGHTGBM_VERSION}",
            "narwhals_requirement": f"narwhals=={NARWHALS_VERSION}",
            "lightgbm_wheel_sha256": LIGHTGBM_WHEEL_SHA256,
            "narwhals_wheel_sha256": NARWHALS_WHEEL_SHA256,
            "stage196_version_hash_import_and_pip_check_required": True,
            "stage195_dependency_change": False,
        },
        "first_stage_pool": {
            "construction_reused_from_stage194": True,
            "feature_representations": list(REPRESENTATIONS),
            "safety_estimators": list(POOL_SAFETY_ESTIMATORS),
            "safety_targets": ["citation_loss", "f1_loss"],
            "joint_safety_risk": "max(p(citation_loss), p(f1_loss))",
            "pool_cap": POOL_CAP,
            "pool_builder_count": pool_builder_count,
            "baseline_unioned_after_cap": True,
            "pool_expansion_search_enabled": False,
        },
        "gain_ranker": {
            "construction_reused_from_stage194": True,
            "estimator": "lightgbm.LGBMRanker",
            "objective": "lambdarank",
            "relevance_labels": {"unsafe": 0, "safe_zero": 1, "strict_gain": 2},
            "label_gain": [0, 1, 4],
            "feature_representations": list(REPRESENTATIONS),
            "tree_profiles": list(TREE_PROFILES),
            "gain_model_count_per_partition": gain_model_count,
            "heldout_labels_used_for_fit_or_early_stopping": False,
        },
        "cost_sensitive_unsafe_head": {
            "estimator": "lightgbm.LGBMClassifier",
            "objective": "binary",
            "positive_label": "unsafe",
            "feature_representations": list(REPRESENTATIONS),
            "tree_profiles": list(TREE_PROFILES),
            "scale_pos_weights": list(SCALE_POS_WEIGHTS),
            "is_unbalance": False,
            "class_weight": None,
            "risk_model_count_per_partition": risk_model_count,
            "probability_calibration_required": False,
            "reason_calibration_not_required": (
                "weighted binary probabilities may be poor and Stage196 uses only "
                "deterministic within-question risk order, never an absolute threshold"
            ),
            "heldout_labels_used_for_fit_or_early_stopping": False,
        },
        "safety_first_frontier": {
            "source_pool": "complete Stage194 cap-16 safety pool including baseline union",
            "risk_order": [
                "ascending unsafe-head score",
                "canonical runtime action order",
            ],
            "safest_prefix_sizes": list(SAFEST_PREFIX_SIZES),
            "baseline_unioned_after_prefix": True,
            "winner_order": [
                "descending LambdaMART gain score",
                "ascending unsafe-head score",
                "canonical runtime action order",
            ],
            "gain_risk_utility_blend_used": False,
            "absolute_probability_threshold_used": False,
            "runtime_gold_filter_used": False,
            "retry_used": False,
            "fallback_used": False,
            "frontier_diagnostics_required": [
                "strict-opportunity frontier recall",
                "unsafe action retention rate",
                "mean frontier size",
                "baseline inclusion rate",
            ],
        },
        "factor_decoupling": {
            "gain_and_risk_feature_representations_independent": True,
            "gain_and_risk_tree_profiles_independent": True,
            "reason": (
                "Stage194 tied gain and risk representation/profile despite measuring "
                "different targets; Stage196 tests their Cartesian product without "
                "additional model fits"
            ),
        },
        "candidate_grid": {
            "pool_builders": pool_builder_count,
            "gain_models": gain_model_count,
            "risk_models": risk_model_count,
            "safest_prefix_sizes": list(SAFEST_PREFIX_SIZES),
            "policy_config_count": policy_count,
            "models_shared_across_policy_configs": True,
        },
        "cross_validation": {
            "outer_fold_count": OUTER_FOLDS,
            "inner_fold_count": INNER_FOLDS,
            "inner_partition_count": OUTER_FOLDS * INNER_FOLDS,
            "maximum_outer_refit_count": OUTER_FOLDS,
            "pool_safety_fits_per_partition": pool_fits,
            "gain_ranker_fits_per_partition": gain_model_count,
            "unsafe_head_fits_per_partition": risk_model_count,
            "model_fits_per_partition": fits_per_partition,
            "maximum_model_fit_count": partition_count * fits_per_partition,
            "maximum_lightgbm_tree_count": (
                partition_count * (gain_model_count + risk_model_count) * 300
            ),
            "no_inner_eligible_config_behavior": (
                "record no-eligible and do not evaluate a weaker outer configuration"
            ),
            "no_retry": True,
            "no_fallback": True,
        },
        "inner_selection": _inner_selection(),
        "advancement_gates": _advancement_gates(),
        "resource_contract": {
            "minimum_preflight_system_available_memory_gib": 4.0,
            "threshold_revision_basis": (
                "user rejected the Stage193 6 GiB gate; Stage194 completed the full grid "
                "without OOM from 5.595 GiB preflight and observed 3.756 GiB minimum free"
            ),
            "cpu_device": True,
            "physical_cpu_threads": 8,
            "one_representation_and_one_weighted_risk_model_materialized_at_a_time": True,
            "event_driven_resource_statistics_required": True,
            "process_wait_contract": (
                "one PowerShell Wait-Process call for the formal PID until natural exit"
            ),
            "insufficient_memory_behavior": (
                "do not start; request resource clearance instead of reducing the grid"
            ),
        },
        "authorization_boundary": {
            "stage196_train_only_experiment_authorized": True,
            "development_evaluation_authorized": False,
            "test_evaluation_authorized": False,
            "full_train_policy_selection_authorized": False,
            "runtime_e2e_authorized": False,
            "replacement_policy_selection_authorized": False,
            "default_runtime_activation_authorized": False,
            "stage178b_authorized": False,
        },
    }


def _inner_selection() -> dict[str, Any]:
    return {
        "eligibility_constraints": [
            "aggregate citation delta >= 0",
            "aggregate mean F1 delta >= 0",
            "citation nonregression in at least 3 of 4 inner folds",
            "F1 nonregression in at least 3 of 4 inner folds",
            "changed-question count >= 10% of inner questions",
            "strict-success count >= 8% of inner questions",
            "strict-success precision >= 0.65",
            "aggregate first-stage pool recall >= 0.95",
            "first-stage pool recall >= 0.90 in at least 3 of 4 folds",
            "aggregate conditional strict capture >= 0.68",
            "conditional strict capture >= 0.60 in at least 3 of 4 folds",
            "aggregate unsafe selection rate <= 0.25",
            "unsafe selection rate <= 0.35 in at least 3 of 4 folds",
        ],
        "thresholds_unchanged_from_stage194": True,
        "frontier_diagnostics_are_not_weaker_substitute_gates": True,
        "lexicographic_objective": [
            "maximize strict-success count",
            "maximize conditional strict capture",
            "maximize strict-success precision",
            "minimize unsafe selection count",
            "maximize strict-opportunity frontier recall",
            "minimize F1-regression action count",
            "minimize citation-loss action count",
            "maximize gold-citation delta",
            "maximize mean F1 delta",
            "maximize repaired Stage182 regressions",
            "deterministic candidate name",
        ],
        "weaker_ineligible_candidate_substitution": False,
    }


def _advancement_gates() -> list[dict[str, Any]]:
    rows = (
        ("outer_folds_with_inner_eligible_config", ">=", 5, "count"),
        ("gold_citation_delta", ">=", 5, "count"),
        ("mean_f1_delta", ">=", 0.005249, "rate"),
        ("citation_bootstrap_ci95_lower", ">=", 0.0, "count"),
        ("f1_bootstrap_ci95_lower", ">=", 0.0, "rate"),
        ("citation_nonregressing_outer_folds", ">=", 4, "count"),
        ("f1_nonregressing_outer_folds", ">=", 4, "count"),
        ("strict_success_count", ">=", 37, "count"),
        ("strict_success_precision", ">=", 0.65, "rate"),
        ("citation_loss_action_count", "<=", 4, "count"),
        ("f1_regression_action_count", "<=", 27, "count"),
        ("stage182_regression_repair_rate", ">=", 0.50, "rate"),
        ("new_f1_regression_rate", "<=", 0.02, "rate"),
        ("changed_question_count", ">=", 37, "count"),
        ("strict_opportunity_pool_recall", ">=", 0.95, "rate"),
        ("conditional_ranker_strict_capture", ">=", 0.68, "rate"),
        ("unsafe_selection_rate", "<=", 0.25, "rate"),
    )
    return [
        {"metric": metric, "operator": operator, "threshold": threshold, "unit": unit}
        for metric, operator, threshold, unit in rows
    ]


def _evidence_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    cv = report["safety_constrained_lambdamart_nested_cv"]
    top = []
    for fold_id, fold in cv["outer_folds"].items():
        candidate = fold["top_inner_candidates"][0]
        evaluation = candidate["evaluation"]
        diagnostics = candidate["diagnostics"]
        top.append(
            {
                "outer_fold_id": fold_id,
                "eligible_config_count": fold["eligible_config_count"],
                "strict_opportunity_pool_recall": diagnostics["strict_opportunity_pool_recall"],
                "conditional_ranker_strict_capture": diagnostics[
                    "conditional_ranker_strict_capture"
                ],
                "strict_success_precision": evaluation["strict_success_precision"],
                "unsafe_selection_rate": diagnostics["unsafe_selection_rate"],
            }
        )
    resources = report["resource_consumption"]
    return {
        "source_status": report["decision"]["status"],
        "source_experiment_valid": report["decision"]["experiment_valid"],
        "source_candidate_family_accepted": report["decision"]["candidate_family_accepted"],
        "outer_fold_count": len(cv["outer_folds"]),
        "outer_folds_with_inner_eligible_config": sum(
            row["eligible_config_count"] > 0 for row in cv["outer_folds"].values()
        ),
        "top_inner_candidates": top,
        "top_pool_recall_range": _range(row["strict_opportunity_pool_recall"] for row in top),
        "top_conditional_capture_range": _range(
            row["conditional_ranker_strict_capture"] for row in top
        ),
        "top_strict_precision_range": _range(row["strict_success_precision"] for row in top),
        "top_unsafe_rate_range": _range(row["unsafe_selection_rate"] for row in top),
        "closest_boundary_context": max(
            top,
            key=lambda row: (
                row["conditional_ranker_strict_capture"],
                row["strict_success_precision"],
                -row["unsafe_selection_rate"],
            ),
        ),
        "source_execution": {
            "model_fit_count": cv["execution"]["model_fit_count"],
            "lightgbm_tree_count": cv["execution"]["tree_count"],
            "private_prediction_count": cv["execution"]["private_prediction_count"],
        },
        "source_resources_gib": {
            "peak_working_set": _gib(resources["process_peak_working_set_bytes"]),
            "peak_private_usage": _gib(resources["process_peak_private_usage_bytes"]),
            "minimum_system_free": _gib(resources["minimum_system_available_memory_bytes"]),
        },
        "source_process_guard_count": len(report["process_guards"]),
        "source_process_guards_passed": sum(row["passed"] for row in report["process_guards"]),
    }


def _guard_checks(
    preliminary: Mapping[str, Any], source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    protocol = preliminary["frozen_protocol"]
    evidence = preliminary["evidence_summary"]
    boundaries = preliminary["execution_boundaries"]
    checks = (
        ("user_confirmed", preliminary["user_confirmation"]["confirmed"] is True),
        ("source_sha256_matches", preliminary["source_file"]["sha256"] == STAGE194_SHA256),
        ("source_is_stage194", source.get("stage") == "Stage 194"),
        ("source_experiment_valid", evidence["source_experiment_valid"] is True),
        ("source_candidate_family_rejected", evidence["source_candidate_family_accepted"] is False),
        (
            "source_status_is_insufficient",
            evidence["source_status"] == "stage194_safety_constrained_lambdamart_insufficient",
        ),
        ("source_all_five_outer_contexts_present", evidence["outer_fold_count"] == 5),
        (
            "source_no_inner_eligible_context",
            evidence["outer_folds_with_inner_eligible_config"] == 0,
        ),
        (
            "source_all_process_guards_passed",
            evidence["source_process_guard_count"] == 33
            and evidence["source_process_guards_passed"] == 33,
        ),
        (
            "source_development_closed",
            source["execution_boundaries"]["development_loaded"] is False,
        ),
        ("source_test_closed", source["execution_boundaries"]["test_loaded"] is False),
        ("pool_recall_bottleneck_cleared", evidence["top_pool_recall_range"]["minimum"] >= 0.95),
        ("unsafe_rate_still_above_gate", evidence["top_unsafe_rate_range"]["minimum"] > 0.25),
        (
            "conditional_capture_near_gate",
            evidence["top_conditional_capture_range"]["maximum"] >= 0.68,
        ),
        ("strict_precision_near_gate", evidence["top_strict_precision_range"]["maximum"] >= 0.65),
        ("pool_cap_is_16", protocol["first_stage_pool"]["pool_cap"] == 16),
        ("pool_builder_count_is_4", protocol["first_stage_pool"]["pool_builder_count"] == 4),
        (
            "pool_expansion_disabled",
            protocol["first_stage_pool"]["pool_expansion_search_enabled"] is False,
        ),
        ("gain_model_count_is_4", protocol["gain_ranker"]["gain_model_count_per_partition"] == 4),
        (
            "risk_weights_are_frozen",
            protocol["cost_sensitive_unsafe_head"]["scale_pos_weights"] == [1.0, 2.0, 4.0],
        ),
        (
            "risk_model_count_is_12",
            protocol["cost_sensitive_unsafe_head"]["risk_model_count_per_partition"] == 12,
        ),
        ("is_unbalance_disabled", protocol["cost_sensitive_unsafe_head"]["is_unbalance"] is False),
        ("class_weight_is_none", protocol["cost_sensitive_unsafe_head"]["class_weight"] is None),
        (
            "absolute_probability_threshold_disabled",
            protocol["safety_first_frontier"]["absolute_probability_threshold_used"] is False,
        ),
        (
            "prefix_sizes_are_frozen",
            protocol["safety_first_frontier"]["safest_prefix_sizes"] == [2, 4, 8, 12, 16],
        ),
        (
            "baseline_unioned_after_prefix",
            protocol["safety_first_frontier"]["baseline_unioned_after_prefix"] is True,
        ),
        (
            "utility_blend_removed",
            protocol["safety_first_frontier"]["gain_risk_utility_blend_used"] is False,
        ),
        (
            "gain_risk_representations_decoupled",
            protocol["factor_decoupling"]["gain_and_risk_feature_representations_independent"]
            is True,
        ),
        (
            "gain_risk_profiles_decoupled",
            protocol["factor_decoupling"]["gain_and_risk_tree_profiles_independent"] is True,
        ),
        ("policy_grid_is_960", protocol["candidate_grid"]["policy_config_count"] == 960),
        (
            "models_shared",
            protocol["candidate_grid"]["models_shared_across_policy_configs"] is True,
        ),
        (
            "fits_per_partition_is_24",
            protocol["cross_validation"]["model_fits_per_partition"] == 24,
        ),
        (
            "maximum_fit_count_is_600",
            protocol["cross_validation"]["maximum_model_fit_count"] == 600,
        ),
        (
            "maximum_tree_count_is_120000",
            protocol["cross_validation"]["maximum_lightgbm_tree_count"] == 120000,
        ),
        (
            "thresholds_unchanged",
            protocol["inner_selection"]["thresholds_unchanged_from_stage194"] is True,
        ),
        (
            "no_weaker_substitution",
            protocol["inner_selection"]["weaker_ineligible_candidate_substitution"] is False,
        ),
        ("advancement_gate_count_is_17", len(protocol["advancement_gates"]) == 17),
        (
            "memory_threshold_revised_to_4_gib",
            protocol["resource_contract"]["minimum_preflight_system_available_memory_gib"] == 4.0,
        ),
        ("cpu_threads_are_8", protocol["resource_contract"]["physical_cpu_threads"] == 8),
        (
            "stage196_authorized",
            protocol["authorization_boundary"]["stage196_train_only_experiment_authorized"] is True,
        ),
        (
            "development_not_authorized",
            protocol["authorization_boundary"]["development_evaluation_authorized"] is False,
        ),
        (
            "test_not_authorized",
            protocol["authorization_boundary"]["test_evaluation_authorized"] is False,
        ),
        (
            "runtime_e2e_not_authorized",
            protocol["authorization_boundary"]["runtime_e2e_authorized"] is False,
        ),
        (
            "full_train_not_authorized",
            protocol["authorization_boundary"]["full_train_policy_selection_authorized"] is False,
        ),
        (
            "replacement_not_authorized",
            protocol["authorization_boundary"]["replacement_policy_selection_authorized"] is False,
        ),
        (
            "default_not_authorized",
            protocol["authorization_boundary"]["default_runtime_activation_authorized"] is False,
        ),
        (
            "stage178b_not_authorized",
            protocol["authorization_boundary"]["stage178b_authorized"] is False,
        ),
        ("public_only", boundaries["loaded_public_reports_only"] is True),
        ("no_train_rows", boundaries["train_rows_loaded"] is False),
        ("no_dev_rows", boundaries["development_loaded"] is False),
        ("no_test_rows", boundaries["test_loaded"] is False),
        ("no_lightgbm_import", boundaries["lightgbm_imported"] is False),
        ("no_dependency_change", boundaries["dependency_installed_or_changed"] is False),
        ("zero_model_fits", boundaries["model_fit_count"] == 0),
        ("zero_private_predictions", boundaries["private_predictions_materialized"] == 0),
        ("no_policy_evaluation", boundaries["policy_evaluation_run"] is False),
        ("no_retry", boundaries["retry_action_count"] == 0),
        ("no_fallback", boundaries["fallback_action_count"] == 0),
    )
    return [_gate(name, passed) for name, passed in checks]


def _decision(guards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = bool(guards) and all(row["passed"] for row in guards)
    return {
        "status": (
            "stage195_safety_first_frontier_protocol_frozen"
            if valid
            else "stage195_safety_first_frontier_protocol_invalid"
        ),
        "protocol_valid": valid,
        "stage196_train_only_experiment_authorized": valid,
        "development_opened": False,
        "test_opened": False,
        "runtime_e2e_authorized": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "default_runtime_activation": False,
    }


def write_stage195_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage195Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = report["evidence_summary"]
    protocol = report["frozen_protocol"]
    top = evidence["top_inner_candidates"]
    charts = {
        "stage195_source_top_pool_recall.svg": _fold_chart(
            top, "Stage 195 source top pool recall", "strict_opportunity_pool_recall"
        ),
        "stage195_source_top_capture.svg": _fold_chart(
            top, "Stage 195 source top conditional capture", "conditional_ranker_strict_capture"
        ),
        "stage195_source_top_precision.svg": _fold_chart(
            top, "Stage 195 source top strict precision", "strict_success_precision"
        ),
        "stage195_source_top_unsafe.svg": _fold_chart(
            top, "Stage 195 source top unsafe rate", "unsafe_selection_rate"
        ),
        "stage195_factor_counts.svg": _chart(
            "Stage 195 frozen factor counts",
            tuple(
                BarDatum(name, value, str(value))
                for name, value in (
                    ("pool builders", protocol["candidate_grid"]["pool_builders"]),
                    ("gain models", protocol["candidate_grid"]["gain_models"]),
                    ("risk models", protocol["candidate_grid"]["risk_models"]),
                    ("prefix sizes", len(protocol["candidate_grid"]["safest_prefix_sizes"])),
                    ("policy configs", protocol["candidate_grid"]["policy_config_count"]),
                )
            ),
            "count",
        ),
        "stage195_fit_budget.svg": _chart(
            "Stage 195 frozen fit budget",
            tuple(
                BarDatum(name, value, str(value))
                for name, value in (
                    (
                        "pool safety per partition",
                        protocol["cross_validation"]["pool_safety_fits_per_partition"],
                    ),
                    (
                        "gain rankers per partition",
                        protocol["cross_validation"]["gain_ranker_fits_per_partition"],
                    ),
                    (
                        "unsafe heads per partition",
                        protocol["cross_validation"]["unsafe_head_fits_per_partition"],
                    ),
                    (
                        "all fits per partition",
                        protocol["cross_validation"]["model_fits_per_partition"],
                    ),
                    ("maximum all fits", protocol["cross_validation"]["maximum_model_fit_count"]),
                )
            ),
            "model fit count",
        ),
        "stage195_prefix_sizes.svg": _chart(
            "Stage 195 frozen safest-prefix sizes",
            tuple(BarDatum(str(value), value, str(value)) for value in SAFEST_PREFIX_SIZES),
            "actions before baseline union",
        ),
        "stage195_risk_weights.svg": _chart(
            "Stage 195 frozen unsafe positive-class weights",
            tuple(
                BarDatum(f"weight {value:g}", value, f"{value:g}") for value in SCALE_POS_WEIGHTS
            ),
            "scale_pos_weight",
        ),
        "stage195_advancement_gates.svg": _chart(
            "Stage 195 unchanged advancement thresholds",
            tuple(
                BarDatum(
                    row["metric"], float(row["threshold"]), f"{row['operator']} {row['threshold']}"
                )
                for row in protocol["advancement_gates"]
            ),
            "threshold magnitude",
            margin_left=820,
        ),
        "stage195_guard_checks.svg": _chart(
            "Stage 195 protocol guard checks",
            tuple(
                BarDatum(row["name"], float(row["passed"]), "pass" if row["passed"] else "fail")
                for row in report["guard_checks"]
            ),
            "1 means passed",
            margin_left=900,
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage195Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _fold_chart(rows: Sequence[Mapping[str, Any]], title: str, metric: str) -> str:
    return _chart(
        title,
        tuple(BarDatum(row["outer_fold_id"], row[metric], f"{row[metric]:.3f}") for row in rows),
        "rate",
    )


def _chart(title: str, bars: Sequence[BarDatum], x_label: str, *, margin_left: int = 440) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        width=1680,
        margin_left=margin_left,
        margin_right=260,
    )


def _range(values: Iterable[float]) -> dict[str, float]:
    materialized = [float(value) for value in values]
    return {"minimum": min(materialized), "maximum": max(materialized)}


def _gib(value: int) -> float:
    return round(value / 1024**3, 6)


def _fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "byte_size": resolved.stat().st_size,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _forbidden_keys_found(value: Any) -> set[str]:
    found = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in FORBIDDEN_PUBLIC_KEYS:
                found.add(str(key))
            found.update(_forbidden_keys_found(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            found.update(_forbidden_keys_found(nested))
    return found


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}
