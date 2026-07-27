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

STAGE = "Stage 202"
CREATED_AT = "2026-07-27"
PROTOCOL_ID = "primeqa_hybrid_top1_joint_objective_protocol_v1"
NEXT_STAGE = "Stage 203"
STAGE201_SHA256 = "b5ca57e6b7f6c0798adff91ba8615579b24435b846922088a86828801ceaa015"
STAGE199_SHA256 = "5b933f524fff1bceb4d4d842e4f3a1aec3160aa3ed337131444ec1b7c2699fee"
SAFETY_WEIGHTS = (0.0, 0.5, 1.0, 2.0)
PRECISION_WEIGHTS = (0.0, 0.5, 1.0, 2.0)
OUTER_FOLDS = 5
INNER_FOLDS = 4
SOURCE_MODELS_PER_PARTITION = 4
SOURCE_LIGHTGBM_MODELS_PER_PARTITION = 2
TREES_PER_LIGHTGBM_MODEL = 300
POOL_CAP = 16
MINIMUM_AVAILABLE_MEMORY_GIB = 4.0
FORBIDDEN_PUBLIC_KEYS = {
    "action_id",
    "candidate_actions",
    "complete_pool",
    "document_id",
    "document_text",
    "feature_rows",
    "predictions",
    "question_key",
    "question_text",
    "selected_action",
}


@dataclass(frozen=True)
class Stage202Visualization:
    name: str
    path: str


def freeze_top1_joint_objective_protocol(
    *,
    stage201_report_path: Path,
    stage199_report_path: Path,
    user_confirmed: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage 203 train-only Top-1 joint-objective experiment."""

    started_at = time.perf_counter()
    stage201 = _load_json(stage201_report_path)
    stage199 = _load_json(stage199_report_path)
    source_files = {
        "stage201_failure_attribution": _fingerprint(stage201_report_path),
        "stage199_joint_risk_winner": _fingerprint(stage199_report_path),
    }
    loaded_at = time.perf_counter()
    evidence = _evidence_summary(stage201, stage199)
    protocol = _frozen_protocol(evidence["source_trajectories"])
    preliminary: dict[str, Any] = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "protocol_id": PROTOCOL_ID,
        "protocol_scope": (
            "Aggregate-only preregistration of a train-only grouped Top-1 joint-objective "
            "experiment. Stage202 reads only public Stage199 and Stage201 reports, loads no "
            "split rows or documents, imports no LightGBM runtime, fits no model, evaluates "
            "no policy, opens no development or test data, relaxes no gate, adds no fallback, "
            "and changes no runtime default."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed),
            "selected_route": "A_top1_grouped_joint_constraint_objective",
            "confirmation_note": confirmation_note,
        },
        "source_files": source_files,
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
            "private_action_rows_materialized": 0,
            "private_feature_rows_materialized": 0,
            "private_predictions_materialized": 0,
            "objective_evaluation_run": False,
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
    guards = _guard_checks(preliminary, stage201, stage199)
    checked_at = time.perf_counter()
    report = {
        **preliminary,
        "guard_checks": guards,
        "decision": _decision(guards),
        "timing_seconds": {
            "load_public_reports": round(loaded_at - started_at, 6),
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
            "question_level_rows_persisted": False,
            "public_report_safe": not forbidden,
        },
    }


def _frozen_protocol(source_trajectories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    objectives = _objective_specs()
    custom_count = len(objectives)
    candidate_count = custom_count + 1
    inner_partition_count = OUTER_FOLDS * INNER_FOLDS
    inner_fits = SOURCE_MODELS_PER_PARTITION + custom_count
    outer_fits = SOURCE_MODELS_PER_PARTITION + 1
    inner_lightgbm = SOURCE_LIGHTGBM_MODELS_PER_PARTITION + custom_count
    outer_lightgbm = SOURCE_LIGHTGBM_MODELS_PER_PARTITION + 1
    return {
        "experiment_name": "train_only_grouped_top1_joint_objective_nested_cv",
        "split_contract": {
            "selection_split": "train",
            "outer_fold_count": OUTER_FOLDS,
            "inner_fold_count": INNER_FOLDS,
            "all_actions_for_one_question_remain_in_one_fold": True,
            "heldout_group_labels_used_for_fit_or_early_stopping": False,
            "sequential_train_research_not_claimed_as_unbiased_final_test": True,
            "development_opened": False,
            "test_opened": False,
        },
        "source_trajectory_contract": {
            "source": "Stage199 fixed Stage196 source spec in each outer context",
            "outer_context_count": len(source_trajectories),
            "trajectories": list(source_trajectories),
            "pool_feature_representation_fixed": True,
            "pool_safety_estimator_fixed": True,
            "objective_feature_representation_equals_source_gain_representation": True,
            "objective_tree_profile_equals_source_gain_tree_profile": True,
            "source_gain_and_risk_models_fit_only_for_exact_control": True,
            "old_risk_signal_and_winner_rule_grid_reopened": False,
            "exact_control_reproduces_stage196_decision": True,
        },
        "candidate_pool_contract": {
            "pool_cap": POOL_CAP,
            "pool_builder": "Stage196 citation-loss and F1-loss safety heads",
            "baseline_unioned_after_cap": True,
            "custom_objective_scores_complete_fixed_pool": True,
            "source_risk_frontier_applied_to_custom_candidates": False,
            "pool_builder_researched_or_retuned": False,
            "runtime_gold_filter_used": False,
            "canonical_action_order_final_tie_break": True,
            "retry_used": False,
            "fallback_used": False,
        },
        "target_encoding": {
            "unsafe": 0,
            "safe_zero": 1,
            "baseline": 2,
            "strict_success": 3,
            "unsafe_definition": "citation_delta < 0 or F1 delta < -1e-12",
            "safe_zero_definition": "citation_delta == 0 and abs(F1 delta) <= 1e-12",
            "baseline_identified_by_action_family": True,
            "target_labels_available_only_inside training partition": True,
            "target_encoding_used_at_runtime": False,
        },
        "top1_joint_objective": {
            "estimator": "lightgbm.LGBMRanker",
            "objective_api": "custom grouped objective(y_true, y_pred, weight, group)",
            "prediction_distribution": "stable within-question softmax over scalar scores",
            "capture_target": (
                "uniform over strict-success actions when present; otherwise one-hot baseline"
            ),
            "safety_target": "uniform over all non-unsafe actions",
            "precision_target": "uniform over strict-success actions plus baseline",
            "combined_target": (
                "(capture + safety_weight*safety + precision_weight*precision) / "
                "(1 + safety_weight + precision_weight)"
            ),
            "loss": "cross entropy from combined fixed target to predicted group softmax",
            "gradient": "predicted_probability - combined_target_probability",
            "diagonal_hessian": "max(p * (1 - p), 1e-6)",
            "full_softmax_hessian_approximated_diagonally": True,
            "question_balanced_weighting": True,
            "one_scalar_score_per_action": True,
            "winner": "highest score in complete fixed pool",
            "runtime_pairwise_tournament": False,
            "runtime_risk_threshold": False,
            "early_stopping": False,
            "tree_count": TREES_PER_LIGHTGBM_MODEL,
            "cpu_device": True,
            "deterministic": True,
            "physical_cpu_threads": 8,
        },
        "objective_factorial": {
            "safety_weights": list(SAFETY_WEIGHTS),
            "precision_weights": list(PRECISION_WEIGHTS),
            "custom_objectives": objectives,
            "custom_objective_count": custom_count,
            "exact_source_control_count": 1,
            "candidate_config_count_per_outer_context": candidate_count,
            "strict_only_cell": "top1_safety_0.00__precision_0.00",
            "safety_only_ablation_count": len(SAFETY_WEIGHTS) - 1,
            "precision_only_ablation_count": len(PRECISION_WEIGHTS) - 1,
            "full_joint_cell_count": (len(SAFETY_WEIGHTS) - 1) * (len(PRECISION_WEIGHTS) - 1),
            "paired_deltas_against_exact_control_required": True,
            "paired_deltas_against_strict_only_required": True,
            "directional_penalty_response_diagnostics_required": True,
            "directional_response_is_acceptance_gate": False,
        },
        "cross_validation": {
            "outer_fold_count": OUTER_FOLDS,
            "inner_fold_count": INNER_FOLDS,
            "inner_partition_count": inner_partition_count,
            "maximum_outer_refit_count": OUTER_FOLDS,
            "source_model_fits_per_inner_partition": SOURCE_MODELS_PER_PARTITION,
            "custom_objective_fits_per_inner_partition": custom_count,
            "model_fits_per_inner_partition": inner_fits,
            "maximum_model_fit_count": inner_partition_count * inner_fits
            + OUTER_FOLDS * outer_fits,
            "lightgbm_models_per_inner_partition": inner_lightgbm,
            "maximum_lightgbm_tree_count": (
                inner_partition_count * inner_lightgbm + OUTER_FOLDS * outer_lightgbm
            )
            * TREES_PER_LIGHTGBM_MODEL,
            "all_16_objectives_fit_on_each_inner_training_partition": True,
            "outer_refit_only_selected_eligible_config": True,
            "no_inner_eligible_config_behavior": (
                "record no-eligible and do not evaluate a weaker outer configuration"
            ),
            "no_retry": True,
            "no_fallback": True,
        },
        "objective_validity_guards": {
            "group_sizes_sum_to_rows": True,
            "exactly_one_baseline_per_question": True,
            "each_target_distribution_sums_to_one": True,
            "combined_target_sums_to_one": True,
            "gradient_and_hessian_finite": True,
            "hessian_strictly_positive": True,
            "training_groups_only_in_objective_closure": True,
            "heldout_labels_excluded_from_objective": True,
            "prediction_row_alignment_exact": True,
            "deterministic_repeatability_check_required": True,
        },
        "inner_selection": _inner_selection(),
        "advancement_gates": _advancement_gates(),
        "resource_contract": {
            "minimum_preflight_system_available_memory_gib": MINIMUM_AVAILABLE_MEMORY_GIB,
            "cpu_device": True,
            "physical_cpu_threads": 8,
            "one_custom_objective_model_materialized_at_a_time": True,
            "shared_sparse_matrices_released_per_partition": True,
            "event_driven_resource_statistics_required": True,
            "process_wait_contract": (
                "one PowerShell Wait-Process call for the formal PID until natural exit"
            ),
            "insufficient_memory_behavior": (
                "do not start; request resource clearance instead of reducing the grid"
            ),
        },
        "authorization_boundary": {
            "stage203_train_only_experiment_authorized": True,
            "development_evaluation_authorized": False,
            "test_evaluation_authorized": False,
            "full_train_policy_selection_authorized": False,
            "runtime_e2e_authorized": False,
            "replacement_policy_selection_authorized": False,
            "default_runtime_activation_authorized": False,
            "stage178b_authorized": False,
        },
    }


def _objective_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": f"top1_safety_{safety:.2f}__precision_{precision:.2f}",
            "safety_weight": safety,
            "precision_weight": precision,
            "ablation_family": (
                "strict_only"
                if safety == 0.0 and precision == 0.0
                else "safety_only"
                if safety > 0.0 and precision == 0.0
                else "precision_only"
                if safety == 0.0 and precision > 0.0
                else "full_joint"
            ),
        }
        for safety in SAFETY_WEIGHTS
        for precision in PRECISION_WEIGHTS
    ]


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
        "thresholds_unchanged_from_stage196": True,
        "lexicographic_objective": [
            "maximize strict-success count",
            "maximize conditional strict capture",
            "maximize strict-success precision",
            "minimize unsafe selection count",
            "maximize strict-opportunity pool recall",
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


def _evidence_summary(stage201: Mapping[str, Any], stage199: Mapping[str, Any]) -> dict[str, Any]:
    attribution = stage201["failure_attribution"]
    questions = attribution["question_context_attribution"]["aggregate"]
    constraints = attribution["constraint_attribution"]["constraints"]
    finding = attribution["diagnostic_finding"]
    stage199_cv = stage199["joint_risk_winner_nested_cv"]
    trajectories = [
        {
            "outer_context": fold_id,
            "source_spec": row["source_spec"],
            "control_reproduction_exact": row["control_reproduction_exact"],
        }
        for fold_id, row in sorted(stage199_cv["outer_contexts"].items())
    ]
    return {
        "stage201_status": stage201["decision"]["status"],
        "stage201_experiment_valid": stage201["decision"]["experiment_valid"],
        "stage201_diagnostic_complete": stage201["decision"]["diagnostic_complete"],
        "stage201_development_opened": stage201["decision"]["development_opened"],
        "stage201_test_opened": stage201["decision"]["test_opened"],
        "stage201_process_guard_count": len(stage201["process_guards"]),
        "stage201_process_guards_passed": sum(row["passed"] for row in stage201["process_guards"]),
        "stage199_status": stage199["decision"]["status"],
        "stage199_experiment_valid": stage199["decision"]["experiment_valid"],
        "stage199_candidate_family_accepted": stage199["decision"]["candidate_family_accepted"],
        "stage199_report_hash_referenced_by_stage201": stage201["source_authorization"][
            "stage199_formal_report"
        ]["sha256"],
        "population": attribution["population"],
        "selected_outcome_counts": questions["selected_outcome_counts"],
        "selected_outcome_partition_exact": questions["selected_outcome_partition_exact"],
        "strict_opportunity_mechanism_counts": questions["strict_opportunity_mechanism_counts"],
        "strict_opportunity_partition_exact": questions["strict_opportunity_partition_exact"],
        "lower_risk_strict_alternative_count": questions["ranking_conflicts"][
            "lower_risk_strict_alternative_count"
        ],
        "higher_gain_strict_alternative_count": questions["ranking_conflicts"][
            "higher_gain_strict_alternative_count"
        ],
        "constraint_failure_counts": {
            name: row["failure_count"] for name, row in constraints.items()
        },
        "research_scores": finding["failure_count_score_by_research_axis"],
        "recommended_next_research": finding["recommended_next_research"],
        "dominant_failed_constraint": finding["dominant_failed_constraint"],
        "dominant_failure_mechanism": finding["dominant_failure_mechanism"],
        "source_trajectories": trajectories,
        "all_control_reproductions_exact": all(
            row["control_reproduction_exact"] for row in trajectories
        ),
    }


def _guard_checks(
    preliminary: Mapping[str, Any],
    stage201: Mapping[str, Any],
    stage199: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evidence = preliminary["evidence_summary"]
    protocol = preliminary["frozen_protocol"]
    objective = protocol["top1_joint_objective"]
    grid = protocol["objective_factorial"]
    cv = protocol["cross_validation"]
    pool = protocol["candidate_pool_contract"]
    validity = protocol["objective_validity_guards"]
    authorization = protocol["authorization_boundary"]
    boundaries = preliminary["execution_boundaries"]
    checks = (
        ("user_confirmed_route_a", preliminary["user_confirmation"]["confirmed"] is True),
        (
            "stage201_sha256_matches",
            preliminary["source_files"]["stage201_failure_attribution"]["sha256"]
            == STAGE201_SHA256,
        ),
        (
            "stage199_sha256_matches",
            preliminary["source_files"]["stage199_joint_risk_winner"]["sha256"] == STAGE199_SHA256,
        ),
        ("source_is_stage201", stage201.get("stage") == "Stage 201"),
        ("control_source_is_stage199", stage199.get("stage") == "Stage 199"),
        ("stage201_valid", evidence["stage201_experiment_valid"] is True),
        ("stage201_complete", evidence["stage201_diagnostic_complete"] is True),
        (
            "stage201_status_complete",
            evidence["stage201_status"]
            == "stage201_joint_risk_winner_failure_attribution_complete",
        ),
        ("stage201_development_closed", evidence["stage201_development_opened"] is False),
        ("stage201_test_closed", evidence["stage201_test_opened"] is False),
        (
            "stage201_all_guards_passed",
            evidence["stage201_process_guard_count"] == 36
            and evidence["stage201_process_guards_passed"] == 36,
        ),
        ("stage199_valid", evidence["stage199_experiment_valid"] is True),
        (
            "stage199_status_insufficient",
            evidence["stage199_status"] == "stage199_joint_risk_winner_insufficient",
        ),
        ("stage199_family_not_accepted", evidence["stage199_candidate_family_accepted"] is False),
        (
            "stage201_references_exact_stage199",
            evidence["stage199_report_hash_referenced_by_stage201"] == STAGE199_SHA256,
        ),
        (
            "question_population_exact",
            evidence["population"]["question_cell_context_count"] == 41_440,
        ),
        ("outcome_partition_exact", evidence["selected_outcome_partition_exact"] is True),
        (
            "opportunity_partition_exact",
            evidence["strict_opportunity_partition_exact"] is True,
        ),
        (
            "winner_miss_is_dominant_failure",
            evidence["dominant_failure_mechanism"] == "winner_selection_miss",
        ),
        (
            "winner_misses_observed",
            evidence["strict_opportunity_mechanism_counts"]["winner_selection_miss"] == 15_294,
        ),
        (
            "objective_research_recommended",
            evidence["recommended_next_research"] == "objective_research",
        ),
        (
            "capture_is_dominant_constraint",
            evidence["dominant_failed_constraint"] == "conditional_ranker_strict_capture",
        ),
        (
            "capture_failures_observed",
            evidence["constraint_failure_counts"]["conditional_ranker_strict_capture"] == 135,
        ),
        (
            "precision_failures_observed",
            evidence["constraint_failure_counts"]["strict_success_precision"] == 125,
        ),
        (
            "unsafe_failures_observed",
            evidence["constraint_failure_counts"]["unsafe_selection_rate"] == 116,
        ),
        ("five_source_trajectories", len(evidence["source_trajectories"]) == 5),
        ("source_controls_exact", evidence["all_control_reproductions_exact"] is True),
        ("pool_cap_is_16", pool["pool_cap"] == 16),
        ("pool_builder_fixed", pool["pool_builder_researched_or_retuned"] is False),
        ("baseline_unioned", pool["baseline_unioned_after_cap"] is True),
        (
            "custom_scores_complete_pool",
            pool["custom_objective_scores_complete_fixed_pool"] is True,
        ),
        (
            "old_frontier_not_applied_to_custom",
            pool["source_risk_frontier_applied_to_custom_candidates"] is False,
        ),
        ("no_runtime_gold_filter", pool["runtime_gold_filter_used"] is False),
        ("objective_is_grouped_ranker", objective["estimator"] == "lightgbm.LGBMRanker"),
        ("objective_outputs_scalar", objective["one_scalar_score_per_action"] is True),
        ("objective_has_no_tournament", objective["runtime_pairwise_tournament"] is False),
        ("objective_has_no_risk_threshold", objective["runtime_risk_threshold"] is False),
        ("objective_has_positive_hessian_floor", "1e-6" in objective["diagonal_hessian"]),
        ("objective_is_deterministic", objective["deterministic"] is True),
        ("objective_uses_cpu", objective["cpu_device"] is True),
        ("safety_weights_exact", grid["safety_weights"] == list(SAFETY_WEIGHTS)),
        ("precision_weights_exact", grid["precision_weights"] == list(PRECISION_WEIGHTS)),
        ("sixteen_custom_objectives", grid["custom_objective_count"] == 16),
        ("seventeen_candidate_configs", grid["candidate_config_count_per_outer_context"] == 17),
        ("nine_full_joint_cells", grid["full_joint_cell_count"] == 9),
        ("paired_control_deltas", grid["paired_deltas_against_exact_control_required"] is True),
        ("paired_strict_only_deltas", grid["paired_deltas_against_strict_only_required"] is True),
        (
            "directional_diagnostics_not_gate",
            grid["directional_response_is_acceptance_gate"] is False,
        ),
        ("twenty_inner_partitions", cv["inner_partition_count"] == 20),
        ("twenty_fits_per_inner", cv["model_fits_per_inner_partition"] == 20),
        ("maximum_fit_count_is_425", cv["maximum_model_fit_count"] == 425),
        ("maximum_tree_count_is_112500", cv["maximum_lightgbm_tree_count"] == 112_500),
        (
            "outer_refit_selected_only",
            cv["outer_refit_only_selected_eligible_config"] is True,
        ),
        ("all_group_guards_enabled", all(validity.values())),
        (
            "inner_thresholds_unchanged",
            protocol["inner_selection"]["thresholds_unchanged_from_stage196"] is True,
        ),
        (
            "no_weaker_substitution",
            protocol["inner_selection"]["weaker_ineligible_candidate_substitution"] is False,
        ),
        (
            "thirteen_inner_constraints",
            len(protocol["inner_selection"]["eligibility_constraints"]) == 13,
        ),
        ("seventeen_advancement_gates", len(protocol["advancement_gates"]) == 17),
        (
            "memory_threshold_is_4_gib",
            protocol["resource_contract"]["minimum_preflight_system_available_memory_gib"] == 4.0,
        ),
        (
            "single_wait_process_contract",
            "one PowerShell Wait-Process" in protocol["resource_contract"]["process_wait_contract"],
        ),
        ("stage203_authorized", authorization["stage203_train_only_experiment_authorized"] is True),
        ("development_not_authorized", authorization["development_evaluation_authorized"] is False),
        ("test_not_authorized", authorization["test_evaluation_authorized"] is False),
        (
            "full_train_not_authorized",
            authorization["full_train_policy_selection_authorized"] is False,
        ),
        ("runtime_not_authorized", authorization["runtime_e2e_authorized"] is False),
        (
            "replacement_not_authorized",
            authorization["replacement_policy_selection_authorized"] is False,
        ),
        ("default_not_authorized", authorization["default_runtime_activation_authorized"] is False),
        ("stage178b_not_authorized", authorization["stage178b_authorized"] is False),
        ("public_only", boundaries["loaded_public_reports_only"] is True),
        ("no_train_rows", boundaries["train_rows_loaded"] is False),
        ("no_dev_rows", boundaries["development_loaded"] is False),
        ("no_test_rows", boundaries["test_loaded"] is False),
        ("no_lightgbm_import", boundaries["lightgbm_imported"] is False),
        ("no_dependency_change", boundaries["dependency_installed_or_changed"] is False),
        ("zero_model_fits", boundaries["model_fit_count"] == 0),
        ("zero_private_rows", boundaries["private_action_rows_materialized"] == 0),
        ("zero_private_predictions", boundaries["private_predictions_materialized"] == 0),
        ("no_objective_evaluation", boundaries["objective_evaluation_run"] is False),
        ("no_policy_evaluation", boundaries["policy_evaluation_run"] is False),
        ("no_retry", boundaries["retry_action_count"] == 0),
        ("no_fallback", boundaries["fallback_action_count"] == 0),
    )
    return [_gate(name, passed) for name, passed in checks]


def _decision(guards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = bool(guards) and all(row["passed"] for row in guards)
    return {
        "status": (
            "stage202_top1_joint_objective_protocol_frozen"
            if valid
            else "stage202_top1_joint_objective_protocol_invalid"
        ),
        "protocol_valid": valid,
        "stage203_train_only_experiment_authorized": valid,
        "development_opened": False,
        "test_opened": False,
        "runtime_e2e_authorized": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "default_runtime_activation": False,
    }


def write_stage202_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage202Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = report["evidence_summary"]
    protocol = report["frozen_protocol"]
    grid = protocol["objective_factorial"]
    cv = protocol["cross_validation"]
    charts = {
        "stage202_source_blockers.svg": _chart(
            "Stage 202 source eligibility blockers",
            [
                _count_bar(name, value)
                for name, value in sorted(
                    evidence["constraint_failure_counts"].items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
            margin_left=700,
        ),
        "stage202_source_mechanisms.svg": _chart(
            "Stage 202 strict-opportunity mechanisms",
            [
                _count_bar(name, value)
                for name, value in evidence["strict_opportunity_mechanism_counts"].items()
            ],
            margin_left=580,
        ),
        "stage202_research_scores.svg": _chart(
            "Stage 202 frozen research-axis evidence",
            [_count_bar(name, value) for name, value in evidence["research_scores"].items()],
        ),
        "stage202_objective_grid.svg": _chart(
            "Stage 202 Top-1 objective grid",
            [
                _count_bar("strict-only cells", 1),
                _count_bar("safety-only cells", grid["safety_only_ablation_count"]),
                _count_bar("precision-only cells", grid["precision_only_ablation_count"]),
                _count_bar("full joint cells", grid["full_joint_cell_count"]),
                _count_bar("exact controls", grid["exact_source_control_count"]),
            ],
        ),
        "stage202_penalty_values.svg": _chart(
            "Stage 202 frozen objective weights",
            [
                *[_rate_bar(f"safety {value:.2f}", value) for value in SAFETY_WEIGHTS],
                *[_rate_bar(f"precision {value:.2f}", value) for value in PRECISION_WEIGHTS],
            ],
        ),
        "stage202_pool_contract.svg": _chart(
            "Stage 202 fixed candidate path",
            [
                _count_bar("pool cap", protocol["candidate_pool_contract"]["pool_cap"]),
                _bool_bar(
                    "score complete pool",
                    protocol["candidate_pool_contract"][
                        "custom_objective_scores_complete_fixed_pool"
                    ],
                ),
                _bool_bar(
                    "apply old risk frontier",
                    protocol["candidate_pool_contract"][
                        "source_risk_frontier_applied_to_custom_candidates"
                    ],
                ),
            ],
        ),
        "stage202_fit_budget.svg": _chart(
            "Stage 202 maximum Stage 203 budget",
            [
                _count_bar("inner partitions", cv["inner_partition_count"]),
                _count_bar("fits per inner", cv["model_fits_per_inner_partition"]),
                _count_bar("maximum fits", cv["maximum_model_fit_count"]),
                _count_bar("maximum LightGBM trees", cv["maximum_lightgbm_tree_count"]),
            ],
        ),
        "stage202_advancement_gates.svg": _chart(
            "Stage 202 retained advancement thresholds",
            [
                BarDatum(
                    row["metric"],
                    float(row["threshold"]),
                    f"{row['operator']} {row['threshold']}",
                )
                for row in protocol["advancement_gates"]
            ],
            margin_left=720,
        ),
        "stage202_authorization.svg": _chart(
            "Stage 202 authorization boundary",
            [_bool_bar(name, value) for name, value in protocol["authorization_boundary"].items()],
            margin_left=720,
        ),
        "stage202_guard_checks.svg": _chart(
            "Stage 202 guard checks",
            [_bool_bar(row["name"], row["passed"]) for row in report["guard_checks"]],
            margin_left=800,
        ),
    }
    visualizations = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        visualizations.append(Stage202Visualization(name, str(path)))
    return tuple(visualizations)


def _chart(
    title: str,
    bars: Sequence[BarDatum],
    *,
    margin_left: int = 520,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label="frozen aggregate value",
        width=1680,
        margin_left=margin_left,
        margin_right=220,
    )


def _count_bar(name: str, value: int) -> BarDatum:
    return BarDatum(name, float(value), str(value))


def _rate_bar(name: str, value: float) -> BarDatum:
    return BarDatum(name, value, f"{value:.6f}")


def _bool_bar(name: str, value: bool) -> BarDatum:
    return BarDatum(name, float(value), str(value).lower())


def _fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _forbidden_keys_found(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in FORBIDDEN_PUBLIC_KEYS:
                found.add(str(key))
            found.update(_forbidden_keys_found(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            found.update(_forbidden_keys_found(child))
    return found


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}
