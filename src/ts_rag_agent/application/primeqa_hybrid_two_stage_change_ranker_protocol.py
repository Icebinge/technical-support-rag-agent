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

STAGE = "Stage 205"
CREATED_AT = "2026-07-27"
PROTOCOL_ID = "primeqa_hybrid_two_stage_change_ranker_protocol_v1"
NEXT_STAGE = "Stage 206"
STAGE204_SHA256 = "3757cc7a84a7a70beddd151228fbe39157fa546db2ce11da4996e66eefd19fe8"
STAGE202_SHA256 = "0818f0ae7186eea19d4137023b87963010e76e9cbf20e8f1ef61576a60569bdb"
RANKER_FAMILIES = ("strict_binary", "strict_safety_graded")
TARGET_CHANGE_COVERAGES = (0.25, 0.40, 0.55, 0.70, 0.85)
OUTER_FOLDS = 5
INNER_FOLDS = 4
GATE_CROSSFIT_FOLDS = 4
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
    "gate_training_rows",
    "predictions",
    "question_key",
    "question_text",
    "selected_action",
}


@dataclass(frozen=True)
class Stage205Visualization:
    name: str
    path: str


def freeze_two_stage_change_ranker_protocol(
    *,
    stage204_report_path: Path,
    stage202_protocol_path: Path,
    user_confirmed: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage 206 train-only two-stage nested-CV experiment."""

    started_at = time.perf_counter()
    stage204 = _load_json(stage204_report_path)
    stage202 = _load_json(stage202_protocol_path)
    sources = {
        "stage204_failure_attribution": _fingerprint(stage204_report_path),
        "stage202_source_protocol": _fingerprint(stage202_protocol_path),
    }
    loaded_at = time.perf_counter()
    evidence = _evidence_summary(stage204, stage202)
    protocol = _frozen_protocol(evidence["source_trajectories"])
    preliminary: dict[str, Any] = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "protocol_id": PROTOCOL_ID,
        "protocol_scope": (
            "Aggregate-only preregistration of a train-only two-stage change/abstain gate "
            "and baseline-excluded conditional action ranker experiment. Stage205 reads only "
            "public Stage204 and Stage202 reports, loads no split rows or documents, imports "
            "no model runtime, fits no model, generates no prediction, evaluates no policy, "
            "opens no development or test data, relaxes no gate, adds no fallback, and changes "
            "no runtime behavior."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed),
            "selected_route": "A_separate_change_abstain_gate_and_conditional_ranker",
            "confirmation_note": confirmation_note,
        },
        "source_files": sources,
        "evidence_summary": evidence,
        "frozen_protocol": protocol,
        "execution_boundaries": {
            "loaded_public_reports_only": True,
            "train_rows_loaded": False,
            "development_loaded": False,
            "test_loaded": False,
            "lightgbm_imported": False,
            "sklearn_imported": False,
            "dependency_installed_or_changed": False,
            "model_fit_count": 0,
            "private_action_rows_materialized": 0,
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
    guards = _guard_checks(preliminary, stage204, stage202)
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
    rankers = _ranker_specs()
    policies = _policy_specs()
    inner_partition_count = OUTER_FOLDS * INNER_FOLDS
    ranker_fits_per_family = GATE_CROSSFIT_FOLDS + 1
    ranker_fits = len(rankers) * ranker_fits_per_family
    gate_fits = len(rankers)
    fits_per_inner = SOURCE_MODELS_PER_PARTITION + ranker_fits + gate_fits
    lightgbm_per_inner = SOURCE_LIGHTGBM_MODELS_PER_PARTITION + ranker_fits + gate_fits
    outer_fits = SOURCE_MODELS_PER_PARTITION + ranker_fits_per_family + 1
    outer_lightgbm = SOURCE_LIGHTGBM_MODELS_PER_PARTITION + ranker_fits_per_family + 1
    return {
        "experiment_name": "train_only_two_stage_change_ranker_nested_cv",
        "split_contract": {
            "selection_split": "train",
            "outer_fold_count": OUTER_FOLDS,
            "inner_fold_count": INNER_FOLDS,
            "gate_crossfit_fold_count": GATE_CROSSFIT_FOLDS,
            "all_actions_for_one_question_remain_in_one_fold": True,
            "gate_crossfit_partitions_are_question_grouped": True,
            "heldout_labels_used_for_fit_or_early_stopping": False,
            "sequential_train_research_not_claimed_as_unbiased_final_test": True,
            "development_opened": False,
            "test_opened": False,
        },
        "source_trajectory_contract": {
            "source": "Stage202 exact Stage196 source trajectory per outer context",
            "outer_context_count": len(source_trajectories),
            "trajectories": list(source_trajectories),
            "pool_builder_fixed_per_outer_context": True,
            "source_control_fixed_per_outer_context": True,
            "feature_representations_fixed_per_outer_context": True,
            "tree_profiles_fixed_per_outer_context": True,
            "old_objective_grid_reopened": False,
            "exact_control_reproduces_stage196_decision": True,
        },
        "candidate_pool_contract": {
            "pool_cap": POOL_CAP,
            "pool_builder": "Stage196 citation-loss and F1-loss safety heads",
            "baseline_unioned_after_cap": True,
            "conditional_ranker_pool": "all fixed-pool nonbaseline actions",
            "baseline_excluded_from_conditional_ranker_fit": True,
            "baseline_excluded_from_conditional_ranker_softmax": True,
            "baseline_excluded_from_conditional_ranker_winner": True,
            "at_least_one_nonbaseline_action_required": True,
            "source_risk_frontier_applied_to_two_stage_candidates": False,
            "pool_builder_researched_or_retuned": False,
            "runtime_gold_filter_used": False,
            "canonical_action_order_final_tie_break": True,
            "retry_used": False,
            "fallback_used": False,
        },
        "outcome_encoding": {
            "unsafe": 0,
            "safe_zero": 1,
            "strict_success": 2,
            "baseline": "excluded from conditional labels",
            "unsafe_definition": "citation_delta < 0 or F1 delta < -1e-12",
            "safe_zero_definition": "citation_delta == 0 and abs(F1 delta) <= 1e-12",
            "strict_success_definition": "nonbaseline strict_expected action",
            "labels_available_only_inside fitting partition": True,
            "gold_labels_used_at_runtime": False,
        },
        "conditional_ranker": {
            "estimator": "lightgbm.LGBMRanker",
            "objective": "lambdarank",
            "question_grouped": True,
            "feature_representation": (
                "source gain_feature_representation from the outer-context trajectory"
            ),
            "tree_profile": "source gain_tree_profile from the outer-context trajectory",
            "families": rankers,
            "family_count": len(rankers),
            "winner": "highest ranker score among nonbaseline actions",
            "within_question_normalized_top1_top2_margin_exported_to_gate": True,
            "raw_absolute_ranker_score_exported_to_gate": False,
            "question_balanced_weighting": True,
            "early_stopping": False,
            "tree_count": TREES_PER_LIGHTGBM_MODEL,
            "cpu_device": True,
            "deterministic": True,
            "physical_cpu_threads": 8,
        },
        "change_abstain_gate": {
            "estimator": "lightgbm.LGBMClassifier",
            "objective": "binary",
            "positive_label": "cross-fitted conditional winner is strict_success",
            "negative_label": "cross-fitted conditional winner is safe_zero or unsafe",
            "training_winners_are_ranker_oof_only": True,
            "same_fit_ranker_winner_used_for_gate_training": False,
            "gate_features": [
                "conditional winner runtime features",
                "baseline runtime features",
                "winner-minus-baseline numeric feature deltas",
                "within-question min-max normalized conditional top1-minus-top2 margin",
                "source citation-loss and F1-loss safety scores",
                "source within-question safety ranks",
            ],
            "tree_profile": "source risk_tree_profile from the outer-context trajectory",
            "gold_or_outcome_features_used": False,
            "raw_absolute_ranker_scores_used": False,
            "class_weight": "balanced",
            "threshold_method": "training OOF score order statistic for target change coverage",
            "target_change_coverages": list(TARGET_CHANGE_COVERAGES),
            "threshold_learned_inside_each_training_partition": True,
            "threshold_reused_without_heldout_tuning": True,
            "decision": (
                "change to conditional winner when gate score meets threshold; else baseline"
            ),
            "baseline_is_intentional_abstention": True,
            "probability_calibration_claimed": False,
            "early_stopping": False,
            "tree_count": TREES_PER_LIGHTGBM_MODEL,
            "cpu_device": True,
            "deterministic": True,
        },
        "cross_fitting_contract": {
            "inner_training_partition_split_into_gate_crossfit_folds": GATE_CROSSFIT_FOLDS,
            "gate_crossfit_assignment": (
                "SHA-256(outer context, inner heldout context, stable question identifier, "
                "seed 205) modulo 4"
            ),
            "gate_crossfit_assignment_is_deterministic": True,
            "ranker_fit_on_crossfit_complement": True,
            "ranker_predicts_crossfit_heldout_questions_once": True,
            "gate_training_has_exactly_one_oof_winner_per_question": True,
            "gate_label_derived_after_oof_winner_selection": True,
            "full_inner_ranker_fit_only_after_gate_rows_are_complete": True,
            "full_inner_ranker_selects_inner_heldout_winner": True,
            "gate_predicts_inner_heldout_without_refit": True,
            "outer_refit_repeats_same_crossfit_pipeline": True,
            "question_overlap_between_gate_fit_and_ranker_prediction_fold": False,
            "no_heldout_threshold_tuning": True,
        },
        "factorial_ablation": {
            "ranker_family_count": len(rankers),
            "target_change_coverage_count": len(TARGET_CHANGE_COVERAGES),
            "two_stage_policy_count": len(policies),
            "exact_source_control_count": 1,
            "candidate_config_count_per_outer_context": len(policies) + 1,
            "policies": policies,
            "paired_deltas_against_exact_control_required": True,
            "ranker_family_paired_comparison_required": True,
            "coverage_curve_monotonicity_diagnostic_required": True,
            "coverage_curve_monotonicity_is_acceptance_gate": False,
        },
        "cross_validation": {
            "outer_fold_count": OUTER_FOLDS,
            "inner_fold_count": INNER_FOLDS,
            "inner_partition_count": inner_partition_count,
            "ranker_fits_per_family_per_partition": ranker_fits_per_family,
            "conditional_ranker_fits_per_inner_partition": ranker_fits,
            "gate_fits_per_inner_partition": gate_fits,
            "source_model_fits_per_inner_partition": SOURCE_MODELS_PER_PARTITION,
            "model_fits_per_inner_partition": fits_per_inner,
            "maximum_outer_refit_count": OUTER_FOLDS,
            "maximum_model_fit_count": inner_partition_count * fits_per_inner
            + OUTER_FOLDS * outer_fits,
            "lightgbm_models_per_inner_partition": lightgbm_per_inner,
            "maximum_lightgbm_tree_count": (
                inner_partition_count * lightgbm_per_inner + OUTER_FOLDS * outer_lightgbm
            )
            * TREES_PER_LIGHTGBM_MODEL,
            "outer_refit_only_selected_eligible_config": True,
            "no_inner_eligible_config_behavior": (
                "record no-eligible and do not evaluate a weaker outer configuration"
            ),
            "no_retry": True,
            "no_fallback": True,
        },
        "validity_guards": {
            "question_groups_disjoint_in_every_split": True,
            "exactly_one_baseline_per_question_before_exclusion": True,
            "at_least_one_nonbaseline_action_per_question": True,
            "baseline_absent_from_ranker_rows": True,
            "ranker_labels_fit_partition_only": True,
            "gate_winners_are_oof": True,
            "one_gate_row_per_training_question": True,
            "gate_labels_have_both_classes": True,
            "gate_features_are_runtime_available": True,
            "threshold_uses_training_oof_scores_only": True,
            "prediction_row_alignment_exact": True,
            "deterministic_repeatability_check_required": True,
        },
        "inner_selection": _inner_selection(),
        "advancement_gates": _advancement_gates(),
        "resource_contract": {
            "minimum_preflight_system_available_memory_gib": MINIMUM_AVAILABLE_MEMORY_GIB,
            "cpu_device": True,
            "physical_cpu_threads": 8,
            "one_ranker_family_materialized_at_a_time": True,
            "crossfit_models_released_after_oof_prediction": True,
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
            "stage206_train_only_experiment_authorized": True,
            "development_evaluation_authorized": False,
            "test_evaluation_authorized": False,
            "full_train_policy_selection_authorized": False,
            "runtime_e2e_authorized": False,
            "replacement_policy_selection_authorized": False,
            "default_runtime_activation_authorized": False,
            "stage178b_authorized": False,
        },
    }


def _ranker_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "strict_binary",
            "labels": {"unsafe": 0, "safe_zero": 0, "strict_success": 1},
            "label_gain": [0, 1],
            "purpose": "direct strict-success discrimination",
        },
        {
            "name": "strict_safety_graded",
            "labels": {"unsafe": 0, "safe_zero": 1, "strict_success": 2},
            "label_gain": [0, 1, 4],
            "purpose": "strict-first ranking with safe-zero above unsafe",
        },
    ]


def _policy_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": f"{ranker}__change_c{int(round(coverage * 100)):02d}",
            "ranker_family": ranker,
            "target_change_coverage": coverage,
        }
        for ranker in RANKER_FAMILIES
        for coverage in TARGET_CHANGE_COVERAGES
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


def _evidence_summary(stage204: Mapping[str, Any], stage202: Mapping[str, Any]) -> dict[str, Any]:
    attribution = stage204["failure_attribution"]
    control = attribution["control_to_custom"]["aggregate"]
    precision = attribution["precision_adjacent_attribution"]["aggregate"]
    safety = attribution["safety_adjacent_attribution"]["aggregate"]
    mechanics = attribution["target_mechanics"]
    finding = attribution["diagnostic_finding"]
    trajectories = stage202["frozen_protocol"]["source_trajectory_contract"]["trajectories"]
    return {
        "stage204_status": stage204["decision"]["status"],
        "stage204_experiment_valid": stage204["decision"]["experiment_valid"],
        "stage204_diagnostic_complete": stage204["decision"]["diagnostic_complete"],
        "stage204_development_opened": stage204["decision"]["development_opened"],
        "stage204_test_opened": stage204["decision"]["test_opened"],
        "stage204_process_guard_count": len(stage204["process_guards"]),
        "stage204_process_guards_passed": sum(row["passed"] for row in stage204["process_guards"]),
        "stage202_status": stage202["decision"]["status"],
        "stage202_protocol_valid": stage202["decision"]["protocol_valid"],
        "stage203_hash_referenced_by_stage204": stage204["source_authorization"][
            "stage203_formal_report"
        ]["sha256"],
        "population": attribution["population"],
        "control_custom_winner_flip_count": control["winner_flip_count"],
        "control_custom_net_strict_count": control["net_strict_count"],
        "control_custom_net_unsafe_count": control["net_unsafe_count"],
        "control_custom_net_baseline_count": control["net_baseline_count"],
        "precision_net_strict_count": precision["net_strict_count"],
        "precision_net_unsafe_count": precision["net_unsafe_count"],
        "precision_net_baseline_count": precision["net_baseline_count"],
        "safety_net_strict_count": safety["net_strict_count"],
        "safety_net_unsafe_count": safety["net_unsafe_count"],
        "safety_net_baseline_count": safety["net_baseline_count"],
        "strict_opportunity_count": mechanics["strict_opportunity_count"],
        "no_strict_opportunity_count": mechanics["no_strict_opportunity_count"],
        "mean_strict_actions_per_question": mechanics["strict_action_count"]["mean"],
        "target_mass_exact": mechanics["precision_component_mass_sum_exact"],
        "dominant_precision_outcome_change": finding["dominant_precision_outcome_change"],
        "recommended_next_research": finding["recommended_next_research"],
        "finding_is_causal_claim": finding["finding_is_causal_claim"],
        "source_trajectories": trajectories,
        "all_control_reproductions_exact": all(
            row["control_reproduction_exact"] for row in trajectories
        ),
    }


def _guard_checks(
    preliminary: Mapping[str, Any],
    stage204: Mapping[str, Any],
    stage202: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evidence = preliminary["evidence_summary"]
    protocol = preliminary["frozen_protocol"]
    pool = protocol["candidate_pool_contract"]
    ranker = protocol["conditional_ranker"]
    gate = protocol["change_abstain_gate"]
    crossfit = protocol["cross_fitting_contract"]
    factorial = protocol["factorial_ablation"]
    cv = protocol["cross_validation"]
    validity = protocol["validity_guards"]
    authorization = protocol["authorization_boundary"]
    boundaries = preliminary["execution_boundaries"]
    checks = (
        ("user_confirmed_route_a", preliminary["user_confirmation"]["confirmed"] is True),
        (
            "stage204_sha256_matches",
            preliminary["source_files"]["stage204_failure_attribution"]["sha256"]
            == STAGE204_SHA256,
        ),
        (
            "stage202_sha256_matches",
            preliminary["source_files"]["stage202_source_protocol"]["sha256"] == STAGE202_SHA256,
        ),
        ("source_is_stage204", stage204.get("stage") == "Stage 204"),
        ("trajectory_source_is_stage202", stage202.get("stage") == "Stage 202"),
        ("stage204_valid", evidence["stage204_experiment_valid"] is True),
        ("stage204_complete", evidence["stage204_diagnostic_complete"] is True),
        (
            "stage204_status_complete",
            evidence["stage204_status"]
            == "stage204_top1_joint_objective_failure_attribution_complete",
        ),
        ("stage204_development_closed", evidence["stage204_development_opened"] is False),
        ("stage204_test_closed", evidence["stage204_test_opened"] is False),
        (
            "stage204_all_guards_passed",
            evidence["stage204_process_guard_count"] == 39
            and evidence["stage204_process_guards_passed"] == 39,
        ),
        ("stage202_valid", evidence["stage202_protocol_valid"] is True),
        (
            "stage202_status_frozen",
            evidence["stage202_status"] == "stage202_top1_joint_objective_protocol_frozen",
        ),
        (
            "two_stage_research_recommended",
            evidence["recommended_next_research"]
            == "separate_change_abstain_head_and_conditional_strict_ranker_protocol",
        ),
        (
            "strict_to_baseline_is_dominant_change",
            evidence["dominant_precision_outcome_change"] == "strict_success__to__baseline",
        ),
        ("finding_not_claimed_causal", evidence["finding_is_causal_claim"] is False),
        ("strict_opportunity_population_exact", evidence["strict_opportunity_count"] == 1439),
        ("mean_strict_actions_observed", evidence["mean_strict_actions_per_question"] > 8.0),
        ("target_mass_exact", evidence["target_mass_exact"] is True),
        ("precision_lost_strict", evidence["precision_net_strict_count"] == -1934),
        ("precision_added_baseline", evidence["precision_net_baseline_count"] == 3154),
        ("five_source_trajectories", len(evidence["source_trajectories"]) == 5),
        ("source_controls_exact", evidence["all_control_reproductions_exact"] is True),
        ("pool_cap_is_16", pool["pool_cap"] == 16),
        ("pool_builder_fixed", pool["pool_builder_researched_or_retuned"] is False),
        (
            "baseline_excluded_from_ranker_fit",
            pool["baseline_excluded_from_conditional_ranker_fit"] is True,
        ),
        (
            "baseline_excluded_from_ranker_softmax",
            pool["baseline_excluded_from_conditional_ranker_softmax"] is True,
        ),
        (
            "baseline_excluded_from_ranker_winner",
            pool["baseline_excluded_from_conditional_ranker_winner"] is True,
        ),
        ("no_old_frontier", pool["source_risk_frontier_applied_to_two_stage_candidates"] is False),
        ("no_runtime_gold_filter", pool["runtime_gold_filter_used"] is False),
        ("two_ranker_families", ranker["family_count"] == 2),
        (
            "ranker_family_names_exact",
            [row["name"] for row in ranker["families"]] == list(RANKER_FAMILIES),
        ),
        ("ranker_is_grouped", ranker["question_grouped"] is True),
        ("ranker_is_deterministic", ranker["deterministic"] is True),
        (
            "raw_ranker_scores_excluded_from_gate",
            ranker["raw_absolute_ranker_score_exported_to_gate"] is False
            and gate["raw_absolute_ranker_scores_used"] is False,
        ),
        ("gate_target_is_cross_fitted", gate["training_winners_are_ranker_oof_only"] is True),
        (
            "same_fit_winner_forbidden",
            gate["same_fit_ranker_winner_used_for_gate_training"] is False,
        ),
        ("gate_has_no_gold_features", gate["gold_or_outcome_features_used"] is False),
        ("coverage_values_exact", gate["target_change_coverages"] == list(TARGET_CHANGE_COVERAGES)),
        (
            "threshold_training_only",
            gate["threshold_learned_inside_each_training_partition"] is True,
        ),
        ("no_probability_calibration_claim", gate["probability_calibration_claimed"] is False),
        (
            "gate_oof_one_row_per_question",
            crossfit["gate_training_has_exactly_one_oof_winner_per_question"] is True,
        ),
        (
            "gate_question_leakage_forbidden",
            crossfit["question_overlap_between_gate_fit_and_ranker_prediction_fold"] is False,
        ),
        (
            "outer_refit_repeats_crossfit",
            crossfit["outer_refit_repeats_same_crossfit_pipeline"] is True,
        ),
        (
            "crossfit_assignment_deterministic",
            crossfit["gate_crossfit_assignment_is_deterministic"] is True,
        ),
        ("ten_two_stage_policies", factorial["two_stage_policy_count"] == 10),
        ("eleven_candidate_configs", factorial["candidate_config_count_per_outer_context"] == 11),
        (
            "paired_control_deltas",
            factorial["paired_deltas_against_exact_control_required"] is True,
        ),
        ("twenty_inner_partitions", cv["inner_partition_count"] == 20),
        ("sixteen_fits_per_inner", cv["model_fits_per_inner_partition"] == 16),
        ("maximum_fit_count_is_370", cv["maximum_model_fit_count"] == 370),
        ("maximum_tree_count_is_96000", cv["maximum_lightgbm_tree_count"] == 96_000),
        ("all_validity_guards_enabled", all(validity.values())),
        (
            "thirteen_inner_constraints",
            len(protocol["inner_selection"]["eligibility_constraints"]) == 13,
        ),
        (
            "inner_thresholds_unchanged",
            protocol["inner_selection"]["thresholds_unchanged_from_stage196"] is True,
        ),
        (
            "no_weaker_substitution",
            protocol["inner_selection"]["weaker_ineligible_candidate_substitution"] is False,
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
        ("stage206_authorized", authorization["stage206_train_only_experiment_authorized"] is True),
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
        ("no_sklearn_import", boundaries["sklearn_imported"] is False),
        ("no_dependency_change", boundaries["dependency_installed_or_changed"] is False),
        ("zero_model_fits", boundaries["model_fit_count"] == 0),
        ("zero_private_rows", boundaries["private_action_rows_materialized"] == 0),
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
            "stage205_two_stage_change_ranker_protocol_frozen"
            if valid
            else "stage205_two_stage_change_ranker_protocol_invalid"
        ),
        "protocol_valid": valid,
        "stage206_train_only_experiment_authorized": valid,
        "development_opened": False,
        "test_opened": False,
        "runtime_e2e_authorized": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "default_runtime_activation": False,
    }


def write_stage205_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage205Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = report["evidence_summary"]
    protocol = report["frozen_protocol"]
    cv = protocol["cross_validation"]
    charts = {
        "stage205_source_transition.svg": _chart(
            "Stage 205 source failure transition",
            [
                _count_bar(
                    "control custom net strict", evidence["control_custom_net_strict_count"]
                ),
                _count_bar(
                    "control custom net unsafe", evidence["control_custom_net_unsafe_count"]
                ),
                _count_bar(
                    "control custom net baseline", evidence["control_custom_net_baseline_count"]
                ),
                _count_bar("precision net strict", evidence["precision_net_strict_count"]),
                _count_bar("precision net unsafe", evidence["precision_net_unsafe_count"]),
                _count_bar("precision net baseline", evidence["precision_net_baseline_count"]),
            ],
        ),
        "stage205_opportunity.svg": _chart(
            "Stage 205 strict opportunity evidence",
            [
                _count_bar("strict opportunity", evidence["strict_opportunity_count"]),
                _count_bar("no strict opportunity", evidence["no_strict_opportunity_count"]),
                _rate_bar("mean strict actions", evidence["mean_strict_actions_per_question"]),
            ],
        ),
        "stage205_architecture.svg": _chart(
            "Stage 205 frozen two-stage architecture",
            [
                _bool_bar(
                    "baseline excluded from ranker",
                    protocol["candidate_pool_contract"][
                        "baseline_excluded_from_conditional_ranker_fit"
                    ],
                ),
                _bool_bar("ranker winner feeds gate", True),
                _bool_bar(
                    "gate winner is cross-fitted",
                    protocol["change_abstain_gate"]["training_winners_are_ranker_oof_only"],
                ),
                _bool_bar(
                    "baseline is abstention",
                    protocol["change_abstain_gate"]["baseline_is_intentional_abstention"],
                ),
            ],
        ),
        "stage205_ranker_families.svg": _chart(
            "Stage 205 conditional ranker labels",
            [
                BarDatum(row["name"], float(max(row["label_gain"])), str(row["label_gain"]))
                for row in protocol["conditional_ranker"]["families"]
            ],
        ),
        "stage205_change_coverages.svg": _chart(
            "Stage 205 target change coverage grid",
            [_rate_bar(f"coverage {value:.2f}", value) for value in TARGET_CHANGE_COVERAGES],
        ),
        "stage205_candidate_grid.svg": _chart(
            "Stage 205 candidate configuration grid",
            [
                _count_bar(
                    "ranker families", protocol["factorial_ablation"]["ranker_family_count"]
                ),
                _count_bar(
                    "coverage levels",
                    protocol["factorial_ablation"]["target_change_coverage_count"],
                ),
                _count_bar(
                    "two-stage policies", protocol["factorial_ablation"]["two_stage_policy_count"]
                ),
                _count_bar(
                    "exact controls", protocol["factorial_ablation"]["exact_source_control_count"]
                ),
            ],
        ),
        "stage205_crossfit.svg": _chart(
            "Stage 205 leakage-resistant cross-fitting",
            [
                _count_bar("outer folds", cv["outer_fold_count"]),
                _count_bar("inner folds", cv["inner_fold_count"]),
                _count_bar("gate crossfit folds", GATE_CROSSFIT_FOLDS),
                _bool_bar(
                    "OOF gate winners only",
                    protocol["cross_fitting_contract"][
                        "gate_training_has_exactly_one_oof_winner_per_question"
                    ],
                ),
            ],
        ),
        "stage205_fit_budget.svg": _chart(
            "Stage 205 maximum Stage 206 budget",
            [
                _count_bar("inner partitions", cv["inner_partition_count"]),
                _count_bar("fits per inner", cv["model_fits_per_inner_partition"]),
                _count_bar("maximum fits", cv["maximum_model_fit_count"]),
                _count_bar("maximum LightGBM trees", cv["maximum_lightgbm_tree_count"]),
            ],
        ),
        "stage205_authorization.svg": _chart(
            "Stage 205 authorization boundary",
            [_bool_bar(name, value) for name, value in protocol["authorization_boundary"].items()],
            margin_left=760,
        ),
        "stage205_guard_checks.svg": _chart(
            "Stage 205 guard checks",
            [_bool_bar(row["name"], row["passed"]) for row in report["guard_checks"]],
            margin_left=760,
        ),
    }
    visualizations = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        visualizations.append(Stage205Visualization(name, str(path)))
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
        margin_right=260,
    )


def _count_bar(name: str, value: int) -> BarDatum:
    return BarDatum(name, float(value), str(value))


def _rate_bar(name: str, value: float) -> BarDatum:
    return BarDatum(name, float(value), f"{value:.6f}")


def _bool_bar(name: str, value: bool) -> BarDatum:
    return BarDatum(name, float(value), str(value).lower())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(path),
        "resolved_size_bytes": resolved.stat().st_size,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


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
