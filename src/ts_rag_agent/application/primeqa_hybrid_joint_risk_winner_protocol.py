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

STAGE = "Stage 198"
CREATED_AT = "2026-07-27"
PROTOCOL_ID = "primeqa_hybrid_joint_risk_winner_protocol_v1"
NEXT_STAGE = "Stage 199"
STAGE197_SHA256 = "c56f4af1b408a07e295a10f7decd2c8a0313f814f16955fd149a170355646d9d"
RISK_SIGNAL_FAMILIES = (
    "source_weighted_classifier",
    "decomposed_loss_risk",
    "pairwise_safety_ranker",
    "decomposed_pairwise_rank_fusion",
)
RANK_RISK_PENALTIES = (0.25, 0.5, 1.0, 2.0)
GAIN_SHORTLIST_SIZES = (2, 4)
OUTER_FOLDS = 5
INNER_FOLDS = 4
MODELS_PER_PARTITION = 5
LIGHTGBM_MODELS_PER_PARTITION = 3
TREES_PER_LIGHTGBM_MODEL = 300
MINIMUM_AVAILABLE_MEMORY_GIB = 4.0
FORBIDDEN_PUBLIC_KEYS = {
    "action_id",
    "candidate_actions",
    "complete_pool",
    "document_text",
    "feature_rows",
    "frontier",
    "predictions",
    "question_key",
    "question_text",
    "selected_actions",
}


@dataclass(frozen=True)
class Stage198Visualization:
    name: str
    path: str


def freeze_joint_risk_winner_protocol(
    *,
    stage197_report_path: Path,
    user_confirmed: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage 199 train-only joint risk-signal and winner-rule experiment."""

    started_at = time.perf_counter()
    source = _load_json(stage197_report_path)
    source_file = _fingerprint(stage197_report_path)
    loaded_at = time.perf_counter()
    evidence = _evidence_summary(source)
    protocol = _frozen_protocol(evidence["source_trajectories"])
    preliminary: dict[str, Any] = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "protocol_id": PROTOCOL_ID,
        "protocol_scope": (
            "Aggregate-only freeze for a train-only factorial ablation of risk signals "
            "and final winner rules. Stage198 reads only the public Stage197 report, "
            "loads no split rows or documents, imports no LightGBM runtime, fits no model, "
            "evaluates no policy, opens no development or test data, adds no fallback, "
            "and changes no runtime default."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed),
            "selected_route": "A_joint_risk_signal_and_winner_rule_factorial",
            "confirmation_note": confirmation_note,
        },
        "source_file": source_file,
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


def _frozen_protocol(source_trajectories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    winner_policies = _winner_policies()
    policy_count = len(RISK_SIGNAL_FAMILIES) * len(winner_policies)
    total_partition_count = OUTER_FOLDS * INNER_FOLDS + OUTER_FOLDS
    return {
        "experiment_name": "train_only_joint_risk_signal_winner_rule_nested_cv",
        "split_contract": {
            "selection_split": "train",
            "outer_fold_count": OUTER_FOLDS,
            "inner_fold_count": INNER_FOLDS,
            "all_actions_for_one_question_remain_in_one_fold": True,
            "sequential_train_research_not_claimed_as_unbiased_final_test": True,
            "development_opened": False,
            "test_opened": False,
        },
        "source_trajectory_contract": {
            "source": "published Stage196 top-inner spec carried through Stage197",
            "outer_context_count": len(source_trajectories),
            "trajectories": list(source_trajectories),
            "pool_builder_fixed_per_outer_context": True,
            "gain_ranker_fixed_per_outer_context": True,
            "safest_prefix_fixed_per_outer_context": True,
            "source_classifier_weight_fixed_per_outer_context": True,
            "old_factor_research_reopened": False,
            "control_reproduces_stage196_top_inner": True,
        },
        "risk_signal_factor": {
            "families": list(RISK_SIGNAL_FAMILIES),
            "family_count": len(RISK_SIGNAL_FAMILIES),
            "source_weighted_classifier": {
                "role": "exact Stage196 control",
                "estimator": "lightgbm.LGBMClassifier",
                "objective": "binary",
                "positive_label": "unsafe",
                "representation_profile_and_weight": "fixed by source trajectory",
                "absolute_probability_threshold_used": False,
            },
            "decomposed_loss_risk": {
                "score": "max(citation_loss_probability, f1_loss_probability)",
                "uses_existing_source_pool_safety_heads": True,
                "additional_model_fit_count": 0,
                "absolute_probability_threshold_used": False,
            },
            "pairwise_safety_ranker": {
                "estimator": "lightgbm.LGBMRanker",
                "objective": "lambdarank",
                "question_grouped": True,
                "relevance_labels": {"unsafe": 0, "non_unsafe": 1},
                "label_gain": [0, 1],
                "lambdarank_truncation_level": 16,
                "representation_and_tree_profile": "fixed from source risk trajectory",
                "higher_model_score_means_safer": True,
                "heldout_labels_used_for_fit_or_early_stopping": False,
            },
            "decomposed_pairwise_rank_fusion": {
                "inputs": ["decomposed_loss_risk", "pairwise_safety_ranker"],
                "method": "mean deterministic within-question normalized risk-rank fraction",
                "additional_model_fit_count": 0,
                "score_calibration_required": False,
            },
            "all_signals_used_only_as_within_question_order": True,
        },
        "winner_rule_factor": {
            "policies": winner_policies,
            "policy_count": len(winner_policies),
            "gain_only_control_count": 1,
            "rank_utility_penalties": list(RANK_RISK_PENALTIES),
            "gain_shortlist_sizes": list(GAIN_SHORTLIST_SIZES),
            "raw_gain_and_risk_scores_never_added": True,
            "runtime_gold_filter_used": False,
        },
        "frontier_contract": {
            "source_pool_cap": 16,
            "risk_signal_rebuilds_safest_prefix": True,
            "prefix_size_fixed_by_source_trajectory": True,
            "baseline_unioned_after_prefix": True,
            "control_risk_and_control_winner_exactly_reproduce_source": True,
            "canonical_action_order_final_tie_break": True,
            "retry_used": False,
            "fallback_used": False,
        },
        "factorial_ablation": {
            "risk_signal_count": len(RISK_SIGNAL_FAMILIES),
            "winner_rule_count": len(winner_policies),
            "policy_config_count_per_outer_context": policy_count,
            "control_cell": "source_weighted_classifier x gain_only",
            "risk_only_cells": "alternate risk signal x gain_only",
            "winner_only_cells": "source weighted classifier x alternate winner rule",
            "joint_cells": "alternate risk signal x alternate winner rule",
            "factor_aggregates_required": True,
            "paired_deltas_against_control_required": True,
            "models_shared_across_policy_configs": True,
        },
        "cross_validation": {
            "outer_fold_count": OUTER_FOLDS,
            "inner_fold_count": INNER_FOLDS,
            "inner_partition_count": OUTER_FOLDS * INNER_FOLDS,
            "maximum_outer_refit_count": OUTER_FOLDS,
            "source_pool_safety_fits_per_partition": 2,
            "source_gain_ranker_fits_per_partition": 1,
            "source_unsafe_classifier_fits_per_partition": 1,
            "pairwise_safety_ranker_fits_per_partition": 1,
            "model_fits_per_partition": MODELS_PER_PARTITION,
            "maximum_model_fit_count": total_partition_count * MODELS_PER_PARTITION,
            "lightgbm_models_per_partition": LIGHTGBM_MODELS_PER_PARTITION,
            "maximum_lightgbm_tree_count": (
                total_partition_count * LIGHTGBM_MODELS_PER_PARTITION * TREES_PER_LIGHTGBM_MODEL
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
            "minimum_preflight_system_available_memory_gib": MINIMUM_AVAILABLE_MEMORY_GIB,
            "cpu_device": True,
            "physical_cpu_threads": 8,
            "one_risk_model_materialized_at_a_time": True,
            "event_driven_resource_statistics_required": True,
            "process_wait_contract": (
                "one PowerShell Wait-Process call for the formal PID until natural exit"
            ),
            "insufficient_memory_behavior": (
                "do not start; request resource clearance instead of reducing the grid"
            ),
        },
        "authorization_boundary": {
            "stage199_train_only_experiment_authorized": True,
            "development_evaluation_authorized": False,
            "test_evaluation_authorized": False,
            "full_train_policy_selection_authorized": False,
            "runtime_e2e_authorized": False,
            "replacement_policy_selection_authorized": False,
            "default_runtime_activation_authorized": False,
            "stage178b_authorized": False,
        },
    }


def _winner_policies() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "name": "gain_only",
            "family": "control",
            "order": ["descending gain score", "ascending risk", "canonical action order"],
        }
    ]
    rows.extend(
        {
            "name": f"rank_utility_{penalty:.2f}",
            "family": "rank_utility",
            "risk_penalty": penalty,
            "objective": (
                "minimize normalized gain-rank fraction + risk_penalty * "
                "normalized risk-rank fraction"
            ),
            "tie_break": ["ascending risk", "descending gain", "canonical action order"],
        }
        for penalty in RANK_RISK_PENALTIES
    )
    rows.extend(
        {
            "name": f"gain_shortlist_{size}_then_risk",
            "family": "gain_shortlist_then_risk",
            "gain_shortlist_size": size,
            "selection": (
                "take top-k frontier actions by gain, then choose lowest risk within shortlist"
            ),
            "baseline_forced_into_shortlist": False,
            "tie_break": ["descending gain", "canonical action order"],
        }
        for size in GAIN_SHORTLIST_SIZES
    )
    return rows


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
    attribution = report["surviving_unsafe_winner_attribution"]
    aggregate = attribution["aggregate"]
    metrics = attribution["unsafe_head_prediction_metrics"]
    trajectories = [
        {
            "outer_context": outer_context,
            "source_spec": row["spec"],
            "top_inner_reconstruction_exact": row["top_inner_reconstruction_exact"],
        }
        for outer_context, row in sorted(attribution["outer_contexts"].items())
    ]
    return {
        "source_status": report["decision"]["status"],
        "source_experiment_valid": report["decision"]["experiment_valid"],
        "source_diagnostic_complete": report["decision"]["diagnostic_complete"],
        "source_development_opened": report["decision"]["development_opened"],
        "source_test_opened": report["decision"]["test_opened"],
        "source_process_guard_count": len(report["process_guards"]),
        "source_process_guards_passed": sum(row["passed"] for row in report["process_guards"]),
        "question_context_count": aggregate["question_context_count"],
        "unsafe_winner_context_count": aggregate["unsafe_winner_context_count"],
        "unsafe_winner_rate": aggregate["unsafe_winner_rate"],
        "mechanism_counts": aggregate["mechanism_counts"],
        "mechanism_partition_exact": aggregate["mechanism_partition_exact"],
        "dominant_mechanism": aggregate["dominant_mechanism"],
        "unsafe_head_roc_auc": metrics["roc_auc"],
        "unsafe_head_average_precision": metrics["average_precision"],
        "source_trajectories": trajectories,
        "all_source_trajectories_exact": all(
            row["top_inner_reconstruction_exact"] for row in trajectories
        ),
    }


def _guard_checks(
    preliminary: Mapping[str, Any], source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    evidence = preliminary["evidence_summary"]
    protocol = preliminary["frozen_protocol"]
    boundaries = preliminary["execution_boundaries"]
    risk = protocol["risk_signal_factor"]
    winner = protocol["winner_rule_factor"]
    cv = protocol["cross_validation"]
    authorization = protocol["authorization_boundary"]
    checks = (
        ("user_confirmed_route_a", preliminary["user_confirmation"]["confirmed"] is True),
        ("source_sha256_matches", preliminary["source_file"]["sha256"] == STAGE197_SHA256),
        ("source_is_stage197", source.get("stage") == "Stage 197"),
        ("source_experiment_valid", evidence["source_experiment_valid"] is True),
        ("source_diagnostic_complete", evidence["source_diagnostic_complete"] is True),
        (
            "source_status_complete",
            evidence["source_status"] == "stage197_surviving_unsafe_winner_attribution_complete",
        ),
        ("source_development_closed", evidence["source_development_opened"] is False),
        ("source_test_closed", evidence["source_test_opened"] is False),
        (
            "source_all_guards_passed",
            evidence["source_process_guard_count"] == 29
            and evidence["source_process_guards_passed"] == 29,
        ),
        ("source_mechanism_partition_exact", evidence["mechanism_partition_exact"] is True),
        ("source_has_unsafe_winners", evidence["unsafe_winner_context_count"] > 0),
        (
            "gain_dominance_observed",
            evidence["mechanism_counts"].get("final_gain_dominance", 0) > 0,
        ),
        (
            "risk_side_failures_observed",
            evidence["mechanism_counts"].get("risk_ordering_failure", 0)
            + evidence["mechanism_counts"].get("risk_frontier_exclusion", 0)
            > 0,
        ),
        ("unsafe_head_not_perfect", evidence["unsafe_head_roc_auc"] < 1.0),
        ("five_source_trajectories", len(evidence["source_trajectories"]) == 5),
        ("source_trajectories_exact", evidence["all_source_trajectories_exact"] is True),
        (
            "old_factors_fixed",
            protocol["source_trajectory_contract"]["old_factor_research_reopened"] is False,
        ),
        (
            "control_reproduces_source",
            protocol["source_trajectory_contract"]["control_reproduces_stage196_top_inner"] is True,
        ),
        ("four_risk_signals", risk["family_count"] == 4),
        ("risk_signal_names_exact", risk["families"] == list(RISK_SIGNAL_FAMILIES)),
        (
            "classifier_control_present",
            risk["source_weighted_classifier"]["role"] == "exact Stage196 control",
        ),
        (
            "decomposed_risk_reuses_heads",
            risk["decomposed_loss_risk"]["additional_model_fit_count"] == 0,
        ),
        (
            "pairwise_ranker_question_grouped",
            risk["pairwise_safety_ranker"]["question_grouped"] is True,
        ),
        (
            "pairwise_ranker_binary_relevance",
            risk["pairwise_safety_ranker"]["relevance_labels"] == {"unsafe": 0, "non_unsafe": 1},
        ),
        (
            "pairwise_no_heldout_fit",
            risk["pairwise_safety_ranker"]["heldout_labels_used_for_fit_or_early_stopping"]
            is False,
        ),
        (
            "fusion_is_rank_based",
            "normalized risk-rank" in risk["decomposed_pairwise_rank_fusion"]["method"],
        ),
        (
            "no_absolute_risk_threshold",
            risk["all_signals_used_only_as_within_question_order"] is True,
        ),
        ("seven_winner_rules", winner["policy_count"] == 7),
        ("one_gain_control", winner["gain_only_control_count"] == 1),
        ("rank_penalties_exact", winner["rank_utility_penalties"] == list(RANK_RISK_PENALTIES)),
        ("shortlist_sizes_exact", winner["gain_shortlist_sizes"] == list(GAIN_SHORTLIST_SIZES)),
        ("raw_scores_not_added", winner["raw_gain_and_risk_scores_never_added"] is True),
        (
            "factorial_grid_is_28",
            protocol["factorial_ablation"]["policy_config_count_per_outer_context"] == 28,
        ),
        (
            "paired_control_deltas_required",
            protocol["factorial_ablation"]["paired_deltas_against_control_required"] is True,
        ),
        (
            "models_shared",
            protocol["factorial_ablation"]["models_shared_across_policy_configs"] is True,
        ),
        ("twenty_inner_partitions", cv["inner_partition_count"] == 20),
        ("five_models_per_partition", cv["model_fits_per_partition"] == 5),
        ("maximum_fit_count_is_125", cv["maximum_model_fit_count"] == 125),
        ("maximum_tree_count_is_22500", cv["maximum_lightgbm_tree_count"] == 22_500),
        (
            "no_weaker_substitution",
            protocol["inner_selection"]["weaker_ineligible_candidate_substitution"] is False,
        ),
        (
            "inner_thresholds_unchanged",
            protocol["inner_selection"]["thresholds_unchanged_from_stage196"] is True,
        ),
        ("advancement_gate_count_is_17", len(protocol["advancement_gates"]) == 17),
        (
            "memory_threshold_is_4_gib",
            protocol["resource_contract"]["minimum_preflight_system_available_memory_gib"] == 4.0,
        ),
        ("stage199_authorized", authorization["stage199_train_only_experiment_authorized"] is True),
        ("development_not_authorized", authorization["development_evaluation_authorized"] is False),
        ("test_not_authorized", authorization["test_evaluation_authorized"] is False),
        (
            "full_train_not_authorized",
            authorization["full_train_policy_selection_authorized"] is False,
        ),
        ("runtime_e2e_not_authorized", authorization["runtime_e2e_authorized"] is False),
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
            "stage198_joint_risk_winner_protocol_frozen"
            if valid
            else "stage198_joint_risk_winner_protocol_invalid"
        ),
        "protocol_valid": valid,
        "stage199_train_only_experiment_authorized": valid,
        "development_opened": False,
        "test_opened": False,
        "runtime_e2e_authorized": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "default_runtime_activation": False,
    }


def write_stage198_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage198Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = report["evidence_summary"]
    protocol = report["frozen_protocol"]
    cv = protocol["cross_validation"]
    charts = {
        "stage198_source_mechanisms.svg": _chart(
            "Stage 198 source unsafe-winner mechanisms",
            [_count_bar(name, value) for name, value in evidence["mechanism_counts"].items()],
        ),
        "stage198_source_risk_metrics.svg": _chart(
            "Stage 198 source unsafe-head metrics",
            [
                _rate_bar("ROC AUC", evidence["unsafe_head_roc_auc"]),
                _rate_bar("average precision", evidence["unsafe_head_average_precision"]),
            ],
        ),
        "stage198_factor_counts.svg": _chart(
            "Stage 198 frozen factorial counts",
            [
                _count_bar("risk signals", protocol["factorial_ablation"]["risk_signal_count"]),
                _count_bar("winner rules", protocol["factorial_ablation"]["winner_rule_count"]),
                _count_bar(
                    "policy cells per outer",
                    protocol["factorial_ablation"]["policy_config_count_per_outer_context"],
                ),
            ],
        ),
        "stage198_rank_penalties.svg": _chart(
            "Stage 198 frozen rank-risk penalties",
            [_rate_bar(f"lambda {value:.2f}", value) for value in RANK_RISK_PENALTIES],
        ),
        "stage198_shortlist_sizes.svg": _chart(
            "Stage 198 frozen gain shortlist sizes",
            [_count_bar(f"top {value}", value) for value in GAIN_SHORTLIST_SIZES],
        ),
        "stage198_fit_budget.svg": _chart(
            "Stage 198 maximum fit budget",
            [
                _count_bar("inner partitions", cv["inner_partition_count"]),
                _count_bar("fits per partition", cv["model_fits_per_partition"]),
                _count_bar("maximum fits", cv["maximum_model_fit_count"]),
                _count_bar("maximum LightGBM trees", cv["maximum_lightgbm_tree_count"]),
            ],
        ),
        "stage198_source_trajectories.svg": _chart(
            "Stage 198 frozen source trajectories",
            [
                _count_bar(row["outer_context"], int(row["top_inner_reconstruction_exact"]))
                for row in evidence["source_trajectories"]
            ],
        ),
        "stage198_advancement_gates.svg": _chart(
            "Stage 198 retained advancement thresholds",
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
        "stage198_authorization.svg": _chart(
            "Stage 198 authorization boundary",
            [_bool_bar(name, value) for name, value in protocol["authorization_boundary"].items()],
            margin_left=720,
        ),
        "stage198_guard_checks.svg": _chart(
            "Stage 198 guard checks",
            [_bool_bar(row["name"], row["passed"]) for row in report["guard_checks"]],
            margin_left=800,
        ),
    }
    visualizations = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        visualizations.append(Stage198Visualization(name, str(path)))
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
