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

STAGE = "Stage 200"
CREATED_AT = "2026-07-27"
PROTOCOL_ID = "primeqa_hybrid_joint_risk_winner_failure_attribution_protocol_v1"
NEXT_STAGE = "Stage 201"
STAGE199_SHA256 = "5b933f524fff1bceb4d4d842e4f3a1aec3160aa3ed337131444ec1b7c2699fee"
OUTER_CONTEXT_COUNT = 5
INNER_FOLD_COUNT = 4
RISK_SIGNAL_COUNT = 4
WINNER_RULE_COUNT = 7
POLICY_CELL_COUNT = RISK_SIGNAL_COUNT * WINNER_RULE_COUNT
INNER_PARTITION_COUNT = OUTER_CONTEXT_COUNT * INNER_FOLD_COUNT
MODEL_FITS_PER_PARTITION = 5
LIGHTGBM_MODELS_PER_PARTITION = 3
TREES_PER_LIGHTGBM_MODEL = 300
MINIMUM_AVAILABLE_MEMORY_GIB = 4.0
FORBIDDEN_PUBLIC_KEYS = {
    "action_id",
    "candidate_actions",
    "complete_pool",
    "document_id",
    "document_text",
    "feature_rows",
    "frontier",
    "prediction_rows",
    "question_key",
    "question_text",
    "selected_action",
}


@dataclass(frozen=True)
class Stage200Visualization:
    name: str
    path: str


def freeze_joint_risk_winner_failure_attribution_protocol(
    *,
    stage199_report_path: Path,
    user_confirmed: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage 201 train-only eligibility-failure attribution."""

    started_at = time.perf_counter()
    source = _load_json(stage199_report_path)
    source_file = _fingerprint(stage199_report_path)
    loaded_at = time.perf_counter()
    evidence = _evidence_summary(source)
    protocol = _frozen_protocol(evidence)
    preliminary: dict[str, Any] = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "protocol_id": PROTOCOL_ID,
        "protocol_scope": (
            "Aggregate-only freeze for Stage201 train-only attribution of why all "
            "Stage199 joint risk/winner cells failed inner eligibility. Stage200 reads "
            "only the public Stage199 report, loads no split rows or documents, imports "
            "no LightGBM runtime, fits no model, evaluates no new policy, opens no "
            "development or test data, relaxes no constraint, adds no fallback, and "
            "changes no runtime default."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed),
            "selected_route": "train_only_inner_eligibility_failure_attribution",
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
            "private_action_rows_materialized": 0,
            "private_predictions_materialized": 0,
            "diagnostic_oracle_run": False,
            "constraint_relaxation_run": False,
            "new_policy_search_run": False,
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
            "question_level_diagnostics_persisted": False,
            "public_report_safe": not forbidden,
        },
    }


def _frozen_protocol(evidence: Mapping[str, Any]) -> dict[str, Any]:
    outer_cell_context_count = OUTER_CONTEXT_COUNT * POLICY_CELL_COUNT
    question_cell_context_count = evidence["inner_question_context_count"] * POLICY_CELL_COUNT
    return {
        "diagnostic_name": "train_only_joint_risk_winner_inner_eligibility_attribution",
        "split_contract": {
            "selection_split": "train",
            "outer_context_count": OUTER_CONTEXT_COUNT,
            "inner_fold_count": INNER_FOLD_COUNT,
            "all_actions_for_one_question_remain_in_one_fold": True,
            "sequential_train_research_not_claimed_as_unbiased_final_test": True,
            "development_opened": False,
            "test_opened": False,
        },
        "source_reproduction_contract": {
            "source": "Stage199 exact train-only joint risk/winner nested CV",
            "source_report_sha256": STAGE199_SHA256,
            "all_28_cells_rebuilt_in_each_outer_context": True,
            "exact_control_reproduction_required": True,
            "exact_top_inner_ordering_required": True,
            "exact_cell_aggregate_reproduction_required": True,
            "exact_zero_eligible_counts_required": True,
            "source_policy_or_threshold_changed": False,
        },
        "diagnostic_population": {
            "outer_cell_context_count": outer_cell_context_count,
            "fold_cell_context_count": outer_cell_context_count * INNER_FOLD_COUNT,
            "inner_question_context_count": evidence["inner_question_context_count"],
            "question_cell_context_count": question_cell_context_count,
            "policy_cell_count_per_outer_context": POLICY_CELL_COUNT,
            "risk_signal_count": RISK_SIGNAL_COUNT,
            "winner_rule_count": WINNER_RULE_COUNT,
            "all_cells_included": True,
            "top_candidate_only_analysis": False,
        },
        "constraint_attribution": {
            "constraints": _eligibility_constraints(),
            "constraint_count": len(_eligibility_constraints()),
            "signed_margin_convention": (
                "nonnegative passes; lower bounds use observed-threshold; upper bounds "
                "use threshold-observed"
            ),
            "required_aggregates": [
                "failure count and rate by constraint",
                "failure count by outer context, risk signal, and winner rule",
                "pairwise constraint co-failure count and Jaccard matrix",
                "exact failed-constraint-set frequency",
                "failed-constraint-count distribution",
                "signed-margin min, q25, median, q75, max, and near-boundary count",
                "single-constraint-removal pass count",
                "Pareto-nondominated cell count for capture and unsafe rate",
            ],
            "single_constraint_removal_is_diagnostic_only": True,
            "constraint_threshold_relaxation_authorized": False,
        },
        "fold_attribution": {
            "unit": "inner fold x outer context x policy cell",
            "fold_cell_context_count": outer_cell_context_count * INNER_FOLD_COUNT,
            "per_fold_metrics": [
                "gold citation delta",
                "mean F1 delta",
                "pool recall",
                "conditional strict capture",
                "unsafe selection rate",
            ],
            "required_aggregates": [
                "violation counts by held-in fold",
                "worst-fold frequency by metric",
                "cross-fold range and standard deviation by cell",
                "cells passing aggregate but failing minimum-fold-count constraints",
            ],
            "fold_ids_public_but_question_membership_private": True,
        },
        "question_context_attribution": {
            "unit": "inner-OOF question context x policy cell",
            "question_cell_context_count": question_cell_context_count,
            "selected_outcome_partition": [
                "baseline",
                "strict_success",
                "safe_zero",
                "unsafe_citation_only",
                "unsafe_f1_only",
                "unsafe_citation_and_f1",
            ],
            "strict_opportunity_partition": [
                "no_strict_opportunity",
                "safety_pool_exclusion",
                "risk_frontier_exclusion",
                "winner_selection_miss",
                "strict_selected",
            ],
            "ranking_conflict_aggregates": [
                "selected gain and risk ranks",
                "best strict alternative gain and risk ranks",
                "selected-minus-best-strict gain rank gap",
                "selected-minus-best-strict risk rank gap",
                "lower-risk strict alternative availability",
                "higher-gain strict alternative availability",
            ],
            "context_aggregates": [
                "pool size",
                "frontier size",
                "strict opportunity count",
                "unsafe candidate count",
                "selected action family",
                "source outer context",
                "risk signal",
                "winner rule",
            ],
            "gold_labels_used_for_diagnostic_partition_only": True,
            "oracle_used_as_runtime_rule": False,
            "question_text_or_identifier_public": False,
            "question_level_rows_public": False,
        },
        "conclusion_contract": {
            "required_questions": [
                "Which constraints are necessary and sufficient blockers most often?",
                "Are failures dominated by a stable fold or by cross-fold variance?",
                "Which risk/winner factors move capture without reducing unsafe enough?",
                "Which loss type survives when safer winner rules reduce selection rate?",
                "Is the next intervention model, objective, or representation research?",
            ],
            "causal_claims_authorized": False,
            "best_ineligible_cell_may_be_promoted": False,
            "new_candidate_family_may_be_selected": False,
        },
        "execution_budget": {
            "inner_partition_count": INNER_PARTITION_COUNT,
            "model_fits_per_partition": MODEL_FITS_PER_PARTITION,
            "exact_model_fit_count": INNER_PARTITION_COUNT * MODEL_FITS_PER_PARTITION,
            "lightgbm_models_per_partition": LIGHTGBM_MODELS_PER_PARTITION,
            "exact_lightgbm_tree_count": (
                INNER_PARTITION_COUNT * LIGHTGBM_MODELS_PER_PARTITION * TREES_PER_LIGHTGBM_MODEL
            ),
            "exact_private_prediction_count": evidence["source_private_prediction_count"],
            "outer_refit_count": 0,
            "additional_diagnostic_model_fit_count": 0,
            "models_shared_across_28_cells": True,
            "streaming_aggregate_required": True,
            "private_question_cell_rows_persisted": False,
            "retry_used": False,
            "fallback_used": False,
        },
        "resource_contract": {
            "minimum_preflight_system_available_memory_gib": MINIMUM_AVAILABLE_MEMORY_GIB,
            "cpu_device": True,
            "physical_cpu_threads": 8,
            "one_partition_model_bundle_materialized_at_a_time": True,
            "event_driven_resource_statistics_required": True,
            "process_wait_contract": (
                "one PowerShell Wait-Process call for the formal PID until natural exit"
            ),
            "insufficient_memory_behavior": (
                "do not start; request resource clearance instead of reducing diagnostics"
            ),
        },
        "authorization_boundary": {
            "stage201_train_only_attribution_authorized": True,
            "development_evaluation_authorized": False,
            "test_evaluation_authorized": False,
            "new_policy_search_authorized": False,
            "constraint_relaxation_authorized": False,
            "full_train_policy_selection_authorized": False,
            "runtime_e2e_authorized": False,
            "replacement_policy_selection_authorized": False,
            "default_runtime_activation_authorized": False,
            "stage178b_authorized": False,
        },
    }


def _eligibility_constraints() -> list[dict[str, Any]]:
    return [
        _constraint("citation_delta", "evaluation.gold_citation_delta", ">=", 0.0),
        _constraint("mean_f1_delta", "evaluation.mean_f1_delta", ">=", 0.0),
        _constraint(
            "citation_nonregressing_fold_count",
            "evaluation.citation_nonregressing_fold_count",
            ">=",
            3,
        ),
        _constraint(
            "f1_nonregressing_fold_count",
            "evaluation.f1_nonregressing_fold_count",
            ">=",
            3,
        ),
        {
            **_constraint(
                "changed_question_count",
                "evaluation.changed_question_count",
                ">=",
                0.10,
            ),
            "threshold_expression": "ceil(0.10 * inner_question_count)",
        },
        {
            **_constraint(
                "strict_success_count",
                "evaluation.strict_success_count",
                ">=",
                0.08,
            ),
            "threshold_expression": "ceil(0.08 * inner_question_count)",
        },
        _constraint(
            "strict_success_precision",
            "evaluation.strict_success_precision",
            ">=",
            0.65,
        ),
        _constraint(
            "strict_opportunity_pool_recall",
            "diagnostics.strict_opportunity_pool_recall",
            ">=",
            0.95,
        ),
        _constraint(
            "folds_meeting_pool_recall_minimum",
            "diagnostics.folds_meeting_pool_recall_minimum",
            ">=",
            3,
            fold_threshold=0.90,
        ),
        _constraint(
            "conditional_ranker_strict_capture",
            "diagnostics.conditional_ranker_strict_capture",
            ">=",
            0.68,
        ),
        _constraint(
            "folds_meeting_conditional_capture_minimum",
            "diagnostics.folds_meeting_conditional_capture_minimum",
            ">=",
            3,
            fold_threshold=0.60,
        ),
        _constraint(
            "unsafe_selection_rate",
            "diagnostics.unsafe_selection_rate",
            "<=",
            0.25,
        ),
        _constraint(
            "folds_meeting_unsafe_rate_maximum",
            "diagnostics.folds_meeting_unsafe_rate_maximum",
            ">=",
            3,
            fold_threshold=0.35,
        ),
    ]


def _constraint(
    name: str,
    source: str,
    operator: str,
    threshold: int | float,
    *,
    fold_threshold: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "source": source,
        "operator": operator,
        "threshold": threshold,
        "signed_margin": ("observed - threshold" if operator == ">=" else "threshold - observed"),
    }
    if fold_threshold is not None:
        row["per_fold_threshold"] = fold_threshold
        row["required_passing_fold_count"] = 3
    return row


def _evidence_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    cv = report["joint_risk_winner_nested_cv"]
    outer = cv["outer_contexts"]
    top = [row["top_inner_candidates"][0] for row in outer.values()]
    risk_metrics = cv["complete_pool_risk_metrics"]
    return {
        "source_status": report["decision"]["status"],
        "source_experiment_valid": report["decision"]["experiment_valid"],
        "source_candidate_family_accepted": report["decision"]["candidate_family_accepted"],
        "source_development_opened": report["decision"]["development_opened"],
        "source_test_opened": report["decision"]["test_opened"],
        "source_process_guard_count": len(report["process_guards"]),
        "source_process_guards_passed": sum(row["passed"] for row in report["process_guards"]),
        "outer_context_count": len(outer),
        "outer_eligible_counts": {
            name: row["eligible_config_count"] for name, row in sorted(outer.items())
        },
        "all_controls_reproduced_exactly": cv["execution"]["all_controls_reproduced_exactly"],
        "policy_cell_count": len(cv["cell_aggregates"]),
        "risk_signal_count": len(cv["risk_signal_factor_aggregates"]),
        "winner_rule_count": len(cv["winner_rule_factor_aggregates"]),
        "inner_question_context_count": sum(row["inner_question_count"] for row in outer.values()),
        "top_inner_conditional_capture_min": min(
            row["diagnostics"]["conditional_ranker_strict_capture"] for row in top
        ),
        "top_inner_conditional_capture_max": max(
            row["diagnostics"]["conditional_ranker_strict_capture"] for row in top
        ),
        "top_inner_unsafe_rate_min": min(
            row["diagnostics"]["unsafe_selection_rate"] for row in top
        ),
        "top_inner_unsafe_rate_max": max(
            row["diagnostics"]["unsafe_selection_rate"] for row in top
        ),
        "complete_pool_risk_roc_auc": {
            name: row["roc_auc"] for name, row in sorted(risk_metrics.items())
        },
        "source_model_fit_count": cv["execution"]["model_fit_count"],
        "source_tree_count": cv["execution"]["tree_count"],
        "source_private_prediction_count": cv["execution"]["private_prediction_count"],
        "source_advancement_gate_pass_count": cv["advancement_gate_pass_count"],
        "source_advancement_gate_count": len(cv["advancement_gates"]),
    }


def _guard_checks(
    preliminary: Mapping[str, Any], source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    evidence = preliminary["evidence_summary"]
    protocol = preliminary["frozen_protocol"]
    population = protocol["diagnostic_population"]
    attribution = protocol["constraint_attribution"]
    execution = protocol["execution_budget"]
    authorization = protocol["authorization_boundary"]
    boundaries = preliminary["execution_boundaries"]
    checks = (
        ("user_confirmed_stage200", preliminary["user_confirmation"]["confirmed"] is True),
        ("source_sha256_matches", preliminary["source_file"]["sha256"] == STAGE199_SHA256),
        ("source_is_stage199", source.get("stage") == "Stage 199"),
        ("source_experiment_valid", evidence["source_experiment_valid"] is True),
        (
            "source_status_insufficient",
            evidence["source_status"] == "stage199_joint_risk_winner_insufficient",
        ),
        ("source_candidate_rejected", evidence["source_candidate_family_accepted"] is False),
        ("source_development_closed", evidence["source_development_opened"] is False),
        ("source_test_closed", evidence["source_test_opened"] is False),
        (
            "source_all_guards_passed",
            evidence["source_process_guard_count"] == 34
            and evidence["source_process_guards_passed"] == 34,
        ),
        ("source_five_outer_contexts", evidence["outer_context_count"] == 5),
        (
            "source_all_outer_contexts_ineligible",
            list(evidence["outer_eligible_counts"].values()) == [0, 0, 0, 0, 0],
        ),
        ("source_all_controls_exact", evidence["all_controls_reproduced_exactly"] is True),
        ("source_28_cells", evidence["policy_cell_count"] == 28),
        ("source_four_risk_signals", evidence["risk_signal_count"] == 4),
        ("source_seven_winner_rules", evidence["winner_rule_count"] == 7),
        ("source_has_question_contexts", evidence["inner_question_context_count"] == 1480),
        ("source_capture_below_or_near_gate", evidence["top_inner_conditional_capture_min"] < 0.68),
        ("source_unsafe_above_gate", evidence["top_inner_unsafe_rate_min"] > 0.25),
        ("source_fit_count_exact", evidence["source_model_fit_count"] == 100),
        ("source_tree_count_exact", evidence["source_tree_count"] == 18_000),
        (
            "source_private_prediction_count_exact",
            evidence["source_private_prediction_count"] == 245_960,
        ),
        ("population_140_outer_cells", population["outer_cell_context_count"] == 140),
        ("population_560_fold_cells", population["fold_cell_context_count"] == 560),
        (
            "population_41440_question_cells",
            population["question_cell_context_count"] == 41_440,
        ),
        ("all_cells_included", population["all_cells_included"] is True),
        ("not_top_only", population["top_candidate_only_analysis"] is False),
        ("thirteen_constraints", attribution["constraint_count"] == 13),
        (
            "constraint_names_unique",
            len({row["name"] for row in attribution["constraints"]}) == 13,
        ),
        (
            "signed_margin_defined",
            all(row["signed_margin"] for row in attribution["constraints"]),
        ),
        (
            "single_removal_diagnostic_only",
            attribution["single_constraint_removal_is_diagnostic_only"] is True,
        ),
        (
            "constraint_relaxation_not_authorized",
            attribution["constraint_threshold_relaxation_authorized"] is False,
        ),
        (
            "question_partition_complete",
            len(protocol["question_context_attribution"]["selected_outcome_partition"]) == 6,
        ),
        (
            "opportunity_partition_complete",
            len(protocol["question_context_attribution"]["strict_opportunity_partition"]) == 5,
        ),
        (
            "oracle_diagnostic_only",
            protocol["question_context_attribution"]["oracle_used_as_runtime_rule"] is False,
        ),
        (
            "question_rows_private",
            protocol["question_context_attribution"]["question_level_rows_public"] is False,
        ),
        ("exact_reproduction_fit_count", execution["exact_model_fit_count"] == 100),
        ("exact_reproduction_tree_count", execution["exact_lightgbm_tree_count"] == 18_000),
        (
            "exact_reproduction_prediction_count",
            execution["exact_private_prediction_count"] == 245_960,
        ),
        ("no_outer_refit", execution["outer_refit_count"] == 0),
        ("no_additional_diagnostic_fit", execution["additional_diagnostic_model_fit_count"] == 0),
        ("streaming_required", execution["streaming_aggregate_required"] is True),
        (
            "private_question_rows_not_persisted",
            execution["private_question_cell_rows_persisted"] is False,
        ),
        ("execution_no_retry", execution["retry_used"] is False),
        ("execution_no_fallback", execution["fallback_used"] is False),
        (
            "stage201_train_only_authorized",
            authorization["stage201_train_only_attribution_authorized"] is True,
        ),
        ("development_not_authorized", authorization["development_evaluation_authorized"] is False),
        ("test_not_authorized", authorization["test_evaluation_authorized"] is False),
        (
            "new_policy_search_not_authorized",
            authorization["new_policy_search_authorized"] is False,
        ),
        (
            "full_train_not_authorized",
            authorization["full_train_policy_selection_authorized"] is False,
        ),
        ("runtime_not_authorized", authorization["runtime_e2e_authorized"] is False),
        ("default_not_authorized", authorization["default_runtime_activation_authorized"] is False),
        ("loaded_public_only", boundaries["loaded_public_reports_only"] is True),
        ("train_not_loaded", boundaries["train_rows_loaded"] is False),
        ("development_not_loaded", boundaries["development_loaded"] is False),
        ("test_not_loaded", boundaries["test_loaded"] is False),
        ("lightgbm_not_imported", boundaries["lightgbm_imported"] is False),
        ("no_model_fit", boundaries["model_fit_count"] == 0),
        ("no_private_rows", boundaries["private_action_rows_materialized"] == 0),
        ("no_private_predictions", boundaries["private_predictions_materialized"] == 0),
        ("no_diagnostic_oracle_run", boundaries["diagnostic_oracle_run"] is False),
        ("no_constraint_relaxation_run", boundaries["constraint_relaxation_run"] is False),
        ("no_policy_search_run", boundaries["new_policy_search_run"] is False),
        ("no_runtime_e2e", boundaries["runtime_e2e_run"] is False),
        ("no_runtime_default", boundaries["runtime_registered_as_default"] is False),
        ("no_retry", boundaries["retry_action_count"] == 0),
        ("no_fallback", boundaries["fallback_action_count"] == 0),
    )
    return [_gate(name, passed) for name, passed in checks]


def _decision(guards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = bool(guards) and all(row["passed"] for row in guards)
    return {
        "status": (
            "stage200_joint_risk_winner_failure_attribution_protocol_frozen"
            if valid
            else "stage200_joint_risk_winner_failure_attribution_protocol_invalid"
        ),
        "protocol_valid": valid,
        "stage201_train_only_attribution_authorized": valid,
        "development_opened": False,
        "test_opened": False,
        "new_policy_search_authorized": False,
        "constraint_relaxation_authorized": False,
        "runtime_e2e_authorized": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "default_runtime_activation": False,
    }


def write_stage200_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage200Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = report["evidence_summary"]
    protocol = report["frozen_protocol"]
    population = protocol["diagnostic_population"]
    execution = protocol["execution_budget"]
    constraints = protocol["constraint_attribution"]["constraints"]
    boundaries = report["execution_boundaries"]
    charts = {
        "stage200_source_outer_eligible.svg": _chart(
            "Stage 200 source eligible cells by outer context",
            [_count_bar(name, value) for name, value in evidence["outer_eligible_counts"].items()],
        ),
        "stage200_source_top_capture_range.svg": _chart(
            "Stage 200 source top-inner capture range",
            [
                _rate_bar("minimum", evidence["top_inner_conditional_capture_min"]),
                _rate_bar("required", 0.68),
                _rate_bar("maximum", evidence["top_inner_conditional_capture_max"]),
            ],
        ),
        "stage200_source_top_unsafe_range.svg": _chart(
            "Stage 200 source top-inner unsafe-rate range",
            [
                _rate_bar("required maximum", 0.25),
                _rate_bar("minimum", evidence["top_inner_unsafe_rate_min"]),
                _rate_bar("maximum", evidence["top_inner_unsafe_rate_max"]),
            ],
        ),
        "stage200_source_risk_auc.svg": _chart(
            "Stage 200 source complete-pool unsafe ROC AUC",
            [
                _rate_bar(name, value)
                for name, value in evidence["complete_pool_risk_roc_auc"].items()
            ],
            margin_left=620,
        ),
        "stage200_diagnostic_populations.svg": _chart(
            "Stage 200 frozen diagnostic populations",
            [
                _count_bar("outer-cell contexts", population["outer_cell_context_count"]),
                _count_bar("fold-cell contexts", population["fold_cell_context_count"]),
                _count_bar("question-cell contexts", population["question_cell_context_count"]),
            ],
        ),
        "stage200_constraint_operators.svg": _chart(
            "Stage 200 frozen eligibility constraints",
            [
                _count_bar(f"{row['name']} {row['operator']}", index)
                for index, row in enumerate(constraints, start=1)
            ],
            margin_left=760,
        ),
        "stage200_execution_budget.svg": _chart(
            "Stage 200 frozen Stage 201 execution budget",
            [
                _count_bar("model fits", execution["exact_model_fit_count"]),
                _count_bar("LightGBM trees", execution["exact_lightgbm_tree_count"]),
                _count_bar("private predictions", execution["exact_private_prediction_count"]),
            ],
        ),
        "stage200_authorization.svg": _chart(
            "Stage 200 authorization boundary",
            [_bool_bar(name, value) for name, value in protocol["authorization_boundary"].items()],
            margin_left=720,
        ),
        "stage200_execution_boundaries.svg": _chart(
            "Stage 200 observed execution boundaries",
            [
                _count_bar("model fits", boundaries["model_fit_count"]),
                _count_bar("private action rows", boundaries["private_action_rows_materialized"]),
                _count_bar("private predictions", boundaries["private_predictions_materialized"]),
                _count_bar("retry actions", boundaries["retry_action_count"]),
                _count_bar("fallback actions", boundaries["fallback_action_count"]),
            ],
        ),
        "stage200_guard_checks.svg": _chart(
            "Stage 200 protocol guard checks",
            [_bool_bar(row["name"], row["passed"]) for row in report["guard_checks"]],
            margin_left=760,
        ),
    }
    visualizations = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        visualizations.append(Stage200Visualization(name, str(path)))
    return tuple(visualizations)


def _fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_size": path.stat().st_size,
    }


def _chart(title: str, bars: Sequence[BarDatum], *, margin_left: int = 520) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label="aggregate value",
        width=1680,
        margin_left=margin_left,
        margin_right=220,
    )


def _count_bar(name: str, value: int | float) -> BarDatum:
    return BarDatum(name, float(value), str(value))


def _rate_bar(name: str, value: float) -> BarDatum:
    return BarDatum(name, value, f"{value:.6f}")


def _bool_bar(name: str, value: bool) -> BarDatum:
    return BarDatum(name, float(value), str(value).lower())


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


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}
