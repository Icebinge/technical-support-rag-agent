from __future__ import annotations

import gc
import json
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application import composition_two_stage_change_ranker_cv as core
from ts_rag_agent.application import primeqa_hybrid_composition_action_audit as stage181
from ts_rag_agent.application import primeqa_hybrid_composition_dual_target_cv as stage182
from ts_rag_agent.application import primeqa_hybrid_composition_f1_risk_attribution as stage183
from ts_rag_agent.application import primeqa_hybrid_iterative_router_calibration as stage169
from ts_rag_agent.application import primeqa_hybrid_safety_constrained_lambdamart_cv as stage194
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application.composition_action_audit import ActionAuditRow
from ts_rag_agent.application.composition_dual_target_policy import (
    DualTargetPrediction,
    SelectedAction,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

STAGE = "Stage 206"
CREATED_AT = "2026-07-27"
ANALYSIS_ID = "primeqa_hybrid_two_stage_change_ranker_nested_cv_v1"
STAGE205_SHA256 = "0988f97e7e30e6772cc7a7c9738a9e1d285f698629b5b0ccf48c1701e36de02a"
STAGE199_SHA256 = "5b933f524fff1bceb4d4d842e4f3a1aec3160aa3ed337131444ec1b7c2699fee"
MINIMUM_AVAILABLE_MEMORY_BYTES = 4 * 1024**3
FORBIDDEN_PUBLIC_KEYS = stage194.FORBIDDEN_PUBLIC_KEYS | {
    "candidate_actions",
    "complete_pool",
    "feature_rows",
    "gate_features",
    "gate_training_rows",
    "prediction_rows",
    "question_key",
    "ranker_predictions",
    "selected_action",
    "source_safety_predictions",
}


@dataclass(frozen=True)
class Stage206Visualization:
    name: str
    path: str


class Stage206SourceReproductionError(ValueError):
    def __init__(self, evidence: Mapping[str, Any]) -> None:
        self.evidence = dict(evidence)
        failed = ", ".join(self.evidence.get("failed_checks", ())) or "unknown"
        super().__init__(f"Stage206 did not reproduce formal Stage182 checks: {failed}")


def run_stage206_two_stage_change_ranker_cv(
    *,
    stage205_protocol_path: Path,
    stage199_report_path: Path,
    lightgbm_wheel_path: Path,
    narwhals_wheel_path: Path,
    stage182_report_path: Path,
    stage181_report_path: Path,
    stage180_report_path: Path,
    stage179_report_path: Path,
    stage178_public_path: Path,
    stage178_private_path: Path,
    stage178_alignment_path: Path,
    stage128_protocol_path: Path,
    stage125_protocol_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    documents_path: Path,
    encoder_batch_size: int = 64,
    progress_sink: stage182.ProgressSink | None = None,
    preflight_failure_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Reproduce Stage182 and run the authorized Stage206 train-only experiment."""

    started_at = time.perf_counter()
    stage205_fingerprint = stage173._resolved_fingerprint(stage205_protocol_path)
    stage199_fingerprint = stage173._resolved_fingerprint(stage199_report_path)
    if stage205_fingerprint["sha256"] != STAGE205_SHA256:
        raise ValueError("Stage206 Stage205 protocol hash mismatch")
    if stage199_fingerprint["sha256"] != STAGE199_SHA256:
        raise ValueError("Stage206 Stage199 report hash mismatch")
    formal_stage205 = _load_json(stage205_protocol_path)
    formal_stage199 = _load_json(stage199_report_path)
    _authorize_stage205_protocol(formal_stage205)
    _authorize_stage199_report(formal_stage199)
    dependency_preflight = stage194._dependency_preflight(
        lightgbm_wheel_path=lightgbm_wheel_path,
        narwhals_wheel_path=narwhals_wheel_path,
    )
    formal_stage182 = _load_json(stage182_report_path)
    stage183._authorize_stage182_report(formal_stage182)
    authorized_at = time.perf_counter()

    import torch

    tracker = stage169.Stage169ResourceTracker(torch_module=torch)
    preflight_snapshot = tracker.capture("stage206_preflight")
    if preflight_snapshot.system_available_memory_bytes < MINIMUM_AVAILABLE_MEMORY_BYTES:
        raise RuntimeError("Stage206 requires at least 4 GiB available system memory")
    _emit(progress_sink, phase="stage205_protocol_dependencies_and_memory_authorized")
    private: dict[str, Any] = {}

    def capture(
        action_rows: Sequence[ActionAuditRow],
        selected_actions: Sequence[SelectedAction],
        outer_predictions: Sequence[DualTargetPrediction],
    ) -> None:
        private["action_rows"] = tuple(action_rows)
        private["selected_actions"] = tuple(selected_actions)
        private["outer_prediction_count"] = len(outer_predictions)

    reproduced_stage182 = stage182.run_stage182_composition_dual_target_cv(
        stage181_report_path=stage181_report_path,
        stage180_report_path=stage180_report_path,
        stage179_report_path=stage179_report_path,
        stage178_public_path=stage178_public_path,
        stage178_private_path=stage178_private_path,
        stage178_alignment_path=stage178_alignment_path,
        stage128_protocol_path=stage128_protocol_path,
        stage125_protocol_path=stage125_protocol_path,
        stage80_report_path=stage80_report_path,
        train_split_path=train_split_path,
        documents_path=documents_path,
        encoder_batch_size=encoder_batch_size,
        progress_sink=progress_sink,
        private_diagnostic_sink=capture,
    )
    tracker.capture("stage182_reproduced")
    reproduced_at = time.perf_counter()
    reproduction = stage183._stage182_reproduction(
        formal=formal_stage182,
        reproduced=reproduced_stage182,
        selected_actions=private["selected_actions"],
    )
    if not reproduction["passed"]:
        failure_report = _preflight_failure_report(
            stage205_fingerprint=stage205_fingerprint,
            stage199_fingerprint=stage199_fingerprint,
            reproduction=reproduction,
        )
        if preflight_failure_sink is not None:
            preflight_failure_sink(failure_report)
        raise Stage206SourceReproductionError(reproduction)
    gc.collect()
    tracker.capture("stage182_temporary_resources_released")

    def cv_progress(event: Mapping[str, Any]) -> None:
        if event.get("phase") in {
            "stage206_inner_partition_complete",
            "stage206_outer_context_complete",
            "stage206_outer_context_no_eligible_config",
        }:
            tracker.capture(str(event["phase"]))
        _emit(progress_sink, **dict(event))

    nested_cv = core.run_two_stage_change_ranker_nested_cv(
        action_rows=private["action_rows"],
        stage182_selected_actions=private["selected_actions"],
        stage205_protocol=formal_stage205,
        stage199_report=formal_stage199,
        progress_sink=cv_progress,
    )
    tracker.capture("two_stage_change_ranker_nested_cv_complete")
    analyzed_at = time.perf_counter()
    report: dict[str, Any] = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "analysis_id": ANALYSIS_ID,
        "analysis_scope": (
            "Train-only five-by-four nested CV of the frozen two-stage conditional "
            "ranker and change/abstain gate grid plus one exact Stage196 control. "
            "Development and test remain closed; no fallback, retry, weaker candidate, "
            "runtime E2E, full-train selection, replacement, or default activation occurs."
        ),
        "source_authorization": {
            "stage205_protocol": stage205_fingerprint,
            "stage199_control_report": stage199_fingerprint,
            "stage182_rerun_sources": reproduced_stage182["source_authorization"],
        },
        "dependency_preflight": dependency_preflight,
        "resource_preflight": {
            "frozen_minimum_available_memory_bytes": MINIMUM_AVAILABLE_MEMORY_BYTES,
            "actual_available_memory_bytes": preflight_snapshot.system_available_memory_bytes,
            "frozen_threshold_met": True,
            "memory_override_authorized": False,
            "candidate_grid_reduced": False,
            "fallback_enabled": False,
        },
        "frozen_protocol": formal_stage205["frozen_protocol"],
        "stage182_reproduction": reproduction,
        "two_stage_change_ranker_nested_cv": nested_cv,
        "runtime": reproduced_stage182["runtime"],
        "resource_consumption": stage181._resource_summary(tracker.snapshots),
        "timing_seconds": {
            "source_dependency_and_memory_authorization": round(authorized_at - started_at, 6),
            "stage182_reproduction": round(reproduced_at - authorized_at, 6),
            "two_stage_change_ranker_nested_cv": round(analyzed_at - reproduced_at, 6),
            "wall": round(analyzed_at - started_at, 6),
        },
        "execution_boundaries": {
            "train_loaded": True,
            "development_loaded": False,
            "test_loaded": False,
            "captured_action_row_count": len(private["action_rows"]),
            "captured_stage182_selected_action_count": len(private["selected_actions"]),
            "captured_stage182_outer_prediction_count": private["outer_prediction_count"],
            "stage182_model_head_fit_count": reproduced_stage182["execution_boundaries"][
                "dual_target_model_head_fit_count"
            ],
            "stage206_model_fit_count": nested_cv["execution"]["model_fit_count"],
            "stage206_lightgbm_tree_count": nested_cv["execution"]["tree_count"],
            "stage206_private_prediction_count": nested_cv["execution"]["private_prediction_count"],
            "gold_used_only_for_training_targets_and_offline_evaluation": True,
            "full_train_policy_selected": False,
            "replacement_policy_selected": False,
            "runtime_registered_as_default": False,
            "runtime_e2e_run": False,
            "stage178b_run": False,
            "retry_action_count": reproduced_stage182["execution_boundaries"]["retry_action_count"],
            "fallback_action_count": reproduced_stage182["execution_boundaries"][
                "fallback_action_count"
            ],
        },
    }
    forbidden = sorted(_forbidden_keys_found(report))
    report["public_safe_contract"] = {
        "forbidden_keys": sorted(FORBIDDEN_PUBLIC_KEYS),
        "forbidden_keys_found": forbidden,
        "private_training_rows_persisted": False,
        "private_predictions_persisted": False,
    }
    report["process_guards"] = _process_guards(report, forbidden)
    valid = all(row["passed"] for row in report["process_guards"])
    accepted = nested_cv["candidate_family_accepted"] if valid else False
    report["decision"] = {
        "status": (
            "stage206_two_stage_change_ranker_candidate_found"
            if valid and accepted
            else "stage206_two_stage_change_ranker_insufficient"
            if valid
            else "stage206_two_stage_change_ranker_invalid"
        ),
        "experiment_valid": valid,
        "candidate_family_accepted": accepted,
        "development_opened": False,
        "test_opened": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "runtime_e2e_authorized": False,
        "default_runtime_activation": False,
    }
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report


def write_stage206_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage206Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv = report["two_stage_change_ranker_nested_cv"]
    resources = report["resource_consumption"]
    top_rows = [
        (fold_id, row["top_inner_candidates"][0]) for fold_id, row in cv["outer_contexts"].items()
    ]
    charts = {
        "stage206_outer_eligible_counts.svg": _chart(
            "Stage 206 inner-eligible configs by outer context",
            [
                _count_bar(fold_id, row["eligible_config_count"])
                for fold_id, row in cv["outer_contexts"].items()
            ],
        ),
        "stage206_control_reproduction.svg": _chart(
            "Stage 206 exact control reproduction",
            [
                _bool_bar(fold_id, row["control_reproduction_exact"])
                for fold_id, row in cv["outer_contexts"].items()
            ],
        ),
        "stage206_top_inner_unsafe.svg": _chart(
            "Stage 206 top-inner unsafe rate",
            [
                _rate_bar(fold_id, candidate["diagnostics"]["unsafe_selection_rate"])
                for fold_id, candidate in top_rows
            ],
        ),
        "stage206_top_inner_capture.svg": _chart(
            "Stage 206 top-inner conditional strict capture",
            [
                _rate_bar(
                    fold_id,
                    candidate["diagnostics"]["conditional_ranker_strict_capture"],
                )
                for fold_id, candidate in top_rows
            ],
        ),
        "stage206_top_inner_precision.svg": _chart(
            "Stage 206 top-inner strict precision",
            [
                _rate_bar(fold_id, candidate["evaluation"]["strict_success_precision"])
                for fold_id, candidate in top_rows
            ],
        ),
        "stage206_ranker_family_unsafe.svg": _chart(
            "Stage 206 ranker-family mean unsafe rate",
            [
                _rate_bar(name, row["mean_unsafe_selection_rate"])
                for name, row in cv["ranker_family_aggregates"].items()
            ],
        ),
        "stage206_ranker_family_capture.svg": _chart(
            "Stage 206 ranker-family mean capture",
            [
                _rate_bar(name, row["mean_conditional_capture"])
                for name, row in cv["ranker_family_aggregates"].items()
            ],
        ),
        "stage206_ranker_family_pre_gate_strict.svg": _chart(
            "Stage 206 pre-gate conditional winner strict rate",
            [
                _rate_bar(name, row["mean_pre_gate_ranker_strict_rate"])
                for name, row in cv["ranker_family_aggregates"].items()
                if "mean_pre_gate_ranker_strict_rate" in row
            ],
        ),
        "stage206_ranker_family_pre_gate_unsafe.svg": _chart(
            "Stage 206 pre-gate conditional winner unsafe rate",
            [
                _rate_bar(name, row["mean_pre_gate_ranker_unsafe_rate"])
                for name, row in cv["ranker_family_aggregates"].items()
                if "mean_pre_gate_ranker_unsafe_rate" in row
            ],
        ),
        "stage206_coverage_realized_change.svg": _chart(
            "Stage 206 realized change coverage",
            [
                _rate_bar(name, row["mean_realized_change_coverage"])
                for name, row in cv["coverage_aggregates"].items()
            ],
        ),
        "stage206_coverage_unsafe.svg": _chart(
            "Stage 206 target-coverage mean unsafe rate",
            [
                _rate_bar(name, row["mean_unsafe_selection_rate"])
                for name, row in cv["coverage_aggregates"].items()
            ],
        ),
        "stage206_coverage_capture.svg": _chart(
            "Stage 206 target-coverage mean conditional capture",
            [
                _rate_bar(name, row["mean_conditional_capture"])
                for name, row in cv["coverage_aggregates"].items()
            ],
        ),
        "stage206_coverage_precision.svg": _chart(
            "Stage 206 target-coverage mean strict precision",
            [
                _rate_bar(name, row["mean_strict_success_precision"])
                for name, row in cv["coverage_aggregates"].items()
            ],
        ),
        "stage206_gate_heldout_roc_auc.svg": _chart(
            "Stage 206 heldout gate ROC AUC by target coverage",
            [
                _rate_bar(name, row["mean_heldout_gate_roc_auc"])
                for name, row in cv["coverage_aggregates"].items()
            ],
        ),
        "stage206_gate_heldout_average_precision.svg": _chart(
            "Stage 206 heldout gate average precision by target coverage",
            [
                _rate_bar(name, row["mean_heldout_gate_average_precision"])
                for name, row in cv["coverage_aggregates"].items()
            ],
        ),
        "stage206_selected_ranker_families.svg": _chart(
            "Stage 206 selected outer ranker families",
            [
                _count_bar(name, value)
                for name, value in cv["selected_ranker_family_counts"].items()
            ],
            margin_left=720,
        ),
        "stage206_advancement_gates.svg": _chart(
            "Stage 206 advancement gates",
            [_bool_bar(row["name"], row["passed"]) for row in cv["advancement_gates"]],
            margin_left=720,
        ),
        "stage206_execution.svg": _chart(
            "Stage 206 execution counts",
            [
                _count_bar("model fits", cv["execution"]["model_fit_count"]),
                _count_bar("source trees", cv["execution"]["source_tree_count"]),
                _count_bar(
                    "conditional ranker trees",
                    cv["execution"]["conditional_ranker_tree_count"],
                ),
                _count_bar("gate trees", cv["execution"]["gate_tree_count"]),
                _count_bar("private predictions", cv["execution"]["private_prediction_count"]),
            ],
        ),
        "stage206_resources.svg": _chart(
            "Stage 206 resource consumption (GiB)",
            [
                _gib_bar("peak working set", resources["process_peak_working_set_bytes"]),
                _gib_bar("peak private usage", resources["process_peak_private_usage_bytes"]),
                _gib_bar(
                    "minimum system available",
                    resources["minimum_system_available_memory_bytes"],
                ),
            ],
        ),
        "stage206_process_guards.svg": _chart(
            "Stage 206 process guards",
            [_bool_bar(row["name"], row["passed"]) for row in report["process_guards"]],
            margin_left=820,
        ),
    }
    visualizations = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        visualizations.append(Stage206Visualization(name, str(path)))
    return tuple(visualizations)


def write_stage206_report_bundle(
    *,
    report: Mapping[str, Any],
    output_path: Path,
    visualization_dir: Path,
) -> dict[str, Any]:
    core_report = dict(report)
    _write_json_atomic(output_path, core_report)
    visualizations = write_stage206_visualizations(
        report=core_report,
        output_dir=visualization_dir,
    )
    final_report = {
        **core_report,
        "visualizations": [{"name": item.name, "path": item.path} for item in visualizations],
    }
    _write_json_atomic(output_path, final_report)
    return final_report


def write_stage206_preflight_failure(*, report: Mapping[str, Any], output_path: Path) -> None:
    _write_json_atomic(output_path, report)


def _authorize_stage205_protocol(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 205":
        raise ValueError("Stage206 requires Stage205")
    decision = report.get("decision", {})
    if decision.get("protocol_valid") is not True:
        raise ValueError("Stage206 requires a valid Stage205 protocol")
    if decision.get("stage206_train_only_experiment_authorized") is not True:
        raise ValueError("Stage205 did not authorize Stage206")
    if decision.get("development_opened") is not False or decision.get("test_opened") is not False:
        raise ValueError("Stage206 requires closed development and test sets")
    if len(report.get("guard_checks", [])) != 84 or not all(
        row.get("passed") is True for row in report["guard_checks"]
    ):
        raise ValueError("Stage206 requires all 84 Stage205 guards")
    frozen = report.get("frozen_protocol", {})
    factorial = frozen.get("factorial_ablation", {})
    crossfit = frozen.get("cross_fitting_contract", {})
    cross_validation = frozen.get("cross_validation", {})
    if factorial.get("two_stage_policy_count") != 10:
        raise ValueError("Stage206 requires the frozen 10-policy grid")
    if factorial.get("candidate_config_count_per_outer_context") != 11:
        raise ValueError("Stage206 requires the frozen 11-config candidate family")
    if crossfit.get("source_safety_predictions_for_gate_winners_are_oof_only") is not True:
        raise ValueError("Stage206 requires strict OOF source-safety gate features")
    if crossfit.get("same_fit_source_safety_predictions_for_gate_training") is not False:
        raise ValueError("Stage206 rejects same-fit source-safety gate features")
    if cross_validation.get("model_fits_per_inner_partition") != 24:
        raise ValueError("Stage206 requires the amended 24-fit partition budget")
    if cross_validation.get("maximum_model_fit_count") != 570:
        raise ValueError("Stage206 requires the amended 570-fit maximum")
    if cross_validation.get("maximum_lightgbm_tree_count") != 96_000:
        raise ValueError("Stage206 requires the frozen 96,000-tree maximum")


def _authorize_stage199_report(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 199":
        raise ValueError("Stage206 requires Stage199 control evidence")
    decision = report.get("decision", {})
    if decision.get("experiment_valid") is not True:
        raise ValueError("Stage206 requires valid Stage199 evidence")
    if decision.get("candidate_family_accepted") is not False:
        raise ValueError("Stage206 requires the insufficient Stage199 candidate family")
    if decision.get("development_opened") is not False or decision.get("test_opened") is not False:
        raise ValueError("Stage206 requires closed Stage199 development and test sets")


def _process_guards(report: Mapping[str, Any], forbidden: Sequence[str]) -> list[dict[str, Any]]:
    cv = report["two_stage_change_ranker_nested_cv"]
    execution = cv["execution"]
    boundaries = report["execution_boundaries"]
    eligible_outer = sum(row["outer_evaluated"] for row in cv["outer_contexts"].values())
    two_stage_outer = sum(
        row["outer_evaluated"] and row["selected_spec"]["ranker_family"] != "exact_control"
        for row in cv["outer_contexts"].values()
    )
    source_partition_count = 20 + eligible_outer
    expected_source_fits = 4 * source_partition_count
    expected_source_safety_crossfit_fits = 160 + 8 * two_stage_outer
    expected_ranker_fits = 200 + 5 * two_stage_outer
    expected_gate_fits = 40 + two_stage_outer
    expected_model_fits = (
        expected_source_fits
        + expected_source_safety_crossfit_fits
        + expected_ranker_fits
        + expected_gate_fits
    )
    expected_lightgbm_fits = 280 + 2 * eligible_outer + 6 * two_stage_outer
    action_count = cv["dataset"]["action_count"]
    nonbaseline_action_count = sum(cv["dataset"]["fold_nonbaseline_action_counts"].values())
    question_count = cv["dataset"]["question_count"]
    evaluated_action_count = sum(
        cv["dataset"]["fold_action_counts"][fold_id]
        for fold_id, row in cv["outer_contexts"].items()
        if row["outer_evaluated"]
    )
    expected_source_safety_oof_predictions = 24 * action_count
    expected_ranker_oof_predictions = 24 * nonbaseline_action_count
    expected_gate_training_questions = 24 * question_count
    expected_private_predictions = (
        40 * action_count + 32 * nonbaseline_action_count + 32 * question_count
    )
    for fold_id, row in cv["outer_contexts"].items():
        if not row["outer_evaluated"] or row["selected_spec"]["ranker_family"] == "exact_control":
            continue
        training_action_count = action_count - cv["dataset"]["fold_action_counts"][fold_id]
        training_nonbaseline_count = (
            nonbaseline_action_count - cv["dataset"]["fold_nonbaseline_action_counts"][fold_id]
        )
        training_question_count = question_count - cv["dataset"]["fold_question_counts"][fold_id]
        heldout_nonbaseline_count = cv["dataset"]["fold_nonbaseline_action_counts"][fold_id]
        heldout_question_count = cv["dataset"]["fold_question_counts"][fold_id]
        expected_source_safety_oof_predictions += 2 * training_action_count
        expected_ranker_oof_predictions += training_nonbaseline_count
        expected_gate_training_questions += training_question_count
        expected_private_predictions += (
            2 * training_action_count
            + training_nonbaseline_count
            + training_question_count
            + heldout_nonbaseline_count
            + heldout_question_count
        )
    expected_private_predictions += 4 * evaluated_action_count
    checks = (
        ("stage182_reproduction_exact", report["stage182_reproduction"]["passed"] is True),
        ("train_only", boundaries["train_loaded"] is True),
        ("development_closed", boundaries["development_loaded"] is False),
        ("test_closed", boundaries["test_loaded"] is False),
        ("five_outer_contexts", len(cv["outer_contexts"]) == 5),
        ("all_controls_reproduced", execution["all_controls_reproduced_exactly"] is True),
        ("five_control_reproductions", execution["control_reproduction_count"] == 5),
        ("eleven_cells", len(cv["cell_aggregates"]) == 11),
        ("three_ranker_families", len(cv["ranker_family_aggregates"]) == 3),
        ("five_target_coverages", len(cv["coverage_aggregates"]) == 5),
        (
            "source_safety_oof_contract_enabled",
            cv["protocol"]["source_safety_gate_features_oof"] is True,
        ),
        ("fallback_disabled_in_protocol", cv["protocol"]["fallback_enabled"] is False),
        (
            "source_fit_count_exact",
            execution["source_model_fit_count"] == expected_source_fits,
        ),
        (
            "source_safety_crossfit_count_exact",
            execution["source_safety_crossfit_fit_count"] == expected_source_safety_crossfit_fits,
        ),
        (
            "conditional_ranker_fit_count_exact",
            execution["conditional_ranker_fit_count"] == expected_ranker_fits,
        ),
        ("gate_fit_count_exact", execution["gate_fit_count"] == expected_gate_fits),
        ("model_fit_count_exact", execution["model_fit_count"] == expected_model_fits),
        (
            "lightgbm_fit_count_exact",
            execution["lightgbm_model_fit_count"] == expected_lightgbm_fits,
        ),
        (
            "source_group_validation_count_exact",
            execution["source_group_contract_validation_count"] == source_partition_count,
        ),
        (
            "ranker_group_validation_count_exact",
            execution["ranker_group_contract_validation_count"] == expected_ranker_fits,
        ),
        (
            "source_safety_oof_prediction_count_exact",
            execution["source_safety_oof_prediction_count"]
            == expected_source_safety_oof_predictions,
        ),
        (
            "ranker_oof_prediction_count_exact",
            execution["ranker_oof_prediction_count"] == expected_ranker_oof_predictions,
        ),
        (
            "gate_training_question_count_exact",
            execution["gate_training_question_count"] == expected_gate_training_questions,
        ),
        (
            "source_tree_count_exact",
            execution["source_tree_count"] == 600 * source_partition_count,
        ),
        (
            "conditional_ranker_tree_count_within_budget",
            expected_ranker_fits
            <= execution["conditional_ranker_tree_count"]
            <= 300 * expected_ranker_fits,
        ),
        (
            "gate_tree_count_within_budget",
            expected_gate_fits <= execution["gate_tree_count"] <= 300 * expected_gate_fits,
        ),
        (
            "tree_parts_sum_exact",
            execution["tree_count"]
            == execution["source_tree_count"]
            + execution["conditional_ranker_tree_count"]
            + execution["gate_tree_count"],
        ),
        ("tree_budget_respected", execution["tree_count"] <= 96_000),
        ("fit_budget_respected", execution["model_fit_count"] <= 570),
        (
            "private_prediction_count_exact",
            execution["private_prediction_count"] == expected_private_predictions,
        ),
        ("advancement_gate_count", len(cv["advancement_gates"]) == 17),
        (
            "candidate_acceptance_matches_gates",
            cv["candidate_family_accepted"]
            == all(row["passed"] for row in cv["advancement_gates"]),
        ),
        ("no_public_training_rows", execution["public_training_rows_written"] == 0),
        ("no_public_prediction_rows", execution["public_prediction_rows_written"] == 0),
        ("no_full_train_selection", boundaries["full_train_policy_selected"] is False),
        ("no_runtime_default", boundaries["runtime_registered_as_default"] is False),
        ("no_runtime_e2e", boundaries["runtime_e2e_run"] is False),
        ("no_stage178b", boundaries["stage178b_run"] is False),
        (
            "gold_runtime_exclusion_declared",
            boundaries["gold_used_only_for_training_targets_and_offline_evaluation"] is True,
        ),
        ("no_retry", boundaries["retry_action_count"] == 0),
        ("no_fallback", boundaries["fallback_action_count"] == 0),
        ("no_forbidden_public_keys", not forbidden),
    )
    return [_gate(name, passed) for name, passed in checks]


def _preflight_failure_report(
    *,
    stage205_fingerprint: Mapping[str, Any],
    stage199_fingerprint: Mapping[str, Any],
    reproduction: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "status": "stage206_stage182_reproduction_failed",
        "source_authorization": {
            "stage205_protocol": dict(stage205_fingerprint),
            "stage199_control_report": dict(stage199_fingerprint),
        },
        "stage182_reproduction": dict(reproduction),
        "execution_boundaries": {
            "stage206_model_fit_count": 0,
            "development_loaded": False,
            "test_loaded": False,
            "retry_action_count": 0,
            "fallback_action_count": 0,
        },
    }


def _chart(
    title: str,
    bars: Sequence[BarDatum],
    *,
    margin_left: int = 520,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label="aggregate value",
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


def _gib_bar(name: str, byte_count: int) -> BarDatum:
    value = byte_count / 1024**3
    return BarDatum(name, value, f"{value:.3f}")


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


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(value, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(progress_sink: stage182.ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
