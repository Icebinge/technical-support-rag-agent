from __future__ import annotations

import gc
import json
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application import composition_top1_joint_objective_cv as core
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

STAGE = "Stage 203"
CREATED_AT = "2026-07-27"
ANALYSIS_ID = "primeqa_hybrid_top1_joint_objective_nested_cv_v1"
STAGE202_SHA256 = "0818f0ae7186eea19d4137023b87963010e76e9cbf20e8f1ef61576a60569bdb"
STAGE199_SHA256 = "5b933f524fff1bceb4d4d842e4f3a1aec3160aa3ed337131444ec1b7c2699fee"
MINIMUM_AVAILABLE_MEMORY_BYTES = 4 * 1024**3
FORBIDDEN_PUBLIC_KEYS = stage194.FORBIDDEN_PUBLIC_KEYS | {
    "candidate_actions",
    "complete_pool",
    "feature_rows",
    "objective_predictions",
    "prediction_rows",
    "question_key",
    "selected_action",
}


@dataclass(frozen=True)
class Stage203Visualization:
    name: str
    path: str


class Stage203SourceReproductionError(ValueError):
    def __init__(self, evidence: Mapping[str, Any]) -> None:
        self.evidence = dict(evidence)
        failed = ", ".join(self.evidence.get("failed_checks", ())) or "unknown"
        super().__init__(f"Stage203 did not reproduce formal Stage182 checks: {failed}")


def run_stage203_top1_joint_objective_cv(
    *,
    stage202_protocol_path: Path,
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
    diagnostic_sink: core.DiagnosticSink | None = None,
) -> dict[str, Any]:
    """Reproduce Stage182 and run the authorized Stage203 train-only experiment."""

    started_at = time.perf_counter()
    stage202_fingerprint = stage173._resolved_fingerprint(stage202_protocol_path)
    stage199_fingerprint = stage173._resolved_fingerprint(stage199_report_path)
    if stage202_fingerprint["sha256"] != STAGE202_SHA256:
        raise ValueError("Stage203 Stage202 protocol hash mismatch")
    if stage199_fingerprint["sha256"] != STAGE199_SHA256:
        raise ValueError("Stage203 Stage199 report hash mismatch")
    formal_stage202 = _load_json(stage202_protocol_path)
    formal_stage199 = _load_json(stage199_report_path)
    _authorize_stage202_protocol(formal_stage202)
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
    preflight_snapshot = tracker.capture("stage203_preflight")
    if preflight_snapshot.system_available_memory_bytes < MINIMUM_AVAILABLE_MEMORY_BYTES:
        raise RuntimeError("Stage203 requires at least 4 GiB available system memory")
    _emit(progress_sink, phase="stage202_protocol_dependencies_and_memory_authorized")
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
            stage202_fingerprint=stage202_fingerprint,
            stage199_fingerprint=stage199_fingerprint,
            reproduction=reproduction,
        )
        if preflight_failure_sink is not None:
            preflight_failure_sink(failure_report)
        raise Stage203SourceReproductionError(reproduction)
    gc.collect()
    tracker.capture("stage182_temporary_resources_released")

    def cv_progress(event: Mapping[str, Any]) -> None:
        if event.get("phase") in {
            "stage203_inner_partition_complete",
            "stage203_outer_context_complete",
            "stage203_outer_context_no_eligible_config",
        }:
            tracker.capture(str(event["phase"]))
        _emit(progress_sink, **dict(event))

    nested_cv = core.run_top1_joint_objective_nested_cv(
        action_rows=private["action_rows"],
        stage182_selected_actions=private["selected_actions"],
        stage202_protocol=formal_stage202,
        stage199_report=formal_stage199,
        progress_sink=cv_progress,
        diagnostic_sink=diagnostic_sink,
    )
    tracker.capture("top1_joint_objective_nested_cv_complete")
    analyzed_at = time.perf_counter()
    report: dict[str, Any] = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "analysis_id": ANALYSIS_ID,
        "analysis_scope": (
            "Train-only five-by-four nested CV of the frozen 4x4 grouped Top-1 "
            "safety/precision objective grid plus one exact Stage196 control. "
            "Development and test remain closed; no fallback, retry, weaker candidate, "
            "runtime E2E, full-train selection, replacement, or default activation occurs."
        ),
        "source_authorization": {
            "stage202_protocol": stage202_fingerprint,
            "stage199_control_report": stage199_fingerprint,
            "stage182_rerun_sources": reproduced_stage182["source_authorization"],
        },
        "dependency_preflight": dependency_preflight,
        "resource_preflight": {
            "frozen_minimum_available_memory_bytes": MINIMUM_AVAILABLE_MEMORY_BYTES,
            "actual_available_memory_bytes": preflight_snapshot.system_available_memory_bytes,
            "frozen_threshold_met": True,
            "memory_override_authorized": False,
            "objective_grid_reduced": False,
            "fallback_enabled": False,
        },
        "frozen_protocol": formal_stage202["frozen_protocol"],
        "stage182_reproduction": reproduction,
        "top1_joint_objective_nested_cv": nested_cv,
        "runtime": reproduced_stage182["runtime"],
        "resource_consumption": stage181._resource_summary(tracker.snapshots),
        "timing_seconds": {
            "source_dependency_and_memory_authorization": round(authorized_at - started_at, 6),
            "stage182_reproduction": round(reproduced_at - authorized_at, 6),
            "top1_joint_objective_nested_cv": round(analyzed_at - reproduced_at, 6),
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
            "stage203_model_fit_count": nested_cv["execution"]["model_fit_count"],
            "stage203_lightgbm_tree_count": nested_cv["execution"]["tree_count"],
            "stage203_private_prediction_count": nested_cv["execution"]["private_prediction_count"],
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
            "stage203_top1_joint_objective_candidate_found"
            if valid and accepted
            else "stage203_top1_joint_objective_insufficient"
            if valid
            else "stage203_top1_joint_objective_invalid"
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


def write_stage203_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage203Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv = report["top1_joint_objective_nested_cv"]
    resources = report["resource_consumption"]
    top_rows = [
        (fold_id, row["top_inner_candidates"][0]) for fold_id, row in cv["outer_contexts"].items()
    ]
    charts = {
        "stage203_outer_eligible_counts.svg": _chart(
            "Stage 203 inner-eligible configs by outer context",
            [
                _count_bar(fold_id, row["eligible_config_count"])
                for fold_id, row in cv["outer_contexts"].items()
            ],
        ),
        "stage203_control_reproduction.svg": _chart(
            "Stage 203 exact control reproduction",
            [
                _bool_bar(fold_id, row["control_reproduction_exact"])
                for fold_id, row in cv["outer_contexts"].items()
            ],
        ),
        "stage203_top_inner_unsafe.svg": _chart(
            "Stage 203 top-inner unsafe rate",
            [
                _rate_bar(fold_id, candidate["diagnostics"]["unsafe_selection_rate"])
                for fold_id, candidate in top_rows
            ],
        ),
        "stage203_top_inner_capture.svg": _chart(
            "Stage 203 top-inner conditional strict capture",
            [
                _rate_bar(
                    fold_id,
                    candidate["diagnostics"]["conditional_ranker_strict_capture"],
                )
                for fold_id, candidate in top_rows
            ],
        ),
        "stage203_top_inner_precision.svg": _chart(
            "Stage 203 top-inner strict precision",
            [
                _rate_bar(fold_id, candidate["evaluation"]["strict_success_precision"])
                for fold_id, candidate in top_rows
            ],
        ),
        "stage203_safety_weight_unsafe.svg": _chart(
            "Stage 203 safety-weight mean unsafe rate",
            [
                _rate_bar(name, row["mean_unsafe_selection_rate"])
                for name, row in cv["safety_weight_aggregates"].items()
            ],
        ),
        "stage203_safety_weight_capture.svg": _chart(
            "Stage 203 safety-weight mean capture",
            [
                _rate_bar(name, row["mean_conditional_capture"])
                for name, row in cv["safety_weight_aggregates"].items()
            ],
        ),
        "stage203_precision_weight_precision.svg": _chart(
            "Stage 203 precision-weight mean strict precision",
            [
                _rate_bar(name, row["mean_strict_success_precision"])
                for name, row in cv["precision_weight_aggregates"].items()
            ],
        ),
        "stage203_ablation_unsafe.svg": _chart(
            "Stage 203 ablation-family mean unsafe rate",
            [
                _rate_bar(name, row["mean_unsafe_selection_rate"])
                for name, row in cv["ablation_family_aggregates"].items()
            ],
        ),
        "stage203_ablation_capture.svg": _chart(
            "Stage 203 ablation-family mean capture",
            [
                _rate_bar(name, row["mean_conditional_capture"])
                for name, row in cv["ablation_family_aggregates"].items()
            ],
        ),
        "stage203_directional_response.svg": _chart(
            "Stage 203 adjacent penalty responses",
            [
                _count_bar(
                    "safety comparisons",
                    cv["directional_penalty_response"]["safety_adjacent_comparison_count"],
                ),
                _count_bar(
                    "unsafe nonincreasing",
                    cv["directional_penalty_response"]["safety_nonincreasing_unsafe_count"],
                ),
                _count_bar(
                    "precision comparisons",
                    cv["directional_penalty_response"]["precision_adjacent_comparison_count"],
                ),
                _count_bar(
                    "precision nondecreasing",
                    cv["directional_penalty_response"][
                        "precision_nondecreasing_strict_precision_count"
                    ],
                ),
            ],
        ),
        "stage203_selected_specs.svg": _chart(
            "Stage 203 selected outer configs",
            [_count_bar(name, value) for name, value in cv["selected_spec_counts"].items()],
            margin_left=720,
        ),
        "stage203_advancement_gates.svg": _chart(
            "Stage 203 advancement gates",
            [_bool_bar(row["name"], row["passed"]) for row in cv["advancement_gates"]],
            margin_left=720,
        ),
        "stage203_execution.svg": _chart(
            "Stage 203 execution counts",
            [
                _count_bar("model fits", cv["execution"]["model_fit_count"]),
                _count_bar("source trees", cv["execution"]["source_tree_count"]),
                _count_bar("custom trees", cv["execution"]["custom_objective_tree_count"]),
                _count_bar("private predictions", cv["execution"]["private_prediction_count"]),
            ],
        ),
        "stage203_resources.svg": _chart(
            "Stage 203 resource consumption (GiB)",
            [
                _gib_bar("peak working set", resources["process_peak_working_set_bytes"]),
                _gib_bar("peak private usage", resources["process_peak_private_usage_bytes"]),
                _gib_bar(
                    "minimum system available",
                    resources["minimum_system_available_memory_bytes"],
                ),
            ],
        ),
        "stage203_process_guards.svg": _chart(
            "Stage 203 process guards",
            [_bool_bar(row["name"], row["passed"]) for row in report["process_guards"]],
            margin_left=820,
        ),
    }
    visualizations = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        visualizations.append(Stage203Visualization(name, str(path)))
    return tuple(visualizations)


def write_stage203_report_bundle(
    *,
    report: Mapping[str, Any],
    output_path: Path,
    visualization_dir: Path,
) -> dict[str, Any]:
    core_report = dict(report)
    _write_json_atomic(output_path, core_report)
    visualizations = write_stage203_visualizations(
        report=core_report,
        output_dir=visualization_dir,
    )
    final_report = {
        **core_report,
        "visualizations": [{"name": item.name, "path": item.path} for item in visualizations],
    }
    _write_json_atomic(output_path, final_report)
    return final_report


def write_stage203_preflight_failure(*, report: Mapping[str, Any], output_path: Path) -> None:
    _write_json_atomic(output_path, report)


def _authorize_stage202_protocol(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 202":
        raise ValueError("Stage203 requires Stage202")
    decision = report.get("decision", {})
    if decision.get("protocol_valid") is not True:
        raise ValueError("Stage203 requires a valid Stage202 protocol")
    if decision.get("stage203_train_only_experiment_authorized") is not True:
        raise ValueError("Stage202 did not authorize Stage203")
    if decision.get("development_opened") is not False or decision.get("test_opened") is not False:
        raise ValueError("Stage203 requires closed development and test sets")
    if len(report.get("guard_checks", [])) != 81 or not all(
        row.get("passed") is True for row in report["guard_checks"]
    ):
        raise ValueError("Stage203 requires all 81 Stage202 guards")
    frozen = report.get("frozen_protocol", {})
    if frozen.get("objective_factorial", {}).get("custom_objective_count") != 16:
        raise ValueError("Stage203 requires the frozen 16-objective grid")
    if frozen.get("objective_factorial", {}).get("candidate_config_count_per_outer_context") != 17:
        raise ValueError("Stage203 requires the frozen 17-config candidate family")


def _authorize_stage199_report(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 199":
        raise ValueError("Stage203 requires Stage199 control evidence")
    decision = report.get("decision", {})
    if decision.get("experiment_valid") is not True:
        raise ValueError("Stage203 requires valid Stage199 evidence")
    if decision.get("candidate_family_accepted") is not False:
        raise ValueError("Stage203 requires the insufficient Stage199 candidate family")
    if decision.get("development_opened") is not False or decision.get("test_opened") is not False:
        raise ValueError("Stage203 requires closed Stage199 development and test sets")


def _process_guards(report: Mapping[str, Any], forbidden: Sequence[str]) -> list[dict[str, Any]]:
    cv = report["top1_joint_objective_nested_cv"]
    execution = cv["execution"]
    boundaries = report["execution_boundaries"]
    eligible_outer = sum(row["outer_evaluated"] for row in cv["outer_contexts"].values())
    custom_outer = execution["outer_custom_objective_refit_count"]
    source_partition_count = 20 + eligible_outer
    expected_source_fits = 4 * source_partition_count
    expected_custom_fits = 320 + custom_outer
    expected_model_fits = expected_source_fits + expected_custom_fits
    evaluated_action_count = sum(
        cv["dataset"]["fold_action_counts"][fold_id]
        for fold_id, row in cv["outer_contexts"].items()
        if row["outer_evaluated"]
    )
    custom_outer_prediction_count = sum(
        cv["dataset"]["fold_action_counts"][fold_id]
        for fold_id, row in cv["outer_contexts"].items()
        if row["outer_evaluated"] and row["selected_spec"]["ablation_family"] != "exact_control"
    )
    expected_private_predictions = (
        80 * cv["dataset"]["action_count"]
        + 4 * evaluated_action_count
        + custom_outer_prediction_count
    )
    checks = (
        ("stage182_reproduction_exact", report["stage182_reproduction"]["passed"] is True),
        ("train_only", boundaries["train_loaded"] is True),
        ("development_closed", boundaries["development_loaded"] is False),
        ("test_closed", boundaries["test_loaded"] is False),
        ("five_outer_contexts", len(cv["outer_contexts"]) == 5),
        ("all_controls_reproduced", execution["all_controls_reproduced_exactly"] is True),
        ("five_control_reproductions", execution["control_reproduction_count"] == 5),
        ("seventeen_cells", len(cv["cell_aggregates"]) == 17),
        ("five_ablation_families", len(cv["ablation_family_aggregates"]) == 5),
        ("four_safety_weights", len(cv["safety_weight_aggregates"]) == 4),
        ("four_precision_weights", len(cv["precision_weight_aggregates"]) == 4),
        (
            "twelve_safety_comparisons",
            cv["directional_penalty_response"]["safety_adjacent_comparison_count"] == 12,
        ),
        (
            "twelve_precision_comparisons",
            cv["directional_penalty_response"]["precision_adjacent_comparison_count"] == 12,
        ),
        ("source_fit_count_exact", execution["source_model_fit_count"] == expected_source_fits),
        ("custom_fit_count_exact", execution["custom_objective_fit_count"] == expected_custom_fits),
        ("model_fit_count_exact", execution["model_fit_count"] == expected_model_fits),
        (
            "pool_safety_fit_count_exact",
            execution["pool_safety_fit_count"] == 2 * source_partition_count,
        ),
        ("gain_fit_count_exact", execution["gain_ranker_fit_count"] == source_partition_count),
        (
            "classifier_fit_count_exact",
            execution["classifier_risk_fit_count"] == source_partition_count,
        ),
        (
            "source_tree_count_exact",
            execution["source_tree_count"] == 600 * source_partition_count,
        ),
        (
            "custom_tree_count_within_budget",
            0 <= execution["custom_objective_tree_count"] <= 300 * expected_custom_fits,
        ),
        (
            "tree_parts_sum_exact",
            execution["tree_count"]
            == execution["source_tree_count"] + execution["custom_objective_tree_count"],
        ),
        ("tree_budget_respected", execution["tree_count"] <= 112_500),
        ("fit_budget_respected", execution["model_fit_count"] <= 425),
        (
            "group_contract_count_exact",
            execution["group_contract_validation_count"]
            == source_partition_count + expected_custom_fits,
        ),
        (
            "callback_count_within_budget",
            expected_custom_fits
            <= execution["objective_callback_call_count"]
            <= 300 * expected_custom_fits,
        ),
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
        ("no_retry", boundaries["retry_action_count"] == 0),
        ("no_fallback", boundaries["fallback_action_count"] == 0),
        ("no_forbidden_public_keys", not forbidden),
    )
    return [_gate(name, passed) for name, passed in checks]


def _preflight_failure_report(
    *,
    stage202_fingerprint: Mapping[str, Any],
    stage199_fingerprint: Mapping[str, Any],
    reproduction: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "status": "stage203_stage182_reproduction_failed",
        "source_authorization": {
            "stage202_protocol": dict(stage202_fingerprint),
            "stage199_control_report": dict(stage199_fingerprint),
        },
        "stage182_reproduction": dict(reproduction),
        "execution_boundaries": {
            "stage203_model_fit_count": 0,
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
