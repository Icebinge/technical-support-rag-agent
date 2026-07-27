from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application import composition_joint_risk_winner_cv as stage199_core
from ts_rag_agent.application import primeqa_hybrid_composition_dual_target_cv as stage182
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application import primeqa_hybrid_top1_joint_objective_cv as stage203
from ts_rag_agent.application.composition_top1_joint_objective_failure_attribution import (
    Top1JointObjectiveFailureAttributor,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

STAGE = "Stage 204"
CREATED_AT = "2026-07-27"
ANALYSIS_ID = "primeqa_hybrid_top1_joint_objective_failure_attribution_v1"
STAGE203_SHA256 = "b675d61a2c79d9fcd74639f6a9e4caf1de3da29205018e4b34343fae79340317"
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
class Stage204Visualization:
    name: str
    path: str


def run_stage204_top1_joint_objective_failure_attribution(
    *,
    stage203_report_path: Path,
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
) -> dict[str, Any]:
    """Reproduce Stage203 once and stream private ranking flips into Stage204."""

    started_at = time.perf_counter()
    stage203_fingerprint = stage173._resolved_fingerprint(stage203_report_path)
    if stage203_fingerprint["sha256"] != STAGE203_SHA256:
        raise ValueError("Stage204 Stage203 report hash mismatch")
    formal_stage203 = _load_json(stage203_report_path)
    _authorize_stage203_report(formal_stage203)
    attributor = Top1JointObjectiveFailureAttributor()
    authorized_at = time.perf_counter()

    try:
        reproduced_stage203 = stage203.run_stage203_top1_joint_objective_cv(
            stage202_protocol_path=stage202_protocol_path,
            stage199_report_path=stage199_report_path,
            lightgbm_wheel_path=lightgbm_wheel_path,
            narwhals_wheel_path=narwhals_wheel_path,
            stage182_report_path=stage182_report_path,
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
            diagnostic_sink=attributor,
        )
    except stage203.Stage203SourceReproductionError as error:
        failure = _preflight_failure_report(
            stage203_fingerprint=stage203_fingerprint,
            reproduction_evidence=error.evidence,
        )
        if preflight_failure_sink is not None:
            preflight_failure_sink(failure)
        raise

    reproduction = _stage203_reproduction(formal_stage203, reproduced_stage203)
    if not reproduction["passed"]:
        raise ValueError("Stage204 did not exactly reproduce frozen Stage203 evidence")
    attribution = attributor.report()
    analyzed_at = time.perf_counter()
    report: dict[str, Any] = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "analysis_id": ANALYSIS_ID,
        "analysis_scope": (
            "Train-only streaming attribution of every Stage203 control-to-custom and "
            "adjacent-weight question-level winner transition. Development and test remain "
            "closed; no new model search, gate relaxation, retry, fallback, full-train "
            "selection, replacement, runtime E2E, or default activation occurs."
        ),
        "source_authorization": {
            "stage203_formal_report": stage203_fingerprint,
            "stage203_rerun_sources": reproduced_stage203["source_authorization"],
        },
        "stage203_reproduction": reproduction,
        "failure_attribution": attribution,
        "resource_preflight": reproduced_stage203["resource_preflight"],
        "resource_consumption": reproduced_stage203["resource_consumption"],
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "stage203_reproduction_and_streaming_attribution": round(
                analyzed_at - authorized_at, 6
            ),
            "wall": round(analyzed_at - started_at, 6),
        },
        "execution_boundaries": {
            "train_loaded": True,
            "development_loaded": False,
            "test_loaded": False,
            "stage203_model_fit_count": reproduced_stage203["execution_boundaries"][
                "stage203_model_fit_count"
            ],
            "stage203_lightgbm_tree_count": reproduced_stage203["execution_boundaries"][
                "stage203_lightgbm_tree_count"
            ],
            "stage203_private_prediction_count": reproduced_stage203["execution_boundaries"][
                "stage203_private_prediction_count"
            ],
            "additional_diagnostic_model_fit_count": 0,
            "outer_refit_count": 0,
            "private_question_rows_persisted": False,
            "new_model_search_run": False,
            "same_weight_grid_search_run": False,
            "constraint_relaxation_run": False,
            "full_train_policy_selected": False,
            "replacement_policy_selected": False,
            "runtime_e2e_run": False,
            "runtime_registered_as_default": False,
            "stage178b_run": False,
            "retry_action_count": reproduced_stage203["execution_boundaries"]["retry_action_count"],
            "fallback_action_count": reproduced_stage203["execution_boundaries"][
                "fallback_action_count"
            ],
        },
    }
    forbidden = sorted(_forbidden_keys_found(report))
    report["public_safe_contract"] = {
        "forbidden_public_keys": sorted(FORBIDDEN_PUBLIC_KEYS),
        "forbidden_keys_found": forbidden,
        "question_level_rows_persisted": False,
        "private_decisions_persisted": False,
        "public_report_safe": not forbidden,
    }
    report["process_guards"] = _process_guards(report, formal_stage203, forbidden)
    valid = all(row["passed"] for row in report["process_guards"])
    report["decision"] = {
        "status": (
            "stage204_top1_joint_objective_failure_attribution_complete"
            if valid
            else "stage204_top1_joint_objective_failure_attribution_invalid"
        ),
        "experiment_valid": valid,
        "diagnostic_complete": valid,
        "recommended_next_research": attribution["diagnostic_finding"]["recommended_next_research"],
        "development_opened": False,
        "test_opened": False,
        "new_model_search_authorized": False,
        "same_weight_grid_search_authorized": False,
        "constraint_relaxation_authorized": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "runtime_e2e_authorized": False,
        "default_runtime_activation": False,
    }
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report


def write_stage204_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage204Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    attribution = report["failure_attribution"]
    candidates = attribution["control_to_custom"]["by_candidate"]
    precision = attribution["precision_adjacent_attribution"]["aggregate"]
    safety = attribution["safety_adjacent_attribution"]["aggregate"]
    target = attribution["target_mechanics"]
    population = attribution["population"]
    resources = report["resource_consumption"]
    charts = {
        "stage204_candidate_winner_flips.svg": _chart(
            "Stage 204 control-to-custom winner flips",
            [_count_bar(name, row["winner_flip_count"]) for name, row in candidates.items()],
            margin_left=620,
        ),
        "stage204_candidate_strict_losses.svg": _chart(
            "Stage 204 strict losses versus control",
            [_count_bar(name, row["strict_loss_count"]) for name, row in candidates.items()],
            margin_left=620,
        ),
        "stage204_candidate_strict_gains.svg": _chart(
            "Stage 204 strict gains versus control",
            [_count_bar(name, row["strict_gain_count"]) for name, row in candidates.items()],
            margin_left=620,
        ),
        "stage204_candidate_safety_repairs.svg": _chart(
            "Stage 204 unsafe repairs versus control",
            [_count_bar(name, row["unsafe_repair_count"]) for name, row in candidates.items()],
            margin_left=620,
        ),
        "stage204_candidate_unsafe_regressions.svg": _chart(
            "Stage 204 unsafe regressions versus control",
            [_count_bar(name, row["unsafe_regression_count"]) for name, row in candidates.items()],
            margin_left=620,
        ),
        "stage204_candidate_baseline_additions.svg": _chart(
            "Stage 204 baseline additions versus control",
            [_count_bar(name, row["baseline_addition_count"]) for name, row in candidates.items()],
            margin_left=620,
        ),
        "stage204_precision_adjacent_flips.svg": _chart(
            "Stage 204 adjacent precision-weight outcomes",
            [
                _count_bar("strict losses", precision["strict_loss_count"]),
                _count_bar("strict gains", precision["strict_gain_count"]),
                _count_bar("unsafe repairs", precision["unsafe_repair_count"]),
                _count_bar("unsafe regressions", precision["unsafe_regression_count"]),
                _count_bar("baseline additions", precision["baseline_addition_count"]),
                _count_bar("baseline removals", precision["baseline_removal_count"]),
            ],
        ),
        "stage204_safety_adjacent_flips.svg": _chart(
            "Stage 204 adjacent safety-weight outcomes",
            [
                _count_bar("strict losses", safety["strict_loss_count"]),
                _count_bar("strict gains", safety["strict_gain_count"]),
                _count_bar("unsafe repairs", safety["unsafe_repair_count"]),
                _count_bar("unsafe regressions", safety["unsafe_regression_count"]),
                _count_bar("baseline additions", safety["baseline_addition_count"]),
                _count_bar("baseline removals", safety["baseline_removal_count"]),
            ],
        ),
        "stage204_target_mass.svg": _chart(
            "Stage 204 target component mean mass",
            [
                _value_bar(
                    "precision baseline",
                    target["precision_component_baseline_mass"]["mean"],
                ),
                _value_bar(
                    "precision strict total",
                    target["precision_component_total_strict_mass"]["mean"],
                ),
                _value_bar("safety baseline", target["safety_component_baseline_mass"]["mean"]),
            ],
        ),
        "stage204_population.svg": _chart(
            "Stage 204 diagnostic population",
            [_count_bar(name, value) for name, value in population.items()],
            margin_left=620,
        ),
        "stage204_resources.svg": _chart(
            "Stage 204 peak resource consumption",
            [
                _value_bar(
                    "working set GiB",
                    resources["process_peak_working_set_bytes"] / (1024**3),
                ),
                _value_bar(
                    "private usage GiB",
                    resources["process_peak_private_usage_bytes"] / (1024**3),
                ),
                _value_bar(
                    "minimum available GiB",
                    resources["minimum_system_available_memory_bytes"] / (1024**3),
                ),
            ],
        ),
        "stage204_process_guards.svg": _chart(
            "Stage 204 process guards",
            [_count_bar(row["name"], int(row["passed"])) for row in report["process_guards"]],
            margin_left=680,
        ),
    }
    visualizations = []
    for name, svg in charts.items():
        ET.fromstring(svg)
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        visualizations.append(Stage204Visualization(name=name, path=str(path)))
    return tuple(visualizations)


def write_stage204_report_bundle(
    *, report: Mapping[str, Any], output_path: Path, visualization_dir: Path
) -> dict[str, Any]:
    core_report = dict(report)
    _write_json_atomic(output_path, core_report)
    visualizations = write_stage204_visualizations(report=core_report, output_dir=visualization_dir)
    final = {
        **core_report,
        "visualizations": [{"name": row.name, "path": row.path} for row in visualizations],
    }
    _write_json_atomic(output_path, final)
    return final


def write_stage204_preflight_failure(*, report: Mapping[str, Any], output_path: Path) -> None:
    _write_json_atomic(output_path, report)


def _authorize_stage203_report(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 203":
        raise ValueError("Stage204 requires Stage203")
    decision = report.get("decision", {})
    if decision.get("experiment_valid") is not True:
        raise ValueError("Stage204 requires valid Stage203 evidence")
    if decision.get("candidate_family_accepted") is not False:
        raise ValueError("Stage204 requires the insufficient Stage203 family")
    if decision.get("development_opened") is not False or decision.get("test_opened") is not False:
        raise ValueError("Stage204 requires closed development and test sets")
    guards = report.get("process_guards", [])
    if len(guards) != 38 or not all(row.get("passed") is True for row in guards):
        raise ValueError("Stage204 requires all 38 Stage203 process guards")
    cv = report.get("top1_joint_objective_nested_cv", {})
    if len(cv.get("cell_aggregates", {})) != 17:
        raise ValueError("Stage204 requires all 17 Stage203 cells")
    if any(
        row.get("outer_evaluated") is not False for row in cv.get("outer_contexts", {}).values()
    ):
        raise ValueError("Stage204 requires the five ineligible Stage203 outer contexts")


def _stage203_reproduction(
    formal: Mapping[str, Any], reproduced: Mapping[str, Any]
) -> dict[str, Any]:
    formal_cv = formal["top1_joint_objective_nested_cv"]
    actual_cv = reproduced["top1_joint_objective_nested_cv"]
    checks = {
        "status": reproduced["decision"]["status"] == formal["decision"]["status"],
        "decision": _close(reproduced["decision"], formal["decision"]),
        "stage182_reproduction": _close(
            reproduced["stage182_reproduction"], formal["stage182_reproduction"]
        ),
        "dataset": _close(actual_cv["dataset"], formal_cv["dataset"]),
        "outer_contexts": _close(actual_cv["outer_contexts"], formal_cv["outer_contexts"]),
        "aggregate": _close(actual_cv["aggregate"], formal_cv["aggregate"]),
        "aggregate_diagnostics": _close(
            actual_cv["aggregate_diagnostics"], formal_cv["aggregate_diagnostics"]
        ),
        "paired_bootstrap": _close(actual_cv["paired_bootstrap"], formal_cv["paired_bootstrap"]),
        "cell_aggregates": _close(actual_cv["cell_aggregates"], formal_cv["cell_aggregates"]),
        "ablation_aggregates": _close(
            actual_cv["ablation_family_aggregates"], formal_cv["ablation_family_aggregates"]
        ),
        "safety_weight_aggregates": _close(
            actual_cv["safety_weight_aggregates"], formal_cv["safety_weight_aggregates"]
        ),
        "precision_weight_aggregates": _close(
            actual_cv["precision_weight_aggregates"], formal_cv["precision_weight_aggregates"]
        ),
        "directional_response": _close(
            actual_cv["directional_penalty_response"], formal_cv["directional_penalty_response"]
        ),
        "advancement_gates": _close(actual_cv["advancement_gates"], formal_cv["advancement_gates"]),
        "execution_counts": _close(
            _stable_execution(actual_cv["execution"]),
            _stable_execution(formal_cv["execution"]),
        ),
        "process_guards": _close(reproduced["process_guards"], formal["process_guards"]),
    }
    return {
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "passed": all(checks.values()),
    }


def _stable_execution(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: child for key, child in value.items() if key not in {"fit_seconds", "wall_seconds"}
    }


def _process_guards(
    report: Mapping[str, Any], formal_stage203: Mapping[str, Any], forbidden: Sequence[str]
) -> list[dict[str, Any]]:
    attribution = report["failure_attribution"]
    population = attribution["population"]
    aggregate = attribution["control_to_custom"]["aggregate"]
    precision = attribution["precision_adjacent_attribution"]["aggregate"]
    safety = attribution["safety_adjacent_attribution"]["aggregate"]
    target = attribution["target_mechanics"]
    boundaries = report["execution_boundaries"]
    formal_execution = formal_stage203["execution_boundaries"]
    candidate_reports = attribution["control_to_custom"]["by_candidate"].values()
    checks = (
        ("stage203_reproduction_exact", report["stage203_reproduction"]["passed"] is True),
        ("train_only", boundaries["train_loaded"] is True),
        ("development_closed", boundaries["development_loaded"] is False),
        ("test_closed", boundaries["test_loaded"] is False),
        ("five_outer_contexts", population["outer_context_count"] == 5),
        ("eighty_five_outer_cells", population["outer_cell_context_count"] == 85),
        ("eighty_custom_outer_cells", population["custom_outer_cell_context_count"] == 80),
        ("sixteen_candidate_reports", len(tuple(candidate_reports)) == 16),
        ("question_context_count_exact", population["question_context_count"] == 1480),
        (
            "control_custom_comparison_count_exact",
            population["control_custom_question_comparison_count"] == 23_680,
        ),
        (
            "precision_adjacent_comparison_count_exact",
            population["precision_adjacent_question_comparison_count"] == 17_760,
        ),
        (
            "safety_adjacent_comparison_count_exact",
            population["safety_adjacent_question_comparison_count"] == 17_760,
        ),
        ("control_left_partition_exact", aggregate["left_partition_exact"] is True),
        ("custom_right_partition_exact", aggregate["right_partition_exact"] is True),
        (
            "control_custom_transition_partition_exact",
            aggregate["transition_partition_exact"] is True,
        ),
        (
            "candidate_partitions_exact",
            all(
                row["left_partition_exact"]
                and row["right_partition_exact"]
                and row["transition_partition_exact"]
                for row in attribution["control_to_custom"]["by_candidate"].values()
            ),
        ),
        (
            "twelve_precision_pairs",
            len(attribution["precision_adjacent_attribution"]["by_pair"]) == 12,
        ),
        ("twelve_safety_pairs", len(attribution["safety_adjacent_attribution"]["by_pair"]) == 12),
        ("precision_transition_partition_exact", precision["transition_partition_exact"] is True),
        ("safety_transition_partition_exact", safety["transition_partition_exact"] is True),
        ("target_question_count_exact", target["question_context_count"] == 1480),
        ("target_mass_exact", target["precision_component_mass_sum_exact"] is True),
        (
            "stage203_fit_count_reproduced",
            boundaries["stage203_model_fit_count"] == formal_execution["stage203_model_fit_count"],
        ),
        (
            "stage203_tree_count_reproduced",
            boundaries["stage203_lightgbm_tree_count"]
            == formal_execution["stage203_lightgbm_tree_count"],
        ),
        (
            "stage203_prediction_count_reproduced",
            boundaries["stage203_private_prediction_count"]
            == formal_execution["stage203_private_prediction_count"],
        ),
        ("no_additional_diagnostic_fits", boundaries["additional_diagnostic_model_fit_count"] == 0),
        ("no_outer_refit", boundaries["outer_refit_count"] == 0),
        ("no_private_question_rows", boundaries["private_question_rows_persisted"] is False),
        ("no_new_model_search", boundaries["new_model_search_run"] is False),
        ("no_same_grid_search", boundaries["same_weight_grid_search_run"] is False),
        ("no_constraint_relaxation", boundaries["constraint_relaxation_run"] is False),
        ("no_full_train_selection", boundaries["full_train_policy_selected"] is False),
        ("no_replacement", boundaries["replacement_policy_selected"] is False),
        ("no_runtime_e2e", boundaries["runtime_e2e_run"] is False),
        ("no_runtime_default", boundaries["runtime_registered_as_default"] is False),
        ("no_stage178b", boundaries["stage178b_run"] is False),
        ("no_retry", boundaries["retry_action_count"] == 0),
        ("no_fallback", boundaries["fallback_action_count"] == 0),
        ("no_forbidden_public_keys", not forbidden),
    )
    return [{"name": name, "passed": bool(passed)} for name, passed in checks]


def _preflight_failure_report(
    *, stage203_fingerprint: Mapping[str, Any], reproduction_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "analysis_id": ANALYSIS_ID,
        "status": "stage204_stage203_source_reproduction_failed",
        "source_authorization": {"stage203_formal_report": dict(stage203_fingerprint)},
        "stage182_reproduction": dict(reproduction_evidence),
        "execution_boundaries": {
            "stage203_model_fit_count": 0,
            "additional_diagnostic_model_fit_count": 0,
            "development_loaded": False,
            "test_loaded": False,
            "retry_action_count": 0,
            "fallback_action_count": 0,
        },
    }


def _close(left: Any, right: Any) -> bool:
    return stage199_core._nested_close(left, right)


def _chart(
    title: str,
    data: Sequence[BarDatum],
    *,
    margin_left: int = 420,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=tuple(data),
        x_label="aggregate value",
        width=1680,
        margin_left=margin_left,
        margin_right=220,
    )


def _count_bar(label: str, value: int) -> BarDatum:
    return BarDatum(label, float(value), str(value))


def _value_bar(label: str, value: float) -> BarDatum:
    return BarDatum(label, float(value), f"{value:.6f}")


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _emit(sink: stage182.ProgressSink | None, **event: Any) -> None:
    if sink is not None:
        sink(event)
