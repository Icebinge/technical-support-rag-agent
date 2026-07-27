from __future__ import annotations

import gc
import json
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from ts_rag_agent.application.composition_joint_risk_winner_cv import (
    DiagnosticSink,
    run_joint_risk_winner_nested_cv,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

STAGE = "Stage 199"
CREATED_AT = "2026-07-27"
ANALYSIS_ID = "primeqa_hybrid_joint_risk_winner_nested_cv_v1"
STAGE198_SHA256 = "62658919388603cdd2c85432399d45dfd0f50148ea4e521119f35a0d2e2e3330"
STAGE197_SHA256 = "c56f4af1b408a07e295a10f7decd2c8a0313f814f16955fd149a170355646d9d"
MINIMUM_AVAILABLE_MEMORY_BYTES = 4 * 1024**3
FORBIDDEN_PUBLIC_KEYS = stage194.FORBIDDEN_PUBLIC_KEYS | {
    "candidate_actions",
    "classifier_risk_predictions",
    "complete_pool",
    "frontier",
    "pairwise_safety_predictions",
    "question_key",
    "risk_predictions",
    "unsafe_score",
}


@dataclass(frozen=True)
class Stage199Visualization:
    name: str
    path: str


class Stage199SourceReproductionError(ValueError):
    """Expose structured preflight evidence without weakening the exact guard."""

    def __init__(self, evidence: Mapping[str, Any]) -> None:
        self.evidence = dict(evidence)
        failed = ", ".join(self.evidence.get("failed_checks", ())) or "unknown"
        super().__init__(f"Stage199 did not reproduce formal Stage182 checks: {failed}")


def run_stage199_joint_risk_winner_cv(
    *,
    stage198_protocol_path: Path,
    stage197_report_path: Path,
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
    diagnostic_sink: DiagnosticSink | None = None,
) -> dict[str, Any]:
    """Reproduce Stage182 and run the authorized Stage199 train-only nested CV."""

    started_at = time.perf_counter()
    stage198_fingerprint = stage173._resolved_fingerprint(stage198_protocol_path)
    if stage198_fingerprint["sha256"] != STAGE198_SHA256:
        raise ValueError("Stage199 Stage198 protocol hash mismatch")
    stage197_fingerprint = stage173._resolved_fingerprint(stage197_report_path)
    if stage197_fingerprint["sha256"] != STAGE197_SHA256:
        raise ValueError("Stage199 Stage197 report hash mismatch")
    formal_stage198 = _load_json(stage198_protocol_path)
    formal_stage197 = _load_json(stage197_report_path)
    _authorize_stage198_protocol(formal_stage198)
    _authorize_stage197_report(formal_stage197)
    dependency_preflight = stage194._dependency_preflight(
        lightgbm_wheel_path=lightgbm_wheel_path,
        narwhals_wheel_path=narwhals_wheel_path,
    )
    formal_stage182 = _load_json(stage182_report_path)
    stage183._authorize_stage182_report(formal_stage182)
    authorized_at = time.perf_counter()

    import torch

    tracker = stage169.Stage169ResourceTracker(torch_module=torch)
    preflight_snapshot = tracker.capture("stage199_preflight")
    if preflight_snapshot.system_available_memory_bytes < MINIMUM_AVAILABLE_MEMORY_BYTES:
        raise RuntimeError("Stage199 requires at least 4 GiB available system memory")
    _emit(progress_sink, phase="stage198_protocol_dependencies_and_memory_authorized")
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
        raise Stage199SourceReproductionError(reproduction)
    gc.collect()
    tracker.capture("stage182_temporary_resources_released")

    def cv_progress(event: Mapping[str, Any]) -> None:
        if event.get("phase") in {
            "stage199_inner_partition_complete",
            "stage199_outer_context_complete",
            "stage199_outer_context_no_eligible_config",
        }:
            tracker.capture(str(event["phase"]))
        _emit(progress_sink, **dict(event))

    nested_cv = run_joint_risk_winner_nested_cv(
        action_rows=private["action_rows"],
        stage182_selected_actions=private["selected_actions"],
        stage198_protocol=formal_stage198,
        stage197_report=formal_stage197,
        progress_sink=cv_progress,
        diagnostic_sink=diagnostic_sink,
    )
    tracker.capture("joint_risk_winner_nested_cv_complete")
    analyzed_at = time.perf_counter()
    report: dict[str, Any] = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "analysis_id": ANALYSIS_ID,
        "analysis_scope": (
            "Train-only five-by-four nested CV of the frozen 4x7 joint risk-signal "
            "and winner-rule factorial. Development and test remain closed; no fallback, "
            "retry, weaker candidate, runtime E2E, full-train selection, replacement, "
            "or default activation occurs."
        ),
        "source_authorization": {
            "stage198_protocol": stage198_fingerprint,
            "stage197_report": stage197_fingerprint,
            "stage182_rerun_sources": reproduced_stage182["source_authorization"],
        },
        "dependency_preflight": dependency_preflight,
        "resource_preflight": {
            "frozen_minimum_available_memory_bytes": MINIMUM_AVAILABLE_MEMORY_BYTES,
            "actual_available_memory_bytes": preflight_snapshot.system_available_memory_bytes,
            "frozen_threshold_met": True,
            "memory_override_authorized": False,
            "factorial_grid_reduced": False,
            "fallback_enabled": False,
        },
        "frozen_protocol": formal_stage198["frozen_protocol"],
        "stage182_reproduction": reproduction,
        "joint_risk_winner_nested_cv": nested_cv,
        "runtime": reproduced_stage182["runtime"],
        "resource_consumption": stage181._resource_summary(tracker.snapshots),
        "timing_seconds": {
            "source_dependency_and_memory_authorization": round(authorized_at - started_at, 6),
            "stage182_reproduction": round(reproduced_at - authorized_at, 6),
            "joint_risk_winner_nested_cv": round(analyzed_at - reproduced_at, 6),
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
            "stage199_model_fit_count": nested_cv["execution"]["model_fit_count"],
            "stage199_lightgbm_tree_count": nested_cv["execution"]["tree_count"],
            "stage199_private_prediction_count": nested_cv["execution"]["private_prediction_count"],
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
            "stage199_joint_risk_winner_candidate_family_found"
            if valid and accepted
            else "stage199_joint_risk_winner_insufficient"
            if valid
            else "stage199_joint_risk_winner_invalid"
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


def write_stage199_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage199Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv = report["joint_risk_winner_nested_cv"]
    resources = report["resource_consumption"]
    top_rows = [
        (fold_id, row["top_inner_candidates"][0]) for fold_id, row in cv["outer_contexts"].items()
    ]
    best_paired = sorted(
        cv["cell_aggregates"].values(),
        key=lambda row: (
            row["paired_vs_control"]["unsafe_selected_count_delta"],
            -row["strict_success_count"],
            row["spec"]["name"],
        ),
    )[:10]
    charts = {
        "stage199_outer_eligible_counts.svg": _chart(
            "Stage 199 inner-eligible cells by outer context",
            [
                _count_bar(fold_id, row["eligible_config_count"])
                for fold_id, row in cv["outer_contexts"].items()
            ],
        ),
        "stage199_control_reproduction.svg": _chart(
            "Stage 199 exact control reproduction",
            [
                _bool_bar(fold_id, row["control_reproduction_exact"])
                for fold_id, row in cv["outer_contexts"].items()
            ],
        ),
        "stage199_top_inner_unsafe.svg": _chart(
            "Stage 199 top-inner unsafe rate",
            [
                _rate_bar(fold_id, candidate["diagnostics"]["unsafe_selection_rate"])
                for fold_id, candidate in top_rows
            ],
        ),
        "stage199_top_inner_capture.svg": _chart(
            "Stage 199 top-inner conditional strict capture",
            [
                _rate_bar(
                    fold_id,
                    candidate["diagnostics"]["conditional_ranker_strict_capture"],
                )
                for fold_id, candidate in top_rows
            ],
        ),
        "stage199_risk_signal_unsafe.svg": _chart(
            "Stage 199 risk-signal marginal mean unsafe rate",
            [
                _rate_bar(name, row["mean_unsafe_selection_rate"])
                for name, row in cv["risk_signal_factor_aggregates"].items()
            ],
            margin_left=620,
        ),
        "stage199_winner_rule_unsafe.svg": _chart(
            "Stage 199 winner-rule marginal mean unsafe rate",
            [
                _rate_bar(name, row["mean_unsafe_selection_rate"])
                for name, row in cv["winner_rule_factor_aggregates"].items()
            ],
            margin_left=620,
        ),
        "stage199_risk_signal_capture.svg": _chart(
            "Stage 199 risk-signal marginal mean capture",
            [
                _rate_bar(name, row["mean_conditional_capture"])
                for name, row in cv["risk_signal_factor_aggregates"].items()
            ],
            margin_left=620,
        ),
        "stage199_winner_rule_capture.svg": _chart(
            "Stage 199 winner-rule marginal mean capture",
            [
                _rate_bar(name, row["mean_conditional_capture"])
                for name, row in cv["winner_rule_factor_aggregates"].items()
            ],
            margin_left=620,
        ),
        "stage199_complete_pool_risk_auc.svg": _chart(
            "Stage 199 complete-pool unsafe ROC AUC",
            [
                _rate_bar(name, row["roc_auc"])
                for name, row in cv["complete_pool_risk_metrics"].items()
            ],
            margin_left=620,
        ),
        "stage199_best_paired_unsafe_delta.svg": _chart(
            "Stage 199 best cell unsafe-count deltas vs control",
            [
                _signed_bar(
                    row["spec"]["name"],
                    row["paired_vs_control"]["unsafe_selected_count_delta"],
                )
                for row in best_paired
            ],
            margin_left=900,
        ),
        "stage199_selected_factors.svg": _chart(
            "Stage 199 selected outer factor counts",
            [
                *(
                    _count_bar(f"risk: {name}", value)
                    for name, value in cv["selected_risk_signal_counts"].items()
                ),
                *(
                    _count_bar(f"winner: {name}", value)
                    for name, value in cv["selected_winner_rule_counts"].items()
                ),
            ],
            margin_left=720,
        ),
        "stage199_advancement_gates.svg": _chart(
            "Stage 199 advancement gates",
            [_bool_bar(row["name"], row["passed"]) for row in cv["advancement_gates"]],
            margin_left=720,
        ),
        "stage199_execution.svg": _chart(
            "Stage 199 execution counts",
            [
                _count_bar("model fits", cv["execution"]["model_fit_count"]),
                _count_bar("LightGBM trees", cv["execution"]["tree_count"]),
                _count_bar("private predictions", cv["execution"]["private_prediction_count"]),
            ],
        ),
        "stage199_resources.svg": _chart(
            "Stage 199 resource consumption (GiB)",
            [
                _gib_bar("peak working set", resources["process_peak_working_set_bytes"]),
                _gib_bar("peak private usage", resources["process_peak_private_usage_bytes"]),
                _gib_bar(
                    "minimum system available",
                    resources["minimum_system_available_memory_bytes"],
                ),
            ],
        ),
        "stage199_process_guards.svg": _chart(
            "Stage 199 process guards",
            [_bool_bar(row["name"], row["passed"]) for row in report["process_guards"]],
            margin_left=820,
        ),
    }
    visualizations = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        visualizations.append(Stage199Visualization(name, str(path)))
    return tuple(visualizations)


def write_stage199_report_bundle(
    *,
    report: Mapping[str, Any],
    output_path: Path,
    visualization_dir: Path,
) -> dict[str, Any]:
    """Persist core evidence before running visualization postprocessing."""
    core_report = dict(report)
    _write_json_atomic(output_path, core_report)
    visualizations = write_stage199_visualizations(
        report=core_report,
        output_dir=visualization_dir,
    )
    final_report = {
        **core_report,
        "visualizations": [{"name": item.name, "path": item.path} for item in visualizations],
    }
    _write_json_atomic(output_path, final_report)
    return final_report


def _authorize_stage198_protocol(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 198":
        raise ValueError("Stage199 requires Stage198")
    decision = report.get("decision", {})
    if decision.get("protocol_valid") is not True:
        raise ValueError("Stage199 requires a valid Stage198 protocol")
    if decision.get("stage199_train_only_experiment_authorized") is not True:
        raise ValueError("Stage198 did not authorize Stage199")
    if decision.get("development_opened") is not False or decision.get("test_opened") is not False:
        raise ValueError("Stage199 requires closed development and test sets")
    if len(report.get("guard_checks", [])) != 62 or not all(
        row.get("passed") is True for row in report["guard_checks"]
    ):
        raise ValueError("Stage199 requires all 62 Stage198 guards")


def _authorize_stage197_report(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 197":
        raise ValueError("Stage199 requires Stage197 control evidence")
    decision = report.get("decision", {})
    if (
        decision.get("experiment_valid") is not True
        or decision.get("diagnostic_complete") is not True
    ):
        raise ValueError("Stage199 requires valid Stage197 evidence")
    if decision.get("development_opened") is not False or decision.get("test_opened") is not False:
        raise ValueError("Stage199 requires closed Stage197 development and test sets")


def _process_guards(report: Mapping[str, Any], forbidden: Sequence[str]) -> list[dict[str, Any]]:
    cv = report["joint_risk_winner_nested_cv"]
    execution = cv["execution"]
    boundaries = report["execution_boundaries"]
    eligible_outer = sum(row["outer_evaluated"] for row in cv["outer_contexts"].values())
    evaluated_action_count = sum(
        cv["dataset"]["fold_action_counts"][fold_id]
        for fold_id, row in cv["outer_contexts"].items()
        if row["outer_evaluated"]
    )
    expected_fit_count = 100 + 5 * eligible_outer
    expected_tree_count = 18_000 + 900 * eligible_outer
    expected_private_predictions = 245_960 + 5 * evaluated_action_count
    risk_metric_counts = {
        row["action_context_count"] for row in cv["complete_pool_risk_metrics"].values()
    }
    checks = (
        ("stage182_reproduction_exact", report["stage182_reproduction"]["passed"] is True),
        ("train_only", boundaries["train_loaded"] is True),
        ("development_closed", boundaries["development_loaded"] is False),
        ("test_closed", boundaries["test_loaded"] is False),
        ("five_outer_contexts", len(cv["outer_contexts"]) == 5),
        ("all_controls_reproduced", execution["all_controls_reproduced_exactly"] is True),
        ("five_control_reproductions", execution["control_reproduction_count"] == 5),
        ("twenty_eight_cells", len(cv["cell_aggregates"]) == 28),
        ("four_risk_factor_rows", len(cv["risk_signal_factor_aggregates"]) == 4),
        ("seven_winner_factor_rows", len(cv["winner_rule_factor_aggregates"]) == 7),
        ("four_risk_metric_rows", len(cv["complete_pool_risk_metrics"]) == 4),
        ("risk_metric_populations_equal", len(risk_metric_counts) == 1),
        ("risk_metric_population_nonzero", next(iter(risk_metric_counts), 0) > 0),
        ("model_fit_count_exact", execution["model_fit_count"] == expected_fit_count),
        (
            "pool_safety_fit_count_exact",
            execution["pool_safety_fit_count"] == 2 * expected_fit_count // 5,
        ),
        ("gain_fit_count_exact", execution["gain_ranker_fit_count"] == expected_fit_count // 5),
        (
            "classifier_fit_count_exact",
            execution["classifier_risk_fit_count"] == expected_fit_count // 5,
        ),
        (
            "pairwise_fit_count_exact",
            execution["pairwise_safety_fit_count"] == expected_fit_count // 5,
        ),
        ("tree_count_exact", execution["tree_count"] == expected_tree_count),
        (
            "group_contract_count_exact",
            execution["group_contract_validation_count"] == 2 * expected_fit_count // 5,
        ),
        (
            "private_prediction_count_exact",
            execution["private_prediction_count"] == expected_private_predictions,
        ),
        ("fit_budget_respected", execution["model_fit_count"] <= 125),
        ("tree_budget_respected", execution["tree_count"] <= 22_500),
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


def _signed_bar(name: str, value: int | float) -> BarDatum:
    return BarDatum(name, float(value), f"{value:+}")


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
