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
from ts_rag_agent.application.composition_safety_first_frontier import (
    run_safety_first_frontier_nested_cv,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

STAGE = "Stage 196"
CREATED_AT = "2026-07-27"
ANALYSIS_ID = "primeqa_hybrid_safety_first_frontier_nested_cv_v1"
STAGE195_SHA256 = "dc02e8423d633481802e42c6d52e85b9e1bda58861d1fc61492819b027b2c637"
MINIMUM_AVAILABLE_MEMORY_BYTES = 4 * 1024**3
FORBIDDEN_PUBLIC_KEYS = stage194.FORBIDDEN_PUBLIC_KEYS | {
    "complete_pool",
    "frontier",
    "gain_predictions",
    "risk_predictions",
    "unsafe_score",
}


@dataclass(frozen=True)
class Stage196Visualization:
    name: str
    path: str


def run_stage196_safety_first_frontier_cv(
    *,
    stage195_protocol_path: Path,
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
) -> dict[str, Any]:
    """Reproduce Stage 182 and run the authorized Stage 196 train-only CV."""

    started_at = time.perf_counter()
    protocol_fingerprint = stage173._resolved_fingerprint(stage195_protocol_path)
    if protocol_fingerprint["sha256"] != STAGE195_SHA256:
        raise ValueError("Stage196 Stage195 protocol hash mismatch")
    formal_protocol = _load_json(stage195_protocol_path)
    _authorize_stage195_protocol(formal_protocol)
    dependency_preflight = stage194._dependency_preflight(
        lightgbm_wheel_path=lightgbm_wheel_path,
        narwhals_wheel_path=narwhals_wheel_path,
    )
    formal_stage182 = _load_json(stage182_report_path)
    stage183._authorize_stage182_report(formal_stage182)
    authorized_at = time.perf_counter()

    import torch

    tracker = stage169.Stage169ResourceTracker(torch_module=torch)
    preflight_snapshot = tracker.capture("stage196_preflight")
    if preflight_snapshot.system_available_memory_bytes < MINIMUM_AVAILABLE_MEMORY_BYTES:
        raise RuntimeError("Stage196 requires at least 4 GiB available system memory")
    _emit(progress_sink, phase="stage195_protocol_dependencies_and_memory_authorized")
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
        raise ValueError("Stage196 did not reproduce the formal Stage182 result")
    gc.collect()
    tracker.capture("stage182_temporary_resources_released")

    def cv_progress(event: Mapping[str, Any]) -> None:
        phase = str(event.get("phase", "stage196_cv_event"))
        if phase in {
            "inner_partition_complete",
            "outer_fold_complete",
            "outer_fold_no_eligible_config",
        }:
            tracker.capture(phase)
        _emit(progress_sink, **dict(event))

    nested_cv = run_safety_first_frontier_nested_cv(
        action_rows=private["action_rows"],
        stage182_selected_actions=private["selected_actions"],
        progress_sink=cv_progress,
    )
    tracker.capture("safety_first_frontier_nested_cv_complete")
    analyzed_at = time.perf_counter()
    report: dict[str, Any] = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "analysis_id": ANALYSIS_ID,
        "analysis_scope": (
            "Train-only five-by-four nested cross-validation of 960 fixed cap-16, "
            "cost-sensitive unsafe-head, and safest-prefix frontier configurations. "
            "Development and test remain closed; no fallback, retry, weaker candidate, "
            "runtime E2E, full-train selection, replacement, or default activation occurs."
        ),
        "source_authorization": {
            "stage195_protocol": protocol_fingerprint,
            "stage182_rerun_sources": reproduced_stage182["source_authorization"],
        },
        "dependency_preflight": dependency_preflight,
        "resource_preflight": {
            "frozen_minimum_available_memory_bytes": MINIMUM_AVAILABLE_MEMORY_BYTES,
            "actual_available_memory_bytes": preflight_snapshot.system_available_memory_bytes,
            "frozen_threshold_met": True,
            "memory_override_authorized": False,
            "model_grid_reduced": False,
            "fallback_enabled": False,
        },
        "frozen_protocol": formal_protocol["frozen_protocol"],
        "stage182_reproduction": reproduction,
        "safety_first_frontier_nested_cv": nested_cv,
        "runtime": reproduced_stage182["runtime"],
        "resource_consumption": stage181._resource_summary(tracker.snapshots),
        "timing_seconds": {
            "source_dependency_and_memory_authorization": round(authorized_at - started_at, 6),
            "stage182_reproduction": round(reproduced_at - authorized_at, 6),
            "safety_first_frontier_nested_cv": round(analyzed_at - reproduced_at, 6),
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
            "stage196_model_fit_count": nested_cv["execution"]["model_fit_count"],
            "stage196_lightgbm_tree_count": nested_cv["execution"]["tree_count"],
            "stage196_private_prediction_count": nested_cv["execution"]["private_prediction_count"],
            "stage196_public_training_rows_written": 0,
            "stage196_public_prediction_rows_written": 0,
            "one_representation_and_one_weighted_risk_model_at_a_time": True,
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
    report["process_guards"] = _process_guards(report=report, forbidden=forbidden)
    valid = all(row["passed"] for row in report["process_guards"])
    accepted = nested_cv["candidate_family_accepted"] if valid else False
    report["decision"] = {
        "status": (
            "stage196_safety_first_frontier_candidate_family_found"
            if valid and accepted
            else "stage196_safety_first_frontier_insufficient"
            if valid
            else "stage196_safety_first_frontier_invalid"
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


def write_stage196_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage196Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv = report["safety_first_frontier_nested_cv"]
    aggregate = cv["aggregate"]
    diagnostics = cv["aggregate_diagnostics"]
    resources = report["resource_consumption"]
    charts = {
        "stage196_inner_eligible_configs.svg": _outer_count_chart(
            cv,
            "Stage 196 inner-eligible configurations",
            "eligible_config_count",
        ),
        "stage196_top_inner_pool_recall.svg": _top_inner_diagnostic_chart(
            cv, "Stage 196 top-inner pool recall", "strict_opportunity_pool_recall"
        ),
        "stage196_top_inner_frontier_recall.svg": _top_inner_diagnostic_chart(
            cv, "Stage 196 top-inner frontier recall", "strict_opportunity_frontier_recall"
        ),
        "stage196_top_inner_conditional_capture.svg": _top_inner_diagnostic_chart(
            cv,
            "Stage 196 top-inner conditional strict capture",
            "conditional_ranker_strict_capture",
        ),
        "stage196_top_inner_unsafe_rate.svg": _top_inner_diagnostic_chart(
            cv, "Stage 196 top-inner unsafe selection rate", "unsafe_selection_rate"
        ),
        "stage196_top_inner_unsafe_retention.svg": _top_inner_diagnostic_chart(
            cv, "Stage 196 top-inner unsafe-action retention", "unsafe_action_retention_rate"
        ),
        "stage196_top_inner_frontier_size.svg": _top_inner_diagnostic_chart(
            cv, "Stage 196 top-inner mean frontier size", "mean_frontier_size"
        ),
        "stage196_aggregate_diagnostics.svg": _chart(
            "Stage 196 aggregate frontier diagnostics",
            tuple(
                BarDatum(name, diagnostics[name], f"{diagnostics[name]:.3f}")
                for name in (
                    "strict_opportunity_pool_recall",
                    "strict_opportunity_frontier_recall",
                    "conditional_ranker_strict_capture",
                    "unsafe_selection_rate",
                    "unsafe_action_retention_rate",
                )
            ),
            "rate",
            margin_left=650,
        ),
        "stage196_aggregate_outcomes.svg": _chart(
            "Stage 196 aggregate selected-action outcomes",
            tuple(
                BarDatum(name, value, str(value))
                for name, value in (
                    ("changed questions", aggregate["changed_question_count"]),
                    ("strict successes", aggregate["strict_success_count"]),
                    ("citation gains", aggregate["citation_gain_action_count"]),
                    ("citation losses", aggregate["citation_loss_action_count"]),
                    ("F1 regressions", aggregate["f1_regression_action_count"]),
                )
            ),
            "question or action count",
        ),
        "stage196_top_inner_prefixes.svg": _top_inner_spec_chart(
            cv,
            "Stage 196 top-inner safest prefixes",
            "safest_prefix_size",
        ),
        "stage196_top_inner_risk_weights.svg": _top_inner_spec_chart(
            cv,
            "Stage 196 top-inner unsafe positive-class weights",
            "scale_pos_weight",
        ),
        "stage196_execution_counts.svg": _chart(
            "Stage 196 model execution",
            tuple(
                BarDatum(name, value, str(value))
                for name, value in (
                    ("all model fits", cv["execution"]["model_fit_count"]),
                    ("pool safety fits", cv["execution"]["pool_safety_fit_count"]),
                    ("LambdaMART fits", cv["execution"]["lambdamart_fit_count"]),
                    ("unsafe-head fits", cv["execution"]["unsafe_head_fit_count"]),
                    ("LightGBM trees", cv["execution"]["tree_count"]),
                )
            ),
            "count",
        ),
        "stage196_memory_gib.svg": _chart(
            "Stage 196 memory usage",
            (
                _gib_bar("peak working set", resources["process_peak_working_set_bytes"]),
                _gib_bar("peak private usage", resources["process_peak_private_usage_bytes"]),
                _gib_bar("minimum system free", resources["minimum_system_available_memory_bytes"]),
            ),
            "GiB",
        ),
        "stage196_advancement_gates.svg": _pass_chart(
            "Stage 196 advancement gates", cv["advancement_gates"]
        ),
        "stage196_process_guards.svg": _pass_chart(
            "Stage 196 process guards", report["process_guards"]
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage196Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _authorize_stage195_protocol(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 195":
        raise ValueError("Stage196 requires the Stage195 protocol")
    decision = report.get("decision", {})
    if decision.get("status") != "stage195_safety_first_frontier_protocol_frozen":
        raise ValueError("Stage196 requires the frozen Stage195 protocol")
    if decision.get("protocol_valid") is not True:
        raise ValueError("Stage196 requires a valid Stage195 protocol")
    if decision.get("stage196_train_only_experiment_authorized") is not True:
        raise ValueError("Stage195 did not authorize Stage196")
    if len(report.get("guard_checks", [])) != 58 or not all(
        row.get("passed") is True for row in report.get("guard_checks", [])
    ):
        raise ValueError("Stage195 guard checks drifted")
    protocol = report.get("frozen_protocol", {})
    if protocol.get("candidate_grid", {}).get("policy_config_count") != 960:
        raise ValueError("Stage195 policy grid drifted")
    cv = protocol.get("cross_validation", {})
    if cv.get("maximum_model_fit_count") != 600:
        raise ValueError("Stage195 fit budget drifted")
    if cv.get("maximum_lightgbm_tree_count") != 120_000:
        raise ValueError("Stage195 tree budget drifted")
    if protocol.get("first_stage_pool", {}).get("pool_cap") != 16:
        raise ValueError("Stage195 pool cap drifted")
    risk = protocol.get("cost_sensitive_unsafe_head", {})
    if risk.get("scale_pos_weights") != [1.0, 2.0, 4.0]:
        raise ValueError("Stage195 risk weights drifted")
    frontier = protocol.get("safety_first_frontier", {})
    if frontier.get("safest_prefix_sizes") != [2, 4, 8, 12, 16]:
        raise ValueError("Stage195 prefix sizes drifted")
    if frontier.get("fallback_used") is not False:
        raise ValueError("Stage195 fallback boundary drifted")
    if (
        protocol.get("resource_contract", {}).get("minimum_preflight_system_available_memory_gib")
        != 4.0
    ):
        raise ValueError("Stage195 memory boundary drifted")


def _process_guards(*, report: Mapping[str, Any], forbidden: Sequence[str]) -> list[dict[str, Any]]:
    boundaries = report["execution_boundaries"]
    cv = report["safety_first_frontier_nested_cv"]
    eligible_outer = sum(row["outer_evaluated"] for row in cv["outer_folds"].values())
    partition_count = 20 + eligible_outer
    expected_fits = 24 * partition_count
    expected_trees = 16 * 300 * partition_count
    checks = (
        ("stage195_protocol_hash_matches", True),
        ("stage195_protocol_authorized", True),
        ("dependency_preflight_passed", True),
        (
            "four_gib_resource_preflight_passed",
            report["resource_preflight"]["frozen_threshold_met"],
        ),
        ("no_memory_override", not report["resource_preflight"]["memory_override_authorized"]),
        ("grid_not_reduced", not report["resource_preflight"]["model_grid_reduced"]),
        ("stage182_reproduction_passed", report["stage182_reproduction"]["passed"]),
        ("captured_action_row_count_is_12298", boundaries["captured_action_row_count"] == 12298),
        (
            "captured_stage182_selected_action_count_is_129",
            boundaries["captured_stage182_selected_action_count"] == 129,
        ),
        ("question_count_is_370", cv["dataset"]["question_count"] == 370),
        ("policy_config_count_is_960", cv["protocol"]["policy_config_count"] == 960),
        ("advancement_gate_count_is_17", len(cv["advancement_gates"]) == 17),
        ("model_fit_count_exact", cv["execution"]["model_fit_count"] == expected_fits),
        ("model_fit_count_within_600", cv["execution"]["model_fit_count"] <= 600),
        (
            "pool_safety_fit_count_exact",
            cv["execution"]["pool_safety_fit_count"] == 8 * partition_count,
        ),
        (
            "lambdamart_fit_count_exact",
            cv["execution"]["lambdamart_fit_count"] == 4 * partition_count,
        ),
        (
            "unsafe_head_fit_count_exact",
            cv["execution"]["unsafe_head_fit_count"] == 12 * partition_count,
        ),
        ("lightgbm_tree_count_exact", cv["execution"]["tree_count"] == expected_trees),
        ("lightgbm_tree_count_within_120000", cv["execution"]["tree_count"] <= 120_000),
        (
            "group_contracts_exact",
            cv["execution"]["group_contract_validation_count"] == 2 * partition_count,
        ),
        (
            "one_representation_and_weighted_risk_model_at_a_time",
            boundaries["one_representation_and_one_weighted_risk_model_at_a_time"],
        ),
        (
            "private_rows_not_public",
            boundaries["stage196_public_training_rows_written"] == 0
            and boundaries["stage196_public_prediction_rows_written"] == 0,
        ),
        ("private_predictions_exist", boundaries["stage196_private_prediction_count"] > 0),
        ("train_loaded", boundaries["train_loaded"]),
        ("development_closed", not boundaries["development_loaded"]),
        ("test_closed", not boundaries["test_loaded"]),
        (
            "gold_offline_only",
            boundaries["gold_used_only_for_training_targets_and_offline_evaluation"],
        ),
        ("full_train_policy_not_selected", not boundaries["full_train_policy_selected"]),
        ("replacement_policy_not_selected", not boundaries["replacement_policy_selected"]),
        ("runtime_e2e_not_run", not boundaries["runtime_e2e_run"]),
        ("default_runtime_unchanged", not boundaries["runtime_registered_as_default"]),
        ("stage178b_not_run", not boundaries["stage178b_run"]),
        ("no_retry", boundaries["retry_action_count"] == 0),
        ("no_fallback", boundaries["fallback_action_count"] == 0),
        ("public_report_safe", not forbidden),
    )
    return [_gate(name, passed) for name, passed in checks]


def _outer_count_chart(cv: Mapping[str, Any], title: str, metric: str) -> str:
    return _chart(
        title,
        tuple(
            BarDatum(fold, row[metric], str(row[metric])) for fold, row in cv["outer_folds"].items()
        ),
        "count",
    )


def _top_inner_diagnostic_chart(cv: Mapping[str, Any], title: str, metric: str) -> str:
    return _chart(
        title,
        tuple(
            BarDatum(
                fold,
                row["top_inner_candidates"][0]["diagnostics"][metric],
                f"{row['top_inner_candidates'][0]['diagnostics'][metric]:.3f}",
            )
            for fold, row in cv["outer_folds"].items()
        ),
        "rate",
    )


def _top_inner_spec_chart(cv: Mapping[str, Any], title: str, field: str) -> str:
    return _chart(
        title,
        tuple(
            BarDatum(
                fold,
                float(row["top_inner_candidates"][0]["spec"][field]),
                str(row["top_inner_candidates"][0]["spec"][field]),
            )
            for fold, row in cv["outer_folds"].items()
        ),
        field,
    )


def _pass_chart(title: str, rows: Sequence[Mapping[str, Any]]) -> str:
    return _chart(
        title,
        tuple(
            BarDatum(row["name"], float(row["passed"]), "pass" if row["passed"] else "fail")
            for row in rows
        ),
        "1 means passed",
        margin_left=940,
    )


def _gib_bar(name: str, byte_count: int) -> BarDatum:
    value = byte_count / 1024**3
    return BarDatum(name, value, f"{value:.3f}")


def _chart(
    title: str,
    data: Sequence[BarDatum],
    x_label: str,
    *,
    margin_left: int = 440,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=data,
        x_label=x_label,
        width=1680,
        margin_left=margin_left,
        margin_right=220,
    )


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


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(progress_sink: stage182.ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
