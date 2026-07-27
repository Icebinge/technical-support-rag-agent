from __future__ import annotations

import gc
import importlib.metadata
import json
import subprocess
import sys
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
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application.composition_action_audit import ActionAuditRow
from ts_rag_agent.application.composition_dual_target_policy import (
    DualTargetPrediction,
    SelectedAction,
)
from ts_rag_agent.application.composition_safety_constrained_lambdamart import (
    run_safety_constrained_lambdamart_nested_cv,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

STAGE = "Stage 194"
CREATED_AT = "2026-07-27"
ANALYSIS_ID = "primeqa_hybrid_safety_constrained_lambdamart_nested_cv_v1"
STAGE193_SHA256 = "3124f186166fb8d04886c75801d271f82fb9b317a54026f97f055c10cefa9930"
LIGHTGBM_WHEEL_SHA256 = "f42d1e5b32b6f170e606d7c689c6165671da98d7bf37f1addec2623efc8740c9"
NARWHALS_WHEEL_SHA256 = "42fdedf44e5b2ca7505630d45b4ac3058f38d8485cba9fe1652ca23152df7489"
MINIMUM_AVAILABLE_MEMORY_BYTES = 6 * 1024**3
FORBIDDEN_PUBLIC_KEYS = stage183._FORBIDDEN_PUBLIC_KEYS | {
    "candidate_actions",
    "citation_loss_probability",
    "document_text",
    "f1_loss_probability",
    "feature_rows",
    "gain_score",
    "group_labels",
    "predictions",
    "question_text",
    "selected_actions",
    "unsafe_probability",
}


@dataclass(frozen=True)
class Stage194Visualization:
    name: str
    path: str


def run_stage194_safety_constrained_lambdamart_cv(
    *,
    stage193_protocol_path: Path,
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
    allow_below_frozen_memory_threshold: bool = False,
    memory_override_note: str = "",
    progress_sink: stage182.ProgressSink | None = None,
) -> dict[str, Any]:
    """Reproduce Stage 182 and run the authorized Stage 194 train-only CV."""

    started_at = time.perf_counter()
    stage193_fingerprint = stage173._resolved_fingerprint(stage193_protocol_path)
    if stage193_fingerprint["sha256"] != STAGE193_SHA256:
        raise ValueError("Stage194 Stage193 protocol hash mismatch")
    formal_stage193 = _load_json(stage193_protocol_path)
    _authorize_stage193_protocol(formal_stage193)
    dependency_preflight = _dependency_preflight(
        lightgbm_wheel_path=lightgbm_wheel_path,
        narwhals_wheel_path=narwhals_wheel_path,
    )
    formal_stage182 = _load_json(stage182_report_path)
    stage183._authorize_stage182_report(formal_stage182)
    authorized_at = time.perf_counter()

    import torch

    tracker = stage169.Stage169ResourceTracker(torch_module=torch)
    preflight_snapshot = tracker.capture("stage194_preflight")
    frozen_memory_threshold_met = (
        preflight_snapshot.system_available_memory_bytes >= MINIMUM_AVAILABLE_MEMORY_BYTES
    )
    if not frozen_memory_threshold_met and not allow_below_frozen_memory_threshold:
        raise RuntimeError("Stage194 requires at least 6 GiB available system memory")
    if not frozen_memory_threshold_met and not memory_override_note.strip():
        raise ValueError("Stage194 memory-threshold override requires an explicit note")
    _emit(progress_sink, phase="stage193_protocol_and_dependencies_authorized")
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
        raise ValueError("Stage194 did not reproduce the formal Stage182 result")
    gc.collect()
    tracker.capture("stage182_temporary_resources_released")

    def cv_progress(event: Mapping[str, Any]) -> None:
        phase = str(event.get("phase", "stage194_cv_event"))
        if phase in {
            "inner_partition_complete",
            "outer_fold_complete",
            "outer_fold_no_eligible_config",
        }:
            tracker.capture(phase)
        _emit(progress_sink, **dict(event))

    nested_cv = run_safety_constrained_lambdamart_nested_cv(
        action_rows=private["action_rows"],
        stage182_selected_actions=private["selected_actions"],
        progress_sink=cv_progress,
    )
    tracker.capture("safety_constrained_lambdamart_nested_cv_complete")
    analyzed_at = time.perf_counter()
    report: dict[str, Any] = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "analysis_id": ANALYSIS_ID,
        "analysis_scope": (
            "Train-only five-by-four nested cross-validation of 64 fixed cap-16 "
            "safety-pool, LambdaMART, and independent unsafe-risk configurations. "
            "Development and test remain closed; no fallback, runtime E2E, full-train "
            "selection, replacement decision, or default activation is performed."
        ),
        "source_authorization": {
            "stage193_protocol": stage193_fingerprint,
            "stage182_rerun_sources": reproduced_stage182["source_authorization"],
        },
        "dependency_preflight": dependency_preflight,
        "resource_preflight": {
            "frozen_minimum_available_memory_bytes": MINIMUM_AVAILABLE_MEMORY_BYTES,
            "actual_available_memory_bytes": preflight_snapshot.system_available_memory_bytes,
            "frozen_threshold_met": frozen_memory_threshold_met,
            "explicit_user_override_authorized": (
                bool(allow_below_frozen_memory_threshold)
                if not frozen_memory_threshold_met
                else False
            ),
            "override_note": (
                memory_override_note.strip() if not frozen_memory_threshold_met else ""
            ),
            "model_grid_reduced": False,
            "fallback_enabled": False,
        },
        "frozen_protocol": formal_stage193["frozen_protocol"],
        "stage182_reproduction": reproduction,
        "safety_constrained_lambdamart_nested_cv": nested_cv,
        "runtime": reproduced_stage182["runtime"],
        "resource_consumption": stage181._resource_summary(tracker.snapshots),
        "timing_seconds": {
            "source_and_dependency_authorization": round(authorized_at - started_at, 6),
            "stage182_reproduction": round(reproduced_at - authorized_at, 6),
            "safety_constrained_lambdamart_nested_cv": round(analyzed_at - reproduced_at, 6),
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
            "stage194_model_fit_count": nested_cv["execution"]["model_fit_count"],
            "stage194_lightgbm_tree_count": nested_cv["execution"]["tree_count"],
            "stage194_private_prediction_count": nested_cv["execution"]["private_prediction_count"],
            "stage194_public_training_rows_written": 0,
            "stage194_public_prediction_rows_written": 0,
            "one_representation_materialized_at_a_time": True,
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
        "private_group_labels_persisted": False,
        "private_predictions_persisted": False,
    }
    report["process_guards"] = _process_guards(report=report, forbidden=forbidden)
    valid = all(row["passed"] for row in report["process_guards"])
    accepted = nested_cv["candidate_family_accepted"] if valid else False
    report["decision"] = {
        "status": (
            "stage194_safety_constrained_lambdamart_candidate_family_found"
            if valid and accepted
            else "stage194_safety_constrained_lambdamart_insufficient"
            if valid
            else "stage194_safety_constrained_lambdamart_invalid"
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


def write_stage194_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage194Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv = report["safety_constrained_lambdamart_nested_cv"]
    aggregate = cv["aggregate"]
    diagnostics = cv["aggregate_diagnostics"]
    resources = report["resource_consumption"]
    charts = {
        "stage194_inner_eligible_configs.svg": _chart(
            "Stage 194 inner-eligible configurations by outer fold",
            tuple(
                BarDatum(fold, row["eligible_config_count"], str(row["eligible_config_count"]))
                for fold, row in cv["outer_folds"].items()
            ),
            "eligible configuration count",
        ),
        "stage194_outer_pool_recall.svg": _outer_diagnostic_chart(
            cv, "Stage 194 outer pool recall", "strict_opportunity_pool_recall"
        ),
        "stage194_outer_conditional_capture.svg": _outer_diagnostic_chart(
            cv,
            "Stage 194 outer conditional strict capture",
            "conditional_ranker_strict_capture",
        ),
        "stage194_outer_unsafe_rate.svg": _outer_diagnostic_chart(
            cv, "Stage 194 outer unsafe selection rate", "unsafe_selection_rate"
        ),
        "stage194_aggregate_diagnostics.svg": _chart(
            "Stage 194 aggregate safety and ranking diagnostics",
            tuple(
                BarDatum(name, value, f"{value:.3f}")
                for name, value in (
                    ("pool recall", diagnostics["strict_opportunity_pool_recall"]),
                    ("conditional capture", diagnostics["conditional_ranker_strict_capture"]),
                    ("actual capture", diagnostics["actual_strict_opportunity_capture"]),
                    ("unsafe selection", diagnostics["unsafe_selection_rate"]),
                )
            ),
            "rate",
        ),
        "stage194_aggregate_outcomes.svg": _chart(
            "Stage 194 aggregate selected-action outcomes",
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
        "stage194_selected_profiles.svg": _chart(
            "Stage 194 selected tree profiles",
            tuple(
                BarDatum(
                    name,
                    cv["selected_profile_counts"].get(name, 0),
                    str(cv["selected_profile_counts"].get(name, 0)),
                )
                for name in ("conservative", "moderate")
            ),
            "outer-fold selection count",
        ),
        "stage194_selected_penalties.svg": _chart(
            "Stage 194 selected risk penalties",
            tuple(
                BarDatum(
                    name,
                    cv["selected_penalty_counts"].get(name, 0),
                    str(cv["selected_penalty_counts"].get(name, 0)),
                )
                for name in ("0.25", "0.50", "1.00", "2.00")
            ),
            "outer-fold selection count",
        ),
        "stage194_execution_counts.svg": _chart(
            "Stage 194 model execution",
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
        "stage194_memory_gib.svg": _chart(
            "Stage 194 memory usage",
            (
                _gib_bar("peak working set", resources["process_peak_working_set_bytes"]),
                _gib_bar("peak private usage", resources["process_peak_private_usage_bytes"]),
                _gib_bar("minimum system free", resources["minimum_system_available_memory_bytes"]),
            ),
            "GiB",
        ),
        "stage194_advancement_gates.svg": _chart(
            "Stage 194 advancement gates",
            tuple(
                BarDatum(row["name"], float(row["passed"]), "pass" if row["passed"] else "fail")
                for row in cv["advancement_gates"]
            ),
            "1 means passed",
            margin_left=940,
        ),
        "stage194_process_guards.svg": _chart(
            "Stage 194 process guards",
            tuple(
                BarDatum(row["name"], float(row["passed"]), "pass" if row["passed"] else "fail")
                for row in report["process_guards"]
            ),
            "1 means passed",
            margin_left=940,
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage194Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _authorize_stage193_protocol(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 193":
        raise ValueError("Stage194 requires the Stage193 protocol")
    decision = report.get("decision", {})
    if decision.get("status") != "stage193_safety_constrained_lambdamart_protocol_frozen":
        raise ValueError("Stage194 requires the frozen Stage193 protocol")
    if decision.get("protocol_valid") is not True:
        raise ValueError("Stage194 requires a valid Stage193 protocol")
    if decision.get("stage194_train_only_experiment_authorized") is not True:
        raise ValueError("Stage193 did not authorize Stage194")
    if len(report.get("guard_checks", [])) != 59 or not all(
        row.get("passed") is True for row in report.get("guard_checks", [])
    ):
        raise ValueError("Stage193 guard checks drifted")
    protocol = report.get("frozen_protocol", {})
    if protocol.get("candidate_grid", {}).get("policy_config_count") != 64:
        raise ValueError("Stage193 policy grid drifted")
    if protocol.get("cross_validation", {}).get("maximum_model_fit_count") != 400:
        raise ValueError("Stage193 fit budget drifted")
    if protocol.get("first_stage_pool", {}).get("pool_cap") != 16:
        raise ValueError("Stage193 pool cap drifted")
    if protocol.get("action_contract", {}).get("fallback_enabled") is not False:
        raise ValueError("Stage193 fallback boundary drifted")


def _dependency_preflight(
    *, lightgbm_wheel_path: Path, narwhals_wheel_path: Path
) -> dict[str, Any]:
    lightgbm_wheel = stage173._resolved_fingerprint(lightgbm_wheel_path)
    narwhals_wheel = stage173._resolved_fingerprint(narwhals_wheel_path)
    if lightgbm_wheel["sha256"] != LIGHTGBM_WHEEL_SHA256:
        raise ValueError("Stage194 LightGBM wheel hash mismatch")
    if narwhals_wheel["sha256"] != NARWHALS_WHEEL_SHA256:
        raise ValueError("Stage194 Narwhals wheel hash mismatch")
    lightgbm_version = importlib.metadata.version("lightgbm")
    narwhals_version = importlib.metadata.version("narwhals")
    if lightgbm_version != "4.7.0" or narwhals_version != "2.24.0":
        raise ValueError("Stage194 dependency version drifted")
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        check=False,
        text=True,
    )
    if pip_check.returncode != 0:
        raise RuntimeError(f"Stage194 pip check failed: {pip_check.stdout}{pip_check.stderr}")
    import lightgbm
    import narwhals

    return {
        "lightgbm": {
            "installed_version": lightgbm.__version__,
            "wheel": lightgbm_wheel,
            "import_passed": True,
        },
        "narwhals": {
            "installed_version": narwhals.__version__,
            "wheel": narwhals_wheel,
            "import_passed": True,
            "user_selected_exact_version": True,
        },
        "pip_check": {
            "passed": True,
            "output": pip_check.stdout.strip(),
        },
    }


def _process_guards(*, report: Mapping[str, Any], forbidden: Sequence[str]) -> list[dict[str, Any]]:
    boundaries = report["execution_boundaries"]
    cv = report["safety_constrained_lambdamart_nested_cv"]
    reproduction = report["stage182_reproduction"]
    eligible_outer = sum(row["outer_evaluated"] for row in cv["outer_folds"].values())
    expected_fits = 320 + 16 * eligible_outer
    expected_representation_fits = expected_fits // 8
    return [
        _gate("stage193_protocol_hash_matches", True),
        _gate("stage193_protocol_authorized", True),
        _gate("dependency_preflight_passed", True),
        _gate(
            "resource_preflight_authorized",
            report["resource_preflight"]["frozen_threshold_met"] is True
            or report["resource_preflight"]["explicit_user_override_authorized"] is True,
        ),
        _gate(
            "resource_override_did_not_reduce_grid_or_add_fallback",
            report["resource_preflight"]["model_grid_reduced"] is False
            and report["resource_preflight"]["fallback_enabled"] is False,
        ),
        _gate("stage182_reproduction_passed", reproduction["passed"] is True),
        _gate(
            "captured_action_row_count_is_12298", boundaries["captured_action_row_count"] == 12298
        ),
        _gate(
            "captured_stage182_selected_action_count_is_129",
            boundaries["captured_stage182_selected_action_count"] == 129,
        ),
        _gate("question_count_is_370", cv["dataset"]["question_count"] == 370),
        _gate("policy_config_count_is_64", cv["protocol"]["policy_config_count"] == 64),
        _gate("advancement_gate_count_is_17", len(cv["advancement_gates"]) == 17),
        _gate(
            "model_fit_count_matches_completed_partitions",
            cv["execution"]["model_fit_count"] == expected_fits,
        ),
        _gate("model_fit_count_within_400", cv["execution"]["model_fit_count"] <= 400),
        _gate(
            "pool_safety_fit_count_exact",
            cv["execution"]["pool_safety_fit_count"] == expected_fits // 2,
        ),
        _gate(
            "lambdamart_fit_count_exact",
            cv["execution"]["lambdamart_fit_count"] == expected_fits // 4,
        ),
        _gate(
            "unsafe_head_fit_count_exact",
            cv["execution"]["unsafe_head_fit_count"] == expected_fits // 4,
        ),
        _gate(
            "all_group_contracts_validated",
            cv["execution"]["group_contract_validation_count"] == expected_representation_fits,
        ),
        _gate("lightgbm_trees_exist", cv["execution"]["tree_count"] > 0),
        _gate(
            "one_representation_at_a_time",
            boundaries["one_representation_materialized_at_a_time"] is True,
        ),
        _gate(
            "private_rows_not_public",
            boundaries["stage194_public_training_rows_written"] == 0
            and boundaries["stage194_public_prediction_rows_written"] == 0,
        ),
        _gate("private_predictions_exist", boundaries["stage194_private_prediction_count"] > 0),
        _gate("train_loaded", boundaries["train_loaded"] is True),
        _gate("development_closed", boundaries["development_loaded"] is False),
        _gate("test_closed", boundaries["test_loaded"] is False),
        _gate(
            "gold_offline_only",
            boundaries["gold_used_only_for_training_targets_and_offline_evaluation"] is True,
        ),
        _gate("full_train_policy_not_selected", boundaries["full_train_policy_selected"] is False),
        _gate(
            "replacement_policy_not_selected", boundaries["replacement_policy_selected"] is False
        ),
        _gate("runtime_e2e_not_run", boundaries["runtime_e2e_run"] is False),
        _gate("default_runtime_unchanged", boundaries["runtime_registered_as_default"] is False),
        _gate("stage178b_not_run", boundaries["stage178b_run"] is False),
        _gate("no_retry", boundaries["retry_action_count"] == 0),
        _gate("no_fallback", boundaries["fallback_action_count"] == 0),
        _gate("public_report_safe", not forbidden),
    ]


def _outer_diagnostic_chart(cv: Mapping[str, Any], title: str, metric: str) -> str:
    return _chart(
        title,
        tuple(
            BarDatum(
                fold,
                (row["outer_diagnostics"] or {}).get(metric, 0.0),
                (
                    f"{row['outer_diagnostics'][metric]:.3f}"
                    if row["outer_diagnostics"] is not None
                    else "not evaluated"
                ),
            )
            for fold, row in cv["outer_folds"].items()
        ),
        "rate",
    )


def _gib_bar(name: str, byte_count: int) -> BarDatum:
    value = byte_count / 1024**3
    return BarDatum(name, value, f"{value:.3f}")


def _chart(
    title: str,
    data: Sequence[BarDatum],
    x_label: str,
    *,
    margin_left: int = 420,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=data,
        x_label=x_label,
        width=1600,
        margin_left=margin_left,
        margin_right=180,
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
