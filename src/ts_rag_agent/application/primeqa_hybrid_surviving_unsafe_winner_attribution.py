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
from ts_rag_agent.application.composition_surviving_unsafe_winner_attribution import (
    run_surviving_unsafe_winner_attribution,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

STAGE = "Stage 197"
CREATED_AT = "2026-07-27"
ANALYSIS_ID = "primeqa_hybrid_surviving_unsafe_winner_attribution_v1"
STAGE196_SHA256 = "e5a44fbc76acaa053ca809174b1f3f767afe31a4d55e0e05d0b6708aee41fa01"
MINIMUM_AVAILABLE_MEMORY_BYTES = 4 * 1024**3
FORBIDDEN_PUBLIC_KEYS = stage194.FORBIDDEN_PUBLIC_KEYS | {
    "complete_pool",
    "frontier",
    "question_key",
    "risk_predictions",
    "unsafe_score",
}


@dataclass(frozen=True)
class Stage197Visualization:
    name: str
    path: str


def run_stage197_surviving_unsafe_winner_attribution(
    *,
    stage196_report_path: Path,
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
    """Reproduce Stage182 and attribute Stage196 top-inner unsafe winners."""

    started_at = time.perf_counter()
    stage196_fingerprint = stage173._resolved_fingerprint(stage196_report_path)
    if stage196_fingerprint["sha256"] != STAGE196_SHA256:
        raise ValueError("Stage197 Stage196 report hash mismatch")
    formal_stage196 = _load_json(stage196_report_path)
    _authorize_stage196_report(formal_stage196)
    dependency_preflight = stage194._dependency_preflight(
        lightgbm_wheel_path=lightgbm_wheel_path,
        narwhals_wheel_path=narwhals_wheel_path,
    )
    formal_stage182 = _load_json(stage182_report_path)
    stage183._authorize_stage182_report(formal_stage182)
    authorized_at = time.perf_counter()

    import torch

    tracker = stage169.Stage169ResourceTracker(torch_module=torch)
    preflight_snapshot = tracker.capture("stage197_preflight")
    if preflight_snapshot.system_available_memory_bytes < MINIMUM_AVAILABLE_MEMORY_BYTES:
        raise RuntimeError("Stage197 requires at least 4 GiB available system memory")
    _emit(progress_sink, phase="stage196_report_dependencies_and_memory_authorized")
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
        raise ValueError("Stage197 did not reproduce the formal Stage182 result")
    gc.collect()
    tracker.capture("stage182_temporary_resources_released")

    def attribution_progress(event: Mapping[str, Any]) -> None:
        if event.get("phase") in {
            "stage197_inner_partition_complete",
            "stage197_outer_context_complete",
        }:
            tracker.capture(str(event["phase"]))
        _emit(progress_sink, **dict(event))

    attribution = run_surviving_unsafe_winner_attribution(
        action_rows=private["action_rows"],
        stage182_selected_actions=private["selected_actions"],
        stage196_report=formal_stage196,
        progress_sink=attribution_progress,
    )
    tracker.capture("surviving_unsafe_winner_attribution_complete")
    analyzed_at = time.perf_counter()
    report: dict[str, Any] = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "analysis_id": ANALYSIS_ID,
        "analysis_scope": (
            "Train-only reconstruction and aggregate attribution of unsafe winners from "
            "the five published Stage196 top-inner configurations. Development and test "
            "remain closed; no search, fallback, runtime selection, or default activation occurs."
        ),
        "source_authorization": {
            "stage196_report": stage196_fingerprint,
            "stage182_rerun_sources": reproduced_stage182["source_authorization"],
        },
        "dependency_preflight": dependency_preflight,
        "resource_preflight": {
            "frozen_minimum_available_memory_bytes": MINIMUM_AVAILABLE_MEMORY_BYTES,
            "actual_available_memory_bytes": preflight_snapshot.system_available_memory_bytes,
            "frozen_threshold_met": True,
            "memory_override_authorized": False,
            "fallback_enabled": False,
        },
        "frozen_diagnostic_protocol": _diagnostic_protocol(),
        "stage182_reproduction": reproduction,
        "surviving_unsafe_winner_attribution": attribution,
        "runtime": reproduced_stage182["runtime"],
        "resource_consumption": stage181._resource_summary(tracker.snapshots),
        "timing_seconds": {
            "source_dependency_and_memory_authorization": round(authorized_at - started_at, 6),
            "stage182_reproduction": round(reproduced_at - authorized_at, 6),
            "unsafe_winner_attribution": round(analyzed_at - reproduced_at, 6),
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
            "stage197_model_fit_count": attribution["execution"]["model_fit_count"],
            "stage197_lightgbm_tree_count": attribution["execution"]["tree_count"],
            "stage197_private_prediction_count": attribution["execution"][
                "private_prediction_count"
            ],
            "gold_used_only_for_training_targets_and_offline_attribution": True,
            "oracle_used_as_runtime_rule": False,
            "new_policy_search_run": False,
            "full_train_policy_selected": False,
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
    report["decision"] = {
        "status": (
            "stage197_surviving_unsafe_winner_attribution_complete"
            if valid
            else "stage197_surviving_unsafe_winner_attribution_invalid"
        ),
        "experiment_valid": valid,
        "diagnostic_complete": valid,
        "dominant_mechanism": (
            attribution["diagnostic_finding"]["dominant_mechanism"] if valid else None
        ),
        "recommended_next_focus": (
            attribution["diagnostic_finding"]["recommended_next_focus"] if valid else None
        ),
        "development_opened": False,
        "test_opened": False,
        "runtime_e2e_authorized": False,
        "default_runtime_activation": False,
    }
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report


def write_stage197_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage197Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    attribution = report["surviving_unsafe_winner_attribution"]
    aggregate = attribution["aggregate"]
    resources = report["resource_consumption"]
    rows = {
        "stage197_unsafe_winners_by_outer.svg": _chart(
            "Stage 197 unsafe winners by outer context",
            [
                _count_bar(name, value["unsafe_winner_attribution"]["unsafe_winner_context_count"])
                for name, value in attribution["outer_contexts"].items()
            ],
        ),
        "stage197_mechanism_partition.svg": _chart(
            "Stage 197 exact unsafe-winner mechanism partition",
            [_count_bar(name, value) for name, value in aggregate["mechanism_counts"].items()],
        ),
        "stage197_risk_rank_buckets.svg": _chart(
            "Stage 197 unsafe-winner risk ranks within complete pool",
            [
                _count_bar(name, value)
                for name, value in aggregate["risk_rank_bucket_counts"].items()
            ],
        ),
        "stage197_gain_rank_buckets.svg": _chart(
            "Stage 197 unsafe-winner gain ranks within frontier",
            [
                _count_bar(name, value)
                for name, value in aggregate["gain_rank_bucket_counts"].items()
            ],
        ),
        "stage197_loss_types.svg": _chart(
            "Stage 197 unsafe-winner loss types",
            [_count_bar(name, value) for name, value in aggregate["loss_type_counts"].items()],
        ),
        "stage197_oracle_repairability.svg": _chart(
            "Stage 197 offline oracle repairability",
            [
                _count_bar(
                    "unsafe with strict opportunity",
                    aggregate["unsafe_with_strict_opportunity_count"],
                ),
                _count_bar(
                    "strict frontier alternative", aggregate["oracle_strict_repairable_count"]
                ),
                _count_bar(
                    "lower-risk strict alternative",
                    aggregate["lower_risk_strict_alternative_count"],
                ),
            ],
        ),
        "stage197_unsafe_head_metrics.svg": _chart(
            "Stage 197 unsafe-head discrimination",
            [
                _rate_bar("ROC AUC", attribution["unsafe_head_prediction_metrics"]["roc_auc"]),
                _rate_bar(
                    "average precision",
                    attribution["unsafe_head_prediction_metrics"]["average_precision"],
                ),
            ],
        ),
        "stage197_execution.svg": _chart(
            "Stage 197 focused reconstruction execution",
            [
                _count_bar("partitions", attribution["execution"]["partition_count"]),
                _count_bar("model fits", attribution["execution"]["model_fit_count"]),
                _count_bar("LightGBM trees", attribution["execution"]["tree_count"]),
            ],
        ),
        "stage197_resources.svg": _chart(
            "Stage 197 resource consumption (GiB)",
            [
                _gib_bar("peak working set", resources["process_peak_working_set_bytes"]),
                _gib_bar("peak private usage", resources["process_peak_private_usage_bytes"]),
                _gib_bar(
                    "minimum system available", resources["minimum_system_available_memory_bytes"]
                ),
            ],
        ),
        "stage197_process_guards.svg": _chart(
            "Stage 197 process guards",
            [
                BarDatum(row["name"], float(row["passed"]), str(int(row["passed"])))
                for row in report["process_guards"]
            ],
        ),
    }
    artifacts = []
    for name, svg in rows.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        artifacts.append(Stage197Visualization(name, str(path)))
    return tuple(artifacts)


def _authorize_stage196_report(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 196":
        raise ValueError("Stage197 requires Stage196")
    decision = report.get("decision", {})
    if decision.get("experiment_valid") is not True:
        raise ValueError("Stage197 requires a valid Stage196 result")
    if decision.get("candidate_family_accepted") is not False:
        raise ValueError("Stage197 expects the insufficient Stage196 result")
    if decision.get("development_opened") is not False or decision.get("test_opened") is not False:
        raise ValueError("Stage197 requires closed development and test sets")
    if not all(row.get("passed") is True for row in report.get("process_guards", [])):
        raise ValueError("Stage197 requires all Stage196 process guards to pass")
    outer = report["safety_first_frontier_nested_cv"]["outer_folds"]
    if len(outer) != 5 or any(not row.get("top_inner_candidates") for row in outer.values()):
        raise ValueError("Stage197 requires five published Stage196 top-inner candidates")


def _diagnostic_protocol() -> dict[str, Any]:
    return {
        "stage196_report_reproduced_by_exact_top_inner_evidence": True,
        "development_and_test_closed": True,
        "top_inner_spec_count": 5,
        "inner_partition_count": 20,
        "model_fits_per_partition": 4,
        "maximum_model_fit_count": 80,
        "maximum_lightgbm_tree_count": 12_000,
        "mechanism_partition": [
            "no_strict_opportunity",
            "safety_pool_exclusion",
            "risk_frontier_exclusion",
            "final_gain_dominance",
            "risk_ordering_failure",
        ],
        "dominant_mechanism_rule": "largest exact mutually-exclusive count; lexical tie-break",
        "oracle_is_diagnostic_only": True,
        "fallback_enabled": False,
    }


def _process_guards(report: Mapping[str, Any], forbidden: Sequence[str]) -> list[dict[str, Any]]:
    attribution = report["surviving_unsafe_winner_attribution"]
    execution = attribution["execution"]
    boundaries = report["execution_boundaries"]
    aggregate = attribution["aggregate"]
    return [
        _gate("stage182_reproduction_exact", report["stage182_reproduction"]["passed"] is True),
        _gate("train_only", boundaries["train_loaded"] is True),
        _gate("development_closed", boundaries["development_loaded"] is False),
        _gate("test_closed", boundaries["test_loaded"] is False),
        _gate("five_outer_contexts", len(attribution["outer_contexts"]) == 5),
        _gate("twenty_inner_partitions", execution["partition_count"] == 20),
        _gate(
            "all_top_inner_reconstructions_exact",
            execution["all_top_inner_reconstructions_exact"] is True,
        ),
        _gate("five_top_inner_reconstructions", execution["top_inner_reconstruction_count"] == 5),
        _gate("exact_model_fit_count", execution["model_fit_count"] == 80),
        _gate("exact_pool_safety_fit_count", execution["pool_safety_fit_count"] == 40),
        _gate("exact_lambdamart_fit_count", execution["lambdamart_fit_count"] == 20),
        _gate("exact_unsafe_head_fit_count", execution["unsafe_head_fit_count"] == 20),
        _gate("tree_budget_respected", execution["tree_count"] <= 12_000),
        _gate("exact_group_contract_count", execution["group_contract_validation_count"] == 20),
        _gate("exact_private_prediction_count", execution["private_prediction_count"] == 196_768),
        _gate("mechanism_partition_exact", aggregate["mechanism_partition_exact"] is True),
        _gate("unsafe_winners_observed", aggregate["unsafe_winner_context_count"] > 0),
        _gate(
            "unsafe_head_metrics_available",
            attribution["unsafe_head_prediction_metrics"]["action_context_count"] > 0,
        ),
        _gate("no_public_training_rows", execution["public_training_rows_written"] == 0),
        _gate("no_public_prediction_rows", execution["public_prediction_rows_written"] == 0),
        _gate("oracle_not_runtime", boundaries["oracle_used_as_runtime_rule"] is False),
        _gate("no_new_policy_search", boundaries["new_policy_search_run"] is False),
        _gate("no_full_train_selection", boundaries["full_train_policy_selected"] is False),
        _gate("no_runtime_default", boundaries["runtime_registered_as_default"] is False),
        _gate("no_runtime_e2e", boundaries["runtime_e2e_run"] is False),
        _gate("no_stage178b", boundaries["stage178b_run"] is False),
        _gate("no_retry", boundaries["retry_action_count"] == 0),
        _gate("no_fallback", boundaries["fallback_action_count"] == 0),
        _gate("no_forbidden_public_keys", not forbidden),
    ]


def _count_bar(name: str, value: int) -> BarDatum:
    return BarDatum(name, float(value), str(value))


def _rate_bar(name: str, value: float) -> BarDatum:
    return BarDatum(name, value, f"{value:.6f}")


def _gib_bar(name: str, byte_count: int) -> BarDatum:
    value = byte_count / 1024**3
    return BarDatum(name, value, f"{value:.3f}")


def _chart(title: str, data: Sequence[BarDatum]) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=data,
        x_label="aggregate value",
        width=1680,
        margin_left=520,
        margin_right=220,
    )


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


def _emit(progress_sink: stage182.ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
