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
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application.composition_action_audit import ActionAuditRow
from ts_rag_agent.application.composition_dual_target_policy import (
    DualTargetPrediction,
    SelectedAction,
)
from ts_rag_agent.application.composition_joint_constraint_ranking import (
    run_joint_constraint_nested_cv,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

STAGE = "Stage 186"
CREATED_AT = "2026-07-23"
ANALYSIS_ID = "primeqa_hybrid_joint_constraint_ranking_nested_cv_v1"
STAGE185_SHA256 = "742ea385e76faa950677941d760f321c834cb23cbfb054458a66ed17807b837e"
FORBIDDEN_PUBLIC_KEYS = stage183._FORBIDDEN_PUBLIC_KEYS | {
    "candidate_actions",
    "citation_loss_probability",
    "f1_loss_probability",
    "feature_rows",
    "predictions",
    "selected_actions",
    "strict_gain_probability",
}


@dataclass(frozen=True)
class Stage186Visualization:
    """One aggregate Stage 186 visualization."""

    name: str
    path: str


def run_stage186_joint_constraint_ranking_cv(
    *,
    stage185_protocol_path: Path,
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
    """Reproduce Stage 182 and run the authorized Stage 186 train nested CV."""

    started_at = time.perf_counter()
    stage185_fingerprint = stage173._resolved_fingerprint(stage185_protocol_path)
    if stage185_fingerprint["sha256"] != STAGE185_SHA256:
        raise ValueError("Stage186 Stage185 protocol hash mismatch")
    formal_stage185 = _load_json(stage185_protocol_path)
    _authorize_stage185_protocol(formal_stage185)
    formal_stage182 = _load_json(stage182_report_path)
    stage183._authorize_stage182_report(formal_stage182)
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="stage185_protocol_authorized")

    import torch

    tracker = stage169.Stage169ResourceTracker(torch_module=torch)
    tracker.capture("analysis_start")
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
        raise ValueError("Stage186 did not reproduce the formal Stage182 result")
    gc.collect()
    tracker.capture("stage182_temporary_resources_released")

    nested_cv = run_joint_constraint_nested_cv(
        action_rows=private["action_rows"],
        stage182_selected_actions=private["selected_actions"],
        progress_sink=progress_sink,
    )
    tracker.capture("joint_constraint_nested_cv_complete")
    analyzed_at = time.perf_counter()
    report: dict[str, Any] = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "analysis_id": ANALYSIS_ID,
        "analysis_scope": (
            "Train-only five-by-four nested cross-validation of 72 citation/F1 "
            "joint-constraint action-ranking configurations. No development/test, "
            "runtime E2E, replacement-policy selection, fallback, or default activation."
        ),
        "source_authorization": {
            "stage185_protocol": stage185_fingerprint,
            "stage182_rerun_sources": reproduced_stage182["source_authorization"],
        },
        "frozen_protocol": formal_stage185["frozen_protocol"],
        "stage182_reproduction": reproduction,
        "joint_constraint_nested_cv": nested_cv,
        "runtime": reproduced_stage182["runtime"],
        "resource_consumption": stage181._resource_summary(tracker.snapshots),
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "stage182_reproduction": round(reproduced_at - authorized_at, 6),
            "joint_constraint_nested_cv": round(analyzed_at - reproduced_at, 6),
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
            "stage186_model_head_fit_count": nested_cv["execution"]["model_head_fit_count"],
            "stage186_private_prediction_count": nested_cv["execution"]["private_prediction_count"],
            "stage186_public_prediction_rows_written": nested_cv["execution"][
                "public_prediction_rows_written"
            ],
            "gold_used_only_for_training_targets_and_offline_evaluation": True,
            "candidate_family_accepted": nested_cv["candidate_family_accepted"],
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
        "private_action_rows_persisted": False,
        "private_predictions_persisted": False,
    }
    report["process_guards"] = _process_guards(report=report, forbidden=forbidden)
    valid = all(row["passed"] for row in report["process_guards"])
    accepted = nested_cv["candidate_family_accepted"] if valid else False
    report["decision"] = {
        "status": (
            "stage186_joint_constraint_ranking_candidate_family_found"
            if valid and accepted
            else "stage186_joint_constraint_ranking_insufficient"
            if valid
            else "stage186_joint_constraint_ranking_invalid"
        ),
        "experiment_valid": valid,
        "candidate_family_accepted": accepted,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "runtime_e2e_authorized": False,
        "development_opened": False,
        "test_opened": False,
        "default_runtime_activation": False,
    }
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report


def write_stage186_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage186Visualization, ...]:
    """Write and XML-validate aggregate Stage 186 charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cv = report["joint_constraint_nested_cv"]
    aggregate = cv["aggregate"]
    head_metrics = cv["head_metrics"]
    resources = report["resource_consumption"]
    charts = {
        "stage186_inner_eligible_configs.svg": _chart(
            "Stage 186 inner-eligible configurations by outer fold",
            tuple(
                BarDatum(
                    fold_id,
                    row["eligible_config_count"],
                    str(row["eligible_config_count"]),
                )
                for fold_id, row in cv["outer_folds"].items()
            ),
            "eligible configuration count",
        ),
        "stage186_outer_citation_delta.svg": _chart(
            "Stage 186 outer-fold gold-citation delta",
            tuple(
                BarDatum(
                    fold_id,
                    (row["outer_evaluation"] or {}).get("gold_citation_delta", 0),
                    str((row["outer_evaluation"] or {}).get("gold_citation_delta", "not run")),
                )
                for fold_id, row in cv["outer_folds"].items()
            ),
            "gold-citation delta",
        ),
        "stage186_outer_f1_delta.svg": _chart(
            "Stage 186 outer-fold mean F1 delta",
            tuple(
                BarDatum(
                    fold_id,
                    (row["outer_evaluation"] or {}).get("mean_f1_delta", 0.0),
                    (
                        f"{row['outer_evaluation']['mean_f1_delta']:.4f}"
                        if row["outer_evaluation"]
                        else "not run"
                    ),
                )
                for fold_id, row in cv["outer_folds"].items()
            ),
            "mean answerable F1 delta",
        ),
        "stage186_regression_flow.svg": _chart(
            "Stage 186 Stage 182 regression flow",
            (
                BarDatum(
                    "reference regressions",
                    aggregate["reference_regression_count"],
                    str(aggregate["reference_regression_count"]),
                ),
                BarDatum(
                    "repaired regressions",
                    aggregate["repaired_reference_regression_count"],
                    str(aggregate["repaired_reference_regression_count"]),
                ),
                BarDatum(
                    "new regressions",
                    aggregate["new_f1_regression_count"],
                    str(aggregate["new_f1_regression_count"]),
                ),
            ),
            "question count",
        ),
        "stage186_selected_head_auc.svg": _chart(
            "Stage 186 selected-bundle outer head AUC",
            tuple(
                BarDatum(
                    target,
                    (head_metrics[target] or {}).get("roc_auc") or 0.0,
                    (
                        f"{head_metrics[target]['roc_auc']:.3f}"
                        if head_metrics[target] and head_metrics[target]["roc_auc"] is not None
                        else "not available"
                    ),
                )
                for target in ("citation_loss", "f1_loss", "strict_gain")
            ),
            "outer held-out ROC AUC",
        ),
        "stage186_advancement_gates.svg": _chart(
            "Stage 186 advancement gates",
            tuple(
                BarDatum(
                    row["name"],
                    float(row["passed"]),
                    "pass" if row["passed"] else "fail",
                )
                for row in cv["advancement_gates"]
            ),
            "1 means passed",
            margin_left=900,
        ),
        "stage186_execution_counts.svg": _chart(
            "Stage 186 execution counts",
            (
                BarDatum(
                    "model head fits",
                    cv["execution"]["model_head_fit_count"],
                    str(cv["execution"]["model_head_fit_count"]),
                ),
                BarDatum(
                    "policy configurations",
                    cv["protocol"]["policy_config_count"],
                    str(cv["protocol"]["policy_config_count"]),
                ),
                BarDatum(
                    "inner partitions",
                    cv["protocol"]["outer_fold_count"] * cv["protocol"]["inner_fold_count"],
                    str(cv["protocol"]["outer_fold_count"] * cv["protocol"]["inner_fold_count"]),
                ),
                BarDatum(
                    "outer refits",
                    sum(row["outer_evaluated"] for row in cv["outer_folds"].values()),
                    str(sum(row["outer_evaluated"] for row in cv["outer_folds"].values())),
                ),
            ),
            "count",
        ),
        "stage186_memory_gib.svg": _chart(
            "Stage 186 memory usage",
            (
                BarDatum(
                    "peak working set",
                    resources["process_peak_working_set_bytes"] / (1024**3),
                    f"{resources['process_peak_working_set_bytes'] / (1024**3):.3f} GiB",
                ),
                BarDatum(
                    "peak private usage",
                    resources["process_peak_private_usage_bytes"] / (1024**3),
                    f"{resources['process_peak_private_usage_bytes'] / (1024**3):.3f} GiB",
                ),
                BarDatum(
                    "minimum system free",
                    resources["minimum_system_available_memory_bytes"] / (1024**3),
                    (f"{resources['minimum_system_available_memory_bytes'] / (1024**3):.3f} GiB"),
                ),
            ),
            "GiB",
        ),
        "stage186_process_guards.svg": _chart(
            "Stage 186 process guards",
            tuple(
                BarDatum(
                    row["name"],
                    float(row["passed"]),
                    "pass" if row["passed"] else "fail",
                )
                for row in report["process_guards"]
            ),
            "1 means passed",
            margin_left=920,
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage186Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _authorize_stage185_protocol(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 185":
        raise ValueError("Stage186 requires the Stage185 protocol")
    decision = report.get("decision", {})
    if decision.get("status") != "stage185_joint_constraint_ranking_protocol_frozen":
        raise ValueError("Stage186 requires the frozen Stage185 protocol")
    if decision.get("protocol_valid") is not True:
        raise ValueError("Stage186 requires a valid Stage185 protocol")
    if decision.get("stage186_train_only_experiment_authorized") is not True:
        raise ValueError("Stage185 did not authorize Stage186")
    if not all(row.get("passed") is True for row in report.get("guard_checks", [])):
        raise ValueError("Stage185 guard checks must all pass")
    protocol = report.get("frozen_protocol", {})
    if protocol.get("candidate_grid", {}).get("policy_config_count") != 72:
        raise ValueError("Stage185 policy grid drifted")
    if protocol.get("cross_validation", {}).get("maximum_model_head_fit_count") != 300:
        raise ValueError("Stage185 fit budget drifted")
    if protocol.get("reference_action_contract", {}).get("fallback_enabled") is not False:
        raise ValueError("Stage185 fallback boundary drifted")


def _process_guards(
    *,
    report: Mapping[str, Any],
    forbidden: Sequence[str],
) -> list[dict[str, Any]]:
    boundaries = report["execution_boundaries"]
    reproduction = report["stage182_reproduction"]
    cv = report["joint_constraint_nested_cv"]
    eligible_outer_folds = sum(row["outer_evaluated"] for row in cv["outer_folds"].values())
    expected_fits = 240 + 12 * eligible_outer_folds
    return [
        _gate("stage185_protocol_hash_matches", True),
        _gate("stage185_protocol_authorized", True),
        _gate("stage182_reproduction_passed", reproduction["passed"] is True),
        _gate(
            "stage182_reproduction_check_count_is_10",
            len(reproduction["checks"]) == 10,
        ),
        _gate(
            "captured_action_row_count_is_12298",
            boundaries["captured_action_row_count"] == 12298,
        ),
        _gate(
            "captured_stage182_selected_action_count_is_129",
            boundaries["captured_stage182_selected_action_count"] == 129,
        ),
        _gate("policy_config_count_is_72", cv["protocol"]["policy_config_count"] == 72),
        _gate("question_count_is_370", cv["dataset"]["question_count"] == 370),
        _gate("reference_action_count_is_370", cv["dataset"]["reference_action_count"] == 370),
        _gate(
            "reference_regression_count_is_55",
            cv["dataset"]["reference_regression_count"] == 55,
        ),
        _gate(
            "model_head_fit_count_matches_completed_partitions",
            cv["execution"]["model_head_fit_count"] == expected_fits,
        ),
        _gate(
            "model_head_fit_count_within_300",
            cv["execution"]["model_head_fit_count"] <= 300,
        ),
        _gate(
            "private_predictions_not_public",
            boundaries["stage186_private_prediction_count"] > 0
            and boundaries["stage186_public_prediction_rows_written"] == 0,
        ),
        _gate("train_loaded", boundaries["train_loaded"] is True),
        _gate("development_closed", boundaries["development_loaded"] is False),
        _gate("test_closed", boundaries["test_loaded"] is False),
        _gate(
            "gold_offline_only",
            boundaries["gold_used_only_for_training_targets_and_offline_evaluation"] is True,
        ),
        _gate(
            "replacement_policy_not_selected",
            boundaries["replacement_policy_selected"] is False,
        ),
        _gate("runtime_e2e_not_run", boundaries["runtime_e2e_run"] is False),
        _gate("default_runtime_unchanged", boundaries["runtime_registered_as_default"] is False),
        _gate("stage178b_not_run", boundaries["stage178b_run"] is False),
        _gate("no_retry", boundaries["retry_action_count"] == 0),
        _gate("no_fallback", boundaries["fallback_action_count"] == 0),
        _gate("public_report_safe", not forbidden),
    ]


def _chart(
    title: str,
    bars: Sequence[BarDatum],
    x_label: str,
    *,
    margin_left: int = 720,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        width=1680,
        margin_left=margin_left,
        margin_right=260,
    )


def _forbidden_keys_found(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        found = {str(key) for key in value if str(key) in FORBIDDEN_PUBLIC_KEYS}
        for nested in value.values():
            found.update(_forbidden_keys_found(nested))
        return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        found: set[str] = set()
        for nested in value:
            found.update(_forbidden_keys_found(nested))
        return found
    return set()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(progress_sink: stage182.ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
