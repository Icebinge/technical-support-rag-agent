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
from ts_rag_agent.application.composition_f1_representation_cv import (
    run_f1_representation_cv,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 184"
_CREATED_AT = "2026-07-23"
_ANALYSIS_ID = "primeqa_hybrid_composition_f1_representation_cv_v1"
_STAGE183_SHA256 = "4dd611c9a759fd791288886c638bd9ec36b7564328eb53f2fef1742544540f1a"
_FORBIDDEN_PUBLIC_KEYS = stage183._FORBIDDEN_PUBLIC_KEYS | {
    "risk_score",
    "safety_score",
    "predictions",
    "selected_actions",
}


@dataclass(frozen=True)
class Stage184Visualization:
    name: str
    path: str


def run_stage184_composition_f1_representation_cv(
    *,
    stage183_report_path: Path,
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
    """Reproduce Stage 182 and screen Stage 184 F1 representations on train OOF."""

    started_at = time.perf_counter()
    stage183_fingerprint = stage173._resolved_fingerprint(stage183_report_path)
    if stage183_fingerprint["sha256"] != _STAGE183_SHA256:
        raise ValueError("Stage184 Stage183 report hash mismatch")
    formal_stage183 = _load_json(stage183_report_path)
    _authorize_stage183_report(formal_stage183)
    formal_stage182 = _load_json(stage182_report_path)
    stage183._authorize_stage182_report(formal_stage182)
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="stage183_source_authorized")

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
        raise ValueError("Stage184 did not reproduce the formal Stage182 result")
    gc.collect()
    tracker.capture("stage182_temporary_resources_released")

    representation_cv = run_f1_representation_cv(
        action_rows=private["action_rows"],
        stage182_selected_actions=private["selected_actions"],
        progress_sink=progress_sink,
    )
    tracker.capture("representation_cv_complete")
    analyzed_at = time.perf_counter()
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train-only five-fold OOF screening of runtime-visible relative, ordinal, "
            "quantile, and pairwise F1-risk representations without selecting a policy."
        ),
        "source_authorization": {
            "stage183": stage183_fingerprint,
            "stage182_rerun_sources": reproduced_stage182["source_authorization"],
        },
        "frozen_protocol": _frozen_protocol(),
        "stage182_reproduction": reproduction,
        "f1_representation_cv": representation_cv,
        "runtime": reproduced_stage182["runtime"],
        "resource_consumption": stage181._resource_summary(tracker.snapshots),
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "stage182_reproduction": round(reproduced_at - authorized_at, 6),
            "f1_representation_cv": round(analyzed_at - reproduced_at, 6),
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
            "stage184_model_fit_count": representation_cv["execution"]["model_fit_count"],
            "stage184_private_prediction_count": representation_cv["execution"][
                "private_prediction_count"
            ],
            "stage184_public_prediction_rows_written": representation_cv["execution"][
                "public_prediction_rows_written"
            ],
            "gold_used_only_for_training_targets_and_offline_evaluation": True,
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
        "forbidden_keys": sorted(_FORBIDDEN_PUBLIC_KEYS),
        "forbidden_keys_found": forbidden,
    }
    report["process_guards"] = _process_guards(report=report, forbidden=forbidden)
    valid = all(row["passed"] for row in report["process_guards"])
    accepted = (
        representation_cv["selection"]["candidate_accepted_for_nested_policy_experiment"]
        if valid
        else False
    )
    report["decision"] = {
        "status": (
            "stage184_f1_representation_cv_candidate_found"
            if valid and accepted
            else "stage184_f1_representation_cv_insufficient"
            if valid
            else "stage184_f1_representation_cv_invalid"
        ),
        "experiment_valid": valid,
        "selected_representation": representation_cv["selection"]["selected_candidate"]
        if valid
        else None,
        "representation_candidate_accepted": accepted,
        "nested_policy_experiment_authorized": accepted,
        "replacement_policy_selected": False,
        "runtime_e2e_authorized": False,
        "development_opened": False,
        "test_opened": False,
        "default_runtime_activation": False,
    }
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report


def _authorize_stage183_report(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 183":
        raise ValueError("Stage184 requires the Stage183 report")
    decision = report.get("decision", {})
    if decision.get("status") != "stage183_f1_risk_failure_attribution_complete":
        raise ValueError("Stage184 requires the completed Stage183 attribution")
    if decision.get("diagnostic_complete") is not True:
        raise ValueError("Stage184 requires a valid Stage183 diagnostic")
    if decision.get("primary_bottleneck") != "f1_risk_separability_and_ranking":
        raise ValueError("Stage184 requires the Stage183 F1-risk bottleneck")
    if not all(row.get("passed") is True for row in report.get("process_guards", [])):
        raise ValueError("Stage183 process guards must all pass")
    boundaries = report.get("execution_boundaries", {})
    if (
        boundaries.get("development_loaded") is not False
        or boundaries.get("test_loaded") is not False
    ):
        raise ValueError("Stage183 development/test boundary drifted")


def _frozen_protocol() -> dict[str, Any]:
    return {
        "formal_stage183_sha256": _STAGE183_SHA256,
        "stage182_exact_reproduction_required": True,
        "representation_candidates": 8,
        "selection_scope": "representation only, no answer-composition policy",
        "minimum_risk_auc": 0.62,
        "minimum_auc_gain_vs_best_raw": 0.03,
        "minimum_fold_auc_nonregression": "4/5",
        "minimum_stage182_safe_alternative_top3_rate": 0.70,
        "minimum_stage182_safe_alternative_top5_rate": 0.85,
        "development_and_test_closed": True,
        "fallback_strategy_enabled": False,
        "runtime_e2e_enabled": False,
    }


def _process_guards(
    *,
    report: Mapping[str, Any],
    forbidden: Sequence[str],
) -> list[dict[str, Any]]:
    boundaries = report["execution_boundaries"]
    cv = report["f1_representation_cv"]
    return [
        _gate(
            "formal_stage183_authorized",
            report["source_authorization"]["stage183"]["sha256"] == _STAGE183_SHA256,
        ),
        _gate("formal_stage182_reproduced", report["stage182_reproduction"]["passed"]),
        _gate("exact_action_rows", boundaries["captured_action_row_count"] == 12_298),
        _gate(
            "exact_selected_actions", boundaries["captured_stage182_selected_action_count"] == 129
        ),
        _gate(
            "exact_selected_regressions", cv["dataset"]["stage182_selected_regression_count"] == 55
        ),
        _gate("exact_nonbaseline_actions", cv["dataset"]["action_count"] == 11_928),
        _gate(
            "exact_representation_candidates", cv["protocol"]["representation_candidate_count"] == 8
        ),
        _gate("exact_stage184_model_fits", boundaries["stage184_model_fit_count"] == 60),
        _gate(
            "no_public_prediction_rows", boundaries["stage184_public_prediction_rows_written"] == 0
        ),
        _gate("one_runtime_resource_build", report["runtime"]["resource_factory_build_count"] == 1),
        _gate(
            "exact_score_provider_calls",
            report["runtime"]["precomputed_score_provider"]["call_count"] == 562,
        ),
        _gate(
            "exact_score_provider_pairs",
            report["runtime"]["precomputed_score_provider"]["pair_count"] == 9_714,
        ),
        _gate("development_closed", boundaries["development_loaded"] is False),
        _gate("test_closed", boundaries["test_loaded"] is False),
        _gate("no_replacement_policy_selected", boundaries["replacement_policy_selected"] is False),
        _gate("default_runtime_unchanged", boundaries["runtime_registered_as_default"] is False),
        _gate("runtime_e2e_not_run", boundaries["runtime_e2e_run"] is False),
        _gate("stage178b_not_run", boundaries["stage178b_run"] is False),
        _gate("no_retry", boundaries["retry_action_count"] == 0),
        _gate("no_fallback", boundaries["fallback_action_count"] == 0),
        _gate("public_report_safe", not forbidden),
    ]


def write_stage184_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage184Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv = report["f1_representation_cv"]
    representations = cv["representations"]
    selection = cv["selection"]
    selected = selection["selected_candidate"]
    raw = selection["best_raw_reference"]
    charts = {
        "representation_risk_auc.svg": _chart(
            "Stage 184 F1-risk ROC AUC by representation",
            tuple(
                BarDatum(name, row["aggregate"]["roc_auc"], f"{row['aggregate']['roc_auc']:.3f}")
                for name, row in representations.items()
            ),
            "five-fold OOF ROC AUC",
        ),
        "representation_average_precision.svg": _chart(
            "Stage 184 F1-risk average precision",
            tuple(
                BarDatum(
                    name,
                    row["aggregate"]["average_precision"],
                    f"{row['aggregate']['average_precision']:.3f}",
                )
                for name, row in representations.items()
            ),
            "five-fold OOF average precision",
        ),
        "safe_alternative_top3.svg": _headroom_chart(representations, depth=3),
        "safe_alternative_top5.svg": _headroom_chart(representations, depth=5),
        "selected_vs_raw_fold_auc.svg": _chart(
            "Stage 184 selected versus raw fold ROC AUC",
            tuple(
                BarDatum(
                    f"{model} {fold_id}",
                    representations[model]["folds"][fold_id]["roc_auc"],
                    f"{representations[model]['folds'][fold_id]['roc_auc']:.3f}",
                )
                for fold_id in representations[raw]["folds"]
                for model in (raw, selected)
            ),
            "held-out fold ROC AUC",
        ),
        "selected_quality_gates.svg": _chart(
            "Stage 184 selected representation quality gates",
            tuple(
                BarDatum(row["name"], float(row["passed"]), "pass" if row["passed"] else "fail")
                for row in selection["quality_gates"]
            ),
            "gate result",
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage184Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _headroom_chart(representations: Mapping[str, Mapping[str, Any]], *, depth: int) -> str:
    key = f"safe_alternative_top{depth}_rate"
    return _chart(
        f"Stage 184 safe alternative in F1-safety top {depth}",
        tuple(
            BarDatum(
                name,
                row["stage182_regression_headroom"][key],
                f"{row['stage182_regression_headroom'][key]:.3f}",
            )
            for name, row in representations.items()
        ),
        "fraction of 55 Stage 182 regressions",
    )


def _chart(title: str, bars: Sequence[BarDatum], x_label: str) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        width=1560,
        margin_left=720,
    )


def _forbidden_keys_found(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        found = {str(key) for key in value if str(key) in _FORBIDDEN_PUBLIC_KEYS}
        for nested in value.values():
            found.update(_forbidden_keys_found(nested))
        return found
    if isinstance(value, (list, tuple)):
        found = set()
        for nested in value:
            found.update(_forbidden_keys_found(nested))
        return found
    return set()


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _emit(progress_sink: stage182.ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
