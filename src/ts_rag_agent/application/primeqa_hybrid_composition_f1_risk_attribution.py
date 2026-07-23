from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application import primeqa_hybrid_composition_dual_target_cv as stage182
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application.composition_action_audit import ActionAuditRow
from ts_rag_agent.application.composition_dual_target_policy import (
    DualTargetPrediction,
    SelectedAction,
)
from ts_rag_agent.application.composition_f1_risk_attribution import (
    run_f1_risk_attribution,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 183"
_CREATED_AT = "2026-07-23"
_ANALYSIS_ID = "primeqa_hybrid_composition_f1_risk_attribution_v1"
_STAGE182_SHA256 = "c3dfbe7484a604b8491bed0531fc82b20bd092016fd7ddf303955b7c7c89044a"
_FORBIDDEN_PUBLIC_KEYS = stage182._FORBIDDEN_PUBLIC_KEYS | {
    "question_key",
    "action_id",
    "selected_indices",
}


@dataclass(frozen=True)
class Stage183Visualization:
    name: str
    path: str


def run_stage183_composition_f1_risk_attribution(
    *,
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
    """Reproduce Stage 182 and attribute its selected-action F1 regressions."""

    started_at = time.perf_counter()
    stage182_fingerprint = stage173._resolved_fingerprint(stage182_report_path)
    if stage182_fingerprint["sha256"] != _STAGE182_SHA256:
        raise ValueError("Stage183 Stage182 report hash mismatch")
    formal_stage182 = _load_json(stage182_report_path)
    _authorize_stage182_report(formal_stage182)
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="stage182_source_authorized")

    private: dict[str, Any] = {}

    def capture(
        action_rows: Sequence[ActionAuditRow],
        selected_actions: Sequence[SelectedAction],
        outer_predictions: Sequence[DualTargetPrediction],
    ) -> None:
        private["action_rows"] = tuple(action_rows)
        private["selected_actions"] = tuple(selected_actions)
        private["outer_predictions"] = tuple(outer_predictions)

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
    reproduced_at = time.perf_counter()
    reproduction = _stage182_reproduction(
        formal=formal_stage182,
        reproduced=reproduced_stage182,
        selected_actions=private["selected_actions"],
    )
    if not reproduction["passed"]:
        raise ValueError("Stage183 did not reproduce the formal Stage182 result")

    attribution = run_f1_risk_attribution(
        action_rows=private["action_rows"],
        selected_actions=private["selected_actions"],
        outer_predictions=private["outer_predictions"],
        outer_fold_reports=reproduced_stage182["dual_target_nested_cv"]["outer_folds"],
    )
    analyzed_at = time.perf_counter()
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train-only attribution of Stage182 selected-action F1 regressions and "
            "inner-policy instability without fitting a replacement model."
        ),
        "source_authorization": {
            "stage182": stage182_fingerprint,
            "stage182_rerun_sources": reproduced_stage182["source_authorization"],
        },
        "frozen_protocol": _frozen_protocol(),
        "stage182_reproduction": reproduction,
        "f1_risk_attribution": attribution,
        "runtime": reproduced_stage182["runtime"],
        "resource_consumption": reproduced_stage182["resource_consumption"],
        "timing_seconds": {
            "stage182_source_authorization": round(authorized_at - started_at, 6),
            "stage182_reproduction": round(reproduced_at - authorized_at, 6),
            "f1_risk_attribution": round(analyzed_at - reproduced_at, 6),
            "wall": round(analyzed_at - started_at, 6),
        },
        "execution_boundaries": {
            "train_loaded": True,
            "development_loaded": False,
            "test_loaded": False,
            "captured_action_row_count": len(private["action_rows"]),
            "captured_selected_action_count": len(private["selected_actions"]),
            "captured_outer_prediction_count": len(private["outer_predictions"]),
            "stage182_model_head_fit_count": reproduced_stage182["execution_boundaries"][
                "dual_target_model_head_fit_count"
            ],
            "attribution_new_model_fit_count": 0,
            "gold_used_only_for_training_labels_and_offline_attribution": True,
            "runtime_policy_selected": False,
            "runtime_registered_as_default": False,
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
    report["process_guards"] = _process_guards(
        report=report,
        formal_stage182=formal_stage182,
        forbidden=forbidden,
    )
    valid = all(row["passed"] for row in report["process_guards"])
    report["decision"] = {
        "status": (
            "stage183_f1_risk_failure_attribution_complete"
            if valid
            else "stage183_f1_risk_failure_attribution_invalid"
        ),
        "diagnostic_complete": valid,
        "primary_bottleneck": attribution["diagnostic_findings"]["primary_bottleneck"]
        if valid
        else None,
        "replacement_policy_selected": False,
        "runtime_e2e_authorized": False,
        "development_opened": False,
        "test_opened": False,
        "default_runtime_activation": False,
    }
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report


def _authorize_stage182_report(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 182":
        raise ValueError("Stage183 requires the Stage182 report")
    decision = report.get("decision", {})
    if decision.get("status") != "stage182_dual_target_nested_cv_insufficient":
        raise ValueError("Stage183 requires the frozen insufficient Stage182 result")
    if decision.get("experiment_valid") is not True:
        raise ValueError("Stage183 requires a valid Stage182 experiment")
    if not all(row.get("passed") is True for row in report.get("process_guards", [])):
        raise ValueError("Stage182 process guards must all pass")
    boundaries = report.get("execution_boundaries", {})
    if (
        boundaries.get("development_loaded") is not False
        or boundaries.get("test_loaded") is not False
    ):
        raise ValueError("Stage182 development/test boundary drifted")


def _stage182_reproduction(
    *,
    formal: Mapping[str, Any],
    reproduced: Mapping[str, Any],
    selected_actions: Sequence[SelectedAction],
) -> dict[str, Any]:
    formal_nested = formal["dual_target_nested_cv"]
    actual_nested = reproduced["dual_target_nested_cv"]
    formal_aggregate = formal_nested["aggregate"]
    actual_aggregate = actual_nested["aggregate"]
    checks = {
        "status": reproduced["decision"]["status"] == formal["decision"]["status"],
        "stage181_reproduction": reproduced["stage181_reproduction"]["passed"],
        "selected_question_count": (
            actual_aggregate["selected_question_count"]
            == formal_aggregate["selected_question_count"]
        ),
        "strict_expected_count": (
            actual_aggregate["strict_expected_count"] == formal_aggregate["strict_expected_count"]
        ),
        "f1_regression_action_count": (
            actual_aggregate["f1_regression_action_count"]
            == formal_aggregate["f1_regression_action_count"]
        ),
        "gold_citation_delta": (
            actual_aggregate["gold_citation_delta"] == formal_aggregate["gold_citation_delta"]
        ),
        "mean_f1_delta": (
            actual_aggregate["mean_f1_delta_all_questions"]
            == formal_aggregate["mean_f1_delta_all_questions"]
        ),
        "selected_specs": (
            actual_nested["selected_spec_counts"] == formal_nested["selected_spec_counts"]
        ),
        "bootstrap": (actual_nested["paired_bootstrap"] == formal_nested["paired_bootstrap"]),
        "private_selected_count": (
            len(selected_actions) == formal_aggregate["selected_question_count"]
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "actual_selected_question_count": actual_aggregate["selected_question_count"],
        "actual_f1_regression_action_count": actual_aggregate["f1_regression_action_count"],
    }


def _frozen_protocol() -> dict[str, Any]:
    return {
        "formal_stage182_sha256": _STAGE182_SHA256,
        "stage182_result_reproduced_before_attribution": True,
        "attribution_targets": [
            "selected F1-regression concentration",
            "risk-probability calibration",
            "runtime-feature separation",
            "same-candidate safe-alternative headroom",
            "no-inner-eligible-fold failure reasons",
        ],
        "risk_auc_weak_threshold": 0.65,
        "selected_regression_rate_high_threshold": 0.25,
        "safe_alternative_headroom_threshold": 0.50,
        "replacement_model_fit_enabled": False,
        "replacement_policy_selection_enabled": False,
        "development_and_test_closed": True,
        "fallback_strategy_enabled": False,
        "runtime_e2e_enabled": False,
    }


def _process_guards(
    *,
    report: Mapping[str, Any],
    formal_stage182: Mapping[str, Any],
    forbidden: Sequence[str],
) -> list[dict[str, Any]]:
    boundaries = report["execution_boundaries"]
    attribution = report["f1_risk_attribution"]
    expected_outer_predictions = sum(
        fold["heldout_head_metrics"]["action_count"]
        for fold in formal_stage182["dual_target_nested_cv"]["outer_folds"].values()
        if fold["heldout_head_metrics"] is not None
    )
    return [
        _gate("formal_stage182_reproduced", report["stage182_reproduction"]["passed"]),
        _gate("exact_action_rows", boundaries["captured_action_row_count"] == 12_298),
        _gate(
            "exact_selected_actions",
            boundaries["captured_selected_action_count"] == 129,
        ),
        _gate(
            "exact_outer_predictions",
            boundaries["captured_outer_prediction_count"] == expected_outer_predictions,
        ),
        _gate(
            "exact_selected_f1_regressions",
            attribution["selected_action_summary"]["f1_regression_count"] == 55,
        ),
        _gate("no_new_attribution_model_fit", boundaries["attribution_new_model_fit_count"] == 0),
        _gate(
            "stage182_model_fits_reproduced",
            boundaries["stage182_model_head_fit_count"] == 88,
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
        _gate("no_runtime_policy_selected", boundaries["runtime_policy_selected"] is False),
        _gate("default_runtime_unchanged", boundaries["runtime_registered_as_default"] is False),
        _gate("stage178b_not_run", boundaries["stage178b_run"] is False),
        _gate("no_retry", boundaries["retry_action_count"] == 0),
        _gate("no_fallback", boundaries["fallback_action_count"] == 0),
        _gate("public_report_safe", not forbidden),
    ]


def write_stage183_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage183Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    attribution = report["f1_risk_attribution"]
    concentration = attribution["selected_regression_concentration"]
    calibration = attribution["risk_calibration"]["selected_action_population"]
    headroom = attribution["safe_alternative_headroom"]
    severity = attribution["selected_action_summary"]["severity"]
    features = attribution["runtime_feature_separation"]["top_features"][:12]
    no_eligible = attribution["no_inner_eligible_fold_attribution"]
    charts = {
        "f1_regression_rate_by_action_family.svg": _rate_chart(
            "Stage 183 selected F1-regression rate by action family",
            concentration["by_action_family"],
        ),
        "f1_regression_rate_by_route.svg": _rate_chart(
            "Stage 183 selected F1-regression rate by route",
            concentration["by_route"],
        ),
        "selected_risk_calibration.svg": _chart(
            "Stage 183 selected-action predicted versus observed F1 risk",
            tuple(
                BarDatum(
                    f"{row['lower']:.1f}-{row['upper']:.1f} predicted",
                    row["mean_predicted_risk"],
                    f"{row['mean_predicted_risk']:.3f}",
                )
                for row in calibration["bins"]
            )
            + tuple(
                BarDatum(
                    f"{row['lower']:.1f}-{row['upper']:.1f} observed",
                    row["observed_regression_rate"],
                    f"{row['observed_regression_rate']:.3f}",
                )
                for row in calibration["bins"]
            ),
            "risk probability or observed rate",
        ),
        "safe_alternative_headroom.svg": _chart(
            "Stage 183 safer-alternative headroom on regressed selections",
            (
                BarDatum(
                    "any strict alternative",
                    headroom["any_strict_alternative_rate"],
                    f"{headroom['any_strict_alternative_rate']:.3f}",
                ),
                BarDatum(
                    "same/better citation and F1-safe",
                    headroom["same_or_better_citation_safe_alternative_rate"],
                    f"{headroom['same_or_better_citation_safe_alternative_rate']:.3f}",
                ),
                BarDatum(
                    "safe citation-gain alternative",
                    headroom["safe_citation_gain_alternative_rate"],
                    f"{headroom['safe_citation_gain_alternative_rate']:.3f}",
                ),
                BarDatum(
                    "same/better safe in model top3",
                    headroom["same_or_better_safe_alternative_in_model_top3_rate"],
                    f"{headroom['same_or_better_safe_alternative_in_model_top3_rate']:.3f}",
                ),
                BarDatum(
                    "same/better safe in model top5",
                    headroom["same_or_better_safe_alternative_in_model_top5_rate"],
                    f"{headroom['same_or_better_safe_alternative_in_model_top5_rate']:.3f}",
                ),
            ),
            "fraction of 55 regressed selections",
        ),
        "selected_f1_regression_severity.svg": _chart(
            "Stage 183 selected F1-regression severity",
            tuple(
                BarDatum(name, float(value), str(value))
                for name, value in severity.items()
                if name != "distribution"
            ),
            "selected action count",
        ),
        "runtime_feature_univariate_auc.svg": _chart(
            "Runtime-visible feature AUC for F1 risk (Stage 183)",
            tuple(
                BarDatum(
                    row["feature"],
                    row["oriented_univariate_auc"],
                    f"{row['oriented_univariate_auc']:.3f}",
                )
                for row in features
            ),
            "oriented univariate ROC AUC",
        ),
        "no_eligible_fold_failure_reasons.svg": _chart(
            "Stage 183 no-inner-eligible-policy failure reasons",
            tuple(
                BarDatum(name, float(value), str(value))
                for name, value in no_eligible["failure_reason_counts"].items()
            ),
            "candidate failure count",
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage183Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _rate_chart(title: str, rows: Mapping[str, Mapping[str, Any]]) -> str:
    return _chart(
        title,
        tuple(
            BarDatum(
                name,
                row["f1_regression_rate"],
                (
                    f"{row['f1_regression_rate']:.3f} "
                    f"({row['f1_regression_count']}/{row['selected_action_count']})"
                ),
            )
            for name, row in rows.items()
        ),
        "selected F1-regression rate",
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
