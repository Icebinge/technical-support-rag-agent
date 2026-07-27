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
from ts_rag_agent.application import primeqa_hybrid_joint_risk_winner_cv as stage199
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application.composition_joint_risk_winner_failure_attribution import (
    JointRiskWinnerFailureAttributor,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

STAGE = "Stage 201"
CREATED_AT = "2026-07-27"
ANALYSIS_ID = "primeqa_hybrid_joint_risk_winner_failure_attribution_v1"
STAGE200_SHA256 = "9edf5f3ba725bba501ebe0325ae1a072288a219e5ef655a932ae7722fcf2cf32"
STAGE199_SHA256 = "5b933f524fff1bceb4d4d842e4f3a1aec3160aa3ed337131444ec1b7c2699fee"
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
class Stage201Visualization:
    name: str
    path: str


def run_stage201_joint_risk_winner_failure_attribution(
    *,
    stage200_protocol_path: Path,
    stage199_report_path: Path,
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
    preflight_failure_sink: Callable[[Mapping[str, Any]], None] | None = None,
    recovery_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reproduce Stage199 once and stream its private evidence into Stage201."""

    started_at = time.perf_counter()
    stage200_fingerprint = stage173._resolved_fingerprint(stage200_protocol_path)
    stage199_fingerprint = stage173._resolved_fingerprint(stage199_report_path)
    if stage200_fingerprint["sha256"] != STAGE200_SHA256:
        raise ValueError("Stage201 Stage200 protocol hash mismatch")
    if stage199_fingerprint["sha256"] != STAGE199_SHA256:
        raise ValueError("Stage201 Stage199 report hash mismatch")
    formal_stage200 = _load_json(stage200_protocol_path)
    formal_stage199 = _load_json(stage199_report_path)
    _authorize_stage200_protocol(formal_stage200)
    _authorize_stage199_report(formal_stage199)
    constraints = formal_stage200["frozen_protocol"]["constraint_attribution"]["constraints"]
    attributor = JointRiskWinnerFailureAttributor(constraints=constraints)
    authorized_at = time.perf_counter()

    try:
        reproduced_stage199 = stage199.run_stage199_joint_risk_winner_cv(
            stage198_protocol_path=stage198_protocol_path,
            stage197_report_path=stage197_report_path,
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
    except stage199.Stage199SourceReproductionError as error:
        failure_report = _preflight_failure_report(
            stage200_fingerprint=stage200_fingerprint,
            stage199_fingerprint=stage199_fingerprint,
            reproduction_evidence=error.evidence,
            recovery_context=recovery_context,
        )
        if preflight_failure_sink is not None:
            preflight_failure_sink(failure_report)
        raise
    reproduction = _stage199_reproduction(formal_stage199, reproduced_stage199)
    if not reproduction["passed"]:
        raise ValueError("Stage201 did not exactly reproduce frozen Stage199 evidence")
    attribution = attributor.report()
    analyzed_at = time.perf_counter()
    report: dict[str, Any] = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "analysis_id": ANALYSIS_ID,
        "analysis_scope": (
            "Train-only aggregate failure attribution over all 140 Stage199 outer-cell "
            "contexts, 560 fold-cell contexts, and 41,440 question-cell contexts. "
            "Development and test remain closed; no constraint is relaxed and no new "
            "policy, fallback, replacement, runtime E2E, or default activation is run."
        ),
        "source_authorization": {
            "stage200_protocol": stage200_fingerprint,
            "stage199_formal_report": stage199_fingerprint,
            "stage199_rerun_sources": reproduced_stage199["source_authorization"],
        },
        "stage199_reproduction": reproduction,
        "formal_run_history": _successful_run_history(recovery_context),
        "failure_attribution": attribution,
        "resource_preflight": reproduced_stage199["resource_preflight"],
        "resource_consumption": reproduced_stage199["resource_consumption"],
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "stage199_reproduction_and_streaming_attribution": round(
                analyzed_at - authorized_at, 6
            ),
            "wall": round(analyzed_at - started_at, 6),
        },
        "execution_boundaries": {
            "train_loaded": True,
            "development_loaded": False,
            "test_loaded": False,
            "stage199_model_fit_count": reproduced_stage199["execution_boundaries"][
                "stage199_model_fit_count"
            ],
            "stage199_lightgbm_tree_count": reproduced_stage199["execution_boundaries"][
                "stage199_lightgbm_tree_count"
            ],
            "stage199_private_prediction_count": reproduced_stage199["execution_boundaries"][
                "stage199_private_prediction_count"
            ],
            "additional_diagnostic_model_fit_count": 0,
            "outer_refit_count": 0,
            "private_question_cell_rows_persisted": False,
            "new_policy_search_run": False,
            "constraint_relaxation_run": False,
            "full_train_policy_selected": False,
            "replacement_policy_selected": False,
            "runtime_e2e_run": False,
            "runtime_registered_as_default": False,
            "stage178b_run": False,
            "retry_action_count": reproduced_stage199["execution_boundaries"]["retry_action_count"],
            "fallback_action_count": reproduced_stage199["execution_boundaries"][
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
    report["process_guards"] = _process_guards(report, formal_stage200, forbidden)
    valid = all(row["passed"] for row in report["process_guards"])
    report["decision"] = {
        "status": (
            "stage201_joint_risk_winner_failure_attribution_complete"
            if valid
            else "stage201_joint_risk_winner_failure_attribution_invalid"
        ),
        "experiment_valid": valid,
        "diagnostic_complete": valid,
        "recommended_next_research": attribution["diagnostic_finding"]["recommended_next_research"],
        "development_opened": False,
        "test_opened": False,
        "new_policy_search_authorized": False,
        "constraint_relaxation_authorized": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "runtime_e2e_authorized": False,
        "default_runtime_activation": False,
    }
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report


def write_stage201_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage201Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    attribution = report["failure_attribution"]
    constraints = attribution["constraint_attribution"]["constraints"]
    folds = attribution["fold_attribution"]
    questions = attribution["question_context_attribution"]
    resources = report["resource_consumption"]
    charts: dict[str, str] = {
        "stage201_constraint_failures.svg": _chart(
            "Stage 201 eligibility failures by constraint",
            [_count_bar(name, row["failure_count"]) for name, row in constraints.items()],
            margin_left=650,
        ),
        "stage201_near_boundary.svg": _chart(
            "Stage 201 failed cells near each boundary",
            [_count_bar(name, row["near_boundary_count"]) for name, row in constraints.items()],
            margin_left=650,
        ),
        "stage201_single_constraint_removal.svg": _chart(
            "Stage 201 cells repaired by one diagnostic removal",
            [
                _count_bar(name, value)
                for name, value in attribution["constraint_attribution"][
                    "single_constraint_removal_pass_counts"
                ].items()
            ],
            margin_left=650,
        ),
        "stage201_failed_constraint_count.svg": _chart(
            "Stage 201 failed-constraint count distribution",
            [
                _count_bar(f"{name} failed", value)
                for name, value in attribution["constraint_attribution"][
                    "failed_constraint_count_distribution"
                ].items()
            ],
        ),
        "stage201_pareto_counts.svg": _chart(
            "Stage 201 capture-unsafe Pareto cells by outer context",
            [
                _count_bar(name, value)
                for name, value in attribution["constraint_attribution"][
                    "pareto_nondominated_cell_count_by_outer_context"
                ].items()
            ],
        ),
        "stage201_fold_violation_totals.svg": _chart(
            "Stage 201 fold-level violation totals",
            [
                _count_bar(metric, sum(values.values()))
                for metric, values in folds["violation_counts_by_metric_and_fold"].items()
            ],
            margin_left=600,
        ),
        "stage201_aggregate_pass_fold_fail.svg": _chart(
            "Stage 201 aggregate-pass but fold-count-fail cells",
            [
                _count_bar(metric, value)
                for metric, value in folds["aggregate_pass_but_fold_count_fail_by_metric"].items()
            ],
            margin_left=600,
        ),
        "stage201_selected_outcomes.svg": _chart(
            "Stage 201 selected outcome partition",
            [
                _count_bar(name, value)
                for name, value in questions["aggregate"]["selected_outcome_counts"].items()
            ],
            margin_left=650,
        ),
        "stage201_opportunity_mechanisms.svg": _chart(
            "Stage 201 strict-opportunity mechanism partition",
            [
                _count_bar(name, value)
                for name, value in questions["aggregate"][
                    "strict_opportunity_mechanism_counts"
                ].items()
            ],
            margin_left=540,
        ),
        "stage201_risk_signal_unsafe.svg": _chart(
            "Stage 201 unsafe winner rate by risk signal",
            [
                _rate_bar(
                    name,
                    _unsafe_rate(row["selected_outcome_counts"]),
                )
                for name, row in questions["by_risk_signal"].items()
            ],
            margin_left=620,
        ),
        "stage201_winner_rule_unsafe.svg": _chart(
            "Stage 201 unsafe winner rate by winner rule",
            [
                _rate_bar(name, _unsafe_rate(row["selected_outcome_counts"]))
                for name, row in questions["by_winner_rule"].items()
            ],
            margin_left=620,
        ),
        "stage201_risk_signal_strict.svg": _chart(
            "Stage 201 strict selected rate by risk signal",
            [
                _rate_bar(
                    name,
                    _ratio(
                        row["selected_outcome_counts"]["strict_success"],
                        row["question_cell_context_count"],
                    ),
                )
                for name, row in questions["by_risk_signal"].items()
            ],
            margin_left=620,
        ),
        "stage201_winner_rule_strict.svg": _chart(
            "Stage 201 strict selected rate by winner rule",
            [
                _rate_bar(
                    name,
                    _ratio(
                        row["selected_outcome_counts"]["strict_success"],
                        row["question_cell_context_count"],
                    ),
                )
                for name, row in questions["by_winner_rule"].items()
            ],
            margin_left=620,
        ),
        "stage201_research_axis_scores.svg": _chart(
            "Stage 201 research-axis failure scores",
            [
                _count_bar(name, value)
                for name, value in attribution["diagnostic_finding"][
                    "failure_count_score_by_research_axis"
                ].items()
            ],
        ),
        "stage201_execution.svg": _chart(
            "Stage 201 exact execution counts",
            [
                _count_bar(
                    "model fits",
                    report["execution_boundaries"]["stage199_model_fit_count"],
                ),
                _count_bar(
                    "LightGBM trees",
                    report["execution_boundaries"]["stage199_lightgbm_tree_count"],
                ),
                _count_bar(
                    "private predictions",
                    report["execution_boundaries"]["stage199_private_prediction_count"],
                ),
                _count_bar(
                    "question-cell contexts",
                    attribution["population"]["question_cell_context_count"],
                ),
            ],
        ),
        "stage201_resources.svg": _chart(
            "Stage 201 resource consumption",
            [
                _gib_bar("peak working set", resources["process_peak_working_set_bytes"]),
                _gib_bar("peak private usage", resources["process_peak_private_usage_bytes"]),
                _gib_bar(
                    "minimum system available",
                    resources["minimum_system_available_memory_bytes"],
                ),
            ],
        ),
        "stage201_process_guards.svg": _chart(
            "Stage 201 process guards",
            [
                _count_bar("passed", sum(row["passed"] for row in report["process_guards"])),
                _count_bar("total", len(report["process_guards"])),
            ],
        ),
    }
    visualizations = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        visualizations.append(Stage201Visualization(name, str(path)))
    return tuple(visualizations)


def write_stage201_report_bundle(
    *, report: Mapping[str, Any], output_path: Path, visualization_dir: Path
) -> dict[str, Any]:
    core_report = dict(report)
    _write_json_atomic(output_path, core_report)
    visualizations = write_stage201_visualizations(report=core_report, output_dir=visualization_dir)
    final_report = {
        **core_report,
        "visualizations": [{"name": item.name, "path": item.path} for item in visualizations],
    }
    _write_json_atomic(output_path, final_report)
    return final_report


def write_stage201_preflight_failure(*, report: Mapping[str, Any], output_path: Path) -> None:
    """Persist a structured failed preflight before propagating its exception."""

    _write_json_atomic(output_path, report)


def _preflight_failure_report(
    *,
    stage200_fingerprint: Mapping[str, Any],
    stage199_fingerprint: Mapping[str, Any],
    reproduction_evidence: Mapping[str, Any],
    recovery_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "analysis_id": ANALYSIS_ID,
        "artifact_type": "stage201_preflight_failure_evidence",
        "source_authorization": {
            "stage200_protocol": dict(stage200_fingerprint),
            "stage199_formal_report": dict(stage199_fingerprint),
        },
        "failed_phase": "stage182_exact_reproduction_preflight",
        "process_id": os.getpid(),
        "stage182_reproduction": dict(reproduction_evidence),
        "recovery_context": dict(recovery_context or {}),
        "execution_boundaries": {
            "stage199_model_fit_count": 0,
            "stage201_attributed_outer_context_count": 0,
            "development_loaded": False,
            "test_loaded": False,
            "constraint_relaxation_run": False,
            "new_policy_search_run": False,
            "retry_action_count": 0,
            "fallback_action_count": 0,
        },
        "decision": {
            "status": "stage201_preflight_failed",
            "experiment_valid": False,
            "diagnostic_complete": False,
            "formal_report_created": False,
        },
    }


def _successful_run_history(
    recovery_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not recovery_context:
        return {
            "formal_attempt_count": 1,
            "prior_failed_attempt_count": 0,
            "user_authorized_corrected_rerun": False,
            "attempts": [
                {
                    "attempt": 1,
                    "outcome": "completed",
                    "process_id": os.getpid(),
                    "stage199_model_fit_count": 100,
                }
            ],
        }
    prior = dict(recovery_context["prior_failed_attempt"])
    return {
        "formal_attempt_count": 2,
        "prior_failed_attempt_count": 1,
        "user_authorized_corrected_rerun": (
            recovery_context.get("user_authorized_corrected_rerun") is True
        ),
        "authorization_selection": recovery_context.get("authorization_selection"),
        "attempts": [
            {"attempt": 1, **prior},
            {
                "attempt": 2,
                "outcome": "completed",
                "process_id": os.getpid(),
                "stage199_model_fit_count": 100,
            },
        ],
    }


def _authorize_stage200_protocol(report: Mapping[str, Any]) -> None:
    decision = report.get("decision", {})
    if report.get("stage") != "Stage 200":
        raise ValueError("Stage201 requires Stage200")
    if (
        decision.get("protocol_valid") is not True
        or decision.get("stage201_train_only_attribution_authorized") is not True
    ):
        raise ValueError("Stage200 did not authorize Stage201")
    if decision.get("development_opened") is not False or decision.get("test_opened") is not False:
        raise ValueError("Stage201 requires closed development and test sets")
    guards = report.get("guard_checks", [])
    if len(guards) != 66 or not all(row.get("passed") is True for row in guards):
        raise ValueError("Stage201 requires all 66 Stage200 guards")


def _authorize_stage199_report(report: Mapping[str, Any]) -> None:
    decision = report.get("decision", {})
    if report.get("stage") != "Stage 199":
        raise ValueError("Stage201 requires Stage199")
    if decision.get("experiment_valid") is not True:
        raise ValueError("Stage201 requires valid Stage199 evidence")
    if decision.get("candidate_family_accepted") is not False:
        raise ValueError("Stage201 requires the frozen insufficient Stage199 result")
    if decision.get("development_opened") is not False or decision.get("test_opened") is not False:
        raise ValueError("Stage201 requires closed Stage199 development and test sets")


def _stage199_reproduction(
    formal: Mapping[str, Any], reproduced: Mapping[str, Any]
) -> dict[str, Any]:
    formal_cv = formal["joint_risk_winner_nested_cv"]
    reproduced_cv = reproduced["joint_risk_winner_nested_cv"]
    evidence_fields = (
        "dataset",
        "outer_contexts",
        "aggregate",
        "aggregate_diagnostics",
        "paired_bootstrap",
        "unavailable_bootstrap",
        "cell_aggregates",
        "risk_signal_factor_aggregates",
        "winner_rule_factor_aggregates",
        "complete_pool_risk_metrics",
        "selected_risk_signal_counts",
        "selected_winner_rule_counts",
        "advancement_gates",
        "candidate_family_accepted",
    )
    checks = {
        name: stage199_core._nested_close(reproduced_cv.get(name), formal_cv.get(name))
        for name in evidence_fields
    }
    execution_fields = (
        "model_fit_count",
        "pool_safety_fit_count",
        "gain_ranker_fit_count",
        "classifier_risk_fit_count",
        "pairwise_safety_fit_count",
        "tree_count",
        "group_contract_validation_count",
        "control_reproduction_count",
        "all_controls_reproduced_exactly",
        "private_prediction_count",
        "public_training_rows_written",
        "public_prediction_rows_written",
        "feature_count_by_representation",
    )
    checks["execution"] = stage199_core._nested_close(
        {name: reproduced_cv["execution"][name] for name in execution_fields},
        {name: formal_cv["execution"][name] for name in execution_fields},
    )
    return {
        "source_report_sha256": STAGE199_SHA256,
        "checks": checks,
        "passed_check_count": sum(checks.values()),
        "check_count": len(checks),
        "passed": all(checks.values()),
        "timing_and_resource_values_excluded_from_equality": True,
    }


def _process_guards(
    report: Mapping[str, Any],
    protocol: Mapping[str, Any],
    forbidden: Sequence[str],
) -> list[dict[str, Any]]:
    attribution = report["failure_attribution"]
    population = attribution["population"]
    questions = attribution["question_context_attribution"]
    boundaries = report["execution_boundaries"]
    run_history = report["formal_run_history"]
    frozen_population = protocol["frozen_protocol"]["diagnostic_population"]
    checks = (
        ("stage199_reproduced_exactly", report["stage199_reproduction"]["passed"] is True),
        ("train_loaded", boundaries["train_loaded"] is True),
        ("development_closed", boundaries["development_loaded"] is False),
        ("test_closed", boundaries["test_loaded"] is False),
        ("five_outer_contexts", population["outer_context_count"] == 5),
        (
            "outer_cell_population_exact",
            population["outer_cell_context_count"] == frozen_population["outer_cell_context_count"],
        ),
        (
            "fold_cell_population_exact",
            population["fold_cell_context_count"] == frozen_population["fold_cell_context_count"],
        ),
        (
            "question_cell_population_exact",
            population["question_cell_context_count"]
            == frozen_population["question_cell_context_count"],
        ),
        (
            "thirteen_constraints",
            len(attribution["constraint_attribution"]["constraints"]) == 13,
        ),
        (
            "constraint_population_exact",
            all(
                row["pass_count"] + row["failure_count"] == population["outer_cell_context_count"]
                for row in attribution["constraint_attribution"]["constraints"].values()
            ),
        ),
        (
            "failed_count_distribution_exact",
            sum(
                attribution["constraint_attribution"][
                    "failed_constraint_count_distribution"
                ].values()
            )
            == population["outer_cell_context_count"],
        ),
        (
            "selected_outcome_partition_exact",
            questions["aggregate"]["selected_outcome_partition_exact"] is True,
        ),
        (
            "opportunity_partition_exact",
            questions["aggregate"]["strict_opportunity_partition_exact"] is True,
        ),
        (
            "all_factor_outcome_partitions_exact",
            all(
                row["selected_outcome_partition_exact"]
                and row["strict_opportunity_partition_exact"]
                for dimension in ("by_outer_context", "by_risk_signal", "by_winner_rule")
                for row in questions[dimension].values()
            ),
        ),
        ("near_boundary_route_a", attribution["near_boundary_contract"]["selection"] == "A"),
        (
            "stage200_not_modified_by_clarification",
            attribution["near_boundary_contract"]["stage200_artifact_modified"] is False,
        ),
        (
            "formal_attempt_history_valid",
            run_history["formal_attempt_count"] == run_history["prior_failed_attempt_count"] + 1,
        ),
        (
            "prior_failures_transparently_recorded",
            len(run_history["attempts"]) == run_history["formal_attempt_count"],
        ),
        (
            "corrected_rerun_authorized_when_needed",
            run_history["prior_failed_attempt_count"] == 0
            or run_history["user_authorized_corrected_rerun"] is True,
        ),
        (
            "prior_attempts_have_zero_stage199_fits",
            all(row["stage199_model_fit_count"] == 0 for row in run_history["attempts"][:-1]),
        ),
        ("model_fit_count_exact", boundaries["stage199_model_fit_count"] == 100),
        ("tree_count_exact", boundaries["stage199_lightgbm_tree_count"] == 18_000),
        (
            "private_prediction_count_exact",
            boundaries["stage199_private_prediction_count"] == 245_960,
        ),
        (
            "no_additional_diagnostic_fit",
            boundaries["additional_diagnostic_model_fit_count"] == 0,
        ),
        ("no_outer_refit", boundaries["outer_refit_count"] == 0),
        (
            "no_question_rows_persisted",
            boundaries["private_question_cell_rows_persisted"] is False,
        ),
        ("no_new_policy_search", boundaries["new_policy_search_run"] is False),
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


def _chart(title: str, bars: Sequence[BarDatum], *, margin_left: int = 520) -> str:
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


def _gib_bar(name: str, value: int) -> BarDatum:
    gib = value / 1024**3
    return BarDatum(name, gib, f"{gib:.3f}")


def _unsafe_rate(outcomes: Mapping[str, int]) -> float:
    unsafe = sum(value for name, value in outcomes.items() if name.startswith("unsafe_"))
    return _ratio(unsafe, sum(outcomes.values()))


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator / denominator), 6) if denominator else 0.0


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
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _emit(progress_sink: stage182.ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
