from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application import primeqa_hybrid_gain_sensitive_ranking_cv as stage188
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application.composition_gain_sensitive_failure_attribution import (
    GainSensitiveFailureAttributionAccumulator,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

STAGE = "Stage 189"
CREATED_AT = "2026-07-27"
ANALYSIS_ID = "primeqa_hybrid_gain_sensitive_failure_attribution_v1"
STAGE188_SHA256 = "c68946d08750d0e07dadee7f70780048615919d79fe617520f17df078f1c6bcc"
FORBIDDEN_PUBLIC_KEYS = stage188.FORBIDDEN_PUBLIC_KEYS | {
    "frontier_actions",
    "inner_predictions",
    "question_decisions",
}


@dataclass(frozen=True)
class Stage189Visualization:
    """One aggregate Stage 189 visualization."""

    name: str
    path: str


def run_stage189_gain_sensitive_failure_attribution(
    *,
    stage188_report_path: Path,
    stage187_protocol_path: Path,
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
    progress_sink: Any = None,
) -> dict[str, Any]:
    """Reproduce Stage 188 and attribute its inner-OOF opportunity losses."""

    started_at = time.perf_counter()
    stage188_fingerprint = stage173._resolved_fingerprint(stage188_report_path)
    if stage188_fingerprint["sha256"] != STAGE188_SHA256:
        raise ValueError("Stage189 Stage188 report hash mismatch")
    formal_stage188 = _load_json(stage188_report_path)
    _authorize_stage188_report(formal_stage188)
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="stage188_source_authorized")

    accumulator = GainSensitiveFailureAttributionAccumulator()
    reproduced_stage188 = stage188.run_stage188_gain_sensitive_ranking_cv(
        stage187_protocol_path=stage187_protocol_path,
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
        inner_diagnostic_sink=accumulator.consume,
    )
    reproduced_at = time.perf_counter()
    reproduction = _stage188_reproduction(
        formal=formal_stage188,
        reproduced=reproduced_stage188,
    )
    if not reproduction["passed"]:
        raise ValueError("Stage189 did not reproduce the formal Stage188 result")

    attribution = accumulator.finalize()
    attributed_at = time.perf_counter()
    next_protocol = _stage190_protocol(attribution)
    report: dict[str, Any] = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "analysis_id": ANALYSIS_ID,
        "analysis_scope": (
            "Train-only Stage188 inner-OOF opportunity decomposition across all 32 "
            "frozen configurations. Private predictions are consumed one outer context "
            "at a time and only aggregate counts are persisted."
        ),
        "source_authorization": {
            "stage188": stage188_fingerprint,
            "stage188_rerun_sources": reproduced_stage188["source_authorization"],
        },
        "frozen_diagnostic_protocol": _diagnostic_protocol(),
        "stage188_reproduction": reproduction,
        "gain_sensitive_failure_attribution": attribution,
        "stage190_protocol": next_protocol,
        "runtime": reproduced_stage188["runtime"],
        "resource_consumption": reproduced_stage188["resource_consumption"],
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "stage188_reproduction_with_streaming_attribution": round(
                reproduced_at - authorized_at,
                6,
            ),
            "attribution_finalize": round(attributed_at - reproduced_at, 6),
            "wall": round(attributed_at - started_at, 6),
        },
        "execution_boundaries": {
            "train_loaded": True,
            "development_loaded": False,
            "test_loaded": False,
            "stage188_model_fit_count": reproduced_stage188["execution_boundaries"][
                "stage188_model_fit_count"
            ],
            "attribution_new_model_fit_count": attribution["execution"]["new_model_fit_count"],
            "private_bundle_prediction_count_consumed": attribution["execution"][
                "private_bundle_prediction_count_consumed"
            ],
            "public_action_rows_written": attribution["execution"]["public_action_rows_written"],
            "public_prediction_rows_written": attribution["execution"][
                "public_prediction_rows_written"
            ],
            "gold_used_only_for_training_targets_and_offline_attribution": True,
            "replacement_policy_selected": False,
            "runtime_e2e_run": False,
            "runtime_registered_as_default": False,
            "stage178b_run": False,
            "retry_action_count": reproduced_stage188["execution_boundaries"]["retry_action_count"],
            "fallback_action_count": reproduced_stage188["execution_boundaries"][
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
    protocol_frozen = valid and next_protocol["protocol_frozen"]
    report["decision"] = {
        "status": (
            "stage189_gain_sensitive_failure_attribution_complete"
            if valid
            else "stage189_gain_sensitive_failure_attribution_invalid"
        ),
        "diagnostic_complete": valid,
        "primary_bottleneck": attribution["diagnostic_findings"]["primary_bottleneck"]
        if valid
        else None,
        "stage190_protocol_frozen": protocol_frozen,
        "stage190_train_only_experiment_authorized": protocol_frozen,
        "replacement_policy_selected": False,
        "runtime_e2e_authorized": False,
        "development_opened": False,
        "test_opened": False,
        "default_runtime_activation": False,
    }
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report


def _authorize_stage188_report(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 188":
        raise ValueError("Stage189 requires the Stage188 report")
    decision = report.get("decision", {})
    if decision.get("status") != "stage188_gain_sensitive_ranking_insufficient":
        raise ValueError("Stage189 requires the insufficient Stage188 result")
    if decision.get("experiment_valid") is not True:
        raise ValueError("Stage189 requires a valid Stage188 experiment")
    if decision.get("candidate_family_accepted") is not False:
        raise ValueError("Stage189 requires the rejected Stage188 candidate family")
    if not all(row.get("passed") is True for row in report.get("process_guards", [])):
        raise ValueError("Stage188 process guards must all pass")
    boundaries = report.get("execution_boundaries", {})
    if (
        boundaries.get("development_loaded") is not False
        or boundaries.get("test_loaded") is not False
        or boundaries.get("fallback_action_count") != 0
    ):
        raise ValueError("Stage188 data or fallback boundary drifted")
    outer_folds = report.get("gain_sensitive_nested_cv", {}).get("outer_folds", {})
    if len(outer_folds) != 5 or any(row.get("outer_evaluated") for row in outer_folds.values()):
        raise ValueError("Stage189 requires five Stage188 no-eligible outer folds")


def _stage188_reproduction(
    *,
    formal: Mapping[str, Any],
    reproduced: Mapping[str, Any],
) -> dict[str, Any]:
    formal_cv = formal["gain_sensitive_nested_cv"]
    actual_cv = reproduced["gain_sensitive_nested_cv"]
    checks = {
        "status": reproduced["decision"]["status"] == formal["decision"]["status"],
        "experiment_valid": reproduced["decision"]["experiment_valid"] is True,
        "stage182_reproduction": reproduced["stage182_reproduction"]["passed"] is True,
        "dataset": actual_cv["dataset"] == formal_cv["dataset"],
        "outer_fold_ids": set(actual_cv["outer_folds"]) == set(formal_cv["outer_folds"]),
        "eligible_config_counts": all(
            actual_cv["outer_folds"][fold_id]["eligible_config_count"]
            == formal_cv["outer_folds"][fold_id]["eligible_config_count"]
            for fold_id in formal_cv["outer_folds"]
        ),
        "top_candidate_specs": all(
            actual_cv["outer_folds"][fold_id]["top_inner_candidates"][0]["spec"]["name"]
            == formal_cv["outer_folds"][fold_id]["top_inner_candidates"][0]["spec"]["name"]
            for fold_id in formal_cv["outer_folds"]
        ),
        "top_candidate_metrics": all(
            _nested_close(
                actual_cv["outer_folds"][fold_id]["top_inner_candidates"][0]["evaluation"],
                formal_cv["outer_folds"][fold_id]["top_inner_candidates"][0]["evaluation"],
            )
            for fold_id in formal_cv["outer_folds"]
        ),
        "no_outer_evaluation": all(
            row["outer_evaluated"] is False for row in actual_cv["outer_folds"].values()
        ),
        "model_fit_count": actual_cv["execution"]["model_fit_count"] == 240,
        "private_prediction_count": (
            actual_cv["execution"]["private_prediction_count"]
            == formal_cv["execution"]["private_prediction_count"]
        ),
        "process_guards": all(row["passed"] for row in reproduced["process_guards"]),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "tolerance": {"absolute": 1e-6, "relative": 1e-9},
        "actual_model_fit_count": actual_cv["execution"]["model_fit_count"],
        "actual_private_prediction_count": actual_cv["execution"]["private_prediction_count"],
    }


def _diagnostic_protocol() -> dict[str, Any]:
    return {
        "formal_stage188_sha256": STAGE188_SHA256,
        "stage188_reproduced_before_accepting_attribution": True,
        "population": "all Stage188 inner-OOF question contexts",
        "configuration_scope": "all 32 frozen Stage188 configurations",
        "opportunity_categories": [
            "frontier_exclusion",
            "frontier_retained_ranker_miss",
            "strict_selected",
        ],
        "partition_requires_exact_equality": True,
        "primary_bottleneck_rule": "larger lost-opportunity count; exact tie is mixed",
        "factor_aggregates": [
            "feature_representation",
            "safety_estimator",
            "gain_ranker",
            "safety_frontier_margin",
        ],
        "new_model_fit_enabled": False,
        "development_and_test_closed": True,
        "fallback_enabled": False,
        "runtime_e2e_enabled": False,
    }


def _stage190_protocol(attribution: Mapping[str, Any]) -> dict[str, Any]:
    branch = attribution["diagnostic_findings"]["stage190_design_branch"]
    if branch["name"] != "baseline_referenced_strict_change_gate":
        return {
            "protocol_frozen": False,
            "reason": (
                "The predeclared Stage190 baseline-contrast protocol is authorized only "
                "when Stage189 identifies gain_ranker_miss as the primary bottleneck."
            ),
            "diagnostic_branch": branch,
        }
    return {
        "protocol_frozen": True,
        "name": "baseline_referenced_strict_change_gate_nested_cv_v1",
        "objective": (
            "Replace Stage188 gain ordering with a candidate-versus-baseline "
            "strict-improvement score while retaining the relative safety frontier."
        ),
        "data_boundary": {
            "train_only": True,
            "frozen_question_grouped_outer_folds": 5,
            "inner_folds_per_outer": 4,
            "development_loaded": False,
            "test_loaded": False,
            "gold_runtime_features": False,
        },
        "feature_representations": [
            "raw_runtime_plus_candidate_minus_baseline",
            "question_relative_runtime_plus_candidate_minus_baseline",
        ],
        "safety_estimators": [
            "class_balanced_logistic",
            "histogram_gradient_boosting",
        ],
        "strict_change_estimators": [
            "class_balanced_logistic",
            "histogram_gradient_boosting",
        ],
        "strict_change_target": (
            "citation_delta >= 0 and F1_delta >= 0 with at least one strict gain"
        ),
        "baseline_contrast_feature_contract": {
            "retain_original_runtime_features": True,
            "numeric_feature_difference": "candidate value minus baseline value",
            "categorical_feature_indicator": "candidate value equals baseline value",
            "baseline_contrast_values": "zero numeric differences and true equality flags",
            "gold_outcomes_in_features": False,
        },
        "estimator_hyperparameters": {
            "class_balanced_logistic": {
                "class_weight": "balanced",
                "solver": "liblinear",
                "max_iter": 2000,
                "random_state": 190,
            },
            "histogram_gradient_boosting": {
                "learning_rate": 0.05,
                "max_iter": 200,
                "max_leaf_nodes": 15,
                "l2_regularization": 1.0,
                "random_state": 190,
            },
            "question_balanced_sample_weights": True,
            "probability_calibration": "none",
        },
        "safety_frontier_margins": [0.0, 0.02, 0.05, 0.1],
        "strict_change_thresholds": [0.5, 0.6, 0.7, 0.8, 0.9],
        "candidate_grid_count": 160,
        "fit_contract": {
            "safety_heads_per_representation": 4,
            "strict_change_heads_per_representation": 2,
            "representations": 2,
            "fits_per_partition": 12,
            "inner_partitions": 20,
            "outer_refits": 5,
            "maximum_model_fit_count": 300,
        },
        "selection_contract": {
            "candidate_generator": "Stage188 relative safety frontier",
            "within_frontier_order": (
                "descending strict-change score, ascending joint safety excess, "
                "canonical action order"
            ),
            "change_rule": (
                "select the best nonbaseline candidate only when its strict-change "
                "score meets the configured threshold; otherwise select baseline"
            ),
            "baseline_abstention_is_policy_semantics": True,
            "fallback_enabled": False,
        },
        "inner_eligibility": {
            "gold_citation_delta_minimum": 0,
            "mean_f1_delta_minimum": 0.0,
            "citation_nonregressing_folds_minimum": 3,
            "f1_nonregressing_folds_minimum": 3,
            "changed_question_rate_minimum": 0.10,
            "strict_success_rate_minimum": 0.08,
            "strict_success_precision_minimum": 0.60,
        },
        "advancement_gates": {
            "reuse_stage188_fourteen_gates_exactly": True,
            "post_hoc_relaxation_allowed": False,
        },
        "no_eligible_behavior": (
            "record no-eligible and do not evaluate a weaker outer configuration"
        ),
        "runtime_e2e_authorized": False,
        "full_train_policy_selection_authorized": False,
        "default_runtime_activation": False,
    }


def _process_guards(
    *,
    report: Mapping[str, Any],
    forbidden: Sequence[str],
) -> list[dict[str, Any]]:
    reproduction = report["stage188_reproduction"]
    attribution = report["gain_sensitive_failure_attribution"]
    execution = attribution["execution"]
    boundaries = report["execution_boundaries"]
    trajectory = attribution["top_ineligible_trajectory"]
    protocol = report["stage190_protocol"]
    return [
        _gate("stage188_report_hash_matches", True),
        _gate("stage188_report_authorized", True),
        _gate("stage188_reproduction_passed", reproduction["passed"]),
        _gate("stage188_reproduction_check_count_is_12", len(reproduction["checks"]) == 12),
        _gate("five_snapshots_consumed", execution["snapshot_count"] == 5),
        _gate("all_32_configurations_analyzed", len(attribution["configuration_aggregates"]) == 32),
        _gate(
            "private_prediction_count_matches_stage188",
            boundaries["private_bundle_prediction_count_consumed"]
            == reproduction["actual_private_prediction_count"],
        ),
        _gate("stage188_model_fit_count_is_240", boundaries["stage188_model_fit_count"] == 240),
        _gate("no_new_attribution_model_fit", boundaries["attribution_new_model_fit_count"] == 0),
        _gate(
            "opportunity_partition_exact", trajectory["opportunity_partition"]["partition_exact"]
        ),
        _gate(
            "all_configuration_partitions_exact",
            execution["all_configuration_partitions_exact"],
        ),
        _gate("strict_opportunities_exist", trajectory["strict_opportunity_context_count"] > 0),
        _gate(
            "stage190_branch_declared",
            bool(protocol.get("protocol_frozen") or protocol.get("reason")),
        ),
        _gate("public_action_rows_zero", boundaries["public_action_rows_written"] == 0),
        _gate("public_prediction_rows_zero", boundaries["public_prediction_rows_written"] == 0),
        _gate("development_closed", boundaries["development_loaded"] is False),
        _gate("test_closed", boundaries["test_loaded"] is False),
        _gate("no_replacement_policy", boundaries["replacement_policy_selected"] is False),
        _gate("runtime_e2e_not_run", boundaries["runtime_e2e_run"] is False),
        _gate("default_runtime_unchanged", boundaries["runtime_registered_as_default"] is False),
        _gate("stage178b_not_run", boundaries["stage178b_run"] is False),
        _gate("no_retry", boundaries["retry_action_count"] == 0),
        _gate("no_fallback", boundaries["fallback_action_count"] == 0),
        _gate("public_report_safe", not forbidden),
    ]


def write_stage189_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage189Visualization, ...]:
    """Write and XML-validate aggregate Stage 189 charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    attribution = report["gain_sensitive_failure_attribution"]
    trajectory = attribution["top_ineligible_trajectory"]
    partition = trajectory["opportunity_partition"]
    factor = attribution["factor_aggregates"]
    resources = report["resource_consumption"]
    charts = {
        "stage189_opportunity_partition.svg": _chart(
            "Stage 189 strict-opportunity loss partition",
            (
                _count_bar("frontier exclusion", partition["frontier_exclusion_context_count"]),
                _count_bar("ranker miss after retention", partition["ranker_miss_context_count"]),
                _count_bar("strict selected", partition["strict_selected_context_count"]),
            ),
            "top-ineligible trajectory question-context count",
        ),
        "stage189_trajectory_rates.svg": _rate_chart(
            "Stage 189 top-ineligible trajectory rates",
            trajectory,
        ),
        "stage189_fold_frontier_exclusion.svg": _fold_partition_chart(
            attribution,
            metric="frontier_exclusion_context_count",
            title="Stage 189 frontier exclusions by outer context",
        ),
        "stage189_fold_ranker_miss.svg": _fold_partition_chart(
            attribution,
            metric="ranker_miss_context_count",
            title="Stage 189 retained strict opportunities missed by ranker",
        ),
        "stage189_margin_frontier_recall.svg": _factor_rate_chart(
            factor["safety_frontier_margin"],
            metric="frontier_strict_question_recall",
            title="Stage 189 frontier strict-question recall by margin",
        ),
        "stage189_margin_ranker_capture.svg": _factor_rate_chart(
            factor["safety_frontier_margin"],
            metric="conditional_ranker_strict_capture",
            title="Stage 189 conditional ranker capture by margin",
        ),
        "stage189_ranker_capture.svg": _factor_rate_chart(
            factor["gain_ranker"],
            metric="conditional_ranker_strict_capture",
            title="Stage 189 conditional strict capture by gain ranker",
        ),
        "stage189_safety_frontier_recall.svg": _factor_rate_chart(
            factor["safety_estimator"],
            metric="frontier_strict_question_recall",
            title="Stage 189 frontier strict-question recall by safety estimator",
        ),
        "stage189_best_configuration_rates.svg": _chart(
            "Stage 189 family-best diagnostic rates",
            tuple(
                BarDatum(name, row["value"], f"{row['value']:.3f}")
                for name, row in attribution["family_best_configurations"].items()
            ),
            "best aggregate rate among 32 configurations",
            margin_left=360,
        ),
        "stage189_execution_counts.svg": _chart(
            "Stage 189 execution counts",
            (
                _count_bar(
                    "Stage 188 model fits", boundaries_value(report, "stage188_model_fit_count")
                ),
                _count_bar(
                    "new attribution fits",
                    boundaries_value(report, "attribution_new_model_fit_count"),
                ),
                _count_bar("outer snapshots", attribution["execution"]["snapshot_count"]),
                _count_bar("configurations", len(attribution["configuration_aggregates"])),
            ),
            "count",
        ),
        "stage189_memory_gib.svg": _chart(
            "Stage 189 memory usage",
            (
                _gib_bar("peak working set", resources["process_peak_working_set_bytes"]),
                _gib_bar("peak private usage", resources["process_peak_private_usage_bytes"]),
                _gib_bar("minimum system free", resources["minimum_system_available_memory_bytes"]),
            ),
            "GiB",
        ),
        "stage189_process_guards.svg": _chart(
            "Stage 189 process guards",
            tuple(
                BarDatum(row["name"], float(row["passed"]), "pass" if row["passed"] else "fail")
                for row in report["process_guards"]
            ),
            "1 means passed",
            margin_left=880,
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage189Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def boundaries_value(report: Mapping[str, Any], name: str) -> int:
    return int(report["execution_boundaries"][name])


def _rate_chart(title: str, metrics: Mapping[str, Any]) -> str:
    names = (
        "frontier_strict_question_recall",
        "conditional_ranker_strict_capture",
        "actual_strict_opportunity_capture",
        "baseline_change_strict_precision",
        "unsafe_selection_rate",
    )
    return _chart(
        title,
        tuple(BarDatum(name, metrics[name], f"{metrics[name]:.3f}") for name in names),
        "rate",
        margin_left=360,
    )


def _fold_partition_chart(
    attribution: Mapping[str, Any],
    *,
    metric: str,
    title: str,
) -> str:
    return _chart(
        title,
        tuple(
            _count_bar(
                fold_id,
                row["top_ineligible_diagnostics"]["opportunity_partition"][metric],
            )
            for fold_id, row in attribution["outer_folds"].items()
        ),
        "question-context count",
    )


def _factor_rate_chart(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    metric: str,
    title: str,
) -> str:
    return _chart(
        title,
        tuple(BarDatum(name, row[metric], f"{row[metric]:.3f}") for name, row in rows.items()),
        metric,
        margin_left=360,
    )


def _count_bar(name: str, value: int) -> BarDatum:
    return BarDatum(name, value, str(value))


def _gib_bar(name: str, byte_count: int) -> BarDatum:
    gib = byte_count / 1024**3
    return BarDatum(name, gib, f"{gib:.3f} GiB")


def _chart(
    title: str,
    bars: Sequence[BarDatum],
    x_label: str,
    *,
    margin_left: int = 330,
) -> str:
    return render_horizontal_bar_chart_svg(
        title,
        bars,
        x_label,
        width=1680,
        margin_left=margin_left,
        margin_right=260,
    )


def _nested_close(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_nested_close(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _nested_close(actual_value, expected_value)
                for actual_value, expected_value in zip(actual, expected, strict=True)
            )
        )
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)):
            return False
        difference = abs(float(actual) - expected)
        return difference <= 1e-6 + 1e-9 * abs(expected)
    return actual == expected


def _forbidden_keys_found(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in FORBIDDEN_PUBLIC_KEYS:
                found.add(str(key))
            found.update(_forbidden_keys_found(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_forbidden_keys_found(child))
    return found


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(progress_sink: Any, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
