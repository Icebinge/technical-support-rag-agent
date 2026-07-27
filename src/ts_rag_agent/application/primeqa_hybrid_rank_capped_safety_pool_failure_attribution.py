from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application import primeqa_hybrid_rank_capped_safety_pool_cv as stage191
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application.composition_rank_capped_safety_pool_failure_attribution import (
    RankCappedSafetyPoolFailureAttributionAccumulator,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

STAGE = "Stage 192"
CREATED_AT = "2026-07-27"
ANALYSIS_ID = "primeqa_hybrid_rank_capped_safety_pool_failure_attribution_v1"
STAGE191_SHA256 = "1747bd9a47a7f233b97e62e38550fc61d8eee8c3ea54cd063c32a66ee14f29d9"
FORBIDDEN_PUBLIC_KEYS = stage191.FORBIDDEN_PUBLIC_KEYS | {
    "inner_predictions",
    "pool_decisions",
    "question_decisions",
}


@dataclass(frozen=True)
class Stage192Visualization:
    """One public-safe Stage 192 visualization."""

    name: str
    path: str


def run_stage192_rank_capped_safety_pool_failure_attribution(
    *,
    stage191_report_path: Path,
    stage190_protocol_path: Path,
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
    """Reproduce Stage 191 and attribute its inner-OOF opportunity losses."""

    started_at = time.perf_counter()
    stage191_fingerprint = stage173._resolved_fingerprint(stage191_report_path)
    if stage191_fingerprint["sha256"] != STAGE191_SHA256:
        raise ValueError("Stage192 Stage191 report hash mismatch")
    formal_stage191 = _load_json(stage191_report_path)
    _authorize_stage191_report(formal_stage191)
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="stage191_source_authorized")

    accumulator = RankCappedSafetyPoolFailureAttributionAccumulator()
    reproduced_stage191 = stage191.run_stage191_rank_capped_safety_pool_cv(
        stage190_protocol_path=stage190_protocol_path,
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
    reproduction = _stage191_reproduction(
        formal=formal_stage191,
        reproduced=reproduced_stage191,
    )
    if not reproduction["passed"]:
        raise ValueError("Stage192 did not reproduce the formal Stage191 result")

    attribution = accumulator.finalize()
    attributed_at = time.perf_counter()
    report: dict[str, Any] = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "analysis_id": ANALYSIS_ID,
        "analysis_scope": (
            "Train-only Stage191 inner-OOF opportunity decomposition across all 32 "
            "frozen configurations. Private predictions are consumed one outer context "
            "at a time and only aggregate counts are persisted."
        ),
        "source_authorization": {
            "stage191": stage191_fingerprint,
            "stage191_rerun_sources": reproduced_stage191["source_authorization"],
        },
        "frozen_diagnostic_protocol": _diagnostic_protocol(),
        "stage191_reproduction": reproduction,
        "rank_capped_safety_pool_failure_attribution": attribution,
        "runtime": reproduced_stage191["runtime"],
        "resource_consumption": reproduced_stage191["resource_consumption"],
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "stage191_reproduction_with_streaming_attribution": round(
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
            "stage191_model_fit_count": reproduced_stage191["execution_boundaries"][
                "stage191_model_fit_count"
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
            "retry_action_count": reproduced_stage191["execution_boundaries"]["retry_action_count"],
            "fallback_action_count": reproduced_stage191["execution_boundaries"][
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
    report["decision"] = {
        "status": (
            "stage192_rank_capped_safety_pool_failure_attribution_complete"
            if valid
            else "stage192_rank_capped_safety_pool_failure_attribution_invalid"
        ),
        "diagnostic_complete": valid,
        "primary_bottleneck": (
            attribution["diagnostic_findings"]["primary_bottleneck"] if valid else None
        ),
        "next_protocol_frozen": False,
        "next_train_only_experiment_authorized": False,
        "replacement_policy_selected": False,
        "runtime_e2e_authorized": False,
        "development_opened": False,
        "test_opened": False,
        "default_runtime_activation": False,
    }
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report


def _authorize_stage191_report(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 191":
        raise ValueError("Stage192 requires the Stage191 report")
    decision = report.get("decision", {})
    if decision.get("status") != "stage191_rank_capped_safety_pool_insufficient":
        raise ValueError("Stage192 requires the insufficient Stage191 result")
    if decision.get("experiment_valid") is not True:
        raise ValueError("Stage192 requires a valid Stage191 experiment")
    if decision.get("candidate_family_accepted") is not False:
        raise ValueError("Stage192 requires the rejected Stage191 candidate family")
    if not all(row.get("passed") is True for row in report.get("process_guards", [])):
        raise ValueError("Stage191 process guards must all pass")
    boundaries = report.get("execution_boundaries", {})
    if (
        boundaries.get("development_loaded") is not False
        or boundaries.get("test_loaded") is not False
        or boundaries.get("fallback_action_count") != 0
    ):
        raise ValueError("Stage191 data or fallback boundary drifted")
    outer_folds = report.get("rank_capped_safety_pool_nested_cv", {}).get("outer_folds", {})
    if (
        len(outer_folds) != 5
        or sum(row.get("outer_evaluated") is True for row in outer_folds.values()) != 4
    ):
        raise ValueError("Stage192 requires the Stage191 four-of-five outer result")


def _stage191_reproduction(
    *,
    formal: Mapping[str, Any],
    reproduced: Mapping[str, Any],
) -> dict[str, Any]:
    formal_cv = formal["rank_capped_safety_pool_nested_cv"]
    actual_cv = reproduced["rank_capped_safety_pool_nested_cv"]
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
        "selected_specs": all(
            actual_cv["outer_folds"][fold_id]["selected_spec"]
            == formal_cv["outer_folds"][fold_id]["selected_spec"]
            for fold_id in formal_cv["outer_folds"]
        ),
        "selected_inner_metrics": all(
            _nested_close(
                actual_cv["outer_folds"][fold_id]["selected_inner_evaluation"],
                formal_cv["outer_folds"][fold_id]["selected_inner_evaluation"],
            )
            and _nested_close(
                actual_cv["outer_folds"][fold_id]["selected_inner_pool_metrics"],
                formal_cv["outer_folds"][fold_id]["selected_inner_pool_metrics"],
            )
            for fold_id in formal_cv["outer_folds"]
        ),
        "outer_metrics": all(
            _nested_close(
                actual_cv["outer_folds"][fold_id]["outer_evaluation"],
                formal_cv["outer_folds"][fold_id]["outer_evaluation"],
            )
            and _nested_close(
                actual_cv["outer_folds"][fold_id]["outer_pool_metrics"],
                formal_cv["outer_folds"][fold_id]["outer_pool_metrics"],
            )
            for fold_id in formal_cv["outer_folds"]
        ),
        "aggregate": _nested_close(actual_cv["aggregate"], formal_cv["aggregate"]),
        "aggregate_pool": _nested_close(
            actual_cv["aggregate_pool_metrics"], formal_cv["aggregate_pool_metrics"]
        ),
        "advancement_gates": actual_cv["advancement_gates"] == formal_cv["advancement_gates"],
        "model_fit_count": actual_cv["execution"]["model_fit_count"] == 288,
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
        "formal_stage191_sha256": STAGE191_SHA256,
        "stage191_reproduced_before_accepting_attribution": True,
        "population": "all Stage191 inner-OOF question contexts",
        "policy_config_count": 32,
        "reference_trajectory": (
            "selected eligible spec when available, otherwise top ineligible spec"
        ),
        "opportunity_partition": [
            "candidate pool exclusion",
            "retained strict opportunity missed by ranker",
            "strict action selected",
        ],
        "new_model_fit_count": 0,
        "private_rows_persisted": False,
        "development_opened": False,
        "test_opened": False,
        "fallback_enabled": False,
    }


def _process_guards(
    *,
    report: Mapping[str, Any],
    forbidden: Sequence[str],
) -> list[dict[str, Any]]:
    reproduction = report["stage191_reproduction"]
    attribution = report["rank_capped_safety_pool_failure_attribution"]
    boundaries = report["execution_boundaries"]
    return [
        _gate("stage191_report_hash_matches", True),
        _gate("stage191_report_authorized", True),
        _gate("stage191_reproduction_passed", reproduction["passed"] is True),
        _gate("stage191_reproduction_check_count_is_15", len(reproduction["checks"]) == 15),
        _gate("five_snapshots_consumed", attribution["execution"]["snapshot_count"] == 5),
        _gate("all_32_configurations_analyzed", len(attribution["configuration_aggregates"]) == 32),
        _gate(
            "private_prediction_count_matches_stage191",
            attribution["execution"]["private_bundle_prediction_count_consumed"] == 393536,
        ),
        _gate("stage191_model_fit_count_is_288", boundaries["stage191_model_fit_count"] == 288),
        _gate("no_new_attribution_model_fit", boundaries["attribution_new_model_fit_count"] == 0),
        _gate(
            "opportunity_partition_exact",
            attribution["diagnostic_findings"]["opportunity_partition_exact"] is True,
        ),
        _gate(
            "all_configuration_partitions_exact",
            attribution["execution"]["all_configuration_partitions_exact"] is True,
        ),
        _gate(
            "strict_opportunities_exist",
            attribution["reference_trajectory"]["strict_opportunity_context_count"] > 0,
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


def write_stage192_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage192Visualization, ...]:
    """Write and XML-validate aggregate Stage 192 charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    attribution = report["rank_capped_safety_pool_failure_attribution"]
    reference = attribution["reference_trajectory"]
    resources = report["resource_consumption"]
    partition = reference["opportunity_partition"]
    charts = {
        "stage192_opportunity_partition.svg": _chart(
            "Stage 192 strict-opportunity loss partition",
            tuple(
                _count_bar(name, value)
                for name, value in (
                    ("pool exclusion", partition["pool_exclusion_context_count"]),
                    ("ranker miss after retention", partition["ranker_miss_context_count"]),
                    ("strict selected", partition["strict_selected_context_count"]),
                )
            ),
            "reference-trajectory question-context count",
        ),
        "stage192_reference_rates.svg": _rate_chart(
            "Stage 192 reference-trajectory rates",
            reference,
        ),
        "stage192_fold_pool_exclusion.svg": _fold_partition_chart(
            attribution,
            title="Stage 192 pool exclusions by outer context",
            metric="pool_exclusion_context_count",
        ),
        "stage192_fold_ranker_miss.svg": _fold_partition_chart(
            attribution,
            title="Stage 192 retained strict opportunities missed by ranker",
            metric="ranker_miss_context_count",
        ),
        "stage192_cap_pool_recall.svg": _factor_rate_chart(
            attribution,
            factor="pool_cap",
            metric="strict_opportunity_pool_recall",
            title="Stage 192 strict-opportunity pool recall by cap",
        ),
        "stage192_cap_ranker_capture.svg": _factor_rate_chart(
            attribution,
            factor="pool_cap",
            metric="conditional_ranker_strict_capture",
            title="Stage 192 conditional ranker capture by cap",
        ),
        "stage192_ranker_capture.svg": _factor_rate_chart(
            attribution,
            factor="gain_ranker",
            metric="conditional_ranker_strict_capture",
            title="Stage 192 conditional strict capture by ranker",
        ),
        "stage192_safety_unsafe_rate.svg": _factor_rate_chart(
            attribution,
            factor="safety_estimator",
            metric="unsafe_selection_rate",
            title="Stage 192 unsafe selection rate by safety estimator",
        ),
        "stage192_miss_breakdown.svg": _chart(
            "Stage 192 within-pool ranker-miss winners",
            (
                _count_bar(
                    "safe-zero winner",
                    reference["ranker_miss_breakdown"]["safe_zero_winner_context_count"],
                ),
                _count_bar(
                    "unsafe winner",
                    reference["ranker_miss_breakdown"]["unsafe_winner_context_count"],
                ),
            ),
            "question-context count",
        ),
        "stage192_execution_counts.svg": _chart(
            "Stage 192 execution counts",
            (
                _count_bar(
                    "Stage 191 model fits", boundaries_value(report, "stage191_model_fit_count")
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
        "stage192_memory_gib.svg": _chart(
            "Stage 192 memory usage",
            (
                _gib_bar("peak working set", resources["process_peak_working_set_bytes"]),
                _gib_bar("peak private usage", resources["process_peak_private_usage_bytes"]),
                _gib_bar("minimum system free", resources["minimum_system_available_memory_bytes"]),
            ),
            "GiB",
        ),
        "stage192_process_guards.svg": _chart(
            "Stage 192 process guards",
            tuple(
                BarDatum(row["name"], float(row["passed"]), "pass" if row["passed"] else "fail")
                for row in report["process_guards"]
            ),
            "1 means passed",
            margin_left=930,
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage192Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def boundaries_value(report: Mapping[str, Any], name: str) -> int:
    return int(report["execution_boundaries"][name])


def _rate_chart(title: str, metrics: Mapping[str, Any]) -> str:
    return _chart(
        title,
        tuple(
            BarDatum(name, metrics[name], f"{metrics[name]:.3f}")
            for name in (
                "strict_opportunity_pool_recall",
                "conditional_ranker_strict_capture",
                "actual_strict_opportunity_capture",
                "baseline_change_strict_precision",
                "unsafe_selection_rate",
            )
        ),
        "rate",
    )


def _fold_partition_chart(
    attribution: Mapping[str, Any],
    *,
    title: str,
    metric: str,
) -> str:
    return _chart(
        title,
        tuple(
            _count_bar(
                fold_id,
                row["reference_diagnostics"]["opportunity_partition"][metric],
            )
            for fold_id, row in attribution["outer_folds"].items()
        ),
        "question-context count",
    )


def _factor_rate_chart(
    attribution: Mapping[str, Any],
    *,
    factor: str,
    metric: str,
    title: str,
) -> str:
    factor_rows = attribution["factor_aggregates"][factor]
    factor_names = list(factor_rows)
    if factor == "pool_cap":
        factor_names.sort(key=lambda value: (value == "all", int(value) if value != "all" else 0))
    return _chart(
        title,
        tuple(
            BarDatum(name, factor_rows[name][metric], f"{factor_rows[name][metric]:.3f}")
            for name in factor_names
        ),
        metric,
    )


def _count_bar(name: str, value: int) -> BarDatum:
    return BarDatum(name, value, str(value))


def _gib_bar(name: str, byte_count: int) -> BarDatum:
    value = byte_count / (1024**3)
    return BarDatum(name, value, f"{value:.3f} GiB")


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


def _nested_close(actual: Any, expected: Any) -> bool:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _nested_close(actual[key], expected[key]) for key in actual
        )
    if (
        isinstance(actual, Sequence)
        and isinstance(expected, Sequence)
        and not isinstance(actual, (str, bytes))
        and not isinstance(expected, (str, bytes))
    ):
        return len(actual) == len(expected) and all(
            _nested_close(left, right) for left, right in zip(actual, expected, strict=True)
        )
    if isinstance(actual, float) or isinstance(expected, float):
        return abs(float(actual) - float(expected)) <= 1e-6 + 1e-9 * abs(float(expected))
    return actual == expected


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


def _emit(progress_sink: Any, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
