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
from ts_rag_agent.application.composition_rank_capped_safety_pool import (
    RankCappedSafetyPoolInnerDiagnosticSink,
    run_rank_capped_safety_pool_nested_cv,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

STAGE = "Stage 191"
CREATED_AT = "2026-07-27"
ANALYSIS_ID = "primeqa_hybrid_rank_capped_safety_pool_nested_cv_v1"
STAGE190_SHA256 = "6558798d6cee0cedb7b01fb864cda749e3f8e63793535ce764acbfaabbb6e07b"
FORBIDDEN_PUBLIC_KEYS = stage183._FORBIDDEN_PUBLIC_KEYS | {
    "candidate_actions",
    "citation_loss_probability",
    "f1_loss_probability",
    "feature_rows",
    "gain_score",
    "listwise_targets",
    "pair_rows",
    "pool_actions",
    "predictions",
    "selected_actions",
}


@dataclass(frozen=True)
class Stage191Visualization:
    """One aggregate Stage 191 visualization."""

    name: str
    path: str


def run_stage191_rank_capped_safety_pool_cv(
    *,
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
    progress_sink: stage182.ProgressSink | None = None,
    inner_diagnostic_sink: RankCappedSafetyPoolInnerDiagnosticSink | None = None,
) -> dict[str, Any]:
    """Reproduce Stage 182 and run the authorized Stage 191 train-only CV."""

    started_at = time.perf_counter()
    stage190_fingerprint = stage173._resolved_fingerprint(stage190_protocol_path)
    if stage190_fingerprint["sha256"] != STAGE190_SHA256:
        raise ValueError("Stage191 Stage190 protocol hash mismatch")
    formal_stage190 = _load_json(stage190_protocol_path)
    _authorize_stage190_protocol(formal_stage190)
    formal_stage182 = _load_json(stage182_report_path)
    stage183._authorize_stage182_report(formal_stage182)
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="stage190_protocol_authorized")

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
        raise ValueError("Stage191 did not reproduce the formal Stage182 result")
    gc.collect()
    tracker.capture("stage182_temporary_resources_released")

    nested_cv = run_rank_capped_safety_pool_nested_cv(
        action_rows=private["action_rows"],
        stage182_selected_actions=private["selected_actions"],
        progress_sink=progress_sink,
        inner_diagnostic_sink=inner_diagnostic_sink,
    )
    tracker.capture("rank_capped_safety_pool_nested_cv_complete")
    analyzed_at = time.perf_counter()
    report: dict[str, Any] = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "analysis_id": ANALYSIS_ID,
        "analysis_scope": (
            "Train-only five-by-four nested cross-validation of 32 rank-capped "
            "safety-pool configurations. Strict-opportunity pool recall is measured "
            "before downstream gain-ranker quality. No development/test, runtime E2E, "
            "full-train selection, fallback, or default activation."
        ),
        "source_authorization": {
            "stage190_protocol": stage190_fingerprint,
            "stage182_rerun_sources": reproduced_stage182["source_authorization"],
        },
        "frozen_protocol": formal_stage190["frozen_protocol"],
        "stage182_reproduction": reproduction,
        "rank_capped_safety_pool_nested_cv": nested_cv,
        "runtime": reproduced_stage182["runtime"],
        "resource_consumption": stage181._resource_summary(tracker.snapshots),
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "stage182_reproduction": round(reproduced_at - authorized_at, 6),
            "rank_capped_safety_pool_nested_cv": round(analyzed_at - reproduced_at, 6),
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
            "stage191_model_fit_count": nested_cv["execution"]["model_fit_count"],
            "stage191_comparable_pair_count": nested_cv["execution"][
                "comparable_pair_count_across_fits"
            ],
            "stage191_listwise_question_fit_count": nested_cv["execution"][
                "listwise_question_fit_count"
            ],
            "stage191_private_prediction_count": nested_cv["execution"]["private_prediction_count"],
            "stage191_public_pair_rows_written": nested_cv["execution"]["public_pair_rows_written"],
            "stage191_public_listwise_targets_written": nested_cv["execution"][
                "public_listwise_targets_written"
            ],
            "stage191_public_prediction_rows_written": nested_cv["execution"][
                "public_prediction_rows_written"
            ],
            "gold_used_only_for_training_targets_and_offline_evaluation": True,
            "all_comparable_pairs_retained_without_sampling": True,
            "all_listwise_actions_retained_without_sampling": True,
            "candidate_family_accepted": nested_cv["candidate_family_accepted"],
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
        "private_action_rows_persisted": False,
        "private_pool_rows_persisted": False,
        "private_pair_rows_persisted": False,
        "private_listwise_targets_persisted": False,
        "private_predictions_persisted": False,
    }
    report["process_guards"] = _process_guards(report=report, forbidden=forbidden)
    valid = all(row["passed"] for row in report["process_guards"])
    accepted = nested_cv["candidate_family_accepted"] if valid else False
    report["decision"] = {
        "status": (
            "stage191_rank_capped_safety_pool_candidate_family_found"
            if valid and accepted
            else "stage191_rank_capped_safety_pool_insufficient"
            if valid
            else "stage191_rank_capped_safety_pool_invalid"
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


def write_stage191_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage191Visualization, ...]:
    """Write and XML-validate aggregate Stage 191 charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cv = report["rank_capped_safety_pool_nested_cv"]
    aggregate = cv["aggregate"]
    pool = cv["aggregate_pool_metrics"]
    resources = report["resource_consumption"]
    charts = {
        "stage191_inner_eligible_configs.svg": _chart(
            "Stage 191 inner-eligible configurations by outer fold",
            tuple(
                BarDatum(fold_id, row["eligible_config_count"], str(row["eligible_config_count"]))
                for fold_id, row in cv["outer_folds"].items()
            ),
            "eligible configuration count",
        ),
        "stage191_outer_pool_recall.svg": _outer_pool_metric_chart(
            cv,
            title="Stage 191 outer strict-opportunity pool recall",
            metric="strict_opportunity_pool_recall",
            x_label="question-level strict-opportunity pool recall",
        ),
        "stage191_outer_pool_size.svg": _outer_pool_metric_chart(
            cv,
            title="Stage 191 outer mean candidate-pool size",
            metric="mean_pool_size",
            x_label="actions per question",
        ),
        "stage191_selected_pool_caps.svg": _chart(
            "Stage 191 selected pool caps",
            tuple(
                BarDatum(
                    cap,
                    cv["selected_pool_cap_counts"].get(cap, 0),
                    str(cv["selected_pool_cap_counts"].get(cap, 0)),
                )
                for cap in ("4", "8", "16", "all")
            ),
            "outer-fold selection count",
        ),
        "stage191_outer_citation_delta.svg": _outer_evaluation_chart(
            cv,
            title="Stage 191 outer gold-citation delta",
            metric="gold_citation_delta",
            x_label="gold-citation delta",
            decimals=0,
        ),
        "stage191_outer_f1_delta.svg": _outer_evaluation_chart(
            cv,
            title="Stage 191 outer mean F1 delta",
            metric="mean_f1_delta",
            x_label="mean answerable F1 delta",
            decimals=4,
        ),
        "stage191_outer_strict_success.svg": _outer_evaluation_chart(
            cv,
            title="Stage 191 outer strict-success actions",
            metric="strict_success_count",
            x_label="strict-success action count",
            decimals=0,
        ),
        "stage191_aggregate_outcomes.svg": _chart(
            "Stage 191 aggregate selected-action outcomes",
            tuple(
                BarDatum(name, value, str(value))
                for name, value in (
                    ("changed questions", aggregate["changed_question_count"]),
                    ("strict-success actions", aggregate["strict_success_count"]),
                    ("citation-gain actions", aggregate["citation_gain_action_count"]),
                    ("citation-loss actions", aggregate["citation_loss_action_count"]),
                    ("F1-regression actions", aggregate["f1_regression_action_count"]),
                )
            ),
            "question or selected-action count",
        ),
        "stage191_pool_summary.svg": _chart(
            "Stage 191 aggregate candidate-pool diagnostics",
            (
                BarDatum(
                    "strict-opportunity recall",
                    pool["strict_opportunity_pool_recall"],
                    f"{pool['strict_opportunity_pool_recall']:.3f}",
                ),
                BarDatum(
                    "strict-action retention",
                    pool["strict_action_retention_rate"],
                    f"{pool['strict_action_retention_rate']:.3f}",
                ),
                BarDatum(
                    "baseline inclusion",
                    pool["baseline_in_pool_rate"],
                    f"{pool['baseline_in_pool_rate']:.3f}",
                ),
            ),
            "rate",
        ),
        "stage191_prediction_auc.svg": _chart(
            "Stage 191 selected-bundle held-out ROC AUC",
            tuple(
                BarDatum(
                    target,
                    (cv["prediction_metrics"][target] or {}).get("roc_auc") or 0.0,
                    (
                        f"{cv['prediction_metrics'][target]['roc_auc']:.3f}"
                        if cv["prediction_metrics"][target]
                        and cv["prediction_metrics"][target]["roc_auc"] is not None
                        else "not available"
                    ),
                )
                for target in ("citation_loss", "f1_loss", "strict_gain")
            ),
            "ROC AUC",
        ),
        "stage191_advancement_gates.svg": _chart(
            "Stage 191 advancement gates",
            tuple(
                BarDatum(row["name"], float(row["passed"]), "pass" if row["passed"] else "fail")
                for row in cv["advancement_gates"]
            ),
            "1 means passed",
            margin_left=920,
        ),
        "stage191_execution_counts.svg": _chart(
            "Stage 191 execution counts",
            tuple(
                BarDatum(name, value, str(value))
                for name, value in (
                    ("model fits", cv["execution"]["model_fit_count"]),
                    ("policy configurations", cv["protocol"]["policy_config_count"]),
                    ("inner partitions", 20),
                    (
                        "outer refits",
                        sum(row["outer_evaluated"] for row in cv["outer_folds"].values()),
                    ),
                )
            ),
            "count",
        ),
        "stage191_pair_counts.svg": _chart(
            "Stage 191 pair construction across fits",
            (
                BarDatum(
                    "comparable pairs",
                    cv["execution"]["comparable_pair_count_across_fits"],
                    str(cv["execution"]["comparable_pair_count_across_fits"]),
                ),
                BarDatum(
                    "omitted incomparable pairs",
                    cv["execution"]["omitted_incomparable_pair_count_across_fits"],
                    str(cv["execution"]["omitted_incomparable_pair_count_across_fits"]),
                ),
            ),
            "unordered within-question pair count",
        ),
        "stage191_memory_gib.svg": _chart(
            "Stage 191 memory usage",
            (
                _gib_bar("peak working set", resources["process_peak_working_set_bytes"]),
                _gib_bar("peak private usage", resources["process_peak_private_usage_bytes"]),
                _gib_bar("minimum system free", resources["minimum_system_available_memory_bytes"]),
            ),
            "GiB",
        ),
        "stage191_process_guards.svg": _chart(
            "Stage 191 process guards",
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
        written.append(Stage191Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _authorize_stage190_protocol(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 190":
        raise ValueError("Stage191 requires the Stage190 protocol")
    decision = report.get("decision", {})
    if decision.get("status") != "stage190_rank_capped_safety_pool_protocol_frozen":
        raise ValueError("Stage191 requires the frozen Stage190 protocol")
    if decision.get("protocol_valid") is not True:
        raise ValueError("Stage191 requires a valid Stage190 protocol")
    if decision.get("stage191_train_only_experiment_authorized") is not True:
        raise ValueError("Stage190 did not authorize Stage191")
    if not all(row.get("passed") is True for row in report.get("guard_checks", [])):
        raise ValueError("Stage190 guard checks must all pass")
    protocol = report.get("frozen_protocol", {})
    if protocol.get("candidate_grid", {}).get("policy_config_count") != 32:
        raise ValueError("Stage190 policy grid drifted")
    if protocol.get("candidate_grid", {}).get("pool_caps") != [4, 8, 16, "all"]:
        raise ValueError("Stage190 pool caps drifted")
    if protocol.get("cross_validation", {}).get("maximum_model_fit_count") != 300:
        raise ValueError("Stage190 fit budget drifted")
    if protocol.get("action_contract", {}).get("fallback_enabled") is not False:
        raise ValueError("Stage190 fallback boundary drifted")
    recall = protocol.get("inner_selection", {}).get("pool_recall_constraints", {})
    if recall.get("aggregate_strict_opportunity_pool_recall_minimum") != 0.8:
        raise ValueError("Stage190 aggregate pool-recall gate drifted")
    if recall.get("per_fold_strict_opportunity_pool_recall_minimum") != 0.7:
        raise ValueError("Stage190 per-fold pool-recall gate drifted")
    if recall.get("folds_meeting_per_fold_minimum") != 3:
        raise ValueError("Stage190 pool-recall fold count drifted")


def _process_guards(
    *,
    report: Mapping[str, Any],
    forbidden: Sequence[str],
) -> list[dict[str, Any]]:
    boundaries = report["execution_boundaries"]
    reproduction = report["stage182_reproduction"]
    cv = report["rank_capped_safety_pool_nested_cv"]
    eligible_outer_folds = sum(row["outer_evaluated"] for row in cv["outer_folds"].values())
    expected_fits = 240 + 12 * eligible_outer_folds
    inner_pool_baseline_rates = [
        candidate["pool_metrics"]["baseline_in_pool_rate"]
        for row in cv["outer_folds"].values()
        for candidate in row["top_inner_candidates"]
    ]
    return [
        _gate("stage190_protocol_hash_matches", True),
        _gate("stage190_protocol_authorized", True),
        _gate("stage182_reproduction_passed", reproduction["passed"] is True),
        _gate("stage182_reproduction_check_count_is_10", len(reproduction["checks"]) == 10),
        _gate(
            "captured_action_row_count_is_12298", boundaries["captured_action_row_count"] == 12298
        ),
        _gate(
            "captured_stage182_selected_action_count_is_129",
            boundaries["captured_stage182_selected_action_count"] == 129,
        ),
        _gate("policy_config_count_is_32", cv["protocol"]["policy_config_count"] == 32),
        _gate("pool_caps_are_frozen", cv["protocol"]["pool_caps"] == [4, 8, 16, "all"]),
        _gate("advancement_gate_count_is_15", len(cv["advancement_gates"]) == 15),
        _gate("question_count_is_370", cv["dataset"]["question_count"] == 370),
        _gate("reference_action_count_is_370", cv["dataset"]["reference_action_count"] == 370),
        _gate(
            "reference_regression_count_is_55", cv["dataset"]["reference_regression_count"] == 55
        ),
        _gate(
            "model_fit_count_matches_completed_partitions",
            cv["execution"]["model_fit_count"] == expected_fits,
        ),
        _gate("model_fit_count_within_300", cv["execution"]["model_fit_count"] <= 300),
        _gate(
            "baseline_in_all_observed_pools",
            bool(inner_pool_baseline_rates)
            and all(value == 1.0 for value in inner_pool_baseline_rates),
        ),
        _gate(
            "all_comparable_pairs_retained",
            boundaries["stage191_comparable_pair_count"] > 0
            and boundaries["all_comparable_pairs_retained_without_sampling"] is True,
        ),
        _gate(
            "all_listwise_actions_retained",
            boundaries["stage191_listwise_question_fit_count"] > 0
            and boundaries["all_listwise_actions_retained_without_sampling"] is True,
        ),
        _gate(
            "private_training_rows_not_public",
            boundaries["stage191_public_pair_rows_written"] == 0
            and boundaries["stage191_public_listwise_targets_written"] == 0
            and boundaries["stage191_public_prediction_rows_written"] == 0,
        ),
        _gate("private_predictions_exist", boundaries["stage191_private_prediction_count"] > 0),
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


def _outer_evaluation_chart(
    cv: Mapping[str, Any],
    *,
    title: str,
    metric: str,
    x_label: str,
    decimals: int,
) -> str:
    bars = []
    for fold_id, row in cv["outer_folds"].items():
        evaluation = row["outer_evaluation"]
        value = evaluation[metric] if evaluation else 0
        label = f"{value:.{decimals}f}" if evaluation else "not run"
        bars.append(BarDatum(fold_id, value, label))
    return _chart(title, tuple(bars), x_label)


def _outer_pool_metric_chart(
    cv: Mapping[str, Any],
    *,
    title: str,
    metric: str,
    x_label: str,
) -> str:
    bars = []
    for fold_id, row in cv["outer_folds"].items():
        pool = row["outer_pool_metrics"]
        value = pool[metric] if pool else 0
        label = f"{value:.3f}" if pool else "not run"
        bars.append(BarDatum(fold_id, value, label))
    return _chart(title, tuple(bars), x_label)


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
