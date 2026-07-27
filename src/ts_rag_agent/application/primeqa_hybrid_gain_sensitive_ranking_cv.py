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
from ts_rag_agent.application.composition_gain_sensitive_ranking import (
    run_gain_sensitive_nested_cv,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

STAGE = "Stage 188"
CREATED_AT = "2026-07-23"
ANALYSIS_ID = "primeqa_hybrid_gain_sensitive_ranking_nested_cv_v1"
STAGE187_SHA256 = "b6125e28f532774dd2137374f6a236520f71e247c774eca1e4d8c078f31e21b2"
FORBIDDEN_PUBLIC_KEYS = stage183._FORBIDDEN_PUBLIC_KEYS | {
    "candidate_actions",
    "citation_loss_probability",
    "f1_loss_probability",
    "feature_rows",
    "gain_score",
    "listwise_targets",
    "pair_rows",
    "predictions",
    "selected_actions",
}


@dataclass(frozen=True)
class Stage188Visualization:
    """One aggregate Stage 188 visualization."""

    name: str
    path: str


def run_stage188_gain_sensitive_ranking_cv(
    *,
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
    progress_sink: stage182.ProgressSink | None = None,
) -> dict[str, Any]:
    """Reproduce Stage 182 and run the authorized Stage 188 train nested CV."""

    started_at = time.perf_counter()
    stage187_fingerprint = stage173._resolved_fingerprint(stage187_protocol_path)
    if stage187_fingerprint["sha256"] != STAGE187_SHA256:
        raise ValueError("Stage188 Stage187 protocol hash mismatch")
    formal_stage187 = _load_json(stage187_protocol_path)
    _authorize_stage187_protocol(formal_stage187)
    formal_stage182 = _load_json(stage182_report_path)
    stage183._authorize_stage182_report(formal_stage182)
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="stage187_protocol_authorized")

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
        raise ValueError("Stage188 did not reproduce the formal Stage182 result")
    gc.collect()
    tracker.capture("stage182_temporary_resources_released")

    nested_cv = run_gain_sensitive_nested_cv(
        action_rows=private["action_rows"],
        stage182_selected_actions=private["selected_actions"],
        progress_sink=progress_sink,
    )
    tracker.capture("gain_sensitive_nested_cv_complete")
    analyzed_at = time.perf_counter()
    report: dict[str, Any] = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "analysis_id": ANALYSIS_ID,
        "analysis_scope": (
            "Train-only five-by-four nested cross-validation of 32 gain-sensitive "
            "within-question ranking configurations with all comparable pairs and "
            "complete ListNet lists. No development/test, runtime E2E, full-train "
            "selection, fallback, or default activation."
        ),
        "user_confirmed_implementation_choice": {
            "choice": "A",
            "listnet_scaler": "StandardScaler(with_mean=False)",
            "patience_improvement_tolerance": 1e-12,
            "changes_candidate_grid_or_fit_budget": False,
        },
        "source_authorization": {
            "stage187_protocol": stage187_fingerprint,
            "stage182_rerun_sources": reproduced_stage182["source_authorization"],
        },
        "frozen_protocol": formal_stage187["frozen_protocol"],
        "stage182_reproduction": reproduction,
        "gain_sensitive_nested_cv": nested_cv,
        "runtime": reproduced_stage182["runtime"],
        "resource_consumption": stage181._resource_summary(tracker.snapshots),
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "stage182_reproduction": round(reproduced_at - authorized_at, 6),
            "gain_sensitive_nested_cv": round(analyzed_at - reproduced_at, 6),
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
            "stage188_model_fit_count": nested_cv["execution"]["model_fit_count"],
            "stage188_comparable_pair_count": nested_cv["execution"][
                "comparable_pair_count_across_fits"
            ],
            "stage188_omitted_incomparable_pair_count": nested_cv["execution"][
                "omitted_incomparable_pair_count_across_fits"
            ],
            "stage188_listwise_question_fit_count": nested_cv["execution"][
                "listwise_question_fit_count"
            ],
            "stage188_private_prediction_count": nested_cv["execution"]["private_prediction_count"],
            "stage188_public_pair_rows_written": nested_cv["execution"]["public_pair_rows_written"],
            "stage188_public_listwise_targets_written": nested_cv["execution"][
                "public_listwise_targets_written"
            ],
            "stage188_public_prediction_rows_written": nested_cv["execution"][
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
        "private_pair_rows_persisted": False,
        "private_listwise_targets_persisted": False,
        "private_predictions_persisted": False,
    }
    report["process_guards"] = _process_guards(report=report, forbidden=forbidden)
    valid = all(row["passed"] for row in report["process_guards"])
    accepted = nested_cv["candidate_family_accepted"] if valid else False
    report["decision"] = {
        "status": (
            "stage188_gain_sensitive_ranking_candidate_family_found"
            if valid and accepted
            else "stage188_gain_sensitive_ranking_insufficient"
            if valid
            else "stage188_gain_sensitive_ranking_invalid"
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


def write_stage188_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage188Visualization, ...]:
    """Write and XML-validate aggregate Stage 188 charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cv = report["gain_sensitive_nested_cv"]
    aggregate = cv["aggregate"]
    metrics = cv["prediction_metrics"]
    resources = report["resource_consumption"]
    charts = {
        "stage188_inner_eligible_configs.svg": _chart(
            "Stage 188 inner-eligible configurations by outer fold",
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
        "stage188_outer_citation_delta.svg": _outer_metric_chart(
            cv,
            title="Stage 188 outer-fold gold-citation delta",
            metric="gold_citation_delta",
            x_label="gold-citation delta",
            decimals=0,
        ),
        "stage188_outer_f1_delta.svg": _outer_metric_chart(
            cv,
            title="Stage 188 outer-fold mean F1 delta",
            metric="mean_f1_delta",
            x_label="mean answerable F1 delta",
            decimals=4,
        ),
        "stage188_outer_strict_success.svg": _outer_metric_chart(
            cv,
            title="Stage 188 outer-fold strict-success actions",
            metric="strict_success_count",
            x_label="strict-success action count",
            decimals=0,
        ),
        "stage188_aggregate_outcomes.svg": _chart(
            "Stage 188 aggregate selected-action outcomes",
            tuple(
                BarDatum(name, value, str(value))
                for name, value in (
                    ("changed questions", aggregate["changed_question_count"]),
                    ("strict-success actions", aggregate["strict_success_count"]),
                    ("citation-gain actions", aggregate["citation_gain_action_count"]),
                    ("citation-loss actions", aggregate["citation_loss_action_count"]),
                    ("F1-regression actions", aggregate["f1_regression_action_count"]),
                    (
                        "repaired Stage 182 regressions",
                        aggregate["repaired_reference_regression_count"],
                    ),
                )
            ),
            "question or selected-action count",
        ),
        "stage188_prediction_auc.svg": _chart(
            "Stage 188 selected-bundle held-out ROC AUC",
            tuple(
                BarDatum(
                    target,
                    (metrics[target] or {}).get("roc_auc") or 0.0,
                    (
                        f"{metrics[target]['roc_auc']:.3f}"
                        if metrics[target] and metrics[target]["roc_auc"] is not None
                        else "not available"
                    ),
                )
                for target in ("citation_loss", "f1_loss", "strict_gain")
            ),
            "ROC AUC",
        ),
        "stage188_selected_rankers.svg": _chart(
            "Stage 188 selected gain rankers",
            tuple(
                BarDatum(name, value, str(value))
                for name, value in (
                    (
                        "pairwise_pareto_logistic",
                        cv["selected_ranker_counts"].get("pairwise_pareto_logistic", 0),
                    ),
                    (
                        "linear_listnet_top_frontier",
                        cv["selected_ranker_counts"].get("linear_listnet_top_frontier", 0),
                    ),
                )
            ),
            "outer-fold selection count",
        ),
        "stage188_advancement_gates.svg": _chart(
            "Stage 188 advancement gates",
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
        "stage188_execution_counts.svg": _chart(
            "Stage 188 execution counts",
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
        "stage188_pair_counts.svg": _chart(
            "Stage 188 pair construction across fits",
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
        "stage188_memory_gib.svg": _chart(
            "Stage 188 memory usage",
            (
                _gib_bar(
                    "peak working set",
                    resources["process_peak_working_set_bytes"],
                ),
                _gib_bar(
                    "peak private usage",
                    resources["process_peak_private_usage_bytes"],
                ),
                _gib_bar(
                    "minimum system free",
                    resources["minimum_system_available_memory_bytes"],
                ),
            ),
            "GiB",
        ),
        "stage188_process_guards.svg": _chart(
            "Stage 188 process guards",
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
        written.append(Stage188Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _authorize_stage187_protocol(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 187":
        raise ValueError("Stage188 requires the Stage187 protocol")
    decision = report.get("decision", {})
    if decision.get("status") != "stage187_gain_sensitive_ranking_protocol_frozen":
        raise ValueError("Stage188 requires the frozen Stage187 protocol")
    if decision.get("protocol_valid") is not True:
        raise ValueError("Stage188 requires a valid Stage187 protocol")
    if decision.get("stage188_train_only_experiment_authorized") is not True:
        raise ValueError("Stage187 did not authorize Stage188")
    if not all(row.get("passed") is True for row in report.get("guard_checks", [])):
        raise ValueError("Stage187 guard checks must all pass")
    protocol = report.get("frozen_protocol", {})
    if protocol.get("candidate_grid", {}).get("policy_config_count") != 32:
        raise ValueError("Stage187 policy grid drifted")
    if protocol.get("cross_validation", {}).get("maximum_model_fit_count") != 300:
        raise ValueError("Stage187 fit budget drifted")
    if protocol.get("action_contract", {}).get("fallback_enabled") is not False:
        raise ValueError("Stage187 fallback boundary drifted")
    if (
        protocol.get("gain_ranker_contract", {})
        .get("pairwise_pareto_logistic", {})
        .get("pair_sampling")
        is not False
    ):
        raise ValueError("Stage187 pair sampling boundary drifted")


def _process_guards(
    *,
    report: Mapping[str, Any],
    forbidden: Sequence[str],
) -> list[dict[str, Any]]:
    boundaries = report["execution_boundaries"]
    reproduction = report["stage182_reproduction"]
    cv = report["gain_sensitive_nested_cv"]
    eligible_outer_folds = sum(row["outer_evaluated"] for row in cv["outer_folds"].values())
    expected_fits = 240 + 12 * eligible_outer_folds
    return [
        _gate("stage187_protocol_hash_matches", True),
        _gate("stage187_protocol_authorized", True),
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
        _gate("policy_config_count_is_32", cv["protocol"]["policy_config_count"] == 32),
        _gate("question_count_is_370", cv["dataset"]["question_count"] == 370),
        _gate(
            "reference_action_count_is_370",
            cv["dataset"]["reference_action_count"] == 370,
        ),
        _gate(
            "reference_regression_count_is_55",
            cv["dataset"]["reference_regression_count"] == 55,
        ),
        _gate(
            "model_fit_count_matches_completed_partitions",
            cv["execution"]["model_fit_count"] == expected_fits,
        ),
        _gate(
            "model_fit_count_within_300",
            cv["execution"]["model_fit_count"] <= 300,
        ),
        _gate(
            "all_comparable_pairs_retained",
            boundaries["stage188_comparable_pair_count"] > 0
            and boundaries["all_comparable_pairs_retained_without_sampling"] is True,
        ),
        _gate(
            "all_listwise_actions_retained",
            boundaries["stage188_listwise_question_fit_count"] > 0
            and boundaries["all_listwise_actions_retained_without_sampling"] is True,
        ),
        _gate(
            "private_training_rows_not_public",
            boundaries["stage188_public_pair_rows_written"] == 0
            and boundaries["stage188_public_listwise_targets_written"] == 0
            and boundaries["stage188_public_prediction_rows_written"] == 0,
        ),
        _gate(
            "private_predictions_exist",
            boundaries["stage188_private_prediction_count"] > 0,
        ),
        _gate(
            "user_confirmed_listnet_scaling_choice_a",
            report["user_confirmed_implementation_choice"]["choice"] == "A"
            and cv["protocol"]["user_confirmed_scaling_choice"] == "A",
        ),
        _gate("train_loaded", boundaries["train_loaded"] is True),
        _gate("development_closed", boundaries["development_loaded"] is False),
        _gate("test_closed", boundaries["test_loaded"] is False),
        _gate(
            "gold_offline_only",
            boundaries["gold_used_only_for_training_targets_and_offline_evaluation"] is True,
        ),
        _gate(
            "full_train_policy_not_selected",
            boundaries["full_train_policy_selected"] is False,
        ),
        _gate(
            "replacement_policy_not_selected",
            boundaries["replacement_policy_selected"] is False,
        ),
        _gate("runtime_e2e_not_run", boundaries["runtime_e2e_run"] is False),
        _gate(
            "default_runtime_unchanged",
            boundaries["runtime_registered_as_default"] is False,
        ),
        _gate("stage178b_not_run", boundaries["stage178b_run"] is False),
        _gate("no_retry", boundaries["retry_action_count"] == 0),
        _gate("no_fallback", boundaries["fallback_action_count"] == 0),
        _gate("public_report_safe", not forbidden),
    ]


def _outer_metric_chart(
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
