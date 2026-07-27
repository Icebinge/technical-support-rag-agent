from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

STAGE = "Stage 190"
CREATED_AT = "2026-07-27"
PROTOCOL_ID = "primeqa_hybrid_rank_capped_safety_pool_protocol_v1"
NEXT_STAGE = "Stage 191"
STAGE189_SHA256 = "48af548168e4e40972c4082fc24bec822ce264427f12c56b98a8d0966df2e5a0"
FEATURE_REPRESENTATIONS = ("raw_runtime", "question_relative_runtime")
SAFETY_ESTIMATORS = ("class_balanced_logistic", "histogram_gradient_boosting")
SAFETY_TARGETS = ("citation_loss", "f1_loss")
GAIN_RANKERS = ("pairwise_pareto_logistic", "linear_listnet_top_frontier")
POOL_CAPS: tuple[int | str, ...] = (4, 8, 16, "all")
OUTER_FOLD_COUNT = 5
INNER_FOLD_COUNT = 4
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "action_id",
        "answer",
        "answer_doc_id",
        "candidate_actions",
        "citation_delta",
        "document_id",
        "document_text",
        "f1_delta",
        "feature_rows",
        "gold_answer",
        "gold_document_id",
        "outcome_class",
        "predictions",
        "question_id",
        "question_key",
        "question_text",
        "strict_expected",
    }
)


@dataclass(frozen=True)
class Stage190Visualization:
    """One public-safe Stage 190 protocol visualization."""

    name: str
    path: str


def freeze_rank_capped_safety_pool_protocol(
    *,
    stage189_report_path: Path,
    user_confirmed: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage 191 train-only rank-capped safety-pool protocol."""

    started_at = time.perf_counter()
    stage189_report = _load_json_object(stage189_report_path)
    source_file = _fingerprint(stage189_report_path)
    loaded_at = time.perf_counter()
    evidence = _evidence_summary(stage189_report)
    protocol = _frozen_protocol()
    preliminary = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "protocol_id": PROTOCOL_ID,
        "protocol_scope": (
            "Aggregate-only freeze for a train-only nested-CV experiment that replaces "
            "Stage188's narrow relative-risk margin with a rank-capped safety pool. "
            "This stage loads no split rows or documents, fits no model, evaluates no "
            "policy, keeps development and test closed, adds no fallback, and changes "
            "no runtime default."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed),
            "confirmation_note": confirmation_note,
        },
        "source_file": source_file,
        "evidence_summary": evidence,
        "frozen_protocol": protocol,
        "execution_boundaries": {
            "loaded_public_reports_only": True,
            "train_rows_loaded": False,
            "development_loaded": False,
            "test_loaded": False,
            "model_fit_count": 0,
            "pair_rows_materialized": 0,
            "listwise_questions_materialized": 0,
            "policy_evaluation_run": False,
            "replacement_policy_selected": False,
            "runtime_e2e_run": False,
            "runtime_registered_as_default": False,
            "stage178b_run": False,
            "retry_action_count": 0,
            "fallback_action_count": 0,
        },
    }
    guard_checks = _guard_checks(preliminary, stage189_report=stage189_report)
    checked_at = time.perf_counter()
    decision = _decision(guard_checks)
    report = {
        **preliminary,
        "guard_checks": guard_checks,
        "decision": decision,
        "timing_seconds": {
            "load_public_report": round(loaded_at - started_at, 6),
            "freeze_and_guard": round(checked_at - loaded_at, 6),
            "total": round(checked_at - started_at, 6),
        },
    }
    forbidden = sorted(_forbidden_keys_found(report))
    return {
        **report,
        "public_safe_contract": {
            "forbidden_public_keys": sorted(FORBIDDEN_PUBLIC_KEYS),
            "forbidden_keys_found": forbidden,
            "private_action_rows_persisted": False,
            "private_pair_rows_persisted": False,
            "private_listwise_targets_persisted": False,
            "private_predictions_persisted": False,
            "public_report_safe": not forbidden,
        },
    }


def _frozen_protocol() -> dict[str, Any]:
    policy_config_count = (
        len(FEATURE_REPRESENTATIONS) * len(SAFETY_ESTIMATORS) * len(GAIN_RANKERS) * len(POOL_CAPS)
    )
    fits_per_representation = len(SAFETY_ESTIMATORS) * len(SAFETY_TARGETS) + len(GAIN_RANKERS)
    fits_per_partition = len(FEATURE_REPRESENTATIONS) * fits_per_representation
    inner_partition_count = OUTER_FOLD_COUNT * INNER_FOLD_COUNT
    outer_refit_count = OUTER_FOLD_COUNT
    return {
        "experiment_name": "train_only_rank_capped_safety_pool_nested_cv",
        "split_contract": {
            "selection_split": "train",
            "frozen_question_grouped_outer_folds": OUTER_FOLD_COUNT,
            "inner_folds_per_outer_fold": INNER_FOLD_COUNT,
            "development_opened": False,
            "test_opened": False,
            "all_actions_for_one_question_remain_in_one_fold": True,
        },
        "action_contract": {
            "candidate_set": (
                "all unique runtime-generatable Stage181 actions including the unique "
                "original baseline action"
            ),
            "baseline_always_unioned_into_pool": True,
            "empty_pool_possible": False,
            "fallback_enabled": False,
            "deterministic_tie_break": "canonical runtime action generation order",
        },
        "model_contract": {
            "feature_representations": list(FEATURE_REPRESENTATIONS),
            "safety_estimators": list(SAFETY_ESTIMATORS),
            "safety_targets": list(SAFETY_TARGETS),
            "gain_rankers": list(GAIN_RANKERS),
            "stage188_features_models_labels_and_hyperparameters_reused_exactly": True,
            "all_comparable_pairs_retained_without_sampling": True,
            "all_listwise_actions_retained_without_sampling": True,
            "gold_outcomes_available_to_runtime": False,
        },
        "rank_capped_safety_pool": {
            "joint_safety_risk": "max(p(citation_loss), p(f1_loss))",
            "ordering": [
                "ascending joint safety risk",
                "ascending p(citation_loss) + p(f1_loss)",
                "canonical runtime action order",
            ],
            "pool_caps": list(POOL_CAPS),
            "pool_rule": (
                "take the first cap actions under safety ordering, or every action for "
                "the all cap, then union the unique original baseline"
            ),
            "selection_inside_pool": [
                "maximize the frozen Stage188 gain-ranker score",
                "minimize joint safety risk",
                "canonical runtime action order",
            ],
            "runtime_gold_filter_used": False,
            "fallback_branch_used": False,
        },
        "candidate_grid": {
            "feature_representations": list(FEATURE_REPRESENTATIONS),
            "safety_estimators": list(SAFETY_ESTIMATORS),
            "gain_rankers": list(GAIN_RANKERS),
            "pool_caps": list(POOL_CAPS),
            "policy_config_count": policy_config_count,
            "safety_predictions_shared_across_rankers_and_caps": True,
            "gain_scores_shared_across_safety_estimators_and_caps": True,
        },
        "cross_validation": {
            "outer_fold_count": OUTER_FOLD_COUNT,
            "inner_fold_count": INNER_FOLD_COUNT,
            "inner_partition_count": inner_partition_count,
            "outer_refit_count": outer_refit_count,
            "model_fits_per_representation_per_partition": fits_per_representation,
            "model_fits_per_partition": fits_per_partition,
            "maximum_model_fit_count": (
                (inner_partition_count + outer_refit_count) * fits_per_partition
            ),
            "inner_selection_uses_only_inner_oof_predictions": True,
            "outer_fold_used_once_after_inner_selection": True,
            "no_inner_eligible_config_behavior": (
                "record no-eligible and do not evaluate a weaker outer configuration"
            ),
            "no_retry": True,
            "no_fallback": True,
        },
        "inner_selection": {
            "pool_recall_constraints": {
                "aggregate_strict_opportunity_pool_recall_minimum": 0.80,
                "per_fold_strict_opportunity_pool_recall_minimum": 0.70,
                "folds_meeting_per_fold_minimum": 3,
                "gold_used_for_offline_eligibility_only": True,
            },
            "existing_stage188_constraints_reused": [
                "aggregate citation delta >= 0",
                "aggregate mean F1 delta >= 0",
                "citation nonregression in at least 3 of 4 inner folds",
                "F1 nonregression in at least 3 of 4 inner folds",
                "changed-question count >= 10% of inner questions",
                "strict-success count >= 8% of inner questions",
                "strict-success precision >= 0.60",
            ],
            "lexicographic_objective": [
                "maximize strict-success count",
                "maximize strict-success precision",
                "maximize strict-opportunity pool recall",
                "minimize F1-regression action count",
                "minimize citation-loss action count",
                "maximize gold-citation delta",
                "maximize mean F1 delta",
                "maximize repaired Stage182 regressions",
                "deterministic candidate name",
            ],
            "weaker_ineligible_candidate_substitution": False,
        },
        "advancement_gates": _advancement_gates(),
        "resource_contract": {
            "sparse_pair_construction_reused": True,
            "dense_histogram_matrix_released_before_pair_construction": True,
            "gpu_required": False,
            "insufficient_memory_behavior": (
                "do not start; request resource clearance instead of reducing protocol"
            ),
            "process_monitoring": (
                "one PowerShell Wait-Process call for the formal PID until natural exit"
            ),
        },
        "authorization_boundary": {
            "stage191_train_only_experiment_may_run_if_protocol_guards_pass": True,
            "development_evaluation_authorized": False,
            "test_evaluation_authorized": False,
            "runtime_e2e_authorized": False,
            "full_train_policy_selection_authorized": False,
            "replacement_policy_selection_authorized": False,
            "default_runtime_activation_authorized": False,
            "stage178b_authorized": False,
        },
    }


def _advancement_gates() -> list[dict[str, Any]]:
    return [
        _threshold("outer_folds_with_inner_eligible_config", ">=", 5, "count"),
        _threshold("gold_citation_delta", ">=", 5, "count"),
        _threshold("mean_f1_delta", ">=", 0.005249, "rate"),
        _threshold("citation_bootstrap_ci95_lower", ">=", 0.0, "count"),
        _threshold("f1_bootstrap_ci95_lower", ">=", 0.0, "rate"),
        _threshold("citation_nonregressing_outer_folds", ">=", 4, "count"),
        _threshold("f1_nonregressing_outer_folds", ">=", 4, "count"),
        _threshold("strict_success_count", ">=", 37, "count"),
        _threshold("strict_success_precision", ">=", 0.65, "rate"),
        _threshold("citation_loss_action_count", "<=", 4, "count"),
        _threshold("f1_regression_action_count", "<=", 27, "count"),
        _threshold("stage182_regression_repair_rate", ">=", 0.50, "rate"),
        _threshold("new_f1_regression_rate", "<=", 0.02, "rate"),
        _threshold("changed_question_count", ">=", 37, "count"),
        _threshold("strict_opportunity_pool_recall", ">=", 0.80, "rate"),
    ]


def _evidence_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    attribution = report["gain_sensitive_failure_attribution"]
    trajectory = attribution["top_ineligible_trajectory"]
    partition = trajectory["opportunity_partition"]
    margin = attribution["factor_aggregates"]["safety_frontier_margin"]
    ranker = attribution["factor_aggregates"]["gain_ranker"]
    best = attribution["family_best_configurations"]
    return {
        "primary_bottleneck": report["decision"]["primary_bottleneck"],
        "question_context_count": trajectory["question_context_count"],
        "strict_opportunity_context_count": trajectory["strict_opportunity_context_count"],
        "frontier_exclusion_context_count": partition["frontier_exclusion_context_count"],
        "ranker_miss_context_count": partition["ranker_miss_context_count"],
        "strict_selected_context_count": partition["strict_selected_context_count"],
        "opportunity_partition_exact": partition["partition_exact"],
        "frontier_strict_question_recall": trajectory["frontier_strict_question_recall"],
        "conditional_ranker_strict_capture": trajectory["conditional_ranker_strict_capture"],
        "strict_action_retention_rate": trajectory["strict_action_retention_rate"],
        "mean_frontier_size": trajectory["mean_frontier_size"],
        "filter_harm_context_count": trajectory["filter_harm_context_count"],
        "filter_rescue_context_count": trajectory["filter_rescue_context_count"],
        "margin_frontier_recall": {
            name: row["frontier_strict_question_recall"] for name, row in margin.items()
        },
        "ranker_conditional_capture": {
            name: row["conditional_ranker_strict_capture"] for name, row in ranker.items()
        },
        "best_frontier_recall_configuration": best["frontier_strict_question_recall"],
        "private_prediction_count_consumed": attribution["execution"][
            "private_bundle_prediction_count_consumed"
        ],
        "new_attribution_model_fit_count": attribution["execution"]["new_model_fit_count"],
    }


def _guard_checks(
    report: Mapping[str, Any],
    *,
    stage189_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evidence = report["evidence_summary"]
    protocol = report["frozen_protocol"]
    boundaries = report["execution_boundaries"]
    source = report["source_file"]
    source_decision = stage189_report["decision"]
    return [
        _gate("user_confirmed", report["user_confirmation"]["confirmed"] is True),
        _gate("stage189_hash_matches", source["sha256"] == STAGE189_SHA256),
        _gate(
            "stage189_status_complete",
            source_decision["status"] == "stage189_gain_sensitive_failure_attribution_complete",
        ),
        _gate("stage189_diagnostic_complete", source_decision["diagnostic_complete"] is True),
        _gate(
            "stage189_primary_bottleneck_is_frontier",
            evidence["primary_bottleneck"] == "safety_frontier_exclusion",
        ),
        _gate("opportunity_count_is_1456", evidence["strict_opportunity_context_count"] == 1456),
        _gate(
            "frontier_exclusion_count_is_1280", evidence["frontier_exclusion_context_count"] == 1280
        ),
        _gate("ranker_miss_count_is_27", evidence["ranker_miss_context_count"] == 27),
        _gate("strict_selected_count_is_149", evidence["strict_selected_context_count"] == 149),
        _gate("opportunity_partition_exact", evidence["opportunity_partition_exact"] is True),
        _gate("frontier_recall_matches", evidence["frontier_strict_question_recall"] == 0.120879),
        _gate(
            "conditional_ranker_capture_matches",
            evidence["conditional_ranker_strict_capture"] == 0.846591,
        ),
        _gate(
            "filter_harm_dominates_rescue",
            evidence["filter_harm_context_count"] > evidence["filter_rescue_context_count"],
        ),
        _gate(
            "largest_margin_has_highest_recall",
            evidence["margin_frontier_recall"]["0.10"]
            == max(evidence["margin_frontier_recall"].values()),
        ),
        _gate(
            "pairwise_conditional_capture_exceeds_listnet",
            evidence["ranker_conditional_capture"]["pairwise_pareto_logistic"]
            > evidence["ranker_conditional_capture"]["linear_listnet_top_frontier"],
        ),
        _gate("policy_config_count_is_32", protocol["candidate_grid"]["policy_config_count"] == 32),
        _gate("pool_caps_frozen", protocol["candidate_grid"]["pool_caps"] == [4, 8, 16, "all"]),
        _gate(
            "baseline_always_in_pool",
            protocol["action_contract"]["baseline_always_unioned_into_pool"] is True,
        ),
        _gate(
            "aggregate_pool_recall_gate_is_0_80",
            protocol["inner_selection"]["pool_recall_constraints"][
                "aggregate_strict_opportunity_pool_recall_minimum"
            ]
            == 0.80,
        ),
        _gate(
            "maximum_fit_count_is_300",
            protocol["cross_validation"]["maximum_model_fit_count"] == 300,
        ),
        _gate("advancement_gate_count_is_15", len(protocol["advancement_gates"]) == 15),
        _gate(
            "no_pair_sampling",
            protocol["model_contract"]["all_comparable_pairs_retained_without_sampling"] is True,
        ),
        _gate(
            "no_list_sampling",
            protocol["model_contract"]["all_listwise_actions_retained_without_sampling"] is True,
        ),
        _gate("public_reports_only", boundaries["loaded_public_reports_only"] is True),
        _gate("train_rows_not_loaded", boundaries["train_rows_loaded"] is False),
        _gate("development_closed", boundaries["development_loaded"] is False),
        _gate("test_closed", boundaries["test_loaded"] is False),
        _gate("no_model_fit", boundaries["model_fit_count"] == 0),
        _gate("no_policy_evaluation", boundaries["policy_evaluation_run"] is False),
        _gate("no_runtime_e2e", boundaries["runtime_e2e_run"] is False),
        _gate("default_runtime_unchanged", boundaries["runtime_registered_as_default"] is False),
        _gate("stage178b_not_run", boundaries["stage178b_run"] is False),
        _gate("no_retry", boundaries["retry_action_count"] == 0),
        _gate("no_fallback", boundaries["fallback_action_count"] == 0),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = all(row["passed"] for row in guard_checks)
    return {
        "status": (
            "stage190_rank_capped_safety_pool_protocol_frozen"
            if valid
            else "stage190_rank_capped_safety_pool_protocol_invalid"
        ),
        "protocol_valid": valid,
        "stage191_train_only_experiment_authorized": valid,
        "development_opened": False,
        "test_opened": False,
        "runtime_e2e_authorized": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "default_runtime_activation": False,
    }


def write_stage190_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage190Visualization, ...]:
    """Write and XML-validate aggregate Stage 190 protocol charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = report["evidence_summary"]
    protocol = report["frozen_protocol"]
    charts = {
        "stage190_opportunity_partition.svg": _chart(
            "Stage 190 source opportunity partition",
            tuple(
                _count_bar(name, value)
                for name, value in (
                    ("frontier exclusion", evidence["frontier_exclusion_context_count"]),
                    ("ranker miss", evidence["ranker_miss_context_count"]),
                    ("strict selected", evidence["strict_selected_context_count"]),
                )
            ),
            "question-context count",
        ),
        "stage190_source_rates.svg": _chart(
            "Stage 190 source diagnostic rates",
            tuple(
                BarDatum(name, value, f"{value:.3f}")
                for name, value in (
                    (
                        "frontier strict-question recall",
                        evidence["frontier_strict_question_recall"],
                    ),
                    ("conditional ranker capture", evidence["conditional_ranker_strict_capture"]),
                    ("strict action retention", evidence["strict_action_retention_rate"]),
                )
            ),
            "rate",
            margin_left=360,
        ),
        "stage190_margin_recall.svg": _chart(
            "Stage 189 frontier recall by frozen margin",
            tuple(
                BarDatum(name, value, f"{value:.3f}")
                for name, value in evidence["margin_frontier_recall"].items()
            ),
            "strict-question recall",
        ),
        "stage190_candidate_grid.svg": _chart(
            "Stage 191 frozen candidate grid",
            tuple(
                _count_bar(name, value)
                for name, value in (
                    ("feature representations", len(FEATURE_REPRESENTATIONS)),
                    ("safety estimators", len(SAFETY_ESTIMATORS)),
                    ("gain rankers", len(GAIN_RANKERS)),
                    ("pool caps", len(POOL_CAPS)),
                    ("policy configurations", protocol["candidate_grid"]["policy_config_count"]),
                )
            ),
            "count",
        ),
        "stage190_fit_budget.svg": _chart(
            "Stage 191 frozen fit budget",
            tuple(
                _count_bar(name, value)
                for name, value in (
                    ("inner partitions", protocol["cross_validation"]["inner_partition_count"]),
                    ("outer refits", protocol["cross_validation"]["outer_refit_count"]),
                    (
                        "fits per partition",
                        protocol["cross_validation"]["model_fits_per_partition"],
                    ),
                    ("maximum model fits", protocol["cross_validation"]["maximum_model_fit_count"]),
                )
            ),
            "count",
        ),
        "stage190_pool_recall_gates.svg": _chart(
            "Stage 191 frozen pool-recall eligibility",
            (
                BarDatum("aggregate minimum", 0.80, "0.80"),
                BarDatum("per-fold minimum", 0.70, "0.70"),
                BarDatum("folds required", 3.0 / 4.0, "3 / 4"),
            ),
            "rate or fold fraction",
        ),
        "stage190_decision_flags.svg": _chart(
            "Stage 190 protocol decision flags",
            tuple(
                BarDatum(name, float(value), "true" if value else "false")
                for name, value in report["decision"].items()
                if isinstance(value, bool)
            ),
            "1 means true",
            margin_left=760,
        ),
        "stage190_guard_checks.svg": _chart(
            "Stage 190 protocol guard checks",
            tuple(
                BarDatum(row["name"], float(row["passed"]), "pass" if row["passed"] else "fail")
                for row in report["guard_checks"]
            ),
            "1 means passed",
            margin_left=900,
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage190Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _threshold(name: str, operator: str, threshold: int | float, unit: str) -> dict[str, Any]:
    return {"name": name, "operator": operator, "threshold": threshold, "unit": unit}


def _count_bar(name: str, value: int) -> BarDatum:
    return BarDatum(name, value, str(value))


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


def _fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "byte_size": resolved.stat().st_size,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _load_json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object at {path}")
    return value


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


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}
