from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from sklearn.metrics import roc_auc_score

from ts_rag_agent.application.composition_action_audit import ActionAuditRow
from ts_rag_agent.application.composition_dual_target_policy import (
    DualTargetPolicySpec,
    DualTargetPrediction,
    SelectedAction,
    dual_target_utility,
    stage182_policy_specs,
)

_F1_TOLERANCE = 1e-12
_RISK_AUC_WEAK_THRESHOLD = 0.65
_SELECTED_REGRESSION_RATE_HIGH_THRESHOLD = 0.25
_SAFE_ALTERNATIVE_HEADROOM_THRESHOLD = 0.50
_RISK_BINS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0))


def run_f1_risk_attribution(
    *,
    action_rows: Sequence[ActionAuditRow],
    selected_actions: Sequence[SelectedAction],
    outer_predictions: Sequence[DualTargetPrediction],
    outer_fold_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Explain Stage 182 F1 regressions without selecting a replacement policy."""

    selected = tuple(selected_actions)
    predictions = tuple(outer_predictions)
    if len({row.row.question_key for row in selected}) != len(selected):
        raise ValueError("selected actions must be unique by question")
    selected_prediction_index = {
        (prediction.row.question_key, prediction.row.action.action_id): prediction
        for prediction in predictions
    }
    if len(selected_prediction_index) != len(predictions):
        raise ValueError("outer predictions must be unique by question and action")
    if any(
        (row.row.question_key, row.row.action.action_id) not in selected_prediction_index
        for row in selected
    ):
        raise ValueError("every selected action requires its outer prediction")

    rows_by_question: dict[str, list[ActionAuditRow]] = defaultdict(list)
    for row in action_rows:
        if row.action.family != "baseline":
            rows_by_question[row.question_key].append(row)
    predictions_by_question: dict[str, list[DualTargetPrediction]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_question[prediction.row.question_key].append(prediction)

    specs = {spec.name: spec for spec in stage182_policy_specs()}
    spec_by_fold = _selected_specs_by_fold(outer_fold_reports, specs=specs)
    regressions = tuple(row for row in selected if row.row.f1_delta < -_F1_TOLERANCE)
    safe_headroom = _safe_alternative_headroom(
        regressions=regressions,
        rows_by_question=rows_by_question,
        predictions_by_question=predictions_by_question,
        spec_by_fold=spec_by_fold,
    )
    calibration = {
        "selected_action_population": _risk_calibration(
            tuple(
                selected_prediction_index[(row.row.question_key, row.row.action.action_id)]
                for row in selected
            )
        ),
        "all_outer_actions": _risk_calibration(predictions),
    }
    head_ranges = _head_metric_ranges(outer_fold_reports)
    no_eligible = _no_eligible_fold_attribution(outer_fold_reports)
    selected_regression_rate = _ratio(len(regressions), len(selected))
    findings = {
        "weak_f1_risk_head_separability": (
            head_ranges["f1_regression"]["maximum_roc_auc"] < _RISK_AUC_WEAK_THRESHOLD
        ),
        "selected_f1_regression_rate_high": (
            selected_regression_rate > _SELECTED_REGRESSION_RATE_HIGH_THRESHOLD
        ),
        "safe_alternative_headroom_high": (
            safe_headroom["same_or_better_citation_safe_alternative_rate"]
            >= _SAFE_ALTERNATIVE_HEADROOM_THRESHOLD
        ),
        "outer_fold_without_inner_eligible_policy": bool(no_eligible["fold_count"]),
    }
    primary = _primary_bottleneck(findings)
    return {
        "protocol": {
            "scope": "frozen Stage182 outer predictions and selected actions",
            "new_model_fit_count": 0,
            "policy_selection_enabled": False,
            "risk_auc_weak_threshold": _RISK_AUC_WEAK_THRESHOLD,
            "selected_regression_rate_high_threshold": (_SELECTED_REGRESSION_RATE_HIGH_THRESHOLD),
            "safe_alternative_headroom_threshold": (_SAFE_ALTERNATIVE_HEADROOM_THRESHOLD),
            "development_and_test_closed": True,
            "fallback_enabled": False,
        },
        "selected_action_summary": {
            "selected_action_count": len(selected),
            "strict_expected_count": sum(row.row.strict_expected for row in selected),
            "f1_regression_count": len(regressions),
            "f1_regression_rate": selected_regression_rate,
            "mean_predicted_f1_risk": _mean(row.f1_regression_probability for row in selected),
            "mean_observed_f1_delta": _mean(row.row.f1_delta for row in selected),
            "severity": _severity_summary(regressions),
        },
        "head_metric_ranges": head_ranges,
        "risk_calibration": calibration,
        "selected_regression_concentration": {
            "by_fold": _group_selected(selected, key=lambda row: row.row.fold_id),
            "by_route": _group_selected(selected, key=lambda row: row.row.route),
            "by_action_family": _group_selected(selected, key=lambda row: row.row.action.family),
            "by_policy": _group_selected(
                selected,
                key=lambda row: outer_fold_reports[row.row.fold_id]["selected_spec"],
            ),
        },
        "safe_alternative_headroom": safe_headroom,
        "runtime_feature_separation": _runtime_feature_separation(predictions),
        "no_inner_eligible_fold_attribution": no_eligible,
        "diagnostic_findings": {
            **findings,
            "primary_bottleneck": primary,
            "runtime_e2e_authorized": False,
            "next_design_scope": (
                "runtime-visible F1-risk representation and ranking only"
                if primary == "f1_risk_separability_and_ranking"
                else "action-space and F1-risk representation review"
            ),
        },
    }


def _selected_specs_by_fold(
    outer_fold_reports: Mapping[str, Mapping[str, Any]],
    *,
    specs: Mapping[str, DualTargetPolicySpec],
) -> dict[str, DualTargetPolicySpec]:
    selected = {}
    for fold_id, report in outer_fold_reports.items():
        name = report["selected_spec"]
        if name is None:
            continue
        if name not in specs:
            raise ValueError(f"unknown Stage182 selected policy: {name}")
        selected[fold_id] = specs[name]
    return selected


def _risk_calibration(
    predictions: Sequence[DualTargetPrediction],
) -> dict[str, Any]:
    bins = []
    weighted_error = 0.0
    for lower, upper in _RISK_BINS:
        members = [
            row
            for row in predictions
            if lower <= row.f1_regression_probability < upper
            or (upper == 1.0 and row.f1_regression_probability == 1.0)
        ]
        observed = _ratio(
            sum(row.row.f1_delta < -_F1_TOLERANCE for row in members),
            len(members),
        )
        predicted = _mean(row.f1_regression_probability for row in members)
        weighted_error += len(members) * abs(observed - predicted)
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "action_count": len(members),
                "mean_predicted_risk": predicted,
                "observed_regression_rate": observed,
                "mean_f1_delta": _mean(row.row.f1_delta for row in members),
            }
        )
    labels = [int(row.row.f1_delta < -_F1_TOLERANCE) for row in predictions]
    scores = [row.f1_regression_probability for row in predictions]
    brier = _mean((score - label) ** 2 for score, label in zip(scores, labels, strict=True))
    return {
        "action_count": len(predictions),
        "observed_regression_count": sum(labels),
        "observed_regression_rate": _ratio(sum(labels), len(labels)),
        "mean_predicted_risk": _mean(scores),
        "expected_calibration_error": round(weighted_error / len(predictions), 6)
        if predictions
        else 0.0,
        "brier_score": brier,
        "bins": bins,
    }


def _safe_alternative_headroom(
    *,
    regressions: Sequence[SelectedAction],
    rows_by_question: Mapping[str, Sequence[ActionAuditRow]],
    predictions_by_question: Mapping[str, Sequence[DualTargetPrediction]],
    spec_by_fold: Mapping[str, DualTargetPolicySpec],
) -> dict[str, Any]:
    any_strict = 0
    same_or_better_safe = 0
    safe_citation_gain = 0
    safe_top3 = 0
    safe_top5 = 0
    margins = []
    for selected in regressions:
        question_rows = rows_by_question[selected.row.question_key]
        same_or_better = [
            row
            for row in question_rows
            if row.f1_delta >= -_F1_TOLERANCE and row.citation_delta >= selected.row.citation_delta
        ]
        any_strict += any(row.strict_expected for row in question_rows)
        same_or_better_safe += bool(same_or_better)
        safe_citation_gain += any(
            row.f1_delta >= -_F1_TOLERANCE and row.citation_delta > 0 for row in question_rows
        )
        spec = spec_by_fold[selected.row.fold_id]
        ranked = sorted(
            predictions_by_question[selected.row.question_key],
            key=lambda row: (
                -dual_target_utility(row, spec.score_mode),
                -row.citation_gain_probability,
                row.f1_regression_probability,
                row.row.action.action_id,
            ),
        )
        safe_ids = {row.action.action_id for row in same_or_better}
        safe_top3 += any(row.row.action.action_id in safe_ids for row in ranked[:3])
        safe_top5 += any(row.row.action.action_id in safe_ids for row in ranked[:5])
        if same_or_better:
            prediction_by_action = {
                row.row.action.action_id: row
                for row in predictions_by_question[selected.row.question_key]
            }
            best_safe_utility = max(
                dual_target_utility(prediction_by_action[row.action.action_id], spec.score_mode)
                for row in same_or_better
            )
            margins.append(selected.utility - best_safe_utility)
    count = len(regressions)
    return {
        "regressed_selected_action_count": count,
        "questions_with_any_strict_alternative": any_strict,
        "any_strict_alternative_rate": _ratio(any_strict, count),
        "questions_with_same_or_better_citation_safe_alternative": same_or_better_safe,
        "same_or_better_citation_safe_alternative_rate": _ratio(same_or_better_safe, count),
        "questions_with_safe_citation_gain_alternative": safe_citation_gain,
        "safe_citation_gain_alternative_rate": _ratio(safe_citation_gain, count),
        "same_or_better_safe_alternative_in_model_top3": safe_top3,
        "same_or_better_safe_alternative_in_model_top3_rate": _ratio(safe_top3, count),
        "same_or_better_safe_alternative_in_model_top5": safe_top5,
        "same_or_better_safe_alternative_in_model_top5_rate": _ratio(safe_top5, count),
        "selected_minus_best_safe_utility_margin": _distribution(margins),
    }


def _runtime_feature_separation(
    predictions: Sequence[DualTargetPrediction],
) -> dict[str, Any]:
    labels = [int(row.row.f1_delta < -_F1_TOLERANCE) for row in predictions]
    feature_names = sorted(
        {
            str(name)
            for prediction in predictions
            for name, value in prediction.row.runtime_features.items()
            if isinstance(value, (bool, int, float))
        }
    )
    summaries = []
    for name in feature_names:
        values = [
            float(prediction.row.runtime_features.get(name, 0.0)) for prediction in predictions
        ]
        positive = [value for value, label in zip(values, labels, strict=True) if label]
        negative = [value for value, label in zip(values, labels, strict=True) if not label]
        if not positive or not negative or len(set(values)) < 2:
            continue
        auc = roc_auc_score(labels, values)
        oriented_auc = max(float(auc), 1.0 - float(auc))
        pooled = math.sqrt((statistics.pvariance(positive) + statistics.pvariance(negative)) / 2.0)
        standardized_difference = (
            abs(statistics.fmean(positive) - statistics.fmean(negative)) / pooled
            if pooled > 0
            else 0.0
        )
        summaries.append(
            {
                "feature": name,
                "regression_mean": round(statistics.fmean(positive), 6),
                "nonregression_mean": round(statistics.fmean(negative), 6),
                "oriented_univariate_auc": round(oriented_auc, 6),
                "absolute_standardized_mean_difference": round(standardized_difference, 6),
            }
        )
    summaries.sort(
        key=lambda row: (
            row["oriented_univariate_auc"],
            row["absolute_standardized_mean_difference"],
            row["feature"],
        ),
        reverse=True,
    )
    return {
        "evaluated_numeric_feature_count": len(summaries),
        "top_features": summaries[:20],
        "maximum_oriented_univariate_auc": (
            summaries[0]["oriented_univariate_auc"] if summaries else None
        ),
    }


def _group_selected(
    selected: Sequence[SelectedAction],
    *,
    key: Callable[[SelectedAction], Any],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[SelectedAction]] = defaultdict(list)
    for row in selected:
        grouped[str(key(row))].append(row)
    return {
        name: {
            "selected_action_count": len(rows),
            "f1_regression_count": sum(row.row.f1_delta < -_F1_TOLERANCE for row in rows),
            "f1_regression_rate": _ratio(
                sum(row.row.f1_delta < -_F1_TOLERANCE for row in rows), len(rows)
            ),
            "mean_f1_delta": _mean(row.row.f1_delta for row in rows),
            "gold_citation_delta": sum(row.row.citation_delta for row in rows),
            "mean_predicted_f1_risk": _mean(row.f1_regression_probability for row in rows),
        }
        for name, rows in sorted(grouped.items())
    }


def _severity_summary(regressions: Sequence[SelectedAction]) -> dict[str, Any]:
    deltas = [row.row.f1_delta for row in regressions]
    return {
        "mild_above_minus_0_01": sum(delta > -0.01 for delta in deltas),
        "moderate_minus_0_05_to_minus_0_01": sum(-0.05 < delta <= -0.01 for delta in deltas),
        "large_minus_0_10_to_minus_0_05": sum(-0.10 < delta <= -0.05 for delta in deltas),
        "severe_at_or_below_minus_0_10": sum(delta <= -0.10 for delta in deltas),
        "distribution": _distribution(deltas),
    }


def _head_metric_ranges(
    outer_fold_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    values: dict[str, list[float]] = {
        "citation_gain": [],
        "f1_regression": [],
    }
    by_model: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"citation_gain": [], "f1_regression": []}
    )
    for report in outer_fold_reports.values():
        for model, metrics in report["inner_head_metrics"].items():
            for target in values:
                auc = metrics["aggregate"][target]["roc_auc"]
                if auc is not None:
                    values[target].append(float(auc))
                    by_model[model][target].append(float(auc))
    return {
        target: {
            "minimum_roc_auc": round(min(target_values), 6),
            "maximum_roc_auc": round(max(target_values), 6),
            "mean_roc_auc": _mean(target_values),
            "by_model": {
                model: {
                    "minimum_roc_auc": round(min(model_targets[target]), 6),
                    "maximum_roc_auc": round(max(model_targets[target]), 6),
                    "mean_roc_auc": _mean(model_targets[target]),
                }
                for model, model_targets in sorted(by_model.items())
            },
        }
        for target, target_values in values.items()
    }


def _no_eligible_fold_attribution(
    outer_fold_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    reports = {}
    reason_totals = Counter()
    for fold_id, report in outer_fold_reports.items():
        if report["selected_spec"] is not None:
            continue
        candidates = []
        for candidate in report["candidate_leaderboard"]:
            evaluation = candidate["evaluation"]
            fold_count = len(evaluation["folds"])
            reasons = []
            if not evaluation["strict_aggregate_pass"]:
                reasons.append("aggregate_strict_a_failed")
            if evaluation["citation_nonregressing_fold_count"] < fold_count:
                reasons.append("citation_fold_nonregression_failed")
            if evaluation["f1_nonregressing_fold_count"] < fold_count:
                reasons.append("f1_fold_nonregression_failed")
            reason_totals.update(reasons)
            candidates.append(
                {
                    "name": candidate["name"],
                    "failure_reasons": reasons,
                    "gold_citation_delta": evaluation["gold_citation_delta"],
                    "mean_f1_delta": evaluation["mean_f1_delta_all_questions"],
                    "citation_nonregressing_fold_count": evaluation[
                        "citation_nonregressing_fold_count"
                    ],
                    "f1_nonregressing_fold_count": evaluation["f1_nonregressing_fold_count"],
                }
            )
        candidates.sort(
            key=lambda row: (
                len(row["failure_reasons"]),
                -row["citation_nonregressing_fold_count"],
                -row["f1_nonregressing_fold_count"],
                -row["gold_citation_delta"],
                -row["mean_f1_delta"],
                row["name"],
            )
        )
        reports[fold_id] = {
            "candidate_count": len(candidates),
            "failure_reason_counts": dict(
                sorted(
                    Counter(
                        reason for row in candidates for reason in row["failure_reasons"]
                    ).items()
                )
            ),
            "closest_candidates": candidates[:8],
        }
    return {
        "fold_count": len(reports),
        "folds": reports,
        "failure_reason_counts": dict(sorted(reason_totals.items())),
    }


def _primary_bottleneck(findings: Mapping[str, bool]) -> str:
    if (
        findings["weak_f1_risk_head_separability"]
        and findings["selected_f1_regression_rate_high"]
        and findings["safe_alternative_headroom_high"]
    ):
        return "f1_risk_separability_and_ranking"
    if findings["weak_f1_risk_head_separability"]:
        return "f1_risk_representation"
    return "action_space_or_selection_stability"


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {
            "minimum": None,
            "median": None,
            "p95": None,
            "maximum": None,
            "mean": None,
        }
    ordered = sorted(values)
    return {
        "minimum": round(float(ordered[0]), 6),
        "median": round(float(statistics.median(ordered)), 6),
        "p95": round(float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]), 6),
        "maximum": round(float(ordered[-1]), 6),
        "mean": round(float(statistics.fmean(ordered)), 6),
    }


def _mean(values: Any) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(statistics.fmean(materialized), 6)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)
