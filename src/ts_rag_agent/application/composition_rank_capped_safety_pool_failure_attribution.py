from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from ts_rag_agent.application.composition_gain_sensitive_ranking import (
    GainSensitivePrediction,
)
from ts_rag_agent.application.composition_rank_capped_safety_pool import (
    RankCappedSafetyPoolInnerOOFSnapshot,
    RankCappedSafetyPoolPolicySpec,
    build_rank_capped_safety_pool_decisions,
    stage191_policy_specs,
)

_F1_TOLERANCE = 1e-12


@dataclass
class _PolicyTotals:
    question_context_count: int = 0
    strict_opportunity_context_count: int = 0
    pool_exclusion_context_count: int = 0
    ranker_miss_context_count: int = 0
    strict_selected_context_count: int = 0
    baseline_changed_context_count: int = 0
    safe_zero_selected_context_count: int = 0
    unsafe_selected_context_count: int = 0
    ranker_miss_safe_zero_context_count: int = 0
    ranker_miss_unsafe_context_count: int = 0
    all_action_count: int = 0
    strict_action_count: int = 0
    pool_action_count: int = 0
    pool_strict_action_count: int = 0
    baseline_in_pool_context_count: int = 0
    selected_citation_delta: int = 0
    selected_f1_delta_sum: float = 0.0
    selected_citation_loss_context_count: int = 0
    selected_f1_regression_context_count: int = 0

    def merge(self, other: _PolicyTotals) -> None:
        for field in fields(self):
            setattr(self, field.name, getattr(self, field.name) + getattr(other, field.name))

    def report(self) -> dict[str, Any]:
        retained_opportunities = self.ranker_miss_context_count + self.strict_selected_context_count
        partition_total = (
            self.pool_exclusion_context_count
            + self.ranker_miss_context_count
            + self.strict_selected_context_count
        )
        return {
            "question_context_count": self.question_context_count,
            "strict_opportunity_context_count": self.strict_opportunity_context_count,
            "opportunity_partition": {
                "pool_exclusion_context_count": self.pool_exclusion_context_count,
                "ranker_miss_context_count": self.ranker_miss_context_count,
                "strict_selected_context_count": self.strict_selected_context_count,
                "partition_total": partition_total,
                "partition_exact": partition_total == self.strict_opportunity_context_count,
            },
            "strict_opportunity_pool_recall": _ratio(
                retained_opportunities,
                self.strict_opportunity_context_count,
            ),
            "conditional_ranker_strict_capture": _ratio(
                self.strict_selected_context_count,
                retained_opportunities,
            ),
            "actual_strict_opportunity_capture": _ratio(
                self.strict_selected_context_count,
                self.strict_opportunity_context_count,
            ),
            "baseline_change_strict_precision": _ratio(
                self.strict_selected_context_count,
                self.baseline_changed_context_count,
            ),
            "baseline_changed_context_count": self.baseline_changed_context_count,
            "safe_zero_selected_context_count": self.safe_zero_selected_context_count,
            "unsafe_selected_context_count": self.unsafe_selected_context_count,
            "unsafe_selection_rate": _ratio(
                self.unsafe_selected_context_count,
                self.question_context_count,
            ),
            "ranker_miss_breakdown": {
                "safe_zero_winner_context_count": self.ranker_miss_safe_zero_context_count,
                "unsafe_winner_context_count": self.ranker_miss_unsafe_context_count,
                "partition_exact": (
                    self.ranker_miss_safe_zero_context_count + self.ranker_miss_unsafe_context_count
                    == self.ranker_miss_context_count
                ),
            },
            "all_action_count": self.all_action_count,
            "strict_action_count": self.strict_action_count,
            "pool_action_count": self.pool_action_count,
            "pool_strict_action_count": self.pool_strict_action_count,
            "strict_action_retention_rate": _ratio(
                self.pool_strict_action_count,
                self.strict_action_count,
            ),
            "mean_pool_size": _ratio(
                self.pool_action_count,
                self.question_context_count,
            ),
            "baseline_in_pool_rate": _ratio(
                self.baseline_in_pool_context_count,
                self.question_context_count,
            ),
            "selected_gold_citation_delta": self.selected_citation_delta,
            "selected_mean_f1_delta": _ratio(
                self.selected_f1_delta_sum,
                self.question_context_count,
            ),
            "selected_citation_loss_context_count": (self.selected_citation_loss_context_count),
            "selected_f1_regression_context_count": (self.selected_f1_regression_context_count),
        }


class RankCappedSafetyPoolFailureAttributionAccumulator:
    """Stream Stage 191 private snapshots into public-safe aggregates."""

    def __init__(self) -> None:
        specs = stage191_policy_specs()
        self._specs = {spec.name: spec for spec in specs}
        self._fold_reports: dict[str, dict[str, Any]] = {}
        self._config_totals = {name: _PolicyTotals() for name in self._specs}
        self._reference_totals = _PolicyTotals()
        self._private_prediction_count = 0

    def consume(self, snapshot: RankCappedSafetyPoolInnerOOFSnapshot) -> None:
        if snapshot.outer_fold_id in self._fold_reports:
            raise ValueError("Stage192 received a duplicate outer-fold snapshot")
        if snapshot.reference_spec_name not in self._specs:
            raise ValueError("Stage192 reference spec is outside the Stage191 grid")
        config_reports: dict[str, dict[str, Any]] = {}
        fold_totals: dict[str, _PolicyTotals] = {}
        for spec in self._specs.values():
            predictions = snapshot.predictions_by_bundle.get(spec.bundle_name)
            if predictions is None:
                raise ValueError(f"Stage192 missing prediction bundle {spec.bundle_name}")
            totals = _diagnose_policy_totals(predictions, spec)
            if totals.question_context_count != snapshot.question_count:
                raise ValueError("Stage192 snapshot question count drifted")
            fold_totals[spec.name] = totals
            self._config_totals[spec.name].merge(totals)
            config_reports[spec.name] = totals.report()

        reference_totals = fold_totals[snapshot.reference_spec_name]
        self._reference_totals.merge(reference_totals)
        self._private_prediction_count += sum(
            len(rows) for rows in snapshot.predictions_by_bundle.values()
        )
        self._fold_reports[snapshot.outer_fold_id] = {
            "inner_fold_ids": list(snapshot.inner_fold_ids),
            "question_context_count": snapshot.question_count,
            "eligible_config_count": snapshot.eligible_config_count,
            "reference_spec": snapshot.reference_spec_name,
            "reference_kind": snapshot.reference_kind,
            "reference_diagnostics": reference_totals.report(),
            "best_configurations": _best_configurations(config_reports),
            "configuration_diagnostics": dict(sorted(config_reports.items())),
        }

    def finalize(self) -> dict[str, Any]:
        if len(self._fold_reports) != 5:
            raise ValueError("Stage192 requires exactly five outer-fold snapshots")
        config_reports = {
            name: totals.report() for name, totals in sorted(self._config_totals.items())
        }
        reference_report = self._reference_totals.report()
        primary_bottleneck = _primary_bottleneck(reference_report)
        return {
            "protocol": {
                "diagnostic_population": "Stage191 inner-OOF question contexts",
                "outer_context_count": 5,
                "policy_config_count": len(self._specs),
                "opportunity_partition": (
                    "strict action excluded by pool, retained but missed by ranker, "
                    "or selected as strict"
                ),
                "reference_trajectory": (
                    "selected eligible spec when available, otherwise top ineligible spec"
                ),
                "gold_scope": "train-only offline attribution labels",
                "new_model_fit_count": 0,
                "private_rows_persisted": False,
            },
            "outer_folds": dict(sorted(self._fold_reports.items())),
            "reference_trajectory": reference_report,
            "configuration_aggregates": config_reports,
            "factor_aggregates": _factor_aggregates(self._specs, self._config_totals),
            "family_best_configurations": _best_configurations(config_reports),
            "diagnostic_findings": {
                "primary_bottleneck": primary_bottleneck,
                "pool_exclusion_context_count": reference_report["opportunity_partition"][
                    "pool_exclusion_context_count"
                ],
                "ranker_miss_context_count": reference_report["opportunity_partition"][
                    "ranker_miss_context_count"
                ],
                "strict_selected_context_count": reference_report["opportunity_partition"][
                    "strict_selected_context_count"
                ],
                "opportunity_partition_exact": reference_report["opportunity_partition"][
                    "partition_exact"
                ],
                "recommended_next_focus": (
                    "within_pool_ranking"
                    if primary_bottleneck == "within_pool_ranker_miss"
                    else "candidate_pool"
                    if primary_bottleneck == "candidate_pool_exclusion"
                    else "mixed"
                ),
            },
            "execution": {
                "snapshot_count": len(self._fold_reports),
                "private_bundle_prediction_count_consumed": self._private_prediction_count,
                "new_model_fit_count": 0,
                "public_action_rows_written": 0,
                "public_prediction_rows_written": 0,
                "all_configuration_partitions_exact": all(
                    report["opportunity_partition"]["partition_exact"]
                    and report["ranker_miss_breakdown"]["partition_exact"]
                    for report in config_reports.values()
                ),
            },
        }


def diagnose_rank_capped_safety_pool_policy(
    predictions: tuple[GainSensitivePrediction, ...],
    spec: RankCappedSafetyPoolPolicySpec,
) -> dict[str, Any]:
    """Diagnose one private prediction bundle for focused tests."""

    return _diagnose_policy_totals(predictions, spec).report()


def _diagnose_policy_totals(
    predictions: tuple[GainSensitivePrediction, ...],
    spec: RankCappedSafetyPoolPolicySpec,
) -> _PolicyTotals:
    decisions = build_rank_capped_safety_pool_decisions(predictions, spec)
    predictions_by_question: dict[str, list[GainSensitivePrediction]] = {}
    for prediction in predictions:
        predictions_by_question.setdefault(prediction.row.question_key, []).append(prediction)
    totals = _PolicyTotals()
    for decision in decisions:
        question_predictions = predictions_by_question[decision.question_key]
        strict_predictions = [row for row in question_predictions if row.row.strict_expected]
        pool_strict = [row for row in decision.pool if row.row.strict_expected]
        winner = decision.winner.row
        totals.question_context_count += 1
        totals.all_action_count += len(question_predictions)
        totals.strict_action_count += len(strict_predictions)
        totals.pool_action_count += len(decision.pool)
        totals.pool_strict_action_count += len(pool_strict)
        totals.baseline_in_pool_context_count += decision.baseline in decision.pool
        totals.baseline_changed_context_count += (
            winner.action.action_id != decision.baseline.row.action.action_id
        )
        totals.selected_citation_delta += winner.citation_delta
        totals.selected_f1_delta_sum += winner.f1_delta
        totals.selected_citation_loss_context_count += winner.citation_delta < 0
        totals.selected_f1_regression_context_count += winner.f1_delta < -_F1_TOLERANCE
        if winner.strict_expected:
            pass
        elif winner.citation_delta == 0 and abs(winner.f1_delta) <= _F1_TOLERANCE:
            totals.safe_zero_selected_context_count += 1
        else:
            totals.unsafe_selected_context_count += 1

        if not strict_predictions:
            continue
        totals.strict_opportunity_context_count += 1
        if not pool_strict:
            totals.pool_exclusion_context_count += 1
        elif winner.strict_expected:
            totals.strict_selected_context_count += 1
        else:
            totals.ranker_miss_context_count += 1
            if winner.citation_delta == 0 and abs(winner.f1_delta) <= _F1_TOLERANCE:
                totals.ranker_miss_safe_zero_context_count += 1
            else:
                totals.ranker_miss_unsafe_context_count += 1
    return totals


def _best_configurations(config_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        metric: _best_configuration(config_reports, metric)
        for metric in (
            "strict_opportunity_pool_recall",
            "conditional_ranker_strict_capture",
            "actual_strict_opportunity_capture",
            "baseline_change_strict_precision",
        )
    }


def _best_configuration(
    config_reports: dict[str, dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    name, report = min(
        config_reports.items(),
        key=lambda item: (-item[1][metric], item[0]),
    )
    return {"spec": name, "metric": metric, "value": report[metric]}


def _factor_aggregates(
    specs: dict[str, RankCappedSafetyPoolPolicySpec],
    totals: dict[str, _PolicyTotals],
) -> dict[str, Any]:
    dimensions = {
        "feature_representation": lambda spec: spec.feature_representation,
        "safety_estimator": lambda spec: spec.safety_estimator,
        "gain_ranker": lambda spec: spec.gain_ranker,
        "pool_cap": lambda spec: str(spec.pool_cap),
    }
    result = {}
    for dimension, getter in dimensions.items():
        grouped: dict[str, _PolicyTotals] = {}
        for name, spec in specs.items():
            value = getter(spec)
            grouped.setdefault(value, _PolicyTotals()).merge(totals[name])
        result[dimension] = {value: grouped[value].report() for value in sorted(grouped)}
    return result


def _primary_bottleneck(report: dict[str, Any]) -> str:
    partition = report["opportunity_partition"]
    pool_exclusion = partition["pool_exclusion_context_count"]
    ranker_miss = partition["ranker_miss_context_count"]
    if pool_exclusion > ranker_miss:
        return "candidate_pool_exclusion"
    if ranker_miss > pool_exclusion:
        return "within_pool_ranker_miss"
    return "mixed"


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator / denominator), 6) if denominator else 0.0
