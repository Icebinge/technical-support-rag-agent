from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Any

from ts_rag_agent.application.composition_gain_sensitive_ranking import (
    GainSensitiveInnerOOFSnapshot,
    GainSensitivePolicySpec,
    GainSensitivePrediction,
    build_gain_sensitive_question_decisions,
    stage188_policy_specs,
)

_F1_TOLERANCE = 1e-12


@dataclass
class _PolicyTotals:
    question_context_count: int = 0
    strict_opportunity_context_count: int = 0
    frontier_exclusion_context_count: int = 0
    ranker_miss_context_count: int = 0
    strict_selected_context_count: int = 0
    changed_context_count: int = 0
    safe_zero_selected_context_count: int = 0
    unsafe_selected_context_count: int = 0
    unfiltered_strict_selected_context_count: int = 0
    filter_harm_context_count: int = 0
    filter_rescue_context_count: int = 0
    all_action_count: int = 0
    strict_action_count: int = 0
    frontier_action_count: int = 0
    frontier_strict_action_count: int = 0
    frontier_contains_baseline_context_count: int = 0
    selected_citation_delta: int = 0
    selected_f1_delta_sum: float = 0.0
    selected_citation_loss_context_count: int = 0
    selected_f1_regression_context_count: int = 0

    def merge(self, other: _PolicyTotals) -> None:
        for field in fields(self):
            setattr(self, field.name, getattr(self, field.name) + getattr(other, field.name))

    def report(self) -> dict[str, Any]:
        frontier_opportunity_count = (
            self.ranker_miss_context_count + self.strict_selected_context_count
        )
        partition_total = (
            self.frontier_exclusion_context_count
            + self.ranker_miss_context_count
            + self.strict_selected_context_count
        )
        return {
            "question_context_count": self.question_context_count,
            "strict_opportunity_context_count": self.strict_opportunity_context_count,
            "opportunity_partition": {
                "frontier_exclusion_context_count": self.frontier_exclusion_context_count,
                "ranker_miss_context_count": self.ranker_miss_context_count,
                "strict_selected_context_count": self.strict_selected_context_count,
                "partition_total": partition_total,
                "partition_exact": partition_total == self.strict_opportunity_context_count,
            },
            "frontier_strict_question_recall": _ratio(
                frontier_opportunity_count,
                self.strict_opportunity_context_count,
            ),
            "conditional_ranker_strict_capture": _ratio(
                self.strict_selected_context_count,
                frontier_opportunity_count,
            ),
            "actual_strict_opportunity_capture": _ratio(
                self.strict_selected_context_count,
                self.strict_opportunity_context_count,
            ),
            "baseline_change_strict_precision": _ratio(
                self.strict_selected_context_count,
                self.changed_context_count,
            ),
            "baseline_changed_context_count": self.changed_context_count,
            "safe_zero_selected_context_count": self.safe_zero_selected_context_count,
            "unsafe_selected_context_count": self.unsafe_selected_context_count,
            "unsafe_selection_rate": _ratio(
                self.unsafe_selected_context_count,
                self.question_context_count,
            ),
            "unfiltered_ranker_strict_capture": _ratio(
                self.unfiltered_strict_selected_context_count,
                self.strict_opportunity_context_count,
            ),
            "filter_harm_context_count": self.filter_harm_context_count,
            "filter_rescue_context_count": self.filter_rescue_context_count,
            "all_action_count": self.all_action_count,
            "strict_action_count": self.strict_action_count,
            "frontier_action_count": self.frontier_action_count,
            "frontier_strict_action_count": self.frontier_strict_action_count,
            "strict_action_retention_rate": _ratio(
                self.frontier_strict_action_count,
                self.strict_action_count,
            ),
            "mean_frontier_size": _ratio(
                self.frontier_action_count,
                self.question_context_count,
            ),
            "frontier_contains_baseline_rate": _ratio(
                self.frontier_contains_baseline_context_count,
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


class GainSensitiveFailureAttributionAccumulator:
    """Consumes one private Stage 188 fold snapshot at a time and keeps aggregates."""

    def __init__(self) -> None:
        specs = stage188_policy_specs()
        self._specs = {spec.name: spec for spec in specs}
        self._fold_reports: dict[str, dict[str, Any]] = {}
        self._config_totals = {name: _PolicyTotals() for name in self._specs}
        self._reference_totals = _PolicyTotals()
        self._private_prediction_count = 0

    def consume(self, snapshot: GainSensitiveInnerOOFSnapshot) -> None:
        """Summarize one outer context without retaining its private predictions."""

        if snapshot.outer_fold_id in self._fold_reports:
            raise ValueError("Stage189 received a duplicate outer-fold snapshot")
        if snapshot.top_candidate_spec_name not in self._specs:
            raise ValueError("Stage189 top candidate is outside the Stage188 grid")
        config_reports: dict[str, dict[str, Any]] = {}
        fold_totals: dict[str, _PolicyTotals] = {}
        for spec in self._specs.values():
            predictions = snapshot.predictions_by_bundle.get(spec.bundle_name)
            if predictions is None:
                raise ValueError(f"Stage189 missing prediction bundle {spec.bundle_name}")
            totals = _diagnose_policy_totals(predictions, spec)
            if totals.question_context_count != snapshot.question_count:
                raise ValueError("Stage189 snapshot question count drifted")
            fold_totals[spec.name] = totals
            self._config_totals[spec.name].merge(totals)
            config_reports[spec.name] = totals.report()

        reference_totals = fold_totals[snapshot.top_candidate_spec_name]
        self._reference_totals.merge(reference_totals)
        self._private_prediction_count += sum(
            len(rows) for rows in snapshot.predictions_by_bundle.values()
        )
        self._fold_reports[snapshot.outer_fold_id] = {
            "inner_fold_ids": list(snapshot.inner_fold_ids),
            "question_context_count": snapshot.question_count,
            "eligible_config_count": snapshot.eligible_config_count,
            "top_ineligible_spec": snapshot.top_candidate_spec_name,
            "top_ineligible_diagnostics": reference_totals.report(),
            "best_configurations": _best_configurations(config_reports),
            "configuration_diagnostics": dict(sorted(config_reports.items())),
        }

    def finalize(self) -> dict[str, Any]:
        """Build the public-safe Stage 189 aggregate report."""

        if len(self._fold_reports) != 5:
            raise ValueError("Stage189 requires exactly five outer-fold snapshots")
        config_reports = {
            name: totals.report() for name, totals in sorted(self._config_totals.items())
        }
        reference_report = self._reference_totals.report()
        primary_bottleneck = _primary_bottleneck(reference_report)
        return {
            "protocol": {
                "diagnostic_population": "Stage188 inner-OOF question contexts",
                "outer_context_count": 5,
                "policy_config_count": len(self._specs),
                "opportunity_partition": (
                    "strict action excluded by frontier, retained but missed by ranker, "
                    "or selected as strict"
                ),
                "bottleneck_rule": (
                    "the larger lost-opportunity count between frontier exclusion and "
                    "ranker miss; exact tie is mixed"
                ),
                "gold_scope": "train-only offline attribution labels",
                "new_model_fit_count": 0,
                "private_rows_persisted": False,
            },
            "outer_folds": dict(sorted(self._fold_reports.items())),
            "top_ineligible_trajectory": reference_report,
            "configuration_aggregates": config_reports,
            "factor_aggregates": _factor_aggregates(self._specs, self._config_totals),
            "family_best_configurations": _best_configurations(config_reports),
            "diagnostic_findings": {
                "primary_bottleneck": primary_bottleneck,
                "frontier_exclusion_context_count": reference_report["opportunity_partition"][
                    "frontier_exclusion_context_count"
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
                "stage190_design_branch": _stage190_design_branch(primary_bottleneck),
            },
            "execution": {
                "snapshot_count": len(self._fold_reports),
                "private_bundle_prediction_count_consumed": self._private_prediction_count,
                "new_model_fit_count": 0,
                "public_action_rows_written": 0,
                "public_prediction_rows_written": 0,
                "all_configuration_partitions_exact": all(
                    report["opportunity_partition"]["partition_exact"]
                    for report in config_reports.values()
                ),
            },
        }


def diagnose_gain_sensitive_policy(
    predictions: tuple[GainSensitivePrediction, ...],
    spec: GainSensitivePolicySpec,
) -> dict[str, Any]:
    """Diagnose one policy on one held-out inner-OOF population."""

    return _diagnose_policy_totals(predictions, spec).report()


def _diagnose_policy_totals(
    predictions: tuple[GainSensitivePrediction, ...],
    spec: GainSensitivePolicySpec,
) -> _PolicyTotals:
    totals = _PolicyTotals()
    grouped_predictions: dict[str, list[GainSensitivePrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped_predictions[prediction.row.question_key].append(prediction)
    decisions = build_gain_sensitive_question_decisions(predictions, spec)
    for decision in decisions:
        if decision.baseline is None:
            raise ValueError("Stage189 diagnostics require one baseline prediction per question")
        all_predictions = grouped_predictions[decision.question_key]
        strict_predictions = [row for row in all_predictions if row.row.strict_expected]
        frontier_strict = [row for row in decision.frontier if row.row.strict_expected]
        has_strict_opportunity = bool(strict_predictions)
        winner_is_strict = decision.winner.row.strict_expected
        unfiltered_is_strict = decision.unfiltered_winner.row.strict_expected

        totals.question_context_count += 1
        totals.all_action_count += len(all_predictions)
        totals.strict_action_count += len(strict_predictions)
        totals.frontier_action_count += len(decision.frontier)
        totals.frontier_strict_action_count += len(frontier_strict)
        totals.frontier_contains_baseline_context_count += decision.baseline in decision.frontier
        totals.changed_context_count += (
            decision.winner.row.action.action_id != decision.baseline.row.action.action_id
        )
        totals.strict_selected_context_count += winner_is_strict
        totals.unfiltered_strict_selected_context_count += unfiltered_is_strict
        totals.selected_citation_delta += decision.winner.row.citation_delta
        totals.selected_f1_delta_sum += decision.winner.row.f1_delta
        totals.selected_citation_loss_context_count += decision.winner.row.citation_delta < 0
        totals.selected_f1_regression_context_count += decision.winner.row.f1_delta < -_F1_TOLERANCE

        winner_is_unsafe = (
            decision.winner.row.citation_delta < 0 or decision.winner.row.f1_delta < -_F1_TOLERANCE
        )
        totals.unsafe_selected_context_count += winner_is_unsafe
        totals.safe_zero_selected_context_count += not winner_is_strict and not winner_is_unsafe
        totals.filter_harm_context_count += unfiltered_is_strict and not winner_is_strict
        totals.filter_rescue_context_count += winner_is_strict and not unfiltered_is_strict

        if has_strict_opportunity:
            totals.strict_opportunity_context_count += 1
            if not frontier_strict:
                totals.frontier_exclusion_context_count += 1
            elif winner_is_strict:
                pass
            else:
                totals.ranker_miss_context_count += 1
    return totals


def _best_configurations(config_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "frontier_strict_question_recall",
        "conditional_ranker_strict_capture",
        "actual_strict_opportunity_capture",
        "baseline_change_strict_precision",
    )
    return {metric: _best_configuration(config_reports, metric=metric) for metric in metrics}


def _best_configuration(
    config_reports: dict[str, dict[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    name, report = min(
        config_reports.items(),
        key=lambda item: (
            -item[1][metric],
            item[1]["unsafe_selected_context_count"],
            item[0],
        ),
    )
    return {
        "spec": name,
        "metric": metric,
        "value": report[metric],
        "unsafe_selected_context_count": report["unsafe_selected_context_count"],
    }


def _factor_aggregates(
    specs: dict[str, GainSensitivePolicySpec],
    totals_by_spec: dict[str, _PolicyTotals],
) -> dict[str, Any]:
    factor_getters = {
        "feature_representation": lambda spec: spec.feature_representation,
        "safety_estimator": lambda spec: spec.safety_estimator,
        "gain_ranker": lambda spec: spec.gain_ranker,
        "safety_frontier_margin": lambda spec: f"{spec.safety_frontier_margin:.2f}",
    }
    result = {}
    for factor_name, getter in factor_getters.items():
        grouped: dict[str, _PolicyTotals] = defaultdict(_PolicyTotals)
        configuration_counts: dict[str, int] = defaultdict(int)
        for name, spec in specs.items():
            value = getter(spec)
            grouped[value].merge(totals_by_spec[name])
            configuration_counts[value] += 1
        result[factor_name] = {
            value: {
                "configuration_count": configuration_counts[value],
                **totals.report(),
            }
            for value, totals in sorted(grouped.items())
        }
    return result


def _primary_bottleneck(reference_report: dict[str, Any]) -> str:
    partition = reference_report["opportunity_partition"]
    excluded = partition["frontier_exclusion_context_count"]
    missed = partition["ranker_miss_context_count"]
    if excluded > missed:
        return "safety_frontier_exclusion"
    if missed > excluded:
        return "gain_ranker_miss"
    return "mixed_frontier_and_ranker"


def _stage190_design_branch(primary_bottleneck: str) -> dict[str, Any]:
    if primary_bottleneck == "gain_ranker_miss":
        return {
            "name": "baseline_referenced_strict_change_gate",
            "retain_stage188_frontier_as_candidate_generator": True,
            "replace_gain_ordering": True,
            "required_components": [
                "candidate-versus-baseline contrast features",
                "strict-improvement score",
                "explicit learned change threshold",
                "baseline abstention as policy semantics",
            ],
        }
    if primary_bottleneck == "safety_frontier_exclusion":
        return {
            "name": "recall_constrained_safety_frontier",
            "retain_stage188_frontier_as_candidate_generator": False,
            "replace_gain_ordering": False,
            "required_components": [
                "inner-OOF strict-action retention objective",
                "recall-constrained safety threshold",
                "separate citation and F1 loss ceilings",
                "no post-hoc threshold relaxation",
            ],
        }
    return {
        "name": "joint_frontier_and_baseline_change_redesign",
        "retain_stage188_frontier_as_candidate_generator": False,
        "replace_gain_ordering": True,
        "required_components": [
            "recall-constrained safety threshold",
            "candidate-versus-baseline contrast features",
            "strict-improvement score",
            "explicit learned change threshold",
        ],
    }


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0
