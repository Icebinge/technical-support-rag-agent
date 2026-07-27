from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ts_rag_agent.application import composition_safety_constrained_lambdamart as stage194
from ts_rag_agent.application.composition_joint_risk_winner_cv import (
    JointRiskWinnerCandidateSnapshot,
    JointRiskWinnerDiagnosticSnapshot,
)
from ts_rag_agent.application.composition_safety_first_frontier import (
    FrontierActionPrediction,
    SafetyFirstFrontierDecision,
)

_F1_TOLERANCE = 1e-12
_COUNT_NEAR_BOUNDARY = {
    "citation_delta",
    "citation_nonregressing_fold_count",
    "f1_nonregressing_fold_count",
    "changed_question_count",
    "strict_success_count",
    "folds_meeting_pool_recall_minimum",
    "folds_meeting_conditional_capture_minimum",
    "folds_meeting_unsafe_rate_maximum",
}
_FOLD_METRICS = {
    "gold_citation_delta": (">=", 0.0),
    "mean_f1_delta": (">=", 0.0),
    "strict_opportunity_pool_recall": (">=", 0.90),
    "conditional_ranker_strict_capture": (">=", 0.60),
    "unsafe_selection_rate": ("<=", 0.35),
}
_SELECTED_OUTCOMES = (
    "baseline",
    "strict_success",
    "safe_zero",
    "unsafe_citation_only",
    "unsafe_f1_only",
    "unsafe_citation_and_f1",
)
_OPPORTUNITY_MECHANISMS = (
    "no_strict_opportunity",
    "safety_pool_exclusion",
    "risk_frontier_exclusion",
    "winner_selection_miss",
    "strict_selected",
)


@dataclass
class _ConstraintAccumulator:
    name: str
    margins: list[float] = field(default_factory=list)
    failure_count: int = 0
    near_boundary_count: int = 0
    failures_by_outer: Counter[str] = field(default_factory=Counter)
    failures_by_risk: Counter[str] = field(default_factory=Counter)
    failures_by_winner: Counter[str] = field(default_factory=Counter)

    def add(
        self,
        *,
        margin: float,
        near_tolerance: float,
        outer_fold_id: str,
        risk_signal: str,
        winner_rule: str,
    ) -> None:
        self.margins.append(margin)
        self.near_boundary_count += int(-near_tolerance <= margin < 0.0)
        if margin >= -_F1_TOLERANCE:
            return
        self.failure_count += 1
        self.failures_by_outer[outer_fold_id] += 1
        self.failures_by_risk[risk_signal] += 1
        self.failures_by_winner[winner_rule] += 1

    def report(self, population: int) -> dict[str, Any]:
        return {
            "failure_count": self.failure_count,
            "failure_rate": _ratio(self.failure_count, population),
            "pass_count": population - self.failure_count,
            "near_boundary_count": self.near_boundary_count,
            "signed_margin_distribution": _distribution(self.margins),
            "failures_by_outer_context": dict(sorted(self.failures_by_outer.items())),
            "failures_by_risk_signal": dict(sorted(self.failures_by_risk.items())),
            "failures_by_winner_rule": dict(sorted(self.failures_by_winner.items())),
        }


@dataclass
class _QuestionAccumulator:
    context_count: int = 0
    outcome_counts: Counter[str] = field(default_factory=Counter)
    mechanism_counts: Counter[str] = field(default_factory=Counter)
    selected_action_families: Counter[str] = field(default_factory=Counter)
    pool_sizes: list[float] = field(default_factory=list)
    frontier_sizes: list[float] = field(default_factory=list)
    strict_opportunity_counts: list[float] = field(default_factory=list)
    unsafe_candidate_counts: list[float] = field(default_factory=list)
    selected_gain_ranks: list[float] = field(default_factory=list)
    selected_risk_ranks: list[float] = field(default_factory=list)
    best_strict_gain_ranks: list[float] = field(default_factory=list)
    best_strict_risk_ranks: list[float] = field(default_factory=list)
    gain_rank_gaps: list[float] = field(default_factory=list)
    risk_rank_gaps: list[float] = field(default_factory=list)
    lower_risk_strict_alternative_count: int = 0
    higher_gain_strict_alternative_count: int = 0
    strict_alternative_context_count: int = 0

    def add(self, decision: SafetyFirstFrontierDecision) -> None:
        winner = decision.winner
        self.context_count += 1
        self.outcome_counts[_selected_outcome(decision)] += 1
        self.mechanism_counts[_opportunity_mechanism(decision)] += 1
        self.selected_action_families[winner.row.action.family] += 1
        self.pool_sizes.append(float(len(decision.complete_pool)))
        self.frontier_sizes.append(float(len(decision.frontier)))
        strict_pool = [row for row in decision.complete_pool if row.row.strict_expected]
        strict_frontier = [row for row in decision.frontier if row.row.strict_expected]
        self.strict_opportunity_counts.append(float(len(strict_pool)))
        self.unsafe_candidate_counts.append(
            float(sum(stage194._is_unsafe(row.row) for row in decision.complete_pool))
        )

        gain_order = sorted(
            decision.frontier,
            key=lambda row: (-row.gain_score, row.unsafe_score, row.row.action.action_id),
        )
        risk_order = sorted(
            decision.complete_pool,
            key=lambda row: (row.unsafe_score, -row.gain_score, row.row.action.action_id),
        )
        selected_gain_rank = _rank(gain_order, winner)
        selected_risk_rank = _rank(risk_order, winner)
        self.selected_gain_ranks.append(float(selected_gain_rank))
        self.selected_risk_ranks.append(float(selected_risk_rank))
        if not strict_frontier:
            return

        self.strict_alternative_context_count += 1
        best_strict = min(
            strict_frontier,
            key=lambda row: (-row.gain_score, row.unsafe_score, row.row.action.action_id),
        )
        best_gain_rank = _rank(gain_order, best_strict)
        best_risk_rank = _rank(risk_order, best_strict)
        self.best_strict_gain_ranks.append(float(best_gain_rank))
        self.best_strict_risk_ranks.append(float(best_risk_rank))
        self.gain_rank_gaps.append(float(selected_gain_rank - best_gain_rank))
        self.risk_rank_gaps.append(float(selected_risk_rank - best_risk_rank))
        self.lower_risk_strict_alternative_count += int(
            any(row.unsafe_score < winner.unsafe_score for row in strict_frontier)
        )
        self.higher_gain_strict_alternative_count += int(
            any(row.gain_score > winner.gain_score for row in strict_frontier)
        )

    def report(self) -> dict[str, Any]:
        outcomes = {name: self.outcome_counts[name] for name in _SELECTED_OUTCOMES}
        mechanisms = {name: self.mechanism_counts[name] for name in _OPPORTUNITY_MECHANISMS}
        return {
            "question_cell_context_count": self.context_count,
            "selected_outcome_counts": outcomes,
            "selected_outcome_partition_total": sum(outcomes.values()),
            "selected_outcome_partition_exact": sum(outcomes.values()) == self.context_count,
            "strict_opportunity_mechanism_counts": mechanisms,
            "strict_opportunity_partition_total": sum(mechanisms.values()),
            "strict_opportunity_partition_exact": (sum(mechanisms.values()) == self.context_count),
            "selected_action_family_counts": dict(sorted(self.selected_action_families.items())),
            "context_distributions": {
                "pool_size": _distribution(self.pool_sizes),
                "frontier_size": _distribution(self.frontier_sizes),
                "strict_opportunity_count": _distribution(self.strict_opportunity_counts),
                "unsafe_candidate_count": _distribution(self.unsafe_candidate_counts),
            },
            "ranking_conflicts": {
                "selected_gain_rank": _distribution(self.selected_gain_ranks),
                "selected_risk_rank": _distribution(self.selected_risk_ranks),
                "best_strict_gain_rank": _distribution(self.best_strict_gain_ranks),
                "best_strict_risk_rank": _distribution(self.best_strict_risk_ranks),
                "selected_minus_best_strict_gain_rank_gap": _distribution(self.gain_rank_gaps),
                "selected_minus_best_strict_risk_rank_gap": _distribution(self.risk_rank_gaps),
                "strict_alternative_context_count": self.strict_alternative_context_count,
                "lower_risk_strict_alternative_count": (self.lower_risk_strict_alternative_count),
                "lower_risk_strict_alternative_rate": _ratio(
                    self.lower_risk_strict_alternative_count,
                    self.strict_alternative_context_count,
                ),
                "higher_gain_strict_alternative_count": (self.higher_gain_strict_alternative_count),
                "higher_gain_strict_alternative_rate": _ratio(
                    self.higher_gain_strict_alternative_count,
                    self.strict_alternative_context_count,
                ),
            },
        }


class JointRiskWinnerFailureAttributor:
    """Aggregate Stage199 private snapshots without persisting question-level rows."""

    def __init__(self, *, constraints: Sequence[Mapping[str, Any]]) -> None:
        self._constraints = tuple(dict(row) for row in constraints)
        self._constraint_totals = {
            str(row["name"]): _ConstraintAccumulator(str(row["name"])) for row in constraints
        }
        self._outer_cell_count = 0
        self._fold_cell_count = 0
        self._question_totals = _QuestionAccumulator()
        self._question_by_outer: dict[str, _QuestionAccumulator] = defaultdict(_QuestionAccumulator)
        self._question_by_risk: dict[str, _QuestionAccumulator] = defaultdict(_QuestionAccumulator)
        self._question_by_winner: dict[str, _QuestionAccumulator] = defaultdict(
            _QuestionAccumulator
        )
        self._failed_set_counts: Counter[str] = Counter()
        self._failed_count_distribution: Counter[str] = Counter()
        self._single_removal_pass_counts: Counter[str] = Counter()
        self._cofailure_counts: Counter[tuple[str, str]] = Counter()
        self._factor_cell_counts: dict[str, Counter[str]] = {
            "risk_signal": Counter(),
            "winner_rule": Counter(),
        }
        self._factor_failure_counts: dict[str, Counter[str]] = {
            "risk_signal": Counter(),
            "winner_rule": Counter(),
        }
        self._fold_violations: dict[str, Counter[str]] = defaultdict(Counter)
        self._worst_fold_frequency: dict[str, Counter[str]] = defaultdict(Counter)
        self._cross_fold_ranges: dict[str, list[float]] = defaultdict(list)
        self._cross_fold_stddevs: dict[str, list[float]] = defaultdict(list)
        self._aggregate_pass_fold_failure: Counter[str] = Counter()
        self._pareto_counts_by_outer: dict[str, int] = {}
        self._snapshot_count = 0

    def __call__(self, snapshot: JointRiskWinnerDiagnosticSnapshot) -> None:
        if len(snapshot.candidates) != 28:
            raise ValueError("Stage201 requires all 28 policy cells per outer context")
        if snapshot.outer_fold_id in self._pareto_counts_by_outer:
            raise ValueError(f"Duplicate Stage201 outer context {snapshot.outer_fold_id}")
        self._snapshot_count += 1
        self._pareto_counts_by_outer[snapshot.outer_fold_id] = _pareto_count(snapshot.candidates)
        for candidate in snapshot.candidates:
            self._add_candidate(snapshot, candidate)

    def _add_candidate(
        self,
        snapshot: JointRiskWinnerDiagnosticSnapshot,
        candidate: JointRiskWinnerCandidateSnapshot,
    ) -> None:
        self._outer_cell_count += 1
        risk_signal = str(candidate.spec["risk_signal"])
        winner_rule = str(candidate.spec["winner_rule"])
        self._factor_cell_counts["risk_signal"][risk_signal] += 1
        self._factor_cell_counts["winner_rule"][winner_rule] += 1
        failures: list[str] = []
        for constraint in self._constraints:
            name = str(constraint["name"])
            observed = float(_resolve_candidate_value(candidate, str(constraint["source"])))
            threshold = _constraint_threshold(
                constraint, inner_question_count=snapshot.inner_question_count
            )
            margin = _signed_margin(observed, threshold, str(constraint["operator"]))
            self._constraint_totals[name].add(
                margin=margin,
                near_tolerance=_near_tolerance(name),
                outer_fold_id=snapshot.outer_fold_id,
                risk_signal=risk_signal,
                winner_rule=winner_rule,
            )
            if margin < -_F1_TOLERANCE:
                failures.append(name)

        failure_key = "none" if not failures else "+".join(sorted(failures))
        self._failed_set_counts[failure_key] += 1
        self._failed_count_distribution[str(len(failures))] += 1
        if len(failures) == 1:
            self._single_removal_pass_counts[failures[0]] += 1
        if failures:
            self._factor_failure_counts["risk_signal"][risk_signal] += 1
            self._factor_failure_counts["winner_rule"][winner_rule] += 1
        for left_index, left in enumerate(sorted(failures)):
            for right in sorted(failures)[left_index + 1 :]:
                self._cofailure_counts[(left, right)] += 1

        self._add_folds(snapshot, candidate)
        for decision in candidate.decisions:
            self._question_totals.add(decision)
            self._question_by_outer[snapshot.outer_fold_id].add(decision)
            self._question_by_risk[risk_signal].add(decision)
            self._question_by_winner[winner_rule].add(decision)

    def _add_folds(
        self,
        snapshot: JointRiskWinnerDiagnosticSnapshot,
        candidate: JointRiskWinnerCandidateSnapshot,
    ) -> None:
        evaluation_folds = candidate.evaluation["folds"]
        diagnostic_folds = candidate.diagnostics["folds"]
        values_by_metric: dict[str, list[float]] = defaultdict(list)
        for fold_id in snapshot.inner_fold_ids:
            self._fold_cell_count += 1
            for metric, (operator, threshold) in _FOLD_METRICS.items():
                source = (
                    evaluation_folds
                    if metric
                    in {
                        "gold_citation_delta",
                        "mean_f1_delta",
                    }
                    else diagnostic_folds
                )
                value = float(source[fold_id][metric])
                values_by_metric[metric].append(value)
                if _signed_margin(value, threshold, operator) < -_F1_TOLERANCE:
                    self._fold_violations[metric][fold_id] += 1
        for metric, values in values_by_metric.items():
            operator = _FOLD_METRICS[metric][0]
            worst_value = max(values) if operator == "<=" else min(values)
            for index, value in enumerate(values):
                if abs(value - worst_value) <= _F1_TOLERANCE:
                    self._worst_fold_frequency[metric][snapshot.inner_fold_ids[index]] += 1
            self._cross_fold_ranges[metric].append(max(values) - min(values))
            self._cross_fold_stddevs[metric].append(
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )

        aggregate_pairs = {
            "gold_citation_delta": (
                float(candidate.evaluation["gold_citation_delta"]) >= 0.0,
                int(candidate.evaluation["citation_nonregressing_fold_count"]) < 3,
            ),
            "mean_f1_delta": (
                float(candidate.evaluation["mean_f1_delta"]) >= 0.0,
                int(candidate.evaluation["f1_nonregressing_fold_count"]) < 3,
            ),
            "strict_opportunity_pool_recall": (
                float(candidate.diagnostics["strict_opportunity_pool_recall"]) >= 0.95,
                int(candidate.diagnostics["folds_meeting_pool_recall_minimum"]) < 3,
            ),
            "conditional_ranker_strict_capture": (
                float(candidate.diagnostics["conditional_ranker_strict_capture"]) >= 0.68,
                int(candidate.diagnostics["folds_meeting_conditional_capture_minimum"]) < 3,
            ),
            "unsafe_selection_rate": (
                float(candidate.diagnostics["unsafe_selection_rate"]) <= 0.25,
                int(candidate.diagnostics["folds_meeting_unsafe_rate_maximum"]) < 3,
            ),
        }
        for metric, (aggregate_passed, fold_failed) in aggregate_pairs.items():
            self._aggregate_pass_fold_failure[metric] += int(aggregate_passed and fold_failed)

    def report(self) -> dict[str, Any]:
        constraints = {
            name: totals.report(self._outer_cell_count)
            for name, totals in self._constraint_totals.items()
        }
        recommendation = _research_recommendation(constraints, self._question_totals.report())
        return {
            "near_boundary_contract": {
                "selection": "A",
                "confirmation_source": "user_confirmation_after_stage200_freeze",
                "count_constraint_failed_margin_tolerance": 1.0,
                "rate_constraint_failed_margin_tolerance": 0.01,
                "mean_f1_delta_failed_margin_tolerance": 0.001,
                "stage200_artifact_modified": False,
            },
            "population": {
                "outer_context_count": self._snapshot_count,
                "outer_cell_context_count": self._outer_cell_count,
                "fold_cell_context_count": self._fold_cell_count,
                "question_cell_context_count": self._question_totals.context_count,
            },
            "constraint_attribution": {
                "constraints": constraints,
                "failed_constraint_set_counts": dict(
                    sorted(self._failed_set_counts.items(), key=lambda row: (-row[1], row[0]))
                ),
                "failed_constraint_count_distribution": dict(
                    sorted(self._failed_count_distribution.items(), key=lambda row: int(row[0]))
                ),
                "single_constraint_removal_pass_counts": {
                    name: self._single_removal_pass_counts[name] for name in self._constraint_totals
                },
                "cofailure": _cofailure_report(self._constraint_totals, self._cofailure_counts),
                "pareto_nondominated_cell_count_by_outer_context": dict(
                    sorted(self._pareto_counts_by_outer.items())
                ),
                "pareto_nondominated_cell_count_total": sum(self._pareto_counts_by_outer.values()),
            },
            "fold_attribution": {
                "violation_counts_by_metric_and_fold": {
                    metric: dict(sorted(counts.items()))
                    for metric, counts in sorted(self._fold_violations.items())
                },
                "worst_fold_frequency_by_metric": {
                    metric: dict(sorted(counts.items()))
                    for metric, counts in sorted(self._worst_fold_frequency.items())
                },
                "cross_fold_range_distribution_by_metric": {
                    metric: _distribution(values)
                    for metric, values in sorted(self._cross_fold_ranges.items())
                },
                "cross_fold_stddev_distribution_by_metric": {
                    metric: _distribution(values)
                    for metric, values in sorted(self._cross_fold_stddevs.items())
                },
                "aggregate_pass_but_fold_count_fail_by_metric": dict(
                    sorted(self._aggregate_pass_fold_failure.items())
                ),
            },
            "question_context_attribution": {
                "aggregate": self._question_totals.report(),
                "by_outer_context": {
                    name: totals.report()
                    for name, totals in sorted(self._question_by_outer.items())
                },
                "by_risk_signal": {
                    name: totals.report() for name, totals in sorted(self._question_by_risk.items())
                },
                "by_winner_rule": {
                    name: totals.report()
                    for name, totals in sorted(self._question_by_winner.items())
                },
            },
            "factor_attribution": {
                dimension: {
                    value: {
                        "cell_count": count,
                        "ineligible_cell_count": self._factor_failure_counts[dimension][value],
                    }
                    for value, count in sorted(counts.items())
                }
                for dimension, counts in self._factor_cell_counts.items()
            },
            "diagnostic_finding": recommendation,
            "privacy_contract": {
                "question_level_rows_persisted": False,
                "private_decisions_persisted": False,
                "streaming_aggregate_used": True,
            },
        }


def _resolve_candidate_value(
    candidate: JointRiskWinnerCandidateSnapshot, source: str
) -> int | float:
    root_name, key = source.split(".", maxsplit=1)
    root = candidate.evaluation if root_name == "evaluation" else candidate.diagnostics
    return root[key]


def _constraint_threshold(constraint: Mapping[str, Any], *, inner_question_count: int) -> float:
    expression = constraint.get("threshold_expression")
    if expression == "ceil(0.10 * inner_question_count)":
        return float(math.ceil(0.10 * inner_question_count))
    if expression == "ceil(0.08 * inner_question_count)":
        return float(math.ceil(0.08 * inner_question_count))
    return float(constraint["threshold"])


def _signed_margin(observed: float, threshold: float, operator: str) -> float:
    if operator == ">=":
        return observed - threshold
    if operator == "<=":
        return threshold - observed
    raise ValueError(f"Unsupported Stage201 constraint operator: {operator}")


def _near_tolerance(name: str) -> float:
    if name in _COUNT_NEAR_BOUNDARY:
        return 1.0
    if name == "mean_f1_delta":
        return 0.001
    return 0.01


def _selected_outcome(decision: SafetyFirstFrontierDecision) -> str:
    row = decision.winner.row
    if decision.winner == decision.baseline:
        return "baseline"
    if row.strict_expected:
        return "strict_success"
    if stage194._is_safe_zero(row):
        return "safe_zero"
    citation = row.citation_delta < 0
    f1 = row.f1_delta < -_F1_TOLERANCE
    if citation and f1:
        return "unsafe_citation_and_f1"
    if citation:
        return "unsafe_citation_only"
    if f1:
        return "unsafe_f1_only"
    raise ValueError("Stage201 selected outcome is outside the frozen exact partition")


def _opportunity_mechanism(decision: SafetyFirstFrontierDecision) -> str:
    if decision.winner.row.strict_expected:
        return "strict_selected"
    if not decision.strict_opportunity:
        return "no_strict_opportunity"
    if not any(row.row.strict_expected for row in decision.complete_pool):
        return "safety_pool_exclusion"
    if not any(row.row.strict_expected for row in decision.frontier):
        return "risk_frontier_exclusion"
    return "winner_selection_miss"


def _rank(rows: Sequence[FrontierActionPrediction], target: FrontierActionPrediction) -> int:
    return rows.index(target) + 1


def _pareto_count(candidates: Sequence[JointRiskWinnerCandidateSnapshot]) -> int:
    points = [
        (
            float(row.diagnostics["conditional_ranker_strict_capture"]),
            float(row.diagnostics["unsafe_selection_rate"]),
        )
        for row in candidates
    ]
    return sum(
        not any(
            other_capture >= capture
            and other_unsafe <= unsafe
            and (other_capture > capture or other_unsafe < unsafe)
            for other_index, (other_capture, other_unsafe) in enumerate(points)
            if other_index != index
        )
        for index, (capture, unsafe) in enumerate(points)
    )


def _cofailure_report(
    totals: Mapping[str, _ConstraintAccumulator],
    counts: Mapping[tuple[str, str], int],
) -> dict[str, Any]:
    rows = []
    names = tuple(totals)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            count = counts.get((left, right), counts.get((right, left), 0))
            union = totals[left].failure_count + totals[right].failure_count - count
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "cofailure_count": count,
                    "jaccard": _ratio(count, union),
                }
            )
    return {
        "pair_count": len(rows),
        "pairs": sorted(rows, key=lambda row: (-row["cofailure_count"], row["left"], row["right"])),
    }


def _research_recommendation(
    constraints: Mapping[str, Mapping[str, Any]],
    questions: Mapping[str, Any],
) -> dict[str, Any]:
    groups = {
        "model_research": (
            "strict_opportunity_pool_recall",
            "folds_meeting_pool_recall_minimum",
            "conditional_ranker_strict_capture",
            "folds_meeting_conditional_capture_minimum",
        ),
        "objective_research": (
            "strict_success_precision",
            "unsafe_selection_rate",
            "folds_meeting_unsafe_rate_maximum",
        ),
        "representation_research": (
            "citation_delta",
            "mean_f1_delta",
            "citation_nonregressing_fold_count",
            "f1_nonregressing_fold_count",
        ),
    }
    scores = {
        group: sum(int(constraints[name]["failure_count"]) for name in names)
        for group, names in groups.items()
    }
    selected = min(scores, key=lambda name: (-scores[name], name))
    mechanisms = questions["strict_opportunity_mechanism_counts"]
    failure_mechanisms = {
        name: count for name, count in mechanisms.items() if name != "strict_selected"
    }
    return {
        "recommended_next_research": selected,
        "failure_count_score_by_research_axis": scores,
        "scoring_rule": "sum of failed outer-cell constraints in each frozen research axis",
        "dominant_failed_constraint": min(
            constraints,
            key=lambda name: (-int(constraints[name]["failure_count"]), name),
        ),
        "dominant_question_partition": min(
            mechanisms,
            key=lambda name: (-int(mechanisms[name]), name),
        ),
        "dominant_failure_mechanism": min(
            failure_mechanisms,
            key=lambda name: (-int(failure_mechanisms[name]), name),
        ),
        "causal_claim": False,
        "new_policy_selected": False,
        "constraint_relaxed": False,
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "minimum": round(ordered[0], 6) if ordered else 0.0,
        "q25": round(_quantile(ordered, 0.25), 6) if ordered else 0.0,
        "median": round(_quantile(ordered, 0.50), 6) if ordered else 0.0,
        "q75": round(_quantile(ordered, 0.75), 6) if ordered else 0.0,
        "maximum": round(ordered[-1], 6) if ordered else 0.0,
        "mean": round(statistics.fmean(ordered), 6) if ordered else 0.0,
    }


def _quantile(ordered: Sequence[float], probability: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator / denominator), 6) if denominator else 0.0
