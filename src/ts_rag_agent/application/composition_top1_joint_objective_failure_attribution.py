from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from ts_rag_agent.application import composition_safety_constrained_lambdamart as stage194
from ts_rag_agent.application.composition_safety_first_frontier import (
    SafetyFirstFrontierDecision,
)
from ts_rag_agent.application.composition_top1_joint_objective_cv import (
    Top1ObjectiveCandidateSnapshot,
    Top1ObjectiveDiagnosticSnapshot,
)

_CONTROL_NAME = "stage196_exact_control"
_SAFETY_WEIGHTS = (0.0, 0.5, 1.0, 2.0)
_PRECISION_WEIGHTS = (0.0, 0.5, 1.0, 2.0)
_OUTCOMES = (
    "baseline",
    "strict_success",
    "safe_zero",
    "unsafe_citation_only",
    "unsafe_f1_only",
    "unsafe_citation_and_f1",
)


@dataclass
class _TransitionAccumulator:
    context_count: int = 0
    same_winner_count: int = 0
    left_outcomes: Counter[str] = field(default_factory=Counter)
    right_outcomes: Counter[str] = field(default_factory=Counter)
    transitions: Counter[str] = field(default_factory=Counter)
    strict_gain_count: int = 0
    strict_loss_count: int = 0
    unsafe_repair_count: int = 0
    unsafe_regression_count: int = 0
    baseline_addition_count: int = 0
    baseline_removal_count: int = 0
    safe_zero_addition_count: int = 0
    safe_zero_removal_count: int = 0

    def add(
        self,
        left: SafetyFirstFrontierDecision,
        right: SafetyFirstFrontierDecision,
    ) -> None:
        _validate_decision_pair(left, right)
        left_row = left.winner.row
        right_row = right.winner.row
        left_outcome = _selected_outcome(left)
        right_outcome = _selected_outcome(right)
        left_strict = bool(left_row.strict_expected)
        right_strict = bool(right_row.strict_expected)
        left_unsafe = stage194._is_unsafe(left_row)
        right_unsafe = stage194._is_unsafe(right_row)
        left_baseline = left_row.action.family == "baseline"
        right_baseline = right_row.action.family == "baseline"
        left_safe_zero = stage194._is_safe_zero(left_row)
        right_safe_zero = stage194._is_safe_zero(right_row)

        self.context_count += 1
        self.same_winner_count += int(stage194._row_key(left_row) == stage194._row_key(right_row))
        self.left_outcomes[left_outcome] += 1
        self.right_outcomes[right_outcome] += 1
        self.transitions[f"{left_outcome}__to__{right_outcome}"] += 1
        self.strict_gain_count += int(not left_strict and right_strict)
        self.strict_loss_count += int(left_strict and not right_strict)
        self.unsafe_repair_count += int(left_unsafe and not right_unsafe)
        self.unsafe_regression_count += int(not left_unsafe and right_unsafe)
        self.baseline_addition_count += int(not left_baseline and right_baseline)
        self.baseline_removal_count += int(left_baseline and not right_baseline)
        self.safe_zero_addition_count += int(not left_safe_zero and right_safe_zero)
        self.safe_zero_removal_count += int(left_safe_zero and not right_safe_zero)

    def report(self) -> dict[str, Any]:
        left = {name: self.left_outcomes[name] for name in _OUTCOMES}
        right = {name: self.right_outcomes[name] for name in _OUTCOMES}
        return {
            "context_count": self.context_count,
            "same_winner_count": self.same_winner_count,
            "winner_flip_count": self.context_count - self.same_winner_count,
            "winner_flip_rate": _ratio(
                self.context_count - self.same_winner_count, self.context_count
            ),
            "left_outcome_counts": left,
            "right_outcome_counts": right,
            "left_partition_exact": sum(left.values()) == self.context_count,
            "right_partition_exact": sum(right.values()) == self.context_count,
            "transition_counts": dict(
                sorted(self.transitions.items(), key=lambda row: (-row[1], row[0]))
            ),
            "transition_partition_exact": sum(self.transitions.values()) == self.context_count,
            "strict_gain_count": self.strict_gain_count,
            "strict_loss_count": self.strict_loss_count,
            "net_strict_count": self.strict_gain_count - self.strict_loss_count,
            "unsafe_repair_count": self.unsafe_repair_count,
            "unsafe_regression_count": self.unsafe_regression_count,
            "net_unsafe_count": self.unsafe_regression_count - self.unsafe_repair_count,
            "baseline_addition_count": self.baseline_addition_count,
            "baseline_removal_count": self.baseline_removal_count,
            "net_baseline_count": self.baseline_addition_count - self.baseline_removal_count,
            "safe_zero_addition_count": self.safe_zero_addition_count,
            "safe_zero_removal_count": self.safe_zero_removal_count,
        }


@dataclass
class _DirectionalAccumulator:
    transitions: _TransitionAccumulator = field(default_factory=_TransitionAccumulator)
    strict_precision_deltas: list[float] = field(default_factory=list)
    unsafe_rate_deltas: list[float] = field(default_factory=list)
    strict_capture_deltas: list[float] = field(default_factory=list)

    def add(
        self,
        left: Top1ObjectiveCandidateSnapshot,
        right: Top1ObjectiveCandidateSnapshot,
    ) -> None:
        for left_decision, right_decision in _aligned_decisions(left, right):
            self.transitions.add(left_decision, right_decision)
        self.strict_precision_deltas.append(
            float(right.evaluation["strict_success_precision"])
            - float(left.evaluation["strict_success_precision"])
        )
        self.unsafe_rate_deltas.append(
            float(right.diagnostics["unsafe_selection_rate"])
            - float(left.diagnostics["unsafe_selection_rate"])
        )
        self.strict_capture_deltas.append(
            float(right.diagnostics["conditional_ranker_strict_capture"])
            - float(left.diagnostics["conditional_ranker_strict_capture"])
        )

    def report(self) -> dict[str, Any]:
        return {
            **self.transitions.report(),
            "strict_precision_delta": _distribution(self.strict_precision_deltas),
            "unsafe_rate_delta": _distribution(self.unsafe_rate_deltas),
            "strict_capture_delta": _distribution(self.strict_capture_deltas),
            "strict_precision_nondecreasing_cell_count": sum(
                value >= 0.0 for value in self.strict_precision_deltas
            ),
            "unsafe_rate_nonincreasing_cell_count": sum(
                value <= 0.0 for value in self.unsafe_rate_deltas
            ),
            "strict_capture_nondecreasing_cell_count": sum(
                value >= 0.0 for value in self.strict_capture_deltas
            ),
            "cell_comparison_count": len(self.strict_precision_deltas),
        }


@dataclass
class _TargetMechanicsAccumulator:
    context_count: int = 0
    pool_sizes: list[float] = field(default_factory=list)
    strict_counts: list[float] = field(default_factory=list)
    nonunsafe_counts: list[float] = field(default_factory=list)
    strict_count_frequency: Counter[str] = field(default_factory=Counter)
    no_strict_count: int = 0
    precision_baseline_masses: list[float] = field(default_factory=list)
    precision_strict_masses: list[float] = field(default_factory=list)
    safety_baseline_masses: list[float] = field(default_factory=list)

    def add(self, decision: SafetyFirstFrontierDecision) -> None:
        rows = tuple(item.row for item in decision.complete_pool)
        strict_count = sum(row.strict_expected for row in rows)
        nonunsafe_count = sum(not stage194._is_unsafe(row) for row in rows)
        if nonunsafe_count < 1:
            raise ValueError("Stage204 target mechanics require a non-unsafe baseline")
        self.context_count += 1
        self.pool_sizes.append(float(len(rows)))
        self.strict_counts.append(float(strict_count))
        self.nonunsafe_counts.append(float(nonunsafe_count))
        self.strict_count_frequency[str(strict_count)] += 1
        self.no_strict_count += int(strict_count == 0)
        denominator = strict_count + 1
        self.precision_baseline_masses.append(1.0 / denominator)
        self.precision_strict_masses.append(strict_count / denominator)
        self.safety_baseline_masses.append(1.0 / nonunsafe_count)

    def report(self) -> dict[str, Any]:
        return {
            "question_context_count": self.context_count,
            "no_strict_opportunity_count": self.no_strict_count,
            "strict_opportunity_count": self.context_count - self.no_strict_count,
            "strict_count_frequency": dict(
                sorted(self.strict_count_frequency.items(), key=lambda row: int(row[0]))
            ),
            "pool_size": _distribution(self.pool_sizes),
            "strict_action_count": _distribution(self.strict_counts),
            "nonunsafe_action_count": _distribution(self.nonunsafe_counts),
            "precision_component_baseline_mass": _distribution(self.precision_baseline_masses),
            "precision_component_total_strict_mass": _distribution(self.precision_strict_masses),
            "safety_component_baseline_mass": _distribution(self.safety_baseline_masses),
            "precision_component_mass_sum_exact": all(
                abs(left + right - 1.0) <= 1e-12
                for left, right in zip(
                    self.precision_baseline_masses,
                    self.precision_strict_masses,
                    strict=True,
                )
            ),
        }


class Top1JointObjectiveFailureAttributor:
    """Stream Stage203 private snapshots into public aggregate flip evidence."""

    def __init__(self) -> None:
        self._snapshot_count = 0
        self._outer_ids: set[str] = set()
        self._outer_cell_count = 0
        self._custom_cell_count = 0
        self._question_context_count = 0
        self._aggregate = _TransitionAccumulator()
        self._by_candidate: dict[str, _TransitionAccumulator] = defaultdict(_TransitionAccumulator)
        self._by_outer: dict[str, _TransitionAccumulator] = defaultdict(_TransitionAccumulator)
        self._by_safety_weight: dict[str, _TransitionAccumulator] = defaultdict(
            _TransitionAccumulator
        )
        self._by_precision_weight: dict[str, _TransitionAccumulator] = defaultdict(
            _TransitionAccumulator
        )
        self._candidate_specs: dict[str, dict[str, Any]] = {}
        self._precision_adjacent = _DirectionalAccumulator()
        self._precision_pairs: dict[str, _DirectionalAccumulator] = defaultdict(
            _DirectionalAccumulator
        )
        self._safety_adjacent = _DirectionalAccumulator()
        self._safety_pairs: dict[str, _DirectionalAccumulator] = defaultdict(
            _DirectionalAccumulator
        )
        self._targets = _TargetMechanicsAccumulator()

    def __call__(self, snapshot: Top1ObjectiveDiagnosticSnapshot) -> None:
        if len(snapshot.candidates) != 17:
            raise ValueError("Stage204 requires all 17 Stage203 cells per outer context")
        if snapshot.outer_fold_id in self._outer_ids:
            raise ValueError(f"Duplicate Stage204 outer context {snapshot.outer_fold_id}")
        candidates = {str(row.spec["name"]): row for row in snapshot.candidates}
        if len(candidates) != 17 or _CONTROL_NAME not in candidates:
            raise ValueError("Stage204 requires unique Stage203 cells and exact control")
        control = candidates[_CONTROL_NAME]
        if len(control.decisions) != snapshot.inner_question_count:
            raise ValueError("Stage204 control decision count differs from inner questions")

        self._snapshot_count += 1
        self._outer_ids.add(snapshot.outer_fold_id)
        self._outer_cell_count += len(candidates)
        self._question_context_count += snapshot.inner_question_count
        for decision in control.decisions:
            self._targets.add(decision)

        for name, candidate in sorted(candidates.items()):
            if name == _CONTROL_NAME:
                continue
            self._custom_cell_count += 1
            spec = dict(candidate.spec)
            self._candidate_specs[name] = spec
            safety_key = _weight_key(float(spec["safety_weight"]))
            precision_key = _weight_key(float(spec["precision_weight"]))
            for left, right in _aligned_decisions(control, candidate):
                self._aggregate.add(left, right)
                self._by_candidate[name].add(left, right)
                self._by_outer[snapshot.outer_fold_id].add(left, right)
                self._by_safety_weight[safety_key].add(left, right)
                self._by_precision_weight[precision_key].add(left, right)

        for safety_weight in _SAFETY_WEIGHTS:
            for lower, higher in zip(_PRECISION_WEIGHTS, _PRECISION_WEIGHTS[1:], strict=False):
                left = candidates[_objective_name(safety_weight, lower)]
                right = candidates[_objective_name(safety_weight, higher)]
                key = f"safety_{safety_weight:.2f}__precision_{lower:.2f}_to_{higher:.2f}"
                self._precision_adjacent.add(left, right)
                self._precision_pairs[key].add(left, right)

        for precision_weight in _PRECISION_WEIGHTS:
            for lower, higher in zip(_SAFETY_WEIGHTS, _SAFETY_WEIGHTS[1:], strict=False):
                left = candidates[_objective_name(lower, precision_weight)]
                right = candidates[_objective_name(higher, precision_weight)]
                key = f"precision_{precision_weight:.2f}__safety_{lower:.2f}_to_{higher:.2f}"
                self._safety_adjacent.add(left, right)
                self._safety_pairs[key].add(left, right)

    def report(self) -> dict[str, Any]:
        aggregate = self._aggregate.report()
        precision = self._precision_adjacent.report()
        safety = self._safety_adjacent.report()
        return {
            "confirmation_contract": {
                "selection": "A",
                "confirmation_source": "user_confirmation_after_stage203",
                "same_weight_grid_reopened": False,
                "new_model_search_run": False,
                "constraint_relaxation_run": False,
            },
            "population": {
                "outer_context_count": self._snapshot_count,
                "outer_cell_context_count": self._outer_cell_count,
                "custom_outer_cell_context_count": self._custom_cell_count,
                "question_context_count": self._question_context_count,
                "control_custom_question_comparison_count": aggregate["context_count"],
                "precision_adjacent_question_comparison_count": precision["context_count"],
                "safety_adjacent_question_comparison_count": safety["context_count"],
            },
            "control_to_custom": {
                "aggregate": aggregate,
                "by_candidate": {
                    name: {
                        "spec": self._candidate_specs[name],
                        **totals.report(),
                    }
                    for name, totals in sorted(self._by_candidate.items())
                },
                "by_outer_context": {
                    name: totals.report() for name, totals in sorted(self._by_outer.items())
                },
                "by_safety_weight": {
                    name: totals.report() for name, totals in sorted(self._by_safety_weight.items())
                },
                "by_precision_weight": {
                    name: totals.report()
                    for name, totals in sorted(self._by_precision_weight.items())
                },
            },
            "precision_adjacent_attribution": {
                "aggregate": precision,
                "by_pair": {
                    name: totals.report() for name, totals in sorted(self._precision_pairs.items())
                },
            },
            "safety_adjacent_attribution": {
                "aggregate": safety,
                "by_pair": {
                    name: totals.report() for name, totals in sorted(self._safety_pairs.items())
                },
            },
            "target_mechanics": self._targets.report(),
            "diagnostic_finding": _diagnostic_finding(aggregate, precision, safety),
            "privacy_contract": {
                "question_level_rows_persisted": False,
                "private_decisions_persisted": False,
                "streaming_aggregate_used": True,
            },
        }


def _aligned_decisions(
    left: Top1ObjectiveCandidateSnapshot,
    right: Top1ObjectiveCandidateSnapshot,
) -> tuple[tuple[SafetyFirstFrontierDecision, SafetyFirstFrontierDecision], ...]:
    left_index = _decision_index(left.decisions)
    right_index = _decision_index(right.decisions)
    if set(left_index) != set(right_index):
        raise ValueError("Stage204 candidate question sets differ")
    return tuple((left_index[key], right_index[key]) for key in sorted(left_index))


def _decision_index(
    decisions: tuple[SafetyFirstFrontierDecision, ...],
) -> dict[str, SafetyFirstFrontierDecision]:
    index = {row.question_key: row for row in decisions}
    if len(index) != len(decisions):
        raise ValueError("Stage204 observed duplicate question decisions")
    return index


def _validate_decision_pair(
    left: SafetyFirstFrontierDecision,
    right: SafetyFirstFrontierDecision,
) -> None:
    if left.question_key != right.question_key:
        raise ValueError("Stage204 compared different questions")
    left_pool = {stage194._row_key(item.row) for item in left.complete_pool}
    right_pool = {stage194._row_key(item.row) for item in right.complete_pool}
    if left_pool != right_pool:
        raise ValueError("Stage204 requires an identical candidate pool for each comparison")


def _selected_outcome(decision: SafetyFirstFrontierDecision) -> str:
    row = decision.winner.row
    if row.action.family == "baseline":
        return "baseline"
    if row.strict_expected:
        return "strict_success"
    if stage194._is_safe_zero(row):
        return "safe_zero"
    citation_loss = row.citation_delta < 0
    f1_loss = row.f1_delta < 0.0
    if citation_loss and f1_loss:
        return "unsafe_citation_and_f1"
    if citation_loss:
        return "unsafe_citation_only"
    if f1_loss:
        return "unsafe_f1_only"
    raise ValueError("Stage204 selected outcome is outside the frozen partition")


def _objective_name(safety_weight: float, precision_weight: float) -> str:
    return f"top1_safety_{safety_weight:.2f}__precision_{precision_weight:.2f}"


def _weight_key(value: float) -> str:
    return f"{value:.2f}"


def _diagnostic_finding(
    aggregate: dict[str, Any],
    precision: dict[str, Any],
    safety: dict[str, Any],
) -> dict[str, Any]:
    transitions = precision["transition_counts"]
    most_frequent_precision_transition = next(iter(transitions)) if transitions else "none"
    changed_transitions = {
        name: count
        for name, count in transitions.items()
        if len(parts := name.split("__to__", maxsplit=1)) == 2 and parts[0] != parts[1]
    }
    dominant_precision_outcome_change = (
        max(changed_transitions, key=lambda name: (changed_transitions[name], name))
        if changed_transitions
        else "none"
    )
    precision_displaces_more_strict_than_it_recovers = (
        precision["strict_loss_count"] > precision["strict_gain_count"]
    )
    safety_repairs_more_than_it_regresses = (
        safety["unsafe_repair_count"] > safety["unsafe_regression_count"]
    )
    recommendation = (
        "separate_change_abstain_head_and_conditional_strict_ranker_protocol"
        if precision_displaces_more_strict_than_it_recovers
        else "recalibrate_grouped_objective_before_new_model_search"
    )
    return {
        "control_custom_net_strict_count": aggregate["net_strict_count"],
        "control_custom_net_unsafe_count": aggregate["net_unsafe_count"],
        "precision_displaces_more_strict_than_it_recovers": (
            precision_displaces_more_strict_than_it_recovers
        ),
        "safety_repairs_more_than_it_regresses": safety_repairs_more_than_it_regresses,
        "most_frequent_precision_transition": most_frequent_precision_transition,
        "dominant_precision_outcome_change": dominant_precision_outcome_change,
        "recommended_next_research": recommendation,
        "finding_is_causal_claim": False,
    }


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "minimum": 0.0, "mean": 0.0, "maximum": 0.0, "stddev": 0.0}
    return {
        "count": len(values),
        "minimum": round(min(values), 6),
        "mean": round(statistics.fmean(values), 6),
        "maximum": round(max(values), 6),
        "stddev": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0
