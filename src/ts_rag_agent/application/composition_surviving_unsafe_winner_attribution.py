from __future__ import annotations

import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ts_rag_agent.application import composition_safety_constrained_lambdamart as stage194
from ts_rag_agent.application.composition_action_audit import ActionAuditRow
from ts_rag_agent.application.composition_dual_target_policy import SelectedAction
from ts_rag_agent.application.composition_f1_representation_cv import (
    build_composition_feature_indices,
)
from ts_rag_agent.application.composition_gain_sensitive_ranking import (
    _binary_metrics,
    build_stage182_reference_rows,
)
from ts_rag_agent.application.composition_joint_constraint_ranking import (
    evaluate_selected_actions,
)
from ts_rag_agent.application.composition_safety_first_frontier import (
    SafetyFirstFrontierDecision,
    _spec_from_dict,
    evaluate_safety_first_frontier_policy,
    fit_predict_safety_first_spec,
)

ProgressSink = Callable[[Mapping[str, Any]], None]
_F1_TOLERANCE = 1e-12


@dataclass
class _AttributionTotals:
    question_context_count: int = 0
    unsafe_winner_context_count: int = 0
    unsafe_with_strict_opportunity_count: int = 0
    loss_types: Counter[str] = field(default_factory=Counter)
    mechanism_counts: Counter[str] = field(default_factory=Counter)
    risk_rank_buckets: Counter[str] = field(default_factory=Counter)
    gain_rank_buckets: Counter[str] = field(default_factory=Counter)
    frontier_sizes: Counter[str] = field(default_factory=Counter)
    lower_risk_strict_alternative_count: int = 0
    safe_zero_alternative_count: int = 0
    oracle_strict_repairable_count: int = 0
    winner_minus_best_strict_gain_gaps: list[float] = field(default_factory=list)
    winner_minus_safest_strict_risk_gaps: list[float] = field(default_factory=list)

    def add(self, decision: SafetyFirstFrontierDecision) -> None:
        self.question_context_count += 1
        winner = decision.winner
        if not stage194._is_unsafe(winner.row):
            return
        self.unsafe_winner_context_count += 1
        self.loss_types[_loss_type(winner.row)] += 1
        self.frontier_sizes[str(len(decision.frontier))] += 1
        pool_risk_order = sorted(
            decision.complete_pool,
            key=lambda row: (row.unsafe_score, row.row.action.action_id),
        )
        frontier_gain_order = sorted(
            decision.frontier,
            key=lambda row: (-row.gain_score, row.unsafe_score, row.row.action.action_id),
        )
        risk_rank = pool_risk_order.index(winner) + 1
        gain_rank = frontier_gain_order.index(winner) + 1
        self.risk_rank_buckets[_rank_bucket(risk_rank)] += 1
        self.gain_rank_buckets[_gain_rank_bucket(gain_rank)] += 1
        self.safe_zero_alternative_count += int(
            any(stage194._is_safe_zero(row.row) for row in decision.frontier)
        )

        if not decision.strict_opportunity:
            self.mechanism_counts["no_strict_opportunity"] += 1
            return
        self.unsafe_with_strict_opportunity_count += 1
        pool_strict = [row for row in decision.complete_pool if row.row.strict_expected]
        frontier_strict = [row for row in decision.frontier if row.row.strict_expected]
        if not pool_strict:
            self.mechanism_counts["safety_pool_exclusion"] += 1
            return
        if not frontier_strict:
            self.mechanism_counts["risk_frontier_exclusion"] += 1
            return

        self.oracle_strict_repairable_count += 1
        best_gain_strict = min(
            frontier_strict,
            key=lambda row: (-row.gain_score, row.unsafe_score, row.row.action.action_id),
        )
        safest_strict = min(
            frontier_strict,
            key=lambda row: (row.unsafe_score, -row.gain_score, row.row.action.action_id),
        )
        lower_risk_strict = any(row.unsafe_score < winner.unsafe_score for row in frontier_strict)
        self.lower_risk_strict_alternative_count += int(lower_risk_strict)
        mechanism = "final_gain_dominance" if lower_risk_strict else "risk_ordering_failure"
        self.mechanism_counts[mechanism] += 1
        self.winner_minus_best_strict_gain_gaps.append(
            float(winner.gain_score - best_gain_strict.gain_score)
        )
        self.winner_minus_safest_strict_risk_gaps.append(
            float(winner.unsafe_score - safest_strict.unsafe_score)
        )

    def report(self) -> dict[str, Any]:
        mechanisms = dict(sorted(self.mechanism_counts.items()))
        dominant = (
            min(
                mechanisms,
                key=lambda name: (-mechanisms[name], name),
            )
            if mechanisms
            else "none"
        )
        return {
            "question_context_count": self.question_context_count,
            "unsafe_winner_context_count": self.unsafe_winner_context_count,
            "unsafe_winner_rate": _ratio(
                self.unsafe_winner_context_count, self.question_context_count
            ),
            "unsafe_with_strict_opportunity_count": self.unsafe_with_strict_opportunity_count,
            "loss_type_counts": dict(sorted(self.loss_types.items())),
            "mechanism_counts": mechanisms,
            "mechanism_partition_total": sum(mechanisms.values()),
            "mechanism_partition_exact": (
                sum(mechanisms.values()) == self.unsafe_winner_context_count
            ),
            "dominant_mechanism": dominant,
            "risk_rank_bucket_counts": _complete_buckets(
                self.risk_rank_buckets, ("1", "2", "3-4", "5-8", "9+")
            ),
            "gain_rank_bucket_counts": _complete_buckets(
                self.gain_rank_buckets, ("1", "2", "3-4", "5+")
            ),
            "lower_risk_strict_alternative_count": self.lower_risk_strict_alternative_count,
            "lower_risk_strict_alternative_rate": _ratio(
                self.lower_risk_strict_alternative_count,
                self.unsafe_with_strict_opportunity_count,
            ),
            "safe_zero_alternative_count": self.safe_zero_alternative_count,
            "oracle_strict_repairable_count": self.oracle_strict_repairable_count,
            "oracle_strict_repairable_rate": _ratio(
                self.oracle_strict_repairable_count,
                self.unsafe_with_strict_opportunity_count,
            ),
            "frontier_size_counts": dict(sorted(self.frontier_sizes.items())),
            "score_gap_summaries": {
                "winner_minus_best_strict_gain": _distribution(
                    self.winner_minus_best_strict_gain_gaps
                ),
                "winner_minus_safest_strict_risk": _distribution(
                    self.winner_minus_safest_strict_risk_gaps
                ),
            },
        }


def run_surviving_unsafe_winner_attribution(
    *,
    action_rows: Sequence[ActionAuditRow],
    stage182_selected_actions: Sequence[SelectedAction],
    stage196_report: Mapping[str, Any],
    progress_sink: ProgressSink | None = None,
) -> dict[str, Any]:
    """Rebuild the five published top-inner specs and attribute unsafe winners."""

    started_at = time.perf_counter()
    rows = tuple(action_rows)
    fold_ids = tuple(sorted({row.fold_id for row in rows}))
    if len(fold_ids) != 5:
        raise ValueError("Stage197 requires exactly five frozen folds")
    references = build_stage182_reference_rows(rows, stage182_selected_actions)
    base_features = build_composition_feature_indices(rows)
    feature_indices = {
        "raw_runtime": base_features["raw"],
        "question_relative_runtime": base_features["question_relative"],
    }
    formal_outer = stage196_report["safety_first_frontier_nested_cv"]["outer_folds"]
    totals = _AttributionTotals()
    outer_reports: dict[str, Any] = {}
    risk_labels: list[int] = []
    risk_scores: list[float] = []
    execution = Counter()
    feature_counts: dict[str, int] = {}

    for outer_fold_id in fold_ids:
        formal_top = formal_outer[outer_fold_id]["top_inner_candidates"][0]
        spec = _spec_from_dict(formal_top["spec"])
        outer_training = tuple(row for row in rows if row.fold_id != outer_fold_id)
        inner_fold_ids = tuple(fold for fold in fold_ids if fold != outer_fold_id)
        safety_predictions = []
        gain_predictions = []
        risk_predictions = []
        for inner_fold_id in inner_fold_ids:
            training = tuple(row for row in outer_training if row.fold_id != inner_fold_id)
            heldout = tuple(row for row in outer_training if row.fold_id == inner_fold_id)
            result = fit_predict_safety_first_spec(
                training,
                heldout,
                feature_indices,
                spec,
            )
            safety_predictions.extend(result.safety_predictions)
            gain_predictions.extend(result.gain_predictions)
            risk_predictions.extend(result.risk_predictions)
            execution["model_fit_count"] += result.model_fit_count
            execution["pool_safety_fit_count"] += 2
            execution["lambdamart_fit_count"] += 1
            execution["unsafe_head_fit_count"] += 1
            execution["tree_count"] += result.tree_count
            execution["group_contract_validation_count"] += result.group_contract_validation_count
            execution["private_prediction_count"] += 4 * len(heldout)
            execution["partition_count"] += 1
            for name, count in result.feature_count_by_representation.items():
                feature_counts[name] = max(feature_counts.get(name, 0), count)
            _emit(
                progress_sink,
                phase="stage197_inner_partition_complete",
                outer_fold_id=outer_fold_id,
                inner_fold_id=inner_fold_id,
                cumulative_model_fit_count=execution["model_fit_count"],
                cumulative_tree_count=execution["tree_count"],
            )

        decisions, diagnostics = evaluate_safety_first_frontier_policy(
            safety_predictions,
            gain_predictions,
            risk_predictions,
            spec,
            expected_fold_ids=inner_fold_ids,
        )
        evaluation = evaluate_selected_actions(
            selected_rows=tuple(decision.winner.row for decision in decisions),
            references=references,
            expected_fold_ids=inner_fold_ids,
        )
        reconstruction_exact = _nested_close(
            evaluation, formal_top["evaluation"]
        ) and _nested_close(diagnostics, formal_top["diagnostics"])
        if not reconstruction_exact:
            raise ValueError(f"Stage197 did not reproduce {outer_fold_id} top-inner evidence")
        fold_totals = _AttributionTotals()
        for decision in decisions:
            fold_totals.add(decision)
            totals.add(decision)
        risk_labels.extend(int(stage194._is_unsafe(row.row)) for row in risk_predictions)
        risk_scores.extend(row.score for row in risk_predictions)
        outer_reports[outer_fold_id] = {
            "spec": formal_top["spec"],
            "inner_fold_ids": list(inner_fold_ids),
            "top_inner_reconstruction_exact": True,
            "top_inner_evaluation": evaluation,
            "top_inner_diagnostics": diagnostics,
            "unsafe_winner_attribution": fold_totals.report(),
        }
        _emit(
            progress_sink,
            phase="stage197_outer_context_complete",
            outer_fold_id=outer_fold_id,
            unsafe_winner_context_count=fold_totals.unsafe_winner_context_count,
        )

    aggregate = totals.report()
    recommendation = _recommendation(aggregate["dominant_mechanism"])
    return {
        "protocol": {
            "diagnostic_population": "Stage196 top-inner train-only OOF question contexts",
            "outer_context_count": 5,
            "inner_partition_count": 20,
            "published_top_spec_per_outer_context": 1,
            "models_per_partition": 4,
            "oracle_scope": "offline attribution upper bound only",
            "fallback_enabled": False,
        },
        "outer_contexts": dict(sorted(outer_reports.items())),
        "aggregate": aggregate,
        "unsafe_head_prediction_metrics": {
            "action_context_count": len(risk_labels),
            **_binary_metrics(risk_labels, risk_scores),
        },
        "diagnostic_finding": {
            "dominant_mechanism": aggregate["dominant_mechanism"],
            "recommended_next_focus": recommendation,
            "recommendation_basis": "largest exact mutually-exclusive mechanism count",
            "oracle_is_not_a_runtime_candidate": True,
        },
        "execution": {
            **dict(sorted(execution.items())),
            "feature_count_by_representation": dict(sorted(feature_counts.items())),
            "top_inner_reconstruction_count": len(outer_reports),
            "all_top_inner_reconstructions_exact": all(
                row["top_inner_reconstruction_exact"] for row in outer_reports.values()
            ),
            "public_training_rows_written": 0,
            "public_prediction_rows_written": 0,
            "wall_seconds": round(time.perf_counter() - started_at, 6),
        },
    }


def _loss_type(row: ActionAuditRow) -> str:
    citation = row.citation_delta < 0
    f1 = row.f1_delta < -_F1_TOLERANCE
    if citation and f1:
        return "citation_and_f1"
    if citation:
        return "citation_only"
    return "f1_only"


def _rank_bucket(rank: int) -> str:
    if rank <= 2:
        return str(rank)
    if rank <= 4:
        return "3-4"
    if rank <= 8:
        return "5-8"
    return "9+"


def _gain_rank_bucket(rank: int) -> str:
    if rank <= 2:
        return str(rank)
    if rank <= 4:
        return "3-4"
    return "5+"


def _complete_buckets(values: Counter[str], names: Sequence[str]) -> dict[str, int]:
    return {name: values[name] for name in names}


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": round(float(statistics.fmean(values)), 6) if values else 0.0,
        "median": round(float(statistics.median(values)), 6) if values else 0.0,
        "minimum": round(float(min(values)), 6) if values else 0.0,
        "maximum": round(float(max(values)), 6) if values else 0.0,
    }


def _recommendation(mechanism: str) -> str:
    return {
        "final_gain_dominance": "risk-aware_final_winner_rule",
        "risk_ordering_failure": "unsafe_head_discrimination",
        "risk_frontier_exclusion": "unsafe_head_discrimination",
        "safety_pool_exclusion": "safety_pool_recall",
        "no_strict_opportunity": "candidate_action_generation",
        "none": "no_change",
    }[mechanism]


def _nested_close(actual: Any, expected: Any) -> bool:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _nested_close(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, Sequence) and not isinstance(actual, (str, bytes)):
        return (
            isinstance(expected, Sequence)
            and not isinstance(expected, (str, bytes))
            and len(actual) == len(expected)
            and all(
                _nested_close(left, right) for left, right in zip(actual, expected, strict=True)
            )
        )
    if isinstance(actual, float) or isinstance(expected, float):
        return abs(float(actual) - float(expected)) <= 1e-6
    return actual == expected


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator / denominator), 6) if denominator else 0.0


def _emit(progress_sink: ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
