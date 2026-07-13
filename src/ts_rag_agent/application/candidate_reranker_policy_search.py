from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from ts_rag_agent.application.candidate_reranker_cv import (
    CandidateRerankerSelection,
    cross_validated_candidate_reranker_selections,
)


@dataclass(frozen=True)
class CandidateRerankerPolicyConfig:
    """Offline constraints for deciding whether to replace the top candidate."""

    name: str
    max_selected_rank: int
    blocked_routes: tuple[str, ...]
    min_score_margin_vs_top_candidate: float
    protect_top1_candidate_score_min: float | None


@dataclass(frozen=True)
class CandidateRerankerPolicyDecision:
    """One question-level constrained reranker decision."""

    split: str
    question_id: str
    question_route: str
    action: str
    decision_reasons: list[str]
    baseline_candidate_id: str
    baseline_candidate_rank: int
    baseline_candidate_token_f1: float
    model_candidate_id: str
    model_candidate_rank: int
    model_candidate_token_f1: float
    model_score_margin_vs_top_candidate: float
    final_candidate_id: str
    final_candidate_rank: int
    final_candidate_token_f1: float
    oracle_candidate_id: str
    oracle_candidate_rank: int
    oracle_candidate_token_f1: float
    f1_delta_vs_top_candidate: float
    final_is_gold_document: bool
    final_is_oracle_best_f1: bool
    gold_document_candidate_count: int
    final_missed_gold_document: bool
    final_deep_rank: bool


@dataclass(frozen=True)
class CandidateRerankerPolicyMetrics:
    """Aggregate metrics for one constrained reranker policy."""

    question_count: int
    baseline_average_token_f1: float
    unconstrained_model_average_token_f1: float
    policy_average_token_f1: float
    oracle_average_token_f1: float
    average_delta_vs_top_candidate: float
    unconstrained_model_delta_vs_top_candidate: float
    oracle_gap_closed_rate: float
    replacement_count: int
    replacement_rate: float
    improved_count: int
    regressed_count: int
    tied_count: int
    regressed_rate: float
    regression_reduction_vs_unconstrained: int
    final_missed_gold_document_count: int
    final_missed_gold_document_rate: float
    final_deep_rank_count: int
    final_deep_rank_rate: float
    final_oracle_best_count: int
    final_oracle_best_rate: float
    decision_reason_counts: dict[str, int]


@dataclass(frozen=True)
class CandidateRerankerPolicyEvaluation:
    """One policy's metrics and segment breakdowns."""

    config: CandidateRerankerPolicyConfig
    metrics: CandidateRerankerPolicyMetrics
    route_metrics: list[CandidateRerankerPolicyMetricsBySegment]
    selected_rank_metrics: list[CandidateRerankerPolicyMetricsBySegment]


@dataclass(frozen=True)
class CandidateRerankerPolicyMetricsBySegment:
    """Compact metrics for a route or selected-rank segment."""

    segment_name: str
    question_count: int
    replacement_count: int
    replacement_rate: float
    average_delta_vs_top_candidate: float
    regressed_count: int
    regressed_rate: float
    final_missed_gold_document_count: int
    final_missed_gold_document_rate: float


@dataclass(frozen=True)
class CandidateRerankerPolicySearchResult:
    """Full constrained reranker policy search result."""

    model_name: str
    fold_count: int
    deep_rank_min: int
    policy_count: int
    unconstrained_metrics: CandidateRerankerPolicyMetrics
    best_average_delta_policy: CandidateRerankerPolicyEvaluation
    best_regression_reduction_policy: CandidateRerankerPolicyEvaluation | None
    top_policies: list[CandidateRerankerPolicyEvaluation]
    search_space: dict[str, Any]
    analysis_scope: str


DEFAULT_MAX_SELECTED_RANK_GRID = (2, 3, 5, 10, 25)
DEFAULT_MIN_SCORE_MARGIN_GRID = (0.0, 0.05, 0.1, 0.2, 0.3)
DEFAULT_PROTECT_TOP1_CANDIDATE_SCORE_MIN_GRID = (None, 90.0, 110.0, 140.0, 170.0)
DEFAULT_BLOCKED_ROUTE_SETS = (
    (),
    ("how_to_or_lookup",),
    ("how_to_or_lookup", "limitation_or_restriction"),
    ("how_to_or_lookup", "security_bulletin_affected_product"),
)


def search_candidate_reranker_policies(
    rows: Sequence[Mapping[str, Any]],
    model_name: str = "logistic_best_candidate",
    fold_count: int = 5,
    max_selected_rank_grid: Sequence[int] = DEFAULT_MAX_SELECTED_RANK_GRID,
    min_score_margin_grid: Sequence[float] = DEFAULT_MIN_SCORE_MARGIN_GRID,
    protect_top1_candidate_score_min_grid: Sequence[
        float | None
    ] = DEFAULT_PROTECT_TOP1_CANDIDATE_SCORE_MIN_GRID,
    blocked_route_sets: Sequence[Sequence[str]] = DEFAULT_BLOCKED_ROUTE_SETS,
    deep_rank_min: int = 6,
    top_policy_limit: int = 25,
) -> CandidateRerankerPolicySearchResult:
    """Search offline constraints over grouped-CV reranker selections."""

    _validate_search_options(
        max_selected_rank_grid=max_selected_rank_grid,
        min_score_margin_grid=min_score_margin_grid,
        protect_top1_candidate_score_min_grid=protect_top1_candidate_score_min_grid,
        blocked_route_sets=blocked_route_sets,
        deep_rank_min=deep_rank_min,
        top_policy_limit=top_policy_limit,
    )
    row_index = _build_row_index(rows)
    selections = cross_validated_candidate_reranker_selections(
        rows=rows,
        model_name=model_name,
        fold_count=fold_count,
    )
    unconstrained_policy = CandidateRerankerPolicyConfig(
        name="unconstrained_model_selection",
        max_selected_rank=25,
        blocked_routes=(),
        min_score_margin_vs_top_candidate=0.0,
        protect_top1_candidate_score_min=None,
    )
    unconstrained_evaluation = _evaluate_policy(
        config=unconstrained_policy,
        selections=selections,
        row_index=row_index,
        deep_rank_min=deep_rank_min,
    )

    policy_evaluations = [
        _evaluate_policy(
            config=config,
            selections=selections,
            row_index=row_index,
            deep_rank_min=deep_rank_min,
        )
        for config in _policy_grid(
            max_selected_rank_grid=max_selected_rank_grid,
            min_score_margin_grid=min_score_margin_grid,
            protect_top1_candidate_score_min_grid=protect_top1_candidate_score_min_grid,
            blocked_route_sets=blocked_route_sets,
        )
    ]
    ranked_policies = sorted(
        policy_evaluations,
        key=_average_delta_policy_key,
        reverse=True,
    )
    regression_reduction_candidates = [
        evaluation
        for evaluation in policy_evaluations
        if (
            evaluation.metrics.average_delta_vs_top_candidate > 0
            and evaluation.metrics.regressed_count
            < unconstrained_evaluation.metrics.regressed_count
        )
    ]
    best_regression_reduction_policy = (
        max(regression_reduction_candidates, key=_regression_reduction_policy_key)
        if regression_reduction_candidates
        else None
    )

    return CandidateRerankerPolicySearchResult(
        model_name=model_name,
        fold_count=fold_count,
        deep_rank_min=deep_rank_min,
        policy_count=len(policy_evaluations),
        unconstrained_metrics=unconstrained_evaluation.metrics,
        best_average_delta_policy=ranked_policies[0],
        best_regression_reduction_policy=best_regression_reduction_policy,
        top_policies=ranked_policies[:top_policy_limit],
        search_space={
            "max_selected_rank_grid": list(max_selected_rank_grid),
            "min_score_margin_grid": list(min_score_margin_grid),
            "protect_top1_candidate_score_min_grid": list(
                protect_top1_candidate_score_min_grid
            ),
            "blocked_route_sets": [list(route_set) for route_set in blocked_route_sets],
        },
        analysis_scope=(
            "Offline grouped-CV policy search only. Constraints decide whether to "
            "replace the original top candidate in the analysis report; no runtime "
            "behavior is changed."
        ),
    )


def candidate_reranker_policy_search_to_dict(
    result: CandidateRerankerPolicySearchResult,
) -> dict[str, Any]:
    """Convert a policy-search result to a JSON-safe dictionary."""

    return asdict(result)


def evaluate_candidate_reranker_policy_from_selections(
    config: CandidateRerankerPolicyConfig,
    selections: Sequence[CandidateRerankerSelection],
    rows: Sequence[Mapping[str, Any]],
    deep_rank_min: int = 6,
) -> CandidateRerankerPolicyEvaluation:
    """Evaluate one policy from already-computed grouped-CV selections."""

    decisions = candidate_reranker_policy_decisions_from_selections(
        config=config,
        selections=selections,
        rows=rows,
        deep_rank_min=deep_rank_min,
    )
    return CandidateRerankerPolicyEvaluation(
        config=config,
        metrics=summarize_candidate_reranker_policy_decisions(
            decisions=decisions,
            selections=selections,
        ),
        route_metrics=_segment_metrics(
            decisions=decisions,
            segment_fn=lambda decision: decision.question_route,
        ),
        selected_rank_metrics=_segment_metrics(
            decisions=decisions,
            segment_fn=lambda decision: _rank_bucket_label(decision.final_candidate_rank),
        ),
    )


def candidate_reranker_policy_decisions_from_selections(
    config: CandidateRerankerPolicyConfig,
    selections: Sequence[CandidateRerankerSelection],
    rows: Sequence[Mapping[str, Any]],
    deep_rank_min: int = 6,
) -> list[CandidateRerankerPolicyDecision]:
    """Apply one constrained policy to grouped-CV selections."""

    row_index = _build_row_index(rows)
    return [
        _policy_decision(
            config=config,
            selection=selection,
            question_rows=row_index[_question_key(selection.split, selection.question_id)],
            deep_rank_min=deep_rank_min,
        )
        for selection in selections
    ]


def summarize_candidate_reranker_policy_decisions(
    decisions: Sequence[CandidateRerankerPolicyDecision],
    selections: Sequence[CandidateRerankerSelection],
) -> CandidateRerankerPolicyMetrics:
    """Summarize policy decisions against the original grouped-CV selections."""

    return _metrics(decisions=decisions, selections=selections)


def _evaluate_policy(
    config: CandidateRerankerPolicyConfig,
    selections: Sequence[CandidateRerankerSelection],
    row_index: Mapping[str, list[Mapping[str, Any]]],
    deep_rank_min: int,
) -> CandidateRerankerPolicyEvaluation:
    decisions = [
        _policy_decision(
            config=config,
            selection=selection,
            question_rows=row_index[_question_key(selection.split, selection.question_id)],
            deep_rank_min=deep_rank_min,
        )
        for selection in selections
    ]
    return CandidateRerankerPolicyEvaluation(
        config=config,
        metrics=_metrics(decisions=decisions, selections=selections),
        route_metrics=_segment_metrics(
            decisions=decisions,
            segment_fn=lambda decision: decision.question_route,
        ),
        selected_rank_metrics=_segment_metrics(
            decisions=decisions,
            segment_fn=lambda decision: _rank_bucket_label(decision.final_candidate_rank),
        ),
    )


def _policy_decision(
    config: CandidateRerankerPolicyConfig,
    selection: CandidateRerankerSelection,
    question_rows: Sequence[Mapping[str, Any]],
    deep_rank_min: int,
) -> CandidateRerankerPolicyDecision:
    decision_reasons = _decision_reasons(
        config=config,
        selection=selection,
        baseline_row=_row_by_candidate_id(question_rows, selection.baseline_candidate_id),
    )
    replace_candidate = not decision_reasons
    final_candidate_id = (
        selection.selected_candidate_id
        if replace_candidate
        else selection.baseline_candidate_id
    )
    final_candidate_rank = (
        selection.selected_candidate_rank
        if replace_candidate
        else selection.baseline_candidate_rank
    )
    final_candidate_token_f1 = (
        selection.selected_candidate_token_f1
        if replace_candidate
        else selection.baseline_candidate_token_f1
    )
    final_is_gold_document = (
        selection.selected_is_gold_document
        if replace_candidate
        else selection.baseline_is_gold_document
    )
    final_is_oracle_best = (
        selection.selected_is_oracle_best_f1
        if replace_candidate
        else selection.baseline_is_oracle_best_f1
    )
    gold_document_candidate_count = sum(_is_gold_document(row) for row in question_rows)

    return CandidateRerankerPolicyDecision(
        split=selection.split,
        question_id=selection.question_id,
        question_route=selection.question_route,
        action="replace_with_model_candidate" if replace_candidate else "keep_top_candidate",
        decision_reasons=decision_reasons or ["accepted"],
        baseline_candidate_id=selection.baseline_candidate_id,
        baseline_candidate_rank=selection.baseline_candidate_rank,
        baseline_candidate_token_f1=selection.baseline_candidate_token_f1,
        model_candidate_id=selection.selected_candidate_id,
        model_candidate_rank=selection.selected_candidate_rank,
        model_candidate_token_f1=selection.selected_candidate_token_f1,
        model_score_margin_vs_top_candidate=selection.score_margin_vs_top_candidate,
        final_candidate_id=final_candidate_id,
        final_candidate_rank=final_candidate_rank,
        final_candidate_token_f1=final_candidate_token_f1,
        oracle_candidate_id=selection.oracle_candidate_id,
        oracle_candidate_rank=selection.oracle_candidate_rank,
        oracle_candidate_token_f1=selection.oracle_candidate_token_f1,
        f1_delta_vs_top_candidate=round(
            final_candidate_token_f1 - selection.baseline_candidate_token_f1,
            4,
        ),
        final_is_gold_document=final_is_gold_document,
        final_is_oracle_best_f1=final_is_oracle_best,
        gold_document_candidate_count=gold_document_candidate_count,
        final_missed_gold_document=(
            gold_document_candidate_count > 0 and not final_is_gold_document
        ),
        final_deep_rank=final_candidate_rank >= deep_rank_min,
    )


def _decision_reasons(
    config: CandidateRerankerPolicyConfig,
    selection: CandidateRerankerSelection,
    baseline_row: Mapping[str, Any],
) -> list[str]:
    if selection.selected_candidate_id == selection.baseline_candidate_id:
        return ["model_selected_top_candidate"]

    reasons = []
    if selection.question_route in config.blocked_routes:
        reasons.append("route_blocked")
    if selection.selected_candidate_rank > config.max_selected_rank:
        reasons.append("selected_rank_exceeds_limit")
    if selection.score_margin_vs_top_candidate < config.min_score_margin_vs_top_candidate:
        reasons.append("score_margin_below_min")
    if (
        config.protect_top1_candidate_score_min is not None
        and _runtime_feature_float(baseline_row, "candidate_score")
        >= config.protect_top1_candidate_score_min
    ):
        reasons.append("top1_candidate_score_protected")
    return reasons


def _metrics(
    decisions: Sequence[CandidateRerankerPolicyDecision],
    selections: Sequence[CandidateRerankerSelection],
) -> CandidateRerankerPolicyMetrics:
    question_count = len(decisions)
    baseline_f1_values = [
        decision.baseline_candidate_token_f1 for decision in decisions
    ]
    unconstrained_f1_values = [
        selection.selected_candidate_token_f1 for selection in selections
    ]
    policy_f1_values = [decision.final_candidate_token_f1 for decision in decisions]
    oracle_f1_values = [decision.oracle_candidate_token_f1 for decision in decisions]
    policy_delta_values = [decision.f1_delta_vs_top_candidate for decision in decisions]
    unconstrained_delta_values = [
        selection.selected_candidate_token_f1 - selection.baseline_candidate_token_f1
        for selection in selections
    ]
    oracle_delta_values = [
        decision.oracle_candidate_token_f1 - decision.baseline_candidate_token_f1
        for decision in decisions
    ]
    outcome_counts = Counter(_outcome(delta) for delta in policy_delta_values)
    reason_counts = Counter(
        reason
        for decision in decisions
        for reason in decision.decision_reasons
    )
    replacement_count = sum(
        decision.action == "replace_with_model_candidate" for decision in decisions
    )
    final_missed_gold_count = sum(decision.final_missed_gold_document for decision in decisions)
    final_deep_rank_count = sum(decision.final_deep_rank for decision in decisions)
    final_oracle_best_count = sum(decision.final_is_oracle_best_f1 for decision in decisions)

    return CandidateRerankerPolicyMetrics(
        question_count=question_count,
        baseline_average_token_f1=_rounded_mean(baseline_f1_values),
        unconstrained_model_average_token_f1=_rounded_mean(unconstrained_f1_values),
        policy_average_token_f1=_rounded_mean(policy_f1_values),
        oracle_average_token_f1=_rounded_mean(oracle_f1_values),
        average_delta_vs_top_candidate=_rounded_mean(policy_delta_values),
        unconstrained_model_delta_vs_top_candidate=_rounded_mean(
            unconstrained_delta_values
        ),
        oracle_gap_closed_rate=_safe_ratio(
            sum(policy_delta_values),
            sum(oracle_delta_values),
        ),
        replacement_count=replacement_count,
        replacement_rate=_ratio(replacement_count, question_count),
        improved_count=outcome_counts["improved"],
        regressed_count=outcome_counts["regressed"],
        tied_count=outcome_counts["tied"],
        regressed_rate=_ratio(outcome_counts["regressed"], question_count),
        regression_reduction_vs_unconstrained=(
            sum(delta < 0 for delta in unconstrained_delta_values)
            - outcome_counts["regressed"]
        ),
        final_missed_gold_document_count=final_missed_gold_count,
        final_missed_gold_document_rate=_ratio(final_missed_gold_count, question_count),
        final_deep_rank_count=final_deep_rank_count,
        final_deep_rank_rate=_ratio(final_deep_rank_count, question_count),
        final_oracle_best_count=final_oracle_best_count,
        final_oracle_best_rate=_ratio(final_oracle_best_count, question_count),
        decision_reason_counts=dict(sorted(reason_counts.items())),
    )


def _segment_metrics(
    decisions: Sequence[CandidateRerankerPolicyDecision],
    segment_fn,
) -> list[CandidateRerankerPolicyMetricsBySegment]:
    decisions_by_segment: dict[str, list[CandidateRerankerPolicyDecision]] = defaultdict(list)
    for decision in decisions:
        decisions_by_segment[str(segment_fn(decision))].append(decision)

    metrics = []
    for segment_name, segment_decisions in decisions_by_segment.items():
        replacement_count = sum(
            decision.action == "replace_with_model_candidate"
            for decision in segment_decisions
        )
        delta_values = [
            decision.f1_delta_vs_top_candidate for decision in segment_decisions
        ]
        regressed_count = sum(delta < 0 for delta in delta_values)
        missed_gold_count = sum(
            decision.final_missed_gold_document for decision in segment_decisions
        )
        metrics.append(
            CandidateRerankerPolicyMetricsBySegment(
                segment_name=segment_name,
                question_count=len(segment_decisions),
                replacement_count=replacement_count,
                replacement_rate=_ratio(replacement_count, len(segment_decisions)),
                average_delta_vs_top_candidate=_rounded_mean(delta_values),
                regressed_count=regressed_count,
                regressed_rate=_ratio(regressed_count, len(segment_decisions)),
                final_missed_gold_document_count=missed_gold_count,
                final_missed_gold_document_rate=_ratio(
                    missed_gold_count,
                    len(segment_decisions),
                ),
            )
        )
    return sorted(
        metrics,
        key=lambda item: (item.average_delta_vs_top_candidate, -item.regressed_count),
        reverse=True,
    )


def _policy_grid(
    max_selected_rank_grid: Sequence[int],
    min_score_margin_grid: Sequence[float],
    protect_top1_candidate_score_min_grid: Sequence[float | None],
    blocked_route_sets: Sequence[Sequence[str]],
) -> list[CandidateRerankerPolicyConfig]:
    policies = []
    for max_selected_rank in max_selected_rank_grid:
        for min_score_margin in min_score_margin_grid:
            for protect_top1_score in protect_top1_candidate_score_min_grid:
                for blocked_routes in blocked_route_sets:
                    normalized_blocked_routes = tuple(sorted(set(blocked_routes)))
                    policies.append(
                        CandidateRerankerPolicyConfig(
                            name=_policy_name(
                                max_selected_rank=max_selected_rank,
                                min_score_margin=min_score_margin,
                                protect_top1_score=protect_top1_score,
                                blocked_routes=normalized_blocked_routes,
                            ),
                            max_selected_rank=max_selected_rank,
                            blocked_routes=normalized_blocked_routes,
                            min_score_margin_vs_top_candidate=min_score_margin,
                            protect_top1_candidate_score_min=protect_top1_score,
                        )
                    )
    return policies


def _policy_name(
    max_selected_rank: int,
    min_score_margin: float,
    protect_top1_score: float | None,
    blocked_routes: Sequence[str],
) -> str:
    protected = "none" if protect_top1_score is None else str(int(protect_top1_score))
    routes = "none" if not blocked_routes else "+".join(blocked_routes)
    return (
        f"rank_lte_{max_selected_rank}__margin_gte_{min_score_margin:g}"
        f"__top1_score_protect_{protected}__blocked_{routes}"
    )


def _average_delta_policy_key(
    evaluation: CandidateRerankerPolicyEvaluation,
) -> tuple[float, int, float, int, int]:
    metrics = evaluation.metrics
    return (
        metrics.average_delta_vs_top_candidate,
        -metrics.regressed_count,
        metrics.oracle_gap_closed_rate,
        metrics.replacement_count,
        metrics.final_oracle_best_count,
    )


def _regression_reduction_policy_key(
    evaluation: CandidateRerankerPolicyEvaluation,
) -> tuple[int, float, float, int]:
    metrics = evaluation.metrics
    return (
        metrics.regression_reduction_vs_unconstrained,
        metrics.average_delta_vs_top_candidate,
        metrics.oracle_gap_closed_rate,
        metrics.replacement_count,
    )


def _build_row_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    row_index: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        row_index[_question_key(row["split"], row["question_id"])].append(row)
    return dict(row_index)


def _row_by_candidate_id(
    rows: Sequence[Mapping[str, Any]],
    candidate_id: str,
) -> Mapping[str, Any]:
    for row in rows:
        if row["candidate_id"] == candidate_id:
            return row
    raise ValueError(f"Missing candidate row: {candidate_id}")


def _is_gold_document(row: Mapping[str, Any]) -> bool:
    gold_labels = row.get("gold_labels")
    if not isinstance(gold_labels, Mapping):
        raise ValueError("row gold_labels must be an object")
    return bool(gold_labels["is_gold_document"])


def _runtime_feature_float(row: Mapping[str, Any], feature_name: str) -> float:
    runtime_features = row.get("runtime_features")
    if not isinstance(runtime_features, Mapping):
        raise ValueError("row runtime_features must be an object")
    return float(runtime_features[feature_name])


def _rank_bucket_label(rank: int) -> str:
    if rank == 1:
        return "rank_1"
    if rank == 2:
        return "rank_2"
    if rank == 3:
        return "rank_3"
    if 4 <= rank <= 5:
        return "rank_4_5"
    if 6 <= rank <= 10:
        return "rank_6_10"
    return "rank_11_plus"


def _outcome(delta: float) -> str:
    if delta > 0:
        return "improved"
    if delta < 0:
        return "regressed"
    return "tied"


def _question_key(split: Any, question_id: Any) -> str:
    return f"{split}::{question_id}"


def _rounded_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(mean(values), 4)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _validate_search_options(
    max_selected_rank_grid: Sequence[int],
    min_score_margin_grid: Sequence[float],
    protect_top1_candidate_score_min_grid: Sequence[float | None],
    blocked_route_sets: Sequence[Sequence[str]],
    deep_rank_min: int,
    top_policy_limit: int,
) -> None:
    if not max_selected_rank_grid:
        raise ValueError("max_selected_rank_grid must not be empty")
    if any(rank <= 0 for rank in max_selected_rank_grid):
        raise ValueError("max_selected_rank_grid values must be positive")
    if not min_score_margin_grid:
        raise ValueError("min_score_margin_grid must not be empty")
    if any(margin < 0 for margin in min_score_margin_grid):
        raise ValueError("min_score_margin_grid values must be non-negative")
    if not protect_top1_candidate_score_min_grid:
        raise ValueError("protect_top1_candidate_score_min_grid must not be empty")
    if any(
        threshold is not None and threshold < 0
        for threshold in protect_top1_candidate_score_min_grid
    ):
        raise ValueError(
            "protect_top1_candidate_score_min_grid values must be non-negative"
        )
    if not blocked_route_sets:
        raise ValueError("blocked_route_sets must not be empty")
    if deep_rank_min <= 1:
        raise ValueError("deep_rank_min must be greater than 1")
    if top_policy_limit <= 0:
        raise ValueError("top_policy_limit must be positive")
