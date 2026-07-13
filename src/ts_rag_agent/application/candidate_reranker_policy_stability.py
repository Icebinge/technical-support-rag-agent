from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from ts_rag_agent.application.candidate_reranker_cv import (
    CandidateRerankerSelection,
    cross_validated_candidate_reranker_selections,
)
from ts_rag_agent.application.candidate_reranker_policy_search import (
    CandidateRerankerPolicyConfig,
    CandidateRerankerPolicyEvaluation,
    CandidateRerankerPolicyMetrics,
    CandidateRerankerPolicyMetricsBySegment,
    candidate_reranker_policy_decisions_from_selections,
    evaluate_candidate_reranker_policy_from_selections,
    summarize_candidate_reranker_policy_decisions,
)


@dataclass(frozen=True)
class CandidateRerankerPolicyDelta:
    """Metric deltas between a primary policy and a challenger policy."""

    primary_policy_name: str
    challenger_policy_name: str
    average_delta_difference: float
    policy_average_f1_difference: float
    oracle_gap_closed_difference: float
    replacement_count_difference: int
    regressed_count_difference: int
    final_missed_gold_document_count_difference: int
    final_deep_rank_count_difference: int


@dataclass(frozen=True)
class CandidateRerankerPolicyFoldStability:
    """Fold-level metrics for one fixed constrained policy."""

    policy_name: str
    fold_index: int
    metrics: CandidateRerankerPolicyMetrics


@dataclass(frozen=True)
class CandidateRerankerPolicyRouteComparison:
    """Route-level comparison between primary and challenger policies."""

    route: str
    question_count: int
    primary_average_delta: float
    challenger_average_delta: float
    average_delta_difference: float
    primary_replacement_count: int
    challenger_replacement_count: int
    replacement_count_difference: int
    primary_regressed_count: int
    challenger_regressed_count: int
    regressed_count_difference: int
    primary_final_missed_gold_document_count: int
    challenger_final_missed_gold_document_count: int
    missed_gold_document_count_difference: int


@dataclass(frozen=True)
class CandidateRerankerPolicyStabilityResult:
    """Stability comparison for fixed constrained candidate-reranker policies."""

    model_name: str
    fold_count: int
    primary_policy: CandidateRerankerPolicyEvaluation
    challenger_policy: CandidateRerankerPolicyEvaluation
    primary_vs_challenger: CandidateRerankerPolicyDelta
    fold_metrics: list[CandidateRerankerPolicyFoldStability]
    route_comparisons: list[CandidateRerankerPolicyRouteComparison]
    findings: list[str]
    analysis_scope: str


STAGE35_BEST_POLICY = CandidateRerankerPolicyConfig(
    name=(
        "rank_lte_5__margin_gte_0.05__top1_score_protect_none"
        "__blocked_how_to_or_lookup+security_bulletin_affected_product"
    ),
    max_selected_rank=5,
    blocked_routes=("how_to_or_lookup", "security_bulletin_affected_product"),
    min_score_margin_vs_top_candidate=0.05,
    protect_top1_candidate_score_min=None,
)
STAGE35_BLOCK_HOW_TO_ONLY_POLICY = CandidateRerankerPolicyConfig(
    name=(
        "rank_lte_5__margin_gte_0.05__top1_score_protect_none"
        "__blocked_how_to_or_lookup"
    ),
    max_selected_rank=5,
    blocked_routes=("how_to_or_lookup",),
    min_score_margin_vs_top_candidate=0.05,
    protect_top1_candidate_score_min=None,
)


def analyze_candidate_reranker_policy_stability(
    rows: Sequence[Mapping[str, Any]],
    model_name: str = "logistic_best_candidate",
    fold_count: int = 5,
    primary_policy: CandidateRerankerPolicyConfig = STAGE35_BEST_POLICY,
    challenger_policy: CandidateRerankerPolicyConfig = STAGE35_BLOCK_HOW_TO_ONLY_POLICY,
    deep_rank_min: int = 6,
) -> CandidateRerankerPolicyStabilityResult:
    """Compare fixed constrained policies across folds and routes."""

    selections = cross_validated_candidate_reranker_selections(
        rows=rows,
        model_name=model_name,
        fold_count=fold_count,
    )
    primary_evaluation = evaluate_candidate_reranker_policy_from_selections(
        config=primary_policy,
        selections=selections,
        rows=rows,
        deep_rank_min=deep_rank_min,
    )
    challenger_evaluation = evaluate_candidate_reranker_policy_from_selections(
        config=challenger_policy,
        selections=selections,
        rows=rows,
        deep_rank_min=deep_rank_min,
    )
    fold_metrics = _fold_metrics(
        rows=rows,
        selections=selections,
        policies=(primary_policy, challenger_policy),
        fold_count=fold_count,
        deep_rank_min=deep_rank_min,
    )
    route_comparisons = _route_comparisons(
        primary_routes=primary_evaluation.route_metrics,
        challenger_routes=challenger_evaluation.route_metrics,
    )

    return CandidateRerankerPolicyStabilityResult(
        model_name=model_name,
        fold_count=fold_count,
        primary_policy=primary_evaluation,
        challenger_policy=challenger_evaluation,
        primary_vs_challenger=_policy_delta(
            primary=primary_evaluation,
            challenger=challenger_evaluation,
        ),
        fold_metrics=fold_metrics,
        route_comparisons=route_comparisons,
        findings=_findings(
            primary=primary_evaluation,
            challenger=challenger_evaluation,
            route_comparisons=route_comparisons,
        ),
        analysis_scope=(
            "Offline stability analysis only. It compares fixed constrained policies "
            "over the same grouped-CV selections and does not change runtime behavior."
        ),
    )


def candidate_reranker_policy_stability_to_dict(
    result: CandidateRerankerPolicyStabilityResult,
) -> dict[str, Any]:
    """Convert a policy-stability result to a JSON-safe dictionary."""

    return asdict(result)


def _fold_metrics(
    rows: Sequence[Mapping[str, Any]],
    selections: Sequence[CandidateRerankerSelection],
    policies: Sequence[CandidateRerankerPolicyConfig],
    fold_count: int,
    deep_rank_min: int,
) -> list[CandidateRerankerPolicyFoldStability]:
    fold_by_question_key = _fold_by_question_key(selections=selections, fold_count=fold_count)
    selections_by_fold: dict[int, list[CandidateRerankerSelection]] = defaultdict(list)
    for selection in selections:
        selections_by_fold[
            fold_by_question_key[_question_key(selection.split, selection.question_id)]
        ].append(selection)

    results = []
    for policy in policies:
        decisions = candidate_reranker_policy_decisions_from_selections(
            config=policy,
            selections=selections,
            rows=rows,
            deep_rank_min=deep_rank_min,
        )
        decisions_by_key = {
            _question_key(decision.split, decision.question_id): decision
            for decision in decisions
        }
        for fold_index in range(fold_count):
            fold_selections = selections_by_fold[fold_index]
            fold_decisions = [
                decisions_by_key[_question_key(selection.split, selection.question_id)]
                for selection in fold_selections
            ]
            results.append(
                CandidateRerankerPolicyFoldStability(
                    policy_name=policy.name,
                    fold_index=fold_index,
                    metrics=summarize_candidate_reranker_policy_decisions(
                        decisions=fold_decisions,
                        selections=fold_selections,
                    ),
                )
            )
    return sorted(results, key=lambda item: (item.policy_name, item.fold_index))


def _route_comparisons(
    primary_routes: Sequence[CandidateRerankerPolicyMetricsBySegment],
    challenger_routes: Sequence[CandidateRerankerPolicyMetricsBySegment],
) -> list[CandidateRerankerPolicyRouteComparison]:
    primary_by_route = {metric.segment_name: metric for metric in primary_routes}
    challenger_by_route = {metric.segment_name: metric for metric in challenger_routes}
    route_names = sorted(set(primary_by_route) | set(challenger_by_route))
    comparisons = []
    for route_name in route_names:
        primary = primary_by_route[route_name]
        challenger = challenger_by_route[route_name]
        comparisons.append(
            CandidateRerankerPolicyRouteComparison(
                route=route_name,
                question_count=max(primary.question_count, challenger.question_count),
                primary_average_delta=primary.average_delta_vs_top_candidate,
                challenger_average_delta=challenger.average_delta_vs_top_candidate,
                average_delta_difference=round(
                    primary.average_delta_vs_top_candidate
                    - challenger.average_delta_vs_top_candidate,
                    4,
                ),
                primary_replacement_count=primary.replacement_count,
                challenger_replacement_count=challenger.replacement_count,
                replacement_count_difference=(
                    primary.replacement_count - challenger.replacement_count
                ),
                primary_regressed_count=primary.regressed_count,
                challenger_regressed_count=challenger.regressed_count,
                regressed_count_difference=(
                    primary.regressed_count - challenger.regressed_count
                ),
                primary_final_missed_gold_document_count=(
                    primary.final_missed_gold_document_count
                ),
                challenger_final_missed_gold_document_count=(
                    challenger.final_missed_gold_document_count
                ),
                missed_gold_document_count_difference=(
                    primary.final_missed_gold_document_count
                    - challenger.final_missed_gold_document_count
                ),
            )
        )
    return sorted(
        comparisons,
        key=lambda item: (abs(item.average_delta_difference), item.question_count),
        reverse=True,
    )


def _policy_delta(
    primary: CandidateRerankerPolicyEvaluation,
    challenger: CandidateRerankerPolicyEvaluation,
) -> CandidateRerankerPolicyDelta:
    primary_metrics = primary.metrics
    challenger_metrics = challenger.metrics
    return CandidateRerankerPolicyDelta(
        primary_policy_name=primary.config.name,
        challenger_policy_name=challenger.config.name,
        average_delta_difference=round(
            primary_metrics.average_delta_vs_top_candidate
            - challenger_metrics.average_delta_vs_top_candidate,
            4,
        ),
        policy_average_f1_difference=round(
            primary_metrics.policy_average_token_f1
            - challenger_metrics.policy_average_token_f1,
            4,
        ),
        oracle_gap_closed_difference=round(
            primary_metrics.oracle_gap_closed_rate - challenger_metrics.oracle_gap_closed_rate,
            4,
        ),
        replacement_count_difference=(
            primary_metrics.replacement_count - challenger_metrics.replacement_count
        ),
        regressed_count_difference=(
            primary_metrics.regressed_count - challenger_metrics.regressed_count
        ),
        final_missed_gold_document_count_difference=(
            primary_metrics.final_missed_gold_document_count
            - challenger_metrics.final_missed_gold_document_count
        ),
        final_deep_rank_count_difference=(
            primary_metrics.final_deep_rank_count
            - challenger_metrics.final_deep_rank_count
        ),
    )


def _findings(
    primary: CandidateRerankerPolicyEvaluation,
    challenger: CandidateRerankerPolicyEvaluation,
    route_comparisons: Sequence[CandidateRerankerPolicyRouteComparison],
) -> list[str]:
    findings = []
    delta = _policy_delta(primary=primary, challenger=challenger)
    if abs(delta.average_delta_difference) <= 0.001:
        findings.append(
            "Primary policy improves average delta by at most 0.001 over the simpler "
            "challenger, so the extra blocked route is not materially validated."
        )
    if delta.regressed_count_difference <= -1:
        findings.append(
            "Primary policy reduces regression count versus the simpler challenger."
        )
    affected_product = next(
        (
            comparison
            for comparison in route_comparisons
            if comparison.route == "security_bulletin_affected_product"
        ),
        None,
    )
    if affected_product and affected_product.question_count <= 1:
        findings.append(
            "security_bulletin_affected_product has only one question, so blocking it "
            "should be treated as sample-size sensitive rather than stable policy evidence."
        )
    if primary.metrics.final_deep_rank_count == 0 and challenger.metrics.final_deep_rank_count == 0:
        findings.append("Both compared policies eliminate deep-rank selections.")
    return findings


def _fold_by_question_key(
    selections: Sequence[CandidateRerankerSelection],
    fold_count: int,
) -> dict[str, int]:
    question_keys = sorted(
        {_question_key(selection.split, selection.question_id) for selection in selections}
    )
    return {
        question_key: index % fold_count
        for index, question_key in enumerate(question_keys)
    }


def _question_key(split: Any, question_id: Any) -> str:
    return f"{split}::{question_id}"
