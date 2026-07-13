from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from ts_rag_agent.application.route_aware_composition_policy import (
    RouteAwareCompositionPolicy,
    RouteAwareCompositionSummary,
    analyze_route_aware_composition_policy,
)

DEFAULT_INSTALL_MARGIN_GRID = (45.0, 50.0, 60.0, 70.0, 80.0, 100.0)


@dataclass(frozen=True)
class RouteAwareCVThresholdScore:
    """One candidate threshold's score on one fold partition."""

    install_upgrade_score_margin_min: float
    total_cases: int
    average_baseline_f1: float
    average_policy_f1: float
    average_f1_delta: float
    baseline_gold_citation_count: int
    policy_gold_citation_count: int
    citation_delta: int
    changed_answer_count: int
    f1_improved_count: int
    f1_regressed_count: int
    citation_lost_count: int


@dataclass(frozen=True)
class RouteAwareCVFoldResult:
    """Train/validation result for one deterministic cross-validation fold."""

    fold_index: int
    train_case_count: int
    validation_case_count: int
    selected_install_upgrade_score_margin_min: float
    selection_reason: str
    train_scores: list[RouteAwareCVThresholdScore]
    validation_score: RouteAwareCVThresholdScore


@dataclass(frozen=True)
class RouteAwareCVAggregate:
    """Aggregate validation performance across folds."""

    fold_count: int
    total_validation_cases: int
    average_baseline_f1: float
    average_policy_f1: float
    average_f1_delta: float
    baseline_gold_citation_count: int
    policy_gold_citation_count: int
    citation_delta: int
    changed_answer_count: int
    f1_improved_count: int
    f1_regressed_count: int
    citation_lost_count: int
    selected_margin_counts: dict[str, int]


@dataclass(frozen=True)
class RouteAwareCVResult:
    """Full route-aware composition threshold cross-validation result."""

    policy_name: str
    fold_count: int
    install_upgrade_score_margin_grid: list[float]
    min_train_average_f1_gain: float
    min_train_citation_delta: int
    aggregate_validation: RouteAwareCVAggregate
    folds: list[RouteAwareCVFoldResult]


def cross_validate_route_aware_composition_policy(
    answer_gap_report: dict[str, Any],
    install_upgrade_score_margin_grid: list[float] | None = None,
    fold_count: int = 5,
    min_train_average_f1_gain: float = 0.0,
    min_train_citation_delta: int = 0,
    strong_first_score_min: float = 100.0,
    strong_first_score_ratio_min: float = 1.15,
    strong_first_score_margin_min: float = 20.0,
    enable_how_to_top1: bool = False,
    max_top1_retrieval_rank: int = 3,
    duplicate_threshold: float = 0.96,
) -> RouteAwareCVResult:
    """Cross-validate the install-route top1 margin over answer-gap cases."""

    raw_cases = answer_gap_report.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("answer_gap_report must contain a list field named 'cases'")

    cases = [case for case in raw_cases if isinstance(case, dict)]
    margins = _normalize_margin_grid(install_upgrade_score_margin_grid)
    _validate_cv_options(
        cases=cases,
        fold_count=fold_count,
        min_train_average_f1_gain=min_train_average_f1_gain,
        min_train_citation_delta=min_train_citation_delta,
        strong_first_score_min=strong_first_score_min,
        strong_first_score_ratio_min=strong_first_score_ratio_min,
        strong_first_score_margin_min=strong_first_score_margin_min,
        max_top1_retrieval_rank=max_top1_retrieval_rank,
        duplicate_threshold=duplicate_threshold,
    )

    folds = _build_deterministic_folds(cases, fold_count)
    fold_results = []
    for fold_index, validation_cases in enumerate(folds):
        train_cases = [
            case
            for index, fold_cases in enumerate(folds)
            if index != fold_index
            for case in fold_cases
        ]
        train_scores = [
            _score_cases(
                cases=train_cases,
                install_upgrade_score_margin_min=margin,
                strong_first_score_min=strong_first_score_min,
                strong_first_score_ratio_min=strong_first_score_ratio_min,
                strong_first_score_margin_min=strong_first_score_margin_min,
                enable_how_to_top1=enable_how_to_top1,
                max_top1_retrieval_rank=max_top1_retrieval_rank,
                duplicate_threshold=duplicate_threshold,
            )
            for margin in margins
        ]
        selected_score, selection_reason = _select_train_score(
            scores=train_scores,
            min_train_average_f1_gain=min_train_average_f1_gain,
            min_train_citation_delta=min_train_citation_delta,
        )
        validation_score = _score_cases(
            cases=validation_cases,
            install_upgrade_score_margin_min=selected_score.install_upgrade_score_margin_min,
            strong_first_score_min=strong_first_score_min,
            strong_first_score_ratio_min=strong_first_score_ratio_min,
            strong_first_score_margin_min=strong_first_score_margin_min,
            enable_how_to_top1=enable_how_to_top1,
            max_top1_retrieval_rank=max_top1_retrieval_rank,
            duplicate_threshold=duplicate_threshold,
        )
        fold_results.append(
            RouteAwareCVFoldResult(
                fold_index=fold_index,
                train_case_count=len(train_cases),
                validation_case_count=len(validation_cases),
                selected_install_upgrade_score_margin_min=(
                    selected_score.install_upgrade_score_margin_min
                ),
                selection_reason=selection_reason,
                train_scores=train_scores,
                validation_score=validation_score,
            )
        )

    return RouteAwareCVResult(
        policy_name=RouteAwareCompositionPolicy.name,
        fold_count=fold_count,
        install_upgrade_score_margin_grid=margins,
        min_train_average_f1_gain=min_train_average_f1_gain,
        min_train_citation_delta=min_train_citation_delta,
        aggregate_validation=_aggregate_validation_scores(fold_results),
        folds=fold_results,
    )


def route_aware_cv_result_to_dict(result: RouteAwareCVResult) -> dict[str, Any]:
    """Convert a CV result to a JSON-safe dictionary."""

    return {
        "policy_name": result.policy_name,
        "fold_count": result.fold_count,
        "install_upgrade_score_margin_grid": result.install_upgrade_score_margin_grid,
        "selection_constraints": {
            "min_train_average_f1_gain": result.min_train_average_f1_gain,
            "min_train_citation_delta": result.min_train_citation_delta,
        },
        "aggregate_validation": asdict(result.aggregate_validation),
        "folds": [
            {
                "fold_index": fold.fold_index,
                "train_case_count": fold.train_case_count,
                "validation_case_count": fold.validation_case_count,
                "selected_install_upgrade_score_margin_min": (
                    fold.selected_install_upgrade_score_margin_min
                ),
                "selection_reason": fold.selection_reason,
                "train_scores": [asdict(score) for score in fold.train_scores],
                "validation_score": asdict(fold.validation_score),
            }
            for fold in result.folds
        ],
    }


def _score_cases(
    cases: list[dict[str, Any]],
    install_upgrade_score_margin_min: float,
    strong_first_score_min: float,
    strong_first_score_ratio_min: float,
    strong_first_score_margin_min: float,
    enable_how_to_top1: bool,
    max_top1_retrieval_rank: int,
    duplicate_threshold: float,
) -> RouteAwareCVThresholdScore:
    policy = RouteAwareCompositionPolicy(
        strong_first_score_min=strong_first_score_min,
        strong_first_score_ratio_min=strong_first_score_ratio_min,
        strong_first_score_margin_min=strong_first_score_margin_min,
        install_upgrade_score_margin_min=install_upgrade_score_margin_min,
        enable_how_to_top1=enable_how_to_top1,
        max_top1_retrieval_rank=max_top1_retrieval_rank,
        duplicate_threshold=duplicate_threshold,
    )
    result = analyze_route_aware_composition_policy(
        answer_gap_report={"cases": cases},
        policy=policy,
        min_average_f1_gain=0.0,
        max_allowed_citation_loss=10**9,
        sample_limit_per_bucket=0,
    )
    return _summary_to_threshold_score(
        summary=result.summary,
        install_upgrade_score_margin_min=install_upgrade_score_margin_min,
    )


def _summary_to_threshold_score(
    summary: RouteAwareCompositionSummary,
    install_upgrade_score_margin_min: float,
) -> RouteAwareCVThresholdScore:
    return RouteAwareCVThresholdScore(
        install_upgrade_score_margin_min=install_upgrade_score_margin_min,
        total_cases=summary.total_cases,
        average_baseline_f1=summary.average_baseline_f1,
        average_policy_f1=summary.average_policy_f1,
        average_f1_delta=summary.average_f1_delta,
        baseline_gold_citation_count=summary.baseline_gold_citation_count,
        policy_gold_citation_count=summary.policy_gold_citation_count,
        citation_delta=summary.citation_delta,
        changed_answer_count=summary.changed_answer_count,
        f1_improved_count=summary.f1_improved_count,
        f1_regressed_count=summary.f1_regressed_count,
        citation_lost_count=summary.citation_lost_count,
    )


def _select_train_score(
    scores: list[RouteAwareCVThresholdScore],
    min_train_average_f1_gain: float,
    min_train_citation_delta: int,
) -> tuple[RouteAwareCVThresholdScore, str]:
    accepted_scores = [
        score
        for score in scores
        if (
            score.average_f1_delta >= min_train_average_f1_gain
            and score.citation_delta >= min_train_citation_delta
        )
    ]
    if accepted_scores:
        return (
            max(accepted_scores, key=_accepted_score_key),
            "selected best F1 among thresholds satisfying train citation/F1 constraints",
        )

    return (
        max(scores, key=_fallback_score_key),
        "selected best available threshold because no candidate satisfied train constraints",
    )


def _accepted_score_key(score: RouteAwareCVThresholdScore) -> tuple:
    return (
        score.average_f1_delta,
        score.citation_delta,
        -score.citation_lost_count,
        -score.f1_regressed_count,
        -score.changed_answer_count,
        score.install_upgrade_score_margin_min,
    )


def _fallback_score_key(score: RouteAwareCVThresholdScore) -> tuple:
    return (
        score.citation_delta,
        score.average_f1_delta,
        -score.citation_lost_count,
        -score.f1_regressed_count,
        -score.changed_answer_count,
        score.install_upgrade_score_margin_min,
    )


def _aggregate_validation_scores(
    fold_results: list[RouteAwareCVFoldResult],
) -> RouteAwareCVAggregate:
    validation_scores = [fold.validation_score for fold in fold_results]
    total_cases = sum(score.total_cases for score in validation_scores)
    baseline_f1_sum = sum(
        score.average_baseline_f1 * score.total_cases for score in validation_scores
    )
    policy_f1_sum = sum(
        score.average_policy_f1 * score.total_cases for score in validation_scores
    )
    selected_margin_counts = Counter(
        str(fold.selected_install_upgrade_score_margin_min) for fold in fold_results
    )

    average_baseline_f1 = _safe_average(baseline_f1_sum, total_cases)
    average_policy_f1 = _safe_average(policy_f1_sum, total_cases)
    return RouteAwareCVAggregate(
        fold_count=len(fold_results),
        total_validation_cases=total_cases,
        average_baseline_f1=average_baseline_f1,
        average_policy_f1=average_policy_f1,
        average_f1_delta=round(average_policy_f1 - average_baseline_f1, 4),
        baseline_gold_citation_count=sum(
            score.baseline_gold_citation_count for score in validation_scores
        ),
        policy_gold_citation_count=sum(
            score.policy_gold_citation_count for score in validation_scores
        ),
        citation_delta=sum(score.citation_delta for score in validation_scores),
        changed_answer_count=sum(score.changed_answer_count for score in validation_scores),
        f1_improved_count=sum(score.f1_improved_count for score in validation_scores),
        f1_regressed_count=sum(score.f1_regressed_count for score in validation_scores),
        citation_lost_count=sum(score.citation_lost_count for score in validation_scores),
        selected_margin_counts=dict(sorted(selected_margin_counts.items())),
    )


def _build_deterministic_folds(
    cases: list[dict[str, Any]],
    fold_count: int,
) -> list[list[dict[str, Any]]]:
    folds: list[list[dict[str, Any]]] = [[] for _ in range(fold_count)]
    sorted_cases = sorted(cases, key=lambda case: str(case.get("question_id", "")))
    for index, case in enumerate(sorted_cases):
        folds[index % fold_count].append(case)
    return folds


def _normalize_margin_grid(
    install_upgrade_score_margin_grid: list[float] | None,
) -> list[float]:
    raw_margins = install_upgrade_score_margin_grid or list(DEFAULT_INSTALL_MARGIN_GRID)
    margins = sorted({float(margin) for margin in raw_margins})
    if not margins:
        raise ValueError("install_upgrade_score_margin_grid must not be empty")
    if any(margin < 0 for margin in margins):
        raise ValueError("install_upgrade_score_margin_grid values must be non-negative")
    return margins


def _validate_cv_options(
    cases: list[dict[str, Any]],
    fold_count: int,
    min_train_average_f1_gain: float,
    min_train_citation_delta: int,
    strong_first_score_min: float,
    strong_first_score_ratio_min: float,
    strong_first_score_margin_min: float,
    max_top1_retrieval_rank: int,
    duplicate_threshold: float,
) -> None:
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if fold_count > len(cases):
        raise ValueError("fold_count must be no larger than the number of cases")
    if min_train_average_f1_gain < 0:
        raise ValueError("min_train_average_f1_gain must be non-negative")
    if strong_first_score_min < 0:
        raise ValueError("strong_first_score_min must be non-negative")
    if strong_first_score_ratio_min < 1:
        raise ValueError("strong_first_score_ratio_min must be at least 1")
    if strong_first_score_margin_min < 0:
        raise ValueError("strong_first_score_margin_min must be non-negative")
    if max_top1_retrieval_rank <= 0:
        raise ValueError("max_top1_retrieval_rank must be positive")
    if not 0 <= duplicate_threshold <= 1:
        raise ValueError("duplicate_threshold must be between 0 and 1")
    if not isinstance(min_train_citation_delta, int):
        raise TypeError("min_train_citation_delta must be an integer")


def _safe_average(numerator: float, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
