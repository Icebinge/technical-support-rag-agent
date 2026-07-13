from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class AnswerGapPriorityCase:
    """Representative case for one answer-gap priority group."""

    question_id: str
    question_title: str
    question_route: str
    bucket: str
    selected_answer_f1: float
    best_gold_window_f1: float
    gold_window_f1_gap: float
    gold_retrieval_rank: int | None
    selected_gold_candidate_count: int


@dataclass(frozen=True)
class AnswerGapPriorityGroup:
    """Aggregated answer-gap priority metrics for one route/bucket slice."""

    group_key: str
    group_type: str
    total_cases: int
    gold_in_context_count: int
    selected_gold_citation_count: int
    gold_in_context_not_selected_count: int
    gold_window_beats_selected_count: int
    selected_answer_low_overlap_count: int
    average_selected_answer_f1: float
    average_best_gold_window_f1: float
    average_gold_window_f1_gap: float
    priority_score: float
    sample_cases: list[AnswerGapPriorityCase]


@dataclass(frozen=True)
class AnswerGapPriorityAnalysis:
    """Cross-report priority analysis for answer-gap cases."""

    source_count: int
    total_cases: int
    bucket_counts: dict[str, int]
    route_counts: dict[str, int]
    route_priorities: list[AnswerGapPriorityGroup]
    bucket_priorities: list[AnswerGapPriorityGroup]
    route_bucket_priorities: list[AnswerGapPriorityGroup]
    top_priority_summary: str


def analyze_answer_gap_priorities(
    answer_gap_reports: list[dict[str, Any]],
    min_cases: int = 1,
    sample_limit_per_group: int = 5,
) -> AnswerGapPriorityAnalysis:
    """Rank answer-gap buckets and routes across one or more answer-gap reports."""

    if not answer_gap_reports:
        raise ValueError("answer_gap_reports must not be empty")
    if min_cases <= 0:
        raise ValueError("min_cases must be positive")
    if sample_limit_per_group < 0:
        raise ValueError("sample_limit_per_group must be non-negative")

    cases = []
    for report in answer_gap_reports:
        raw_cases = report.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError("each answer-gap report must contain a list field named 'cases'")
        cases.extend(case for case in raw_cases if isinstance(case, dict))

    route_groups = _build_groups(
        cases=cases,
        group_type="route",
        key_func=lambda case: str(case.get("question_route", "")),
        min_cases=min_cases,
        sample_limit_per_group=sample_limit_per_group,
    )
    bucket_groups = _build_groups(
        cases=cases,
        group_type="bucket",
        key_func=lambda case: str(case.get("bucket", "")),
        min_cases=min_cases,
        sample_limit_per_group=sample_limit_per_group,
    )
    route_bucket_groups = _build_groups(
        cases=cases,
        group_type="route_bucket",
        key_func=lambda case: (
            f"{case.get('question_route', '')}::{case.get('bucket', '')}"
        ),
        min_cases=min_cases,
        sample_limit_per_group=sample_limit_per_group,
    )
    return AnswerGapPriorityAnalysis(
        source_count=len(answer_gap_reports),
        total_cases=len(cases),
        bucket_counts=dict(Counter(str(case.get("bucket", "")) for case in cases)),
        route_counts=dict(Counter(str(case.get("question_route", "")) for case in cases)),
        route_priorities=route_groups,
        bucket_priorities=bucket_groups,
        route_bucket_priorities=route_bucket_groups,
        top_priority_summary=_build_top_priority_summary(route_bucket_groups),
    )


def answer_gap_priority_analysis_to_dict(
    analysis: AnswerGapPriorityAnalysis,
) -> dict[str, Any]:
    """Convert answer-gap priority analysis to a JSON-safe dictionary."""

    return {
        "source_count": analysis.source_count,
        "total_cases": analysis.total_cases,
        "bucket_counts": analysis.bucket_counts,
        "route_counts": analysis.route_counts,
        "priority_score_note": (
            "Heuristic only: larger values indicate more cases where gold evidence "
            "is available or stronger than selected evidence. It is not a model metric."
        ),
        "top_priority_summary": analysis.top_priority_summary,
        "route_priorities": [asdict(group) for group in analysis.route_priorities],
        "bucket_priorities": [asdict(group) for group in analysis.bucket_priorities],
        "route_bucket_priorities": [
            asdict(group) for group in analysis.route_bucket_priorities
        ],
    }


def _build_groups(
    cases: list[dict[str, Any]],
    group_type: str,
    key_func,
    min_cases: int,
    sample_limit_per_group: int,
) -> list[AnswerGapPriorityGroup]:
    cases_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        cases_by_key[key_func(case)].append(case)

    groups = [
        _build_group(
            group_key=group_key,
            group_type=group_type,
            cases=group_cases,
            sample_limit_per_group=sample_limit_per_group,
        )
        for group_key, group_cases in cases_by_key.items()
        if len(group_cases) >= min_cases
    ]
    return sorted(
        groups,
        key=lambda group: (
            -group.priority_score,
            -group.gold_in_context_not_selected_count,
            -group.gold_window_beats_selected_count,
            group.group_key,
        ),
    )


def _build_group(
    group_key: str,
    group_type: str,
    cases: list[dict[str, Any]],
    sample_limit_per_group: int,
) -> AnswerGapPriorityGroup:
    selected_f1_values = [_safe_float(case.get("selected_answer_token_f1")) for case in cases]
    best_window_f1_values = [
        _safe_float(case.get("best_gold_window", {}).get("token_f1"))
        for case in cases
        if isinstance(case.get("best_gold_window"), dict)
    ]
    gold_window_gaps = [
        max(
            0.0,
            _safe_float(case.get("best_gold_window", {}).get("token_f1"))
            - _safe_float(case.get("selected_answer_token_f1")),
        )
        for case in cases
        if isinstance(case.get("best_gold_window"), dict)
    ]

    total_cases = len(cases)
    gold_in_context_count = sum(1 for case in cases if case.get("gold_retrieval_rank") is not None)
    selected_gold_citation_count = sum(
        1 for case in cases if _safe_int(case.get("selected_gold_candidate_count")) > 0
    )
    gold_in_context_not_selected_count = _count_bucket(cases, "gold_in_context_not_selected")
    gold_window_beats_selected_count = _count_bucket(
        cases,
        "gold_window_beats_selected_answer",
    )
    selected_answer_low_overlap_count = _count_bucket(cases, "selected_answer_low_overlap")
    average_gold_window_f1_gap = _average(gold_window_gaps)
    priority_score = _priority_score(
        gold_in_context_not_selected_count=gold_in_context_not_selected_count,
        gold_window_beats_selected_count=gold_window_beats_selected_count,
        selected_answer_low_overlap_count=selected_answer_low_overlap_count,
        average_gold_window_f1_gap=average_gold_window_f1_gap,
    )

    return AnswerGapPriorityGroup(
        group_key=group_key,
        group_type=group_type,
        total_cases=total_cases,
        gold_in_context_count=gold_in_context_count,
        selected_gold_citation_count=selected_gold_citation_count,
        gold_in_context_not_selected_count=gold_in_context_not_selected_count,
        gold_window_beats_selected_count=gold_window_beats_selected_count,
        selected_answer_low_overlap_count=selected_answer_low_overlap_count,
        average_selected_answer_f1=_average(selected_f1_values),
        average_best_gold_window_f1=_average(best_window_f1_values),
        average_gold_window_f1_gap=average_gold_window_f1_gap,
        priority_score=priority_score,
        sample_cases=_select_sample_cases(cases, sample_limit_per_group),
    )


def _priority_score(
    gold_in_context_not_selected_count: int,
    gold_window_beats_selected_count: int,
    selected_answer_low_overlap_count: int,
    average_gold_window_f1_gap: float,
) -> float:
    issue_count = (
        2 * gold_in_context_not_selected_count
        + gold_window_beats_selected_count
        + selected_answer_low_overlap_count
    )
    return round(issue_count * (1 + average_gold_window_f1_gap), 4)


def _build_top_priority_summary(groups: list[AnswerGapPriorityGroup]) -> str:
    if not groups:
        return "No priority group met the minimum case threshold."
    top = groups[0]
    return (
        f"{top.group_key} has the highest heuristic priority score "
        f"({top.priority_score}) across {top.total_cases} cases."
    )


def _select_sample_cases(
    cases: list[dict[str, Any]],
    sample_limit: int,
) -> list[AnswerGapPriorityCase]:
    if sample_limit == 0:
        return []

    priority_buckets = {
        "gold_in_context_not_selected",
        "gold_window_beats_selected_answer",
        "selected_answer_low_overlap",
    }
    priority_cases = [
        case for case in cases if str(case.get("bucket", "")) in priority_buckets
    ]
    candidate_cases = priority_cases or cases
    ranked_cases = sorted(
        candidate_cases,
        key=lambda case: (
            -_gold_window_f1_gap(case),
            case.get("bucket") != "gold_in_context_not_selected",
            str(case.get("question_id", "")),
        ),
    )
    return [_case_to_sample(case) for case in ranked_cases[:sample_limit]]


def _case_to_sample(case: dict[str, Any]) -> AnswerGapPriorityCase:
    selected_f1 = _safe_float(case.get("selected_answer_token_f1"))
    best_window_f1 = _safe_float(case.get("best_gold_window", {}).get("token_f1"))
    return AnswerGapPriorityCase(
        question_id=str(case.get("question_id", "")),
        question_title=str(case.get("question_title", "")),
        question_route=str(case.get("question_route", "")),
        bucket=str(case.get("bucket", "")),
        selected_answer_f1=selected_f1,
        best_gold_window_f1=best_window_f1,
        gold_window_f1_gap=round(max(0.0, best_window_f1 - selected_f1), 4),
        gold_retrieval_rank=_safe_optional_int(case.get("gold_retrieval_rank")),
        selected_gold_candidate_count=_safe_int(case.get("selected_gold_candidate_count")),
    )


def _gold_window_f1_gap(case: dict[str, Any]) -> float:
    if not isinstance(case.get("best_gold_window"), dict):
        return 0.0
    return max(
        0.0,
        _safe_float(case.get("best_gold_window", {}).get("token_f1"))
        - _safe_float(case.get("selected_answer_token_f1")),
    )


def _count_bucket(cases: list[dict[str, Any]], bucket: str) -> int:
    return sum(1 for case in cases if case.get("bucket") == bucket)


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(mean(values), 4)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
