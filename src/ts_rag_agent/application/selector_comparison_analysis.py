from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SelectorComparisonCase:
    """One question-level comparison between two selector reports."""

    question_id: str
    question_type: str
    winner: str
    baseline_f1: float
    challenger_f1: float
    f1_delta: float
    baseline_gold_cited: bool
    challenger_gold_cited: bool
    baseline_question_route: str
    challenger_question_route: str
    baseline_selected_selector_name: str
    challenger_selected_selector_name: str
    baseline_bucket: str
    challenger_bucket: str
    question_title: str
    gold_answer: str
    baseline_selected_answer: str
    challenger_selected_answer: str


@dataclass(frozen=True)
class SelectorComparisonSummary:
    """Aggregate selector comparison metrics."""

    baseline_label: str
    challenger_label: str
    total_compared: int
    baseline_wins: int
    challenger_wins: int
    ties: int
    avg_f1_delta: float
    baseline_gold_citation_count: int
    challenger_gold_citation_count: int
    winner_counts: dict[str, int]
    question_type_counts: dict[str, int]
    question_type_win_counts: dict[str, dict[str, int]]
    challenger_question_route_counts: dict[str, int]
    challenger_route_win_counts: dict[str, dict[str, int]]
    challenger_selected_selector_counts: dict[str, int]


@dataclass(frozen=True)
class SelectorComparisonResult:
    """Full selector comparison output."""

    summary: SelectorComparisonSummary
    baseline_win_cases: list[SelectorComparisonCase]
    challenger_win_cases: list[SelectorComparisonCase]
    tie_cases: list[SelectorComparisonCase]


def compare_selector_reports(
    baseline_report: dict[str, Any],
    challenger_report: dict[str, Any],
    baseline_label: str,
    challenger_label: str,
    f1_win_margin: float = 0.03,
    sample_limit_per_bucket: int = 20,
) -> SelectorComparisonResult:
    """Compare two answer-gap reports question by question."""

    if f1_win_margin < 0:
        raise ValueError("f1_win_margin must be non-negative")
    if sample_limit_per_bucket < 0:
        raise ValueError("sample_limit_per_bucket must be non-negative")

    baseline_cases = _cases_by_question_id(baseline_report)
    challenger_cases = _cases_by_question_id(challenger_report)
    shared_question_ids = sorted(baseline_cases.keys() & challenger_cases.keys())
    comparisons = [
        _compare_case(
            baseline_case=baseline_cases[question_id],
            challenger_case=challenger_cases[question_id],
            f1_win_margin=f1_win_margin,
        )
        for question_id in shared_question_ids
    ]

    return SelectorComparisonResult(
        summary=_build_summary(
            comparisons=comparisons,
            baseline_label=baseline_label,
            challenger_label=challenger_label,
        ),
        baseline_win_cases=_select_cases(
            comparisons=comparisons,
            winner="baseline",
            sample_limit=sample_limit_per_bucket,
        ),
        challenger_win_cases=_select_cases(
            comparisons=comparisons,
            winner="challenger",
            sample_limit=sample_limit_per_bucket,
        ),
        tie_cases=_select_cases(
            comparisons=comparisons,
            winner="tie",
            sample_limit=sample_limit_per_bucket,
        ),
    )


def comparison_result_to_dict(result: SelectorComparisonResult) -> dict[str, Any]:
    """Convert a comparison result to a JSON-safe dictionary."""

    return {
        "summary": asdict(result.summary),
        "baseline_win_cases": [asdict(case) for case in result.baseline_win_cases],
        "challenger_win_cases": [asdict(case) for case in result.challenger_win_cases],
        "tie_cases": [asdict(case) for case in result.tie_cases],
    }


def classify_question_type(question_title: str, question_text: str, gold_answer: str) -> str:
    """Assign a coarse question type for selector error analysis."""

    combined = f"{question_title}\n{question_text}\n{gold_answer}".lower()
    if any(token in combined for token in ("cve-", "cveid", "cvss", "security bulletin")):
        return "security_bulletin"
    if any(token in combined for token in ("limitation", "restriction", "not supported")):
        return "limitation_or_restriction"
    if any(token in combined for token in (".pdf", "attached", "/support/docview", "http://", "https://")):
        return "attachment_or_link"
    if any(token in combined for token in ("exception", "trace", "dump", "javacore", "error ")):
        return "error_or_log"
    if any(token in combined for token in ("install", "upgrade", "configure", "migration")):
        return "install_upgrade_config"
    if question_title.lower().startswith(("how ", "how do", "how can", "what is", "where ")):
        return "how_to_or_lookup"
    return "other"


def _cases_by_question_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError("report must contain a list field named 'cases'")

    return {
        str(case["question_id"]): case
        for case in cases
        if isinstance(case, dict) and "question_id" in case
    }


def _compare_case(
    baseline_case: dict[str, Any],
    challenger_case: dict[str, Any],
    f1_win_margin: float,
) -> SelectorComparisonCase:
    baseline_f1 = float(baseline_case.get("selected_answer_token_f1", 0.0))
    challenger_f1 = float(challenger_case.get("selected_answer_token_f1", 0.0))
    f1_delta = round(challenger_f1 - baseline_f1, 4)
    baseline_gold_cited = int(baseline_case.get("selected_gold_candidate_count", 0)) > 0
    challenger_gold_cited = int(challenger_case.get("selected_gold_candidate_count", 0)) > 0

    winner = _choose_winner(
        f1_delta=f1_delta,
        f1_win_margin=f1_win_margin,
        baseline_gold_cited=baseline_gold_cited,
        challenger_gold_cited=challenger_gold_cited,
    )
    question_title = str(baseline_case.get("question_title", ""))
    question_text = str(baseline_case.get("question_text", ""))
    gold_answer = str(baseline_case.get("gold_answer", ""))
    question_type = classify_question_type(question_title, question_text, gold_answer)

    return SelectorComparisonCase(
        question_id=str(baseline_case.get("question_id", "")),
        question_type=question_type,
        winner=winner,
        baseline_f1=baseline_f1,
        challenger_f1=challenger_f1,
        f1_delta=f1_delta,
        baseline_gold_cited=baseline_gold_cited,
        challenger_gold_cited=challenger_gold_cited,
        baseline_question_route=str(baseline_case.get("question_route", question_type)),
        challenger_question_route=str(challenger_case.get("question_route", question_type)),
        baseline_selected_selector_name=str(
            baseline_case.get("selected_selector_name", "")
        ),
        challenger_selected_selector_name=str(
            challenger_case.get("selected_selector_name", "")
        ),
        baseline_bucket=str(baseline_case.get("bucket", "")),
        challenger_bucket=str(challenger_case.get("bucket", "")),
        question_title=question_title,
        gold_answer=_truncate(gold_answer),
        baseline_selected_answer=_selected_answer_text(baseline_case),
        challenger_selected_answer=_selected_answer_text(challenger_case),
    )


def _choose_winner(
    f1_delta: float,
    f1_win_margin: float,
    baseline_gold_cited: bool,
    challenger_gold_cited: bool,
) -> str:
    if f1_delta >= f1_win_margin:
        return "challenger"
    if f1_delta <= -f1_win_margin:
        return "baseline"
    if challenger_gold_cited and not baseline_gold_cited:
        return "challenger"
    if baseline_gold_cited and not challenger_gold_cited:
        return "baseline"
    return "tie"


def _build_summary(
    comparisons: list[SelectorComparisonCase],
    baseline_label: str,
    challenger_label: str,
) -> SelectorComparisonSummary:
    winner_counts = Counter(case.winner for case in comparisons)
    question_type_counts = Counter(case.question_type for case in comparisons)
    question_type_win_counts: dict[str, Counter[str]] = defaultdict(Counter)
    challenger_route_win_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for case in comparisons:
        question_type_win_counts[case.question_type][case.winner] += 1
        challenger_route_win_counts[case.challenger_question_route][case.winner] += 1

    avg_f1_delta = (
        round(sum(case.f1_delta for case in comparisons) / len(comparisons), 4)
        if comparisons
        else 0.0
    )

    return SelectorComparisonSummary(
        baseline_label=baseline_label,
        challenger_label=challenger_label,
        total_compared=len(comparisons),
        baseline_wins=winner_counts.get("baseline", 0),
        challenger_wins=winner_counts.get("challenger", 0),
        ties=winner_counts.get("tie", 0),
        avg_f1_delta=avg_f1_delta,
        baseline_gold_citation_count=sum(case.baseline_gold_cited for case in comparisons),
        challenger_gold_citation_count=sum(
            case.challenger_gold_cited for case in comparisons
        ),
        winner_counts=dict(winner_counts),
        question_type_counts=dict(question_type_counts),
        question_type_win_counts={
            question_type: dict(counts)
            for question_type, counts in sorted(question_type_win_counts.items())
        },
        challenger_question_route_counts=dict(
            Counter(case.challenger_question_route for case in comparisons)
        ),
        challenger_route_win_counts={
            question_route: dict(counts)
            for question_route, counts in sorted(challenger_route_win_counts.items())
        },
        challenger_selected_selector_counts=dict(
            Counter(case.challenger_selected_selector_name for case in comparisons)
        ),
    )


def _select_cases(
    comparisons: list[SelectorComparisonCase],
    winner: str,
    sample_limit: int,
) -> list[SelectorComparisonCase]:
    if sample_limit == 0:
        return []

    selected = [case for case in comparisons if case.winner == winner]
    if winner == "challenger":
        selected.sort(key=lambda case: (-case.f1_delta, case.question_id))
    elif winner == "baseline":
        selected.sort(key=lambda case: (case.f1_delta, case.question_id))
    else:
        selected.sort(key=lambda case: (abs(case.f1_delta), case.question_id))
    return selected[:sample_limit]


def _selected_answer_text(case: dict[str, Any]) -> str:
    selected_candidates = case.get("selected_candidates", [])
    if not isinstance(selected_candidates, list):
        return ""

    return _truncate(
        " ".join(
            str(candidate.get("sentence", ""))
            for candidate in selected_candidates
            if isinstance(candidate, dict)
        )
    )


def _truncate(text: str, max_chars: int = 500) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."
