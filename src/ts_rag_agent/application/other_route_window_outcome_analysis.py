from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class OtherRouteWindowCase:
    """One other-route comparison between baseline and answer-window selectors."""

    source_label: str
    question_id: str
    subtype: str
    winner: str
    baseline_f1: float
    challenger_f1: float
    f1_delta: float
    baseline_gold_cited: bool
    challenger_gold_cited: bool
    question_title: str


@dataclass(frozen=True)
class OtherRouteSubtypeSummary:
    """Aggregate outcome for one other-route subtype."""

    source_label: str
    subtype: str
    total_cases: int
    baseline_wins: int
    challenger_wins: int
    ties: int
    avg_f1_delta: float
    baseline_gold_citation_count: int
    challenger_gold_citation_count: int
    citation_delta: int
    recommendation: str
    sample_cases: list[OtherRouteWindowCase]


@dataclass(frozen=True)
class OtherRouteWindowOutcomeAnalysis:
    """Cross-report analysis of answer-window outcomes on the other route."""

    source_count: int
    total_cases: int
    subtype_counts: dict[str, int]
    source_counts: dict[str, int]
    overall_subtype_summaries: list[OtherRouteSubtypeSummary]
    source_subtype_summaries: list[OtherRouteSubtypeSummary]
    stable_answer_window_subtypes: list[str]
    mixed_subtypes: list[str]
    top_summary: str


def analyze_other_route_window_outcomes(
    baseline_reports: list[dict[str, Any]],
    challenger_reports: list[dict[str, Any]],
    source_labels: list[str],
    min_cases: int = 3,
    f1_win_margin: float = 0.03,
    sample_limit_per_subtype: int = 5,
) -> OtherRouteWindowOutcomeAnalysis:
    """Analyze whether answer-window helps stable question subtypes."""

    _validate_inputs(
        baseline_reports=baseline_reports,
        challenger_reports=challenger_reports,
        source_labels=source_labels,
        min_cases=min_cases,
        f1_win_margin=f1_win_margin,
        sample_limit_per_subtype=sample_limit_per_subtype,
    )

    cases = []
    for source_label, baseline_report, challenger_report in zip(
        source_labels,
        baseline_reports,
        challenger_reports,
        strict=True,
    ):
        cases.extend(
            _compare_other_route_cases(
                source_label=source_label,
                baseline_report=baseline_report,
                challenger_report=challenger_report,
                f1_win_margin=f1_win_margin,
            )
        )

    overall_summaries = _build_subtype_summaries(
        cases=cases,
        source_label="all",
        min_cases=min_cases,
        sample_limit_per_subtype=sample_limit_per_subtype,
    )
    source_summaries = []
    for source_label in source_labels:
        source_cases = [case for case in cases if case.source_label == source_label]
        source_summaries.extend(
            _build_subtype_summaries(
                cases=source_cases,
                source_label=source_label,
                min_cases=min_cases,
                sample_limit_per_subtype=sample_limit_per_subtype,
            )
        )

    stable_answer_window_subtypes = _stable_answer_window_subtypes(
        source_summaries=source_summaries,
        source_labels=source_labels,
    )
    mixed_subtypes = _mixed_subtypes(
        overall_summaries=overall_summaries,
        source_summaries=source_summaries,
        stable_answer_window_subtypes=stable_answer_window_subtypes,
    )

    return OtherRouteWindowOutcomeAnalysis(
        source_count=len(source_labels),
        total_cases=len(cases),
        subtype_counts=dict(Counter(case.subtype for case in cases)),
        source_counts=dict(Counter(case.source_label for case in cases)),
        overall_subtype_summaries=overall_summaries,
        source_subtype_summaries=source_summaries,
        stable_answer_window_subtypes=stable_answer_window_subtypes,
        mixed_subtypes=mixed_subtypes,
        top_summary=_build_top_summary(overall_summaries, stable_answer_window_subtypes),
    )


def other_route_window_outcome_analysis_to_dict(
    analysis: OtherRouteWindowOutcomeAnalysis,
) -> dict[str, Any]:
    """Convert analysis to a JSON-safe dictionary."""

    return {
        "source_count": analysis.source_count,
        "total_cases": analysis.total_cases,
        "subtype_counts": analysis.subtype_counts,
        "source_counts": analysis.source_counts,
        "recommendation_note": (
            "Heuristic only: subtype labels use question title/text features that are "
            "available at runtime, but this report does not change runtime routing."
        ),
        "stable_answer_window_subtypes": analysis.stable_answer_window_subtypes,
        "mixed_subtypes": analysis.mixed_subtypes,
        "top_summary": analysis.top_summary,
        "overall_subtype_summaries": [
            asdict(summary) for summary in analysis.overall_subtype_summaries
        ],
        "source_subtype_summaries": [
            asdict(summary) for summary in analysis.source_subtype_summaries
        ],
    }


def classify_other_route_subtype(question_title: str, question_text: str = "") -> str:
    """Classify an other-route question using only runtime-visible text."""

    combined = f"{question_title}\n{question_text}".lower()
    title = question_title.strip().lower()

    if _contains_any(
        combined,
        (
            "support's guide",
            "support guide",
            "attached",
            "attachment",
            ".pdf",
            "download",
            "where can i download",
        ),
    ):
        return "support_or_download"
    if _contains_any(
        combined,
        (
            "how to",
            "steps",
            "procedure",
            "hide",
            "delete",
            "create",
            "modify",
            "change",
            "migrate",
            "convert",
            "enable",
            "disable",
            "retrieve",
            "open module",
        ),
    ):
        return "procedure_or_change"
    if _contains_any(
        combined,
        (
            "configuration",
            "configure",
            "property",
            "parameter",
            "xml",
            "libpath",
            "output_type",
            "setting",
            "timeout",
        ),
    ):
        return "configuration_or_property"
    if title.startswith(("can ", "is ", "does ", "do ", "would ", "want ")) or (
        "supported" in combined or "compliant" in combined
    ):
        return "capability_or_support"
    if title.startswith("why ") or _contains_any(
        combined,
        (
            "unable",
            "can't",
            "cannot",
            "not able",
            "fail",
            "failed",
            "problem",
            "refused",
            "unavailable",
        ),
    ):
        return "failure_or_behavior"
    if title.startswith(("what exactly", "what is")):
        return "definition_or_lookup"
    return "general_other"


def _compare_other_route_cases(
    source_label: str,
    baseline_report: dict[str, Any],
    challenger_report: dict[str, Any],
    f1_win_margin: float,
) -> list[OtherRouteWindowCase]:
    baseline_cases = _cases_by_question_id(baseline_report)
    challenger_cases = _cases_by_question_id(challenger_report)
    cases = []

    for question_id in sorted(baseline_cases.keys() & challenger_cases.keys()):
        baseline_case = baseline_cases[question_id]
        challenger_case = challenger_cases[question_id]
        if str(challenger_case.get("question_route", "")) != "other":
            continue

        baseline_f1 = _safe_float(baseline_case.get("selected_answer_token_f1"))
        challenger_f1 = _safe_float(challenger_case.get("selected_answer_token_f1"))
        f1_delta = round(challenger_f1 - baseline_f1, 4)
        baseline_gold_cited = _safe_int(
            baseline_case.get("selected_gold_candidate_count")
        ) > 0
        challenger_gold_cited = _safe_int(
            challenger_case.get("selected_gold_candidate_count")
        ) > 0
        question_title = str(baseline_case.get("question_title", ""))
        question_text = str(baseline_case.get("question_text", ""))
        cases.append(
            OtherRouteWindowCase(
                source_label=source_label,
                question_id=question_id,
                subtype=classify_other_route_subtype(question_title, question_text),
                winner=_choose_winner(
                    f1_delta=f1_delta,
                    f1_win_margin=f1_win_margin,
                    baseline_gold_cited=baseline_gold_cited,
                    challenger_gold_cited=challenger_gold_cited,
                ),
                baseline_f1=baseline_f1,
                challenger_f1=challenger_f1,
                f1_delta=f1_delta,
                baseline_gold_cited=baseline_gold_cited,
                challenger_gold_cited=challenger_gold_cited,
                question_title=question_title,
            )
        )

    return cases


def _build_subtype_summaries(
    cases: list[OtherRouteWindowCase],
    source_label: str,
    min_cases: int,
    sample_limit_per_subtype: int,
) -> list[OtherRouteSubtypeSummary]:
    cases_by_subtype: dict[str, list[OtherRouteWindowCase]] = defaultdict(list)
    for case in cases:
        cases_by_subtype[case.subtype].append(case)

    summaries = [
        _build_subtype_summary(
            source_label=source_label,
            subtype=subtype,
            cases=subtype_cases,
            sample_limit_per_subtype=sample_limit_per_subtype,
        )
        for subtype, subtype_cases in cases_by_subtype.items()
        if len(subtype_cases) >= min_cases
    ]
    return sorted(
        summaries,
        key=lambda summary: (
            summary.recommendation != "candidate_answer_window",
            -summary.avg_f1_delta,
            -summary.total_cases,
            summary.subtype,
        ),
    )


def _build_subtype_summary(
    source_label: str,
    subtype: str,
    cases: list[OtherRouteWindowCase],
    sample_limit_per_subtype: int,
) -> OtherRouteSubtypeSummary:
    winner_counts = Counter(case.winner for case in cases)
    baseline_gold_citation_count = sum(case.baseline_gold_cited for case in cases)
    challenger_gold_citation_count = sum(case.challenger_gold_cited for case in cases)
    avg_f1_delta = round(mean(case.f1_delta for case in cases), 4)
    recommendation = _recommendation(
        baseline_wins=winner_counts.get("baseline", 0),
        challenger_wins=winner_counts.get("challenger", 0),
        avg_f1_delta=avg_f1_delta,
    )

    return OtherRouteSubtypeSummary(
        source_label=source_label,
        subtype=subtype,
        total_cases=len(cases),
        baseline_wins=winner_counts.get("baseline", 0),
        challenger_wins=winner_counts.get("challenger", 0),
        ties=winner_counts.get("tie", 0),
        avg_f1_delta=avg_f1_delta,
        baseline_gold_citation_count=baseline_gold_citation_count,
        challenger_gold_citation_count=challenger_gold_citation_count,
        citation_delta=challenger_gold_citation_count - baseline_gold_citation_count,
        recommendation=recommendation,
        sample_cases=_select_sample_cases(cases, recommendation, sample_limit_per_subtype),
    )


def _stable_answer_window_subtypes(
    source_summaries: list[OtherRouteSubtypeSummary],
    source_labels: list[str],
) -> list[str]:
    summaries_by_subtype: dict[str, dict[str, OtherRouteSubtypeSummary]] = defaultdict(dict)
    for summary in source_summaries:
        summaries_by_subtype[summary.subtype][summary.source_label] = summary

    stable_subtypes = []
    for subtype, summaries_by_source in summaries_by_subtype.items():
        if set(summaries_by_source) != set(source_labels):
            continue
        if all(
            summary.recommendation == "candidate_answer_window"
            for summary in summaries_by_source.values()
        ):
            stable_subtypes.append(subtype)
    return sorted(stable_subtypes)


def _mixed_subtypes(
    overall_summaries: list[OtherRouteSubtypeSummary],
    source_summaries: list[OtherRouteSubtypeSummary],
    stable_answer_window_subtypes: list[str],
) -> list[str]:
    stable = set(stable_answer_window_subtypes)
    mixed = {
        summary.subtype
        for summary in overall_summaries
        if summary.subtype not in stable
        and summary.recommendation == "candidate_answer_window"
    }
    recommendations_by_subtype: dict[str, set[str]] = defaultdict(set)
    for summary in source_summaries:
        recommendations_by_subtype[summary.subtype].add(summary.recommendation)
    for subtype, recommendations in recommendations_by_subtype.items():
        if subtype not in stable and len(recommendations) > 1:
            mixed.add(subtype)
    return sorted(mixed)


def _build_top_summary(
    overall_summaries: list[OtherRouteSubtypeSummary],
    stable_answer_window_subtypes: list[str],
) -> str:
    if stable_answer_window_subtypes:
        return (
            "Stable answer-window candidate subtype(s): "
            f"{', '.join(stable_answer_window_subtypes)}."
        )
    if not overall_summaries:
        return "No other-route subtype met the minimum case threshold."
    top = overall_summaries[0]
    return (
        f"No subtype was a stable cross-source answer-window candidate. "
        f"Best overall subtype was {top.subtype} with avg F1 delta "
        f"{top.avg_f1_delta:+.4f} across {top.total_cases} cases."
    )


def _recommendation(
    baseline_wins: int,
    challenger_wins: int,
    avg_f1_delta: float,
) -> str:
    if avg_f1_delta > 0 and challenger_wins > baseline_wins:
        return "candidate_answer_window"
    if avg_f1_delta < 0 and baseline_wins >= challenger_wins:
        return "keep_baseline"
    return "mixed_or_insufficient"


def _select_sample_cases(
    cases: list[OtherRouteWindowCase],
    recommendation: str,
    sample_limit: int,
) -> list[OtherRouteWindowCase]:
    if sample_limit == 0:
        return []
    if recommendation == "candidate_answer_window":
        ranked_cases = sorted(cases, key=lambda case: (-case.f1_delta, case.question_id))
    elif recommendation == "keep_baseline":
        ranked_cases = sorted(cases, key=lambda case: (case.f1_delta, case.question_id))
    else:
        ranked_cases = sorted(
            cases,
            key=lambda case: (-abs(case.f1_delta), case.question_id),
        )
    return ranked_cases[:sample_limit]


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


def _cases_by_question_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError("each report must contain a list field named 'cases'")
    return {
        str(case["question_id"]): case
        for case in cases
        if isinstance(case, dict) and "question_id" in case
    }


def _validate_inputs(
    baseline_reports: list[dict[str, Any]],
    challenger_reports: list[dict[str, Any]],
    source_labels: list[str],
    min_cases: int,
    f1_win_margin: float,
    sample_limit_per_subtype: int,
) -> None:
    if not baseline_reports:
        raise ValueError("baseline_reports must not be empty")
    if len(baseline_reports) != len(challenger_reports):
        raise ValueError("baseline_reports and challenger_reports must have the same length")
    if len(baseline_reports) != len(source_labels):
        raise ValueError("source_labels length must match report lists")
    if len(set(source_labels)) != len(source_labels):
        raise ValueError("source_labels must be unique")
    if min_cases <= 0:
        raise ValueError("min_cases must be positive")
    if f1_win_margin < 0:
        raise ValueError("f1_win_margin must be non-negative")
    if sample_limit_per_subtype < 0:
        raise ValueError("sample_limit_per_subtype must be non-negative")


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


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
