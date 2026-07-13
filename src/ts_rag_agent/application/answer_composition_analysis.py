from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from ts_rag_agent.application.text_metrics import token_f1


@dataclass(frozen=True)
class CompositionCandidate:
    """One selected evidence candidate from an answer-gap report."""

    document_id: str
    retrieval_rank: int | None
    sentence: str
    candidate_token_f1: float


@dataclass(frozen=True)
class CompositionCase:
    """One question-level answer-composition analysis case."""

    question_id: str
    question_route: str
    selected_selector_name: str
    selected_candidate_count: int
    selected_document_count: int
    duplicate_candidate_count: int
    current_f1: float
    top1_f1: float
    top2_f1: float
    dedup_top3_f1: float
    best_single_oracle_f1: float
    best_prefix_oracle_f1: float
    best_same_doc_prefix_oracle_f1: float
    best_oracle_strategy: str
    best_oracle_answer: str
    question_title: str
    current_answer: str


@dataclass(frozen=True)
class CompositionSummary:
    """Aggregate answer-composition metrics."""

    total_cases: int
    average_current_f1: float
    average_top1_f1: float
    average_top2_f1: float
    average_dedup_top3_f1: float
    average_best_single_oracle_f1: float
    average_best_prefix_oracle_f1: float
    average_best_same_doc_prefix_oracle_f1: float
    average_best_oracle_gain: float
    top1_beats_current: int
    top2_beats_current: int
    dedup_top3_beats_current: int
    best_single_oracle_beats_current: int
    best_prefix_oracle_beats_current: int
    best_same_doc_prefix_oracle_beats_current: int
    multi_document_answer_count: int
    duplicate_answer_count: int
    question_route_counts: dict[str, int]
    best_oracle_strategy_counts: dict[str, int]


@dataclass(frozen=True)
class CompositionAnalysisResult:
    """Full answer-composition analysis result."""

    summary: CompositionSummary
    top1_gain_cases: list[CompositionCase]
    dedup_gain_cases: list[CompositionCase]
    oracle_gap_cases: list[CompositionCase]


def analyze_answer_composition_report(
    answer_gap_report: dict[str, Any],
    f1_gain_margin: float = 0.03,
    sample_limit_per_bucket: int = 20,
) -> CompositionAnalysisResult:
    """Analyze answer composition choices from an answer-gap JSON report."""

    if f1_gain_margin < 0:
        raise ValueError("f1_gain_margin must be non-negative")
    if sample_limit_per_bucket < 0:
        raise ValueError("sample_limit_per_bucket must be non-negative")

    raw_cases = answer_gap_report.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("answer_gap_report must contain a list field named 'cases'")

    cases = [
        _analyze_case(raw_case)
        for raw_case in raw_cases
        if isinstance(raw_case, dict)
    ]
    return CompositionAnalysisResult(
        summary=_build_summary(cases, f1_gain_margin),
        top1_gain_cases=_select_cases(
            cases=cases,
            score_fn=lambda case: case.top1_f1 - case.current_f1,
            f1_gain_margin=f1_gain_margin,
            sample_limit=sample_limit_per_bucket,
        ),
        dedup_gain_cases=_select_cases(
            cases=cases,
            score_fn=lambda case: case.dedup_top3_f1 - case.current_f1,
            f1_gain_margin=f1_gain_margin,
            sample_limit=sample_limit_per_bucket,
        ),
        oracle_gap_cases=_select_cases(
            cases=cases,
            score_fn=lambda case: case.best_prefix_oracle_f1 - case.current_f1,
            f1_gain_margin=f1_gain_margin,
            sample_limit=sample_limit_per_bucket,
        ),
    )


def composition_result_to_dict(result: CompositionAnalysisResult) -> dict[str, Any]:
    """Convert composition analysis to a JSON-safe dictionary."""

    return {
        "summary": asdict(result.summary),
        "top1_gain_cases": [asdict(case) for case in result.top1_gain_cases],
        "dedup_gain_cases": [asdict(case) for case in result.dedup_gain_cases],
        "oracle_gap_cases": [asdict(case) for case in result.oracle_gap_cases],
    }


def _analyze_case(raw_case: dict[str, Any]) -> CompositionCase:
    gold_answer = str(raw_case.get("gold_answer", ""))
    candidates = _load_candidates(raw_case)
    current_answer = _join_candidate_sentences(candidates)
    current_f1 = float(raw_case.get("selected_answer_token_f1", 0.0))
    if current_f1 == 0.0 and current_answer:
        current_f1 = token_f1(current_answer, gold_answer)

    top1_answer = _join_candidate_sentences(candidates[:1])
    top2_answer = _join_candidate_sentences(candidates[:2])
    dedup_candidates = _deduplicate_candidates(candidates)
    dedup_top3_answer = _join_candidate_sentences(dedup_candidates[:3])

    single_oracle_answer = _best_single_candidate_answer(candidates, gold_answer)
    prefix_oracle_answer = _best_prefix_answer(candidates, gold_answer)
    same_doc_oracle_answer = _best_same_document_prefix_answer(candidates, gold_answer)
    oracle_answers = {
        "best_single": single_oracle_answer,
        "best_prefix": prefix_oracle_answer,
        "best_same_doc_prefix": same_doc_oracle_answer,
    }
    best_oracle_strategy, best_oracle_answer = max(
        oracle_answers.items(),
        key=lambda item: token_f1(item[1], gold_answer),
    )

    return CompositionCase(
        question_id=str(raw_case.get("question_id", "")),
        question_route=str(raw_case.get("question_route", "")),
        selected_selector_name=str(raw_case.get("selected_selector_name", "")),
        selected_candidate_count=len(candidates),
        selected_document_count=len({candidate.document_id for candidate in candidates}),
        duplicate_candidate_count=len(candidates) - len(_deduplicate_candidates(candidates)),
        current_f1=round(current_f1, 4),
        top1_f1=round(token_f1(top1_answer, gold_answer), 4),
        top2_f1=round(token_f1(top2_answer, gold_answer), 4),
        dedup_top3_f1=round(token_f1(dedup_top3_answer, gold_answer), 4),
        best_single_oracle_f1=round(token_f1(single_oracle_answer, gold_answer), 4),
        best_prefix_oracle_f1=round(token_f1(prefix_oracle_answer, gold_answer), 4),
        best_same_doc_prefix_oracle_f1=round(token_f1(same_doc_oracle_answer, gold_answer), 4),
        best_oracle_strategy=best_oracle_strategy,
        best_oracle_answer=_truncate(best_oracle_answer),
        question_title=str(raw_case.get("question_title", "")),
        current_answer=_truncate(current_answer),
    )


def _load_candidates(raw_case: dict[str, Any]) -> list[CompositionCandidate]:
    raw_candidates = raw_case.get("selected_candidates", [])
    if not isinstance(raw_candidates, list):
        return []

    candidates = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            continue
        candidates.append(
            CompositionCandidate(
                document_id=str(raw_candidate.get("document_id", "")),
                retrieval_rank=_safe_int(raw_candidate.get("retrieval_rank")),
                sentence=str(raw_candidate.get("sentence", "")),
                candidate_token_f1=float(raw_candidate.get("candidate_token_f1", 0.0)),
            )
        )
    return candidates


def _join_candidate_sentences(candidates: list[CompositionCandidate]) -> str:
    return " ".join(candidate.sentence for candidate in candidates if candidate.sentence)


def _deduplicate_candidates(
    candidates: list[CompositionCandidate],
    duplicate_threshold: float = 0.9,
) -> list[CompositionCandidate]:
    kept = []
    kept_token_sets: list[set[str]] = []
    for candidate in candidates:
        token_set = set(_tokens(candidate.sentence))
        if not token_set:
            continue
        if any(
            _jaccard(token_set, kept_set) >= duplicate_threshold
            for kept_set in kept_token_sets
        ):
            continue
        kept.append(candidate)
        kept_token_sets.append(token_set)
    return kept


def _best_single_candidate_answer(
    candidates: list[CompositionCandidate],
    gold_answer: str,
) -> str:
    if not candidates:
        return ""
    return max(candidates, key=lambda candidate: token_f1(candidate.sentence, gold_answer)).sentence


def _best_prefix_answer(
    candidates: list[CompositionCandidate],
    gold_answer: str,
) -> str:
    prefixes = [
        _join_candidate_sentences(candidates[:prefix_size])
        for prefix_size in range(1, len(candidates) + 1)
    ]
    return _best_answer(prefixes, gold_answer)


def _best_same_document_prefix_answer(
    candidates: list[CompositionCandidate],
    gold_answer: str,
) -> str:
    prefixes = []
    document_ids = []
    for candidate in candidates:
        if candidate.document_id not in document_ids:
            document_ids.append(candidate.document_id)

    for document_id in document_ids:
        document_candidates = [
            candidate for candidate in candidates if candidate.document_id == document_id
        ]
        prefixes.extend(
            _join_candidate_sentences(document_candidates[:prefix_size])
            for prefix_size in range(1, len(document_candidates) + 1)
        )
    return _best_answer(prefixes, gold_answer)


def _best_answer(answers: list[str], gold_answer: str) -> str:
    if not answers:
        return ""
    return max(answers, key=lambda answer: token_f1(answer, gold_answer))


def _build_summary(
    cases: list[CompositionCase],
    f1_gain_margin: float,
) -> CompositionSummary:
    return CompositionSummary(
        total_cases=len(cases),
        average_current_f1=_average(case.current_f1 for case in cases),
        average_top1_f1=_average(case.top1_f1 for case in cases),
        average_top2_f1=_average(case.top2_f1 for case in cases),
        average_dedup_top3_f1=_average(case.dedup_top3_f1 for case in cases),
        average_best_single_oracle_f1=_average(
            case.best_single_oracle_f1 for case in cases
        ),
        average_best_prefix_oracle_f1=_average(
            case.best_prefix_oracle_f1 for case in cases
        ),
        average_best_same_doc_prefix_oracle_f1=_average(
            case.best_same_doc_prefix_oracle_f1 for case in cases
        ),
        average_best_oracle_gain=_average(
            case.best_prefix_oracle_f1 - case.current_f1 for case in cases
        ),
        top1_beats_current=_count_gain(
            cases,
            lambda case: case.top1_f1 - case.current_f1,
            f1_gain_margin,
        ),
        top2_beats_current=_count_gain(
            cases,
            lambda case: case.top2_f1 - case.current_f1,
            f1_gain_margin,
        ),
        dedup_top3_beats_current=_count_gain(
            cases,
            lambda case: case.dedup_top3_f1 - case.current_f1,
            f1_gain_margin,
        ),
        best_single_oracle_beats_current=_count_gain(
            cases,
            lambda case: case.best_single_oracle_f1 - case.current_f1,
            f1_gain_margin,
        ),
        best_prefix_oracle_beats_current=_count_gain(
            cases,
            lambda case: case.best_prefix_oracle_f1 - case.current_f1,
            f1_gain_margin,
        ),
        best_same_doc_prefix_oracle_beats_current=_count_gain(
            cases,
            lambda case: case.best_same_doc_prefix_oracle_f1 - case.current_f1,
            f1_gain_margin,
        ),
        multi_document_answer_count=sum(1 for case in cases if case.selected_document_count > 1),
        duplicate_answer_count=sum(1 for case in cases if case.duplicate_candidate_count > 0),
        question_route_counts=dict(Counter(case.question_route for case in cases)),
        best_oracle_strategy_counts=dict(Counter(case.best_oracle_strategy for case in cases)),
    )


def _select_cases(
    cases: list[CompositionCase],
    score_fn,
    f1_gain_margin: float,
    sample_limit: int,
) -> list[CompositionCase]:
    if sample_limit == 0:
        return []

    selected = [case for case in cases if score_fn(case) >= f1_gain_margin]
    selected.sort(key=lambda case: (-score_fn(case), case.question_id))
    return selected[:sample_limit]


def _count_gain(cases: list[CompositionCase], score_fn, f1_gain_margin: float) -> int:
    return sum(1 for case in cases if score_fn(case) >= f1_gain_margin)


def _average(values) -> float:
    materialized_values = list(values)
    if not materialized_values:
        return 0.0
    return round(sum(materialized_values) / len(materialized_values), 4)


def _tokens(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truncate(text: str, max_chars: int = 700) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."
