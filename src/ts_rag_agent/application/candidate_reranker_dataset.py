from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from ts_rag_agent.application.evidence_selection import (
    SentenceEvidenceCandidate,
    SentenceEvidenceSelector,
    classify_question_route,
    split_sentences,
    tokenize_text,
)
from ts_rag_agent.application.text_metrics import token_f1
from ts_rag_agent.domain.dataset import PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


@dataclass(frozen=True)
class CandidateRerankerDatasetRow:
    """One candidate-level row for offline reranker training data."""

    split: str
    question_id: str
    candidate_id: str
    candidate_rank: int
    runtime_features: dict[str, Any]
    gold_labels: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CandidateRerankerQuestionSummary:
    """Question-level oracle summary for the candidate pool."""

    split: str
    question_id: str
    question_route: str
    candidate_count: int
    gold_document_candidate_count: int
    top_candidate_token_f1: float
    best_candidate_token_f1: float
    best_candidate_rank: int | None
    oracle_gain_vs_top_candidate: float


@dataclass(frozen=True)
class CandidateRerankerDatasetSummary:
    """Aggregate summary for a candidate-reranker dataset build."""

    splits: list[str]
    selector_name: str
    total_questions: int
    total_rows: int
    questions_by_split: dict[str, int]
    rows_by_split: dict[str, int]
    rows_by_route: dict[str, int]
    average_rows_per_question: float
    average_top_candidate_token_f1: float
    average_best_candidate_token_f1: float
    average_oracle_gain_vs_top_candidate: float
    questions_with_gold_document_candidate: int
    gold_document_candidate_rows: int


@dataclass(frozen=True)
class CandidateRerankerDatasetBuild:
    """Full candidate-reranker dataset build result."""

    summary: CandidateRerankerDatasetSummary
    rows: list[CandidateRerankerDatasetRow]
    question_summaries: list[CandidateRerankerQuestionSummary]


def build_candidate_reranker_dataset(
    split_questions: dict[str, Sequence[PrimeQAQuestion]],
    search_fn: Callable[[PrimeQAQuestion, int], Sequence[RetrievalResult]],
    evidence_selector: SentenceEvidenceSelector,
    retrieval_top_k: int = 5,
    candidate_limit: int = 25,
    min_candidate_score: float = 2.0,
) -> CandidateRerankerDatasetBuild:
    """Build an offline feature dataset with runtime features and gold labels."""

    _validate_options(
        retrieval_top_k=retrieval_top_k,
        candidate_limit=candidate_limit,
        min_candidate_score=min_candidate_score,
    )
    if not split_questions:
        raise ValueError("split_questions must not be empty")

    rows = []
    question_summaries = []
    for split, questions in split_questions.items():
        for question in questions:
            if not question.answerable:
                continue

            retrieval_results = list(search_fn(question, retrieval_top_k))
            ranked_candidates = evidence_selector.rank_sentence_candidates(
                question,
                retrieval_results,
            )
            eligible_candidates = [
                candidate
                for candidate in ranked_candidates
                if candidate.score >= min_candidate_score
            ][:candidate_limit]
            question_rows = build_candidate_rows_for_question(
                split=split,
                question=question,
                candidates=eligible_candidates,
                selector_name=evidence_selector.name,
            )
            rows.extend(question_rows)
            question_summaries.append(
                summarize_question_candidates(
                    split=split,
                    question=question,
                    rows=question_rows,
                )
            )

    return CandidateRerankerDatasetBuild(
        summary=summarize_candidate_reranker_dataset(
            rows=rows,
            question_summaries=question_summaries,
            selector_name=evidence_selector.name,
        ),
        rows=rows,
        question_summaries=question_summaries,
    )


def build_candidate_rows_for_question(
    split: str,
    question: PrimeQAQuestion,
    candidates: Sequence[SentenceEvidenceCandidate],
    selector_name: str,
) -> list[CandidateRerankerDatasetRow]:
    """Build candidate rows for one question from an already-ranked pool."""

    question_route = classify_question_route(question)
    candidate_f1_values = [
        round(token_f1(candidate.sentence, question.answer), 4)
        for candidate in candidates
    ]
    best_f1 = max(candidate_f1_values, default=0.0)
    best_rank = _best_rank(candidate_f1_values)

    return [
        CandidateRerankerDatasetRow(
            split=split,
            question_id=question.id,
            candidate_id=f"{question.id}::candidate_{candidate_rank:03d}",
            candidate_rank=candidate_rank,
            runtime_features=_runtime_features(
                question=question,
                candidate=candidate,
                question_route=question_route,
                selector_name=selector_name,
            ),
            gold_labels=_gold_labels(
                question=question,
                candidate=candidate,
                candidate_token_f1=candidate_f1,
                best_candidate_token_f1=best_f1,
                is_best_candidate=candidate_rank == best_rank,
            ),
            metadata=_metadata(
                question=question,
                candidate=candidate,
                question_route=question_route,
            ),
        )
        for candidate_rank, (candidate, candidate_f1) in enumerate(
            zip(candidates, candidate_f1_values, strict=True),
            start=1,
        )
    ]


def summarize_question_candidates(
    split: str,
    question: PrimeQAQuestion,
    rows: Sequence[CandidateRerankerDatasetRow],
) -> CandidateRerankerQuestionSummary:
    """Summarize candidate labels for one question."""

    candidate_f1_values = [
        float(row.gold_labels["candidate_token_f1"])
        for row in rows
    ]
    best_f1 = max(candidate_f1_values, default=0.0)
    top_f1 = candidate_f1_values[0] if candidate_f1_values else 0.0
    best_rank = _best_rank(candidate_f1_values)
    gold_candidate_count = sum(
        bool(row.gold_labels["is_gold_document"])
        for row in rows
    )
    return CandidateRerankerQuestionSummary(
        split=split,
        question_id=question.id,
        question_route=classify_question_route(question),
        candidate_count=len(rows),
        gold_document_candidate_count=gold_candidate_count,
        top_candidate_token_f1=round(top_f1, 4),
        best_candidate_token_f1=round(best_f1, 4),
        best_candidate_rank=best_rank,
        oracle_gain_vs_top_candidate=round(best_f1 - top_f1, 4),
    )


def summarize_candidate_reranker_dataset(
    rows: Sequence[CandidateRerankerDatasetRow],
    question_summaries: Sequence[CandidateRerankerQuestionSummary],
    selector_name: str,
) -> CandidateRerankerDatasetSummary:
    """Summarize a candidate-reranker dataset build."""

    questions_by_split = Counter(summary.split for summary in question_summaries)
    rows_by_split = Counter(row.split for row in rows)
    rows_by_route = Counter(
        str(row.runtime_features["question_route"])
        for row in rows
    )
    total_questions = len(question_summaries)
    total_rows = len(rows)
    return CandidateRerankerDatasetSummary(
        splits=sorted(questions_by_split),
        selector_name=selector_name,
        total_questions=total_questions,
        total_rows=total_rows,
        questions_by_split=dict(questions_by_split),
        rows_by_split=dict(rows_by_split),
        rows_by_route=dict(rows_by_route),
        average_rows_per_question=_rounded_mean(
            [summary.candidate_count for summary in question_summaries]
        ),
        average_top_candidate_token_f1=_rounded_mean(
            [summary.top_candidate_token_f1 for summary in question_summaries]
        ),
        average_best_candidate_token_f1=_rounded_mean(
            [summary.best_candidate_token_f1 for summary in question_summaries]
        ),
        average_oracle_gain_vs_top_candidate=_rounded_mean(
            [summary.oracle_gain_vs_top_candidate for summary in question_summaries]
        ),
        questions_with_gold_document_candidate=sum(
            summary.gold_document_candidate_count > 0
            for summary in question_summaries
        ),
        gold_document_candidate_rows=sum(
            bool(row.gold_labels["is_gold_document"])
            for row in rows
        ),
    )


def candidate_reranker_dataset_build_to_dict(
    build: CandidateRerankerDatasetBuild,
) -> dict[str, Any]:
    """Convert a dataset build to JSON-safe metadata."""

    return {
        "summary": asdict(build.summary),
        "question_summaries": [
            asdict(summary)
            for summary in build.question_summaries
        ],
        "feature_contract": {
            "runtime_features": (
                "May be used by a reranker at runtime. These fields do not use "
                "gold answers or gold document labels."
            ),
            "gold_labels": (
                "Offline labels only. These fields use gold answers or gold "
                "document ids and must not be used as runtime features."
            ),
            "metadata": (
                "Inspection fields only. Text values may be truncated and should "
                "not be treated as model-ready numeric features."
            ),
        },
    }


def candidate_reranker_row_to_dict(
    row: CandidateRerankerDatasetRow,
) -> dict[str, Any]:
    """Convert one candidate row to a JSON-safe dictionary."""

    return asdict(row)


def _runtime_features(
    question: PrimeQAQuestion,
    candidate: SentenceEvidenceCandidate,
    question_route: str,
    selector_name: str,
) -> dict[str, Any]:
    question_terms = _content_terms(tokenize_text(question.full_question))
    title_terms = _content_terms(tokenize_text(candidate.retrieval_result.document.title))
    candidate_terms = _content_terms(tokenize_text(candidate.sentence))
    overlap_terms = question_terms & candidate_terms
    title_overlap_terms = question_terms & title_terms
    sentence_count = len(split_sentences(candidate.sentence))
    candidate_tokens = tokenize_text(candidate.sentence)
    return {
        "selector_name": selector_name,
        "question_route": question_route,
        "retrieval_rank": candidate.retrieval_result.rank,
        "retrieval_score": round(candidate.retrieval_result.score, 4),
        "candidate_score": round(candidate.score, 4),
        "candidate_token_count": len(candidate_tokens),
        "candidate_sentence_count": sentence_count,
        "question_token_count": len(tokenize_text(question.full_question)),
        "query_term_count": len(question_terms),
        "query_overlap_count": len(overlap_terms),
        "query_overlap_ratio": _ratio(len(overlap_terms), len(question_terms)),
        "candidate_query_coverage_ratio": _ratio(len(overlap_terms), len(candidate_terms)),
        "title_query_overlap_count": len(title_overlap_terms),
        "title_query_overlap_ratio": _ratio(len(title_overlap_terms), len(question_terms)),
        "answer_signal_score": _answer_signal_score(candidate.sentence),
        "problem_noise_score": _problem_noise_score(candidate.sentence),
        "has_answer_heading": _has_answer_heading(candidate.sentence),
        "has_problem_heading": _has_problem_heading(candidate.sentence),
        "has_question_heading": _has_question_heading(candidate.sentence),
        "has_url": bool(re.search(r"https?://|www\.", candidate.sentence.lower())),
        "has_trace_noise": bool(
            re.search(
                r"\b(trace|stack|dump|exception|heapdump|javacore)\b",
                candidate.sentence.lower(),
            )
        ),
        "symbol_ratio": _symbol_ratio(candidate.sentence),
    }


def _gold_labels(
    question: PrimeQAQuestion,
    candidate: SentenceEvidenceCandidate,
    candidate_token_f1: float,
    best_candidate_token_f1: float,
    is_best_candidate: bool,
) -> dict[str, Any]:
    return {
        "candidate_token_f1": candidate_token_f1,
        "is_gold_document": candidate.retrieval_result.document.id == question.answer_doc_id,
        "is_best_candidate_for_question": is_best_candidate,
        "best_candidate_token_f1_for_question": best_candidate_token_f1,
        "f1_gap_to_best_candidate": round(
            max(0.0, best_candidate_token_f1 - candidate_token_f1),
            4,
        ),
    }


def _metadata(
    question: PrimeQAQuestion,
    candidate: SentenceEvidenceCandidate,
    question_route: str,
) -> dict[str, Any]:
    document = candidate.retrieval_result.document
    return {
        "question_title": _truncate(question.title),
        "question_route": question_route,
        "document_id": document.id,
        "document_title": _truncate(document.title),
        "candidate_sentence": _truncate(candidate.sentence),
    }


def _best_rank(candidate_f1_values: Sequence[float]) -> int | None:
    if not candidate_f1_values:
        return None
    best_f1 = max(candidate_f1_values)
    return candidate_f1_values.index(best_f1) + 1


def _answer_signal_score(text: str) -> float:
    normalized = text.lower()
    score = 0.0
    if _has_answer_heading(text):
        score += 2.0
    if re.search(r"\b(workaround|fix|corrective action|local fix)\b", normalized):
        score += 1.2
    if re.search(
        r"\b(install|upgrade|configure|restart|set|enable|disable|apply|run|use)\b",
        normalized,
    ):
        score += 0.8
    if re.search(r"\b(required|must|should|recommended|supported)\b", normalized):
        score += 0.4
    return round(score, 4)


def _problem_noise_score(text: str) -> float:
    normalized = text.lower()
    score = 0.0
    if _has_problem_heading(text):
        score += 1.0
    if _has_question_heading(text):
        score += 0.7
    if re.search(r"\b(trace|stack|dump|exception|heapdump|javacore)\b", normalized):
        score += 0.5
    if len(tokenize_text(text)) > 120:
        score += 0.8
    return round(score, 4)


def _has_answer_heading(text: str) -> bool:
    return bool(
        re.search(
            r"\b(resolving the problem|resolution|solution|answer)\b",
            text.lower(),
        )
    )


def _has_problem_heading(text: str) -> bool:
    return bool(
        re.search(
            r"\b(problem\(abstract\)|symptom|environment|"
            r"diagnosing the problem|collecting data|steps to reproduce)\b",
            text.lower(),
        )
    )


def _has_question_heading(text: str) -> bool:
    return bool(re.search(r"\b(question|technote \(faq\))\b", text.lower()))


def _symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    symbol_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
    return round(symbol_count / len(text), 4)


def _content_terms(tokens: list[str]) -> set[str]:
    return {
        token
        for token in tokens
        if token not in _STOPWORDS and len(token) > 1
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _rounded_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(mean(values), 4)


def _truncate(text: str, max_chars: int = 700) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."


def _validate_options(
    retrieval_top_k: int,
    candidate_limit: int,
    min_candidate_score: float,
) -> None:
    if retrieval_top_k <= 0:
        raise ValueError("retrieval_top_k must be positive")
    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive")
    if min_candidate_score < 0:
        raise ValueError("min_candidate_score must be non-negative")


_STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "get",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "with",
    "you",
}
