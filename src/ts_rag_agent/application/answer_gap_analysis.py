from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean

from ts_rag_agent.application.evidence_selection import (
    SentenceEvidenceCandidate,
    split_sentences,
    trace_selector_route,
)
from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator
from ts_rag_agent.application.retrieval_evaluation import Retriever
from ts_rag_agent.application.text_metrics import token_f1
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult

ANSWER_GAP_BUCKET_DEFINITIONS = {
    "gold_not_in_context": "Gold document was not retrieved in the selected top-k.",
    "gold_in_context_not_selected": (
        "Gold document was retrieved, but the answer used only non-gold evidence."
    ),
    "gold_window_beats_selected_answer": (
        "A contiguous sentence window inside the gold document has clearly higher "
        "token F1 than the selected answer."
    ),
    "gold_sentence_beats_selected_answer": (
        "A single sentence inside the gold document has clearly higher token F1 "
        "than the selected answer."
    ),
    "selected_answer_low_overlap": (
        "The selected answer cites the gold document but still has low token overlap."
    ),
    "selected_answer_reasonable_overlap": (
        "The selected answer cites the gold document and no stronger local gold "
        "sentence/window gap was detected."
    ),
    "gold_document_missing": "The labeled gold document is not available locally.",
}


@dataclass(frozen=True)
class GoldEvidenceSpan:
    """Best matching sentence or contiguous sentence window from the gold document."""

    text: str
    token_f1: float
    sentence_start: int
    sentence_count: int


@dataclass(frozen=True)
class AnswerGapCase:
    """One answerable question's gap between selected evidence and gold answer."""

    question: PrimeQAQuestion
    retrieval_top_k: int
    gold_retrieval_rank: int | None
    selected_candidates: list[SentenceEvidenceCandidate]
    selected_answer_text: str
    selected_answer_token_f1: float
    selected_gold_candidate_count: int
    question_route: str
    selected_selector_name: str
    route_reason: str
    best_gold_sentence: GoldEvidenceSpan | None
    best_gold_window: GoldEvidenceSpan | None
    best_selected_gold_candidate_token_f1: float | None
    bucket: str


@dataclass(frozen=True)
class AnswerGapSummary:
    """Aggregate answer-gap metrics for one experiment configuration."""

    total_answerable_questions: int
    gold_document_available: int
    gold_in_context: int
    selected_gold_citation: int
    bucket_counts: dict[str, int]
    average_selected_answer_token_f1: float
    average_best_gold_sentence_token_f1: float
    average_best_gold_window_token_f1: float
    selected_answer_low_overlap: int
    gold_sentence_beats_selected_answer: int
    gold_window_beats_selected_answer: int
    question_route_counts: dict[str, int]
    selected_selector_counts: dict[str, int]


@dataclass(frozen=True)
class AnswerGapAnalysisResult:
    """Full answer-gap analysis result."""

    summary: AnswerGapSummary
    cases: list[AnswerGapCase]


class AnswerGapAnalyzer:
    """Compare selected RAG evidence with answer-like text inside gold documents."""

    def __init__(
        self,
        retriever: Retriever,
        answer_generator: ExtractiveAnswerGenerator,
        documents_by_id: Mapping[str, PrimeQADocument],
        min_gold_sentence_chars: int = 24,
        max_window_sentences: int = 3,
        f1_gap_margin: float = 0.05,
        low_f1_threshold: float = 0.2,
    ) -> None:
        if min_gold_sentence_chars <= 0:
            raise ValueError("min_gold_sentence_chars must be positive")
        if max_window_sentences <= 0:
            raise ValueError("max_window_sentences must be positive")
        if f1_gap_margin < 0:
            raise ValueError("f1_gap_margin must be non-negative")
        if low_f1_threshold < 0:
            raise ValueError("low_f1_threshold must be non-negative")

        self._retriever = retriever
        self._answer_generator = answer_generator
        self._documents_by_id = documents_by_id
        self._min_gold_sentence_chars = min_gold_sentence_chars
        self._max_window_sentences = max_window_sentences
        self._f1_gap_margin = f1_gap_margin
        self._low_f1_threshold = low_f1_threshold

    def analyze(
        self,
        questions: Sequence[PrimeQAQuestion],
        retrieval_top_k: int = 5,
        sample_limit: int = 30,
    ) -> AnswerGapAnalysisResult:
        """Analyze answerable questions and keep representative gap cases."""

        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be positive")
        if sample_limit < 0:
            raise ValueError("sample_limit must be non-negative")

        answerable_questions = [question for question in questions if question.answerable]
        all_cases = [
            self._analyze_question(question, retrieval_top_k)
            for question in answerable_questions
        ]
        return AnswerGapAnalysisResult(
            summary=_build_summary(all_cases),
            cases=_select_cases(all_cases, sample_limit),
        )

    def _analyze_question(
        self,
        question: PrimeQAQuestion,
        retrieval_top_k: int,
    ) -> AnswerGapCase:
        retrieval_results = self._retriever.search(question.full_question, top_k=retrieval_top_k)
        route_trace = trace_selector_route(
            question,
            self._answer_generator.evidence_selector_name,
        )
        selected_candidates = self._select_answer_candidates(question, retrieval_results)
        selected_answer_text = " ".join(candidate.sentence for candidate in selected_candidates)
        selected_answer_token_f1 = token_f1(selected_answer_text, question.answer)
        selected_gold_candidates = [
            candidate
            for candidate in selected_candidates
            if candidate.retrieval_result.document.id == question.answer_doc_id
        ]
        selected_gold_candidate_f1_values = [
            token_f1(candidate.sentence, question.answer)
            for candidate in selected_gold_candidates
        ]
        gold_document = (
            self._documents_by_id.get(question.answer_doc_id)
            if question.answer_doc_id
            else None
        )
        best_gold_sentence = self._best_gold_span(
            question=question,
            gold_document=gold_document,
            max_window_sentences=1,
        )
        best_gold_window = self._best_gold_span(
            question=question,
            gold_document=gold_document,
            max_window_sentences=self._max_window_sentences,
        )
        gold_retrieval_rank = _find_gold_retrieval_rank(
            question.answer_doc_id,
            retrieval_results,
        )
        bucket = _categorize_case(
            gold_document=gold_document,
            gold_retrieval_rank=gold_retrieval_rank,
            selected_gold_candidate_count=len(selected_gold_candidates),
            selected_answer_token_f1=selected_answer_token_f1,
            best_gold_sentence=best_gold_sentence,
            best_gold_window=best_gold_window,
            f1_gap_margin=self._f1_gap_margin,
            low_f1_threshold=self._low_f1_threshold,
        )

        return AnswerGapCase(
            question=question,
            retrieval_top_k=retrieval_top_k,
            gold_retrieval_rank=gold_retrieval_rank,
            selected_candidates=selected_candidates,
            selected_answer_text=selected_answer_text,
            selected_answer_token_f1=round(selected_answer_token_f1, 4),
            selected_gold_candidate_count=len(selected_gold_candidates),
            question_route=route_trace.question_route,
            selected_selector_name=route_trace.selected_selector_name,
            route_reason=route_trace.route_reason,
            best_gold_sentence=best_gold_sentence,
            best_gold_window=best_gold_window,
            best_selected_gold_candidate_token_f1=round(max(selected_gold_candidate_f1_values), 4)
            if selected_gold_candidate_f1_values
            else None,
            bucket=bucket,
        )

    def _select_answer_candidates(
        self,
        question: PrimeQAQuestion,
        retrieval_results: list[RetrievalResult],
    ) -> list[SentenceEvidenceCandidate]:
        candidates = self._answer_generator.rank_sentence_candidates(question, retrieval_results)
        return [
            candidate
            for candidate in candidates
            if candidate.score >= self._answer_generator.min_sentence_score
        ][: self._answer_generator.max_sentences]

    def _best_gold_span(
        self,
        question: PrimeQAQuestion,
        gold_document: PrimeQADocument | None,
        max_window_sentences: int,
    ) -> GoldEvidenceSpan | None:
        if gold_document is None:
            return None

        sentences = [
            " ".join(sentence.split())
            for sentence in split_sentences(gold_document.text)
            if len(" ".join(sentence.split())) >= self._min_gold_sentence_chars
        ]
        if not sentences:
            return None

        best_span = None
        max_size = min(max_window_sentences, len(sentences))
        for window_size in range(1, max_size + 1):
            for start_index in range(0, len(sentences) - window_size + 1):
                text = " ".join(sentences[start_index : start_index + window_size])
                score = token_f1(text, question.answer)
                if best_span is None or score > best_span.token_f1:
                    best_span = GoldEvidenceSpan(
                        text=text,
                        token_f1=round(score, 4),
                        sentence_start=start_index,
                        sentence_count=window_size,
                    )
        return best_span


def _categorize_case(
    gold_document: PrimeQADocument | None,
    gold_retrieval_rank: int | None,
    selected_gold_candidate_count: int,
    selected_answer_token_f1: float,
    best_gold_sentence: GoldEvidenceSpan | None,
    best_gold_window: GoldEvidenceSpan | None,
    f1_gap_margin: float,
    low_f1_threshold: float,
) -> str:
    if gold_document is None:
        return "gold_document_missing"
    if gold_retrieval_rank is None:
        return "gold_not_in_context"
    if selected_gold_candidate_count == 0:
        return "gold_in_context_not_selected"
    if (
        best_gold_window is not None
        and best_gold_window.token_f1 >= selected_answer_token_f1 + f1_gap_margin
    ):
        return "gold_window_beats_selected_answer"
    if (
        best_gold_sentence is not None
        and best_gold_sentence.token_f1 >= selected_answer_token_f1 + f1_gap_margin
    ):
        return "gold_sentence_beats_selected_answer"
    if selected_answer_token_f1 < low_f1_threshold:
        return "selected_answer_low_overlap"
    return "selected_answer_reasonable_overlap"


def _find_gold_retrieval_rank(
    gold_doc_id: str | None,
    retrieval_results: list[RetrievalResult],
) -> int | None:
    if gold_doc_id is None:
        return None
    for retrieval_result in retrieval_results:
        if retrieval_result.document.id == gold_doc_id:
            return retrieval_result.rank
    return None


def _build_summary(cases: list[AnswerGapCase]) -> AnswerGapSummary:
    bucket_counts = Counter(case.bucket for case in cases)
    selected_answer_f1_values = [case.selected_answer_token_f1 for case in cases]
    best_gold_sentence_f1_values = [
        case.best_gold_sentence.token_f1
        for case in cases
        if case.best_gold_sentence is not None
    ]
    best_gold_window_f1_values = [
        case.best_gold_window.token_f1
        for case in cases
        if case.best_gold_window is not None
    ]
    question_route_counts = Counter(case.question_route for case in cases)
    selected_selector_counts = Counter(case.selected_selector_name for case in cases)

    return AnswerGapSummary(
        total_answerable_questions=len(cases),
        gold_document_available=sum(
            1 for case in cases if case.bucket != "gold_document_missing"
        ),
        gold_in_context=sum(1 for case in cases if case.gold_retrieval_rank is not None),
        selected_gold_citation=sum(
            1 for case in cases if case.selected_gold_candidate_count > 0
        ),
        bucket_counts={
            bucket_name: bucket_counts.get(bucket_name, 0)
            for bucket_name in ANSWER_GAP_BUCKET_DEFINITIONS
        },
        average_selected_answer_token_f1=_rounded_mean(selected_answer_f1_values),
        average_best_gold_sentence_token_f1=_rounded_mean(best_gold_sentence_f1_values),
        average_best_gold_window_token_f1=_rounded_mean(best_gold_window_f1_values),
        selected_answer_low_overlap=bucket_counts.get("selected_answer_low_overlap", 0),
        gold_sentence_beats_selected_answer=bucket_counts.get(
            "gold_sentence_beats_selected_answer",
            0,
        ),
        gold_window_beats_selected_answer=bucket_counts.get(
            "gold_window_beats_selected_answer",
            0,
        ),
        question_route_counts=dict(question_route_counts),
        selected_selector_counts=dict(selected_selector_counts),
    )


def _select_cases(cases: list[AnswerGapCase], sample_limit: int) -> list[AnswerGapCase]:
    if sample_limit == 0:
        return []
    priority = {
        "gold_window_beats_selected_answer": 0,
        "gold_sentence_beats_selected_answer": 1,
        "gold_in_context_not_selected": 2,
        "selected_answer_low_overlap": 3,
        "gold_not_in_context": 4,
        "gold_document_missing": 5,
        "selected_answer_reasonable_overlap": 6,
    }
    return sorted(
        cases,
        key=lambda case: (
            priority.get(case.bucket, 99),
            case.selected_answer_token_f1,
            case.question.id,
        ),
    )[:sample_limit]


def _rounded_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(mean(values), 4)


def span_to_dict(span: GoldEvidenceSpan | None) -> dict | None:
    """Convert a gold span to a JSON-safe dictionary."""

    if span is None:
        return None
    return {
        "text": _truncate(span.text),
        "token_f1": span.token_f1,
        "sentence_start": span.sentence_start,
        "sentence_count": span.sentence_count,
    }


def candidate_to_dict(
    candidate: SentenceEvidenceCandidate,
    gold_answer: str,
) -> dict:
    """Convert a selected candidate to a JSON-safe dictionary."""

    retrieval_result = candidate.retrieval_result
    return {
        "document_id": retrieval_result.document.id,
        "title": retrieval_result.document.title,
        "retrieval_rank": retrieval_result.rank,
        "candidate_score": round(candidate.score, 4),
        "candidate_token_f1": round(token_f1(candidate.sentence, gold_answer), 4),
        "sentence": _truncate(candidate.sentence),
    }


def case_to_dict(case: AnswerGapCase) -> dict:
    """Convert an answer-gap case to a JSON-safe dictionary."""

    return {
        "question_id": case.question.id,
        "question_title": case.question.title,
        "question_text": _truncate(case.question.text),
        "gold_answer_doc_id": case.question.answer_doc_id,
        "gold_answer": _truncate(case.question.answer),
        "retrieval_top_k": case.retrieval_top_k,
        "gold_retrieval_rank": case.gold_retrieval_rank,
        "bucket": case.bucket,
        "selected_answer_token_f1": case.selected_answer_token_f1,
        "selected_gold_candidate_count": case.selected_gold_candidate_count,
        "question_route": case.question_route,
        "selected_selector_name": case.selected_selector_name,
        "route_reason": case.route_reason,
        "best_selected_gold_candidate_token_f1": case.best_selected_gold_candidate_token_f1,
        "selected_doc_ids": [
            candidate.retrieval_result.document.id for candidate in case.selected_candidates
        ],
        "selected_candidates": [
            candidate_to_dict(candidate, case.question.answer)
            for candidate in case.selected_candidates
        ],
        "best_gold_sentence": span_to_dict(case.best_gold_sentence),
        "best_gold_window": span_to_dict(case.best_gold_window),
    }


def _truncate(text: str, max_chars: int = 700) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."
