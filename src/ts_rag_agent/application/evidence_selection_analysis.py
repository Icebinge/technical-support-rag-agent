from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, median

from ts_rag_agent.application.rag_answering import (
    ExtractiveAnswerGenerator,
    SentenceEvidenceCandidate,
)
from ts_rag_agent.application.retrieval_evaluation import Retriever
from ts_rag_agent.domain.dataset import PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult

EVIDENCE_SELECTION_BUCKET_DEFINITIONS = {
    "gold_not_in_context": "gold 文档没有进入当前 top-k 检索上下文。",
    "gold_cited": "gold 文档进入上下文，并被抽取式回答器引用。",
    "gold_in_context_no_sentence_candidate": (
        "gold 文档进入上下文，但没有任何句子通过候选生成，通常是无关键词重合或句子过短。"
    ),
    "gold_candidate_below_min_sentence_score": (
        "gold 文档有候选句，但最高分低于生成器最低证据分。"
    ),
    "gold_candidate_loses_to_wrong_sentences": (
        "gold 文档候选句达到最低分，但被更靠前的错误文档候选句挤出最终答案。"
    ),
    "unknown_gold_in_context_not_cited": "gold 文档在上下文中但未被引用，且未被当前规则解释。",
}


@dataclass(frozen=True)
class EvidenceSelectionCase:
    """单个问题的证据选择分析结果。"""

    question: PrimeQAQuestion
    retrieval_top_k: int
    gold_retrieval_rank: int | None
    selected_candidates: list[SentenceEvidenceCandidate]
    best_gold_candidate: SentenceEvidenceCandidate | None
    best_non_gold_candidate: SentenceEvidenceCandidate | None
    best_gold_candidate_rank: int | None
    bucket: str
    gold_newly_available_after_base_k: bool


@dataclass(frozen=True)
class EvidenceSelectionTopKSummary:
    """某个 retrieval_top_k 下的证据选择统计。"""

    retrieval_top_k: int
    total_answerable_questions: int
    bucket_counts: dict[str, int]
    gold_in_context: int
    gold_cited: int
    gold_in_context_not_cited: int
    gold_newly_available_after_base_k: int
    gold_newly_available_but_not_cited: int
    gold_candidate_score_stats: dict[str, float | None]
    gold_candidate_rank_stats: dict[str, float | None]


@dataclass(frozen=True)
class EvidenceSelectionAnalysisResult:
    """完整证据选择分析结果。"""

    summaries: list[EvidenceSelectionTopKSummary]
    cases_by_top_k: dict[int, list[EvidenceSelectionCase]]


class EvidenceSelectionAnalyzer:
    """分析检索上下文里的 gold 文档为什么没有被最终引用。"""

    def __init__(
        self,
        retriever: Retriever,
        answer_generator: ExtractiveAnswerGenerator,
        base_top_k: int = 5,
    ) -> None:
        if base_top_k <= 0:
            raise ValueError("base_top_k must be positive")

        self._retriever = retriever
        self._answer_generator = answer_generator
        self._base_top_k = base_top_k

    def analyze(
        self,
        questions: Sequence[PrimeQAQuestion],
        retrieval_top_k_values: Sequence[int],
        sample_limit_per_top_k: int = 20,
    ) -> EvidenceSelectionAnalysisResult:
        """分析多个 top-k 下 gold 文档进入上下文后是否被证据选择器用到。"""

        _validate_options(
            retrieval_top_k_values=retrieval_top_k_values,
            sample_limit_per_top_k=sample_limit_per_top_k,
        )

        sorted_top_k_values = sorted(set(retrieval_top_k_values))
        max_top_k = max(sorted_top_k_values)
        answerable_questions = [question for question in questions if question.answerable]
        cases_by_top_k: dict[int, list[EvidenceSelectionCase]] = defaultdict(list)
        all_cases_by_top_k: dict[int, list[EvidenceSelectionCase]] = defaultdict(list)

        for question in answerable_questions:
            retrieval_results = self._retriever.search(question.full_question, top_k=max_top_k)
            for retrieval_top_k in sorted_top_k_values:
                top_k_results = retrieval_results[:retrieval_top_k]
                case = self._analyze_question(
                    question=question,
                    retrieval_top_k=retrieval_top_k,
                    retrieval_results=top_k_results,
                )
                all_cases_by_top_k[retrieval_top_k].append(case)
                should_keep_sample = (
                    _should_keep_case(case)
                    and len(cases_by_top_k[retrieval_top_k]) < sample_limit_per_top_k
                )
                if should_keep_sample:
                    cases_by_top_k[retrieval_top_k].append(case)

        summaries = [
            _build_summary(
                retrieval_top_k=retrieval_top_k,
                cases=all_cases_by_top_k[retrieval_top_k],
            )
            for retrieval_top_k in sorted_top_k_values
        ]
        return EvidenceSelectionAnalysisResult(
            summaries=summaries,
            cases_by_top_k=dict(cases_by_top_k),
        )

    def _analyze_question(
        self,
        question: PrimeQAQuestion,
        retrieval_top_k: int,
        retrieval_results: list[RetrievalResult],
    ) -> EvidenceSelectionCase:
        candidates = self._answer_generator.rank_sentence_candidates(question, retrieval_results)
        selected_candidates = [
            candidate
            for candidate in candidates
            if candidate.score >= self._answer_generator.min_sentence_score
        ][: self._answer_generator.max_sentences]
        selected_doc_ids = {
            candidate.retrieval_result.document.id for candidate in selected_candidates
        }

        gold_doc_id = question.answer_doc_id
        gold_retrieval_rank = _find_gold_retrieval_rank(gold_doc_id, retrieval_results)
        best_gold_candidate = _find_first_candidate_for_document(candidates, gold_doc_id)
        best_non_gold_candidate = _find_first_non_gold_candidate(candidates, gold_doc_id)
        best_gold_candidate_rank = _find_candidate_rank(candidates, best_gold_candidate)
        bucket = _categorize_case(
            gold_doc_id=gold_doc_id,
            gold_retrieval_rank=gold_retrieval_rank,
            selected_doc_ids=selected_doc_ids,
            best_gold_candidate=best_gold_candidate,
            best_gold_candidate_rank=best_gold_candidate_rank,
            min_sentence_score=self._answer_generator.min_sentence_score,
            max_sentences=self._answer_generator.max_sentences,
        )

        return EvidenceSelectionCase(
            question=question,
            retrieval_top_k=retrieval_top_k,
            gold_retrieval_rank=gold_retrieval_rank,
            selected_candidates=selected_candidates,
            best_gold_candidate=best_gold_candidate,
            best_non_gold_candidate=best_non_gold_candidate,
            best_gold_candidate_rank=best_gold_candidate_rank,
            bucket=bucket,
            gold_newly_available_after_base_k=(
                gold_retrieval_rank is not None
                and self._base_top_k < gold_retrieval_rank <= retrieval_top_k
            ),
        )


def _validate_options(
    retrieval_top_k_values: Sequence[int],
    sample_limit_per_top_k: int,
) -> None:
    if not retrieval_top_k_values:
        raise ValueError("retrieval_top_k_values must not be empty")
    if any(value <= 0 for value in retrieval_top_k_values):
        raise ValueError("retrieval_top_k_values must be positive")
    if sample_limit_per_top_k < 0:
        raise ValueError("sample_limit_per_top_k must be non-negative")


def _categorize_case(
    gold_doc_id: str | None,
    gold_retrieval_rank: int | None,
    selected_doc_ids: set[str],
    best_gold_candidate: SentenceEvidenceCandidate | None,
    best_gold_candidate_rank: int | None,
    min_sentence_score: float,
    max_sentences: int,
) -> str:
    if gold_doc_id is None or gold_retrieval_rank is None:
        return "gold_not_in_context"
    if gold_doc_id in selected_doc_ids:
        return "gold_cited"
    if best_gold_candidate is None:
        return "gold_in_context_no_sentence_candidate"
    if best_gold_candidate.score < min_sentence_score:
        return "gold_candidate_below_min_sentence_score"
    if best_gold_candidate_rank is not None and best_gold_candidate_rank > max_sentences:
        return "gold_candidate_loses_to_wrong_sentences"
    return "unknown_gold_in_context_not_cited"


def _build_summary(
    retrieval_top_k: int,
    cases: list[EvidenceSelectionCase],
) -> EvidenceSelectionTopKSummary:
    bucket_counts = Counter(case.bucket for case in cases)
    gold_in_context_cases = [case for case in cases if case.gold_retrieval_rank is not None]
    gold_cited_cases = [case for case in cases if case.bucket == "gold_cited"]
    gold_newly_available_cases = [
        case for case in cases if case.gold_newly_available_after_base_k
    ]
    gold_newly_available_but_not_cited_cases = [
        case
        for case in gold_newly_available_cases
        if case.bucket != "gold_cited"
    ]

    return EvidenceSelectionTopKSummary(
        retrieval_top_k=retrieval_top_k,
        total_answerable_questions=len(cases),
        bucket_counts={
            bucket_name: bucket_counts.get(bucket_name, 0)
            for bucket_name in EVIDENCE_SELECTION_BUCKET_DEFINITIONS
        },
        gold_in_context=len(gold_in_context_cases),
        gold_cited=len(gold_cited_cases),
        gold_in_context_not_cited=len(gold_in_context_cases) - len(gold_cited_cases),
        gold_newly_available_after_base_k=len(gold_newly_available_cases),
        gold_newly_available_but_not_cited=len(gold_newly_available_but_not_cited_cases),
        gold_candidate_score_stats=_gold_candidate_score_stats(cases),
        gold_candidate_rank_stats=_gold_candidate_rank_stats(cases),
    )


def _should_keep_case(case: EvidenceSelectionCase) -> bool:
    return case.gold_retrieval_rank is not None and case.bucket != "gold_cited"


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


def _find_first_candidate_for_document(
    candidates: list[SentenceEvidenceCandidate],
    document_id: str | None,
) -> SentenceEvidenceCandidate | None:
    if document_id is None:
        return None
    return next(
        (
            candidate
            for candidate in candidates
            if candidate.retrieval_result.document.id == document_id
        ),
        None,
    )


def _find_first_non_gold_candidate(
    candidates: list[SentenceEvidenceCandidate],
    gold_doc_id: str | None,
) -> SentenceEvidenceCandidate | None:
    return next(
        (
            candidate
            for candidate in candidates
            if candidate.retrieval_result.document.id != gold_doc_id
        ),
        None,
    )


def _find_candidate_rank(
    candidates: list[SentenceEvidenceCandidate],
    target_candidate: SentenceEvidenceCandidate | None,
) -> int | None:
    if target_candidate is None:
        return None
    for index, candidate in enumerate(candidates, start=1):
        if candidate == target_candidate:
            return index
    return None


def _gold_candidate_score_stats(
    cases: list[EvidenceSelectionCase],
) -> dict[str, float | None]:
    scores = [
        case.best_gold_candidate.score
        for case in cases
        if case.best_gold_candidate is not None
    ]
    return _numeric_stats(scores)


def _gold_candidate_rank_stats(
    cases: list[EvidenceSelectionCase],
) -> dict[str, float | None]:
    ranks = [
        case.best_gold_candidate_rank
        for case in cases
        if case.best_gold_candidate_rank is not None
    ]
    return _numeric_stats(ranks)


def _numeric_stats(values: list[float | int]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": round(float(min(values)), 4),
        "median": round(float(median(values)), 4),
        "mean": round(float(mean(values)), 4),
        "max": round(float(max(values)), 4),
    }


def candidate_to_dict(candidate: SentenceEvidenceCandidate | None) -> dict | None:
    """将候选证据句转换成可写入 JSON 的结构。"""

    if candidate is None:
        return None
    retrieval_result = candidate.retrieval_result
    return {
        "document_id": retrieval_result.document.id,
        "title": retrieval_result.document.title,
        "retrieval_rank": retrieval_result.rank,
        "retrieval_score": round(retrieval_result.score, 4),
        "candidate_score": round(candidate.score, 4),
        "overlap_terms": list(candidate.overlap_terms),
        "sentence": _truncate(candidate.sentence),
    }


def case_to_dict(case: EvidenceSelectionCase) -> dict:
    """将单条分析案例转换成可写入 JSON 的结构。"""

    return {
        "question_id": case.question.id,
        "question_title": case.question.title,
        "question_text": _truncate(case.question.text),
        "gold_answer_doc_id": case.question.answer_doc_id,
        "gold_retrieval_rank": case.gold_retrieval_rank,
        "bucket": case.bucket,
        "gold_newly_available_after_base_k": case.gold_newly_available_after_base_k,
        "selected_doc_ids": [
            candidate.retrieval_result.document.id for candidate in case.selected_candidates
        ],
        "selected_candidates": [
            candidate_to_dict(candidate) for candidate in case.selected_candidates
        ],
        "best_gold_candidate_rank": case.best_gold_candidate_rank,
        "best_gold_candidate": candidate_to_dict(case.best_gold_candidate),
        "best_non_gold_candidate": candidate_to_dict(case.best_non_gold_candidate),
        "gold_answer": _truncate(case.question.answer),
    }


def _truncate(text: str, max_chars: int = 700) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."
