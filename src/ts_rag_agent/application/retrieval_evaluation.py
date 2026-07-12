from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from ts_rag_agent.domain.dataset import PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalMetrics, RetrievalResult


class Retriever(Protocol):
    """可被统一评估的检索器接口。"""

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """返回与查询最相关的 top-k 文档。"""


def evaluate_retrieval(
    questions: Iterable[PrimeQAQuestion],
    retriever: Retriever,
    top_k_values: tuple[int, ...] = (1, 5, 10),
) -> RetrievalMetrics:
    """在有答案文档标注的问题上计算 hit@k 和 MRR。"""

    if not top_k_values:
        raise ValueError("top_k_values must not be empty")
    if any(top_k <= 0 for top_k in top_k_values):
        raise ValueError("top_k_values must be positive")

    question_list = list(questions)
    evaluated = [
        question
        for question in question_list
        if question.answerable and question.answer_doc_id is not None
    ]
    max_k = max(top_k_values)
    hit_counts = {top_k: 0 for top_k in top_k_values}
    reciprocal_rank_sum = 0.0

    for question in evaluated:
        results = retriever.search(question.full_question, top_k=max_k)
        result_doc_ids = [result.document.id for result in results]

        for top_k in top_k_values:
            if question.answer_doc_id in result_doc_ids[:top_k]:
                hit_counts[top_k] += 1

        if question.answer_doc_id in result_doc_ids:
            rank = result_doc_ids.index(question.answer_doc_id) + 1
            reciprocal_rank_sum += 1 / rank

    evaluated_count = len(evaluated)
    return RetrievalMetrics(
        total_questions=len(question_list),
        evaluated_questions=evaluated_count,
        hit_at_k={
            top_k: round(count / evaluated_count, 4) if evaluated_count else 0.0
            for top_k, count in hit_counts.items()
        },
        mrr=round(reciprocal_rank_sum / evaluated_count, 4) if evaluated_count else 0.0,
    )
