from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ts_rag_agent.application.answer_verification import AnswerVerifier
from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator, evaluate_answers
from ts_rag_agent.application.retrieval_evaluation import Retriever
from ts_rag_agent.domain.answer import (
    AnswerEvaluationMetrics,
    AnswerVerificationResult,
    GeneratedAnswer,
)
from ts_rag_agent.domain.dataset import PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


@dataclass(frozen=True)
class VerifiedRAGQuestionResult:
    """单个问题在生成、验证、评估链路里的完整结果。"""

    question: PrimeQAQuestion
    retrieval_results: list[RetrievalResult]
    original_answer: GeneratedAnswer
    verification_result: AnswerVerificationResult

    @property
    def verified_answer(self) -> GeneratedAnswer:
        """返回经过验证器处理后的最终答案。"""

        return self.verification_result.verified_answer


@dataclass(frozen=True)
class VerifiedRAGEvaluationResult:
    """答案验证实验的一次完整评估结果。"""

    question_results: list[VerifiedRAGQuestionResult]
    original_metrics: AnswerEvaluationMetrics
    verified_metrics: AnswerEvaluationMetrics
    answerable_gold_doc_in_context: int

    @property
    def reason_counts(self) -> dict[str, int]:
        """统计验证器给出的拒答或通过原因。"""

        counter: Counter[str] = Counter()
        for question_result in self.question_results:
            counter.update(question_result.verification_result.reasons)
        return dict(sorted(counter.items()))

    @property
    def newly_refused_count(self) -> int:
        """统计原始答案未拒答、但验证后被拒答的样本数。"""

        return sum(
            1
            for question_result in self.question_results
            if (
                not question_result.original_answer.refused
                and question_result.verified_answer.refused
            )
        )


class VerifiedRAGEvaluator:
    """运行检索、抽取式回答、答案验证和指标统计。"""

    def __init__(
        self,
        retriever: Retriever,
        answer_generator: ExtractiveAnswerGenerator,
        answer_verifier: AnswerVerifier,
        retrieval_top_k: int = 5,
    ) -> None:
        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be positive")

        self._retriever = retriever
        self._answer_generator = answer_generator
        self._answer_verifier = answer_verifier
        self._retrieval_top_k = retrieval_top_k

    def evaluate(self, questions: Sequence[PrimeQAQuestion]) -> VerifiedRAGEvaluationResult:
        """在一批问题上评估验证前后 RAG 答案的变化。"""

        question_results = []
        original_answers = []
        verified_answers = []
        answerable_gold_doc_in_context = 0

        for question in questions:
            retrieval_results = self._retriever.search(
                question.full_question,
                top_k=self._retrieval_top_k,
            )
            if question.answerable:
                retrieved_doc_ids = {result.document.id for result in retrieval_results}
                if question.answer_doc_id in retrieved_doc_ids:
                    answerable_gold_doc_in_context += 1

            original_answer = self._answer_generator.generate(question, retrieval_results)
            verification_result = self._answer_verifier.verify(original_answer, retrieval_results)

            question_results.append(
                VerifiedRAGQuestionResult(
                    question=question,
                    retrieval_results=retrieval_results,
                    original_answer=original_answer,
                    verification_result=verification_result,
                )
            )
            original_answers.append(original_answer)
            verified_answers.append(verification_result.verified_answer)

        return VerifiedRAGEvaluationResult(
            question_results=question_results,
            original_metrics=evaluate_answers(questions, original_answers),
            verified_metrics=evaluate_answers(questions, verified_answers),
            answerable_gold_doc_in_context=answerable_gold_doc_in_context,
        )
