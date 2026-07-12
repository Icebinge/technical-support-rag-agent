from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ts_rag_agent.application.answer_verification import AnswerVerifier
from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator, evaluate_answers
from ts_rag_agent.application.retrieval_evaluation import Retriever
from ts_rag_agent.application.verified_rag_evaluation import (
    VerifiedRAGEvaluationResult,
    VerifiedRAGQuestionResult,
)
from ts_rag_agent.application.verified_rag_quality_analysis import (
    analyze_verified_rag_quality,
)
from ts_rag_agent.domain.answer import AnswerEvaluationMetrics, GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


@dataclass(frozen=True)
class ThresholdSweepConfig:
    """一次验证阈值扫描的参数组合。"""

    retrieval_top_k: int
    min_evidence_score: float
    max_citation_rank: int


@dataclass(frozen=True)
class ThresholdSweepSummary:
    """一次参数组合对应的指标摘要。"""

    config: ThresholdSweepConfig
    original_metrics: AnswerEvaluationMetrics
    verified_metrics: AnswerEvaluationMetrics
    answerable_gold_doc_in_context: int
    reason_counts: dict[str, int]
    newly_refused_count: int
    quality_analysis: dict


@dataclass(frozen=True)
class ThresholdSweepResult:
    """完整阈值扫描结果。"""

    summaries: list[ThresholdSweepSummary]
    pareto_candidate_indices: list[int]


@dataclass(frozen=True)
class _BaseQuestionResult:
    """固定 retrieval_top_k 后，验证前可复用的单题结果。"""

    question: PrimeQAQuestion
    retrieval_results: list[RetrievalResult]
    original_answer: GeneratedAnswer


@dataclass(frozen=True)
class _BaseRAGResult:
    """固定 retrieval_top_k 后，验证前可复用的 RAG 结果。"""

    retrieval_top_k: int
    question_results: list[_BaseQuestionResult]
    original_metrics: AnswerEvaluationMetrics
    answerable_gold_doc_in_context: int


class VerifiedRAGThresholdSweeper:
    """对 RAG 验证层参数做网格扫描，并复用检索和原始答案。"""

    def __init__(
        self,
        retriever: Retriever,
        answer_generator: ExtractiveAnswerGenerator,
        min_citations: int = 1,
    ) -> None:
        if min_citations <= 0:
            raise ValueError("min_citations must be positive")

        self._retriever = retriever
        self._answer_generator = answer_generator
        self._min_citations = min_citations
        self._base_cache: dict[int, _BaseRAGResult] = {}

    def sweep(
        self,
        questions: Sequence[PrimeQAQuestion],
        retrieval_top_k_values: Sequence[int],
        min_evidence_scores: Sequence[float],
        max_citation_ranks: Sequence[int],
        sample_limit_per_bucket: int = 0,
    ) -> ThresholdSweepResult:
        """扫描多个验证参数组合，并返回带 Pareto 候选标记的结果。"""

        _validate_sweep_values(
            retrieval_top_k_values=retrieval_top_k_values,
            min_evidence_scores=min_evidence_scores,
            max_citation_ranks=max_citation_ranks,
            sample_limit_per_bucket=sample_limit_per_bucket,
        )

        summaries = []
        for retrieval_top_k in retrieval_top_k_values:
            base_result = self._get_or_build_base_result(questions, retrieval_top_k)
            for min_evidence_score in min_evidence_scores:
                for max_citation_rank in max_citation_ranks:
                    config = ThresholdSweepConfig(
                        retrieval_top_k=retrieval_top_k,
                        min_evidence_score=min_evidence_score,
                        max_citation_rank=max_citation_rank,
                    )
                    summaries.append(
                        self._evaluate_config(
                            base_result=base_result,
                            config=config,
                            sample_limit_per_bucket=sample_limit_per_bucket,
                        )
                    )

        return ThresholdSweepResult(
            summaries=summaries,
            pareto_candidate_indices=_find_pareto_candidate_indices(summaries),
        )

    def _get_or_build_base_result(
        self,
        questions: Sequence[PrimeQAQuestion],
        retrieval_top_k: int,
    ) -> _BaseRAGResult:
        if retrieval_top_k not in self._base_cache:
            self._base_cache[retrieval_top_k] = self._build_base_result(
                questions=questions,
                retrieval_top_k=retrieval_top_k,
            )
        return self._base_cache[retrieval_top_k]

    def _build_base_result(
        self,
        questions: Sequence[PrimeQAQuestion],
        retrieval_top_k: int,
    ) -> _BaseRAGResult:
        question_results = []
        original_answers = []
        answerable_gold_doc_in_context = 0

        for question in questions:
            retrieval_results = self._retriever.search(
                question.full_question,
                top_k=retrieval_top_k,
            )
            if question.answerable:
                retrieved_doc_ids = {result.document.id for result in retrieval_results}
                if question.answer_doc_id in retrieved_doc_ids:
                    answerable_gold_doc_in_context += 1

            original_answer = self._answer_generator.generate(question, retrieval_results)
            question_results.append(
                _BaseQuestionResult(
                    question=question,
                    retrieval_results=retrieval_results,
                    original_answer=original_answer,
                )
            )
            original_answers.append(original_answer)

        return _BaseRAGResult(
            retrieval_top_k=retrieval_top_k,
            question_results=question_results,
            original_metrics=evaluate_answers(questions, original_answers),
            answerable_gold_doc_in_context=answerable_gold_doc_in_context,
        )

    def _evaluate_config(
        self,
        base_result: _BaseRAGResult,
        config: ThresholdSweepConfig,
        sample_limit_per_bucket: int,
    ) -> ThresholdSweepSummary:
        verifier = AnswerVerifier(
            min_citations=self._min_citations,
            min_evidence_score=config.min_evidence_score,
            max_citation_rank=config.max_citation_rank,
        )
        question_results = []
        verified_answers = []

        for base_question_result in base_result.question_results:
            verification_result = verifier.verify(
                base_question_result.original_answer,
                base_question_result.retrieval_results,
            )
            question_results.append(
                VerifiedRAGQuestionResult(
                    question=base_question_result.question,
                    retrieval_results=base_question_result.retrieval_results,
                    original_answer=base_question_result.original_answer,
                    verification_result=verification_result,
                )
            )
            verified_answers.append(verification_result.verified_answer)

        evaluation = VerifiedRAGEvaluationResult(
            question_results=question_results,
            original_metrics=base_result.original_metrics,
            verified_metrics=evaluate_answers(
                [result.question for result in base_result.question_results],
                verified_answers,
            ),
            answerable_gold_doc_in_context=base_result.answerable_gold_doc_in_context,
        )
        quality_analysis = analyze_verified_rag_quality(
            evaluation,
            min_evidence_score=config.min_evidence_score,
            sample_limit_per_bucket=sample_limit_per_bucket,
        )
        return ThresholdSweepSummary(
            config=config,
            original_metrics=evaluation.original_metrics,
            verified_metrics=evaluation.verified_metrics,
            answerable_gold_doc_in_context=evaluation.answerable_gold_doc_in_context,
            reason_counts=evaluation.reason_counts,
            newly_refused_count=evaluation.newly_refused_count,
            quality_analysis=quality_analysis,
        )


def _validate_sweep_values(
    retrieval_top_k_values: Sequence[int],
    min_evidence_scores: Sequence[float],
    max_citation_ranks: Sequence[int],
    sample_limit_per_bucket: int,
) -> None:
    if not retrieval_top_k_values:
        raise ValueError("retrieval_top_k_values must not be empty")
    if any(value <= 0 for value in retrieval_top_k_values):
        raise ValueError("retrieval_top_k_values must be positive")
    if not min_evidence_scores:
        raise ValueError("min_evidence_scores must not be empty")
    if any(value < 0 for value in min_evidence_scores):
        raise ValueError("min_evidence_scores must be non-negative")
    if not max_citation_ranks:
        raise ValueError("max_citation_ranks must not be empty")
    if any(value <= 0 for value in max_citation_ranks):
        raise ValueError("max_citation_ranks must be positive")
    if sample_limit_per_bucket < 0:
        raise ValueError("sample_limit_per_bucket must be non-negative")


def _find_pareto_candidate_indices(summaries: list[ThresholdSweepSummary]) -> list[int]:
    return [
        index
        for index, summary in enumerate(summaries)
        if not any(_dominates(other, summary) for other in summaries)
    ]


def _dominates(candidate: ThresholdSweepSummary, target: ThresholdSweepSummary) -> bool:
    candidate_metrics = candidate.verified_metrics
    target_metrics = target.verified_metrics
    checks = [
        candidate_metrics.answerable_refusal_rate <= target_metrics.answerable_refusal_rate,
        candidate_metrics.unanswerable_refusal_rate >= target_metrics.unanswerable_refusal_rate,
        candidate_metrics.gold_doc_citation_rate >= target_metrics.gold_doc_citation_rate,
        candidate_metrics.average_token_f1 >= target_metrics.average_token_f1,
    ]
    strict_checks = [
        candidate_metrics.answerable_refusal_rate < target_metrics.answerable_refusal_rate,
        candidate_metrics.unanswerable_refusal_rate > target_metrics.unanswerable_refusal_rate,
        candidate_metrics.gold_doc_citation_rate > target_metrics.gold_doc_citation_rate,
        candidate_metrics.average_token_f1 > target_metrics.average_token_f1,
    ]
    return all(checks) and any(strict_checks)
