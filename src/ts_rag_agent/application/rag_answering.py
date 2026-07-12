from __future__ import annotations

from collections.abc import Sequence

from ts_rag_agent.application.evidence_selection import (
    OverlapSentenceEvidenceSelector,
    SentenceEvidenceCandidate,
    SentenceEvidenceSelector,
)
from ts_rag_agent.application.text_metrics import token_f1
from ts_rag_agent.domain.answer import AnswerCitation, AnswerEvaluationMetrics, GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


class ExtractiveAnswerGenerator:
    """Builds citation-bearing extractive answers from retrieved documents."""

    def __init__(
        self,
        max_sentences: int = 3,
        min_sentence_score: float = 2.0,
        min_sentence_chars: int = 24,
        evidence_selector: SentenceEvidenceSelector | None = None,
    ) -> None:
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive")
        if min_sentence_score < 0:
            raise ValueError("min_sentence_score must be non-negative")
        if min_sentence_chars <= 0:
            raise ValueError("min_sentence_chars must be positive")

        self._max_sentences = max_sentences
        self._min_sentence_score = min_sentence_score
        self._min_sentence_chars = min_sentence_chars
        self._evidence_selector = evidence_selector or OverlapSentenceEvidenceSelector(
            min_sentence_chars=min_sentence_chars
        )

    @property
    def max_sentences(self) -> int:
        """Maximum number of evidence sentences used in one answer."""

        return self._max_sentences

    @property
    def min_sentence_score(self) -> float:
        """Minimum candidate score required before an answer is generated."""

        return self._min_sentence_score

    @property
    def evidence_selector_name(self) -> str:
        """Name of the sentence evidence selector used by this generator."""

        return self._evidence_selector.name

    def generate(
        self,
        question: PrimeQAQuestion,
        retrieval_results: Sequence[RetrievalResult],
    ) -> GeneratedAnswer:
        """Generate one extractive answer from retrieved documents."""

        sentence_candidates = self.rank_sentence_candidates(question, retrieval_results)
        selected = [
            candidate
            for candidate in sentence_candidates
            if candidate.score >= self._min_sentence_score
        ][: self._max_sentences]

        if not selected:
            return GeneratedAnswer(
                question_id=question.id,
                answer="I do not have enough retrieved evidence to answer this question.",
                citations=[],
                refused=True,
            )

        answer_parts = [
            f"{candidate.sentence} [{candidate.retrieval_result.document.id}]"
            for candidate in selected
        ]
        citations = [
            AnswerCitation(
                document_id=candidate.retrieval_result.document.id,
                title=candidate.retrieval_result.document.title,
                retrieval_rank=candidate.retrieval_result.rank,
                evidence_score=round(candidate.score, 4),
            )
            for candidate in selected
        ]
        return GeneratedAnswer(
            question_id=question.id,
            answer=" ".join(answer_parts),
            citations=citations,
            refused=False,
        )

    def rank_sentence_candidates(
        self,
        question: PrimeQAQuestion,
        retrieval_results: Sequence[RetrievalResult],
    ) -> list[SentenceEvidenceCandidate]:
        """Return ranked evidence candidates for analysis and answer generation."""

        return self._evidence_selector.rank_sentence_candidates(question, retrieval_results)


def evaluate_answers(
    questions: Sequence[PrimeQAQuestion],
    answers: Sequence[GeneratedAnswer],
) -> AnswerEvaluationMetrics:
    """Evaluate citation and refusal behavior for generated RAG answers."""

    answer_by_question_id = {answer.question_id: answer for answer in answers}
    answerable_questions = [question for question in questions if question.answerable]
    unanswerable_questions = [question for question in questions if not question.answerable]
    generated_answerable = []
    refused_answerable = 0
    refused_unanswerable = 0
    gold_doc_cited = 0
    token_f1_values = []

    for question in answerable_questions:
        answer = answer_by_question_id[question.id]
        if answer.refused:
            refused_answerable += 1
            continue

        generated_answerable.append(question)
        cited_doc_ids = {citation.document_id for citation in answer.citations}
        if question.answer_doc_id in cited_doc_ids:
            gold_doc_cited += 1
        token_f1_values.append(token_f1(answer.answer, question.answer))

    for question in unanswerable_questions:
        answer = answer_by_question_id[question.id]
        if answer.refused:
            refused_unanswerable += 1

    generated_count = len(generated_answerable)
    return AnswerEvaluationMetrics(
        total_questions=len(questions),
        answerable_questions=len(answerable_questions),
        unanswerable_questions=len(unanswerable_questions),
        generated_answerable_questions=generated_count,
        refused_answerable_questions=refused_answerable,
        refused_unanswerable_questions=refused_unanswerable,
        gold_doc_citation_rate=round(gold_doc_cited / generated_count, 4)
        if generated_count
        else 0.0,
        answerable_refusal_rate=round(refused_answerable / len(answerable_questions), 4)
        if answerable_questions
        else 0.0,
        unanswerable_refusal_rate=round(refused_unanswerable / len(unanswerable_questions), 4)
        if unanswerable_questions
        else 0.0,
        average_token_f1=round(sum(token_f1_values) / len(token_f1_values), 4)
        if token_f1_values
        else 0.0,
    )
