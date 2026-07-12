from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ts_rag_agent.domain.answer import AnswerCitation, AnswerEvaluationMetrics, GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


@dataclass(frozen=True)
class SentenceEvidenceCandidate:
    """抽取式回答器识别出的候选证据句。"""

    sentence: str
    retrieval_result: RetrievalResult
    score: float
    overlap_terms: tuple[str, ...]


class ExtractiveAnswerGenerator:
    """从检索文档中抽取句子并生成带引用答案。"""

    def __init__(
        self,
        max_sentences: int = 3,
        min_sentence_score: float = 2.0,
        min_sentence_chars: int = 24,
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
        self._sentence_cache: dict[str, list[str]] = {}

    @property
    def max_sentences(self) -> int:
        """每个答案最多抽取的证据句数量。"""

        return self._max_sentences

    @property
    def min_sentence_score(self) -> float:
        """生成答案所需的最低证据句分数。"""

        return self._min_sentence_score

    def generate(
        self,
        question: PrimeQAQuestion,
        retrieval_results: Sequence[RetrievalResult],
    ) -> GeneratedAnswer:
        """基于检索结果生成一个抽取式答案。"""

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
        """返回按分数排序的候选证据句，用于生成答案和错误分析。"""

        query_terms = set(_tokenize(question.full_question))
        candidates = []
        seen_sentences = set()

        for retrieval_result in retrieval_results:
            for sentence in self._get_document_sentences(
                retrieval_result.document.id,
                retrieval_result.document.text,
            ):
                normalized_sentence = " ".join(sentence.split())
                if len(normalized_sentence) < self._min_sentence_chars:
                    continue
                if normalized_sentence in seen_sentences:
                    continue

                seen_sentences.add(normalized_sentence)
                sentence_terms = set(_tokenize(normalized_sentence))
                overlap_terms = tuple(sorted(query_terms & sentence_terms))
                if not overlap_terms:
                    continue

                score = len(overlap_terms) / math.log2(retrieval_result.rank + 1)
                candidates.append(
                    SentenceEvidenceCandidate(
                        sentence=normalized_sentence,
                        retrieval_result=retrieval_result,
                        score=score,
                        overlap_terms=overlap_terms,
                    )
                )

        return sorted(
            candidates,
            key=lambda candidate: (
                -candidate.score,
                candidate.retrieval_result.rank,
                candidate.retrieval_result.document.id,
            ),
        )

    def _get_document_sentences(self, document_id: str, text: str) -> list[str]:
        """缓存文档切句结果，避免阈值扫描时重复处理同一篇文档。"""

        if document_id not in self._sentence_cache:
            self._sentence_cache[document_id] = _split_sentences(text)
        return self._sentence_cache[document_id]


def evaluate_answers(
    questions: Sequence[PrimeQAQuestion],
    answers: Sequence[GeneratedAnswer],
) -> AnswerEvaluationMetrics:
    """评估抽取式 RAG 答案的引用和拒答表现。"""

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
        token_f1_values.append(_token_f1(answer.answer, question.answer))

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


def _split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        if sentence.strip()
    ]


def _token_f1(prediction: str, gold: str) -> float:
    prediction_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold)
    if not prediction_tokens or not gold_tokens:
        return 0.0

    prediction_counts = _count_tokens(prediction_tokens)
    gold_counts = _count_tokens(gold_tokens)
    overlap = sum(
        min(prediction_counts[token], gold_counts[token])
        for token in prediction_counts.keys() & gold_counts.keys()
    )
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _count_tokens(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_+#]+(?:[.+#-][a-z0-9_+#]+)*", text.lower())
