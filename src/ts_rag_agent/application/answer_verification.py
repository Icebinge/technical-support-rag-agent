from __future__ import annotations

from collections.abc import Sequence

from ts_rag_agent.domain.answer import AnswerVerificationResult, GeneratedAnswer
from ts_rag_agent.domain.retrieval import RetrievalResult


class AnswerVerifier:
    """基于证据强度和引用合法性的回答验证器。"""

    def __init__(
        self,
        min_citations: int = 1,
        min_evidence_score: float = 8.0,
        max_citation_rank: int = 3,
    ) -> None:
        if min_citations <= 0:
            raise ValueError("min_citations must be positive")
        if min_evidence_score < 0:
            raise ValueError("min_evidence_score must be non-negative")
        if max_citation_rank <= 0:
            raise ValueError("max_citation_rank must be positive")

        self._min_citations = min_citations
        self._min_evidence_score = min_evidence_score
        self._max_citation_rank = max_citation_rank

    def verify(
        self,
        answer: GeneratedAnswer,
        retrieval_results: Sequence[RetrievalResult],
    ) -> AnswerVerificationResult:
        """验证答案是否有足够证据支持。"""

        if answer.refused:
            return AnswerVerificationResult(
                original_answer=answer,
                verified_answer=answer,
                citation_context_valid=True,
                reasons=["already_refused"],
            )

        context_doc_ids = {result.document.id for result in retrieval_results}
        citation_doc_ids = [citation.document_id for citation in answer.citations]
        citation_context_valid = all(doc_id in context_doc_ids for doc_id in citation_doc_ids)
        reasons = []

        if len(answer.citations) < self._min_citations:
            reasons.append("not_enough_citations")
        if not citation_context_valid:
            reasons.append("citation_not_in_retrieved_context")
        if answer.citations:
            max_evidence_score = max(citation.evidence_score for citation in answer.citations)
            best_citation_rank = min(citation.retrieval_rank for citation in answer.citations)
            if max_evidence_score < self._min_evidence_score:
                reasons.append("weak_evidence_score")
            if best_citation_rank > self._max_citation_rank:
                reasons.append("citation_rank_too_low")

        if reasons:
            verified_answer = GeneratedAnswer(
                question_id=answer.question_id,
                answer="I do not have enough verified evidence to answer this question.",
                citations=[],
                refused=True,
            )
        else:
            verified_answer = answer
            reasons.append("verified")

        return AnswerVerificationResult(
            original_answer=answer,
            verified_answer=verified_answer,
            citation_context_valid=citation_context_valid,
            reasons=reasons,
        )
