from ts_rag_agent.application.answer_verification import AnswerVerifier
from ts_rag_agent.domain.answer import AnswerCitation, GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.domain.retrieval import RetrievalResult


def test_answer_verifier_keeps_strong_contextual_citation():
    document = PrimeQADocument(id="doc-a", title="A", text="A")
    answer = GeneratedAnswer(
        question_id="q1",
        answer="Answer [doc-a]",
        citations=[
            AnswerCitation(
                document_id="doc-a",
                title="A",
                retrieval_rank=1,
                evidence_score=8.0,
            )
        ],
        refused=False,
    )
    verifier = AnswerVerifier(min_evidence_score=8.0, max_citation_rank=3)

    result = verifier.verify(answer, [RetrievalResult(document=document, score=10.0, rank=1)])

    assert result.verified_answer.refused is False
    assert result.reasons == ["verified"]


def test_answer_verifier_refuses_weak_evidence():
    document = PrimeQADocument(id="doc-a", title="A", text="A")
    answer = GeneratedAnswer(
        question_id="q1",
        answer="Answer [doc-a]",
        citations=[
            AnswerCitation(
                document_id="doc-a",
                title="A",
                retrieval_rank=1,
                evidence_score=3.0,
            )
        ],
        refused=False,
    )
    verifier = AnswerVerifier(min_evidence_score=8.0, max_citation_rank=3)

    result = verifier.verify(answer, [RetrievalResult(document=document, score=10.0, rank=1)])

    assert result.verified_answer.refused is True
    assert "weak_evidence_score" in result.reasons


def test_answer_verifier_rejects_citation_outside_context():
    retrieved_document = PrimeQADocument(id="doc-a", title="A", text="A")
    answer = GeneratedAnswer(
        question_id="q1",
        answer="Answer [doc-b]",
        citations=[
            AnswerCitation(
                document_id="doc-b",
                title="B",
                retrieval_rank=1,
                evidence_score=9.0,
            )
        ],
        refused=False,
    )
    verifier = AnswerVerifier(min_evidence_score=8.0, max_citation_rank=3)

    result = verifier.verify(
        answer,
        [RetrievalResult(document=retrieved_document, score=10.0, rank=1)],
    )

    assert result.citation_context_valid is False
    assert result.verified_answer.refused is True
    assert "citation_not_in_retrieved_context" in result.reasons
