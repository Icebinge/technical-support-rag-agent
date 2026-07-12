from ts_rag_agent.application.verified_rag_evaluation import (
    VerifiedRAGEvaluationResult,
    VerifiedRAGQuestionResult,
)
from ts_rag_agent.application.verified_rag_quality_analysis import (
    analyze_verified_rag_quality,
)
from ts_rag_agent.domain.answer import (
    AnswerCitation,
    AnswerEvaluationMetrics,
    AnswerVerificationResult,
    GeneratedAnswer,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


def test_quality_analysis_buckets_newly_refused_cases():
    evaluation = VerifiedRAGEvaluationResult(
        question_results=[
            _newly_refused_result(
                question=_question("q1", answerable=False),
                retrieved_doc_ids=["doc-a"],
                cited_doc_id="doc-a",
            ),
            _newly_refused_result(
                question=_question("q2", answerable=True, answer_doc_id="gold-doc"),
                retrieved_doc_ids=["doc-a"],
                cited_doc_id="doc-a",
            ),
            _newly_refused_result(
                question=_question("q3", answerable=True, answer_doc_id="gold-doc"),
                retrieved_doc_ids=["gold-doc"],
                cited_doc_id="gold-doc",
            ),
            _newly_refused_result(
                question=_question("q4", answerable=True, answer_doc_id="gold-doc"),
                retrieved_doc_ids=["gold-doc", "doc-a"],
                cited_doc_id="doc-a",
            ),
        ],
        original_metrics=_empty_metrics(),
        verified_metrics=_empty_metrics(),
        answerable_gold_doc_in_context=2,
    )

    analysis = analyze_verified_rag_quality(
        evaluation,
        min_evidence_score=8.0,
        sample_limit_per_bucket=2,
    )

    assert analysis["newly_refused"]["total"] == 4
    assert analysis["newly_refused"]["bucket_counts"] == {
        "reasonable_refusal_unanswerable": 1,
        "safe_refusal_retrieval_miss": 1,
        "possible_threshold_over_refusal_gold_cited": 1,
        "evidence_selection_miss_gold_available": 1,
        "unknown_new_refusal": 0,
    }
    assert analysis["newly_refused"]["answerable_gold_in_context"] == 2
    assert analysis["newly_refused"]["answerable_gold_cited"] == 1


def _newly_refused_result(
    question: PrimeQAQuestion,
    retrieved_doc_ids: list[str],
    cited_doc_id: str,
) -> VerifiedRAGQuestionResult:
    retrieval_results = [
        RetrievalResult(
            document=PrimeQADocument(id=doc_id, title=doc_id, text=f"{doc_id} text"),
            score=10.0 - index,
            rank=index + 1,
        )
        for index, doc_id in enumerate(retrieved_doc_ids)
    ]
    original_answer = GeneratedAnswer(
        question_id=question.id,
        answer=f"answer [{cited_doc_id}]",
        citations=[
            AnswerCitation(
                document_id=cited_doc_id,
                title=cited_doc_id,
                retrieval_rank=1,
                evidence_score=7.0,
            )
        ],
        refused=False,
    )
    verified_answer = GeneratedAnswer(
        question_id=question.id,
        answer="I do not have enough verified evidence to answer this question.",
        citations=[],
        refused=True,
    )
    return VerifiedRAGQuestionResult(
        question=question,
        retrieval_results=retrieval_results,
        original_answer=original_answer,
        verification_result=AnswerVerificationResult(
            original_answer=original_answer,
            verified_answer=verified_answer,
            citation_context_valid=True,
            reasons=["weak_evidence_score"],
        ),
    )


def _question(
    question_id: str,
    answerable: bool,
    answer_doc_id: str | None = None,
) -> PrimeQAQuestion:
    return PrimeQAQuestion(
        id=question_id,
        title=f"Question {question_id}",
        text="How to solve it?",
        answer="Gold answer." if answerable else "",
        answerable=answerable,
        answer_doc_id=answer_doc_id,
        doc_ids=[answer_doc_id] if answer_doc_id else [],
    )


def _empty_metrics() -> AnswerEvaluationMetrics:
    return AnswerEvaluationMetrics(
        total_questions=0,
        answerable_questions=0,
        unanswerable_questions=0,
        generated_answerable_questions=0,
        refused_answerable_questions=0,
        refused_unanswerable_questions=0,
        gold_doc_citation_rate=0.0,
        answerable_refusal_rate=0.0,
        unanswerable_refusal_rate=0.0,
        average_token_f1=0.0,
    )
