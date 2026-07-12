from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator, evaluate_answers
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


def test_extractive_answer_generator_returns_cited_sentence():
    question = PrimeQAQuestion(
        id="q1",
        title="How do I restart service A?",
        text="The service stopped after upgrade.",
        answer="Restart service A from the control panel.",
        answerable=True,
        answer_doc_id="doc-a",
        doc_ids=["doc-a"],
    )
    document = PrimeQADocument(
        id="doc-a",
        title="Restart service A",
        text="Restart service A from the control panel. Database setup is unrelated.",
    )
    generator = ExtractiveAnswerGenerator(
        max_sentences=1,
        min_sentence_score=1.0,
        min_sentence_chars=8,
    )

    answer = generator.generate(
        question,
        [RetrievalResult(document=document, score=10.0, rank=1)],
    )

    assert answer.refused is False
    assert "Restart service A" in answer.answer
    assert "[doc-a]" in answer.answer
    assert answer.citations[0].document_id == "doc-a"


def test_extractive_answer_generator_refuses_without_evidence():
    question = PrimeQAQuestion(
        id="q1",
        title="How do I restart service A?",
        text="",
        answer="Restart service A.",
        answerable=True,
        answer_doc_id="doc-a",
        doc_ids=["doc-a"],
    )
    document = PrimeQADocument(id="doc-b", title="Database", text="Install database drivers.")
    generator = ExtractiveAnswerGenerator(
        max_sentences=1,
        min_sentence_score=1.0,
        min_sentence_chars=8,
    )

    answer = generator.generate(
        question,
        [RetrievalResult(document=document, score=10.0, rank=1)],
    )

    assert answer.refused is True
    assert answer.citations == []


def test_evaluate_answers_reports_citation_and_refusal_metrics():
    questions = [
        PrimeQAQuestion(
            id="q1",
            title="How do I restart service A?",
            text="",
            answer="Restart service A.",
            answerable=True,
            answer_doc_id="doc-a",
            doc_ids=["doc-a"],
        ),
        PrimeQAQuestion(
            id="q2",
            title="What is the private password?",
            text="",
            answer="",
            answerable=False,
            answer_doc_id=None,
            doc_ids=[],
        ),
    ]
    document = PrimeQADocument(id="doc-a", title="Restart service A", text="Restart service A.")
    generator = ExtractiveAnswerGenerator(
        max_sentences=1,
        min_sentence_score=1.0,
        min_sentence_chars=8,
    )
    answers = [
        generator.generate(questions[0], [RetrievalResult(document=document, score=10.0, rank=1)]),
        generator.generate(questions[1], []),
    ]

    metrics = evaluate_answers(questions, answers)

    assert metrics.total_questions == 2
    assert metrics.answerable_questions == 1
    assert metrics.unanswerable_questions == 1
    assert metrics.gold_doc_citation_rate == 1.0
    assert metrics.unanswerable_refusal_rate == 1.0
