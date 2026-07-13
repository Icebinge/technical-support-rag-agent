from ts_rag_agent.application.answer_composition import (
    RouteAwareAnswerCompositionPolicy,
    create_answer_composition_policy,
)
from ts_rag_agent.application.evidence_selection import SentenceEvidenceCandidate
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


def test_extractive_answer_generator_can_use_route_aware_composition_policy():
    question = PrimeQAQuestion(
        id="q1",
        title="How do I install product A?",
        text="",
        answer="Install product A with the setup command.",
        answerable=True,
        answer_doc_id="doc-a",
        doc_ids=["doc-a"],
    )
    first_document = PrimeQADocument(
        id="doc-a",
        title="Install product A",
        text="Install product A with the setup command.",
    )
    second_document = PrimeQADocument(
        id="doc-b",
        title="Product A overview",
        text="Product A overview text.",
    )
    selector = _StaticEvidenceSelector(
        [
            _candidate(first_document, 140.0, "Install product A with the setup command."),
            _candidate(second_document, 80.0, "Product A overview text."),
        ]
    )
    generator = ExtractiveAnswerGenerator(
        max_sentences=3,
        min_sentence_score=1.0,
        evidence_selector=selector,
        composition_policy=RouteAwareAnswerCompositionPolicy(),
    )

    answer = generator.generate(
        question,
        [
            RetrievalResult(document=first_document, score=10.0, rank=1),
            RetrievalResult(document=second_document, score=9.0, rank=2),
        ],
    )

    assert answer.refused is False
    assert answer.answer == "Install product A with the setup command. [doc-a]"
    assert [citation.document_id for citation in answer.citations] == ["doc-a"]


def test_answer_composition_policy_factory_rejects_unknown_policy():
    try:
        create_answer_composition_policy("unknown")
    except ValueError as exc:
        assert "composition_policy must be one of" in str(exc)
    else:
        raise AssertionError("unknown composition policy should fail")


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


class _StaticEvidenceSelector:
    name = "static"

    def __init__(self, candidates: list[SentenceEvidenceCandidate]) -> None:
        self._candidates = candidates

    def rank_sentence_candidates(self, question, retrieval_results):
        return self._candidates


def _candidate(
    document: PrimeQADocument,
    score: float,
    sentence: str,
) -> SentenceEvidenceCandidate:
    return SentenceEvidenceCandidate(
        sentence=sentence,
        retrieval_result=RetrievalResult(document=document, score=score, rank=1),
        score=score,
        overlap_terms=("install",),
    )
