from ts_rag_agent.application.answer_gap_analysis import AnswerGapAnalyzer
from ts_rag_agent.application.evidence_selection import (
    BM25SentenceEvidenceSelector,
    OverlapSentenceEvidenceSelector,
)
from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator
from ts_rag_agent.application.text_metrics import token_f1
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


class _StaticRetriever:
    def __init__(self, documents: list[PrimeQADocument]) -> None:
        self._documents = documents

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        return [
            RetrievalResult(document=document, score=10.0 - index, rank=index + 1)
            for index, document in enumerate(self._documents[:top_k])
        ]


def test_token_f1_scores_partial_overlap():
    score = token_f1(
        "Restart service A from the control panel.",
        "Restart service A from the control panel after upgrade.",
    )

    assert 0 < score < 1


def test_answer_gap_detects_gold_window_that_beats_selected_sentence():
    question = PrimeQAQuestion(
        id="q1",
        title="How do I restart service A after upgrade?",
        text="",
        answer="Restart service A from the control panel after upgrade.",
        answerable=True,
        answer_doc_id="gold",
        doc_ids=["gold"],
    )
    gold_document = PrimeQADocument(
        id="gold",
        title="Restart service A",
        text="Restart service A. Use the control panel after upgrade.",
    )
    analyzer = AnswerGapAnalyzer(
        retriever=_StaticRetriever([gold_document]),
        answer_generator=ExtractiveAnswerGenerator(
            max_sentences=1,
            min_sentence_score=0.0,
            min_sentence_chars=1,
            evidence_selector=BM25SentenceEvidenceSelector(
                min_sentence_chars=1,
                max_candidates_per_document=1,
            ),
        ),
        documents_by_id={"gold": gold_document},
        min_gold_sentence_chars=1,
        max_window_sentences=2,
        f1_gap_margin=0.05,
    )

    result = analyzer.analyze([question], retrieval_top_k=1, sample_limit=5)

    assert result.summary.total_answerable_questions == 1
    assert result.summary.gold_in_context == 1
    assert result.summary.selected_gold_citation == 1
    assert result.summary.gold_window_beats_selected_answer == 1
    assert result.summary.question_route_counts["install_upgrade_config"] == 1
    assert result.summary.selected_selector_counts["bm25_sentence"] == 1
    assert result.cases[0].best_gold_window is not None
    assert result.cases[0].best_gold_window.sentence_count == 2
    assert result.cases[0].question_route == "install_upgrade_config"
    assert result.cases[0].selected_selector_name == "bm25_sentence"


def test_answer_gap_detects_gold_in_context_not_selected():
    question = PrimeQAQuestion(
        id="q2",
        title="How do I restart service A?",
        text="The service failed after upgrade.",
        answer="Use the official service A restart procedure.",
        answerable=True,
        answer_doc_id="gold",
        doc_ids=["gold", "wrong"],
    )
    wrong_document = PrimeQADocument(
        id="wrong",
        title="Noisy restart trace",
        text="Restart service A failed after upgrade with restart service A trace.",
    )
    gold_document = PrimeQADocument(
        id="gold",
        title="Official restart procedure",
        text="Use the official service A restart procedure.",
    )
    analyzer = AnswerGapAnalyzer(
        retriever=_StaticRetriever([wrong_document, gold_document]),
        answer_generator=ExtractiveAnswerGenerator(
            max_sentences=1,
            min_sentence_score=0.0,
            min_sentence_chars=1,
            evidence_selector=OverlapSentenceEvidenceSelector(min_sentence_chars=1),
        ),
        documents_by_id={"wrong": wrong_document, "gold": gold_document},
        min_gold_sentence_chars=1,
        max_window_sentences=1,
    )

    result = analyzer.analyze([question], retrieval_top_k=2, sample_limit=5)

    assert result.summary.gold_in_context == 1
    assert result.summary.selected_gold_citation == 0
    assert result.summary.bucket_counts["gold_in_context_not_selected"] == 1
    assert result.cases[0].bucket == "gold_in_context_not_selected"
