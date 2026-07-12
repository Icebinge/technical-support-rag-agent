from ts_rag_agent.application.evidence_selection_analysis import EvidenceSelectionAnalyzer
from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator
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


def test_evidence_selection_analysis_detects_gold_candidate_losing_to_wrong_sentence():
    question = PrimeQAQuestion(
        id="q1",
        title="How do I restart service A after upgrade?",
        text="",
        answer="Restart service A from the control panel.",
        answerable=True,
        answer_doc_id="gold-doc",
        doc_ids=["gold-doc"],
    )
    wrong_document = PrimeQADocument(
        id="wrong-doc",
        title="Upgrade service A",
        text="Upgrade service A after upgrade.",
    )
    gold_document = PrimeQADocument(
        id="gold-doc",
        title="Restart service A",
        text="Restart service A from the control panel.",
    )
    analyzer = EvidenceSelectionAnalyzer(
        retriever=_StaticRetriever([wrong_document, gold_document]),
        answer_generator=ExtractiveAnswerGenerator(
            max_sentences=1,
            min_sentence_score=1.0,
            min_sentence_chars=8,
        ),
        base_top_k=1,
    )

    analysis = analyzer.analyze(
        questions=[question],
        retrieval_top_k_values=[1, 2],
        sample_limit_per_top_k=5,
    )

    top1_summary, top2_summary = analysis.summaries
    assert top1_summary.bucket_counts["gold_not_in_context"] == 1
    assert top2_summary.bucket_counts["gold_candidate_loses_to_wrong_sentences"] == 1
    assert top2_summary.gold_newly_available_after_base_k == 1
    assert top2_summary.gold_newly_available_but_not_cited == 1
    assert analysis.cases_by_top_k[2][0].best_gold_candidate is not None
