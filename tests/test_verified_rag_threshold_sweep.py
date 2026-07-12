from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator
from ts_rag_agent.application.verified_rag_threshold_sweep import (
    VerifiedRAGThresholdSweeper,
)
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


def test_threshold_sweeper_reuses_base_results_and_compares_configs():
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
        text="Restart service A from the control panel.",
    )
    sweeper = VerifiedRAGThresholdSweeper(
        retriever=_StaticRetriever([document]),
        answer_generator=ExtractiveAnswerGenerator(
            max_sentences=1,
            min_sentence_score=1.0,
            min_sentence_chars=8,
        ),
    )

    result = sweeper.sweep(
        questions=[question],
        retrieval_top_k_values=[1],
        min_evidence_scores=[1.0, 20.0],
        max_citation_ranks=[1],
    )

    assert len(result.summaries) == 2
    assert result.summaries[0].verified_metrics.answerable_refusal_rate == 0.0
    assert result.summaries[1].verified_metrics.answerable_refusal_rate == 1.0
    assert result.pareto_candidate_indices == [0]
