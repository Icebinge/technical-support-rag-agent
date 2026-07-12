from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.reranker import (
    RerankingRetriever,
    build_reranker_document_text,
)


@dataclass
class FixedCandidateRetriever:
    """测试用的固定候选召回器。"""

    results: list[RetrievalResult]

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        return self.results[:top_k]


class FakePairScorer:
    """测试用的成对打分器。"""

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        scores = []
        for _query, document_text in pairs:
            if "correct" in document_text:
                scores.append(2.0)
            else:
                scores.append(1.0)
        return np.asarray(scores, dtype=np.float32)


def test_reranking_retriever_reorders_candidates():
    doc_a = PrimeQADocument(id="doc-a", title="Wrong", text="not enough evidence")
    doc_b = PrimeQADocument(id="doc-b", title="Correct", text="correct answer evidence")
    candidate_retriever = FixedCandidateRetriever(
        [
            RetrievalResult(document=doc_a, score=10.0, rank=1),
            RetrievalResult(document=doc_b, score=8.0, rank=2),
        ]
    )
    reranker = RerankingRetriever(
        candidate_retriever=candidate_retriever,
        scorer=FakePairScorer(),
        candidate_top_k=2,
    )

    results = reranker.search("question", top_k=2)

    assert [result.document.id for result in results] == ["doc-b", "doc-a"]
    assert results[0].rank == 1
    assert results[0].score > results[1].score


def test_build_reranker_document_text_truncates_document():
    document = PrimeQADocument(id="doc-a", title="Title", text="x" * 100)

    text = build_reranker_document_text(document, document_text_max_chars=12)

    assert text == "Title\n\nxxxxx"
