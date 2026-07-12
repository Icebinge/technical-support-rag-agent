from dataclasses import dataclass

from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.hybrid_retriever import HybridRetriever


@dataclass
class FixedRetriever:
    """测试用的固定排序检索器。"""

    results: list[RetrievalResult]

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        return self.results[:top_k]


def test_hybrid_retriever_uses_reciprocal_rank_fusion():
    doc_a = PrimeQADocument(id="doc-a", title="A", text="A")
    doc_b = PrimeQADocument(id="doc-b", title="B", text="B")
    doc_c = PrimeQADocument(id="doc-c", title="C", text="C")
    sparse = FixedRetriever(
        [
            RetrievalResult(document=doc_a, score=10.0, rank=1),
            RetrievalResult(document=doc_b, score=9.0, rank=2),
        ]
    )
    dense = FixedRetriever(
        [
            RetrievalResult(document=doc_b, score=0.9, rank=1),
            RetrievalResult(document=doc_c, score=0.8, rank=2),
        ]
    )
    hybrid = HybridRetriever(sparse, dense, candidate_top_k=2, rrf_k=60)

    results = hybrid.search("query", top_k=3)

    assert [result.document.id for result in results] == ["doc-b", "doc-a", "doc-c"]
    assert results[0].rank == 1
    assert results[0].score > results[1].score
