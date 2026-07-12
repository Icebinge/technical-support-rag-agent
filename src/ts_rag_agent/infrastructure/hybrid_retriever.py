from __future__ import annotations

from collections import defaultdict

from ts_rag_agent.application.retrieval_evaluation import Retriever
from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.domain.retrieval import RetrievalResult


class HybridRetriever:
    """用 RRF 融合稀疏检索和稠密检索的结果。"""

    def __init__(
        self,
        sparse_retriever: Retriever,
        dense_retriever: Retriever,
        candidate_top_k: int = 100,
        rrf_k: int = 60,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        if candidate_top_k <= 0:
            raise ValueError("candidate_top_k must be positive")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if sparse_weight < 0 or dense_weight < 0:
            raise ValueError("retriever weights must be non-negative")

        self._sparse_retriever = sparse_retriever
        self._dense_retriever = dense_retriever
        self._candidate_top_k = candidate_top_k
        self._rrf_k = rrf_k
        self._sparse_weight = sparse_weight
        self._dense_weight = dense_weight

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """返回稀疏和稠密检索融合后的 top-k 文档。"""

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        documents: dict[str, PrimeQADocument] = {}
        scores: dict[str, float] = defaultdict(float)
        self._accumulate_scores(
            results=self._sparse_retriever.search(query, top_k=self._candidate_top_k),
            documents=documents,
            scores=scores,
            weight=self._sparse_weight,
        )
        self._accumulate_scores(
            results=self._dense_retriever.search(query, top_k=self._candidate_top_k),
            documents=documents,
            scores=scores,
            weight=self._dense_weight,
        )

        ranked_doc_ids = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
        return [
            RetrievalResult(document=documents[doc_id], score=scores[doc_id], rank=rank)
            for rank, doc_id in enumerate(ranked_doc_ids[:top_k], start=1)
        ]

    def _accumulate_scores(
        self,
        results: list[RetrievalResult],
        documents: dict[str, PrimeQADocument],
        scores: dict[str, float],
        weight: float,
    ) -> None:
        for result in results:
            doc_id = result.document.id
            documents[doc_id] = result.document
            scores[doc_id] += weight / (self._rrf_k + result.rank)
