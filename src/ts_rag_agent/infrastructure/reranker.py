from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ts_rag_agent.application.retrieval_evaluation import Retriever
from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.domain.retrieval import RetrievalResult


class PairScoringModel(Protocol):
    """query-document 成对打分模型接口。"""

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """对一批 query-document 文本对输出相关性分数。"""


class CrossEncoderPairScoringModel:
    """基于 sentence-transformers CrossEncoder 的重排模型封装。"""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        max_length: int | None = None,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._batch_size = batch_size
        self._model = CrossEncoder(model_name, max_length=max_length, device=device)

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        scores = self._model.predict(
            list(pairs),
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return np.asarray(scores, dtype=np.float32)


class RerankingRetriever:
    """先召回候选文档，再用 pair scorer 重排的检索器。"""

    def __init__(
        self,
        candidate_retriever: Retriever,
        scorer: PairScoringModel,
        candidate_top_k: int = 50,
        document_text_max_chars: int = 1600,
    ) -> None:
        if candidate_top_k <= 0:
            raise ValueError("candidate_top_k must be positive")
        if document_text_max_chars <= 0:
            raise ValueError("document_text_max_chars must be positive")

        self._candidate_retriever = candidate_retriever
        self._scorer = scorer
        self._candidate_top_k = candidate_top_k
        self._document_text_max_chars = document_text_max_chars

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """返回重排后的 top-k 文档。"""

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        candidates = self._candidate_retriever.search(query, top_k=self._candidate_top_k)
        if not candidates:
            return []

        pairs = [
            (query, build_reranker_document_text(candidate.document, self._document_text_max_chars))
            for candidate in candidates
        ]
        scores = self._scorer.predict(pairs)
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].document.id),
        )

        return [
            RetrievalResult(document=candidate.document, score=float(score), rank=rank)
            for rank, (candidate, score) in enumerate(ranked[:top_k], start=1)
        ]


def build_reranker_document_text(
    document: PrimeQADocument,
    document_text_max_chars: int = 1600,
) -> str:
    """构建给 reranker 使用的文档文本。"""

    return f"{document.title}\n\n{document.text}"[:document_text_max_chars]
