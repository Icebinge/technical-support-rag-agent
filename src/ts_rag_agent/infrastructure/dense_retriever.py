from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np

from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.domain.retrieval import RetrievalResult


class TextEmbeddingModel(Protocol):
    """文本向量模型接口。"""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """把一批文本编码成二维向量矩阵。"""


class SentenceTransformerEmbeddingModel:
    """基于 sentence-transformers 的本地 embedding 模型封装。"""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        device: str | None = None,
        show_progress_bar: bool = False,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._batch_size = batch_size
        self._show_progress_bar = show_progress_bar
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=self._show_progress_bar,
        )
        return np.asarray(embeddings, dtype=np.float32)


class DenseRetriever:
    """基于文本向量相似度的稠密检索器。"""

    def __init__(
        self,
        encoder: TextEmbeddingModel,
        document_text_max_chars: int = 1600,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        if document_text_max_chars <= 0:
            raise ValueError("document_text_max_chars must be positive")

        self._encoder = encoder
        self._document_text_max_chars = document_text_max_chars
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._documents: list[PrimeQADocument] = []
        self._embeddings: np.ndarray | None = None

    def fit(self, documents: Iterable[PrimeQADocument]) -> None:
        """编码文档并建立内存向量索引。"""

        document_list = list(documents)
        texts = [
            _document_search_text(
                document=document,
                max_chars=self._document_text_max_chars,
                prefix=self._document_prefix,
            )
            for document in document_list
        ]
        embeddings = self._encoder.encode(texts)
        self.fit_embeddings(document_list, embeddings)

    def fit_embeddings(
        self,
        documents: Iterable[PrimeQADocument],
        embeddings: np.ndarray,
    ) -> None:
        """从已计算好的文档向量建立索引。"""

        document_list = list(documents)
        raw_embedding_matrix = np.asarray(embeddings, dtype=np.float32)
        if raw_embedding_matrix.ndim != 2:
            raise ValueError("embeddings must be a 2D matrix")

        embedding_matrix = _normalize_rows(raw_embedding_matrix)
        if embedding_matrix.ndim != 2:
            raise ValueError("embeddings must be a 2D matrix")
        if len(document_list) != embedding_matrix.shape[0]:
            raise ValueError("document count must match embedding row count")

        self._documents = document_list
        self._embeddings = embedding_matrix

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """返回与查询向量最相似的 top-k 文档。"""

        if self._embeddings is None:
            raise RuntimeError("DenseRetriever.fit() must be called before search().")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self._documents:
            return []

        query_embedding = self._encoder.encode([f"{self._query_prefix}{query}"])
        query_vector = _normalize_rows(np.asarray(query_embedding, dtype=np.float32))[0]
        scores = self._embeddings @ query_vector
        top_indices = np.argsort(-scores, kind="stable")[:top_k]

        return [
            RetrievalResult(
                document=self._documents[int(index)],
                score=float(scores[int(index)]),
                rank=rank,
            )
            for rank, index in enumerate(top_indices, start=1)
        ]


def build_document_texts(
    documents: Iterable[PrimeQADocument],
    document_text_max_chars: int = 1600,
    document_prefix: str = "",
) -> list[str]:
    """构建用于 dense embedding 的文档文本。"""

    return [
        _document_search_text(
            document=document,
            max_chars=document_text_max_chars,
            prefix=document_prefix,
        )
        for document in documents
    ]


def _document_search_text(document: PrimeQADocument, max_chars: int, prefix: str) -> str:
    text = f"{prefix}{document.title}\n\n{document.text}"
    return text[:max_chars]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    return matrix / safe_norms
