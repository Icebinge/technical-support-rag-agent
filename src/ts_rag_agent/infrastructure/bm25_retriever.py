from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable

import numpy as np

from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.exact_top_k import (
    exact_top_k_indices,
    full_sort_top_k_indices,
)

Tokenizer = Callable[[str], list[str]]


class BM25Retriever:
    """基于 BM25 的轻量文档检索器。"""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self._k1 = k1
        self._b = b
        self._tokenizer = tokenizer or tokenize_text
        self._documents: list[PrimeQADocument] = []
        self._document_ids: list[str] = []
        self._doc_lengths = np.empty(0, dtype=np.float64)
        self._length_normalizers = np.empty(0, dtype=np.float64)
        self._avg_doc_length = 0.0
        self._idf: dict[str, float] = {}
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._is_fitted = False

    def fit(self, documents: Iterable[PrimeQADocument]) -> None:
        """构建 BM25 倒排索引。"""

        self._documents = list(documents)
        self._document_ids = [document.id for document in self._documents]
        doc_lengths = []
        self._idf = {}
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for doc_index, document in enumerate(self._documents):
            tokens = self._tokenizer(_document_search_text(document))
            term_counts = Counter(tokens)
            doc_lengths.append(len(tokens))

            for term, term_frequency in term_counts.items():
                postings[term].append((doc_index, term_frequency))

        document_count = len(self._documents)
        self._doc_lengths = np.asarray(doc_lengths, dtype=np.float64)
        total_length = float(self._doc_lengths.sum())
        self._avg_doc_length = total_length / document_count if document_count else 0.0
        self._length_normalizers = self._build_length_normalizers(self._doc_lengths)
        self._postings = {
            term: (
                np.fromiter((item[0] for item in term_postings), dtype=np.int32),
                np.fromiter((item[1] for item in term_postings), dtype=np.float64),
            )
            for term, term_postings in postings.items()
        }
        self._idf = {
            term: _compute_idf(document_count, len(term_postings))
            for term, term_postings in postings.items()
        }
        self._is_fitted = True

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """返回与查询最相关的 top-k 文档。"""

        return self._search(query, top_k=top_k, use_full_sort_reference=False)

    def search_full_sort_reference(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Run the historical full eligible-row sort for equivalence validation."""

        return self._search(query, top_k=top_k, use_full_sort_reference=True)

    def _search(
        self,
        query: str,
        *,
        top_k: int,
        use_full_sort_reference: bool,
    ) -> list[RetrievalResult]:

        if not self._is_fitted:
            raise RuntimeError("BM25Retriever.fit() must be called before search().")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_terms = self._tokenizer(query)
        if not query_terms or not self._documents:
            return []

        scores = np.zeros(len(self._documents), dtype=np.float64)
        for term in query_terms:
            idf = self._idf.get(term)
            if idf is None:
                continue

            doc_indices, term_frequencies = self._postings[term]
            scores[doc_indices] += self._score_terms(
                idf,
                term_frequencies,
                self._length_normalizers[doc_indices],
            )

        selector = full_sort_top_k_indices if use_full_sort_reference else exact_top_k_indices
        ranked_indices = selector(
            scores,
            top_k=top_k,
            eligible_indices=np.flatnonzero(scores),
            string_tie_breaks=self._document_ids,
        )
        return [
            RetrievalResult(
                document=self._documents[int(doc_index)],
                score=float(scores[doc_index]),
                rank=rank,
            )
            for rank, doc_index in enumerate(ranked_indices, start=1)
        ]

    def _score_terms(
        self,
        idf: float,
        term_frequencies: np.ndarray,
        length_normalizers: np.ndarray,
    ) -> np.ndarray:
        numerator = term_frequencies * (self._k1 + 1)
        denominator = term_frequencies + length_normalizers
        return idf * numerator / denominator

    def _build_length_normalizers(self, doc_lengths: np.ndarray) -> np.ndarray:
        length_normalizer = np.full(doc_lengths.shape, 1 - self._b, dtype=np.float64)
        if self._avg_doc_length:
            length_normalizer += self._b * doc_lengths / self._avg_doc_length
        return self._k1 * length_normalizer


def tokenize_text(text: str) -> list[str]:
    """将英文技术文本切分为适合 BM25 的小写词项。"""

    return re.findall(r"[a-z0-9_+#]+(?:[.+#-][a-z0-9_+#]+)*", text.lower())


def _document_search_text(document: PrimeQADocument) -> str:
    return f"{document.title}\n\n{document.text}"


def _compute_idf(document_count: int, document_frequency: int) -> float:
    return math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
