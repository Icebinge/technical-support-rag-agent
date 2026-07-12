from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable

from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.domain.retrieval import RetrievalResult

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
        self._doc_lengths: list[int] = []
        self._avg_doc_length = 0.0
        self._idf: dict[str, float] = {}
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._is_fitted = False

    def fit(self, documents: Iterable[PrimeQADocument]) -> None:
        """构建 BM25 倒排索引。"""

        self._documents = list(documents)
        self._doc_lengths = []
        self._idf = {}
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for doc_index, document in enumerate(self._documents):
            tokens = self._tokenizer(_document_search_text(document))
            term_counts = Counter(tokens)
            self._doc_lengths.append(len(tokens))

            for term, term_frequency in term_counts.items():
                postings[term].append((doc_index, term_frequency))

        document_count = len(self._documents)
        total_length = sum(self._doc_lengths)
        self._avg_doc_length = total_length / document_count if document_count else 0.0
        self._postings = dict(postings)
        self._idf = {
            term: _compute_idf(document_count, len(term_postings))
            for term, term_postings in self._postings.items()
        }
        self._is_fitted = True

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """返回与查询最相关的 top-k 文档。"""

        if not self._is_fitted:
            raise RuntimeError("BM25Retriever.fit() must be called before search().")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_terms = self._tokenizer(query)
        if not query_terms or not self._documents:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in query_terms:
            idf = self._idf.get(term)
            if idf is None:
                continue

            for doc_index, term_frequency in self._postings[term]:
                scores[doc_index] += self._score_term(
                    idf,
                    term_frequency,
                    self._doc_lengths[doc_index],
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], self._documents[item[0]].id))
        return [
            RetrievalResult(document=self._documents[doc_index], score=score, rank=rank)
            for rank, (doc_index, score) in enumerate(ranked[:top_k], start=1)
        ]

    def _score_term(self, idf: float, term_frequency: int, doc_length: int) -> float:
        length_normalizer = 1 - self._b
        if self._avg_doc_length:
            length_normalizer += self._b * doc_length / self._avg_doc_length

        numerator = term_frequency * (self._k1 + 1)
        denominator = term_frequency + self._k1 * length_normalizer
        return idf * numerator / denominator


def tokenize_text(text: str) -> list[str]:
    """将英文技术文本切分为适合 BM25 的小写词项。"""

    return re.findall(r"[a-z0-9_+#]+(?:[.+#-][a-z0-9_+#]+)*", text.lower())


def _document_search_text(document: PrimeQADocument) -> str:
    return f"{document.title}\n\n{document.text}"


def _compute_idf(document_count: int, document_frequency: int) -> float:
    return math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
