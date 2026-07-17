from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np

from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQADocumentSection
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.bm25_retriever import tokenize_text


class SectionBM25Retriever:
    """先检索 section，再聚合回父文档的 BM25 检索器。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self._k1 = k1
        self._b = b
        self._documents: dict[str, PrimeQADocument] = {}
        self._document_ids: list[str] = []
        self._document_indices: dict[str, int] = {}
        self._sections: list[PrimeQADocumentSection] = []
        self._section_lengths = np.empty(0, dtype=np.float64)
        self._section_document_indices = np.empty(0, dtype=np.int32)
        self._avg_section_length = 0.0
        self._idf: dict[str, float] = {}
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._is_fitted = False

    def fit(
        self,
        documents: Iterable[PrimeQADocument],
        sections_by_document: dict[str, list[PrimeQADocumentSection]],
    ) -> None:
        """构建 section 级 BM25 倒排索引。"""

        self._documents = {document.id: document for document in documents}
        self._document_ids = list(self._documents)
        self._document_indices = {
            document_id: index for index, document_id in enumerate(self._document_ids)
        }
        self._sections = [
            section
            for document_id in self._documents
            for section in sections_by_document.get(document_id, [])
            if section.text.strip()
        ]
        section_lengths = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for section_index, section in enumerate(self._sections):
            document = self._documents[section.document_id]
            tokens = tokenize_text(_section_search_text(document, section))
            term_counts = Counter(tokens)
            section_lengths.append(len(tokens))

            for term, term_frequency in term_counts.items():
                postings[term].append((section_index, term_frequency))

        section_count = len(self._sections)
        self._section_lengths = np.asarray(section_lengths, dtype=np.float64)
        self._section_document_indices = np.asarray(
            [self._document_indices[section.document_id] for section in self._sections],
            dtype=np.int32,
        )
        total_length = float(self._section_lengths.sum())
        self._avg_section_length = total_length / section_count if section_count else 0.0
        self._postings = {
            term: (
                np.fromiter((item[0] for item in term_postings), dtype=np.int32),
                np.fromiter((item[1] for item in term_postings), dtype=np.float64),
            )
            for term, term_postings in postings.items()
        }
        self._idf = {
            term: _compute_idf(section_count, len(term_postings))
            for term, term_postings in postings.items()
        }
        self._is_fitted = True

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """返回按 section 分数聚合后的父文档 top-k。"""

        if not self._is_fitted:
            raise RuntimeError("SectionBM25Retriever.fit() must be called before search().")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_terms = tokenize_text(query)
        if not query_terms or not self._sections:
            return []

        section_scores = np.zeros(len(self._sections), dtype=np.float64)
        touched_sections = np.zeros(len(self._sections), dtype=np.bool_)
        for term in query_terms:
            idf = self._idf.get(term)
            if idf is None:
                continue

            section_indices, term_frequencies = self._postings[term]
            section_scores[section_indices] += self._score_terms(
                idf,
                term_frequencies,
                self._section_lengths[section_indices],
            )
            touched_sections[section_indices] = True

        touched_section_indices = np.flatnonzero(touched_sections)
        document_scores = np.full(len(self._document_ids), -np.inf, dtype=np.float64)
        if touched_section_indices.size:
            np.maximum.at(
                document_scores,
                self._section_document_indices[touched_section_indices],
                section_scores[touched_section_indices],
            )
        touched_document_indices = np.flatnonzero(np.isfinite(document_scores))
        ranked_document_indices = sorted(
            touched_document_indices,
            key=lambda index: (-float(document_scores[index]), self._document_ids[int(index)]),
        )
        return [
            RetrievalResult(
                document=self._documents[self._document_ids[int(document_index)]],
                score=float(document_scores[document_index]),
                rank=rank,
            )
            for rank, document_index in enumerate(
                ranked_document_indices[:top_k],
                start=1,
            )
        ]

    def _score_terms(
        self,
        idf: float,
        term_frequencies: np.ndarray,
        section_lengths: np.ndarray,
    ) -> np.ndarray:
        length_normalizer = 1 - self._b
        if self._avg_section_length:
            length_normalizer += self._b * section_lengths / self._avg_section_length

        numerator = term_frequencies * (self._k1 + 1)
        denominator = term_frequencies + self._k1 * length_normalizer
        return idf * numerator / denominator


def _section_search_text(document: PrimeQADocument, section: PrimeQADocumentSection) -> str:
    return f"{document.title}\n\n{section.section_id}\n\n{section.text}"


def _compute_idf(section_count: int, document_frequency: int) -> float:
    return math.log(1 + (section_count - document_frequency + 0.5) / (document_frequency + 0.5))
