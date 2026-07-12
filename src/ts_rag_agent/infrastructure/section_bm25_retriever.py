from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable

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
        self._sections: list[PrimeQADocumentSection] = []
        self._section_lengths: list[int] = []
        self._avg_section_length = 0.0
        self._idf: dict[str, float] = {}
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._is_fitted = False

    def fit(
        self,
        documents: Iterable[PrimeQADocument],
        sections_by_document: dict[str, list[PrimeQADocumentSection]],
    ) -> None:
        """构建 section 级 BM25 倒排索引。"""

        self._documents = {document.id: document for document in documents}
        self._sections = [
            section
            for document_id in self._documents
            for section in sections_by_document.get(document_id, [])
            if section.text.strip()
        ]
        self._section_lengths = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for section_index, section in enumerate(self._sections):
            document = self._documents[section.document_id]
            tokens = tokenize_text(_section_search_text(document, section))
            term_counts = Counter(tokens)
            self._section_lengths.append(len(tokens))

            for term, term_frequency in term_counts.items():
                postings[term].append((section_index, term_frequency))

        section_count = len(self._sections)
        total_length = sum(self._section_lengths)
        self._avg_section_length = total_length / section_count if section_count else 0.0
        self._postings = dict(postings)
        self._idf = {
            term: _compute_idf(section_count, len(term_postings))
            for term, term_postings in self._postings.items()
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

        section_scores: dict[int, float] = defaultdict(float)
        for term in query_terms:
            idf = self._idf.get(term)
            if idf is None:
                continue

            for section_index, term_frequency in self._postings[term]:
                section_scores[section_index] += self._score_term(
                    idf,
                    term_frequency,
                    self._section_lengths[section_index],
                )

        document_scores: dict[str, float] = {}
        for section_index, section_score in section_scores.items():
            section = self._sections[section_index]
            current_score = document_scores.get(section.document_id)
            if current_score is None or section_score > current_score:
                document_scores[section.document_id] = section_score

        ranked_doc_ids = sorted(
            document_scores,
            key=lambda doc_id: (-document_scores[doc_id], doc_id),
        )
        return [
            RetrievalResult(
                document=self._documents[doc_id],
                score=document_scores[doc_id],
                rank=rank,
            )
            for rank, doc_id in enumerate(ranked_doc_ids[:top_k], start=1)
        ]

    def _score_term(self, idf: float, term_frequency: int, section_length: int) -> float:
        length_normalizer = 1 - self._b
        if self._avg_section_length:
            length_normalizer += self._b * section_length / self._avg_section_length

        numerator = term_frequency * (self._k1 + 1)
        denominator = term_frequency + self._k1 * length_normalizer
        return idf * numerator / denominator


def _section_search_text(document: PrimeQADocument, section: PrimeQADocumentSection) -> str:
    return f"{document.title}\n\n{section.section_id}\n\n{section.text}"


def _compute_idf(section_count: int, document_frequency: int) -> float:
    return math.log(1 + (section_count - document_frequency + 0.5) / (document_frequency + 0.5))
