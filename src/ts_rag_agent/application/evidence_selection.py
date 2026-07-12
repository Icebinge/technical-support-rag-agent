from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ts_rag_agent.domain.dataset import PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


@dataclass(frozen=True)
class SentenceEvidenceCandidate:
    """One sentence-level evidence candidate selected from retrieved documents."""

    sentence: str
    retrieval_result: RetrievalResult
    score: float
    overlap_terms: tuple[str, ...]


class SentenceEvidenceSelector(Protocol):
    """Ranks sentence evidence candidates for an extractive RAG answerer."""

    @property
    def name(self) -> str:
        """Stable selector name used in experiment reports."""

    def rank_sentence_candidates(
        self,
        question: PrimeQAQuestion,
        retrieval_results: Sequence[RetrievalResult],
    ) -> list[SentenceEvidenceCandidate]:
        """Return ranked evidence candidates for one question."""


class OverlapSentenceEvidenceSelector:
    """Original query-term-overlap selector kept as a baseline."""

    name = "overlap_sentence"

    def __init__(self, min_sentence_chars: int = 24) -> None:
        if min_sentence_chars <= 0:
            raise ValueError("min_sentence_chars must be positive")

        self._min_sentence_chars = min_sentence_chars
        self._sentence_cache: dict[str, list[str]] = {}

    def rank_sentence_candidates(
        self,
        question: PrimeQAQuestion,
        retrieval_results: Sequence[RetrievalResult],
    ) -> list[SentenceEvidenceCandidate]:
        query_terms = set(tokenize_text(question.full_question))
        candidates = []
        seen_sentences = set()

        for retrieval_result in retrieval_results:
            for sentence in self._get_document_sentences(
                retrieval_result.document.id,
                retrieval_result.document.text,
            ):
                normalized_sentence = normalize_sentence(sentence)
                if len(normalized_sentence) < self._min_sentence_chars:
                    continue
                if normalized_sentence in seen_sentences:
                    continue

                seen_sentences.add(normalized_sentence)
                sentence_terms = set(tokenize_text(normalized_sentence))
                overlap_terms = tuple(sorted(query_terms & sentence_terms))
                if not overlap_terms:
                    continue

                score = len(overlap_terms) / math.log2(retrieval_result.rank + 1)
                candidates.append(
                    SentenceEvidenceCandidate(
                        sentence=normalized_sentence,
                        retrieval_result=retrieval_result,
                        score=score,
                        overlap_terms=overlap_terms,
                    )
                )

        return sorted(
            candidates,
            key=lambda candidate: (
                -candidate.score,
                candidate.retrieval_result.rank,
                candidate.retrieval_result.document.id,
            ),
        )

    def _get_document_sentences(self, document_id: str, text: str) -> list[str]:
        if document_id not in self._sentence_cache:
            self._sentence_cache[document_id] = split_sentences(text)
        return self._sentence_cache[document_id]


class BM25SentenceEvidenceSelector:
    """BM25-style sentence selector with IDF weighting and noise penalties."""

    name = "bm25_sentence"

    def __init__(
        self,
        min_sentence_chars: int = 24,
        max_candidates_per_document: int = 1,
        k1: float = 1.2,
        b: float = 0.35,
        score_scale: float = 2.5,
    ) -> None:
        if min_sentence_chars <= 0:
            raise ValueError("min_sentence_chars must be positive")
        if max_candidates_per_document <= 0:
            raise ValueError("max_candidates_per_document must be positive")
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        if score_scale <= 0:
            raise ValueError("score_scale must be positive")

        self._min_sentence_chars = min_sentence_chars
        self._max_candidates_per_document = max_candidates_per_document
        self._k1 = k1
        self._b = b
        self._score_scale = score_scale
        self._sentence_cache: dict[str, list[str]] = {}

    def rank_sentence_candidates(
        self,
        question: PrimeQAQuestion,
        retrieval_results: Sequence[RetrievalResult],
    ) -> list[SentenceEvidenceCandidate]:
        query_terms = _content_terms(tokenize_text(question.full_question))
        if not query_terms:
            return []

        sentence_rows = self._collect_sentence_rows(query_terms, retrieval_results)
        if not sentence_rows:
            return []

        sentence_count = len(sentence_rows)
        avg_sentence_length = sum(len(row.terms) for row in sentence_rows) / sentence_count
        idf_by_term = _compute_idf_by_term(query_terms, sentence_rows)

        candidates = [
            self._score_sentence_row(
                row=row,
                query_terms=query_terms,
                idf_by_term=idf_by_term,
                avg_sentence_length=avg_sentence_length,
            )
            for row in sentence_rows
        ]
        ranked_candidates = sorted(
            candidates,
            key=lambda candidate: (
                -candidate.score,
                candidate.retrieval_result.rank,
                candidate.retrieval_result.document.id,
            ),
        )
        return _cap_candidates_per_document(
            ranked_candidates,
            max_candidates_per_document=self._max_candidates_per_document,
        )

    def _collect_sentence_rows(
        self,
        query_terms: set[str],
        retrieval_results: Sequence[RetrievalResult],
    ) -> list[_SentenceRow]:
        rows = []
        seen_sentences = set()

        for retrieval_result in retrieval_results:
            for sentence in self._get_document_sentences(
                retrieval_result.document.id,
                retrieval_result.document.text,
            ):
                normalized_sentence = normalize_sentence(sentence)
                if len(normalized_sentence) < self._min_sentence_chars:
                    continue
                if normalized_sentence in seen_sentences:
                    continue

                terms = _content_terms(tokenize_text(normalized_sentence))
                overlap_terms = tuple(sorted(query_terms & terms))
                if not overlap_terms:
                    continue

                seen_sentences.add(normalized_sentence)
                rows.append(
                    _SentenceRow(
                        sentence=normalized_sentence,
                        retrieval_result=retrieval_result,
                        terms=terms,
                        overlap_terms=overlap_terms,
                    )
                )

        return rows

    def _score_sentence_row(
        self,
        row: _SentenceRow,
        query_terms: set[str],
        idf_by_term: dict[str, float],
        avg_sentence_length: float,
    ) -> SentenceEvidenceCandidate:
        term_counts = Counter(row.terms)
        sentence_length = len(row.terms)
        length_normalizer = 1 - self._b
        if avg_sentence_length:
            length_normalizer += self._b * sentence_length / avg_sentence_length

        bm25_score = 0.0
        for term in query_terms:
            term_frequency = term_counts.get(term, 0)
            if term_frequency == 0:
                continue

            numerator = term_frequency * (self._k1 + 1)
            denominator = term_frequency + self._k1 * length_normalizer
            bm25_score += idf_by_term[term] * numerator / denominator

        retrieval_prior = 1 / math.log2(row.retrieval_result.rank + 1)
        overlap_bonus = 0.25 * len(row.overlap_terms)
        score = (bm25_score + overlap_bonus) * (1 + 0.35 * retrieval_prior)
        score *= _sentence_noise_penalty(row.sentence)
        score *= self._score_scale

        return SentenceEvidenceCandidate(
            sentence=row.sentence,
            retrieval_result=row.retrieval_result,
            score=score,
            overlap_terms=row.overlap_terms,
        )

    def _get_document_sentences(self, document_id: str, text: str) -> list[str]:
        if document_id not in self._sentence_cache:
            self._sentence_cache[document_id] = split_sentences(text)
        return self._sentence_cache[document_id]


def create_sentence_evidence_selector(
    selector_name: str,
    min_sentence_chars: int = 24,
    max_candidates_per_document: int = 1,
) -> SentenceEvidenceSelector:
    """Create a sentence evidence selector from a stable experiment name."""

    normalized_name = selector_name.strip().lower().replace("-", "_")
    if normalized_name in {"overlap", "overlap_sentence"}:
        return OverlapSentenceEvidenceSelector(min_sentence_chars=min_sentence_chars)
    if normalized_name in {"bm25", "bm25_sentence"}:
        return BM25SentenceEvidenceSelector(
            min_sentence_chars=min_sentence_chars,
            max_candidates_per_document=max_candidates_per_document,
        )

    raise ValueError(
        "selector_name must be one of: overlap, overlap_sentence, bm25, bm25_sentence"
    )


@dataclass(frozen=True)
class _SentenceRow:
    sentence: str
    retrieval_result: RetrievalResult
    terms: set[str]
    overlap_terms: tuple[str, ...]


STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "get",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "with",
    "you",
}


def split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        if sentence.strip()
    ]


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_+#]+(?:[.+#-][a-z0-9_+#]+)*", text.lower())


def normalize_sentence(sentence: str) -> str:
    return " ".join(sentence.split())


def _content_terms(tokens: list[str]) -> set[str]:
    return {
        token
        for token in tokens
        if token not in STOPWORDS and not _is_noisy_token(token)
    }


def _is_noisy_token(token: str) -> bool:
    if len(token) > 40:
        return True
    if len(token) == 1 and not token.isdigit() and token not in {"c", "r"}:
        return True
    return False


def _compute_idf_by_term(query_terms: set[str], rows: list[_SentenceRow]) -> dict[str, float]:
    sentence_count = len(rows)
    document_frequency = {
        term: sum(1 for row in rows if term in row.terms)
        for term in query_terms
    }
    return {
        term: math.log(
            1 + (sentence_count - frequency + 0.5) / (frequency + 0.5)
        )
        for term, frequency in document_frequency.items()
    }


def _cap_candidates_per_document(
    candidates: list[SentenceEvidenceCandidate],
    max_candidates_per_document: int,
) -> list[SentenceEvidenceCandidate]:
    kept_candidates = []
    counts_by_document_id: Counter[str] = Counter()

    for candidate in candidates:
        document_id = candidate.retrieval_result.document.id
        if counts_by_document_id[document_id] >= max_candidates_per_document:
            continue

        kept_candidates.append(candidate)
        counts_by_document_id[document_id] += 1

    return kept_candidates


def _sentence_noise_penalty(sentence: str) -> float:
    penalty = 1.0
    if len(sentence) > 260:
        penalty *= 0.72
    if len(sentence) > 500:
        penalty *= 0.55
    if _contains_path_like_text(sentence):
        penalty *= 0.7
    if _contains_dump_or_trace_text(sentence):
        penalty *= 0.75
    if _symbol_ratio(sentence) > 0.18:
        penalty *= 0.8
    return penalty


def _contains_path_like_text(sentence: str) -> bool:
    return bool(re.search(r"([a-z]:\\|/[^ ]+/|\\[^ ]+\\)", sentence.lower()))


def _contains_dump_or_trace_text(sentence: str) -> bool:
    return bool(
        re.search(
            r"\b(trace|stack|dump|exception|thread|timestamp|heapdump|javacore)\b",
            sentence.lower(),
        )
    )


def _symbol_ratio(sentence: str) -> float:
    if not sentence:
        return 0.0
    symbol_count = sum(1 for char in sentence if not char.isalnum() and not char.isspace())
    return symbol_count / len(sentence)
