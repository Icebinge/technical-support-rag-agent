from __future__ import annotations

import re
from collections.abc import Sequence

from ts_rag_agent.application.evidence_selection import (
    HybridRoutingEvidenceSelector,
    SentenceEvidenceCandidate,
    SentenceEvidenceSelector,
    classify_question_route,
    normalize_sentence,
    split_sentences,
    tokenize_text,
)
from ts_rag_agent.domain.dataset import PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


class LocalWindowRerankEvidenceSelector:
    """Rerank compact local windows around already-selected evidence candidates."""

    def __init__(
        self,
        base_selector: SentenceEvidenceSelector | None = None,
        min_sentence_chars: int = 24,
        max_candidates_per_document: int = 3,
        selected_candidate_limit: int = 3,
        max_window_sentences: int = 3,
        rerank_routes: tuple[str, ...] = ("other",),
    ) -> None:
        if min_sentence_chars <= 0:
            raise ValueError("min_sentence_chars must be positive")
        if max_candidates_per_document <= 0:
            raise ValueError("max_candidates_per_document must be positive")
        if selected_candidate_limit <= 0:
            raise ValueError("selected_candidate_limit must be positive")
        if max_window_sentences <= 0:
            raise ValueError("max_window_sentences must be positive")
        if not rerank_routes:
            raise ValueError("rerank_routes must not be empty")

        self._base_selector = base_selector or HybridRoutingEvidenceSelector(
            min_sentence_chars=min_sentence_chars,
            answer_aware_max_candidates_per_document=max_candidates_per_document,
            section_span_max_candidates_per_document=1,
        )
        self._min_sentence_chars = min_sentence_chars
        self._selected_candidate_limit = selected_candidate_limit
        self._max_window_sentences = max_window_sentences
        self._rerank_routes = tuple(sorted(set(rerank_routes)))
        self._name = (
            f"local_window_rerank_{self._base_selector.name}_"
            f"routes_{'_'.join(self._rerank_routes)}_"
            f"top{selected_candidate_limit}_mws{max_window_sentences}"
        )

    @property
    def name(self) -> str:
        """Stable selector name used in experiment reports."""

        return self._name

    def rank_sentence_candidates(
        self,
        question: PrimeQAQuestion,
        retrieval_results: Sequence[RetrievalResult],
    ) -> list[SentenceEvidenceCandidate]:
        """Return base candidates with local reranked replacements for target routes."""

        base_candidates = self._base_selector.rank_sentence_candidates(
            question,
            retrieval_results,
        )
        question_route = classify_question_route(question)
        if question_route not in self._rerank_routes:
            return base_candidates

        reranked_candidates = []
        for index, candidate in enumerate(base_candidates):
            if index >= self._selected_candidate_limit:
                reranked_candidates.append(candidate)
                continue

            reranked_candidates.append(self._rerank_candidate(question, candidate))
        return reranked_candidates

    def _rerank_candidate(
        self,
        question: PrimeQAQuestion,
        candidate: SentenceEvidenceCandidate,
    ) -> SentenceEvidenceCandidate:
        local_windows = self._build_local_windows(candidate)
        if not local_windows:
            return candidate

        question_terms = _content_terms(tokenize_text(question.full_question))
        anchor_terms = _content_terms(tokenize_text(candidate.sentence))
        best_window = max(
            local_windows,
            key=lambda window: (
                _score_local_window(
                    window=window,
                    question_terms=question_terms,
                    anchor_terms=anchor_terms,
                ),
                -len(tokenize_text(window)),
                window,
            ),
        )
        overlap_terms = tuple(
            sorted(question_terms & _content_terms(tokenize_text(best_window)))
        )
        return SentenceEvidenceCandidate(
            sentence=best_window,
            retrieval_result=candidate.retrieval_result,
            score=candidate.score,
            overlap_terms=overlap_terms,
        )

    def _build_local_windows(
        self,
        candidate: SentenceEvidenceCandidate,
    ) -> list[str]:
        document_sentences = [
            normalized_sentence
            for sentence in split_sentences(candidate.retrieval_result.document.text)
            if len(normalized_sentence := normalize_sentence(sentence))
            >= self._min_sentence_chars
        ]
        anchor_indices = _find_anchor_indices(
            candidate_sentence=candidate.sentence,
            document_sentences=document_sentences,
        )
        if not anchor_indices:
            return []

        windows = []
        seen_windows = set()
        for anchor_index in anchor_indices:
            for window in _build_anchor_windows(
                sentences=document_sentences,
                anchor_index=anchor_index,
                max_window_sentences=self._max_window_sentences,
            ):
                if window in seen_windows:
                    continue
                seen_windows.add(window)
                windows.append(window)
        return windows


def _find_anchor_indices(
    candidate_sentence: str,
    document_sentences: list[str],
) -> list[int]:
    candidate_sentences = [
        normalize_sentence(sentence)
        for sentence in split_sentences(candidate_sentence)
        if normalize_sentence(sentence)
    ]
    if not candidate_sentences:
        return []

    exact_matches = [
        index
        for index, document_sentence in enumerate(document_sentences)
        if document_sentence in candidate_sentences
    ]
    if exact_matches:
        return exact_matches

    normalized_candidate = normalize_sentence(candidate_sentence)
    containment_matches = [
        index
        for index, document_sentence in enumerate(document_sentences)
        if document_sentence in normalized_candidate
        or normalized_candidate in document_sentence
    ]
    if containment_matches:
        return containment_matches

    candidate_terms = _content_terms(tokenize_text(normalized_candidate))
    if not candidate_terms:
        return []

    scored_indices = [
        (
            _overlap_ratio(candidate_terms, _content_terms(tokenize_text(sentence))),
            index,
        )
        for index, sentence in enumerate(document_sentences)
    ]
    best_score, best_index = max(scored_indices, default=(0.0, -1))
    if best_score < 0.6 or best_index < 0:
        return []
    return [best_index]


def _build_anchor_windows(
    sentences: list[str],
    anchor_index: int,
    max_window_sentences: int,
) -> list[str]:
    windows = []
    min_start = max(0, anchor_index - max_window_sentences + 1)
    max_end = min(len(sentences), anchor_index + max_window_sentences)
    for start_index in range(min_start, anchor_index + 1):
        for end_index in range(anchor_index + 1, max_end + 1):
            if end_index - start_index > max_window_sentences:
                continue
            windows.append(" ".join(sentences[start_index:end_index]))
    return windows


def _score_local_window(
    window: str,
    question_terms: set[str],
    anchor_terms: set[str],
) -> float:
    window_terms = _content_terms(tokenize_text(window))
    query_overlap = len(question_terms & window_terms)
    anchor_coverage = _overlap_ratio(anchor_terms, window_terms)
    answer_signal = _answer_signal_score(window)
    compactness = _compactness_score(window)
    noise_penalty = _local_noise_penalty(window)
    return (
        query_overlap
        + 1.5 * anchor_coverage
        + answer_signal
        + compactness
        - noise_penalty
    )


def _answer_signal_score(text: str) -> float:
    normalized = text.lower()
    score = 0.0
    if re.search(r"\b(resolving the problem|resolution|solution|answer)\b", normalized):
        score += 2.0
    if re.search(r"\b(workaround|fix|corrective action|local fix)\b", normalized):
        score += 1.2
    if re.search(
        r"\b(install|upgrade|configure|restart|set|enable|disable|apply|run|use)\b",
        normalized,
    ):
        score += 0.8
    if re.search(r"\b(required|must|should|recommended|supported)\b", normalized):
        score += 0.4
    return score


def _compactness_score(text: str) -> float:
    token_count = len(tokenize_text(text))
    sentence_count = len(split_sentences(text))
    if sentence_count == 1:
        return 0.4
    if sentence_count == 2 and token_count <= 80:
        return 0.3
    if sentence_count == 3 and token_count <= 100:
        return 0.1
    return -0.2


def _local_noise_penalty(text: str) -> float:
    token_count = len(tokenize_text(text))
    penalty = 0.0
    if token_count > 100:
        penalty += 0.8
    if token_count > 140:
        penalty += 1.2
    if re.search(r"\b(trace|stack|dump|exception|heapdump|javacore)\b", text.lower()):
        penalty += 0.4
    return penalty


def _content_terms(tokens: list[str]) -> set[str]:
    return {
        token
        for token in tokens
        if token not in _STOPWORDS and len(token) > 1
    }


def _overlap_ratio(source_terms: set[str], target_terms: set[str]) -> float:
    if not source_terms:
        return 0.0
    return len(source_terms & target_terms) / len(source_terms)


_STOPWORDS = {
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
