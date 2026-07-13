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


@dataclass(frozen=True)
class SelectorRouteTrace:
    """Explain which selector was used for one question."""

    question_route: str
    selected_selector_name: str
    route_reason: str


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
        score *= _sentence_noise_penalty(row.scoring_text or row.sentence)
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


class AnswerAwareBM25SentenceEvidenceSelector(BM25SentenceEvidenceSelector):
    """BM25 selector with lightweight answer-section and noise structure signals."""

    name = "answer_aware_bm25_sentence"

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
                if not overlap_terms and not _has_answer_section_signal(normalized_sentence):
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
        candidate = super()._score_sentence_row(
            row=row,
            query_terms=query_terms,
            idf_by_term=idf_by_term,
            avg_sentence_length=avg_sentence_length,
        )
        scoring_text = row.scoring_text or row.sentence
        adjusted_score = candidate.score + _answer_section_bonus(scoring_text)
        adjusted_score *= _answer_section_multiplier(scoring_text)
        adjusted_score *= _problem_statement_penalty(scoring_text)
        return SentenceEvidenceCandidate(
            sentence=candidate.sentence,
            retrieval_result=candidate.retrieval_result,
            score=adjusted_score,
            overlap_terms=candidate.overlap_terms,
        )


class SectionSpanBM25SentenceEvidenceSelector(AnswerAwareBM25SentenceEvidenceSelector):
    """Answer-aware selector that scores windows inside document sections."""

    name = "section_span_bm25_sentence"

    def __init__(
        self,
        min_sentence_chars: int = 24,
        max_candidates_per_document: int = 1,
        k1: float = 1.2,
        b: float = 0.35,
        score_scale: float = 2.5,
        max_window_sentences: int = 3,
    ) -> None:
        if max_window_sentences <= 0:
            raise ValueError("max_window_sentences must be positive")

        super().__init__(
            min_sentence_chars=min_sentence_chars,
            max_candidates_per_document=max_candidates_per_document,
            k1=k1,
            b=b,
            score_scale=score_scale,
        )
        self._max_window_sentences = max_window_sentences

    def _collect_sentence_rows(
        self,
        query_terms: set[str],
        retrieval_results: Sequence[RetrievalResult],
    ) -> list[_SentenceRow]:
        rows = []
        seen_spans = set()

        for retrieval_result in retrieval_results:
            sections = _split_document_sections(retrieval_result.document.text)
            for section in sections:
                sentences = [
                    normalize_sentence(sentence)
                    for sentence in split_sentences(section.text)
                    if len(normalize_sentence(sentence)) >= self._min_sentence_chars
                ]
                for span in _build_sentence_windows(sentences, self._max_window_sentences):
                    if span in seen_spans:
                        continue

                    scoring_text = _build_section_scoring_text(
                        heading=section.heading,
                        span=span,
                    )
                    terms = _content_terms(tokenize_text(span))
                    overlap_terms = tuple(sorted(query_terms & terms))
                    if not overlap_terms and not _has_answer_section_signal(scoring_text):
                        continue

                    seen_spans.add(span)
                    rows.append(
                        _SentenceRow(
                            sentence=span,
                            retrieval_result=retrieval_result,
                            terms=terms,
                            overlap_terms=overlap_terms,
                            scoring_text=scoring_text,
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
        candidate = super()._score_sentence_row(
            row=row,
            query_terms=query_terms,
            idf_by_term=idf_by_term,
            avg_sentence_length=avg_sentence_length,
        )
        scoring_text = row.scoring_text or row.sentence
        adjusted_score = candidate.score + _section_span_answer_pattern_bonus(scoring_text)
        adjusted_score *= _section_span_background_penalty(scoring_text)
        adjusted_score *= _section_span_length_penalty(row.sentence)
        return SentenceEvidenceCandidate(
            sentence=candidate.sentence,
            retrieval_result=candidate.retrieval_result,
            score=adjusted_score,
            overlap_terms=candidate.overlap_terms,
        )


class HybridRoutingEvidenceSelector:
    """Route question types to the selector that currently handles them best."""

    def __init__(
        self,
        min_sentence_chars: int = 24,
        answer_aware_max_candidates_per_document: int = 3,
        section_span_max_candidates_per_document: int = 1,
    ) -> None:
        if answer_aware_max_candidates_per_document <= 0:
            raise ValueError("answer_aware_max_candidates_per_document must be positive")
        if section_span_max_candidates_per_document <= 0:
            raise ValueError("section_span_max_candidates_per_document must be positive")

        self._name = (
            "hybrid_routing_answer_aware_"
            f"mcpd{answer_aware_max_candidates_per_document}_"
            f"section_span_mcpd{section_span_max_candidates_per_document}"
        )
        self._answer_aware_selector = AnswerAwareBM25SentenceEvidenceSelector(
            min_sentence_chars=min_sentence_chars,
            max_candidates_per_document=answer_aware_max_candidates_per_document,
        )
        self._section_span_selector = SectionSpanBM25SentenceEvidenceSelector(
            min_sentence_chars=min_sentence_chars,
            max_candidates_per_document=section_span_max_candidates_per_document,
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
        """Route one question to the best known non-LLM selector."""

        route_trace = trace_selector_route(question, self.name)
        if route_trace.selected_selector_name == self._section_span_selector.name:
            return self._section_span_selector.rank_sentence_candidates(
                question,
                retrieval_results,
            )
        return self._answer_aware_selector.rank_sentence_candidates(
            question,
            retrieval_results,
        )


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
    if normalized_name in {
        "answer_aware",
        "answer_aware_bm25",
        "answer_aware_bm25_sentence",
    }:
        return AnswerAwareBM25SentenceEvidenceSelector(
            min_sentence_chars=min_sentence_chars,
            max_candidates_per_document=max_candidates_per_document,
        )
    if normalized_name in {
        "section_span",
        "section_span_bm25",
        "section_span_bm25_sentence",
    }:
        return SectionSpanBM25SentenceEvidenceSelector(
            min_sentence_chars=min_sentence_chars,
            max_candidates_per_document=max_candidates_per_document,
        )
    if normalized_name in {
        "hybrid",
        "hybrid_routing",
        "hybrid_selector",
        "routing",
    }:
        return HybridRoutingEvidenceSelector(
            min_sentence_chars=min_sentence_chars,
            answer_aware_max_candidates_per_document=max_candidates_per_document,
            section_span_max_candidates_per_document=1,
        )

    raise ValueError(
        "selector_name must be one of: overlap, overlap_sentence, bm25, "
        "bm25_sentence, answer_aware, answer_aware_bm25_sentence, section_span, "
        "hybrid_routing"
    )


@dataclass(frozen=True)
class _SentenceRow:
    sentence: str
    retrieval_result: RetrievalResult
    terms: set[str]
    overlap_terms: tuple[str, ...]
    scoring_text: str = ""


@dataclass(frozen=True)
class _DocumentSection:
    heading: str
    text: str


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


def _answer_section_multiplier(sentence: str) -> float:
    normalized = sentence.lower()
    multiplier = 1.0
    if re.search(r"\b(resolving the problem|resolution|solution|answer)\b", normalized):
        multiplier *= 2.5
    if re.search(r"\b(workaround|fix|corrective action|local fix)\b", normalized):
        multiplier *= 1.6
    if re.search(r"\b(cause|caused by|root cause)\b", normalized):
        multiplier *= 1.3
    if re.search(r"\b(install|upgrade|configure|set|enable|disable|use|apply)\b", normalized):
        multiplier *= 1.1
    return multiplier


def _answer_section_bonus(sentence: str) -> float:
    normalized = sentence.lower()
    if re.search(r"\b(resolving the problem|resolution|solution|answer)\b", normalized):
        return 25.0
    if re.search(r"\b(workaround|fix|corrective action|local fix)\b", normalized):
        return 12.0
    if re.search(r"\b(cause|caused by|root cause)\b", normalized):
        return 10.0
    return 0.0


def _has_answer_section_signal(sentence: str) -> bool:
    return _answer_section_bonus(sentence) > 0


def _problem_statement_penalty(sentence: str) -> float:
    normalized = sentence.lower()
    penalty = 1.0
    if "problem(abstract)" in normalized or re.search(
        r"\b(problem summary|symptom|question|environment|error description)\b",
        normalized,
    ):
        penalty *= 0.12
    if re.search(r"\b(diagnosing the problem|collecting data|steps to reproduce)\b", normalized):
        penalty *= 0.5
    if re.search(
        r"\b(trace|stack|dump|exception|javacore|heapdump|0section|1xhexc)\b",
        normalized,
    ):
        penalty *= 0.25
    if " null " in f" {normalized} ":
        penalty *= 0.6
    return penalty


SECTION_HEADING_PATTERN = re.compile(
    r"\b("
    r"problem\(abstract\)|problem summary|symptom|error description|question|"
    r"environment|summary|affected products and versions|remediation/fixes|"
    r"cause|answer|resolving the problem|resolution|solution|"
    r"workaround|fix|corrective action|local fix|diagnosing the problem|"
    r"collecting data|steps to reproduce"
    r")\b",
    re.IGNORECASE,
)


def _split_document_sections(text: str) -> list[_DocumentSection]:
    matches = list(SECTION_HEADING_PATTERN.finditer(text))
    if not matches:
        return [_DocumentSection(heading="", text=text)]

    sections = []
    if matches[0].start() > 0:
        prefix = text[: matches[0].start()].strip()
        if prefix:
            sections.append(_DocumentSection(heading="", text=prefix))

    for index, match in enumerate(matches):
        section_start = match.end()
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[section_start:section_end].strip()
        if section_text:
            sections.append(
                _DocumentSection(
                    heading=normalize_sentence(match.group(1)),
                    text=section_text,
                )
            )

    return sections


def _build_sentence_windows(sentences: list[str], max_window_sentences: int) -> list[str]:
    windows = []
    max_size = min(max_window_sentences, len(sentences))
    for window_size in range(1, max_size + 1):
        for start_index in range(0, len(sentences) - window_size + 1):
            windows.append(" ".join(sentences[start_index : start_index + window_size]))
    return windows


def _build_section_scoring_text(heading: str, span: str) -> str:
    if not heading:
        return span
    return f"{heading} {span}"


def _section_span_answer_pattern_bonus(text: str) -> float:
    normalized = text.lower()
    bonus = 0.0
    if re.search(r"\bcveid\b|\bcvss base score\b|\bcvss vector\b", normalized):
        bonus += 45.0
    if re.search(r"\b(description:|description)\b", normalized) and "cve" in normalized:
        bonus += 20.0
    if re.search(r"\b(restriction|limitation|known current limitation)\b", normalized):
        bonus += 22.0
    if re.search(r"\b(rfe|enhancement)\b", normalized):
        bonus += 16.0
    if re.search(r"\b(use|set|enable|disable|install|apply|code|configure)\b", normalized):
        bonus += 6.0
    return bonus


def _section_span_background_penalty(text: str) -> float:
    normalized = text.lower()
    penalty = 1.0
    if re.search(r"\bsummary\b", normalized):
        penalty *= 0.6
    if re.search(r"\baffected products and versions\b", normalized):
        penalty *= 0.55
    if re.search(r"\bremediation/fixes\b", normalized):
        penalty *= 0.8
    return penalty


def _section_span_length_penalty(text: str) -> float:
    token_count = len(tokenize_text(text))
    if token_count > 180:
        return 0.65
    if token_count > 120:
        return 0.8
    return 1.0


def classify_question_route(question: PrimeQAQuestion) -> str:
    """Classify a question for selector routing without using gold answers."""

    combined = question.full_question.lower()
    if any(token in combined for token in ("cve-", "cveid", "cvss", "security bulletin")):
        return "security_bulletin"
    if any(token in combined for token in ("limitation", "restriction", "not supported")):
        return "limitation_or_restriction"
    if any(token in combined for token in ("exception", "trace", "dump", "javacore", "error ")):
        return "error_or_log"
    if any(token in combined for token in ("install", "upgrade", "configure", "migration")):
        return "install_upgrade_config"
    if question.title.lower().startswith(("how ", "how do", "how can", "what is", "where ")):
        return "how_to_or_lookup"
    return "other"


def trace_selector_route(
    question: PrimeQAQuestion,
    selector_name: str,
) -> SelectorRouteTrace:
    """Trace selector routing without using gold answers."""

    question_route = classify_question_route(question)
    if selector_name.startswith("hybrid_routing"):
        if question_route in {"security_bulletin", "limitation_or_restriction"}:
            return SelectorRouteTrace(
                question_route=question_route,
                selected_selector_name="section_span_bm25_sentence",
                route_reason=f"{question_route} routed to section-span",
            )
        return SelectorRouteTrace(
            question_route=question_route,
            selected_selector_name="answer_aware_bm25_sentence",
            route_reason=f"{question_route} routed to answer-aware",
        )

    return SelectorRouteTrace(
        question_route=question_route,
        selected_selector_name=selector_name,
        route_reason="selector does not use routing",
    )
