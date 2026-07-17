import math
from collections import Counter, defaultdict

import pytest

from ts_rag_agent.application.retrieval_evaluation import evaluate_retrieval
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQADocumentSection, PrimeQAQuestion
from ts_rag_agent.infrastructure.bm25_retriever import tokenize_text
from ts_rag_agent.infrastructure.section_bm25_retriever import SectionBM25Retriever


def test_section_bm25_returns_parent_document():
    documents = [
        PrimeQADocument(id="doc-a", title="Service guide", text="Full service guide."),
        PrimeQADocument(id="doc-b", title="Database guide", text="Full database guide."),
    ]
    sections = {
        "doc-a": [
            PrimeQADocumentSection(
                document_id="doc-a",
                section_id="restart",
                text="Restart service A from the control panel.",
            )
        ],
        "doc-b": [
            PrimeQADocumentSection(
                document_id="doc-b",
                section_id="driver",
                text="Install database driver packages.",
            )
        ],
    }
    retriever = SectionBM25Retriever()
    retriever.fit(documents, sections)

    results = retriever.search("restart service A", top_k=1)

    assert results[0].document.id == "doc-a"


def test_section_bm25_can_use_common_evaluator():
    documents = [
        PrimeQADocument(id="doc-a", title="Service guide", text="Full service guide."),
        PrimeQADocument(id="doc-b", title="Database guide", text="Full database guide."),
    ]
    sections = {
        "doc-a": [
            PrimeQADocumentSection(
                document_id="doc-a",
                section_id="restart",
                text="Restart service A from the control panel.",
            )
        ],
        "doc-b": [
            PrimeQADocumentSection(
                document_id="doc-b",
                section_id="driver",
                text="Install database driver packages.",
            )
        ],
    }
    questions = [
        PrimeQAQuestion(
            id="q1",
            title="How do I restart service A?",
            text="",
            answer="Restart service A.",
            answerable=True,
            answer_doc_id="doc-a",
            doc_ids=["doc-a"],
        )
    ]
    retriever = SectionBM25Retriever()
    retriever.fit(documents, sections)

    metrics = evaluate_retrieval(questions, retriever, top_k_values=(1,))

    assert metrics.hit_at_k == {1: 1.0}


def test_vectorized_section_bm25_matches_scalar_rollup_formula():
    documents = [
        PrimeQADocument(id="doc-c", title="Adapter guide", text="unused"),
        PrimeQADocument(id="doc-a", title="Adapter guide", text="unused"),
        PrimeQADocument(id="doc-b", title="Database guide", text="unused"),
    ]
    sections = {
        "doc-c": [
            PrimeQADocumentSection(
                document_id="doc-c",
                section_id="restart",
                text="adapter token restart restart",
            ),
            PrimeQADocumentSection(
                document_id="doc-c",
                section_id="install",
                text="adapter package install",
            ),
        ],
        "doc-a": [
            PrimeQADocumentSection(
                document_id="doc-a",
                section_id="restart",
                text="adapter token restart service",
            )
        ],
        "doc-b": [
            PrimeQADocumentSection(
                document_id="doc-b",
                section_id="driver",
                text="database driver package",
            )
        ],
    }
    retriever = SectionBM25Retriever(k1=1.5, b=0.75)
    retriever.fit(documents, sections)

    for query in ("adapter token restart", "adapter adapter", "database package"):
        actual = retriever.search(query, top_k=3)
        full_sort_reference = retriever.search_full_sort_reference(query, top_k=3)
        expected = _scalar_section_bm25_search(
            documents,
            sections,
            query=query,
            top_k=3,
            k1=1.5,
            b=0.75,
        )

        assert [result.document.id for result in actual] == [
            result.document.id for result in full_sort_reference
        ]
        assert [result.score for result in actual] == [
            result.score for result in full_sort_reference
        ]
        assert [result.document.id for result in actual] == [row[0] for row in expected]
        assert [result.score for result in actual] == pytest.approx(
            [row[1] for row in expected],
            rel=1e-12,
            abs=1e-12,
        )


def _scalar_section_bm25_search(
    documents: list[PrimeQADocument],
    sections_by_document: dict[str, list[PrimeQADocumentSection]],
    *,
    query: str,
    top_k: int,
    k1: float,
    b: float,
) -> list[tuple[str, float]]:
    document_by_id = {document.id: document for document in documents}
    sections = [section for document in documents for section in sections_by_document[document.id]]
    section_tokens = [
        tokenize_text(
            f"{document_by_id[section.document_id].title}\n\n{section.section_id}\n\n{section.text}"
        )
        for section in sections
    ]
    section_count = len(sections)
    average_length = sum(map(len, section_tokens)) / section_count
    document_frequency = Counter(term for tokens in section_tokens for term in set(tokens))
    section_scores: dict[int, float] = defaultdict(float)
    for term in tokenize_text(query):
        frequency = document_frequency.get(term)
        if frequency is None:
            continue
        idf = math.log(1 + (section_count - frequency + 0.5) / (frequency + 0.5))
        for index, tokens in enumerate(section_tokens):
            term_frequency = tokens.count(term)
            if term_frequency == 0:
                continue
            length_normalizer = 1 - b + b * len(tokens) / average_length
            section_scores[index] += (
                idf * (term_frequency * (k1 + 1)) / (term_frequency + k1 * length_normalizer)
            )
    document_scores: dict[str, float] = {}
    for section_index, score in section_scores.items():
        document_id = sections[section_index].document_id
        document_scores[document_id] = max(score, document_scores.get(document_id, score))
    ranked = sorted(document_scores, key=lambda doc_id: (-document_scores[doc_id], doc_id))
    return [(doc_id, document_scores[doc_id]) for doc_id in ranked[:top_k]]
