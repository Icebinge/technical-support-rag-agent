from ts_rag_agent.application.retrieval_evaluation import evaluate_retrieval
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQADocumentSection, PrimeQAQuestion
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
