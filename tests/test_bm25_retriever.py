from ts_rag_agent.application.retrieval_evaluation import evaluate_retrieval
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.infrastructure.bm25_retriever import (
    BM25Retriever,
    tokenize_text,
)


def test_bm25_search_returns_most_relevant_document():
    retriever = BM25Retriever()
    retriever.fit(
        [
            PrimeQADocument(
                id="doc-a",
                title="Restart service A",
                text="Use the control panel to restart service A after an upgrade.",
            ),
            PrimeQADocument(
                id="doc-b",
                title="Service upgrade notes",
                text="Check service logs after upgrade before calling support.",
            ),
            PrimeQADocument(
                id="doc-c",
                title="Password policy",
                text="Rotate administrator passwords every 90 days.",
            ),
        ]
    )

    results = retriever.search("service A stopped after upgrade", top_k=2)

    assert [result.document.id for result in results] == ["doc-a", "doc-b"]
    assert results[0].rank == 1
    assert results[0].score > results[1].score


def test_evaluate_retrieval_computes_hit_at_k_and_mrr():
    retriever = BM25Retriever()
    retriever.fit(
        [
            PrimeQADocument(id="doc-a", title="Restart service A", text="Restart service A."),
            PrimeQADocument(
                id="doc-b",
                title="Database driver",
                text="Install the database driver.",
            ),
            PrimeQADocument(id="doc-c", title="Password policy", text="Rotate passwords."),
        ]
    )
    questions = [
        PrimeQAQuestion(
            id="q1",
            title="How do I restart service A?",
            text="The service stopped.",
            answer="Restart service A.",
            answerable=True,
            answer_doc_id="doc-a",
            doc_ids=["doc-a"],
        ),
        PrimeQAQuestion(
            id="q2",
            title="How do I fix the database driver?",
            text="The database connection fails.",
            answer="Install the database driver.",
            answerable=True,
            answer_doc_id="doc-b",
            doc_ids=["doc-b"],
        ),
        PrimeQAQuestion(
            id="q3",
            title="What is the private admin password?",
            text="",
            answer="",
            answerable=False,
            answer_doc_id=None,
            doc_ids=[],
        ),
    ]

    metrics = evaluate_retrieval(questions, retriever, top_k_values=(1, 2))

    assert metrics.total_questions == 3
    assert metrics.evaluated_questions == 2
    assert metrics.hit_at_k == {1: 1.0, 2: 1.0}
    assert metrics.mrr == 1.0


def test_search_requires_fit():
    retriever = BM25Retriever()

    try:
        retriever.search("service")
    except RuntimeError as exc:
        assert "fit" in str(exc)
    else:
        raise AssertionError("search should fail before fit")


def test_tokenize_text_keeps_common_technical_tokens():
    tokens = tokenize_text("WebSphere MQ 7.0.1.5 supports C++ and app-scan_source.")

    assert "websphere" in tokens
    assert "mq" in tokens
    assert "7.0.1.5" in tokens
    assert "c++" in tokens
    assert "app-scan_source" in tokens
