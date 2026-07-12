from collections.abc import Sequence

import numpy as np

from ts_rag_agent.application.retrieval_evaluation import evaluate_retrieval
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.infrastructure.dense_retriever import DenseRetriever, build_document_texts


class FakeEmbeddingModel:
    """测试用的确定性 embedding 模型。"""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self._encode_one(text) for text in texts], dtype=np.float32)

    def _encode_one(self, text: str) -> list[float]:
        normalized = text.lower()
        return [
            float("service" in normalized or "restart" in normalized),
            float("database" in normalized or "driver" in normalized),
            float("password" in normalized),
        ]


def test_dense_search_returns_most_similar_document():
    retriever = DenseRetriever(FakeEmbeddingModel())
    retriever.fit(
        [
            PrimeQADocument(id="doc-a", title="Restart service A", text="Restart service A."),
            PrimeQADocument(id="doc-b", title="Database driver", text="Install the driver."),
            PrimeQADocument(id="doc-c", title="Password policy", text="Rotate passwords."),
        ]
    )

    results = retriever.search("service restart problem", top_k=2)

    assert results[0].document.id == "doc-a"
    assert results[0].rank == 1
    assert results[0].score > results[1].score


def test_dense_evaluate_retrieval_reuses_common_metrics():
    retriever = DenseRetriever(FakeEmbeddingModel())
    retriever.fit(
        [
            PrimeQADocument(id="doc-a", title="Restart service A", text="Restart service A."),
            PrimeQADocument(id="doc-b", title="Database driver", text="Install the driver."),
        ]
    )
    questions = [
        PrimeQAQuestion(
            id="q1",
            title="How do I restart the service?",
            text="",
            answer="Restart service A.",
            answerable=True,
            answer_doc_id="doc-a",
            doc_ids=["doc-a"],
        ),
        PrimeQAQuestion(
            id="q2",
            title="What is the private password?",
            text="",
            answer="",
            answerable=False,
            answer_doc_id=None,
            doc_ids=[],
        ),
    ]

    metrics = evaluate_retrieval(questions, retriever, top_k_values=(1,))

    assert metrics.total_questions == 2
    assert metrics.evaluated_questions == 1
    assert metrics.hit_at_k == {1: 1.0}
    assert metrics.mrr == 1.0


def test_fit_embeddings_validates_shape():
    retriever = DenseRetriever(FakeEmbeddingModel())
    documents = [PrimeQADocument(id="doc-a", title="A", text="A")]

    try:
        retriever.fit_embeddings(documents, np.asarray([1.0, 0.0], dtype=np.float32))
    except ValueError as exc:
        assert "2D" in str(exc)
    else:
        raise AssertionError("fit_embeddings should reject a 1D embedding vector")


def test_build_document_texts_truncates_long_documents():
    documents = [PrimeQADocument(id="doc-a", title="Title", text="x" * 100)]

    texts = build_document_texts(documents, document_text_max_chars=12)

    assert texts == ["Title\n\nxxxxx"]


def test_build_document_texts_applies_document_prefix():
    documents = [PrimeQADocument(id="doc-a", title="Title", text="Body")]

    texts = build_document_texts(
        documents,
        document_text_max_chars=30,
        document_prefix="passage: ",
    )

    assert texts == ["passage: Title\n\nBody"]
