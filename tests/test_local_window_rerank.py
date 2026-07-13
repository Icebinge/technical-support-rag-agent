from collections.abc import Sequence

from ts_rag_agent.application.evidence_selection import (
    SentenceEvidenceCandidate,
    create_sentence_evidence_selector,
    trace_selector_route,
)
from ts_rag_agent.application.local_window_rerank import (
    LocalWindowRerankEvidenceSelector,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


def test_local_window_rerank_expands_candidate_inside_local_document_window():
    question = PrimeQAQuestion(
        id="q1",
        title="Profile panel blank",
        text="Need restart profile tool after opening a blank panel.",
        answer="Install the missing adwaita libraries and restart the profile tool.",
        answerable=True,
        answer_doc_id="gold",
        doc_ids=["gold"],
    )
    document = PrimeQADocument(
        id="gold",
        title="Profile panel blank",
        text=(
            "The profile tool opens to a blank panel. "
            "RESOLUTION Install the missing adwaita libraries. "
            "Restart the profile tool."
        ),
    )
    retrieval_result = RetrievalResult(document=document, score=10.0, rank=1)
    selector = LocalWindowRerankEvidenceSelector(
        base_selector=_StaticSelector(
            sentence="RESOLUTION Install the missing adwaita libraries.",
            retrieval_result=retrieval_result,
        ),
        min_sentence_chars=8,
        selected_candidate_limit=1,
        max_window_sentences=3,
    )

    candidates = selector.rank_sentence_candidates(question, [retrieval_result])

    assert "Install the missing adwaita libraries" in candidates[0].sentence
    assert "Restart the profile tool" in candidates[0].sentence
    assert candidates[0].retrieval_result.document.id == "gold"
    assert candidates[0].score == 10.0


def test_local_window_rerank_keeps_non_target_route_on_base_selector():
    question = PrimeQAQuestion(
        id="q1",
        title="How do I restart profile tool?",
        text="",
        answer="Restart the profile tool.",
        answerable=True,
        answer_doc_id="gold",
        doc_ids=["gold"],
    )
    document = PrimeQADocument(
        id="gold",
        title="Profile restart",
        text=(
            "RESOLUTION Install the missing adwaita libraries. "
            "Restart the profile tool."
        ),
    )
    retrieval_result = RetrievalResult(document=document, score=10.0, rank=1)
    selector = LocalWindowRerankEvidenceSelector(
        base_selector=_StaticSelector(
            sentence="RESOLUTION Install the missing adwaita libraries.",
            retrieval_result=retrieval_result,
        ),
        min_sentence_chars=8,
        selected_candidate_limit=1,
        max_window_sentences=2,
    )

    candidates = selector.rank_sentence_candidates(question, [retrieval_result])

    assert candidates[0].sentence == "RESOLUTION Install the missing adwaita libraries."


def test_selector_factory_creates_local_window_rerank_selector():
    selector = create_sentence_evidence_selector(
        selector_name="local-window-rerank",
        min_sentence_chars=8,
        max_candidates_per_document=3,
    )

    assert selector.name.startswith("local_window_rerank_hybrid_routing")


def test_trace_selector_route_explains_local_window_rerank_route():
    question = PrimeQAQuestion(
        id="q1",
        title="Profile panel blank",
        text="Need restart profile tool.",
        answer="",
        answerable=True,
        answer_doc_id="gold",
        doc_ids=["gold"],
    )

    trace = trace_selector_route(
        question,
        "local_window_rerank_hybrid_routing_answer_aware_mcpd3_section_span_mcpd1",
    )

    assert trace.question_route == "other"
    assert trace.selected_selector_name == "local_window_rerank"
    assert "local-window" in trace.route_reason


class _StaticSelector:
    name = "static_selector"

    def __init__(
        self,
        sentence: str,
        retrieval_result: RetrievalResult,
    ) -> None:
        self._sentence = sentence
        self._retrieval_result = retrieval_result

    def rank_sentence_candidates(
        self,
        question: PrimeQAQuestion,
        retrieval_results: Sequence[RetrievalResult],
    ) -> list[SentenceEvidenceCandidate]:
        return [
            SentenceEvidenceCandidate(
                sentence=self._sentence,
                retrieval_result=self._retrieval_result,
                score=10.0,
                overlap_terms=(),
            )
        ]
