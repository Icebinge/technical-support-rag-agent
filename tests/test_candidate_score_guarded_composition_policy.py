from ts_rag_agent.application.candidate_score_guarded_composition_policy import (
    CandidateScoreGuardedRerankerCompositionPolicy,
)
from ts_rag_agent.application.evidence_selection import SentenceEvidenceCandidate
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


def test_candidate_score_guarded_reranker_rewrites_leading_candidate_when_score_passes():
    policy = CandidateScoreGuardedRerankerCompositionPolicy(
        scorer=_FixedScorer([0.2, 0.95, 0.1]),
        selector_name="test_selector",
    )
    candidates = [
        _candidate("doc-top", score=100.0, sentence="Top ranked baseline answer."),
        _candidate("doc-selected", score=80.0, sentence="Selected stronger answer."),
        _candidate("doc-third", score=70.0, sentence="Third answer."),
    ]

    decision = policy.select(
        question=_question(),
        candidates=candidates,
        max_sentences=3,
    )

    assert decision.selected_candidates == [candidates[1], candidates[0], candidates[2]]
    assert decision.reason == "candidate_score_gte_60_accepted"
    assert policy.last_trace is not None
    assert policy.last_trace.action == "replace_with_model_candidate"
    assert policy.last_trace.selected_candidate_rank == 2


def test_candidate_score_guarded_reranker_blocks_low_score_replacement():
    policy = CandidateScoreGuardedRerankerCompositionPolicy(
        scorer=_FixedScorer([0.2, 0.95, 0.1]),
        selector_name="test_selector",
    )
    candidates = [
        _candidate("doc-top", score=100.0, sentence="Top ranked baseline answer."),
        _candidate("doc-low", score=59.9, sentence="Low score model pick."),
        _candidate("doc-third", score=70.0, sentence="Third answer."),
    ]

    decision = policy.select(
        question=_question(),
        candidates=candidates,
        max_sentences=3,
    )

    assert decision.selected_candidates == candidates
    assert decision.reason == "candidate_score_gte_60_blocked"
    assert policy.last_trace is not None
    assert policy.last_trace.action == "keep_top_candidate"
    assert policy.last_trace.selected_candidate_score == 59.9


def test_candidate_score_guarded_reranker_blocks_rank_contained_replacement():
    policy = CandidateScoreGuardedRerankerCompositionPolicy(
        scorer=_FixedScorer([0.2, 0.95, 0.1]),
        selector_name="test_selector",
        rank_contained_max_retrieval_rank=3,
    )
    candidates = [
        _candidate(
            "doc-top",
            score=100.0,
            sentence="Top ranked baseline answer.",
            retrieval_rank=4,
        ),
        _candidate(
            "doc-selected",
            score=80.0,
            sentence="Selected stronger answer.",
            retrieval_rank=1,
        ),
        _candidate("doc-third", score=70.0, sentence="Third answer.", retrieval_rank=2),
    ]

    decision = policy.select(
        question=_question(),
        candidates=candidates,
        max_sentences=3,
    )

    assert policy.name == "candidate_score_gte_60_rank_contained_guarded_reranker"
    assert decision.selected_candidates == candidates
    assert decision.reason == "selected_citation_rank_exceeds_limit"
    assert policy.last_trace is not None
    assert policy.last_trace.action == "keep_top_candidate"
    assert policy.last_trace.proposed_worst_retrieval_rank == 4
    assert policy.last_trace.rank_contained_max_retrieval_rank == 3


class _FixedScorer:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def score(self, examples) -> list[float]:
        return self._scores[: len(examples)]


def _question() -> PrimeQAQuestion:
    return PrimeQAQuestion(
        id="q1",
        title="How do I configure Product A?",
        text="I need the supported setup steps.",
        answer="Selected stronger answer.",
        answerable=True,
        answer_doc_id="doc-selected",
    )


def _candidate(
    document_id: str,
    score: float,
    sentence: str,
    retrieval_rank: int = 1,
) -> SentenceEvidenceCandidate:
    return SentenceEvidenceCandidate(
        sentence=sentence,
        retrieval_result=RetrievalResult(
            document=PrimeQADocument(
                id=document_id,
                title=f"Document {document_id}",
                text=sentence,
            ),
            score=100.0,
            rank=retrieval_rank,
        ),
        score=score,
        overlap_terms=("configure",),
    )
