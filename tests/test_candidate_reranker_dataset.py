from collections.abc import Sequence

from ts_rag_agent.application.candidate_reranker_dataset import (
    build_candidate_reranker_dataset,
    build_candidate_rows_for_question,
    candidate_reranker_dataset_build_to_dict,
)
from ts_rag_agent.application.evidence_selection import SentenceEvidenceCandidate
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


def test_candidate_rows_separate_runtime_features_from_gold_labels():
    question = PrimeQAQuestion(
        id="q1",
        title="Profile blank panel",
        text="How do I fix the blank profile panel?",
        answer="Install the missing libraries and restart the profile tool.",
        answerable=True,
        answer_doc_id="gold",
        doc_ids=["gold"],
    )
    gold_result = RetrievalResult(
        document=PrimeQADocument(
            id="gold",
            title="Profile blank panel",
            text="",
        ),
        score=10.0,
        rank=1,
    )
    distractor_result = RetrievalResult(
        document=PrimeQADocument(
            id="distractor",
            title="Profile tracing",
            text="",
        ),
        score=9.0,
        rank=2,
    )
    candidates = [
        _candidate(
            "RESOLUTION Install the missing libraries and restart the profile tool.",
            gold_result,
            score=20.0,
        ),
        _candidate(
            "SYMPTOM The profile panel is blank and trace files are generated.",
            distractor_result,
            score=15.0,
        ),
    ]

    rows = build_candidate_rows_for_question(
        split="dev",
        question=question,
        candidates=candidates,
        selector_name="test_selector",
    )

    assert rows[0].gold_labels["is_gold_document"] is True
    assert rows[0].gold_labels["is_best_candidate_for_question"] is True
    assert rows[1].runtime_features["has_problem_heading"] is True
    assert "candidate_token_f1" not in rows[0].runtime_features
    assert "is_gold_document" not in rows[0].runtime_features
    assert rows[0].metadata["document_id"] == "gold"


def test_build_candidate_reranker_dataset_summarizes_oracle_gain():
    question = PrimeQAQuestion(
        id="q1",
        title="Profile blank panel",
        text="How do I fix the blank profile panel?",
        answer="Restart the profile tool.",
        answerable=True,
        answer_doc_id="gold",
        doc_ids=["gold"],
    )
    retrieval_result = RetrievalResult(
        document=PrimeQADocument(
            id="gold",
            title="Profile blank panel",
            text="",
        ),
        score=10.0,
        rank=1,
    )
    selector = _StaticSelector(
        candidates=[
            _candidate("Install the missing libraries.", retrieval_result, score=10.0),
            _candidate("Restart the profile tool.", retrieval_result, score=9.0),
        ]
    )

    build = build_candidate_reranker_dataset(
        split_questions={"dev": [question]},
        search_fn=lambda question, top_k: [retrieval_result],
        evidence_selector=selector,
        retrieval_top_k=5,
        candidate_limit=10,
        min_candidate_score=2.0,
    )
    build_dict = candidate_reranker_dataset_build_to_dict(build)

    assert build.summary.total_questions == 1
    assert build.summary.total_rows == 2
    assert build.summary.average_oracle_gain_vs_top_candidate > 0
    assert build.question_summaries[0].best_candidate_rank == 2
    assert build_dict["feature_contract"]["gold_labels"].startswith("Offline labels")


def test_build_candidate_reranker_dataset_filters_by_min_score_and_limit():
    question = PrimeQAQuestion(
        id="q1",
        title="Profile blank panel",
        text="",
        answer="Restart the profile tool.",
        answerable=True,
        answer_doc_id="gold",
        doc_ids=["gold"],
    )
    retrieval_result = RetrievalResult(
        document=PrimeQADocument(
            id="gold",
            title="Profile blank panel",
            text="",
        ),
        score=10.0,
        rank=1,
    )
    selector = _StaticSelector(
        candidates=[
            _candidate("Restart the profile tool.", retrieval_result, score=5.0),
            _candidate("Install missing libraries.", retrieval_result, score=4.0),
            _candidate("Low score candidate.", retrieval_result, score=1.0),
        ]
    )

    build = build_candidate_reranker_dataset(
        split_questions={"dev": [question]},
        search_fn=lambda question, top_k: [retrieval_result],
        evidence_selector=selector,
        retrieval_top_k=5,
        candidate_limit=1,
        min_candidate_score=2.0,
    )

    assert build.summary.total_rows == 1
    assert build.rows[0].candidate_rank == 1


class _StaticSelector:
    name = "static_selector"

    def __init__(self, candidates: list[SentenceEvidenceCandidate]) -> None:
        self._candidates = candidates

    def rank_sentence_candidates(
        self,
        question: PrimeQAQuestion,
        retrieval_results: Sequence[RetrievalResult],
    ) -> list[SentenceEvidenceCandidate]:
        return self._candidates


def _candidate(
    sentence: str,
    retrieval_result: RetrievalResult,
    score: float,
) -> SentenceEvidenceCandidate:
    return SentenceEvidenceCandidate(
        sentence=sentence,
        retrieval_result=retrieval_result,
        score=score,
        overlap_terms=(),
    )
