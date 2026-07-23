from __future__ import annotations

import inspect

import pytest

from ts_rag_agent.application.candidate_reranker_cv import CandidateRerankerExample
from ts_rag_agent.application.citation_aware_composition_policy import (
    CitationAwareCompositionPolicy,
    CitationAwareCompositionSpec,
    DualTargetCandidateModel,
    stage180_policy_specs,
)
from ts_rag_agent.application.evidence_selection import SentenceEvidenceCandidate
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQARuntimeQuery
from ts_rag_agent.domain.retrieval import RetrievalResult


def test_stage180_policy_family_is_frozen_and_complete() -> None:
    specs = stage180_policy_specs()

    assert len(specs) == 14
    assert len({spec.policy_id for spec in specs}) == 14
    assert sum(spec.family == "dual_target" for spec in specs) == 6
    assert sum(spec.family != "dual_target" for spec in specs) == 8


def test_score_rank_policy_enforces_document_diversity() -> None:
    question = PrimeQARuntimeQuery(id="q", title="Fix service", text="How do I fix it?")
    candidates = [
        _candidate("a", rank=2, score=12.0, sentence="First answer sentence."),
        _candidate("a", rank=2, score=11.0, sentence="Second answer sentence."),
        _candidate("b", rank=1, score=10.0, sentence="Third answer sentence."),
    ]
    policy = CitationAwareCompositionPolicy(
        spec=CitationAwareCompositionSpec("test", "score_rank", 1, rank_power=0.0)
    )

    decision = policy.select(question, candidates, 3)

    assert [row.retrieval_result.document.id for row in decision.selected_candidates] == [
        "a",
        "b",
    ]


def test_context_rank_coverage_selects_top_context_documents() -> None:
    question = PrimeQARuntimeQuery(id="q", title="Fix service", text="How do I fix it?")
    candidates = [
        _candidate(
            str(rank),
            rank=rank,
            score=20.0 - rank,
            sentence=f"Candidate sentence for document {rank} with enough text.",
        )
        for rank in (4, 2, 1, 3)
    ]
    policy = CitationAwareCompositionPolicy(
        spec=CitationAwareCompositionSpec(
            "coverage",
            "context_rank_coverage",
            1,
        )
    )

    decision = policy.select(question, candidates, 3)

    assert [row.retrieval_result.rank for row in decision.selected_candidates] == [1, 2, 3]


def test_dual_target_model_rejects_single_class_citation_training() -> None:
    example = CandidateRerankerExample(
        split="train",
        question_id="q",
        candidate_id="c",
        candidate_rank=1,
        question_route="other",
        runtime_features={"candidate_score": 1.0},
        candidate_token_f1=0.0,
        is_best_candidate_for_question=True,
        is_gold_document=False,
    )

    with pytest.raises(ValueError, match="positive and negative"):
        DualTargetCandidateModel().fit([example])


def test_dual_target_model_fits_and_scores_runtime_features() -> None:
    examples = [
        CandidateRerankerExample(
            split="train",
            question_id=f"q{index}",
            candidate_id=f"c{index}",
            candidate_rank=index + 1,
            question_route="how_to_or_lookup",
            runtime_features={
                "candidate_score": float(index + 1),
                "retrieval_rank": index + 1,
                "question_route": "how_to_or_lookup",
            },
            candidate_token_f1=float(index % 2),
            is_best_candidate_for_question=index % 2 == 1,
            is_gold_document=index % 2 == 1,
        )
        for index in range(4)
    ]
    model = DualTargetCandidateModel()
    model.fit(examples)
    question = PrimeQARuntimeQuery(id="runtime", title="Fix service", text="How do I fix it?")
    candidates = [_candidate("a", rank=1, score=12.0, sentence="Use this fix command.")]

    scores = model.score_runtime(
        question=question,
        candidates=candidates,
        selector_name="bm25_sentence",
    )

    assert len(scores) == 1
    assert all(0 <= value <= 1 for value in scores[0])


def test_runtime_policy_source_does_not_read_gold_fields() -> None:
    source = inspect.getsource(CitationAwareCompositionPolicy.select)

    assert "answer_doc_id" not in source
    assert "question.answer" not in source
    assert "is_gold_document" not in source


def _candidate(
    document_id: str,
    *,
    rank: int,
    score: float,
    sentence: str,
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
            rank=rank,
        ),
        score=score,
        overlap_terms=("answer",),
    )
