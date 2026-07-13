from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ts_rag_agent.application.evidence_selection import (
    SentenceEvidenceCandidate,
    classify_question_route,
)
from ts_rag_agent.application.route_aware_composition_policy import (
    CompositionPolicyCandidate,
    RouteAwareCompositionPolicy,
)
from ts_rag_agent.domain.dataset import PrimeQAQuestion


@dataclass(frozen=True)
class AnswerCompositionDecision:
    """Final evidence sentence selection for one generated answer."""

    selected_candidates: list[SentenceEvidenceCandidate]
    question_route: str
    strategy: str
    reason: str


class AnswerCompositionPolicy(Protocol):
    """Chooses final answer evidence from already-ranked sentence candidates."""

    @property
    def name(self) -> str:
        """Stable policy name used in experiment reports."""

    def select(
        self,
        question: PrimeQAQuestion,
        candidates: Sequence[SentenceEvidenceCandidate],
        max_sentences: int,
    ) -> AnswerCompositionDecision:
        """Select final answer candidates without using gold labels."""


class TopKAnswerCompositionPolicy:
    """Current runtime behavior: keep the top scored candidates."""

    name = "top_k"

    def select(
        self,
        question: PrimeQAQuestion,
        candidates: Sequence[SentenceEvidenceCandidate],
        max_sentences: int,
    ) -> AnswerCompositionDecision:
        _validate_max_sentences(max_sentences)
        return AnswerCompositionDecision(
            selected_candidates=list(candidates[:max_sentences]),
            question_route=classify_question_route(question),
            strategy="top_k",
            reason=f"kept the top {max_sentences} ranked candidates",
        )


class RouteAwareAnswerCompositionPolicy:
    """Runtime adapter for the Stage 21 route-aware composition policy."""

    def __init__(
        self,
        route_policy: RouteAwareCompositionPolicy | None = None,
    ) -> None:
        self._route_policy = route_policy or RouteAwareCompositionPolicy()

    @property
    def name(self) -> str:
        """Stable policy name used in experiment reports."""

        return self._route_policy.name

    def select(
        self,
        question: PrimeQAQuestion,
        candidates: Sequence[SentenceEvidenceCandidate],
        max_sentences: int,
    ) -> AnswerCompositionDecision:
        _validate_max_sentences(max_sentences)
        capped_candidates = list(candidates[:max_sentences])
        question_route = classify_question_route(question)
        policy_pairs = [
            (_to_policy_candidate(candidate), candidate)
            for candidate in capped_candidates
        ]
        policy_candidates = [
            policy_candidate for policy_candidate, _runtime_candidate in policy_pairs
        ]
        selected_policy_candidates, strategy, reason = self._route_policy.select(
            question_route,
            policy_candidates,
        )
        runtime_candidates_by_policy_id = {
            id(policy_candidate): runtime_candidate
            for policy_candidate, runtime_candidate in policy_pairs
        }

        return AnswerCompositionDecision(
            selected_candidates=[
                runtime_candidates_by_policy_id[id(policy_candidate)]
                for policy_candidate in selected_policy_candidates
            ],
            question_route=question_route,
            strategy=strategy,
            reason=reason,
        )


def create_answer_composition_policy(policy_name: str) -> AnswerCompositionPolicy:
    """Create an answer-composition policy from a stable CLI name."""

    normalized_name = policy_name.strip().lower().replace("-", "_")
    if normalized_name in {"top_k", "top3", "default"}:
        return TopKAnswerCompositionPolicy()
    if normalized_name in {
        "route_aware",
        "route_aware_top1_direct_otherwise_top3",
        RouteAwareCompositionPolicy.name,
    }:
        return RouteAwareAnswerCompositionPolicy()

    raise ValueError(
        "composition_policy must be one of: top-k, top3, default, route-aware, "
        f"{RouteAwareCompositionPolicy.name}"
    )


def _to_policy_candidate(
    candidate: SentenceEvidenceCandidate,
) -> CompositionPolicyCandidate:
    retrieval_result = candidate.retrieval_result
    return CompositionPolicyCandidate(
        document_id=retrieval_result.document.id,
        title=retrieval_result.document.title,
        retrieval_rank=retrieval_result.rank,
        sentence=candidate.sentence,
        candidate_score=candidate.score,
    )


def _validate_max_sentences(max_sentences: int) -> None:
    if max_sentences <= 0:
        raise ValueError("max_sentences must be positive")
