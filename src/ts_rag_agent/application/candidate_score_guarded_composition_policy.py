from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ts_rag_agent.application.answer_composition import AnswerCompositionDecision
from ts_rag_agent.application.candidate_reranker_cv import (
    SCORER_FACTORIES,
    CandidateRerankerExample,
    CandidateRerankerScorer,
    candidate_reranker_rows_to_examples,
)
from ts_rag_agent.application.candidate_reranker_dataset import (
    build_candidate_runtime_features,
)
from ts_rag_agent.application.candidate_reranker_policy_search import (
    CandidateRerankerPolicyConfig,
)
from ts_rag_agent.application.candidate_score_guarded_policy_evaluation import (
    default_stage39_policy_specs,
)
from ts_rag_agent.application.candidate_score_guarded_policy_split_validation import (
    CANDIDATE_SCORE_GTE_60_LABEL,
)
from ts_rag_agent.application.evidence_selection import (
    SentenceEvidenceCandidate,
    classify_question_route,
)
from ts_rag_agent.domain.dataset import PrimeQAQuestion


@dataclass(frozen=True)
class RuntimeCandidateRerankerDecisionTrace:
    """Runtime trace for one candidate-score guarded reranker decision."""

    action: str
    reason: str
    selected_candidate_rank: int
    selected_candidate_score: float
    model_score_margin_vs_top_candidate: float
    proposed_worst_retrieval_rank: int | None = None
    rank_contained_max_retrieval_rank: int | None = None
    preserve_baseline_out_of_rank_docs: bool = False
    protected_baseline_out_of_rank_document_ids: tuple[str, ...] = ()
    dropped_protected_baseline_out_of_rank_document_ids: tuple[str, ...] = ()


class CandidateScoreGuardedRerankerCompositionPolicy:
    """Runtime answer composition using a trained reranker plus score>=60 guard."""

    def __init__(
        self,
        scorer: CandidateRerankerScorer,
        selector_name: str,
        policy_config: CandidateRerankerPolicyConfig | None = None,
        rank_contained_max_retrieval_rank: int | None = None,
        preserve_baseline_out_of_rank_docs: bool = False,
    ) -> None:
        if (
            rank_contained_max_retrieval_rank is not None
            and rank_contained_max_retrieval_rank <= 0
        ):
            raise ValueError("rank_contained_max_retrieval_rank must be positive")
        if preserve_baseline_out_of_rank_docs and rank_contained_max_retrieval_rank is None:
            raise ValueError(
                "preserve_baseline_out_of_rank_docs requires "
                "rank_contained_max_retrieval_rank"
            )
        self._scorer = scorer
        self._selector_name = selector_name
        self._policy_config = policy_config or _candidate_score_gte60_config()
        self._rank_contained_max_retrieval_rank = rank_contained_max_retrieval_rank
        self._preserve_baseline_out_of_rank_docs = preserve_baseline_out_of_rank_docs
        self._last_trace: RuntimeCandidateRerankerDecisionTrace | None = None

    @property
    def name(self) -> str:
        """Stable policy name used in experiment reports."""

        if self._preserve_baseline_out_of_rank_docs:
            return (
                "candidate_score_gte_60_rank_contained_"
                "preserve_baseline_out_of_rank_guarded_reranker"
            )
        if self._rank_contained_max_retrieval_rank is not None:
            return "candidate_score_gte_60_rank_contained_guarded_reranker"
        return "candidate_score_gte_60_guarded_reranker"

    @property
    def last_trace(self) -> RuntimeCandidateRerankerDecisionTrace | None:
        """Most recent decision trace for diagnostics and tests."""

        return self._last_trace

    def select(
        self,
        question: PrimeQAQuestion,
        candidates: Sequence[SentenceEvidenceCandidate],
        max_sentences: int,
    ) -> AnswerCompositionDecision:
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive")
        if not candidates:
            self._last_trace = RuntimeCandidateRerankerDecisionTrace(
                action="keep_top_candidate",
                reason="no_candidates",
                selected_candidate_rank=0,
                selected_candidate_score=0.0,
                model_score_margin_vs_top_candidate=0.0,
                proposed_worst_retrieval_rank=None,
                rank_contained_max_retrieval_rank=self._rank_contained_max_retrieval_rank,
                preserve_baseline_out_of_rank_docs=(
                    self._preserve_baseline_out_of_rank_docs
                ),
            )
            return AnswerCompositionDecision(
                selected_candidates=[],
                question_route=classify_question_route(question),
                strategy=self.name,
                reason="no candidates available",
            )

        question_route = classify_question_route(question)
        runtime_examples = _runtime_examples(
            question=question,
            candidates=candidates,
            selector_name=self._selector_name,
            question_route=question_route,
        )
        scores = self._scorer.score(runtime_examples)
        selected_index = max(
            range(len(runtime_examples)),
            key=lambda index: (scores[index], -runtime_examples[index].candidate_rank),
        )
        baseline_score = scores[0]
        selected_score = scores[selected_index]
        selected_rank = selected_index + 1
        selected_candidate = candidates[selected_index]
        margin = round(selected_score - baseline_score, 6)
        blocked_reason = _blocked_reason(
            config=self._policy_config,
            question_route=question_route,
            selected_rank=selected_rank,
            selected_candidate_score=selected_candidate.score,
            model_score_margin_vs_top_candidate=margin,
        )
        if blocked_reason is None:
            baseline_candidates = list(candidates[:max_sentences])
            proposed_candidates = _leading_rewrite_candidates(
                leading_candidate=selected_candidate,
                baseline_candidates=candidates,
                limit=max_sentences,
            )
            proposed_worst_retrieval_rank = _worst_retrieval_rank(proposed_candidates)
            rank_contained_blocked_reason = _rank_contained_blocked_reason(
                proposed_candidates=proposed_candidates,
                max_retrieval_rank=self._rank_contained_max_retrieval_rank,
            )
            preservation_blocked_reason = _baseline_out_of_rank_preservation_blocked_reason(
                baseline_candidates=baseline_candidates,
                proposed_candidates=proposed_candidates,
                max_retrieval_rank=self._rank_contained_max_retrieval_rank,
                enabled=self._preserve_baseline_out_of_rank_docs,
            )
            if rank_contained_blocked_reason is None:
                if preservation_blocked_reason is None:
                    selected_candidates = proposed_candidates
                    action = "replace_with_model_candidate"
                    reason = "candidate_score_gte_60_accepted"
                else:
                    selected_candidates = baseline_candidates
                    action = "keep_top_candidate"
                    reason = preservation_blocked_reason
            else:
                selected_candidates = baseline_candidates
                action = "keep_top_candidate"
                reason = rank_contained_blocked_reason
        else:
            selected_candidates = list(candidates[:max_sentences])
            action = "keep_top_candidate"
            reason = blocked_reason
            proposed_worst_retrieval_rank = None
            proposed_candidates = selected_candidates

        protected_docs = _baseline_out_of_rank_document_ids(
            baseline_candidates=list(candidates[:max_sentences]),
            max_retrieval_rank=self._rank_contained_max_retrieval_rank,
            enabled=self._preserve_baseline_out_of_rank_docs,
        )
        proposed_docs = _document_ids(proposed_candidates)
        self._last_trace = RuntimeCandidateRerankerDecisionTrace(
            action=action,
            reason=reason,
            selected_candidate_rank=selected_rank,
            selected_candidate_score=round(selected_candidate.score, 4),
            model_score_margin_vs_top_candidate=margin,
            proposed_worst_retrieval_rank=proposed_worst_retrieval_rank,
            rank_contained_max_retrieval_rank=self._rank_contained_max_retrieval_rank,
            preserve_baseline_out_of_rank_docs=(
                self._preserve_baseline_out_of_rank_docs
            ),
            protected_baseline_out_of_rank_document_ids=tuple(sorted(protected_docs)),
            dropped_protected_baseline_out_of_rank_document_ids=tuple(
                sorted(protected_docs - proposed_docs)
            ),
        )
        return AnswerCompositionDecision(
            selected_candidates=selected_candidates,
            question_route=question_route,
            strategy=self.name,
            reason=reason,
        )


def fit_candidate_score_guarded_reranker_composition_policy(
    rows: Sequence[Mapping[str, Any]],
    selector_name: str,
    model_name: str = "logistic_best_candidate",
    train_split: str = "train",
    rank_contained_max_retrieval_rank: int | None = None,
    preserve_baseline_out_of_rank_docs: bool = False,
) -> CandidateScoreGuardedRerankerCompositionPolicy:
    """Fit a runtime candidate-score guarded reranker composition policy."""

    normalized_train_split = train_split.strip().lower()
    if not normalized_train_split:
        raise ValueError("train_split must not be empty")
    if model_name not in SCORER_FACTORIES:
        raise ValueError(f"Unknown candidate reranker model: {model_name}")
    examples = [
        example
        for example in candidate_reranker_rows_to_examples(rows)
        if example.split == normalized_train_split
    ]
    if not examples:
        raise ValueError(f"No candidate-reranker rows found for split: {train_split}")
    scorer = SCORER_FACTORIES[model_name]().fit(examples)
    return CandidateScoreGuardedRerankerCompositionPolicy(
        scorer=scorer,
        selector_name=selector_name,
        rank_contained_max_retrieval_rank=rank_contained_max_retrieval_rank,
        preserve_baseline_out_of_rank_docs=preserve_baseline_out_of_rank_docs,
    )


def _runtime_examples(
    question: PrimeQAQuestion,
    candidates: Sequence[SentenceEvidenceCandidate],
    selector_name: str,
    question_route: str,
) -> list[CandidateRerankerExample]:
    return [
        CandidateRerankerExample(
            split="runtime",
            question_id=question.id,
            candidate_id=f"{question.id}::runtime_candidate_{rank:03d}",
            candidate_rank=rank,
            question_route=question_route,
            runtime_features=build_candidate_runtime_features(
                question=question,
                candidate=candidate,
                question_route=question_route,
                selector_name=selector_name,
            ),
            candidate_token_f1=0.0,
            is_best_candidate_for_question=False,
            is_gold_document=False,
        )
        for rank, candidate in enumerate(candidates, start=1)
    ]


def _blocked_reason(
    config: CandidateRerankerPolicyConfig,
    question_route: str,
    selected_rank: int,
    selected_candidate_score: float,
    model_score_margin_vs_top_candidate: float,
) -> str | None:
    if selected_rank == 1:
        return "model_selected_top_candidate"
    if question_route in config.blocked_routes:
        return "route_blocked"
    if selected_rank > config.max_selected_rank:
        return "selected_rank_exceeds_limit"
    if model_score_margin_vs_top_candidate < config.min_score_margin_vs_top_candidate:
        return "score_margin_below_min"
    if (
        config.min_selected_candidate_score is not None
        and selected_candidate_score < config.min_selected_candidate_score
    ):
        return "candidate_score_gte_60_blocked"
    return None


def _leading_rewrite_candidates(
    leading_candidate: SentenceEvidenceCandidate,
    baseline_candidates: Sequence[SentenceEvidenceCandidate],
    limit: int,
) -> list[SentenceEvidenceCandidate]:
    selected = [leading_candidate]
    for candidate in baseline_candidates:
        if candidate is leading_candidate:
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _rank_contained_blocked_reason(
    proposed_candidates: Sequence[SentenceEvidenceCandidate],
    max_retrieval_rank: int | None,
) -> str | None:
    if max_retrieval_rank is None:
        return None
    if any(
        candidate.retrieval_result.rank > max_retrieval_rank
        for candidate in proposed_candidates
    ):
        return "selected_citation_rank_exceeds_limit"
    return None


def _baseline_out_of_rank_preservation_blocked_reason(
    baseline_candidates: Sequence[SentenceEvidenceCandidate],
    proposed_candidates: Sequence[SentenceEvidenceCandidate],
    max_retrieval_rank: int | None,
    enabled: bool,
) -> str | None:
    protected_docs = _baseline_out_of_rank_document_ids(
        baseline_candidates=baseline_candidates,
        max_retrieval_rank=max_retrieval_rank,
        enabled=enabled,
    )
    if not protected_docs:
        return None
    dropped_docs = protected_docs - _document_ids(proposed_candidates)
    if dropped_docs:
        return "baseline_out_of_rank_document_dropped"
    return None


def _baseline_out_of_rank_document_ids(
    baseline_candidates: Sequence[SentenceEvidenceCandidate],
    max_retrieval_rank: int | None,
    enabled: bool,
) -> set[str]:
    if not enabled:
        return set()
    if max_retrieval_rank is None:
        raise ValueError(
            "max_retrieval_rank is required when preserving baseline out-of-rank docs"
        )
    return {
        candidate.retrieval_result.document.id
        for candidate in baseline_candidates
        if candidate.retrieval_result.rank > max_retrieval_rank
    }


def _document_ids(candidates: Sequence[SentenceEvidenceCandidate]) -> set[str]:
    return {candidate.retrieval_result.document.id for candidate in candidates}


def _worst_retrieval_rank(
    candidates: Sequence[SentenceEvidenceCandidate],
) -> int | None:
    if not candidates:
        return None
    return max(candidate.retrieval_result.rank for candidate in candidates)


def _candidate_score_gte60_config() -> CandidateRerankerPolicyConfig:
    for label, config in default_stage39_policy_specs():
        if label == CANDIDATE_SCORE_GTE_60_LABEL:
            return config
    raise ValueError("default Stage 39 policies missing candidate_score_gte_60")
