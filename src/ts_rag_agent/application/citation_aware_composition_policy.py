from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ts_rag_agent.application.answer_composition import AnswerCompositionDecision
from ts_rag_agent.application.candidate_reranker_cv import CandidateRerankerExample
from ts_rag_agent.application.candidate_reranker_dataset import (
    build_candidate_runtime_features,
)
from ts_rag_agent.application.evidence_selection import (
    SentenceEvidenceCandidate,
    classify_question_route,
)
from ts_rag_agent.domain.dataset import PrimeQAQuery

PolicyFamily = Literal["score_rank", "context_rank_coverage", "dual_target"]


@dataclass(frozen=True)
class CitationAwareCompositionSpec:
    """One frozen runtime-visible composition policy configuration."""

    policy_id: str
    family: PolicyFamily
    document_cap: int
    rank_power: float = 0.0
    citation_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.document_cap <= 0:
            raise ValueError("document_cap must be positive")
        if self.rank_power < 0:
            raise ValueError("rank_power must be non-negative")
        if not 0 <= self.citation_weight <= 1:
            raise ValueError("citation_weight must be between 0 and 1")
        if self.family == "dual_target" and self.citation_weight in {0.0, 1.0}:
            raise ValueError("dual_target requires both model heads")


class DualTargetCandidateModel:
    """Runtime scorer fitted from offline citation and answer-fidelity labels."""

    def __init__(self) -> None:
        self._citation_pipeline = Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=180,
                    ),
                ),
            ]
        )
        self._f1_pipeline = Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )
        self._fitted = False

    def fit(self, examples: Sequence[CandidateRerankerExample]) -> None:
        if not examples:
            raise ValueError("dual-target fit requires examples")
        citation_labels = [int(example.is_gold_document) for example in examples]
        if len(set(citation_labels)) != 2:
            raise ValueError("citation head requires positive and negative labels")
        features = _feature_dicts(examples)
        self._citation_pipeline.fit(features, citation_labels)
        self._f1_pipeline.fit(
            features,
            [example.candidate_token_f1 for example in examples],
        )
        self._fitted = True

    def score_runtime(
        self,
        *,
        question: PrimeQAQuery,
        candidates: Sequence[SentenceEvidenceCandidate],
        selector_name: str,
    ) -> tuple[tuple[float, float], ...]:
        if not self._fitted:
            raise RuntimeError("dual-target model must be fitted before scoring")
        route = classify_question_route(question)
        features = [
            build_candidate_runtime_features(
                question=question,
                candidate=candidate,
                question_route=route,
                selector_name=selector_name,
            )
            for candidate in candidates
        ]
        probabilities = self._citation_pipeline.predict_proba(features)
        positive_index = list(self._citation_pipeline.classes_).index(1)
        f1_scores = self._f1_pipeline.predict(features)
        return tuple(
            (
                _clip(float(probability[positive_index])),
                _clip(float(f1_score)),
            )
            for probability, f1_score in zip(probabilities, f1_scores, strict=True)
        )


class CitationAwareCompositionPolicy:
    """Select answer sentences using only runtime-visible candidate signals."""

    def __init__(
        self,
        *,
        spec: CitationAwareCompositionSpec,
        selector_name: str = "bm25_sentence",
        model: DualTargetCandidateModel | None = None,
    ) -> None:
        if spec.family == "dual_target" and model is None:
            raise ValueError("dual_target policy requires a fitted model")
        if spec.family != "dual_target" and model is not None:
            raise ValueError("rule policy must not receive a learned model")
        self._spec = spec
        self._selector_name = selector_name
        self._model = model

    @property
    def name(self) -> str:
        return self._spec.policy_id

    @property
    def spec(self) -> CitationAwareCompositionSpec:
        return self._spec

    def select(
        self,
        question: PrimeQAQuery,
        candidates: Sequence[SentenceEvidenceCandidate],
        max_sentences: int,
    ) -> AnswerCompositionDecision:
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive")
        available = tuple(candidates)
        if not available:
            return AnswerCompositionDecision(
                selected_candidates=[],
                question_route=classify_question_route(question),
                strategy=self.name,
                reason="no eligible sentence candidates",
            )

        if self._spec.family == "context_rank_coverage":
            selected = _context_rank_coverage(
                available,
                limit=max_sentences,
                document_cap=self._spec.document_cap,
            )
        else:
            scores = self._scores(question=question, candidates=available)
            ranked = sorted(
                zip(available, scores, strict=True),
                key=lambda row: (
                    -row[1],
                    row[0].retrieval_result.rank,
                    row[0].retrieval_result.document.id,
                    row[0].sentence,
                ),
            )
            selected = _take_with_document_cap(
                (candidate for candidate, _score in ranked),
                limit=max_sentences,
                document_cap=self._spec.document_cap,
            )

        return AnswerCompositionDecision(
            selected_candidates=list(selected),
            question_route=classify_question_route(question),
            strategy=self.name,
            reason="selected by frozen runtime-visible citation-aware policy",
        )

    def _scores(
        self,
        *,
        question: PrimeQAQuery,
        candidates: Sequence[SentenceEvidenceCandidate],
    ) -> tuple[float, ...]:
        if self._spec.family == "score_rank":
            return tuple(
                candidate.score
                / math.log2(candidate.retrieval_result.rank + 1) ** self._spec.rank_power
                for candidate in candidates
            )
        if self._spec.family == "dual_target":
            assert self._model is not None
            heads = self._model.score_runtime(
                question=question,
                candidates=candidates,
                selector_name=self._selector_name,
            )
            weight = self._spec.citation_weight
            return tuple(
                weight * citation_score + (1 - weight) * f1_score
                for citation_score, f1_score in heads
            )
        raise ValueError(f"Unsupported score family: {self._spec.family}")


def stage180_policy_specs() -> tuple[CitationAwareCompositionSpec, ...]:
    """Return the frozen Stage 180 rule and learned policy family."""

    rules = (
        CitationAwareCompositionSpec("rule_score_rank_p000_cap1", "score_rank", 1, 0.0),
        CitationAwareCompositionSpec("rule_score_rank_p025_cap1", "score_rank", 1, 0.25),
        CitationAwareCompositionSpec("rule_score_rank_p050_cap1", "score_rank", 1, 0.5),
        CitationAwareCompositionSpec("rule_score_rank_p100_cap1", "score_rank", 1, 1.0),
        CitationAwareCompositionSpec("rule_score_rank_p025_cap3", "score_rank", 3, 0.25),
        CitationAwareCompositionSpec("rule_score_rank_p050_cap3", "score_rank", 3, 0.5),
        CitationAwareCompositionSpec("rule_score_rank_p100_cap3", "score_rank", 3, 1.0),
        CitationAwareCompositionSpec(
            "rule_context_rank_coverage_top3",
            "context_rank_coverage",
            1,
        ),
    )
    learned = tuple(
        CitationAwareCompositionSpec(
            policy_id=f"dual_c{int(weight * 100):02d}_f{int((1 - weight) * 100):02d}_cap{cap}",
            family="dual_target",
            document_cap=cap,
            citation_weight=weight,
        )
        for cap in (1, 2)
        for weight in (0.25, 0.5, 0.75)
    )
    return rules + learned


def _context_rank_coverage(
    candidates: Sequence[SentenceEvidenceCandidate],
    *,
    limit: int,
    document_cap: int,
) -> tuple[SentenceEvidenceCandidate, ...]:
    selected = []
    for rank in range(1, limit + 1):
        matching = [
            candidate for candidate in candidates if candidate.retrieval_result.rank == rank
        ]
        if matching:
            selected.append(matching[0])
    selected_ids = {id(candidate) for candidate in selected}
    return _take_with_document_cap(
        (*selected, *(candidate for candidate in candidates if id(candidate) not in selected_ids)),
        limit=limit,
        document_cap=document_cap,
    )


def _take_with_document_cap(
    candidates: Iterable[SentenceEvidenceCandidate],
    *,
    limit: int,
    document_cap: int,
) -> tuple[SentenceEvidenceCandidate, ...]:
    selected = []
    counts: Counter[str] = Counter()
    for candidate in candidates:
        document_id = candidate.retrieval_result.document.id
        if counts[document_id] >= document_cap:
            continue
        selected.append(candidate)
        counts[document_id] += 1
        if len(selected) >= limit:
            break
    return tuple(selected)


def _feature_dicts(examples: Sequence[CandidateRerankerExample]) -> list[dict[str, Any]]:
    return [dict(example.runtime_features) for example in examples]


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))
