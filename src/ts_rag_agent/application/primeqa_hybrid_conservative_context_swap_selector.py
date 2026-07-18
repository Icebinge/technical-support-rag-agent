from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    CANDIDATE_POOL_DEPTH,
    CONTEXT_DEPTH,
    ContextCandidateRecord,
    ContextSelection,
    ModelFamily,
    RuntimeCandidateScorer,
)

CONSERVATIVE_SWAP_PROTOCOL_ID = "primeqa_hybrid_conservative_context_swap_selector_v1"
PROTECTED_PREFIX_AND_BUDGET = ((8, 2), (9, 1))
MARGIN_THRESHOLD_QUANTILES = (0.50, 0.70, 0.80, 0.90, 0.95)


@dataclass(frozen=True)
class ConservativeSwapSelectorConfig:
    """One frozen Stage162 conservative RRF Top10 swap family."""

    config_id: str
    model_family: ModelFamily
    protected_prefix_depth: int
    promotion_budget: int
    context_depth: int = CONTEXT_DEPTH
    candidate_pool_depth: int = CANDIDATE_POOL_DEPTH

    def __post_init__(self) -> None:
        if (self.protected_prefix_depth, self.promotion_budget) not in (
            PROTECTED_PREFIX_AND_BUDGET
        ):
            raise ValueError("prefix and promotion budget are outside the frozen Stage162 set")
        if self.context_depth != CONTEXT_DEPTH:
            raise ValueError("Stage162 context depth must remain 10")
        if self.candidate_pool_depth != CANDIDATE_POOL_DEPTH:
            raise ValueError("Stage162 candidate pool depth must remain 200")
        if self.context_depth - self.protected_prefix_depth != self.promotion_budget:
            raise ValueError("promotion budget must equal the unprotected Top10 slot count")


@dataclass(frozen=True)
class ScoredSwapOpportunity:
    """One model-scored challenger/incumbent replacement opportunity."""

    incumbent: ContextCandidateRecord
    challenger: ContextCandidateRecord
    incumbent_score: float
    challenger_score: float

    @property
    def margin(self) -> float:
        return self.challenger_score - self.incumbent_score


@dataclass(frozen=True)
class ConservativeSwapPlan:
    """A scored Top10 plan that can be materialized at a frozen margin threshold."""

    config: ConservativeSwapSelectorConfig
    protected: tuple[ContextCandidateRecord, ...]
    incumbents: tuple[ContextCandidateRecord, ...]
    opportunities: tuple[ScoredSwapOpportunity, ...]

    def select(self, *, margin_threshold: float) -> ContextSelection:
        if margin_threshold < 0.0:
            raise ValueError("Stage162 margin threshold must be nonnegative")
        accepted = tuple(
            opportunity
            for opportunity in self.opportunities
            if opportunity.margin > margin_threshold
        )
        displaced_ids = {opportunity.incumbent.document_id for opportunity in accepted}
        retained = tuple(
            record for record in self.incumbents if record.document_id not in displaced_ids
        )
        promoted = tuple(opportunity.challenger for opportunity in accepted)
        selected = tuple(
            sorted([*self.protected, *retained, *promoted], key=lambda record: record.baseline_rank)
        )
        if len(selected) != CONTEXT_DEPTH:
            raise RuntimeError("Stage162 conservative swap did not preserve Top10 depth")
        expected_prefix = tuple(record.document_id for record in self.protected)
        observed_prefix = tuple(
            record.document_id for record in selected[: self.config.protected_prefix_depth]
        )
        return ContextSelection(
            selected=selected,
            protected_prefix_violation_count=int(observed_prefix != expected_prefix),
            tail_promotion_count=sum(record.baseline_rank > CONTEXT_DEPTH for record in selected),
        )


class ConservativeContextSwapSelector:
    """Score only the unprotected RRF Top10 incumbents and Top11-200 challengers."""

    def __init__(
        self,
        *,
        config: ConservativeSwapSelectorConfig,
        scorer: RuntimeCandidateScorer,
    ) -> None:
        self._config = config
        self._scorer = scorer

    @property
    def config(self) -> ConservativeSwapSelectorConfig:
        return self._config

    def plan(self, records: Sequence[ContextCandidateRecord]) -> ConservativeSwapPlan:
        ordered = sorted(records, key=lambda record: record.baseline_rank)
        if len(ordered) != self._config.candidate_pool_depth:
            raise ValueError("Stage162 selector requires an exact top200 candidate pool")
        protected = tuple(ordered[: self._config.protected_prefix_depth])
        incumbents = tuple(ordered[self._config.protected_prefix_depth : CONTEXT_DEPTH])
        challengers = tuple(ordered[CONTEXT_DEPTH:])
        scorable = (*incumbents, *challengers)
        scores = self._scorer.score(scorable)
        score_by_document = {
            record.document_id: float(score) for record, score in zip(scorable, scores, strict=True)
        }
        weakest_incumbents = sorted(
            incumbents,
            key=lambda record: (
                score_by_document[record.document_id],
                -record.baseline_rank,
                record.document_id,
            ),
        )
        strongest_challengers = sorted(
            challengers,
            key=lambda record: (
                -score_by_document[record.document_id],
                record.baseline_rank,
                record.document_id,
            ),
        )
        opportunities = tuple(
            ScoredSwapOpportunity(
                incumbent=incumbent,
                challenger=challenger,
                incumbent_score=score_by_document[incumbent.document_id],
                challenger_score=score_by_document[challenger.document_id],
            )
            for incumbent, challenger in zip(
                weakest_incumbents[: self._config.promotion_budget],
                strongest_challengers[: self._config.promotion_budget],
                strict=True,
            )
        )
        return ConservativeSwapPlan(
            config=self._config,
            protected=protected,
            incumbents=incumbents,
            opportunities=opportunities,
        )

    def select(
        self,
        records: Sequence[ContextCandidateRecord],
        *,
        margin_threshold: float,
    ) -> ContextSelection:
        return self.plan(records).select(margin_threshold=margin_threshold)


def frozen_stage162_swap_configs() -> tuple[ConservativeSwapSelectorConfig, ...]:
    """Return the four user-authorized Stage162 model and swap-budget families."""

    configs = []
    for protected_prefix_depth, promotion_budget in PROTECTED_PREFIX_AND_BUDGET:
        configs.extend(
            (
                ConservativeSwapSelectorConfig(
                    config_id=(
                        f"rrf_prefix{protected_prefix_depth}_budget{promotion_budget}_"
                        "pairwise_logistic_nested_margin_v1"
                    ),
                    model_family="pairwise_logistic",
                    protected_prefix_depth=protected_prefix_depth,
                    promotion_budget=promotion_budget,
                ),
                ConservativeSwapSelectorConfig(
                    config_id=(
                        f"rrf_prefix{protected_prefix_depth}_budget{promotion_budget}_"
                        "histogram_gbdt_nested_margin_v1"
                    ),
                    model_family="pointwise_histogram_gbdt",
                    protected_prefix_depth=protected_prefix_depth,
                    promotion_budget=promotion_budget,
                ),
            )
        )
    return tuple(configs)


def margin_threshold_candidates(
    plans: Sequence[ConservativeSwapPlan],
) -> tuple[float, ...]:
    """Build the frozen zero-plus-positive-margin-quantile threshold grid."""

    positive_margins = sorted(
        opportunity.margin
        for plan in plans
        for opportunity in plan.opportunities
        if opportunity.margin > 0.0
    )
    candidates = {0.0}
    if positive_margins:
        values = np.asarray(positive_margins, dtype=np.float64)
        candidates.update(
            round(float(np.quantile(values, quantile, method="linear")), 12)
            for quantile in MARGIN_THRESHOLD_QUANTILES
        )
    return tuple(sorted(candidates))
