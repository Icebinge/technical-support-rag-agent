from __future__ import annotations

from collections.abc import Sequence

import pytest

from ts_rag_agent.application.primeqa_hybrid_conservative_context_swap_selector import (
    ConservativeContextSwapSelector,
    ConservativeSwapSelectorConfig,
    frozen_stage162_swap_configs,
    margin_threshold_candidates,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    ContextCandidateRecord,
    ScorerFitSummary,
)


class _UtilityScorer:
    def fit(
        self,
        records: Sequence[ContextCandidateRecord],
        *,
        protected_prefix_depth: int,
    ) -> ScorerFitSummary:
        return ScorerFitSummary(
            model_family="fake",
            training_group_count=1,
            positive_candidate_count=1,
            negative_candidate_count=1,
            training_example_count=2,
            feature_count=1,
        )

    def score(self, records: Sequence[ContextCandidateRecord]) -> list[float]:
        return [float(record.features["utility"]) for record in records]


def test_stage162_frozen_grid_has_exact_four_conservative_configs() -> None:
    configs = frozen_stage162_swap_configs()

    assert len(configs) == 4
    assert {(config.protected_prefix_depth, config.promotion_budget) for config in configs} == {
        (8, 2),
        (9, 1),
    }
    assert {config.model_family for config in configs} == {
        "pairwise_logistic",
        "pointwise_histogram_gbdt",
    }


def test_stage162_two_swap_selector_preserves_prefix_and_respects_margin() -> None:
    config = ConservativeSwapSelectorConfig(
        config_id="prefix8_budget2",
        model_family="pairwise_logistic",
        protected_prefix_depth=8,
        promotion_budget=2,
    )
    selector = ConservativeContextSwapSelector(config=config, scorer=_UtilityScorer())
    plan = selector.plan(_candidate_pool())

    two_swaps = plan.select(margin_threshold=0.0)
    one_swap = plan.select(margin_threshold=0.7)

    assert [record.baseline_rank for record in two_swaps.selected] == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        50,
        60,
    ]
    assert [record.baseline_rank for record in one_swap.selected] == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        50,
    ]
    assert two_swaps.tail_promotion_count == 2
    assert one_swap.tail_promotion_count == 1
    assert two_swaps.protected_prefix_violation_count == 0
    assert one_swap.protected_prefix_violation_count == 0


def test_stage162_margin_comparison_is_strict_and_can_keep_rrf_incumbent() -> None:
    config = ConservativeSwapSelectorConfig(
        config_id="prefix9_budget1",
        model_family="pairwise_logistic",
        protected_prefix_depth=9,
        promotion_budget=1,
    )
    plan = ConservativeContextSwapSelector(config=config, scorer=_UtilityScorer()).plan(
        _candidate_pool()
    )
    exact_margin = plan.opportunities[0].margin

    selection = plan.select(margin_threshold=exact_margin)

    assert [record.baseline_rank for record in selection.selected] == list(range(1, 11))
    assert selection.tail_promotion_count == 0


def test_stage162_threshold_grid_is_zero_plus_frozen_positive_quantiles() -> None:
    config = ConservativeSwapSelectorConfig(
        config_id="prefix8_budget2",
        model_family="pairwise_logistic",
        protected_prefix_depth=8,
        promotion_budget=2,
    )
    selector = ConservativeContextSwapSelector(config=config, scorer=_UtilityScorer())
    plans = tuple(selector.plan(_candidate_pool(offset=offset)) for offset in range(6))

    thresholds = margin_threshold_candidates(plans)

    assert thresholds[0] == 0.0
    assert 2 <= len(thresholds) <= 6
    assert thresholds == tuple(sorted(set(thresholds)))
    assert all(threshold >= 0.0 for threshold in thresholds)


def test_stage162_config_rejects_unfrozen_prefix_budget_pair() -> None:
    with pytest.raises(ValueError, match="outside the frozen"):
        ConservativeSwapSelectorConfig(
            config_id="invalid",
            model_family="pairwise_logistic",
            protected_prefix_depth=8,
            promotion_budget=1,
        )


def _candidate_pool(offset: int = 0) -> tuple[ContextCandidateRecord, ...]:
    utilities = {9: 0.2, 10: 0.1, 50: 1.0 + offset * 0.01, 60: 0.8}
    return tuple(
        ContextCandidateRecord(
            sample_id=f"sample_{offset}",
            fold_id="fold_0",
            document_id=f"sample_{offset}_doc_{rank:03d}",
            baseline_rank=rank,
            answerable=True,
            is_gold=rank == 50,
            features={"utility": utilities.get(rank, -float(rank))},
        )
        for rank in range(1, 201)
    )
