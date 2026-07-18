from __future__ import annotations

from collections.abc import Sequence

import pytest

from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    CONTEXT_DEPTH,
    RUNTIME_FEATURE_NAMES,
    ContextCandidateRecord,
    PairwiseLogisticCandidateScorer,
    PointwiseHistogramCandidateScorer,
    ProtectedContextSelectorConfig,
    ProtectedPrefixContextSelector,
    ScorerFitSummary,
    frozen_stage161_selector_configs,
    select_current_query_overlap_top10,
    select_original_rrf_top10,
)


class _BaselineReverseScorer:
    def fit(
        self,
        records: Sequence[ContextCandidateRecord],
        *,
        protected_prefix_depth: int,
    ) -> ScorerFitSummary:
        raise AssertionError("selection-only test must not fit")

    def score(self, records: Sequence[ContextCandidateRecord]) -> list[float]:
        return [float(record.baseline_rank) for record in records]


def test_stage161_frozen_grid_has_exact_six_configs() -> None:
    configs = frozen_stage161_selector_configs()

    assert len(configs) == 6
    assert {config.protected_prefix_depth for config in configs} == {3, 5, 7}
    assert {config.model_family for config in configs} == {
        "pairwise_logistic",
        "pointwise_histogram_gbdt",
    }
    assert all(config.context_depth == CONTEXT_DEPTH for config in configs)
    assert len({config.config_id for config in configs}) == 6


@pytest.mark.parametrize("prefix_depth", [3, 5, 7])
def test_selector_preserves_prefix_and_learns_only_remaining_slots(
    prefix_depth: int,
) -> None:
    records = _candidate_pool(sample_id="sample", fold_id="fold_0", gold_rank=80)
    config = ProtectedContextSelectorConfig(
        config_id=f"test_prefix_{prefix_depth}",
        model_family="pairwise_logistic",
        protected_prefix_depth=prefix_depth,
    )

    selection = ProtectedPrefixContextSelector(
        config=config,
        scorer=_BaselineReverseScorer(),
    ).select(records)

    assert [record.baseline_rank for record in selection.selected[:prefix_depth]] == list(
        range(1, prefix_depth + 1)
    )
    assert [record.baseline_rank for record in selection.selected[prefix_depth:]] == list(
        range(200, 200 - (CONTEXT_DEPTH - prefix_depth), -1)
    )
    assert selection.protected_prefix_violation_count == 0
    assert selection.tail_promotion_count == CONTEXT_DEPTH - prefix_depth


def test_stage161_controls_reproduce_rrf_and_query_overlap_ordering() -> None:
    records = _candidate_pool(sample_id="sample", fold_id="fold_0", gold_rank=80)
    overlap = select_current_query_overlap_top10(records)
    rrf = select_original_rrf_top10(records)

    assert [record.baseline_rank for record in rrf.selected] == list(range(1, 11))
    assert [record.baseline_rank for record in overlap.selected] == list(range(200, 190, -1))
    assert overlap.tail_promotion_count == 10


@pytest.mark.parametrize(
    "scorer",
    [PairwiseLogisticCandidateScorer(), PointwiseHistogramCandidateScorer()],
)
def test_stage161_scorers_fit_runtime_features_and_promote_gold(scorer) -> None:
    records = tuple(
        record
        for sample_index in range(12)
        for record in _candidate_pool(
            sample_id=f"sample_{sample_index}",
            fold_id=f"fold_{sample_index % 3}",
            gold_rank=25 + sample_index,
            pool_depth=45,
        )
    )

    summary = scorer.fit(records, protected_prefix_depth=5)
    sample_records = [record for record in records if record.sample_id == "sample_0"]
    scores = scorer.score(sample_records[5:])
    gold_score = scores[19]

    assert summary.training_group_count == 12
    assert summary.positive_candidate_count == 12
    assert summary.negative_candidate_count >= 12
    assert summary.feature_count == len(RUNTIME_FEATURE_NAMES)
    assert gold_score == max(scores)


def test_stage161_config_rejects_non_frozen_prefix() -> None:
    with pytest.raises(ValueError, match="outside the frozen"):
        ProtectedContextSelectorConfig(
            config_id="invalid",
            model_family="pairwise_logistic",
            protected_prefix_depth=4,
        )


def _candidate_pool(
    *,
    sample_id: str,
    fold_id: str,
    gold_rank: int,
    pool_depth: int = 200,
) -> tuple[ContextCandidateRecord, ...]:
    return tuple(
        ContextCandidateRecord(
            sample_id=sample_id,
            fold_id=fold_id,
            document_id=f"{sample_id}_doc_{rank:03d}",
            baseline_rank=rank,
            answerable=True,
            is_gold=rank == gold_rank,
            features={
                feature_name: (
                    100.0
                    if rank == gold_rank
                    else float(rank)
                    if feature_name == "current_query_overlap_combined_score"
                    else 0.0
                )
                for feature_name in RUNTIME_FEATURE_NAMES
            },
        )
        for rank in range(1, pool_depth + 1)
    )
