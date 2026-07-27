from __future__ import annotations

import pytest

from ts_rag_agent.application import (
    composition_rank_capped_safety_pool_failure_attribution as attribution,
)
from ts_rag_agent.application.composition_action_audit import (
    ActionAuditRow,
    CompositionAction,
)
from ts_rag_agent.application.composition_gain_sensitive_ranking import (
    GainSensitivePrediction,
)
from ts_rag_agent.application.composition_rank_capped_safety_pool import (
    RankCappedSafetyPoolInnerOOFSnapshot,
    RankCappedSafetyPoolPolicySpec,
    stage191_policy_specs,
)


def test_diagnostic_partitions_all_strict_opportunity_paths() -> None:
    predictions = (
        _prediction(_row("excluded", "baseline", family="baseline"), 0.10, 0.0),
        _prediction(_row("excluded", "d1"), 0.11, 0.0),
        _prediction(_row("excluded", "d2"), 0.12, 0.0),
        _prediction(_row("excluded", "d3"), 0.13, 0.0),
        _prediction(_row("excluded", "strict", strict=True), 0.90, 2.0),
        _prediction(_row("missed", "baseline", family="baseline"), 0.10, 2.0),
        _prediction(_row("missed", "strict", strict=True), 0.20, 1.0),
        _prediction(_row("selected", "baseline", family="baseline"), 0.10, 0.0),
        _prediction(_row("selected", "strict", strict=True), 0.20, 2.0),
    )

    report = attribution.diagnose_rank_capped_safety_pool_policy(
        predictions,
        _spec(),
    )

    assert report["strict_opportunity_context_count"] == 3
    assert report["opportunity_partition"] == {
        "pool_exclusion_context_count": 1,
        "ranker_miss_context_count": 1,
        "strict_selected_context_count": 1,
        "partition_total": 3,
        "partition_exact": True,
    }
    assert report["ranker_miss_breakdown"]["partition_exact"] is True


def test_accumulator_consumes_five_snapshots_and_all_configurations() -> None:
    accumulator = attribution.RankCappedSafetyPoolFailureAttributionAccumulator()
    predictions = (
        _prediction(_row("q", "baseline", family="baseline"), 0.1, 0.0),
        _prediction(_row("q", "strict", strict=True), 0.2, 1.0),
    )
    specs = stage191_policy_specs()
    bundle_names = {spec.bundle_name for spec in specs}
    for index in range(1, 6):
        accumulator.consume(
            RankCappedSafetyPoolInnerOOFSnapshot(
                outer_fold_id=f"fold_{index}",
                inner_fold_ids=tuple(f"fold_{other}" for other in range(1, 6) if other != index),
                question_count=1,
                predictions_by_bundle={name: predictions for name in bundle_names},
                reference_spec_name=specs[0].name,
                reference_kind="selected_eligible",
                eligible_config_count=1,
            )
        )

    report = accumulator.finalize()

    assert report["execution"]["snapshot_count"] == 5
    assert len(report["configuration_aggregates"]) == 32
    assert report["execution"]["all_configuration_partitions_exact"] is True
    assert (
        report["reference_trajectory"]["opportunity_partition"]["strict_selected_context_count"]
        == 5
    )


def test_inner_oof_snapshot_freezes_private_prediction_bundles() -> None:
    source = {"bundle": []}
    snapshot = RankCappedSafetyPoolInnerOOFSnapshot(
        outer_fold_id="fold_1",
        inner_fold_ids=["fold_2"],
        question_count=0,
        predictions_by_bundle=source,
        reference_spec_name="spec",
        reference_kind="top_ineligible",
        eligible_config_count=0,
    )

    source["late"] = []

    assert snapshot.inner_fold_ids == ("fold_2",)
    assert snapshot.predictions_by_bundle == {"bundle": ()}
    with pytest.raises(TypeError):
        snapshot.predictions_by_bundle["late"] = ()


def _spec() -> RankCappedSafetyPoolPolicySpec:
    return RankCappedSafetyPoolPolicySpec(
        name="test",
        feature_representation="raw_runtime",
        safety_estimator="class_balanced_logistic",
        gain_ranker="pairwise_pareto_logistic",
        pool_cap=4,
    )


def _row(
    question_key: str,
    action_id: str,
    *,
    family: str = "test",
    strict: bool = False,
) -> ActionAuditRow:
    return ActionAuditRow(
        question_key=question_key,
        fold_id="fold_1",
        route="other",
        action=CompositionAction(
            action_id=action_id,
            family=family,
            aliases=(),
            selected_indices=(0,),
            matches_stage180=False,
        ),
        runtime_features={},
        outcome_class="test",
        strict_expected=strict,
        citation_delta=1 if strict else 0,
        f1_delta=0.1 if strict else 0.0,
    )


def _prediction(
    row: ActionAuditRow,
    risk: float,
    gain: float,
) -> GainSensitivePrediction:
    return GainSensitivePrediction(
        row=row,
        citation_loss_probability=risk,
        f1_loss_probability=risk,
        gain_score=gain,
    )
