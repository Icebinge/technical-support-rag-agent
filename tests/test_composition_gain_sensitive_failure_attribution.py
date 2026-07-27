from __future__ import annotations

from ts_rag_agent.application.composition_action_audit import ActionAuditRow, CompositionAction
from ts_rag_agent.application.composition_gain_sensitive_failure_attribution import (
    GainSensitiveFailureAttributionAccumulator,
    diagnose_gain_sensitive_policy,
)
from ts_rag_agent.application.composition_gain_sensitive_ranking import (
    GainSensitiveInnerOOFSnapshot,
    GainSensitivePolicySpec,
    GainSensitivePrediction,
    stage188_policy_specs,
)


def test_policy_diagnostic_exactly_partitions_strict_opportunities() -> None:
    predictions = _diagnostic_predictions()

    report = diagnose_gain_sensitive_policy(predictions, _spec())

    assert report["strict_opportunity_context_count"] == 4
    assert report["opportunity_partition"] == {
        "frontier_exclusion_context_count": 1,
        "ranker_miss_context_count": 2,
        "strict_selected_context_count": 1,
        "partition_total": 4,
        "partition_exact": True,
    }
    assert report["frontier_strict_question_recall"] == 0.75
    assert report["conditional_ranker_strict_capture"] == 0.333333
    assert report["actual_strict_opportunity_capture"] == 0.25
    assert report["baseline_change_strict_precision"] == 1.0
    assert report["filter_harm_context_count"] == 1
    assert report["filter_rescue_context_count"] == 0


def test_accumulator_streams_five_snapshots_and_identifies_ranker_miss() -> None:
    accumulator = GainSensitiveFailureAttributionAccumulator()
    specs = stage188_policy_specs()
    bundles = {spec.bundle_name: _diagnostic_predictions() for spec in specs}

    for index in range(1, 6):
        accumulator.consume(
            GainSensitiveInnerOOFSnapshot(
                outer_fold_id=f"fold_{index}",
                inner_fold_ids=tuple(f"fold_{inner}" for inner in range(1, 6) if inner != index),
                question_count=4,
                predictions_by_bundle=bundles,
                top_candidate_spec_name=specs[0].name,
                eligible_config_count=0,
            )
        )

    report = accumulator.finalize()

    assert len(report["outer_folds"]) == 5
    assert len(report["configuration_aggregates"]) == 32
    assert report["execution"]["private_bundle_prediction_count_consumed"] == 320
    assert report["execution"]["all_configuration_partitions_exact"] is True
    assert report["diagnostic_findings"]["primary_bottleneck"] == "gain_ranker_miss"
    assert (
        report["diagnostic_findings"]["stage190_design_branch"]["name"]
        == "baseline_referenced_strict_change_gate"
    )
    assert report["factor_aggregates"]["safety_frontier_margin"]["0.00"]["configuration_count"] == 8


def _diagnostic_predictions() -> tuple[GainSensitivePrediction, ...]:
    rows = []
    definitions = {
        "q_excluded": ((0.10, 0.10, 0.0), (0.90, 0.90, 2.0)),
        "q_missed_a": ((0.10, 0.10, 2.0), (0.10, 0.10, 1.0)),
        "q_missed_b": ((0.10, 0.10, 2.0), (0.10, 0.10, 1.0)),
        "q_selected": ((0.10, 0.10, 0.0), (0.10, 0.10, 2.0)),
    }
    for question_key, (baseline_scores, strict_scores) in definitions.items():
        baseline = _row(question_key, "baseline", strict=False, family="baseline")
        strict = _row(question_key, "strict", strict=True)
        rows.extend(
            (
                GainSensitivePrediction(baseline, *baseline_scores),
                GainSensitivePrediction(strict, *strict_scores),
            )
        )
    return tuple(rows)


def _spec() -> GainSensitivePolicySpec:
    return GainSensitivePolicySpec(
        name="test",
        feature_representation="raw_runtime",
        safety_estimator="class_balanced_logistic",
        gain_ranker="pairwise_pareto_logistic",
        safety_frontier_margin=0.0,
    )


def _row(
    question_key: str,
    action_id: str,
    *,
    strict: bool,
    family: str = "test",
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
        runtime_features={"score": 1.0},
        outcome_class="test",
        strict_expected=strict,
        citation_delta=1 if strict else 0,
        f1_delta=0.1 if strict else 0.0,
    )
