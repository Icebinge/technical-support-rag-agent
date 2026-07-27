from __future__ import annotations

from ts_rag_agent.application import composition_joint_risk_winner_cv as policy
from ts_rag_agent.application import composition_safety_constrained_lambdamart as stage194
from ts_rag_agent.application import composition_safety_first_frontier as stage196
from ts_rag_agent.application.composition_action_audit import ActionAuditRow, CompositionAction
from ts_rag_agent.application.composition_gain_sensitive_ranking import (
    build_stage182_reference_rows,
)
from ts_rag_agent.application.composition_joint_constraint_ranking import (
    evaluate_selected_actions,
)


def test_stage199_factorial_grid_is_complete_and_unique() -> None:
    specs = policy.stage199_policy_specs()

    assert len(specs) == 28
    assert len({spec.name for spec in specs}) == 28
    assert {spec.risk_signal for spec in specs} == set(policy._RISK_SIGNALS)
    assert {spec.winner_rule.name for spec in specs} == {
        "gain_only",
        "rank_utility_0.25",
        "rank_utility_0.50",
        "rank_utility_1.00",
        "rank_utility_2.00",
        "gain_shortlist_2_then_risk",
        "gain_shortlist_4_then_risk",
    }


def test_exact_control_reproduces_stage196_decision() -> None:
    rows, safety, gain, classifier, pairwise = _predictions()
    source = _source_spec(prefix=4)

    stage196_decision = stage196.build_safety_first_frontier_decisions(
        safety,
        gain,
        classifier,
        source,
    )[0]
    stage199_decision = policy.build_joint_risk_winner_decisions(
        safety,
        gain,
        classifier,
        pairwise,
        source,
        _policy("source_weighted_classifier", "gain_only"),
    )[0]

    assert stage199_decision.winner.row == stage196_decision.winner.row
    assert stage199_decision.complete_pool == stage196_decision.complete_pool
    assert stage199_decision.frontier == stage196_decision.frontier
    assert {row.question_key for row in rows} == {"q1"}


def test_rank_utility_can_replace_unsafe_gain_winner_with_safer_strict() -> None:
    _, safety, gain, classifier, pairwise = _predictions()

    decision = policy.build_joint_risk_winner_decisions(
        safety,
        gain,
        classifier,
        pairwise,
        _source_spec(prefix=4),
        _policy("source_weighted_classifier", "rank_utility_1.00"),
    )[0]

    assert decision.winner.row.action.action_id == "strict"


def test_gain_shortlist_then_risk_does_not_force_baseline() -> None:
    _, safety, gain, classifier, pairwise = _predictions()

    decision = policy.build_joint_risk_winner_decisions(
        safety,
        gain,
        classifier,
        pairwise,
        _source_spec(prefix=4),
        _policy("source_weighted_classifier", "gain_shortlist_2_then_risk"),
    )[0]

    assert decision.winner.row.action.action_id == "strict"


def test_real_joint_partition_fit_trains_five_models() -> None:
    training = tuple(
        row
        for question_index in range(60)
        for row in (
            _row(f"train_{question_index}", "baseline", family="baseline"),
            _row(
                f"train_{question_index}",
                "strict",
                strict=True,
                citation_delta=1,
                f1_delta=0.1,
            ),
            _row(
                f"train_{question_index}",
                "unsafe",
                citation_delta=-1,
                f1_delta=-0.1,
            ),
        )
    )
    heldout = tuple(
        row
        for question_index in range(10)
        for row in (
            _row(f"heldout_{question_index}", "baseline", family="baseline"),
            _row(f"heldout_{question_index}", "strict", strict=True, citation_delta=1),
            _row(f"heldout_{question_index}", "unsafe", f1_delta=-0.1),
        )
    )
    feature_index = {
        (row.question_key, row.action.action_id): {
            "strict_hint": float(row.strict_expected),
            "unsafe_hint": float(row.action.action_id == "unsafe"),
        }
        for row in (*training, *heldout)
    }

    result = policy.fit_predict_joint_partition(
        training,
        heldout,
        {"raw_runtime": feature_index, "question_relative_runtime": feature_index},
        _source_spec(prefix=4),
    )

    assert result.model_fit_count == 5
    assert result.tree_count == 900
    assert result.group_contract_validation_count == 2
    assert len(result.safety_predictions) == len(heldout)
    assert len(result.gain_predictions) == len(heldout)
    assert len(result.classifier_risk_predictions) == len(heldout)
    assert len(result.pairwise_safety_predictions) == len(heldout)


def test_nested_cv_reproduces_controls_and_uses_full_125_fit_budget() -> None:
    rows = tuple(
        row
        for fold_index in range(1, 6)
        for question_index in range(10)
        for row in (
            _row(
                f"q{fold_index}_{question_index}",
                "baseline",
                family="baseline",
                fold_id=f"fold_{fold_index}",
            ),
            _row(
                f"q{fold_index}_{question_index}",
                "strict",
                strict=True,
                citation_delta=1,
                f1_delta=0.1,
                fold_id=f"fold_{fold_index}",
            ),
            _row(
                f"q{fold_index}_{question_index}",
                "unsafe",
                citation_delta=-1,
                f1_delta=-0.1,
                fold_id=f"fold_{fold_index}",
            ),
        )
    )
    source = _source_spec(prefix=3)
    stage198 = {
        "frozen_protocol": {
            "source_trajectory_contract": {
                "trajectories": [
                    {
                        "outer_context": f"fold_{index}",
                        "source_spec": stage196._spec_dict(source),
                    }
                    for index in range(1, 6)
                ]
            }
        }
    }
    references = build_stage182_reference_rows(rows, ())
    source_outer = {}
    control_spec = _policy("source_weighted_classifier", "gain_only")
    for outer_index in range(1, 6):
        outer_fold = f"fold_{outer_index}"
        inner_fold_ids = tuple(f"fold_{index}" for index in range(1, 6) if index != outer_index)
        inner_rows = tuple(row for row in rows if row.fold_id != outer_fold)
        result = _fake_fit_predictor((), inner_rows, {}, source)
        decisions, diagnostics = policy.evaluate_joint_risk_winner_policy(
            result.safety_predictions,
            result.gain_predictions,
            result.classifier_risk_predictions,
            result.pairwise_safety_predictions,
            source,
            control_spec,
            expected_fold_ids=inner_fold_ids,
        )
        evaluation = evaluate_selected_actions(
            selected_rows=tuple(decision.winner.row for decision in decisions),
            references=references,
            expected_fold_ids=inner_fold_ids,
        )
        source_outer[outer_fold] = {
            "top_inner_evaluation": evaluation,
            "top_inner_diagnostics": diagnostics,
        }
    stage197 = {"surviving_unsafe_winner_attribution": {"outer_contexts": source_outer}}

    report = policy.run_joint_risk_winner_nested_cv(
        action_rows=rows,
        stage182_selected_actions=(),
        stage198_protocol=stage198,
        stage197_report=stage197,
        partition_fit_predictor=_fake_fit_predictor,
    )

    assert report["execution"]["model_fit_count"] == 125
    assert report["execution"]["tree_count"] == 22_500
    assert report["execution"]["private_prediction_count"] == 3_750
    assert report["execution"]["all_controls_reproduced_exactly"] is True
    assert len(report["cell_aggregates"]) == 28
    assert all(row["outer_evaluated"] for row in report["outer_contexts"].values())


def _predictions():
    rows = (
        _row("q1", "baseline", family="baseline"),
        _row("q1", "strict", strict=True, citation_delta=1, f1_delta=0.1),
        _row("q1", "unsafe", f1_delta=-0.1),
        _row("q1", "safe_zero"),
    )
    safety = tuple(stage194.SafetyPrediction(row, 0.1, 0.1) for row in rows)
    gain_scores = {"baseline": 0.0, "strict": 0.9, "unsafe": 1.0, "safe_zero": 0.2}
    risk_scores = {"baseline": 0.0, "strict": 0.1, "unsafe": 0.9, "safe_zero": 0.2}
    pairwise_scores = {"baseline": 1.0, "strict": 0.9, "unsafe": 0.0, "safe_zero": 0.8}
    gain = tuple(stage196.GainPrediction(row, gain_scores[row.action.action_id]) for row in rows)
    classifier = tuple(
        stage196.UnsafePrediction(row, risk_scores[row.action.action_id]) for row in rows
    )
    pairwise = tuple(
        policy.PairwiseSafetyPrediction(row, pairwise_scores[row.action.action_id]) for row in rows
    )
    return rows, safety, gain, classifier, pairwise


def _source_spec(*, prefix: int) -> stage196.SafetyFirstFrontierPolicySpec:
    return stage196.SafetyFirstFrontierPolicySpec(
        "source",
        "raw_runtime",
        "class_balanced_logistic",
        "raw_runtime",
        "conservative",
        "raw_runtime",
        "conservative",
        1.0,
        prefix,
    )


def _policy(risk_signal: policy.RiskSignal, winner_name: str) -> policy.JointRiskWinnerPolicySpec:
    return next(
        row
        for row in policy.stage199_policy_specs()
        if row.risk_signal == risk_signal and row.winner_rule.name == winner_name
    )


def _row(
    question_key: str,
    action_id: str,
    *,
    family: str = "test",
    strict: bool = False,
    citation_delta: int = 0,
    f1_delta: float = 0.0,
    fold_id: str = "fold_1",
) -> ActionAuditRow:
    return ActionAuditRow(
        question_key=question_key,
        fold_id=fold_id,
        route="other",
        action=CompositionAction(
            action_id=action_id,
            family=family,
            aliases=(),
            selected_indices=(0,),
            matches_stage180=False,
        ),
        runtime_features={"score": float(strict)},
        outcome_class="test",
        strict_expected=strict,
        citation_delta=citation_delta,
        f1_delta=f1_delta,
    )


def _fake_fit_predictor(_training, heldout, _feature_indices, _source_spec):
    safety = tuple(
        stage194.SafetyPrediction(
            row,
            0.01 if row.strict_expected else 0.9 if row.action.action_id == "unsafe" else 0.1,
            0.01 if row.strict_expected else 0.9 if row.action.action_id == "unsafe" else 0.1,
        )
        for row in heldout
    )
    gain = tuple(
        stage196.GainPrediction(
            row,
            1.0 if row.strict_expected else -1.0 if row.action.action_id == "unsafe" else 0.0,
        )
        for row in heldout
    )
    classifier = tuple(
        stage196.UnsafePrediction(
            row,
            0.0 if row.strict_expected else 0.9 if row.action.action_id == "unsafe" else 0.1,
        )
        for row in heldout
    )
    pairwise = tuple(
        policy.PairwiseSafetyPrediction(
            row,
            1.0 if row.strict_expected else 0.0 if row.action.action_id == "unsafe" else 0.5,
        )
        for row in heldout
    )
    return policy.JointPartitionResult(
        safety,
        gain,
        classifier,
        pairwise,
        {"raw_runtime": 2},
        5,
        900,
        2,
    )
