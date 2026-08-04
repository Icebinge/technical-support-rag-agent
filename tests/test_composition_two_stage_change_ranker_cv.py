from __future__ import annotations

import pytest

from ts_rag_agent.application import composition_safety_constrained_lambdamart as stage194
from ts_rag_agent.application import composition_safety_first_frontier as stage196
from ts_rag_agent.application import composition_two_stage_change_ranker_cv as policy
from ts_rag_agent.application.composition_action_audit import ActionAuditRow, CompositionAction
from ts_rag_agent.application.composition_f1_representation_cv import (
    build_composition_feature_indices,
)
from ts_rag_agent.application.composition_gain_sensitive_ranking import (
    build_stage182_reference_rows,
)
from ts_rag_agent.application.composition_joint_constraint_ranking import (
    evaluate_selected_actions,
)


def test_stage206_policy_grid_is_complete_and_unique() -> None:
    specs = policy.stage206_policy_specs()

    assert len(specs) == 10
    assert len({spec.name for spec in specs}) == 10
    assert {spec.ranker_family for spec in specs} == {
        "strict_binary",
        "strict_safety_graded",
    }
    assert {spec.target_change_coverage for spec in specs} == {
        0.25,
        0.40,
        0.55,
        0.70,
        0.85,
    }


def test_gate_crossfit_assignment_is_deterministic_and_populates_four_folds() -> None:
    first = {
        question: policy.gate_crossfit_index(
            question,
            outer_context="fold_1",
            heldout_context="fold_2",
        )
        for question in (f"question_{index}" for index in range(200))
    }
    second = {
        question: policy.gate_crossfit_index(
            question,
            outer_context="fold_1",
            heldout_context="fold_2",
        )
        for question in first
    }

    assert first == second
    assert set(first.values()) == {0, 1, 2, 3}
    assert any(
        value
        != policy.gate_crossfit_index(
            question,
            outer_context="fold_1",
            heldout_context="fold_3",
        )
        for question, value in first.items()
    )


def test_conditional_candidate_excludes_baseline_and_gate_excludes_raw_rank_score() -> None:
    rows = (
        _row("q1", "baseline", family="baseline"),
        _row("q1", "strict", strict=True, citation_delta=1, f1_delta=0.1),
        _row("q1", "safe_zero"),
        _row("q1", "unsafe", citation_delta=-1),
    )
    safety = tuple(stage194.SafetyPrediction(row, 0.1, 0.1) for row in rows)
    ranker = tuple(
        policy.RankerScorePrediction(row, score)
        for row, score in zip(rows[1:], (9.0, 3.0, -4.0), strict=True)
    )
    feature_indices = {
        representation: {
            (row.question_key, row.action.action_id): {"runtime_value": float(index)}
            for index, row in enumerate(rows)
        }
        for representation in ("raw_runtime", "question_relative_runtime")
    }

    candidate = policy._build_question_candidates(safety, ranker, feature_indices)[0]

    assert candidate.candidate.row.action.action_id == "strict"
    assert candidate.candidate.row.action.family != "baseline"
    assert candidate.baseline.row.action.family == "baseline"
    assert candidate.gate_features["ranker_normalized_top1_top2_margin"] == pytest.approx(6 / 13)
    assert "ranker_absolute_score" not in candidate.gate_features
    assert all("gain_score" not in key for key in candidate.gate_features)


def test_coverage_threshold_uses_training_order_statistic() -> None:
    scores = [0.9, 0.8, 0.7, 0.6]

    assert policy._coverage_threshold(scores, 0.25) == 0.9
    assert policy._coverage_threshold(scores, 0.40) == 0.8
    assert policy._coverage_threshold(scores, 0.85) == 0.6


def test_real_partition_enforces_strict_oof_fit_and_prediction_budgets() -> None:
    training = tuple(
        row
        for question_index in range(80)
        for row in _question_rows(f"train_{question_index}", question_index)
    )
    heldout = tuple(
        row
        for question_index in range(20)
        for row in _question_rows(f"heldout_{question_index}", question_index)
    )
    base_features = build_composition_feature_indices((*training, *heldout))
    feature_indices = {
        "raw_runtime": base_features["raw"],
        "question_relative_runtime": base_features["question_relative"],
    }
    policies = (
        policy.TwoStagePolicySpec("strict_binary__change_c25", "strict_binary", 0.25),
        policy.TwoStagePolicySpec(
            "strict_safety_graded__change_c25",
            "strict_safety_graded",
            0.25,
        ),
    )

    result = policy.fit_predict_two_stage_partition(
        training,
        heldout,
        feature_indices,
        _source_spec(),
        policies,
        "outer_smoke",
        "heldout_smoke",
    )

    assert result.model_fit_count == 24
    assert result.source_model_fit_count == 4
    assert result.source_safety_crossfit_fit_count == 8
    assert result.conditional_ranker_fit_count == 10
    assert result.gate_fit_count == 2
    assert result.lightgbm_model_fit_count == 14
    assert result.source_group_contract_validation_count == 1
    assert result.ranker_group_contract_validation_count == 10
    assert result.source_safety_oof_prediction_count == 800
    assert result.ranker_oof_prediction_count == 640
    assert result.gate_training_question_count == 160
    assert result.private_prediction_count == 2_200
    assert set(result.policy_decisions) == {row.name for row in policies}
    assert all(len(decisions) == 20 for decisions in result.policy_decisions.values())
    assert all(
        decision.complete_pool and decision.baseline.row.action.family == "baseline"
        for decisions in result.policy_decisions.values()
        for decision in decisions
    )


def test_nested_cv_reproduces_controls_and_uses_full_strict_budget() -> None:
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
    source = _source_spec()
    stage205 = {
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
    for outer_index in range(1, 6):
        outer_fold = f"fold_{outer_index}"
        inner_fold_ids = tuple(f"fold_{index}" for index in range(1, 6) if index != outer_index)
        inner_rows = tuple(row for row in rows if row.fold_id != outer_fold)
        result = _fake_fit_predictor((), inner_rows, {}, source, (), outer_fold, outer_fold)
        decisions = stage196.build_safety_first_frontier_decisions(
            result.safety_predictions,
            result.gain_predictions,
            result.risk_predictions,
            source,
        )
        source_outer[outer_fold] = {
            "control_reproduction_exact": True,
            "top_inner_candidates": [
                {
                    "spec": {
                        "risk_signal": "source_weighted_classifier",
                        "winner_rule": "gain_only",
                    },
                    "evaluation": evaluate_selected_actions(
                        selected_rows=tuple(decision.winner.row for decision in decisions),
                        references=references,
                        expected_fold_ids=inner_fold_ids,
                    ),
                    "diagnostics": policy.stage203.evaluate_top1_decisions(
                        decisions,
                        expected_fold_ids=inner_fold_ids,
                    ),
                }
            ],
        }
    stage199 = {"joint_risk_winner_nested_cv": {"outer_contexts": source_outer}}

    report = policy.run_two_stage_change_ranker_nested_cv(
        action_rows=rows,
        stage182_selected_actions=(),
        stage205_protocol=stage205,
        stage199_report=stage199,
        partition_fit_predictor=_fake_fit_predictor,
    )

    assert report["execution"]["model_fit_count"] == 570
    assert report["execution"]["source_safety_crossfit_fit_count"] == 200
    assert report["execution"]["conditional_ranker_fit_count"] == 225
    assert report["execution"]["gate_fit_count"] == 45
    assert report["execution"]["tree_count"] == 96_000
    assert report["execution"]["private_prediction_count"] == 13_350
    assert report["execution"]["all_controls_reproduced_exactly"] is True
    assert len(report["cell_aggregates"]) == 11
    assert all(row["outer_evaluated"] for row in report["outer_contexts"].values())


def _source_spec() -> stage196.SafetyFirstFrontierPolicySpec:
    return stage196.SafetyFirstFrontierPolicySpec(
        "source",
        "raw_runtime",
        "class_balanced_logistic",
        "question_relative_runtime",
        "conservative",
        "raw_runtime",
        "conservative",
        1.0,
        16,
    )


def _question_rows(question_key: str, question_index: int) -> tuple[ActionAuditRow, ...]:
    strict_signal = 1.0 if question_index % 2 == 0 else 0.0
    safe_signal = 1.0 - strict_signal
    return (
        _row(question_key, "baseline", family="baseline", rank_signal=0.0),
        _row(
            question_key,
            "strict",
            strict=True,
            citation_delta=1,
            f1_delta=0.1,
            rank_signal=strict_signal,
        ),
        _row(question_key, "safe_zero", rank_signal=safe_signal),
        _row(question_key, "citation_loss", citation_delta=-1, rank_signal=0.0),
        _row(question_key, "f1_loss", f1_delta=-0.1, rank_signal=0.0),
    )


def _row(
    question_key: str,
    action_id: str,
    *,
    family: str = "test",
    strict: bool = False,
    citation_delta: int = 0,
    f1_delta: float = 0.0,
    rank_signal: float = 0.0,
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
        runtime_features={
            "citation_loss_hint": float(citation_delta < 0),
            "f1_loss_hint": float(f1_delta < 0),
            "baseline_hint": float(family == "baseline"),
            "rank_signal": rank_signal,
        },
        outcome_class="test",
        strict_expected=strict,
        citation_delta=citation_delta,
        f1_delta=f1_delta,
    )


def _fake_fit_predictor(
    training,
    heldout,
    _feature_indices,
    _source_spec,
    policy_specs,
    _outer_context,
    _heldout_context,
):
    families = {spec.ranker_family for spec in policy_specs}
    family_count = len(families)
    safety = tuple(stage194.SafetyPrediction(row, 0.1, 0.1) for row in heldout)
    gain = tuple(
        stage196.GainPrediction(
            row,
            1.0 if row.action.family == "baseline" else 0.9 if row.strict_expected else 0.0,
        )
        for row in heldout
    )
    risk = tuple(
        stage196.UnsafePrediction(row, 0.9 if stage194._is_unsafe(row) else 0.1) for row in heldout
    )
    decisions = _strict_decisions(heldout)
    question_count = len({row.question_key for row in heldout})
    training_question_count = len({row.question_key for row in training})
    nonbaseline_training = sum(row.action.family != "baseline" for row in training)
    nonbaseline_heldout = sum(row.action.family != "baseline" for row in heldout)
    gate_diagnostics = {
        spec.name: {
            "ranker_family": spec.ranker_family,
            "target_change_coverage": spec.target_change_coverage,
            "training_gate": {"roc_auc": 0.8, "average_precision": 0.8},
            "heldout_gate": {"roc_auc": 0.8, "average_precision": 0.8},
            "heldout_question_count": question_count,
            "realized_change_count": question_count,
            "realized_change_coverage": 1.0,
            "pre_gate_ranker_strict_count": question_count,
            "pre_gate_ranker_strict_rate": 1.0,
            "pre_gate_ranker_unsafe_count": 0,
            "pre_gate_ranker_unsafe_rate": 0.0,
        }
        for spec in policy_specs
    }
    source_safety_oof = 2 * len(training) if family_count else 0
    ranker_oof = family_count * nonbaseline_training
    private_predictions = 4 * len(heldout)
    if family_count:
        private_predictions += (
            source_safety_oof
            + ranker_oof
            + family_count * training_question_count
            + family_count * nonbaseline_heldout
            + family_count * question_count
        )
    return policy.TwoStagePartitionResult(
        safety_predictions=safety,
        gain_predictions=gain,
        risk_predictions=risk,
        policy_decisions={spec.name: decisions for spec in policy_specs},
        gate_diagnostics=gate_diagnostics,
        feature_count_by_representation={"raw_runtime": 4},
        model_fit_count=4 + 8 * bool(family_count) + 6 * family_count,
        source_model_fit_count=4,
        source_safety_crossfit_fit_count=8 * bool(family_count),
        conditional_ranker_fit_count=5 * family_count,
        gate_fit_count=family_count,
        lightgbm_model_fit_count=2 + 6 * family_count,
        tree_count=600 + 1_800 * family_count,
        source_tree_count=600,
        conditional_ranker_tree_count=1_500 * family_count,
        gate_tree_count=300 * family_count,
        source_group_contract_validation_count=1,
        ranker_group_contract_validation_count=5 * family_count,
        source_safety_oof_prediction_count=source_safety_oof,
        ranker_oof_prediction_count=ranker_oof,
        gate_training_question_count=family_count * training_question_count,
        private_prediction_count=private_predictions,
    )


def _strict_decisions(rows) -> tuple[stage196.SafetyFirstFrontierDecision, ...]:
    grouped = stage194._group_rows(rows)
    decisions = []
    for question_key, question_rows in sorted(grouped.items()):
        predictions = tuple(
            stage196.FrontierActionPrediction(
                row,
                0.1,
                0.1,
                1.0 if row.strict_expected else 0.0,
                0.1,
            )
            for row in question_rows
        )
        baseline = next(row for row in predictions if row.row.action.family == "baseline")
        winner = next(row for row in predictions if row.row.strict_expected)
        decisions.append(
            stage196.SafetyFirstFrontierDecision(
                question_key=question_key,
                baseline=baseline,
                complete_pool=predictions,
                frontier=predictions,
                winner=winner,
                strict_opportunity=True,
                action_count=len(predictions),
                strict_action_count=1,
            )
        )
    return tuple(decisions)
