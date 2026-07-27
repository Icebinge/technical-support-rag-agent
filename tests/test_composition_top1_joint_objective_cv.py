from __future__ import annotations

import numpy as np
import pytest

from ts_rag_agent.application import composition_safety_constrained_lambdamart as stage194
from ts_rag_agent.application import composition_safety_first_frontier as stage196
from ts_rag_agent.application import composition_top1_joint_objective_cv as policy
from ts_rag_agent.application.composition_action_audit import ActionAuditRow, CompositionAction
from ts_rag_agent.application.composition_gain_sensitive_ranking import (
    build_stage182_reference_rows,
)
from ts_rag_agent.application.composition_joint_constraint_ranking import (
    evaluate_selected_actions,
)


def test_stage203_objective_grid_is_complete_and_unique() -> None:
    specs = policy.stage203_objective_specs()

    assert len(specs) == 16
    assert len({spec.name for spec in specs}) == 16
    assert {(spec.safety_weight, spec.precision_weight) for spec in specs} == {
        (safety, precision)
        for safety in policy._SAFETY_WEIGHTS
        for precision in policy._PRECISION_WEIGHTS
    }
    assert {spec.ablation_family for spec in specs} == {
        "strict_only",
        "safety_only",
        "precision_only",
        "full_joint",
    }


def test_grouped_target_exactly_mixes_capture_safety_and_precision() -> None:
    labels = np.asarray(
        [
            policy._BASELINE_LABEL,
            policy._STRICT_LABEL,
            policy._UNSAFE_LABEL,
            policy._SAFE_ZERO_LABEL,
            policy._BASELINE_LABEL,
            policy._SAFE_ZERO_LABEL,
            policy._UNSAFE_LABEL,
        ],
        dtype=np.int8,
    )

    target = policy.build_grouped_top1_target(
        labels=labels,
        group_sizes=[4, 3],
        safety_weight=1.0,
        precision_weight=1.0,
    )

    assert target == pytest.approx([5 / 18, 11 / 18, 0, 1 / 9, 5 / 6, 1 / 6, 0])
    assert target[:4].sum() == pytest.approx(1.0)
    assert target[4:].sum() == pytest.approx(1.0)


def test_strict_only_target_uses_baseline_when_group_has_no_strict_action() -> None:
    labels = [
        policy._BASELINE_LABEL,
        policy._SAFE_ZERO_LABEL,
        policy._UNSAFE_LABEL,
    ]

    target = policy.build_grouped_top1_target(
        labels=labels,
        group_sizes=[3],
        safety_weight=0.0,
        precision_weight=0.0,
    )

    assert target.tolist() == [1.0, 0.0, 0.0]


def test_grouped_objective_returns_finite_positive_derivatives() -> None:
    labels = np.asarray(
        [policy._BASELINE_LABEL, policy._STRICT_LABEL, policy._UNSAFE_LABEL] * 2,
        dtype=np.int8,
    )
    groups = np.asarray([3, 3], dtype=np.int32)
    weights = np.full(6, 1 / 3, dtype=np.float64)
    objective = policy.GroupedTop1Objective(
        labels=labels,
        group_sizes=groups,
        sample_weights=weights,
        spec=policy.Top1ObjectiveSpec("joint", 1.0, 1.0, "full_joint"),
    )

    gradient, hessian = objective(labels, np.zeros(6), weights, groups)

    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(hessian))
    assert np.all(hessian > 0)
    assert gradient[:3].sum() == pytest.approx(0.0)
    assert gradient[3:].sum() == pytest.approx(0.0)
    assert objective.diagnostics()["callback_call_count"] == 1
    assert objective.diagnostics()["target_sum_max_error"] == 0.0


def test_group_contract_rejects_multiple_baselines() -> None:
    with pytest.raises(ValueError, match="exactly one baseline"):
        policy.build_grouped_top1_target(
            labels=[policy._BASELINE_LABEL, policy._BASELINE_LABEL, policy._STRICT_LABEL],
            group_sizes=[3],
            safety_weight=1.0,
            precision_weight=1.0,
        )


def test_top1_decision_scores_complete_pool_without_source_frontier() -> None:
    rows = (
        _row("q1", "baseline", family="baseline"),
        _row("q1", "strict", strict=True, citation_delta=1, f1_delta=0.1),
        _row("q1", "unsafe", citation_delta=-1),
        _row("q1", "safe_zero"),
    )
    safety = tuple(stage194.SafetyPrediction(row, 0.1, 0.1) for row in rows)
    scores = {"baseline": 0.0, "strict": 2.0, "unsafe": 1.0, "safe_zero": 0.5}
    predictions = tuple(
        policy.Top1ScorePrediction(row, scores[row.action.action_id]) for row in rows
    )

    decision = policy.build_top1_decisions(safety, predictions)[0]

    assert decision.winner.row.action.action_id == "strict"
    assert decision.frontier == decision.complete_pool
    assert len(decision.complete_pool) == 4
    assert decision.baseline.row.action.family == "baseline"


def test_real_partition_fits_source_and_requested_custom_objectives() -> None:
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
            _row(f"train_{question_index}", "citation_loss", citation_delta=-1),
            _row(f"train_{question_index}", "f1_loss", f1_delta=-0.1),
            _row(f"train_{question_index}", "safe_zero"),
        )
    )
    heldout = tuple(
        row
        for question_index in range(10)
        for row in (
            _row(f"heldout_{question_index}", "baseline", family="baseline"),
            _row(
                f"heldout_{question_index}",
                "strict",
                strict=True,
                citation_delta=1,
                f1_delta=0.1,
            ),
            _row(f"heldout_{question_index}", "citation_loss", citation_delta=-1),
            _row(f"heldout_{question_index}", "f1_loss", f1_delta=-0.1),
            _row(f"heldout_{question_index}", "safe_zero"),
        )
    )
    feature_index = {
        (row.question_key, row.action.action_id): {
            "strict_hint": float(row.strict_expected),
            "unsafe_hint": float(stage194._is_unsafe(row)),
            "baseline_hint": float(row.action.family == "baseline"),
        }
        for row in (*training, *heldout)
    }
    specs = (policy.stage203_objective_specs()[0], policy.stage203_objective_specs()[-1])

    result = policy.fit_predict_top1_partition(
        training,
        heldout,
        {"raw_runtime": feature_index, "question_relative_runtime": feature_index},
        _source_spec(prefix=4),
        specs,
    )

    assert result.model_fit_count == 6
    assert result.source_model_fit_count == 4
    assert result.custom_objective_fit_count == 2
    assert 602 <= result.tree_count <= 1200
    assert result.source_tree_count == 600
    assert 2 <= result.custom_objective_tree_count <= 600
    assert result.group_contract_validation_count == 3
    assert 2 <= result.objective_callback_call_count <= 600
    assert set(result.objective_predictions) == {spec.name for spec in specs}
    assert all(len(values) == len(heldout) for values in result.objective_predictions.values())


def test_nested_cv_reproduces_controls_and_uses_full_425_fit_budget() -> None:
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
    stage202 = {
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
        result = _fake_fit_predictor((), inner_rows, {}, source, policy.stage203_objective_specs())
        decisions = stage196.build_safety_first_frontier_decisions(
            result.safety_predictions,
            result.gain_predictions,
            result.risk_predictions,
            source,
        )
        control_evaluation = evaluate_selected_actions(
            selected_rows=tuple(decision.winner.row for decision in decisions),
            references=references,
            expected_fold_ids=inner_fold_ids,
        )
        control_diagnostics = policy.evaluate_top1_decisions(
            decisions,
            expected_fold_ids=inner_fold_ids,
        )
        source_outer[outer_fold] = {
            "control_reproduction_exact": True,
            "top_inner_candidates": [
                {
                    "spec": {
                        "risk_signal": "source_weighted_classifier",
                        "winner_rule": "gain_only",
                    },
                    "evaluation": control_evaluation,
                    "diagnostics": control_diagnostics,
                }
            ],
        }
    stage199 = {"joint_risk_winner_nested_cv": {"outer_contexts": source_outer}}

    report = policy.run_top1_joint_objective_nested_cv(
        action_rows=rows,
        stage182_selected_actions=(),
        stage202_protocol=stage202,
        stage199_report=stage199,
        partition_fit_predictor=_fake_fit_predictor,
    )

    assert report["execution"]["model_fit_count"] == 425
    assert report["execution"]["custom_objective_fit_count"] == 325
    assert report["execution"]["outer_custom_objective_refit_count"] == 5
    assert report["execution"]["tree_count"] == 112_500
    assert report["execution"]["group_contract_validation_count"] == 350
    assert report["execution"]["objective_callback_call_count"] == 97_500
    assert report["execution"]["private_prediction_count"] == 12_750
    assert report["execution"]["all_controls_reproduced_exactly"] is True
    assert len(report["cell_aggregates"]) == 17
    assert all(row["outer_evaluated"] for row in report["outer_contexts"].values())
    assert report["directional_penalty_response"]["safety_adjacent_comparison_count"] == 12
    assert report["directional_penalty_response"]["precision_adjacent_comparison_count"] == 12
    assert report["candidate_family_accepted"] == all(
        row["passed"] for row in report["advancement_gates"]
    )


def test_stage199_control_evidence_rejects_missing_control_candidate() -> None:
    with pytest.raises(ValueError, match="exactly one published Stage199 control"):
        policy._stage199_control_evidence(
            {
                "control_reproduction_exact": True,
                "top_inner_candidates": [],
            }
        )


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


def _fake_fit_predictor(_training, heldout, _feature_indices, _source_spec, objective_specs):
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
    objective = {
        spec.name: tuple(
            policy.Top1ScorePrediction(
                row,
                1.0 if row.strict_expected else 0.0 if row.action.family == "baseline" else -1.0,
            )
            for row in heldout
        )
        for spec in objective_specs
    }
    return policy.Top1PartitionResult(
        safety_predictions=safety,
        gain_predictions=gain,
        risk_predictions=risk,
        objective_predictions=objective,
        feature_count_by_representation={"raw_runtime": 2},
        model_fit_count=4 + len(objective_specs),
        source_model_fit_count=4,
        custom_objective_fit_count=len(objective_specs),
        tree_count=600 + 300 * len(objective_specs),
        source_tree_count=600,
        custom_objective_tree_count=300 * len(objective_specs),
        group_contract_validation_count=1 + len(objective_specs),
        objective_callback_call_count=300 * len(objective_specs),
    )
