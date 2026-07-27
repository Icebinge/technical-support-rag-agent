from __future__ import annotations

from ts_rag_agent.application import (
    composition_safety_constrained_lambdamart as stage194,
)
from ts_rag_agent.application import composition_safety_first_frontier as policy
from ts_rag_agent.application.composition_action_audit import ActionAuditRow, CompositionAction


def test_stage196_policy_grid_is_complete_decoupled_and_unique() -> None:
    specs = policy.stage196_policy_specs()

    assert len(specs) == 960
    assert len({spec.name for spec in specs}) == 960
    assert {spec.scale_pos_weight for spec in specs} == {1.0, 2.0, 4.0}
    assert {spec.safest_prefix_size for spec in specs} == {2, 4, 8, 12, 16}
    assert any(
        spec.gain_feature_representation != spec.risk_feature_representation
        and spec.gain_tree_profile != spec.risk_tree_profile
        for spec in specs
    )


def test_safest_prefix_unions_baseline_then_selects_highest_gain() -> None:
    rows = (
        _row("q1", "baseline", family="baseline"),
        *(_row("q1", f"a{index}", strict=index == 1) for index in range(1, 18)),
    )
    safety = tuple(
        stage194.SafetyPrediction(
            row,
            0.99 if row.action.family == "baseline" else index / 100,
            0.99 if row.action.family == "baseline" else index / 100,
        )
        for index, row in enumerate(rows)
    )
    gain = tuple(
        policy.GainPrediction(row, 10.0 if row.action.action_id == "a1" else 0.0) for row in rows
    )
    risk = tuple(
        policy.UnsafePrediction(row, 0.0 if row.action.action_id == "a1" else 1.0) for row in rows
    )

    decision = policy.build_safety_first_frontier_decisions(
        safety,
        gain,
        risk,
        _spec(prefix=2),
    )[0]

    assert len(decision.complete_pool) == 17
    assert len(decision.frontier) == 3
    assert decision.baseline in decision.frontier
    assert decision.winner.row.action.action_id == "a1"


def test_frontier_diagnostics_expose_filtered_strict_and_unsafe_actions() -> None:
    rows = (
        _row("q1", "baseline", family="baseline", fold_id="fold_1"),
        _row("q1", "strict", strict=True, citation_delta=1, fold_id="fold_1"),
        _row("q1", "unsafe", f1_delta=-0.2, fold_id="fold_1"),
    )
    safety = tuple(stage194.SafetyPrediction(row, 0.1, 0.1) for row in rows)
    gain = tuple(policy.GainPrediction(row, float(row.strict_expected)) for row in rows)
    risk = tuple(
        policy.UnsafePrediction(
            row,
            {"baseline": 0.0, "unsafe": 0.1, "strict": 0.9}[row.action.action_id],
        )
        for row in rows
    )

    _, report = policy.evaluate_safety_first_frontier_policy(
        safety,
        gain,
        risk,
        _spec(prefix=2),
        expected_fold_ids=("fold_1",),
    )

    assert report["strict_opportunity_pool_recall"] == 1.0
    assert report["strict_opportunity_frontier_recall"] == 0.0
    assert report["unsafe_action_retention_rate"] == 1.0
    assert report["baseline_in_frontier_rate"] == 1.0


def test_nested_cv_uses_frozen_600_fit_budget() -> None:
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
        )
    )

    report = policy.run_safety_first_frontier_nested_cv(
        action_rows=rows,
        stage182_selected_actions=(),
        representation_fit_predictor=_fake_fit_predictor,
    )

    assert report["execution"]["model_fit_count"] == 600
    assert report["execution"]["pool_safety_fit_count"] == 200
    assert report["execution"]["lambdamart_fit_count"] == 100
    assert report["execution"]["unsafe_head_fit_count"] == 300
    assert report["execution"]["tree_count"] == 120_000
    assert report["protocol"]["policy_config_count"] == 960
    assert all(row["outer_evaluated"] for row in report["outer_folds"].values())
    assert report["aggregate_diagnostics"]["strict_opportunity_frontier_recall"] == 1.0
    assert report["aggregate_diagnostics"]["unsafe_selection_rate"] == 0.0
    assert len(report["advancement_gates"]) == 17


def test_real_lightgbm_partition_fit_covers_all_weighted_heads() -> None:
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
            _row(
                f"heldout_{question_index}",
                "strict",
                strict=True,
                citation_delta=1,
                f1_delta=0.1,
            ),
            _row(
                f"heldout_{question_index}",
                "unsafe",
                citation_delta=-1,
                f1_delta=-0.1,
            ),
        )
    )
    feature_index = {
        (row.question_key, row.action.action_id): {
            "score": row.runtime_features["score"],
            "unsafe_hint": float(row.action.action_id == "unsafe"),
        }
        for row in (*training, *heldout)
    }

    result = policy.fit_predict_safety_first_representation(
        training,
        heldout,
        feature_index,
        "raw_runtime",
    )

    assert result.model_fit_count == 12
    assert result.tree_count == 2_400
    assert result.group_contract_validation_count == 1
    assert set(result.predictions.risk_by_profile_and_weight) == {
        "conservative__1.0",
        "conservative__2.0",
        "conservative__4.0",
        "moderate__1.0",
        "moderate__2.0",
        "moderate__4.0",
    }
    assert all(
        len(rows) == len(heldout) for rows in result.predictions.risk_by_profile_and_weight.values()
    )


def test_real_focused_spec_fit_trains_only_four_required_models() -> None:
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
            "score": row.runtime_features["score"],
            "unsafe_hint": float(row.action.action_id == "unsafe"),
        }
        for row in (*training, *heldout)
    }

    result = policy.fit_predict_safety_first_spec(
        training,
        heldout,
        {"raw_runtime": feature_index, "question_relative_runtime": feature_index},
        _spec(prefix=4),
    )

    assert result.model_fit_count == 4
    assert result.tree_count == 600
    assert result.group_contract_validation_count == 1
    assert len(result.safety_predictions) == len(heldout)
    assert len(result.gain_predictions) == len(heldout)
    assert len(result.risk_predictions) == len(heldout)


def _spec(*, prefix: int) -> policy.SafetyFirstFrontierPolicySpec:
    return policy.SafetyFirstFrontierPolicySpec(
        name="test",
        pool_feature_representation="raw_runtime",
        pool_safety_estimator="class_balanced_logistic",
        gain_feature_representation="raw_runtime",
        gain_tree_profile="conservative",
        risk_feature_representation="raw_runtime",
        risk_tree_profile="conservative",
        scale_pos_weight=1.0,
        safest_prefix_size=prefix,
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


def _fake_fit_predictor(_training, heldout, _feature_index, _representation):
    safety = {
        estimator: tuple(
            stage194.SafetyPrediction(
                row,
                0.01 if row.strict_expected else 0.1,
                0.01 if row.strict_expected else 0.1,
            )
            for row in heldout
        )
        for estimator in ("class_balanced_logistic", "histogram_gradient_boosting")
    }
    gain = {
        profile: tuple(
            policy.GainPrediction(row, 1.0 if row.strict_expected else 0.0) for row in heldout
        )
        for profile in ("conservative", "moderate")
    }
    risk = {
        f"{profile}__{weight:.1f}": tuple(
            policy.UnsafePrediction(row, 0.01 if row.strict_expected else 0.1) for row in heldout
        )
        for profile in ("conservative", "moderate")
        for weight in (1.0, 2.0, 4.0)
    }
    return policy.RepresentationPartitionResult(
        predictions=policy.RepresentationPredictions(safety, gain, risk),
        feature_count=1,
        model_fit_count=12,
        tree_count=2_400,
        group_contract_validation_count=1,
    )
