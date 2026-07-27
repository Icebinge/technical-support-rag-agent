from __future__ import annotations

from ts_rag_agent.application import composition_safety_constrained_lambdamart as policy
from ts_rag_agent.application.composition_action_audit import (
    ActionAuditRow,
    CompositionAction,
)


def test_stage194_policy_grid_is_complete_and_unique() -> None:
    specs = policy.stage194_policy_specs()

    assert len(specs) == 64
    assert len({spec.name for spec in specs}) == 64
    assert {spec.risk_penalty for spec in specs} == {0.25, 0.5, 1.0, 2.0}
    assert {spec.tree_profile for spec in specs} == {"conservative", "moderate"}
    assert {spec.pool_feature_representation for spec in specs} == {
        "raw_runtime",
        "question_relative_runtime",
    }


def test_selection_caps_pool_unions_baseline_and_combines_gain_with_risk() -> None:
    rows = (
        _row("q1", "baseline", family="baseline"),
        *(_row("q1", f"a{index}") for index in range(1, 18)),
    )
    safety = tuple(
        policy.SafetyPrediction(
            row,
            citation_loss_probability=(0.99 if row.action.family == "baseline" else index / 100),
            f1_loss_probability=(0.99 if row.action.family == "baseline" else index / 100),
        )
        for index, row in enumerate(rows)
    )
    reranker = tuple(
        policy.RerankerPrediction(
            row,
            gain_score=(100.0 if row.action.action_id == "a1" else 0.0),
            unsafe_probability=(0.0 if row.action.action_id == "a1" else 1.0),
        )
        for row in rows
    )

    decision = policy.build_safety_constrained_decisions(
        safety,
        reranker,
        _spec(risk_penalty=1.0),
    )[0]

    assert len(decision.pool) == 17
    assert any(row.row.action.family == "baseline" for row in decision.pool)
    assert decision.winner.row.action.action_id == "a1"


def test_policy_diagnostics_partition_strict_opportunities_and_unsafe_winners() -> None:
    rows = (
        _row("q1", "baseline", family="baseline", fold_id="fold_1"),
        _row("q1", "strict", strict=True, citation_delta=1, fold_id="fold_1"),
        _row("q2", "baseline", family="baseline", fold_id="fold_2"),
        _row("q2", "unsafe", f1_delta=-0.2, fold_id="fold_2"),
    )
    safety = tuple(policy.SafetyPrediction(row, 0.1, 0.1) for row in rows)
    reranker = tuple(
        policy.RerankerPrediction(
            row,
            gain_score=float(row.action.action_id in {"strict", "unsafe"}),
            unsafe_probability=float(row.action.action_id == "unsafe"),
        )
        for row in rows
    )

    _, report = policy.evaluate_safety_constrained_policy(
        safety,
        reranker,
        _spec(risk_penalty=0.25),
        expected_fold_ids=("fold_1", "fold_2"),
    )

    assert report["strict_opportunity_question_count"] == 1
    assert report["pool_recalled_question_count"] == 1
    assert report["strict_selected_question_count"] == 1
    assert report["strict_opportunity_pool_recall"] == 1.0
    assert report["conditional_ranker_strict_capture"] == 1.0
    assert report["unsafe_selected_question_count"] == 1
    assert report["unsafe_selection_rate"] == 0.5
    assert report["baseline_in_pool_rate"] == 1.0


def test_nested_cv_uses_frozen_400_fit_budget() -> None:
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

    report = policy.run_safety_constrained_lambdamart_nested_cv(
        action_rows=rows,
        stage182_selected_actions=(),
        representation_fitter=_fake_fitter,
    )

    assert report["execution"]["model_fit_count"] == 400
    assert report["execution"]["pool_safety_fit_count"] == 200
    assert report["execution"]["lambdamart_fit_count"] == 100
    assert report["execution"]["unsafe_head_fit_count"] == 100
    assert report["protocol"]["policy_config_count"] == 64
    assert all(row["outer_evaluated"] for row in report["outer_folds"].values())
    assert report["aggregate_diagnostics"]["strict_opportunity_pool_recall"] == 1.0
    assert report["aggregate_diagnostics"]["conditional_ranker_strict_capture"] == 1.0
    assert report["aggregate_diagnostics"]["unsafe_selection_rate"] == 0.0
    assert len(report["advancement_gates"]) == 17


def _spec(*, risk_penalty: float) -> policy.SafetyConstrainedLambdaMARTPolicySpec:
    return policy.SafetyConstrainedLambdaMARTPolicySpec(
        name="test",
        pool_feature_representation="raw_runtime",
        pool_safety_estimator="class_balanced_logistic",
        reranker_feature_representation="raw_runtime",
        tree_profile="conservative",
        risk_penalty=risk_penalty,
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


class _FakeRepresentation:
    def __init__(self, representation: policy.FeatureRepresentation) -> None:
        self.feature_representation = representation
        self.feature_count = 1
        self.model_fit_count = 8
        self.diagnostics = {"tree_count": 1_200}

    def predict(self, rows, _feature_index):
        safety = {
            estimator: tuple(
                policy.SafetyPrediction(
                    row,
                    0.01 if row.strict_expected else 0.1,
                    0.01 if row.strict_expected else 0.1,
                )
                for row in rows
            )
            for estimator in (
                "class_balanced_logistic",
                "histogram_gradient_boosting",
            )
        }
        reranker = {
            profile: tuple(
                policy.RerankerPrediction(
                    row,
                    1.0 if row.strict_expected else 0.0,
                    0.01 if row.strict_expected else 0.1,
                )
                for row in rows
            )
            for profile in ("conservative", "moderate")
        }
        return policy.RepresentationPredictions(safety, reranker)


def _fake_fitter(_rows, _feature_index, representation):
    return _FakeRepresentation(representation)
