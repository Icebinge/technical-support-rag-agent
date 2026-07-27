from __future__ import annotations

from ts_rag_agent.application import composition_rank_capped_safety_pool as pool
from ts_rag_agent.application.composition_action_audit import (
    ActionAuditRow,
    CompositionAction,
)
from ts_rag_agent.application.composition_gain_sensitive_ranking import (
    GainSensitivePrediction,
)


def test_stage191_policy_grid_is_complete_and_unique() -> None:
    specs = pool.stage191_policy_specs()

    assert len(specs) == 32
    assert len({spec.name for spec in specs}) == 32
    assert {spec.pool_cap for spec in specs} == {4, 8, 16, "all"}
    assert {spec.feature_representation for spec in specs} == {
        "raw_runtime",
        "question_relative_runtime",
    }


def test_safety_pool_caps_by_risk_unions_baseline_and_uses_gain_inside_pool() -> None:
    predictions = (
        _prediction(_row("q1", "baseline", family="baseline"), risk=0.90, gain=100.0),
        _prediction(_row("q1", "a"), risk=0.10, gain=1.0),
        _prediction(_row("q1", "b"), risk=0.20, gain=2.0),
        _prediction(_row("q1", "c"), risk=0.30, gain=50.0),
    )
    spec = _spec(pool_cap=2)

    decision = pool.build_rank_capped_safety_pool_decisions(predictions, spec)[0]

    assert [row.row.action.action_id for row in decision.pool] == ["a", "b", "baseline"]
    assert decision.winner.row.action.action_id == "baseline"


def test_pool_recall_is_question_level_and_reported_by_fold() -> None:
    predictions = (
        _prediction(_row("q1", "baseline", family="baseline", fold_id="fold_1"), 0.1, 0.0),
        _prediction(_row("q1", "strict", strict=True, fold_id="fold_1"), 0.2, 1.0),
        _prediction(_row("q2", "baseline", family="baseline", fold_id="fold_2"), 0.1, 0.0),
        _prediction(_row("q2", "unsafe", fold_id="fold_2"), 0.2, 1.0),
    )

    metrics = pool.evaluate_rank_capped_safety_pool(
        predictions,
        _spec(pool_cap=4),
        expected_fold_ids=("fold_1", "fold_2"),
    )

    assert metrics["strict_opportunity_question_count"] == 1
    assert metrics["recalled_strict_opportunity_question_count"] == 1
    assert metrics["strict_opportunity_pool_recall"] == 1.0
    assert metrics["folds"]["fold_1"]["strict_opportunity_pool_recall"] == 1.0
    assert metrics["folds"]["fold_2"]["strict_opportunity_pool_recall"] == 0.0
    assert metrics["baseline_in_pool_rate"] == 1.0


def test_nested_cv_uses_frozen_fit_budget_and_pool_gate() -> None:
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

    report = pool.run_rank_capped_safety_pool_nested_cv(
        action_rows=rows,
        stage182_selected_actions=(),
        representation_fitter=_fake_representation_fitter,
    )

    assert report["execution"]["model_fit_count"] == 300
    assert report["protocol"]["policy_config_count"] == 32
    assert all(row["outer_evaluated"] for row in report["outer_folds"].values())
    assert report["aggregate_pool_metrics"]["strict_opportunity_pool_recall"] == 1.0
    assert report["aggregate_pool_metrics"]["baseline_in_pool_rate"] == 1.0
    assert report["aggregate"]["strict_success_count"] == 50
    assert len(report["advancement_gates"]) == 15
    gates = {row["name"]: row["passed"] for row in report["advancement_gates"]}
    assert gates["strict_opportunity_pool_recall_at_least_0_80"] is True
    assert gates["stage182_regression_repair_rate_at_least_0_50"] is False
    assert report["candidate_family_accepted"] is False


def _spec(*, pool_cap: pool.PoolCap) -> pool.RankCappedSafetyPoolPolicySpec:
    return pool.RankCappedSafetyPoolPolicySpec(
        name=f"test_pool_{pool_cap}",
        feature_representation="raw_runtime",
        safety_estimator="class_balanced_logistic",
        gain_ranker="pairwise_pareto_logistic",
        pool_cap=pool_cap,
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


class _FakeRepresentation:
    def __init__(self, feature_representation: str) -> None:
        self.feature_representation = feature_representation
        self.feature_count = 1
        self.model_fit_count = 6
        self.diagnostics = {
            "pairwise": {
                "comparable_pair_count": 1,
                "omitted_incomparable_pair_count": 0,
            },
            "listnet": {"question_count": 1, "completed_iterations": 1},
        }

    def predict(self, rows, _feature_index):
        result = {}
        for estimator in ("class_balanced_logistic", "histogram_gradient_boosting"):
            for ranker in ("pairwise_pareto_logistic", "linear_listnet_top_frontier"):
                name = f"{self.feature_representation}__{estimator}__{ranker}"
                result[name] = tuple(
                    GainSensitivePrediction(
                        row=row,
                        citation_loss_probability=0.1 if row.strict_expected else 0.2,
                        f1_loss_probability=0.1 if row.strict_expected else 0.2,
                        gain_score=1.0 if row.strict_expected else 0.0,
                    )
                    for row in rows
                )
        return result


def _fake_representation_fitter(_rows, _feature_indices):
    return {
        name: _FakeRepresentation(name) for name in ("raw_runtime", "question_relative_runtime")
    }
