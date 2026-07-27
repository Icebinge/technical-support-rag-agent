from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from ts_rag_agent.application import composition_gain_sensitive_ranking as ranking
from ts_rag_agent.application.composition_action_audit import (
    ActionAuditRow,
    CompositionAction,
)
from ts_rag_agent.application.composition_gain_sensitive_ranking import (
    GainSensitivePolicySpec,
    GainSensitivePrediction,
    build_listnet_training_target,
    build_pairwise_training_data,
    fit_gain_sensitive_representations,
    pairwise_preference,
    select_gain_sensitive_actions,
    stage188_policy_specs,
)


def test_stage188_policy_grid_is_complete_and_unique() -> None:
    specs = stage188_policy_specs()

    assert len(specs) == 32
    assert len({spec.name for spec in specs}) == 32
    assert {spec.feature_representation for spec in specs} == {
        "raw_runtime",
        "question_relative_runtime",
    }
    assert {spec.safety_estimator for spec in specs} == {
        "class_balanced_logistic",
        "histogram_gradient_boosting",
    }
    assert {spec.gain_ranker for spec in specs} == {
        "pairwise_pareto_logistic",
        "linear_listnet_top_frontier",
    }
    assert {spec.safety_frontier_margin for spec in specs} == {
        0.0,
        0.02,
        0.05,
        0.10,
    }


def test_pairwise_preference_uses_tiers_and_omits_tradeoffs() -> None:
    strict_citation = _row("q1", "strict_citation", citation_delta=1, f1_delta=0.0)
    strict_f1 = _row("q1", "strict_f1", citation_delta=0, f1_delta=0.2)
    safe_zero = _row("q1", "safe_zero", citation_delta=0, f1_delta=0.0)
    unsafe = _row("q1", "unsafe", citation_delta=0, f1_delta=-0.1)

    assert pairwise_preference(strict_citation, safe_zero) == 1
    assert pairwise_preference(safe_zero, unsafe) == 1
    assert pairwise_preference(unsafe, strict_f1) == -1
    assert pairwise_preference(strict_citation, strict_f1) == 0


def test_pairwise_training_emits_both_orientations_without_sampling() -> None:
    rows = (
        _row("q1", "baseline", citation_delta=0, f1_delta=0.0, family="baseline"),
        _row("q1", "strict", citation_delta=1, f1_delta=0.1),
        _row("q1", "tradeoff", citation_delta=0, f1_delta=0.2),
    )
    matrix = csr_matrix(np.asarray([[0.0], [1.0], [2.0]]))

    training = build_pairwise_training_data(rows, matrix)

    assert training.comparable_pair_count == 2
    assert training.omitted_incomparable_pair_count == 1
    assert training.matrix.shape == (4, 1)
    assert training.labels.tolist().count(1) == 2
    assert training.labels.tolist().count(0) == 2
    assert np.isclose(float(np.sum(training.weights)), 1.0)


def test_listnet_target_uses_highest_tier_pareto_frontier() -> None:
    rows = (
        _row("q1", "baseline", citation_delta=0, f1_delta=0.0, family="baseline"),
        _row("q1", "strict_dominated", citation_delta=0, f1_delta=0.1),
        _row("q1", "strict_frontier", citation_delta=1, f1_delta=0.1),
        _row("q2", "baseline", citation_delta=0, f1_delta=0.0, family="baseline"),
        _row("q2", "unsafe", citation_delta=0, f1_delta=-0.1),
    )

    target = build_listnet_training_target(rows)

    assert target.frontier_action_count == 2
    assert target.probabilities.tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]
    assert all(
        np.isclose(float(np.sum(target.probabilities[positions])), 1.0)
        for positions in target.question_positions
    )


def test_relative_safety_frontier_is_nonempty_and_gain_primary() -> None:
    low_citation = _row("q1", "a_low_citation", citation_delta=0, f1_delta=0.0)
    compromise = _row("q1", "b_compromise", citation_delta=1, f1_delta=0.1)
    low_f1 = _row("q1", "c_low_f1", citation_delta=0, f1_delta=0.0)
    predictions = (
        GainSensitivePrediction(low_citation, 0.10, 0.30, 0.1),
        GainSensitivePrediction(compromise, 0.16, 0.16, 0.9),
        GainSensitivePrediction(low_f1, 0.30, 0.10, 0.2),
    )
    spec = _spec(margin=0.05)

    selected = select_gain_sensitive_actions(predictions, spec)

    assert selected == (compromise,)


def test_shared_encoder_fits_safety_and_both_gain_rankers() -> None:
    rows = (
        _row(
            "q1",
            "baseline",
            citation_delta=0,
            f1_delta=0.0,
            family="baseline",
            score=0.0,
        ),
        _row("q1", "strict", citation_delta=1, f1_delta=0.2, score=2.0),
        _row(
            "q2",
            "baseline",
            citation_delta=0,
            f1_delta=0.0,
            family="baseline",
            score=0.0,
        ),
        _row("q2", "unsafe", citation_delta=-1, f1_delta=-0.2, score=-2.0),
    )
    base = ranking.build_composition_feature_indices(rows)
    feature_indices = {
        "raw_runtime": base["raw"],
        "question_relative_runtime": base["question_relative"],
    }

    representations = fit_gain_sensitive_representations(rows, feature_indices)

    assert set(representations) == {"raw_runtime", "question_relative_runtime"}
    assert sum(item.model_fit_count for item in representations.values()) == 12
    for representation in representations.values():
        predictions = representation.predict(
            rows,
            feature_indices[representation.feature_representation],
        )
        assert len(predictions) == 4
        for bundle_rows in predictions.values():
            assert len(bundle_rows) == len(rows)
            assert all(0.0 <= row.citation_loss_probability <= 1.0 for row in bundle_rows)
            assert all(0.0 <= row.f1_loss_probability <= 1.0 for row in bundle_rows)
            assert all(np.isfinite(row.gain_score) for row in bundle_rows)
        assert representation.diagnostics["pairwise"]["comparable_pair_count"] == 2
        assert representation.diagnostics["listnet"]["completed_iterations"] > 0


def test_nested_cv_uses_all_partitions_and_frozen_fit_budget() -> None:
    rows = tuple(
        row
        for fold_index in range(1, 6)
        for question_index in range(2)
        for row in (
            _row(
                f"q{fold_index}_{question_index}",
                "baseline",
                citation_delta=0,
                f1_delta=0.0,
                family="baseline",
                fold_id=f"fold_{fold_index}",
                score=0.0,
            ),
            _row(
                f"q{fold_index}_{question_index}",
                "strict",
                citation_delta=1,
                f1_delta=0.1,
                fold_id=f"fold_{fold_index}",
                score=1.0,
            ),
        )
    )

    report = ranking.run_gain_sensitive_nested_cv(
        action_rows=rows,
        stage182_selected_actions=(),
        representation_fitter=_fake_representation_fitter,
    )

    assert report["execution"]["model_fit_count"] == 300
    assert report["protocol"]["policy_config_count"] == 32
    assert all(row["outer_evaluated"] for row in report["outer_folds"].values())
    assert report["aggregate"]["strict_success_count"] == 10
    assert report["aggregate"]["strict_success_precision"] == 1.0
    assert report["aggregate"]["gold_citation_delta"] == 10
    assert report["aggregate"]["mean_f1_delta"] == 0.1


def _spec(*, margin: float) -> GainSensitivePolicySpec:
    return GainSensitivePolicySpec(
        name="test",
        feature_representation="raw_runtime",
        safety_estimator="class_balanced_logistic",
        gain_ranker="pairwise_pareto_logistic",
        safety_frontier_margin=margin,
    )


def _row(
    question_key: str,
    action_id: str,
    *,
    citation_delta: int,
    f1_delta: float,
    family: str = "test",
    score: float = 1.0,
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
        runtime_features={"score": score},
        outcome_class="test",
        strict_expected=(
            citation_delta >= 0 and f1_delta >= -1e-12 and (citation_delta > 0 or f1_delta > 1e-12)
        ),
        citation_delta=citation_delta,
        f1_delta=f1_delta,
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
            "listnet": {
                "question_count": 1,
                "completed_iterations": 1,
            },
        }

    def predict(self, rows, _feature_index):
        result = {}
        for estimator in (
            "class_balanced_logistic",
            "histogram_gradient_boosting",
        ):
            for ranker in (
                "pairwise_pareto_logistic",
                "linear_listnet_top_frontier",
            ):
                name = f"{self.feature_representation}__{estimator}__{ranker}"
                result[name] = tuple(
                    GainSensitivePrediction(
                        row=row,
                        citation_loss_probability=0.1,
                        f1_loss_probability=0.1,
                        gain_score=1.0 if row.action.action_id == "strict" else 0.0,
                    )
                    for row in rows
                )
        return result


def _fake_representation_fitter(_rows, _feature_indices):
    return {
        name: _FakeRepresentation(name) for name in ("raw_runtime", "question_relative_runtime")
    }
