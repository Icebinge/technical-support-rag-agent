from __future__ import annotations

from ts_rag_agent.application.composition_action_audit import (
    ActionAuditRow,
    CompositionAction,
)
from ts_rag_agent.application.composition_dual_target_policy import SelectedAction
from ts_rag_agent.application.composition_f1_representation_cv import (
    _question_relative_feature_index,
    run_f1_representation_cv,
    stage184_representation_specs,
)


def test_stage184_representation_grid_covers_four_target_kinds() -> None:
    specs = stage184_representation_specs()

    assert len(specs) == 8
    assert {spec.target_kind for spec in specs} == {
        "binary",
        "ordinal",
        "pairwise",
        "quantile",
    }
    assert {spec.feature_space for spec in specs} == {"raw", "question_relative"}


def test_question_relative_features_are_label_free_and_question_local() -> None:
    rows = (
        _row("fold_1", "q1", "a", signal=1.0, f1_delta=-0.2, citation_delta=1),
        _row("fold_1", "q1", "b", signal=3.0, f1_delta=0.2, citation_delta=1),
        _row("fold_1", "q2", "a", signal=10.0, f1_delta=-0.1, citation_delta=0),
        _row("fold_1", "q2", "b", signal=20.0, f1_delta=0.1, citation_delta=0),
    )

    features = _question_relative_feature_index(rows)

    assert features[("q1", "a")]["relative_delta_mean__signal"] == -1.0
    assert features[("q2", "a")]["relative_delta_mean__signal"] == -5.0
    assert features[("q1", "b")]["relative_percentile__signal"] == 1.0
    assert all("f1_delta" not in row and "citation_delta" not in row for row in features.values())


def test_five_fold_representation_cv_scores_all_candidates_without_public_rows() -> None:
    rows = []
    selected = []
    for fold_index in range(1, 6):
        fold_id = f"fold_{fold_index}"
        for question_index in range(2):
            question = f"q{fold_index}_{question_index}"
            risk = _row(
                fold_id,
                question,
                "risk",
                signal=3.0,
                f1_delta=-0.2,
                citation_delta=1,
            )
            rows.extend(
                [
                    risk,
                    _row(
                        fold_id,
                        question,
                        "safe",
                        signal=1.0,
                        f1_delta=0.1,
                        citation_delta=1,
                    ),
                    _row(
                        fold_id,
                        question,
                        "gain",
                        signal=0.0,
                        f1_delta=0.2,
                        citation_delta=0,
                    ),
                ]
            )
            if question_index == 0:
                selected.append(
                    SelectedAction(
                        row=risk,
                        utility=0.7,
                        citation_gain_probability=0.8,
                        f1_regression_probability=0.1,
                    )
                )

    result = run_f1_representation_cv(
        action_rows=rows,
        stage182_selected_actions=selected,
    )

    assert result["dataset"]["action_count"] == 30
    assert result["dataset"]["stage182_selected_regression_count"] == 5
    assert len(result["representations"]) == 8
    assert result["execution"]["model_fit_count"] == 60
    assert result["execution"]["private_prediction_count"] == 240
    assert result["execution"]["public_prediction_rows_written"] == 0
    assert all(
        report["stage182_regression_headroom"]["safe_alternative_top3_count"] == 5
        for report in result["representations"].values()
    )


def _row(
    fold_id: str,
    question: str,
    action_id: str,
    *,
    signal: float,
    f1_delta: float,
    citation_delta: int,
) -> ActionAuditRow:
    return ActionAuditRow(
        question_key=question,
        fold_id=fold_id,
        route="other",
        action=CompositionAction(
            action_id=action_id,
            family="replace_slot_1",
            aliases=("replace_slot_1",),
            selected_indices=(0,),
            matches_stage180=False,
        ),
        runtime_features={
            "signal": signal,
            "question_route": "other",
            "action_family": "replace_slot_1",
            "added_score_mean": signal,
            "selected_score_mean": 1.0,
        },
        outcome_class="synthetic",
        strict_expected=citation_delta >= 0
        and f1_delta >= 0
        and (citation_delta > 0 or f1_delta > 0),
        citation_delta=citation_delta,
        f1_delta=f1_delta,
    )
