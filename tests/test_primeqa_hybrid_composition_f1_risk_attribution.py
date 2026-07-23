from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from ts_rag_agent.application import (
    primeqa_hybrid_composition_f1_risk_attribution as stage183,
)


def test_stage182_authorization_requires_valid_insufficient_report() -> None:
    report = {
        "stage": "Stage 182",
        "decision": {
            "status": "stage182_dual_target_nested_cv_insufficient",
            "experiment_valid": True,
        },
        "process_guards": [{"name": "guard", "passed": True}],
        "execution_boundaries": {
            "development_loaded": False,
            "test_loaded": False,
        },
    }

    stage183._authorize_stage182_report(report)

    report["decision"]["experiment_valid"] = False
    with pytest.raises(ValueError, match="valid Stage182"):
        stage183._authorize_stage182_report(report)


def test_stage182_reproduction_requires_exact_private_and_public_metrics() -> None:
    aggregate = {
        "selected_question_count": 129,
        "strict_expected_count": 69,
        "f1_regression_action_count": 55,
        "gold_citation_delta": 5,
        "mean_f1_delta_all_questions": 0.005249,
    }
    formal = {
        "decision": {"status": "stage182_dual_target_nested_cv_insufficient"},
        "dual_target_nested_cv": {
            "aggregate": aggregate,
            "selected_spec_counts": {"policy": 4, "no_eligible_policy": 1},
            "paired_bootstrap": {"seed": 182},
        },
    }
    reproduced = {
        "decision": {"status": "stage182_dual_target_nested_cv_insufficient"},
        "stage181_reproduction": {"passed": True},
        "dual_target_nested_cv": {
            "aggregate": dict(aggregate),
            "selected_spec_counts": {"policy": 4, "no_eligible_policy": 1},
            "paired_bootstrap": {"seed": 182},
        },
    }

    result = stage183._stage182_reproduction(
        formal=formal,
        reproduced=reproduced,
        selected_actions=[object()] * 129,
    )

    assert result["passed"] is True
    assert all(result["checks"].values())


def test_stage183_public_safety_rejects_private_rows() -> None:
    assert stage183._forbidden_keys_found({"outer_predictions": [], "question_key": "private"}) == {
        "outer_predictions",
        "question_key",
    }
    assert (
        stage183._forbidden_keys_found(
            {"risk_calibration": {"selected_action_population": {"action_count": 129}}}
        )
        == set()
    )
    assert stage183._forbidden_keys_found({"aggregate": {"count": 55}}) == set()


def test_stage183_visualizations_are_well_formed(tmp_path) -> None:
    group = {
        "selected_action_count": 10,
        "f1_regression_count": 4,
        "f1_regression_rate": 0.4,
    }
    report = {
        "f1_risk_attribution": {
            "selected_regression_concentration": {
                "by_action_family": {"replace_slot_1": group},
                "by_route": {"other": group},
            },
            "risk_calibration": {
                "selected_action_population": {
                    "bins": [
                        {
                            "lower": 0.0,
                            "upper": 0.2,
                            "mean_predicted_risk": 0.1,
                            "observed_regression_rate": 0.3,
                        }
                    ]
                }
            },
            "safe_alternative_headroom": {
                "any_strict_alternative_rate": 0.9,
                "same_or_better_citation_safe_alternative_rate": 0.7,
                "safe_citation_gain_alternative_rate": 0.2,
                "same_or_better_safe_alternative_in_model_top3_rate": 0.5,
                "same_or_better_safe_alternative_in_model_top5_rate": 0.6,
            },
            "selected_action_summary": {
                "severity": {
                    "mild_above_minus_0_01": 1,
                    "moderate_minus_0_05_to_minus_0_01": 2,
                    "large_minus_0_10_to_minus_0_05": 3,
                    "severe_at_or_below_minus_0_10": 4,
                    "distribution": {},
                }
            },
            "runtime_feature_separation": {
                "top_features": [{"feature": "signal", "oriented_univariate_auc": 0.61}]
            },
            "no_inner_eligible_fold_attribution": {
                "failure_reason_counts": {"f1_fold_nonregression_failed": 12}
            },
        }
    }

    visuals = stage183.write_stage183_visualizations(report=report, output_dir=tmp_path)

    assert len(visuals) == 7
    for visual in visuals:
        ET.parse(visual.path)
