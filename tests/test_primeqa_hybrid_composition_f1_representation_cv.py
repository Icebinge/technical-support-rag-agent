from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from ts_rag_agent.application import (
    primeqa_hybrid_composition_f1_representation_cv as stage184,
)


def test_stage183_authorization_requires_completed_f1_risk_diagnostic() -> None:
    report = {
        "stage": "Stage 183",
        "decision": {
            "status": "stage183_f1_risk_failure_attribution_complete",
            "diagnostic_complete": True,
            "primary_bottleneck": "f1_risk_separability_and_ranking",
        },
        "process_guards": [{"name": "guard", "passed": True}],
        "execution_boundaries": {
            "development_loaded": False,
            "test_loaded": False,
        },
    }

    stage184._authorize_stage183_report(report)

    report["decision"]["primary_bottleneck"] = "other"
    with pytest.raises(ValueError, match="F1-risk bottleneck"):
        stage184._authorize_stage183_report(report)


def test_stage184_public_safety_rejects_private_predictions() -> None:
    assert stage184._forbidden_keys_found({"predictions": [], "risk_score": 0.2}) == {
        "predictions",
        "risk_score",
    }
    assert stage184._forbidden_keys_found({"aggregate": {"roc_auc": 0.64}}) == set()


def test_stage184_visualizations_are_well_formed(tmp_path) -> None:
    fold_metrics = {f"fold_{index}": {"roc_auc": 0.60 + index / 100} for index in range(1, 6)}
    representations = {
        "raw_logistic_binary": _representation(0.58, 0.55, 0.53, 0.78, fold_metrics),
        "relative_pairwise_logistic": _representation(0.64, 0.61, 0.75, 0.90, fold_metrics),
    }
    report = {
        "f1_representation_cv": {
            "representations": representations,
            "selection": {
                "best_raw_reference": "raw_logistic_binary",
                "selected_candidate": "relative_pairwise_logistic",
                "quality_gates": [
                    {"name": "risk_auc_at_least_0_62", "passed": True},
                    {"name": "fold_auc_nonregression_at_least_4_of_5", "passed": False},
                ],
            },
        }
    }

    visuals = stage184.write_stage184_visualizations(report=report, output_dir=tmp_path)

    assert len(visuals) == 6
    for visual in visuals:
        ET.parse(visual.path)


def _representation(
    auc: float,
    average_precision: float,
    top3: float,
    top5: float,
    folds: dict,
) -> dict:
    return {
        "aggregate": {
            "roc_auc": auc,
            "average_precision": average_precision,
        },
        "folds": folds,
        "stage182_regression_headroom": {
            "safe_alternative_top3_rate": top3,
            "safe_alternative_top5_rate": top5,
        },
    }
