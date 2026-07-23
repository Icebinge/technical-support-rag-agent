from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ts_rag_agent.application import primeqa_hybrid_joint_constraint_ranking_cv as analysis


def test_authorize_stage185_protocol_accepts_frozen_contract() -> None:
    analysis._authorize_stage185_protocol(_stage185_protocol())


def test_authorize_stage185_protocol_rejects_fallback() -> None:
    report = _stage185_protocol()
    report["frozen_protocol"]["reference_action_contract"]["fallback_enabled"] = True

    with pytest.raises(ValueError, match="fallback"):
        analysis._authorize_stage185_protocol(report)


def test_stage186_visualizations_are_valid_svg(tmp_path: Path) -> None:
    report = _stage186_report()

    visualizations = analysis.write_stage186_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert len(visualizations) == 9
    for visualization in visualizations:
        root = ET.parse(visualization.path).getroot()
        assert root.tag.endswith("svg")
        assert "Poppins" in Path(visualization.path).read_text(encoding="utf-8")


def _stage185_protocol() -> dict:
    return {
        "stage": "Stage 185",
        "decision": {
            "status": "stage185_joint_constraint_ranking_protocol_frozen",
            "protocol_valid": True,
            "stage186_train_only_experiment_authorized": True,
        },
        "guard_checks": [{"name": "all", "passed": True}],
        "frozen_protocol": {
            "candidate_grid": {"policy_config_count": 72},
            "cross_validation": {"maximum_model_head_fit_count": 300},
            "reference_action_contract": {"fallback_enabled": False},
        },
    }


def _stage186_report() -> dict:
    outer_folds = {
        f"fold_{index}": {
            "eligible_config_count": index,
            "outer_evaluated": True,
            "outer_evaluation": {
                "gold_citation_delta": index - 2,
                "mean_f1_delta": (index - 2) / 100,
            },
        }
        for index in range(1, 6)
    }
    return {
        "joint_constraint_nested_cv": {
            "protocol": {
                "outer_fold_count": 5,
                "inner_fold_count": 4,
                "policy_config_count": 72,
            },
            "outer_folds": outer_folds,
            "aggregate": {
                "reference_regression_count": 55,
                "repaired_reference_regression_count": 30,
                "new_f1_regression_count": 4,
            },
            "head_metrics": {
                "citation_loss": {"roc_auc": 0.7},
                "f1_loss": {"roc_auc": 0.6},
                "strict_gain": {"roc_auc": 0.65},
            },
            "advancement_gates": [
                {"name": "gate_a", "passed": True},
                {"name": "gate_b", "passed": False},
            ],
            "execution": {
                "model_head_fit_count": 300,
                "private_prediction_count": 1000,
            },
        },
        "resource_consumption": {
            "process_peak_working_set_bytes": 4 * 1024**3,
            "process_peak_private_usage_bytes": 3.7 * 1024**3,
            "minimum_system_available_memory_bytes": 2 * 1024**3,
        },
        "process_guards": [
            {"name": "guard_a", "passed": True},
            {"name": "guard_b", "passed": True},
        ],
    }
