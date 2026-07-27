from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ts_rag_agent.application import primeqa_hybrid_rank_capped_safety_pool_cv as analysis


def test_authorize_stage190_protocol_accepts_frozen_contract() -> None:
    analysis._authorize_stage190_protocol(_stage190_protocol())


def test_authorize_stage190_protocol_rejects_fallback() -> None:
    report = _stage190_protocol()
    report["frozen_protocol"]["action_contract"]["fallback_enabled"] = True

    with pytest.raises(ValueError, match="fallback"):
        analysis._authorize_stage190_protocol(report)


def test_authorize_stage190_protocol_rejects_pool_recall_drift() -> None:
    report = _stage190_protocol()
    report["frozen_protocol"]["inner_selection"]["pool_recall_constraints"][
        "aggregate_strict_opportunity_pool_recall_minimum"
    ] = 0.7

    with pytest.raises(ValueError, match="pool-recall"):
        analysis._authorize_stage190_protocol(report)


def test_stage191_visualizations_are_valid_svg(tmp_path: Path) -> None:
    visualizations = analysis.write_stage191_visualizations(
        report=_stage191_report(),
        output_dir=tmp_path,
    )

    assert len(visualizations) == 15
    for visualization in visualizations:
        root = ET.parse(visualization.path).getroot()
        assert root.tag.endswith("svg")
        assert "Poppins" in Path(visualization.path).read_text(encoding="utf-8")


def _stage190_protocol() -> dict:
    return {
        "stage": "Stage 190",
        "decision": {
            "status": "stage190_rank_capped_safety_pool_protocol_frozen",
            "protocol_valid": True,
            "stage191_train_only_experiment_authorized": True,
        },
        "guard_checks": [{"name": "all", "passed": True}],
        "frozen_protocol": {
            "candidate_grid": {
                "policy_config_count": 32,
                "pool_caps": [4, 8, 16, "all"],
            },
            "cross_validation": {"maximum_model_fit_count": 300},
            "action_contract": {"fallback_enabled": False},
            "inner_selection": {
                "pool_recall_constraints": {
                    "aggregate_strict_opportunity_pool_recall_minimum": 0.8,
                    "per_fold_strict_opportunity_pool_recall_minimum": 0.7,
                    "folds_meeting_per_fold_minimum": 3,
                }
            },
        },
    }


def _stage191_report() -> dict:
    outer_folds = {
        f"fold_{index}": {
            "eligible_config_count": index,
            "outer_evaluated": True,
            "outer_evaluation": {
                "gold_citation_delta": index,
                "mean_f1_delta": index / 100,
                "strict_success_count": index + 2,
            },
            "outer_pool_metrics": {
                "strict_opportunity_pool_recall": 0.8 + index / 100,
                "mean_pool_size": 8 + index,
            },
        }
        for index in range(1, 6)
    }
    return {
        "rank_capped_safety_pool_nested_cv": {
            "protocol": {"policy_config_count": 32},
            "outer_folds": outer_folds,
            "aggregate": {
                "changed_question_count": 100,
                "strict_success_count": 70,
                "citation_gain_action_count": 20,
                "citation_loss_action_count": 2,
                "f1_regression_action_count": 10,
            },
            "aggregate_pool_metrics": {
                "strict_opportunity_pool_recall": 0.85,
                "strict_action_retention_rate": 0.6,
                "baseline_in_pool_rate": 1.0,
            },
            "prediction_metrics": {
                "citation_loss": {"roc_auc": 0.8},
                "f1_loss": {"roc_auc": 0.7},
                "strict_gain": {"roc_auc": 0.75},
            },
            "selected_pool_cap_counts": {"4": 1, "8": 2, "16": 1, "all": 1},
            "advancement_gates": [
                {"name": "gate_a", "passed": True},
                {"name": "gate_b", "passed": False},
            ],
            "execution": {
                "model_fit_count": 300,
                "comparable_pair_count_across_fits": 1000,
                "omitted_incomparable_pair_count_across_fits": 500,
            },
        },
        "resource_consumption": {
            "process_peak_working_set_bytes": 4 * 1024**3,
            "process_peak_private_usage_bytes": 3.5 * 1024**3,
            "minimum_system_available_memory_bytes": 2 * 1024**3,
        },
        "process_guards": [
            {"name": "guard_a", "passed": True},
            {"name": "guard_b", "passed": True},
        ],
    }
