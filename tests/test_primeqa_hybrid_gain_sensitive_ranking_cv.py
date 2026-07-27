from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ts_rag_agent.application import primeqa_hybrid_gain_sensitive_ranking_cv as analysis


def test_authorize_stage187_protocol_accepts_frozen_contract() -> None:
    analysis._authorize_stage187_protocol(_stage187_protocol())


def test_authorize_stage187_protocol_rejects_fallback() -> None:
    report = _stage187_protocol()
    report["frozen_protocol"]["action_contract"]["fallback_enabled"] = True

    with pytest.raises(ValueError, match="fallback"):
        analysis._authorize_stage187_protocol(report)


def test_authorize_stage187_protocol_rejects_pair_sampling() -> None:
    report = _stage187_protocol()
    report["frozen_protocol"]["gain_ranker_contract"]["pairwise_pareto_logistic"][
        "pair_sampling"
    ] = True

    with pytest.raises(ValueError, match="pair sampling"):
        analysis._authorize_stage187_protocol(report)


def test_stage188_visualizations_are_valid_svg(tmp_path: Path) -> None:
    visualizations = analysis.write_stage188_visualizations(
        report=_stage188_report(),
        output_dir=tmp_path,
    )

    assert len(visualizations) == 12
    for visualization in visualizations:
        root = ET.parse(visualization.path).getroot()
        assert root.tag.endswith("svg")
        assert "Poppins" in Path(visualization.path).read_text(encoding="utf-8")


def _stage187_protocol() -> dict:
    return {
        "stage": "Stage 187",
        "decision": {
            "status": "stage187_gain_sensitive_ranking_protocol_frozen",
            "protocol_valid": True,
            "stage188_train_only_experiment_authorized": True,
        },
        "guard_checks": [{"name": "all", "passed": True}],
        "frozen_protocol": {
            "candidate_grid": {"policy_config_count": 32},
            "cross_validation": {"maximum_model_fit_count": 300},
            "action_contract": {"fallback_enabled": False},
            "gain_ranker_contract": {"pairwise_pareto_logistic": {"pair_sampling": False}},
        },
    }


def _stage188_report() -> dict:
    outer_folds = {
        f"fold_{index}": {
            "eligible_config_count": index,
            "outer_evaluated": True,
            "outer_evaluation": {
                "gold_citation_delta": index,
                "mean_f1_delta": index / 100,
                "strict_success_count": index + 2,
            },
        }
        for index in range(1, 6)
    }
    return {
        "gain_sensitive_nested_cv": {
            "protocol": {"policy_config_count": 32},
            "outer_folds": outer_folds,
            "aggregate": {
                "changed_question_count": 100,
                "strict_success_count": 70,
                "citation_gain_action_count": 20,
                "citation_loss_action_count": 2,
                "f1_regression_action_count": 10,
                "repaired_reference_regression_count": 30,
            },
            "prediction_metrics": {
                "citation_loss": {"roc_auc": 0.8},
                "f1_loss": {"roc_auc": 0.7},
                "strict_gain": {"roc_auc": 0.75},
            },
            "selected_ranker_counts": {
                "pairwise_pareto_logistic": 3,
                "linear_listnet_top_frontier": 2,
            },
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
