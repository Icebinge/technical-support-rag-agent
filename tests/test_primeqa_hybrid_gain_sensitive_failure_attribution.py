from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ts_rag_agent.application import (
    primeqa_hybrid_gain_sensitive_failure_attribution as analysis,
)


def test_authorize_stage188_report_accepts_valid_insufficient_result() -> None:
    analysis._authorize_stage188_report(_stage188_report())


def test_authorize_stage188_report_rejects_fallback() -> None:
    report = _stage188_report()
    report["execution_boundaries"]["fallback_action_count"] = 1

    with pytest.raises(ValueError, match="boundary"):
        analysis._authorize_stage188_report(report)


def test_stage190_protocol_freezes_only_for_ranker_miss() -> None:
    ranker_protocol = analysis._stage190_protocol(_attribution("gain_ranker_miss"))
    frontier_protocol = analysis._stage190_protocol(_attribution("safety_frontier_exclusion"))

    assert ranker_protocol["protocol_frozen"] is True
    assert ranker_protocol["candidate_grid_count"] == 160
    assert ranker_protocol["fit_contract"]["maximum_model_fit_count"] == 300
    assert ranker_protocol["selection_contract"]["fallback_enabled"] is False
    assert frontier_protocol["protocol_frozen"] is False


def test_stage189_visualizations_are_valid_svg(tmp_path: Path) -> None:
    visualizations = analysis.write_stage189_visualizations(
        report=_visual_report(),
        output_dir=tmp_path,
    )

    assert len(visualizations) == 12
    for visualization in visualizations:
        root = ET.parse(visualization.path).getroot()
        assert root.tag.endswith("svg")
        assert "Poppins" in Path(visualization.path).read_text(encoding="utf-8")


def _stage188_report() -> dict:
    return {
        "stage": "Stage 188",
        "decision": {
            "status": "stage188_gain_sensitive_ranking_insufficient",
            "experiment_valid": True,
            "candidate_family_accepted": False,
        },
        "process_guards": [{"name": "all", "passed": True}],
        "execution_boundaries": {
            "development_loaded": False,
            "test_loaded": False,
            "fallback_action_count": 0,
        },
        "gain_sensitive_nested_cv": {
            "outer_folds": {f"fold_{index}": {"outer_evaluated": False} for index in range(1, 6)}
        },
    }


def _attribution(primary: str) -> dict:
    return {
        "diagnostic_findings": {
            "stage190_design_branch": {
                "name": (
                    "baseline_referenced_strict_change_gate"
                    if primary == "gain_ranker_miss"
                    else "recall_constrained_safety_frontier"
                )
            }
        }
    }


def _metrics() -> dict:
    return {
        "frontier_strict_question_recall": 0.8,
        "conditional_ranker_strict_capture": 0.3,
        "actual_strict_opportunity_capture": 0.24,
        "baseline_change_strict_precision": 0.4,
        "unsafe_selection_rate": 0.02,
    }


def _visual_report() -> dict:
    metrics = _metrics()
    fold = {
        "top_ineligible_diagnostics": {
            "opportunity_partition": {
                "frontier_exclusion_context_count": 10,
                "ranker_miss_context_count": 30,
            }
        }
    }
    factors = {
        "safety_frontier_margin": {
            "0.00": metrics,
            "0.10": metrics,
        },
        "gain_ranker": {
            "pairwise_pareto_logistic": metrics,
            "linear_listnet_top_frontier": metrics,
        },
        "safety_estimator": {
            "class_balanced_logistic": metrics,
            "histogram_gradient_boosting": metrics,
        },
    }
    return {
        "gain_sensitive_failure_attribution": {
            "top_ineligible_trajectory": {
                **metrics,
                "opportunity_partition": {
                    "frontier_exclusion_context_count": 50,
                    "ranker_miss_context_count": 150,
                    "strict_selected_context_count": 70,
                },
            },
            "outer_folds": {f"fold_{index}": fold for index in range(1, 6)},
            "factor_aggregates": factors,
            "family_best_configurations": {
                "frontier_recall": {"value": 0.9},
                "ranker_capture": {"value": 0.4},
            },
            "configuration_aggregates": {str(index): {} for index in range(32)},
            "execution": {"snapshot_count": 5},
        },
        "execution_boundaries": {
            "stage188_model_fit_count": 240,
            "attribution_new_model_fit_count": 0,
        },
        "resource_consumption": {
            "process_peak_working_set_bytes": 5 * 1024**3,
            "process_peak_private_usage_bytes": 4 * 1024**3,
            "minimum_system_available_memory_bytes": 2 * 1024**3,
        },
        "process_guards": [
            {"name": "guard_a", "passed": True},
            {"name": "guard_b", "passed": True},
        ],
    }
