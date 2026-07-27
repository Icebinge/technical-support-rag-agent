from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ts_rag_agent.application import (
    primeqa_hybrid_surviving_unsafe_winner_attribution as analysis,
)


def test_authorize_stage196_accepts_valid_insufficient_result() -> None:
    analysis._authorize_stage196_report(_stage196_report())


def test_authorize_stage196_rejects_open_test() -> None:
    report = _stage196_report()
    report["decision"]["test_opened"] = True

    with pytest.raises(ValueError, match="closed development and test"):
        analysis._authorize_stage196_report(report)


def test_stage197_visualizations_are_valid_svg(tmp_path: Path) -> None:
    visualizations = analysis.write_stage197_visualizations(
        report=_stage197_report(), output_dir=tmp_path
    )

    assert len(visualizations) == 10
    for visualization in visualizations:
        root = ET.parse(visualization.path).getroot()
        assert root.tag.endswith("svg")
        assert "Poppins" in Path(visualization.path).read_text(encoding="utf-8")


def _stage196_report() -> dict:
    outer = {
        f"fold_{index}": {"top_inner_candidates": [{"spec": {"name": "top"}}]}
        for index in range(1, 6)
    }
    return {
        "stage": "Stage 196",
        "decision": {
            "experiment_valid": True,
            "candidate_family_accepted": False,
            "development_opened": False,
            "test_opened": False,
        },
        "process_guards": [{"name": "all", "passed": True}],
        "safety_first_frontier_nested_cv": {"outer_folds": outer},
    }


def _stage197_report() -> dict:
    aggregate = {
        "mechanism_counts": {"final_gain_dominance": 20, "risk_ordering_failure": 10},
        "risk_rank_bucket_counts": {"1": 10, "2": 8, "3-4": 7, "5-8": 4, "9+": 1},
        "gain_rank_bucket_counts": {"1": 30, "2": 0, "3-4": 0, "5+": 0},
        "loss_type_counts": {"citation_only": 2, "f1_only": 28},
        "unsafe_with_strict_opportunity_count": 30,
        "oracle_strict_repairable_count": 25,
        "lower_risk_strict_alternative_count": 20,
    }
    outer = {
        f"fold_{index}": {"unsafe_winner_attribution": {"unsafe_winner_context_count": index * 10}}
        for index in range(1, 6)
    }
    return {
        "surviving_unsafe_winner_attribution": {
            "aggregate": aggregate,
            "outer_contexts": outer,
            "unsafe_head_prediction_metrics": {
                "roc_auc": 0.8,
                "average_precision": 0.7,
            },
            "execution": {
                "partition_count": 20,
                "model_fit_count": 80,
                "tree_count": 12_000,
            },
        },
        "resource_consumption": {
            "process_peak_working_set_bytes": 3 * 1024**3,
            "process_peak_private_usage_bytes": 4 * 1024**3,
            "minimum_system_available_memory_bytes": 5 * 1024**3,
        },
        "process_guards": [
            {"name": "guard_a", "passed": True},
            {"name": "guard_b", "passed": True},
        ],
    }
