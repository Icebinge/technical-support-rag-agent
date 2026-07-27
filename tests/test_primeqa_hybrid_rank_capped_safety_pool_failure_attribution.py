from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ts_rag_agent.application import (
    primeqa_hybrid_rank_capped_safety_pool_failure_attribution as analysis,
)


def test_authorize_stage191_report_accepts_insufficient_result() -> None:
    analysis._authorize_stage191_report(_stage191_report())


def test_authorize_stage191_report_rejects_test_access() -> None:
    report = _stage191_report()
    report["execution_boundaries"]["test_loaded"] = True

    with pytest.raises(ValueError, match="boundary"):
        analysis._authorize_stage191_report(report)


def test_stage192_visualizations_are_valid_svg(tmp_path: Path) -> None:
    visualizations = analysis.write_stage192_visualizations(
        report=_stage192_report(),
        output_dir=tmp_path,
    )

    assert len(visualizations) == 12
    for visualization in visualizations:
        root = ET.parse(visualization.path).getroot()
        assert root.tag.endswith("svg")
        assert "Poppins" in Path(visualization.path).read_text(encoding="utf-8")

    cap_chart = (tmp_path / "stage192_cap_pool_recall.svg").read_text(encoding="utf-8")
    assert cap_chart.index(">4</text>") < cap_chart.index(">8</text>")
    assert cap_chart.index(">8</text>") < cap_chart.index(">16</text>")
    assert cap_chart.index(">16</text>") < cap_chart.index(">all</text>")


def _stage191_report() -> dict:
    return {
        "stage": "Stage 191",
        "decision": {
            "status": "stage191_rank_capped_safety_pool_insufficient",
            "experiment_valid": True,
            "candidate_family_accepted": False,
        },
        "process_guards": [{"name": "all", "passed": True}],
        "execution_boundaries": {
            "development_loaded": False,
            "test_loaded": False,
            "fallback_action_count": 0,
        },
        "rank_capped_safety_pool_nested_cv": {
            "outer_folds": {
                f"fold_{index}": {"outer_evaluated": index != 2} for index in range(1, 6)
            }
        },
    }


def _stage192_report() -> dict:
    reference = {
        "strict_opportunity_pool_recall": 0.95,
        "conditional_ranker_strict_capture": 0.55,
        "actual_strict_opportunity_capture": 0.52,
        "baseline_change_strict_precision": 0.56,
        "unsafe_selection_rate": 0.35,
        "ranker_miss_breakdown": {
            "safe_zero_winner_context_count": 100,
            "unsafe_winner_context_count": 500,
        },
        "opportunity_partition": {
            "pool_exclusion_context_count": 50,
            "ranker_miss_context_count": 600,
            "strict_selected_context_count": 700,
        },
    }
    factor_row = {
        "strict_opportunity_pool_recall": 0.9,
        "conditional_ranker_strict_capture": 0.5,
        "unsafe_selection_rate": 0.3,
    }
    return {
        "rank_capped_safety_pool_failure_attribution": {
            "reference_trajectory": reference,
            "outer_folds": {
                f"fold_{index}": {
                    "reference_diagnostics": {
                        "opportunity_partition": reference["opportunity_partition"]
                    }
                }
                for index in range(1, 6)
            },
            "factor_aggregates": {
                "pool_cap": {
                    "16": factor_row,
                    "4": factor_row,
                    "all": factor_row,
                    "8": factor_row,
                },
                "gain_ranker": {"pairwise": factor_row, "listnet": factor_row},
                "safety_estimator": {"logistic": factor_row, "histogram": factor_row},
            },
            "configuration_aggregates": {f"config_{index}": {} for index in range(32)},
            "execution": {"snapshot_count": 5},
        },
        "execution_boundaries": {
            "stage191_model_fit_count": 288,
            "attribution_new_model_fit_count": 0,
        },
        "resource_consumption": {
            "process_peak_working_set_bytes": 5 * 1024**3,
            "process_peak_private_usage_bytes": 3.5 * 1024**3,
            "minimum_system_available_memory_bytes": 3 * 1024**3,
        },
        "process_guards": [
            {"name": "guard_a", "passed": True},
            {"name": "guard_b", "passed": True},
        ],
    }
