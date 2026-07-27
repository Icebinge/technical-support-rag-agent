from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ts_rag_agent.application import primeqa_hybrid_joint_risk_winner_cv as analysis


def test_authorize_stage198_accepts_valid_protocol() -> None:
    analysis._authorize_stage198_protocol(_stage198_report())


def test_authorize_stage198_rejects_open_test() -> None:
    report = _stage198_report()
    report["decision"]["test_opened"] = True

    with pytest.raises(ValueError, match="closed development and test"):
        analysis._authorize_stage198_protocol(report)


def test_authorize_stage197_rejects_incomplete_diagnostic() -> None:
    report = _stage197_report()
    report["decision"]["diagnostic_complete"] = False

    with pytest.raises(ValueError, match="valid Stage197"):
        analysis._authorize_stage197_report(report)


def test_stage199_visualizations_are_valid_svg(tmp_path: Path) -> None:
    visualizations = analysis.write_stage199_visualizations(
        report=_stage199_report(), output_dir=tmp_path
    )

    assert len(visualizations) == 15
    for visualization in visualizations:
        root = ET.parse(visualization.path).getroot()
        assert root.tag.endswith("svg")
        assert "Poppins" in Path(visualization.path).read_text(encoding="utf-8")


def test_stage199_report_bundle_preserves_core_report_when_visualization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "stage199.json"

    def fail_visualization(**_: object) -> list[object]:
        raise RuntimeError("visualization failed")

    monkeypatch.setattr(analysis, "write_stage199_visualizations", fail_visualization)

    with pytest.raises(RuntimeError, match="visualization failed"):
        analysis.write_stage199_report_bundle(
            report=_stage199_report(),
            output_path=output_path,
            visualization_dir=tmp_path / "visuals",
        )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["joint_risk_winner_nested_cv"]["advancement_gates"][0] == {
        "name": "gate_0",
        "passed": True,
    }
    assert "visualizations" not in persisted


def _stage198_report() -> dict:
    return {
        "stage": "Stage 198",
        "decision": {
            "protocol_valid": True,
            "stage199_train_only_experiment_authorized": True,
            "development_opened": False,
            "test_opened": False,
        },
        "guard_checks": [{"name": f"guard_{index}", "passed": True} for index in range(62)],
    }


def _stage197_report() -> dict:
    return {
        "stage": "Stage 197",
        "decision": {
            "experiment_valid": True,
            "diagnostic_complete": True,
            "development_opened": False,
            "test_opened": False,
        },
    }


def _stage199_report() -> dict:
    factor_risk = {
        name: {"mean_unsafe_selection_rate": 0.2, "mean_conditional_capture": 0.7}
        for name in ("classifier", "decomposed", "pairwise", "fusion")
    }
    factor_winner = {
        name: {"mean_unsafe_selection_rate": 0.2, "mean_conditional_capture": 0.7}
        for name in (
            "gain_only",
            "rank_utility_0.25",
            "rank_utility_0.50",
            "rank_utility_1.00",
            "rank_utility_2.00",
            "shortlist_2",
            "shortlist_4",
        )
    }
    outer = {
        f"fold_{index}": {
            "eligible_config_count": index,
            "control_reproduction_exact": True,
            "top_inner_candidates": [
                {
                    "diagnostics": {
                        "unsafe_selection_rate": 0.2,
                        "conditional_ranker_strict_capture": 0.7,
                    }
                }
            ],
        }
        for index in range(1, 6)
    }
    cell_aggregates = {
        f"cell_{index}": {
            "spec": {"name": f"cell_{index}"},
            "strict_success_count": 100 - index,
            "paired_vs_control": {"unsafe_selected_count_delta": -index},
        }
        for index in range(28)
    }
    return {
        "joint_risk_winner_nested_cv": {
            "outer_contexts": outer,
            "cell_aggregates": cell_aggregates,
            "risk_signal_factor_aggregates": factor_risk,
            "winner_rule_factor_aggregates": factor_winner,
            "complete_pool_risk_metrics": {name: {"roc_auc": 0.7} for name in factor_risk},
            "selected_risk_signal_counts": {"pairwise": 3},
            "selected_winner_rule_counts": {"rank_utility_1.00": 3},
            "advancement_gates": [{"name": f"gate_{index}", "passed": True} for index in range(17)],
            "execution": {
                "model_fit_count": 125,
                "tree_count": 22_500,
                "private_prediction_count": 307_450,
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
