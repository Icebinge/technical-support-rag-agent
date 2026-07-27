from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ts_rag_agent.application import (
    primeqa_hybrid_top1_joint_objective_failure_attribution as analysis,
)


def test_stage204_authorizes_only_valid_insufficient_stage203() -> None:
    report = _formal_stage203()

    analysis._authorize_stage203_report(report)

    report["decision"]["candidate_family_accepted"] = True
    with pytest.raises(ValueError, match="insufficient Stage203"):
        analysis._authorize_stage203_report(report)


def test_stage204_reproduction_ignores_dynamic_timing_but_not_evidence() -> None:
    formal = _formal_stage203()
    reproduced = json.loads(json.dumps(formal))
    reproduced["top1_joint_objective_nested_cv"]["execution"]["fit_seconds"] = 999.0
    reproduced["top1_joint_objective_nested_cv"]["execution"]["wall_seconds"] = 1000.0

    result = analysis._stage203_reproduction(formal, reproduced)

    assert result["passed"] is True
    reproduced["top1_joint_objective_nested_cv"]["dataset"]["question_count"] = 369
    result = analysis._stage203_reproduction(formal, reproduced)
    assert result["passed"] is False
    assert "dataset" in result["failed_checks"]


def test_stage204_process_guards_require_exact_population_and_boundaries() -> None:
    report = _stage204_report()
    formal = _formal_stage203()

    guards = analysis._process_guards(report, formal, ())

    assert len(guards) == 39
    assert all(row["passed"] for row in guards)


def test_stage204_visualizations_are_complete_svg(tmp_path: Path) -> None:
    report = _stage204_report()
    report["process_guards"] = analysis._process_guards(report, _formal_stage203(), ())

    visualizations = analysis.write_stage204_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert len(visualizations) == 12
    for visualization in visualizations:
        ET.parse(visualization.path)


def test_stage204_core_report_survives_visualization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "stage204.json"

    def fail_visualization(**_: object) -> tuple[analysis.Stage204Visualization, ...]:
        raise RuntimeError("visual failure")

    monkeypatch.setattr(analysis, "write_stage204_visualizations", fail_visualization)
    with pytest.raises(RuntimeError, match="visual failure"):
        analysis.write_stage204_report_bundle(
            report={"stage": "Stage 204", "decision": {"experiment_valid": True}},
            output_path=output,
            visualization_dir=tmp_path / "visuals",
        )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["stage"] == "Stage 204"
    assert "visualizations" not in persisted


def test_stage204_preflight_failure_is_persisted(tmp_path: Path) -> None:
    output = tmp_path / "failure.json"
    failure = {"stage": "Stage 204", "status": "failed"}

    analysis.write_stage204_preflight_failure(report=failure, output_path=output)

    assert json.loads(output.read_text(encoding="utf-8")) == failure


def _formal_stage203() -> dict:
    cv = {
        "dataset": {"question_count": 370},
        "outer_contexts": {f"fold_{index}": {"outer_evaluated": False} for index in range(1, 6)},
        "aggregate": {},
        "aggregate_diagnostics": {},
        "paired_bootstrap": {},
        "cell_aggregates": {f"cell_{index}": {} for index in range(17)},
        "ablation_family_aggregates": {},
        "safety_weight_aggregates": {},
        "precision_weight_aggregates": {},
        "directional_penalty_response": {},
        "advancement_gates": [],
        "execution": {
            "model_fit_count": 400,
            "tree_count": 108_000,
            "fit_seconds": 1.0,
            "wall_seconds": 2.0,
        },
    }
    return {
        "stage": "Stage 203",
        "decision": {
            "status": "stage203_top1_joint_objective_insufficient",
            "experiment_valid": True,
            "candidate_family_accepted": False,
            "development_opened": False,
            "test_opened": False,
        },
        "stage182_reproduction": {},
        "top1_joint_objective_nested_cv": cv,
        "process_guards": [{"name": f"guard_{index}", "passed": True} for index in range(38)],
        "execution_boundaries": {
            "stage203_model_fit_count": 400,
            "stage203_lightgbm_tree_count": 108_000,
            "stage203_private_prediction_count": 983_840,
        },
    }


def _stage204_report() -> dict:
    transition = {
        "context_count": 23_680,
        "left_partition_exact": True,
        "right_partition_exact": True,
        "transition_partition_exact": True,
        "winner_flip_count": 1,
        "strict_loss_count": 1,
        "strict_gain_count": 0,
        "unsafe_repair_count": 1,
        "unsafe_regression_count": 0,
        "baseline_addition_count": 1,
        "baseline_removal_count": 0,
    }
    candidate = {
        **transition,
        "context_count": 1480,
        "spec": {"name": "candidate"},
    }
    directional = {
        **transition,
        "context_count": 17_760,
    }
    return {
        "stage203_reproduction": {"passed": True},
        "failure_attribution": {
            "population": {
                "outer_context_count": 5,
                "outer_cell_context_count": 85,
                "custom_outer_cell_context_count": 80,
                "question_context_count": 1480,
                "control_custom_question_comparison_count": 23_680,
                "precision_adjacent_question_comparison_count": 17_760,
                "safety_adjacent_question_comparison_count": 17_760,
            },
            "control_to_custom": {
                "aggregate": transition,
                "by_candidate": {f"candidate_{index}": candidate for index in range(16)},
            },
            "precision_adjacent_attribution": {
                "aggregate": directional,
                "by_pair": {f"pair_{index}": {} for index in range(12)},
            },
            "safety_adjacent_attribution": {
                "aggregate": directional,
                "by_pair": {f"pair_{index}": {} for index in range(12)},
            },
            "target_mechanics": {
                "question_context_count": 1480,
                "precision_component_mass_sum_exact": True,
                "precision_component_baseline_mass": {"mean": 0.1},
                "precision_component_total_strict_mass": {"mean": 0.9},
                "safety_component_baseline_mass": {"mean": 0.2},
            },
        },
        "execution_boundaries": {
            "train_loaded": True,
            "development_loaded": False,
            "test_loaded": False,
            "stage203_model_fit_count": 400,
            "stage203_lightgbm_tree_count": 108_000,
            "stage203_private_prediction_count": 983_840,
            "additional_diagnostic_model_fit_count": 0,
            "outer_refit_count": 0,
            "private_question_rows_persisted": False,
            "new_model_search_run": False,
            "same_weight_grid_search_run": False,
            "constraint_relaxation_run": False,
            "full_train_policy_selected": False,
            "replacement_policy_selected": False,
            "runtime_e2e_run": False,
            "runtime_registered_as_default": False,
            "stage178b_run": False,
            "retry_action_count": 0,
            "fallback_action_count": 0,
        },
        "resource_consumption": {
            "process_peak_working_set_bytes": 1,
            "process_peak_private_usage_bytes": 1,
            "minimum_system_available_memory_bytes": 1,
        },
    }
