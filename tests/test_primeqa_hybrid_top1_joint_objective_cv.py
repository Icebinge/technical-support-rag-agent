from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ts_rag_agent.application import primeqa_hybrid_top1_joint_objective_cv as analysis


def test_authorize_stage202_accepts_valid_protocol() -> None:
    analysis._authorize_stage202_protocol(_stage202_report())


def test_authorize_stage202_rejects_open_test() -> None:
    report = _stage202_report()
    report["decision"]["test_opened"] = True

    with pytest.raises(ValueError, match="closed development and test"):
        analysis._authorize_stage202_protocol(report)


def test_authorize_stage199_rejects_accepted_source_family() -> None:
    report = _stage199_report()
    report["decision"]["candidate_family_accepted"] = True

    with pytest.raises(ValueError, match="insufficient Stage199"):
        analysis._authorize_stage199_report(report)


def test_stage203_process_guards_accept_exact_maximum_budget() -> None:
    report = _stage203_report()

    guards = analysis._process_guards(report, ())

    assert all(row["passed"] for row in guards)
    assert {row["name"] for row in guards} >= {
        "model_fit_count_exact",
        "custom_tree_count_within_budget",
        "private_prediction_count_exact",
        "callback_count_within_budget",
    }


def test_stage203_visualizations_are_valid_svg(tmp_path: Path) -> None:
    report = _stage203_report()
    report["process_guards"] = analysis._process_guards(report, ())

    visualizations = analysis.write_stage203_visualizations(report=report, output_dir=tmp_path)

    assert len(visualizations) == 16
    for visualization in visualizations:
        root = ET.parse(visualization.path).getroot()
        assert root.tag.endswith("svg")
        assert "Poppins" in Path(visualization.path).read_text(encoding="utf-8")


def test_stage203_report_bundle_preserves_core_when_visualization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "stage203.json"

    def fail_visualization(**_: object) -> list[object]:
        raise RuntimeError("visualization failed")

    monkeypatch.setattr(analysis, "write_stage203_visualizations", fail_visualization)

    with pytest.raises(RuntimeError, match="visualization failed"):
        analysis.write_stage203_report_bundle(
            report=_stage203_report(),
            output_path=output_path,
            visualization_dir=tmp_path / "visuals",
        )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["top1_joint_objective_nested_cv"]["execution"]["model_fit_count"] == 425
    assert "visualizations" not in persisted


def test_stage203_preflight_failure_is_persisted(tmp_path: Path) -> None:
    path = tmp_path / "failure.json"
    report = {
        "stage": "Stage 203",
        "status": "stage203_stage182_reproduction_failed",
        "stage182_reproduction": {"failed_checks": ["bootstrap"]},
    }

    analysis.write_stage203_preflight_failure(report=report, output_path=path)

    assert json.loads(path.read_text(encoding="utf-8")) == report


def _stage202_report() -> dict:
    return {
        "stage": "Stage 202",
        "decision": {
            "protocol_valid": True,
            "stage203_train_only_experiment_authorized": True,
            "development_opened": False,
            "test_opened": False,
        },
        "guard_checks": [{"name": f"guard_{index}", "passed": True} for index in range(81)],
        "frozen_protocol": {
            "objective_factorial": {
                "custom_objective_count": 16,
                "candidate_config_count_per_outer_context": 17,
            }
        },
    }


def _stage199_report() -> dict:
    return {
        "stage": "Stage 199",
        "decision": {
            "experiment_valid": True,
            "candidate_family_accepted": False,
            "development_opened": False,
            "test_opened": False,
        },
    }


def _stage203_report() -> dict:
    outer = {
        f"fold_{index}": {
            "eligible_config_count": 3,
            "control_reproduction_exact": True,
            "selected_spec": {
                "name": "top1_safety_1.00__precision_1.00",
                "ablation_family": "full_joint",
            },
            "top_inner_candidates": [
                {
                    "evaluation": {"strict_success_precision": 0.7},
                    "diagnostics": {
                        "unsafe_selection_rate": 0.2,
                        "conditional_ranker_strict_capture": 0.7,
                    },
                }
            ],
            "outer_evaluated": True,
        }
        for index in range(1, 6)
    }
    factor = {
        name: {
            "mean_unsafe_selection_rate": 0.2,
            "mean_conditional_capture": 0.7,
            "mean_strict_success_precision": 0.7,
        }
        for name in ("exact_control", "strict_only", "safety_only", "precision_only", "full_joint")
    }
    weight_factor = {
        f"{value:.2f}": {
            "mean_unsafe_selection_rate": 0.2,
            "mean_conditional_capture": 0.7,
            "mean_strict_success_precision": 0.7,
        }
        for value in (0.0, 0.5, 1.0, 2.0)
    }
    cv = {
        "dataset": {
            "action_count": 100,
            "fold_action_counts": {f"fold_{index}": 20 for index in range(1, 6)},
        },
        "outer_contexts": outer,
        "cell_aggregates": {f"cell_{index}": {} for index in range(17)},
        "ablation_family_aggregates": factor,
        "safety_weight_aggregates": weight_factor,
        "precision_weight_aggregates": weight_factor,
        "directional_penalty_response": {
            "safety_adjacent_comparison_count": 12,
            "safety_nonincreasing_unsafe_count": 10,
            "precision_adjacent_comparison_count": 12,
            "precision_nondecreasing_strict_precision_count": 9,
        },
        "selected_spec_counts": {"top1_safety_1.00__precision_1.00": 5},
        "advancement_gates": [{"name": f"gate_{index}", "passed": True} for index in range(17)],
        "candidate_family_accepted": True,
        "execution": {
            "model_fit_count": 425,
            "source_model_fit_count": 100,
            "pool_safety_fit_count": 50,
            "gain_ranker_fit_count": 25,
            "classifier_risk_fit_count": 25,
            "custom_objective_fit_count": 325,
            "outer_custom_objective_refit_count": 5,
            "source_tree_count": 15_000,
            "custom_objective_tree_count": 97_500,
            "tree_count": 112_500,
            "group_contract_validation_count": 350,
            "objective_callback_call_count": 97_500,
            "control_reproduction_count": 5,
            "all_controls_reproduced_exactly": True,
            "private_prediction_count": 8_500,
            "public_training_rows_written": 0,
            "public_prediction_rows_written": 0,
        },
    }
    return {
        "stage182_reproduction": {"passed": True},
        "top1_joint_objective_nested_cv": cv,
        "execution_boundaries": {
            "train_loaded": True,
            "development_loaded": False,
            "test_loaded": False,
            "full_train_policy_selected": False,
            "runtime_registered_as_default": False,
            "runtime_e2e_run": False,
            "stage178b_run": False,
            "retry_action_count": 0,
            "fallback_action_count": 0,
        },
        "resource_consumption": {
            "process_peak_working_set_bytes": 3 * 1024**3,
            "process_peak_private_usage_bytes": 4 * 1024**3,
            "minimum_system_available_memory_bytes": 5 * 1024**3,
        },
    }
