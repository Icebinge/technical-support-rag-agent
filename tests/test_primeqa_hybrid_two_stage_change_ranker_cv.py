from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ts_rag_agent.application import primeqa_hybrid_two_stage_change_ranker_cv as analysis


def test_authorize_stage205_accepts_strict_protocol() -> None:
    analysis._authorize_stage205_protocol(_stage205_report())


def test_authorize_stage205_rejects_same_fit_source_safety() -> None:
    report = _stage205_report()
    report["frozen_protocol"]["cross_fitting_contract"][
        "same_fit_source_safety_predictions_for_gate_training"
    ] = True

    with pytest.raises(ValueError, match="same-fit source-safety"):
        analysis._authorize_stage205_protocol(report)


def test_authorize_stage199_rejects_accepted_source_family() -> None:
    report = _stage199_report()
    report["decision"]["candidate_family_accepted"] = True

    with pytest.raises(ValueError, match="insufficient Stage199"):
        analysis._authorize_stage199_report(report)


def test_stage206_process_guards_accept_exact_maximum_budget() -> None:
    report = _stage206_report()

    guards = analysis._process_guards(report, ())

    assert all(row["passed"] for row in guards)
    assert {row["name"] for row in guards} >= {
        "source_safety_crossfit_count_exact",
        "conditional_ranker_fit_count_exact",
        "gate_fit_count_exact",
        "private_prediction_count_exact",
        "tree_budget_respected",
    }


def test_stage206_visualizations_are_valid_svg(tmp_path: Path) -> None:
    report = _stage206_report()
    report["process_guards"] = analysis._process_guards(report, ())

    visualizations = analysis.write_stage206_visualizations(report=report, output_dir=tmp_path)

    assert len(visualizations) == 20
    for visualization in visualizations:
        root = ET.parse(visualization.path).getroot()
        assert root.tag.endswith("svg")
        assert "Poppins" in Path(visualization.path).read_text(encoding="utf-8")


def test_stage206_report_bundle_preserves_core_when_visualization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "stage206.json"

    def fail_visualization(**_: object) -> list[object]:
        raise RuntimeError("visualization failed")

    monkeypatch.setattr(analysis, "write_stage206_visualizations", fail_visualization)

    with pytest.raises(RuntimeError, match="visualization failed"):
        analysis.write_stage206_report_bundle(
            report=_stage206_report(),
            output_path=output_path,
            visualization_dir=tmp_path / "visuals",
        )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["two_stage_change_ranker_nested_cv"]["execution"]["model_fit_count"] == 570
    assert "visualizations" not in persisted


def test_stage206_preflight_failure_is_persisted(tmp_path: Path) -> None:
    path = tmp_path / "failure.json"
    report = {
        "stage": "Stage 206",
        "status": "stage206_stage182_reproduction_failed",
        "stage182_reproduction": {"failed_checks": ["bootstrap"]},
    }

    analysis.write_stage206_preflight_failure(report=report, output_path=path)

    assert json.loads(path.read_text(encoding="utf-8")) == report


def _stage205_report() -> dict:
    return {
        "stage": "Stage 205",
        "decision": {
            "protocol_valid": True,
            "stage206_train_only_experiment_authorized": True,
            "development_opened": False,
            "test_opened": False,
        },
        "guard_checks": [{"name": f"guard_{index}", "passed": True} for index in range(84)],
        "frozen_protocol": {
            "factorial_ablation": {
                "two_stage_policy_count": 10,
                "candidate_config_count_per_outer_context": 11,
            },
            "cross_fitting_contract": {
                "source_safety_predictions_for_gate_winners_are_oof_only": True,
                "same_fit_source_safety_predictions_for_gate_training": False,
            },
            "cross_validation": {
                "model_fits_per_inner_partition": 24,
                "maximum_model_fit_count": 570,
                "maximum_lightgbm_tree_count": 96_000,
            },
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


def _stage206_report() -> dict:
    outer = {
        f"fold_{index}": {
            "eligible_config_count": 3,
            "control_reproduction_exact": True,
            "selected_spec": {
                "name": "strict_binary__change_c55",
                "ranker_family": "strict_binary",
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
    ranker_factors = {
        "exact_control": {
            "mean_unsafe_selection_rate": 0.3,
            "mean_conditional_capture": 0.6,
            "mean_strict_success_precision": 0.6,
        },
        **{
            name: {
                "mean_unsafe_selection_rate": 0.2,
                "mean_conditional_capture": 0.7,
                "mean_strict_success_precision": 0.7,
                "mean_pre_gate_ranker_strict_rate": 0.72,
                "mean_pre_gate_ranker_unsafe_rate": 0.18,
            }
            for name in ("strict_binary", "strict_safety_graded")
        },
    }
    coverage_factors = {
        str(value): {
            "mean_unsafe_selection_rate": 0.2,
            "mean_conditional_capture": 0.7,
            "mean_strict_success_precision": 0.7,
            "mean_realized_change_coverage": value,
            "mean_heldout_gate_roc_auc": 0.74,
            "mean_heldout_gate_average_precision": 0.71,
        }
        for value in (0.25, 0.40, 0.55, 0.70, 0.85)
    }
    cv = {
        "protocol": {
            "source_safety_gate_features_oof": True,
            "fallback_enabled": False,
        },
        "dataset": {
            "action_count": 100,
            "question_count": 20,
            "fold_action_counts": {f"fold_{index}": 20 for index in range(1, 6)},
            "fold_nonbaseline_action_counts": {f"fold_{index}": 16 for index in range(1, 6)},
            "fold_question_counts": {f"fold_{index}": 4 for index in range(1, 6)},
        },
        "outer_contexts": outer,
        "cell_aggregates": {f"cell_{index}": {} for index in range(11)},
        "ranker_family_aggregates": ranker_factors,
        "coverage_aggregates": coverage_factors,
        "selected_spec_counts": {"strict_binary__change_c55": 5},
        "selected_ranker_family_counts": {"strict_binary": 5},
        "advancement_gates": [{"name": f"gate_{index}", "passed": True} for index in range(17)],
        "candidate_family_accepted": True,
        "execution": {
            "model_fit_count": 570,
            "source_model_fit_count": 100,
            "source_safety_crossfit_fit_count": 200,
            "conditional_ranker_fit_count": 225,
            "gate_fit_count": 45,
            "lightgbm_model_fit_count": 320,
            "source_tree_count": 15_000,
            "conditional_ranker_tree_count": 67_500,
            "gate_tree_count": 13_500,
            "tree_count": 96_000,
            "source_group_contract_validation_count": 25,
            "ranker_group_contract_validation_count": 225,
            "source_safety_oof_prediction_count": 3_200,
            "ranker_oof_prediction_count": 2_240,
            "gate_training_question_count": 560,
            "control_reproduction_count": 5,
            "all_controls_reproduced_exactly": True,
            "private_prediction_count": 8_900,
            "public_training_rows_written": 0,
            "public_prediction_rows_written": 0,
        },
    }
    return {
        "stage182_reproduction": {"passed": True},
        "two_stage_change_ranker_nested_cv": cv,
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
            "gold_used_only_for_training_targets_and_offline_evaluation": True,
        },
        "resource_consumption": {
            "process_peak_working_set_bytes": 3 * 1024**3,
            "process_peak_private_usage_bytes": 4 * 1024**3,
            "minimum_system_available_memory_bytes": 5 * 1024**3,
        },
    }
