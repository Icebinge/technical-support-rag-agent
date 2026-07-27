from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ts_rag_agent.application import (
    primeqa_hybrid_joint_risk_winner_failure_attribution as analysis,
)


def test_stage201_authorizes_frozen_sources() -> None:
    analysis._authorize_stage200_protocol(_stage200_report())
    analysis._authorize_stage199_report(_stage199_report())


def test_stage201_rejects_open_test_set() -> None:
    report = _stage200_report()
    report["decision"]["test_opened"] = True

    with pytest.raises(ValueError, match="closed development and test"):
        analysis._authorize_stage200_protocol(report)


def test_stage201_reproduction_excludes_only_timing() -> None:
    formal = _stage199_report()
    reproduced = _stage199_report()
    reproduced["joint_risk_winner_nested_cv"]["execution"]["wall_seconds"] = 999.0

    result = analysis._stage199_reproduction(formal, reproduced)

    assert result["passed"] is True
    assert result["passed_check_count"] == result["check_count"]
    assert result["timing_and_resource_values_excluded_from_equality"] is True


def test_stage201_visualizations_are_valid_svg(tmp_path: Path) -> None:
    visualizations = analysis.write_stage201_visualizations(
        report=_stage201_report(), output_dir=tmp_path
    )

    assert len(visualizations) == 17
    for visualization in visualizations:
        root = ET.parse(visualization.path).getroot()
        assert root.tag.endswith("svg")
        assert "Poppins" in Path(visualization.path).read_text(encoding="utf-8")


def test_stage201_bundle_preserves_core_when_visualization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "stage201.json"

    def fail_visualization(**_: object) -> list[object]:
        raise RuntimeError("visualization failed")

    monkeypatch.setattr(analysis, "write_stage201_visualizations", fail_visualization)

    with pytest.raises(RuntimeError, match="visualization failed"):
        analysis.write_stage201_report_bundle(
            report=_stage201_report(),
            output_path=output_path,
            visualization_dir=tmp_path / "visuals",
        )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["failure_attribution"]["population"]["outer_cell_context_count"] == 140
    assert "visualizations" not in persisted


def test_stage201_preflight_failure_is_structured_and_persisted(tmp_path: Path) -> None:
    report = analysis._preflight_failure_report(
        stage200_fingerprint={"sha256": "stage200"},
        stage199_fingerprint={"sha256": "stage199"},
        reproduction_evidence={
            "passed": False,
            "failed_checks": ["bootstrap"],
            "comparison_values": {"bootstrap": {"formal": {"seed": 1}, "actual": {"seed": 2}}},
        },
        recovery_context={"authorization_selection": "A"},
    )
    path = tmp_path / "preflight_failure.json"

    analysis.write_stage201_preflight_failure(report=report, output_path=path)

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["decision"]["status"] == "stage201_preflight_failed"
    assert persisted["stage182_reproduction"]["failed_checks"] == ["bootstrap"]
    assert persisted["execution_boundaries"]["stage199_model_fit_count"] == 0


def test_stage201_success_history_preserves_authorized_prior_failure() -> None:
    history = analysis._successful_run_history(
        {
            "user_authorized_corrected_rerun": True,
            "authorization_selection": "A",
            "prior_failed_attempt": {
                "outcome": "failed",
                "stage199_model_fit_count": 0,
            },
        }
    )

    assert history["formal_attempt_count"] == 2
    assert history["prior_failed_attempt_count"] == 1
    assert history["attempts"][0]["outcome"] == "failed"
    assert history["attempts"][1]["outcome"] == "completed"


def _stage200_report() -> dict:
    return {
        "stage": "Stage 200",
        "decision": {
            "protocol_valid": True,
            "stage201_train_only_attribution_authorized": True,
            "development_opened": False,
            "test_opened": False,
        },
        "guard_checks": [{"name": f"guard_{index}", "passed": True} for index in range(66)],
    }


def _stage199_report() -> dict:
    cv = {
        name: {"value": name}
        for name in (
            "dataset",
            "outer_contexts",
            "aggregate",
            "aggregate_diagnostics",
            "paired_bootstrap",
            "unavailable_bootstrap",
            "cell_aggregates",
            "risk_signal_factor_aggregates",
            "winner_rule_factor_aggregates",
            "complete_pool_risk_metrics",
            "selected_risk_signal_counts",
            "selected_winner_rule_counts",
            "advancement_gates",
        )
    }
    cv["candidate_family_accepted"] = False
    cv["execution"] = {
        "model_fit_count": 100,
        "pool_safety_fit_count": 40,
        "gain_ranker_fit_count": 20,
        "classifier_risk_fit_count": 20,
        "pairwise_safety_fit_count": 20,
        "tree_count": 18_000,
        "group_contract_validation_count": 40,
        "control_reproduction_count": 5,
        "all_controls_reproduced_exactly": True,
        "private_prediction_count": 245_960,
        "public_training_rows_written": 0,
        "public_prediction_rows_written": 0,
        "feature_count_by_representation": {"raw": 1},
        "wall_seconds": 1.0,
    }
    return {
        "stage": "Stage 199",
        "decision": {
            "experiment_valid": True,
            "candidate_family_accepted": False,
            "development_opened": False,
            "test_opened": False,
        },
        "joint_risk_winner_nested_cv": cv,
    }


def _stage201_report() -> dict:
    constraint_names = [f"constraint_{index}" for index in range(13)]
    constraints = {
        name: {"failure_count": index, "near_boundary_count": index // 2}
        for index, name in enumerate(constraint_names)
    }
    outcomes = {
        "baseline": 10,
        "strict_success": 20,
        "safe_zero": 5,
        "unsafe_citation_only": 2,
        "unsafe_f1_only": 3,
        "unsafe_citation_and_f1": 1,
    }
    question_row = {
        "question_cell_context_count": sum(outcomes.values()),
        "selected_outcome_counts": outcomes,
    }
    return {
        "failure_attribution": {
            "population": {"outer_cell_context_count": 140, "question_cell_context_count": 41_440},
            "constraint_attribution": {
                "constraints": constraints,
                "single_constraint_removal_pass_counts": {
                    name: index for index, name in enumerate(constraint_names)
                },
                "failed_constraint_count_distribution": {"0": 1, "1": 20, "2": 119},
                "pareto_nondominated_cell_count_by_outer_context": {
                    f"fold_{index}": index for index in range(1, 6)
                },
            },
            "fold_attribution": {
                "violation_counts_by_metric_and_fold": {
                    "capture": {"fold_1": 10, "fold_2": 8},
                    "unsafe": {"fold_1": 12, "fold_2": 9},
                },
                "aggregate_pass_but_fold_count_fail_by_metric": {
                    "capture": 4,
                    "unsafe": 3,
                },
            },
            "question_context_attribution": {
                "aggregate": {
                    **question_row,
                    "strict_opportunity_mechanism_counts": {
                        "strict_selected": 20,
                        "winner_selection_miss": 21,
                    },
                },
                "by_risk_signal": {"risk_a": question_row, "risk_b": question_row},
                "by_winner_rule": {"winner_a": question_row, "winner_b": question_row},
            },
            "diagnostic_finding": {
                "failure_count_score_by_research_axis": {
                    "model_research": 30,
                    "objective_research": 40,
                    "representation_research": 20,
                }
            },
        },
        "execution_boundaries": {
            "stage199_model_fit_count": 100,
            "stage199_lightgbm_tree_count": 18_000,
            "stage199_private_prediction_count": 245_960,
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
