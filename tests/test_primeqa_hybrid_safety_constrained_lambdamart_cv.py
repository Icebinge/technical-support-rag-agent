from __future__ import annotations

from pathlib import Path

import pytest

from ts_rag_agent.application import (
    primeqa_hybrid_safety_constrained_lambdamart_cv as analysis,
)


def test_authorize_stage193_protocol_accepts_frozen_contract() -> None:
    analysis._authorize_stage193_protocol(_stage193_protocol())


def test_authorize_stage193_protocol_rejects_fallback() -> None:
    report = _stage193_protocol()
    report["frozen_protocol"]["action_contract"]["fallback_enabled"] = True

    with pytest.raises(ValueError, match="fallback"):
        analysis._authorize_stage193_protocol(report)


def test_authorize_stage193_protocol_rejects_fit_budget_drift() -> None:
    report = _stage193_protocol()
    report["frozen_protocol"]["cross_validation"]["maximum_model_fit_count"] = 399

    with pytest.raises(ValueError, match="fit budget"):
        analysis._authorize_stage193_protocol(report)


def test_stage194_visualizations_are_valid_svg(tmp_path: Path) -> None:
    report = _stage194_report()

    visualizations = analysis.write_stage194_visualizations(report=report, output_dir=tmp_path)

    assert len(visualizations) == 12
    assert all(
        Path(item.path).read_text(encoding="utf-8").startswith("<svg") for item in visualizations
    )


def _stage193_protocol() -> dict:
    return {
        "stage": "Stage 193",
        "decision": {
            "status": "stage193_safety_constrained_lambdamart_protocol_frozen",
            "protocol_valid": True,
            "stage194_train_only_experiment_authorized": True,
        },
        "guard_checks": [{"passed": True} for _ in range(59)],
        "frozen_protocol": {
            "candidate_grid": {"policy_config_count": 64},
            "cross_validation": {"maximum_model_fit_count": 400},
            "first_stage_pool": {"pool_cap": 16},
            "action_contract": {"fallback_enabled": False},
        },
    }


def _stage194_report() -> dict:
    outer = {
        f"fold_{index}": {
            "eligible_config_count": index,
            "outer_diagnostics": {
                "strict_opportunity_pool_recall": 0.96,
                "conditional_ranker_strict_capture": 0.70,
                "unsafe_selection_rate": 0.20,
            },
        }
        for index in range(1, 6)
    }
    return {
        "safety_constrained_lambdamart_nested_cv": {
            "outer_folds": outer,
            "aggregate": {
                "changed_question_count": 50,
                "strict_success_count": 40,
                "citation_gain_action_count": 8,
                "citation_loss_action_count": 2,
                "f1_regression_action_count": 20,
            },
            "aggregate_diagnostics": {
                "strict_opportunity_pool_recall": 0.96,
                "conditional_ranker_strict_capture": 0.70,
                "actual_strict_opportunity_capture": 0.672,
                "unsafe_selection_rate": 0.20,
            },
            "selected_profile_counts": {"conservative": 3, "moderate": 2},
            "selected_penalty_counts": {"0.25": 1, "0.50": 2, "1.00": 2},
            "execution": {
                "model_fit_count": 400,
                "pool_safety_fit_count": 200,
                "lambdamart_fit_count": 100,
                "unsafe_head_fit_count": 100,
                "tree_count": 60_000,
            },
            "advancement_gates": [
                {"name": f"gate_{index}", "passed": index % 2 == 0} for index in range(17)
            ],
        },
        "resource_consumption": {
            "process_peak_working_set_bytes": 3 * 1024**3,
            "process_peak_private_usage_bytes": 4 * 1024**3,
            "minimum_system_available_memory_bytes": 7 * 1024**3,
        },
        "process_guards": [{"name": f"guard_{index}", "passed": True} for index in range(32)],
    }
