from __future__ import annotations

from pathlib import Path

import pytest

from ts_rag_agent.application import primeqa_hybrid_safety_first_frontier_cv as analysis


def test_authorize_stage195_protocol_accepts_frozen_contract() -> None:
    analysis._authorize_stage195_protocol(_stage195_protocol())


def test_authorize_stage195_protocol_rejects_fallback() -> None:
    report = _stage195_protocol()
    report["frozen_protocol"]["safety_first_frontier"]["fallback_used"] = True

    with pytest.raises(ValueError, match="fallback"):
        analysis._authorize_stage195_protocol(report)


def test_authorize_stage195_protocol_rejects_fit_budget_drift() -> None:
    report = _stage195_protocol()
    report["frozen_protocol"]["cross_validation"]["maximum_model_fit_count"] = 599

    with pytest.raises(ValueError, match="fit budget"):
        analysis._authorize_stage195_protocol(report)


def test_authorize_stage195_protocol_rejects_memory_boundary_drift() -> None:
    report = _stage195_protocol()
    report["frozen_protocol"]["resource_contract"][
        "minimum_preflight_system_available_memory_gib"
    ] = 3.0

    with pytest.raises(ValueError, match="memory"):
        analysis._authorize_stage195_protocol(report)


def test_stage196_visualizations_are_valid_svg(tmp_path: Path) -> None:
    visualizations = analysis.write_stage196_visualizations(
        report=_stage196_report(),
        output_dir=tmp_path,
    )

    assert len(visualizations) == 15
    assert all(
        Path(item.path).read_text(encoding="utf-8").startswith("<svg") for item in visualizations
    )


def _stage195_protocol() -> dict:
    return {
        "stage": "Stage 195",
        "decision": {
            "status": "stage195_safety_first_frontier_protocol_frozen",
            "protocol_valid": True,
            "stage196_train_only_experiment_authorized": True,
        },
        "guard_checks": [{"passed": True} for _ in range(58)],
        "frozen_protocol": {
            "candidate_grid": {"policy_config_count": 960},
            "cross_validation": {
                "maximum_model_fit_count": 600,
                "maximum_lightgbm_tree_count": 120_000,
            },
            "first_stage_pool": {"pool_cap": 16},
            "cost_sensitive_unsafe_head": {"scale_pos_weights": [1.0, 2.0, 4.0]},
            "safety_first_frontier": {
                "safest_prefix_sizes": [2, 4, 8, 12, 16],
                "fallback_used": False,
            },
            "resource_contract": {"minimum_preflight_system_available_memory_gib": 4.0},
        },
    }


def _stage196_report() -> dict:
    diagnostics = {
        "strict_opportunity_pool_recall": 0.99,
        "strict_opportunity_frontier_recall": 0.80,
        "conditional_ranker_strict_capture": 0.70,
        "unsafe_selection_rate": 0.20,
        "unsafe_action_retention_rate": 0.30,
        "mean_frontier_size": 5.0,
    }
    outer = {
        f"fold_{index}": {
            "eligible_config_count": index,
            "outer_diagnostics": diagnostics,
            "top_inner_candidates": [
                {
                    "spec": {
                        "safest_prefix_size": 4,
                        "scale_pos_weight": 2.0,
                    },
                    "diagnostics": diagnostics,
                }
            ],
        }
        for index in range(1, 6)
    }
    return {
        "safety_first_frontier_nested_cv": {
            "outer_folds": outer,
            "aggregate": {
                "changed_question_count": 50,
                "strict_success_count": 40,
                "citation_gain_action_count": 8,
                "citation_loss_action_count": 2,
                "f1_regression_action_count": 20,
            },
            "aggregate_diagnostics": diagnostics,
            "selected_prefix_counts": {"4": 3, "8": 2},
            "selected_risk_weight_counts": {"2.0": 4, "4.0": 1},
            "execution": {
                "model_fit_count": 600,
                "pool_safety_fit_count": 200,
                "lambdamart_fit_count": 100,
                "unsafe_head_fit_count": 300,
                "tree_count": 120_000,
            },
            "advancement_gates": [
                {"name": f"gate_{index}", "passed": index % 2 == 0} for index in range(17)
            ],
        },
        "resource_consumption": {
            "process_peak_working_set_bytes": 3 * 1024**3,
            "process_peak_private_usage_bytes": 4 * 1024**3,
            "minimum_system_available_memory_bytes": 5 * 1024**3,
        },
        "process_guards": [{"name": f"guard_{index}", "passed": True} for index in range(35)],
    }
