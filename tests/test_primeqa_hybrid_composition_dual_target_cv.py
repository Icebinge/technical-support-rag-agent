from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from ts_rag_agent.application import primeqa_hybrid_composition_dual_target_cv as stage182


def test_stage181_authorization_requires_completed_closed_report() -> None:
    report = {
        "stage": "Stage 181",
        "decision": {"status": "stage181_counterfactual_action_audit_complete"},
        "process_guards": [{"name": "guard", "passed": True}],
        "execution_boundaries": {
            "development_loaded": False,
            "test_loaded": False,
        },
    }

    stage182._authorize_stage181_report(report)

    report["execution_boundaries"]["test_loaded"] = True
    with pytest.raises(ValueError, match="development/test"):
        stage182._authorize_stage181_report(report)


def test_stage181_reproduction_checks_all_frozen_aggregates() -> None:
    stage181_report = {
        "composition_dataset": {"action_row_count_including_baseline": 12_298},
        "action_audit": {
            "nonbaseline_action_count": 11_928,
            "strict_expected_action_count": 5_668,
            "outcome_class_counts": {"dual_gain": 257},
            "oracle": {
                "gold_citation_delta": 58,
                "mean_answerable_f1_delta": 0.111694,
            },
        },
        "stage180_selected_action_audit": {
            "gold_citation_delta": 31,
            "mean_answerable_f1_delta": -0.000493,
        },
    }
    result = stage182._stage181_reproduction(
        stage181_report=stage181_report,
        action_summary={
            "nonbaseline_action_count": 11_928,
            "strict_expected_action_count": 5_668,
            "outcome_class_counts": {"dual_gain": 257},
            "oracle": {
                "gold_citation_delta": 58,
                "mean_answerable_f1_delta": 0.111694,
            },
        },
        stage180_summary={
            "gold_citation_delta": 31,
            "mean_answerable_f1_delta": -0.000493,
        },
        build_diagnostics={"action_row_count_including_baseline": 12_298},
    )

    assert result["passed"] is True
    assert all(result["checks"].values())


def test_public_safety_detects_private_dual_target_rows() -> None:
    assert stage182._forbidden_keys_found({"selected_actions": [], "outer_predictions": []}) == {
        "outer_predictions",
        "selected_actions",
    }
    assert stage182._forbidden_keys_found({"aggregate": {"count": 3}}) == set()


def test_stage182_visualizations_are_well_formed(tmp_path) -> None:
    fold = {
        "heldout_policy_evaluation": {
            "gold_citation_delta": 1,
            "mean_f1_delta_all_questions": 0.01,
            "question_coverage": 0.25,
            "strict_expected_precision": 0.75,
        }
    }
    report = {
        "dual_target_nested_cv": {
            "outer_folds": {f"fold_{index}": fold for index in range(1, 6)},
            "aggregate": {
                "gold_citation_delta": 5,
                "mean_f1_delta_all_questions": 0.01,
            },
        },
        "stage181_single_target_benchmark": {
            "top1_gold_citation_delta": 5,
            "top1_mean_answerable_f1_delta": 0.007358,
        },
        "quality_gates": [{"name": "strict_a", "passed": True}],
    }

    visuals = stage182.write_stage182_visualizations(report=report, output_dir=tmp_path)

    assert len(visuals) == 6
    for visual in visuals:
        ET.parse(visual.path)
