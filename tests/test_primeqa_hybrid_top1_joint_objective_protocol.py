from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application import primeqa_hybrid_top1_joint_objective_protocol as protocol


def test_freeze_authorizes_only_stage203_train_experiment(tmp_path: Path, monkeypatch) -> None:
    stage201_path, stage199_path = _write_sources(tmp_path)
    monkeypatch.setattr(protocol, "STAGE201_SHA256", _hash(stage201_path))
    monkeypatch.setattr(protocol, "STAGE199_SHA256", _hash(stage199_path))

    report = protocol.freeze_top1_joint_objective_protocol(
        stage201_report_path=stage201_path,
        stage199_report_path=stage199_path,
        user_confirmed=True,
        confirmation_note="user selected A",
    )

    assert report["decision"] == {
        "status": "stage202_top1_joint_objective_protocol_frozen",
        "protocol_valid": True,
        "stage203_train_only_experiment_authorized": True,
        "development_opened": False,
        "test_opened": False,
        "runtime_e2e_authorized": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "default_runtime_activation": False,
    }
    frozen = report["frozen_protocol"]
    assert frozen["objective_factorial"]["custom_objective_count"] == 16
    assert frozen["objective_factorial"]["candidate_config_count_per_outer_context"] == 17
    assert frozen["cross_validation"]["maximum_model_fit_count"] == 425
    assert frozen["cross_validation"]["maximum_lightgbm_tree_count"] == 112_500
    assert (
        frozen["candidate_pool_contract"]["source_risk_frontier_applied_to_custom_candidates"]
        is False
    )
    assert frozen["candidate_pool_contract"]["fallback_used"] is False
    assert len(frozen["inner_selection"]["eligibility_constraints"]) == 13
    assert len(frozen["advancement_gates"]) == 17
    assert all(row["passed"] for row in report["guard_checks"])
    assert report["public_safe_contract"]["public_report_safe"] is True


def test_objective_grid_is_complete_unique_and_contains_ablation_families() -> None:
    specs = protocol._objective_specs()

    assert len(specs) == 16
    assert len({row["name"] for row in specs}) == 16
    assert {(row["safety_weight"], row["precision_weight"]) for row in specs} == {
        (safety, precision)
        for safety in protocol.SAFETY_WEIGHTS
        for precision in protocol.PRECISION_WEIGHTS
    }
    assert {row["ablation_family"] for row in specs} == {
        "strict_only",
        "safety_only",
        "precision_only",
        "full_joint",
    }


def test_unconfirmed_route_blocks_stage203(tmp_path: Path, monkeypatch) -> None:
    stage201_path, stage199_path = _write_sources(tmp_path)
    monkeypatch.setattr(protocol, "STAGE201_SHA256", _hash(stage201_path))
    monkeypatch.setattr(protocol, "STAGE199_SHA256", _hash(stage199_path))

    report = protocol.freeze_top1_joint_objective_protocol(
        stage201_report_path=stage201_path,
        stage199_report_path=stage199_path,
        user_confirmed=False,
        confirmation_note="not confirmed",
    )

    assert report["decision"]["protocol_valid"] is False
    assert _failed(report) == {"user_confirmed_route_a"}


def test_open_test_source_blocks_stage203(tmp_path: Path, monkeypatch) -> None:
    stage201_path, stage199_path = _write_sources(tmp_path, test_opened=True)
    monkeypatch.setattr(protocol, "STAGE201_SHA256", _hash(stage201_path))
    monkeypatch.setattr(protocol, "STAGE199_SHA256", _hash(stage199_path))

    report = protocol.freeze_top1_joint_objective_protocol(
        stage201_report_path=stage201_path,
        stage199_report_path=stage199_path,
        user_confirmed=True,
        confirmation_note="A",
    )

    assert report["decision"]["protocol_valid"] is False
    assert _failed(report) == {"stage201_test_closed"}


def test_inexact_question_partition_blocks_stage203(tmp_path: Path, monkeypatch) -> None:
    stage201_path, stage199_path = _write_sources(tmp_path, partition_exact=False)
    monkeypatch.setattr(protocol, "STAGE201_SHA256", _hash(stage201_path))
    monkeypatch.setattr(protocol, "STAGE199_SHA256", _hash(stage199_path))

    report = protocol.freeze_top1_joint_objective_protocol(
        stage201_report_path=stage201_path,
        stage199_report_path=stage199_path,
        user_confirmed=True,
        confirmation_note="A",
    )

    assert report["decision"]["protocol_valid"] is False
    assert _failed(report) == {"opportunity_partition_exact", "outcome_partition_exact"}


def test_source_hash_mismatch_blocks_stage203(tmp_path: Path) -> None:
    stage201_path, stage199_path = _write_sources(tmp_path)

    report = protocol.freeze_top1_joint_objective_protocol(
        stage201_report_path=stage201_path,
        stage199_report_path=stage199_path,
        user_confirmed=True,
        confirmation_note="A",
    )

    assert report["decision"]["protocol_valid"] is False
    assert _failed(report) == {
        "stage199_sha256_matches",
        "stage201_references_exact_stage199",
        "stage201_sha256_matches",
    }


def test_visualizations_are_valid_svg(tmp_path: Path, monkeypatch) -> None:
    stage201_path, stage199_path = _write_sources(tmp_path)
    monkeypatch.setattr(protocol, "STAGE201_SHA256", _hash(stage201_path))
    monkeypatch.setattr(protocol, "STAGE199_SHA256", _hash(stage199_path))
    report = protocol.freeze_top1_joint_objective_protocol(
        stage201_report_path=stage201_path,
        stage199_report_path=stage199_path,
        user_confirmed=True,
        confirmation_note="A",
    )

    visualizations = protocol.write_stage202_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    assert len(visualizations) == 10
    for visualization in visualizations:
        root = ET.parse(visualization.path).getroot()
        assert root.tag.endswith("svg")
        assert "Poppins" in Path(visualization.path).read_text(encoding="utf-8")


def _failed(report: dict) -> set[str]:
    return {row["name"] for row in report["guard_checks"] if not row["passed"]}


def _write_sources(
    tmp_path: Path,
    *,
    test_opened: bool = False,
    partition_exact: bool = True,
) -> tuple[Path, Path]:
    stage199_path = tmp_path / "stage199.json"
    stage199_path.write_text(json.dumps(_stage199_report()), encoding="utf-8")
    stage199_hash = _hash(stage199_path)
    stage201_path = tmp_path / "stage201.json"
    stage201_path.write_text(
        json.dumps(
            _stage201_report(
                stage199_hash=stage199_hash,
                test_opened=test_opened,
                partition_exact=partition_exact,
            )
        ),
        encoding="utf-8",
    )
    return stage201_path, stage199_path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage199_report() -> dict:
    outer = {
        f"fold_{index}": {
            "source_spec": {
                "name": f"source_{index}",
                "pool_feature_representation": "question_relative_runtime",
                "pool_safety_estimator": "class_balanced_logistic",
                "gain_feature_representation": "raw_runtime",
                "gain_tree_profile": "moderate",
                "risk_feature_representation": "question_relative_runtime",
                "risk_tree_profile": "moderate",
                "scale_pos_weight": 2.0,
                "safest_prefix_size": 4,
            },
            "control_reproduction_exact": True,
        }
        for index in range(1, 6)
    }
    return {
        "stage": "Stage 199",
        "decision": {
            "status": "stage199_joint_risk_winner_insufficient",
            "experiment_valid": True,
            "candidate_family_accepted": False,
        },
        "joint_risk_winner_nested_cv": {"outer_contexts": outer},
    }


def _stage201_report(*, stage199_hash: str, test_opened: bool, partition_exact: bool) -> dict:
    constraints = {
        "citation_delta": {"failure_count": 0},
        "mean_f1_delta": {"failure_count": 0},
        "citation_nonregressing_fold_count": {"failure_count": 12},
        "f1_nonregressing_fold_count": {"failure_count": 1},
        "changed_question_count": {"failure_count": 0},
        "strict_success_count": {"failure_count": 8},
        "strict_success_precision": {"failure_count": 125},
        "strict_opportunity_pool_recall": {"failure_count": 0},
        "folds_meeting_pool_recall_minimum": {"failure_count": 0},
        "conditional_ranker_strict_capture": {"failure_count": 135},
        "folds_meeting_conditional_capture_minimum": {"failure_count": 89},
        "unsafe_selection_rate": {"failure_count": 116},
        "folds_meeting_unsafe_rate_maximum": {"failure_count": 51},
    }
    return {
        "stage": "Stage 201",
        "decision": {
            "status": "stage201_joint_risk_winner_failure_attribution_complete",
            "experiment_valid": True,
            "diagnostic_complete": True,
            "development_opened": False,
            "test_opened": test_opened,
        },
        "source_authorization": {
            "stage199_formal_report": {"sha256": stage199_hash},
        },
        "process_guards": [{"name": f"guard_{index}", "passed": True} for index in range(36)],
        "failure_attribution": {
            "population": {
                "outer_context_count": 5,
                "outer_cell_context_count": 140,
                "fold_cell_context_count": 560,
                "question_cell_context_count": 41_440,
            },
            "constraint_attribution": {"constraints": constraints},
            "question_context_attribution": {
                "aggregate": {
                    "selected_outcome_counts": {
                        "baseline": 6270,
                        "strict_success": 22_044,
                        "safe_zero": 1176,
                        "unsafe_citation_only": 354,
                        "unsafe_f1_only": 11_134,
                        "unsafe_citation_and_f1": 462,
                    },
                    "selected_outcome_partition_exact": partition_exact,
                    "strict_opportunity_mechanism_counts": {
                        "no_strict_opportunity": 672,
                        "safety_pool_exclusion": 476,
                        "risk_frontier_exclusion": 2954,
                        "winner_selection_miss": 15_294,
                        "strict_selected": 22_044,
                    },
                    "strict_opportunity_partition_exact": partition_exact,
                    "ranking_conflicts": {
                        "lower_risk_strict_alternative_count": 10_477,
                        "higher_gain_strict_alternative_count": 11_700,
                    },
                }
            },
            "diagnostic_finding": {
                "recommended_next_research": "objective_research",
                "failure_count_score_by_research_axis": {
                    "model_research": 224,
                    "objective_research": 292,
                    "representation_research": 13,
                },
                "dominant_failed_constraint": "conditional_ranker_strict_capture",
                "dominant_failure_mechanism": "winner_selection_miss",
            },
        },
    }
