from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application import primeqa_hybrid_two_stage_change_ranker_protocol as protocol


def test_freeze_authorizes_only_stage206_train_experiment(tmp_path: Path, monkeypatch) -> None:
    stage204_path, stage202_path = _write_sources(tmp_path)
    monkeypatch.setattr(protocol, "STAGE204_SHA256", _hash(stage204_path))
    monkeypatch.setattr(protocol, "STAGE202_SHA256", _hash(stage202_path))

    report = protocol.freeze_two_stage_change_ranker_protocol(
        stage204_report_path=stage204_path,
        stage202_protocol_path=stage202_path,
        user_confirmed=True,
        confirmation_note="user selected A",
    )

    assert report["decision"] == {
        "status": "stage205_two_stage_change_ranker_protocol_frozen",
        "protocol_valid": True,
        "stage206_train_only_experiment_authorized": True,
        "development_opened": False,
        "test_opened": False,
        "runtime_e2e_authorized": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "default_runtime_activation": False,
    }
    frozen = report["frozen_protocol"]
    assert frozen["factorial_ablation"]["two_stage_policy_count"] == 10
    assert frozen["factorial_ablation"]["candidate_config_count_per_outer_context"] == 11
    assert frozen["cross_validation"]["model_fits_per_inner_partition"] == 16
    assert frozen["cross_validation"]["maximum_model_fit_count"] == 370
    assert frozen["cross_validation"]["maximum_lightgbm_tree_count"] == 96_000
    assert len(frozen["inner_selection"]["eligibility_constraints"]) == 13
    assert len(frozen["advancement_gates"]) == 17
    assert all(row["passed"] for row in report["guard_checks"])
    assert report["public_safe_contract"]["public_report_safe"] is True


def test_ranker_and_coverage_factorial_is_complete_and_unique() -> None:
    rankers = protocol._ranker_specs()
    policies = protocol._policy_specs()

    assert [row["name"] for row in rankers] == list(protocol.RANKER_FAMILIES)
    assert len(policies) == 10
    assert len({row["name"] for row in policies}) == 10
    assert {(row["ranker_family"], row["target_change_coverage"]) for row in policies} == {
        (ranker, coverage)
        for ranker in protocol.RANKER_FAMILIES
        for coverage in protocol.TARGET_CHANGE_COVERAGES
    }


def test_gate_training_contract_forbids_same_fit_ranker_winners() -> None:
    frozen = protocol._frozen_protocol(_trajectories())
    gate = frozen["change_abstain_gate"]
    crossfit = frozen["cross_fitting_contract"]

    assert gate["training_winners_are_ranker_oof_only"] is True
    assert gate["same_fit_ranker_winner_used_for_gate_training"] is False
    assert gate["raw_absolute_ranker_scores_used"] is False
    assert crossfit["question_overlap_between_gate_fit_and_ranker_prediction_fold"] is False
    assert crossfit["gate_crossfit_assignment_is_deterministic"] is True
    assert crossfit["gate_training_has_exactly_one_oof_winner_per_question"] is True
    assert frozen["candidate_pool_contract"]["baseline_excluded_from_conditional_ranker_fit"]
    assert frozen["candidate_pool_contract"]["fallback_used"] is False


def test_unconfirmed_route_blocks_stage206(tmp_path: Path, monkeypatch) -> None:
    stage204_path, stage202_path = _write_sources(tmp_path)
    monkeypatch.setattr(protocol, "STAGE204_SHA256", _hash(stage204_path))
    monkeypatch.setattr(protocol, "STAGE202_SHA256", _hash(stage202_path))

    report = protocol.freeze_two_stage_change_ranker_protocol(
        stage204_report_path=stage204_path,
        stage202_protocol_path=stage202_path,
        user_confirmed=False,
        confirmation_note="not confirmed",
    )

    assert report["decision"]["protocol_valid"] is False
    assert _failed(report) == {"user_confirmed_route_a"}


def test_open_test_source_blocks_stage206(tmp_path: Path, monkeypatch) -> None:
    stage204_path, stage202_path = _write_sources(tmp_path, test_opened=True)
    monkeypatch.setattr(protocol, "STAGE204_SHA256", _hash(stage204_path))
    monkeypatch.setattr(protocol, "STAGE202_SHA256", _hash(stage202_path))

    report = protocol.freeze_two_stage_change_ranker_protocol(
        stage204_report_path=stage204_path,
        stage202_protocol_path=stage202_path,
        user_confirmed=True,
        confirmation_note="A",
    )

    assert report["decision"]["protocol_valid"] is False
    assert _failed(report) == {"stage204_test_closed"}


def test_wrong_recommendation_blocks_stage206(tmp_path: Path, monkeypatch) -> None:
    stage204_path, stage202_path = _write_sources(tmp_path, recommendation="different_route")
    monkeypatch.setattr(protocol, "STAGE204_SHA256", _hash(stage204_path))
    monkeypatch.setattr(protocol, "STAGE202_SHA256", _hash(stage202_path))

    report = protocol.freeze_two_stage_change_ranker_protocol(
        stage204_report_path=stage204_path,
        stage202_protocol_path=stage202_path,
        user_confirmed=True,
        confirmation_note="A",
    )

    assert report["decision"]["protocol_valid"] is False
    assert _failed(report) == {"two_stage_research_recommended"}


def test_source_hash_mismatch_blocks_stage206(tmp_path: Path) -> None:
    stage204_path, stage202_path = _write_sources(tmp_path)

    report = protocol.freeze_two_stage_change_ranker_protocol(
        stage204_report_path=stage204_path,
        stage202_protocol_path=stage202_path,
        user_confirmed=True,
        confirmation_note="A",
    )

    assert report["decision"]["protocol_valid"] is False
    assert _failed(report) == {"stage202_sha256_matches", "stage204_sha256_matches"}


def test_visualizations_are_valid_svg(tmp_path: Path, monkeypatch) -> None:
    stage204_path, stage202_path = _write_sources(tmp_path)
    monkeypatch.setattr(protocol, "STAGE204_SHA256", _hash(stage204_path))
    monkeypatch.setattr(protocol, "STAGE202_SHA256", _hash(stage202_path))
    report = protocol.freeze_two_stage_change_ranker_protocol(
        stage204_report_path=stage204_path,
        stage202_protocol_path=stage202_path,
        user_confirmed=True,
        confirmation_note="A",
    )

    visualizations = protocol.write_stage205_visualizations(
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
    recommendation: str = "separate_change_abstain_head_and_conditional_strict_ranker_protocol",
) -> tuple[Path, Path]:
    stage204_path = tmp_path / "stage204.json"
    stage204_path.write_text(
        json.dumps(_stage204_report(test_opened=test_opened, recommendation=recommendation)),
        encoding="utf-8",
    )
    stage202_path = tmp_path / "stage202.json"
    stage202_path.write_text(json.dumps(_stage202_report()), encoding="utf-8")
    return stage204_path, stage202_path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage204_report(*, test_opened: bool, recommendation: str) -> dict:
    return {
        "stage": "Stage 204",
        "source_authorization": {"stage203_formal_report": {"sha256": "stage203"}},
        "decision": {
            "status": "stage204_top1_joint_objective_failure_attribution_complete",
            "experiment_valid": True,
            "diagnostic_complete": True,
            "development_opened": False,
            "test_opened": test_opened,
        },
        "process_guards": [{"name": f"guard_{index}", "passed": True} for index in range(39)],
        "failure_attribution": {
            "population": {"question_context_count": 1480},
            "control_to_custom": {
                "aggregate": {
                    "winner_flip_count": 20_186,
                    "net_strict_count": -8_055,
                    "net_unsafe_count": -3_731,
                    "net_baseline_count": 12_125,
                }
            },
            "precision_adjacent_attribution": {
                "aggregate": {
                    "net_strict_count": -1_934,
                    "net_unsafe_count": -1_105,
                    "net_baseline_count": 3_154,
                }
            },
            "safety_adjacent_attribution": {
                "aggregate": {
                    "net_strict_count": -1_382,
                    "net_unsafe_count": -843,
                    "net_baseline_count": 2_313,
                }
            },
            "target_mechanics": {
                "strict_opportunity_count": 1_439,
                "no_strict_opportunity_count": 41,
                "strict_action_count": {"mean": 8.043919},
                "precision_component_mass_sum_exact": True,
            },
            "diagnostic_finding": {
                "dominant_precision_outcome_change": "strict_success__to__baseline",
                "recommended_next_research": recommendation,
                "finding_is_causal_claim": False,
            },
        },
    }


def _stage202_report() -> dict:
    return {
        "stage": "Stage 202",
        "decision": {
            "status": "stage202_top1_joint_objective_protocol_frozen",
            "protocol_valid": True,
        },
        "frozen_protocol": {"source_trajectory_contract": {"trajectories": _trajectories()}},
    }


def _trajectories() -> list[dict]:
    return [
        {
            "outer_context": f"fold_{index}",
            "source_spec": {"name": f"source_{index}"},
            "control_reproduction_exact": True,
        }
        for index in range(1, 6)
    ]
