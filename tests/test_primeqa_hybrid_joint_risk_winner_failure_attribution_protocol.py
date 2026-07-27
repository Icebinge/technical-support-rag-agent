from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application import (
    primeqa_hybrid_joint_risk_winner_failure_attribution_protocol as protocol,
)


def test_freeze_authorizes_only_stage201_train_attribution(tmp_path: Path, monkeypatch) -> None:
    path = _write_stage199(tmp_path)
    monkeypatch.setattr(protocol, "STAGE199_SHA256", _hash(path))

    report = protocol.freeze_joint_risk_winner_failure_attribution_protocol(
        stage199_report_path=path,
        user_confirmed=True,
        confirmation_note="user approved next stage",
    )

    assert report["decision"] == {
        "status": "stage200_joint_risk_winner_failure_attribution_protocol_frozen",
        "protocol_valid": True,
        "stage201_train_only_attribution_authorized": True,
        "development_opened": False,
        "test_opened": False,
        "new_policy_search_authorized": False,
        "constraint_relaxation_authorized": False,
        "runtime_e2e_authorized": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "default_runtime_activation": False,
    }
    frozen = report["frozen_protocol"]
    assert frozen["diagnostic_population"] == {
        "outer_cell_context_count": 140,
        "fold_cell_context_count": 560,
        "inner_question_context_count": 1480,
        "question_cell_context_count": 41_440,
        "policy_cell_count_per_outer_context": 28,
        "risk_signal_count": 4,
        "winner_rule_count": 7,
        "all_cells_included": True,
        "top_candidate_only_analysis": False,
    }
    assert frozen["constraint_attribution"]["constraint_count"] == 13
    assert frozen["execution_budget"]["exact_model_fit_count"] == 100
    assert frozen["execution_budget"]["exact_lightgbm_tree_count"] == 18_000
    assert frozen["execution_budget"]["additional_diagnostic_model_fit_count"] == 0
    assert frozen["execution_budget"]["fallback_used"] is False
    assert len(report["guard_checks"]) == 66
    assert all(row["passed"] for row in report["guard_checks"])
    assert report["public_safe_contract"]["public_report_safe"] is True


def test_constraint_catalog_preserves_stage199_thresholds() -> None:
    constraints = {row["name"]: row for row in protocol._eligibility_constraints()}

    assert len(constraints) == 13
    assert constraints["strict_success_precision"]["threshold"] == 0.65
    assert constraints["strict_opportunity_pool_recall"]["threshold"] == 0.95
    assert constraints["conditional_ranker_strict_capture"]["threshold"] == 0.68
    assert constraints["unsafe_selection_rate"] == {
        "name": "unsafe_selection_rate",
        "source": "diagnostics.unsafe_selection_rate",
        "operator": "<=",
        "threshold": 0.25,
        "signed_margin": "threshold - observed",
    }
    assert constraints["changed_question_count"]["threshold_expression"] == (
        "ceil(0.10 * inner_question_count)"
    )


def test_unconfirmed_stage_blocks_stage201(tmp_path: Path, monkeypatch) -> None:
    path = _write_stage199(tmp_path)
    monkeypatch.setattr(protocol, "STAGE199_SHA256", _hash(path))

    report = protocol.freeze_joint_risk_winner_failure_attribution_protocol(
        stage199_report_path=path,
        user_confirmed=False,
        confirmation_note="not confirmed",
    )

    assert report["decision"]["protocol_valid"] is False
    assert _failed(report) == {"user_confirmed_stage200"}


def test_source_with_eligible_cell_blocks_failure_attribution(tmp_path: Path, monkeypatch) -> None:
    path = _write_stage199(tmp_path, eligible_count=1)
    monkeypatch.setattr(protocol, "STAGE199_SHA256", _hash(path))

    report = protocol.freeze_joint_risk_winner_failure_attribution_protocol(
        stage199_report_path=path,
        user_confirmed=True,
        confirmation_note="confirmed",
    )

    assert report["decision"]["protocol_valid"] is False
    assert _failed(report) == {"source_all_outer_contexts_ineligible"}


def test_source_hash_mismatch_blocks_stage201(tmp_path: Path) -> None:
    path = _write_stage199(tmp_path)

    report = protocol.freeze_joint_risk_winner_failure_attribution_protocol(
        stage199_report_path=path,
        user_confirmed=True,
        confirmation_note="confirmed",
    )

    assert report["decision"]["protocol_valid"] is False
    assert _failed(report) == {"source_sha256_matches"}


def test_visualizations_are_valid_svg(tmp_path: Path, monkeypatch) -> None:
    path = _write_stage199(tmp_path)
    monkeypatch.setattr(protocol, "STAGE199_SHA256", _hash(path))
    report = protocol.freeze_joint_risk_winner_failure_attribution_protocol(
        stage199_report_path=path,
        user_confirmed=True,
        confirmation_note="confirmed",
    )

    visualizations = protocol.write_stage200_visualizations(
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


def _write_stage199(tmp_path: Path, *, eligible_count: int = 0) -> Path:
    path = tmp_path / "stage199.json"
    path.write_text(
        json.dumps(_stage199_report(eligible_count=eligible_count)),
        encoding="utf-8",
    )
    return path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage199_report(*, eligible_count: int) -> dict:
    inner_counts = [295, 299, 299, 290, 297]
    capture = [0.671280, 0.667832, 0.647059, 0.685512, 0.681507]
    unsafe = [0.311864, 0.305085, 0.327759, 0.272414, 0.295681]
    outer = {
        f"fold_{index}": {
            "inner_question_count": inner_counts[index - 1],
            "eligible_config_count": eligible_count if index == 1 else 0,
            "top_inner_candidates": [
                {
                    "diagnostics": {
                        "conditional_ranker_strict_capture": capture[index - 1],
                        "unsafe_selection_rate": unsafe[index - 1],
                    }
                }
            ],
        }
        for index in range(1, 6)
    }
    return {
        "stage": "Stage 199",
        "decision": {
            "status": "stage199_joint_risk_winner_insufficient",
            "experiment_valid": True,
            "candidate_family_accepted": False,
            "development_opened": False,
            "test_opened": False,
        },
        "process_guards": [{"name": f"guard_{index}", "passed": True} for index in range(34)],
        "joint_risk_winner_nested_cv": {
            "outer_contexts": outer,
            "cell_aggregates": {f"cell_{index}": {} for index in range(28)},
            "risk_signal_factor_aggregates": {f"risk_{index}": {} for index in range(4)},
            "winner_rule_factor_aggregates": {f"winner_{index}": {} for index in range(7)},
            "complete_pool_risk_metrics": {
                "source_weighted_classifier": {"roc_auc": 0.590635},
                "decomposed_loss_risk": {"roc_auc": 0.597567},
                "pairwise_safety_ranker": {"roc_auc": 0.598221},
                "decomposed_pairwise_rank_fusion": {"roc_auc": 0.603601},
            },
            "execution": {
                "all_controls_reproduced_exactly": True,
                "model_fit_count": 100,
                "tree_count": 18_000,
                "private_prediction_count": 245_960,
            },
            "advancement_gate_pass_count": 4,
            "advancement_gates": [
                {"name": f"gate_{index}", "passed": index < 4} for index in range(17)
            ],
        },
    }
