from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application import primeqa_hybrid_joint_risk_winner_protocol as protocol


def test_freeze_authorizes_only_stage199_train_experiment(tmp_path: Path, monkeypatch) -> None:
    path = _write_stage197(tmp_path)
    monkeypatch.setattr(protocol, "STAGE197_SHA256", _hash(path))

    report = protocol.freeze_joint_risk_winner_protocol(
        stage197_report_path=path,
        user_confirmed=True,
        confirmation_note="user selected A",
    )

    assert report["decision"] == {
        "status": "stage198_joint_risk_winner_protocol_frozen",
        "protocol_valid": True,
        "stage199_train_only_experiment_authorized": True,
        "development_opened": False,
        "test_opened": False,
        "runtime_e2e_authorized": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "default_runtime_activation": False,
    }
    frozen = report["frozen_protocol"]
    assert frozen["risk_signal_factor"]["family_count"] == 4
    assert frozen["winner_rule_factor"]["policy_count"] == 7
    assert frozen["factorial_ablation"]["policy_config_count_per_outer_context"] == 28
    assert frozen["cross_validation"]["maximum_model_fit_count"] == 125
    assert frozen["cross_validation"]["maximum_lightgbm_tree_count"] == 22_500
    assert frozen["frontier_contract"]["fallback_used"] is False
    assert len(frozen["advancement_gates"]) == 17
    assert len(report["guard_checks"]) == 62
    assert all(row["passed"] for row in report["guard_checks"])
    assert report["public_safe_contract"]["public_report_safe"] is True


def test_unconfirmed_route_blocks_stage199(tmp_path: Path, monkeypatch) -> None:
    path = _write_stage197(tmp_path)
    monkeypatch.setattr(protocol, "STAGE197_SHA256", _hash(path))

    report = protocol.freeze_joint_risk_winner_protocol(
        stage197_report_path=path,
        user_confirmed=False,
        confirmation_note="not confirmed",
    )

    assert report["decision"]["protocol_valid"] is False
    assert _failed(report) == {"user_confirmed_route_a"}


def test_source_test_access_blocks_stage199(tmp_path: Path, monkeypatch) -> None:
    path = _write_stage197(tmp_path, test_opened=True)
    monkeypatch.setattr(protocol, "STAGE197_SHA256", _hash(path))

    report = protocol.freeze_joint_risk_winner_protocol(
        stage197_report_path=path,
        user_confirmed=True,
        confirmation_note="A",
    )

    assert report["decision"]["protocol_valid"] is False
    assert _failed(report) == {"source_test_closed"}


def test_inexact_source_partition_blocks_stage199(tmp_path: Path, monkeypatch) -> None:
    path = _write_stage197(tmp_path, partition_exact=False)
    monkeypatch.setattr(protocol, "STAGE197_SHA256", _hash(path))

    report = protocol.freeze_joint_risk_winner_protocol(
        stage197_report_path=path,
        user_confirmed=True,
        confirmation_note="A",
    )

    assert report["decision"]["protocol_valid"] is False
    assert _failed(report) == {"source_mechanism_partition_exact"}


def test_source_hash_mismatch_blocks_stage199(tmp_path: Path) -> None:
    path = _write_stage197(tmp_path)

    report = protocol.freeze_joint_risk_winner_protocol(
        stage197_report_path=path,
        user_confirmed=True,
        confirmation_note="A",
    )

    assert report["decision"]["protocol_valid"] is False
    assert _failed(report) == {"source_sha256_matches"}


def test_visualizations_are_valid_svg(tmp_path: Path, monkeypatch) -> None:
    path = _write_stage197(tmp_path)
    monkeypatch.setattr(protocol, "STAGE197_SHA256", _hash(path))
    report = protocol.freeze_joint_risk_winner_protocol(
        stage197_report_path=path,
        user_confirmed=True,
        confirmation_note="A",
    )

    visualizations = protocol.write_stage198_visualizations(
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


def _write_stage197(
    tmp_path: Path,
    *,
    test_opened: bool = False,
    partition_exact: bool = True,
) -> Path:
    path = tmp_path / "stage197.json"
    path.write_text(
        json.dumps(
            _stage197_report(
                test_opened=test_opened,
                partition_exact=partition_exact,
            )
        ),
        encoding="utf-8",
    )
    return path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage197_report(*, test_opened: bool, partition_exact: bool) -> dict:
    outer = {
        f"fold_{index}": {
            "spec": {
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
            "top_inner_reconstruction_exact": True,
        }
        for index in range(1, 6)
    }
    return {
        "stage": "Stage 197",
        "decision": {
            "status": "stage197_surviving_unsafe_winner_attribution_complete",
            "experiment_valid": True,
            "diagnostic_complete": True,
            "development_opened": False,
            "test_opened": test_opened,
        },
        "process_guards": [{"name": f"guard_{index}", "passed": True} for index in range(29)],
        "surviving_unsafe_winner_attribution": {
            "aggregate": {
                "question_context_count": 1480,
                "unsafe_winner_context_count": 465,
                "unsafe_winner_rate": 0.314189,
                "mechanism_counts": {
                    "final_gain_dominance": 181,
                    "risk_frontier_exclusion": 97,
                    "risk_ordering_failure": 172,
                    "safety_pool_exclusion": 15,
                },
                "mechanism_partition_exact": partition_exact,
                "dominant_mechanism": "final_gain_dominance",
            },
            "unsafe_head_prediction_metrics": {
                "roc_auc": 0.589114,
                "average_precision": 0.562431,
            },
            "outer_contexts": outer,
        },
    }
