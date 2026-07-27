from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application import primeqa_hybrid_safety_first_frontier_protocol as protocol


def test_freeze_protocol_authorizes_only_stage196_train_experiment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = _write_stage194(tmp_path)
    monkeypatch.setattr(protocol, "STAGE194_SHA256", _hash(path))

    report = protocol.freeze_safety_first_frontier_protocol(
        stage194_report_path=path,
        user_confirmed=True,
        confirmation_note="continue recommended Stage 195",
    )

    assert report["decision"] == {
        "status": "stage195_safety_first_frontier_protocol_frozen",
        "protocol_valid": True,
        "stage196_train_only_experiment_authorized": True,
        "development_opened": False,
        "test_opened": False,
        "runtime_e2e_authorized": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "default_runtime_activation": False,
    }
    frozen = report["frozen_protocol"]
    assert frozen["candidate_grid"]["policy_config_count"] == 960
    assert frozen["cross_validation"]["maximum_model_fit_count"] == 600
    assert frozen["cross_validation"]["maximum_lightgbm_tree_count"] == 120_000
    assert frozen["cost_sensitive_unsafe_head"]["scale_pos_weights"] == [1.0, 2.0, 4.0]
    assert frozen["safety_first_frontier"]["safest_prefix_sizes"] == [2, 4, 8, 12, 16]
    assert frozen["safety_first_frontier"]["fallback_used"] is False
    assert len(frozen["advancement_gates"]) == 17
    assert len(report["guard_checks"]) == 58
    assert all(row["passed"] for row in report["guard_checks"])
    assert report["public_safe_contract"]["public_report_safe"] is True


def test_accepted_stage194_candidate_blocks_stage196(tmp_path: Path, monkeypatch) -> None:
    path = _write_stage194(tmp_path, candidate_accepted=True)
    monkeypatch.setattr(protocol, "STAGE194_SHA256", _hash(path))

    report = protocol.freeze_safety_first_frontier_protocol(
        stage194_report_path=path,
        user_confirmed=True,
        confirmation_note="continue",
    )

    assert report["decision"]["protocol_valid"] is False
    assert report["decision"]["stage196_train_only_experiment_authorized"] is False
    assert _failed(report) == {"source_candidate_family_rejected"}


def test_source_test_access_blocks_stage196(tmp_path: Path, monkeypatch) -> None:
    path = _write_stage194(tmp_path, test_loaded=True)
    monkeypatch.setattr(protocol, "STAGE194_SHA256", _hash(path))

    report = protocol.freeze_safety_first_frontier_protocol(
        stage194_report_path=path,
        user_confirmed=True,
        confirmation_note="continue",
    )

    assert report["decision"]["protocol_valid"] is False
    assert _failed(report) == {"source_test_closed"}


def test_source_hash_mismatch_blocks_stage196(tmp_path: Path) -> None:
    path = _write_stage194(tmp_path)

    report = protocol.freeze_safety_first_frontier_protocol(
        stage194_report_path=path,
        user_confirmed=True,
        confirmation_note="continue",
    )

    assert report["decision"]["protocol_valid"] is False
    assert _failed(report) == {"source_sha256_matches"}


def test_already_safe_source_blocks_unnecessary_stage196(tmp_path: Path, monkeypatch) -> None:
    path = _write_stage194(tmp_path, unsafe_gate_satisfied=True)
    monkeypatch.setattr(protocol, "STAGE194_SHA256", _hash(path))

    report = protocol.freeze_safety_first_frontier_protocol(
        stage194_report_path=path,
        user_confirmed=True,
        confirmation_note="continue",
    )

    assert report["decision"]["protocol_valid"] is False
    assert _failed(report) == {"unsafe_rate_still_above_gate"}


def test_visualizations_are_valid_svg(tmp_path: Path, monkeypatch) -> None:
    path = _write_stage194(tmp_path)
    monkeypatch.setattr(protocol, "STAGE194_SHA256", _hash(path))
    report = protocol.freeze_safety_first_frontier_protocol(
        stage194_report_path=path,
        user_confirmed=True,
        confirmation_note="continue",
    )

    visualizations = protocol.write_stage195_visualizations(
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


def _write_stage194(
    tmp_path: Path,
    *,
    candidate_accepted: bool = False,
    test_loaded: bool = False,
    unsafe_gate_satisfied: bool = False,
) -> Path:
    path = tmp_path / "stage194.json"
    path.write_text(
        json.dumps(
            _stage194_report(
                candidate_accepted=candidate_accepted,
                test_loaded=test_loaded,
                unsafe_gate_satisfied=unsafe_gate_satisfied,
            )
        ),
        encoding="utf-8",
    )
    return path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage194_report(
    *,
    candidate_accepted: bool,
    test_loaded: bool,
    unsafe_gate_satisfied: bool,
) -> dict:
    fold_metrics = (
        (0.989691, 0.670139, 0.657343, 0.318644),
        (0.986207, 0.629758, 0.609756, 0.342373),
        (0.989619, 0.657343, 0.651786, 0.314189),
        (0.989510, 0.681979, 0.659420, 0.300000),
        (0.986486, 0.633562, 0.621993, 0.342193),
    )
    outer_folds = {}
    for index, (pool, capture, precision, unsafe) in enumerate(fold_metrics, start=1):
        if unsafe_gate_satisfied:
            unsafe = 0.25
        outer_folds[f"fold_{index}"] = {
            "eligible_config_count": 0,
            "top_inner_candidates": [
                {
                    "evaluation": {"strict_success_precision": precision},
                    "diagnostics": {
                        "strict_opportunity_pool_recall": pool,
                        "conditional_ranker_strict_capture": capture,
                        "unsafe_selection_rate": unsafe,
                    },
                }
            ],
        }
    gib = 1024**3
    return {
        "stage": "Stage 194",
        "decision": {
            "status": "stage194_safety_constrained_lambdamart_insufficient",
            "experiment_valid": True,
            "candidate_family_accepted": candidate_accepted,
        },
        "process_guards": [{"name": f"guard_{index}", "passed": True} for index in range(33)],
        "execution_boundaries": {
            "development_loaded": False,
            "test_loaded": test_loaded,
        },
        "safety_constrained_lambdamart_nested_cv": {
            "outer_folds": outer_folds,
            "execution": {
                "model_fit_count": 320,
                "tree_count": 48_000,
                "private_prediction_count": 393_536,
            },
        },
        "resource_consumption": {
            "process_peak_working_set_bytes": int(3.762 * gib),
            "process_peak_private_usage_bytes": int(4.179 * gib),
            "minimum_system_available_memory_bytes": int(3.756 * gib),
        },
    }
