from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application import (
    primeqa_hybrid_safety_constrained_lambdamart_protocol as protocol,
)


def test_freeze_protocol_authorizes_only_stage194_train_experiment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = _write_stage192(tmp_path)
    monkeypatch.setattr(protocol, "STAGE192_SHA256", _hash(path))

    report = protocol.freeze_safety_constrained_lambdamart_protocol(
        stage192_report_path=path,
        user_confirmed=True,
        confirmation_note="selected A",
    )

    assert report["decision"] == {
        "status": "stage193_safety_constrained_lambdamart_protocol_frozen",
        "protocol_valid": True,
        "stage194_dependency_provisioning_authorized": True,
        "stage194_train_only_experiment_authorized": True,
        "development_opened": False,
        "test_opened": False,
        "runtime_e2e_authorized": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "default_runtime_activation": False,
    }
    frozen = report["frozen_protocol"]
    assert frozen["dependency_contract"]["package_requirement"] == "lightgbm==4.7.0"
    assert frozen["first_stage_pool"]["pool_cap"] == 16
    assert frozen["candidate_grid"]["policy_config_count"] == 64
    assert frozen["cross_validation"]["maximum_model_fit_count"] == 400
    assert frozen["relevance_contract"]["label_gain"] == [0, 1, 4]
    assert frozen["within_pool_selection"]["risk_penalties"] == [0.25, 0.5, 1.0, 2.0]
    assert frozen["action_contract"]["fallback_enabled"] is False
    assert len(frozen["advancement_gates"]) == 17
    assert len(report["guard_checks"]) == 59
    assert all(row["passed"] for row in report["guard_checks"])
    assert report["public_safe_contract"]["public_report_safe"] is True


def test_wrong_bottleneck_blocks_stage194(tmp_path: Path, monkeypatch) -> None:
    path = _write_stage192(tmp_path, bottleneck="candidate_pool_exclusion")
    monkeypatch.setattr(protocol, "STAGE192_SHA256", _hash(path))

    report = protocol.freeze_safety_constrained_lambdamart_protocol(
        stage192_report_path=path,
        user_confirmed=True,
        confirmation_note="selected A",
    )

    assert report["decision"]["protocol_valid"] is False
    assert report["decision"]["stage194_train_only_experiment_authorized"] is False
    failed = {row["name"] for row in report["guard_checks"] if not row["passed"]}
    assert failed == {"stage192_bottleneck_is_within_pool_ranker"}


def test_source_test_access_blocks_stage194(tmp_path: Path, monkeypatch) -> None:
    path = _write_stage192(tmp_path, test_loaded=True)
    monkeypatch.setattr(protocol, "STAGE192_SHA256", _hash(path))

    report = protocol.freeze_safety_constrained_lambdamart_protocol(
        stage192_report_path=path,
        user_confirmed=True,
        confirmation_note="selected A",
    )

    assert report["decision"]["protocol_valid"] is False
    failed = {row["name"] for row in report["guard_checks"] if not row["passed"]}
    assert failed == {"source_test_was_closed"}


def test_visualizations_are_valid_svg(tmp_path: Path, monkeypatch) -> None:
    path = _write_stage192(tmp_path)
    monkeypatch.setattr(protocol, "STAGE192_SHA256", _hash(path))
    report = protocol.freeze_safety_constrained_lambdamart_protocol(
        stage192_report_path=path,
        user_confirmed=True,
        confirmation_note="selected A",
    )

    visualizations = protocol.write_stage193_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    assert len(visualizations) == 9
    for visualization in visualizations:
        root = ET.parse(visualization.path).getroot()
        assert root.tag.endswith("svg")
        assert "Poppins" in Path(visualization.path).read_text(encoding="utf-8")


def _write_stage192(
    tmp_path: Path,
    *,
    bottleneck: str = "within_pool_ranker_miss",
    test_loaded: bool = False,
) -> Path:
    path = tmp_path / "stage192.json"
    path.write_text(
        json.dumps(_stage192_report(bottleneck=bottleneck, test_loaded=test_loaded)),
        encoding="utf-8",
    )
    return path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage192_report(*, bottleneck: str, test_loaded: bool) -> dict:
    factor_row = {
        "strict_opportunity_pool_recall": 0.95,
        "conditional_ranker_strict_capture": 0.5,
        "unsafe_selection_rate": 0.3,
    }
    return {
        "stage": "Stage 192",
        "decision": {
            "status": "stage192_rank_capped_safety_pool_failure_attribution_complete",
            "diagnostic_complete": True,
            "primary_bottleneck": bottleneck,
        },
        "process_guards": [{"name": f"guard_{index}", "passed": True} for index in range(23)],
        "execution_boundaries": {
            "development_loaded": False,
            "test_loaded": test_loaded,
        },
        "rank_capped_safety_pool_failure_attribution": {
            "reference_trajectory": {
                "question_context_count": 1480,
                "strict_opportunity_context_count": 1456,
                "opportunity_partition": {
                    "pool_exclusion_context_count": 58,
                    "ranker_miss_context_count": 500,
                    "strict_selected_context_count": 898,
                    "partition_total": 1456,
                    "partition_exact": True,
                },
                "strict_opportunity_pool_recall": 0.960165,
                "conditional_ranker_strict_capture": 0.642346,
                "actual_strict_opportunity_capture": 0.616758,
                "baseline_change_strict_precision": 0.6113,
                "unsafe_selection_rate": 0.352703,
                "ranker_miss_breakdown": {
                    "safe_zero_winner_context_count": 35,
                    "unsafe_winner_context_count": 465,
                },
            },
            "factor_aggregates": {
                "pool_cap": {
                    "16": {
                        **factor_row,
                        "strict_opportunity_pool_recall": 0.986607,
                    },
                    "all": {
                        **factor_row,
                        "strict_opportunity_pool_recall": 1.0,
                    },
                },
                "gain_ranker": {
                    "linear_listnet_top_frontier": {
                        **factor_row,
                        "conditional_ranker_strict_capture": 0.313853,
                        "unsafe_selection_rate": 0.191132,
                    },
                    "pairwise_pareto_logistic": {
                        **factor_row,
                        "conditional_ranker_strict_capture": 0.562369,
                        "unsafe_selection_rate": 0.319172,
                    },
                },
                "feature_representation": {
                    "question_relative_runtime": {
                        **factor_row,
                        "conditional_ranker_strict_capture": 0.298252,
                        "unsafe_selection_rate": 0.167821,
                    },
                    "raw_runtime": {
                        **factor_row,
                        "conditional_ranker_strict_capture": 0.578815,
                        "unsafe_selection_rate": 0.342483,
                    },
                },
            },
            "execution": {
                "private_bundle_prediction_count_consumed": 393536,
                "new_model_fit_count": 0,
            },
        },
    }
