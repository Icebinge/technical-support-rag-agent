from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application import (
    primeqa_hybrid_rank_capped_safety_pool_protocol as protocol,
)


def test_freeze_protocol_authorizes_only_stage191_train_experiment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = _write_stage189(tmp_path)
    monkeypatch.setattr(protocol, "STAGE189_SHA256", _hash(path))

    report = protocol.freeze_rank_capped_safety_pool_protocol(
        stage189_report_path=path,
        user_confirmed=True,
        confirmation_note="confirmed",
    )

    assert report["decision"] == {
        "status": "stage190_rank_capped_safety_pool_protocol_frozen",
        "protocol_valid": True,
        "stage191_train_only_experiment_authorized": True,
        "development_opened": False,
        "test_opened": False,
        "runtime_e2e_authorized": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "default_runtime_activation": False,
    }
    frozen = report["frozen_protocol"]
    assert frozen["candidate_grid"]["policy_config_count"] == 32
    assert frozen["candidate_grid"]["pool_caps"] == [4, 8, 16, "all"]
    assert frozen["cross_validation"]["maximum_model_fit_count"] == 300
    assert frozen["action_contract"]["fallback_enabled"] is False
    assert (
        frozen["inner_selection"]["pool_recall_constraints"][
            "aggregate_strict_opportunity_pool_recall_minimum"
        ]
        == 0.8
    )
    assert len(frozen["advancement_gates"]) == 15
    assert len(report["guard_checks"]) == 34
    assert all(row["passed"] for row in report["guard_checks"])
    assert report["public_safe_contract"]["public_report_safe"] is True


def test_non_frontier_bottleneck_blocks_stage191(tmp_path: Path, monkeypatch) -> None:
    path = _write_stage189(tmp_path, bottleneck="gain_ranker_miss")
    monkeypatch.setattr(protocol, "STAGE189_SHA256", _hash(path))

    report = protocol.freeze_rank_capped_safety_pool_protocol(
        stage189_report_path=path,
        user_confirmed=True,
        confirmation_note="confirmed",
    )

    assert report["decision"]["protocol_valid"] is False
    assert report["decision"]["stage191_train_only_experiment_authorized"] is False
    failed = {row["name"] for row in report["guard_checks"] if not row["passed"]}
    assert failed == {"stage189_primary_bottleneck_is_frontier"}


def test_visualizations_are_valid_svg(tmp_path: Path, monkeypatch) -> None:
    path = _write_stage189(tmp_path)
    monkeypatch.setattr(protocol, "STAGE189_SHA256", _hash(path))
    report = protocol.freeze_rank_capped_safety_pool_protocol(
        stage189_report_path=path,
        user_confirmed=True,
        confirmation_note="confirmed",
    )

    visualizations = protocol.write_stage190_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    assert len(visualizations) == 8
    for visualization in visualizations:
        root = ET.parse(visualization.path).getroot()
        assert root.tag.endswith("svg")
        assert "Poppins" in Path(visualization.path).read_text(encoding="utf-8")


def _write_stage189(
    tmp_path: Path,
    *,
    bottleneck: str = "safety_frontier_exclusion",
) -> Path:
    path = tmp_path / "stage189.json"
    path.write_text(json.dumps(_stage189_report(bottleneck)), encoding="utf-8")
    return path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage189_report(bottleneck: str) -> dict:
    margin = {
        "0.00": {"frontier_strict_question_recall": 0.014423},
        "0.02": {"frontier_strict_question_recall": 0.02318},
        "0.05": {"frontier_strict_question_recall": 0.03932},
        "0.10": {"frontier_strict_question_recall": 0.079499},
    }
    ranker = {
        "linear_listnet_top_frontier": {"conditional_ranker_strict_capture": 0.50933},
        "pairwise_pareto_logistic": {"conditional_ranker_strict_capture": 0.754116},
    }
    return {
        "stage": "Stage 189",
        "decision": {
            "status": "stage189_gain_sensitive_failure_attribution_complete",
            "diagnostic_complete": True,
            "primary_bottleneck": bottleneck,
        },
        "gain_sensitive_failure_attribution": {
            "top_ineligible_trajectory": {
                "question_context_count": 1480,
                "strict_opportunity_context_count": 1456,
                "opportunity_partition": {
                    "frontier_exclusion_context_count": 1280,
                    "ranker_miss_context_count": 27,
                    "strict_selected_context_count": 149,
                    "partition_total": 1456,
                    "partition_exact": True,
                },
                "frontier_strict_question_recall": 0.120879,
                "conditional_ranker_strict_capture": 0.846591,
                "strict_action_retention_rate": 0.030655,
                "mean_frontier_size": 1.725676,
                "filter_harm_context_count": 688,
                "filter_rescue_context_count": 14,
            },
            "factor_aggregates": {
                "safety_frontier_margin": margin,
                "gain_ranker": ranker,
            },
            "family_best_configurations": {
                "frontier_strict_question_recall": {
                    "spec": "best",
                    "metric": "frontier_strict_question_recall",
                    "value": 0.135989,
                    "unsafe_selected_context_count": 77,
                }
            },
            "execution": {
                "private_bundle_prediction_count_consumed": 393536,
                "new_model_fit_count": 0,
            },
        },
    }
