from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application import primeqa_hybrid_joint_constraint_ranking_protocol as protocol


def test_freeze_protocol_authorizes_only_stage186_train_experiment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _write_source_reports(tmp_path)
    monkeypatch.setattr(protocol, "EXPECTED_SOURCE_SHA256", _hashes(paths))

    report = protocol.freeze_joint_constraint_ranking_protocol(
        **paths,
        user_confirmed=True,
        confirmation_note="confirmed",
    )

    assert report["decision"] == {
        "status": "stage185_joint_constraint_ranking_protocol_frozen",
        "protocol_valid": True,
        "stage186_train_only_experiment_authorized": True,
        "development_opened": False,
        "test_opened": False,
        "runtime_e2e_authorized": False,
        "replacement_policy_selected": False,
        "default_runtime_activation": False,
    }
    frozen = report["frozen_protocol"]
    assert frozen["candidate_grid"]["policy_config_count"] == 72
    assert frozen["cross_validation"]["maximum_model_head_fit_count"] == 300
    assert frozen["reference_action_contract"]["fallback_enabled"] is False
    assert frozen["cross_validation"]["no_fallback"] is True
    assert len(report["guard_checks"]) == 27
    assert all(row["passed"] for row in report["guard_checks"])
    assert report["public_safe_contract"]["public_report_safe"] is True


def test_invalid_stage184_status_blocks_stage186(tmp_path: Path, monkeypatch) -> None:
    paths = _write_source_reports(tmp_path, stage184_status="unexpected")
    monkeypatch.setattr(protocol, "EXPECTED_SOURCE_SHA256", _hashes(paths))

    report = protocol.freeze_joint_constraint_ranking_protocol(
        **paths,
        user_confirmed=True,
        confirmation_note="confirmed",
    )

    assert report["decision"]["protocol_valid"] is False
    assert report["decision"]["stage186_train_only_experiment_authorized"] is False
    failed = {row["name"] for row in report["guard_checks"] if not row["passed"]}
    assert failed == {"stage184_report_status_matches"}


def test_visualizations_are_valid_svg(tmp_path: Path, monkeypatch) -> None:
    paths = _write_source_reports(tmp_path)
    monkeypatch.setattr(protocol, "EXPECTED_SOURCE_SHA256", _hashes(paths))
    report = protocol.freeze_joint_constraint_ranking_protocol(
        **paths,
        user_confirmed=True,
        confirmation_note="confirmed",
    )

    visualizations = protocol.write_stage185_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    assert len(visualizations) == 7
    for visualization in visualizations:
        root = ET.parse(visualization.path).getroot()
        assert root.tag.endswith("svg")
        assert "Poppins" in Path(visualization.path).read_text(encoding="utf-8")


def _write_source_reports(
    tmp_path: Path,
    *,
    stage184_status: str = "stage184_f1_representation_cv_insufficient",
) -> dict[str, Path]:
    reports = {
        "stage181_report": _stage181_report(),
        "stage182_report": _stage182_report(),
        "stage183_report": _stage183_report(),
        "stage184_report": _stage184_report(stage184_status),
    }
    paths = {}
    for name, report in reports.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths[f"{name}_path"] = path
    return paths


def _hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {
        name.removesuffix("_path"): hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def _stage181_report() -> dict:
    return {
        "decision": {"status": "stage181_counterfactual_action_audit_complete"},
        "action_audit": {
            "question_count": 370,
            "nonbaseline_action_count": 11928,
            "strict_expected_action_count": 5668,
            "questions_with_strict_expected_action": 364,
            "oracle": {
                "gold_citation_delta": 58,
                "mean_answerable_f1_delta": 0.111694,
            },
        },
    }


def _stage182_report() -> dict:
    return {
        "decision": {"status": "stage182_dual_target_nested_cv_insufficient"},
        "dual_target_nested_cv": {
            "aggregate": {
                "selected_question_count": 129,
                "strict_expected_precision": 0.534884,
                "citation_loss_action_count": 4,
                "f1_regression_action_count": 55,
                "gold_citation_delta": 5,
                "mean_f1_delta_all_questions": 0.005249,
            }
        },
    }


def _stage183_report() -> dict:
    return {
        "decision": {"status": "stage183_f1_risk_failure_attribution_complete"},
        "f1_risk_attribution": {
            "selected_action_summary": {"f1_regression_rate": 0.426357},
            "safe_alternative_headroom": {
                "same_or_better_citation_safe_alternative_rate": 1.0,
                "same_or_better_safe_alternative_in_model_top3_rate": 0.527273,
                "same_or_better_safe_alternative_in_model_top5_rate": 0.781818,
            },
            "diagnostic_findings": {"primary_bottleneck": "f1_risk_separability_and_ranking"},
        },
    }


def _stage184_report(status: str) -> dict:
    representations = {
        "raw_logistic_binary": _representation(0.570751, 0.763636, 0.818182),
        "relative_hist_binary": _representation(0.592808, 0.763636, 0.836364),
        "relative_hist_ordinal": _representation(0.594294, 0.690909, 0.836364),
    }
    return {
        "decision": {
            "status": status,
            "experiment_valid": True,
            "representation_candidate_accepted": False,
        },
        "f1_representation_cv": {
            "representations": representations,
            "selection": {
                "best_raw_reference": "raw_logistic_binary",
                "selected_candidate": "relative_hist_ordinal",
                "candidate_accepted_for_nested_policy_experiment": False,
                "selected_roc_auc": 0.594294,
                "roc_auc_gain_vs_best_raw": 0.023543,
                "quality_gate_pass_count": 1,
            },
        },
    }


def _representation(auc: float, top3: float, top5: float) -> dict:
    return {
        "aggregate": {"roc_auc": auc},
        "stage182_regression_headroom": {
            "safe_alternative_top3_rate": top3,
            "safe_alternative_top5_rate": top5,
        },
    }
