from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application import (
    primeqa_hybrid_gain_sensitive_ranking_protocol as protocol,
)


def test_freeze_protocol_authorizes_only_stage188_train_experiment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _write_source_reports(tmp_path)
    monkeypatch.setattr(protocol, "EXPECTED_SOURCE_SHA256", _hashes(paths))

    report = protocol.freeze_gain_sensitive_ranking_protocol(
        **paths,
        user_confirmed=True,
        confirmation_note="confirmed",
    )

    assert report["decision"] == {
        "status": "stage187_gain_sensitive_ranking_protocol_frozen",
        "protocol_valid": True,
        "stage188_train_only_experiment_authorized": True,
        "development_opened": False,
        "test_opened": False,
        "runtime_e2e_authorized": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "default_runtime_activation": False,
    }
    frozen = report["frozen_protocol"]
    assert frozen["candidate_grid"]["policy_config_count"] == 32
    assert frozen["cross_validation"]["maximum_model_fit_count"] == 300
    assert frozen["action_contract"]["fallback_enabled"] is False
    assert frozen["cross_validation"]["no_fallback"] is True
    assert (
        frozen["relative_safety_frontier"][
            "mathematically_nonempty_for_every_nonempty_candidate_set"
        ]
        is True
    )
    assert len(report["guard_checks"]) == 41
    assert all(row["passed"] for row in report["guard_checks"])
    assert report["public_safe_contract"]["public_report_safe"] is True


def test_invalid_stage186_status_blocks_stage188(tmp_path: Path, monkeypatch) -> None:
    paths = _write_source_reports(tmp_path, stage186_status="unexpected")
    monkeypatch.setattr(protocol, "EXPECTED_SOURCE_SHA256", _hashes(paths))

    report = protocol.freeze_gain_sensitive_ranking_protocol(
        **paths,
        user_confirmed=True,
        confirmation_note="confirmed",
    )

    assert report["decision"]["protocol_valid"] is False
    assert report["decision"]["stage188_train_only_experiment_authorized"] is False
    failed = {row["name"] for row in report["guard_checks"] if not row["passed"]}
    assert failed == {"stage186_report_status_matches"}


def test_protocol_omits_incomparable_pair_preferences(tmp_path: Path, monkeypatch) -> None:
    paths = _write_source_reports(tmp_path)
    monkeypatch.setattr(protocol, "EXPECTED_SOURCE_SHA256", _hashes(paths))

    report = protocol.freeze_gain_sensitive_ranking_protocol(
        **paths,
        user_confirmed=True,
        confirmation_note="confirmed",
    )

    gain = report["frozen_protocol"]["gain_ranker_contract"]
    pairwise = gain["pairwise_pareto_logistic"]
    assert "omit incomparable" in " ".join(pairwise["preference_rule"])
    assert pairwise["pair_sampling"] is False
    assert gain["incomparable_tradeoff_preference_fabricated"] is False


def test_visualizations_are_valid_svg(tmp_path: Path, monkeypatch) -> None:
    paths = _write_source_reports(tmp_path)
    monkeypatch.setattr(protocol, "EXPECTED_SOURCE_SHA256", _hashes(paths))
    report = protocol.freeze_gain_sensitive_ranking_protocol(
        **paths,
        user_confirmed=True,
        confirmation_note="confirmed",
    )

    visualizations = protocol.write_stage187_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    assert len(visualizations) == 8
    for visualization in visualizations:
        root = ET.parse(visualization.path).getroot()
        assert root.tag.endswith("svg")
        assert "Poppins" in Path(visualization.path).read_text(encoding="utf-8")


def _write_source_reports(
    tmp_path: Path,
    *,
    stage186_status: str = "stage186_joint_constraint_ranking_insufficient",
) -> dict[str, Path]:
    reports = {
        "stage181_report": _stage181_report(),
        "stage182_report": _stage182_report(),
        "stage183_report": _stage183_report(),
        "stage184_report": _stage184_report(),
        "stage185_report": _stage185_report(),
        "stage186_report": _stage186_report(stage186_status),
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
                "strict_expected_count": 69,
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
            "safe_alternative_headroom": {
                "same_or_better_citation_safe_alternative_rate": 1.0,
                "same_or_better_safe_alternative_in_model_top3_rate": 0.527273,
                "same_or_better_safe_alternative_in_model_top5_rate": 0.781818,
            },
            "diagnostic_findings": {"primary_bottleneck": "f1_risk_separability_and_ranking"},
        },
    }


def _stage184_report() -> dict:
    return {
        "decision": {"status": "stage184_f1_representation_cv_insufficient"},
        "f1_representation_cv": {
            "selection": {
                "selected_candidate": "relative_hist_ordinal",
                "selected_roc_auc": 0.594294,
                "quality_gate_pass_count": 1,
            }
        },
    }


def _stage185_report() -> dict:
    return {
        "decision": {"status": "stage185_joint_constraint_ranking_protocol_frozen"},
        "frozen_protocol": {
            "candidate_grid": {"policy_config_count": 72},
            "cross_validation": {"maximum_model_head_fit_count": 300},
            "advancement_gates": [
                {
                    "name": "strict_success_precision",
                    "operator": ">=",
                    "threshold": 0.65,
                    "unit": "rate",
                }
            ],
        },
    }


def _stage186_report(status: str) -> dict:
    def head(auc: float, ap: float) -> dict[str, float]:
        return {"roc_auc": auc, "average_precision": ap}

    gates = [
        {"name": "gate_a", "passed": True},
        {"name": "strict_success_precision_at_least_0_65", "passed": False},
    ]
    return {
        "decision": {
            "status": status,
            "experiment_valid": True,
            "candidate_family_accepted": False,
        },
        "joint_constraint_nested_cv": {
            "aggregate": {
                "question_count": 370,
                "changed_question_count": 130,
                "strict_success_count": 0,
                "strict_success_precision": 0.0,
                "citation_gain_action_count": 0,
                "citation_loss_action_count": 0,
                "f1_regression_action_count": 0,
                "gold_citation_delta": 0,
                "mean_f1_delta": 0.0,
                "citation_delta_vs_reference": -5,
                "mean_f1_delta_vs_reference": -0.005249,
                "repaired_reference_regression_count": 53,
            },
            "head_metrics": {
                "citation_loss": head(0.863981, 0.360039),
                "f1_loss": head(0.605203, 0.537792),
                "strict_gain": head(0.602056, 0.543826),
            },
            "selected_spec_counts": {
                (
                    "raw_runtime__class_balanced_logistic__"
                    "max_safety_risk_lexicographic__safety_0.02__gain_0.00"
                ): 5
            },
            "advancement_gates": gates,
            "advancement_gate_pass_count": 1,
        },
    }
