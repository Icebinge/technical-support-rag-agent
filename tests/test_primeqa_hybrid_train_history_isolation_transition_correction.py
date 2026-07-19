from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.audit_primeqa_hybrid_train_history_isolation_transitions import app
from ts_rag_agent.application import (
    primeqa_hybrid_train_history_isolation_transition_correction as correction,
)


def test_direction_excluded_snapshot_removes_nested_transition_counts() -> None:
    value = {
        "effect": {
            "synthetic_refusal_to_isolated_false_answer_count": 3,
            "synthetic_false_answer_to_isolated_refusal_count": 22,
            "discordant_pair_count": 25,
        },
        "other": [{"synthetic_refusal_to_isolated_false_answer_count": 1, "rate": 0.2}],
    }

    snapshot = correction._without_directional_transition_counts(value)

    assert snapshot == {
        "effect": {"discordant_pair_count": 25},
        "other": [{"rate": 0.2}],
    }
    assert correction._directional_summary_count(value) == 1


def test_correction_guards_require_exact_22_to_3_direction() -> None:
    report = _guard_report()

    checks = {check["name"]: check["passed"] for check in correction._correction_guards(report)}

    assert all(checks.values())
    report["correction"]["corrected_post_first_unanswerable"][
        "synthetic_refusal_to_isolated_false_answer_count"
    ] = 3
    checks = {check["name"]: check["passed"] for check in correction._correction_guards(report)}
    assert checks["directional_transition_labels_corrected"] is False


def test_authorization_rejects_unknown_source_hash() -> None:
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        correction._authorize_fingerprints(
            {
                "original_public": {"sha256": "0" * 64},
                "original_private": {"sha256": correction._ORIGINAL_PRIVATE_BYTE_SHA256},
            }
        )


def test_canonical_json_hash_is_order_independent() -> None:
    assert correction._canonical_json_sha256({"a": 1, "b": 2}) == (
        correction._canonical_json_sha256({"b": 2, "a": 1})
    )


def test_correction_visualizations_write_two_parseable_svgs(tmp_path: Path) -> None:
    report = _guard_report()

    visualizations = correction.write_stage165_transition_correction_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert len(visualizations) == 2
    for visualization in visualizations:
        ET.parse(visualization.path)


def test_cli_requires_explicit_stage_continuation_confirmation() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "stage-continuation" in result.stdout
    assert "--dev" not in result.stdout
    assert "--test" not in result.stdout


def test_public_report_contains_no_case_content_keys() -> None:
    report = _guard_report()

    public_safe = correction.stage165._public_safe_contract(report)

    assert public_safe["public_safe"] is True
    assert public_safe["forbidden_keys_found"] == []


def _guard_report() -> dict:
    original_decision = {
        "status": correction._ORIGINAL_STATUS,
        "candidate_eligible_for_frozen_dev_validation": False,
    }
    return {
        "user_confirmation": {"stage_continuation_confirmed": True},
        "source_authorization": {
            "fingerprints_before": {
                "original_public": {"sha256": correction._ORIGINAL_PUBLIC_SHA256},
                "original_private": {"sha256": correction._ORIGINAL_PRIVATE_BYTE_SHA256},
            },
            "original_private_canonical_sha256": (correction._ORIGINAL_PRIVATE_CANONICAL_SHA256),
        },
        "correction": {
            "original_post_first_unanswerable": {
                "synthetic_refusal_to_isolated_false_answer_count": 3,
                "synthetic_false_answer_to_isolated_refusal_count": 22,
            },
            "corrected_post_first_unanswerable": {
                "synthetic_refusal_to_isolated_false_answer_count": 22,
                "synthetic_false_answer_to_isolated_refusal_count": 3,
            },
        },
        "metric_integrity": {
            "direction_excluded_snapshot_changed": False,
            "direction_excluded_snapshot_sha256_before": "a" * 64,
            "direction_excluded_snapshot_sha256_after": "a" * 64,
            "false_answer_rate_difference_before": 0.126667,
            "false_answer_rate_difference_after": 0.126667,
            "mcnemar_p_before": 0.000157,
            "mcnemar_p_after": 0.000157,
            "original_decision": original_decision,
            "corrected_decision": original_decision,
            "decision_changed": False,
        },
        "execution_counts": {
            "public_reports_loaded": 1,
            "private_reports_loaded": 1,
            "observation_rows_read": 1124,
            "train_rows_loaded": 0,
            "development_rows_loaded": 0,
            "test_rows_loaded": 0,
            "documents_loaded": 0,
            "retrieval_runs": 0,
            "agent_runs": 0,
            "model_generation_runs": 0,
            "feature_fit_runs": 0,
            "threshold_search_runs": 0,
        },
        "closed_boundaries": {
            "development_loaded": False,
            "test_loaded": False,
            "policy_selected": False,
            "runtime_registered_as_default": False,
            "fallback_strategies_enabled": False,
        },
    }
