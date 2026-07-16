from __future__ import annotations

import json
from pathlib import Path

from ts_rag_agent.application import (
    primeqa_hybrid_append_candidate_evidence_shortlist_selected_config_review as stage133_review,
)

review_selected_config = (
    stage133_review.review_primeqa_hybrid_append_candidate_evidence_shortlist_selected_config
)
write_visualizations = (
    stage133_review
    .write_primeqa_hybrid_append_candidate_evidence_shortlist_selected_config_review_visualizations
)


def test_selected_sidecar_review_supports_agent_protocol_not_runtime_default(
    tmp_path: Path,
) -> None:
    stage132_path = _write_stage132_fixture(tmp_path)

    report = review_selected_config(
        stage132_report_path=stage132_path,
        user_confirmed_review=True,
        confirmation_note="unit test confirmed Stage133 selected sidecar review",
    )
    visualizations = write_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 133"
    assert report["review_id"] == (
        "primeqa_hybrid_append_candidate_evidence_shortlist_"
        "selected_config_review_v1"
    )
    assert report["selected_config_review"]["classification"] == (
        "safe_but_neutral_sidecar"
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_append_candidate_evidence_shortlist_"
        "selected_config_review_completed"
    )
    assert report["decision"]["recommended_next_direction"] == (
        "freeze_stage116_answer_context_plus_stage128_sidecar_agent_protocol"
    )
    assert report["decision"]["selected_config_supported_for_agent_protocol_design"]
    assert report["decision"]["selected_config_supported_for_runtime_defaultization"] is False
    assert report["decision"]["selected_config_supported_for_answer_context_replacement"] is False
    assert report["replacement_route_review"]["all_replacement_configs_failed"] is True
    assert report["agent_design_review"]["sidecar_contract"][
        "primary_answer_context_changed"
    ] is False
    assert report["agent_design_review"]["sidecar_contract"][
        "append_candidates_can_generate_answer_text"
    ] is False
    assert all(check["passed"] for check in report["guard_checks"])
    assert report["public_safe_contract"]["forbidden_keys_found"] == []
    assert "Private fixture question" not in serialized
    assert "Private fixture answer" not in serialized
    assert '"question_text":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage133_selected_sidecar_train_dev_deltas.svg",
        "stage133_replacement_route_risk.svg",
        "stage133_sidecar_value_flags.svg",
        "stage133_agent_design_decision_flags.svg",
        "stage133_guard_check_status.svg",
    }


def test_selected_sidecar_review_blocks_without_confirmation(tmp_path: Path) -> None:
    stage132_path = _write_stage132_fixture(tmp_path)

    report = review_selected_config(
        stage132_report_path=stage132_path,
        user_confirmed_review=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage133_review"]["passed"] is False
    assert report["decision"]["status"].endswith("_blocked")


def test_selected_sidecar_review_blocks_wrong_stage132_status(tmp_path: Path) -> None:
    stage132_path = _write_stage132_fixture(
        tmp_path,
        decision_status=(
            "primeqa_hybrid_append_candidate_evidence_shortlist_validation_blocked"
        ),
    )

    report = review_selected_config(
        stage132_report_path=stage132_path,
        user_confirmed_review=True,
        confirmation_note="unit test confirmed Stage133 selected sidecar review",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage132_validation_completed"]["passed"] is False
    assert report["decision"]["can_continue_train_dev_development"] is False


def test_selected_sidecar_review_blocks_if_replacement_route_did_not_fail(
    tmp_path: Path,
) -> None:
    stage132_path = _write_stage132_fixture(tmp_path, replacement_guard_passed=True)

    report = review_selected_config(
        stage132_report_path=stage132_path,
        user_confirmed_review=True,
        confirmation_note="unit test confirmed Stage133 selected sidecar review",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["replacement_append_answer_context_route_should_stop"]["passed"] is False
    assert report["decision"]["selected_config_supported_for_agent_protocol_design"] is False


def _write_stage132_fixture(
    tmp_path: Path,
    *,
    decision_status: str = (
        "primeqa_hybrid_append_candidate_evidence_shortlist_validation_completed"
    ),
    replacement_guard_passed: bool = False,
) -> Path:
    path = tmp_path / "stage132.json"
    path.write_text(
        json.dumps(
            _stage132_report(
                decision_status=decision_status,
                replacement_guard_passed=replacement_guard_passed,
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _stage132_report(*, decision_status: str, replacement_guard_passed: bool) -> dict:
    return {
        "stage": "Stage 132",
        "analysis_id": "primeqa_hybrid_append_candidate_evidence_shortlist_validation_v1",
        "train_selection": {
            "selection_split": "train",
            "dev_used_for_selection": False,
            "dev_used_for_retuning": False,
            "candidate_count": 3,
            "eligible_config_count": 1,
            "selected_config_id": "prefix10_append_sidecar_probe_v1",
            "selected_profile_id": "stage132_prefix10_append_sidecar_probe_v1",
            "selected_family_id": "stage131_append_candidate_evidence_shortlist_redesign",
            "selected_train_summary": {
                "verified_average_token_f1_delta": 0.0,
                "verified_gold_citation_count_delta": 0,
                "gold_hit_count_at_profile_depth_delta": 9,
                "changed_verified_answer_rate": 0.0,
            },
            "selection_ranking": [
                {
                    "config_id": "prefix10_append_sidecar_probe_v1",
                    "profile_id": "stage132_prefix10_append_sidecar_probe_v1",
                    "protected_prefix_slots": 10,
                    "replacement_append_slots": 0,
                    "train_verified_f1_delta_vs_stage116": 0.0,
                    "train_gold_citation_count_delta_vs_stage116": 0,
                    "train_target_depth_gold_hit_delta_vs_stage116": 9,
                    "train_changed_answer_rate_vs_stage116": 0.0,
                    "guard_passed": True,
                    "failed_checks": [],
                },
                _replacement_row(
                    "prefix9_append1_high_precision_v1",
                    "stage132_prefix9_append1_high_precision_v1",
                    replacement_guard_passed,
                ),
                _replacement_row(
                    "prefix8_append2_balanced_probe_v1",
                    "stage132_prefix8_append2_balanced_probe_v1",
                    replacement_guard_passed,
                ),
            ],
        },
        "dev_report_observations": {
            "validation_split": "dev",
            "status": "reported_not_used_for_selection",
            "dev_used_for_selection": False,
            "dev_used_for_retuning": False,
            "selected_config_id": "prefix10_append_sidecar_probe_v1",
            "selected_profile_id": "stage132_prefix10_append_sidecar_probe_v1",
            "selected_dev_summary": {
                "deltas_vs_stage116_control": {
                    "verified_average_token_f1_delta": 0.0,
                    "verified_gold_citation_count_delta": 0,
                    "gold_hit_count_at_profile_depth_delta": 1,
                },
                "changed_verified_answer_rate_vs_stage116": 0.0,
            },
            "config_dev_reviews": [
                _dev_review(
                    "prefix9_append1_high_precision_v1",
                    "stage132_prefix9_append1_high_precision_v1",
                    -0.0036,
                    -2,
                    0.4132,
                    6,
                    -6,
                ),
                _dev_review(
                    "prefix8_append2_balanced_probe_v1",
                    "stage132_prefix8_append2_balanced_probe_v1",
                    -0.0053,
                    -2,
                    0.4132,
                    8,
                    -8,
                ),
            ],
        },
        "profile_reports": {
            "stage132_prefix10_append_sidecar_probe_v1": {
                "shortlist_config": {
                    "protected_prefix_slots": 10,
                    "replacement_append_slots": 0,
                    "append_sidecar_slots": 3,
                    "append_sidecar_can_generate_answer_text": False,
                    "append_sidecar_can_support_citation_verification": True,
                }
            }
        },
        "guard_checks": [
            {"name": f"stage132_guard_{index}", "passed": True}
            for index in range(1, 23)
        ],
        "decision": {
            "status": decision_status,
            "recommended_next_direction": (
                "review_append_candidate_evidence_shortlist_selected_config"
            ),
            "selected_config_id": "prefix10_append_sidecar_probe_v1",
            "selected_profile_id": "stage132_prefix10_append_sidecar_probe_v1",
            "eligible_config_count": 1,
            "can_continue_train_dev_development": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "runtime_defaultization_allowed_now": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "public_safe_contract": {
            "forbidden_keys_found": [],
        },
        "private_fixture_strings": [
            "Private fixture question",
            "Private fixture answer",
        ],
    }


def _replacement_row(config_id: str, profile_id: str, guard_passed: bool) -> dict:
    return {
        "config_id": config_id,
        "profile_id": profile_id,
        "protected_prefix_slots": 9,
        "replacement_append_slots": 1,
        "train_verified_f1_delta_vs_stage116": 0.0015,
        "train_gold_citation_count_delta_vs_stage116": 0,
        "train_changed_answer_rate_vs_stage116": 0.3932,
        "guard_passed": guard_passed,
        "failed_checks": []
        if guard_passed
        else [
            "append_selected_citations_do_not_displace_prefix_like_citations_without_gold_gain"
        ],
    }


def _dev_review(
    config_id: str,
    profile_id: str,
    f1_delta: float,
    citation_delta: int,
    churn: float,
    append_citations: int,
    prefix_delta: int,
) -> dict:
    return {
        "config_id": config_id,
        "profile_id": profile_id,
        "deltas_vs_stage116_control": {
            "verified_average_token_f1_delta": f1_delta,
            "verified_gold_citation_count_delta": citation_delta,
        },
        "changed_verified_answer_rate_vs_stage116": churn,
        "selected_citation_region_shift": {
            "append_region_selected_citation_count": append_citations,
            "prefix_like_selected_citation_delta": prefix_delta,
        },
    }
