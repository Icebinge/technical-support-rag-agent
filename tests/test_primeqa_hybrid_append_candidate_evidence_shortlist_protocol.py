from __future__ import annotations

import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_append_candidate_evidence_shortlist_protocol import (
    freeze_primeqa_hybrid_append_candidate_evidence_shortlist_protocol,
    write_primeqa_hybrid_append_candidate_evidence_shortlist_protocol_visualizations,
)


def test_append_candidate_evidence_shortlist_protocol_freezes_redesign_contract(
    tmp_path: Path,
) -> None:
    stage130_path = _write_stage130_fixture(tmp_path)

    report = freeze_primeqa_hybrid_append_candidate_evidence_shortlist_protocol(
        stage130_review_path=stage130_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmed Stage131 protocol freeze",
    )
    visualizations = (
        write_primeqa_hybrid_append_candidate_evidence_shortlist_protocol_visualizations(
            report=report,
            output_dir=tmp_path / "visuals",
        )
    )

    serialized = json.dumps(report, ensure_ascii=False)
    frozen = report["frozen_protocol"]
    plan = frozen["selection_and_validation_plan"]
    configs = {config["config_id"]: config for config in frozen["candidate_shortlist_configs"]}

    assert report["stage"] == "Stage 131"
    assert report["protocol_id"] == (
        "primeqa_hybrid_append_candidate_evidence_shortlist_redesign_protocol_v1"
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_append_candidate_evidence_shortlist_"
        "redesign_protocol_frozen"
    )
    assert report["decision"]["recommended_next_direction"] == (
        "run_append_candidate_evidence_shortlist_train_cv_dev_validation"
    )
    assert frozen["source_candidate_pool_contract"]["stage116_prefix_depth"] == 200
    assert frozen["source_candidate_pool_contract"]["stage128_candidate_pool_depth"] == 400
    assert frozen["source_candidate_pool_contract"]["append_candidates_are_supplemental"]
    assert configs["prefix10_append_sidecar_probe_v1"]["replacement_append_slots"] == 0
    assert configs["prefix9_append1_high_precision_v1"]["protected_prefix_slots"] == 9
    assert configs["prefix8_append2_balanced_probe_v1"]["replacement_append_slots"] == 2
    assert all(
        config["append_sidecar_can_generate_answer_text"] is False
        for config in configs.values()
    )
    assert plan["selection_split"] == "train"
    assert plan["validation_split"] == "dev"
    assert plan["primary_train_cv_guard"] == (
        "gold_citation_count_delta_vs_stage116_non_negative"
    )
    assert plan["test_rules"]["final_test_metrics_allowed"] is False
    assert plan["runtime_rules"]["default_runtime_policy"] == "unchanged"
    assert plan["runtime_rules"]["fallback_strategies_enabled"] is False
    assert report["decision"]["stage128_direct_agent_integration_path_remains_blocked"]
    assert report["decision"]["runtime_defaultization_allowed_now"] is False
    assert report["public_safe_contract"]["forbidden_keys_found"] == []
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Private fixture question" not in serialized
    assert "Private fixture answer" not in serialized
    assert '"question_text":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage131_source_failure_pressure.svg",
        "stage131_shortlist_candidate_budgets.svg",
        "stage131_validation_guard_thresholds.svg",
        "stage131_protocol_decision_flags.svg",
        "stage131_guard_check_status.svg",
    }


def test_append_candidate_evidence_shortlist_protocol_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    stage130_path = _write_stage130_fixture(tmp_path)

    report = freeze_primeqa_hybrid_append_candidate_evidence_shortlist_protocol(
        stage130_review_path=stage130_path,
        user_confirmed_protocol=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage131_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_append_candidate_evidence_shortlist_"
        "redesign_protocol_blocked"
    )


def test_append_candidate_evidence_shortlist_protocol_blocks_wrong_stage130_status(
    tmp_path: Path,
) -> None:
    stage130_path = _write_stage130_fixture(
        tmp_path,
        decision_status=(
            "primeqa_hybrid_stage129_agent_integration_failure_review_blocked"
        ),
    )

    report = freeze_primeqa_hybrid_append_candidate_evidence_shortlist_protocol(
        stage130_review_path=stage130_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmed Stage131 protocol freeze",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage130_review_completed"]["passed"] is False
    assert report["decision"]["can_run_append_shortlist_validation_now"] is False


def test_append_candidate_evidence_shortlist_protocol_blocks_wrong_next_direction(
    tmp_path: Path,
) -> None:
    stage130_path = _write_stage130_fixture(
        tmp_path,
        recommended_next_direction="some_other_route",
    )

    report = freeze_primeqa_hybrid_append_candidate_evidence_shortlist_protocol(
        stage130_review_path=stage130_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmed Stage131 protocol freeze",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage130_recommends_stage131_protocol"]["passed"] is False
    assert report["decision"]["status"].endswith("_blocked")


def test_append_candidate_evidence_shortlist_protocol_blocks_if_direct_path_unblocked(
    tmp_path: Path,
) -> None:
    stage130_path = _write_stage130_fixture(
        tmp_path,
        stage128_direct_blocked=False,
    )

    report = freeze_primeqa_hybrid_append_candidate_evidence_shortlist_protocol(
        stage130_review_path=stage130_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmed Stage131 protocol freeze",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage130_blocks_direct_stage128_integration"]["passed"] is False
    assert report["decision"]["can_continue_train_dev_development"] is False


def test_append_candidate_evidence_shortlist_protocol_blocks_missing_required_pattern(
    tmp_path: Path,
) -> None:
    stage130_path = _write_stage130_fixture(
        tmp_path,
        failure_pattern_ids=("recall_gain_not_citation_safe",),
    )

    report = freeze_primeqa_hybrid_append_candidate_evidence_shortlist_protocol(
        stage130_review_path=stage130_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmed Stage131 protocol freeze",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage130_failure_patterns_are_present"]["passed"] is False
    assert report["decision"]["can_run_append_shortlist_validation_now"] is False


def _write_stage130_fixture(
    tmp_path: Path,
    *,
    decision_status: str = (
        "primeqa_hybrid_stage129_agent_integration_failure_review_completed"
    ),
    recommended_next_direction: str = (
        "freeze_append_candidate_evidence_shortlist_redesign_protocol"
    ),
    stage128_direct_blocked: bool = True,
    failure_pattern_ids: tuple[str, ...] = (
        "recall_gain_not_citation_safe",
        "append_region_displaces_prefix_evidence",
        "changed_answer_churn_too_high",
        "dev_report_confirms_risk_direction",
    ),
) -> Path:
    path = tmp_path / "stage130.json"
    path.write_text(
        json.dumps(
            _stage130_report(
                decision_status=decision_status,
                recommended_next_direction=recommended_next_direction,
                stage128_direct_blocked=stage128_direct_blocked,
                failure_pattern_ids=failure_pattern_ids,
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _stage130_report(
    *,
    decision_status: str,
    recommended_next_direction: str,
    stage128_direct_blocked: bool,
    failure_pattern_ids: tuple[str, ...],
) -> dict:
    return {
        "stage": "Stage 130",
        "review_id": "primeqa_hybrid_stage129_agent_integration_failure_review_v1",
        "train_cv_failure_review": _split_review(
            split="train_cv",
            gold_hit_delta=9,
            gold_citation_delta=-1,
            f1_delta=0.0003,
            changed_answer_rate=0.3932,
            append_selected=42,
            prefix_like_delta=-42,
        ),
        "dev_report_only_review": _split_review(
            split="dev",
            gold_hit_delta=1,
            gold_citation_delta=-2,
            f1_delta=-0.0036,
            changed_answer_rate=0.4132,
            append_selected=12,
            prefix_like_delta=-12,
        ),
        "failure_patterns": [
            {
                "pattern_id": pattern_id,
                "severity": "blocking"
                if pattern_id == "recall_gain_not_citation_safe"
                else "high",
            }
            for pattern_id in failure_pattern_ids
        ],
        "action_boundary": {
            "stage128_direct_agent_integration_path_blocked": stage128_direct_blocked,
            "test_remains_locked": True,
            "runtime_default_policy": "unchanged",
            "fallback_strategies_enabled": False,
        },
        "guard_checks": [
            {"name": f"stage130_guard_{index}", "passed": True}
            for index in range(1, 13)
        ],
        "decision": {
            "status": decision_status,
            "recommended_next_direction": recommended_next_direction,
            "can_continue_train_dev_development": True,
            "stage128_direct_agent_integration_path_blocked": stage128_direct_blocked,
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


def _split_review(
    *,
    split: str,
    gold_hit_delta: int,
    gold_citation_delta: int,
    f1_delta: float,
    changed_answer_rate: float,
    append_selected: int,
    prefix_like_delta: int,
) -> dict:
    return {
        "split": split,
        "candidate_vs_control_deltas": {
            "gold_hit_count_at_profile_depth_delta": gold_hit_delta,
            "verified_gold_citation_count_delta": gold_citation_delta,
            "verified_average_token_f1_delta": f1_delta,
        },
        "changed_verified_answer_rate_vs_control": changed_answer_rate,
        "selected_citation_region_shift": {
            "append_region_selected_citation_count": append_selected,
            "prefix_like_selected_citation_delta": prefix_like_delta,
        },
    }
