from __future__ import annotations

import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_protocol import (
    freeze_primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol,
    write_primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol_visualizations,
)


def test_sidecar_protocol_freezes_stage116_primary_context_and_sidecar_boundaries(
    tmp_path: Path,
) -> None:
    sources = _write_source_fixtures(tmp_path)

    report = freeze_primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol(
        stage128_protocol_path=sources["stage128"],
        stage129_validation_path=sources["stage129"],
        stage133_review_path=sources["stage133"],
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmed Stage134 sidecar protocol",
    )
    visualizations = (
        write_primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol_visualizations(
            report=report,
            output_dir=tmp_path / "visuals",
        )
    )

    serialized = json.dumps(report, ensure_ascii=False)
    frozen = report["frozen_protocol"]
    channels = frozen["agent_channel_contract"]
    primary = channels["primary_answer_context_channel"]
    sidecar = channels["sidecar_observation_channel"]
    policy = frozen["agent_consumer_policy"]
    validation = frozen["validation_plan"]

    assert report["stage"] == "Stage 134"
    assert report["protocol_id"] == (
        "primeqa_hybrid_stage116_answer_context_stage128_sidecar_agent_protocol_v1"
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_stage116_answer_context_stage128_sidecar_agent_protocol_frozen"
    )
    assert report["decision"]["recommended_next_direction"] == (
        "run_stage116_answer_context_stage128_sidecar_observation_train_cv_dev_validation"
    )
    assert primary["answer_context_depth"] == 10
    assert primary["sidecar_candidates_included"] is False
    assert primary["may_be_reordered_by_sidecar"] is False
    assert primary["may_be_replaced_by_sidecar"] is False
    assert sidecar["candidate_pool_depth"] == 400
    assert sidecar["observation_slots"] == 3
    assert sidecar["allowed_to_generate_answer_text"] is False
    assert sidecar["allowed_to_replace_primary_context"] is False
    assert sidecar["allowed_to_support_agent_observation"] is True
    assert sidecar["allowed_to_support_future_citation_verification"] is True
    assert _blocked(policy, "sidecar_answer_text_generation")
    assert _blocked(policy, "sidecar_primary_context_replacement")
    assert _blocked(policy, "direct_stage128_all400_answer_context")
    assert _blocked(policy, "runtime_default_retrieval_route")
    assert _blocked(policy, "fallback_strategy_route")
    assert all(
        item["requires_stage135_validation"] is True
        for item in policy["allowed_train_dev_consumers"]
    )
    assert validation["selection_split"] == "train"
    assert validation["validation_split"] == "dev"
    assert validation["test_rules"]["final_test_metrics_allowed"] is False
    assert validation["runtime_rules"]["default_runtime_policy"] == "unchanged"
    assert validation["runtime_rules"]["fallback_strategies_enabled"] is False
    assert report["decision"]["runtime_defaultization_allowed_now"] is False
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["sidecar_can_generate_answer_text"] is False
    assert report["decision"]["sidecar_can_replace_primary_context"] is False
    assert report["decision"]["direct_stage128_all400_answer_context_remains_blocked"] is True
    assert report["public_safe_contract"]["forbidden_keys_found"] == []
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Private fixture question" not in serialized
    assert "Private fixture answer" not in serialized
    assert '"question_text":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage134_protocol_components.svg",
        "stage134_sidecar_train_dev_signals.svg",
        "stage134_channel_permission_flags.svg",
        "stage134_risk_boundary_flags.svg",
        "stage134_protocol_decision_flags.svg",
        "stage134_guard_check_status.svg",
    }


def test_sidecar_protocol_blocks_without_confirmation(tmp_path: Path) -> None:
    sources = _write_source_fixtures(tmp_path)

    report = freeze_primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol(
        stage128_protocol_path=sources["stage128"],
        stage129_validation_path=sources["stage129"],
        stage133_review_path=sources["stage133"],
        user_confirmed_protocol=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage134_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_stage116_answer_context_stage128_sidecar_agent_protocol_blocked"
    )


def test_sidecar_protocol_blocks_if_direct_stage128_path_was_not_failed(
    tmp_path: Path,
) -> None:
    sources = _write_source_fixtures(
        tmp_path,
        stage129_status="primeqa_hybrid_agent_retrieval_integration_validation_completed",
        train_cv_validation_passed=True,
        train_cv_failed_checks=[],
    )

    report = freeze_primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol(
        stage128_protocol_path=sources["stage128"],
        stage129_validation_path=sources["stage129"],
        stage133_review_path=sources["stage133"],
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmed Stage134 sidecar protocol",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage129_direct_stage128_integration_remains_blocked"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_stage116_answer_context_stage128_sidecar_agent_protocol_blocked"
    )


def test_sidecar_protocol_blocks_if_selected_sidecar_can_generate_answer_text(
    tmp_path: Path,
) -> None:
    sources = _write_source_fixtures(tmp_path, sidecar_can_generate_answer=True)

    report = freeze_primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol(
        stage128_protocol_path=sources["stage128"],
        stage129_validation_path=sources["stage129"],
        stage133_review_path=sources["stage133"],
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmed Stage134 sidecar protocol",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert (
        checks["stage133_sidecar_cannot_generate_or_replace_answer_context"]["passed"]
        is False
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_stage116_answer_context_stage128_sidecar_agent_protocol_blocked"
    )


def _blocked(policy: dict, consumer_id: str) -> bool:
    return any(
        item["consumer_id"] == consumer_id and item["blocked"] is True
        for item in policy["blocked_consumers"]
    )


def _write_source_fixtures(
    tmp_path: Path,
    *,
    stage129_status: str = (
        "primeqa_hybrid_agent_retrieval_integration_validation_blocked_or_failed"
    ),
    train_cv_validation_passed: bool = False,
    train_cv_failed_checks: list[str] | None = None,
    sidecar_can_generate_answer: bool = False,
) -> dict[str, Path]:
    paths = {
        "stage128": tmp_path / "stage128.json",
        "stage129": tmp_path / "stage129.json",
        "stage133": tmp_path / "stage133.json",
    }
    paths["stage128"].write_text(
        json.dumps(_stage128_protocol(), ensure_ascii=False),
        encoding="utf-8",
    )
    paths["stage129"].write_text(
        json.dumps(
            _stage129_validation(
                status=stage129_status,
                train_cv_validation_passed=train_cv_validation_passed,
                train_cv_failed_checks=(
                    train_cv_failed_checks
                    if train_cv_failed_checks is not None
                    else ["gold_citation_count_delta_vs_stage116_non_negative"]
                ),
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    paths["stage133"].write_text(
        json.dumps(
            _stage133_review(sidecar_can_generate_answer=sidecar_can_generate_answer),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return paths


def _stage128_protocol() -> dict:
    return {
        "stage": "Stage 128",
        "protocol_id": "primeqa_hybrid_agent_retrieval_integration_protocol_v1",
        "frozen_protocol": {
            "agent_retrieval_contract": {
                "candidate_pool_output_depth": 400,
                "candidate_pool_is_not_automatic_answer_context": True,
                "answer_context_policy": "unchanged_until_stage129_validation",
                "rank_regions": [
                    {
                        "region_id": "stage116_immutable_prefix",
                        "rank_start": 1,
                        "rank_end": 200,
                        "may_reorder": False,
                        "may_drop": False,
                        "may_insert_expansion_candidate": False,
                    },
                    {
                        "region_id": "stage128_append_expansion",
                        "rank_start": 201,
                        "rank_end": 400,
                        "source": "prefix_existing_dense_broad_append200_v1",
                        "append_budget": 200,
                        "may_insert_before_rank_201": False,
                    },
                ],
            },
        },
        "guard_checks": [
            {"name": f"stage128_guard_{index}", "passed": True}
            for index in range(17)
        ],
        "decision": {
            "status": "primeqa_hybrid_agent_retrieval_integration_protocol_frozen",
            "selected_config_id": "prefix_existing_dense_broad_append200_v1",
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
    }


def _stage129_validation(
    *,
    status: str,
    train_cv_validation_passed: bool,
    train_cv_failed_checks: list[str],
) -> dict:
    return {
        "stage": "Stage 129",
        "analysis_id": "primeqa_hybrid_agent_retrieval_integration_validation_v1",
        "train_cv_validation": {
            "deltas_vs_stage116_control": {
                "verified_average_token_f1_delta": 0.0003,
                "verified_gold_citation_count_delta": -1,
                "gold_hit_count_at_profile_depth_delta": 9,
                "changed_verified_answers": 221,
            },
        },
        "dev_report_observations": {
            "dev_changed_verified_answers_vs_stage116_control": 50,
        },
        "guard_checks": [
            {"name": f"stage129_guard_{index}", "passed": index != 21}
            for index in range(1, 22)
        ],
        "decision": {
            "analysis_id": "primeqa_hybrid_agent_retrieval_integration_validation_v1",
            "selected_profile_id": "stage128_prefix_append_top400_agent_pool",
            "train_cv_validation_passed": train_cv_validation_passed,
            "train_cv_failed_checks": train_cv_failed_checks,
            "dev_validation_status": "reported_not_used_for_selection",
            "dev_gate_status": "report_only_no_runtime_or_test_gate",
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "runtime_defaultization_allowed_now": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
            "status": status,
            "failed_checks": ["stage129_agent_answer_quality_train_cv_guard"],
        },
        "public_safe_contract": {
            "forbidden_keys_found": [],
        },
    }


def _stage133_review(*, sidecar_can_generate_answer: bool) -> dict:
    return {
        "stage": "Stage 133",
        "review_id": (
            "primeqa_hybrid_append_candidate_evidence_shortlist_"
            "selected_config_review_v1"
        ),
        "selected_config_review": {
            "config_id": "prefix10_append_sidecar_probe_v1",
            "profile_id": "stage132_prefix10_append_sidecar_probe_v1",
            "classification": "safe_but_neutral_sidecar",
            "train": {
                "verified_average_token_f1_delta": 0.0,
                "verified_gold_citation_count_delta": 0,
                "gold_hit_count_at_profile_depth_delta": 9,
                "changed_verified_answer_rate": 0.0,
            },
            "dev": {
                "verified_average_token_f1_delta": 0.0,
                "verified_gold_citation_count_delta": 0,
                "gold_hit_count_at_profile_depth_delta": 1,
                "changed_verified_answer_rate": 0.0,
            },
            "shortlist_config": {
                "protected_prefix_slots": 10,
                "replacement_append_slots": 0,
                "append_sidecar_slots": 3,
                "append_sidecar_can_generate_answer_text": sidecar_can_generate_answer,
                "append_sidecar_can_support_citation_verification": True,
            },
            "value_assessment": {
                "answer_quality_improved": False,
                "gold_citation_improved": False,
                "retrieval_coverage_improved": True,
                "answer_context_preserved": True,
                "dev_direction_confirms_neutral_safety": True,
            },
        },
        "agent_design_review": {
            "sidecar_contract": {
                "primary_answer_context_source": "Stage116 top200 evidence shortlist behavior",
                "primary_answer_context_changed": False,
                "append_candidates_can_generate_answer_text": sidecar_can_generate_answer,
                "append_candidates_can_replace_prefix_slots": False,
                "append_candidates_can_support_agent_observation": True,
                "append_candidates_can_support_future_citation_verification": True,
                "candidate_pool_depth_available_to_agent_sidecar": 400,
            },
        },
        "guard_checks": [
            {"name": f"stage133_guard_{index}", "passed": True}
            for index in range(14)
        ],
        "decision": {
            "review_id": (
                "primeqa_hybrid_append_candidate_evidence_shortlist_"
                "selected_config_review_v1"
            ),
            "selected_config_id": "prefix10_append_sidecar_probe_v1",
            "selected_profile_id": "stage132_prefix10_append_sidecar_probe_v1",
            "selected_config_classification": "safe_but_neutral_sidecar",
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "runtime_defaultization_allowed_now": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
            "status": (
                "primeqa_hybrid_append_candidate_evidence_shortlist_"
                "selected_config_review_completed"
            ),
            "selected_config_supported_for_agent_protocol_design": True,
            "selected_config_supported_for_runtime_defaultization": False,
            "selected_config_supported_for_answer_context_replacement": False,
            "replacement_append_answer_context_route_stopped": True,
            "recommended_next_direction": (
                "freeze_stage116_answer_context_plus_stage128_sidecar_agent_protocol"
            ),
        },
        "public_safe_contract": {
            "forbidden_keys_found": [],
        },
        "private_fixture_strings": [
            "Private fixture question",
            "Private fixture answer",
        ],
    }
