from __future__ import annotations

import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_agent_retrieval_integration_protocol import (
    freeze_primeqa_hybrid_agent_retrieval_integration_protocol,
    write_primeqa_hybrid_agent_retrieval_integration_protocol_visualizations,
)


def test_agent_retrieval_integration_protocol_freezes_candidate_pool_contract(
    tmp_path: Path,
) -> None:
    stage127_path = _write_stage127_fixture(tmp_path)

    report = freeze_primeqa_hybrid_agent_retrieval_integration_protocol(
        stage127_review_path=stage127_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmed Stage128 protocol",
    )
    visualizations = (
        write_primeqa_hybrid_agent_retrieval_integration_protocol_visualizations(
            report=report,
            output_dir=tmp_path / "visuals",
        )
    )

    serialized = json.dumps(report, ensure_ascii=False)
    frozen = report["frozen_protocol"]
    retrieval_contract = frozen["agent_retrieval_contract"]
    consumer_policy = frozen["agent_consumer_policy"]
    validation = frozen["validation_plan"]

    assert report["stage"] == "Stage 128"
    assert report["protocol_id"] == (
        "primeqa_hybrid_agent_retrieval_integration_protocol_v1"
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_agent_retrieval_integration_protocol_frozen"
    )
    assert report["decision"]["recommended_next_direction"] == (
        "run_agent_retrieval_integration_train_cv_dev_validation"
    )
    assert retrieval_contract["candidate_pool_output_depth"] == 400
    assert retrieval_contract["candidate_pool_is_not_automatic_answer_context"] is True
    assert retrieval_contract["rank_regions"][0]["rank_end"] == 200
    assert retrieval_contract["rank_regions"][0]["may_reorder"] is False
    assert retrieval_contract["rank_regions"][1]["rank_start"] == 201
    assert retrieval_contract["rank_regions"][1]["append_budget"] == 200
    assert any(
        row["consumer_id"] == "direct_answer_context_all_400"
        and row["blocked"] is True
        for row in consumer_policy["blocked_consumers"]
    )
    assert all(
        row["requires_stage129_validation"] is True
        for row in consumer_policy["allowed_consumers"]
    )
    assert validation["selection_split"] == "train"
    assert validation["validation_split"] == "dev"
    assert validation["test_rules"]["final_test_metrics_allowed"] is False
    assert validation["runtime_rules"]["default_runtime_policy"] == "unchanged"
    assert validation["runtime_rules"]["fallback_strategies_enabled"] is False
    assert report["decision"]["runtime_defaultization_allowed_now"] is False
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["public_safe_contract"]["forbidden_keys_found"] == []
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Private fixture question" not in serialized
    assert "Private fixture answer" not in serialized
    assert '"question_text":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage128_selected_config_value.svg",
        "stage128_candidate_pool_contract.svg",
        "stage128_agent_consumer_policy.svg",
        "stage128_risk_review_flags.svg",
        "stage128_protocol_decision_flags.svg",
        "stage128_guard_check_status.svg",
    }


def test_agent_retrieval_integration_protocol_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    stage127_path = _write_stage127_fixture(tmp_path)

    report = freeze_primeqa_hybrid_agent_retrieval_integration_protocol(
        stage127_review_path=stage127_path,
        user_confirmed_protocol=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage128_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_agent_retrieval_integration_protocol_blocked"
    )


def test_agent_retrieval_integration_protocol_blocks_if_stage127_wrong_status(
    tmp_path: Path,
) -> None:
    stage127_path = _write_stage127_fixture(
        tmp_path,
        decision_status=(
            "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_"
            "selected_config_review_blocked"
        ),
    )

    report = freeze_primeqa_hybrid_agent_retrieval_integration_protocol(
        stage127_review_path=stage127_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmed Stage128 protocol",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage127_review_completed"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_agent_retrieval_integration_protocol_blocked"
    )


def _write_stage127_fixture(
    tmp_path: Path,
    *,
    decision_status: str = (
        "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_"
        "selected_config_review_completed"
    ),
) -> Path:
    path = tmp_path / "stage127.json"
    path.write_text(
        json.dumps(_stage127_report(decision_status=decision_status), ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _stage127_report(*, decision_status: str) -> dict:
    return {
        "stage": "Stage 127",
        "review_id": (
            "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_"
            "selected_config_review_v1"
        ),
        "selected_config_review": {
            "config_id": "prefix_existing_dense_broad_append200_v1",
            "family_id": "stage116_prefix_existing_dense_append_family_v1",
            "train": {
                "target_depth_hit_count_gain_vs_stage116_top200": 9,
                "incremental_gain_rate": 0.0243,
                "hit_at_200_loss_count": 0,
                "prefix_identity_violation_count": 0,
            },
            "dev": {
                "target_depth_hit_count_gain_vs_stage116_top200": 1,
                "incremental_gain_rate": 0.0132,
                "hit_at_200_loss_count": 0,
                "prefix_identity_violation_count": 0,
            },
        },
        "agent_design_review": {
            "retrieval_contract": {
                "selected_config_id": "prefix_existing_dense_broad_append200_v1",
                "baseline_prefix_depth": 200,
                "append_start_rank": 201,
                "append_budget": 200,
                "target_pool_depth": 400,
            },
            "cost_profile": {
                "candidate_depth_multiplier_vs_stage116": 2.0,
                "additional_candidates_per_query": 200,
                "channel_count": 7,
                "channel_families": {
                    "dense_cache": 2,
                    "lexical_bm25": 1,
                    "lexical_exact_token": 1,
                    "lexical_section_rollup": 1,
                    "lexical_weighted_document": 2,
                },
            },
            "risk_review": {
                "dev_gain_is_smaller_than_train_gain": True,
                "best_dev_config_differs_from_train_selected": True,
                "best_dev_config_id": "prefix_query_variant_append100_v1",
                "best_dev_target_depth_gain": 5,
                "answer_quality_not_measured": True,
                "final_test_not_run": True,
                "runtime_default_unchanged": True,
            },
        },
        "guard_checks": [
            {"name": f"stage127_guard_{index}", "passed": True}
            for index in range(1, 16)
        ],
        "decision": {
            "status": decision_status,
            "recommended_next_direction": (
                "freeze_agent_retrieval_integration_protocol_for_selected_"
                "prefix_expansion"
            ),
            "selected_config_id": "prefix_existing_dense_broad_append200_v1",
            "selected_family_id": "stage116_prefix_existing_dense_append_family_v1",
            "selected_config_supported_for_agent_protocol_design": True,
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
