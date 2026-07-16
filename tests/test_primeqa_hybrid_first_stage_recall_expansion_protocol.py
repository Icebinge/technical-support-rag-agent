from __future__ import annotations

import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_first_stage_recall_expansion_protocol import (
    freeze_primeqa_hybrid_first_stage_recall_expansion_protocol,
    write_primeqa_hybrid_first_stage_recall_expansion_protocol_visualizations,
)


def test_first_stage_recall_expansion_protocol_freezes_broader_pool_route(
    tmp_path: Path,
) -> None:
    stage122_path = _write_stage122_fixture(tmp_path)

    report = freeze_primeqa_hybrid_first_stage_recall_expansion_protocol(
        stage122_report_path=stage122_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = (
        write_primeqa_hybrid_first_stage_recall_expansion_protocol_visualizations(
            report=report,
            output_dir=tmp_path / "visuals",
        )
    )

    serialized = json.dumps(report, ensure_ascii=False)
    frozen = report["frozen_protocol"]
    configs = frozen["candidate_configs"]
    selection = frozen["selection_rules"]
    selected_review = report["stage122_summary"]["selected_config_review"]
    blocked_review = report["stage122_summary"]["blocked_signal_config_review"]
    assert report["stage"] == "Stage 123"
    assert report["protocol_id"] == (
        "primeqa_hybrid_first_stage_recall_expansion_protocol_v1"
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_first_stage_recall_expansion_protocol_frozen"
    )
    assert report["decision"]["recommended_next_direction"] == (
        "run_first_stage_recall_expansion_train_cv_dev_validation"
    )
    assert frozen["baseline_candidate_pool_contract"]["baseline_pool_depth"] == 200
    assert frozen["candidate_generation_contract"]["model_download_allowed"] is False
    assert frozen["candidate_generation_contract"][
        "oracle_document_metadata_allowed"
    ] is False
    assert selected_review["train_hit20_recovery_count"] == 4
    assert selected_review["dev_hit20_recovery_count"] == 0
    assert blocked_review["train_hit20_recovery_count"] == 11
    assert blocked_review["dev_hit20_regression_count"] == 1
    assert len(configs) == 7
    assert all(
        config["candidate_generation"]["target_pool_depth"] > 200
        for config in configs
    )
    assert all(
        config["candidate_generation"]["target_pool_depth"] <= 400
        and config["candidate_generation"]["channel_top_k"] <= 400
        for config in configs
    )
    assert all(
        config["feature_sources"]["requires_model_download"] is False
        and config["feature_sources"]["requires_new_embedding_build"] is False
        and config["feature_sources"]["uses_oracle_document_metadata"] is False
        and config["feature_sources"]["uses_test_membership"] is False
        for config in configs
    )
    assert selection["selection_split"] == "train"
    assert selection["minimum_train_folds"] == 5
    assert selection["guard_thresholds"]["maximum_train_cv_hit_at_200_loss_count"] == 0
    assert selection["dev_rules"]["dev_selection_allowed"] is False
    assert selection["test_rules"]["final_test_metrics_allowed"] is False
    assert selection["runtime_rules"]["default_runtime_policy"] == "unchanged"
    assert selection["runtime_rules"]["fallback_strategies_enabled"] is False
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["public_safe_contract"]["forbidden_keys_found"] == []
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Private fixture question text" not in serialized
    assert "Private fixture answer text" not in serialized
    assert '"question_text":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage123_stage122_signal_summary.svg",
        "stage123_candidate_family_counts.svg",
        "stage123_target_pool_depths.svg",
        "stage123_channel_top_k_budgets.svg",
        "stage123_guard_thresholds.svg",
        "stage123_protocol_decision_flags.svg",
        "stage123_guard_check_status.svg",
    }


def test_first_stage_recall_expansion_protocol_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    stage122_path = _write_stage122_fixture(tmp_path)

    report = freeze_primeqa_hybrid_first_stage_recall_expansion_protocol(
        stage122_report_path=stage122_path,
        user_confirmed_protocol=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage123_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_first_stage_recall_expansion_protocol_blocked"
    )


def test_first_stage_recall_expansion_protocol_blocks_if_stage122_not_completed(
    tmp_path: Path,
) -> None:
    stage122_path = _write_stage122_fixture(
        tmp_path,
        stage122_status="primeqa_hybrid_fast_filter_screening_changed_case_review_blocked",
    )

    report = freeze_primeqa_hybrid_first_stage_recall_expansion_protocol(
        stage122_report_path=stage122_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage122_changed_case_review_completed"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_first_stage_recall_expansion_protocol_blocked"
    )


def _write_stage122_fixture(
    tmp_path: Path,
    *,
    stage122_status: str = (
        "primeqa_hybrid_fast_filter_screening_changed_case_review_completed"
    ),
) -> Path:
    path = tmp_path / "stage122.json"
    path.write_text(
        json.dumps(_stage122_report(stage122_status=stage122_status), ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _stage122_report(*, stage122_status: str) -> dict:
    return {
        "stage": "Stage 122",
        "analysis_id": "primeqa_hybrid_fast_filter_screening_changed_case_review_v1",
        "config_reviews": [
            {
                "config_id": "special_token_exact_window40_rule_selector_v1",
                "interpretation": {
                    "status": "safe_but_weak",
                    "runtime_defaultization_supported": False,
                },
                "split_reviews": {
                    "train_cv": {
                        "changed_case_count": 40,
                        "improved_count": 4,
                        "regressed_count": 36,
                        "hit20_recovery_count": 4,
                        "hit20_regression_count": 3,
                    },
                    "dev": {
                        "changed_case_count": 8,
                        "improved_count": 0,
                        "regressed_count": 8,
                        "hit20_recovery_count": 0,
                        "hit20_regression_count": 0,
                    },
                },
            },
            {
                "config_id": "top10_locked_route_vote_window50_pairwise_logistic_v1",
                "interpretation": {
                    "status": "positive_signal_but_guard_risky",
                    "runtime_defaultization_supported": False,
                },
                "split_reviews": {
                    "train_cv": {
                        "changed_case_count": 40,
                        "improved_count": 11,
                        "regressed_count": 29,
                        "hit20_recovery_count": 11,
                        "hit20_regression_count": 7,
                    },
                    "dev": {
                        "changed_case_count": 8,
                        "improved_count": 2,
                        "regressed_count": 6,
                        "hit20_recovery_count": 2,
                        "hit20_regression_count": 1,
                    },
                },
            },
        ],
        "cross_config_findings": {
            "reviewed_config_count": 2,
            "blocked_signal_has_real_hit20_recoveries": True,
            "blocked_signal_has_guard_relevant_regressions": True,
            "selected_config_is_low_change": False,
        },
        "decision": {
            "status": stage122_status,
            "recommended_next_direction": "design_first_stage_recall_expansion_protocol",
            "can_continue_train_dev_development": True,
            "runtime_defaultization_supported": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "public_safe_contract": {
            "forbidden_keys_found": [],
            "raw_candidate_rows_written": False,
        },
        "private_fixture_strings": [
            "Private fixture question text",
            "Private fixture answer text",
        ],
    }
