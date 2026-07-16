from __future__ import annotations

import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_fast_filter_screening_protocol import (
    freeze_primeqa_hybrid_fast_filter_screening_protocol,
    write_primeqa_hybrid_fast_filter_screening_protocol_visualizations,
)


def test_fast_filter_screening_protocol_freezes_conservative_route(
    tmp_path: Path,
) -> None:
    stage119_path = _write_stage119_fixture(tmp_path)

    report = freeze_primeqa_hybrid_fast_filter_screening_protocol(
        stage119_report_path=stage119_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = write_primeqa_hybrid_fast_filter_screening_protocol_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    frozen = report["frozen_protocol"]
    configs = frozen["candidate_configs"]
    selection = frozen["selection_rules"]
    assert report["stage"] == "Stage 120"
    assert report["protocol_id"] == "primeqa_hybrid_fast_filter_screening_protocol_v1"
    assert report["decision"]["status"] == (
        "primeqa_hybrid_fast_filter_screening_protocol_frozen"
    )
    assert report["decision"]["recommended_next_direction"] == (
        "run_fast_filter_screening_train_cv_dev_validation"
    )
    assert frozen["fixed_candidate_pool_contract"]["candidate_pool_depth"] == 200
    assert frozen["fixed_candidate_pool_contract"][
        "screening_may_reorder_entire_top200"
    ] is False
    assert len(configs) == 6
    assert all(
        config["safety_constraints"]["full_top200_rerank_allowed"] is False
        for config in configs
    )
    assert all(
        config["fast_filter"]["protected_prefix_depth"] >= 5
        and config["safety_constraints"]["promotion_budget_top10"] <= 1
        for config in configs
    )
    assert selection["selection_split"] == "train"
    assert selection["dev_rules"]["dev_selection_allowed"] is False
    assert selection["test_rules"]["final_test_metrics_allowed"] is False
    assert selection["runtime_rules"]["default_runtime_policy"] == "unchanged"
    assert selection["runtime_rules"]["fallback_strategies_enabled"] is False
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["public_safe_contract"]["forbidden_keys_found"] == []
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Private fixture answer text" not in serialized
    assert '"question_text":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage120_stage119_stop_summary.svg",
        "stage120_candidate_family_counts.svg",
        "stage120_fast_filter_window_sizes.svg",
        "stage120_promotion_budgets.svg",
        "stage120_guard_thresholds.svg",
        "stage120_protocol_decision_flags.svg",
        "stage120_guard_check_status.svg",
    }


def test_fast_filter_screening_protocol_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    stage119_path = _write_stage119_fixture(tmp_path)

    report = freeze_primeqa_hybrid_fast_filter_screening_protocol(
        stage119_report_path=stage119_path,
        user_confirmed_protocol=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage120_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_fast_filter_screening_protocol_blocked"
    )


def test_fast_filter_screening_protocol_blocks_if_stage119_not_stopped(
    tmp_path: Path,
) -> None:
    stage119_path = _write_stage119_fixture(
        tmp_path,
        stage119_status="primeqa_hybrid_second_stage_reranking_stop_decision_blocked",
    )

    report = freeze_primeqa_hybrid_fast_filter_screening_protocol(
        stage119_report_path=stage119_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage119_stop_decision_completed"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_fast_filter_screening_protocol_blocked"
    )


def _write_stage119_fixture(
    tmp_path: Path,
    *,
    stage119_status: str = "primeqa_hybrid_second_stage_reranking_family_stopped",
) -> Path:
    path = tmp_path / "stage119.json"
    path.write_text(
        json.dumps(_stage119_report(stage119_status=stage119_status), ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _stage119_report(*, stage119_status: str) -> dict:
    return {
        "stage": "Stage 119",
        "stopped_family": {
            "family_id": "second_stage_reranking_candidate_family",
            "source_protocol_id": "primeqa_hybrid_second_stage_reranking_protocol_v1",
            "source_analysis_id": (
                "primeqa_hybrid_second_stage_reranking_train_cv_dev_validation_v1"
            ),
            "stage118_summary": {
                "selectable_config_count": 0,
                "config_count": 8,
                "train_top200_gold_present_rate": 0.9324,
                "dev_top200_gold_present_rate": 0.9079,
                "train_candidate_record_count_in_memory": 74000,
                "dev_candidate_record_count_in_memory": 15200,
                "raw_candidate_rows_written": False,
            },
            "candidate_family_summary": {
                "channel_rank_feature_reranker_family_v1": {
                    "config_count": 2,
                    "train_cv_selectable_config_count": 0,
                    "best_train_cv_objective_config_id": (
                        "crf_lexical_routes_first_v1"
                    ),
                    "best_train_cv_mrr_at_20_config_id": (
                        "crf_lexical_routes_first_v1"
                    ),
                    "train_cv_guard_failure_reasons": {
                        "train_cv_top10_regression_count_within_guard": 2,
                    },
                }
            },
            "train_cv_positive_signal_but_blocked_configs": [
                {
                    "config_id": "crf_lexical_routes_first_v1",
                    "train_cv_mrr_at_20_delta": 0.0102,
                    "failed_train_cv_guards": [
                        "train_cv_top10_regression_count_within_guard"
                    ],
                }
            ],
            "dev_report_observations": {
                "dev_used_for_selection": False,
                "dev_used_for_retuning": False,
                "dev_observations_are_non_adoptable": True,
            },
        },
        "decision": {
            "status": stage119_status,
            "stopped_family_id": "second_stage_reranking_candidate_family",
            "recommended_next_direction": "user_confirmed_next_research_direction_required",
            "requires_user_confirmation_before_next_protocol": True,
            "can_continue_train_dev_development": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "public_safe_contract": {
            "forbidden_keys_found": [],
            "test_split_loaded": False,
            "final_test_metrics_run": False,
        },
        "private_fixture_strings": ["Private fixture answer text"],
    }
