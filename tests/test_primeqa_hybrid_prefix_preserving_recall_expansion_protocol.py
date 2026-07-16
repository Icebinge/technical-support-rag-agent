from __future__ import annotations

import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_prefix_preserving_recall_expansion_protocol import (
    freeze_primeqa_hybrid_prefix_preserving_recall_expansion_protocol,
    write_primeqa_hybrid_prefix_preserving_recall_expansion_protocol_visualizations,
)


def test_prefix_preserving_recall_expansion_protocol_freezes_append_only_route(
    tmp_path: Path,
) -> None:
    stage124_path = _write_stage124_fixture(tmp_path)

    report = freeze_primeqa_hybrid_prefix_preserving_recall_expansion_protocol(
        stage124_report_path=stage124_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = (
        write_primeqa_hybrid_prefix_preserving_recall_expansion_protocol_visualizations(
            report=report,
            output_dir=tmp_path / "visuals",
        )
    )

    serialized = json.dumps(report, ensure_ascii=False)
    frozen = report["frozen_protocol"]
    configs = frozen["candidate_configs"]
    selection = frozen["selection_rules"]
    prefix = frozen["baseline_prefix_contract"]
    assert report["stage"] == "Stage 125"
    assert report["protocol_id"] == (
        "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_protocol_v1"
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_protocol_frozen"
    )
    assert report["decision"]["recommended_next_direction"] == (
        "run_stage116_prefix_preserving_recall_expansion_train_cv_dev_validation"
    )
    assert prefix["prefix_depth"] == 200
    assert prefix["ranks_1_to_200_must_remain_identical"] is True
    assert prefix["prefix_documents_may_be_reordered"] is False
    assert prefix["prefix_documents_may_be_dropped"] is False
    assert len(configs) == 6
    assert all(
        config["prefix_preservation"]["ranks_1_to_200_must_remain_identical"] is True
        and config["prefix_preservation"]["may_reorder_prefix"] is False
        and config["prefix_preservation"]["may_drop_prefix_documents"] is False
        and config["prefix_preservation"]["may_insert_before_rank_201"] is False
        for config in configs
    )
    assert all(
        config["append_generation"]["append_start_rank"] == 201
        and config["append_generation"]["target_pool_depth"] in {300, 400}
        and config["append_generation"]["append_budget"]
        == config["append_generation"]["target_pool_depth"] - 200
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
    assert (
        selection["guard_thresholds"][
            "maximum_train_cv_prefix_identity_violation_count"
        ]
        == 0
    )
    assert selection["guard_thresholds"]["maximum_train_cv_hit_at_200_loss_count"] == 0
    assert selection["dev_rules"]["dev_selection_allowed"] is False
    assert selection["test_rules"]["final_test_metrics_allowed"] is False
    assert selection["runtime_rules"]["default_runtime_policy"] == "unchanged"
    assert selection["runtime_rules"]["fallback_strategies_enabled"] is False
    assert report["public_safe_contract"]["forbidden_keys_found"] == []
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Private fixture question text" not in serialized
    assert "Private fixture answer text" not in serialized
    assert '"question_text":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage125_stage124_blocked_signal_summary.svg",
        "stage125_candidate_family_counts.svg",
        "stage125_append_budgets.svg",
        "stage125_target_pool_depths.svg",
        "stage125_guard_thresholds.svg",
        "stage125_protocol_decision_flags.svg",
        "stage125_guard_check_status.svg",
    }


def test_prefix_preserving_recall_expansion_protocol_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    stage124_path = _write_stage124_fixture(tmp_path)

    report = freeze_primeqa_hybrid_prefix_preserving_recall_expansion_protocol(
        stage124_report_path=stage124_path,
        user_confirmed_protocol=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage125_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_protocol_blocked"
    )


def test_prefix_preserving_recall_expansion_protocol_blocks_if_stage124_wrong_status(
    tmp_path: Path,
) -> None:
    stage124_path = _write_stage124_fixture(
        tmp_path,
        stage124_status="primeqa_hybrid_first_stage_recall_expansion_validation_blocked",
    )

    report = freeze_primeqa_hybrid_prefix_preserving_recall_expansion_protocol(
        stage124_report_path=stage124_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage124_validation_completed_no_selection"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_protocol_blocked"
    )


def _write_stage124_fixture(
    tmp_path: Path,
    *,
    stage124_status: str = (
        "primeqa_hybrid_first_stage_recall_expansion_validation_completed_no_selection"
    ),
) -> Path:
    path = tmp_path / "stage124.json"
    path.write_text(
        json.dumps(_stage124_report(stage124_status=stage124_status), ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _stage124_report(*, stage124_status: str) -> dict:
    return {
        "stage": "Stage 124",
        "analysis_id": (
            "primeqa_hybrid_first_stage_recall_expansion_train_cv_dev_validation_v1"
        ),
        "baseline_by_split": {
            "train": {
                "evaluated_questions": 370,
                "hit_at_200_count": 345,
                "hit_at_200": 0.9324,
                "pool_depth": 200,
            },
            "dev": {
                "evaluated_questions": 76,
                "hit_at_200_count": 69,
                "hit_at_200": 0.9079,
                "pool_depth": 200,
            },
        },
        "train_selection": {
            "candidate_count": 7,
            "eligible_config_count": 0,
            "selected_config_id": None,
            "selected_family_id": None,
        },
        "config_reviews": [
            _config_review(
                config_id="rrf_same_routes_top400_k60_v1",
                family_id="rrf_depth_expansion_family_v1",
                target_pool_depth=400,
                train_gain=9,
                train_delta=1,
                train_losses=1,
                dev_gain=1,
                dev_delta=0,
                dev_losses=0,
            ),
            _config_review(
                config_id="existing_dense_cache_broad_union_top400_v1",
                family_id="existing_dense_cache_union_family_v1",
                target_pool_depth=400,
                train_gain=9,
                train_delta=1,
                train_losses=1,
                dev_gain=1,
                dev_delta=0,
                dev_losses=0,
            ),
            _config_review(
                config_id="rrf_same_routes_top300_k60_v1",
                family_id="rrf_depth_expansion_family_v1",
                target_pool_depth=300,
                train_gain=7,
                train_delta=-1,
                train_losses=1,
                dev_gain=0,
                dev_delta=0,
                dev_losses=0,
            ),
        ],
        "decision": {
            "status": stage124_status,
            "recommended_next_direction": (
                "design_stage116_prefix_preserving_recall_expansion_protocol"
            ),
            "selected_config_id": None,
            "selected_family_id": None,
            "positive_target_depth_signal_blocked_by_hit_at_200_loss": True,
            "can_continue_train_dev_development": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "public_safe_contract": {
            "forbidden_keys_found": [],
        },
        "private_fixture_strings": [
            "Private fixture question text",
            "Private fixture answer text",
        ],
    }


def _config_review(
    *,
    config_id: str,
    family_id: str,
    target_pool_depth: int,
    train_gain: int,
    train_delta: int,
    train_losses: int,
    dev_gain: int,
    dev_delta: int,
    dev_losses: int,
) -> dict:
    return {
        "config_id": config_id,
        "family_id": family_id,
        "target_pool_depth": target_pool_depth,
        "split_reviews": {
            "train": {
                "target_depth_hit_count_gain_vs_baseline_top200": train_gain,
                "hit_at_200_delta_vs_baseline": train_delta,
                "hit_at_200_loss_count": train_losses,
            },
            "dev": {
                "target_depth_hit_count_gain_vs_baseline_top200": dev_gain,
                "hit_at_200_delta_vs_baseline": dev_delta,
                "hit_at_200_loss_count": dev_losses,
            },
        },
        "train_cv_guard": {"passed": False},
    }

