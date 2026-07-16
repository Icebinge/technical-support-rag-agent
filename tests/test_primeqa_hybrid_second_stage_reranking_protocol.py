from __future__ import annotations

import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_second_stage_reranking_protocol import (
    freeze_primeqa_hybrid_second_stage_reranking_protocol,
    write_primeqa_hybrid_second_stage_reranking_protocol_visualizations,
)


def test_second_stage_reranking_protocol_freezes_top200_contract(
    tmp_path: Path,
) -> None:
    stage116_path = _write_stage116_fixture(tmp_path)

    report = freeze_primeqa_hybrid_second_stage_reranking_protocol(
        stage116_report_path=stage116_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = write_primeqa_hybrid_second_stage_reranking_protocol_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    frozen = report["frozen_protocol"]
    selection = frozen["selection_rules"]
    pool_contract = frozen["fixed_candidate_pool_contract"]
    assert report["stage"] == "Stage 117"
    assert report["protocol_id"] == (
        "primeqa_hybrid_second_stage_reranking_protocol_v1"
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_second_stage_reranking_protocol_frozen"
    )
    assert report["decision"]["recommended_next_direction"] == (
        "run_second_stage_reranking_train_cv_dev_validation"
    )
    assert pool_contract["candidate_pool_depth"] == 200
    assert pool_contract["reranker_may_reorder_pool"] is True
    assert pool_contract["reranker_may_add_documents"] is False
    assert len(frozen["candidate_configs"]) == 8
    assert [family["family_id"] for family in frozen["candidate_families"]] == [
        "channel_rank_feature_reranker_family_v1",
        "lexical_document_feature_reranker_family_v1",
        "supervised_lightweight_reranker_family_v1",
    ]
    assert selection["selection_split"] == "train"
    assert selection["minimum_train_folds"] == 5
    assert selection["dev_rules"]["dev_selection_allowed"] is False
    assert selection["dev_rules"]["dev_threshold_tuning_allowed"] is False
    assert selection["test_rules"]["test_access_allowed"] is False
    assert selection["test_rules"]["final_test_metrics_allowed"] is False
    assert selection["runtime_rules"]["default_runtime_policy"] == "unchanged"
    assert selection["runtime_rules"]["fallback_strategies_enabled"] is False
    assert frozen["candidate_artifact_contract"][
        "candidate_rows_not_built_in_stage117"
    ] is True
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["public_safe_contract"]["forbidden_keys_found"] == []
    assert all(check["passed"] for check in report["guard_checks"])
    assert '"question_text":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage117_stage116_candidate_pool_recall.svg",
        "stage117_candidate_family_priorities.svg",
        "stage117_candidate_config_counts.svg",
        "stage117_objective_weights.svg",
        "stage117_guard_thresholds.svg",
        "stage117_protocol_decision_flags.svg",
        "stage117_guard_check_status.svg",
    }


def test_second_stage_reranking_protocol_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    stage116_path = _write_stage116_fixture(tmp_path)

    report = freeze_primeqa_hybrid_second_stage_reranking_protocol(
        stage116_report_path=stage116_path,
        user_confirmed_protocol=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage117_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_second_stage_reranking_protocol_blocked"
    )


def test_second_stage_reranking_protocol_blocks_if_stage116_not_completed(
    tmp_path: Path,
) -> None:
    stage116_path = _write_stage116_fixture(
        tmp_path,
        decision_status="primeqa_hybrid_high_recall_union_candidate_pool_blocked",
    )

    report = freeze_primeqa_hybrid_second_stage_reranking_protocol(
        stage116_report_path=stage116_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage116_completed"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_second_stage_reranking_protocol_blocked"
    )


def _write_stage116_fixture(
    tmp_path: Path,
    *,
    decision_status: str = "primeqa_hybrid_high_recall_union_candidate_pool_completed",
) -> Path:
    report = {
        "stage": "Stage 116",
        "created_at": "2026-07-16",
        "analysis_id": "primeqa_hybrid_high_recall_union_candidate_pool_v1",
        "split_contract": {
            "split_name": "primeqa_hybrid_stage68_v1",
            "protocol_version": "primeqa_hybrid_split_v1",
            "development_splits": ["train", "dev"],
            "dev_selection_used": False,
            "dev_retuning_used": False,
            "forbidden_final_splits": ["test"],
        },
        "analysis_config": {
            "candidate_pool_id": "stage116_multi_route_union_candidate_pool",
            "channel_top_k": 200,
            "pool_top_k_values": [10, 20, 50, 100, 200],
            "rrf_k": 60,
        },
        "loaded_data_summary": {
            "test_split_loaded": False,
            "split_samples": {
                "train": {"answerable_count": 370},
                "dev": {"answerable_count": 76},
            },
        },
        "dense_channel_preflight": {
            "status": "dense_channels_ready",
            "can_run_without_download": True,
        },
        "channel_catalog": [{"channel_id": f"route_{index}"} for index in range(7)],
        "candidate_pool_metrics_by_split": {
            "train": {
                "hit_at_k": {"100": 0.8973, "200": 0.9324},
                "uncapped_union_hit_rate": 0.9676,
                "candidate_pool_size": {"average": 643.6135, "p95": 778.0},
            },
            "dev": {
                "hit_at_k": {"100": 0.8684, "200": 0.9079},
                "uncapped_union_hit_rate": 0.9474,
                "candidate_pool_size": {"average": 662.2632, "p95": 806.0},
            },
        },
        "comparisons_to_baseline": {
            "dev": {
                "hit@100": {"hit_count_delta": 3},
                "hit@200": {"hit_count_delta": 3},
            }
        },
        "guard_checks": [
            {"name": f"guard_{index}", "passed": True} for index in range(11)
        ],
        "decision": {
            "status": decision_status,
            "recommended_next_direction": (
                "design_second_stage_precision_reranking_protocol_over_stage116_pool"
            ),
            "can_continue_second_stage_precision_experiment": True,
            "analysis_result": {
                "dev_uncapped_union_not_found_count": 4,
            },
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "public_safe_contract": {
            "forbidden_keys_found": [],
        },
    }
    path = tmp_path / "stage116.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
