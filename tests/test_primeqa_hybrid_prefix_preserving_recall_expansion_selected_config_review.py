from __future__ import annotations

import json
from pathlib import Path

from ts_rag_agent.application import (
    primeqa_hybrid_prefix_preserving_recall_expansion_selected_config_review as stage127_review,
)

review_selected_config = (
    stage127_review.review_primeqa_hybrid_prefix_preserving_recall_expansion_selected_config
)
write_visualizations = (
    stage127_review
    .write_primeqa_hybrid_prefix_preserving_recall_expansion_selected_config_review_visualizations
)


def test_selected_config_review_supports_next_agent_protocol_design(
    tmp_path: Path,
) -> None:
    stage126_path = _write_stage126_fixture(tmp_path)

    report = review_selected_config(
        stage126_report_path=stage126_path,
        user_confirmed_review=True,
        confirmation_note="unit test confirmed Stage127 review",
    )
    visualizations = write_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 127"
    assert report["review_id"] == (
        "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_"
        "selected_config_review_v1"
    )
    assert report["selected_config_review"]["config_id"] == (
        "prefix_existing_dense_broad_append200_v1"
    )
    assert report["agent_design_review"]["runtime_defaultization_allowed_now"] is False
    assert report["agent_design_review"]["final_test_gate_allowed_now"] is False
    assert report["agent_design_review"]["retrieval_contract"]["rank_regions"][0][
        "may_reorder"
    ] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_"
        "selected_config_review_completed"
    )
    assert report["decision"]["recommended_next_direction"] == (
        "freeze_agent_retrieval_integration_protocol_for_selected_prefix_expansion"
    )
    assert report["decision"]["runtime_defaultization_allowed_now"] is False
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert all(check["passed"] for check in report["guard_checks"])
    assert report["public_safe_contract"]["forbidden_keys_found"] == []
    assert "Private fixture question" not in serialized
    assert "Private fixture answer" not in serialized
    assert '"question_text":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage127_selected_incremental_recall.svg",
        "stage127_config_train_dev_gain.svg",
        "stage127_boundary_safety.svg",
        "stage127_candidate_pool_shape.svg",
        "stage127_decision_flags.svg",
        "stage127_guard_check_status.svg",
    }


def test_selected_config_review_blocks_without_confirmation(tmp_path: Path) -> None:
    stage126_path = _write_stage126_fixture(tmp_path)

    report = review_selected_config(
        stage126_report_path=stage126_path,
        user_confirmed_review=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage127_review"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_"
        "selected_config_review_blocked"
    )


def test_selected_config_review_blocks_if_stage126_status_is_wrong(
    tmp_path: Path,
) -> None:
    stage126_path = _write_stage126_fixture(
        tmp_path,
        decision_status=(
            "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_"
            "validation_blocked"
        ),
    )

    report = review_selected_config(
        stage126_report_path=stage126_path,
        user_confirmed_review=True,
        confirmation_note="unit test confirmed Stage127 review",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage126_validation_completed"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_"
        "selected_config_review_blocked"
    )


def _write_stage126_fixture(
    tmp_path: Path,
    *,
    decision_status: str = (
        "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_validation_completed"
    ),
) -> Path:
    path = tmp_path / "stage126.json"
    path.write_text(
        json.dumps(_stage126_report(decision_status=decision_status), ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _stage126_report(*, decision_status: str) -> dict:
    return {
        "stage": "Stage 126",
        "analysis_id": (
            "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_validation_v1"
        ),
        "baseline_by_split": {
            "train": {
                "evaluated_questions": 370,
                "pool_depth": 200,
                "hit_at_200_count": 345,
                "hit_at_200": 0.9324,
            },
            "dev": {
                "evaluated_questions": 76,
                "pool_depth": 200,
                "hit_at_200_count": 69,
                "hit_at_200": 0.9079,
            },
        },
        "train_selection": {
            "selection_split": "train",
            "selection_mode": (
                "train_grouped_cross_validation_prefix_preserving_candidate_selection"
            ),
            "dev_used_for_selection": False,
            "dev_used_for_retuning": False,
            "candidate_count": 6,
            "eligible_config_count": 6,
            "selected_config_id": "prefix_existing_dense_broad_append200_v1",
            "selected_family_id": "stage116_prefix_existing_dense_append_family_v1",
        },
        "dev_report_observations": {
            "dev_used_for_selection": False,
            "dev_used_for_retuning": False,
            "dev_reported_only": True,
            "selected_config_id": "prefix_existing_dense_broad_append200_v1",
        },
        "config_reviews": [
            _config_review(
                config_id="prefix_existing_dense_broad_append200_v1",
                family_id="stage116_prefix_existing_dense_append_family_v1",
                append_budget=200,
                target_pool_depth=400,
                train_gain=9,
                dev_gain=1,
            ),
            _config_review(
                config_id="prefix_rrf_same_routes_append200_k60_v1",
                family_id="stage116_prefix_rrf_append_family_v1",
                append_budget=200,
                target_pool_depth=400,
                train_gain=9,
                dev_gain=1,
            ),
            _config_review(
                config_id="prefix_query_variant_append100_v1",
                family_id="stage116_prefix_query_variant_append_family_v1",
                append_budget=100,
                target_pool_depth=300,
                train_gain=7,
                dev_gain=5,
            ),
            _config_review(
                config_id="prefix_rrf_same_routes_append100_k60_v1",
                family_id="stage116_prefix_rrf_append_family_v1",
                append_budget=100,
                target_pool_depth=300,
                train_gain=7,
                dev_gain=0,
            ),
            _config_review(
                config_id="prefix_route_balanced_append200_v1",
                family_id="stage116_prefix_route_balanced_append_family_v1",
                append_budget=200,
                target_pool_depth=400,
                train_gain=7,
                dev_gain=2,
            ),
            _config_review(
                config_id="prefix_rrf_same_routes_append100_k80_v1",
                family_id="stage116_prefix_rrf_append_family_v1",
                append_budget=100,
                target_pool_depth=300,
                train_gain=4,
                dev_gain=0,
            ),
        ],
        "guard_checks": [
            {"name": f"stage126_guard_{index}", "passed": True}
            for index in range(1, 22)
        ],
        "decision": {
            "status": decision_status,
            "recommended_next_direction": (
                "review_stage116_prefix_preserving_recall_expansion_selected_config"
            ),
            "selected_config_id": "prefix_existing_dense_broad_append200_v1",
            "selected_family_id": "stage116_prefix_existing_dense_append_family_v1",
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
            "Private fixture question",
            "Private fixture answer",
        ],
    }


def _config_review(
    *,
    config_id: str,
    family_id: str,
    append_budget: int,
    target_pool_depth: int,
    train_gain: int,
    dev_gain: int,
) -> dict:
    return {
        "config_id": config_id,
        "family_id": family_id,
        "source_stage124_config_id": f"{config_id}__source",
        "append_source_algorithm": "cached_dense_plus_lexical_rrf",
        "route_set": "stage116_lexical_routes_plus_existing_dense_cache_routes",
        "channel_top_k": target_pool_depth,
        "append_budget": append_budget,
        "target_pool_depth": target_pool_depth,
        "split_reviews": {
            "train": _split_review(
                split="train",
                config_id=config_id,
                evaluated=370,
                baseline=345,
                target_hits=345 + train_gain,
                gain=train_gain,
                append_budget=append_budget,
                target_pool_depth=target_pool_depth,
            ),
            "dev": _split_review(
                split="dev",
                config_id=config_id,
                evaluated=76,
                baseline=69,
                target_hits=69 + dev_gain,
                gain=dev_gain,
                append_budget=append_budget,
                target_pool_depth=target_pool_depth,
            ),
        },
        "train_cv_guard": {"passed": True, "failed_checks": []},
    }


def _split_review(
    *,
    split: str,
    config_id: str,
    evaluated: int,
    baseline: int,
    target_hits: int,
    gain: int,
    append_budget: int,
    target_pool_depth: int,
) -> dict:
    fold_metrics = (
        {
            "fold_1": {"target_depth_hit_count_gain_vs_stage116_top200": 2},
            "fold_2": {"target_depth_hit_count_gain_vs_stage116_top200": 3},
            "fold_3": {"target_depth_hit_count_gain_vs_stage116_top200": 2},
            "fold_4": {"target_depth_hit_count_gain_vs_stage116_top200": 0},
            "fold_5": {"target_depth_hit_count_gain_vs_stage116_top200": 2},
        }
        if split == "train" and config_id == "prefix_existing_dense_broad_append200_v1"
        else {}
    )
    return {
        "split": split,
        "config_id": config_id,
        "target_pool_depth": target_pool_depth,
        "append_budget": append_budget,
        "evaluated_questions": evaluated,
        "baseline_hit_at_200_count": baseline,
        "hit_at_200_count": baseline,
        "hit_at_200_delta_vs_stage116_prefix": 0,
        "hit_at_200_loss_count": 0,
        "target_depth_hit_count": target_hits,
        "target_depth_hit_rate": round(target_hits / evaluated, 4),
        "target_depth_hit_count_gain_vs_stage116_top200": gain,
        "appended_gold_recovery_count": gain,
        "prefix_identity_violation_count": 0,
        "append_budget_exceeded_count": 0,
        "append_exhaustion_count": 0,
        "candidate_pool_size": {
            "average": float(target_pool_depth),
            "median": float(target_pool_depth),
            "p95": float(target_pool_depth),
            "max": target_pool_depth,
        },
        "append_count": {
            "average": float(append_budget),
            "median": float(append_budget),
            "p95": float(append_budget),
            "max": append_budget,
            "budget": append_budget,
        },
        "channel_count": 7,
        "channel_families": {
            "dense_cache": 2,
            "lexical_bm25": 1,
            "lexical_exact_token": 1,
            "lexical_section_rollup": 1,
            "lexical_weighted_document": 2,
        },
        "fold_metrics": fold_metrics,
    }
