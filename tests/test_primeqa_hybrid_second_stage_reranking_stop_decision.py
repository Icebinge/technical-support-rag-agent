from __future__ import annotations

import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_second_stage_reranking_stop_decision import (
    decide_primeqa_hybrid_second_stage_reranking_stop,
    write_primeqa_hybrid_second_stage_reranking_stop_visualizations,
)


def test_second_stage_reranking_stop_decision_stops_family(tmp_path: Path) -> None:
    stage118_path = _write_stage118_fixture(tmp_path)

    report = decide_primeqa_hybrid_second_stage_reranking_stop(
        stage118_report_path=stage118_path,
        user_confirmed_stop=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = write_primeqa_hybrid_second_stage_reranking_stop_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 119"
    assert report["decision"]["status"] == (
        "primeqa_hybrid_second_stage_reranking_family_stopped"
    )
    assert report["decision"]["stopped_family_id"] == (
        "second_stage_reranking_candidate_family"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert report["stopped_family"]["stage118_summary"]["selectable_config_count"] == 0
    assert len(report["stopped_family"]["config_stop_evidence"]) == 8
    assert len(
        report["stopped_family"]["train_cv_positive_signal_but_blocked_configs"]
    ) == 2
    assert report["public_safe_contract"]["forbidden_keys_found"] == []
    assert "Private fixture answer text" not in serialized
    assert '"question_text":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage119_train_cv_objective_scores.svg",
        "stage119_train_cv_mrr_at_20_deltas.svg",
        "stage119_train_cv_hit_at_10_deltas.svg",
        "stage119_train_cv_guard_failure_reasons.svg",
        "stage119_selectability_by_family.svg",
        "stage119_stop_decision_flags.svg",
        "stage119_stop_guard_check_status.svg",
    }


def test_second_stage_reranking_stop_decision_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    stage118_path = _write_stage118_fixture(tmp_path)

    report = decide_primeqa_hybrid_second_stage_reranking_stop(
        stage118_report_path=stage118_path,
        user_confirmed_stop=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage119_stop_decision"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_second_stage_reranking_stop_decision_blocked"
    )


def test_second_stage_reranking_stop_decision_blocks_if_stage118_selected_config(
    tmp_path: Path,
) -> None:
    stage118_path = _write_stage118_fixture(
        tmp_path,
        stage118_status=(
            "primeqa_hybrid_second_stage_reranking_completed_train_cv_selected_"
            "dev_reported"
        ),
        selected_config_id="crf_route_agreement_best_rank_v1",
        selectable_config_count=1,
        dev_validation_status="reported_not_used_for_selection",
    )

    report = decide_primeqa_hybrid_second_stage_reranking_stop(
        stage118_report_path=stage118_path,
        user_confirmed_stop=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks[
        "stage118_completed_with_no_train_cv_selectable_config"
    ]["passed"] is False
    assert checks["stage118_selected_no_config"]["passed"] is False
    assert checks["all_stage118_configs_are_train_cv_nonselectable"]["passed"] is False
    assert checks["stage118_dev_report_has_no_selected_config"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_second_stage_reranking_stop_decision_blocked"
    )


def _write_stage118_fixture(
    tmp_path: Path,
    *,
    stage118_status: str = (
        "primeqa_hybrid_second_stage_reranking_completed_no_train_cv_selectable_config"
    ),
    selected_config_id: str | None = None,
    selectable_config_count: int = 0,
    dev_validation_status: str = "no_train_cv_selectable_config",
) -> Path:
    path = tmp_path / "stage118.json"
    path.write_text(
        json.dumps(
            _stage118_report(
                stage118_status=stage118_status,
                selected_config_id=selected_config_id,
                selectable_config_count=selectable_config_count,
                dev_validation_status=dev_validation_status,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _stage118_report(
    *,
    stage118_status: str,
    selected_config_id: str | None,
    selectable_config_count: int,
    dev_validation_status: str,
) -> dict:
    return {
        "stage": "Stage 118",
        "analysis_id": "primeqa_hybrid_second_stage_reranking_train_cv_dev_validation_v1",
        "split_contract": {
            "development_splits": ["train", "dev"],
            "selection_split": "train",
            "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
            "validation_split": "dev",
            "dev_validation_mode": "single_pass_report_only_no_retuning",
            "forbidden_final_splits": ["test"],
        },
        "stage117_summary": _stage117_summary(),
        "candidate_pool_summary": {
            "train": {
                "gold_present_in_top200_rate": 0.9324,
                "candidate_record_count_in_memory": 74000,
                "raw_candidate_rows_written": False,
            },
            "dev": {
                "gold_present_in_top200_rate": 0.9079,
                "candidate_record_count_in_memory": 15200,
                "raw_candidate_rows_written": False,
            },
        },
        "guard_checks": [
            {"name": "stage118_guard", "passed": True},
        ],
        "train_cv_selection": {
            "selection_split": "train",
            "selection_source": "train_cv_only",
            "selected_config_id": selected_config_id,
            "selected_family_id": (
                "channel_rank_feature_reranker_family_v1"
                if selected_config_id
                else None
            ),
            "selectable_config_count": selectable_config_count,
            "config_count": 8,
            "status": (
                "train_cv_selected"
                if selected_config_id
                else "no_train_cv_selectable_config"
            ),
            "dev_used_for_selection": False,
            "dev_used_for_retuning": False,
        },
        "dev_validation": {
            "validation_split": "dev",
            "selected_config_id": selected_config_id,
            "status": dev_validation_status,
            "dev_used_for_selection": False,
            "dev_used_for_retuning": False,
        },
        "config_results": _config_results(selected_config_id),
        "decision": {
            "status": stage118_status,
            "recommended_next_direction": "record_second_stage_reranking_stop_decision",
            "selected_config_id": selected_config_id,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "public_safe_contract": {
            "forbidden_keys_found": [],
        },
        "private_fixture_strings": ["Private fixture answer text"],
    }


def _stage117_summary() -> dict:
    return {
        "stage": "Stage 117",
        "protocol_id": "primeqa_hybrid_second_stage_reranking_protocol_v1",
        "decision_status": "primeqa_hybrid_second_stage_reranking_protocol_frozen",
        "candidate_pool_depth": 200,
        "candidate_config_count": 8,
        "selection_rules": {
            "selection_split": "train",
            "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
            "minimum_train_folds": 5,
            "dev_rules": {
                "dev_selection_allowed": False,
                "dev_retuning_allowed": False,
                "dev_threshold_tuning_allowed": False,
            },
            "test_rules": {
                "test_access_allowed": False,
                "final_test_metrics_allowed": False,
                "test_tuning_allowed": False,
            },
            "runtime_rules": {
                "default_runtime_policy": "unchanged",
                "fallback_strategies_enabled": False,
            },
        },
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
    }


def _config_results(selected_config_id: str | None) -> list[dict]:
    rows = [
        _config(
            config_id="crf_lexical_routes_first_v1",
            family_id="channel_rank_feature_reranker_family_v1",
            method="deterministic_weighted_score",
            objective=-0.0229,
            mrr=0.0102,
            hit10=-0.0162,
            hit20=-0.019,
            selected_config_id=selected_config_id,
            failed=[
                ("train_cv_hit_at_20_regression_rate_within_guard", 0.0324),
                ("train_cv_top10_regression_count_within_guard", 15),
            ],
        ),
        _config(
            config_id="crf_route_agreement_best_rank_v1",
            family_id="channel_rank_feature_reranker_family_v1",
            method="deterministic_weighted_score",
            objective=-5.9883,
            mrr=0.0018,
            hit10=0.0,
            hit20=0.0081,
            selected_config_id=selected_config_id,
            failed=[
                ("train_cv_bm25_top10_gold_demotions_to_below_50_within_guard", 3),
                ("train_cv_hit_at_20_regression_rate_within_guard", 0.0243),
                ("train_cv_top10_regression_count_within_guard", 21),
            ],
        ),
    ]
    for index in range(6):
        rows.append(
            _config(
                config_id=f"blocked_config_{index}",
                family_id=(
                    "lexical_document_feature_reranker_family_v1"
                    if index < 3
                    else "supervised_lightweight_reranker_family_v1"
                ),
                method="deterministic_weighted_score",
                objective=-20.0 - index,
                mrr=-0.05 - index / 100,
                hit10=-0.1,
                hit20=-0.1,
                selected_config_id=selected_config_id,
                failed=[
                    ("train_cv_mrr_at_20_delta_non_negative", -0.05),
                    ("train_cv_top10_regression_count_within_guard", 30 + index),
                ],
            )
        )
    return rows


def _config(
    *,
    config_id: str,
    family_id: str,
    method: str,
    objective: float,
    mrr: float,
    hit10: float,
    hit20: float,
    selected_config_id: str | None,
    failed: list[tuple[str, float | int]],
) -> dict:
    selectable = config_id == selected_config_id
    return {
        "config_id": config_id,
        "family_id": family_id,
        "ranking_method": method,
        "training_status": "succeeded",
        "training_error": None,
        "train_cv_objective_score": objective,
        "train_cv_selectable": selectable,
        "comparisons_to_baseline": {
            "train_cv": {
                "mrr_at_20_delta": mrr,
                "hit@10_delta": hit10,
                "hit@20_delta": hit20,
                "hit@200_delta": 0.0,
                "hit@200_count_delta": 0,
                "missing_count_at_200_delta": 0,
            },
            "dev": {
                "mrr_at_20_delta": mrr / 2,
                "hit@10_delta": hit10 / 2,
                "hit@20_delta": hit20 / 2,
                "hit@200_delta": 0.0,
            },
        },
        "train_cv_selection_guards": _guards(failed, selectable=selectable),
    }


def _guards(
    failed: list[tuple[str, float | int]],
    *,
    selectable: bool,
) -> list[dict]:
    failed_names = {name for name, _ in failed}
    guard_values = {
        "train_cv_hit_at_200_loss_count_within_guard": 0,
        "train_cv_bm25_top10_gold_demotions_to_below_50_within_guard": 0,
        "train_cv_hit_at_20_regression_rate_within_guard": 0.0,
        "train_cv_top10_regression_count_within_guard": 0,
        "train_cv_mrr_at_20_delta_non_negative": 0.1,
    }
    guard_values.update(dict(failed))
    return [
        {
            "name": name,
            "passed": selectable or name not in failed_names,
            "observed": value,
            "expected": "within frozen guard",
        }
        for name, value in guard_values.items()
    ]
