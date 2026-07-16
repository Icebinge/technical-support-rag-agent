from __future__ import annotations

import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_retrieval_index_redesign_stop_decision import (
    decide_primeqa_hybrid_retrieval_index_redesign_stop,
    write_primeqa_hybrid_retrieval_index_redesign_stop_visualizations,
)


def test_retrieval_index_redesign_stop_decision_stops_family(tmp_path: Path) -> None:
    stage114_path = _write_stage114_fixture(tmp_path)

    report = decide_primeqa_hybrid_retrieval_index_redesign_stop(
        stage114_report_path=stage114_path,
        user_confirmed_stop=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = write_primeqa_hybrid_retrieval_index_redesign_stop_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 115"
    assert report["decision"]["status"] == (
        "primeqa_hybrid_retrieval_index_redesign_family_stopped"
    )
    assert report["decision"]["stopped_family_id"] == (
        "retrieval_index_redesign_candidate_family"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert report["stopped_family"]["stage114_summary"]["selectable_config_count"] == 0
    assert len(report["stopped_family"]["config_stop_evidence"]) == 8
    assert len(report["stopped_family"]["train_cv_improved_but_blocked_configs"]) == 4
    assert report["public_safe_contract"]["forbidden_keys_found"] == []
    assert "Private fixture answer text" not in serialized
    assert '"question_text":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage115_train_cv_retrieval_context_miss_deltas.svg",
        "stage115_train_cv_gold_doc_recall_deltas.svg",
        "stage115_train_cv_changed_answer_rates.svg",
        "stage115_train_cv_guard_failure_reasons.svg",
        "stage115_selectability_by_family.svg",
        "stage115_stop_decision_flags.svg",
        "stage115_stop_guard_check_status.svg",
    }


def test_retrieval_index_redesign_stop_decision_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    stage114_path = _write_stage114_fixture(tmp_path)

    report = decide_primeqa_hybrid_retrieval_index_redesign_stop(
        stage114_report_path=stage114_path,
        user_confirmed_stop=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage115_stop_decision"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_retrieval_index_redesign_stop_decision_blocked"
    )


def test_retrieval_index_redesign_stop_decision_blocks_if_stage114_selected_config(
    tmp_path: Path,
) -> None:
    stage114_path = _write_stage114_fixture(
        tmp_path,
        stage114_status=(
            "primeqa_hybrid_retrieval_index_redesign_completed_train_cv_selected_"
            "dev_reported"
        ),
        selected_config_id="unit_selected",
        selectable_config_count=1,
        dev_validation_status="reported_not_used_for_selection",
    )

    report = decide_primeqa_hybrid_retrieval_index_redesign_stop(
        stage114_report_path=stage114_path,
        user_confirmed_stop=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks[
        "stage114_completed_with_no_train_cv_selectable_config"
    ]["passed"] is False
    assert checks["stage114_selected_no_config"]["passed"] is False
    assert checks["stage114_dev_report_has_no_selected_config"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_retrieval_index_redesign_stop_decision_blocked"
    )


def _write_stage114_fixture(
    tmp_path: Path,
    *,
    stage114_status: str = (
        "primeqa_hybrid_retrieval_index_redesign_completed_no_train_cv_selectable_config"
    ),
    selected_config_id: str | None = None,
    selectable_config_count: int = 0,
    dev_validation_status: str = "no_train_cv_selectable_config",
) -> Path:
    path = tmp_path / "stage114.json"
    path.write_text(
        json.dumps(
            _stage114_report(
                stage114_status=stage114_status,
                selected_config_id=selected_config_id,
                selectable_config_count=selectable_config_count,
                dev_validation_status=dev_validation_status,
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _stage114_report(
    *,
    stage114_status: str,
    selected_config_id: str | None,
    selectable_config_count: int,
    dev_validation_status: str,
) -> dict:
    return {
        "stage": "Stage 114",
        "analysis_id": (
            "primeqa_hybrid_retrieval_index_redesign_train_cv_dev_validation_v1"
        ),
        "split_contract": {
            "development_splits": ["train", "dev"],
            "selection_split": "train",
            "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
            "validation_split": "dev",
            "dev_validation_mode": "single_pass_no_retuning",
            "dev_gate_status": "report_only_no_frozen_pass_threshold",
            "forbidden_final_splits": ["test"],
        },
        "stage113_summary": _stage113_summary(),
        "guard_checks": [
            {"name": "stage114_final_test_metrics_not_run", "passed": True},
            {"name": "stage114_runtime_defaults_unchanged", "passed": True},
            {"name": "stage114_fallback_strategies_not_added", "passed": True},
        ],
        "train_cv_selection": {
            "selection_split": "train",
            "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
            "selection_source": "train_cv_only",
            "baseline_train_cv_objective_score": -117.5,
            "selected_config_id": selected_config_id,
            "selected_family_id": (
                "entity_version_error_code_handling_candidate_v1"
                if selected_config_id
                else None
            ),
            "selected_train_cv_objective_delta": -14.0 if selected_config_id else None,
            "selectable_config_count": selectable_config_count,
            "config_count": 8,
        },
        "dev_validation": {
            "validation_split": "dev",
            "selected_config_id": selected_config_id,
            "status": dev_validation_status,
            "dev_validation_passed": None,
            "dev_gate_status": "report_only_no_frozen_pass_threshold",
        },
        "config_results": _config_results(selected_config_id),
        "decision": {
            "status": stage114_status,
            "recommended_next_direction": "record_retrieval_index_redesign_stop_decision",
            "selected_config_id": selected_config_id,
            "selected_family_id": (
                "entity_version_error_code_handling_candidate_v1"
                if selected_config_id
                else None
            ),
            "selectable_config_count": selectable_config_count,
            "dev_validation_status": dev_validation_status,
            "dev_gate_status": "report_only_no_frozen_pass_threshold",
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "private_fixture_strings": ["Private fixture answer text"],
    }


def _stage113_summary() -> dict:
    return {
        "stage": "Stage 113",
        "protocol_id": "primeqa_hybrid_retrieval_index_redesign_protocol_v1",
        "protocol_status": "frozen_requires_user_confirmation_before_train_dev_run",
        "candidate_config_count": 8,
        "selection_rules": {
            "selection_split": "train",
            "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
            "minimum_train_folds": 5,
            "dev_rules": {
                "dev_selection_allowed": False,
                "dev_retuning_allowed": False,
                "dev_threshold_tuning_allowed": False,
                "dev_report_required": True,
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
            config_id="evc_special_token_title_heading_boost_v1",
            family_id="entity_version_error_code_handling_candidate_v1",
            retrieval_mode="weighted_bm25_with_special_token_boost",
            objective_delta=-14.0,
            miss_delta=-4,
            recall_delta=0.0108,
            changed_rate=0.7278,
            dev_f1_delta=0.0067,
            failed=[
                "train_cv_evidence_selection_miss_delta_within_guard",
                "train_cv_gold_span_beats_selected_delta_within_guard",
                "train_cv_changed_answer_rate_within_guard",
            ],
        ),
        _config(
            config_id="evc_special_token_exact_boost_v1",
            family_id="entity_version_error_code_handling_candidate_v1",
            retrieval_mode="bm25_with_runtime_special_token_boost",
            objective_delta=-14.0,
            miss_delta=-4,
            recall_delta=0.0108,
            changed_rate=0.1833,
            dev_f1_delta=0.0041,
            failed=[
                "train_cv_answerability_false_answer_delta_within_guard",
                "train_cv_evidence_selection_miss_delta_within_guard",
            ],
        ),
        _config(
            config_id="thw_title3_heading2_body1_doc_bm25_v1",
            family_id="title_heading_weighted_bm25_candidate_v1",
            retrieval_mode="weighted_document_bm25",
            objective_delta=-10.5,
            miss_delta=-3,
            recall_delta=0.0081,
            changed_rate=0.8327,
            dev_f1_delta=0.0052,
            failed=["train_cv_changed_answer_rate_within_guard"],
        ),
        _config(
            config_id="thw_title2_heading2_body1_doc_bm25_v1",
            family_id="title_heading_weighted_bm25_candidate_v1",
            retrieval_mode="weighted_document_bm25",
            objective_delta=-10.5,
            miss_delta=-3,
            recall_delta=0.0081,
            changed_rate=0.7171,
            dev_f1_delta=0.0057,
            failed=["train_cv_changed_answer_rate_within_guard"],
        ),
        _config(
            config_id="slr_section_top3_rrf_doc_rollup_v1",
            family_id="section_level_index_rollup_candidate_v1",
            retrieval_mode="section_document_rrf",
            objective_delta=31.5,
            miss_delta=9,
            recall_delta=-0.0244,
            changed_rate=0.9804,
            dev_f1_delta=-0.0117,
            failed=["train_cv_retrieval_context_miss_delta_negative"],
        ),
        _config(
            config_id="thw_title_heading_query_view_rrf_v1",
            family_id="title_heading_weighted_bm25_candidate_v1",
            retrieval_mode="document_bm25_rrf",
            objective_delta=73.5,
            miss_delta=21,
            recall_delta=-0.0568,
            changed_rate=0.9893,
            dev_f1_delta=-0.0077,
            failed=["train_cv_retrieval_context_miss_delta_negative"],
        ),
        _config(
            config_id="slr_heading_section_title_rollup_v1",
            family_id="section_level_index_rollup_candidate_v1",
            retrieval_mode="heading_section_title_rollup",
            objective_delta=80.5,
            miss_delta=23,
            recall_delta=-0.0622,
            changed_rate=0.9875,
            dev_f1_delta=-0.0044,
            failed=["train_cv_retrieval_context_miss_delta_negative"],
        ),
        _config(
            config_id="slr_section_top1_doc_rollup_v1",
            family_id="section_level_index_rollup_candidate_v1",
            retrieval_mode="section_bm25_document_rollup",
            objective_delta=91.0,
            miss_delta=26,
            recall_delta=-0.0703,
            changed_rate=0.9840,
            dev_f1_delta=-0.0042,
            failed=["train_cv_retrieval_context_miss_delta_negative"],
        ),
    ]
    if selected_config_id:
        rows[0] = {
            **rows[0],
            "train_cv_selectability": {
                **rows[0]["train_cv_selectability"],
                "selectable": True,
                "guard_failure_reasons": [],
            },
        }
    return rows


def _config(
    *,
    config_id: str,
    family_id: str,
    retrieval_mode: str,
    objective_delta: float,
    miss_delta: int,
    recall_delta: float,
    changed_rate: float,
    dev_f1_delta: float,
    failed: list[str],
) -> dict:
    return {
        "config_id": config_id,
        "family_id": family_id,
        "retrieval_mode": retrieval_mode,
        "selection_eligible": True,
        "objective_score_deltas_by_split": {
            "train_cv": objective_delta,
            "train_full": objective_delta,
            "dev": 0.0,
        },
        "target_bucket_deltas_by_split": {
            "train_cv": {
                "retrieval_context_miss": miss_delta,
                "answerability_false_answer": 0,
                "evidence_selection_miss": 1 if failed else 0,
                "gold_span_beats_selected_answer": 0,
            },
            "train_full": {
                "retrieval_context_miss": miss_delta,
                "answerability_false_answer": 0,
                "evidence_selection_miss": 1 if failed else 0,
                "gold_span_beats_selected_answer": 0,
            },
            "dev": {
                "retrieval_context_miss": 0,
                "answerability_false_answer": 0,
                "evidence_selection_miss": 0,
                "gold_span_beats_selected_answer": 0,
            },
        },
        "retrieval_metric_deltas_by_split": {
            "train_cv": {"gold_doc_recall_at_10": recall_delta},
            "train_full": {"gold_doc_recall_at_10": recall_delta},
            "dev": {"gold_doc_recall_at_10": 0.0},
        },
        "metric_deltas_by_split": {
            "train_cv": {
                "average_token_f1": 0.0,
                "gold_doc_citation_rate": 0.0,
                "answerable_refusal_rate": 0.0,
            },
            "train_full": {
                "average_token_f1": 0.0,
                "gold_doc_citation_rate": 0.0,
                "answerable_refusal_rate": 0.0,
            },
            "dev": {
                "average_token_f1": dev_f1_delta,
                "gold_doc_citation_rate": 0.0,
                "answerable_refusal_rate": 0.0,
            },
        },
        "changed_answer_rates_by_split": {
            "train_cv": changed_rate,
            "train_full": changed_rate,
            "dev": min(changed_rate, 0.99),
        },
        "train_cv_selectability": {
            "selectable": False,
            "observed": {
                "train_cv_average_token_f1_drop": 0.0,
                "train_cv_gold_doc_citation_rate_drop": 0.0,
                "train_cv_answerable_refusal_rate_delta": 0.0,
                "train_cv_answerability_false_answer_delta": 0,
                "train_cv_evidence_selection_miss_delta": 1 if failed else 0,
                "train_cv_gold_span_beats_selected_delta": 0,
                "train_cv_changed_answer_rate": changed_rate,
            },
            "guard_failure_reasons": failed,
        },
    }
