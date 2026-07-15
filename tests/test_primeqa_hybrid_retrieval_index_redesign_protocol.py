import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_retrieval_index_redesign_protocol import (
    freeze_primeqa_hybrid_retrieval_index_redesign_protocol,
    write_primeqa_hybrid_retrieval_index_redesign_protocol_visualizations,
)


def test_retrieval_index_redesign_protocol_freezes_candidate_families(
    tmp_path: Path,
) -> None:
    stage112_path = _write_stage112(tmp_path)

    report = freeze_primeqa_hybrid_retrieval_index_redesign_protocol(
        stage112_report_path=stage112_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = write_primeqa_hybrid_retrieval_index_redesign_protocol_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    frozen = report["frozen_protocol"]
    selection = frozen["selection_rules"]
    assert report["stage"] == "Stage 113"
    assert report["protocol_id"] == (
        "primeqa_hybrid_retrieval_index_redesign_protocol_v1"
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_retrieval_index_redesign_protocol_frozen"
    )
    assert report["decision"]["recommended_next_direction"] == (
        "run_retrieval_index_redesign_train_cv_dev_validation"
    )
    assert [family["family_id"] for family in frozen["candidate_families"]] == [
        "title_heading_weighted_bm25_candidate_v1",
        "section_level_index_rollup_candidate_v1",
        "entity_version_error_code_handling_candidate_v1",
    ]
    assert len(frozen["candidate_configs"]) == 8
    assert selection["selection_split"] == "train"
    assert selection["selection_mode"] == (
        "train_grouped_cross_validation_then_full_train_refit"
    )
    assert selection["dev_rules"]["dev_selection_allowed"] is False
    assert selection["dev_rules"]["dev_threshold_tuning_allowed"] is False
    assert selection["test_rules"]["test_access_allowed"] is False
    assert selection["test_rules"]["final_test_metrics_allowed"] is False
    assert selection["runtime_rules"]["default_runtime_policy"] == "unchanged"
    assert selection["runtime_rules"]["fallback_strategies_enabled"] is False
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert all(check["passed"] for check in report["guard_checks"])
    assert '"question_text":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage113_stage112_primary_root_causes.svg",
        "stage113_stage112_high_signal_dimensions.svg",
        "stage113_candidate_family_priorities.svg",
        "stage113_candidate_config_counts.svg",
        "stage113_selection_guard_thresholds.svg",
        "stage113_protocol_decision_flags.svg",
        "stage113_guard_check_status.svg",
    }


def test_retrieval_index_redesign_protocol_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    stage112_path = _write_stage112(tmp_path)

    report = freeze_primeqa_hybrid_retrieval_index_redesign_protocol(
        stage112_report_path=stage112_path,
        user_confirmed_protocol=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage113_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_retrieval_index_redesign_protocol_blocked"
    )


def test_retrieval_index_redesign_protocol_blocks_if_stage112_not_completed(
    tmp_path: Path,
) -> None:
    stage112_path = _write_stage112(
        tmp_path,
        decision_status=(
            "primeqa_hybrid_retrieval_context_miss_root_cause_audit_blocked"
        ),
    )

    report = freeze_primeqa_hybrid_retrieval_index_redesign_protocol(
        stage112_report_path=stage112_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage112_audit_completed"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_retrieval_index_redesign_protocol_blocked"
    )


def _write_stage112(
    tmp_path: Path,
    *,
    decision_status: str = (
        "primeqa_hybrid_retrieval_context_miss_root_cause_audit_completed"
    ),
) -> Path:
    report = {
        "stage": "Stage 112",
        "analysis_id": "primeqa_hybrid_retrieval_context_miss_root_cause_audit_v1",
        "loaded_data_summary": {
            "document_count": 28482,
            "section_count": 216648,
            "test_split_loaded": False,
        },
        "split_reports": {
            "train": {"audit_case_count": 125},
            "dev": {"audit_case_count": 23},
        },
        "cross_split_summary": {
            "answerable_rows": 446,
            "audit_case_count": 148,
            "audit_case_rate_among_answerable": 0.3318,
            "primary_root_cause_counts": {
                "title_heading_mismatch": 74,
                "query_expression_gap": 65,
                "long_document_score_dilution": 4,
                "entity_version_error_code_mismatch": 3,
                "bm25_field_weighting_or_index_structure": 2,
            },
            "dimension_high_signal_counts": {
                "title_heading_mismatch": 137,
                "bm25_field_weighting_or_index_structure": 121,
                "entity_version_error_code_mismatch": 80,
                "query_expression_gap": 65,
                "long_document_score_dilution": 37,
                "section_boundary_or_span_locality": 10,
            },
            "gold_doc_rank_bucket_counts": {
                "not_found_top50": 110,
                "rank_21_to_50": 24,
                "rank_11_to_20": 14,
            },
            "question_route_counts": {
                "other": 57,
                "error_or_log": 40,
                "install_upgrade_config": 32,
            },
            "common_train_dev_root_causes": [
                "entity_version_error_code_mismatch",
                "query_expression_gap",
                "title_heading_mismatch",
            ],
        },
        "decision": {
            "status": decision_status,
            "recommended_next_stage": "Stage113 test fixture",
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
    }
    path = tmp_path / "stage112.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
