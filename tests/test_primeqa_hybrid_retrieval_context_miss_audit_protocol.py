import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_retrieval_context_miss_audit_protocol import (
    freeze_primeqa_hybrid_retrieval_context_miss_audit_protocol,
    write_primeqa_hybrid_retrieval_context_miss_audit_protocol_visualizations,
)


def test_retrieval_context_miss_audit_protocol_freezes_route_a(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = freeze_primeqa_hybrid_retrieval_context_miss_audit_protocol(
        stage102_report_path=paths["stage102"],
        stage107_report_path=paths["stage107"],
        stage110_report_path=paths["stage110"],
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = (
        write_primeqa_hybrid_retrieval_context_miss_audit_protocol_visualizations(
            report=report,
            output_dir=tmp_path / "visuals",
        )
    )

    serialized = json.dumps(report, ensure_ascii=False)
    frozen = report["frozen_protocol"]
    contract = frozen["stage112_run_contract"]
    assert report["stage"] == "Stage 111"
    assert report["protocol_id"] == (
        "primeqa_hybrid_retrieval_context_miss_audit_protocol_v1"
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_retrieval_context_miss_audit_protocol_frozen"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert [item["dimension_id"] for item in frozen["audit_dimensions"]] == [
        "query_expression_gap",
        "title_heading_mismatch",
        "section_boundary_or_span_locality",
        "long_document_score_dilution",
        "entity_version_error_code_mismatch",
        "bm25_field_weighting_or_index_structure",
    ]
    assert contract["reported_splits"] == ["train", "dev"]
    assert contract["final_test_metrics_allowed"] is False
    assert contract["gold_doc_id_allowed_as_runtime_feature"] is False
    assert contract["stage112_may_use_gold_doc_id_for_offline_labeling"] is True
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Private fixture answer text" not in serialized
    assert '"question_text":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage111_retrieval_context_miss_counts.svg",
        "stage111_audit_dimension_priorities.svg",
        "stage111_stage112_data_access_contract.svg",
        "stage111_protocol_decision_flags.svg",
        "stage111_guard_check_status.svg",
    }


def test_retrieval_context_miss_audit_protocol_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = freeze_primeqa_hybrid_retrieval_context_miss_audit_protocol(
        stage102_report_path=paths["stage102"],
        stage107_report_path=paths["stage107"],
        stage110_report_path=paths["stage110"],
        user_confirmed_protocol=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage111_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_retrieval_context_miss_audit_protocol_blocked"
    )


def test_retrieval_context_miss_audit_protocol_blocks_if_stage110_not_stopped(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(
        tmp_path,
        stage110_status="primeqa_hybrid_failure_pattern_redesign_stop_decision_blocked",
    )

    report = freeze_primeqa_hybrid_retrieval_context_miss_audit_protocol(
        stage102_report_path=paths["stage102"],
        stage107_report_path=paths["stage107"],
        stage110_report_path=paths["stage110"],
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage110_stopped_failure_pattern_redesign_family"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_retrieval_context_miss_audit_protocol_blocked"
    )


def _write_fixture(
    tmp_path: Path,
    *,
    stage110_status: str = "primeqa_hybrid_failure_pattern_redesign_family_stopped",
) -> dict[str, Path]:
    return {
        "stage102": _write_json(tmp_path / "stage102.json", _stage102_report()),
        "stage107": _write_json(tmp_path / "stage107.json", _stage107_report()),
        "stage110": _write_json(
            tmp_path / "stage110.json",
            _stage110_report(stage110_status=stage110_status),
        ),
    }


def _stage102_report() -> dict:
    return {
        "stage": "Stage 102",
        "analysis_id": "answer_pipeline_error_decomposition_train_dev_analysis_v1",
        "aggregate_outputs": {
            "bucket_counts_by_split": {
                "train": {
                    "answerability_false_answer": 180,
                    "retrieval_context_miss": 125,
                    "gold_span_beats_selected_answer": 174,
                },
                "dev": {
                    "answerability_false_answer": 41,
                    "retrieval_context_miss": 23,
                    "gold_span_beats_selected_answer": 41,
                },
            }
        },
        "metrics_by_split": {
            "train": {"verified": {"average_token_f1": 0.2017}},
            "dev": {"verified": {"average_token_f1": 0.2040}},
        },
        "decision": {
            "status": "primeqa_hybrid_answer_pipeline_error_decomposition_completed",
            "recommended_next_direction": "evidence_selection_and_answerability_candidate_design",
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "private_fixture_strings": ["Private fixture answer text"],
    }


def _stage107_report() -> dict:
    return {
        "stage": "Stage 107",
        "protocol_id": "primeqa_hybrid_validation_failure_pattern_analysis_v1",
        "pattern_summary": {
            "dev_failure_overview": {
                "failure_count": 117,
                "failure_rate": 0.9669,
                "answerable_failure_rate": 1.0,
            },
            "dev_retrieval_and_context_profile": {
                "answerable_gold_context_absent_count": 23,
                "answerable_gold_context_absent_rate": 0.3026,
                "answerable_gold_context_present_count": 53,
                "context_present_but_evidence_or_composition_failure_count": 53,
            },
        },
        "decision": {
            "status": "primeqa_hybrid_validation_failure_pattern_analysis_completed",
            "recommended_next_direction": "failure_pattern_driven_train_dev_redesign_protocol",
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
    }


def _stage110_report(*, stage110_status: str) -> dict:
    return {
        "stage": "Stage 110",
        "stopped_family": {
            "stage109_summary": {
                "selectable_config_count": 0,
                "config_count": 7,
            }
        },
        "decision": {
            "status": stage110_status,
            "stopped_family_id": "failure_pattern_redesign_candidate_family",
            "stopped_protocol_id": (
                "primeqa_hybrid_failure_pattern_redesign_protocol_v1"
            ),
            "recommended_next_direction": (
                "user_confirmed_next_research_direction_required"
            ),
            "requires_user_confirmation_before_next_protocol": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
    }


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
