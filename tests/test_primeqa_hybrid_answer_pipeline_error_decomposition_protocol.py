import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_answer_pipeline_error_decomposition_protocol import (
    freeze_primeqa_hybrid_answer_pipeline_error_decomposition_protocol,
    write_primeqa_hybrid_answer_pipeline_protocol_visualizations,
)


def test_answer_pipeline_protocol_freezes_confirmed_stage101(tmp_path):
    stage100_path = _write_json(tmp_path / "stage100.json", _stage100_report())

    report = freeze_primeqa_hybrid_answer_pipeline_error_decomposition_protocol(
        stage100_report_path=stage100_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = write_primeqa_hybrid_answer_pipeline_protocol_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 101"
    assert report["decision"]["status"] == (
        "primeqa_hybrid_answer_pipeline_error_decomposition_protocol_frozen"
    )
    assert report["decision"]["protocol_id"] == (
        "answer_pipeline_error_decomposition_train_dev_v1"
    )
    assert report["decision"][
        "can_run_train_dev_error_decomposition_after_user_confirmation"
    ]
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["can_use_test_for_tuning"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert [
        bucket["bucket_id"]
        for bucket in report["frozen_protocol"]["bucket_assignment_contract"]["buckets"]
    ] == [
        "answerability_false_answer",
        "retrieval_context_miss",
        "evidence_selection_miss",
        "verification_over_refusal",
        "gold_span_beats_selected_answer",
        "low_overlap_gold_cited_answer",
        "answer_supported_and_cited",
    ]
    assert "Restart the database service" not in serialized
    assert "Install the firmware update" not in serialized
    assert "How do I use this private answer pipeline question" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage101_error_bucket_priority_weights.svg",
        "stage101_pipeline_stage_order.svg",
        "stage101_public_case_field_counts.svg",
        "stage101_output_artifact_contract.svg",
        "stage101_protocol_decision_flags.svg",
        "stage101_guard_check_status.svg",
    }


def test_answer_pipeline_protocol_blocks_without_confirmation(tmp_path):
    stage100_path = _write_json(tmp_path / "stage100.json", _stage100_report())

    report = freeze_primeqa_hybrid_answer_pipeline_error_decomposition_protocol(
        stage100_report_path=stage100_path,
        user_confirmed_protocol=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage101_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_answer_pipeline_error_decomposition_protocol_blocked"
    )
    assert (
        report["decision"][
            "can_run_train_dev_error_decomposition_after_user_confirmation"
        ]
        is False
    )


def test_answer_pipeline_protocol_blocks_if_stage100_did_not_recommend_route(tmp_path):
    stage100 = _stage100_report()
    stage100["decision"]["recommended_next_direction"] = "third_wave_retrieval_design"
    stage100_path = _write_json(tmp_path / "stage100.json", stage100)

    report = freeze_primeqa_hybrid_answer_pipeline_error_decomposition_protocol(
        stage100_report_path=stage100_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert (
        checks["stage100_recommends_answer_pipeline_decomposition"]["passed"]
        is False
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_answer_pipeline_error_decomposition_protocol_blocked"
    )


def test_answer_pipeline_protocol_keeps_public_case_fields_sanitized(tmp_path):
    stage100_path = _write_json(tmp_path / "stage100.json", _stage100_report())

    report = freeze_primeqa_hybrid_answer_pipeline_error_decomposition_protocol(
        stage100_report_path=stage100_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )

    fields = set(
        report["frozen_protocol"]["public_safe_output_contract"]["case_sample_fields"]
    )
    assert not (
        fields
        & {
            "question_text",
            "raw_answer_text",
            "gold_answer",
            "document_id",
            "answer_doc_id",
            "retrieved_doc_ids",
            "cited_doc_ids",
            "matched_token_strings",
        }
    )


def _stage100_report() -> dict:
    return {
        "stage": "Stage 100",
        "aggregate_summary": {
            "first_wave_retrieval_candidates_exhausted": True,
            "second_wave_expected_candidate_count": 5,
            "second_wave_stopped_candidate_count": 5,
            "second_wave_all_expected_candidates_stopped": True,
            "runtime_advancing_second_wave_candidate_count": 0,
            "best_second_wave_dev_hit10_delta": 0.0,
            "best_second_wave_top10_net": 0,
            "stage99_route_family_exhausted": True,
            "remaining_actionable_candidate_count": 0,
            "blocked_source_doc_ids_diagnostic_status": (
                "blocked_from_train_dev_experiment"
            ),
            "second_wave_retrieval_route_family_exhausted": True,
        },
        "decision": {
            "status": "primeqa_hybrid_second_wave_route_exhaustion_summary_completed",
            "first_wave_retrieval_candidates_exhausted": True,
            "second_wave_retrieval_route_family_exhausted": True,
            "runtime_advancing_second_wave_candidate_count": 0,
            "remaining_actionable_candidate_count": 0,
            "recommended_next_direction": "answer_pipeline_error_decomposition",
            "requires_user_confirmation_before_next_protocol": True,
            "can_continue_train_dev_development": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        },
        "private_fixture_strings": [
            "Restart the database service",
            "Install the firmware update",
            "How do I use this private answer pipeline question",
        ],
    }


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
