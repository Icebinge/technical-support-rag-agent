import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_evidence_answerability_candidate_protocol import (
    freeze_primeqa_hybrid_evidence_answerability_candidate_protocol,
    write_primeqa_hybrid_evidence_answerability_protocol_visualizations,
)


def test_evidence_answerability_candidate_protocol_freezes_confirmed_design(tmp_path):
    stage102_path = _write_json(tmp_path / "stage102.json", _stage102_report())

    report = freeze_primeqa_hybrid_evidence_answerability_candidate_protocol(
        stage102_report_path=stage102_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = write_primeqa_hybrid_evidence_answerability_protocol_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 103"
    assert report["decision"]["status"] == (
        "primeqa_hybrid_evidence_answerability_candidate_protocol_frozen"
    )
    assert report["decision"]["design_id"] == (
        "evidence_selection_and_answerability_candidate_design_v1"
    )
    assert report["decision"][
        "can_run_train_dev_candidate_comparison_after_user_confirmation"
    ]
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["can_use_test_for_tuning"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert [
        candidate["candidate_id"]
        for candidate in report["frozen_candidate_protocol"]["candidate_policies"]
    ] == [
        "answerability_margin_gate_candidate_v1",
        "evidence_window_reselector_candidate_v1",
        "joint_gate_then_window_candidate_v1",
    ]
    assert report["bottleneck_summary"]["shared_primary_bottleneck_count"] == 2
    assert "private-doc-alpha" not in serialized
    assert "Restart the private database service" not in serialized
    assert "How do I use the private Stage103 fixture" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage103_shared_bottleneck_counts.svg",
        "stage103_shared_bottleneck_rates.svg",
        "stage103_candidate_priority_scores.svg",
        "stage103_candidate_target_case_counts.svg",
        "stage103_candidate_feature_group_counts.svg",
        "stage103_protocol_decision_flags.svg",
        "stage103_guard_check_status.svg",
    }


def test_evidence_answerability_candidate_protocol_blocks_without_confirmation(
    tmp_path,
):
    stage102_path = _write_json(tmp_path / "stage102.json", _stage102_report())

    report = freeze_primeqa_hybrid_evidence_answerability_candidate_protocol(
        stage102_report_path=stage102_path,
        user_confirmed_protocol=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage103_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_evidence_answerability_candidate_protocol_blocked"
    )
    assert (
        report["decision"][
            "can_run_train_dev_candidate_comparison_after_user_confirmation"
        ]
        is False
    )


def test_evidence_answerability_candidate_protocol_blocks_wrong_stage102_route(
    tmp_path,
):
    stage102 = _stage102_report()
    stage102["decision"]["recommended_next_direction"] = "third_wave_retrieval_design"
    stage102_path = _write_json(tmp_path / "stage102.json", stage102)

    report = freeze_primeqa_hybrid_evidence_answerability_candidate_protocol(
        stage102_report_path=stage102_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert (
        checks["stage102_recommends_evidence_answerability_design"]["passed"]
        is False
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_evidence_answerability_candidate_protocol_blocked"
    )


def test_evidence_answerability_candidate_protocol_blocks_missing_shared_bottleneck(
    tmp_path,
):
    stage102 = _stage102_report()
    stage102["aggregate_outputs"]["bucket_counts_by_split"]["dev"][
        "gold_span_beats_selected_answer"
    ] = 0
    stage102["aggregate_outputs"]["bucket_rates_by_split"]["dev"][
        "gold_span_beats_selected_answer"
    ] = 0.0
    stage102_path = _write_json(tmp_path / "stage102.json", stage102)

    report = freeze_primeqa_hybrid_evidence_answerability_candidate_protocol(
        stage102_report_path=stage102_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["shared_gold_span_gap_observed"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_evidence_answerability_candidate_protocol_blocked"
    )


def test_evidence_answerability_candidate_output_contract_is_public_safe(
    tmp_path,
):
    stage102_path = _write_json(tmp_path / "stage102.json", _stage102_report())

    report = freeze_primeqa_hybrid_evidence_answerability_candidate_protocol(
        stage102_report_path=stage102_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )

    fields = set(
        report["frozen_candidate_protocol"]["public_safe_output_contract"][
            "stage104_allowed_case_fields"
        ]
    )
    assert not (
        fields
        & {
            "question_text",
            "raw_answer_text",
            "gold_answer",
            "answer_doc_id",
            "retrieved_doc_ids",
            "cited_doc_ids",
            "source_doc_ids",
            "document_text",
        }
    )


def _stage102_report() -> dict:
    return {
        "stage": "Stage 102",
        "analysis_id": "answer_pipeline_error_decomposition_train_dev_analysis_v1",
        "data_summary": {
            "documents": 10,
            "splits": {
                "train": {
                    "row_count": 100,
                    "answerable_count": 60,
                    "unanswerable_count": 40,
                },
                "dev": {
                    "row_count": 25,
                    "answerable_count": 15,
                    "unanswerable_count": 10,
                },
            },
        },
        "metrics_by_split": {
            "train": {
                "original": {
                    "average_token_f1": 0.19,
                },
                "verified": {
                    "average_token_f1": 0.2,
                    "gold_doc_citation_rate": 0.5,
                    "answerable_refusal_rate": 0.04,
                    "unanswerable_refusal_rate": 0.06,
                    "refused_answerable_questions": 2,
                    "refused_unanswerable_questions": 3,
                },
                "answerable_gold_context_count": 40,
                "answerable_gold_context_rate": 0.67,
            },
            "dev": {
                "original": {
                    "average_token_f1": 0.18,
                },
                "verified": {
                    "average_token_f1": 0.21,
                    "gold_doc_citation_rate": 0.6,
                    "answerable_refusal_rate": 0.08,
                    "unanswerable_refusal_rate": 0.1,
                    "refused_answerable_questions": 1,
                    "refused_unanswerable_questions": 1,
                },
                "answerable_gold_context_count": 10,
                "answerable_gold_context_rate": 0.66,
            },
        },
        "aggregate_outputs": {
            "bucket_counts_by_split": {
                "train": {
                    "answerability_false_answer": 30,
                    "retrieval_context_miss": 20,
                    "evidence_selection_miss": 12,
                    "gold_span_beats_selected_answer": 28,
                },
                "dev": {
                    "answerability_false_answer": 8,
                    "retrieval_context_miss": 4,
                    "evidence_selection_miss": 3,
                    "gold_span_beats_selected_answer": 7,
                },
            },
            "bucket_rates_by_split": {
                "train": {
                    "answerability_false_answer": 0.3,
                    "retrieval_context_miss": 0.2,
                    "evidence_selection_miss": 0.12,
                    "gold_span_beats_selected_answer": 0.28,
                },
                "dev": {
                    "answerability_false_answer": 0.32,
                    "retrieval_context_miss": 0.16,
                    "evidence_selection_miss": 0.12,
                    "gold_span_beats_selected_answer": 0.28,
                },
            },
            "top_priority_buckets": {
                "train": [
                    {
                        "bucket_id": "answerability_false_answer",
                        "case_count": 30,
                        "priority_weight": 1.55,
                        "priority_score": 46.5,
                    },
                    {
                        "bucket_id": "gold_span_beats_selected_answer",
                        "case_count": 28,
                        "priority_weight": 1.45,
                        "priority_score": 40.6,
                    },
                    {
                        "bucket_id": "evidence_selection_miss",
                        "case_count": 12,
                        "priority_weight": 1.7,
                        "priority_score": 20.4,
                    },
                ],
                "dev": [
                    {
                        "bucket_id": "answerability_false_answer",
                        "case_count": 8,
                        "priority_weight": 1.55,
                        "priority_score": 12.4,
                    },
                    {
                        "bucket_id": "gold_span_beats_selected_answer",
                        "case_count": 7,
                        "priority_weight": 1.45,
                        "priority_score": 10.15,
                    },
                    {
                        "bucket_id": "evidence_selection_miss",
                        "case_count": 3,
                        "priority_weight": 1.7,
                        "priority_score": 5.1,
                    },
                ],
            },
        },
        "public_safe_case_samples": {
            "train": {
                "fixture": [
                    {
                        "question_text": "How do I use the private Stage103 fixture?",
                        "answer_doc_id": "private-doc-alpha",
                        "raw_answer_text": "Restart the private database service",
                    }
                ]
            }
        },
        "guard_checks": [
            {"name": "guard_one", "passed": True},
            {"name": "guard_two", "passed": True},
        ],
        "decision": {
            "status": "primeqa_hybrid_answer_pipeline_error_decomposition_completed",
            "analysis_id": "answer_pipeline_error_decomposition_train_dev_analysis_v1",
            "train_top_bucket": "answerability_false_answer",
            "dev_top_bucket": "answerability_false_answer",
            "train_evidence_selection_miss": 12,
            "dev_evidence_selection_miss": 3,
            "train_answerability_false_answer": 30,
            "dev_answerability_false_answer": 8,
            "train_verified_average_token_f1": 0.2,
            "dev_verified_average_token_f1": 0.21,
            "recommended_next_direction": (
                "evidence_selection_and_answerability_candidate_design"
            ),
            "requires_user_confirmation_before_next_protocol": True,
            "can_continue_train_dev_development": True,
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
