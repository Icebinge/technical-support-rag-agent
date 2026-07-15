import json

from ts_rag_agent.application.primeqa_hybrid_structured_query_protocol import (
    freeze_primeqa_hybrid_structured_query_protocol,
    write_primeqa_hybrid_structured_query_protocol_visualizations,
)


def test_structured_query_protocol_freezes_confirmed_candidate(tmp_path):
    stage84_path = tmp_path / "stage84.json"
    stage87_path = tmp_path / "stage87.json"
    stage84_path.write_text(
        json.dumps(_stage84_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stage87_path.write_text(
        json.dumps(_stage87_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = freeze_primeqa_hybrid_structured_query_protocol(
        stage84_report_path=stage84_path,
        stage87_report_path=stage87_path,
        user_confirmed_candidate=True,
        confirmed_candidate_id="structured_query_keyphrase_compaction_design",
        confirmation_note="confirmed in test",
    )
    visualizations = write_primeqa_hybrid_structured_query_protocol_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 88"
    assert report["decision"]["status"] == (
        "primeqa_hybrid_structured_query_protocol_frozen"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["requires_user_confirmation_before_train_dev_run"] is True
    assert report["decision"]["can_use_test_for_tuning"] is False
    assert report["frozen_protocol"]["protocol_id"] == (
        "structured_query_keyphrase_compaction_train_dev_v1"
    )
    assert len(report["frozen_protocol"]["candidate_config_grid"]) == 4
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Restart the database service" not in serialized
    assert "Install the firmware update" not in serialized
    assert "How do I compact a real query" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage88_structured_query_config_token_limits.svg",
        "stage88_structured_query_feature_group_counts.svg",
        "stage88_structured_query_protocol_decision_flags.svg",
        "stage88_structured_query_guard_check_status.svg",
    }


def test_structured_query_protocol_blocks_unconfirmed_candidate(tmp_path):
    stage84_path = tmp_path / "stage84.json"
    stage87_path = tmp_path / "stage87.json"
    stage84_path.write_text(
        json.dumps(_stage84_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stage87_path.write_text(
        json.dumps(_stage87_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = freeze_primeqa_hybrid_structured_query_protocol(
        stage84_report_path=stage84_path,
        stage87_report_path=stage87_path,
        user_confirmed_candidate=False,
        confirmed_candidate_id="structured_query_keyphrase_compaction_design",
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_structured_query_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_structured_query_protocol_blocked"
    )


def test_structured_query_protocol_blocks_candidate_mismatch(tmp_path):
    stage84_path = tmp_path / "stage84.json"
    stage87_path = tmp_path / "stage87.json"
    stage84_path.write_text(
        json.dumps(_stage84_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stage87_path.write_text(
        json.dumps(_stage87_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = freeze_primeqa_hybrid_structured_query_protocol(
        stage84_report_path=stage84_path,
        stage87_report_path=stage87_path,
        user_confirmed_candidate=True,
        confirmed_candidate_id="section_signal_guarded_expansion_design",
        confirmation_note="wrong candidate",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert (
        checks["confirmed_candidate_matches_stage87_next_candidate"]["passed"] is False
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_structured_query_protocol_blocked"
    )


def _stage84_report() -> dict:
    return {
        "stage": "Stage 84",
        "candidate_designs": [
            {
                "candidate_id": "structured_query_keyphrase_compaction_design",
                "status": "recommended_for_train_dev_protocol_design",
                "target_miss_count": 3,
                "target_miss_count_by_split": {"train": 2, "dev": 1},
                "target_metric_contract": [
                    "primary: train-selected dev hit@10 must improve over BM25 baseline",
                    "secondary: top10 regression count must be lower than improvement count",
                    "guard: no query view may be selected by dev-only performance",
                ],
                "runtime_evidence_policy": [
                    "May use runtime question text and deterministic token features.",
                    "Must not use answer document IDs, gold labels, or source DOC_IDS.",
                ],
                "example_raw_question": "How do I compact a real query",
            }
        ],
        "decision": {
            "status": "primeqa_hybrid_second_wave_retrieval_candidate_design_completed",
            "recommended_next_candidate_id": (
                "lexical_cluster_diversity_rerank_design"
            ),
            "requires_user_confirmation_before_train_dev_run": True,
            "can_continue_train_dev_development": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        },
    }


def _stage87_report() -> dict:
    return {
        "stage": "Stage 87",
        "candidate_queue": {
            "next_candidate_summary": {
                "candidate_id": "structured_query_keyphrase_compaction_design",
                "status": "recommended_for_train_dev_protocol_design",
            },
        },
        "decision": {
            "status": "primeqa_hybrid_lexical_cluster_diversity_route_stopped",
            "stopped_candidate_id": "lexical_cluster_diversity_rerank_design",
            "current_route_defaultization": "blocked",
            "next_candidate_id": "structured_query_keyphrase_compaction_design",
            "can_continue_train_dev_development": True,
            "requires_user_confirmation_before_next_protocol": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        },
        "raw_question_text": "Restart the database service",
        "raw_answer_text": "Install the firmware update",
    }
