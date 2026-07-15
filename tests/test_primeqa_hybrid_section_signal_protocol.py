import json

from ts_rag_agent.application.primeqa_hybrid_section_signal_protocol import (
    freeze_primeqa_hybrid_section_signal_protocol,
    write_primeqa_hybrid_section_signal_protocol_visualizations,
)


def test_section_signal_protocol_freezes_confirmed_candidate(tmp_path):
    stage84_path = tmp_path / "stage84.json"
    stage90_path = tmp_path / "stage90.json"
    stage84_path.write_text(
        json.dumps(_stage84_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stage90_path.write_text(
        json.dumps(_stage90_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = freeze_primeqa_hybrid_section_signal_protocol(
        stage84_report_path=stage84_path,
        stage90_report_path=stage90_path,
        user_confirmed_candidate=True,
        confirmed_candidate_id="section_signal_guarded_expansion_design",
        confirmation_note="confirmed in test",
    )
    visualizations = write_primeqa_hybrid_section_signal_protocol_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 91"
    assert report["decision"]["status"] == (
        "primeqa_hybrid_section_signal_protocol_frozen"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["requires_user_confirmation_before_train_dev_run"] is True
    assert report["decision"]["can_use_test_for_tuning"] is False
    assert report["frozen_protocol"]["protocol_id"] == (
        "section_signal_guarded_expansion_train_dev_v1"
    )
    assert len(report["frozen_protocol"]["candidate_config_grid"]) == 4
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Restart the database service" not in serialized
    assert "Install the firmware update" not in serialized
    assert "How do I use this private section signal question" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage91_section_signal_config_promotion_budgets.svg",
        "stage91_section_signal_config_ratio_thresholds.svg",
        "stage91_section_signal_feature_group_counts.svg",
        "stage91_section_signal_protocol_decision_flags.svg",
        "stage91_section_signal_guard_check_status.svg",
    }


def test_section_signal_protocol_blocks_unconfirmed_candidate(tmp_path):
    stage84_path = tmp_path / "stage84.json"
    stage90_path = tmp_path / "stage90.json"
    stage84_path.write_text(
        json.dumps(_stage84_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stage90_path.write_text(
        json.dumps(_stage90_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = freeze_primeqa_hybrid_section_signal_protocol(
        stage84_report_path=stage84_path,
        stage90_report_path=stage90_path,
        user_confirmed_candidate=False,
        confirmed_candidate_id="section_signal_guarded_expansion_design",
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_section_signal_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_section_signal_protocol_blocked"
    )


def test_section_signal_protocol_blocks_candidate_mismatch(tmp_path):
    stage84_path = tmp_path / "stage84.json"
    stage90_path = tmp_path / "stage90.json"
    stage84_path.write_text(
        json.dumps(_stage84_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stage90_path.write_text(
        json.dumps(_stage90_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = freeze_primeqa_hybrid_section_signal_protocol(
        stage84_report_path=stage84_path,
        stage90_report_path=stage90_path,
        user_confirmed_candidate=True,
        confirmed_candidate_id="score_margin_bm25_normalization_gate_design",
        confirmation_note="wrong candidate",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert (
        checks["confirmed_candidate_matches_stage90_next_candidate"]["passed"] is False
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_section_signal_protocol_blocked"
    )


def _stage84_report() -> dict:
    return {
        "stage": "Stage 84",
        "candidate_designs": [
            {
                "candidate_id": "section_signal_guarded_expansion_design",
                "name": "Section signal guarded expansion design",
                "category": "section_signal_gate",
                "status": "recommended_for_train_dev_protocol_design",
                "risk_level": "medium",
                "implementation_readiness": 0.64,
                "prior_signal_key": "section_bm25",
                "prior_signal_score": 0.45,
                "priority_score": 174,
                "target_miss_count": 119,
                "target_miss_count_by_split": {"dev": 17, "train": 102},
                "target_rank_buckets": {
                    "not_found_top50": 110,
                    "rank_11_to_20": 1,
                    "rank_21_to_50": 8,
                },
                "target_metric_contract": [
                    "primary: dev hit@10 must improve over BM25 baseline",
                    "secondary: search-depth improvements must exceed regressions",
                    "guard: section signal must not demote existing BM25 top10 hits by default",
                ],
                "runtime_evidence_policy": [
                    "May use runtime section BM25 scores and document BM25 scores.",
                    "Must not use gold answer rank, source DOC_IDS, or test labels.",
                ],
                "example_raw_question": (
                    "How do I use this private section signal question"
                ),
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


def _stage90_report() -> dict:
    return {
        "stage": "Stage 90",
        "candidate_queue": {
            "next_candidate_summary": {
                "candidate_id": "section_signal_guarded_expansion_design",
                "status": "recommended_for_train_dev_protocol_design",
            },
        },
        "decision": {
            "status": "primeqa_hybrid_structured_query_route_stopped",
            "stopped_candidate_id": "structured_query_keyphrase_compaction_design",
            "current_route_defaultization": "blocked",
            "next_candidate_id": "section_signal_guarded_expansion_design",
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
