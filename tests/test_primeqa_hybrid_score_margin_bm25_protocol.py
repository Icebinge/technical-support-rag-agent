import json

from ts_rag_agent.application.primeqa_hybrid_score_margin_bm25_protocol import (
    freeze_primeqa_hybrid_score_margin_bm25_protocol,
    write_primeqa_hybrid_score_margin_bm25_protocol_visualizations,
)


def test_score_margin_bm25_protocol_freezes_confirmed_candidate(tmp_path):
    stage84_path = tmp_path / "stage84.json"
    stage93_path = tmp_path / "stage93.json"
    stage84_path.write_text(
        json.dumps(_stage84_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stage93_path.write_text(
        json.dumps(_stage93_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = freeze_primeqa_hybrid_score_margin_bm25_protocol(
        stage84_report_path=stage84_path,
        stage93_report_path=stage93_path,
        user_confirmed_candidate=True,
        confirmed_candidate_id="score_margin_bm25_normalization_gate_design",
        confirmation_note="confirmed in test",
    )
    visualizations = write_primeqa_hybrid_score_margin_bm25_protocol_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 94"
    assert report["decision"]["status"] == (
        "primeqa_hybrid_score_margin_bm25_protocol_frozen"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["requires_user_confirmation_before_train_dev_run"] is True
    assert report["decision"]["can_use_test_for_tuning"] is False
    assert report["frozen_protocol"]["protocol_id"] == (
        "score_margin_bm25_normalization_gate_train_dev_v1"
    )
    assert len(report["frozen_protocol"]["candidate_config_grid"]) == 4
    assert all(check["passed"] for check in report["guard_checks"])
    assert (
        report["frozen_protocol"]["historical_signal_policy"][
            "dev_only_b095_observation_can_select_runtime_rule"
        ]
        is False
    )
    assert "Restart the database service" not in serialized
    assert "Install the firmware update" not in serialized
    assert "How do I use this private score-margin question" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage94_score_margin_bm25_config_b_values.svg",
        "stage94_score_margin_bm25_margin_thresholds.svg",
        "stage94_score_margin_bm25_length_thresholds.svg",
        "stage94_score_margin_bm25_feature_group_counts.svg",
        "stage94_score_margin_bm25_protocol_decision_flags.svg",
        "stage94_score_margin_bm25_guard_check_status.svg",
    }


def test_score_margin_bm25_protocol_blocks_unconfirmed_candidate(tmp_path):
    stage84_path = tmp_path / "stage84.json"
    stage93_path = tmp_path / "stage93.json"
    stage84_path.write_text(
        json.dumps(_stage84_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stage93_path.write_text(
        json.dumps(_stage93_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = freeze_primeqa_hybrid_score_margin_bm25_protocol(
        stage84_report_path=stage84_path,
        stage93_report_path=stage93_path,
        user_confirmed_candidate=False,
        confirmed_candidate_id="score_margin_bm25_normalization_gate_design",
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_score_margin_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_score_margin_bm25_protocol_blocked"
    )


def test_score_margin_bm25_protocol_blocks_candidate_mismatch(tmp_path):
    stage84_path = tmp_path / "stage84.json"
    stage93_path = tmp_path / "stage93.json"
    stage84_path.write_text(
        json.dumps(_stage84_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stage93_path.write_text(
        json.dumps(_stage93_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = freeze_primeqa_hybrid_score_margin_bm25_protocol(
        stage84_report_path=stage84_path,
        stage93_report_path=stage93_path,
        user_confirmed_candidate=True,
        confirmed_candidate_id="selective_dense_sparse_low_overlap_gate_design",
        confirmation_note="wrong candidate",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert (
        checks["confirmed_candidate_matches_stage93_next_candidate"]["passed"] is False
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_score_margin_bm25_protocol_blocked"
    )


def _stage84_report() -> dict:
    return {
        "stage": "Stage 84",
        "candidate_designs": [
            {
                "candidate_id": "score_margin_bm25_normalization_gate_design",
                "name": "Score-margin BM25 normalization gate design",
                "category": "lexical_parameter_gate",
                "status": "recommended_for_train_dev_protocol_design",
                "risk_level": "medium",
                "implementation_readiness": 0.7,
                "prior_signal_key": "bm25_k1_b_grid",
                "prior_signal_score": 0.48,
                "priority_score": 171,
                "target_miss_count": 111,
                "target_miss_count_by_split": {"dev": 18, "train": 93},
                "target_rank_buckets": {
                    "not_found_top50": 73,
                    "rank_11_to_20": 14,
                    "rank_21_to_50": 24,
                },
                "rationale": (
                    "Stage82 showed b=0.95 configs with better dev hit@10, "
                    "but those configs were not train-selected."
                ),
                "stage85_protocol_outline": [
                    "Define score-margin and document-length proxy features on train.",
                    "Select any adaptive BM25 normalization rule on train only.",
                    "Validate the selected rule on dev without using dev to choose b.",
                ],
                "target_metric_contract": [
                    "primary: train-selected rule must improve dev hit@10",
                    "secondary: rank 11-50 near misses should decrease",
                    "guard: dev-only b=0.95 observations cannot select a runtime rule",
                ],
                "runtime_evidence_policy": [
                    "May use BM25 scores, document length, and candidate rank features.",
                    "Must not use source DOC_IDS, answer document IDs, or dev-only selection.",
                ],
                "example_raw_question": (
                    "How do I use this private score-margin question"
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


def _stage93_report() -> dict:
    return {
        "stage": "Stage 93",
        "candidate_queue": {
            "next_candidate_summary": {
                "candidate_id": "score_margin_bm25_normalization_gate_design",
                "status": "recommended_for_train_dev_protocol_design",
            },
        },
        "decision": {
            "status": "primeqa_hybrid_section_signal_route_stopped",
            "stopped_candidate_id": "section_signal_guarded_expansion_design",
            "current_route_defaultization": "blocked",
            "next_candidate_id": "score_margin_bm25_normalization_gate_design",
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
