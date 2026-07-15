import json

from ts_rag_agent.application.primeqa_hybrid_lexical_cluster_diversity_protocol import (
    freeze_primeqa_hybrid_lexical_cluster_diversity_protocol,
    write_primeqa_hybrid_lexical_cluster_diversity_protocol_visualizations,
)


def test_lexical_cluster_diversity_protocol_freezes_confirmed_candidate(tmp_path):
    stage84_path = tmp_path / "stage84.json"
    stage84_path.write_text(
        json.dumps(_stage84_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = freeze_primeqa_hybrid_lexical_cluster_diversity_protocol(
        stage84_report_path=stage84_path,
        user_confirmed_candidate=True,
        confirmed_candidate_id="lexical_cluster_diversity_rerank_design",
        confirmation_note="confirmed in test",
    )
    visualizations = (
        write_primeqa_hybrid_lexical_cluster_diversity_protocol_visualizations(
            report=report,
            output_dir=tmp_path / "visuals",
        )
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 85"
    assert report["decision"]["status"] == (
        "primeqa_hybrid_lexical_cluster_diversity_protocol_frozen"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["requires_user_confirmation_before_train_dev_run"] is True
    assert report["frozen_protocol"]["protocol_id"] == (
        "lexical_cluster_diversity_rerank_train_dev_v1"
    )
    assert len(report["frozen_protocol"]["candidate_config_grid"]) == 4
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Restart the database service" not in serialized
    assert "Install the firmware update" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage85_lcdr_candidate_config_penalties.svg",
        "stage85_lcdr_feature_group_counts.svg",
        "stage85_lcdr_protocol_decision_flags.svg",
        "stage85_lcdr_guard_check_status.svg",
    }


def test_lexical_cluster_diversity_protocol_blocks_unconfirmed_candidate(tmp_path):
    stage84_path = tmp_path / "stage84.json"
    stage84_path.write_text(
        json.dumps(_stage84_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = freeze_primeqa_hybrid_lexical_cluster_diversity_protocol(
        stage84_report_path=stage84_path,
        user_confirmed_candidate=False,
        confirmed_candidate_id="lexical_cluster_diversity_rerank_design",
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_recommended_candidate"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_lexical_cluster_diversity_protocol_blocked"
    )


def test_lexical_cluster_diversity_protocol_blocks_candidate_mismatch(tmp_path):
    stage84_path = tmp_path / "stage84.json"
    stage84_path.write_text(
        json.dumps(_stage84_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = freeze_primeqa_hybrid_lexical_cluster_diversity_protocol(
        stage84_report_path=stage84_path,
        user_confirmed_candidate=True,
        confirmed_candidate_id="structured_query_keyphrase_compaction_design",
        confirmation_note="wrong candidate",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["confirmed_candidate_matches_stage84_recommendation"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_lexical_cluster_diversity_protocol_blocked"
    )


def _stage84_report() -> dict:
    return {
        "stage": "Stage 84",
        "candidate_designs": [
            {
                "candidate_id": "lexical_cluster_diversity_rerank_design",
                "status": "recommended_for_train_dev_protocol_design",
                "target_miss_count": 3,
                "target_miss_count_by_split": {"train": 2, "dev": 1},
                "runtime_evidence_policy": [
                    "May use runtime candidate scores.",
                    "Must not use source DOC_IDS.",
                ],
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
