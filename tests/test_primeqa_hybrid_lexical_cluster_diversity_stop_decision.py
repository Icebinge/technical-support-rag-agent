import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_lexical_cluster_diversity_stop_decision import (
    decide_primeqa_hybrid_lexical_cluster_diversity_stop,
    write_primeqa_hybrid_lexical_cluster_diversity_stop_visualizations,
)


def test_lcdr_stop_decision_stops_route_and_selects_next_candidate(tmp_path):
    paths = _write_fixture(tmp_path)

    report = decide_primeqa_hybrid_lexical_cluster_diversity_stop(
        stage84_report_path=paths["stage84_report"],
        stage86_report_path=paths["stage86_report"],
        user_confirmed_stop=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = write_primeqa_hybrid_lexical_cluster_diversity_stop_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 87"
    assert report["decision"]["status"] == (
        "primeqa_hybrid_lexical_cluster_diversity_route_stopped"
    )
    assert report["decision"]["stopped_candidate_id"] == (
        "lexical_cluster_diversity_rerank_design"
    )
    assert report["decision"]["next_candidate_id"] == (
        "structured_query_keyphrase_compaction_design"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Restart the database service" not in serialized
    assert "Install the firmware update" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage87_lcdr_train_dev_hit10_delta.svg",
        "stage87_lcdr_dev_change_counts.svg",
        "stage87_second_wave_remaining_candidate_priority.svg",
        "stage87_lcdr_stop_decision_flags.svg",
        "stage87_lcdr_stop_guard_check_status.svg",
    }


def test_lcdr_stop_decision_blocks_without_user_confirmation(tmp_path):
    paths = _write_fixture(tmp_path)

    report = decide_primeqa_hybrid_lexical_cluster_diversity_stop(
        stage84_report_path=paths["stage84_report"],
        stage86_report_path=paths["stage86_report"],
        user_confirmed_stop=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage87_stop_decision"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_lexical_cluster_diversity_stop_decision_blocked"
    )


def test_lcdr_stop_decision_blocks_if_dev_hit10_improved(tmp_path):
    paths = _write_fixture(tmp_path, dev_hit10_delta=0.02)

    report = decide_primeqa_hybrid_lexical_cluster_diversity_stop(
        stage84_report_path=paths["stage84_report"],
        stage86_report_path=paths["stage86_report"],
        user_confirmed_stop=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage86_train_selected_config_has_no_dev_hit10_gain"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_lexical_cluster_diversity_stop_decision_blocked"
    )


def _write_fixture(
    tmp_path: Path,
    *,
    dev_hit10_delta: float = 0.0,
) -> dict[str, Path]:
    stage84 = {
        "stage": "Stage 84",
        "recommended_execution_order": [
            "lexical_cluster_diversity_rerank_design",
            "structured_query_keyphrase_compaction_design",
            "section_signal_guarded_expansion_design",
        ],
        "candidate_designs": [
            _candidate(
                candidate_id="lexical_cluster_diversity_rerank_design",
                priority_score=210,
                metric_contract=[
                    "primary: dev hit@10 must improve over BM25 baseline",
                    "guard: no title/body text should be written to reports",
                ],
            ),
            _candidate(
                candidate_id="structured_query_keyphrase_compaction_design",
                priority_score=207,
                metric_contract=[
                    "primary: train-selected dev hit@10 must improve over BM25 baseline"
                ],
            ),
            _candidate(
                candidate_id="section_signal_guarded_expansion_design",
                priority_score=174,
                metric_contract=[
                    "primary: train-selected dev hit@10 must improve over BM25 baseline"
                ],
            ),
            {
                "candidate_id": "source_doc_ids_oracle_union_blocked",
                "name": "Source DOC_IDS oracle union",
                "category": "blocked_diagnostic",
                "status": "blocked_from_train_dev_experiment",
                "risk_level": "blocked",
                "implementation_readiness": 0.0,
                "priority_score": 0,
            },
        ],
    }
    stage86 = {
        "stage": "Stage 86",
        "config": {
            "candidate_id": "lexical_cluster_diversity_rerank_design",
            "protocol_id": "lexical_cluster_diversity_rerank_train_dev_v1",
        },
        "train_selection": {
            "selected_train_comparison_to_baseline": {
                "hit@10_delta": 0.0054,
                "top10_improvement_count": 4,
                "top10_regression_count": 2,
            },
            "selected_dev_comparison_to_baseline": {
                "hit@10_delta": dev_hit10_delta,
                "top10_improvement_count": 0,
                "top10_regression_count": 0,
                "rank_up_within_top10_count": 0,
                "rank_down_within_top10_count": 0,
                "not_found_count_at_search_depth_delta": 0,
                "rank_11_to_50_count_delta": 0,
            },
        },
        "decision": {
            "status": "primeqa_hybrid_lexical_cluster_diversity_comparison_completed",
            "selected_config_id": "lcdr_penalty_0_06_title_query_cluster",
            "can_continue_train_dev_development": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        },
        "private_example_strings": [
            "Restart the database service",
            "Install the firmware update",
        ],
    }
    return {
        "stage84_report": _write_json(tmp_path / "stage84.json", stage84),
        "stage86_report": _write_json(tmp_path / "stage86.json", stage86),
    }


def _candidate(
    *,
    candidate_id: str,
    priority_score: int,
    metric_contract: list[str],
) -> dict:
    return {
        "candidate_id": candidate_id,
        "name": candidate_id.replace("_", " "),
        "category": "candidate",
        "status": "recommended_for_train_dev_protocol_design",
        "risk_level": "medium",
        "implementation_readiness": 0.7,
        "priority_score": priority_score,
        "target_miss_count": 10,
        "target_miss_count_by_split": {"dev": 2, "train": 8},
        "target_metric_contract": metric_contract,
        "runtime_evidence_policy": [
            "Must not use source DOC_IDS, answer document IDs, or gold labels."
        ],
    }


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
