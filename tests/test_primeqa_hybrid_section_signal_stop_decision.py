import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_section_signal_stop_decision import (
    decide_primeqa_hybrid_section_signal_stop,
    write_primeqa_hybrid_section_signal_stop_visualizations,
)


def test_section_signal_stop_decision_stops_route_and_selects_next_candidate(tmp_path):
    paths = _write_fixture(tmp_path)

    report = decide_primeqa_hybrid_section_signal_stop(
        stage84_report_path=paths["stage84_report"],
        stage92_report_path=paths["stage92_report"],
        user_confirmed_stop=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = write_primeqa_hybrid_section_signal_stop_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 93"
    assert report["decision"]["status"] == (
        "primeqa_hybrid_section_signal_route_stopped"
    )
    assert report["decision"]["stopped_candidate_id"] == (
        "section_signal_guarded_expansion_design"
    )
    assert report["decision"]["next_candidate_id"] == (
        "score_margin_bm25_normalization_gate_design"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Restart the database service" not in serialized
    assert "Install the firmware update" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage93_section_signal_train_dev_hit10_delta.svg",
        "stage93_section_signal_dev_change_counts.svg",
        "stage93_second_wave_remaining_candidate_priority.svg",
        "stage93_section_signal_stop_decision_flags.svg",
        "stage93_section_signal_stop_guard_check_status.svg",
    }


def test_section_signal_stop_decision_blocks_without_user_confirmation(tmp_path):
    paths = _write_fixture(tmp_path)

    report = decide_primeqa_hybrid_section_signal_stop(
        stage84_report_path=paths["stage84_report"],
        stage92_report_path=paths["stage92_report"],
        user_confirmed_stop=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage93_stop_decision"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_section_signal_stop_decision_blocked"
    )


def test_section_signal_stop_decision_blocks_if_primary_contract_passed(tmp_path):
    paths = _write_fixture(
        tmp_path,
        dev_hit10_delta=0.02,
        primary_contract_passed=True,
    )

    report = decide_primeqa_hybrid_section_signal_stop(
        stage84_report_path=paths["stage84_report"],
        stage92_report_path=paths["stage92_report"],
        user_confirmed_stop=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage92_primary_contract_failed"]["passed"] is False
    assert (
        checks["stage92_train_selected_config_has_no_dev_hit10_gain"]["passed"] is False
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_section_signal_stop_decision_blocked"
    )


def _write_fixture(
    tmp_path: Path,
    *,
    dev_hit10_delta: float = 0.0,
    primary_contract_passed: bool = False,
) -> dict[str, Path]:
    stage84 = {
        "stage": "Stage 84",
        "recommended_execution_order": [
            "lexical_cluster_diversity_rerank_design",
            "structured_query_keyphrase_compaction_design",
            "section_signal_guarded_expansion_design",
            "score_margin_bm25_normalization_gate_design",
            "selective_dense_sparse_low_overlap_gate_design",
        ],
        "candidate_designs": [
            _candidate(
                candidate_id="lexical_cluster_diversity_rerank_design",
                priority_score=210,
                metric_contract=[
                    "primary: dev hit@10 must improve over BM25 baseline",
                ],
            ),
            _candidate(
                candidate_id="structured_query_keyphrase_compaction_design",
                priority_score=207,
                metric_contract=[
                    "primary: train-selected dev hit@10 must improve over BM25 baseline",
                ],
            ),
            _candidate(
                candidate_id="section_signal_guarded_expansion_design",
                priority_score=174,
                metric_contract=[
                    "primary: dev hit@10 must improve over BM25 baseline",
                    "secondary: search-depth improvements must exceed regressions",
                    "guard: section signal must not demote existing BM25 top10 hits by default",
                ],
            ),
            _candidate(
                candidate_id="score_margin_bm25_normalization_gate_design",
                priority_score=171,
                metric_contract=[
                    "primary: train-selected rule must improve dev hit@10",
                ],
            ),
            _candidate(
                candidate_id="selective_dense_sparse_low_overlap_gate_design",
                priority_score=159,
                metric_contract=[
                    "primary: train-selected gated policy must improve dev hit@10",
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
    stage92 = {
        "stage": "Stage 92",
        "config": {
            "candidate_id": "section_signal_guarded_expansion_design",
            "protocol_id": "section_signal_guarded_expansion_train_dev_v1",
        },
        "train_selection": {
            "selected_train_comparison_to_baseline": {
                "hit@10_delta": 0.0,
                "search_depth_net_improvement_count": 1,
                "top10_improvement_count": 1,
                "top10_regression_count": 1,
                "section_signal_promotion_count": 92,
            },
            "selected_dev_comparison_to_baseline": {
                "hit@10_delta": dev_hit10_delta,
                "search_depth_net_improvement_count": 0,
                "top10_improvement_count": 0,
                "top10_regression_count": 0,
                "not_found_count_at_search_depth_delta": 0,
                "rank_11_to_50_count_delta": 0,
                "section_signal_promotion_count": 22,
                "protected_top10_demotion_count": 22,
            },
        },
        "decision": {
            "status": "primeqa_hybrid_section_signal_comparison_completed",
            "selected_config_id": "ssgx_section_top50_injection_guard_v1",
            "primary_contract_passed": primary_contract_passed,
            "secondary_contract_passed": False,
            "guard_contract_passed": True,
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
        "stage92_report": _write_json(tmp_path / "stage92.json", stage92),
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
