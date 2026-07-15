import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_selective_dense_sparse_stop_decision import (
    decide_primeqa_hybrid_selective_dense_sparse_stop,
    write_primeqa_hybrid_selective_dense_sparse_stop_visualizations,
)


def test_selective_dense_sparse_stop_decision_stops_exhausted_route_family(tmp_path):
    paths = _write_fixture(tmp_path)

    report = decide_primeqa_hybrid_selective_dense_sparse_stop(
        stage84_report_path=paths["stage84_report"],
        stage97_report_path=paths["stage97_report"],
        stage98_report_path=paths["stage98_report"],
        user_confirmed_stop=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = write_primeqa_hybrid_selective_dense_sparse_stop_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 99"
    assert report["decision"]["status"] == (
        "primeqa_hybrid_selective_dense_sparse_route_stopped"
    )
    assert report["decision"]["stopped_candidate_id"] == (
        "selective_dense_sparse_low_overlap_gate_design"
    )
    assert report["decision"]["remaining_actionable_candidate_count"] == 0
    assert report["decision"]["route_family_exhausted"] is True
    assert report["decision"]["can_continue_train_dev_development"] is False
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Restart the database service" not in serialized
    assert "Install the firmware update" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage99_selective_dense_sparse_train_dev_hit10_delta.svg",
        "stage99_selective_dense_sparse_dev_contract_deltas.svg",
        "stage99_selective_dense_sparse_gate_actions.svg",
        "stage99_second_wave_route_status.svg",
        "stage99_selective_dense_sparse_stop_decision_flags.svg",
        "stage99_selective_dense_sparse_stop_guard_check_status.svg",
    }


def test_selective_dense_sparse_stop_decision_blocks_without_confirmation(tmp_path):
    paths = _write_fixture(tmp_path)

    report = decide_primeqa_hybrid_selective_dense_sparse_stop(
        stage84_report_path=paths["stage84_report"],
        stage97_report_path=paths["stage97_report"],
        stage98_report_path=paths["stage98_report"],
        user_confirmed_stop=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage99_stop_decision"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_selective_dense_sparse_stop_decision_blocked"
    )


def test_selective_dense_sparse_stop_decision_blocks_if_primary_contract_passed(
    tmp_path,
):
    paths = _write_fixture(
        tmp_path,
        dev_hit10_delta=0.02,
        primary_contract_passed=True,
    )

    report = decide_primeqa_hybrid_selective_dense_sparse_stop(
        stage84_report_path=paths["stage84_report"],
        stage97_report_path=paths["stage97_report"],
        stage98_report_path=paths["stage98_report"],
        user_confirmed_stop=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage98_primary_contract_failed"]["passed"] is False
    assert (
        checks["stage98_train_selected_policy_has_no_dev_hit10_gain"]["passed"]
        is False
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_selective_dense_sparse_stop_decision_blocked"
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
                    "secondary: dev not-found@50 should decrease without hit@1 collapse",
                    "guard: no downloads and no dev-selected gate thresholds",
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
                "target_metric_contract": [
                    "blocked: not eligible for train/dev tuning",
                    "blocked: not eligible for runtime defaultization",
                ],
            },
        ],
    }
    stage97 = {
        "stage": "Stage 97",
        "decision": {
            "status": "primeqa_hybrid_selective_dense_sparse_protocol_frozen",
            "protocol_id": "selective_dense_sparse_low_overlap_gate_train_dev_v1",
            "candidate_id": "selective_dense_sparse_low_overlap_gate_design",
            "requires_user_confirmation_before_train_dev_run": True,
            "can_run_train_dev_metrics_after_user_confirmation": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        },
        "frozen_protocol": {
            "protocol_id": "selective_dense_sparse_low_overlap_gate_train_dev_v1",
            "candidate_id": "selective_dense_sparse_low_overlap_gate_design",
            "dense_cache_contract": {
                "allowed_dense_configs": [{"config_id": "dense_a"}],
                "download_required": False,
                "document_reencoding_allowed": False,
            },
            "candidate_policy_grid": [{"policy_id": "policy_a"}],
            "train_selection_rule": {
                "selection_split": "train",
                "validation_split": "dev",
                "dev_selection_forbidden": True,
                "test_selection_forbidden": True,
            },
            "target_metric_contract": [
                "primary: train-selected gated policy must improve dev hit@10",
                "secondary: dev not-found@50 should decrease without hit@1 collapse",
                "guard: no downloads and no dev-selected gate thresholds",
            ],
        },
    }
    selected_policy_id = "sdsl_minilm_low_overlap_conservative_v1"
    stage98 = {
        "stage": "Stage 98",
        "loaded_data_summary": {
            "test_split_loaded": False,
            "final_metrics_run": False,
        },
        "train_selection": {
            "selected_policy_id": selected_policy_id,
            "selected_train_comparison_to_baseline": {
                "hit@10_delta": 0.0,
                "not_found_count_at_search_depth_delta": 0,
                "gate_activation_count": 2,
                "promotion_count": 2,
            },
        },
        "comparisons_to_baseline": {
            "dev": {
                selected_policy_id: {
                    "hit@1_delta": 0.0,
                    "hit@10_delta": dev_hit10_delta,
                    "top10_improvement_count": 0,
                    "top10_regression_count": 0,
                    "not_found_count_at_search_depth_delta": 0,
                    "gate_activation_count": 0,
                    "promotion_count": 0,
                }
            }
        },
        "artifact_safety": {
            "raw_question_text_written": False,
            "raw_answer_text_written": False,
            "raw_document_text_written": False,
            "raw_document_title_written": False,
            "query_terms_written": False,
            "matched_token_strings_written": False,
            "source_doc_ids_used_as_runtime_evidence": False,
            "answer_doc_ids_used_as_runtime_features": False,
        },
        "decision": {
            "status": "primeqa_hybrid_selective_dense_sparse_comparison_completed",
            "protocol_id": "selective_dense_sparse_low_overlap_gate_train_dev_v1",
            "selected_policy_id": selected_policy_id,
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
        "stage97_report": _write_json(tmp_path / "stage97.json", stage97),
        "stage98_report": _write_json(tmp_path / "stage98.json", stage98),
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
