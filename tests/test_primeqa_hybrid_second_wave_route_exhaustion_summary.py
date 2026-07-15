import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_second_wave_route_exhaustion_summary import (
    summarize_primeqa_hybrid_second_wave_route_exhaustion,
    write_primeqa_hybrid_second_wave_route_exhaustion_visualizations,
)


def test_second_wave_route_exhaustion_recommends_answer_pipeline_direction(tmp_path):
    paths = _write_fixture_reports(tmp_path)

    report = summarize_primeqa_hybrid_second_wave_route_exhaustion(
        **paths,
        user_confirmed_summary=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = write_primeqa_hybrid_second_wave_route_exhaustion_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 100"
    assert report["aggregate_summary"]["second_wave_all_expected_candidates_stopped"]
    assert report["aggregate_summary"]["runtime_advancing_second_wave_candidate_count"] == 0
    assert report["decision"]["recommended_next_direction"] == (
        "answer_pipeline_error_decomposition"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Restart the database service" not in serialized
    assert "Install the firmware update" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage100_second_wave_dev_hit10_deltas.svg",
        "stage100_second_wave_top10_net_changes.svg",
        "stage100_second_wave_route_outcomes.svg",
        "stage100_next_direction_readiness.svg",
        "stage100_route_exhaustion_decision_flags.svg",
        "stage100_route_exhaustion_guard_check_status.svg",
    }


def test_second_wave_route_exhaustion_blocks_without_confirmation(tmp_path):
    paths = _write_fixture_reports(tmp_path)

    report = summarize_primeqa_hybrid_second_wave_route_exhaustion(
        **paths,
        user_confirmed_summary=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage100_summary"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_second_wave_route_exhaustion_summary_blocked"
    )


def test_second_wave_route_exhaustion_blocks_if_candidate_missing(tmp_path):
    paths = _write_fixture_reports(tmp_path, omit_stage99=True)

    report = summarize_primeqa_hybrid_second_wave_route_exhaustion(
        **paths,
        user_confirmed_summary=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["all_second_wave_candidates_have_stop_reports"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_second_wave_route_exhaustion_summary_blocked"
    )


def _write_fixture_reports(tmp_path: Path, *, omit_stage99: bool = False) -> dict[str, Path]:
    stage83 = {
        "stage": "Stage 83",
        "decision": {
            "status": "primeqa_hybrid_retrieval_recall_exhaustion_summary_completed",
            "stage76_allowed_candidates_exhausted": True,
            "can_continue_train_dev_development": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        },
    }
    order = [
        "lexical_cluster_diversity_rerank_design",
        "structured_query_keyphrase_compaction_design",
        "section_signal_guarded_expansion_design",
        "score_margin_bm25_normalization_gate_design",
        "selective_dense_sparse_low_overlap_gate_design",
    ]
    stage84 = {
        "stage": "Stage 84",
        "decision": {
            "status": "primeqa_hybrid_second_wave_retrieval_candidate_design_completed",
            "recommended_execution_order": order,
            "can_continue_train_dev_development": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        },
        "candidate_designs": [
            _candidate(candidate_id=candidate_id, priority_score=200 - index)
            for index, candidate_id in enumerate(order)
        ]
        + [
            {
                "candidate_id": "source_doc_ids_oracle_union_blocked",
                "name": "Source DOC_IDS oracle union",
                "status": "blocked_from_train_dev_experiment",
                "target_miss_count": 148,
                "runtime_evidence_policy": [
                    "Forbidden: source DOC_IDS are not runtime retrieval evidence."
                ],
            }
        ],
    }
    reports = {
        "stage83_report_path": _write_json(tmp_path / "stage83.json", stage83),
        "stage84_report_path": _write_json(tmp_path / "stage84.json", stage84),
        "stage87_report_path": _write_json(
            tmp_path / "stage87.json",
            _stop_report(
                stage="Stage 87",
                status="primeqa_hybrid_lexical_cluster_diversity_route_stopped",
                candidate_id="lexical_cluster_diversity_rerank_design",
                protocol_id="lexical_cluster_diversity_rerank_train_dev_v1",
                summary_key="stage86_summary",
                selected_key="selected_config_id",
                selected_id="lcdr_fixture",
                dev_hit10_delta=0.0,
            ),
        ),
        "stage90_report_path": _write_json(
            tmp_path / "stage90.json",
            _stop_report(
                stage="Stage 90",
                status="primeqa_hybrid_structured_query_route_stopped",
                candidate_id="structured_query_keyphrase_compaction_design",
                protocol_id="structured_query_keyphrase_compaction_train_dev_v1",
                summary_key="stage89_summary",
                selected_key="selected_config_id",
                selected_id="sqkc_fixture",
                dev_hit10_delta=-0.0527,
                dev_top10_improvement_count=1,
                dev_top10_regression_count=5,
            ),
        ),
        "stage93_report_path": _write_json(
            tmp_path / "stage93.json",
            _stop_report(
                stage="Stage 93",
                status="primeqa_hybrid_section_signal_route_stopped",
                candidate_id="section_signal_guarded_expansion_design",
                protocol_id="section_signal_guarded_expansion_train_dev_v1",
                summary_key="stage92_summary",
                selected_key="selected_config_id",
                selected_id="ssgx_fixture",
                dev_hit10_delta=0.0,
            ),
        ),
        "stage96_report_path": _write_json(
            tmp_path / "stage96.json",
            _stop_report(
                stage="Stage 96",
                status="primeqa_hybrid_score_margin_bm25_route_stopped",
                candidate_id="score_margin_bm25_normalization_gate_design",
                protocol_id="score_margin_bm25_normalization_gate_train_dev_v1",
                summary_key="stage95_summary",
                selected_key="selected_config_id",
                selected_id="smbn_fixture",
                dev_hit10_delta=0.0,
            ),
        ),
        "stage99_report_path": _write_json(
            tmp_path / "stage99.json",
            _stop_report(
                stage="Stage 99",
                status="primeqa_hybrid_selective_dense_sparse_route_stopped",
                candidate_id=(
                    "missing_candidate"
                    if omit_stage99
                    else "selective_dense_sparse_low_overlap_gate_design"
                ),
                protocol_id="selective_dense_sparse_low_overlap_gate_train_dev_v1",
                summary_key="stage98_summary",
                selected_key="selected_policy_id",
                selected_id="sdsl_fixture",
                dev_hit10_delta=0.0,
                extra_decision={
                    "remaining_actionable_candidate_count": 0,
                    "route_family_exhausted": True,
                },
            ),
        ),
    }
    return reports


def _stop_report(
    *,
    stage: str,
    status: str,
    candidate_id: str,
    protocol_id: str,
    summary_key: str,
    selected_key: str,
    selected_id: str,
    dev_hit10_delta: float,
    dev_top10_improvement_count: int = 0,
    dev_top10_regression_count: int = 0,
    extra_decision: dict | None = None,
) -> dict:
    decision = {
        "status": status,
        "stopped_candidate_id": candidate_id,
        "stopped_protocol_id": protocol_id,
        "can_continue_train_dev_development": False,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        **(extra_decision or {}),
    }
    return {
        "stage": stage,
        "decision": decision,
        "stopped_route": {
            "candidate_id": candidate_id,
            "protocol_id": protocol_id,
            "stage84_candidate_summary": _candidate(
                candidate_id=candidate_id,
                priority_score=100,
            ),
            summary_key: {
                "status": f"{candidate_id}_comparison_completed",
                "candidate_id": candidate_id,
                "protocol_id": protocol_id,
                selected_key: selected_id,
                "train_hit10_delta": 0.0,
                "dev_hit10_delta": dev_hit10_delta,
                "dev_hit1_delta": 0.0,
                "dev_top10_improvement_count": dev_top10_improvement_count,
                "dev_top10_regression_count": dev_top10_regression_count,
                "dev_not_found_count_at_search_depth_delta": 0,
                "primary_contract_passed": False,
                "secondary_contract_passed": False,
                "guard_contract_passed": True,
                "can_open_final_test_gate_now": False,
                "can_run_final_test_metrics_now": False,
                "can_use_test_for_tuning": False,
                "default_runtime_policy": "unchanged",
            },
            "stop_reason": "Fixture stop reason.",
        },
        "guard_checks": [{"name": "fixture_guard", "passed": True}],
        "private_example_strings": [
            "Restart the database service",
            "Install the firmware update",
        ],
    }


def _candidate(*, candidate_id: str, priority_score: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "name": candidate_id.replace("_", " "),
        "status": "recommended_for_train_dev_protocol_design",
        "priority_score": priority_score,
        "target_miss_count": 10,
        "target_metric_contract": [
            "primary: train-selected dev hit@10 must improve over BM25 baseline"
        ],
    }


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
