import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_retrieval_recall_exhaustion_summary import (
    summarize_primeqa_hybrid_retrieval_recall_exhaustion,
    write_primeqa_hybrid_retrieval_recall_exhaustion_visualizations,
)


def test_retrieval_recall_exhaustion_summary_is_public_safe(tmp_path):
    paths = _write_fixture_reports(tmp_path)

    report = summarize_primeqa_hybrid_retrieval_recall_exhaustion(**paths)
    visualizations = write_primeqa_hybrid_retrieval_recall_exhaustion_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 83"
    assert report["aggregate_summary"]["allowed_candidates_completed"] is True
    assert report["aggregate_summary"]["runtime_advancing_candidate_count"] == 0
    assert report["decision"]["requires_user_confirmation_before_next_route"] is True
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Restart the service from the admin console" not in serialized
    assert "Install the storage driver package" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage83_candidate_dev_hit10_deltas.svg",
        "stage83_candidate_top10_net_changes.svg",
        "stage83_candidate_advancement_status.svg",
        "stage83_next_route_options.svg",
    }


def test_retrieval_recall_exhaustion_blocks_if_candidate_missing(tmp_path):
    paths = _write_fixture_reports(tmp_path)
    stage76 = json.loads(paths["stage76_report_path"].read_text(encoding="utf-8"))
    stage76["candidate_designs"] = stage76["candidate_designs"][:-1]
    paths["stage76_report_path"].write_text(
        json.dumps(stage76, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = summarize_primeqa_hybrid_retrieval_recall_exhaustion(**paths)

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["source_doc_ids_candidate_remains_blocked"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_retrieval_recall_exhaustion_summary_blocked"
    )


def _write_fixture_reports(tmp_path: Path) -> dict[str, Path]:
    stage76 = tmp_path / "stage76.json"
    stage76.write_text(
        json.dumps(
            {
                "stage": "Stage 76",
                "candidate_designs": [
                    {"candidate_id": candidate_id, "status": "recommended"}
                    for candidate_id in [
                        "query_view_ablation_full_title_dedup",
                        "fielded_title_text_bm25_score_fusion",
                        "section_bm25_doc_rollup_train_dev_probe",
                        "dense_sparse_rrf_train_dev_probe",
                        "bm25_k1_b_grid_train_to_dev",
                    ]
                ]
                + [
                    {
                        "candidate_id": "source_doc_ids_oracle_union_blocked",
                        "status": "blocked_from_train_dev_experiment",
                        "target_miss_count": 2,
                        "target_miss_count_by_split": {"train": 1, "dev": 1},
                    }
                ],
                "decision": _decision(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    reports = {
        "stage77_report_path": _write_stage_report(
            tmp_path / "stage77.json",
            stage="Stage 77",
            decision={
                **_decision("primeqa_hybrid_query_view_ablation_completed"),
                "train_selected_view_id": "full_question_dedup_terms",
                "train_selected_dev_hit10_delta": -0.1,
                "train_selected_dev_top10_improvements": 0,
                "train_selected_dev_top10_regressions": 1,
            },
        ),
        "stage78_report_path": _write_stage_report(
            tmp_path / "stage78.json",
            stage="Stage 78",
            decision={
                **_decision("primeqa_hybrid_fielded_bm25_fusion_completed"),
                "train_selected_config_id": "fielded",
                "train_selected_dev_hit10_delta": 0.0,
                "train_selected_dev_top10_improvements": 1,
                "train_selected_dev_top10_regressions": 1,
            },
        ),
        "stage79_report_path": _write_stage_report(
            tmp_path / "stage79.json",
            stage="Stage 79",
            decision={
                **_decision("primeqa_hybrid_section_bm25_doc_rollup_completed"),
                "candidate_config_id": "section",
                "candidate_dev_hit10_delta": -0.2,
                "candidate_dev_top10_improvements": 1,
                "candidate_dev_top10_regressions": 2,
                "candidate_dev_not_found_at_search_depth_delta": 1,
            },
        ),
        "stage80_report_path": _write_stage_report(
            tmp_path / "stage80.json",
            stage="Stage 80",
            decision={
                **_decision("primeqa_hybrid_dense_sparse_rrf_feasibility_completed"),
                "can_run_dense_sparse_rrf_without_download": True,
                "requires_user_confirmation_before_train_dev_run": True,
            },
        ),
        "stage81_report_path": _write_stage_report(
            tmp_path / "stage81.json",
            stage="Stage 81",
            decision={
                **_decision("primeqa_hybrid_dense_sparse_rrf_comparison_completed"),
                "selected_config_id": "dense_sparse",
                "selected_dev_hit10_delta": -0.01,
                "selected_dev_top10_improvements": 3,
                "selected_dev_top10_regressions": 4,
                "selected_dev_not_found_at_search_depth_delta": -6,
            },
        ),
        "stage82_report_path": _write_stage82_report(tmp_path / "stage82.json"),
    }
    return {"stage76_report_path": stage76, **reports}


def _write_stage_report(path: Path, *, stage: str, decision: dict) -> Path:
    path.write_text(
        json.dumps({"stage": stage, "decision": decision}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _write_stage82_report(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "stage": "Stage 82",
                "decision": {
                    **_decision("primeqa_hybrid_bm25_k1_b_grid_completed"),
                    "selected_config_id": "full_document_bm25_baseline",
                    "selected_dev_hit10_delta": 0.0,
                    "selected_dev_top10_improvements": 0,
                    "selected_dev_top10_regressions": 0,
                    "selected_dev_not_found_at_search_depth_delta": 0,
                },
                "metrics_by_split": {
                    "dev": {
                        "full_document_bm25_baseline": {
                            "hit_at_k": {"hit@10": 0.7}
                        },
                        "bm25_grid__k1_1_20__b_0_95": {
                            "hit_at_k": {"hit@10": 0.71}
                        },
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _decision(status: str = "fixture_completed") -> dict:
    return {
        "status": status,
        "can_continue_train_dev_development": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
    }
