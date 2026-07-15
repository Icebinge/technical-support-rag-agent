import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_second_wave_retrieval_candidate_design import (
    design_primeqa_hybrid_second_wave_retrieval_candidates,
    write_primeqa_hybrid_second_wave_retrieval_candidate_design_visualizations,
)


def test_second_wave_candidate_design_requires_public_safe_confirmed_route(tmp_path):
    paths = _write_fixture_reports(tmp_path)

    report = design_primeqa_hybrid_second_wave_retrieval_candidates(
        **paths,
        user_confirmed_route=True,
        confirmation_note="confirmed in test",
    )
    visualizations = (
        write_primeqa_hybrid_second_wave_retrieval_candidate_design_visualizations(
            report=report,
            output_dir=tmp_path / "visuals",
        )
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 84"
    assert report["user_confirmation"]["confirmed"] is True
    assert report["decision"]["status"] == (
        "primeqa_hybrid_second_wave_retrieval_candidate_design_completed"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["requires_user_confirmation_before_train_dev_run"] is True
    assert "structured_query_keyphrase_compaction_design" in (
        report["recommended_execution_order"]
    )
    candidates = {
        candidate["candidate_id"]: candidate
        for candidate in report["candidate_designs"]
    }
    assert (
        candidates["source_doc_ids_oracle_union_blocked"]["status"]
        == "blocked_from_train_dev_experiment"
    )
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Restart the database service" not in serialized
    assert "Install the firmware update" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage84_second_wave_candidate_priority_scores.svg",
        "stage84_second_wave_candidate_target_misses.svg",
        "stage84_second_wave_candidate_dev_targets.svg",
        "stage84_second_wave_candidate_prior_signal_scores.svg",
        "stage84_second_wave_allowed_vs_blocked_candidates.svg",
    }


def test_second_wave_candidate_design_blocks_without_confirmation(tmp_path):
    paths = _write_fixture_reports(tmp_path)

    report = design_primeqa_hybrid_second_wave_retrieval_candidates(
        **paths,
        user_confirmed_route=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage84_recommended_route"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_second_wave_retrieval_candidate_design_blocked"
    )


def _write_fixture_reports(tmp_path: Path) -> dict[str, Path]:
    return {
        "stage75_report_path": _write_json(tmp_path / "stage75.json", _stage75_report()),
        "stage76_report_path": _write_json(
            tmp_path / "stage76.json",
            _stage_report("Stage 76"),
        ),
        "stage77_report_path": _write_json(
            tmp_path / "stage77.json",
            _stage77_report(),
        ),
        "stage78_report_path": _write_json(
            tmp_path / "stage78.json",
            _stage78_report(),
        ),
        "stage79_report_path": _write_json(
            tmp_path / "stage79.json",
            _stage79_report(),
        ),
        "stage80_report_path": _write_json(
            tmp_path / "stage80.json",
            _stage_report("Stage 80"),
        ),
        "stage81_report_path": _write_json(
            tmp_path / "stage81.json",
            _stage81_report(),
        ),
        "stage82_report_path": _write_json(
            tmp_path / "stage82.json",
            _stage82_report(),
        ),
        "stage83_report_path": _write_json(
            tmp_path / "stage83.json",
            _stage83_report(),
        ),
    }


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _stage75_report() -> dict:
    train_cases = [
        _miss_case(
            split="train",
            sample_id="train:1",
            bucket="not_found_top50",
            tags=[
                "gold_doc_not_found_within_top50",
                "top1_query_overlap_exceeds_gold",
                "top10_contains_source_candidate_doc",
            ],
            query_unique_terms=20,
            gold_title_overlap=5,
        )
    ]
    dev_cases = [
        _miss_case(
            split="dev",
            sample_id="dev:1",
            bucket="rank_21_to_50",
            tags=[
                "gold_doc_rank_21_to_50",
                "top1_query_overlap_exceeds_gold",
                "top10_contains_source_candidate_doc",
            ],
            query_unique_terms=18,
            gold_title_overlap=4,
        ),
        _miss_case(
            split="dev",
            sample_id="dev:2",
            bucket="not_found_top50",
            tags=[
                "gold_doc_not_found_within_top50",
                "gold_doc_query_overlap_ratio_lt_0_25",
                "top10_contains_source_candidate_doc",
            ],
            query_unique_terms=12,
            gold_title_overlap=1,
        ),
    ]
    return {
        "stage": "Stage 75",
        "split_contract": _split_contract(),
        "split_reports": {
            "train": {
                "evaluated_questions": 10,
                "hit_at_top_k": 0.9,
                "miss_count": 1,
                "miss_cases": train_cases,
            },
            "dev": {
                "evaluated_questions": 5,
                "hit_at_top_k": 0.6,
                "miss_count": 2,
                "miss_cases": dev_cases,
            },
        },
        "cross_split_summary": {
            "evaluated_questions": 15,
            "hit_at_top_k": 0.8,
            "miss_count": 3,
        },
        "decision": _decision(),
    }


def _miss_case(
    *,
    split: str,
    sample_id: str,
    bucket: str,
    tags: list[str],
    query_unique_terms: int,
    gold_title_overlap: int,
) -> dict:
    return {
        "split": split,
        "sample_id": sample_id,
        "question_route": "other",
        "gold_rank_bucket": bucket,
        "gold_in_source_candidate_doc_ids": True,
        "gold_document_token_count": 1300,
        "gold_query_overlap_count": 4,
        "gold_title_query_overlap_count": gold_title_overlap,
        "query_unique_token_count": query_unique_terms,
        "query_length_bucket": "unique_terms_16_plus"
        if query_unique_terms >= 16
        else "unique_terms_9_to_15",
        "reason_tags": tags,
        "top_results": [
            {
                "rank": rank,
                "score": 100.0 - rank,
                "title_query_overlap_count": gold_title_overlap,
            }
            for rank in range(1, 11)
        ],
    }


def _stage77_report() -> dict:
    return {
        **_stage_report("Stage 77"),
        "decision": {
            **_decision("primeqa_hybrid_query_view_ablation_completed"),
            "train_selected_view_id": "full_question_dedup_terms",
            "train_selected_dev_hit10_delta": -0.04,
            "train_selected_dev_top10_improvements": 1,
            "train_selected_dev_top10_regressions": 4,
        },
    }


def _stage78_report() -> dict:
    return {
        **_stage_report("Stage 78"),
        "decision": {
            **_decision("primeqa_hybrid_fielded_bm25_fusion_completed"),
            "train_selected_config_id": "fielded_title_0_25_text_1_00",
            "train_selected_dev_hit10_delta": 0.0,
            "train_selected_dev_top10_improvements": 1,
            "train_selected_dev_top10_regressions": 1,
        },
        "train_selection": {
            "selected_dev_comparison_to_baseline": {"mrr_delta": 0.02}
        },
    }


def _stage79_report() -> dict:
    return {
        **_stage_report("Stage 79"),
        "decision": {
            **_decision("primeqa_hybrid_section_bm25_doc_rollup_completed"),
            "candidate_config_id": "section",
            "candidate_dev_hit10_delta": -0.05,
            "candidate_dev_top10_improvements": 1,
            "candidate_dev_top10_regressions": 5,
        },
        "comparisons_to_baseline": {
            "dev": {"search_depth_net_improvement_count": -1}
        },
    }


def _stage81_report() -> dict:
    return {
        **_stage_report("Stage 81"),
        "decision": {
            **_decision("primeqa_hybrid_dense_sparse_rrf_comparison_completed"),
            "selected_config_id": "dense_sparse",
            "selected_dev_hit10_delta": -0.01,
            "selected_dev_top10_improvements": 3,
            "selected_dev_top10_regressions": 4,
            "selected_dev_not_found_at_search_depth_delta": -6,
        },
        "comparisons_to_baseline": {
            "dev": {
                "dense_sparse": {"search_depth_net_improvement_count": 6}
            }
        },
    }


def _stage82_report() -> dict:
    return {
        **_stage_report("Stage 82"),
        "decision": {
            **_decision("primeqa_hybrid_bm25_k1_b_grid_completed"),
            "selected_config_id": "full_document_bm25_baseline",
            "selected_dev_hit10_delta": 0.0,
            "selected_dev_top10_improvements": 0,
            "selected_dev_top10_regressions": 0,
        },
        "metrics_by_split": {
            "dev": {
                "full_document_bm25_baseline": {"hit_at_k": {"hit@10": 0.7}}
            }
        },
        "comparisons_to_baseline": {
            "dev": {
                "bm25_grid__k1_1_20__b_0_95": {"hit@10_delta": 0.01}
            }
        },
    }


def _stage83_report() -> dict:
    return {
        **_stage_report("Stage 83"),
        "blocked_candidate": {
            "candidate_id": "source_doc_ids_oracle_union_blocked",
            "status": "blocked_from_train_dev_experiment",
        },
        "candidate_outcomes": [],
        "dev_only_observations": [],
        "decision": {
            **_decision("primeqa_hybrid_retrieval_recall_exhaustion_summary_completed"),
            "stage76_allowed_candidates_exhausted": True,
            "runtime_advancing_candidate_count": 0,
            "recommended_next_route_option": "second_wave_retrieval_candidate_design",
            "requires_user_confirmation_before_next_route": True,
        },
    }


def _stage_report(stage: str) -> dict:
    return {
        "stage": stage,
        "split_contract": _split_contract(),
        "decision": _decision(),
    }


def _split_contract() -> dict:
    return {
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "development_splits": ["train", "dev"],
        "forbidden_final_splits": ["test"],
    }


def _decision(status: str = "fixture_completed") -> dict:
    return {
        "status": status,
        "can_continue_train_dev_development": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
    }
