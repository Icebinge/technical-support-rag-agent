import json

from ts_rag_agent.application.primeqa_hybrid_retrieval_recall_candidate_design import (
    design_primeqa_hybrid_retrieval_recall_candidates,
    write_primeqa_hybrid_retrieval_recall_candidate_design_visualizations,
)


def test_retrieval_recall_candidate_design_builds_train_dev_only_plan(tmp_path):
    stage75_path = tmp_path / "stage75.json"
    stage75_path.write_text(
        json.dumps(_stage75_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = design_primeqa_hybrid_retrieval_recall_candidates(
        stage75_report_path=stage75_path,
    )
    visualizations = write_primeqa_hybrid_retrieval_recall_candidate_design_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    assert report["stage"] == "Stage 76"
    assert report["stage75_summary"]["dev"]["miss_count"] == 2
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert all(check["passed"] for check in report["guard_checks"])
    assert "section_bm25_doc_rollup_train_dev_probe" in (
        report["recommended_execution_order"]
    )
    assert "query_view_ablation_full_title_dedup" in (
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
    assert (
        candidates["source_doc_ids_oracle_union_blocked"][
            "target_miss_count_by_split"
        ]["dev"]
        == 2
    )
    assert (
        candidates["bm25_k1_b_grid_train_to_dev"]["target_miss_count_by_split"][
            "dev"
        ]
        == 1
    )
    assert {artifact.name for artifact in visualizations} == {
        "stage76_candidate_priority_scores.svg",
        "stage76_candidate_target_misses.svg",
        "stage76_candidate_dev_targets.svg",
        "stage76_allowed_vs_blocked_candidates.svg",
    }


def test_retrieval_recall_candidate_design_blocks_bad_source_report(tmp_path):
    stage75_path = tmp_path / "bad_stage75.json"
    report = _stage75_report()
    report["decision"]["default_runtime_policy"] = "changed"
    stage75_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = design_primeqa_hybrid_retrieval_recall_candidates(
        stage75_report_path=stage75_path,
    )

    checks = {check["name"]: check for check in result["guard_checks"]}
    assert checks["source_default_runtime_policy_unchanged"]["passed"] is False
    assert result["decision"]["status"] == (
        "primeqa_hybrid_retrieval_recall_candidate_design_blocked"
    )


def _stage75_report() -> dict:
    train_cases = [
        _miss_case(
            split="train",
            bucket="not_found_top50",
            tags=[
                "gold_doc_not_found_within_top50",
                "top1_query_overlap_exceeds_gold",
                "top10_contains_source_candidate_doc",
            ],
            query_bucket="unique_terms_16_plus",
            doc_tokens=1200,
            title_overlap=2,
        )
    ]
    dev_cases = [
        _miss_case(
            split="dev",
            bucket="not_found_top50",
            tags=[
                "gold_doc_not_found_within_top50",
                "gold_doc_query_overlap_ratio_lt_0_25",
                "top10_contains_source_candidate_doc",
            ],
            query_bucket="unique_terms_16_plus",
            doc_tokens=400,
            title_overlap=0,
        ),
        _miss_case(
            split="dev",
            bucket="rank_11_to_20",
            tags=[
                "gold_doc_rank_11_to_20",
                "top1_query_overlap_exceeds_gold",
                "top10_contains_source_candidate_doc",
            ],
            query_bucket="unique_terms_9_to_15",
            doc_tokens=300,
            title_overlap=5,
        ),
    ]
    return {
        "stage": "Stage 75",
        "split_contract": {
            "split_name": "primeqa_hybrid_stage68_v1",
            "protocol_version": "primeqa_hybrid_split_v1",
            "development_splits": ["train", "dev"],
            "forbidden_final_splits": ["test"],
        },
        "split_reports": {
            "train": {
                "evaluated_questions": 10,
                "hit_at_top_k": 0.9,
                "miss_count": 1,
                "miss_rate": 0.1,
                "miss_cases": train_cases,
            },
            "dev": {
                "evaluated_questions": 5,
                "hit_at_top_k": 0.6,
                "miss_count": 2,
                "miss_rate": 0.4,
                "miss_cases": dev_cases,
            },
        },
        "cross_split_summary": {
            "evaluated_questions": 15,
            "hit_at_top_k": 0.8,
            "miss_count": 3,
            "miss_rate": 0.2,
            "reason_tag_counts": {
                "gold_doc_not_found_within_top50": 2,
                "gold_doc_rank_11_to_20": 1,
            },
            "gold_rank_bucket_counts": {
                "not_found_top50": 2,
                "rank_11_to_20": 1,
            },
            "route_miss_counts": {"other": 2, "error_or_log": 1},
        },
        "guard_checks": [
            {"name": "candidate_rows_have_no_test_split", "passed": True},
        ],
        "decision": {
            "can_run_final_test_metrics_now": False,
            "default_runtime_policy": "unchanged",
        },
    }


def _miss_case(
    *,
    split: str,
    bucket: str,
    tags: list[str],
    query_bucket: str,
    doc_tokens: int,
    title_overlap: int,
) -> dict:
    return {
        "split": split,
        "sample_id": f"{split}:sample",
        "question_route": "other",
        "gold_rank_bucket": bucket,
        "gold_in_source_candidate_doc_ids": True,
        "gold_document_token_count": doc_tokens,
        "gold_title_query_overlap_count": title_overlap,
        "query_length_bucket": query_bucket,
        "reason_tags": tags,
    }
