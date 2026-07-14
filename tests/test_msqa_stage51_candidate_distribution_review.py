import json
from pathlib import Path

import pytest

from ts_rag_agent.application.msqa_stage51_candidate_distribution_review import (
    review_msqa_stage51_candidate_distribution,
    write_msqa_stage51_candidate_distribution_visualizations,
)


def test_candidate_distribution_review_blocks_unaligned_candidate_pool(tmp_path):
    adapter_report = tmp_path / "stage61.json"
    candidate_jsonl = tmp_path / "candidates.jsonl"
    stage31_summary = tmp_path / "stage31.json"
    _write_stage61_report(adapter_report)
    _write_candidate_jsonl(candidate_jsonl, candidate_counts=(8, 9))
    _write_stage31_summary(stage31_summary, candidate_counts=(3, 3))

    report = review_msqa_stage51_candidate_distribution(
        adapter_report_path=adapter_report,
        candidate_jsonl_path=candidate_jsonl,
        stage31_summary_path=stage31_summary,
    )

    assert report["stage"] == "Stage 62"
    assert report["decision"]["status"] == (
        "msqa_stage51_adapter_comparison_blocked_by_candidate_pool_mismatch"
    )
    assert report["decision"]["can_run_stage51_candidate_now"] is False
    assert "candidate_pool_size_aligned_with_stage31" in report["decision"][
        "blocker_checks"
    ]
    assert report["candidate_pool_comparison"][
        "stage61_median_exceeds_stage31_max"
    ] is True


def test_candidate_distribution_review_reports_counts_and_gold_coverage(tmp_path):
    adapter_report = tmp_path / "stage61.json"
    candidate_jsonl = tmp_path / "candidates.jsonl"
    stage31_summary = tmp_path / "stage31.json"
    _write_stage61_report(adapter_report)
    _write_candidate_jsonl(candidate_jsonl, candidate_counts=(4, 6))
    _write_stage31_summary(stage31_summary, candidate_counts=(3, 3))

    report = review_msqa_stage51_candidate_distribution(
        adapter_report_path=adapter_report,
        candidate_jsonl_path=candidate_jsonl,
        stage31_summary_path=stage31_summary,
    )

    stage61_counts = report["stage61_candidate_distribution"][
        "candidate_count_per_query"
    ]
    assert stage61_counts["count"] == 2
    assert stage61_counts["average"] == 5.0
    assert report["stage61_candidate_distribution"]["rows_with_question_key"] == 0
    assert report["stage61_candidate_distribution"]["queries_with_gold_source_candidate"] == 2
    assert report["stage61_candidate_distribution"]["gold_source_candidate_rate"] == 1.0
    assert report["stage61_candidate_distribution"]["retrieval_rank_counts"] == {
        "1": 9,
        "2": 1,
    }


def test_candidate_distribution_review_visualizations_are_written(tmp_path):
    adapter_report = tmp_path / "stage61.json"
    candidate_jsonl = tmp_path / "candidates.jsonl"
    stage31_summary = tmp_path / "stage31.json"
    _write_stage61_report(adapter_report)
    _write_candidate_jsonl(candidate_jsonl, candidate_counts=(4, 5))
    _write_stage31_summary(stage31_summary, candidate_counts=(3, 3))
    report = review_msqa_stage51_candidate_distribution(
        adapter_report_path=adapter_report,
        candidate_jsonl_path=candidate_jsonl,
        stage31_summary_path=stage31_summary,
    )

    artifacts = write_msqa_stage51_candidate_distribution_visualizations(
        report,
        tmp_path / "visuals",
    )

    assert {artifact.name for artifact in artifacts} == {
        "stage62_candidate_count_percentiles.svg",
        "stage62_stage31_vs_stage61_candidate_pool.svg",
        "stage62_candidate_rows_by_retrieval_rank.svg",
        "stage62_fairness_checks.svg",
    }
    for artifact in artifacts:
        assert Path(artifact.path).read_text(encoding="utf-8").startswith("<svg")


def test_candidate_distribution_review_requires_stage61_passed_report(tmp_path):
    adapter_report = tmp_path / "stage61.json"
    candidate_jsonl = tmp_path / "candidates.jsonl"
    stage31_summary = tmp_path / "stage31.json"
    _write_stage61_report(adapter_report, status="blocked")
    _write_candidate_jsonl(candidate_jsonl, candidate_counts=(4, 5))
    _write_stage31_summary(stage31_summary, candidate_counts=(3, 3))

    with pytest.raises(ValueError, match="adapter dry run must have passed"):
        review_msqa_stage51_candidate_distribution(
            adapter_report_path=adapter_report,
            candidate_jsonl_path=candidate_jsonl,
            stage31_summary_path=stage31_summary,
        )


def _write_stage61_report(path: Path, *, status: str | None = None) -> None:
    status = status or "msqa_stage51_candidate_adapter_dry_run_passed"
    report = {
        "stage": "Stage 61",
        "dry_run_summary": {
            "evaluation_samples": 2,
            "candidate_rows": 9,
            "samples_with_candidates": 2,
            "samples_without_candidates": 0,
            "samples_with_gold_source_candidate": 2,
        },
        "source_retrieval_summary": {
            "hit@1": 0.5,
            "hit@2": 1.0,
            "mrr": 0.75,
        },
        "candidate_contract_checks": [
            {"name": "a", "passed": True},
            {"name": "b", "passed": True},
        ],
        "decision": {
            "status": status,
            "stage51_candidate_run_performed": False,
        },
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_candidate_jsonl(path: Path, *, candidate_counts: tuple[int, int]) -> None:
    rows = []
    for query_index, count in enumerate(candidate_counts, start=1):
        query_id = f"q{query_index}"
        for candidate_index in range(1, count + 1):
            source_row_id = query_id if candidate_index == 1 else f"s{candidate_index}"
            rows.append(
                {
                    "query_question_id": query_id,
                    "source_row_id": source_row_id,
                    "gold_source_row_id": query_id,
                    "retrieval_rank": 1 if candidate_index <= 5 else 2,
                    "candidate_score": float(count - candidate_index + 1),
                    "candidate_id": (
                        f"{source_row_id}::processed_answer_sentence::"
                        f"{candidate_index:03d}"
                    ),
                }
            )
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_stage31_summary(
    path: Path,
    *,
    candidate_counts: tuple[int, int],
) -> None:
    report = {
        "dataset": "PrimeQA/TechQA",
        "build_config": {
            "retrieval_top_k": 1,
            "max_candidates_per_document": 3,
            "candidate_limit": 25,
            "evidence_selector": "test_selector",
        },
        "summary": {
            "total_questions": len(candidate_counts),
            "total_rows": sum(candidate_counts),
            "average_rows_per_question": sum(candidate_counts) / len(candidate_counts),
        },
        "question_summaries": [
            {
                "candidate_count": candidate_count,
                "gold_document_candidate_count": 1,
            }
            for candidate_count in candidate_counts
        ],
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
