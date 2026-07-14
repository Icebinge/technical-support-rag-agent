import json
from pathlib import Path

import pytest

from ts_rag_agent.application.msqa_stage51_adapter_comparison import (
    compare_msqa_stage51_capped_adapter,
    write_msqa_stage51_adapter_comparison_visualizations,
)


def test_msqa_stage51_adapter_comparison_uses_capped_pool_without_rebuild(tmp_path):
    paths = _write_fixture(tmp_path)

    report = compare_msqa_stage51_capped_adapter(
        split_jsonl_path=paths["split"],
        candidate_jsonl_path=paths["candidates"],
        adapter_report_path=paths["adapter_report"],
        distribution_report_path=paths["distribution_report"],
        candidate_reranker_dataset_path=paths["reranker_dataset"],
        stage31_summary_path=paths["stage31_summary"],
        model_name="ridge_candidate_token_f1",
        train_split="train",
        max_answer_candidates=2,
        max_citation_rank=1,
        sample_limit=2,
    )

    assert report["stage"] == "Stage 64"
    assert report["comparison_contract"]["candidate_pool_rebuilt"] is False
    assert report["comparison_contract"]["candidate_pool_rows"] == 4
    assert report["comparison_contract"]["candidate_jsonl_rows_with_question_key"] == 0
    assert report["decision"]["stage51_adapter_comparison_run_performed"] is True
    assert report["decision"]["can_defaultize_runtime_now"] is False
    assert report["metrics"]["question_count"] == 2
    assert "stage63_gold_source_candidate_rate" in report[
        "stage63_source_availability_warning"
    ]

    artifacts = write_msqa_stage51_adapter_comparison_visualizations(
        report,
        tmp_path / "visuals",
    )
    assert {artifact.name for artifact in artifacts} == {
        "stage64_msqa_answer_f1.svg",
        "stage64_msqa_answer_f1_delta.svg",
        "stage64_msqa_gold_source_citation.svg",
        "stage64_msqa_decision_reasons.svg",
    }
    for artifact in artifacts:
        assert Path(artifact.path).read_text(encoding="utf-8").startswith("<svg")


def test_msqa_stage51_adapter_comparison_rejects_question_text_in_candidates(tmp_path):
    paths = _write_fixture(tmp_path)
    candidate_rows = [
        json.loads(line)
        for line in paths["candidates"].read_text(encoding="utf-8").splitlines()
    ]
    candidate_rows[0]["question"] = "leaked question"
    paths["candidates"].write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in candidate_rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not contain question text"):
        compare_msqa_stage51_capped_adapter(
            split_jsonl_path=paths["split"],
            candidate_jsonl_path=paths["candidates"],
            adapter_report_path=paths["adapter_report"],
            distribution_report_path=paths["distribution_report"],
            candidate_reranker_dataset_path=paths["reranker_dataset"],
            stage31_summary_path=paths["stage31_summary"],
            model_name="ridge_candidate_token_f1",
            train_split="train",
        )


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "split": tmp_path / "split.jsonl",
        "candidates": tmp_path / "candidates.jsonl",
        "adapter_report": tmp_path / "adapter_report.json",
        "distribution_report": tmp_path / "distribution_report.json",
        "reranker_dataset": tmp_path / "reranker_dataset.jsonl",
        "stage31_summary": tmp_path / "stage31_summary.json",
    }
    _write_split(paths["split"])
    _write_candidates(paths["candidates"])
    _write_adapter_report(paths["adapter_report"])
    _write_distribution_report(paths["distribution_report"])
    _write_reranker_dataset(paths["reranker_dataset"])
    _write_stage31_summary(paths["stage31_summary"])
    return paths


def _write_split(path: Path) -> None:
    rows = [
        {
            "dataset": "microsoft_msqa",
            "split": "msqa_stage57_project_eval_v1",
            "adapter_contract_version": "msqa_eval_adapter_v1",
            "question_id": "q1",
            "answer_id": "a1",
            "question": "How do I reset Azure password?",
            "answer": "Reset the password in Azure portal.",
            "source_url": "https://learn.microsoft.com/q1",
            "metadata": {},
        },
        {
            "dataset": "microsoft_msqa",
            "split": "msqa_stage57_project_eval_v1",
            "adapter_contract_version": "msqa_eval_adapter_v1",
            "question_id": "q2",
            "answer_id": "a2",
            "question": "How do I configure Teams?",
            "answer": "Configure Teams in the admin center.",
            "source_url": "https://learn.microsoft.com/q2",
            "metadata": {},
        },
    ]
    _write_jsonl(path, rows)


def _write_candidates(path: Path) -> None:
    rows = [
        _candidate_row("q1", "a1", "q1", "Reset the password in Azure portal.", 1, 90.0),
        _candidate_row("q1", "a1", "s2", "Open the billing blade.", 2, 30.0),
        _candidate_row("q2", "a2", "q2", "Configure Teams in the admin center.", 1, 85.0),
        _candidate_row("q2", "a2", "s3", "Review unrelated logs.", 2, 20.0),
    ]
    _write_jsonl(path, rows)


def _candidate_row(
    query_id: str,
    answer_id: str,
    source_id: str,
    sentence: str,
    retrieval_rank: int,
    candidate_score: float,
) -> dict:
    candidate_id = f"{source_id}::processed_answer_sentence::001"
    return {
        "query_question_id": query_id,
        "query_answer_id": answer_id,
        "gold_source_row_id": query_id,
        "gold_source_url": f"https://learn.microsoft.com/{query_id}",
        "question_id": source_id,
        "answer_id": f"{answer_id}-{source_id}",
        "source_url": f"https://learn.microsoft.com/{source_id}",
        "source_row_id": source_id,
        "candidate_id": candidate_id,
        "candidate_row_id": f"{query_id}::{candidate_id}",
        "candidate_sentence": sentence,
        "retrieval_rank": retrieval_rank,
        "retrieval_score": 100.0 - retrieval_rank,
        "candidate_score": candidate_score,
        "overlap_terms": ["azure"] if query_id == "q1" else ["teams"],
    }


def _write_adapter_report(path: Path) -> None:
    report = {
        "stage": "Stage 63",
        "adapter_contract": {
            "top_k": 5,
            "max_candidates_per_source_row": 3,
            "effective_candidate_pool_cap": 15,
        },
        "decision": {
            "status": "msqa_stage31_aligned_candidate_adapter_dry_run_passed",
            "stage51_candidate_run_performed": False,
        },
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_distribution_report(path: Path) -> None:
    report = {
        "stage": "Stage 63",
        "adapter_candidate_distribution": {
            "gold_source_candidate_rate": 1.0,
        },
        "stage31_candidate_distribution": {
            "gold_document_candidate_rate": 1.0,
        },
        "candidate_pool_comparison": {
            "gold_candidate_rate_delta_adapter_minus_stage31": 0.0,
        },
        "decision": {
            "status": "msqa_stage51_adapter_comparison_ready_for_user_confirmation",
        },
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_reranker_dataset(path: Path) -> None:
    rows = []
    for split in ("train",):
        for question_id in ("t1", "t2"):
            rows.extend(
                [
                    _reranker_row(split, question_id, 1, 90.0, 1.0, True),
                    _reranker_row(split, question_id, 2, 20.0, 0.0, False),
                ]
            )
    _write_jsonl(path, rows)


def _reranker_row(
    split: str,
    question_id: str,
    rank: int,
    candidate_score: float,
    candidate_token_f1: float,
    is_best: bool,
) -> dict:
    return {
        "split": split,
        "question_id": question_id,
        "candidate_id": f"{question_id}::candidate_{rank:03d}",
        "candidate_rank": rank,
        "runtime_features": {
            "selector_name": "test_selector",
            "question_route": "install_upgrade_config",
            "retrieval_rank": rank,
            "retrieval_score": 100.0 - rank,
            "candidate_score": candidate_score,
            "candidate_token_count": 6,
            "candidate_sentence_count": 1,
            "question_token_count": 6,
            "query_term_count": 3,
            "query_overlap_count": 2 if is_best else 0,
            "query_overlap_ratio": 0.66 if is_best else 0.0,
            "candidate_query_coverage_ratio": 0.5 if is_best else 0.0,
            "title_query_overlap_count": 0,
            "title_query_overlap_ratio": 0.0,
            "answer_signal_score": 1.0 if is_best else 0.0,
            "problem_noise_score": 0.0,
            "has_answer_heading": False,
            "has_problem_heading": False,
            "has_question_heading": False,
            "has_url": False,
            "has_trace_noise": False,
            "symbol_ratio": 0.0,
        },
        "gold_labels": {
            "candidate_token_f1": candidate_token_f1,
            "is_best_candidate_for_question": is_best,
            "is_gold_document": is_best,
        },
        "metadata": {},
    }


def _write_stage31_summary(path: Path) -> None:
    report = {
        "dataset": "PrimeQA/TechQA",
        "build_config": {
            "retrieval_top_k": 5,
            "max_candidates_per_document": 3,
            "candidate_limit": 25,
            "evidence_selector": "test_selector",
        },
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
