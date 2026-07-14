import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_bm25_miss_analysis import (
    run_primeqa_hybrid_bm25_miss_analysis,
    write_primeqa_hybrid_bm25_miss_analysis_visualizations,
)


def test_primeqa_hybrid_bm25_miss_analysis_runs_public_safe_train_dev_only(tmp_path):
    paths = _write_fixture(tmp_path, include_test_candidate=False)

    report = run_primeqa_hybrid_bm25_miss_analysis(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        candidate_dataset_path=paths["candidate_dataset"],
        top_k=1,
        search_depth=3,
    )
    visualizations = write_primeqa_hybrid_bm25_miss_analysis_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 75"
    assert report["split_reports"]["train"]["miss_count"] == 0
    assert report["split_reports"]["dev"]["miss_count"] == 1
    assert report["split_reports"]["dev"]["hit_at_top_k"] == 0.0
    dev_case = report["split_reports"]["dev"]["miss_cases"][0]
    assert dev_case["question_route"] == "install_upgrade_config"
    assert "gold_doc_zero_query_overlap" in dev_case["reason_tags"]
    assert "top1_query_overlap_exceeds_gold" in dev_case["reason_tags"]
    assert all(check["passed"] for check in report["guard_checks"])
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert "Install the storage driver package" not in serialized
    assert "Restart the service from the admin console" not in serialized
    assert "Driver installation troubleshooting" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage75_bm25_miss_count_by_split.svg",
        "stage75_bm25_miss_rate_by_split.svg",
        "stage75_bm25_miss_reason_tags.svg",
        "stage75_bm25_miss_rank_buckets.svg",
        "stage75_bm25_dev_miss_routes.svg",
    }


def test_primeqa_hybrid_bm25_miss_analysis_blocks_test_candidate_rows(tmp_path):
    paths = _write_fixture(tmp_path, include_test_candidate=True)

    report = run_primeqa_hybrid_bm25_miss_analysis(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        candidate_dataset_path=paths["candidate_dataset"],
        top_k=1,
        search_depth=3,
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["candidate_artifact_splits_are_train_dev_only"]["passed"] is False
    assert checks["candidate_rows_have_no_test_split"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_bm25_top10_miss_analysis_blocked"
    )


def _write_fixture(tmp_path: Path, *, include_test_candidate: bool) -> dict[str, Path]:
    train_split = tmp_path / "train.jsonl"
    dev_split = tmp_path / "dev.jsonl"
    _write_jsonl(
        train_split,
        [
            _split_sample(
                sample_id="primeqa_train:TRAIN_Q001",
                assigned_split="train",
                source_split="primeqa_train",
                question_title="How to restart service?",
                answer="Restart the service from the admin console.",
                answer_doc_id="doc-restart",
            )
        ],
    )
    _write_jsonl(
        dev_split,
        [
            _split_sample(
                sample_id="primeqa_dev:DEV_Q001",
                assigned_split="dev",
                source_split="primeqa_dev",
                question_title="How to install driver?",
                answer="Install the storage driver package.",
                answer_doc_id="doc-gold-driver",
            )
        ],
    )
    documents = tmp_path / "documents.json"
    documents.write_text(
        json.dumps(
            {
                "doc-restart": {
                    "id": "doc-restart",
                    "title": "Restart service",
                    "text": "Restart the service from the admin console.",
                    "sections": [],
                },
                "doc-driver-decoy": {
                    "id": "doc-driver-decoy",
                    "title": "Driver installation troubleshooting",
                    "text": "Install driver package error details.",
                    "sections": [],
                },
                "doc-gold-driver": {
                    "id": "doc-gold-driver",
                    "title": "Storage package notice",
                    "text": "Use the archive bundle from the support portal.",
                    "sections": [],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    candidate_rows = [
        _candidate_row(
            split="train",
            question_id="primeqa_train:TRAIN_Q001",
            document_id="doc-restart",
            route="other",
        ),
        _candidate_row(
            split="dev",
            question_id="primeqa_dev:DEV_Q001",
            document_id="doc-gold-driver",
            route="install_upgrade_config",
        ),
    ]
    if include_test_candidate:
        candidate_rows.append(
            _candidate_row(
                split="test",
                question_id="primeqa_test:TEST_Q001",
                document_id="doc-test",
                route="other",
            )
        )
    candidate_dataset = tmp_path / "candidates.jsonl"
    _write_jsonl(candidate_dataset, candidate_rows)
    return {
        "train_split": train_split,
        "dev_split": dev_split,
        "documents": documents,
        "candidate_dataset": candidate_dataset,
    }


def _split_sample(
    *,
    sample_id: str,
    assigned_split: str,
    source_split: str,
    question_title: str,
    answer: str,
    answer_doc_id: str,
) -> dict:
    return {
        "dataset": "primeqa_techqa",
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "assigned_split": assigned_split,
        "split_subtype": f"group_random_{assigned_split}",
        "assignment_reason": "group_random_remainder_split",
        "source_split": source_split,
        "source_row_index": 1,
        "sample_id": sample_id,
        "question_id": sample_id.split(":", maxsplit=1)[1],
        "question_title": question_title,
        "question_text": "",
        "question": question_title,
        "answerable": True,
        "answer": answer,
        "answer_doc_id": answer_doc_id,
        "candidate_doc_ids": [answer_doc_id],
        "answer_span": {"start_offset": 0, "end_offset": len(answer)},
        "metadata": {
            "group_hash": f"{sample_id}:group",
            "candidate_doc_count": 1,
            "candidate_doc_hash": f"{sample_id}:docs",
        },
    }


def _candidate_row(
    *,
    split: str,
    question_id: str,
    document_id: str,
    route: str,
) -> dict:
    return {
        "split": split,
        "question_id": question_id,
        "candidate_id": f"{question_id}::candidate_001",
        "candidate_rank": 1,
        "runtime_features": {
            "selector_name": "fixture",
            "question_route": route,
            "retrieval_rank": 1,
        },
        "gold_labels": {
            "candidate_token_f1": 1.0,
            "is_gold_document": True,
        },
        "metadata": {
            "document_id": document_id,
            "question_route": route,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
