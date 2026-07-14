import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_development_checks import (
    run_primeqa_hybrid_development_checks,
    write_primeqa_hybrid_development_check_visualizations,
)


def test_primeqa_hybrid_development_checks_run_train_dev_only(tmp_path):
    paths = _write_fixture(tmp_path, include_test_candidate=False)

    report = run_primeqa_hybrid_development_checks(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        candidate_dataset_path=paths["candidate_dataset"],
        candidate_summary_path=paths["candidate_summary"],
        top_k_values=(1, 2),
    )
    visualizations = write_primeqa_hybrid_development_check_visualizations(
        report,
        tmp_path / "visuals",
    )

    assert report["stage"] == "Stage 70"
    assert report["bm25_baseline"]["metrics_by_split"]["train"]["evaluated_questions"] == 1
    assert report["bm25_baseline"]["metrics_by_split"]["dev"]["evaluated_questions"] == 1
    assert report["candidate_artifact_checks"]["scan"]["rows_with_test_split"] == 0
    assert all(check["passed"] for check in report["candidate_artifact_checks"]["checks"])
    assert all(check["passed"] for check in report["guard_checks"])
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert {artifact.name for artifact in visualizations} == {
        "stage70_primeqa_bm25_hit_at_k.svg",
        "stage70_primeqa_bm25_mrr.svg",
        "stage70_primeqa_candidate_rows_by_split.svg",
        "stage70_primeqa_candidate_questions_by_split.svg",
    }


def test_primeqa_hybrid_development_checks_block_test_candidate_rows(tmp_path):
    paths = _write_fixture(tmp_path, include_test_candidate=True)

    report = run_primeqa_hybrid_development_checks(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        candidate_dataset_path=paths["candidate_dataset"],
        candidate_summary_path=paths["candidate_summary"],
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["candidate_rows_have_no_test_split"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_train_dev_development_checks_blocked"
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
                question_title="How to restart service?",
                answer="Restart the service from the admin console.",
                answer_doc_id="doc-a",
            )
        ],
    )
    _write_jsonl(
        dev_split,
        [
            _split_sample(
                sample_id="primeqa_dev:DEV_Q001",
                assigned_split="dev",
                question_title="How to install driver?",
                answer="Install the storage driver package.",
                answer_doc_id="doc-b",
            )
        ],
    )
    documents = tmp_path / "documents.json"
    documents.write_text(
        json.dumps(
            {
                "doc-a": {
                    "id": "doc-a",
                    "title": "Restart service",
                    "text": "Restart the service from the admin console.",
                    "sections": [],
                },
                "doc-b": {
                    "id": "doc-b",
                    "title": "Install driver",
                    "text": "Install the storage driver package.",
                    "sections": [],
                },
            }
        ),
        encoding="utf-8",
    )
    candidate_rows = [
        _candidate_row("train", "primeqa_train:TRAIN_Q001", "doc-a"),
        _candidate_row("dev", "primeqa_dev:DEV_Q001", "doc-b"),
    ]
    if include_test_candidate:
        candidate_rows.append(_candidate_row("test", "primeqa_test:TEST_Q001", "doc-c"))
    candidate_dataset = tmp_path / "candidates.jsonl"
    _write_jsonl(candidate_dataset, candidate_rows)
    candidate_summary = tmp_path / "candidate_summary.json"
    candidate_summary.write_text(
        json.dumps(_candidate_summary(candidate_rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "train_split": train_split,
        "dev_split": dev_split,
        "documents": documents,
        "candidate_dataset": candidate_dataset,
        "candidate_summary": candidate_summary,
    }


def _split_sample(
    *,
    sample_id: str,
    assigned_split: str,
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
        "source_split": sample_id.split(":", maxsplit=1)[0],
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


def _candidate_row(split: str, question_id: str, document_id: str) -> dict:
    return {
        "split": split,
        "question_id": question_id,
        "candidate_id": f"{question_id}::candidate_001",
        "candidate_rank": 1,
        "runtime_features": {
            "selector_name": "fixture",
            "question_route": "other",
            "retrieval_rank": 1,
        },
        "gold_labels": {
            "candidate_token_f1": 1.0,
            "is_gold_document": True,
        },
        "metadata": {
            "document_id": document_id,
            "question_route": "other",
        },
    }


def _candidate_summary(rows: list[dict]) -> dict:
    rows_by_split = {}
    question_ids_by_split = {}
    for row in rows:
        rows_by_split[row["split"]] = rows_by_split.get(row["split"], 0) + 1
        question_ids_by_split.setdefault(row["split"], set()).add(row["question_id"])
    return {
        "stage": "Stage 69",
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "candidate_artifact_summary": {
            "summary": {
                "splits": sorted(rows_by_split),
                "selector_name": "fixture",
                "total_questions": sum(
                    len(question_ids) for question_ids in question_ids_by_split.values()
                ),
                "total_rows": len(rows),
                "questions_by_split": {
                    split: len(question_ids)
                    for split, question_ids in question_ids_by_split.items()
                },
                "rows_by_split": rows_by_split,
                "rows_by_route": {"other": len(rows)},
                "average_rows_per_question": 1.0,
                "average_top_candidate_token_f1": 1.0,
                "average_best_candidate_token_f1": 1.0,
                "average_oracle_gain_vs_top_candidate": 0.0,
                "questions_with_gold_document_candidate": len(rows),
                "gold_document_candidate_rows": len(rows),
            }
        },
        "guard_checks": [],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
