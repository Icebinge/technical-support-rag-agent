import json
from pathlib import Path

import pytest

from ts_rag_agent.application.primeqa_hybrid_split_rebuild import (
    rebuild_primeqa_hybrid_train_dev_artifacts,
    write_primeqa_hybrid_rebuild_candidate_artifacts,
    write_primeqa_hybrid_rebuild_question_artifacts,
    write_primeqa_hybrid_rebuild_visualizations,
)


def test_rebuild_primeqa_hybrid_artifacts_uses_train_dev_only(tmp_path):
    paths = _write_fixture(tmp_path)

    bundle = rebuild_primeqa_hybrid_train_dev_artifacts(
        split_paths=paths["splits"],
        documents_path=paths["documents"],
        candidate_splits=("train", "dev"),
        evidence_selector_name="overlap",
        max_candidates_per_document=1,
        min_candidate_score=0.0,
    )
    question_artifacts = write_primeqa_hybrid_rebuild_question_artifacts(
        bundle,
        tmp_path / "question_artifacts",
    )
    candidate_artifact = write_primeqa_hybrid_rebuild_candidate_artifacts(
        bundle,
        dataset_output=tmp_path / "candidates.jsonl",
        summary_output=tmp_path / "candidates.summary.json",
    )
    report = {
        **bundle.report,
        "question_artifacts": question_artifacts,
        "candidate_artifact": candidate_artifact,
    }
    visualizations = write_primeqa_hybrid_rebuild_visualizations(
        report,
        tmp_path / "visuals",
    )

    assert bundle.report["stage"] == "Stage 69"
    assert bundle.report["decision"]["status"] == "primeqa_hybrid_train_dev_rebuild_ready"
    assert bundle.report["decision"]["can_run_final_test_metrics_now"] is False
    assert all(check["passed"] for check in bundle.report["guard_checks"])
    assert bundle.candidate_build.summary.splits == ["dev", "train"]
    assert set(bundle.candidate_build.summary.rows_by_split) == {"dev", "train"}
    assert all(row.split != "test" for row in bundle.candidate_build.rows)
    assert len(question_artifacts) == 3
    assert candidate_artifact["dataset_row_count"] == len(bundle.candidate_build.rows)
    assert Path(candidate_artifact["dataset_path"]).exists()
    assert {artifact.name for artifact in visualizations} == {
        "stage69_primeqa_loaded_split_rows.svg",
        "stage69_primeqa_loaded_answerable_rows.svg",
        "stage69_primeqa_candidate_rows_by_split.svg",
        "stage69_primeqa_candidate_questions_by_split.svg",
    }


def test_rebuild_primeqa_hybrid_artifacts_rejects_test_candidate_split(tmp_path):
    paths = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="Test split is locked"):
        rebuild_primeqa_hybrid_train_dev_artifacts(
            split_paths=paths["splits"],
            documents_path=paths["documents"],
            candidate_splits=("train", "test"),
            evidence_selector_name="overlap",
            min_candidate_score=0.0,
        )


def _write_fixture(tmp_path: Path) -> dict:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    split_paths = {
        "train": split_dir / "train.jsonl",
        "dev": split_dir / "dev.jsonl",
        "test": split_dir / "test.jsonl",
    }
    _write_jsonl(
        split_paths["train"],
        [
            _sample(
                sample_id="primeqa_train:TRAIN_Q001",
                assigned_split="train",
                question_title="How to restart service?",
                answer="Restart the service from the admin console.",
                answer_doc_id="doc-a",
            )
        ],
    )
    _write_jsonl(
        split_paths["dev"],
        [
            _sample(
                sample_id="primeqa_dev:DEV_Q001",
                assigned_split="dev",
                question_title="How to install driver?",
                answer="Install the storage driver package.",
                answer_doc_id="doc-b",
            )
        ],
    )
    _write_jsonl(
        split_paths["test"],
        [
            _sample(
                sample_id="primeqa_train:TRAIN_Q999",
                assigned_split="test",
                split_subtype="document_disjoint",
                question_title="How to configure TLS?",
                answer="Enable TLS in the security configuration.",
                answer_doc_id="doc-c",
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
                "doc-c": {
                    "id": "doc-c",
                    "title": "TLS configuration",
                    "text": "Enable TLS in the security configuration.",
                    "sections": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return {"splits": split_paths, "documents": documents}


def _sample(
    *,
    sample_id: str,
    assigned_split: str,
    question_title: str,
    answer: str,
    answer_doc_id: str,
    split_subtype: str | None = None,
) -> dict:
    return {
        "dataset": "primeqa_techqa",
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "assigned_split": assigned_split,
        "split_subtype": split_subtype or f"group_random_{assigned_split}",
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
