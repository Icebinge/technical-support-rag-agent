import json

import pytest

from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    load_primeqa_hybrid_split_questions,
    load_primeqa_hybrid_split_samples,
    summarize_primeqa_hybrid_split_samples,
    write_primeqa_compatible_question_files,
)


def test_load_primeqa_hybrid_split_samples_uses_collision_safe_question_id(tmp_path):
    split_path = tmp_path / "train.jsonl"
    _write_jsonl(
        split_path,
        [
            _sample(
                sample_id="primeqa_train:TRAIN_Q001",
                question_id="TRAIN_Q001",
                assigned_split="train",
            )
        ],
    )

    samples = load_primeqa_hybrid_split_samples(split_path)
    questions = load_primeqa_hybrid_split_questions({"train": split_path})
    summary = summarize_primeqa_hybrid_split_samples({"train": samples})
    artifacts = write_primeqa_compatible_question_files(
        split_samples={"train": samples},
        output_dir=tmp_path / "questions",
    )

    assert samples[0].sample_id == "primeqa_train:TRAIN_Q001"
    assert questions["train"][0].id == "primeqa_train:TRAIN_Q001"
    assert questions["train"][0].answer_doc_id == "doc-a"
    assert summary["train"]["row_count"] == 1
    assert summary["train"]["unique_candidate_doc_ids"] == 2
    rows = json.loads(open(artifacts[0]["path"], encoding="utf-8").read())
    assert rows[0]["QUESTION_ID"] == "primeqa_train:TRAIN_Q001"
    assert rows[0]["SOURCE_QUESTION_ID"] == "TRAIN_Q001"


def test_primeqa_hybrid_split_loader_rejects_duplicate_sample_ids(tmp_path):
    split_path = tmp_path / "train.jsonl"
    _write_jsonl(
        split_path,
        [
            _sample(sample_id="primeqa_train:TRAIN_Q001", question_id="TRAIN_Q001"),
            _sample(sample_id="primeqa_train:TRAIN_Q001", question_id="TRAIN_Q001"),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate sample_id"):
        load_primeqa_hybrid_split_samples(split_path)


def _sample(
    *,
    sample_id: str,
    question_id: str,
    assigned_split: str = "train",
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
        "question_id": question_id,
        "question_title": "How to restart service?",
        "question_text": "",
        "question": "How to restart service?",
        "answerable": True,
        "answer": "Restart the service from the admin console.",
        "answer_doc_id": "doc-a",
        "candidate_doc_ids": ["doc-a", "doc-b"],
        "answer_span": {"start_offset": 0, "end_offset": 10},
        "metadata": {
            "group_hash": "hash-a",
            "candidate_doc_count": 2,
            "candidate_doc_hash": "hash-b",
        },
    }


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
