import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_split_plan import (
    plan_primeqa_hybrid_split,
    write_primeqa_hybrid_split_assignments,
    write_primeqa_hybrid_split_visualizations,
)


def test_primeqa_hybrid_split_keeps_duplicate_groups_together(tmp_path):
    paths = _write_fixture(tmp_path)

    report = plan_primeqa_hybrid_split(
        train_questions_path=paths["train"],
        dev_questions_path=paths["dev"],
        validation_reference_path=paths["validation"],
        document_disjoint_answer_doc_ratio=0.25,
        seed=7,
    )

    assert report["stage"] == "Stage 67"
    assert report["input_summary"]["row_count"] == 8
    assert report["input_summary"]["duplicate_group_count"] == 1
    assert report["decision"]["can_run_final_metrics_now"] is False
    assert report["decision"]["split_files_finalized"] is False
    assert all(check["passed"] for check in report["leakage_checks"])

    duplicate_rows = [
        row
        for row in report["assignments"]
        if row["question_id"] == "DEV_Q001"
    ]
    assert len(duplicate_rows) == 2
    assert len({row["assigned_split"] for row in duplicate_rows}) == 1


def test_primeqa_hybrid_split_strictly_isolates_selected_documents(tmp_path):
    paths = _write_fixture(tmp_path)

    report = plan_primeqa_hybrid_split(
        train_questions_path=paths["train"],
        dev_questions_path=paths["dev"],
        validation_reference_path=paths["validation"],
        document_disjoint_answer_doc_ratio=0.25,
        seed=7,
    )

    document_summary = report["document_disjoint_summary"]
    assert document_summary["selected_answer_doc_count"] == 2
    assert document_summary["document_disjoint_row_count"] >= 2
    assert document_summary["candidate_doc_intersection_only_group_count"] >= 1
    checks = {check["name"]: check for check in report["leakage_checks"]}
    assert checks[
        "selected_document_candidate_doc_ids_only_in_document_disjoint_test"
    ] == {
        "name": "selected_document_candidate_doc_ids_only_in_document_disjoint_test",
        "passed": True,
        "observed": 0,
        "expected": 0,
    }

    selected_docs = set(document_summary["selected_answer_doc_sample"])
    non_document_rows = [
        row
        for row in report["assignments"]
        if row["split_subtype"] != "document_disjoint"
    ]
    assert all(row["answer_doc_id"] not in selected_docs for row in non_document_rows)


def test_primeqa_hybrid_split_writes_artifacts_without_raw_text(tmp_path):
    paths = _write_fixture(tmp_path)
    report = plan_primeqa_hybrid_split(
        train_questions_path=paths["train"],
        dev_questions_path=paths["dev"],
        validation_reference_path=paths["validation"],
        document_disjoint_answer_doc_ratio=0.25,
        seed=7,
    )

    assignments_path = tmp_path / "assignments.jsonl"
    write_primeqa_hybrid_split_assignments(report, assignments_path)
    rows = [
        json.loads(line)
        for line in assignments_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows
    assert "question_text" not in rows[0]
    assert "answer" not in rows[0]

    artifacts = write_primeqa_hybrid_split_visualizations(report, tmp_path / "visuals")
    assert {artifact.name for artifact in artifacts} == {
        "stage67_primeqa_split_rows.svg",
        "stage67_primeqa_answerable_rows.svg",
        "stage67_primeqa_test_subtypes.svg",
        "stage67_primeqa_source_rows.svg",
    }
    for artifact in artifacts:
        assert Path(artifact.path).read_text(encoding="utf-8").startswith("<svg")


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "train": tmp_path / "training_Q_A.json",
        "dev": tmp_path / "dev_Q_A.json",
        "validation": tmp_path / "validation_reference.json",
    }
    _write_json(
        paths["train"],
        [
            _row("TRAIN_Q001", "How to reset password?", "Y", "doc-a", ["doc-a", "doc-x"]),
            _row("TRAIN_Q002", "How to restart service?", "Y", "doc-b", ["doc-b", "doc-c"]),
            _row("TRAIN_Q003", "Why is login slow?", "N", "-", ["doc-c", "doc-d"]),
            _row("TRAIN_Q004", "How to configure TLS?", "Y", "doc-d", ["doc-d", "doc-e"]),
        ],
    )
    _write_json(
        paths["dev"],
        [
            _row("DEV_Q001", "How to install client?", "Y", "doc-e", ["doc-e", "doc-a"]),
            _row("DEV_Q002", "Why does backup fail?", "Y", "doc-f", ["doc-f", "doc-g"]),
            _row("DEV_Q003", "What means error 500?", "N", "-", ["doc-h", "doc-i"]),
        ],
    )
    _write_json(
        paths["validation"],
        [
            _row("DEV_Q001", "How to install client?", "Y", "doc-e", ["doc-e", "doc-a"]),
        ],
    )
    return paths


def _row(
    question_id: str,
    question_text: str,
    answerable: str,
    document: str,
    doc_ids: list[str],
) -> dict:
    return {
        "QUESTION_ID": question_id,
        "QUESTION_TITLE": question_text,
        "QUESTION_TEXT": "",
        "DOC_IDS": doc_ids,
        "ANSWERABLE": answerable,
        "DOCUMENT": document,
        "START_OFFSET": 0 if answerable == "Y" else "-",
        "END_OFFSET": 10 if answerable == "Y" else "-",
        "ANSWER": "answer text" if answerable == "Y" else "-",
    }


def _write_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
