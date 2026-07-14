import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_split_freeze import (
    freeze_primeqa_hybrid_split,
    write_primeqa_frozen_split_jsonl,
    write_primeqa_hybrid_split_freeze_visualizations,
)


def test_primeqa_hybrid_split_freeze_accepts_stage67_boundary(tmp_path):
    paths = _write_fixture(tmp_path)

    bundle = freeze_primeqa_hybrid_split(
        train_questions_path=paths["train"],
        dev_questions_path=paths["dev"],
        validation_reference_path=paths["validation"],
        document_disjoint_answer_doc_ratio=0.25,
        seed=7,
    )

    assert bundle.report["stage"] == "Stage 68"
    assert bundle.report["frozen_split"]["split_name"] == "primeqa_hybrid_stage68_v1"
    assert bundle.report["frozen_split"]["row_count"] == 8
    assert bundle.report["decision"]["status"] == "primeqa_hybrid_split_frozen_for_rebuild"
    assert bundle.report["decision"]["split_files_finalized"] is True
    assert bundle.report["decision"]["can_run_final_metrics_now"] is False
    assert all(check["passed"] for check in bundle.report["leakage_checks"])
    assert all(check["passed"] for check in bundle.report["freeze_checks"])

    report_text = json.dumps(bundle.report, ensure_ascii=False)
    assert "answer text" not in report_text
    assert "How to reset password?" not in report_text


def test_primeqa_hybrid_split_freeze_writes_jsonl_and_visualizations(tmp_path):
    paths = _write_fixture(tmp_path)
    bundle = freeze_primeqa_hybrid_split(
        train_questions_path=paths["train"],
        dev_questions_path=paths["dev"],
        validation_reference_path=paths["validation"],
        document_disjoint_answer_doc_ratio=0.25,
        seed=7,
    )

    split_artifacts = write_primeqa_frozen_split_jsonl(bundle, tmp_path / "splits")
    report = {**bundle.report, "split_artifacts": split_artifacts}
    visualizations = write_primeqa_hybrid_split_freeze_visualizations(
        report,
        tmp_path / "visuals",
    )

    rows = []
    for artifact in split_artifacts:
        path = Path(artifact["path"])
        text = path.read_text(encoding="utf-8")
        assert "\u2028" not in text
        assert len(artifact["sha256"]) == 64
        rows.extend(json.loads(line) for line in text.splitlines() if line.strip())

    assert len(rows) == 8
    assert {row["assigned_split"] for row in rows} == {"train", "dev", "test"}
    assert any(row["question"] == "How to reset password?" for row in rows)
    assert any(
        "\\u2028" in Path(artifact["path"]).read_text(encoding="utf-8")
        for artifact in split_artifacts
    )
    assert {artifact.name for artifact in visualizations} == {
        "stage68_primeqa_frozen_split_rows.svg",
        "stage68_primeqa_frozen_answerable_rows.svg",
        "stage68_primeqa_frozen_test_subtypes.svg",
        "stage68_primeqa_frozen_source_rows.svg",
    }
    for artifact in visualizations:
        assert Path(artifact.path).read_text(encoding="utf-8").startswith("<svg")


def test_primeqa_hybrid_split_freeze_preserves_strict_document_isolation(tmp_path):
    paths = _write_fixture(tmp_path)

    bundle = freeze_primeqa_hybrid_split(
        train_questions_path=paths["train"],
        dev_questions_path=paths["dev"],
        validation_reference_path=paths["validation"],
        document_disjoint_answer_doc_ratio=0.25,
        seed=7,
    )

    selected_docs = set(
        bundle.report["stage67_plan_summary"]["document_disjoint_summary"][
            "selected_answer_doc_sample"
        ]
    )
    assert selected_docs
    non_document_samples = [
        sample
        for sample in bundle.samples
        if sample["split_subtype"] != "document_disjoint"
    ]
    assert all(sample["answer_doc_id"] not in selected_docs for sample in non_document_samples)
    assert all(
        not selected_docs.intersection(sample["candidate_doc_ids"])
        for sample in non_document_samples
    )


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
            _row(
                "TRAIN_Q004",
                "How to configure TLS?",
                "Y",
                "doc-d",
                ["doc-d", "doc-e"],
                answer="answer\u2028 text",
            ),
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
    answer: str = "answer text",
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
        "ANSWER": answer if answerable == "Y" else "-",
    }


def _write_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
