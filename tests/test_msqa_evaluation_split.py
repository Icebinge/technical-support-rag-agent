import csv
import json
from pathlib import Path

from ts_rag_agent.application.msqa_evaluation_split import (
    build_msqa_evaluation_split_report,
    write_msqa_evaluation_split_visualizations,
    write_msqa_project_split_jsonl,
)


def test_build_msqa_evaluation_split_freezes_safe_test_rows(tmp_path):
    msqa_csv = tmp_path / "msqa.csv"
    _write_msqa_csv(msqa_csv)
    primeqa_train = tmp_path / "training_Q_A.json"
    primeqa_dev = tmp_path / "dev_Q_A.json"
    _write_primeqa_questions(
        primeqa_train,
        [{"QUESTION_ID": "p1", "QUESTION_TEXT": "How do I reset Azure password?"}],
    )
    _write_primeqa_questions(
        primeqa_dev,
        [{"QUESTION_ID": "p2", "QUESTION_TEXT": "How can I configure Teams?"}],
    )

    report = build_msqa_evaluation_split_report(
        msqa_csv_path=msqa_csv,
        primeqa_train_questions_path=primeqa_train,
        primeqa_dev_questions_path=primeqa_dev,
        near_duplicate_threshold=0.75,
        sample_limit=5,
    )

    assert report["adapter_contract"]["answer_field"] == "ProcessedAnswerText"
    assert "Do not fall back" in report["adapter_contract"]["no_fallback_policy"]
    counts = report["primeqa_leakage_audit"]["counts"]
    assert counts["exact_overlap_count"] == 1
    assert counts["near_duplicate_overlap_count"] == 1
    split = report["frozen_split"]
    assert split["source_split_used"] == "test"
    assert split["selected_question_ids"] == ["4"]
    assert split["filter_counts"]["source_split_candidates"] == 4
    assert split["filter_counts"]["excluded_invalid_source_url"] == 1
    assert split["filter_counts"]["excluded_internal_normalized_duplicates"] == 0
    assert split["filter_counts"]["excluded_primeqa_leakage"] == 2
    assert split["filter_counts"]["selected_question_count"] == 1
    assert report["readiness"]["can_run_msqa_topk_baseline_next"] is True
    assert report["readiness"]["can_run_stage51_comparison_now"] is False


def test_msqa_split_outputs_jsonl_and_visualizations(tmp_path):
    msqa_csv = tmp_path / "msqa.csv"
    _write_msqa_csv(msqa_csv)
    primeqa_train = tmp_path / "training_Q_A.json"
    primeqa_dev = tmp_path / "dev_Q_A.json"
    _write_primeqa_questions(primeqa_train, [])
    _write_primeqa_questions(primeqa_dev, [{"QUESTION_ID": "p1", "QUESTION_TEXT": "none"}])
    report = build_msqa_evaluation_split_report(
        msqa_csv_path=msqa_csv,
        primeqa_train_questions_path=primeqa_train,
        primeqa_dev_questions_path=primeqa_dev,
    )
    split_output = tmp_path / "split.jsonl"

    write_msqa_project_split_jsonl(
        report=report,
        msqa_csv_path=msqa_csv,
        output_path=split_output,
    )
    artifacts = write_msqa_evaluation_split_visualizations(report, tmp_path / "visuals")

    rows = [
        json.loads(line)
        for line in split_output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    split_text = split_output.read_text(encoding="utf-8")
    assert "\u2028" not in split_text
    assert "\\u2028" in split_text
    for line in split_text.splitlines():
        if line.strip():
            json.loads(line)
    assert [row["question_id"] for row in rows] == report["frozen_split"][
        "selected_question_ids"
    ]
    assert {artifact.name for artifact in artifacts} == {
        "stage57_msqa_leakage_counts.svg",
        "stage57_msqa_split_filter_counts.svg",
        "stage57_msqa_selected_domain_flags.svg",
        "stage57_msqa_adapter_field_coverage.svg",
    }
    for artifact in artifacts:
        assert Path(artifact.path).read_text(encoding="utf-8").startswith("<svg")


def _write_msqa_csv(path: Path) -> None:
    fields = [
        "QuestionId",
        "AnswerId",
        "Tags",
        "IsAzure",
        "IsM365",
        "IsOther",
        "QuestionText",
        "AnswerText",
        "Url",
        "ProcessedAnswerText",
        "isLong",
        "isShort",
        "Split",
    ]
    rows = [
        {
            "QuestionId": "1",
            "AnswerId": "a1",
            "IsAzure": "True",
            "QuestionText": "How do I reset Azure password?",
            "Url": "https://learn.microsoft.com/en-us/answers/questions/1/example.html",
            "ProcessedAnswerText": "Reset it in Azure.",
            "Split": "test",
        },
        {
            "QuestionId": "2",
            "AnswerId": "a2",
            "IsM365": "True",
            "QuestionText": "How can I configure Teams quickly?",
            "Url": "https://learn.microsoft.com/en-us/answers/questions/2/example.html",
            "ProcessedAnswerText": "Use Teams admin center.",
            "Split": "test",
        },
        {
            "QuestionId": "3",
            "AnswerId": "a3",
            "IsOther": "True",
            "QuestionText": "How do I configure SQL Server?",
            "Url": "https://example.com/questions/3",
            "ProcessedAnswerText": "Use SQL Server configuration manager.",
            "Split": "test",
        },
        {
            "QuestionId": "4",
            "AnswerId": "a4",
            "IsOther": "True",
            "QuestionText": "How do I install a storage driver?",
            "Url": "https://learn.microsoft.com/en-us/answers/questions/4/example.html",
            "ProcessedAnswerText": "Install the storage\u2028driver package.",
            "Split": "test",
        },
        {
            "QuestionId": "5",
            "AnswerId": "a5",
            "IsOther": "True",
            "QuestionText": "How do I install a storage driver?",
            "Url": "https://learn.microsoft.com/en-us/answers/questions/5/example.html",
            "ProcessedAnswerText": "Install the storage driver package.",
            "Split": "train",
        },
        {
            "QuestionId": "6",
            "AnswerId": "a6",
            "IsOther": "True",
            "QuestionText": "How do I configure Exchange?",
            "Url": "https://learn.microsoft.com/en-us/answers/questions/6/example.html",
            "ProcessedAnswerText": "Use Exchange admin center.",
            "Split": "train",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "False") for field in fields})


def _write_primeqa_questions(path: Path, rows: list[dict]) -> None:
    payload = []
    for row in rows:
        payload.append(
            {
                "QUESTION_ID": row["QUESTION_ID"],
                "QUESTION_TITLE": row.get("QUESTION_TITLE", ""),
                "QUESTION_TEXT": row["QUESTION_TEXT"],
                "ANSWER": "answer",
                "ANSWERABLE": "Y",
                "DOCUMENT": "doc",
                "DOC_IDS": ["doc"],
                "START_OFFSET": 0,
                "END_OFFSET": 1,
            }
        )
    path.write_text(json.dumps(payload), encoding="utf-8")
