import csv
import json
from pathlib import Path

from ts_rag_agent.application.msqa_schema_probe import (
    probe_msqa_dataset,
    write_msqa_schema_probe_visualizations,
)


def test_probe_msqa_dataset_reports_schema_links_test_ids_and_exact_overlap(tmp_path):
    msqa_csv = tmp_path / "msqa-32k.csv"
    _write_msqa_csv(msqa_csv)
    test_id_file = tmp_path / "test_id.txt"
    test_id_file.write_text("1\nmissing\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("# MSQA\n", encoding="utf-8")
    primeqa_train = tmp_path / "training_Q_A.json"
    primeqa_dev = tmp_path / "dev_Q_A.json"
    _write_primeqa_questions(
        primeqa_train,
        [{"QUESTION_ID": "p1", "QUESTION_TEXT": "How do I reset Azure password?"}],
    )
    _write_primeqa_questions(
        primeqa_dev,
        [{"QUESTION_ID": "p2", "QUESTION_TEXT": "Different question"}],
    )

    report = probe_msqa_dataset(
        msqa_csv_path=msqa_csv,
        test_id_path=test_id_file,
        readme_path=readme,
        repository_head="abc123",
        primeqa_train_questions_path=primeqa_train,
        primeqa_dev_questions_path=primeqa_dev,
        readme_row_count_claim=3,
        sample_limit=2,
    )

    assert report["schema"]["row_count"] == 3
    assert report["schema"]["unique_question_ids"] == 3
    assert report["schema"]["required_field_missing_counts"]["QuestionText"] == 0
    assert report["schema"]["answer_field_candidates"]["DoubleProcessedAnswerText"][
        "missing_count"
    ] == 1
    assert report["source_link_coverage"]["rows_with_row_url"]["count"] == 3
    assert report["source_link_coverage"]["rows_with_answer_text_link"]["count"] == 2
    assert report["test_id_file"]["test_id_count"] == 2
    assert report["test_id_file"]["test_ids_found_in_csv"] == 1
    assert report["test_id_file"]["test_ids_missing_from_csv"] == ["missing"]
    assert (
        report["primeqa_exact_leakage_precheck"]["exact_overlap_msqa_question_count"]
        == 1
    )
    assert report["readiness"]["can_run_final_metrics_now"] is False
    assert report["readiness"]["default_runtime_policy"] == "unchanged"


def test_probe_msqa_dataset_visualizations_are_written(tmp_path):
    msqa_csv = tmp_path / "msqa-32k.csv"
    _write_msqa_csv(msqa_csv)
    test_id_file = tmp_path / "test_id.txt"
    test_id_file.write_text("1\n2\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("# MSQA\n", encoding="utf-8")
    primeqa_train = tmp_path / "training_Q_A.json"
    primeqa_dev = tmp_path / "dev_Q_A.json"
    _write_primeqa_questions(primeqa_train, [])
    _write_primeqa_questions(
        primeqa_dev,
        [{"QUESTION_ID": "p1", "QUESTION_TEXT": "No overlap"}],
    )
    report = probe_msqa_dataset(
        msqa_csv_path=msqa_csv,
        test_id_path=test_id_file,
        readme_path=readme,
        repository_head="abc123",
        primeqa_train_questions_path=primeqa_train,
        primeqa_dev_questions_path=primeqa_dev,
        readme_row_count_claim=3,
    )

    artifacts = write_msqa_schema_probe_visualizations(report, tmp_path / "visuals")

    assert {artifact.name for artifact in artifacts} == {
        "stage56_msqa_split_distribution.svg",
        "stage56_msqa_source_link_coverage.svg",
        "stage56_msqa_domain_flags.svg",
        "stage56_msqa_test_id_coverage.svg",
        "stage56_msqa_primeqa_exact_overlap.svg",
    }
    for artifact in artifacts:
        assert Path(artifact.path).read_text(encoding="utf-8").startswith("<svg")


def _write_msqa_csv(path: Path) -> None:
    fields = [
        "QuestionId",
        "AnswerId",
        "CreationDate",
        "Score",
        "QuestionScore",
        "AnswerScore",
        "Tags",
        "IsAzure",
        "IsM365",
        "IsOther",
        "QuestionText",
        "AnswerText",
        "Url",
        "ProcessedAnswerText",
        "QuestionTokenLength",
        "AnswerTokenLength",
        "SampleTokenLength",
        "isLong",
        "QuestionNoLinkTokenLen",
        "QuestionNoLinkCharLen",
        "QuestionContainLink",
        "QuestionIsShort",
        "ProcessedAnswerNoLinkTokenLen",
        "ProcessedAnswerNoLinkCharLen",
        "ProcessedAnswerContainLink",
        "ProcessedAnswerIsShort",
        "isShort",
        "Split",
        "DoubleProcessedAnswerText",
    ]
    rows = [
        {
            "QuestionId": "1",
            "AnswerId": "a1",
            "IsAzure": "True",
            "IsM365": "False",
            "IsOther": "False",
            "QuestionText": "How do I reset Azure password?",
            "AnswerText": "Read https://learn.microsoft.com/en-us/azure/example",
            "Url": "https://learn.microsoft.com/en-us/answers/questions/1/example.html",
            "ProcessedAnswerText": "Use https://learn.microsoft.com/en-us/azure/example",
            "isLong": "False",
            "isShort": "False",
            "Split": "test",
            "DoubleProcessedAnswerText": "Use https://learn.microsoft.com/en-us/azure/example",
        },
        {
            "QuestionId": "2",
            "AnswerId": "a2",
            "IsAzure": "False",
            "IsM365": "True",
            "IsOther": "False",
            "QuestionText": "How do I configure Teams?",
            "AnswerText": "Open Teams admin center.",
            "Url": "https://learn.microsoft.com/en-us/answers/questions/2/example.html",
            "ProcessedAnswerText": "Open Teams admin center.",
            "isLong": "False",
            "isShort": "False",
            "Split": "train",
            "DoubleProcessedAnswerText": "",
        },
        {
            "QuestionId": "3",
            "AnswerId": "a3",
            "IsAzure": "False",
            "IsM365": "False",
            "IsOther": "True",
            "QuestionText": "How do I configure SQL Server?",
            "AnswerText": "See https://example.com/sql",
            "Url": "https://learn.microsoft.com/en-us/answers/questions/3/example.html",
            "ProcessedAnswerText": "See SQL docs.",
            "isLong": "False",
            "isShort": "True",
            "Split": "NNN",
            "DoubleProcessedAnswerText": "See SQL docs.",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


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
