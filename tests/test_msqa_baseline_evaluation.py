import csv
import json
from pathlib import Path

from ts_rag_agent.application.msqa_baseline_evaluation import (
    evaluate_msqa_topk_baseline,
    load_msqa_baseline_samples,
    write_msqa_topk_baseline_visualizations,
)


def test_evaluate_msqa_topk_baseline_reports_answer_source_metrics(tmp_path):
    msqa_csv = tmp_path / "msqa.csv"
    _write_msqa_csv(msqa_csv)
    split_jsonl = tmp_path / "split.jsonl"
    _write_split_jsonl(split_jsonl)

    report = evaluate_msqa_topk_baseline(
        msqa_csv_path=msqa_csv,
        split_jsonl_path=split_jsonl,
        top_k_values=(1, 2),
        corpus_modes=("answer_only", "question_answer_page_text"),
        corpus_scope="all_contract_rows",
        sample_limit=3,
    )

    assert report["data"]["corpus_contract_rows"] == 3
    assert report["data"]["frozen_split_samples"] == 2
    variants = {variant["corpus_mode"]: variant for variant in report["variants"]}
    assert variants["answer_only"]["retrieval_metrics"]["hit_at_k"] == {
        "hit@1": 0.5,
        "hit@2": 1.0,
    }
    assert variants["answer_only"]["retrieval_metrics"]["mrr"] == 0.75
    assert variants["question_answer_page_text"]["retrieval_metrics"]["hit_at_k"][
        "hit@1"
    ] == 1.0
    assert report["decision"]["can_run_stage51_candidate_now"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"


def test_msqa_baseline_samples_validate_split_and_contract(tmp_path):
    split_jsonl = tmp_path / "split.jsonl"
    _write_split_jsonl(split_jsonl)

    samples = load_msqa_baseline_samples(split_jsonl)

    assert [sample.question_id for sample in samples] == ["1", "2"]


def test_msqa_baseline_visualizations_are_written(tmp_path):
    msqa_csv = tmp_path / "msqa.csv"
    _write_msqa_csv(msqa_csv)
    split_jsonl = tmp_path / "split.jsonl"
    _write_split_jsonl(split_jsonl)
    report = evaluate_msqa_topk_baseline(
        msqa_csv_path=msqa_csv,
        split_jsonl_path=split_jsonl,
        top_k_values=(1, 2, 10),
        corpus_scope="frozen_split_only",
    )

    artifacts = write_msqa_topk_baseline_visualizations(report, tmp_path / "visuals")

    assert {artifact.name for artifact in artifacts} == {
        "stage58_msqa_hit_at_1.svg",
        "stage58_msqa_hit_at_10.svg",
        "stage58_msqa_mrr.svg",
        "stage58_msqa_top1_answer_f1.svg",
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
            "QuestionText": "How do I reset Azure password?",
            "ProcessedAnswerText": "Reset the password in Azure portal.",
            "Url": "https://learn.microsoft.com/en-us/answers/questions/1/example.html",
            "Split": "test",
        },
        {
            "QuestionId": "2",
            "AnswerId": "a2",
            "QuestionText": "How do I configure Teams?",
            "ProcessedAnswerText": "Configure Teams in the admin center.",
            "Url": "https://learn.microsoft.com/en-us/answers/questions/2/example.html",
            "Split": "test",
        },
        {
            "QuestionId": "0",
            "AnswerId": "a3",
            "QuestionText": "How do I reset Azure gateway?",
            "ProcessedAnswerText": "Reset Azure password in the Azure portal.",
            "Url": "https://learn.microsoft.com/en-us/answers/questions/0/example.html",
            "Split": "train",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "False") for field in fields})


def _write_split_jsonl(path: Path) -> None:
    rows = [
        {
            "dataset": "microsoft_msqa",
            "split": "msqa_stage57_project_eval_v1",
            "adapter_contract_version": "msqa_eval_adapter_v1",
            "question_id": "1",
            "answer_id": "a1",
            "question": "How do I reset Azure password?",
            "answer": "Reset the password in Azure portal.",
            "source_url": "https://learn.microsoft.com/en-us/answers/questions/1/example.html",
            "metadata": {},
        },
        {
            "dataset": "microsoft_msqa",
            "split": "msqa_stage57_project_eval_v1",
            "adapter_contract_version": "msqa_eval_adapter_v1",
            "question_id": "2",
            "answer_id": "a2",
            "question": "How do I configure Teams?",
            "answer": "Configure Teams in the admin center.",
            "source_url": "https://learn.microsoft.com/en-us/answers/questions/2/example.html",
            "metadata": {},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
