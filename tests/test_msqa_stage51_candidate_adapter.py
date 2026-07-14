import json
from collections import Counter
from pathlib import Path

import pytest

from ts_rag_agent.application.msqa_stage51_candidate_adapter import (
    build_msqa_stage51_candidate_adapter_dry_run,
    candidate_row_to_dict,
    write_msqa_stage51_candidate_adapter_visualizations,
    write_msqa_stage51_candidate_jsonl,
)


def test_msqa_stage51_candidate_adapter_builds_confirmed_answer_sentence_rows(
    tmp_path,
):
    split_jsonl = tmp_path / "split.jsonl"
    protocol_report = tmp_path / "protocol.json"
    _write_split_jsonl(split_jsonl)
    _write_protocol_report(protocol_report)

    dry_run = build_msqa_stage51_candidate_adapter_dry_run(
        split_jsonl_path=split_jsonl,
        protocol_report_path=protocol_report,
        confirmed_protocol=True,
        top_k=2,
        min_sentence_chars=1,
    )

    report = dry_run.report
    assert report["stage"] == "Stage 61"
    assert report["user_confirmation"]["confirmed_protocol_option"] == "A"
    assert report["decision"]["can_run_stage51_candidate_now"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert report["adapter_contract"]["retrieval_index_text"] == (
        "ProcessedAnswerText only"
    )
    assert report["adapter_contract"]["excluded_index_text"] == "QuestionText"
    assert report["dry_run_summary"]["evaluation_samples"] == 2
    assert report["dry_run_summary"]["samples_with_candidates"] == 2
    assert report["dry_run_summary"]["candidate_rows"] >= 4
    assert all(check["passed"] for check in report["candidate_contract_checks"])

    first_row = candidate_row_to_dict(dry_run.candidate_rows[0])
    assert "question" not in first_row
    assert first_row["candidate_id"].startswith(
        f"{first_row['source_row_id']}::processed_answer_sentence::"
    )
    assert first_row["question_id"] == first_row["source_row_id"]
    assert first_row["candidate_sentence"]


def test_msqa_stage51_candidate_adapter_requires_confirmation(tmp_path):
    split_jsonl = tmp_path / "split.jsonl"
    protocol_report = tmp_path / "protocol.json"
    _write_split_jsonl(split_jsonl)
    _write_protocol_report(protocol_report)

    with pytest.raises(ValueError, match="confirmed_protocol=True"):
        build_msqa_stage51_candidate_adapter_dry_run(
            split_jsonl_path=split_jsonl,
            protocol_report_path=protocol_report,
            confirmed_protocol=False,
        )


def test_msqa_stage51_candidate_adapter_caps_candidates_per_source_row(tmp_path):
    split_jsonl = tmp_path / "split.jsonl"
    protocol_report = tmp_path / "protocol.json"
    _write_split_jsonl(split_jsonl)
    _write_protocol_report(protocol_report)

    dry_run = build_msqa_stage51_candidate_adapter_dry_run(
        split_jsonl_path=split_jsonl,
        protocol_report_path=protocol_report,
        confirmed_protocol=True,
        top_k=2,
        min_sentence_chars=1,
        max_candidates_per_source_row=1,
        stage_name="Stage 63",
    )

    report = dry_run.report
    assert report["stage"] == "Stage 63"
    assert report["decision"]["status"] == (
        "msqa_stage31_aligned_candidate_adapter_dry_run_passed"
    )
    assert report["adapter_contract"]["max_candidates_per_source_row"] == 1
    assert report["adapter_contract"]["effective_candidate_pool_cap"] == 2
    counts = Counter(
        (row.query_question_id, row.source_row_id) for row in dry_run.candidate_rows
    )
    assert counts
    assert max(counts.values()) == 1
    assert report["dry_run_summary"]["candidate_rows"] <= 4


def test_msqa_stage51_candidate_adapter_rejects_wrong_protocol(tmp_path):
    split_jsonl = tmp_path / "split.jsonl"
    protocol_report = tmp_path / "protocol.json"
    _write_split_jsonl(split_jsonl)
    _write_protocol_report(protocol_report, candidate_construction="wrong")

    with pytest.raises(ValueError, match="candidate construction"):
        build_msqa_stage51_candidate_adapter_dry_run(
            split_jsonl_path=split_jsonl,
            protocol_report_path=protocol_report,
            confirmed_protocol=True,
        )


def test_msqa_stage51_candidate_adapter_writes_jsonl_and_visualizations(tmp_path):
    split_jsonl = tmp_path / "split.jsonl"
    protocol_report = tmp_path / "protocol.json"
    candidate_output = tmp_path / "candidates.jsonl"
    _write_split_jsonl(split_jsonl)
    _write_protocol_report(protocol_report)
    dry_run = build_msqa_stage51_candidate_adapter_dry_run(
        split_jsonl_path=split_jsonl,
        protocol_report_path=protocol_report,
        confirmed_protocol=True,
        top_k=2,
    )

    write_msqa_stage51_candidate_jsonl(
        candidate_rows=dry_run.candidate_rows,
        output_path=candidate_output,
    )
    artifacts = write_msqa_stage51_candidate_adapter_visualizations(
        dry_run.report,
        tmp_path / "visuals",
    )

    rows = [
        json.loads(line)
        for line in candidate_output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == len(dry_run.candidate_rows)
    assert {artifact.name for artifact in artifacts} == {
        "stage61_adapter_candidate_counts.svg",
        "stage61_adapter_source_hit_rates.svg",
        "stage61_adapter_contract_checks.svg",
    }
    for artifact in artifacts:
        assert Path(artifact.path).read_text(encoding="utf-8").startswith("<svg")


def _write_split_jsonl(path: Path) -> None:
    rows = [
        {
            "dataset": "microsoft_msqa",
            "split": "msqa_stage57_project_eval_v1",
            "adapter_contract_version": "msqa_eval_adapter_v1",
            "question_id": "1",
            "answer_id": "a1",
            "question": "How do I reset Azure password?",
            "answer": "Reset the password in Azure portal. Confirm MFA.",
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
            "answer": "Configure Teams in the admin center. Review policies.",
            "source_url": "https://learn.microsoft.com/en-us/answers/questions/2/example.html",
            "metadata": {},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_protocol_report(
    path: Path,
    *,
    candidate_construction: str = "processed_answer_sentence_candidates",
) -> None:
    report = {
        "stage": "Stage 60",
        "recommended_protocol": {
            "source_citation_identity": "msqa_row_source_url",
            "candidate_construction": candidate_construction,
        },
        "decision": {
            "recommended_source_citation_identity": "msqa_row_source_url",
            "recommended_candidate_construction": candidate_construction,
            "requires_user_confirmation": True,
        },
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
