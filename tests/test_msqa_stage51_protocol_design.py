import json
from pathlib import Path

import pytest

from ts_rag_agent.application.msqa_stage51_protocol_design import (
    design_msqa_stage51_protocol,
    write_msqa_stage51_protocol_visualizations,
)


def test_msqa_stage51_protocol_design_recommends_row_sentence_protocol(tmp_path):
    schema_probe = tmp_path / "stage56.json"
    evaluation_split = tmp_path / "stage57.json"
    compatibility_review = tmp_path / "stage59.json"
    _write_stage56_report(schema_probe)
    _write_stage57_report(evaluation_split)
    _write_stage59_report(compatibility_review)

    report = design_msqa_stage51_protocol(
        schema_probe_report_path=schema_probe,
        evaluation_split_report_path=evaluation_split,
        compatibility_review_path=compatibility_review,
    )

    assert report["stage"] == "Stage 60"
    assert report["decision"]["requires_user_confirmation"] is True
    assert report["decision"]["can_run_stage51_candidate_now"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert (
        report["decision"]["recommended_source_citation_identity"]
        == "msqa_row_source_url"
    )
    assert (
        report["decision"]["recommended_candidate_construction"]
        == "processed_answer_sentence_candidates"
    )
    assert (
        report["recommended_protocol"]["protocol_status"]
        == "draft_requires_user_confirmation"
    )


def test_msqa_stage51_protocol_design_blocks_incomplete_link_identity(tmp_path):
    schema_probe = tmp_path / "stage56.json"
    evaluation_split = tmp_path / "stage57.json"
    compatibility_review = tmp_path / "stage59.json"
    _write_stage56_report(schema_probe)
    _write_stage57_report(evaluation_split)
    _write_stage59_report(compatibility_review)

    report = design_msqa_stage51_protocol(
        schema_probe_report_path=schema_probe,
        evaluation_split_report_path=evaluation_split,
        compatibility_review_path=compatibility_review,
    )

    source_options = {
        option["label"]: option
        for option in report["source_citation_identity_options"]
    }
    assert source_options["msqa_row_source_url"]["status"] == (
        "recommended_for_user_confirmation"
    )
    assert source_options["msqa_row_source_url"]["coverage_percent"] == 100.0
    assert source_options["processed_answer_links"]["status"] == "blocked"
    assert source_options["processed_answer_links"]["coverage_percent"] == 61.807
    assert source_options["question_answer_page_text"]["status"] == "rejected"


def test_msqa_stage51_protocol_design_visualizations_are_written(tmp_path):
    schema_probe = tmp_path / "stage56.json"
    evaluation_split = tmp_path / "stage57.json"
    compatibility_review = tmp_path / "stage59.json"
    _write_stage56_report(schema_probe)
    _write_stage57_report(evaluation_split)
    _write_stage59_report(compatibility_review)
    report = design_msqa_stage51_protocol(
        schema_probe_report_path=schema_probe,
        evaluation_split_report_path=evaluation_split,
        compatibility_review_path=compatibility_review,
    )

    artifacts = write_msqa_stage51_protocol_visualizations(
        report,
        tmp_path / "visuals",
    )

    assert {artifact.name for artifact in artifacts} == {
        "stage60_source_identity_scores.svg",
        "stage60_candidate_construction_scores.svg",
        "stage60_source_coverage.svg",
        "stage60_decision_flags.svg",
    }
    for artifact in artifacts:
        assert Path(artifact.path).read_text(encoding="utf-8").startswith("<svg")


def test_msqa_stage51_protocol_design_requires_stage59_block(tmp_path):
    schema_probe = tmp_path / "stage56.json"
    evaluation_split = tmp_path / "stage57.json"
    compatibility_review = tmp_path / "stage59.json"
    _write_stage56_report(schema_probe)
    _write_stage57_report(evaluation_split)
    _write_stage59_report(compatibility_review, can_run_stage51=True)

    with pytest.raises(ValueError, match="expects Stage 59 to block"):
        design_msqa_stage51_protocol(
            schema_probe_report_path=schema_probe,
            evaluation_split_report_path=evaluation_split,
            compatibility_review_path=compatibility_review,
        )


def _write_stage56_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "stage": "Stage 56",
                "source_link_coverage": {
                    "rows_with_row_url": {"count": 100, "percent": 100.0},
                    "rows_with_processed_answer_link": {
                        "count": 62,
                        "percent": 61.807,
                    },
                    "rows_with_processed_answer_learn_link": {
                        "count": 34,
                        "percent": 33.903,
                    },
                    "rows_with_processed_answer_azure_docish_link": {
                        "count": 13,
                        "percent": 13.473,
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_stage57_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "stage": "Stage 57",
                "adapter_contract": {
                    "contract_version": "msqa_eval_adapter_v1",
                    "answer_field": "ProcessedAnswerText",
                    "source_url_field": "Url",
                },
                "frozen_split": {
                    "split_name": "msqa_stage57_project_eval_v1",
                    "filter_counts": {
                        "selected_question_count": 3301,
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_stage59_report(path: Path, *, can_run_stage51: bool = False) -> None:
    path.write_text(
        json.dumps(
            {
                "stage": "Stage 59",
                "compatibility_gate": {
                    "summary": {
                        "blocker_count": 0 if can_run_stage51 else 5,
                    },
                },
                "decision": {
                    "can_run_stage51_candidate_now": can_run_stage51,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
