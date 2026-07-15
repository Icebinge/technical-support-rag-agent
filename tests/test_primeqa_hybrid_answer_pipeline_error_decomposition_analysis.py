import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_answer_pipeline_error_decomposition_analysis import (
    run_primeqa_hybrid_answer_pipeline_error_decomposition,
    write_primeqa_hybrid_answer_pipeline_decomposition_visualizations,
)


def test_answer_pipeline_error_decomposition_runs_train_dev_only(tmp_path):
    paths = _write_fixture_files(tmp_path)

    report = run_primeqa_hybrid_answer_pipeline_error_decomposition(
        stage101_protocol_path=paths["stage101"],
        train_split_path=paths["train"],
        dev_split_path=paths["dev"],
        documents_path=paths["documents"],
        user_confirmed_analysis=True,
        confirmation_note="unit test confirmation",
        retrieval_top_k=1,
        min_evidence_score=0.0,
        sample_limit_per_bucket=2,
    )
    visualizations = write_primeqa_hybrid_answer_pipeline_decomposition_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 102"
    assert report["decision"]["status"] == (
        "primeqa_hybrid_answer_pipeline_error_decomposition_completed"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["data_summary"]["splits"]["train"]["row_count"] == 3
    assert report["data_summary"]["splits"]["dev"]["row_count"] == 3
    assert report["aggregate_outputs"]["bucket_counts_by_split"]["train"][
        "answerability_false_answer"
    ] >= 1
    assert all(check["passed"] for check in report["guard_checks"])
    assert "private-doc-alpha" not in serialized
    assert "private-doc-beta" not in serialized
    assert "Private fixture answer text" not in serialized
    assert "question_text" not in serialized
    assert "answer_doc_id" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage102_bucket_counts_by_split.svg",
        "stage102_pipeline_stage_counts.svg",
        "stage102_answerability_bucket_counts.svg",
        "stage102_verified_metric_rates.svg",
        "stage102_public_case_sample_counts.svg",
        "stage102_guard_check_status.svg",
    }


def test_answer_pipeline_error_decomposition_blocks_without_confirmation(tmp_path):
    paths = _write_fixture_files(tmp_path)

    report = run_primeqa_hybrid_answer_pipeline_error_decomposition(
        stage101_protocol_path=paths["stage101"],
        train_split_path=paths["train"],
        dev_split_path=paths["dev"],
        documents_path=paths["documents"],
        user_confirmed_analysis=False,
        confirmation_note="not confirmed",
        retrieval_top_k=1,
        min_evidence_score=0.0,
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage102_analysis"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_answer_pipeline_error_decomposition_blocked"
    )


def test_answer_pipeline_error_decomposition_blocks_if_stage101_not_frozen(tmp_path):
    paths = _write_fixture_files(tmp_path, stage101_status="blocked")

    report = run_primeqa_hybrid_answer_pipeline_error_decomposition(
        stage101_protocol_path=paths["stage101"],
        train_split_path=paths["train"],
        dev_split_path=paths["dev"],
        documents_path=paths["documents"],
        user_confirmed_analysis=True,
        confirmation_note="unit test confirmation",
        retrieval_top_k=1,
        min_evidence_score=0.0,
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage101_protocol_is_frozen"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_answer_pipeline_error_decomposition_blocked"
    )


def test_answer_pipeline_error_decomposition_public_samples_match_contract(tmp_path):
    paths = _write_fixture_files(tmp_path)

    report = run_primeqa_hybrid_answer_pipeline_error_decomposition(
        stage101_protocol_path=paths["stage101"],
        train_split_path=paths["train"],
        dev_split_path=paths["dev"],
        documents_path=paths["documents"],
        user_confirmed_analysis=True,
        confirmation_note="unit test confirmation",
        retrieval_top_k=1,
        min_evidence_score=0.0,
    )

    expected_fields = [
        "sample_id",
        "split",
        "answerability_label",
        "pipeline_bucket_id",
        "pipeline_stage",
        "retrieval_rank_bucket",
        "retrieval_context_status",
        "citation_status",
        "evidence_selection_status",
        "answer_token_f1_bucket",
        "best_gold_span_f1_bucket",
        "answer_gold_span_gap_bucket",
        "verifier_decision",
        "refusal_reason_code",
        "question_route",
        "evidence_selector_name",
        "composition_policy_id",
        "bucket_confidence_band",
    ]
    for by_bucket in report["public_safe_case_samples"].values():
        for cases in by_bucket.values():
            for case in cases:
                assert list(case) == expected_fields


def _write_fixture_files(
    tmp_path: Path,
    *,
    stage101_status: str = (
        "primeqa_hybrid_answer_pipeline_error_decomposition_protocol_frozen"
    ),
) -> dict[str, Path]:
    paths = {
        "stage101": _write_json(
            tmp_path / "stage101.json",
            _stage101_protocol(status=stage101_status),
        ),
        "documents": _write_json(tmp_path / "documents.json", _documents()),
    }
    for split in ("train", "dev"):
        paths[split] = _write_jsonl(
            tmp_path / f"{split}.jsonl",
            [
                _sample(
                    split=split,
                    sample_id=f"{split}_s1",
                    question_text="How do I restart queue worker after timeout?",
                    answerable=True,
                    answer="Restart the queue worker service.",
                    answer_doc_id="private-doc-alpha",
                ),
                _sample(
                    split=split,
                    sample_id=f"{split}_s2",
                    question_text="How do I fix purple screen calibration?",
                    answerable=True,
                    answer="Private fixture answer text for the isolated document.",
                    answer_doc_id="private-doc-missing-from-query",
                ),
                _sample(
                    split=split,
                    sample_id=f"{split}_s3",
                    question_text="How do I restart queue worker for unknown edition?",
                    answerable=False,
                    answer="",
                    answer_doc_id=None,
                ),
            ],
        )
    return paths


def _stage101_protocol(*, status: str) -> dict:
    return {
        "stage": "Stage 101",
        "decision": {
            "status": status,
            "protocol_id": "answer_pipeline_error_decomposition_train_dev_v1",
            "recommended_direction": "answer_pipeline_error_decomposition",
            "can_run_train_dev_error_decomposition_after_user_confirmation": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "frozen_protocol": {
            "public_safe_output_contract": {
                "case_sample_fields": [
                    "sample_id",
                    "split",
                    "answerability_label",
                    "pipeline_bucket_id",
                    "pipeline_stage",
                    "retrieval_rank_bucket",
                    "retrieval_context_status",
                    "citation_status",
                    "evidence_selection_status",
                    "answer_token_f1_bucket",
                    "best_gold_span_f1_bucket",
                    "answer_gold_span_gap_bucket",
                    "verifier_decision",
                    "refusal_reason_code",
                    "question_route",
                    "evidence_selector_name",
                    "composition_policy_id",
                    "bucket_confidence_band",
                ]
            }
        },
    }


def _documents() -> dict:
    return {
        "private-doc-alpha": {
            "id": "private-doc-alpha",
            "title": "Queue worker restart",
            "text": "Resolution: Restart the queue worker service. Verify the worker logs.",
        },
        "private-doc-beta": {
            "id": "private-doc-beta",
            "title": "Queue worker timeout",
            "text": "Queue worker timeout unknown edition restart troubleshooting.",
        },
        "private-doc-missing-from-query": {
            "id": "private-doc-missing-from-query",
            "title": "Isolated private fixture",
            "text": "Private fixture answer text for the isolated document.",
        },
    }


def _sample(
    *,
    split: str,
    sample_id: str,
    question_text: str,
    answerable: bool,
    answer: str,
    answer_doc_id: str | None,
) -> dict:
    return {
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "assigned_split": split,
        "split_subtype": "random_grouped",
        "source_split": "fixture",
        "sample_id": sample_id,
        "question_id": f"source_{sample_id}",
        "question_title": "Fixture title",
        "question_text": question_text,
        "answerable": answerable,
        "answer": answer,
        "answer_doc_id": answer_doc_id,
        "candidate_doc_ids": tuple(
            doc_id
            for doc_id in (
                "private-doc-alpha",
                "private-doc-beta",
                "private-doc-missing-from-query",
            )
            if doc_id
        ),
        "answer_span": {
            "start_offset": 0 if answerable else None,
            "end_offset": len(answer) if answerable else None,
        },
    }


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    return path
