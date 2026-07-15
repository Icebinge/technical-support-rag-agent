import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_retrieval_context_miss_root_cause_audit import (
    run_primeqa_hybrid_retrieval_context_miss_root_cause_audit,
    write_primeqa_hybrid_retrieval_context_miss_root_cause_audit_visualizations,
)


def test_retrieval_context_miss_root_cause_audit_runs_train_dev_only(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_retrieval_context_miss_root_cause_audit(
        stage111_protocol_path=paths["stage111"],
        train_split_path=paths["train"],
        dev_split_path=paths["dev"],
        documents_path=paths["documents"],
        user_confirmed_audit=True,
        confirmation_note="unit test confirmation",
        sample_limit_per_bucket=2,
    )
    visualizations = (
        write_primeqa_hybrid_retrieval_context_miss_root_cause_audit_visualizations(
            report=report,
            output_dir=tmp_path / "visuals",
        )
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 112"
    assert report["analysis_id"] == (
        "primeqa_hybrid_retrieval_context_miss_root_cause_audit_v1"
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_retrieval_context_miss_root_cause_audit_completed"
    )
    assert report["split_reports"]["train"]["audit_case_count"] == 1
    assert report["split_reports"]["dev"]["audit_case_count"] == 1
    assert report["loaded_data_summary"]["test_split_loaded"] is False
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert '"answer_doc_id":' not in serialized
    assert '"question_text":' not in serialized
    for split_samples in report["public_case_samples"].values():
        for cases in split_samples.values():
            for case in cases:
                assert tuple(case) == (
                    "sample_id",
                    "split",
                    "retrieval_context_miss_root_cause_bucket",
                    "question_route",
                    "gold_doc_rank_bucket",
                    "query_expression_gap_bucket",
                    "title_heading_overlap_bucket",
                    "section_locality_bucket",
                    "document_length_bucket",
                    "entity_version_error_code_bucket",
                    "index_structure_signal_bucket",
                    "confidence_band",
                )
    assert {artifact.name for artifact in visualizations} == {
        "stage112_audit_case_counts_by_split.svg",
        "stage112_primary_root_cause_counts.svg",
        "stage112_dimension_high_signal_counts.svg",
        "stage112_gold_rank_bucket_counts.svg",
        "stage112_question_route_counts.svg",
        "stage112_guard_check_status.svg",
    }


def test_retrieval_context_miss_root_cause_audit_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_retrieval_context_miss_root_cause_audit(
        stage111_protocol_path=paths["stage111"],
        train_split_path=paths["train"],
        dev_split_path=paths["dev"],
        documents_path=paths["documents"],
        user_confirmed_audit=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage112_audit"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_retrieval_context_miss_root_cause_audit_blocked"
    )


def test_retrieval_context_miss_root_cause_audit_blocks_on_count_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path, expected_train_misses=2)

    report = run_primeqa_hybrid_retrieval_context_miss_root_cause_audit(
        stage111_protocol_path=paths["stage111"],
        train_split_path=paths["train"],
        dev_split_path=paths["dev"],
        documents_path=paths["documents"],
        user_confirmed_audit=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["audit_case_counts_match_stage102_retrieval_context_miss"][
        "passed"
    ] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_retrieval_context_miss_root_cause_audit_blocked"
    )


def _write_fixture(
    tmp_path: Path,
    *,
    expected_train_misses: int = 1,
) -> dict[str, Path]:
    documents = {
        "train_top": _document(
            title="alpha error details fix",
            text="alpha error details fix troubleshooting guide",
        ),
        "train_gold_miss": _document(
            title="legacy note",
            text="Hidden value appears here for the supported procedure.",
        ),
        "train_gold_hit": _document(
            title="configure beta portal",
            text="configure beta portal with the supported answer.",
        ),
        "dev_top": _document(
            title="gamma install failure logs",
            text="gamma install failure logs contain the visible message.",
        ),
        "dev_gold_miss": _document(
            title="maintenance reference",
            text="The maintenance reference contains the required answer.",
        ),
    }
    paths = {
        "stage111": _write_json(
            tmp_path / "stage111.json",
            _stage111_report(expected_train_misses=expected_train_misses),
        ),
        "train": tmp_path / "train.jsonl",
        "dev": tmp_path / "dev.jsonl",
        "documents": _write_json(tmp_path / "documents.json", documents),
    }
    _write_jsonl(
        paths["train"],
        [
            _sample(
                sample_id="train-miss",
                split="train",
                question_title="alpha error details",
                question_text="Where are alpha error details?",
                answer_doc_id="train_gold_miss",
                answer="Hidden value appears here",
            ),
            _sample(
                sample_id="train-hit",
                split="train",
                question_title="configure beta portal",
                question_text="How do I configure beta portal?",
                answer_doc_id="train_gold_hit",
                answer="configure beta portal",
            ),
        ],
    )
    _write_jsonl(
        paths["dev"],
        [
            _sample(
                sample_id="dev-miss",
                split="dev",
                question_title="gamma install failure logs",
                question_text="Where are gamma install failure logs?",
                answer_doc_id="dev_gold_miss",
                answer="required answer",
            )
        ],
    )
    return paths


def _stage111_report(*, expected_train_misses: int) -> dict:
    return {
        "stage": "Stage 111",
        "protocol_id": "primeqa_hybrid_retrieval_context_miss_audit_protocol_v1",
        "stage102_summary": {
            "train_retrieval_context_miss_count": expected_train_misses,
            "dev_retrieval_context_miss_count": 1,
        },
        "frozen_protocol": {
            "audit_dimensions": [
                {"dimension_id": "query_expression_gap"},
                {"dimension_id": "title_heading_mismatch"},
                {"dimension_id": "section_boundary_or_span_locality"},
                {"dimension_id": "long_document_score_dilution"},
                {"dimension_id": "entity_version_error_code_mismatch"},
                {"dimension_id": "bm25_field_weighting_or_index_structure"},
            ],
            "stage112_run_contract": {
                "retrieval_depth_for_diagnostic_only": 50,
                "stage112_may_use_gold_doc_id_for_offline_labeling": True,
                "gold_doc_id_allowed_as_runtime_feature": False,
                "reported_splits": ["train", "dev"],
                "selection_or_threshold_tuning_allowed": False,
                "candidate_defaultization_allowed": False,
                "final_test_metrics_allowed": False,
            },
        },
        "decision": {
            "status": "primeqa_hybrid_retrieval_context_miss_audit_protocol_frozen",
            "recommended_next_direction": (
                "run_retrieval_context_miss_root_cause_audit_train_dev"
            ),
            "can_run_train_dev_audit_after_user_confirmation": True,
            "requires_user_confirmation_before_train_dev_audit": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
    }


def _document(*, title: str, text: str) -> dict:
    return {
        "id": "",
        "title": title,
        "text": text,
        "sections": [
            {
                "id": "DETAILS",
                "text": text,
                "start": 0,
                "end": len(text),
            }
        ],
    }


def _sample(
    *,
    sample_id: str,
    split: str,
    question_title: str,
    question_text: str,
    answer_doc_id: str,
    answer: str,
) -> dict:
    return {
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "assigned_split": split,
        "split_subtype": "fixture",
        "source_split": "primeqa_train",
        "sample_id": sample_id,
        "question_id": sample_id,
        "question_title": question_title,
        "question_text": question_text,
        "answerable": True,
        "answer": answer,
        "answer_doc_id": answer_doc_id,
        "candidate_doc_ids": [answer_doc_id],
        "answer_span": {"start_offset": 0, "end_offset": max(1, len(answer))},
    }


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path
