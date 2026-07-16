from __future__ import annotations

import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_retrieval_index_redesign_comparison import (
    run_primeqa_hybrid_retrieval_index_redesign_comparison,
    write_primeqa_hybrid_retrieval_index_redesign_comparison_visualizations,
)


def test_stage114_runs_frozen_configs_without_test_or_raw_public_fields(
    tmp_path: Path,
) -> None:
    protocol_path = tmp_path / "stage113.json"
    stage102_path = tmp_path / "stage102.json"
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    documents_path = tmp_path / "documents.sections.json"

    protocol_path.write_text(
        json.dumps(_stage113_protocol(), ensure_ascii=False),
        encoding="utf-8",
    )
    stage102_path.write_text(
        json.dumps({"stage": "Stage 102", "analysis_id": "stage102"}),
        encoding="utf-8",
    )
    documents_path.write_text(
        json.dumps(_documents(), ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(train_path, [_sample(index, "train") for index in range(1, 6)])
    _write_jsonl(dev_path, [_sample(index, "dev") for index in range(6, 8)])

    report = run_primeqa_hybrid_retrieval_index_redesign_comparison(
        stage113_protocol_path=protocol_path,
        stage102_report_path=stage102_path,
        train_split_path=train_path,
        dev_split_path=dev_path,
        documents_path=documents_path,
        user_confirmed_comparison=True,
        confirmation_note="test confirmation",
        component_depth=5,
    )

    assert report["stage"] == "Stage 114"
    assert report["split_contract"]["forbidden_final_splits"] == ["test"]
    assert report["split_contract"]["dev_gate_status"] == (
        "report_only_no_frozen_pass_threshold"
    )
    assert len(report["config_results"]) == 8
    assert report["train_cv_selection"]["selection_source"] == "train_cv_only"
    assert report["dev_validation"]["dev_validation_passed"] is None
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert not _contains_forbidden_key(
        {
            "baseline": report["baseline_result"],
            "configs": report["config_results"],
            "selection": report["train_cv_selection"],
            "dev": report["dev_validation"],
        }
    )


def test_stage114_writes_visualizations(tmp_path: Path) -> None:
    protocol_path = tmp_path / "stage113.json"
    stage102_path = tmp_path / "stage102.json"
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    documents_path = tmp_path / "documents.sections.json"
    visualization_dir = tmp_path / "visuals"

    protocol_path.write_text(
        json.dumps(_stage113_protocol(), ensure_ascii=False),
        encoding="utf-8",
    )
    stage102_path.write_text(
        json.dumps({"stage": "Stage 102", "analysis_id": "stage102"}),
        encoding="utf-8",
    )
    documents_path.write_text(
        json.dumps(_documents(), ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(train_path, [_sample(index, "train") for index in range(1, 6)])
    _write_jsonl(dev_path, [_sample(index, "dev") for index in range(6, 8)])

    report = run_primeqa_hybrid_retrieval_index_redesign_comparison(
        stage113_protocol_path=protocol_path,
        stage102_report_path=stage102_path,
        train_split_path=train_path,
        dev_split_path=dev_path,
        documents_path=documents_path,
        user_confirmed_comparison=True,
        confirmation_note="test confirmation",
        component_depth=5,
    )
    artifacts = write_primeqa_hybrid_retrieval_index_redesign_comparison_visualizations(
        report=report,
        output_dir=visualization_dir,
    )

    assert len(artifacts) == 8
    assert all(Path(artifact.path).exists() for artifact in artifacts)
    assert all(
        Path(artifact.path).read_text(encoding="utf-8").startswith("<svg")
        for artifact in artifacts
    )


def _stage113_protocol() -> dict:
    return {
        "stage": "Stage 113",
        "protocol_id": "primeqa_hybrid_retrieval_index_redesign_protocol_v1",
        "frozen_protocol": {
            "protocol_status": "frozen_requires_user_confirmation_before_train_dev_run",
            "candidate_configs": [
                {
                    "config_id": "thw_title2_heading2_body1_doc_bm25_v1",
                    "family_id": "title_heading_weighted_bm25_candidate_v1",
                    "retrieval_mode": "weighted_document_bm25",
                    "weights": {"title": 2.0, "section_heading": 2.0, "body": 1.0},
                    "selection_eligible": True,
                },
                {
                    "config_id": "thw_title3_heading2_body1_doc_bm25_v1",
                    "family_id": "title_heading_weighted_bm25_candidate_v1",
                    "retrieval_mode": "weighted_document_bm25",
                    "weights": {"title": 3.0, "section_heading": 2.0, "body": 1.0},
                    "selection_eligible": True,
                },
                {
                    "config_id": "thw_title_heading_query_view_rrf_v1",
                    "family_id": "title_heading_weighted_bm25_candidate_v1",
                    "retrieval_mode": "document_bm25_rrf",
                    "rrf_k": 60,
                    "selection_eligible": True,
                },
                {
                    "config_id": "slr_section_top1_doc_rollup_v1",
                    "family_id": "section_level_index_rollup_candidate_v1",
                    "retrieval_mode": "section_bm25_document_rollup",
                    "selection_eligible": True,
                },
                {
                    "config_id": "slr_section_top3_rrf_doc_rollup_v1",
                    "family_id": "section_level_index_rollup_candidate_v1",
                    "retrieval_mode": "section_document_rrf",
                    "rrf_k": 60,
                    "selection_eligible": True,
                },
                {
                    "config_id": "slr_heading_section_title_rollup_v1",
                    "family_id": "section_level_index_rollup_candidate_v1",
                    "retrieval_mode": "heading_section_title_rollup",
                    "section_heading_weight": 2.0,
                    "document_title_weight": 2.0,
                    "selection_eligible": True,
                },
                {
                    "config_id": "evc_special_token_exact_boost_v1",
                    "family_id": "entity_version_error_code_handling_candidate_v1",
                    "retrieval_mode": "bm25_with_runtime_special_token_boost",
                    "special_token_boost": 1.5,
                    "selection_eligible": True,
                },
                {
                    "config_id": "evc_special_token_title_heading_boost_v1",
                    "family_id": "entity_version_error_code_handling_candidate_v1",
                    "retrieval_mode": "weighted_bm25_with_special_token_boost",
                    "title_weight": 2.0,
                    "heading_weight": 2.0,
                    "special_token_boost": 1.5,
                    "selection_eligible": True,
                },
            ],
            "selection_rules": {
                "minimum_train_folds": 5,
                "guard_thresholds": {
                    "max_train_cv_average_token_f1_drop": 0.005,
                    "max_train_cv_gold_doc_citation_rate_drop": 0.015,
                    "max_train_cv_answerable_refusal_rate_delta": 0.02,
                    "max_train_cv_answerability_false_answer_delta": 0,
                    "max_train_cv_evidence_selection_miss_delta": 0,
                    "max_train_cv_gold_span_beats_selected_delta": 0,
                    "max_train_cv_changed_answer_rate": 0.25,
                },
                "dev_rules": {
                    "dev_selection_allowed": False,
                    "dev_retuning_allowed": False,
                    "dev_threshold_tuning_allowed": False,
                    "dev_report_required": True,
                },
                "runtime_rules": {
                    "default_runtime_policy": "unchanged",
                    "fallback_strategies_enabled": False,
                },
            },
        },
        "decision": {
            "recommended_next_direction": (
                "run_retrieval_index_redesign_train_cv_dev_validation"
            ),
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
    }


def _documents() -> dict:
    return {
        "D1": {
            "id": "D1",
            "title": "ZX-900 APAR reset failure",
            "text": "Install fix pack 7. The ZX-900 APAR reset failure is fixed.",
            "sections": [
                {
                    "id": "Reset failure",
                    "text": "Install fix pack 7 for the ZX-900 APAR reset failure.",
                    "start": 0,
                    "end": 64,
                }
            ],
        },
        "D2": {
            "id": "D2",
            "title": "General logging configuration",
            "text": "Logging configuration uses verbose tracing and rotation.",
            "sections": [
                {
                    "id": "Logging",
                    "text": "Use verbose tracing for diagnostic logs.",
                    "start": 0,
                    "end": 42,
                }
            ],
        },
        "D3": {
            "id": "D3",
            "title": "TLS certificate renewal",
            "text": "Renew the TLS certificate before it expires.",
            "sections": [
                {
                    "id": "Certificates",
                    "text": "Renew the TLS certificate before it expires.",
                    "start": 0,
                    "end": 43,
                }
            ],
        },
    }


def _sample(index: int, split: str) -> dict:
    return {
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "assigned_split": split,
        "split_subtype": "random_grouped",
        "source_split": "training",
        "sample_id": f"{split}_{index}",
        "question_id": f"q{index}",
        "question_title": "ZX-900 APAR reset failure",
        "question_text": f"How do I fix reset failure scenario {index}?",
        "answerable": True,
        "answer": "Install fix pack 7.",
        "answer_doc_id": "D1",
        "candidate_doc_ids": ["D1", "D2", "D3"],
        "answer_span": {"start_offset": 0, "end_offset": 19},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


def _contains_forbidden_key(value: object) -> bool:
    forbidden = {
        "question_text",
        "question_title",
        "answer_text",
        "document_id",
        "answer_doc_id",
        "retrieved_doc_ids",
        "cited_doc_ids",
        "source_doc_ids",
        "query_terms",
        "document_title",
        "document_body",
        "document_text",
    }
    if isinstance(value, dict):
        return any(
            str(key) in forbidden or _contains_forbidden_key(nested)
            for key, nested in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_forbidden_key(nested) for nested in value)
    return False
