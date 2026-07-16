from __future__ import annotations

import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_fast_filter_screening_changed_case_review import (
    review_primeqa_hybrid_fast_filter_screening_changed_cases,
    write_primeqa_hybrid_fast_filter_screening_changed_case_visualizations,
)
from ts_rag_agent.application.primeqa_hybrid_fast_filter_screening_protocol import (
    freeze_primeqa_hybrid_fast_filter_screening_protocol,
)


def test_fast_filter_screening_changed_case_review_runs_public_safe_train_dev_only(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = review_primeqa_hybrid_fast_filter_screening_changed_cases(
        stage121_validation_path=paths["stage121_validation"],
        stage120_protocol_path=paths["stage120_protocol"],
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage80_report_path=None,
        user_confirmed_review=True,
        confirmation_note="unit test confirmed Stage122 changed-case review",
        include_dense_channels=False,
        sample_limit=5,
    )
    visualizations = (
        write_primeqa_hybrid_fast_filter_screening_changed_case_visualizations(
            report=report,
            output_dir=tmp_path / "visuals",
        )
    )

    serialized = json.dumps(report, ensure_ascii=False)
    checks = {check["name"]: check for check in report["guard_checks"]}
    assert report["stage"] == "Stage 122"
    assert report["analysis_id"] == (
        "primeqa_hybrid_fast_filter_screening_changed_case_review_v1"
    )
    assert all(check["passed"] for check in report["guard_checks"])
    assert checks["stage122_reviews_expected_configs"]["passed"] is True
    assert len(report["config_reviews"]) == 2
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert report["public_safe_contract"]["forbidden_keys_found"] == []
    assert "Restart the service from the admin console" not in serialized
    assert "Install the storage driver package" not in serialized
    assert "doc-restart" not in serialized
    assert '"sample_id":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert '"question_text":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage122_changed_case_outcomes.svg",
        "stage122_hit20_transitions.svg",
        "stage122_changed_case_rank_delta.svg",
        "stage122_guard_risk_summary.svg",
        "stage122_decision_flags.svg",
        "stage122_guard_check_status.svg",
    }


def test_fast_filter_screening_changed_case_review_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = review_primeqa_hybrid_fast_filter_screening_changed_cases(
        stage121_validation_path=paths["stage121_validation"],
        stage120_protocol_path=paths["stage120_protocol"],
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage80_report_path=None,
        user_confirmed_review=False,
        confirmation_note="not confirmed",
        include_dense_channels=False,
        sample_limit=5,
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage122_review"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_fast_filter_screening_changed_case_review_blocked"
    )


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    train_split = tmp_path / "train.jsonl"
    dev_split = tmp_path / "dev.jsonl"
    _write_jsonl(
        train_split,
        [
            _split_sample(
                sample_id=f"primeqa_train:TRAIN_Q{index:03d}",
                assigned_split="train",
                question_title=title,
                question_text=query,
                answer=answer,
                answer_doc_id=doc_id,
            )
            for index, (title, query, answer, doc_id) in enumerate(
                [
                    (
                        "Restart service",
                        "admin console restart",
                        "Restart the service from the admin console.",
                        "doc-restart",
                    ),
                    (
                        "Storage driver install",
                        "storage driver bundle",
                        "Install the storage driver package.",
                        "doc-driver",
                    ),
                    (
                        "GPU firmware update",
                        "update gpu firmware",
                        "Apply the GPU firmware update.",
                        "doc-gpu",
                    ),
                    (
                        "Network port reset",
                        "reset network port",
                        "Reset the network port.",
                        "doc-network",
                    ),
                    (
                        "Database backup",
                        "create database backup",
                        "Create a database backup.",
                        "doc-backup",
                    ),
                    (
                        "Cache clear",
                        "clear application cache",
                        "Clear the application cache.",
                        "doc-cache",
                    ),
                ],
                start=1,
            )
        ],
    )
    _write_jsonl(
        dev_split,
        [
            _split_sample(
                sample_id="primeqa_dev:DEV_Q001",
                assigned_split="dev",
                question_title="Storage driver install",
                question_text="bundle storage driver",
                answer="Install the storage driver package.",
                answer_doc_id="doc-driver",
            ),
            _split_sample(
                sample_id="primeqa_dev:DEV_Q002",
                assigned_split="dev",
                question_title="Restart service",
                question_text="restart admin console",
                answer="Restart the service from the admin console.",
                answer_doc_id="doc-restart",
            ),
        ],
    )
    documents = _write_documents(tmp_path)
    stage119_report = _write_stage119_report(tmp_path)
    stage120_protocol = tmp_path / "stage120.json"
    stage120_report = freeze_primeqa_hybrid_fast_filter_screening_protocol(
        stage119_report_path=stage119_report,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmed Stage120 protocol",
    )
    stage120_protocol.write_text(
        json.dumps(stage120_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stage121_validation = _write_stage121_validation(tmp_path)
    return {
        "train_split": train_split,
        "dev_split": dev_split,
        "documents": documents,
        "stage120_protocol": stage120_protocol,
        "stage121_validation": stage121_validation,
    }


def _write_documents(tmp_path: Path) -> Path:
    documents = tmp_path / "documents.json"
    rows = {
        "doc-restart": _document_row(
            title="Restart service",
            text="Restart the service from the admin console.",
            section_id="restart",
        ),
        "doc-driver": _document_row(
            title="Storage driver install",
            text="Install the storage driver package.",
            section_id="driver",
        ),
        "doc-gpu": _document_row(
            title="GPU firmware update",
            text="Apply the GPU firmware update.",
            section_id="gpu",
        ),
        "doc-network": _document_row(
            title="Network port reset",
            text="Reset the network port.",
            section_id="network",
        ),
        "doc-backup": _document_row(
            title="Database backup",
            text="Create a database backup.",
            section_id="backup",
        ),
        "doc-cache": _document_row(
            title="Cache clear",
            text="Clear the application cache.",
            section_id="cache",
        ),
        "doc-decoy": _document_row(
            title="Driver installation troubleshooting",
            text="general installation troubleshooting",
            section_id="decoy",
        ),
    }
    for doc_id, row in rows.items():
        row["id"] = doc_id
    documents.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return documents


def _document_row(*, title: str, text: str, section_id: str) -> dict[str, object]:
    return {
        "title": title,
        "text": text,
        "sections": [
            {
                "id": section_id,
                "text": text,
                "start": 0,
                "end": len(text),
            }
        ],
    }


def _write_stage119_report(tmp_path: Path) -> Path:
    report = {
        "stage": "Stage 119",
        "stopped_family": {
            "family_id": "second_stage_reranking_candidate_family",
            "source_protocol_id": "primeqa_hybrid_second_stage_reranking_protocol_v1",
            "source_analysis_id": (
                "primeqa_hybrid_second_stage_reranking_train_cv_dev_validation_v1"
            ),
            "stage118_summary": {
                "selectable_config_count": 0,
                "config_count": 8,
                "train_top200_gold_present_rate": 0.9324,
                "dev_top200_gold_present_rate": 0.9079,
                "train_candidate_record_count_in_memory": 74000,
                "dev_candidate_record_count_in_memory": 15200,
                "raw_candidate_rows_written": False,
            },
            "candidate_family_summary": {},
            "train_cv_positive_signal_but_blocked_configs": [],
            "dev_report_observations": {
                "dev_used_for_selection": False,
                "dev_used_for_retuning": False,
                "dev_observations_are_non_adoptable": True,
            },
        },
        "decision": {
            "status": "primeqa_hybrid_second_stage_reranking_family_stopped",
            "stopped_family_id": "second_stage_reranking_candidate_family",
            "recommended_next_direction": "user_confirmed_next_research_direction_required",
            "requires_user_confirmation_before_next_protocol": True,
            "can_continue_train_dev_development": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "public_safe_contract": {
            "forbidden_keys_found": [],
            "test_split_loaded": False,
            "final_test_metrics_run": False,
        },
    }
    path = tmp_path / "stage119.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_stage121_validation(tmp_path: Path) -> Path:
    report = {
        "stage": "Stage 121",
        "analysis_id": "primeqa_hybrid_fast_filter_screening_train_cv_dev_validation_v1",
        "train_cv_selection": {
            "selected_config_id": "special_token_exact_window40_rule_selector_v1",
            "selected_family_id": "evidence_density_fast_filter_family_v1",
            "guard_passed_config_count": 2,
            "selectable_config_count": 1,
            "selected_train_cv_comparison": {"hit@20_delta": 0.0027},
        },
        "dev_validation": {
            "dev_comparison_to_baseline": {"hit@20_delta": 0.0},
        },
        "config_results": [
            _stage121_config_result(
                "special_token_exact_window40_rule_selector_v1",
                selectable=True,
            ),
            _stage121_config_result(
                "top10_locked_route_vote_window50_pairwise_logistic_v1",
                selectable=False,
            ),
        ],
        "decision": {
            "status": (
                "primeqa_hybrid_fast_filter_screening_completed_"
                "train_cv_selected_dev_reported"
            ),
            "recommended_next_direction": "review_fast_filter_screening_changed_cases",
            "selected_config_id": "special_token_exact_window40_rule_selector_v1",
            "selected_family_id": "evidence_density_fast_filter_family_v1",
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
    }
    path = tmp_path / "stage121.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _stage121_config_result(config_id: str, *, selectable: bool) -> dict[str, object]:
    return {
        "config_id": config_id,
        "train_cv_selectable": selectable,
        "train_cv_guard_passed": selectable,
        "train_cv_objective_score": 0.0,
        "comparisons_to_baseline": {
            "train_cv": {"hit@20_delta": 0.0, "mrr_at_20_delta": 0.0},
            "dev": {"hit@20_delta": 0.0, "mrr_at_20_delta": 0.0},
        },
    }


def _split_sample(
    *,
    sample_id: str,
    assigned_split: str,
    question_title: str,
    question_text: str,
    answer: str,
    answer_doc_id: str,
) -> dict[str, object]:
    return {
        "dataset": "primeqa_techqa",
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "assigned_split": assigned_split,
        "split_subtype": f"group_random_{assigned_split}",
        "assignment_reason": "group_random_remainder_split",
        "source_split": f"primeqa_{assigned_split}",
        "source_row_index": 1,
        "sample_id": sample_id,
        "question_id": sample_id.split(":", maxsplit=1)[1],
        "question_title": question_title,
        "question_text": question_text,
        "question": f"{question_title}\n\n{question_text}",
        "answerable": True,
        "answer": answer,
        "answer_doc_id": answer_doc_id,
        "candidate_doc_ids": [answer_doc_id],
        "answer_span": {"start_offset": 0, "end_offset": len(answer)},
        "metadata": {
            "group_hash": f"{sample_id}:group",
            "candidate_doc_count": 1,
            "candidate_doc_hash": f"{sample_id}:docs",
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
