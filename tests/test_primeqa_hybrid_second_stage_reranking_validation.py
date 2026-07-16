from __future__ import annotations

import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_second_stage_reranking_protocol import (
    freeze_primeqa_hybrid_second_stage_reranking_protocol,
)
from ts_rag_agent.application.primeqa_hybrid_second_stage_reranking_validation import (
    run_primeqa_hybrid_second_stage_reranking_validation,
    write_primeqa_hybrid_second_stage_reranking_visualizations,
)


def test_second_stage_reranking_validation_runs_train_cv_dev_only(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_second_stage_reranking_validation(
        stage117_protocol_path=paths["stage117_protocol"],
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage80_report_path=None,
        user_confirmed_validation=True,
        confirmation_note="unit test confirmed Stage118 train/dev validation",
        include_dense_channels=False,
    )
    visualizations = write_primeqa_hybrid_second_stage_reranking_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    checks = {check["name"]: check for check in report["guard_checks"]}
    assert report["stage"] == "Stage 118"
    assert set(report["candidate_pool_summary"]) == {"dev", "train"}
    assert report["loaded_data_summary"]["test_split_loaded"] is False
    assert report["baseline_metrics"]["train_cv"]["hit_at_k"]["200"] == 1.0
    assert report["baseline_metrics"]["dev"]["hit_at_k"]["200"] == 1.0
    assert report["analysis_config"]["feature_aliases"] == {
        "bm25_top10_non_gold_indicator": "bm25_top10_indicator"
    }
    assert checks["stage118_protocol_feature_aliases_remove_gold_label_feature"][
        "passed"
    ] is True
    assert all(check["passed"] for check in report["guard_checks"])
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert report["public_safe_contract"]["forbidden_keys_found"] == []
    assert "Restart the service from the admin console" not in serialized
    assert "Install the storage driver package" not in serialized
    assert "Driver installation troubleshooting" not in serialized
    assert '"answer_doc_id":' not in serialized
    assert '"question_text":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage118_train_cv_objective_scores.svg",
        "stage118_train_cv_mrr_at_20_delta.svg",
        "stage118_train_cv_hit_at_10_delta.svg",
        "stage118_train_cv_guard_pass_counts.svg",
        "stage118_dev_selected_config_deltas.svg",
        "stage118_selection_decision_flags.svg",
        "stage118_guard_check_status.svg",
    }


def test_second_stage_reranking_validation_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_second_stage_reranking_validation(
        stage117_protocol_path=paths["stage117_protocol"],
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage80_report_path=None,
        user_confirmed_validation=False,
        confirmation_note="not confirmed",
        include_dense_channels=False,
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage118_validation"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_second_stage_reranking_validation_blocked"
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
    stage116_report = _write_stage116_report(tmp_path)
    stage117_protocol = tmp_path / "stage117.json"
    stage117_report = freeze_primeqa_hybrid_second_stage_reranking_protocol(
        stage116_report_path=stage116_report,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmed Stage117 protocol",
    )
    assert stage117_report["decision"]["status"] == (
        "primeqa_hybrid_second_stage_reranking_protocol_frozen"
    )
    stage117_protocol.write_text(
        json.dumps(stage117_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "train_split": train_split,
        "dev_split": dev_split,
        "documents": documents,
        "stage117_protocol": stage117_protocol,
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


def _write_stage116_report(tmp_path: Path) -> Path:
    report = {
        "stage": "Stage 116",
        "created_at": "2026-07-16",
        "analysis_id": "primeqa_hybrid_high_recall_union_candidate_pool_v1",
        "split_contract": {
            "split_name": "primeqa_hybrid_stage68_v1",
            "protocol_version": "primeqa_hybrid_split_v1",
            "development_splits": ["train", "dev"],
            "dev_selection_used": False,
            "dev_retuning_used": False,
            "forbidden_final_splits": ["test"],
        },
        "analysis_config": {
            "candidate_pool_id": "stage116_multi_route_union_candidate_pool",
            "channel_top_k": 200,
            "pool_top_k_values": [10, 20, 50, 100, 200],
            "rrf_k": 60,
        },
        "loaded_data_summary": {
            "test_split_loaded": False,
            "split_samples": {
                "train": {"answerable_count": 6},
                "dev": {"answerable_count": 2},
            },
        },
        "dense_channel_preflight": {
            "status": "dense_channels_ready",
            "can_run_without_download": True,
        },
        "channel_catalog": [{"channel_id": f"route_{index}"} for index in range(7)],
        "candidate_pool_metrics_by_split": {
            "train": {
                "hit_at_k": {"100": 1.0, "200": 1.0},
                "uncapped_union_hit_rate": 1.0,
                "candidate_pool_size": {"average": 7.0, "p95": 7.0},
            },
            "dev": {
                "hit_at_k": {"100": 1.0, "200": 1.0},
                "uncapped_union_hit_rate": 1.0,
                "candidate_pool_size": {"average": 7.0, "p95": 7.0},
            },
        },
        "comparisons_to_baseline": {
            "dev": {
                "hit@100": {"hit_count_delta": 1},
                "hit@200": {"hit_count_delta": 1},
            }
        },
        "guard_checks": [
            {"name": f"guard_{index}", "passed": True} for index in range(11)
        ],
        "decision": {
            "status": "primeqa_hybrid_high_recall_union_candidate_pool_completed",
            "recommended_next_direction": (
                "design_second_stage_precision_reranking_protocol_over_stage116_pool"
            ),
            "can_continue_second_stage_precision_experiment": True,
            "analysis_result": {
                "dev_uncapped_union_not_found_count": 0,
            },
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "public_safe_contract": {
            "forbidden_keys_found": [],
        },
    }
    path = tmp_path / "stage116.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


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
