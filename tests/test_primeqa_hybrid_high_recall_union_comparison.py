from __future__ import annotations

import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    run_primeqa_hybrid_high_recall_union_comparison,
    write_primeqa_hybrid_high_recall_union_visualizations,
)


def test_high_recall_union_runs_train_dev_only_and_public_safe(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_high_recall_union_comparison(
        stage115_report_path=paths["stage115_report"],
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        user_confirmed_direction=True,
        confirmation_note="unit test confirmed high-recall union direction",
        include_dense_channels=False,
        channel_top_k=2,
        pool_top_k_values=(1, 2),
        train_fold_count=2,
    )
    visualizations = write_primeqa_hybrid_high_recall_union_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 116"
    assert set(report["channel_metrics_by_split"]) == {"dev", "train"}
    assert set(report["candidate_pool_metrics_by_split"]) == {"dev", "train"}
    assert report["loaded_data_summary"]["test_split_loaded"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_high_recall_union_candidate_pool_completed"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert report["train_fold_stability"]["fold_count"] == 2
    assert report["public_safe_contract"]["forbidden_keys_found"] == []
    assert "Restart the service from the admin console" not in serialized
    assert "Install the storage driver package" not in serialized
    assert "Driver installation troubleshooting" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage116_dev_channel_hit_at_100.svg",
        "stage116_dev_union_recall_by_pool_depth.svg",
        "stage116_dev_union_delta_vs_baseline.svg",
        "stage116_dev_marginal_hits_by_channel.svg",
        "stage116_train_fold_union_hit_at_100.svg",
        "stage116_candidate_pool_size_summary.svg",
        "stage116_guard_check_status.svg",
    }


def test_high_recall_union_blocks_without_user_confirmation(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_high_recall_union_comparison(
        stage115_report_path=paths["stage115_report"],
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        user_confirmed_direction=False,
        confirmation_note="not confirmed",
        include_dense_channels=False,
        channel_top_k=2,
        pool_top_k_values=(1, 2),
        train_fold_count=2,
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage116_high_recall_union_direction"]["passed"] is False
    assert report["channel_metrics_by_split"] == {}
    assert report["decision"]["status"] == (
        "primeqa_hybrid_high_recall_union_candidate_pool_blocked"
    )


def test_high_recall_union_blocks_if_stage115_family_not_stopped(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path, stage115_status="wrong_status")

    report = run_primeqa_hybrid_high_recall_union_comparison(
        stage115_report_path=paths["stage115_report"],
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        user_confirmed_direction=True,
        confirmation_note="unit test confirmed high-recall union direction",
        include_dense_channels=False,
        channel_top_k=2,
        pool_top_k_values=(1, 2),
        train_fold_count=2,
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage115_retrieval_index_redesign_family_stopped"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_high_recall_union_candidate_pool_blocked"
    )


def _write_fixture(
    tmp_path: Path,
    *,
    stage115_status: str = "primeqa_hybrid_retrieval_index_redesign_family_stopped",
) -> dict[str, Path]:
    train_split = tmp_path / "train.jsonl"
    dev_split = tmp_path / "dev.jsonl"
    _write_jsonl(
        train_split,
        [
            _split_sample(
                sample_id="primeqa_train:TRAIN_Q001",
                assigned_split="train",
                question_title="Restart service",
                question_text="admin console restart",
                answer="Restart the service from the admin console.",
                answer_doc_id="doc-restart",
            ),
            _split_sample(
                sample_id="primeqa_train:TRAIN_Q002",
                assigned_split="train",
                question_title="Driver install",
                question_text="storage driver bundle",
                answer="Install the storage driver package.",
                answer_doc_id="doc-driver",
            ),
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
            )
        ],
    )
    documents = tmp_path / "documents.json"
    documents.write_text(
        json.dumps(
            {
                "doc-restart": {
                    "id": "doc-restart",
                    "title": "Restart service",
                    "text": "Restart support note.",
                    "sections": [
                        {
                            "id": "restart",
                            "text": "Restart the service from the admin console.",
                            "start": 0,
                            "end": 47,
                        }
                    ],
                },
                "doc-driver": {
                    "id": "doc-driver",
                    "title": "Storage driver install",
                    "text": "Use the storage driver bundle.",
                    "sections": [
                        {
                            "id": "driver",
                            "text": "Install the storage driver package.",
                            "start": 0,
                            "end": 35,
                        }
                    ],
                },
                "doc-decoy": {
                    "id": "doc-decoy",
                    "title": "Driver installation troubleshooting",
                    "text": "general installation troubleshooting",
                    "sections": [],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    stage115_report = tmp_path / "stage115.json"
    stage115_report.write_text(
        json.dumps(
            {
                "stage": "Stage 115",
                "decision": {
                    "status": stage115_status,
                    "stopped_family_id": "retrieval_index_redesign_candidate_family",
                    "recommended_next_direction": (
                        "user_confirmed_next_research_direction_required"
                    ),
                    "can_run_final_test_metrics_now": False,
                    "fallback_strategies_enabled": False,
                    "default_runtime_policy": "unchanged",
                },
                "stopped_family": {
                    "stage114_summary": {"selectable_config_count": 0}
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "train_split": train_split,
        "dev_split": dev_split,
        "documents": documents,
        "stage115_report": stage115_report,
    }


def _split_sample(
    *,
    sample_id: str,
    assigned_split: str,
    question_title: str,
    question_text: str,
    answer: str,
    answer_doc_id: str,
) -> dict:
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
