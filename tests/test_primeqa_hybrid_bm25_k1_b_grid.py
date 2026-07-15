import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_bm25_k1_b_grid import (
    run_primeqa_hybrid_bm25_k1_b_grid,
    write_primeqa_hybrid_bm25_k1_b_grid_visualizations,
)


def test_bm25_k1_b_grid_runs_train_dev_only_and_public_safe(tmp_path):
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_bm25_k1_b_grid(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage76_report_path=paths["stage76_report"],
        stage81_report_path=paths["stage81_report"],
        top_k_values=(1, 5, 10),
        search_depth=10,
    )
    visualizations = write_primeqa_hybrid_bm25_k1_b_grid_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 82"
    assert set(report["metrics_by_split"]) == {"dev", "train"}
    assert len(report["grid_configs"]) == 9
    assert "full_document_bm25_baseline" in report["metrics_by_split"]["dev"]
    assert report["train_selection"]["selected_config_id"] in report["metrics_by_split"][
        "train"
    ]
    assert all(check["passed"] for check in report["guard_checks"])
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["loaded_data_summary"]["test_split_loaded"] is False
    assert "Restart the service from the admin console" not in serialized
    assert "Install the storage driver package" not in serialized
    assert "Driver installation troubleshooting" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage82_bm25_grid_train_hit_at_10.svg",
        "stage82_bm25_grid_dev_hit_at_10.svg",
        "stage82_bm25_grid_dev_delta_hit_at_10.svg",
        "stage82_bm25_grid_dev_near_miss_11_to_50.svg",
        "stage82_bm25_grid_dev_top10_changes.svg",
    }


def test_bm25_k1_b_grid_blocks_unconfirmed_protocol(tmp_path):
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_bm25_k1_b_grid(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage76_report_path=paths["stage76_report"],
        stage81_report_path=paths["stage81_report"],
        user_confirmed_grid_protocol="medium_grid_unconfirmed",
        top_k_values=(1, 5, 10),
        search_depth=10,
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_small_grid_protocol"]["passed"] is False
    assert report["decision"]["status"] == "primeqa_hybrid_bm25_k1_b_grid_blocked"
    assert report["metrics_by_split"] == {}


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
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
                    "text": "Restart the service from the admin console.",
                    "sections": [],
                },
                "doc-driver": {
                    "id": "doc-driver",
                    "title": "Storage driver install",
                    "text": "Use the storage driver bundle.",
                    "sections": [],
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
    stage75_report = tmp_path / "stage75.json"
    stage75_report.write_text(
        json.dumps(
            {
                "stage": "Stage 75",
                "split_reports": {
                    "train": {"hit_at_top_k": 1.0},
                    "dev": {"hit_at_top_k": 1.0},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    stage76_report = tmp_path / "stage76.json"
    stage76_report.write_text(
        json.dumps(
            {
                "stage": "Stage 76",
                "candidate_designs": [
                    {
                        "candidate_id": "bm25_k1_b_grid_train_to_dev",
                        "status": "recommended_for_train_dev_experiment",
                        "target_metric_contract": [
                            "guard: grid values must be fixed before the run"
                        ],
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    stage81_report = tmp_path / "stage81.json"
    stage81_report.write_text(
        json.dumps(
            {
                "stage": "Stage 81",
                "decision": {
                    "can_open_final_test_gate_now": False,
                    "recommended_next_stage": (
                        "Stage 82: move to the remaining Stage76 "
                        "bm25_k1_b_grid_train_to_dev candidate on train/dev"
                    ),
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
        "stage75_report": stage75_report,
        "stage76_report": stage76_report,
        "stage81_report": stage81_report,
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
