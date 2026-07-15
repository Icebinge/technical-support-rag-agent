import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_section_signal_comparison import (
    run_primeqa_hybrid_section_signal_comparison,
    write_primeqa_hybrid_section_signal_comparison_visualizations,
)


def test_section_signal_comparison_runs_train_dev_only_and_public_safe(tmp_path):
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_section_signal_comparison(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage91_report_path=paths["stage91_report"],
        user_confirmed_protocol=True,
        confirmed_protocol_id="section_signal_guarded_expansion_train_dev_v1",
        confirmation_note="confirmed in test",
        top_k_values=(1, 5, 10),
        search_depth=10,
    )
    visualizations = write_primeqa_hybrid_section_signal_comparison_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 92"
    assert set(report["metrics_by_split"]) == {"dev", "train"}
    assert report["decision"]["status"] == (
        "primeqa_hybrid_section_signal_comparison_completed"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Restart the service from the admin console" not in serialized
    assert "Install the storage driver package" not in serialized
    assert "Driver installation troubleshooting" not in serialized
    assert "private section signal protocol string" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage92_section_signal_train_hit_at_10.svg",
        "stage92_section_signal_dev_hit_at_10.svg",
        "stage92_section_signal_dev_delta_hit_at_10.svg",
        "stage92_section_signal_dev_search_depth_net.svg",
        "stage92_section_signal_dev_top10_changes.svg",
        "stage92_section_signal_guard_check_status.svg",
    }


def test_section_signal_comparison_blocks_unconfirmed_protocol(tmp_path):
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_section_signal_comparison(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage91_report_path=paths["stage91_report"],
        user_confirmed_protocol=False,
        confirmed_protocol_id="section_signal_guarded_expansion_train_dev_v1",
        confirmation_note="not confirmed",
        top_k_values=(1, 5, 10),
        search_depth=10,
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_frozen_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_section_signal_comparison_blocked"
    )


def test_section_signal_comparison_blocks_stage75_baseline_mismatch(tmp_path):
    paths = _write_fixture(tmp_path)
    stage75 = json.loads(paths["stage75_report"].read_text(encoding="utf-8"))
    stage75["split_reports"]["train"]["hit_at_top_k"] = 0.0
    paths["stage75_report"].write_text(
        json.dumps(stage75, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = run_primeqa_hybrid_section_signal_comparison(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage91_report_path=paths["stage91_report"],
        user_confirmed_protocol=True,
        confirmed_protocol_id="section_signal_guarded_expansion_train_dev_v1",
        confirmation_note="confirmed in test",
        top_k_values=(1, 5, 10),
        search_depth=10,
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["baseline_train_hit10_matches_stage75"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_section_signal_comparison_blocked"
    )


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
    stage91_report = tmp_path / "stage91.json"
    stage91_report.write_text(
        json.dumps(_stage91_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "train_split": train_split,
        "dev_split": dev_split,
        "documents": documents,
        "stage75_report": stage75_report,
        "stage91_report": stage91_report,
    }


def _stage91_report() -> dict:
    return {
        "stage": "Stage 91",
        "frozen_protocol": {
            "protocol_id": "section_signal_guarded_expansion_train_dev_v1",
            "candidate_id": "section_signal_guarded_expansion_design",
            "candidate_config_grid": [
                {
                    "config_id": "ssgx_shadow_no_top10_demotion_v1",
                    "promotion_mode": "shadow_after_top10",
                    "eligible_baseline_rank_min": 11,
                    "eligible_baseline_rank_max": 50,
                    "section_rank_max": 50,
                    "minimum_section_to_document_score_ratio": 1.1,
                    "maximum_document_score_margin_to_rank10": None,
                    "maximum_top10_promotions_per_query": 0,
                    "protected_bm25_top_rank_count": 10,
                    "demote_existing_bm25_top10": False,
                },
                {
                    "config_id": "ssgx_rank11_20_margin_guard_v1",
                    "promotion_mode": "single_rank10_promotion",
                    "eligible_baseline_rank_min": 11,
                    "eligible_baseline_rank_max": 20,
                    "section_rank_max": 30,
                    "minimum_section_to_document_score_ratio": 1.2,
                    "maximum_document_score_margin_to_rank10": 0.08,
                    "maximum_top10_promotions_per_query": 1,
                    "protected_bm25_top_rank_count": 5,
                    "demote_existing_bm25_top10": True,
                },
                {
                    "config_id": "ssgx_rank21_50_high_confidence_v1",
                    "promotion_mode": "single_rank10_promotion",
                    "eligible_baseline_rank_min": 21,
                    "eligible_baseline_rank_max": 50,
                    "section_rank_max": 20,
                    "minimum_section_to_document_score_ratio": 1.45,
                    "maximum_document_score_margin_to_rank10": 0.05,
                    "maximum_top10_promotions_per_query": 1,
                    "protected_bm25_top_rank_count": 8,
                    "demote_existing_bm25_top10": True,
                },
                {
                    "config_id": "ssgx_section_top50_injection_guard_v1",
                    "promotion_mode": "single_rank10_section_candidate_injection",
                    "eligible_baseline_rank_min": 51,
                    "eligible_baseline_rank_max": None,
                    "section_rank_max": 15,
                    "minimum_section_to_document_score_ratio": 1.6,
                    "maximum_document_score_margin_to_rank10": None,
                    "maximum_top10_promotions_per_query": 1,
                    "protected_bm25_top_rank_count": 8,
                    "demote_existing_bm25_top10": True,
                },
            ],
            "public_safe_changed_case_fields": [
                "sample_id",
                "split",
                "baseline_rank",
                "challenger_rank",
                "config_id",
                "section_signal_bucket",
                "baseline_rank_bucket",
                "section_rank_bucket",
                "score_ratio_bucket",
                "score_margin_bucket",
                "promotion_reason_code",
                "top10_protection_action",
            ],
        },
        "decision": {
            "status": "primeqa_hybrid_section_signal_protocol_frozen",
            "can_run_train_dev_metrics_after_user_confirmation": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        },
        "raw_question_text": "private section signal protocol string",
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
