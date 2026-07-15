import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_score_margin_bm25_comparison import (
    run_primeqa_hybrid_score_margin_bm25_comparison,
    write_primeqa_hybrid_score_margin_bm25_comparison_visualizations,
)


def test_score_margin_bm25_comparison_runs_train_dev_only_and_public_safe(tmp_path):
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_score_margin_bm25_comparison(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage94_report_path=paths["stage94_report"],
        user_confirmed_protocol=True,
        confirmed_protocol_id="score_margin_bm25_normalization_gate_train_dev_v1",
        confirmation_note="confirmed in test",
        top_k_values=(1, 5, 10),
        search_depth=10,
    )
    visualizations = write_primeqa_hybrid_score_margin_bm25_comparison_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 95"
    assert set(report["metrics_by_split"]) == {"dev", "train"}
    assert report["decision"]["status"] == (
        "primeqa_hybrid_score_margin_bm25_comparison_completed"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Restart the service from the admin console" not in serialized
    assert "Install the storage driver package" not in serialized
    assert "Driver installation troubleshooting" not in serialized
    assert "private score-margin protocol string" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage95_score_margin_bm25_train_hit_at_10.svg",
        "stage95_score_margin_bm25_dev_hit_at_10.svg",
        "stage95_score_margin_bm25_dev_delta_hit_at_10.svg",
        "stage95_score_margin_bm25_dev_rank_11_to_50_delta.svg",
        "stage95_score_margin_bm25_dev_top10_changes.svg",
        "stage95_score_margin_bm25_dev_gate_actions.svg",
        "stage95_score_margin_bm25_guard_check_status.svg",
    }


def test_score_margin_bm25_comparison_blocks_unconfirmed_protocol(tmp_path):
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_score_margin_bm25_comparison(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage94_report_path=paths["stage94_report"],
        user_confirmed_protocol=False,
        confirmed_protocol_id="score_margin_bm25_normalization_gate_train_dev_v1",
        confirmation_note="not confirmed",
        top_k_values=(1, 5, 10),
        search_depth=10,
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_frozen_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_score_margin_bm25_comparison_blocked"
    )


def test_score_margin_bm25_comparison_blocks_stage75_baseline_mismatch(tmp_path):
    paths = _write_fixture(tmp_path)
    stage75 = json.loads(paths["stage75_report"].read_text(encoding="utf-8"))
    stage75["split_reports"]["train"]["hit_at_top_k"] = 0.0
    paths["stage75_report"].write_text(
        json.dumps(stage75, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = run_primeqa_hybrid_score_margin_bm25_comparison(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage94_report_path=paths["stage94_report"],
        user_confirmed_protocol=True,
        confirmed_protocol_id="score_margin_bm25_normalization_gate_train_dev_v1",
        confirmation_note="confirmed in test",
        top_k_values=(1, 5, 10),
        search_depth=10,
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["baseline_train_hit10_matches_stage75"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_score_margin_bm25_comparison_blocked"
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
                    "text": "Restart support note admin console restart.",
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
    stage94_report = tmp_path / "stage94.json"
    stage94_report.write_text(
        json.dumps(_stage94_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "train_split": train_split,
        "dev_split": dev_split,
        "documents": documents,
        "stage75_report": stage75_report,
        "stage94_report": stage94_report,
    }


def _stage94_report() -> dict:
    return {
        "stage": "Stage 94",
        "frozen_protocol": {
            "protocol_id": "score_margin_bm25_normalization_gate_train_dev_v1",
            "candidate_id": "score_margin_bm25_normalization_gate_design",
            "candidate_config_grid": [
                {
                    "config_id": "smbn_rank11_20_long_doc_b095_margin_v1",
                    "normalization_view_id": "bm25_k1_1_5_b_0_95_long_doc",
                    "challenger_bm25_k1": 1.5,
                    "challenger_bm25_b": 0.95,
                    "eligible_baseline_rank_min": 11,
                    "eligible_baseline_rank_max": 20,
                    "challenger_rank_max": 10,
                    "maximum_score_margin_to_rank10": 0.05,
                    "length_gate_mode": "long_document_only",
                    "minimum_document_length_ratio_to_average": 1.2,
                    "maximum_document_length_ratio_to_average": None,
                    "maximum_top10_promotions_per_query": 1,
                    "protected_bm25_top_rank_count": 8,
                },
                {
                    "config_id": "smbn_rank21_50_long_doc_b095_high_confidence_v1",
                    "normalization_view_id": "bm25_k1_1_5_b_0_95_long_doc",
                    "challenger_bm25_k1": 1.5,
                    "challenger_bm25_b": 0.95,
                    "eligible_baseline_rank_min": 21,
                    "eligible_baseline_rank_max": 50,
                    "challenger_rank_max": 15,
                    "maximum_score_margin_to_rank10": 0.03,
                    "length_gate_mode": "long_document_only",
                    "minimum_document_length_ratio_to_average": 1.5,
                    "maximum_document_length_ratio_to_average": None,
                    "maximum_top10_promotions_per_query": 1,
                    "protected_bm25_top_rank_count": 8,
                },
                {
                    "config_id": "smbn_rank11_20_short_doc_b055_margin_v1",
                    "normalization_view_id": "bm25_k1_1_5_b_0_55_short_doc",
                    "challenger_bm25_k1": 1.5,
                    "challenger_bm25_b": 0.55,
                    "eligible_baseline_rank_min": 11,
                    "eligible_baseline_rank_max": 20,
                    "challenger_rank_max": 10,
                    "maximum_score_margin_to_rank10": 0.04,
                    "length_gate_mode": "short_document_only",
                    "minimum_document_length_ratio_to_average": None,
                    "maximum_document_length_ratio_to_average": 0.85,
                    "maximum_top10_promotions_per_query": 1,
                    "protected_bm25_top_rank_count": 8,
                },
                {
                    "config_id": "smbn_rank11_50_dual_length_band_margin_v1",
                    "normalization_view_id": (
                        "bm25_k1_1_5_b_0_55_or_0_95_by_length_band"
                    ),
                    "challenger_bm25_k1": 1.5,
                    "challenger_bm25_b": (
                        "0.55_for_short_docs_0.95_for_long_docs"
                    ),
                    "eligible_baseline_rank_min": 11,
                    "eligible_baseline_rank_max": 50,
                    "challenger_rank_max": 12,
                    "maximum_score_margin_to_rank10": 0.02,
                    "length_gate_mode": "outside_length_band_short_or_long",
                    "minimum_document_length_ratio_to_average": 1.35,
                    "maximum_document_length_ratio_to_average": 0.75,
                    "maximum_top10_promotions_per_query": 1,
                    "protected_bm25_top_rank_count": 8,
                },
            ],
            "train_selection_rule": {
                "dev_selection_forbidden": True,
                "test_selection_forbidden": True,
                "stage82_dev_observation_selection_forbidden": True,
            },
            "historical_signal_policy": {
                "dev_only_b095_observation_can_select_runtime_rule": False,
            },
            "public_safe_changed_case_fields": [
                "sample_id",
                "split",
                "baseline_rank",
                "challenger_rank",
                "config_id",
                "normalization_view_id",
                "baseline_rank_bucket",
                "challenger_rank_bucket",
                "score_margin_bucket",
                "document_length_bucket",
                "promotion_reason_code",
            ],
        },
        "decision": {
            "status": "primeqa_hybrid_score_margin_bm25_protocol_frozen",
            "can_run_train_dev_metrics_after_user_confirmation": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        },
        "raw_question_text": "private score-margin protocol string",
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
