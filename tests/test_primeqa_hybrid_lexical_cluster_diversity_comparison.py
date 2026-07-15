import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_lexical_cluster_diversity_comparison import (
    run_primeqa_hybrid_lexical_cluster_diversity_comparison,
    write_primeqa_hybrid_lexical_cluster_diversity_comparison_visualizations,
)


def test_lcdr_comparison_runs_train_dev_only_and_public_safe(tmp_path):
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_lexical_cluster_diversity_comparison(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage85_report_path=paths["stage85_report"],
        user_confirmed_protocol=True,
        confirmed_protocol_id="lexical_cluster_diversity_rerank_train_dev_v1",
        confirmation_note="unit test confirmation",
        top_k_values=(1, 5, 10),
        search_depth=10,
    )
    visualizations = write_primeqa_hybrid_lexical_cluster_diversity_comparison_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 86"
    assert set(report["metrics_by_split"]) == {"dev", "train"}
    assert len(report["candidate_configs"]) == 4
    assert "full_document_bm25_baseline" in report["metrics_by_split"]["dev"]
    assert report["loaded_data_summary"]["test_split_loaded"] is False
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Shared alpha error restart" not in serialized
    assert "Use the isolated recovery fix." not in serialized
    assert "isolated recovery fix" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage86_lcdr_train_hit_at_10.svg",
        "stage86_lcdr_dev_hit_at_10.svg",
        "stage86_lcdr_dev_delta_hit_at_10.svg",
        "stage86_lcdr_dev_top10_changes.svg",
        "stage86_lcdr_dev_answer_duplicate_buckets.svg",
    }


def test_lcdr_comparison_blocks_unconfirmed_protocol_without_metrics(tmp_path):
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_lexical_cluster_diversity_comparison(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage85_report_path=paths["stage85_report"],
        user_confirmed_protocol=False,
        confirmed_protocol_id="lexical_cluster_diversity_rerank_train_dev_v1",
        confirmation_note="not confirmed",
        top_k_values=(1, 5, 10),
        search_depth=10,
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_frozen_protocol"]["passed"] is False
    assert report["metrics_by_split"] == {}
    assert report["decision"]["status"] == (
        "primeqa_hybrid_lexical_cluster_diversity_comparison_blocked"
    )


def test_lcdr_penalty_can_move_duplicate_cluster_answer_within_top10(tmp_path):
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_lexical_cluster_diversity_comparison(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage85_report_path=paths["stage85_report"],
        user_confirmed_protocol=True,
        confirmed_protocol_id="lexical_cluster_diversity_rerank_train_dev_v1",
        confirmation_note="unit test confirmation",
        top_k_values=(1, 5, 10),
        search_depth=10,
    )

    train_comparisons = report["comparisons_to_baseline"]["train"]
    assert any(
        comparison["rank_up_within_top10_count"] > 0
        for comparison in train_comparisons.values()
    )
    selected_id = report["train_selection"]["selected_config_id"]
    assert selected_id in train_comparisons


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    train_split = tmp_path / "train.jsonl"
    dev_split = tmp_path / "dev.jsonl"
    _write_jsonl(
        train_split,
        [
            _split_sample(
                sample_id="primeqa_train:TRAIN_Q001",
                assigned_split="train",
                question_title="Alpha error restart",
                question_text="alpha error restart service",
                answer="Use the isolated recovery fix.",
                answer_doc_id="doc-answer",
            )
        ],
    )
    _write_jsonl(
        dev_split,
        [
            _split_sample(
                sample_id="primeqa_dev:DEV_Q001",
                assigned_split="dev",
                question_title="Alpha error restart",
                question_text="alpha error restart service",
                answer="Use the isolated recovery fix.",
                answer_doc_id="doc-answer",
            )
        ],
    )
    documents = tmp_path / "documents.json"
    documents.write_text(
        json.dumps(
            {
                "doc-duplicate-a": {
                    "id": "doc-duplicate-a",
                    "title": "Shared alpha error restart",
                    "text": "alpha error restart service alpha error restart",
                    "sections": [],
                },
                "doc-duplicate-b": {
                    "id": "doc-duplicate-b",
                    "title": "Shared alpha error restart",
                    "text": "alpha error restart service alpha error restart",
                    "sections": [],
                },
                "doc-answer": {
                    "id": "doc-answer",
                    "title": "Isolated alpha recovery",
                    "text": "alpha error restart service recovery fix alpha error restart",
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
    stage85_report = tmp_path / "stage85.json"
    stage85_report.write_text(
        json.dumps(
            {
                "stage": "Stage 85",
                "frozen_protocol": {
                    "protocol_id": "lexical_cluster_diversity_rerank_train_dev_v1",
                    "candidate_id": "lexical_cluster_diversity_rerank_design",
                    "candidate_config_grid": [
                        _lcdr_config("lcdr_penalty_0_03_title_query_cluster", 0.03),
                        _lcdr_config("lcdr_penalty_0_06_title_query_cluster", 0.06),
                        _lcdr_config("lcdr_penalty_0_09_title_query_cluster", 0.09),
                        _lcdr_config("lcdr_penalty_0_12_title_query_cluster", 0.12),
                    ],
                    "public_safe_changed_case_fields": [
                        "sample_id",
                        "split",
                        "baseline_rank",
                        "challenger_rank",
                        "baseline_cluster_duplicate_index",
                        "challenger_cluster_duplicate_index",
                        "config_id",
                    ],
                },
                "decision": {
                    "can_run_train_dev_metrics_after_user_confirmation": True,
                    "can_run_final_test_metrics_now": False,
                    "can_use_test_for_tuning": False,
                    "default_runtime_policy": "unchanged",
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
        "stage85_report": stage85_report,
    }


def _lcdr_config(config_id: str, duplicate_penalty_weight: float) -> dict:
    return {
        "config_id": config_id,
        "duplicate_penalty_weight": duplicate_penalty_weight,
        "cluster_key": "title_query_overlap_hash",
        "minimum_title_overlap_terms": 3,
        "minimum_cluster_size": 2,
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
