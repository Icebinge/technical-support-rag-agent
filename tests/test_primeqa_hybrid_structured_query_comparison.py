import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_structured_query_comparison import (
    run_primeqa_hybrid_structured_query_comparison,
    write_primeqa_hybrid_structured_query_comparison_visualizations,
)


def test_structured_query_comparison_runs_train_dev_only_and_public_safe(tmp_path):
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_structured_query_comparison(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage88_report_path=paths["stage88_report"],
        user_confirmed_protocol=True,
        confirmed_protocol_id="structured_query_keyphrase_compaction_train_dev_v1",
        confirmation_note="unit test confirmation",
        top_k_values=(1, 5, 10),
        search_depth=20,
    )
    visualizations = write_primeqa_hybrid_structured_query_comparison_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 89"
    assert set(report["metrics_by_split"]) == {"dev", "train"}
    assert len(report["candidate_configs"]) == 4
    assert "full_document_bm25_baseline" in report["metrics_by_split"]["dev"]
    assert report["loaded_data_summary"]["test_split_loaded"] is False
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert "customer urgent help problem" not in serialized
    assert "Use the alpha recovery fix." not in serialized
    assert "alpha recovery fix" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage89_structured_query_train_hit_at_10.svg",
        "stage89_structured_query_dev_hit_at_10.svg",
        "stage89_structured_query_dev_delta_hit_at_10.svg",
        "stage89_structured_query_dev_top10_changes.svg",
        "stage89_structured_query_average_compacted_terms.svg",
    }


def test_structured_query_comparison_blocks_unconfirmed_protocol_without_metrics(
    tmp_path,
):
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_structured_query_comparison(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage88_report_path=paths["stage88_report"],
        user_confirmed_protocol=False,
        confirmed_protocol_id="structured_query_keyphrase_compaction_train_dev_v1",
        confirmation_note="not confirmed",
        top_k_values=(1, 5, 10),
        search_depth=20,
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_frozen_protocol"]["passed"] is False
    assert report["metrics_by_split"] == {}
    assert report["decision"]["status"] == (
        "primeqa_hybrid_structured_query_comparison_blocked"
    )


def test_structured_query_compacts_noisy_full_question(tmp_path):
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_structured_query_comparison(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage88_report_path=paths["stage88_report"],
        user_confirmed_protocol=True,
        confirmed_protocol_id="structured_query_keyphrase_compaction_train_dev_v1",
        confirmation_note="unit test confirmation",
        top_k_values=(1, 5, 10),
        search_depth=20,
    )

    selected_id = report["train_selection"]["selected_config_id"]
    train_metrics = report["metrics_by_split"]["train"]
    assert selected_id in report["comparisons_to_baseline"]["train"]
    assert (
        train_metrics[selected_id]["average_compacted_query_token_count"]
        < train_metrics["full_document_bm25_baseline"]["average_query_token_count"]
    )


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    train_split = tmp_path / "train.jsonl"
    dev_split = tmp_path / "dev.jsonl"
    noisy_question_text = (
        "about after all also any are as at before being but by can could "
        "customer did do does during for from get getting had has have help "
        "how need needs not please problem question saw see should that the "
        "their them there these they this those urgent were what when where "
        "which why will with would your alpha service restart E123"
    )
    noisy_decoy_text = (
        "about after all also any are as at before being but by can could "
        "customer did do does during for from get getting had has have help "
        "how need needs not please problem question saw see should that the "
        "their them there these they this those urgent were what when where "
        "which why will with would your "
    )
    _write_jsonl(
        train_split,
        [
            _split_sample(
                sample_id="primeqa_train:TRAIN_Q001",
                assigned_split="train",
                question_title="Alpha error E123 restart",
                question_text=noisy_question_text,
                answer="Use the alpha recovery fix.",
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
                question_title="Alpha error E123 restart",
                question_text=noisy_question_text,
                answer="Use the alpha recovery fix.",
                answer_doc_id="doc-answer",
            )
        ],
    )
    documents = tmp_path / "documents.json"
    document_rows = {
        f"doc-decoy-{index:02d}": {
            "id": f"doc-decoy-{index:02d}",
            "title": "Customer urgent help problem",
            "text": noisy_decoy_text * 2,
            "sections": [],
        }
        for index in range(12)
    }
    document_rows["doc-answer"] = {
        "id": "doc-answer",
        "title": "Alpha service restart E123",
        "text": "alpha service restart E123 recovery fix",
        "sections": [],
    }
    documents.write_text(
        json.dumps(document_rows, ensure_ascii=False, indent=2),
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
    stage88_report = tmp_path / "stage88.json"
    stage88_report.write_text(
        json.dumps(
            {
                "stage": "Stage 88",
                "frozen_protocol": {
                    "protocol_id": (
                        "structured_query_keyphrase_compaction_train_dev_v1"
                    ),
                    "candidate_id": "structured_query_keyphrase_compaction_design",
                    "candidate_config_grid": _structured_query_grid(),
                    "public_safe_changed_case_fields": [
                        "sample_id",
                        "split",
                        "baseline_rank",
                        "challenger_rank",
                        "config_id",
                        "query_view_id",
                        "query_token_count",
                        "compacted_query_token_count",
                        "token_bucket_counts",
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
        "stage88_report": stage88_report,
    }


def _structured_query_grid() -> list[dict]:
    return [
        {
            "config_id": "sqkc_action_error_product_v1",
            "query_view_id": "action_error_product_version_terms",
            "preserved_feature_buckets": [
                "error_code_or_log_identifier",
                "product_component_or_feature",
                "version_or_platform",
                "action_intent",
                "quoted_or_code_like_terms",
            ],
            "maximum_unique_terms": 18,
            "minimum_unique_terms": 4,
        },
        {
            "config_id": "sqkc_title_guarded_action_error_v1",
            "query_view_id": "title_guarded_action_error_product_terms",
            "preserved_feature_buckets": [
                "title_guard_terms",
                "error_code_or_log_identifier",
                "product_component_or_feature",
                "version_or_platform",
                "action_intent",
            ],
            "minimum_title_terms": 3,
            "maximum_unique_terms": 16,
            "minimum_unique_terms": 4,
        },
        {
            "config_id": "sqkc_error_first_compact_v1",
            "query_view_id": "error_identifier_first_terms",
            "preserved_feature_buckets": [
                "error_code_or_log_identifier",
                "quoted_or_code_like_terms",
                "product_component_or_feature",
                "action_intent",
            ],
            "maximum_unique_terms": 14,
            "minimum_unique_terms": 3,
        },
        {
            "config_id": "sqkc_noun_phrase_compact_v1",
            "query_view_id": "deterministic_noun_phrase_like_terms",
            "preserved_feature_buckets": [
                "deterministic_noun_phrase_like_terms",
                "product_component_or_feature",
                "version_or_platform",
                "action_intent",
            ],
            "noun_phrase_window_size": 2,
            "maximum_unique_terms": 20,
            "minimum_unique_terms": 4,
        },
    ]


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
