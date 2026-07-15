import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_answer_pipeline_error_decomposition_analysis import (
    run_primeqa_hybrid_answer_pipeline_error_decomposition,
)
from ts_rag_agent.application.primeqa_hybrid_failure_pattern_redesign_comparison import (
    run_primeqa_hybrid_failure_pattern_redesign_comparison,
    write_primeqa_hybrid_failure_pattern_redesign_comparison_visualizations,
)
from ts_rag_agent.application.primeqa_hybrid_failure_pattern_redesign_protocol import (
    freeze_primeqa_hybrid_failure_pattern_redesign_protocol,
)


def test_failure_pattern_redesign_comparison_runs_train_cv_dev_validation(
    tmp_path: Path,
) -> None:
    paths = _write_fixture_files(tmp_path)

    report = run_primeqa_hybrid_failure_pattern_redesign_comparison(
        stage108_protocol_path=paths["stage108"],
        stage102_report_path=paths["stage102"],
        train_split_path=paths["train"],
        dev_split_path=paths["dev"],
        documents_path=paths["documents"],
        user_confirmed_comparison=True,
        confirmation_note="unit test confirmation",
        retrieval_top_k=3,
        sample_limit_per_transition=2,
    )
    visualizations = (
        write_primeqa_hybrid_failure_pattern_redesign_comparison_visualizations(
            report=report,
            output_dir=tmp_path / "visuals",
        )
    )

    serialized = json.dumps(report, ensure_ascii=False)
    checks = {check["name"]: check for check in report["guard_checks"]}
    assert report["stage"] == "Stage 109"
    assert report["split_contract"]["development_splits"] == ["train", "dev"]
    assert report["split_contract"]["forbidden_final_splits"] == ["test"]
    assert report["data_summary"]["train_cv"]["fold_count"] == 5
    assert report["train_cv_selection"]["selection_split"] == "train"
    assert report["train_cv_selection"]["selection_mode"] == (
        "train_grouped_cross_validation_then_full_train_refit"
    )
    assert report["dev_validation"]["validation_split"] == "dev"
    assert len(report["config_results"]) == 7
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["can_use_test_for_tuning"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert checks["train_cv_selection_uses_train_only_fields"]["observed"][
        "ranking_keys"
    ] == [
        [
            "rank",
            "config_id",
            "candidate_family_id",
            "train_cv_weighted_target_score",
            "train_cv_weighted_target_delta",
            "train_cv_selectable",
            "train_cv_changed_answer_count",
        ]
    ]
    assert "private-doc-alpha" not in serialized
    assert "private-doc-beta" not in serialized
    assert "Private fixture answer text" not in serialized
    assert '"question_text":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage109_train_cv_weighted_target_deltas.svg",
        "stage109_dev_weighted_target_deltas.svg",
        "stage109_train_cv_selectability.svg",
        "stage109_changed_answer_counts.svg",
        "stage109_dev_metric_deltas.svg",
        "stage109_decision_flags.svg",
        "stage109_guard_check_status.svg",
    }


def test_failure_pattern_redesign_comparison_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    paths = _write_fixture_files(tmp_path)

    report = run_primeqa_hybrid_failure_pattern_redesign_comparison(
        stage108_protocol_path=paths["stage108"],
        stage102_report_path=paths["stage102"],
        train_split_path=paths["train"],
        dev_split_path=paths["dev"],
        documents_path=paths["documents"],
        user_confirmed_comparison=False,
        confirmation_note="not confirmed",
        retrieval_top_k=3,
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage109_comparison"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_failure_pattern_redesign_comparison_blocked"
    )


def _write_fixture_files(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "stage101": _write_json(tmp_path / "stage101.json", _stage101_protocol()),
        "stage107": _write_json(tmp_path / "stage107.json", _stage107_report()),
        "documents": _write_json(tmp_path / "documents.json", _documents()),
    }
    paths["train"] = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            _sample(
                split="train",
                sample_id="train_s1",
                question_text="How do I restart queue worker after timeout?",
                answerable=True,
                answer="Restart the queue worker service.",
                answer_doc_id="private-doc-alpha",
            ),
            _sample(
                split="train",
                sample_id="train_s2",
                question_text="How should I verify queue worker logs after restart?",
                answerable=True,
                answer="Verify worker logs after restarting the queue worker service.",
                answer_doc_id="private-doc-alpha",
            ),
            _sample(
                split="train",
                sample_id="train_s3",
                question_text="How do I increase retry timeout for queue failures?",
                answerable=True,
                answer="Increase the queue retry timeout.",
                answer_doc_id="private-doc-beta",
            ),
            _sample(
                split="train",
                sample_id="train_s4",
                question_text="How do I clear cache corruption after deployment?",
                answerable=True,
                answer="Clear the deployment cache and restart the cache service.",
                answer_doc_id="private-doc-gamma",
            ),
            _sample(
                split="train",
                sample_id="train_s5",
                question_text="How do I rebuild the search index safely?",
                answerable=True,
                answer="Run the index rebuild job and verify the search logs.",
                answer_doc_id="private-doc-delta",
            ),
            _sample(
                split="train",
                sample_id="train_s6",
                question_text="How do I restart queue worker for unknown edition?",
                answerable=False,
                answer="",
                answer_doc_id=None,
            ),
        ],
    )
    paths["dev"] = _write_jsonl(
        tmp_path / "dev.jsonl",
        [
            _sample(
                split="dev",
                sample_id="dev_s1",
                question_text="How do I restart queue worker service?",
                answerable=True,
                answer="Restart the queue worker service.",
                answer_doc_id="private-doc-alpha",
            ),
            _sample(
                split="dev",
                sample_id="dev_s2",
                question_text="How do I repair unknown queue worker product line?",
                answerable=False,
                answer="",
                answer_doc_id=None,
            ),
            _sample(
                split="dev",
                sample_id="dev_s3",
                question_text="How do I rebuild the search index?",
                answerable=True,
                answer="Run the index rebuild job and verify the search logs.",
                answer_doc_id="private-doc-delta",
            ),
        ],
    )
    stage102 = run_primeqa_hybrid_answer_pipeline_error_decomposition(
        stage101_protocol_path=paths["stage101"],
        train_split_path=paths["train"],
        dev_split_path=paths["dev"],
        documents_path=paths["documents"],
        user_confirmed_analysis=True,
        confirmation_note="unit test confirmation",
        retrieval_top_k=3,
        min_evidence_score=7.0,
        sample_limit_per_bucket=2,
    )
    stage108 = freeze_primeqa_hybrid_failure_pattern_redesign_protocol(
        stage107_report_path=paths["stage107"],
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )
    paths["stage102"] = _write_json(tmp_path / "stage102.json", stage102)
    paths["stage108"] = _write_json(tmp_path / "stage108.json", stage108)
    return paths


def _stage101_protocol() -> dict:
    return {
        "stage": "Stage 101",
        "decision": {
            "status": (
                "primeqa_hybrid_answer_pipeline_error_decomposition_protocol_frozen"
            ),
            "protocol_id": "answer_pipeline_error_decomposition_train_dev_v1",
            "recommended_direction": "answer_pipeline_error_decomposition",
            "can_run_train_dev_error_decomposition_after_user_confirmation": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "frozen_protocol": {
            "public_safe_output_contract": {
                "case_sample_fields": [
                    "sample_id",
                    "split",
                    "answerability_label",
                    "pipeline_bucket_id",
                    "pipeline_stage",
                    "retrieval_rank_bucket",
                    "retrieval_context_status",
                    "citation_status",
                    "evidence_selection_status",
                    "answer_token_f1_bucket",
                    "best_gold_span_f1_bucket",
                    "answer_gold_span_gap_bucket",
                    "verifier_decision",
                    "refusal_reason_code",
                    "question_route",
                    "evidence_selector_name",
                    "composition_policy_id",
                    "bucket_confidence_band",
                ]
            }
        },
    }


def _stage107_report() -> dict:
    return {
        "stage": "Stage 107",
        "protocol_id": "primeqa_hybrid_validation_failure_pattern_analysis_v1",
        "pattern_summary": {
            "dev_failure_overview": {
                "failure_count": 117,
                "failure_rate": 0.9669,
                "answerable_failure_count": 76,
                "answerable_failure_rate": 1.0,
                "answerable_non_error_count": 0,
                "unanswerable_false_answer_count": 41,
                "unanswerable_false_answer_rate": 0.9111,
            },
            "dev_retrieval_and_context_profile": {
                "answerable_gold_context_absent_rate": 0.3026,
                "context_present_gold_span_beats_selected_rate": 0.7736,
                "context_present_evidence_selection_miss_rate": 0.2264,
                "answerable_supported_and_cited_count": 0,
            },
            "stage105_candidate_failure_pattern": {
                "dev_better_nonselectable_config_count": 7,
                "train_guard_failure_reasons": {
                    "answerable_refusal_rate_delta_within_guard": 7,
                    "gold_doc_citation_rate_drop_within_guard": 4,
                },
            },
        },
        "guard_checks": [
            {
                "name": "stage107_fixture_guard",
                "passed": True,
                "observed": True,
                "expected": True,
            }
        ],
        "decision": {
            "status": "primeqa_hybrid_validation_failure_pattern_analysis_completed",
            "recommended_next_direction": (
                "failure_pattern_driven_train_dev_redesign_protocol"
            ),
            "stage105_selected_config_was_dev_noop": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
    }


def _documents() -> dict:
    return {
        "private-doc-alpha": {
            "id": "private-doc-alpha",
            "title": "Queue worker restart",
            "text": (
                "Resolution: Restart the queue worker service. "
                "Verify worker logs after restarting the queue worker service."
            ),
        },
        "private-doc-beta": {
            "id": "private-doc-beta",
            "title": "Queue worker timeout",
            "text": (
                "Timeout handling: Increase the queue retry timeout. "
                "Queue worker timeout unknown edition restart troubleshooting."
            ),
        },
        "private-doc-gamma": {
            "id": "private-doc-gamma",
            "title": "Deployment cache repair",
            "text": (
                "Cache repair: Clear the deployment cache and restart the cache service."
            ),
        },
        "private-doc-delta": {
            "id": "private-doc-delta",
            "title": "Search index rebuild",
            "text": "Run the index rebuild job and verify the search logs.",
        },
    }


def _sample(
    *,
    split: str,
    sample_id: str,
    question_text: str,
    answerable: bool,
    answer: str,
    answer_doc_id: str | None,
) -> dict:
    return {
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "assigned_split": split,
        "split_subtype": "random_grouped",
        "source_split": "fixture",
        "sample_id": sample_id,
        "question_id": f"source_{sample_id}",
        "question_title": "Fixture title",
        "question_text": question_text,
        "answerable": answerable,
        "answer": answer,
        "answer_doc_id": answer_doc_id,
        "candidate_doc_ids": [
            "private-doc-alpha",
            "private-doc-beta",
            "private-doc-gamma",
            "private-doc-delta",
        ],
        "answer_span": {
            "start_offset": 0 if answerable else None,
            "end_offset": len(answer) if answerable else None,
        },
    }


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    return path
