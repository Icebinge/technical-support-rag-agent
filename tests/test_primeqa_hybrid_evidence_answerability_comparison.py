import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_answer_pipeline_error_decomposition_analysis import (
    run_primeqa_hybrid_answer_pipeline_error_decomposition,
)
from ts_rag_agent.application.primeqa_hybrid_evidence_answerability_comparison import (
    run_primeqa_hybrid_evidence_answerability_comparison,
    write_primeqa_hybrid_evidence_answerability_comparison_visualizations,
)


def test_evidence_answerability_comparison_runs_train_selected_dev_validated(
    tmp_path: Path,
) -> None:
    paths = _write_fixture_files(tmp_path)

    report = run_primeqa_hybrid_evidence_answerability_comparison(
        stage104_protocol_path=paths["stage104"],
        stage102_report_path=paths["stage102"],
        train_split_path=paths["train"],
        dev_split_path=paths["dev"],
        documents_path=paths["documents"],
        user_confirmed_comparison=True,
        confirmation_note="unit test confirmation",
        sample_limit_per_bucket_transition=2,
    )
    visualizations = (
        write_primeqa_hybrid_evidence_answerability_comparison_visualizations(
            report=report,
            output_dir=tmp_path / "visuals",
        )
    )

    serialized = json.dumps(report, ensure_ascii=False)
    checks = {check["name"]: check for check in report["guard_checks"]}
    assert report["stage"] == "Stage 105"
    assert report["split_contract"]["development_splits"] == ["train", "dev"]
    assert report["split_contract"]["forbidden_final_splits"] == ["test"]
    assert report["decision"]["status"] == (
        "primeqa_hybrid_evidence_answerability_comparison_completed_dev_guard_failed"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["can_use_test_for_tuning"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert report["train_selection"]["selection_split"] == "train"
    assert report["train_selection"]["selected_config_id"] == "unit_baseline_clone"
    assert report["dev_validation"]["validation_split"] == "dev"
    assert report["dev_validation"]["dev_validation_passed"] is False
    assert all(check["passed"] for check in report["guard_checks"])
    assert checks["dev_validation_not_used_for_selection"]["observed"][
        "ranking_keys"
    ] == [
        [
            "rank",
            "config_id",
            "candidate_id",
            "train_weighted_target_score",
            "train_weighted_target_delta",
            "train_selectable",
            "train_changed_answer_count",
        ]
    ]
    assert "private-doc-alpha" not in serialized
    assert "private-doc-beta" not in serialized
    assert "Private fixture answer text" not in serialized
    assert '"question_text":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage105_train_weighted_target_scores.svg",
        "stage105_dev_weighted_target_scores.svg",
        "stage105_train_target_score_deltas.svg",
        "stage105_dev_target_score_deltas.svg",
        "stage105_train_selectability_guards.svg",
        "stage105_changed_answer_counts.svg",
        "stage105_guard_check_status.svg",
    }


def test_evidence_answerability_comparison_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    paths = _write_fixture_files(tmp_path)

    report = run_primeqa_hybrid_evidence_answerability_comparison(
        stage104_protocol_path=paths["stage104"],
        stage102_report_path=paths["stage102"],
        train_split_path=paths["train"],
        dev_split_path=paths["dev"],
        documents_path=paths["documents"],
        user_confirmed_comparison=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage105_comparison"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_evidence_answerability_comparison_blocked"
    )


def _write_fixture_files(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "stage101": _write_json(tmp_path / "stage101.json", _stage101_protocol()),
        "documents": _write_json(tmp_path / "documents.json", _documents()),
    }
    for split in ("train", "dev"):
        paths[split] = _write_jsonl(
            tmp_path / f"{split}.jsonl",
            [
                _sample(
                    split=split,
                    sample_id=f"{split}_s1",
                    question_text="How do I restart queue worker after timeout?",
                    answerable=True,
                    answer="Restart the queue worker service.",
                    answer_doc_id="private-doc-alpha",
                ),
                _sample(
                    split=split,
                    sample_id=f"{split}_s2",
                    question_text="How do I fix purple screen calibration?",
                    answerable=True,
                    answer="Private fixture answer text for the isolated document.",
                    answer_doc_id="private-doc-missing-from-query",
                ),
                _sample(
                    split=split,
                    sample_id=f"{split}_s3",
                    question_text="How do I restart queue worker for unknown edition?",
                    answerable=False,
                    answer="",
                    answer_doc_id=None,
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
        retrieval_top_k=10,
        min_evidence_score=7.0,
        sample_limit_per_bucket=2,
    )
    paths["stage102"] = _write_json(tmp_path / "stage102.json", stage102)
    paths["stage104"] = _write_json(tmp_path / "stage104.json", _stage104_protocol())
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


def _stage104_protocol() -> dict:
    return {
        "stage": "Stage 104",
        "protocol_id": "evidence_answerability_candidate_train_dev_comparison_v1",
        "decision": {
            "status": (
                "primeqa_hybrid_evidence_answerability_comparison_protocol_frozen"
            ),
            "recommended_direction": (
                "evidence_answerability_candidate_train_dev_comparison"
            ),
            "can_run_train_dev_candidate_comparison_after_user_confirmation": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "frozen_protocol": {
            "protocol_status": "frozen",
            "baseline_reference": _baseline_reference(),
            "candidate_config_grid": [_baseline_clone_config()],
            "train_selection_rule": {
                "selection_split": "train",
                "objective": {
                    "answerability_false_answer": 1.55,
                    "gold_span_beats_selected_answer": 1.45,
                    "evidence_selection_miss": 1.70,
                },
                "selectability_guards": {
                    "max_train_answerable_refusal_rate_delta": 0.05,
                    "max_train_average_token_f1_drop": 0.01,
                    "max_train_gold_doc_citation_rate_drop": 0.03,
                },
            },
            "dev_validation_rule": {
                "validation_split": "dev",
                "threshold_tuning_allowed": False,
            },
        },
    }


def _baseline_reference() -> dict:
    return {
        "baseline_id": "stage102_verified_bm25_top10_answer_pipeline",
        "retriever": "BM25",
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "retrieval_top_k": 10,
        "evidence_selector_name": "bm25_sentence",
        "max_candidates_per_document": 3,
        "composition_policy_name": "top_k",
        "max_sentences": 3,
        "min_sentence_score": 2.0,
        "verifier_min_evidence_score": 7.0,
        "verifier_max_citation_rank": 3,
        "verifier_min_citations": 1,
        "source_stage": "Stage 102",
    }


def _baseline_clone_config() -> dict:
    return {
        "config_id": "unit_baseline_clone",
        "candidate_id": "unit_clone_candidate",
        "selector_name": "bm25_sentence",
        "composition_policy_name": "top_k",
        "max_candidates_per_document": 3,
        "max_sentences": 3,
        "min_sentence_score": 2.0,
        "verifier_min_citations": 1,
        "verifier_min_evidence_score": 7.0,
        "verifier_max_citation_rank": 3,
    }


def _documents() -> dict:
    return {
        "private-doc-alpha": {
            "id": "private-doc-alpha",
            "title": "Queue worker restart",
            "text": "Resolution: Restart the queue worker service. Verify worker logs.",
        },
        "private-doc-beta": {
            "id": "private-doc-beta",
            "title": "Queue worker timeout",
            "text": "Queue worker timeout unknown edition restart troubleshooting.",
        },
        "private-doc-missing-from-query": {
            "id": "private-doc-missing-from-query",
            "title": "Isolated private fixture",
            "text": "Private fixture answer text for the isolated document.",
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
            "private-doc-missing-from-query",
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
