import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_validation_failure_pattern_analysis import (
    analyze_primeqa_hybrid_validation_failure_patterns,
    write_primeqa_hybrid_validation_failure_pattern_visualizations,
)


def test_validation_failure_pattern_analysis_runs_public_safe(tmp_path: Path) -> None:
    paths = _write_fixture_files(tmp_path)

    report = analyze_primeqa_hybrid_validation_failure_patterns(
        stage102_report_path=paths["stage102"],
        stage105_report_path=paths["stage105"],
        stage106_decision_path=paths["stage106"],
        user_confirmed_analysis=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = write_primeqa_hybrid_validation_failure_pattern_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    pattern = report["pattern_summary"]
    checks = {check["name"]: check for check in report["guard_checks"]}
    assert report["stage"] == "Stage 107"
    assert report["protocol_id"] == (
        "primeqa_hybrid_validation_failure_pattern_analysis_v1"
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_validation_failure_pattern_analysis_completed"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert pattern["dev_failure_overview"]["failure_count"] == 9
    assert pattern["dev_failure_overview"]["answerable_failure_rate"] == 1.0
    assert pattern["dev_failure_overview"]["unanswerable_false_answer_rate"] == 0.75
    assert pattern["dev_retrieval_and_context_profile"][
        "answerable_gold_context_absent_count"
    ] == 2
    assert pattern["stage105_candidate_failure_pattern"][
        "dev_better_nonselectable_config_count"
    ] == 1
    assert all(check["passed"] for check in report["guard_checks"])
    assert checks["stage107_reads_saved_reports_only"]["passed"] is True
    assert '"question_text":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert "private-doc-alpha" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage107_dev_failure_bucket_counts.svg",
        "stage107_train_dev_bucket_rate_drift.svg",
        "stage107_dev_route_failure_counts.svg",
        "stage107_dev_answerable_failure_flow.svg",
        "stage107_stage105_candidate_behavior.svg",
        "stage107_decision_flags.svg",
        "stage107_guard_check_status.svg",
    }


def test_validation_failure_pattern_analysis_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    paths = _write_fixture_files(tmp_path)

    report = analyze_primeqa_hybrid_validation_failure_patterns(
        stage102_report_path=paths["stage102"],
        stage105_report_path=paths["stage105"],
        stage106_decision_path=paths["stage106"],
        user_confirmed_analysis=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage107_failure_pattern_analysis"][
        "passed"
    ] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_validation_failure_pattern_analysis_blocked"
    )


def _write_fixture_files(tmp_path: Path) -> dict[str, Path]:
    return {
        "stage102": _write_json(tmp_path / "stage102.json", _stage102_report()),
        "stage105": _write_json(tmp_path / "stage105.json", _stage105_report()),
        "stage106": _write_json(tmp_path / "stage106.json", _stage106_report()),
    }


def _stage102_report() -> dict:
    return {
        "stage": "Stage 102",
        "analysis_id": "answer_pipeline_error_decomposition_train_dev_analysis_v1",
        "data_summary": {
            "splits": {
                "train": {
                    "row_count": 20,
                    "answerable_count": 12,
                    "unanswerable_count": 8,
                },
                "dev": {
                    "row_count": 10,
                    "answerable_count": 6,
                    "unanswerable_count": 4,
                },
            }
        },
        "metrics_by_split": {
            "dev": {
                "verified": {
                    "total_questions": 10,
                    "answerable_questions": 6,
                    "unanswerable_questions": 4,
                    "gold_doc_citation_rate": 0.5,
                    "answerable_refusal_rate": 0.0,
                    "unanswerable_refusal_rate": 0.25,
                    "average_token_f1": 0.2,
                }
            }
        },
        "aggregate_outputs": {
            "bucket_counts_by_split": {
                "train": {
                    "answerability_false_answer": 6,
                    "retrieval_context_miss": 4,
                    "evidence_selection_miss": 2,
                    "verification_over_refusal": 0,
                    "gold_span_beats_selected_answer": 5,
                    "low_overlap_gold_cited_answer": 0,
                    "answer_supported_and_cited": 3,
                },
                "dev": {
                    "answerability_false_answer": 3,
                    "retrieval_context_miss": 2,
                    "evidence_selection_miss": 1,
                    "verification_over_refusal": 0,
                    "gold_span_beats_selected_answer": 3,
                    "low_overlap_gold_cited_answer": 0,
                    "answer_supported_and_cited": 1,
                },
            },
            "bucket_rates_by_split": {
                "train": {
                    "answerability_false_answer": 0.3,
                    "retrieval_context_miss": 0.2,
                    "evidence_selection_miss": 0.1,
                    "verification_over_refusal": 0.0,
                    "gold_span_beats_selected_answer": 0.25,
                    "low_overlap_gold_cited_answer": 0.0,
                    "answer_supported_and_cited": 0.15,
                },
                "dev": {
                    "answerability_false_answer": 0.3,
                    "retrieval_context_miss": 0.2,
                    "evidence_selection_miss": 0.1,
                    "verification_over_refusal": 0.0,
                    "gold_span_beats_selected_answer": 0.3,
                    "low_overlap_gold_cited_answer": 0.0,
                    "answer_supported_and_cited": 0.1,
                },
            },
            "answerability_bucket_cross_tab": {
                "dev::answerable": {
                    "answerability_false_answer": 0,
                    "retrieval_context_miss": 2,
                    "evidence_selection_miss": 1,
                    "verification_over_refusal": 0,
                    "gold_span_beats_selected_answer": 3,
                    "low_overlap_gold_cited_answer": 0,
                    "answer_supported_and_cited": 0,
                },
                "dev::unanswerable": {
                    "answerability_false_answer": 3,
                    "retrieval_context_miss": 0,
                    "evidence_selection_miss": 0,
                    "verification_over_refusal": 0,
                    "gold_span_beats_selected_answer": 0,
                    "low_overlap_gold_cited_answer": 0,
                    "answer_supported_and_cited": 1,
                },
            },
            "route_bucket_cross_tab": {
                "dev::other": {
                    "answerability_false_answer": 2,
                    "retrieval_context_miss": 1,
                    "evidence_selection_miss": 1,
                    "verification_over_refusal": 0,
                    "gold_span_beats_selected_answer": 2,
                    "low_overlap_gold_cited_answer": 0,
                    "answer_supported_and_cited": 1,
                },
                "dev::install_upgrade_config": {
                    "answerability_false_answer": 1,
                    "retrieval_context_miss": 1,
                    "evidence_selection_miss": 0,
                    "verification_over_refusal": 0,
                    "gold_span_beats_selected_answer": 1,
                    "low_overlap_gold_cited_answer": 0,
                    "answer_supported_and_cited": 0,
                },
            },
            "retrieval_rank_bucket_distributions": {
                "dev": {
                    "not_applicable": 4,
                    "not_found_top_k": 2,
                    "rank_1": 3,
                    "rank_2_to_3": 1,
                }
            },
            "token_f1_bucket_distributions": {
                "dev": {
                    "f1_0_00_to_0_19": 4,
                    "f1_0_20_to_0_39": 2,
                    "not_applicable": 4,
                }
            },
            "verification_decision_distributions": {
                "dev": {
                    "answered": 9,
                    "refused": 1,
                }
            },
        },
        "guard_checks": [_passed_check("stage102")],
        "decision": {
            "status": "primeqa_hybrid_answer_pipeline_error_decomposition_completed",
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
    }


def _stage105_report() -> dict:
    return {
        "stage": "Stage 105",
        "analysis_id": "evidence_answerability_candidate_train_dev_comparison_v1",
        "config_results": [
            _config_result(
                config_id="selected_noop",
                candidate_id="answerability_margin_gate_candidate_v1",
                selectable=True,
                train_delta=0.0,
                dev_delta=0.0,
                train_changed=1,
                dev_changed=0,
                checks={
                    "answerable_refusal_rate_delta_within_guard": True,
                    "average_token_f1_drop_within_guard": True,
                    "gold_doc_citation_rate_drop_within_guard": True,
                },
            ),
            _config_result(
                config_id="better_blocked",
                candidate_id="evidence_window_reselector_candidate_v1",
                selectable=False,
                train_delta=-5.0,
                dev_delta=-2.0,
                train_changed=20,
                dev_changed=9,
                checks={
                    "answerable_refusal_rate_delta_within_guard": False,
                    "average_token_f1_drop_within_guard": True,
                    "gold_doc_citation_rate_drop_within_guard": False,
                },
            ),
        ],
        "train_selection": {
            "selection_split": "train",
            "selected_config_id": "selected_noop",
            "selected_candidate_id": "answerability_margin_gate_candidate_v1",
            "selected_train_weighted_target_delta": 0.0,
            "selectable_config_count": 1,
            "config_count": 2,
        },
        "dev_validation": {
            "validation_split": "dev",
            "selected_config_id": "selected_noop",
            "selected_candidate_id": "answerability_margin_gate_candidate_v1",
            "dev_weighted_target_delta": 0.0,
            "dev_changed_answer_count": 0,
            "dev_target_bucket_deltas": {
                "answerability_false_answer": 0,
                "gold_span_beats_selected_answer": 0,
                "evidence_selection_miss": 0,
            },
            "dev_metric_deltas": {
                "answerable_refusal_rate": 0.0,
                "unanswerable_refusal_rate": 0.0,
                "gold_doc_citation_rate": 0.0,
                "average_token_f1": 0.0,
            },
            "dev_validation_passed": False,
        },
        "guard_checks": [_passed_check("stage105")],
        "decision": {
            "status": (
                "primeqa_hybrid_evidence_answerability_comparison_completed_dev_guard_failed"
            ),
            "selected_config_id": "selected_noop",
            "selected_candidate_id": "answerability_margin_gate_candidate_v1",
            "selectable_config_count": 1,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
    }


def _stage106_report() -> dict:
    return {
        "stage": "Stage 106",
        "stopped_family": {
            "dev_better_nonselectable_configs": [
                {
                    "config_id": "better_blocked",
                    "train_selectable": False,
                }
            ]
        },
        "guard_checks": [_passed_check("stage106")],
        "decision": {
            "status": "primeqa_hybrid_evidence_answerability_candidate_family_stopped",
            "stopped_family_id": "evidence_answerability_candidate_family",
            "stopped_protocol_id": "evidence_answerability_candidate_train_dev_comparison_v1",
            "current_route_defaultization": "blocked",
            "redesign_required_before_any_runtime_or_test_gate": True,
            "recommended_next_direction": "evidence_answerability_redesign_decision",
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
    }


def _config_result(
    *,
    config_id: str,
    candidate_id: str,
    selectable: bool,
    train_delta: float,
    dev_delta: float,
    train_changed: int,
    dev_changed: int,
    checks: dict[str, bool],
) -> dict:
    return {
        "config_id": config_id,
        "candidate_id": candidate_id,
        "weighted_target_score_deltas_by_split": {
            "train": train_delta,
            "dev": dev_delta,
        },
        "changed_answer_counts_by_split": {
            "train": train_changed,
            "dev": dev_changed,
        },
        "train_selectability": {
            "selectable": selectable,
            "checks": checks,
        },
    }


def _passed_check(name: str) -> dict:
    return {
        "name": name,
        "passed": True,
        "observed": True,
        "expected": True,
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
