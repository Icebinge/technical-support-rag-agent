import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_evidence_answerability_comparison_protocol import (
    freeze_primeqa_hybrid_evidence_answerability_comparison_protocol,
    write_primeqa_hybrid_evidence_answerability_comparison_protocol_visualizations,
)


def test_evidence_answerability_comparison_protocol_freezes_confirmed_grid(tmp_path):
    stage103_path = _write_json(tmp_path / "stage103.json", _stage103_report())

    report = freeze_primeqa_hybrid_evidence_answerability_comparison_protocol(
        stage103_protocol_path=stage103_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = (
        write_primeqa_hybrid_evidence_answerability_comparison_protocol_visualizations(
            report=report,
            output_dir=tmp_path / "visuals",
        )
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 104"
    assert report["decision"]["status"] == (
        "primeqa_hybrid_evidence_answerability_comparison_protocol_frozen"
    )
    assert report["decision"]["protocol_id"] == (
        "evidence_answerability_candidate_train_dev_comparison_v1"
    )
    assert report["decision"][
        "can_run_train_dev_candidate_comparison_after_user_confirmation"
    ]
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["can_use_test_for_tuning"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert len(report["frozen_protocol"]["candidate_config_grid"]) == 9
    assert all(check["passed"] for check in report["guard_checks"])
    assert _config_counts_by_candidate(report) == {
        "answerability_margin_gate_candidate_v1": 3,
        "evidence_window_reselector_candidate_v1": 3,
        "joint_gate_then_window_candidate_v1": 3,
    }
    assert "Private Stage104 fixture question" not in serialized
    assert "private-doc-alpha" not in serialized
    assert "Restart the private queue worker" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage104_config_counts_by_candidate.svg",
        "stage104_config_min_evidence_scores.svg",
        "stage104_config_max_citation_ranks.svg",
        "stage104_selector_mix.svg",
        "stage104_train_selection_guard_thresholds.svg",
        "stage104_protocol_decision_flags.svg",
        "stage104_guard_check_status.svg",
    }


def test_evidence_answerability_comparison_protocol_blocks_without_confirmation(
    tmp_path,
):
    stage103_path = _write_json(tmp_path / "stage103.json", _stage103_report())

    report = freeze_primeqa_hybrid_evidence_answerability_comparison_protocol(
        stage103_protocol_path=stage103_path,
        user_confirmed_protocol=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage104_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_evidence_answerability_comparison_protocol_blocked"
    )


def test_evidence_answerability_comparison_protocol_blocks_unfrozen_stage103(
    tmp_path,
):
    stage103 = _stage103_report()
    stage103["decision"]["status"] = "blocked"
    stage103_path = _write_json(tmp_path / "stage103.json", stage103)

    report = freeze_primeqa_hybrid_evidence_answerability_comparison_protocol(
        stage103_protocol_path=stage103_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage103_protocol_is_frozen"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_evidence_answerability_comparison_protocol_blocked"
    )


def test_evidence_answerability_comparison_protocol_blocks_wrong_direction(
    tmp_path,
):
    stage103 = _stage103_report()
    stage103["decision"]["recommended_direction"] = "run_final_test"
    stage103_path = _write_json(tmp_path / "stage103.json", stage103)

    report = freeze_primeqa_hybrid_evidence_answerability_comparison_protocol(
        stage103_protocol_path=stage103_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage103_recommends_candidate_comparison"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_evidence_answerability_comparison_protocol_blocked"
    )


def test_evidence_answerability_comparison_output_contract_is_public_safe(
    tmp_path,
):
    stage103_path = _write_json(tmp_path / "stage103.json", _stage103_report())

    report = freeze_primeqa_hybrid_evidence_answerability_comparison_protocol(
        stage103_protocol_path=stage103_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )

    contract = report["frozen_protocol"]["public_safe_output_contract"]
    fields = set(contract["stage105_allowed_case_fields"]) | set(
        contract["stage105_allowed_aggregate_fields"]
    )
    assert not (
        fields
        & {
            "question_text",
            "raw_answer_text",
            "gold_answer",
            "answer_doc_id",
            "retrieved_doc_ids",
            "cited_doc_ids",
            "source_doc_ids",
            "document_text",
        }
    )


def _config_counts_by_candidate(report: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for config in report["frozen_protocol"]["candidate_config_grid"]:
        candidate_id = config["candidate_id"]
        counts[candidate_id] = counts.get(candidate_id, 0) + 1
    return counts


def _stage103_report() -> dict:
    return {
        "stage": "Stage 103",
        "design_id": "evidence_selection_and_answerability_candidate_design_v1",
        "bottleneck_summary": {
            "primary_bottleneck_bucket_ids": [
                "answerability_false_answer",
                "gold_span_beats_selected_answer",
            ],
            "secondary_context_bucket_ids": [
                "evidence_selection_miss",
                "retrieval_context_miss",
            ],
            "bucket_rows": [
                {
                    "bucket_id": "answerability_false_answer",
                    "train_count": 180,
                    "dev_count": 41,
                    "combined_count": 221,
                    "combined_priority_score": 342.55,
                    "shared_train_dev": True,
                },
                {
                    "bucket_id": "gold_span_beats_selected_answer",
                    "train_count": 174,
                    "dev_count": 41,
                    "combined_count": 215,
                    "combined_priority_score": 311.75,
                    "shared_train_dev": True,
                },
                {
                    "bucket_id": "evidence_selection_miss",
                    "train_count": 67,
                    "dev_count": 12,
                    "combined_count": 79,
                    "combined_priority_score": 134.3,
                    "shared_train_dev": True,
                },
            ],
        },
        "frozen_candidate_protocol": {
            "candidate_policies": [
                _candidate(
                    candidate_id="answerability_margin_gate_candidate_v1",
                    target_buckets=["answerability_false_answer"],
                    target_combined_case_count=221,
                    priority_score=271.2996,
                ),
                _candidate(
                    candidate_id="evidence_window_reselector_candidate_v1",
                    target_buckets=[
                        "gold_span_beats_selected_answer",
                        "evidence_selection_miss",
                    ],
                    target_combined_case_count=294,
                    priority_score=313.1271,
                ),
                _candidate(
                    candidate_id="joint_gate_then_window_candidate_v1",
                    target_buckets=[
                        "answerability_false_answer",
                        "gold_span_beats_selected_answer",
                        "evidence_selection_miss",
                    ],
                    target_combined_case_count=515,
                    priority_score=366.699,
                ),
            ],
            "blocked_items": [
                {
                    "blocked_item_id": "source_doc_id_oracle_candidate_blocked",
                    "status": "blocked_from_train_dev_experiment",
                }
            ],
            "stage104_train_dev_comparison_contract": {
                "comparison_id": (
                    "evidence_answerability_candidate_train_dev_comparison_v1"
                ),
                "run_mode": "train_dev_only_after_user_confirmation",
                "baseline_reference": "Stage102 verified BM25 top10 answer pipeline",
                "train_selection_rule": {
                    "candidate_thresholds_selected_on": "train_only",
                    "dev_threshold_tuning_allowed": False,
                    "test_access_allowed": False,
                },
                "dev_validation_rule": {
                    "dev_used_for": "single validation of train-selected candidate",
                    "dev_retuning_allowed": False,
                },
                "promotion_rule": {
                    "runtime_default_change_allowed_in_stage104": False,
                    "final_test_gate_remains_closed": True,
                },
                "metric_contract": {},
            },
        },
        "private_fixture_strings": {
            "question_text": "Private Stage104 fixture question",
            "answer_doc_id": "private-doc-alpha",
            "raw_answer_text": "Restart the private queue worker",
        },
        "decision": {
            "status": (
                "primeqa_hybrid_evidence_answerability_candidate_protocol_frozen"
            ),
            "design_id": "evidence_selection_and_answerability_candidate_design_v1",
            "recommended_direction": (
                "evidence_answerability_train_dev_candidate_comparison"
            ),
            "recommended_execution_order": [
                "joint_gate_then_window_candidate_v1",
                "evidence_window_reselector_candidate_v1",
                "answerability_margin_gate_candidate_v1",
            ],
            "requires_user_confirmation_before_train_dev_run": True,
            "can_continue_train_dev_development": True,
            "can_run_train_dev_candidate_comparison_after_user_confirmation": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
    }


def _candidate(
    *,
    candidate_id: str,
    target_buckets: list[str],
    target_combined_case_count: int,
    priority_score: float,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "status": "recommended_for_stage104_train_dev_protocol",
        "risk_level": "medium",
        "target_buckets": target_buckets,
        "target_combined_case_count": target_combined_case_count,
        "priority_score": priority_score,
    }


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
