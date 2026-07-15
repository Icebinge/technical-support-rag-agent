import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_failure_pattern_redesign_protocol import (
    freeze_primeqa_hybrid_failure_pattern_redesign_protocol,
    write_primeqa_hybrid_failure_pattern_redesign_protocol_visualizations,
)


def test_failure_pattern_redesign_protocol_freezes_confirmed_grid(
    tmp_path: Path,
) -> None:
    stage107_path = _write_json(tmp_path / "stage107.json", _stage107_report())

    report = freeze_primeqa_hybrid_failure_pattern_redesign_protocol(
        stage107_report_path=stage107_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = (
        write_primeqa_hybrid_failure_pattern_redesign_protocol_visualizations(
            report=report,
            output_dir=tmp_path / "visuals",
        )
    )

    serialized = json.dumps(report, ensure_ascii=False)
    frozen = report["frozen_protocol"]
    train_rule = frozen["train_selection_rule"]
    dev_rule = frozen["dev_validation_rule"]
    assert report["stage"] == "Stage 108"
    assert report["decision"]["status"] == (
        "primeqa_hybrid_failure_pattern_redesign_protocol_frozen"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert len(frozen["candidate_families"]) == 3
    assert len(frozen["candidate_config_grid"]) == 7
    assert train_rule["selection_split"] == "train"
    assert train_rule["train_cv_fold_count"] == 5
    assert train_rule["objective"]["requires_negative_train_cv_weighted_delta"] is True
    assert train_rule["objective"]["no_op_candidate_selectable"] is False
    assert dev_rule["dev_selection_allowed"] is False
    assert dev_rule["dev_retuning_allowed"] is False
    assert all(check["passed"] for check in report["guard_checks"])
    assert "private-doc-alpha" not in serialized
    assert '"answer_doc_id":' not in serialized
    assert '"question_text":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage108_candidate_family_priorities.svg",
        "stage108_candidate_config_counts.svg",
        "stage108_target_bucket_weights.svg",
        "stage108_train_cv_guard_thresholds.svg",
        "stage108_protocol_decision_flags.svg",
        "stage108_guard_check_status.svg",
    }


def test_failure_pattern_redesign_protocol_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    stage107_path = _write_json(tmp_path / "stage107.json", _stage107_report())

    report = freeze_primeqa_hybrid_failure_pattern_redesign_protocol(
        stage107_report_path=stage107_path,
        user_confirmed_protocol=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage108_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_failure_pattern_redesign_protocol_blocked"
    )


def test_failure_pattern_redesign_protocol_blocks_unfinished_stage107(
    tmp_path: Path,
) -> None:
    stage107 = _stage107_report()
    stage107["decision"]["status"] = "blocked"
    stage107_path = _write_json(tmp_path / "stage107.json", stage107)

    report = freeze_primeqa_hybrid_failure_pattern_redesign_protocol(
        stage107_report_path=stage107_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage107_analysis_completed"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_failure_pattern_redesign_protocol_blocked"
    )


def test_failure_pattern_redesign_protocol_public_safe_contract(
    tmp_path: Path,
) -> None:
    stage107_path = _write_json(tmp_path / "stage107.json", _stage107_report())

    report = freeze_primeqa_hybrid_failure_pattern_redesign_protocol(
        stage107_report_path=stage107_path,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmation",
    )

    contract = report["frozen_protocol"]["public_safe_output_contract"]
    fields = set(contract["allowed_stage109_case_fields"])
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
        "private_fixture_strings": {
            "raw_question_text": "Private Stage108 fixture question",
            "answer_doc_id": "private-doc-alpha",
        },
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
