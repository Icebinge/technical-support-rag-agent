import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_evidence_answerability_stop_decision import (
    decide_primeqa_hybrid_evidence_answerability_stop,
    write_primeqa_hybrid_evidence_answerability_stop_visualizations,
)


def test_evidence_answerability_stop_decision_stops_family(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)

    report = decide_primeqa_hybrid_evidence_answerability_stop(
        stage104_protocol_path=paths["stage104"],
        stage105_report_path=paths["stage105"],
        user_confirmed_stop=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = write_primeqa_hybrid_evidence_answerability_stop_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 106"
    assert report["decision"]["status"] == (
        "primeqa_hybrid_evidence_answerability_candidate_family_stopped"
    )
    assert report["decision"]["stopped_family_id"] == (
        "evidence_answerability_candidate_family"
    )
    assert report["decision"]["redesign_required_before_any_runtime_or_test_gate"]
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert report["stopped_family"]["stage105_summary"]["selected_config_id"] == (
        "unit_selected_noop"
    )
    assert report["stopped_family"]["dev_better_nonselectable_configs"] == [
        {
            "config_id": "unit_dev_better_blocked",
            "candidate_id": "joint_gate_then_window_candidate_v1",
            "dev_weighted_target_delta": -2.0,
            "train_weighted_target_delta": -5.0,
            "train_selectable": False,
            "failed_train_guards": [
                "answerable_refusal_rate_delta_within_guard",
            ],
        }
    ]
    assert "Private fixture answer text" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage106_evidence_answerability_target_deltas.svg",
        "stage106_train_selectability_by_family.svg",
        "stage106_train_guard_failure_reasons.svg",
        "stage106_stop_decision_flags.svg",
        "stage106_stop_guard_check_status.svg",
    }


def test_evidence_answerability_stop_decision_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = decide_primeqa_hybrid_evidence_answerability_stop(
        stage104_protocol_path=paths["stage104"],
        stage105_report_path=paths["stage105"],
        user_confirmed_stop=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage106_stop_decision"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_evidence_answerability_stop_decision_blocked"
    )


def test_evidence_answerability_stop_decision_blocks_if_stage105_passed_dev(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(
        tmp_path,
        stage105_status="primeqa_hybrid_evidence_answerability_comparison_completed",
        dev_validation_passed=True,
        dev_weighted_target_delta=-1.0,
    )

    report = decide_primeqa_hybrid_evidence_answerability_stop(
        stage104_protocol_path=paths["stage104"],
        stage105_report_path=paths["stage105"],
        user_confirmed_stop=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage105_completed_with_dev_guard_failed"]["passed"] is False
    assert checks["stage105_selected_config_failed_dev_validation"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_evidence_answerability_stop_decision_blocked"
    )


def _write_fixture(
    tmp_path: Path,
    *,
    stage105_status: str = (
        "primeqa_hybrid_evidence_answerability_comparison_completed_dev_guard_failed"
    ),
    dev_validation_passed: bool = False,
    dev_weighted_target_delta: float = 0.0,
) -> dict[str, Path]:
    return {
        "stage104": _write_json(tmp_path / "stage104.json", _stage104_report()),
        "stage105": _write_json(
            tmp_path / "stage105.json",
            _stage105_report(
                stage105_status=stage105_status,
                dev_validation_passed=dev_validation_passed,
                dev_weighted_target_delta=dev_weighted_target_delta,
            ),
        ),
    }


def _stage104_report() -> dict:
    return {
        "stage": "Stage 104",
        "protocol_id": "evidence_answerability_candidate_train_dev_comparison_v1",
        "decision": {
            "status": (
                "primeqa_hybrid_evidence_answerability_comparison_protocol_frozen"
            ),
            "recommended_direction": (
                "run_evidence_answerability_train_dev_candidate_comparison"
            ),
            "can_run_train_dev_candidate_comparison_after_user_confirmation": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "frozen_protocol": {
            "candidate_config_grid": [
                {
                    "config_id": f"unit_config_{index}",
                    "candidate_id": candidate_id,
                }
                for index, candidate_id in enumerate(
                    [
                        "answerability_margin_gate_candidate_v1",
                        "answerability_margin_gate_candidate_v1",
                        "answerability_margin_gate_candidate_v1",
                        "evidence_window_reselector_candidate_v1",
                        "evidence_window_reselector_candidate_v1",
                        "evidence_window_reselector_candidate_v1",
                        "joint_gate_then_window_candidate_v1",
                        "joint_gate_then_window_candidate_v1",
                        "joint_gate_then_window_candidate_v1",
                    ],
                    start=1,
                )
            ],
            "train_selection_rule": {
                "selection_split": "train",
                "validation_split": "dev",
                "dev_threshold_tuning_allowed": False,
                "test_access_allowed": False,
            },
            "dev_validation_rule": {
                "dev_selection_allowed": False,
                "dev_retuning_allowed": False,
                "test_access_allowed": False,
            },
            "runtime_feature_contract": {
                "prohibited_runtime_inputs": [
                    "gold answers",
                    "answer document identifiers",
                    "source DOC_IDS",
                ]
            },
            "fallback_strategy_policy": {
                "fallback_strategies_enabled": False,
            },
        },
    }


def _stage105_report(
    *,
    stage105_status: str,
    dev_validation_passed: bool,
    dev_weighted_target_delta: float,
) -> dict:
    return {
        "stage": "Stage 105",
        "analysis_id": "evidence_answerability_candidate_train_dev_comparison_v1",
        "guard_checks": [
            {"name": "stage105_final_test_metrics_not_run", "passed": True},
            {"name": "stage105_default_runtime_policy_unchanged", "passed": True},
        ],
        "train_selection": {
            "selection_split": "train",
            "selected_config_id": "unit_selected_noop",
            "selected_candidate_id": "answerability_margin_gate_candidate_v1",
            "selected_train_weighted_target_delta": 0.0,
            "selectable_config_count": 1,
            "config_count": 2,
        },
        "dev_validation": {
            "selected_config_id": "unit_selected_noop",
            "dev_validation_passed": dev_validation_passed,
            "dev_weighted_target_delta": dev_weighted_target_delta,
            "dev_changed_answer_count": 0,
            "dev_target_bucket_deltas": {
                "answerability_false_answer": 0,
                "gold_span_beats_selected_answer": 0,
                "evidence_selection_miss": 0,
            },
            "dev_metric_deltas": {
                "answerable_refusal_rate": 0.0,
                "gold_doc_citation_rate": 0.0,
                "average_token_f1": 0.0,
            },
        },
        "config_results": [
            _config_result(
                config_id="unit_selected_noop",
                candidate_id="answerability_margin_gate_candidate_v1",
                train_delta=0.0,
                dev_delta=dev_weighted_target_delta,
                selectable=True,
            ),
            _config_result(
                config_id="unit_dev_better_blocked",
                candidate_id="joint_gate_then_window_candidate_v1",
                train_delta=-5.0,
                dev_delta=-2.0,
                selectable=False,
            ),
        ],
        "decision": {
            "status": stage105_status,
            "recommended_next_direction": "evidence_answerability_stop_decision",
            "selected_config_id": "unit_selected_noop",
            "selected_candidate_id": "answerability_margin_gate_candidate_v1",
            "selectable_config_count": 1,
            "dev_validation_passed": dev_validation_passed,
            "dev_weighted_target_delta": dev_weighted_target_delta,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "private_fixture_strings": ["Private fixture answer text"],
    }


def _config_result(
    *,
    config_id: str,
    candidate_id: str,
    train_delta: float,
    dev_delta: float,
    selectable: bool,
) -> dict:
    checks = {
        "answerable_refusal_rate_delta_within_guard": selectable,
        "average_token_f1_drop_within_guard": True,
        "gold_doc_citation_rate_drop_within_guard": True,
    }
    return {
        "config_id": config_id,
        "candidate_id": candidate_id,
        "weighted_target_score_deltas_by_split": {
            "train": train_delta,
            "dev": dev_delta,
        },
        "train_selectability": {
            "selectable": selectable,
            "checks": checks,
        },
    }


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
