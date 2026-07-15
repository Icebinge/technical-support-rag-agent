import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_failure_pattern_redesign_stop_decision import (
    decide_primeqa_hybrid_failure_pattern_redesign_stop,
    write_primeqa_hybrid_failure_pattern_redesign_stop_visualizations,
)


def test_failure_pattern_redesign_stop_decision_stops_family(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = decide_primeqa_hybrid_failure_pattern_redesign_stop(
        stage109_report_path=paths["stage109"],
        user_confirmed_stop=True,
        confirmation_note="unit test confirmation",
    )
    visualizations = write_primeqa_hybrid_failure_pattern_redesign_stop_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 110"
    assert report["decision"]["status"] == (
        "primeqa_hybrid_failure_pattern_redesign_family_stopped"
    )
    assert report["decision"]["stopped_family_id"] == (
        "failure_pattern_redesign_candidate_family"
    )
    assert report["decision"]["redesign_required_before_any_runtime_or_test_gate"]
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert report["stopped_family"]["stage109_summary"]["selectable_config_count"] == 0
    assert len(report["stopped_family"]["config_stop_evidence"]) == 7
    assert len(
        report["stopped_family"]["dev_improved_train_cv_nonselectable_configs"]
    ) == 6
    assert report["stopped_family"]["noop_blocked_configs"] == [
        {
            "config_id": "unit_noop_blocked",
            "candidate_family_id": "context_present_span_composer_candidate_v1",
            "train_cv_weighted_target_delta": 0.0,
            "train_cv_changed_answer_count": 0,
            "failed_train_cv_guards": ["train_cv_weighted_target_delta_negative"],
        }
    ]
    assert report["public_safe_contract"]["forbidden_keys_found"] == []
    assert "Private fixture answer text" not in serialized
    assert '"question_text":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage110_train_cv_weighted_target_deltas.svg",
        "stage110_dev_weighted_target_deltas.svg",
        "stage110_answerable_refusal_deltas.svg",
        "stage110_selectability_by_family.svg",
        "stage110_train_cv_guard_failure_reasons.svg",
        "stage110_stop_decision_flags.svg",
        "stage110_stop_guard_check_status.svg",
    }


def test_failure_pattern_redesign_stop_decision_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = decide_primeqa_hybrid_failure_pattern_redesign_stop(
        stage109_report_path=paths["stage109"],
        user_confirmed_stop=False,
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage110_stop_decision"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_failure_pattern_redesign_stop_decision_blocked"
    )


def test_failure_pattern_redesign_stop_decision_blocks_if_stage109_selected_config(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(
        tmp_path,
        stage109_status="primeqa_hybrid_failure_pattern_redesign_completed_dev_validation_passed",
        selected_config_id="unit_selected",
        selectable_config_count=1,
        dev_validation_passed=True,
        dev_validation_status=None,
    )

    report = decide_primeqa_hybrid_failure_pattern_redesign_stop(
        stage109_report_path=paths["stage109"],
        user_confirmed_stop=True,
        confirmation_note="unit test confirmation",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks[
        "stage109_completed_with_no_train_cv_selectable_config"
    ]["passed"] is False
    assert checks["stage109_selected_no_config"]["passed"] is False
    assert checks["stage109_dev_validation_has_no_selected_config"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_failure_pattern_redesign_stop_decision_blocked"
    )


def _write_fixture(
    tmp_path: Path,
    *,
    stage109_status: str = (
        "primeqa_hybrid_failure_pattern_redesign_completed_no_train_cv_selectable_config"
    ),
    selected_config_id: str | None = None,
    selectable_config_count: int = 0,
    dev_validation_passed: bool = False,
    dev_validation_status: str | None = "no_train_cv_selectable_config",
) -> dict[str, Path]:
    return {
        "stage109": _write_json(
            tmp_path / "stage109.json",
            _stage109_report(
                stage109_status=stage109_status,
                selected_config_id=selected_config_id,
                selectable_config_count=selectable_config_count,
                dev_validation_passed=dev_validation_passed,
                dev_validation_status=dev_validation_status,
            ),
        )
    }


def _stage109_report(
    *,
    stage109_status: str,
    selected_config_id: str | None,
    selectable_config_count: int,
    dev_validation_passed: bool,
    dev_validation_status: str | None,
) -> dict:
    return {
        "stage": "Stage 109",
        "analysis_id": (
            "primeqa_hybrid_failure_pattern_redesign_train_cv_dev_validation_v1"
        ),
        "split_contract": {
            "development_splits": ["train", "dev"],
            "selection_split": "train",
            "validation_split": "dev",
            "forbidden_final_splits": ["test"],
        },
        "stage108_summary": _stage108_summary(),
        "guard_checks": [
            {"name": "stage109_final_test_metrics_not_run", "passed": True},
            {"name": "stage109_runtime_defaults_unchanged", "passed": True},
            {"name": "stage109_fallback_strategies_not_added", "passed": True},
        ],
        "train_cv_selection": {
            "selection_split": "train",
            "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
            "baseline_train_cv_weighted_target_score": 100.0,
            "selected_config_id": selected_config_id,
            "selected_candidate_family_id": (
                "support_aware_answerability_gate_candidate_v1"
                if selected_config_id
                else None
            ),
            "selected_train_cv_weighted_target_delta": (
                -1.0 if selected_config_id else None
            ),
            "selectable_config_count": selectable_config_count,
            "config_count": 7,
        },
        "dev_validation": {
            "validation_split": "dev",
            "selected_config_id": selected_config_id,
            "status": dev_validation_status,
            "dev_validation_passed": dev_validation_passed,
        },
        "config_results": _config_results(selected_config_id),
        "decision": {
            "status": stage109_status,
            "recommended_next_direction": (
                "record_failure_pattern_redesign_stop_decision"
            ),
            "selected_config_id": selected_config_id,
            "selected_candidate_family_id": (
                "support_aware_answerability_gate_candidate_v1"
                if selected_config_id
                else None
            ),
            "selectable_config_count": selectable_config_count,
            "dev_validation_passed": dev_validation_passed,
            "dev_weighted_target_delta": -1.0 if selected_config_id else None,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "private_fixture_strings": ["Private fixture answer text"],
    }


def _stage108_summary() -> dict:
    return {
        "stage": "Stage 108",
        "protocol_id": "primeqa_hybrid_failure_pattern_redesign_protocol_v1",
        "decision_status": "primeqa_hybrid_failure_pattern_redesign_protocol_frozen",
        "candidate_config_count": 7,
        "train_selection_rule": {
            "selection_split": "train",
            "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
            "train_cv_fold_count": 5,
            "objective": {
                "requires_negative_train_cv_weighted_delta": True,
                "no_op_candidate_selectable": False,
            },
            "selectability_guards": {
                "max_train_cv_answerable_refusal_rate_delta": 0.02,
                "max_train_cv_average_token_f1_drop": 0.005,
                "max_train_cv_gold_doc_citation_rate_drop": 0.015,
                "max_train_cv_retrieval_context_miss_delta": 0,
            },
        },
        "dev_validation_rule": {
            "dev_selection_allowed": False,
            "dev_retuning_allowed": False,
            "dev_threshold_tuning_allowed": False,
            "test_access_allowed": False,
        },
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }


def _config_results(selected_config_id: str | None) -> list[dict]:
    configs = [
        _config(
            config_id="unit_joint_strong_blocked",
            family="joint_support_gate_span_composer_candidate_v1",
            component="joint_support_gate_span_composer",
            train_delta=-10.0,
            dev_delta=-3.0,
            changed_count=6,
            answerable_refusal_delta=0.30,
            failed_checks=["answerable_refusal_rate_delta_within_guard"],
        ),
        _config(
            config_id="unit_title_blocked",
            family="context_present_span_composer_candidate_v1",
            component="context_present_span_composer",
            train_delta=-8.0,
            dev_delta=-4.0,
            changed_count=5,
            answerable_refusal_delta=0.20,
            failed_checks=["answerable_refusal_rate_delta_within_guard"],
        ),
        _config(
            config_id="unit_joint_title_blocked",
            family="joint_support_gate_span_composer_candidate_v1",
            component="joint_support_gate_span_composer",
            train_delta=-7.0,
            dev_delta=-2.5,
            changed_count=5,
            answerable_refusal_delta=0.18,
            failed_checks=["answerable_refusal_rate_delta_within_guard"],
        ),
        _config(
            config_id="unit_support_strong_blocked",
            family="support_aware_answerability_gate_candidate_v1",
            component="support_aware_answerability_gate",
            train_delta=-6.0,
            dev_delta=-2.0,
            changed_count=4,
            answerable_refusal_delta=0.12,
            failed_checks=["answerable_refusal_rate_delta_within_guard"],
        ),
        _config(
            config_id="unit_support_mild_blocked",
            family="support_aware_answerability_gate_candidate_v1",
            component="support_aware_answerability_gate",
            train_delta=-2.0,
            dev_delta=-0.5,
            changed_count=2,
            answerable_refusal_delta=0.04,
            failed_checks=["answerable_refusal_rate_delta_within_guard"],
        ),
        _config(
            config_id="unit_context_regression_blocked",
            family="context_present_span_composer_candidate_v1",
            component="context_present_span_composer",
            train_delta=-1.0,
            dev_delta=-0.1,
            changed_count=3,
            answerable_refusal_delta=0.03,
            failed_checks=[
                "answerable_refusal_rate_delta_within_guard",
                "average_token_f1_drop_within_guard",
                "gold_doc_citation_rate_drop_within_guard",
            ],
        ),
        _config(
            config_id="unit_noop_blocked",
            family="context_present_span_composer_candidate_v1",
            component="context_present_span_composer",
            train_delta=0.0,
            dev_delta=0.0,
            changed_count=0,
            answerable_refusal_delta=0.0,
            failed_checks=["train_cv_weighted_target_delta_negative"],
        ),
    ]
    if selected_config_id:
        configs[0]["config_id"] = selected_config_id
        configs[0]["train_cv_selectability"]["selectable"] = True
        configs[0]["train_cv_selectability"]["checks"] = {
            key: True for key in configs[0]["train_cv_selectability"]["checks"]
        }
    return configs


def _config(
    *,
    config_id: str,
    family: str,
    component: str,
    train_delta: float,
    dev_delta: float,
    changed_count: int,
    answerable_refusal_delta: float,
    failed_checks: list[str],
) -> dict:
    checks = {
        "train_cv_weighted_target_delta_negative": train_delta < 0.0,
        "answerable_refusal_rate_delta_within_guard": True,
        "average_token_f1_drop_within_guard": True,
        "gold_doc_citation_rate_drop_within_guard": True,
        "retrieval_context_miss_delta_within_guard": True,
    }
    for name in failed_checks:
        checks[name] = False
    return {
        "config_id": config_id,
        "candidate_family_id": family,
        "component_family": component,
        "weighted_target_score_deltas_by_split": {
            "train_cv": train_delta,
            "train_full": train_delta,
            "dev": dev_delta,
        },
        "changed_answer_counts_by_split": {
            "train_cv": changed_count,
            "train_full": changed_count,
            "dev": changed_count,
        },
        "train_cv_selectability": {
            "selectable": False,
            "observed": {
                "train_cv_weighted_target_delta": train_delta,
                "answerable_refusal_rate_delta": answerable_refusal_delta,
                "average_token_f1_drop": 0.0,
                "gold_doc_citation_rate_drop": 0.0,
                "retrieval_context_miss_delta": 0,
            },
            "checks": checks,
        },
    }


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
