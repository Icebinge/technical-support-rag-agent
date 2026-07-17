from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_activation_validation import (
    _activation_checks,
    _decision,
    _public_safe_contract,
    _source_checks,
    _stage145_baseline_adapter,
    write_primeqa_hybrid_concurrent_runtime_activation_validation_visualizations,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_validation_protocol import (
    ConcurrentRuntimeValidationState,
)


def test_stage146_source_checks_require_final_stage145_evidence() -> None:
    stage145 = _stage145_source()

    checks = _source_checks(
        stage145=stage145,
        source_evidence={"profile_id": "strict_practical_b_concurrency4_v1"},
        source_evaluation=_eligible_evaluation(),
        user_confirmed_validation=True,
        confirmation_note="User confirmed Stage146 application activation validation.",
    )

    assert len(checks) == 7
    assert all(check["passed"] for check in checks)

    stage145["guard_checks"][0]["passed"] = False
    failed = _source_checks(
        stage145=stage145,
        source_evidence={"profile_id": "strict_practical_b_concurrency4_v1"},
        source_evaluation=_eligible_evaluation(),
        user_confirmed_validation=True,
        confirmation_note="User confirmed Stage146 application activation validation.",
    )
    assert failed[1]["passed"] is False


def test_stage146_activation_checks_cover_all_startup_states() -> None:
    checks = _activation_checks(payload=_activation_payload())

    assert len(checks) == 8
    assert all(check["passed"] for check in checks)


def test_stage146_baseline_adapter_uses_first_stage145_pass_and_dev() -> None:
    stage145 = {
        "train_validation": {
            "pass_execution_order": ["first", "second"],
            "pass_reports": {
                "first": {"verified_gold_citation_count": 151},
                "second": {"verified_gold_citation_count": 151},
            },
        },
        "dev_report_only_validation": {"verified_gold_citation_count": 33},
    }

    baseline = _stage145_baseline_adapter(stage145)

    assert baseline["train_runtime_validation"]["verified_gold_citation_count"] == 151
    assert baseline["dev_runtime_report_only_validation"]["verified_gold_citation_count"] == 33


def test_stage146_decision_requires_guards_policy_and_active_startup() -> None:
    checks = [{"name": "all", "passed": True}]

    decision = _decision(
        checks=checks,
        current_policy_state=ConcurrentRuntimeValidationState.ELIGIBLE,
        eligible_startup_active=True,
    )
    blocked = _decision(
        checks=checks,
        current_policy_state=ConcurrentRuntimeValidationState.ELIGIBLE,
        eligible_startup_active=False,
    )

    assert decision["explicit_nondefault_concurrent_activation_available"] is True
    assert decision["concurrent_runtime_activation_allowed_now"] is True
    assert decision["runtime_registered_as_default"] is False
    assert decision["test_gate_opened"] is False
    assert blocked["explicit_nondefault_concurrent_activation_available"] is False


def test_stage146_public_contract_reflects_loaded_splits() -> None:
    blocked = _public_safe_contract({})
    loaded = _public_safe_contract(
        {
            "loaded_data_summary": {
                "train": {"row_count": 562},
                "dev": {"row_count": 121},
            }
        }
    )

    assert blocked["train_split_loaded"] is False
    assert blocked["dev_split_loaded"] is False
    assert loaded["train_split_loaded"] is True
    assert loaded["dev_split_loaded"] is True
    assert loaded["test_split_loaded"] is False
    assert loaded["synthetic_rejected_source_persisted"] is False


def test_stage146_visualizations_parse(tmp_path: Path) -> None:
    report = _visualization_report()

    visualizations = write_primeqa_hybrid_concurrent_runtime_activation_validation_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert len(visualizations) == 9
    for visualization in visualizations:
        ET.parse(visualization.path)


def _stage145_source() -> dict:
    evidence = {"profile_id": "strict_practical_b_concurrency4_v1"}
    evaluation = _eligible_evaluation()
    return {
        "stage": "Stage 145",
        "analysis_id": "primeqa_hybrid_concurrent_runtime_train_cv_dev_validation_v1",
        "guard_checks": [
            {"name": f"stage145_guard_{index}", "passed": True} for index in range(36)
        ],
        "concurrency_policy_evidence": evidence,
        "concurrency_policy_evaluation": evaluation,
        "decision": {
            "status": "primeqa_hybrid_concurrent_runtime_train_cv_dev_validation_passed",
            "concurrent_research_runtime_validation_passed": True,
            "can_wire_explicit_nondefault_concurrent_runtime_now": True,
            "runtime_registered_as_default": False,
            "queue_actions_enabled": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
        },
        "public_safe_contract": {
            "test_split_loaded": False,
            "test_metrics_run": False,
            "forbidden_keys_found": [],
        },
    }


def _eligible_evaluation() -> dict:
    return {
        "state": "eligible",
        "rejection_reasons": [],
        "concurrent_runtime_activated": False,
    }


def _activation_payload() -> dict:
    common = {
        "registered_as_runtime_default": False,
        "test_access_allowed": False,
        "queue_action_count": 0,
        "retry_action_count": 0,
        "fallback_action_count": 0,
    }
    return {
        "activation_contract": {
            "settings_field": "enable_concurrent_sidecar_agent",
            "environment_flag": "TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT",
            "default_enabled": False,
            "explicit_true_required": True,
            "mutually_exclusive_with": "enable_optional_sidecar_agent",
            "registered_as_runtime_default": False,
            "test_access_allowed": False,
            "queue_actions_allowed": False,
            "retry_actions_allowed": False,
            "fallback_strategies_allowed": False,
        },
        "configuration_mutual_exclusion_enforced": True,
        "startup_cases": {
            "disabled": {
                **common,
                "activation_state": "disabled",
                "activation_requested": False,
                "source_validation_state": "not_evaluated_disabled",
                "resources_initialized": False,
                "runtime_activated": False,
                "warmup_request_count": 0,
            },
            "rejected": {
                **common,
                "activation_state": "rejected",
                "activation_requested": True,
                "resources_initialized": False,
                "runtime_activated": False,
                "warmup_request_count": 0,
                "rejection_reasons": [
                    "stage145_saved_evidence_mismatch",
                    "train_end_to_end_p95_exceeds_slo",
                ],
            },
            "eligible": {
                **common,
                "activation_state": "eligible",
                "source_validation_state": "eligible",
                "resources_initialized": True,
                "runtime_activated": True,
                "resource_factory_build_count": 1,
                "warmup_request_count": 1,
                "warmup_arrival_pattern": "warmup_single_request",
                "warmup_candidate_pool_depth": 400,
            },
        },
        "disabled_resource_factory_build_count": 0,
        "rejected_resource_factory_build_count": 0,
        "source_unchanged_after_synthetic_case": True,
        "synthetic_rejected_case": {
            "source_report_modified": False,
            "persisted": False,
        },
    }


def _visualization_report() -> dict:
    latency = {"p95": 0.6, "p99": 0.8}
    scope = {"end_to_end_latency_seconds": latency}
    decision = _decision(
        checks=[{"name": "all", "passed": True}],
        current_policy_state=ConcurrentRuntimeValidationState.ELIGIBLE,
        eligible_startup_active=True,
    )
    return {
        "startup_cases": _activation_payload()["startup_cases"],
        "disabled_resource_factory_build_count": 0,
        "rejected_resource_factory_build_count": 0,
        "resource_factory_build_count": 1,
        "train_validation": {
            "pattern_pooled_reports": {
                "synchronized": scope,
                "jitter": scope,
            },
            "fold_pattern_repetition_reports": {"fold": scope},
            "pass_reports": {"pass": scope},
            "global_pooled_report": scope,
            "combined_runtime_summary": {
                "arrival_offset_error_ms": {
                    "average": 1.0,
                    "p95": 2.0,
                    "p99": 3.0,
                    "max": 4.0,
                }
            },
        },
        "dev_report_only_validation": {"end_to_end_latency_seconds": latency},
        "overload_probe": {
            "attempt_count": 5,
            "admitted_count": 4,
            "rejected_count": 1,
            "rejected_downstream_call_count": 0,
        },
        "decision": decision,
        "guard_checks": [{"name": "all", "passed": True}],
    }
