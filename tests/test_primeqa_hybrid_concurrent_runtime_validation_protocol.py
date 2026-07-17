import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_validation_protocol import (
    ConcurrentRuntimeValidationEvidence,
    ConcurrentRuntimeValidationState,
    StrictPracticalConcurrentRuntimeValidationPolicy,
    _forbidden_keys_found,
    freeze_primeqa_hybrid_concurrent_runtime_validation_protocol,
    write_primeqa_hybrid_concurrent_runtime_protocol_visualizations,
)


def test_profile_b_policy_accepts_exact_boundaries_without_activating() -> None:
    evaluation = StrictPracticalConcurrentRuntimeValidationPolicy().evaluate(_compliant_evidence())

    assert evaluation.state is ConcurrentRuntimeValidationState.ELIGIBLE
    assert evaluation.rejection_reasons == ()
    assert evaluation.concurrent_runtime_activated is False


def test_profile_b_policy_rejects_latency_and_missing_percentile_evidence() -> None:
    evaluation = StrictPracticalConcurrentRuntimeValidationPolicy().evaluate(
        _compliant_evidence(
            train_end_to_end_p95_seconds=0.800001,
            train_end_to_end_p99_seconds=None,
            dev_end_to_end_slo_passed=False,
            dev_end_to_end_p99_seconds=None,
        )
    )

    assert evaluation.state is ConcurrentRuntimeValidationState.REJECTED
    assert "train_end_to_end_p95_exceeds_slo" in evaluation.rejection_reasons
    assert "train_end_to_end_p99_missing" in evaluation.rejection_reasons
    assert "dev_report_only_slo_failed" in evaluation.rejection_reasons
    assert "dev_end_to_end_p99_missing" in evaluation.rejection_reasons


def test_profile_b_policy_rejects_unsafe_overload_behavior() -> None:
    evaluation = StrictPracticalConcurrentRuntimeValidationPolicy().evaluate(
        _compliant_evidence(
            overload_admitted_count=5,
            overload_rejected_count=0,
            overload_rejected_before_downstream=False,
            overload_error_type="RuntimeError",
            queue_action_count=1,
        )
    )

    assert evaluation.state is ConcurrentRuntimeValidationState.REJECTED
    assert "overload_probe_admitted_count_not_four" in evaluation.rejection_reasons
    assert "overload_probe_rejected_count_not_one" in evaluation.rejection_reasons
    assert "overload_rejection_reached_downstream" in evaluation.rejection_reasons
    assert "overload_rejection_error_type_mismatch" in evaluation.rejection_reasons
    assert "queue_action_detected" in evaluation.rejection_reasons


def test_profile_b_policy_rejects_resource_data_and_default_boundary_failures() -> None:
    evaluation = StrictPracticalConcurrentRuntimeValidationPolicy().evaluate(
        _compliant_evidence(
            process_resource_inventory_preserved=False,
            request_local_state_isolated=False,
            synchronized_arrival_schedule_exact=False,
            train_accepted_request_count=3371,
            train_latency_gate_scope_count=38,
            train_behavior_invariants_passed=False,
            dev_loaded_after_train_gate=False,
            dev_report_only_pass_count=2,
            dev_accepted_request_count=120,
            dev_behavior_invariants_passed=False,
            test_split_locked=False,
            runtime_default_unchanged=False,
            retry_action_count=1,
            fallback_action_count=1,
        )
    )

    assert evaluation.state is ConcurrentRuntimeValidationState.REJECTED
    assert "process_resource_inventory_not_preserved" in evaluation.rejection_reasons
    assert "request_local_state_not_isolated" in evaluation.rejection_reasons
    assert "synchronized_arrival_schedule_mismatch" in evaluation.rejection_reasons
    assert "train_accepted_request_count_not_3372" in evaluation.rejection_reasons
    assert "train_latency_gate_scope_count_not_39" in evaluation.rejection_reasons
    assert "train_behavior_invariants_failed" in evaluation.rejection_reasons
    assert "dev_loaded_before_train_gate" in evaluation.rejection_reasons
    assert "dev_report_only_pass_count_not_one" in evaluation.rejection_reasons
    assert "dev_accepted_request_count_not_121" in evaluation.rejection_reasons
    assert "dev_behavior_invariants_failed" in evaluation.rejection_reasons
    assert "test_split_not_locked" in evaluation.rejection_reasons
    assert "runtime_default_changed" in evaluation.rejection_reasons
    assert "retry_action_detected" in evaluation.rejection_reasons
    assert "fallback_action_detected" in evaluation.rejection_reasons


def test_stage144_freeze_passes_without_claiming_concurrent_implementation(
    tmp_path: Path,
) -> None:
    source_path = _write_stage143_source(tmp_path)

    report = freeze_primeqa_hybrid_concurrent_runtime_validation_protocol(
        stage143_validation_path=source_path,
        user_confirmed_protocol=True,
        confirmation_note="User selected strict practical profile B.",
        selected_profile_id="strict_practical_b_concurrency4_v1",
    )

    assert len(report["guard_checks"]) == 29
    assert all(check["passed"] for check in report["guard_checks"])
    decision = report["decision"]
    assert decision["concurrent_runtime_validation_protocol_frozen"] is True
    assert decision["concurrency_validation_policy_executable"] is True
    assert decision["concurrent_runtime_implemented_now"] is False
    assert decision["concurrent_runtime_validation_run"] is False
    assert decision["concurrent_runtime_activation_allowed_now"] is False
    assert decision["test_gate_opened"] is False
    assert decision["default_runtime_policy"] == "unchanged"
    train = report["frozen_protocol"]["train_validation_contract"]
    assert train["accepted_requests_total"] == 3372
    assert train["total_latency_gate_scope_count"] == 39
    assert report["public_safe_contract"]["forbidden_keys_found"] == []


def test_stage144_freeze_blocks_unconfirmed_or_different_profile(tmp_path: Path) -> None:
    source_path = _write_stage143_source(tmp_path)

    report = freeze_primeqa_hybrid_concurrent_runtime_validation_protocol(
        stage143_validation_path=source_path,
        user_confirmed_protocol=False,
        confirmation_note="",
        selected_profile_id="different_profile",
    )

    assert report["decision"]["concurrent_runtime_validation_protocol_frozen"] is False
    assert "stage144_user_confirmed_profile_b" in report["decision"]["failed_checks"]


def test_stage144_freeze_blocks_stage143_resource_or_boundary_drift(tmp_path: Path) -> None:
    source = _stage143_source()
    source["resource_summary"]["dense_model_count"] = 3
    source["decision"]["concurrent_runtime_activation_allowed"] = True
    source_path = tmp_path / "stage143.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    report = freeze_primeqa_hybrid_concurrent_runtime_validation_protocol(
        stage143_validation_path=source_path,
        user_confirmed_protocol=True,
        confirmation_note="User selected strict practical profile B.",
        selected_profile_id="strict_practical_b_concurrency4_v1",
    )

    failed = report["decision"]["failed_checks"]
    assert "stage143_process_resource_inventory_is_complete" in failed
    assert "stage143_concurrency_default_test_retry_fallback_remain_closed" in failed


def test_stage144_public_safety_checks_exact_keys() -> None:
    assert _forbidden_keys_found({"request_count": 4, "cohort_size": 4}) == set()
    assert _forbidden_keys_found({"nested": {"request_id": "private"}}) == {"request_id"}
    assert _forbidden_keys_found({"nested": {"cohort_id": "private"}}) == {"cohort_id"}


def test_stage144_writes_all_visualizations(tmp_path: Path) -> None:
    report = freeze_primeqa_hybrid_concurrent_runtime_validation_protocol(
        stage143_validation_path=_write_stage143_source(tmp_path),
        user_confirmed_protocol=True,
        confirmation_note="User selected strict practical profile B.",
        selected_profile_id="strict_practical_b_concurrency4_v1",
    )

    visualizations = write_primeqa_hybrid_concurrent_runtime_protocol_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    assert {visualization.name for visualization in visualizations} == {
        "stage144_end_to_end_latency_slo.svg",
        "stage144_train_request_budget.svg",
        "stage144_arrival_pattern_offsets.svg",
        "stage144_latency_gate_matrix.svg",
        "stage144_overload_contract.svg",
        "stage144_process_resource_inventory.svg",
        "stage144_decision_flags.svg",
        "stage144_guard_check_status.svg",
    }
    assert all(Path(visualization.path).is_file() for visualization in visualizations)


def _compliant_evidence(**overrides: object) -> ConcurrentRuntimeValidationEvidence:
    values = {
        "profile_id": "strict_practical_b_concurrency4_v1",
        "warm_single_process": True,
        "max_in_flight": 4,
        "synchronized_arrival_schedule_exact": True,
        "jittered_arrival_schedule_exact": True,
        "synchronized_train_repetitions": 3,
        "jittered_train_repetitions": 3,
        "train_accepted_request_count": 3372,
        "train_fold_count": 5,
        "train_latency_gate_scope_count": 39,
        "train_fold_pattern_repetition_gates_passed": True,
        "train_pass_aggregate_gates_passed": True,
        "train_pattern_pooled_gates_passed": True,
        "train_global_pooled_gate_passed": True,
        "train_behavior_invariants_passed": True,
        "train_end_to_end_p95_seconds": 0.8,
        "train_end_to_end_p99_seconds": 1.5,
        "overload_attempt_count": 5,
        "overload_admitted_count": 4,
        "overload_rejected_count": 1,
        "overload_rejected_before_downstream": True,
        "overload_error_type": "PrimeQAHybridConcurrentCapacityExceededError",
        "queue_action_count": 0,
        "retry_action_count": 0,
        "fallback_action_count": 0,
        "process_resource_inventory_preserved": True,
        "request_local_state_isolated": True,
        "dev_loaded_after_train_gate": True,
        "dev_report_only_pass_count": 1,
        "dev_accepted_request_count": 121,
        "dev_end_to_end_slo_passed": True,
        "dev_behavior_invariants_passed": True,
        "dev_end_to_end_p95_seconds": 0.8,
        "dev_end_to_end_p99_seconds": 1.5,
        "test_split_locked": True,
        "runtime_default_unchanged": True,
    }
    values.update(overrides)
    return ConcurrentRuntimeValidationEvidence(**values)  # type: ignore[arg-type]


def _write_stage143_source(tmp_path: Path) -> Path:
    path = tmp_path / "stage143.json"
    path.write_text(json.dumps(_stage143_source()), encoding="utf-8")
    return path


def _stage143_source() -> dict:
    resources = {
        "dense_model_count": 2,
        "dense_embedding_cache_count": 2,
        "lexical_index_count": 4,
        "derived_route_count": 1,
        "candidate_pool_retriever_instance_count": 1,
        "optional_entrypoint_instance_count": 1,
        "resources_built_or_loaded_per_request": False,
    }
    return {
        "stage": "Stage 143",
        "analysis_id": "primeqa_hybrid_optional_sidecar_runtime_wiring_validation_v1",
        "runtime_contract": {
            "runtime_mode": "optional_sidecar_agent_single_request",
            "default_enabled": False,
            "explicit_true_required": True,
            "single_request_only": True,
            "concurrent_request_support_authorized": False,
            "registered_as_runtime_default": False,
            "test_access_allowed": False,
            "retry_actions_allowed": False,
            "fallback_strategies_allowed": False,
            "errors_propagate": True,
        },
        "resource_summary": resources,
        "resource_factory_build_count": 1,
        "train_runtime_validation": _split_report(562, 0.104243, 0.152497),
        "train_fold_reports": {f"fold_{index}": {} for index in range(1, 6)},
        "train_gate_passed_before_dev": True,
        "dev_loaded_only_after_train_gate": True,
        "dev_runtime_report_only_validation": _split_report(121, 0.094431, 0.123178),
        "guard_checks": [
            {"name": f"stage143_guard_{index}", "passed": True} for index in range(28)
        ],
        "decision": {
            "status": "primeqa_hybrid_optional_sidecar_runtime_wiring_validation_passed",
            "optional_runtime_wiring_implemented": True,
            "optional_runtime_activation_validated": True,
            "single_request_runtime_validated": True,
            "concurrent_runtime_activation_allowed": False,
            "runtime_registered_as_default": False,
            "runtime_defaultization_allowed_now": False,
            "test_gate_opened": False,
            "test_metrics_run": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "public_safe_contract": {
            "test_split_loaded": False,
            "test_metrics_run": False,
            "forbidden_keys_found": [],
        },
    }


def _split_report(row_count: int, p95: float, p99: float) -> dict:
    return {
        "row_count": row_count,
        "runtime_request_trace_violation_count": 0,
        "entrypoint_trace_violation_count": 0,
        "exact_five_transition_trace_rate": 1.0,
        "candidate_pool_depth": {"min": 400.0, "max": 400.0},
        "retrieval_latency_seconds": {"p95": p95, "p99": p99},
        "terminal_state_counts": {"complete": row_count, "refuse": 0},
        "retry_action_count": 0,
        "fallback_action_count": 0,
        "recall": {"hit_counts": {"400": 1}, "rates": {"400": 1.0}},
        "verified_metrics": {"average_token_f1": 0.1},
        "verified_gold_citation_count": 1,
    }
