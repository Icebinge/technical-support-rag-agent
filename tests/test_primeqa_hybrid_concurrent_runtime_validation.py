from __future__ import annotations

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_validation import (
    _behavior_matches_stage143,
    _CohortArrivalGate,
    _concurrency_slo_pass,
    _ConcurrentValidationObservation,
    _cross_request_contamination_count,
    _decision,
    _policy_evidence,
    _public_safe_contract,
    _runtime_trace_violation_count,
    _source_checks,
    write_primeqa_hybrid_concurrent_runtime_validation_visualizations,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_validation_protocol import (
    ConcurrentRuntimeValidationState,
    StrictPracticalConcurrentRuntimeValidationPolicy,
)
from ts_rag_agent.domain.answer import AnswerCitation, GeneratedAnswer
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import PrimeQAHybridSplitSample


def test_stage145_source_checks_require_stage144_and_stage143() -> None:
    checks = _source_checks(
        stage144=_stage144(),
        stage143=_stage143(),
        user_confirmed_validation=True,
        confirmation_note="User confirmed Stage145 full concurrent validation.",
    )

    assert len(checks) == 8
    assert all(check["passed"] for check in checks)


def test_stage145_concurrent_trace_matches_exact_allowlist() -> None:
    runtime_trace = _runtime_trace()
    entrypoint_trace = {"terminal_state": "complete"}

    assert (
        _runtime_trace_violation_count(
            runtime_trace=runtime_trace,
            entrypoint_trace=entrypoint_trace,
        )
        == 0
    )

    runtime_trace["request_id"] = "private"
    assert (
        _runtime_trace_violation_count(
            runtime_trace=runtime_trace,
            entrypoint_trace=entrypoint_trace,
        )
        == 2
    )


def test_stage145_profile_b_latency_uses_both_percentiles() -> None:
    assert _concurrency_slo_pass({"p95": 0.8, "p99": 1.5})
    assert not _concurrency_slo_pass({"p95": 0.800001, "p99": 1.0})
    assert not _concurrency_slo_pass({"p95": 0.7, "p99": 1.500001})


def test_stage145_cross_request_digest_comparison_detects_drift() -> None:
    first = _observation(sample_id="sample-a", behavior_digest="stable")
    second = _observation(sample_id="sample-a", behavior_digest="stable")
    changed = _observation(sample_id="sample-a", behavior_digest="changed")
    other = _observation(sample_id="sample-b", behavior_digest="other")

    assert _cross_request_contamination_count([first, second, other]) == 0
    assert _cross_request_contamination_count([first, second, changed, other]) == 1


def test_stage145_cohort_workers_share_one_measured_clock_origin() -> None:
    arrival_gate = _CohortArrivalGate(4)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(arrival_gate.wait_for_release) for _ in range(4)]
        releases = tuple(future.result() for future in futures)

    assert len(set(releases)) == 1


def test_stage145_behavior_parity_requires_stage143_recall_and_agent_values() -> None:
    actual = _split_summary()
    expected = _split_summary()

    assert _behavior_matches_stage143(actual, expected)
    actual["verified_gold_citation_count"] = 150
    assert not _behavior_matches_stage143(actual, expected)


def test_stage145_real_policy_evidence_shape_is_eligible() -> None:
    train = _train_report()
    overload = _overload_report()
    dev = _dev_report()
    evidence = _policy_evidence(
        train_report=train,
        overload_report=overload,
        dev_report=dev,
        dev_loaded_after_train_gate=True,
        resource_summary=_resources(),
        resource_factory_build_count=1,
    )

    evaluation = StrictPracticalConcurrentRuntimeValidationPolicy().evaluate(evidence)

    assert evaluation.state is ConcurrentRuntimeValidationState.ELIGIBLE
    assert evaluation.rejection_reasons == ()
    assert evaluation.concurrent_runtime_activated is False


def test_stage145_public_contract_rejects_request_ids() -> None:
    blocked_contract = _public_safe_contract({"request_count": 4})
    assert blocked_contract["forbidden_keys_found"] == []
    assert blocked_contract["train_split_loaded"] is False
    assert blocked_contract["dev_split_loaded"] is False

    loaded_contract = _public_safe_contract(
        {
            "loaded_data_summary": {
                "train": {"row_count": 562},
                "dev": {"row_count": 121},
            }
        }
    )
    assert loaded_contract["train_split_loaded"] is True
    assert loaded_contract["dev_split_loaded"] is True
    assert _public_safe_contract({"nested": {"request_id": "private"}})["forbidden_keys_found"] == [
        "request_id"
    ]


def test_stage145_decision_and_visualizations(tmp_path: Path) -> None:
    checks = [{"name": "all", "passed": True}]
    decision = _decision(checks, ConcurrentRuntimeValidationState.ELIGIBLE)
    report = _visualization_report(decision=decision, checks=checks)

    visualizations = write_primeqa_hybrid_concurrent_runtime_validation_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert decision["concurrent_research_runtime_validation_passed"] is True
    assert decision["can_wire_explicit_nondefault_concurrent_runtime_now"] is True
    assert decision["concurrent_runtime_registered_for_application_use"] is False
    assert decision["test_gate_opened"] is False
    assert len(visualizations) == 10
    for visualization in visualizations:
        ET.parse(visualization.path)


def _observation(
    *,
    sample_id: str,
    behavior_digest: str,
) -> _ConcurrentValidationObservation:
    sample = PrimeQAHybridSplitSample(
        split_name="stage68",
        protocol_version="v1",
        assigned_split="train",
        split_subtype="grouped",
        source_split="train",
        sample_id=sample_id,
        question_id=f"question-{sample_id}",
        question_title="adapter token",
        question_text="How do I repair the adapter token?",
        answerable=True,
        answer="Apply the documented adapter token repair.",
        answer_doc_id="doc-001",
        candidate_doc_ids=(),
        start_offset=None,
        end_offset=None,
    )
    answer = GeneratedAnswer(
        question_id=sample_id,
        answer="Apply the documented adapter token repair.",
        citations=[
            AnswerCitation(
                document_id="doc-001",
                title="adapter token",
                retrieval_rank=1,
                evidence_score=1.0,
            )
        ],
        refused=False,
    )
    return _ConcurrentValidationObservation(
        sample=sample,
        verified_answer=answer,
        candidate_doc_ids=tuple(f"doc-{index:03d}" for index in range(1, 401)),
        arrival_pattern="synchronized_four_request_burst",
        repetition=1,
        target_arrival_offset_ms=0,
        actual_arrival_offset_ms=0.1,
        end_to_end_latency_seconds=0.2,
        retrieval_latency_seconds=0.1,
        runtime_trace=_runtime_trace(),
        entrypoint_trace={"terminal_state": "complete"},
        runtime_trace_violation_count=0,
        entrypoint_trace_violation_count=0,
        behavior_digest=behavior_digest,
    )


def _runtime_trace() -> dict:
    return {
        "runtime_mode": "optional_sidecar_agent_concurrent_four_request",
        "activation_requested": True,
        "activation_state": "eligible",
        "slo_profile_id": "strict_practical_b_concurrency4_v1",
        "warm_resources_ready": True,
        "concurrency_limit": 4,
        "in_flight_at_admission": 4,
        "admission_state": "admitted",
        "arrival_pattern": "synchronized_four_request_burst",
        "candidate_pool_depth": 400,
        "retrieval_latency_ms": 100.0,
        "end_to_end_latency_ms": 200.0,
        "latency_budget_passed": True,
        "terminal_state": "complete",
    }


def _split_summary() -> dict:
    return {
        "runtime_request_trace_violation_count": 0,
        "entrypoint_trace_violation_count": 0,
        "exact_five_transition_trace_rate": 1.0,
        "candidate_pool_depth": {"min": 400.0, "max": 400.0},
        "recall": {"hit_counts": {"10": 255, "50": 303, "100": 332, "200": 345, "400": 354}},
        "verified_metrics": {"average_token_f1": 0.1946},
        "verified_gold_citation_count": 151,
        "terminal_state_counts": {"complete": 560, "refuse": 2},
        "retry_action_count": 0,
        "fallback_action_count": 0,
    }


def _train_report() -> dict:
    latency = {"p95": 0.8, "p99": 1.5}
    return {
        "accepted_request_count": 3372,
        "synchronized_schedule_exact": True,
        "jittered_schedule_exact": True,
        "repetitions_per_pattern": {
            "synchronized_four_request_burst": 3,
            "deterministic_jitter_0_to_20ms": 3,
        },
        "latency_gate_scope_count": 39,
        "fold_pattern_repetition_reports": {
            f"fold_gate_{index}": {"slo_passed": True} for index in range(30)
        },
        "pass_reports": {f"pass_{index}": {"end_to_end_slo_passed": True} for index in range(6)},
        "pattern_pooled_reports": {
            "synchronized": {"slo_passed": True},
            "jitter": {"slo_passed": True},
        },
        "global_pooled_report": {
            "slo_passed": True,
            "end_to_end_latency_seconds": latency,
        },
        "behavior_invariants_passed": True,
        "cross_request_contamination_count": 0,
        "combined_runtime_summary": {
            "retry_action_count": 0,
            "fallback_action_count": 0,
        },
    }


def _overload_report() -> dict:
    return {
        "attempt_count": 5,
        "admitted_count": 4,
        "rejected_count": 1,
        "rejected_downstream_call_count": 0,
        "rejection_error_type": "PrimeQAHybridConcurrentCapacityExceededError",
        "queue_action_count": 0,
        "retry_action_count": 0,
        "fallback_action_count": 0,
    }


def _dev_report() -> dict:
    return {
        "row_count": 121,
        "measured_pass_count": 1,
        "end_to_end_slo_passed": True,
        "behavior_matches_stage143": True,
        "end_to_end_latency_seconds": {"p95": 0.8, "p99": 1.5},
        "retry_action_count": 0,
        "fallback_action_count": 0,
    }


def _resources() -> dict:
    return {
        "dense_model_count": 2,
        "dense_embedding_cache_count": 2,
        "lexical_index_count": 4,
        "derived_route_count": 1,
        "candidate_pool_retriever_instance_count": 1,
        "optional_entrypoint_instance_count": 1,
        "resources_built_or_loaded_per_request": False,
    }


def _stage144() -> dict:
    return {
        "stage": "Stage 144",
        "guard_checks": [{"passed": True}] * 29,
        "frozen_protocol": {
            "profile": {
                "profile_id": "strict_practical_b_concurrency4_v1",
                "max_in_flight": 4,
                "end_to_end_p95_seconds": 0.8,
                "end_to_end_p99_seconds": 1.5,
            },
            "train_validation_contract": {
                "accepted_requests_total": 3372,
                "complete_measured_pass_count": 6,
                "grouped_fold_count": 5,
                "total_latency_gate_scope_count": 39,
            },
            "overload_contract": {
                "probe_attempt_count": 5,
                "expected_admitted_count": 4,
                "expected_rejected_count": 1,
                "typed_error": "PrimeQAHybridConcurrentCapacityExceededError",
                "queue_allowed": False,
                "retry_allowed": False,
                "fallback_allowed": False,
            },
        },
        "decision": {
            "status": "primeqa_hybrid_concurrent_runtime_validation_protocol_frozen",
            "runtime_registered_as_default": False,
            "runtime_defaultization_allowed_now": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
        },
        "public_safe_contract": {
            "forbidden_keys_found": [],
            "test_split_loaded": False,
            "test_metrics_run": False,
        },
    }


def _stage143() -> dict:
    return {
        "stage": "Stage 143",
        "guard_checks": [{"passed": True}] * 28,
        "decision": {
            "status": "primeqa_hybrid_optional_sidecar_runtime_wiring_validation_passed",
            "single_request_runtime_validated": True,
            "runtime_registered_as_default": False,
            "default_runtime_policy": "unchanged",
        },
        "public_safe_contract": {
            "forbidden_keys_found": [],
            "test_split_loaded": False,
            "test_metrics_run": False,
        },
    }


def _visualization_report(*, decision: dict, checks: list[dict]) -> dict:
    pass_latency = {
        "end_to_end_latency_seconds": {"p50": 0.1, "p95": 0.2, "p99": 0.3, "max": 0.4},
        "throughput_requests_per_second": 10.0,
    }
    train = {
        "accepted_request_count": 3372,
        "pass_reports": {f"pass_{index}": pass_latency for index in range(6)},
        "fold_pattern_repetition_reports": {f"fold_{index}": pass_latency for index in range(30)},
        "pattern_pooled_reports": {
            "sync": pass_latency,
            "jitter": pass_latency,
        },
        "global_pooled_report": pass_latency,
        "all_latency_gate_scopes_passed": True,
        "behavior_invariants_passed": True,
        "cross_request_contamination_count": 0,
    }
    dev = {
        "row_count": 121,
        "end_to_end_latency_seconds": pass_latency["end_to_end_latency_seconds"],
        "end_to_end_slo_passed": True,
        "behavior_matches_stage143": True,
    }
    return {
        "warmup": {"request_count": 1},
        "overload_probe": {
            "attempt_count": 5,
            "admitted_count": 4,
            "rejected_count": 1,
            "rejected_downstream_call_count": 0,
        },
        "train_validation": train,
        "dev_report_only_validation": dev,
        "concurrency_policy_evaluation": {"state": "eligible"},
        "decision": decision,
        "guard_checks": checks,
    }
