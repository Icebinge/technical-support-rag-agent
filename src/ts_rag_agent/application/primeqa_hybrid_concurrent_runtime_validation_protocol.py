from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 144"
_CREATED_AT = "2026-07-17"
_PROTOCOL_ID = "primeqa_hybrid_concurrent_runtime_validation_protocol_v1"
_PROFILE_ID = "strict_practical_b_concurrency4_v1"
_SOURCE_STAGE143_ANALYSIS_ID = "primeqa_hybrid_optional_sidecar_runtime_wiring_validation_v1"
_SOURCE_STAGE143_STATUS = "primeqa_hybrid_optional_sidecar_runtime_wiring_validation_passed"
_NEXT_DIRECTION = "implement_and_run_strict_practical_concurrency4_train_cv_dev_validation"
_TRAIN_ROW_COUNT = 562
_DEV_ROW_COUNT = 121
_TRAIN_FOLD_COUNT = 5
_TRAIN_REPETITIONS_PER_PATTERN = 3
_MAX_IN_FLIGHT = 4
_JITTER_OFFSETS_MS = (0, 7, 13, 20)
_EXPECTED_STAGE143_GUARDS = 28
_CAPACITY_ERROR_TYPE = "PrimeQAHybridConcurrentCapacityExceededError"
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "answer_doc_id",
        "answer_text",
        "candidate_doc_ids",
        "cited_doc_ids",
        "cohort_id",
        "document_body",
        "document_id",
        "document_text",
        "document_title",
        "gold_answer",
        "matched_token_strings",
        "question_id",
        "question_text",
        "question_title",
        "raw_answer_text",
        "raw_document_text",
        "raw_question_text",
        "request_id",
        "retrieved_doc_ids",
        "runtime_content_handle",
        "sample_id",
        "source_doc_ids",
    }
)


class ConcurrentRuntimeValidationState(str, Enum):
    """Outcomes of the executable Stage144 concurrency evidence policy."""

    REJECTED = "rejected"
    ELIGIBLE = "eligible"


@dataclass(frozen=True)
class StrictPracticalConcurrencySlo:
    """User-selected profile B for a warm process with four in-flight requests."""

    profile_id: str = _PROFILE_ID
    max_in_flight: int = _MAX_IN_FLIGHT
    end_to_end_p95_seconds: float = 0.8
    end_to_end_p99_seconds: float = 1.5


@dataclass(frozen=True)
class ConcurrentRuntimeValidationEvidence:
    """Aggregate-only evidence required before concurrent activation is eligible."""

    profile_id: str
    warm_single_process: bool
    max_in_flight: int
    synchronized_arrival_schedule_exact: bool
    jittered_arrival_schedule_exact: bool
    synchronized_train_repetitions: int
    jittered_train_repetitions: int
    train_accepted_request_count: int
    train_fold_count: int
    train_latency_gate_scope_count: int
    train_fold_pattern_repetition_gates_passed: bool
    train_pass_aggregate_gates_passed: bool
    train_pattern_pooled_gates_passed: bool
    train_global_pooled_gate_passed: bool
    train_behavior_invariants_passed: bool
    train_end_to_end_p95_seconds: float | None
    train_end_to_end_p99_seconds: float | None
    overload_attempt_count: int
    overload_admitted_count: int
    overload_rejected_count: int
    overload_rejected_before_downstream: bool
    overload_error_type: str
    queue_action_count: int
    retry_action_count: int
    fallback_action_count: int
    process_resource_inventory_preserved: bool
    request_local_state_isolated: bool
    dev_loaded_after_train_gate: bool
    dev_report_only_pass_count: int
    dev_accepted_request_count: int
    dev_end_to_end_slo_passed: bool
    dev_behavior_invariants_passed: bool
    dev_end_to_end_p95_seconds: float | None
    dev_end_to_end_p99_seconds: float | None
    test_split_locked: bool
    runtime_default_unchanged: bool


@dataclass(frozen=True)
class ConcurrentRuntimeValidationEvaluation:
    """Public-safe policy result; eligibility never activates runtime by itself."""

    state: ConcurrentRuntimeValidationState
    rejection_reasons: tuple[str, ...]
    concurrent_runtime_activated: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "rejection_reasons": list(self.rejection_reasons),
            "concurrent_runtime_activated": self.concurrent_runtime_activated,
        }


class StrictPracticalConcurrentRuntimeValidationPolicy:
    """Fail-closed B-profile policy with no queue, retries, or fallback."""

    def __init__(self, slo: StrictPracticalConcurrencySlo | None = None) -> None:
        self._slo = slo or StrictPracticalConcurrencySlo()

    def evaluate(
        self,
        evidence: ConcurrentRuntimeValidationEvidence,
    ) -> ConcurrentRuntimeValidationEvaluation:
        reasons: list[str] = []
        if evidence.profile_id != self._slo.profile_id:
            reasons.append("concurrency_profile_mismatch")
        if not evidence.warm_single_process:
            reasons.append("warm_single_process_evidence_missing")
        if evidence.max_in_flight != self._slo.max_in_flight:
            reasons.append("max_in_flight_not_four")
        if not evidence.synchronized_arrival_schedule_exact:
            reasons.append("synchronized_arrival_schedule_mismatch")
        if not evidence.jittered_arrival_schedule_exact:
            reasons.append("jittered_arrival_schedule_mismatch")
        if evidence.synchronized_train_repetitions != _TRAIN_REPETITIONS_PER_PATTERN:
            reasons.append("synchronized_train_repetitions_not_three")
        if evidence.jittered_train_repetitions != _TRAIN_REPETITIONS_PER_PATTERN:
            reasons.append("jittered_train_repetitions_not_three")
        if evidence.train_accepted_request_count != 3372:
            reasons.append("train_accepted_request_count_not_3372")
        if evidence.train_fold_count != _TRAIN_FOLD_COUNT:
            reasons.append("train_grouped_five_fold_evidence_missing")
        if evidence.train_latency_gate_scope_count != 39:
            reasons.append("train_latency_gate_scope_count_not_39")
        if not evidence.train_fold_pattern_repetition_gates_passed:
            reasons.append("train_fold_pattern_repetition_gate_failed")
        if not evidence.train_pass_aggregate_gates_passed:
            reasons.append("train_pass_aggregate_gate_failed")
        if not evidence.train_pattern_pooled_gates_passed:
            reasons.append("train_pattern_pooled_gate_failed")
        if not evidence.train_global_pooled_gate_passed:
            reasons.append("train_global_pooled_gate_failed")
        if not evidence.train_behavior_invariants_passed:
            reasons.append("train_behavior_invariants_failed")
        _append_latency_reason(
            reasons,
            split="train",
            percentile="p95",
            observed=evidence.train_end_to_end_p95_seconds,
            limit=self._slo.end_to_end_p95_seconds,
        )
        _append_latency_reason(
            reasons,
            split="train",
            percentile="p99",
            observed=evidence.train_end_to_end_p99_seconds,
            limit=self._slo.end_to_end_p99_seconds,
        )
        if evidence.overload_attempt_count != _MAX_IN_FLIGHT + 1:
            reasons.append("overload_probe_attempt_count_not_five")
        if evidence.overload_admitted_count != _MAX_IN_FLIGHT:
            reasons.append("overload_probe_admitted_count_not_four")
        if evidence.overload_rejected_count != 1:
            reasons.append("overload_probe_rejected_count_not_one")
        if not evidence.overload_rejected_before_downstream:
            reasons.append("overload_rejection_reached_downstream")
        if evidence.overload_error_type != _CAPACITY_ERROR_TYPE:
            reasons.append("overload_rejection_error_type_mismatch")
        if evidence.queue_action_count != 0:
            reasons.append("queue_action_detected")
        if evidence.retry_action_count != 0:
            reasons.append("retry_action_detected")
        if evidence.fallback_action_count != 0:
            reasons.append("fallback_action_detected")
        if not evidence.process_resource_inventory_preserved:
            reasons.append("process_resource_inventory_not_preserved")
        if not evidence.request_local_state_isolated:
            reasons.append("request_local_state_not_isolated")
        if not evidence.dev_loaded_after_train_gate:
            reasons.append("dev_loaded_before_train_gate")
        if evidence.dev_report_only_pass_count != 1:
            reasons.append("dev_report_only_pass_count_not_one")
        if evidence.dev_accepted_request_count != _DEV_ROW_COUNT:
            reasons.append("dev_accepted_request_count_not_121")
        if not evidence.dev_end_to_end_slo_passed:
            reasons.append("dev_report_only_slo_failed")
        if not evidence.dev_behavior_invariants_passed:
            reasons.append("dev_behavior_invariants_failed")
        _append_latency_reason(
            reasons,
            split="dev",
            percentile="p95",
            observed=evidence.dev_end_to_end_p95_seconds,
            limit=self._slo.end_to_end_p95_seconds,
        )
        _append_latency_reason(
            reasons,
            split="dev",
            percentile="p99",
            observed=evidence.dev_end_to_end_p99_seconds,
            limit=self._slo.end_to_end_p99_seconds,
        )
        if not evidence.test_split_locked:
            reasons.append("test_split_not_locked")
        if not evidence.runtime_default_unchanged:
            reasons.append("runtime_default_changed")

        if reasons:
            return ConcurrentRuntimeValidationEvaluation(
                state=ConcurrentRuntimeValidationState.REJECTED,
                rejection_reasons=tuple(reasons),
            )
        return ConcurrentRuntimeValidationEvaluation(
            state=ConcurrentRuntimeValidationState.ELIGIBLE,
            rejection_reasons=(),
        )


@dataclass(frozen=True)
class PrimeQAHybridConcurrentRuntimeProtocolVisualization:
    """One generated Stage144 protocol visualization."""

    name: str
    path: str


def freeze_primeqa_hybrid_concurrent_runtime_validation_protocol(
    *,
    stage143_validation_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
    selected_profile_id: str,
) -> dict[str, Any]:
    """Freeze profile B from the saved public-safe Stage143 aggregate."""

    started_at = time.perf_counter()
    source = _load_json_object(stage143_validation_path)
    loaded_at = time.perf_counter()
    source_summary = _stage143_summary(source)
    slo = StrictPracticalConcurrencySlo()
    frozen_protocol = _frozen_protocol(slo)
    canonical_evaluations = _canonical_evaluations(
        StrictPracticalConcurrentRuntimeValidationPolicy(slo)
    )
    preliminary = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "protocol_scope": (
            "Aggregate-only freeze of the user-selected strict practical B concurrency "
            "validation contract. It reads only the saved Stage143 report and executes "
            "synthetic policy cases. It does not load train, dev, test, questions, "
            "documents, models, indexes, or candidate pools; implement concurrent runtime; "
            "run concurrent requests; change defaults; or add queues, retries, or fallback."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_protocol),
            "confirmation_note": confirmation_note,
            "selected_profile_id": selected_profile_id,
        },
        "source_files": {"stage143_validation": _fingerprint(stage143_validation_path)},
        "stage143_summary": source_summary,
        "benchmark_machine": _benchmark_machine(),
        "frozen_protocol": frozen_protocol,
        "canonical_validation_evaluations": canonical_evaluations,
    }
    guards = _guard_checks(
        report=preliminary,
        source_summary=source_summary,
        frozen_protocol=frozen_protocol,
        canonical_evaluations=canonical_evaluations,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
        selected_profile_id=selected_profile_id,
    )
    checked_at = time.perf_counter()
    report = {
        **preliminary,
        "guard_checks": guards,
        "decision": _decision(guards),
        "timing_seconds": {
            "load_public_stage143_aggregate": round(loaded_at - started_at, 6),
            "freeze_and_guard": round(checked_at - loaded_at, 6),
            "total": round(checked_at - started_at, 6),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_concurrent_runtime_protocol_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridConcurrentRuntimeProtocolVisualization]:
    """Write aggregate-only SVG charts for the frozen Stage144 contract."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage144_end_to_end_latency_slo.svg": render_horizontal_bar_chart_svg(
            title="Stage144 profile B end-to-end latency SLO",
            bars=_latency_slo_bars(report),
            x_label="seconds",
            width=1360,
            margin_left=660,
        ),
        "stage144_train_request_budget.svg": render_horizontal_bar_chart_svg(
            title="Stage144 measured train request budget",
            bars=_train_request_budget_bars(report),
            x_label="accepted requests",
            width=1420,
            margin_left=690,
        ),
        "stage144_arrival_pattern_offsets.svg": render_horizontal_bar_chart_svg(
            title="Stage144 deterministic cohort arrival offsets",
            bars=_arrival_offset_bars(report),
            x_label="milliseconds",
            width=1420,
            margin_left=720,
        ),
        "stage144_latency_gate_matrix.svg": render_horizontal_bar_chart_svg(
            title="Stage144 train latency gate scopes",
            bars=_latency_gate_bars(report),
            x_label="gate scopes",
            width=1400,
            margin_left=700,
        ),
        "stage144_overload_contract.svg": render_horizontal_bar_chart_svg(
            title="Stage144 five-request overload contract",
            bars=_overload_bars(report),
            x_label="requests",
            width=1420,
            margin_left=720,
        ),
        "stage144_process_resource_inventory.svg": render_horizontal_bar_chart_svg(
            title="Stage144 preserved process resource inventory",
            bars=_resource_bars(report),
            x_label="instances",
            width=1500,
            margin_left=780,
        ),
        "stage144_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage144 protocol and runtime decision flags",
            bars=_decision_bars(report),
            x_label="1 means true",
            width=1700,
            margin_left=900,
        ),
        "stage144_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage144 protocol guard checks",
            bars=[
                BarDatum(
                    label=str(check["name"]),
                    value=1.0 if check["passed"] else 0.0,
                    value_label="passed" if check["passed"] else "failed",
                )
                for check in report.get("guard_checks", [])
            ],
            x_label="1 means passed",
            width=2320,
            margin_left=1320,
        ),
    }
    artifacts: list[PrimeQAHybridConcurrentRuntimeProtocolVisualization] = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridConcurrentRuntimeProtocolVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _append_latency_reason(
    reasons: list[str],
    *,
    split: str,
    percentile: str,
    observed: float | None,
    limit: float,
) -> None:
    if observed is None:
        reasons.append(f"{split}_end_to_end_{percentile}_missing")
    elif observed > limit:
        reasons.append(f"{split}_end_to_end_{percentile}_exceeds_slo")


def _stage143_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision") or {}
    guards = report.get("guard_checks") or []
    runtime = report.get("runtime_contract") or {}
    resources = report.get("resource_summary") or {}
    train = report.get("train_runtime_validation") or {}
    dev = report.get("dev_runtime_report_only_validation") or {}
    public_safe = report.get("public_safe_contract") or {}
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "status": decision.get("status"),
        "guard_check_count": len(guards),
        "guard_check_passed_count": sum(bool(check.get("passed")) for check in guards),
        "optional_runtime_wiring_implemented": decision.get("optional_runtime_wiring_implemented"),
        "optional_runtime_activation_validated": decision.get(
            "optional_runtime_activation_validated"
        ),
        "single_request_runtime_validated": decision.get("single_request_runtime_validated"),
        "concurrent_runtime_activation_allowed": decision.get(
            "concurrent_runtime_activation_allowed"
        ),
        "runtime_registered_as_default": decision.get("runtime_registered_as_default"),
        "runtime_defaultization_allowed_now": decision.get("runtime_defaultization_allowed_now"),
        "test_gate_opened": decision.get("test_gate_opened"),
        "test_metrics_run": decision.get("test_metrics_run"),
        "retry_actions_enabled": decision.get("retry_actions_enabled"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "runtime_contract": {
            key: runtime.get(key)
            for key in (
                "runtime_mode",
                "default_enabled",
                "explicit_true_required",
                "single_request_only",
                "concurrent_request_support_authorized",
                "registered_as_runtime_default",
                "test_access_allowed",
                "retry_actions_allowed",
                "fallback_strategies_allowed",
                "errors_propagate",
            )
        },
        "resource_summary": dict(resources),
        "resource_factory_build_count": report.get("resource_factory_build_count"),
        "train": _source_split_summary(train),
        "train_fold_count": len(report.get("train_fold_reports") or {}),
        "train_gate_passed_before_dev": report.get("train_gate_passed_before_dev"),
        "dev_loaded_only_after_train_gate": report.get("dev_loaded_only_after_train_gate"),
        "dev": _source_split_summary(dev),
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
        "source_test_split_loaded": public_safe.get("test_split_loaded"),
        "source_test_metrics_run": public_safe.get("test_metrics_run"),
    }


def _source_split_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "row_count": report.get("row_count"),
        "runtime_request_trace_violation_count": report.get(
            "runtime_request_trace_violation_count"
        ),
        "entrypoint_trace_violation_count": report.get("entrypoint_trace_violation_count"),
        "exact_five_transition_trace_rate": report.get("exact_five_transition_trace_rate"),
        "candidate_pool_depth": dict(report.get("candidate_pool_depth") or {}),
        "retrieval_latency_seconds": dict(report.get("retrieval_latency_seconds") or {}),
        "terminal_state_counts": dict(report.get("terminal_state_counts") or {}),
        "retry_action_count": report.get("retry_action_count"),
        "fallback_action_count": report.get("fallback_action_count"),
        "recall": dict(report.get("recall") or {}),
        "verified_metrics": dict(report.get("verified_metrics") or {}),
        "verified_gold_citation_count": report.get("verified_gold_citation_count"),
    }


def _frozen_protocol(slo: StrictPracticalConcurrencySlo) -> dict[str, Any]:
    train_requests_per_pattern = _TRAIN_ROW_COUNT * _TRAIN_REPETITIONS_PER_PATTERN
    full_cohorts, final_cohort_size = divmod(_TRAIN_ROW_COUNT, _MAX_IN_FLIGHT)
    dev_full_cohorts, dev_final_cohort_size = divmod(_DEV_ROW_COUNT, _MAX_IN_FLIGHT)
    return {
        "protocol_id": _PROTOCOL_ID,
        "profile": {
            **asdict(slo),
            "profile_name": "strict_practical_b",
            "hardware_scope": "current_benchmark_machine_only",
            "process_state": "one_warm_process",
            "startup_in_end_to_end_slo": False,
            "startup_reported_separately": True,
            "percentile_method": "linear_interpolation_at_(n_minus_1)_times_p",
        },
        "latency_measurement_contract": {
            "primary_metric": "request_end_to_end_latency_seconds",
            "measurement_start": "before_nonblocking_admission_attempt",
            "measurement_end": "complete_refuse_or_capacity_rejection_terminal_outcome",
            "accepted_request_scope": (
                "admission_plus_candidate_retrieval_plus_answer_plus_verify_plus_observe"
            ),
            "retrieval_latency_role": "secondary_diagnostic_only",
            "retrieval_latency_additional_gate": False,
            "accepted_and_capacity_rejected_distributions_reported_separately": True,
        },
        "arrival_patterns": {
            "synchronized_four_request_burst": {
                "pattern_id": "synchronized_four_request_burst",
                "cohort_capacity": _MAX_IN_FLIGHT,
                "offsets_ms": [0, 0, 0, 0],
            },
            "deterministic_jitter_0_to_20ms": {
                "pattern_id": "deterministic_jitter_0_to_20ms",
                "cohort_capacity": _MAX_IN_FLIGHT,
                "offsets_ms": list(_JITTER_OFFSETS_MS),
                "schedule_reused_for_each_cohort": True,
            },
        },
        "train_validation_contract": {
            "row_count_per_complete_pass": _TRAIN_ROW_COUNT,
            "grouped_fold_count": _TRAIN_FOLD_COUNT,
            "repetitions_per_arrival_pattern": _TRAIN_REPETITIONS_PER_PATTERN,
            "arrival_pattern_count": 2,
            "complete_measured_pass_count": 6,
            "accepted_requests_per_pattern": train_requests_per_pattern,
            "accepted_requests_total": train_requests_per_pattern * 2,
            "full_four_request_cohorts_per_pass": full_cohorts,
            "final_cohort_size_per_pass": final_cohort_size,
            "fold_pattern_repetition_gate_count": 30,
            "pass_aggregate_gate_count": 6,
            "pattern_pooled_gate_count": 2,
            "global_pooled_gate_count": 1,
            "total_latency_gate_scope_count": 39,
            "every_gate_requires_p95_and_p99": True,
            "all_gates_must_pass": True,
            "train_gate_must_pass_before_dev_load": True,
            "behavior_invariants": {
                "runtime_request_trace_violation_count": 0,
                "entrypoint_trace_violation_count": 0,
                "cross_request_contamination_count": 0,
                "candidate_pool_depth_exactly_400": True,
                "recall_hit_counts_must_match_stage143": True,
                "terminal_counts_must_match_stage143": True,
                "verified_f1_and_gold_citations_must_match_stage143": True,
                "retry_and_fallback_action_counts": 0,
            },
        },
        "overload_contract": {
            "probe_attempt_count": 5,
            "expected_admitted_count": 4,
            "expected_rejected_count": 1,
            "validation_harness_barrier": (
                "hold_four_admitted requests after admission and before retrieval while "
                "the fifth request attempts admission"
            ),
            "rejection_timing": "before_retrieval_agent_or_any_other_downstream_call",
            "typed_error": _CAPACITY_ERROR_TYPE,
            "numeric_rejection_latency_limit_invented": False,
            "proof": "zero downstream calls for the rejected request",
            "queue_allowed": False,
            "retry_allowed": False,
            "fallback_allowed": False,
        },
        "dev_report_only_contract": {
            "load_only_after_complete_train_gate": True,
            "row_count": _DEV_ROW_COUNT,
            "measured_pass_count": 1,
            "schedule": (
                "alternate synchronized and deterministic-jitter cohorts beginning with "
                "synchronized"
            ),
            "full_four_request_cohorts": dev_full_cohorts,
            "final_cohort_size": dev_final_cohort_size,
            "p95_and_p99_must_pass": True,
            "behavior_invariants_must_match_stage143": True,
            "selection_or_retuning_allowed": False,
        },
        "resource_safety_contract": {
            "process_resource_inventory_must_match_stage143": True,
            "resource_factory_build_count": 1,
            "resources_built_or_loaded_per_request": False,
            "admission_controller_instance_count": 1,
            "admission_capacity": _MAX_IN_FLIGHT,
            "shared_long_lived_candidate_retriever": True,
            "request_local_retrieval_profile": True,
            "request_local_agent_state_machine": True,
            "request_local_entrypoint_execution": True,
            "shared_pending_retrieval_profile_allowed": False,
            "cross_request_trace_or_result_contamination_allowed": False,
            "exceptions_propagate_to_originating_request": True,
        },
        "public_request_trace_contract": {
            "allowed_fields": [
                "runtime_mode",
                "activation_requested",
                "activation_state",
                "slo_profile_id",
                "warm_resources_ready",
                "concurrency_limit",
                "in_flight_at_admission",
                "admission_state",
                "arrival_pattern",
                "candidate_pool_depth",
                "retrieval_latency_ms",
                "end_to_end_latency_ms",
                "latency_budget_passed",
                "terminal_state",
            ],
            "question_answer_or_document_content_allowed": False,
            "question_sample_document_request_or_cohort_ids_allowed": False,
            "private_per_request_traces_written": False,
        },
        "locked_boundaries": {
            "concurrent_runtime_implemented_in_this_stage": False,
            "concurrent_runtime_validation_run": False,
            "concurrent_runtime_activation_allowed_now": False,
            "runtime_registered_as_default": False,
            "runtime_defaultization_allowed_now": False,
            "test_split_loaded": False,
            "test_metrics_run": False,
            "queue_actions_enabled": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
    }


def _canonical_evaluations(
    policy: StrictPracticalConcurrentRuntimeValidationPolicy,
) -> dict[str, Any]:
    compliant = _compliant_synthetic_evidence()
    latency_failure = ConcurrentRuntimeValidationEvidence(
        **{
            **asdict(compliant),
            "train_end_to_end_p95_seconds": 0.800001,
            "dev_end_to_end_p99_seconds": None,
        }
    )
    overload_failure = ConcurrentRuntimeValidationEvidence(
        **{
            **asdict(compliant),
            "overload_admitted_count": 5,
            "overload_rejected_count": 0,
            "overload_rejected_before_downstream": False,
            "queue_action_count": 1,
        }
    )
    boundary_failure = ConcurrentRuntimeValidationEvidence(
        **{
            **asdict(compliant),
            "test_split_locked": False,
            "runtime_default_unchanged": False,
            "retry_action_count": 1,
            "fallback_action_count": 1,
        }
    )
    return {
        "synthetic_exact_boundary_compliant": policy.evaluate(compliant).to_public_dict(),
        "synthetic_latency_failure": policy.evaluate(latency_failure).to_public_dict(),
        "synthetic_overload_failure": policy.evaluate(overload_failure).to_public_dict(),
        "synthetic_locked_boundary_failure": policy.evaluate(boundary_failure).to_public_dict(),
    }


def _compliant_synthetic_evidence() -> ConcurrentRuntimeValidationEvidence:
    return ConcurrentRuntimeValidationEvidence(
        profile_id=_PROFILE_ID,
        warm_single_process=True,
        max_in_flight=_MAX_IN_FLIGHT,
        synchronized_arrival_schedule_exact=True,
        jittered_arrival_schedule_exact=True,
        synchronized_train_repetitions=_TRAIN_REPETITIONS_PER_PATTERN,
        jittered_train_repetitions=_TRAIN_REPETITIONS_PER_PATTERN,
        train_accepted_request_count=3372,
        train_fold_count=_TRAIN_FOLD_COUNT,
        train_latency_gate_scope_count=39,
        train_fold_pattern_repetition_gates_passed=True,
        train_pass_aggregate_gates_passed=True,
        train_pattern_pooled_gates_passed=True,
        train_global_pooled_gate_passed=True,
        train_behavior_invariants_passed=True,
        train_end_to_end_p95_seconds=0.8,
        train_end_to_end_p99_seconds=1.5,
        overload_attempt_count=5,
        overload_admitted_count=4,
        overload_rejected_count=1,
        overload_rejected_before_downstream=True,
        overload_error_type=_CAPACITY_ERROR_TYPE,
        queue_action_count=0,
        retry_action_count=0,
        fallback_action_count=0,
        process_resource_inventory_preserved=True,
        request_local_state_isolated=True,
        dev_loaded_after_train_gate=True,
        dev_report_only_pass_count=1,
        dev_accepted_request_count=121,
        dev_end_to_end_slo_passed=True,
        dev_behavior_invariants_passed=True,
        dev_end_to_end_p95_seconds=0.8,
        dev_end_to_end_p99_seconds=1.5,
        test_split_locked=True,
        runtime_default_unchanged=True,
    )


def _guard_checks(
    *,
    report: Mapping[str, Any],
    source_summary: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
    canonical_evaluations: Mapping[str, Any],
    user_confirmed_protocol: bool,
    confirmation_note: str,
    selected_profile_id: str,
) -> list[dict[str, Any]]:
    profile = frozen_protocol.get("profile") or {}
    latency = frozen_protocol.get("latency_measurement_contract") or {}
    patterns = frozen_protocol.get("arrival_patterns") or {}
    train = frozen_protocol.get("train_validation_contract") or {}
    train_behavior = train.get("behavior_invariants") or {}
    overload = frozen_protocol.get("overload_contract") or {}
    dev = frozen_protocol.get("dev_report_only_contract") or {}
    resources = frozen_protocol.get("resource_safety_contract") or {}
    trace = frozen_protocol.get("public_request_trace_contract") or {}
    locked = frozen_protocol.get("locked_boundaries") or {}
    source_runtime = source_summary.get("runtime_contract") or {}
    source_resources = source_summary.get("resource_summary") or {}
    compliant = canonical_evaluations.get("synthetic_exact_boundary_compliant") or {}
    latency_failure = canonical_evaluations.get("synthetic_latency_failure") or {}
    overload_failure = canonical_evaluations.get("synthetic_overload_failure") or {}
    boundary_failure = canonical_evaluations.get("synthetic_locked_boundary_failure") or {}
    return [
        _check(
            "stage144_user_confirmed_profile_b",
            user_confirmed_protocol
            and bool(confirmation_note.strip())
            and selected_profile_id == _PROFILE_ID,
            {
                "confirmed": user_confirmed_protocol,
                "note_present": bool(confirmation_note.strip()),
                "profile": selected_profile_id,
            },
            _PROFILE_ID,
        ),
        _check(
            "stage143_source_identity_and_status_passed",
            source_summary.get("stage") == "Stage 143"
            and source_summary.get("analysis_id") == _SOURCE_STAGE143_ANALYSIS_ID
            and source_summary.get("status") == _SOURCE_STAGE143_STATUS,
            {
                "stage": source_summary.get("stage"),
                "analysis_id": source_summary.get("analysis_id"),
                "status": source_summary.get("status"),
            },
            _SOURCE_STAGE143_STATUS,
        ),
        _check(
            "stage143_all_guards_passed",
            source_summary.get("guard_check_count") == _EXPECTED_STAGE143_GUARDS
            and source_summary.get("guard_check_passed_count") == _EXPECTED_STAGE143_GUARDS,
            {
                "count": source_summary.get("guard_check_count"),
                "passed": source_summary.get("guard_check_passed_count"),
            },
            _EXPECTED_STAGE143_GUARDS,
        ),
        _check(
            "stage143_single_request_runtime_is_validated",
            source_summary.get("optional_runtime_wiring_implemented") is True
            and source_summary.get("optional_runtime_activation_validated") is True
            and source_summary.get("single_request_runtime_validated") is True,
            {
                "wiring": source_summary.get("optional_runtime_wiring_implemented"),
                "activation": source_summary.get("optional_runtime_activation_validated"),
                "single_request": source_summary.get("single_request_runtime_validated"),
            },
            True,
        ),
        _check(
            "stage143_concurrency_default_test_retry_fallback_remain_closed",
            source_summary.get("concurrent_runtime_activation_allowed") is False
            and source_summary.get("runtime_registered_as_default") is False
            and source_summary.get("runtime_defaultization_allowed_now") is False
            and source_summary.get("test_gate_opened") is False
            and source_summary.get("test_metrics_run") is False
            and source_summary.get("retry_actions_enabled") is False
            and source_summary.get("fallback_strategies_enabled") is False
            and source_summary.get("default_runtime_policy") == "unchanged",
            "Stage143 boundary flags",
            "all closed",
        ),
        _check(
            "stage143_runtime_contract_is_single_request_only",
            source_runtime.get("default_enabled") is False
            and source_runtime.get("single_request_only") is True
            and source_runtime.get("concurrent_request_support_authorized") is False
            and source_runtime.get("errors_propagate") is True,
            source_runtime,
            "single-request fail-closed source",
        ),
        _check(
            "stage143_process_resource_inventory_is_complete",
            source_resources
            == {
                "dense_model_count": 2,
                "dense_embedding_cache_count": 2,
                "lexical_index_count": 4,
                "derived_route_count": 1,
                "candidate_pool_retriever_instance_count": 1,
                "optional_entrypoint_instance_count": 1,
                "resources_built_or_loaded_per_request": False,
            }
            and source_summary.get("resource_factory_build_count") == 1,
            {
                "resources": source_resources,
                "build_count": source_summary.get("resource_factory_build_count"),
            },
            "Stage143 exact inventory and one build",
        ),
        _check(
            "stage143_train_dev_order_and_counts_are_exact",
            (source_summary.get("train") or {}).get("row_count") == _TRAIN_ROW_COUNT
            and source_summary.get("train_fold_count") == _TRAIN_FOLD_COUNT
            and source_summary.get("train_gate_passed_before_dev") is True
            and source_summary.get("dev_loaded_only_after_train_gate") is True
            and (source_summary.get("dev") or {}).get("row_count") == _DEV_ROW_COUNT,
            {
                "train_rows": (source_summary.get("train") or {}).get("row_count"),
                "folds": source_summary.get("train_fold_count"),
                "dev_rows": (source_summary.get("dev") or {}).get("row_count"),
            },
            {"train": 562, "folds": 5, "dev": 121},
        ),
        _check(
            "stage143_source_is_public_safe_and_test_locked",
            source_summary.get("public_safe_forbidden_keys_found") == []
            and source_summary.get("source_test_split_loaded") is False
            and source_summary.get("source_test_metrics_run") is False,
            {
                "forbidden": source_summary.get("public_safe_forbidden_keys_found"),
                "test_loaded": source_summary.get("source_test_split_loaded"),
                "test_run": source_summary.get("source_test_metrics_run"),
            },
            "forbidden empty and test false",
        ),
        _check(
            "profile_b_warm_machine_and_concurrency_are_exact",
            profile.get("profile_id") == _PROFILE_ID
            and profile.get("max_in_flight") == 4
            and profile.get("hardware_scope") == "current_benchmark_machine_only"
            and profile.get("process_state") == "one_warm_process"
            and profile.get("startup_in_end_to_end_slo") is False,
            profile,
            "profile B, warm current machine, concurrency four",
        ),
        _check(
            "profile_b_end_to_end_slo_is_exact",
            profile.get("end_to_end_p95_seconds") == 0.8
            and profile.get("end_to_end_p99_seconds") == 1.5,
            {
                "p95": profile.get("end_to_end_p95_seconds"),
                "p99": profile.get("end_to_end_p99_seconds"),
            },
            {"p95": 0.8, "p99": 1.5},
        ),
        _check(
            "end_to_end_measurement_includes_complete_agent_path",
            latency.get("measurement_start") == "before_nonblocking_admission_attempt"
            and latency.get("measurement_end")
            == "complete_refuse_or_capacity_rejection_terminal_outcome"
            and latency.get("accepted_request_scope")
            == "admission_plus_candidate_retrieval_plus_answer_plus_verify_plus_observe"
            and latency.get("retrieval_latency_role") == "secondary_diagnostic_only"
            and latency.get("retrieval_latency_additional_gate") is False,
            latency,
            "full request path; retrieval diagnostic only",
        ),
        _check(
            "synchronized_arrival_pattern_is_exact",
            (patterns.get("synchronized_four_request_burst") or {}).get("offsets_ms")
            == [0, 0, 0, 0],
            patterns.get("synchronized_four_request_burst"),
            [0, 0, 0, 0],
        ),
        _check(
            "jitter_arrival_pattern_is_deterministic_0_to_20ms",
            (patterns.get("deterministic_jitter_0_to_20ms") or {}).get("offsets_ms")
            == [0, 7, 13, 20]
            and (patterns.get("deterministic_jitter_0_to_20ms") or {}).get(
                "schedule_reused_for_each_cohort"
            )
            is True,
            patterns.get("deterministic_jitter_0_to_20ms"),
            [0, 7, 13, 20],
        ),
        _check(
            "train_runs_three_complete_passes_per_pattern",
            train.get("repetitions_per_arrival_pattern") == 3
            and train.get("complete_measured_pass_count") == 6
            and train.get("accepted_requests_per_pattern") == 1686
            and train.get("accepted_requests_total") == 3372,
            {
                "repetitions": train.get("repetitions_per_arrival_pattern"),
                "passes": train.get("complete_measured_pass_count"),
                "per_pattern": train.get("accepted_requests_per_pattern"),
                "total": train.get("accepted_requests_total"),
            },
            {"repetitions": 3, "passes": 6, "per_pattern": 1686, "total": 3372},
        ),
        _check(
            "train_cohort_shape_is_exact",
            train.get("full_four_request_cohorts_per_pass") == 140
            and train.get("final_cohort_size_per_pass") == 2,
            {
                "full": train.get("full_four_request_cohorts_per_pass"),
                "final": train.get("final_cohort_size_per_pass"),
            },
            {"full": 140, "final": 2},
        ),
        _check(
            "train_latency_gate_matrix_is_complete",
            train.get("fold_pattern_repetition_gate_count") == 30
            and train.get("pass_aggregate_gate_count") == 6
            and train.get("pattern_pooled_gate_count") == 2
            and train.get("global_pooled_gate_count") == 1
            and train.get("total_latency_gate_scope_count") == 39
            and train.get("every_gate_requires_p95_and_p99") is True
            and train.get("all_gates_must_pass") is True
            and train_behavior.get("runtime_request_trace_violation_count") == 0
            and train_behavior.get("entrypoint_trace_violation_count") == 0
            and train_behavior.get("cross_request_contamination_count") == 0
            and train_behavior.get("candidate_pool_depth_exactly_400") is True
            and train_behavior.get("recall_hit_counts_must_match_stage143") is True
            and train_behavior.get("terminal_counts_must_match_stage143") is True
            and train_behavior.get("verified_f1_and_gold_citations_must_match_stage143") is True
            and train_behavior.get("retry_and_fallback_action_counts") == 0,
            {
                "fold": train.get("fold_pattern_repetition_gate_count"),
                "pass": train.get("pass_aggregate_gate_count"),
                "pattern": train.get("pattern_pooled_gate_count"),
                "global": train.get("global_pooled_gate_count"),
                "total": train.get("total_latency_gate_scope_count"),
                "behavior_invariants": train_behavior,
            },
            39,
        ),
        _check(
            "overload_probe_admits_four_and_rejects_one",
            overload.get("probe_attempt_count") == 5
            and overload.get("expected_admitted_count") == 4
            and overload.get("expected_rejected_count") == 1,
            {
                "attempts": overload.get("probe_attempt_count"),
                "admitted": overload.get("expected_admitted_count"),
                "rejected": overload.get("expected_rejected_count"),
            },
            {"attempts": 5, "admitted": 4, "rejected": 1},
        ),
        _check(
            "overload_rejection_is_typed_and_before_downstream",
            overload.get("rejection_timing")
            == "before_retrieval_agent_or_any_other_downstream_call"
            and overload.get("typed_error") == _CAPACITY_ERROR_TYPE
            and overload.get("numeric_rejection_latency_limit_invented") is False
            and overload.get("proof") == "zero downstream calls for the rejected request",
            overload,
            "typed pre-downstream rejection proven by zero calls",
        ),
        _check(
            "queue_retry_and_fallback_are_forbidden",
            overload.get("queue_allowed") is False
            and overload.get("retry_allowed") is False
            and overload.get("fallback_allowed") is False,
            {
                "queue": overload.get("queue_allowed"),
                "retry": overload.get("retry_allowed"),
                "fallback": overload.get("fallback_allowed"),
            },
            False,
        ),
        _check(
            "resource_inventory_is_shared_and_request_state_is_local",
            resources.get("process_resource_inventory_must_match_stage143") is True
            and resources.get("resource_factory_build_count") == 1
            and resources.get("resources_built_or_loaded_per_request") is False
            and resources.get("admission_capacity") == 4
            and resources.get("request_local_retrieval_profile") is True
            and resources.get("request_local_agent_state_machine") is True
            and resources.get("request_local_entrypoint_execution") is True
            and resources.get("shared_pending_retrieval_profile_allowed") is False
            and resources.get("cross_request_trace_or_result_contamination_allowed") is False,
            resources,
            "shared heavy resources and request-local mutable state",
        ),
        _check(
            "public_trace_is_allowlisted_without_request_or_content_ids",
            set(trace.get("allowed_fields") or [])
            == {
                "runtime_mode",
                "activation_requested",
                "activation_state",
                "slo_profile_id",
                "warm_resources_ready",
                "concurrency_limit",
                "in_flight_at_admission",
                "admission_state",
                "arrival_pattern",
                "candidate_pool_depth",
                "retrieval_latency_ms",
                "end_to_end_latency_ms",
                "latency_budget_passed",
                "terminal_state",
            }
            and trace.get("question_answer_or_document_content_allowed") is False
            and trace.get("question_sample_document_request_or_cohort_ids_allowed") is False
            and trace.get("private_per_request_traces_written") is False,
            trace,
            "exact aggregate-safe allowlist",
        ),
        _check(
            "dev_is_one_mixed_report_only_pass_after_train",
            dev.get("load_only_after_complete_train_gate") is True
            and dev.get("row_count") == 121
            and dev.get("measured_pass_count") == 1
            and dev.get("full_four_request_cohorts") == 30
            and dev.get("final_cohort_size") == 1
            and dev.get("p95_and_p99_must_pass") is True
            and dev.get("behavior_invariants_must_match_stage143") is True
            and dev.get("selection_or_retuning_allowed") is False,
            dev,
            "one locked mixed dev pass after train",
        ),
        _check(
            "current_runtime_test_default_queue_retry_fallback_remain_closed",
            all(
                locked.get(key) is False
                for key in (
                    "concurrent_runtime_implemented_in_this_stage",
                    "concurrent_runtime_validation_run",
                    "concurrent_runtime_activation_allowed_now",
                    "runtime_registered_as_default",
                    "runtime_defaultization_allowed_now",
                    "test_split_loaded",
                    "test_metrics_run",
                    "queue_actions_enabled",
                    "retry_actions_enabled",
                    "fallback_strategies_enabled",
                )
            )
            and locked.get("default_runtime_policy") == "unchanged",
            locked,
            "all closed and default unchanged",
        ),
        _check(
            "synthetic_exact_boundary_case_is_eligible_without_activation",
            compliant.get("state") == ConcurrentRuntimeValidationState.ELIGIBLE.value
            and compliant.get("rejection_reasons") == []
            and compliant.get("concurrent_runtime_activated") is False,
            compliant,
            "eligible policy result without activation",
        ),
        _check(
            "synthetic_latency_failure_is_rejected",
            latency_failure.get("state") == ConcurrentRuntimeValidationState.REJECTED.value
            and "train_end_to_end_p95_exceeds_slo"
            in (latency_failure.get("rejection_reasons") or [])
            and "dev_end_to_end_p99_missing" in (latency_failure.get("rejection_reasons") or []),
            latency_failure,
            "strict percentile failure reasons",
        ),
        _check(
            "synthetic_overload_failure_is_rejected",
            overload_failure.get("state") == ConcurrentRuntimeValidationState.REJECTED.value
            and "overload_probe_admitted_count_not_four"
            in (overload_failure.get("rejection_reasons") or [])
            and "queue_action_detected" in (overload_failure.get("rejection_reasons") or []),
            overload_failure,
            "overload and queue failures",
        ),
        _check(
            "synthetic_locked_boundary_failure_is_rejected",
            boundary_failure.get("state") == ConcurrentRuntimeValidationState.REJECTED.value
            and "test_split_not_locked" in (boundary_failure.get("rejection_reasons") or [])
            and "runtime_default_changed" in (boundary_failure.get("rejection_reasons") or [])
            and "retry_action_detected" in (boundary_failure.get("rejection_reasons") or [])
            and "fallback_action_detected" in (boundary_failure.get("rejection_reasons") or []),
            boundary_failure,
            "test/default/retry/fallback failures",
        ),
        _check(
            "source_and_protocol_are_public_safe",
            source_summary.get("public_safe_forbidden_keys_found") == []
            and not _forbidden_keys_found(report),
            {
                "source": source_summary.get("public_safe_forbidden_keys_found"),
                "protocol": sorted(_forbidden_keys_found(report)),
            },
            [],
        ),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [str(check["name"]) for check in guard_checks if not check["passed"]]
    base = {
        "protocol_id": _PROTOCOL_ID,
        "selected_profile_id": _PROFILE_ID,
        "max_in_flight": 4,
        "end_to_end_p95_limit_seconds": 0.8,
        "end_to_end_p99_limit_seconds": 1.5,
        "concurrency_validation_policy_executable": True,
        "concurrent_runtime_implemented_now": False,
        "concurrent_runtime_validation_run": False,
        "concurrent_runtime_activation_allowed_now": False,
        "runtime_registered_as_default": False,
        "runtime_defaultization_allowed_now": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
        "queue_actions_enabled": False,
        "retry_actions_enabled": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }
    if failed:
        return {
            **base,
            "status": "primeqa_hybrid_concurrent_runtime_validation_protocol_blocked",
            "failed_checks": failed,
            "concurrent_runtime_validation_protocol_frozen": False,
            "recommended_next_direction": "review_stage144_concurrency_protocol_failures",
        }
    return {
        **base,
        "status": "primeqa_hybrid_concurrent_runtime_validation_protocol_frozen",
        "failed_checks": [],
        "concurrent_runtime_validation_protocol_frozen": True,
        "recommended_next_direction": _NEXT_DIRECTION,
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "aggregate_only": True,
        "synthetic_policy_evaluations_written": True,
        "stage143_aggregate_loaded": True,
        "train_split_loaded": False,
        "dev_split_loaded": False,
        "test_split_loaded": False,
        "test_metrics_run": False,
        "models_or_indexes_loaded": False,
        "runtime_requests_executed": False,
        "concurrent_runtime_requests_executed": False,
        "private_per_request_traces_written": False,
        "raw_questions_written": False,
        "raw_answers_written": False,
        "raw_documents_written": False,
        "raw_candidate_rows_written": False,
        "raw_sample_document_request_or_cohort_ids_written": False,
        "forbidden_keys_found": sorted(_forbidden_keys_found(report)),
    }


def _benchmark_machine() -> dict[str, Any]:
    values = {
        "scope": "current_benchmark_machine_only",
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "machine_architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return {
        **values,
        "anonymous_fingerprint_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "portable_performance_claim_allowed": False,
    }


def _latency_slo_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    profile = (report.get("frozen_protocol") or {}).get("profile") or {}
    return [
        BarDatum(
            label="end-to-end P95 limit",
            value=float(profile.get("end_to_end_p95_seconds") or 0.0),
            value_label=f"{float(profile.get('end_to_end_p95_seconds') or 0):.3f}s",
        ),
        BarDatum(
            label="end-to-end P99 limit",
            value=float(profile.get("end_to_end_p99_seconds") or 0.0),
            value_label=f"{float(profile.get('end_to_end_p99_seconds') or 0):.3f}s",
        ),
    ]


def _train_request_budget_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    train = (report.get("frozen_protocol") or {}).get("train_validation_contract") or {}
    rows = (
        ("synchronized pattern", train.get("accepted_requests_per_pattern")),
        ("deterministic jitter pattern", train.get("accepted_requests_per_pattern")),
        ("all measured train requests", train.get("accepted_requests_total")),
    )
    return [
        BarDatum(label=label, value=float(value or 0), value_label=str(value or 0))
        for label, value in rows
    ]


def _arrival_offset_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    patterns = (report.get("frozen_protocol") or {}).get("arrival_patterns") or {}
    sync = (patterns.get("synchronized_four_request_burst") or {}).get("offsets_ms") or []
    jitter = (patterns.get("deterministic_jitter_0_to_20ms") or {}).get("offsets_ms") or []
    rows = [(f"sync position {index + 1}", value) for index, value in enumerate(sync)]
    rows.extend((f"jitter position {index + 1}", value) for index, value in enumerate(jitter))
    return [
        BarDatum(label=label, value=float(value), value_label=f"{value}ms") for label, value in rows
    ]


def _latency_gate_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    train = (report.get("frozen_protocol") or {}).get("train_validation_contract") or {}
    rows = (
        ("fold x pattern x repetition", train.get("fold_pattern_repetition_gate_count")),
        ("complete pass aggregates", train.get("pass_aggregate_gate_count")),
        ("pattern pooled", train.get("pattern_pooled_gate_count")),
        ("global pooled", train.get("global_pooled_gate_count")),
    )
    return [
        BarDatum(label=label, value=float(value or 0), value_label=str(value or 0))
        for label, value in rows
    ]


def _overload_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    overload = (report.get("frozen_protocol") or {}).get("overload_contract") or {}
    rows = (
        ("simultaneous attempts", overload.get("probe_attempt_count")),
        ("admitted", overload.get("expected_admitted_count")),
        ("capacity rejected", overload.get("expected_rejected_count")),
    )
    return [
        BarDatum(label=label, value=float(value or 0), value_label=str(value or 0))
        for label, value in rows
    ]


def _resource_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    resources = (report.get("stage143_summary") or {}).get("resource_summary") or {}
    keys = (
        "dense_model_count",
        "dense_embedding_cache_count",
        "lexical_index_count",
        "derived_route_count",
        "candidate_pool_retriever_instance_count",
        "optional_entrypoint_instance_count",
    )
    return [
        BarDatum(
            label=key,
            value=float(resources.get(key) or 0),
            value_label=str(resources.get(key) or 0),
        )
        for key in keys
    ]


def _decision_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    keys = (
        "concurrent_runtime_validation_protocol_frozen",
        "concurrency_validation_policy_executable",
        "concurrent_runtime_implemented_now",
        "concurrent_runtime_validation_run",
        "concurrent_runtime_activation_allowed_now",
        "runtime_registered_as_default",
        "runtime_defaultization_allowed_now",
        "test_gate_opened",
        "queue_actions_enabled",
        "retry_actions_enabled",
        "fallback_strategies_enabled",
    )
    return [
        BarDatum(
            label=key,
            value=1.0 if decision.get(key) else 0.0,
            value_label="true" if decision.get(key) else "false",
        )
        for key in keys
    ]


def _forbidden_keys_found(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in _FORBIDDEN_PUBLIC_KEYS:
                found.add(key_text)
            found.update(_forbidden_keys_found(child))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for child in value:
            found.update(_forbidden_keys_found(child))
    return found


def _check(name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest}
