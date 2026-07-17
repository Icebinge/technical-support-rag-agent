from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 147"
_CREATED_AT = "2026-07-17"
_PROTOCOL_ID = "primeqa_hybrid_agent_request_facade_protocol_v1"
_SOURCE_ANALYSIS_ID = "primeqa_hybrid_concurrent_runtime_application_activation_validation_v1"
_SOURCE_STATUS = "primeqa_hybrid_concurrent_runtime_application_activation_validation_passed"
_EXPECTED_SOURCE_GUARDS = 43
_NEXT_DIRECTION = "implement_transport_neutral_agent_request_facade"
_PUBLIC_RUNTIME_TRACE_FIELDS = (
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
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "answer_doc_id",
        "answer_text",
        "candidate_doc_ids",
        "cited_doc_ids",
        "document_body",
        "document_id",
        "document_text",
        "document_title",
        "gold_answer",
        "question_id",
        "question_text",
        "question_title",
        "raw_answer_text",
        "raw_document_text",
        "raw_question_text",
        "request_handle",
        "retrieved_doc_ids",
        "runtime_content_handle",
        "sample_id",
        "source_doc_ids",
    }
)


class AgentRequestFacadeProtocolState(str, Enum):
    """Outcomes of the executable Stage147 facade protocol policy."""

    REJECTED = "rejected"
    ELIGIBLE = "eligible"


@dataclass(frozen=True)
class AgentRequestFacadeProtocolEvidence:
    """Evidence required before implementing the transport-neutral facade."""

    source_stage146_validated: bool
    explicit_nondefault_concurrent_runtime_available: bool
    active_runtime_required: bool
    facade_owns_runtime_resources: bool
    private_request_payload: bool
    runtime_request_is_label_free: bool
    private_response_payload: bool
    public_telemetry_allowlist_only: bool
    public_telemetry_contains_request_content: bool
    public_telemetry_contains_response_content: bool
    capacity_error_mapping_exact: bool
    capacity_rejected_before_downstream: bool
    invalid_request_rejected_before_downstream: bool
    pre_dispatch_cancellation_only: bool
    in_flight_hard_cancellation_allowed: bool
    downstream_errors_propagate: bool
    errors_converted_to_answers: bool
    lifecycle_sequence_exact: bool
    draining_rejects_new_requests: bool
    shutdown_waits_for_in_flight: bool
    implicit_shutdown_timeout_allowed: bool
    force_cancel_allowed: bool
    queue_action_count: int
    retry_action_count: int
    fallback_action_count: int
    network_transport_deferred: bool
    test_split_locked: bool
    runtime_default_unchanged: bool


@dataclass(frozen=True)
class AgentRequestFacadeProtocolEvaluation:
    """Public-safe result; eligibility authorizes only the next implementation step."""

    state: AgentRequestFacadeProtocolState
    rejection_reasons: tuple[str, ...]
    facade_implemented: bool = False
    network_service_implemented: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "rejection_reasons": list(self.rejection_reasons),
            "facade_implemented": self.facade_implemented,
            "network_service_implemented": self.network_service_implemented,
        }


class StrictAgentRequestFacadeProtocolPolicy:
    """Fail-closed facade policy with no queue, retry, fallback, or forced shutdown."""

    def evaluate(
        self,
        evidence: AgentRequestFacadeProtocolEvidence,
    ) -> AgentRequestFacadeProtocolEvaluation:
        reasons: list[str] = []
        _require(reasons, evidence.source_stage146_validated, "stage146_source_not_validated")
        _require(
            reasons,
            evidence.explicit_nondefault_concurrent_runtime_available,
            "explicit_nondefault_concurrent_runtime_unavailable",
        )
        _require(reasons, evidence.active_runtime_required, "active_runtime_not_required")
        _forbid(reasons, evidence.facade_owns_runtime_resources, "facade_owns_runtime_resources")
        _require(reasons, evidence.private_request_payload, "private_request_boundary_missing")
        _require(reasons, evidence.runtime_request_is_label_free, "runtime_request_reads_labels")
        _require(reasons, evidence.private_response_payload, "private_response_boundary_missing")
        _require(
            reasons,
            evidence.public_telemetry_allowlist_only,
            "public_telemetry_allowlist_missing",
        )
        _forbid(
            reasons,
            evidence.public_telemetry_contains_request_content,
            "request_content_exposed_in_public_telemetry",
        )
        _forbid(
            reasons,
            evidence.public_telemetry_contains_response_content,
            "response_content_exposed_in_public_telemetry",
        )
        _require(reasons, evidence.capacity_error_mapping_exact, "capacity_mapping_not_exact")
        _require(
            reasons,
            evidence.capacity_rejected_before_downstream,
            "capacity_rejection_reached_downstream",
        )
        _require(
            reasons,
            evidence.invalid_request_rejected_before_downstream,
            "invalid_request_reached_downstream",
        )
        _require(
            reasons,
            evidence.pre_dispatch_cancellation_only,
            "pre_dispatch_cancellation_contract_missing",
        )
        _forbid(
            reasons,
            evidence.in_flight_hard_cancellation_allowed,
            "in_flight_hard_cancellation_enabled",
        )
        _require(
            reasons,
            evidence.downstream_errors_propagate,
            "downstream_error_propagation_disabled",
        )
        _forbid(reasons, evidence.errors_converted_to_answers, "errors_converted_to_answers")
        _require(reasons, evidence.lifecycle_sequence_exact, "lifecycle_sequence_mismatch")
        _require(
            reasons,
            evidence.draining_rejects_new_requests,
            "draining_accepts_new_requests",
        )
        _require(
            reasons,
            evidence.shutdown_waits_for_in_flight,
            "shutdown_does_not_wait_for_in_flight",
        )
        _forbid(
            reasons,
            evidence.implicit_shutdown_timeout_allowed,
            "implicit_shutdown_timeout_enabled",
        )
        _forbid(reasons, evidence.force_cancel_allowed, "force_cancel_enabled")
        if evidence.queue_action_count != 0:
            reasons.append("queue_action_detected")
        if evidence.retry_action_count != 0:
            reasons.append("retry_action_detected")
        if evidence.fallback_action_count != 0:
            reasons.append("fallback_action_detected")
        _require(
            reasons,
            evidence.network_transport_deferred,
            "network_transport_defined_prematurely",
        )
        _require(reasons, evidence.test_split_locked, "test_split_not_locked")
        _require(reasons, evidence.runtime_default_unchanged, "runtime_default_changed")
        return AgentRequestFacadeProtocolEvaluation(
            state=(
                AgentRequestFacadeProtocolState.REJECTED
                if reasons
                else AgentRequestFacadeProtocolState.ELIGIBLE
            ),
            rejection_reasons=tuple(reasons),
        )


@dataclass(frozen=True)
class PrimeQAHybridAgentRequestFacadeProtocolVisualization:
    """One generated Stage147 aggregate/specification visualization."""

    name: str
    path: str


def freeze_primeqa_hybrid_agent_request_facade_protocol(
    *,
    stage146_validation_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the facade contract from the saved public-safe Stage146 aggregate."""

    started_at = time.perf_counter()
    source = _load_json_object(stage146_validation_path)
    loaded_at = time.perf_counter()
    source_summary = _stage146_summary(source)
    protocol = _frozen_protocol()
    policy = StrictAgentRequestFacadeProtocolPolicy()
    canonical_evaluations = _canonical_evaluations(policy)
    preliminary = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "protocol_scope": (
            "Aggregate-only freeze of a transport-neutral, non-default application Agent "
            "request-facade contract. It reads only the saved Stage146 public aggregate and "
            "runs synthetic policy cases. It does not load train, dev, test, questions, "
            "documents, models, indexes, or candidate pools; instantiate a facade; start a "
            "network service; change defaults; or add queues, retries, fallback, hard "
            "cancellation, forced shutdown, or an implicit shutdown timeout."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_protocol),
            "confirmation_note": confirmation_note,
        },
        "source_files": {"stage146_validation": _fingerprint(stage146_validation_path)},
        "stage146_summary": source_summary,
        "frozen_protocol": protocol,
        "canonical_policy_evaluations": canonical_evaluations,
    }
    guards = _guard_checks(
        report=preliminary,
        source_summary=source_summary,
        protocol=protocol,
        canonical_evaluations=canonical_evaluations,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    checked_at = time.perf_counter()
    report = {
        **preliminary,
        "guard_checks": guards,
        "decision": _decision(guards),
        "timing_seconds": {
            "load_public_stage146_aggregate": round(loaded_at - started_at, 6),
            "freeze_and_guard": round(checked_at - loaded_at, 6),
            "total": round(checked_at - started_at, 6),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_agent_request_facade_protocol_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridAgentRequestFacadeProtocolVisualization]:
    """Write Stage147 aggregate/specification SVG charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage147_source_activation_boundary.svg": _chart(
            "Stage147 source activation boundary",
            _flag_bars(
                report.get("stage146_summary") or {},
                (
                    "source_identity_valid",
                    "all_source_guards_passed",
                    "explicit_nondefault_concurrent_activation_available",
                    "runtime_registered_as_default",
                    "network_service_implemented",
                ),
            ),
        ),
        "stage147_private_call_contract.svg": _chart(
            "Stage147 private call contract fields",
            _contract_count_bars(report, "private_call_contract"),
            x_label="field count",
        ),
        "stage147_public_telemetry_allowlist.svg": _chart(
            "Stage147 public telemetry allowlist",
            _contract_count_bars(report, "public_telemetry_contract"),
            x_label="field count",
        ),
        "stage147_error_mapping.svg": _chart(
            "Stage147 error mapping invariants",
            _section_flag_bars(report, "error_contract"),
        ),
        "stage147_cancellation_boundary.svg": _chart(
            "Stage147 cancellation boundary",
            _section_flag_bars(report, "cancellation_contract"),
        ),
        "stage147_lifecycle.svg": _chart(
            "Stage147 facade lifecycle",
            _lifecycle_bars(report),
            x_label="ordered state",
        ),
        "stage147_shutdown_contract.svg": _chart(
            "Stage147 shutdown invariants",
            _section_flag_bars(report, "shutdown_contract"),
        ),
        "stage147_policy_cases.svg": _chart(
            "Stage147 canonical policy outcomes",
            _policy_case_bars(report),
        ),
        "stage147_decision_flags.svg": _chart(
            "Stage147 protocol decision flags",
            _flag_bars(
                report.get("decision") or {},
                (
                    "agent_request_facade_protocol_frozen",
                    "facade_protocol_policy_executable",
                    "facade_implementation_allowed_next",
                    "facade_implemented_now",
                    "network_service_implemented",
                    "runtime_registered_as_default",
                    "test_gate_opened",
                    "queue_actions_enabled",
                    "retry_actions_enabled",
                    "fallback_strategies_enabled",
                ),
            ),
        ),
        "stage147_guard_check_status.svg": _chart(
            "Stage147 protocol guard checks",
            [
                BarDatum(
                    label=str(check["name"]),
                    value=1.0 if check["passed"] else 0.0,
                    value_label="passed" if check["passed"] else "failed",
                )
                for check in report.get("guard_checks", [])
            ],
            width=2440,
            margin_left=1380,
        ),
    }
    artifacts: list[PrimeQAHybridAgentRequestFacadeProtocolVisualization] = []
    for name, content in charts.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridAgentRequestFacadeProtocolVisualization(name=name, path=str(path))
        )
    return artifacts


def _frozen_protocol() -> dict[str, Any]:
    return {
        "facade_identity": {
            "facade_id": "primeqa_hybrid_transport_neutral_agent_request_facade_v1",
            "runtime_dependency": "active_concurrency_four_runtime_from_stage146_bootstrap",
            "activation": "explicit_nondefault_only",
            "facade_owns_runtime_resources": False,
            "facade_builds_or_loads_resources": False,
        },
        "private_call_contract": {
            "request_fields": ["request_handle", "title", "text", "cancellation_signal"],
            "required_request_fields": ["request_handle", "text"],
            "forbidden_runtime_label_fields": [
                "gold_answer",
                "answerable_label",
                "gold_document_reference",
                "test_membership",
            ],
            "response_fields": ["request_handle", "text", "refused", "citations"],
            "citation_fields": ["document_reference", "title", "rank", "evidence_score"],
            "candidate_pool_exposed": False,
            "raw_action_trace_exposed": False,
            "content_is_public_telemetry": False,
        },
        "public_telemetry_contract": {
            "request_trace_fields": list(_PUBLIC_RUNTIME_TRACE_FIELDS),
            "facade_event_fields": [
                "facade_state",
                "outcome_code",
                "downstream_dispatched",
                "queue_action_count",
                "retry_action_count",
                "fallback_action_count",
            ],
            "contains_request_content": False,
            "contains_response_content": False,
            "contains_request_or_document_identifiers": False,
        },
        "error_contract": {
            "invalid_request_code": "invalid_request",
            "invalid_request_rejected_before_downstream": True,
            "capacity_source_type": "PrimeQAHybridConcurrentCapacityExceededError",
            "capacity_facade_code": "capacity_exceeded",
            "capacity_mapping_exact": True,
            "capacity_rejected_before_downstream": True,
            "lifecycle_rejection_codes": ["facade_not_active", "facade_draining", "facade_closed"],
            "downstream_errors_propagate_unchanged": True,
            "errors_converted_to_answer_payloads": False,
        },
        "cancellation_contract": {
            "cooperative_signal": True,
            "checked_before_runtime_dispatch": True,
            "cancelled_before_dispatch_reaches_downstream": False,
            "in_flight_hard_cancellation_supported": False,
            "in_flight_work_may_be_claimed_cancelled": False,
            "permit_release_owned_by_runtime_finally": True,
        },
        "lifecycle_contract": {
            "states": ["accepting", "draining", "closed"],
            "initial_state_requires_active_runtime": True,
            "transition_sequence": ["accepting_to_draining", "draining_to_closed"],
            "new_calls_allowed_by_state": {
                "accepting_state": True,
                "draining_state": False,
                "closed_state": False,
            },
            "reopen_allowed": False,
        },
        "shutdown_contract": {
            "reject_new_calls_before_wait": True,
            "wait_for_in_flight_naturally": True,
            "implicit_timeout_seconds": None,
            "force_cancel": False,
            "close_runtime_resources": False,
            "process_bootstrap_retains_resource_ownership": True,
        },
        "closed_boundaries": {
            "network_transport_contract_deferred": True,
            "http_status_mapping_deferred": True,
            "request_size_limit_deferred": True,
            "facade_implementation_deferred": True,
            "runtime_defaultization": False,
            "test_access": False,
            "queue_actions": False,
            "retry_actions": False,
            "fallback_strategies": False,
        },
    }


def _compliant_evidence(**overrides: object) -> AgentRequestFacadeProtocolEvidence:
    values: dict[str, object] = {
        "source_stage146_validated": True,
        "explicit_nondefault_concurrent_runtime_available": True,
        "active_runtime_required": True,
        "facade_owns_runtime_resources": False,
        "private_request_payload": True,
        "runtime_request_is_label_free": True,
        "private_response_payload": True,
        "public_telemetry_allowlist_only": True,
        "public_telemetry_contains_request_content": False,
        "public_telemetry_contains_response_content": False,
        "capacity_error_mapping_exact": True,
        "capacity_rejected_before_downstream": True,
        "invalid_request_rejected_before_downstream": True,
        "pre_dispatch_cancellation_only": True,
        "in_flight_hard_cancellation_allowed": False,
        "downstream_errors_propagate": True,
        "errors_converted_to_answers": False,
        "lifecycle_sequence_exact": True,
        "draining_rejects_new_requests": True,
        "shutdown_waits_for_in_flight": True,
        "implicit_shutdown_timeout_allowed": False,
        "force_cancel_allowed": False,
        "queue_action_count": 0,
        "retry_action_count": 0,
        "fallback_action_count": 0,
        "network_transport_deferred": True,
        "test_split_locked": True,
        "runtime_default_unchanged": True,
    }
    values.update(overrides)
    return AgentRequestFacadeProtocolEvidence(**values)  # type: ignore[arg-type]


def _canonical_evaluations(
    policy: StrictAgentRequestFacadeProtocolPolicy,
) -> dict[str, dict[str, Any]]:
    cases = {
        "exact_compliant_protocol": _compliant_evidence(),
        "source_or_default_boundary_drift": _compliant_evidence(
            source_stage146_validated=False,
            runtime_default_unchanged=False,
            test_split_locked=False,
        ),
        "content_boundary_leak": _compliant_evidence(
            runtime_request_is_label_free=False,
            public_telemetry_contains_request_content=True,
            public_telemetry_contains_response_content=True,
        ),
        "unsafe_capacity_or_error_behavior": _compliant_evidence(
            capacity_error_mapping_exact=False,
            capacity_rejected_before_downstream=False,
            errors_converted_to_answers=True,
            queue_action_count=1,
            retry_action_count=1,
            fallback_action_count=1,
        ),
        "unsafe_cancellation_or_shutdown": _compliant_evidence(
            in_flight_hard_cancellation_allowed=True,
            shutdown_waits_for_in_flight=False,
            implicit_shutdown_timeout_allowed=True,
            force_cancel_allowed=True,
        ),
    }
    return {name: policy.evaluate(evidence).to_public_dict() for name, evidence in cases.items()}


def _stage146_summary(source: Mapping[str, Any]) -> dict[str, Any]:
    decision = source.get("decision") or {}
    public = source.get("public_safe_contract") or {}
    checks = source.get("guard_checks") or []
    passed_count = sum(check.get("passed") is True for check in checks)
    identity_valid = (
        source.get("stage") == "Stage 146"
        and source.get("analysis_id") == _SOURCE_ANALYSIS_ID
        and decision.get("status") == _SOURCE_STATUS
    )
    closed = (
        decision.get("runtime_registered_as_default") is False
        and decision.get("runtime_defaultization_allowed_now") is False
        and decision.get("network_service_implemented") is False
        and decision.get("test_gate_opened") is False
        and decision.get("test_metrics_run") is False
        and decision.get("queue_actions_enabled") is False
        and decision.get("retry_actions_enabled") is False
        and decision.get("fallback_strategies_enabled") is False
        and public.get("test_split_loaded") is False
        and public.get("test_metrics_run") is False
        and public.get("forbidden_keys_found") == []
    )
    return {
        "source_identity_valid": identity_valid,
        "source_guard_count": len(checks),
        "source_passed_guard_count": passed_count,
        "all_source_guards_passed": (
            len(checks) == _EXPECTED_SOURCE_GUARDS and passed_count == _EXPECTED_SOURCE_GUARDS
        ),
        "application_activation_bootstrap_implemented": decision.get(
            "application_activation_bootstrap_implemented"
        ),
        "eligible_runtime_full_workload_validation_passed": decision.get(
            "eligible_runtime_full_workload_validation_passed"
        ),
        "explicit_nondefault_concurrent_activation_available": decision.get(
            "explicit_nondefault_concurrent_activation_available"
        ),
        "runtime_registered_as_default": decision.get("runtime_registered_as_default"),
        "network_service_implemented": decision.get("network_service_implemented"),
        "closed_boundaries_preserved": closed,
        "test_split_loaded": public.get("test_split_loaded"),
        "test_metrics_run": public.get("test_metrics_run"),
        "forbidden_keys_found": list(public.get("forbidden_keys_found") or []),
    }


def _guard_checks(
    *,
    report: Mapping[str, Any],
    source_summary: Mapping[str, Any],
    protocol: Mapping[str, Any],
    canonical_evaluations: Mapping[str, Any],
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> list[dict[str, Any]]:
    private = protocol.get("private_call_contract") or {}
    public = protocol.get("public_telemetry_contract") or {}
    errors = protocol.get("error_contract") or {}
    cancellation = protocol.get("cancellation_contract") or {}
    lifecycle = protocol.get("lifecycle_contract") or {}
    shutdown = protocol.get("shutdown_contract") or {}
    closed = protocol.get("closed_boundaries") or {}
    exact = canonical_evaluations.get("exact_compliant_protocol") or {}
    negative_cases = [
        value for key, value in canonical_evaluations.items() if key != "exact_compliant_protocol"
    ]
    checks = [
        _check("stage147_user_confirmed", user_confirmed_protocol, user_confirmed_protocol, True),
        _check(
            "stage147_confirmation_note_present",
            bool(confirmation_note.strip()),
            bool(confirmation_note.strip()),
            True,
        ),
        _check(
            "stage146_source_identity_valid",
            source_summary.get("source_identity_valid") is True,
            source_summary.get("source_identity_valid"),
            True,
        ),
        _check(
            "stage146_all_43_guards_passed",
            source_summary.get("all_source_guards_passed") is True,
            source_summary.get("source_passed_guard_count"),
            _EXPECTED_SOURCE_GUARDS,
        ),
        _check(
            "stage146_application_activation_implemented",
            source_summary.get("application_activation_bootstrap_implemented") is True,
            source_summary.get("application_activation_bootstrap_implemented"),
            True,
        ),
        _check(
            "stage146_full_workload_passed",
            source_summary.get("eligible_runtime_full_workload_validation_passed") is True,
            source_summary.get("eligible_runtime_full_workload_validation_passed"),
            True,
        ),
        _check(
            "stage146_explicit_nondefault_activation_available",
            source_summary.get("explicit_nondefault_concurrent_activation_available") is True,
            source_summary.get("explicit_nondefault_concurrent_activation_available"),
            True,
        ),
        _check(
            "stage146_closed_boundaries_preserved",
            source_summary.get("closed_boundaries_preserved") is True,
            source_summary.get("closed_boundaries_preserved"),
            True,
        ),
        _check(
            "private_request_fields_exact",
            private.get("request_fields")
            == ["request_handle", "title", "text", "cancellation_signal"],
            private.get("request_fields"),
            ["request_handle", "title", "text", "cancellation_signal"],
        ),
        _check(
            "runtime_label_fields_forbidden",
            len(private.get("forbidden_runtime_label_fields") or []) == 4,
            len(private.get("forbidden_runtime_label_fields") or []),
            4,
        ),
        _check(
            "private_response_fields_exact",
            private.get("response_fields") == ["request_handle", "text", "refused", "citations"],
            private.get("response_fields"),
            ["request_handle", "text", "refused", "citations"],
        ),
        _check(
            "candidate_pool_and_raw_trace_not_exposed",
            private.get("candidate_pool_exposed") is False
            and private.get("raw_action_trace_exposed") is False,
            [private.get("candidate_pool_exposed"), private.get("raw_action_trace_exposed")],
            [False, False],
        ),
        _check(
            "public_telemetry_contains_no_content",
            public.get("contains_request_content") is False
            and public.get("contains_response_content") is False
            and public.get("contains_request_or_document_identifiers") is False,
            [
                public.get("contains_request_content"),
                public.get("contains_response_content"),
                public.get("contains_request_or_document_identifiers"),
            ],
            [False, False, False],
        ),
        _check(
            "public_runtime_trace_matches_stage146_allowlist",
            public.get("request_trace_fields") == list(_PUBLIC_RUNTIME_TRACE_FIELDS),
            public.get("request_trace_fields"),
            list(_PUBLIC_RUNTIME_TRACE_FIELDS),
        ),
        _check(
            "capacity_mapping_is_exact",
            errors.get("capacity_source_type") == "PrimeQAHybridConcurrentCapacityExceededError"
            and errors.get("capacity_facade_code") == "capacity_exceeded"
            and errors.get("capacity_mapping_exact") is True,
            [
                errors.get("capacity_source_type"),
                errors.get("capacity_facade_code"),
                errors.get("capacity_mapping_exact"),
            ],
            ["PrimeQAHybridConcurrentCapacityExceededError", "capacity_exceeded", True],
        ),
        _check(
            "invalid_and_capacity_reject_before_downstream",
            errors.get("invalid_request_rejected_before_downstream") is True
            and errors.get("capacity_rejected_before_downstream") is True,
            [
                errors.get("invalid_request_rejected_before_downstream"),
                errors.get("capacity_rejected_before_downstream"),
            ],
            [True, True],
        ),
        _check(
            "downstream_errors_propagate_without_answer_conversion",
            errors.get("downstream_errors_propagate_unchanged") is True
            and errors.get("errors_converted_to_answer_payloads") is False,
            [
                errors.get("downstream_errors_propagate_unchanged"),
                errors.get("errors_converted_to_answer_payloads"),
            ],
            [True, False],
        ),
        _check(
            "cancellation_is_pre_dispatch_only",
            cancellation.get("checked_before_runtime_dispatch") is True
            and cancellation.get("cancelled_before_dispatch_reaches_downstream") is False
            and cancellation.get("in_flight_hard_cancellation_supported") is False,
            [
                cancellation.get("checked_before_runtime_dispatch"),
                cancellation.get("cancelled_before_dispatch_reaches_downstream"),
                cancellation.get("in_flight_hard_cancellation_supported"),
            ],
            [True, False, False],
        ),
        _check(
            "runtime_finally_owns_permit_release",
            cancellation.get("permit_release_owned_by_runtime_finally") is True,
            cancellation.get("permit_release_owned_by_runtime_finally"),
            True,
        ),
        _check(
            "lifecycle_sequence_exact",
            lifecycle.get("states") == ["accepting", "draining", "closed"]
            and lifecycle.get("transition_sequence")
            == ["accepting_to_draining", "draining_to_closed"],
            [lifecycle.get("states"), lifecycle.get("transition_sequence")],
            [
                ["accepting", "draining", "closed"],
                ["accepting_to_draining", "draining_to_closed"],
            ],
        ),
        _check(
            "draining_and_closed_reject_new_calls",
            (lifecycle.get("new_calls_allowed_by_state") or {})
            == {"accepting_state": True, "draining_state": False, "closed_state": False},
            lifecycle.get("new_calls_allowed_by_state"),
            {"accepting_state": True, "draining_state": False, "closed_state": False},
        ),
        _check(
            "facade_cannot_reopen",
            lifecycle.get("reopen_allowed") is False,
            lifecycle.get("reopen_allowed"),
            False,
        ),
        _check(
            "shutdown_rejects_before_natural_wait",
            shutdown.get("reject_new_calls_before_wait") is True
            and shutdown.get("wait_for_in_flight_naturally") is True,
            [
                shutdown.get("reject_new_calls_before_wait"),
                shutdown.get("wait_for_in_flight_naturally"),
            ],
            [True, True],
        ),
        _check(
            "shutdown_has_no_implicit_timeout_or_force_cancel",
            shutdown.get("implicit_timeout_seconds") is None
            and shutdown.get("force_cancel") is False,
            [shutdown.get("implicit_timeout_seconds"), shutdown.get("force_cancel")],
            [None, False],
        ),
        _check(
            "bootstrap_retains_resource_ownership",
            shutdown.get("close_runtime_resources") is False
            and shutdown.get("process_bootstrap_retains_resource_ownership") is True,
            [
                shutdown.get("close_runtime_resources"),
                shutdown.get("process_bootstrap_retains_resource_ownership"),
            ],
            [False, True],
        ),
        _check(
            "network_http_and_size_contracts_deferred",
            closed.get("network_transport_contract_deferred") is True
            and closed.get("http_status_mapping_deferred") is True
            and closed.get("request_size_limit_deferred") is True,
            [
                closed.get("network_transport_contract_deferred"),
                closed.get("http_status_mapping_deferred"),
                closed.get("request_size_limit_deferred"),
            ],
            [True, True, True],
        ),
        _check(
            "default_test_queue_retry_fallback_closed",
            all(
                closed.get(key) is False
                for key in (
                    "runtime_defaultization",
                    "test_access",
                    "queue_actions",
                    "retry_actions",
                    "fallback_strategies",
                )
            ),
            [
                closed.get(key)
                for key in (
                    "runtime_defaultization",
                    "test_access",
                    "queue_actions",
                    "retry_actions",
                    "fallback_strategies",
                )
            ],
            [False, False, False, False, False],
        ),
        _check(
            "compliant_policy_case_is_eligible",
            exact.get("state") == AgentRequestFacadeProtocolState.ELIGIBLE.value
            and exact.get("rejection_reasons") == []
            and exact.get("facade_implemented") is False
            and exact.get("network_service_implemented") is False,
            exact,
            {
                "state": "eligible",
                "rejection_reasons": [],
                "facade_implemented": False,
                "network_service_implemented": False,
            },
        ),
        _check(
            "all_negative_policy_cases_rejected",
            len(negative_cases) == 4
            and all(case.get("state") == "rejected" for case in negative_cases),
            [case.get("state") for case in negative_cases],
            ["rejected"] * 4,
        ),
        _check(
            "source_or_default_drift_reasons_exact",
            (canonical_evaluations.get("source_or_default_boundary_drift") or {}).get(
                "rejection_reasons"
            )
            == [
                "stage146_source_not_validated",
                "test_split_not_locked",
                "runtime_default_changed",
            ],
            (canonical_evaluations.get("source_or_default_boundary_drift") or {}).get(
                "rejection_reasons"
            ),
            [
                "stage146_source_not_validated",
                "test_split_not_locked",
                "runtime_default_changed",
            ],
        ),
        _check(
            "content_leak_reasons_exact",
            (canonical_evaluations.get("content_boundary_leak") or {}).get("rejection_reasons")
            == [
                "runtime_request_reads_labels",
                "request_content_exposed_in_public_telemetry",
                "response_content_exposed_in_public_telemetry",
            ],
            (canonical_evaluations.get("content_boundary_leak") or {}).get("rejection_reasons"),
            [
                "runtime_request_reads_labels",
                "request_content_exposed_in_public_telemetry",
                "response_content_exposed_in_public_telemetry",
            ],
        ),
        _check(
            "capacity_error_negative_case_detected",
            set(
                (canonical_evaluations.get("unsafe_capacity_or_error_behavior") or {}).get(
                    "rejection_reasons"
                )
                or []
            )
            == {
                "capacity_mapping_not_exact",
                "capacity_rejection_reached_downstream",
                "errors_converted_to_answers",
                "queue_action_detected",
                "retry_action_detected",
                "fallback_action_detected",
            },
            (canonical_evaluations.get("unsafe_capacity_or_error_behavior") or {}).get(
                "rejection_reasons"
            ),
            "six exact rejection reasons",
        ),
        _check(
            "cancellation_shutdown_negative_case_detected",
            set(
                (canonical_evaluations.get("unsafe_cancellation_or_shutdown") or {}).get(
                    "rejection_reasons"
                )
                or []
            )
            == {
                "in_flight_hard_cancellation_enabled",
                "shutdown_does_not_wait_for_in_flight",
                "implicit_shutdown_timeout_enabled",
                "force_cancel_enabled",
            },
            (canonical_evaluations.get("unsafe_cancellation_or_shutdown") or {}).get(
                "rejection_reasons"
            ),
            "four exact rejection reasons",
        ),
        _check(
            "protocol_report_contains_no_forbidden_public_keys",
            _forbidden_keys_found(report) == set(),
            sorted(_forbidden_keys_found(report)),
            [],
        ),
    ]
    return checks


def _decision(guards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [str(check.get("name")) for check in guards if check.get("passed") is not True]
    passed = not failed
    return {
        "status": (
            "primeqa_hybrid_agent_request_facade_protocol_frozen"
            if passed
            else "primeqa_hybrid_agent_request_facade_protocol_rejected"
        ),
        "agent_request_facade_protocol_frozen": passed,
        "facade_protocol_policy_executable": passed,
        "facade_implementation_allowed_next": passed,
        "facade_implemented_now": False,
        "network_service_implemented": False,
        "transport_contract_defined": False,
        "runtime_registered_as_default": False,
        "runtime_defaultization_allowed_now": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
        "queue_actions_enabled": False,
        "retry_actions_enabled": False,
        "fallback_strategies_enabled": False,
        "hard_cancellation_enabled": False,
        "implicit_shutdown_timeout_enabled": False,
        "next_direction": _NEXT_DIRECTION if passed else "repair_failed_stage147_guards",
        "failed_checks": failed,
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_kind": "saved_public_stage146_aggregate_only",
        "train_split_loaded": False,
        "dev_split_loaded": False,
        "test_split_loaded": False,
        "test_metrics_run": False,
        "questions_loaded": False,
        "documents_loaded": False,
        "models_loaded": False,
        "indexes_loaded": False,
        "candidate_pools_built": False,
        "synthetic_policy_cases_only": True,
        "forbidden_keys_found": sorted(_forbidden_keys_found(report)),
    }


def _contract_count_bars(report: Mapping[str, Any], section: str) -> list[BarDatum]:
    contract = (report.get("frozen_protocol") or {}).get(section) or {}
    rows = [(key, len(value)) for key, value in contract.items() if isinstance(value, list)]
    return [BarDatum(label=key, value=float(value), value_label=str(value)) for key, value in rows]


def _section_flag_bars(report: Mapping[str, Any], section: str) -> list[BarDatum]:
    contract = (report.get("frozen_protocol") or {}).get(section) or {}
    return _flag_bars(
        contract, tuple(key for key, value in contract.items() if isinstance(value, bool))
    )


def _lifecycle_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    states = ((report.get("frozen_protocol") or {}).get("lifecycle_contract") or {}).get(
        "states"
    ) or []
    return [
        BarDatum(label=str(state), value=float(index), value_label=str(index))
        for index, state in enumerate(states, start=1)
    ]


def _policy_case_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    evaluations = report.get("canonical_policy_evaluations") or {}
    return [
        BarDatum(
            label=str(name),
            value=1.0 if evaluation.get("state") == "eligible" else 0.0,
            value_label=str(evaluation.get("state")),
        )
        for name, evaluation in evaluations.items()
    ]


def _flag_bars(source: Mapping[str, Any], keys: Sequence[str]) -> list[BarDatum]:
    return [
        BarDatum(
            label=key,
            value=1.0 if source.get(key) is True else 0.0,
            value_label="true" if source.get(key) is True else "false",
        )
        for key in keys
    ]


def _chart(
    title: str,
    bars: Sequence[BarDatum],
    *,
    x_label: str = "1 means true",
    width: int = 1740,
    margin_left: int = 900,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        width=width,
        margin_left=margin_left,
    )


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


def _require(reasons: list[str], condition: bool, reason: str) -> None:
    if not condition:
        reasons.append(reason)


def _forbid(reasons: list[str], condition: bool, reason: str) -> None:
    if condition:
        reasons.append(reason)


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
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
