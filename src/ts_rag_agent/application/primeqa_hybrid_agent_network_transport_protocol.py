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

_STAGE = "Stage 149"
_CREATED_AT = "2026-07-17"
_PROTOCOL_ID = "primeqa_hybrid_agent_http_transport_protocol_v1"
_SOURCE_ANALYSIS_ID = "primeqa_hybrid_transport_neutral_agent_request_facade_validation_v1"
_SOURCE_STATUS = "primeqa_hybrid_transport_neutral_agent_request_facade_validation_passed"
_EXPECTED_SOURCE_GUARDS = 37
_FINAL_STATUS = "primeqa_hybrid_agent_http_transport_protocol_frozen"
_NEXT_DIRECTION = "implement_nondefault_local_fastapi_agent_transport"

_MAX_BODY_BYTES = 32 * 1024
_MAX_REQUEST_HANDLE_CHARS = 128
_MAX_TITLE_CHARS = 512
_MAX_TEXT_CHARS = 24 * 1024
_PUBLIC_LOG_FIELDS = (
    "route_id",
    "method",
    "http_status",
    "transport_outcome_code",
    "facade_state",
    "downstream_dispatched",
    "queue_action_count",
    "retry_action_count",
    "fallback_action_count",
    "runtime_mode",
    "activation_state",
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
        "answer_text",
        "authorization",
        "candidate_doc_ids",
        "citations",
        "client_ip",
        "cookie",
        "document_body",
        "document_id",
        "document_reference",
        "document_text",
        "document_title",
        "gold_answer",
        "headers",
        "question_id",
        "question_text",
        "question_title",
        "raw_body",
        "request_handle",
        "response_text",
        "source_doc_ids",
        "text",
        "title",
        "user_agent",
    }
)


class AgentNetworkTransportProtocolState(str, Enum):
    """Outcome of the strict Stage149 transport policy."""

    REJECTED = "rejected"
    ELIGIBLE = "eligible"


@dataclass(frozen=True)
class AgentNetworkTransportProtocolEvidence:
    """Evidence required before implementing the local HTTP adapter."""

    source_stage148_validated: bool
    facade_is_transport_neutral: bool
    local_loopback_only: bool
    remote_exposure_enabled: bool
    exact_versioned_routes: bool
    strict_json_content_type: bool
    unknown_request_fields_rejected: bool
    raw_body_limit_enforced_before_parse: bool
    field_limits_exact: bool
    stable_success_schema: bool
    stable_error_schema: bool
    http_error_mapping_exact: bool
    domain_refusal_is_success: bool
    disconnect_checked_before_dispatch: bool
    in_flight_hard_cancellation_claimed: bool
    lifespan_owns_facade_shutdown: bool
    lifespan_owns_runtime_resources: bool
    natural_shutdown_without_timeout: bool
    liveness_is_resource_independent: bool
    readiness_requires_accepting_facade: bool
    public_logging_allowlist_only: bool
    default_access_log_enabled: bool
    queue_action_count: int
    retry_action_count: int
    fallback_action_count: int
    network_service_implemented: bool
    runtime_default_unchanged: bool
    test_split_locked: bool


@dataclass(frozen=True)
class AgentNetworkTransportProtocolEvaluation:
    """Public-safe protocol eligibility result."""

    state: AgentNetworkTransportProtocolState
    rejection_reasons: tuple[str, ...]
    network_service_implemented: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "rejection_reasons": list(self.rejection_reasons),
            "network_service_implemented": self.network_service_implemented,
        }


class StrictAgentNetworkTransportProtocolPolicy:
    """Fail-closed HTTP policy with no remote exposure or hidden recovery behavior."""

    def evaluate(
        self,
        evidence: AgentNetworkTransportProtocolEvidence,
    ) -> AgentNetworkTransportProtocolEvaluation:
        reasons: list[str] = []
        _require(reasons, evidence.source_stage148_validated, "stage148_source_not_validated")
        _require(reasons, evidence.facade_is_transport_neutral, "facade_not_transport_neutral")
        _require(reasons, evidence.local_loopback_only, "loopback_binding_not_required")
        _forbid(reasons, evidence.remote_exposure_enabled, "remote_exposure_enabled")
        _require(reasons, evidence.exact_versioned_routes, "route_contract_not_exact")
        _require(reasons, evidence.strict_json_content_type, "json_content_type_not_strict")
        _require(
            reasons,
            evidence.unknown_request_fields_rejected,
            "unknown_request_fields_accepted",
        )
        _require(
            reasons,
            evidence.raw_body_limit_enforced_before_parse,
            "raw_body_limit_not_preparse",
        )
        _require(reasons, evidence.field_limits_exact, "field_limits_not_exact")
        _require(reasons, evidence.stable_success_schema, "success_schema_not_stable")
        _require(reasons, evidence.stable_error_schema, "error_schema_not_stable")
        _require(reasons, evidence.http_error_mapping_exact, "http_error_mapping_not_exact")
        _require(reasons, evidence.domain_refusal_is_success, "domain_refusal_mapped_as_error")
        _require(
            reasons,
            evidence.disconnect_checked_before_dispatch,
            "predispatch_disconnect_check_missing",
        )
        _forbid(
            reasons,
            evidence.in_flight_hard_cancellation_claimed,
            "in_flight_hard_cancellation_claimed",
        )
        _require(
            reasons,
            evidence.lifespan_owns_facade_shutdown,
            "lifespan_does_not_shutdown_facade",
        )
        _forbid(
            reasons,
            evidence.lifespan_owns_runtime_resources,
            "transport_lifespan_owns_runtime_resources",
        )
        _require(
            reasons,
            evidence.natural_shutdown_without_timeout,
            "natural_shutdown_contract_missing",
        )
        _require(
            reasons,
            evidence.liveness_is_resource_independent,
            "liveness_depends_on_runtime_resources",
        )
        _require(
            reasons,
            evidence.readiness_requires_accepting_facade,
            "readiness_accepts_nonaccepting_facade",
        )
        _require(
            reasons,
            evidence.public_logging_allowlist_only,
            "public_logging_allowlist_missing",
        )
        _forbid(reasons, evidence.default_access_log_enabled, "default_access_log_enabled")
        if evidence.queue_action_count != 0:
            reasons.append("queue_action_detected")
        if evidence.retry_action_count != 0:
            reasons.append("retry_action_detected")
        if evidence.fallback_action_count != 0:
            reasons.append("fallback_action_detected")
        _forbid(reasons, evidence.network_service_implemented, "network_service_preimplemented")
        _require(reasons, evidence.runtime_default_unchanged, "runtime_default_changed")
        _require(reasons, evidence.test_split_locked, "test_split_not_locked")
        return AgentNetworkTransportProtocolEvaluation(
            state=(
                AgentNetworkTransportProtocolState.REJECTED
                if reasons
                else AgentNetworkTransportProtocolState.ELIGIBLE
            ),
            rejection_reasons=tuple(reasons),
        )


@dataclass(frozen=True)
class PrimeQAHybridAgentNetworkTransportProtocolVisualization:
    """One Stage149 aggregate/specification SVG."""

    name: str
    path: str


def freeze_primeqa_hybrid_agent_network_transport_protocol(
    *,
    stage148_validation_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the HTTP contract from the saved public-safe Stage148 aggregate."""

    started_at = time.perf_counter()
    source = _load_json_object(stage148_validation_path)
    loaded_at = time.perf_counter()
    source_summary = _stage148_summary(source)
    protocol = _frozen_protocol()
    evaluations = _canonical_evaluations(StrictAgentNetworkTransportProtocolPolicy())
    source_unchanged = _load_json_object(stage148_validation_path) == source
    preliminary = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "protocol_scope": (
            "Aggregate-only freeze of a strict, local-loopback HTTP transport around the "
            "Stage148 facade. It reads only the saved Stage148 public aggregate and runs "
            "synthetic policy cases. It does not load train, dev, test, questions, documents, "
            "models, indexes, or candidate pools; start a network service; change defaults; "
            "or add queues, retries, fallback, hard cancellation, forced shutdown, or an "
            "implicit shutdown timeout."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_protocol),
            "confirmation_note": confirmation_note,
        },
        "source_files": {"stage148_validation": _fingerprint(stage148_validation_path)},
        "stage148_summary": source_summary,
        "source_unchanged_after_protocol_freeze": source_unchanged,
        "official_design_references": _official_design_references(),
        "frozen_protocol": protocol,
        "canonical_policy_evaluations": evaluations,
    }
    guards = _guard_checks(
        report=preliminary,
        source_summary=source_summary,
        protocol=protocol,
        evaluations=evaluations,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    checked_at = time.perf_counter()
    report = {
        **preliminary,
        "guard_checks": guards,
        "decision": _decision(guards),
        "timing_seconds": {
            "load_public_stage148_aggregate": round(loaded_at - started_at, 6),
            "freeze_and_guard": round(checked_at - loaded_at, 6),
            "total": round(checked_at - started_at, 6),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_agent_network_transport_protocol_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridAgentNetworkTransportProtocolVisualization]:
    """Write Stage149 aggregate/specification SVG charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = report.get("frozen_protocol") or {}
    charts = {
        "stage149_source_gate.svg": _chart(
            "Stage149 Stage148 source gate",
            _flag_bars(
                report.get("stage148_summary") or {},
                (
                    "source_identity_valid",
                    "all_source_guards_passed",
                    "transport_neutral_facade_implemented",
                    "network_transport_protocol_allowed_next",
                    "network_service_implemented",
                ),
            ),
        ),
        "stage149_http_surface.svg": _chart(
            "Stage149 exact HTTP surface",
            _route_bars(protocol),
            x_label="route count",
        ),
        "stage149_request_limits.svg": _chart(
            "Stage149 strict request limits",
            _limit_bars(protocol),
            x_label="limit value",
            margin_left=760,
        ),
        "stage149_status_mapping.svg": _chart(
            "Stage149 HTTP status mapping",
            _status_bars(protocol),
            x_label="HTTP status",
            margin_left=900,
        ),
        "stage149_disconnect_boundary.svg": _chart(
            "Stage149 disconnect boundary",
            _section_flag_bars(protocol, "disconnect_contract"),
        ),
        "stage149_lifespan_boundary.svg": _chart(
            "Stage149 lifespan ownership",
            _section_flag_bars(protocol, "lifespan_contract"),
        ),
        "stage149_health_semantics.svg": _chart(
            "Stage149 health and readiness semantics",
            _section_flag_bars(protocol, "health_contract"),
        ),
        "stage149_logging_boundary.svg": _chart(
            "Stage149 public logging boundary",
            _logging_bars(protocol),
            x_label="field count",
        ),
        "stage149_policy_cases.svg": _chart(
            "Stage149 canonical policy outcomes",
            _policy_case_bars(report),
        ),
        "stage149_guard_check_status.svg": _chart(
            "Stage149 protocol guard checks",
            [
                BarDatum(
                    label=str(check["name"]),
                    value=1.0 if check["passed"] else 0.0,
                    value_label="passed" if check["passed"] else "failed",
                )
                for check in report.get("guard_checks", [])
            ],
            width=2500,
            margin_left=1420,
        ),
    }
    artifacts: list[PrimeQAHybridAgentNetworkTransportProtocolVisualization] = []
    for name, content in charts.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridAgentNetworkTransportProtocolVisualization(name=name, path=str(path))
        )
    return artifacts


def _frozen_protocol() -> dict[str, Any]:
    return {
        "transport_identity": {
            "transport_id": _PROTOCOL_ID,
            "protocol": "HTTP/1.1",
            "implementation_framework_next": "FastAPI",
            "binding_host": "127.0.0.1",
            "local_loopback_only": True,
            "remote_exposure_enabled": False,
            "tls_terminated_by_application": False,
            "authentication_enabled": False,
            "cors_enabled": False,
            "openapi_and_interactive_docs_enabled": False,
        },
        "route_contract": {
            "routes": [
                {"route_id": "agent_answer", "method": "POST", "path": "/v1/agent/answers"},
                {"route_id": "liveness", "method": "GET", "path": "/health/live"},
                {"route_id": "readiness", "method": "GET", "path": "/health/ready"},
            ],
            "unversioned_agent_route_allowed": False,
            "streaming_response_enabled": False,
            "websocket_enabled": False,
        },
        "request_contract": {
            "content_type": "application/json",
            "charset": "utf-8",
            "request_fields": ["request_handle", "title", "text"],
            "required_fields": ["request_handle", "text"],
            "nullable_fields": ["title"],
            "unknown_fields_rejected": True,
            "coercion_enabled": False,
            "raw_body_max_bytes": _MAX_BODY_BYTES,
            "request_handle_max_chars": _MAX_REQUEST_HANDLE_CHARS,
            "title_max_chars": _MAX_TITLE_CHARS,
            "text_max_chars": _MAX_TEXT_CHARS,
            "nonblank_fields": ["request_handle", "text"],
            "content_length_precheck": True,
            "streaming_body_hard_cap": True,
            "oversize_body_truncated_and_processed": False,
        },
        "response_contract": {
            "success_fields": ["request_handle", "text", "refused", "citations"],
            "citation_fields": ["document_reference", "title", "rank", "evidence_score"],
            "error_envelope_fields": ["error"],
            "error_fields": ["code", "message"],
            "domain_refusal_http_status": 200,
            "domain_refusal_is_transport_error": False,
            "candidate_pool_exposed": False,
            "raw_action_trace_exposed": False,
        },
        "http_error_mapping": {
            "malformed_json": {"status": 400, "code": "malformed_json"},
            "request_body_too_large": {"status": 413, "code": "request_body_too_large"},
            "unsupported_media_type": {"status": 415, "code": "unsupported_media_type"},
            "schema_validation_failed": {"status": 422, "code": "invalid_request"},
            "facade_invalid_request": {"status": 422, "code": "invalid_request"},
            "facade_not_active": {"status": 503, "code": "facade_not_active"},
            "capacity_exceeded": {"status": 503, "code": "capacity_exceeded"},
            "facade_draining": {"status": 503, "code": "facade_draining"},
            "facade_closed": {"status": 503, "code": "facade_closed"},
            "unexpected_downstream_error": {"status": 500, "code": "internal_error"},
            "cancelled_before_dispatch": {"status": None, "code": "client_disconnected"},
        },
        "disconnect_contract": {
            "check_after_body_before_dispatch": True,
            "predispatch_disconnect_sets_facade_cancellation": True,
            "predispatch_disconnect_reaches_runtime": False,
            "response_attempted_after_known_disconnect": False,
            "in_flight_hard_cancellation_supported": False,
            "in_flight_disconnect_claimed_as_cancelled": False,
            "in_flight_work_completes_or_raises_naturally": True,
            "disconnect_race_eliminated": False,
        },
        "lifespan_contract": {
            "fastapi_lifespan_context_required": True,
            "startup_requires_active_stage146_bootstrap": True,
            "facade_created_once_per_process": True,
            "shutdown_calls_facade_shutdown": True,
            "shutdown_rejects_new_before_wait": True,
            "shutdown_waits_for_in_flight_naturally": True,
            "implicit_shutdown_timeout_seconds": None,
            "force_cancel": False,
            "transport_owns_runtime_resources": False,
            "process_bootstrap_retains_runtime_resource_ownership": True,
        },
        "health_contract": {
            "liveness_path": "/health/live",
            "liveness_success_status": 200,
            "liveness_loads_or_probes_models": False,
            "liveness_depends_on_facade_acceptance": False,
            "readiness_path": "/health/ready",
            "readiness_accepting_status": 200,
            "readiness_nonaccepting_status": 503,
            "readiness_requires_facade_accepting": True,
            "readiness_loads_questions_or_documents": False,
        },
        "execution_contract": {
            "event_loop_blocked_by_sync_facade": False,
            "sync_facade_runs_off_event_loop": True,
            "application_owned_waiting_queue": False,
            "runtime_capacity_rejection_remains_nonblocking": True,
            "automatic_retry": False,
            "fallback": False,
            "request_timeout_seconds": None,
        },
        "logging_contract": {
            "default_uvicorn_access_log_enabled": False,
            "structured_allowlist_only": True,
            "allowed_fields": list(_PUBLIC_LOG_FIELDS),
            "request_or_response_content_logged": False,
            "request_or_document_identifiers_logged": False,
            "headers_cookies_or_client_address_logged": False,
            "exception_message_logged_publicly": False,
        },
        "closed_boundaries": {
            "network_service_implemented": False,
            "runtime_registered_as_default": False,
            "remote_deployment_authorized": False,
            "authentication_or_tls_designed": False,
            "test_access": False,
            "queue_actions": False,
            "retry_actions": False,
            "fallback_strategies": False,
            "hard_cancellation": False,
            "implicit_shutdown_timeout": False,
        },
    }


def _official_design_references() -> list[dict[str, str]]:
    return [
        {
            "topic": "lifespan",
            "url": "https://fastapi.tiangolo.com/advanced/events/",
            "applied_fact": "FastAPI lifespan context surrounds startup and shutdown.",
        },
        {
            "topic": "disconnect_detection",
            "url": "https://www.starlette.io/requests/",
            "applied_fact": "Starlette Request exposes asynchronous disconnect detection.",
        },
        {
            "topic": "custom_error_handlers",
            "url": "https://fastapi.tiangolo.com/tutorial/handling-errors/",
            "applied_fact": "FastAPI supports custom exception and validation handlers.",
        },
        {
            "topic": "server_resource_limits",
            "url": "https://www.uvicorn.org/settings/",
            "applied_fact": (
                "Uvicorn documents loopback binding, access logging, and 503 concurrency limits."
            ),
        },
    ]


def _compliant_evidence(**overrides: object) -> AgentNetworkTransportProtocolEvidence:
    values: dict[str, object] = {
        "source_stage148_validated": True,
        "facade_is_transport_neutral": True,
        "local_loopback_only": True,
        "remote_exposure_enabled": False,
        "exact_versioned_routes": True,
        "strict_json_content_type": True,
        "unknown_request_fields_rejected": True,
        "raw_body_limit_enforced_before_parse": True,
        "field_limits_exact": True,
        "stable_success_schema": True,
        "stable_error_schema": True,
        "http_error_mapping_exact": True,
        "domain_refusal_is_success": True,
        "disconnect_checked_before_dispatch": True,
        "in_flight_hard_cancellation_claimed": False,
        "lifespan_owns_facade_shutdown": True,
        "lifespan_owns_runtime_resources": False,
        "natural_shutdown_without_timeout": True,
        "liveness_is_resource_independent": True,
        "readiness_requires_accepting_facade": True,
        "public_logging_allowlist_only": True,
        "default_access_log_enabled": False,
        "queue_action_count": 0,
        "retry_action_count": 0,
        "fallback_action_count": 0,
        "network_service_implemented": False,
        "runtime_default_unchanged": True,
        "test_split_locked": True,
    }
    values.update(overrides)
    return AgentNetworkTransportProtocolEvidence(**values)  # type: ignore[arg-type]


def _canonical_evaluations(
    policy: StrictAgentNetworkTransportProtocolPolicy,
) -> dict[str, dict[str, Any]]:
    cases = {
        "exact_compliant_protocol": _compliant_evidence(),
        "unsafe_exposure_or_schema": _compliant_evidence(
            local_loopback_only=False,
            remote_exposure_enabled=True,
            exact_versioned_routes=False,
            strict_json_content_type=False,
            unknown_request_fields_rejected=False,
        ),
        "unsafe_size_or_error_mapping": _compliant_evidence(
            raw_body_limit_enforced_before_parse=False,
            field_limits_exact=False,
            stable_error_schema=False,
            http_error_mapping_exact=False,
            domain_refusal_is_success=False,
        ),
        "unsafe_disconnect_or_shutdown": _compliant_evidence(
            disconnect_checked_before_dispatch=False,
            in_flight_hard_cancellation_claimed=True,
            lifespan_owns_facade_shutdown=False,
            lifespan_owns_runtime_resources=True,
            natural_shutdown_without_timeout=False,
        ),
        "unsafe_health_logging_or_recovery": _compliant_evidence(
            liveness_is_resource_independent=False,
            readiness_requires_accepting_facade=False,
            public_logging_allowlist_only=False,
            default_access_log_enabled=True,
            queue_action_count=1,
            retry_action_count=1,
            fallback_action_count=1,
        ),
        "unsafe_boundary_drift": _compliant_evidence(
            source_stage148_validated=False,
            network_service_implemented=True,
            runtime_default_unchanged=False,
            test_split_locked=False,
        ),
    }
    return {name: policy.evaluate(evidence).to_public_dict() for name, evidence in cases.items()}


def _stage148_summary(source: Mapping[str, Any]) -> dict[str, Any]:
    decision = source.get("decision") or {}
    public = source.get("public_safe_contract") or {}
    implementation = source.get("implementation_contract") or {}
    checks = source.get("guard_checks") or []
    passed_count = sum(check.get("passed") is True for check in checks)
    identity_valid = (
        source.get("stage") == "Stage 148"
        and source.get("analysis_id") == _SOURCE_ANALYSIS_ID
        and decision.get("status") == _SOURCE_STATUS
    )
    closed = (
        decision.get("network_service_implemented") is False
        and decision.get("runtime_registered_as_default") is False
        and decision.get("test_gate_opened") is False
        and decision.get("queue_actions_enabled") is False
        and decision.get("retry_actions_enabled") is False
        and decision.get("fallback_strategies_enabled") is False
        and public.get("network_service_started") is False
        and public.get("test_split_loaded") is False
        and public.get("forbidden_keys_found") == []
    )
    return {
        "source_identity_valid": identity_valid,
        "source_guard_count": len(checks),
        "source_passed_guard_count": passed_count,
        "all_source_guards_passed": (
            len(checks) == _EXPECTED_SOURCE_GUARDS and passed_count == _EXPECTED_SOURCE_GUARDS
        ),
        "transport_neutral_facade_implemented": decision.get(
            "transport_neutral_facade_implemented"
        ),
        "facade_synthetic_validation_passed": decision.get("facade_synthetic_validation_passed"),
        "label_free_runtime_query_validated": decision.get("label_free_runtime_query_validated"),
        "public_telemetry_allowlists_validated": decision.get(
            "public_telemetry_allowlists_validated"
        ),
        "lifecycle_and_natural_shutdown_validated": decision.get(
            "lifecycle_and_natural_shutdown_validated"
        ),
        "network_transport_protocol_allowed_next": decision.get(
            "network_transport_protocol_allowed_next"
        ),
        "network_service_implemented": decision.get("network_service_implemented"),
        "runtime_registered_as_default": decision.get("runtime_registered_as_default"),
        "facade_network_service_implemented": implementation.get("network_service_implemented"),
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
    evaluations: Mapping[str, Any],
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> list[dict[str, Any]]:
    identity = protocol.get("transport_identity") or {}
    routes = protocol.get("route_contract") or {}
    request = protocol.get("request_contract") or {}
    response = protocol.get("response_contract") or {}
    errors = protocol.get("http_error_mapping") or {}
    disconnect = protocol.get("disconnect_contract") or {}
    lifespan = protocol.get("lifespan_contract") or {}
    health = protocol.get("health_contract") or {}
    execution = protocol.get("execution_contract") or {}
    logging = protocol.get("logging_contract") or {}
    closed = protocol.get("closed_boundaries") or {}
    exact = evaluations.get("exact_compliant_protocol") or {}
    negative = [value for key, value in evaluations.items() if key != "exact_compliant_protocol"]
    checks = [
        _check("stage149_user_confirmed", user_confirmed_protocol, user_confirmed_protocol, True),
        _check(
            "stage149_confirmation_note_present",
            bool(confirmation_note.strip()),
            bool(confirmation_note.strip()),
            True,
        ),
        _check(
            "stage148_source_identity_valid",
            source_summary.get("source_identity_valid") is True,
            source_summary.get("source_identity_valid"),
            True,
        ),
        _check(
            "stage148_all_37_guards_passed",
            source_summary.get("all_source_guards_passed") is True,
            source_summary.get("source_passed_guard_count"),
            _EXPECTED_SOURCE_GUARDS,
        ),
        _check(
            "stage148_facade_implemented_and_validated",
            source_summary.get("transport_neutral_facade_implemented") is True
            and source_summary.get("facade_synthetic_validation_passed") is True,
            [
                source_summary.get("transport_neutral_facade_implemented"),
                source_summary.get("facade_synthetic_validation_passed"),
            ],
            [True, True],
        ),
        _check(
            "stage148_label_free_and_public_safe",
            source_summary.get("label_free_runtime_query_validated") is True
            and source_summary.get("public_telemetry_allowlists_validated") is True,
            [
                source_summary.get("label_free_runtime_query_validated"),
                source_summary.get("public_telemetry_allowlists_validated"),
            ],
            [True, True],
        ),
        _check(
            "stage148_natural_lifecycle_validated",
            source_summary.get("lifecycle_and_natural_shutdown_validated") is True,
            source_summary.get("lifecycle_and_natural_shutdown_validated"),
            True,
        ),
        _check(
            "stage148_authorizes_transport_protocol",
            source_summary.get("network_transport_protocol_allowed_next") is True,
            source_summary.get("network_transport_protocol_allowed_next"),
            True,
        ),
        _check(
            "stage148_closed_boundaries_preserved",
            source_summary.get("closed_boundaries_preserved") is True,
            source_summary.get("closed_boundaries_preserved"),
            True,
        ),
        _check(
            "transport_is_loopback_only",
            identity.get("binding_host") == "127.0.0.1"
            and identity.get("local_loopback_only") is True
            and identity.get("remote_exposure_enabled") is False,
            [
                identity.get("binding_host"),
                identity.get("local_loopback_only"),
                identity.get("remote_exposure_enabled"),
            ],
            ["127.0.0.1", True, False],
        ),
        _check(
            "public_surface_has_exact_three_routes",
            routes.get("routes")
            == [
                {"route_id": "agent_answer", "method": "POST", "path": "/v1/agent/answers"},
                {"route_id": "liveness", "method": "GET", "path": "/health/live"},
                {"route_id": "readiness", "method": "GET", "path": "/health/ready"},
            ],
            routes.get("routes"),
            "exact_three_routes",
        ),
        _check(
            "streaming_websocket_and_unversioned_routes_closed",
            routes.get("unversioned_agent_route_allowed") is False
            and routes.get("streaming_response_enabled") is False
            and routes.get("websocket_enabled") is False,
            [
                routes.get("unversioned_agent_route_allowed"),
                routes.get("streaming_response_enabled"),
                routes.get("websocket_enabled"),
            ],
            [False, False, False],
        ),
        _check(
            "request_schema_exact_and_strict",
            request.get("content_type") == "application/json"
            and request.get("request_fields") == ["request_handle", "title", "text"]
            and request.get("required_fields") == ["request_handle", "text"]
            and request.get("unknown_fields_rejected") is True
            and request.get("coercion_enabled") is False,
            [
                request.get("content_type"),
                request.get("request_fields"),
                request.get("required_fields"),
                request.get("unknown_fields_rejected"),
                request.get("coercion_enabled"),
            ],
            [
                "application/json",
                ["request_handle", "title", "text"],
                ["request_handle", "text"],
                True,
                False,
            ],
        ),
        _check(
            "raw_body_limit_exact_and_preparse",
            request.get("raw_body_max_bytes") == _MAX_BODY_BYTES
            and request.get("content_length_precheck") is True
            and request.get("streaming_body_hard_cap") is True
            and request.get("oversize_body_truncated_and_processed") is False,
            [
                request.get("raw_body_max_bytes"),
                request.get("content_length_precheck"),
                request.get("streaming_body_hard_cap"),
                request.get("oversize_body_truncated_and_processed"),
            ],
            [_MAX_BODY_BYTES, True, True, False],
        ),
        _check(
            "field_limits_exact",
            [
                request.get("request_handle_max_chars"),
                request.get("title_max_chars"),
                request.get("text_max_chars"),
            ]
            == [_MAX_REQUEST_HANDLE_CHARS, _MAX_TITLE_CHARS, _MAX_TEXT_CHARS],
            [
                request.get("request_handle_max_chars"),
                request.get("title_max_chars"),
                request.get("text_max_chars"),
            ],
            [_MAX_REQUEST_HANDLE_CHARS, _MAX_TITLE_CHARS, _MAX_TEXT_CHARS],
        ),
        _check(
            "success_and_citation_schema_match_facade",
            response.get("success_fields") == ["request_handle", "text", "refused", "citations"]
            and response.get("citation_fields")
            == ["document_reference", "title", "rank", "evidence_score"],
            [response.get("success_fields"), response.get("citation_fields")],
            "Stage148 private response contract",
        ),
        _check(
            "stable_two_field_error_envelope",
            response.get("error_envelope_fields") == ["error"]
            and response.get("error_fields") == ["code", "message"],
            [response.get("error_envelope_fields"), response.get("error_fields")],
            [["error"], ["code", "message"]],
        ),
        _check(
            "domain_refusal_is_http_success",
            response.get("domain_refusal_http_status") == 200
            and response.get("domain_refusal_is_transport_error") is False,
            [
                response.get("domain_refusal_http_status"),
                response.get("domain_refusal_is_transport_error"),
            ],
            [200, False],
        ),
        _check(
            "input_error_statuses_exact",
            _status_code(errors, "malformed_json") == 400
            and _status_code(errors, "request_body_too_large") == 413
            and _status_code(errors, "unsupported_media_type") == 415
            and _status_code(errors, "schema_validation_failed") == 422
            and _status_code(errors, "facade_invalid_request") == 422,
            [
                _status_code(errors, "malformed_json"),
                _status_code(errors, "request_body_too_large"),
                _status_code(errors, "unsupported_media_type"),
                _status_code(errors, "schema_validation_failed"),
                _status_code(errors, "facade_invalid_request"),
            ],
            [400, 413, 415, 422, 422],
        ),
        _check(
            "capacity_and_lifecycle_map_to_503",
            all(
                _status_code(errors, key) == 503
                for key in (
                    "facade_not_active",
                    "capacity_exceeded",
                    "facade_draining",
                    "facade_closed",
                )
            ),
            [
                _status_code(errors, key)
                for key in (
                    "facade_not_active",
                    "capacity_exceeded",
                    "facade_draining",
                    "facade_closed",
                )
            ],
            [503, 503, 503, 503],
        ),
        _check(
            "unexpected_error_is_generic_500",
            errors.get("unexpected_downstream_error") == {"status": 500, "code": "internal_error"},
            errors.get("unexpected_downstream_error"),
            {"status": 500, "code": "internal_error"},
        ),
        _check(
            "known_disconnect_has_no_fake_http_response",
            errors.get("cancelled_before_dispatch")
            == {"status": None, "code": "client_disconnected"}
            and disconnect.get("response_attempted_after_known_disconnect") is False,
            [
                errors.get("cancelled_before_dispatch"),
                disconnect.get("response_attempted_after_known_disconnect"),
            ],
            [{"status": None, "code": "client_disconnected"}, False],
        ),
        _check(
            "disconnect_is_predispatch_cooperative_only",
            disconnect.get("check_after_body_before_dispatch") is True
            and disconnect.get("predispatch_disconnect_sets_facade_cancellation") is True
            and disconnect.get("predispatch_disconnect_reaches_runtime") is False
            and disconnect.get("in_flight_hard_cancellation_supported") is False,
            [
                disconnect.get("check_after_body_before_dispatch"),
                disconnect.get("predispatch_disconnect_sets_facade_cancellation"),
                disconnect.get("predispatch_disconnect_reaches_runtime"),
                disconnect.get("in_flight_hard_cancellation_supported"),
            ],
            [True, True, False, False],
        ),
        _check(
            "disconnect_race_is_not_hidden",
            disconnect.get("disconnect_race_eliminated") is False
            and disconnect.get("in_flight_disconnect_claimed_as_cancelled") is False
            and disconnect.get("in_flight_work_completes_or_raises_naturally") is True,
            [
                disconnect.get("disconnect_race_eliminated"),
                disconnect.get("in_flight_disconnect_claimed_as_cancelled"),
                disconnect.get("in_flight_work_completes_or_raises_naturally"),
            ],
            [False, False, True],
        ),
        _check(
            "lifespan_owns_facade_not_runtime_resources",
            lifespan.get("fastapi_lifespan_context_required") is True
            and lifespan.get("shutdown_calls_facade_shutdown") is True
            and lifespan.get("transport_owns_runtime_resources") is False
            and lifespan.get("process_bootstrap_retains_runtime_resource_ownership") is True,
            [
                lifespan.get("fastapi_lifespan_context_required"),
                lifespan.get("shutdown_calls_facade_shutdown"),
                lifespan.get("transport_owns_runtime_resources"),
                lifespan.get("process_bootstrap_retains_runtime_resource_ownership"),
            ],
            [True, True, False, True],
        ),
        _check(
            "shutdown_is_natural_without_timeout_or_force_cancel",
            lifespan.get("shutdown_rejects_new_before_wait") is True
            and lifespan.get("shutdown_waits_for_in_flight_naturally") is True
            and lifespan.get("implicit_shutdown_timeout_seconds") is None
            and lifespan.get("force_cancel") is False,
            [
                lifespan.get("shutdown_rejects_new_before_wait"),
                lifespan.get("shutdown_waits_for_in_flight_naturally"),
                lifespan.get("implicit_shutdown_timeout_seconds"),
                lifespan.get("force_cancel"),
            ],
            [True, True, None, False],
        ),
        _check(
            "liveness_is_resource_independent",
            health.get("liveness_success_status") == 200
            and health.get("liveness_loads_or_probes_models") is False
            and health.get("liveness_depends_on_facade_acceptance") is False,
            [
                health.get("liveness_success_status"),
                health.get("liveness_loads_or_probes_models"),
                health.get("liveness_depends_on_facade_acceptance"),
            ],
            [200, False, False],
        ),
        _check(
            "readiness_requires_accepting_facade",
            health.get("readiness_accepting_status") == 200
            and health.get("readiness_nonaccepting_status") == 503
            and health.get("readiness_requires_facade_accepting") is True,
            [
                health.get("readiness_accepting_status"),
                health.get("readiness_nonaccepting_status"),
                health.get("readiness_requires_facade_accepting"),
            ],
            [200, 503, True],
        ),
        _check(
            "sync_facade_is_off_event_loop_without_app_queue",
            execution.get("event_loop_blocked_by_sync_facade") is False
            and execution.get("sync_facade_runs_off_event_loop") is True
            and execution.get("application_owned_waiting_queue") is False
            and execution.get("runtime_capacity_rejection_remains_nonblocking") is True,
            [
                execution.get("event_loop_blocked_by_sync_facade"),
                execution.get("sync_facade_runs_off_event_loop"),
                execution.get("application_owned_waiting_queue"),
                execution.get("runtime_capacity_rejection_remains_nonblocking"),
            ],
            [False, True, False, True],
        ),
        _check(
            "no_request_timeout_retry_or_fallback",
            execution.get("request_timeout_seconds") is None
            and execution.get("automatic_retry") is False
            and execution.get("fallback") is False,
            [
                execution.get("request_timeout_seconds"),
                execution.get("automatic_retry"),
                execution.get("fallback"),
            ],
            [None, False, False],
        ),
        _check(
            "default_access_log_disabled",
            logging.get("default_uvicorn_access_log_enabled") is False,
            logging.get("default_uvicorn_access_log_enabled"),
            False,
        ),
        _check(
            "structured_logging_allowlist_exact",
            logging.get("structured_allowlist_only") is True
            and logging.get("allowed_fields") == list(_PUBLIC_LOG_FIELDS),
            [logging.get("structured_allowlist_only"), logging.get("allowed_fields")],
            [True, list(_PUBLIC_LOG_FIELDS)],
        ),
        _check(
            "logging_contains_no_private_content_or_identifiers",
            logging.get("request_or_response_content_logged") is False
            and logging.get("request_or_document_identifiers_logged") is False
            and logging.get("headers_cookies_or_client_address_logged") is False
            and logging.get("exception_message_logged_publicly") is False,
            [
                logging.get("request_or_response_content_logged"),
                logging.get("request_or_document_identifiers_logged"),
                logging.get("headers_cookies_or_client_address_logged"),
                logging.get("exception_message_logged_publicly"),
            ],
            [False, False, False, False],
        ),
        _check(
            "all_closed_boundaries_remain_closed",
            all(value is False for value in closed.values()),
            closed,
            "all false",
        ),
        _check(
            "exact_policy_case_is_eligible",
            exact.get("state") == "eligible" and exact.get("rejection_reasons") == [],
            exact,
            {"state": "eligible", "rejection_reasons": []},
        ),
        _check(
            "all_five_negative_cases_are_rejected",
            len(negative) == 5 and all(value.get("state") == "rejected" for value in negative),
            [value.get("state") for value in negative],
            ["rejected"] * 5,
        ),
        _check(
            "unsafe_recovery_case_detects_queue_retry_fallback",
            set(
                (evaluations.get("unsafe_health_logging_or_recovery") or {}).get(
                    "rejection_reasons", []
                )
            )
            >= {"queue_action_detected", "retry_action_detected", "fallback_action_detected"},
            (evaluations.get("unsafe_health_logging_or_recovery") or {}).get(
                "rejection_reasons", []
            ),
            "contains queue/retry/fallback rejection reasons",
        ),
        _check(
            "stage148_source_unchanged_after_protocol_freeze",
            report.get("source_unchanged_after_protocol_freeze") is True,
            report.get("source_unchanged_after_protocol_freeze"),
            True,
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
        "status": _FINAL_STATUS
        if passed
        else "primeqa_hybrid_agent_http_transport_protocol_rejected",
        "failed_checks": failed,
        "agent_http_transport_protocol_frozen": passed,
        "transport_protocol_policy_executable": passed,
        "local_fastapi_implementation_allowed_next": passed,
        "network_service_implemented": False,
        "local_loopback_only": True,
        "remote_deployment_authorized": False,
        "runtime_registered_as_default": False,
        "runtime_defaultization_allowed_now": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
        "queue_actions_enabled": False,
        "retry_actions_enabled": False,
        "fallback_strategies_enabled": False,
        "hard_cancellation_enabled": False,
        "implicit_shutdown_timeout_enabled": False,
        "next_direction": _NEXT_DIRECTION if passed else "repair_failed_stage149_guards",
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_kind": "saved_public_stage148_aggregate_only",
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
        "network_service_started": False,
        "network_port_bound": False,
        "forbidden_keys_found": sorted(_forbidden_keys_found(report)),
    }


def _route_bars(protocol: Mapping[str, Any]) -> list[BarDatum]:
    routes = (protocol.get("route_contract") or {}).get("routes") or []
    return [
        BarDatum(label=str(route["route_id"]), value=1.0, value_label=str(route["method"]))
        for route in routes
    ]


def _limit_bars(protocol: Mapping[str, Any]) -> list[BarDatum]:
    request = protocol.get("request_contract") or {}
    keys = (
        "raw_body_max_bytes",
        "request_handle_max_chars",
        "title_max_chars",
        "text_max_chars",
    )
    return [
        BarDatum(label=key, value=float(request.get(key, 0)), value_label=str(request.get(key, 0)))
        for key in keys
    ]


def _status_bars(protocol: Mapping[str, Any]) -> list[BarDatum]:
    errors = protocol.get("http_error_mapping") or {}
    return [
        BarDatum(
            label=str(name),
            value=float(mapping.get("status") or 0),
            value_label="no response"
            if mapping.get("status") is None
            else str(mapping.get("status")),
        )
        for name, mapping in errors.items()
    ]


def _logging_bars(protocol: Mapping[str, Any]) -> list[BarDatum]:
    logging = protocol.get("logging_contract") or {}
    return [
        BarDatum(
            label="allowed_fields",
            value=float(len(logging.get("allowed_fields") or [])),
            value_label=str(len(logging.get("allowed_fields") or [])),
        ),
        BarDatum(label="private_fields", value=0.0, value_label="0"),
    ]


def _section_flag_bars(protocol: Mapping[str, Any], section: str) -> list[BarDatum]:
    values = protocol.get(section) or {}
    return _flag_bars(
        values, tuple(key for key, value in values.items() if isinstance(value, bool))
    )


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
    width: int = 1800,
    margin_left: int = 940,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        width=width,
        margin_left=margin_left,
    )


def _status_code(errors: Mapping[str, Any], key: str) -> Any:
    return (errors.get(key) or {}).get("status")


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
    return {"name": name, "passed": bool(passed), "observed": observed, "expected": expected}


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
