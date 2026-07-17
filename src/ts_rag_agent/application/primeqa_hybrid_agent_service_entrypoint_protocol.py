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

_STAGE = "Stage 151"
_CREATED_AT = "2026-07-18"
_PROTOCOL_ID = "primeqa_hybrid_local_agent_service_entrypoint_protocol_v1"
_SOURCE_ANALYSIS_ID = "primeqa_hybrid_local_fastapi_agent_transport_validation_v1"
_SOURCE_STATUS = "primeqa_hybrid_local_fastapi_agent_transport_validation_passed"
_EXPECTED_SOURCE_GUARDS = 37
_FINAL_STATUS = "primeqa_hybrid_local_agent_service_entrypoint_protocol_frozen"
_NEXT_DIRECTION = "implement_nondefault_local_agent_service_entrypoint"
_ENTRYPOINT_MODULE = "ts_rag_agent.local_agent_service"
_MIN_PORT = 1024
_MAX_PORT = 65535
_PUBLIC_STARTUP_FIELDS = (
    "entrypoint_id",
    "phase",
    "outcome_code",
    "exit_code",
    "binding_host",
    "binding_port",
    "source_validation_state",
    "runtime_activation_state",
    "resources_initialized",
    "warmup_completed",
    "listener_bound",
    "server_started",
    "shutdown_trigger",
    "transport_state",
    "runtime_registered_as_default",
    "queue_action_count",
    "retry_action_count",
    "fallback_action_count",
)
_STARTUP_ORDER = (
    "parse_exact_cli",
    "load_and_validate_stage150_public_report",
    "validate_explicit_runtime_and_transport_flags",
    "load_and_validate_stage145_public_report",
    "load_frozen_retrieval_protocols",
    "construct_process_resource_factory",
    "build_shared_resources_once",
    "run_builtin_label_free_synthetic_warmup",
    "create_fastapi_app_and_uvicorn_config",
    "prebind_exact_loopback_listener_once",
    "run_uvicorn_server_on_main_thread",
    "uvicorn_stops_accepting_and_waits_http_tasks",
    "lifespan_drains_transport",
    "entrypoint_finally_confirms_listener_closed",
    "release_process_references",
)
_CANONICAL_SOURCE_KEYS = (
    "stage150_http_transport_validation",
    "stage145_concurrent_runtime_validation",
    "stage128_agent_retrieval_protocol",
    "stage125_recall_expansion_protocol",
    "stage80_dense_sparse_report",
    "primeqa_technote_documents",
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
        "exception_message",
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
        "warmup_text",
    }
)


class AgentServiceEntrypointProtocolState(str, Enum):
    """Outcome of the strict Stage151 service composition policy."""

    REJECTED = "rejected"
    ELIGIBLE = "eligible"


@dataclass(frozen=True)
class AgentServiceEntrypointProtocolEvidence:
    """Evidence required before implementing the local service entrypoint."""

    source_stage150_validated: bool
    transport_implemented_and_socket_validated: bool
    transport_disabled_by_default: bool
    service_not_persistently_running: bool
    local_loopback_only: bool
    explicit_port_required: bool
    port_range_exact: bool
    port_zero_allowed: bool
    host_override_allowed: bool
    source_path_cli_overrides_allowed: bool
    runtime_and_transport_flags_explicit: bool
    stage150_checked_before_other_sources: bool
    stage145_revalidated_before_resource_build: bool
    resource_graph_built_once: bool
    label_free_synthetic_warmup: bool
    split_question_rows_loaded_for_warmup: bool
    socket_bound_after_warmup: bool
    socket_prebound_once_without_race: bool
    bind_retry_count: int
    single_process_single_worker: bool
    reload_enabled: bool
    server_runs_on_main_thread: bool
    uvicorn_owns_supported_signal_handlers: bool
    custom_signal_handlers_installed: bool
    lifespan_drains_transport_naturally: bool
    implicit_shutdown_timeout_enabled: bool
    force_cancel_enabled: bool
    entrypoint_closes_listener: bool
    public_startup_logging_allowlist_only: bool
    public_exception_message_enabled: bool
    queue_action_count: int
    retry_action_count: int
    fallback_action_count: int
    service_entrypoint_implemented: bool
    runtime_default_unchanged: bool
    test_split_locked: bool


@dataclass(frozen=True)
class AgentServiceEntrypointProtocolEvaluation:
    """Public-safe entrypoint protocol eligibility result."""

    state: AgentServiceEntrypointProtocolState
    rejection_reasons: tuple[str, ...]
    service_entrypoint_implemented: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "rejection_reasons": list(self.rejection_reasons),
            "service_entrypoint_implemented": self.service_entrypoint_implemented,
        }


class StrictAgentServiceEntrypointProtocolPolicy:
    """Fail-closed local service composition policy with no hidden recovery."""

    def evaluate(
        self,
        evidence: AgentServiceEntrypointProtocolEvidence,
    ) -> AgentServiceEntrypointProtocolEvaluation:
        reasons: list[str] = []
        _require(reasons, evidence.source_stage150_validated, "stage150_source_not_validated")
        _require(
            reasons,
            evidence.transport_implemented_and_socket_validated,
            "transport_or_socket_validation_missing",
        )
        _require(
            reasons,
            evidence.transport_disabled_by_default,
            "transport_default_enabled",
        )
        _require(
            reasons,
            evidence.service_not_persistently_running,
            "persistent_service_already_running",
        )
        _require(reasons, evidence.local_loopback_only, "loopback_binding_not_required")
        _require(reasons, evidence.explicit_port_required, "explicit_port_not_required")
        _require(reasons, evidence.port_range_exact, "port_range_not_exact")
        _forbid(reasons, evidence.port_zero_allowed, "ephemeral_port_allowed")
        _forbid(reasons, evidence.host_override_allowed, "host_override_allowed")
        _forbid(
            reasons,
            evidence.source_path_cli_overrides_allowed,
            "source_path_cli_override_allowed",
        )
        _require(
            reasons,
            evidence.runtime_and_transport_flags_explicit,
            "explicit_activation_flags_missing",
        )
        _require(
            reasons,
            evidence.stage150_checked_before_other_sources,
            "stage150_not_checked_first",
        )
        _require(
            reasons,
            evidence.stage145_revalidated_before_resource_build,
            "stage145_not_revalidated_before_resources",
        )
        _require(reasons, evidence.resource_graph_built_once, "resource_graph_not_single_build")
        _require(
            reasons,
            evidence.label_free_synthetic_warmup,
            "warmup_not_label_free_synthetic",
        )
        _forbid(
            reasons,
            evidence.split_question_rows_loaded_for_warmup,
            "split_question_loaded_for_warmup",
        )
        _require(reasons, evidence.socket_bound_after_warmup, "socket_bound_before_warmup")
        _require(
            reasons,
            evidence.socket_prebound_once_without_race,
            "socket_not_prebound_exactly_once",
        )
        if evidence.bind_retry_count != 0:
            reasons.append("bind_retry_detected")
        _require(
            reasons,
            evidence.single_process_single_worker,
            "single_process_single_worker_not_required",
        )
        _forbid(reasons, evidence.reload_enabled, "reload_enabled")
        _require(reasons, evidence.server_runs_on_main_thread, "server_not_on_main_thread")
        _require(
            reasons,
            evidence.uvicorn_owns_supported_signal_handlers,
            "uvicorn_signal_ownership_missing",
        )
        _forbid(
            reasons,
            evidence.custom_signal_handlers_installed,
            "custom_signal_handler_installed",
        )
        _require(
            reasons,
            evidence.lifespan_drains_transport_naturally,
            "natural_lifespan_drain_missing",
        )
        _forbid(
            reasons,
            evidence.implicit_shutdown_timeout_enabled,
            "implicit_shutdown_timeout_enabled",
        )
        _forbid(reasons, evidence.force_cancel_enabled, "force_cancel_enabled")
        _require(
            reasons,
            evidence.entrypoint_closes_listener,
            "entrypoint_listener_close_missing",
        )
        _require(
            reasons,
            evidence.public_startup_logging_allowlist_only,
            "public_startup_log_allowlist_missing",
        )
        _forbid(
            reasons,
            evidence.public_exception_message_enabled,
            "public_exception_message_enabled",
        )
        if evidence.queue_action_count != 0:
            reasons.append("queue_action_detected")
        if evidence.retry_action_count != 0:
            reasons.append("retry_action_detected")
        if evidence.fallback_action_count != 0:
            reasons.append("fallback_action_detected")
        _forbid(
            reasons,
            evidence.service_entrypoint_implemented,
            "service_entrypoint_preimplemented",
        )
        _require(reasons, evidence.runtime_default_unchanged, "runtime_default_changed")
        _require(reasons, evidence.test_split_locked, "test_split_not_locked")
        return AgentServiceEntrypointProtocolEvaluation(
            state=(
                AgentServiceEntrypointProtocolState.REJECTED
                if reasons
                else AgentServiceEntrypointProtocolState.ELIGIBLE
            ),
            rejection_reasons=tuple(reasons),
        )


@dataclass(frozen=True)
class PrimeQAHybridAgentServiceEntrypointProtocolVisualization:
    """One Stage151 aggregate/specification SVG."""

    name: str
    path: str


def freeze_primeqa_hybrid_agent_service_entrypoint_protocol(
    *,
    stage150_validation_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the local service entrypoint from the saved Stage150 public report."""

    started_at = time.perf_counter()
    source = _load_json_object(stage150_validation_path)
    loaded_at = time.perf_counter()
    source_summary = _stage150_summary(source)
    protocol = _frozen_protocol()
    evaluations = _canonical_evaluations(StrictAgentServiceEntrypointProtocolPolicy())
    source_unchanged = _load_json_object(stage150_validation_path) == source
    preliminary = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "protocol_scope": (
            "Aggregate-only freeze of a strict local service-entrypoint composition around "
            "the Stage150 FastAPI adapter. It reads only the saved Stage150 public report "
            "and runs synthetic policy cases. It does not load train, dev, test, questions, "
            "documents, models, indexes, or candidate pools; bind a port; implement or run "
            "a service entrypoint; change runtime defaults; or add queues, retries, fallback, "
            "reload, multiple workers, custom signal handlers, forced shutdown, or an implicit "
            "shutdown timeout."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_protocol),
            "confirmation_note": confirmation_note,
        },
        "source_files": {"stage150_validation": _fingerprint(stage150_validation_path)},
        "stage150_summary": source_summary,
        "source_unchanged_after_protocol_freeze": source_unchanged,
        "official_design_references": [
            {
                "topic": "uvicorn_programmatic_server_and_binding",
                "url": "https://www.uvicorn.org/settings/",
                "applied_fact": (
                    "Uvicorn supports programmatic configuration, loopback host binding, "
                    "explicit ports, and a configurable graceful-shutdown timeout."
                ),
            },
            {
                "topic": "uvicorn_graceful_process_shutdown",
                "url": "https://www.uvicorn.org/server-behavior/",
                "applied_fact": (
                    "Uvicorn graceful shutdown finalizes connections and background tasks."
                ),
            },
            {
                "topic": "python_signal_main_thread_boundary",
                "url": "https://docs.python.org/3.10/library/signal.html",
                "applied_fact": (
                    "Python signal handlers are installed from the main interpreter thread, "
                    "and supported signals vary by platform."
                ),
            },
        ],
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
        source_unchanged=source_unchanged,
    )
    completed_at = time.perf_counter()
    report = {
        **preliminary,
        "guard_checks": guards,
        "decision": _decision(guards),
        "timing_seconds": {
            "load_public_stage150_aggregate": round(loaded_at - started_at, 6),
            "freeze_and_check_protocol": round(completed_at - loaded_at, 6),
            "total": round(completed_at - started_at, 6),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_agent_service_entrypoint_protocol_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridAgentServiceEntrypointProtocolVisualization]:
    """Write Stage151 public-safe aggregate/specification charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = report.get("frozen_protocol") or {}
    invocation = protocol.get("invocation_contract") or {}
    sources = protocol.get("source_authorization_contract") or {}
    activation = protocol.get("activation_contract") or {}
    composition = protocol.get("runtime_composition_contract") or {}
    warmup = protocol.get("warmup_contract") or {}
    socket_contract = protocol.get("socket_contract") or {}
    process = protocol.get("process_signal_contract") or {}
    shutdown = protocol.get("shutdown_contract") or {}
    charts = {
        "stage151_source_gate.svg": _chart(
            "Stage151 Stage150 source gate",
            _flag_bars(
                report.get("stage150_summary") or {},
                (
                    "source_identity_valid",
                    "all_source_guards_passed",
                    "transport_implemented",
                    "real_loopback_socket_validated",
                    "closed_boundaries_preserved",
                ),
            ),
        ),
        "stage151_invocation_surface.svg": _chart(
            "Stage151 exact invocation surface",
            [
                _bar("required CLI options", len(invocation.get("required_options") or [])),
                _bar("optional CLI options", len(invocation.get("optional_options") or [])),
                _bar("forbidden CLI options", len(invocation.get("forbidden_options") or [])),
            ],
            x_label="option count",
        ),
        "stage151_port_policy.svg": _chart(
            "Stage151 strict local port policy",
            [
                _bar("minimum allowed port", invocation.get("minimum_port", 0)),
                _bar("maximum allowed port", invocation.get("maximum_port", 0)),
                _bar("port zero allowed", invocation.get("port_zero_allowed", False)),
                _bar("host override allowed", invocation.get("host_override_allowed", False)),
            ],
            x_label="port or boolean value",
        ),
        "stage151_source_order.svg": _chart(
            "Stage151 authorization and startup order",
            [
                BarDatum(label=str(name), value=float(index), value_label=str(index))
                for index, name in enumerate(sources.get("startup_order") or [], start=1)
            ],
            x_label="order",
            width=2300,
            margin_left=1200,
        ),
        "stage151_activation_gate.svg": _chart(
            "Stage151 explicit activation gate",
            _flag_bars(
                activation,
                (
                    "concurrent_runtime_flag_must_be_true",
                    "local_http_transport_flag_must_be_true",
                    "stage150_must_pass_before_resource_load",
                    "stage145_must_recompute_eligible",
                ),
            ),
        ),
        "stage151_resource_warmup.svg": _chart(
            "Stage151 resource and warmup boundary",
            [
                _bar("resource build count", composition.get("resource_build_count", 0)),
                _bar("warmup count", warmup.get("warmup_request_count", 0)),
                _bar("label-free synthetic", warmup.get("label_free_synthetic", False)),
                _bar("split rows loaded", warmup.get("split_question_rows_loaded", False)),
            ],
            x_label="count or boolean value",
        ),
        "stage151_socket_process.svg": _chart(
            "Stage151 socket and process model",
            [
                _bar("prebind count", socket_contract.get("listener_bind_attempt_count", 0)),
                _bar("bind retries", socket_contract.get("bind_retry_count", 0)),
                _bar("worker count", process.get("worker_count", 0)),
                _bar("reload enabled", process.get("reload_enabled", False)),
                _bar("main thread server", process.get("server_runs_on_main_thread", False)),
            ],
            x_label="count or boolean value",
        ),
        "stage151_shutdown_signal.svg": _chart(
            "Stage151 signal and natural shutdown",
            _flag_bars(
                {**process, **shutdown},
                (
                    "uvicorn_owns_supported_signal_handlers",
                    "custom_signal_handlers_installed",
                    "uvicorn_closes_listener_before_lifespan_shutdown",
                    "transport_closed_before_process_reference_release",
                    "force_cancel",
                    "signal_exit_code_normalized",
                ),
            ),
        ),
        "stage151_policy_cases.svg": _chart(
            "Stage151 canonical policy cases",
            [
                BarDatum(
                    label=str(name),
                    value=1.0 if row.get("state") == "eligible" else 0.0,
                    value_label=str(row.get("state")),
                )
                for name, row in (report.get("canonical_policy_evaluations") or {}).items()
            ],
        ),
        "stage151_guard_check_status.svg": _chart(
            "Stage151 protocol guard checks",
            [
                BarDatum(
                    label=str(check["name"]),
                    value=1.0 if check["passed"] else 0.0,
                    value_label="passed" if check["passed"] else "failed",
                )
                for check in report.get("guard_checks", [])
            ],
            width=2600,
            margin_left=1480,
        ),
    }
    artifacts: list[PrimeQAHybridAgentServiceEntrypointProtocolVisualization] = []
    for name, content in charts.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridAgentServiceEntrypointProtocolVisualization(
                name=name,
                path=str(path),
            )
        )
    return artifacts


def _frozen_protocol() -> dict[str, Any]:
    return {
        "entrypoint_identity": {
            "entrypoint_id": _PROTOCOL_ID,
            "module": _ENTRYPOINT_MODULE,
            "service_scope": "local_research_only",
            "binding_host": "127.0.0.1",
            "http_protocol": "HTTP/1.1",
            "persistent_service_implemented": False,
        },
        "invocation_contract": {
            "command": f"python -m {_ENTRYPOINT_MODULE} --port <PORT>",
            "required_options": ["--port"],
            "optional_options": [],
            "forbidden_options": [
                "--host",
                "--reload",
                "--workers",
                "--uds",
                "--fd",
                "--stage150-validation",
                "--stage145-validation",
                "--documents",
                "--encoder-device",
                "--encoder-batch-size",
            ],
            "port_has_default": False,
            "minimum_port": _MIN_PORT,
            "maximum_port": _MAX_PORT,
            "port_zero_allowed": False,
            "host_override_allowed": False,
            "unknown_options_rejected": True,
            "source_path_cli_overrides_allowed": False,
        },
        "source_authorization_contract": {
            "canonical_source_keys": list(_CANONICAL_SOURCE_KEYS),
            "source_roots_from_project_settings": True,
            "source_files_read_only": True,
            "source_fingerprints_recorded": True,
            "stage150_checked_first": True,
            "stage145_recomputed_before_resource_build": True,
            "startup_order": list(_STARTUP_ORDER),
            "source_rejection_builds_resources": False,
            "source_rejection_binds_socket": False,
        },
        "activation_contract": {
            "concurrent_runtime_flag": "TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT",
            "local_http_transport_flag": "TS_RAG_ENABLE_LOCAL_AGENT_HTTP_TRANSPORT",
            "concurrent_runtime_flag_must_be_true": True,
            "local_http_transport_flag_must_be_true": True,
            "stage150_must_pass_before_resource_load": True,
            "stage145_must_recompute_eligible": True,
            "runtime_registered_as_default": False,
            "implicit_activation_allowed": False,
        },
        "runtime_composition_contract": {
            "resource_factory": "PrimeQAHybridProcessRuntimeResourceFactory",
            "bootstrap": "PrimeQAHybridConcurrentRuntimeBootstrap",
            "http_app_factory": "create_primeqa_hybrid_agent_http_app",
            "uvicorn_config_factory": "create_primeqa_hybrid_agent_uvicorn_config",
            "resource_build_count": 1,
            "entrypoint_owns_resource_factory_reference": True,
            "bootstrap_result_owns_active_runtime_reference": True,
            "transport_owns_runtime_resources": False,
            "resource_close_interface_claimed": False,
            "process_releases_references_after_server_return": True,
        },
        "warmup_contract": {
            "input_model": "PrimeQARuntimeQuery",
            "source": "built_in_synthetic_label_free",
            "label_free_synthetic": True,
            "warmup_request_count": 1,
            "split_question_rows_loaded": False,
            "gold_fields_present": False,
            "warmup_content_logged": False,
            "warmup_must_complete_before_socket_bind": True,
            "bootstrap_signature_refactor_required": True,
        },
        "socket_contract": {
            "binding_host": "127.0.0.1",
            "listener_owner": "service_entrypoint",
            "listener_created_after_warmup": True,
            "listener_prebound_before_server_run": True,
            "listener_passed_to_uvicorn_server": True,
            "listener_bind_attempt_count": 1,
            "bind_retry_count": 0,
            "alternate_port_fallback": False,
            "listener_closed_after_server_return": True,
            "port_value_written_to_startup_event": True,
            "remote_binding_allowed": False,
        },
        "process_signal_contract": {
            "process_count": 1,
            "worker_count": 1,
            "reload_enabled": False,
            "server_api": "uvicorn.Config_plus_uvicorn.Server.run",
            "server_runs_on_main_thread": True,
            "uvicorn_owns_supported_signal_handlers": True,
            "custom_signal_handlers_installed": False,
            "supported_signals_are_platform_and_uvicorn_defined": True,
            "signal_exit_code_normalized": False,
            "second_signal_force_exit_behavior_overridden": False,
        },
        "shutdown_contract": {
            "uvicorn_stops_accepting_connections": True,
            "uvicorn_closes_listener_before_lifespan_shutdown": True,
            "uvicorn_waits_http_tasks_before_lifespan_shutdown": True,
            "fastapi_lifespan_shutdown_runs": True,
            "transport_lifespan_shutdown_after_http_tasks": True,
            "transport_closed_before_process_reference_release": True,
            "in_flight_work_completes_or_raises_naturally": True,
            "implicit_shutdown_timeout_seconds": None,
            "force_cancel": False,
            "entrypoint_listener_close_is_idempotent_finally": True,
            "runtime_resource_close_interface_claimed": False,
        },
        "exit_status_contract": {
            "clean_server_return": 0,
            "unexpected_composition_failure": 1,
            "cli_contract_invalid": 2,
            "stage150_authorization_rejected": 3,
            "activation_configuration_rejected": 4,
            "stage145_or_runtime_activation_rejected": 5,
            "resource_or_warmup_failure": 6,
            "socket_bind_or_listen_failure": 7,
            "server_or_lifespan_failure": 8,
            "external_signal_exit_code": "not_normalized_platform_and_uvicorn_defined",
            "startup_failure_retry_count": 0,
        },
        "observability_contract": {
            "public_startup_event_fields": list(_PUBLIC_STARTUP_FIELDS),
            "public_startup_event_field_count": len(_PUBLIC_STARTUP_FIELDS),
            "exactly_one_terminal_event": True,
            "request_access_log_enabled": False,
            "request_or_response_content_in_public_event": False,
            "warmup_content_in_public_event": False,
            "source_paths_in_public_event": False,
            "exception_message_in_public_event": False,
            "exception_type_may_be_in_public_event": True,
            "uvicorn_framework_error_logging_is_separate": True,
        },
        "closed_boundaries": {
            "service_entrypoint_implemented": False,
            "network_service_started": False,
            "network_port_bound": False,
            "remote_exposure_authorized": False,
            "runtime_registered_as_default": False,
            "test_gate_opened": False,
            "test_metrics_run": False,
            "queue_actions_enabled": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
        },
    }


def _canonical_evaluations(
    policy: StrictAgentServiceEntrypointProtocolPolicy,
) -> dict[str, dict[str, Any]]:
    cases = {
        "exact_local_entrypoint_contract": _compliant_evidence(),
        "remote_dynamic_server_surface": _compliant_evidence(
            explicit_port_required=False,
            port_range_exact=False,
            port_zero_allowed=True,
            host_override_allowed=True,
            single_process_single_worker=False,
            reload_enabled=True,
        ),
        "source_order_and_override_drift": _compliant_evidence(
            source_path_cli_overrides_allowed=True,
            stage150_checked_before_other_sources=False,
            stage145_revalidated_before_resource_build=False,
        ),
        "label_bearing_split_warmup": _compliant_evidence(
            label_free_synthetic_warmup=False,
            split_question_rows_loaded_for_warmup=True,
            socket_bound_after_warmup=False,
        ),
        "hidden_recovery_and_custom_signal": _compliant_evidence(
            bind_retry_count=1,
            server_runs_on_main_thread=False,
            uvicorn_owns_supported_signal_handlers=False,
            custom_signal_handlers_installed=True,
            implicit_shutdown_timeout_enabled=True,
            force_cancel_enabled=True,
            queue_action_count=1,
            retry_action_count=1,
            fallback_action_count=1,
        ),
        "preimplemented_default_test_open": _compliant_evidence(
            service_not_persistently_running=False,
            service_entrypoint_implemented=True,
            runtime_default_unchanged=False,
            test_split_locked=False,
        ),
    }
    return {name: policy.evaluate(evidence).to_public_dict() for name, evidence in cases.items()}


def _compliant_evidence(**overrides: Any) -> AgentServiceEntrypointProtocolEvidence:
    values: dict[str, Any] = {
        "source_stage150_validated": True,
        "transport_implemented_and_socket_validated": True,
        "transport_disabled_by_default": True,
        "service_not_persistently_running": True,
        "local_loopback_only": True,
        "explicit_port_required": True,
        "port_range_exact": True,
        "port_zero_allowed": False,
        "host_override_allowed": False,
        "source_path_cli_overrides_allowed": False,
        "runtime_and_transport_flags_explicit": True,
        "stage150_checked_before_other_sources": True,
        "stage145_revalidated_before_resource_build": True,
        "resource_graph_built_once": True,
        "label_free_synthetic_warmup": True,
        "split_question_rows_loaded_for_warmup": False,
        "socket_bound_after_warmup": True,
        "socket_prebound_once_without_race": True,
        "bind_retry_count": 0,
        "single_process_single_worker": True,
        "reload_enabled": False,
        "server_runs_on_main_thread": True,
        "uvicorn_owns_supported_signal_handlers": True,
        "custom_signal_handlers_installed": False,
        "lifespan_drains_transport_naturally": True,
        "implicit_shutdown_timeout_enabled": False,
        "force_cancel_enabled": False,
        "entrypoint_closes_listener": True,
        "public_startup_logging_allowlist_only": True,
        "public_exception_message_enabled": False,
        "queue_action_count": 0,
        "retry_action_count": 0,
        "fallback_action_count": 0,
        "service_entrypoint_implemented": False,
        "runtime_default_unchanged": True,
        "test_split_locked": True,
    }
    values.update(overrides)
    return AgentServiceEntrypointProtocolEvidence(**values)


def _stage150_summary(source: Mapping[str, Any]) -> dict[str, Any]:
    decision = source.get("decision") or {}
    public = source.get("public_safe_contract") or {}
    contract = source.get("implementation_contract") or {}
    checks = source.get("guard_checks") or []
    passed_count = sum(check.get("passed") is True for check in checks)
    identity_valid = (
        source.get("stage") == "Stage 150"
        and source.get("analysis_id") == _SOURCE_ANALYSIS_ID
        and decision.get("status") == _SOURCE_STATUS
    )
    transport_exact = (
        contract.get("binding_host") == "127.0.0.1"
        and contract.get("http_protocol") == "HTTP/1.1"
        and contract.get("max_in_flight") == 4
        and contract.get("application_waiting_queue") is False
        and contract.get("request_timeout_seconds") is None
        and contract.get("implicit_shutdown_timeout_seconds") is None
    )
    closed = (
        decision.get("network_service_persistently_running") is False
        and decision.get("runtime_registered_as_default") is False
        and decision.get("remote_exposure_authorized") is False
        and decision.get("test_gate_opened") is False
        and decision.get("queue_actions_enabled") is False
        and decision.get("retry_actions_enabled") is False
        and decision.get("fallback_strategies_enabled") is False
        and public.get("network_service_persistently_running") is False
        and public.get("test_split_loaded") is False
        and public.get("test_metrics_run") is False
        and public.get("private_keys_found") == []
    )
    return {
        "source_identity_valid": identity_valid,
        "source_guard_count": len(checks),
        "source_passed_guard_count": passed_count,
        "all_source_guards_passed": (
            len(checks) == _EXPECTED_SOURCE_GUARDS and passed_count == _EXPECTED_SOURCE_GUARDS
        ),
        "transport_implemented": decision.get("local_fastapi_transport_implemented"),
        "in_process_asgi_validated": decision.get("in_process_asgi_validation_passed"),
        "real_loopback_socket_validated": decision.get("real_loopback_socket_validation_passed"),
        "transport_disabled_by_default": decision.get("disabled_by_default"),
        "local_loopback_only": decision.get("local_loopback_only"),
        "transport_contract_exact": transport_exact,
        "network_service_persistently_running": decision.get(
            "network_service_persistently_running"
        ),
        "runtime_registered_as_default": decision.get("runtime_registered_as_default"),
        "closed_boundaries_preserved": closed,
        "test_split_loaded": public.get("test_split_loaded"),
        "test_metrics_run": public.get("test_metrics_run"),
        "private_keys_found": list(public.get("private_keys_found") or []),
    }


def _guard_checks(
    *,
    report: Mapping[str, Any],
    source_summary: Mapping[str, Any],
    protocol: Mapping[str, Any],
    evaluations: Mapping[str, Mapping[str, Any]],
    user_confirmed_protocol: bool,
    confirmation_note: str,
    source_unchanged: bool,
) -> list[dict[str, Any]]:
    identity = protocol.get("entrypoint_identity") or {}
    invocation = protocol.get("invocation_contract") or {}
    sources = protocol.get("source_authorization_contract") or {}
    activation = protocol.get("activation_contract") or {}
    composition = protocol.get("runtime_composition_contract") or {}
    warmup = protocol.get("warmup_contract") or {}
    socket_contract = protocol.get("socket_contract") or {}
    process = protocol.get("process_signal_contract") or {}
    shutdown = protocol.get("shutdown_contract") or {}
    exits = protocol.get("exit_status_contract") or {}
    observability = protocol.get("observability_contract") or {}
    closed = protocol.get("closed_boundaries") or {}
    expected_states = {
        "exact_local_entrypoint_contract": "eligible",
        "remote_dynamic_server_surface": "rejected",
        "source_order_and_override_drift": "rejected",
        "label_bearing_split_warmup": "rejected",
        "hidden_recovery_and_custom_signal": "rejected",
        "preimplemented_default_test_open": "rejected",
    }
    return [
        _check("stage151_user_confirmed", user_confirmed_protocol, user_confirmed_protocol, True),
        _check(
            "stage151_confirmation_note_present",
            bool(confirmation_note.strip()),
            bool(confirmation_note.strip()),
            True,
        ),
        _check(
            "stage150_source_identity_valid",
            source_summary.get("source_identity_valid") is True,
            source_summary.get("source_identity_valid"),
            True,
        ),
        _check(
            "stage150_all_37_guards_passed",
            source_summary.get("all_source_guards_passed") is True,
            source_summary.get("source_passed_guard_count"),
            _EXPECTED_SOURCE_GUARDS,
        ),
        _check(
            "stage150_transport_and_socket_validation_passed",
            source_summary.get("transport_implemented") is True
            and source_summary.get("in_process_asgi_validated") is True
            and source_summary.get("real_loopback_socket_validated") is True,
            [
                source_summary.get("transport_implemented"),
                source_summary.get("in_process_asgi_validated"),
                source_summary.get("real_loopback_socket_validated"),
            ],
            [True, True, True],
        ),
        _check(
            "stage150_transport_contract_and_defaults_exact",
            source_summary.get("transport_disabled_by_default") is True
            and source_summary.get("local_loopback_only") is True
            and source_summary.get("transport_contract_exact") is True,
            [
                source_summary.get("transport_disabled_by_default"),
                source_summary.get("local_loopback_only"),
                source_summary.get("transport_contract_exact"),
            ],
            [True, True, True],
        ),
        _check(
            "stage150_closed_boundaries_preserved",
            source_summary.get("closed_boundaries_preserved") is True,
            source_summary.get("closed_boundaries_preserved"),
            True,
        ),
        _check(
            "stage150_source_unchanged_after_freeze",
            source_unchanged,
            source_unchanged,
            True,
        ),
        _check(
            "entrypoint_identity_exact",
            identity
            == {
                "entrypoint_id": _PROTOCOL_ID,
                "module": _ENTRYPOINT_MODULE,
                "service_scope": "local_research_only",
                "binding_host": "127.0.0.1",
                "http_protocol": "HTTP/1.1",
                "persistent_service_implemented": False,
            },
            identity,
            "exact local entrypoint identity",
        ),
        _check(
            "entrypoint_cli_has_only_required_port",
            invocation.get("required_options") == ["--port"]
            and invocation.get("optional_options") == []
            and invocation.get("unknown_options_rejected") is True,
            [
                invocation.get("required_options"),
                invocation.get("optional_options"),
                invocation.get("unknown_options_rejected"),
            ],
            [["--port"], [], True],
        ),
        _check(
            "entrypoint_port_range_is_strict_and_non_ephemeral",
            invocation.get("port_has_default") is False
            and invocation.get("minimum_port") == _MIN_PORT
            and invocation.get("maximum_port") == _MAX_PORT
            and invocation.get("port_zero_allowed") is False,
            [
                invocation.get("port_has_default"),
                invocation.get("minimum_port"),
                invocation.get("maximum_port"),
                invocation.get("port_zero_allowed"),
            ],
            [False, _MIN_PORT, _MAX_PORT, False],
        ),
        _check(
            "host_and_source_path_overrides_forbidden",
            invocation.get("host_override_allowed") is False
            and invocation.get("source_path_cli_overrides_allowed") is False
            and all(
                option in (invocation.get("forbidden_options") or [])
                for option in ("--host", "--stage150-validation", "--documents")
            ),
            [
                invocation.get("host_override_allowed"),
                invocation.get("source_path_cli_overrides_allowed"),
                invocation.get("forbidden_options"),
            ],
            "no host or source-path override",
        ),
        _check(
            "canonical_sources_and_startup_order_exact",
            sources.get("canonical_source_keys") == list(_CANONICAL_SOURCE_KEYS)
            and sources.get("startup_order") == list(_STARTUP_ORDER),
            [sources.get("canonical_source_keys"), sources.get("startup_order")],
            [list(_CANONICAL_SOURCE_KEYS), list(_STARTUP_ORDER)],
        ),
        _check(
            "source_rejection_precedes_resources_and_socket",
            sources.get("stage150_checked_first") is True
            and sources.get("stage145_recomputed_before_resource_build") is True
            and sources.get("source_rejection_builds_resources") is False
            and sources.get("source_rejection_binds_socket") is False,
            [
                sources.get("stage150_checked_first"),
                sources.get("stage145_recomputed_before_resource_build"),
                sources.get("source_rejection_builds_resources"),
                sources.get("source_rejection_binds_socket"),
            ],
            [True, True, False, False],
        ),
        _check(
            "activation_requires_both_flags_and_both_sources",
            all(
                activation.get(key) is True
                for key in (
                    "concurrent_runtime_flag_must_be_true",
                    "local_http_transport_flag_must_be_true",
                    "stage150_must_pass_before_resource_load",
                    "stage145_must_recompute_eligible",
                )
            )
            and activation.get("implicit_activation_allowed") is False,
            activation,
            "two explicit flags and two source gates",
        ),
        _check(
            "resource_graph_built_once_with_clear_ownership",
            composition.get("resource_build_count") == 1
            and composition.get("entrypoint_owns_resource_factory_reference") is True
            and composition.get("bootstrap_result_owns_active_runtime_reference") is True
            and composition.get("transport_owns_runtime_resources") is False,
            [
                composition.get("resource_build_count"),
                composition.get("entrypoint_owns_resource_factory_reference"),
                composition.get("bootstrap_result_owns_active_runtime_reference"),
                composition.get("transport_owns_runtime_resources"),
            ],
            [1, True, True, False],
        ),
        _check(
            "resource_close_claim_is_honest",
            composition.get("resource_close_interface_claimed") is False
            and composition.get("process_releases_references_after_server_return") is True,
            [
                composition.get("resource_close_interface_claimed"),
                composition.get("process_releases_references_after_server_return"),
            ],
            [False, True],
        ),
        _check(
            "warmup_is_builtin_label_free_and_split_free",
            warmup.get("input_model") == "PrimeQARuntimeQuery"
            and warmup.get("source") == "built_in_synthetic_label_free"
            and warmup.get("label_free_synthetic") is True
            and warmup.get("warmup_request_count") == 1
            and warmup.get("split_question_rows_loaded") is False
            and warmup.get("gold_fields_present") is False,
            [
                warmup.get("input_model"),
                warmup.get("source"),
                warmup.get("label_free_synthetic"),
                warmup.get("warmup_request_count"),
                warmup.get("split_question_rows_loaded"),
                warmup.get("gold_fields_present"),
            ],
            ["PrimeQARuntimeQuery", "built_in_synthetic_label_free", True, 1, False, False],
        ),
        _check(
            "warmup_type_refactor_required_before_implementation",
            warmup.get("bootstrap_signature_refactor_required") is True
            and warmup.get("warmup_must_complete_before_socket_bind") is True,
            [
                warmup.get("bootstrap_signature_refactor_required"),
                warmup.get("warmup_must_complete_before_socket_bind"),
            ],
            [True, True],
        ),
        _check(
            "listener_prebound_once_after_warmup",
            socket_contract.get("listener_created_after_warmup") is True
            and socket_contract.get("listener_prebound_before_server_run") is True
            and socket_contract.get("listener_passed_to_uvicorn_server") is True
            and socket_contract.get("listener_bind_attempt_count") == 1,
            [
                socket_contract.get("listener_created_after_warmup"),
                socket_contract.get("listener_prebound_before_server_run"),
                socket_contract.get("listener_passed_to_uvicorn_server"),
                socket_contract.get("listener_bind_attempt_count"),
            ],
            [True, True, True, 1],
        ),
        _check(
            "listener_has_no_retry_fallback_or_remote_binding",
            socket_contract.get("bind_retry_count") == 0
            and socket_contract.get("alternate_port_fallback") is False
            and socket_contract.get("remote_binding_allowed") is False,
            [
                socket_contract.get("bind_retry_count"),
                socket_contract.get("alternate_port_fallback"),
                socket_contract.get("remote_binding_allowed"),
            ],
            [0, False, False],
        ),
        _check(
            "process_is_single_worker_main_thread_without_reload",
            process.get("process_count") == 1
            and process.get("worker_count") == 1
            and process.get("reload_enabled") is False
            and process.get("server_runs_on_main_thread") is True,
            [
                process.get("process_count"),
                process.get("worker_count"),
                process.get("reload_enabled"),
                process.get("server_runs_on_main_thread"),
            ],
            [1, 1, False, True],
        ),
        _check(
            "uvicorn_owns_platform_signal_behavior",
            process.get("uvicorn_owns_supported_signal_handlers") is True
            and process.get("custom_signal_handlers_installed") is False
            and process.get("supported_signals_are_platform_and_uvicorn_defined") is True
            and process.get("signal_exit_code_normalized") is False,
            [
                process.get("uvicorn_owns_supported_signal_handlers"),
                process.get("custom_signal_handlers_installed"),
                process.get("supported_signals_are_platform_and_uvicorn_defined"),
                process.get("signal_exit_code_normalized"),
            ],
            [True, False, True, False],
        ),
        _check(
            "shutdown_order_matches_uvicorn_then_lifespan",
            shutdown.get("uvicorn_stops_accepting_connections") is True
            and shutdown.get("uvicorn_closes_listener_before_lifespan_shutdown") is True
            and shutdown.get("uvicorn_waits_http_tasks_before_lifespan_shutdown") is True
            and shutdown.get("fastapi_lifespan_shutdown_runs") is True
            and shutdown.get("transport_lifespan_shutdown_after_http_tasks") is True
            and shutdown.get("transport_closed_before_process_reference_release") is True
            and shutdown.get("in_flight_work_completes_or_raises_naturally") is True
            and shutdown.get("entrypoint_listener_close_is_idempotent_finally") is True,
            [
                shutdown.get("uvicorn_stops_accepting_connections"),
                shutdown.get("uvicorn_closes_listener_before_lifespan_shutdown"),
                shutdown.get("uvicorn_waits_http_tasks_before_lifespan_shutdown"),
                shutdown.get("fastapi_lifespan_shutdown_runs"),
                shutdown.get("transport_lifespan_shutdown_after_http_tasks"),
                shutdown.get("transport_closed_before_process_reference_release"),
                shutdown.get("in_flight_work_completes_or_raises_naturally"),
                shutdown.get("entrypoint_listener_close_is_idempotent_finally"),
            ],
            [True] * 8,
        ),
        _check(
            "shutdown_has_no_timeout_force_cancel_or_false_close_claim",
            shutdown.get("implicit_shutdown_timeout_seconds") is None
            and shutdown.get("force_cancel") is False
            and shutdown.get("runtime_resource_close_interface_claimed") is False,
            [
                shutdown.get("implicit_shutdown_timeout_seconds"),
                shutdown.get("force_cancel"),
                shutdown.get("runtime_resource_close_interface_claimed"),
            ],
            [None, False, False],
        ),
        _check(
            "stable_startup_exit_statuses_and_no_retry",
            [
                exits.get(key)
                for key in (
                    "clean_server_return",
                    "unexpected_composition_failure",
                    "cli_contract_invalid",
                    "stage150_authorization_rejected",
                    "activation_configuration_rejected",
                    "stage145_or_runtime_activation_rejected",
                    "resource_or_warmup_failure",
                    "socket_bind_or_listen_failure",
                    "server_or_lifespan_failure",
                )
            ]
            == list(range(9))
            and exits.get("startup_failure_retry_count") == 0
            and exits.get("external_signal_exit_code")
            == "not_normalized_platform_and_uvicorn_defined",
            exits,
            "exit codes 0..8, no retry, signal code not normalized",
        ),
        _check(
            "startup_log_allowlist_is_exact_and_content_free",
            observability.get("public_startup_event_fields") == list(_PUBLIC_STARTUP_FIELDS)
            and observability.get("public_startup_event_field_count") == len(_PUBLIC_STARTUP_FIELDS)
            and observability.get("exactly_one_terminal_event") is True
            and observability.get("request_access_log_enabled") is False
            and observability.get("request_or_response_content_in_public_event") is False
            and observability.get("warmup_content_in_public_event") is False
            and observability.get("source_paths_in_public_event") is False
            and observability.get("exception_message_in_public_event") is False
            and observability.get("uvicorn_framework_error_logging_is_separate") is True,
            observability,
            "exact 18-field content-free startup event",
        ),
        _check(
            "closed_boundaries_remain_closed",
            all(value is False for value in closed.values()),
            closed,
            "all false",
        ),
        _check(
            "canonical_policy_case_states_exact",
            {name: row.get("state") for name, row in evaluations.items()} == expected_states,
            {name: row.get("state") for name, row in evaluations.items()},
            expected_states,
        ),
        _check(
            "eligible_case_has_no_rejections",
            evaluations.get("exact_local_entrypoint_contract", {}).get("rejection_reasons") == [],
            evaluations.get("exact_local_entrypoint_contract"),
            {"state": "eligible", "rejection_reasons": []},
        ),
        _check(
            "all_unsafe_cases_have_rejection_reasons",
            all(
                bool(row.get("rejection_reasons"))
                for name, row in evaluations.items()
                if name != "exact_local_entrypoint_contract"
            ),
            {
                name: len(row.get("rejection_reasons") or [])
                for name, row in evaluations.items()
                if name != "exact_local_entrypoint_contract"
            },
            "all greater than zero",
        ),
        _check(
            "policy_detects_queue_retry_and_fallback",
            all(
                reason
                in (
                    evaluations.get("hidden_recovery_and_custom_signal", {}).get(
                        "rejection_reasons"
                    )
                    or []
                )
                for reason in (
                    "queue_action_detected",
                    "retry_action_detected",
                    "fallback_action_detected",
                )
            ),
            evaluations.get("hidden_recovery_and_custom_signal"),
            "three hidden-recovery reasons present",
        ),
        _check(
            "public_report_contains_no_private_keys",
            _forbidden_keys_found(report) == set(),
            sorted(_forbidden_keys_found(report)),
            [],
        ),
    ]


def _decision(guards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [str(check.get("name")) for check in guards if check.get("passed") is not True]
    passed = not failed
    return {
        "status": _FINAL_STATUS if passed else "primeqa_hybrid_agent_service_protocol_rejected",
        "failed_checks": failed,
        "agent_service_entrypoint_protocol_frozen": passed,
        "local_service_entrypoint_implementation_allowed_next": passed,
        "bootstrap_warmup_type_refactor_required": passed,
        "service_entrypoint_implemented": False,
        "network_service_started": False,
        "network_port_bound": False,
        "remote_deployment_authorized": False,
        "runtime_registered_as_default": False,
        "runtime_defaultization_allowed_now": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
        "queue_actions_enabled": False,
        "retry_actions_enabled": False,
        "fallback_strategies_enabled": False,
        "custom_signal_handlers_enabled": False,
        "implicit_shutdown_timeout_enabled": False,
        "next_direction": _NEXT_DIRECTION if passed else "repair_failed_stage151_guards",
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_kind": "saved_public_stage150_aggregate_only",
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
        "service_entrypoint_implemented": False,
        "network_service_started": False,
        "network_port_bound": False,
        "signal_handlers_installed": False,
        "forbidden_keys_found": sorted(_forbidden_keys_found(report)),
    }


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


def _bar(label: str, value: Any) -> BarDatum:
    numeric = float(value) if isinstance(value, int | float | bool) else 0.0
    return BarDatum(label=label, value=numeric, value_label=str(value))


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
    width: int = 1900,
    margin_left: int = 980,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        width=width,
        margin_left=margin_left,
    )


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
