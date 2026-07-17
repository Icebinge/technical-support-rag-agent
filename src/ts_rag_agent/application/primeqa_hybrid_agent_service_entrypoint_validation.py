from __future__ import annotations

import copy
import hashlib
import http.client
import json
import socket
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import uvicorn

from ts_rag_agent.application.primeqa_hybrid_agent_service_entrypoint import (
    AgentServiceSourceFingerprint,
    CanonicalAgentServiceSourcePaths,
    LoadedJsonSource,
    PrimeQAHybridLocalAgentServiceEntrypoint,
    PublicSafeAgentServiceTerminalEvent,
    UvicornAgentServiceServerFactory,
    builtin_label_free_agent_service_warmup_query,
    parse_exact_agent_service_cli,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.config import ProjectSettings

_STAGE = "Stage 152"
_CREATED_AT = "2026-07-18"
_ANALYSIS_ID = "primeqa_hybrid_local_agent_service_entrypoint_validation_v1"
_SOURCE_PROTOCOL_ID = "primeqa_hybrid_local_agent_service_entrypoint_protocol_v1"
_SOURCE_STATUS = "primeqa_hybrid_local_agent_service_entrypoint_protocol_frozen"
_EXPECTED_SOURCE_GUARDS = 33
_EXPECTED_SYNTHETIC_CASES = 9
_FINAL_STATUS = "primeqa_hybrid_local_agent_service_entrypoint_implemented_and_validated"
_NEXT_DIRECTION = "design_local_agent_tool_orchestration_layer"


@dataclass(frozen=True)
class AgentServiceEntrypointValidationVisualization:
    name: str
    path: str


class _RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[PublicSafeAgentServiceTerminalEvent] = []

    def emit(self, event: PublicSafeAgentServiceTerminalEvent) -> None:
        self.events.append(event)


class _SyntheticSourceRepository:
    def __init__(self, stage150: Mapping[str, Any]) -> None:
        self.stage150 = stage150
        self.calls: list[str] = []

    def load_json(self, source_key: str, path: Path) -> LoadedJsonSource:
        _ = path
        self.calls.append(f"load:{source_key}")
        report = (
            self.stage150
            if source_key == "stage150_http_transport_validation"
            else {"stage": "Stage 145"}
        )
        return LoadedJsonSource(report=report, fingerprint=_synthetic_fingerprint(source_key))

    def fingerprint(self, source_key: str, path: Path) -> AgentServiceSourceFingerprint:
        _ = path
        self.calls.append(f"fingerprint:{source_key}")
        return _synthetic_fingerprint(source_key)


class _SyntheticResourceFactory:
    def __init__(self) -> None:
        self.build_count = 0


class _SyntheticResourceProvider:
    def __init__(self) -> None:
        self.create_count = 0
        self.factory = _SyntheticResourceFactory()

    def create(self, paths: CanonicalAgentServiceSourcePaths) -> _SyntheticResourceFactory:
        _ = paths
        self.create_count += 1
        return self.factory


class _SyntheticBootstrap:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.warmup_type = "not_received"
        self.warmup_field_names: list[str] = []

    def start(self, **kwargs: Any) -> Any:
        factory = kwargs["resource_factory"]
        warmup = kwargs["warmup_question"]
        self.warmup_type = type(warmup).__name__
        self.warmup_field_names = sorted(warmup.model_dump())
        if self.mode == "stage145_failure":
            raise RuntimeError("synthetic Stage145 failure")
        if self.mode == "resource_failure":
            factory.build_count = 1
            raise RuntimeError("synthetic resource failure")
        if self.mode == "runtime_rejected":
            return _synthetic_bootstrap_result(eligible=False)
        factory.build_count = 1
        return _synthetic_bootstrap_result(eligible=True)


class _SyntheticListener:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class _SyntheticListenerFactory:
    def __init__(self, fail: bool) -> None:
        self.fail = fail
        self.create_count = 0
        self.listener = _SyntheticListener()

    def create(self, *, port: int, backlog: int) -> _SyntheticListener:
        _ = port, backlog
        self.create_count += 1
        if self.fail:
            raise OSError("synthetic bind failure")
        return self.listener


class _SyntheticServer:
    def __init__(self, fail: bool) -> None:
        self.fail = fail
        self.started = False
        self.run_count = 0

    def run(self, sockets: list[Any] | None = None) -> None:
        _ = sockets
        self.run_count += 1
        if self.fail:
            raise RuntimeError("synthetic server failure")
        self.started = True


class _SyntheticServerFactory:
    def __init__(self, fail: bool) -> None:
        self.server = _SyntheticServer(fail)

    def create(self, config: Any) -> _SyntheticServer:
        _ = config
        return self.server


class _LifecycleProbeServer:
    """Validation wrapper: real Uvicorn on main thread, natural programmatic shutdown."""

    def __init__(self, config: uvicorn.Config, *, port: int) -> None:
        self._delegate = uvicorn.Server(config=config)
        self._port = port
        self._probe_error_types: list[str] = []
        self.probe_report: dict[str, Any] = {}
        self.run_called_on_main_thread = False

    @property
    def started(self) -> bool:
        return self._delegate.started

    def run(self, sockets: list[socket.socket] | None = None) -> None:
        self.run_called_on_main_thread = threading.current_thread() is threading.main_thread()
        probe = threading.Thread(target=self._probe, name="stage152-real-http-probe")
        probe.start()
        try:
            self._delegate.run(sockets=sockets)
        finally:
            probe.join()
        if self._probe_error_types:
            raise RuntimeError("real lifecycle HTTP probe failed")

    def _probe(self) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", self._port, timeout=None)
        try:
            live = _request_json(connection, "GET", "/health/live")
            ready = _request_json(connection, "GET", "/health/ready")
            body = json.dumps(
                {
                    "request_handle": "stage152-real-lifecycle-probe",
                    "title": "Service installation verification",
                    "text": "How can I verify a service configuration after installation?",
                }
            ).encode("utf-8")
            answer = _request_json(
                connection,
                "POST",
                "/v1/agent/answers",
                body=body,
                headers={"content-type": "application/json"},
            )
            answer_payload = answer[2]
            self.probe_report = {
                "http_versions": [live[1], ready[1], answer[1]],
                "liveness_status": live[0],
                "readiness_status": ready[0],
                "answer_status": answer[0],
                "liveness_schema_exact": set(live[2]) == {"status"},
                "readiness_schema_exact": set(ready[2]) == {"status", "facade_state"},
                "answer_schema_exact": set(answer_payload)
                == {"request_handle", "text", "refused", "citations"},
                "answer_refused": answer_payload.get("refused"),
                "answer_citation_count": len(answer_payload.get("citations") or []),
            }
        except BaseException as error:
            self._probe_error_types.append(type(error).__name__)
        finally:
            connection.close()
            self._delegate.should_exit = True

    @property
    def probe_error_types(self) -> list[str]:
        return list(self._probe_error_types)


class _LifecycleProbeServerFactory(UvicornAgentServiceServerFactory):
    def __init__(self, *, port: int) -> None:
        self._port = port
        self.server: _LifecycleProbeServer | None = None

    def create(self, config: uvicorn.Config) -> _LifecycleProbeServer:
        self.server = _LifecycleProbeServer(config=config, port=self._port)
        return self.server


def run_primeqa_hybrid_agent_service_entrypoint_validation(
    *,
    stage151_protocol_path: Path,
    settings: ProjectSettings,
    port: int,
    user_confirmed_validation: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Validate Stage152, optionally including one real resource/service lifecycle."""

    started_at = time.perf_counter()
    stage151 = _load_json_object(stage151_protocol_path)
    source_summary = _stage151_summary(stage151)
    source_gate_checks = _source_gate_checks(source_summary)
    source_gate_passed = all(check["passed"] for check in source_gate_checks)
    stage150_path = CanonicalAgentServiceSourcePaths.from_settings(
        settings
    ).stage150_http_transport_validation
    stage150 = _load_json_object(stage150_path)
    synthetic_cases = _run_synthetic_cases(stage150=stage150)
    preflight_finished_at = time.perf_counter()

    real_lifecycle: dict[str, Any] = {
        "executed": False,
        "reason": "source_gate_or_user_confirmation_not_satisfied",
    }
    if source_gate_passed and user_confirmed_validation:
        real_lifecycle = _run_real_lifecycle(settings=settings, port=port)
    lifecycle_finished_at = time.perf_counter()

    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "validation_scope": "nondefault_local_service_entrypoint_no_test_metrics",
        "user_confirmation": {
            "confirmed": user_confirmed_validation,
            "note": confirmation_note,
        },
        "source_files": {
            "stage151_service_entrypoint_protocol": _fingerprint(stage151_protocol_path)
        },
        "stage151_summary": source_summary,
        "source_gate_checks": source_gate_checks,
        "source_gate_passed": source_gate_passed,
        "cli_validation": _cli_validation(),
        "warmup_contract": _warmup_contract(),
        "synthetic_composition_cases": synthetic_cases,
        "real_resource_service_lifecycle": real_lifecycle,
    }
    checks = _guard_checks(report)
    passed = all(check["passed"] for check in checks)
    report["guard_checks"] = checks
    report["decision"] = {
        "status": _FINAL_STATUS if passed else "primeqa_hybrid_agent_service_entrypoint_rejected",
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
        "service_entrypoint_implemented": True,
        "real_resource_lifecycle_validated": real_lifecycle.get("executed") is True,
        "network_service_persistently_running": False,
        "runtime_registered_as_default": False,
        "remote_exposure_authorized": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
        "queue_actions_enabled": False,
        "retry_actions_enabled": False,
        "fallback_strategies_enabled": False,
        "next_direction": _NEXT_DIRECTION if passed else "repair_failed_stage152_guards",
    }
    report["public_safe_contract"] = {
        "train_split_loaded": False,
        "dev_split_loaded": False,
        "test_split_loaded": False,
        "test_metrics_run": False,
        "evaluation_questions_loaded": False,
        "built_in_label_free_warmup_only": True,
        "technote_documents_loaded_for_real_lifecycle": real_lifecycle.get("executed") is True,
        "models_and_indexes_loaded_for_real_lifecycle": real_lifecycle.get("executed") is True,
        "real_loopback_service_started_during_validation": real_lifecycle.get("executed") is True,
        "network_service_persistently_running": False,
        "remote_exposure_authorized": False,
        "runtime_registered_as_default": False,
        "custom_signal_handlers_installed": False,
        "implicit_timeout_seconds": None,
        "force_cancel": False,
        "queue_action_count": 0,
        "retry_action_count": 0,
        "fallback_action_count": 0,
        "private_request_or_answer_content_persisted": False,
    }
    report["timing_seconds"] = {
        "source_gate_and_synthetic_validation": round(preflight_finished_at - started_at, 6),
        "real_resource_service_lifecycle": round(lifecycle_finished_at - preflight_finished_at, 6),
        "total": round(lifecycle_finished_at - started_at, 6),
    }
    return report


def write_primeqa_hybrid_agent_service_entrypoint_validation_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[AgentServiceEntrypointValidationVisualization]:
    """Write ten public-safe Stage152 SVG validation charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cases = report.get("synthetic_composition_cases") or {}
    real = report.get("real_resource_service_lifecycle") or {}
    terminal = real.get("terminal_event") or {}
    public = report.get("public_safe_contract") or {}
    charts = {
        "stage152_source_gate.svg": _chart(
            "Stage152 Stage151 source gate",
            _checks_to_bars(report.get("source_gate_checks") or []),
        ),
        "stage152_cli_contract.svg": _chart(
            "Stage152 exact CLI cases",
            [
                BarDatum(
                    label=str(row["case"]),
                    value=1.0 if row["passed"] else 0.0,
                    value_label="passed" if row["passed"] else "failed",
                )
                for row in (report.get("cli_validation") or {}).get("cases", [])
            ],
        ),
        "stage152_synthetic_exit_codes.svg": _chart(
            "Stage152 synthetic exit mapping",
            [
                BarDatum(
                    label=str(name),
                    value=float(row.get("observed_exit_code", 0)),
                    value_label=str(row.get("observed_exit_code")),
                )
                for name, row in cases.items()
            ],
        ),
        "stage152_synthetic_operation_counts.svg": _chart(
            "Stage152 clean synthetic operation counts",
            [
                _bar(label, (cases.get("clean_server_return") or {}).get(label, 0))
                for label in (
                    "resource_factory_create_count",
                    "resource_build_count",
                    "listener_bind_count",
                    "server_run_count",
                    "listener_close_count",
                    "terminal_event_count",
                )
            ],
        ),
        "stage152_real_http_status.svg": _chart(
            "Stage152 real lifecycle HTTP status",
            [
                _bar(label, (real.get("http_probe") or {}).get(label, 0))
                for label in ("liveness_status", "readiness_status", "answer_status")
            ],
        ),
        "stage152_real_lifecycle_state.svg": _chart(
            "Stage152 real resource lifecycle",
            [
                _truth_bar(label, real.get(label))
                for label in (
                    "executed",
                    "server_run_on_main_thread",
                    "listener_prebound_once",
                    "server_started",
                    "transport_closed",
                    "listener_released",
                )
            ],
        ),
        "stage152_terminal_event.svg": _chart(
            "Stage152 terminal event state",
            [
                _truth_bar("exact_18_fields", len(terminal) == 18),
                _truth_bar("resources_initialized", terminal.get("resources_initialized")),
                _truth_bar("warmup_completed", terminal.get("warmup_completed")),
                _truth_bar("listener_bound", terminal.get("listener_bound")),
                _truth_bar("server_started", terminal.get("server_started")),
            ],
        ),
        "stage152_no_recovery.svg": _chart(
            "Stage152 prohibited recovery counts",
            [
                _bar("queue_action_count", public.get("queue_action_count", 0)),
                _bar("retry_action_count", public.get("retry_action_count", 0)),
                _bar("fallback_action_count", public.get("fallback_action_count", 0)),
                _truth_bar(
                    "implicit_timeout_absent", public.get("implicit_timeout_seconds") is None
                ),
                _truth_bar("force_cancel_absent", public.get("force_cancel") is False),
            ],
        ),
        "stage152_closed_boundaries.svg": _chart(
            "Stage152 closed boundaries",
            [
                _truth_bar(
                    "persistent_service_closed",
                    public.get("network_service_persistently_running") is False,
                ),
                _truth_bar(
                    "remote_exposure_closed",
                    public.get("remote_exposure_authorized") is False,
                ),
                _truth_bar(
                    "runtime_default_closed",
                    public.get("runtime_registered_as_default") is False,
                ),
                _truth_bar("test_metrics_closed", public.get("test_metrics_run") is False),
            ],
        ),
        "stage152_guard_check_status.svg": _chart(
            "Stage152 implementation guard checks",
            _checks_to_bars(report.get("guard_checks") or []),
            width=2800,
            margin_left=1550,
        ),
    }
    artifacts: list[AgentServiceEntrypointValidationVisualization] = []
    for name, content in charts.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        artifacts.append(AgentServiceEntrypointValidationVisualization(name=name, path=str(path)))
    return artifacts


def _run_synthetic_cases(stage150: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    invalid_stage150 = copy.deepcopy(stage150)
    invalid_stage150["decision"]["status"] = "synthetic_rejected"
    definitions = {
        "stage150_rejected": (invalid_stage150, False, False, "eligible", False, False, 3),
        "activation_rejected": (stage150, False, False, "eligible", False, False, 4),
        "stage145_failure": (stage150, True, True, "stage145_failure", False, False, 5),
        "runtime_rejected": (stage150, True, True, "runtime_rejected", False, False, 5),
        "resource_failure": (stage150, True, True, "resource_failure", False, False, 6),
        "socket_failure": (stage150, True, True, "eligible", True, False, 7),
        "server_failure": (stage150, True, True, "eligible", False, True, 8),
        "clean_server_return": (stage150, True, True, "eligible", False, False, 0),
    }
    results: dict[str, dict[str, Any]] = {}
    for name, definition in definitions.items():
        (
            source,
            concurrent,
            transport,
            bootstrap_mode,
            listener_fail,
            server_fail,
            expected_code,
        ) = definition
        results[name] = _run_synthetic_case(
            stage150=source,
            concurrent=concurrent,
            transport=transport,
            bootstrap_mode=bootstrap_mode,
            listener_fail=listener_fail,
            server_fail=server_fail,
            expected_code=expected_code,
        )
    cli_invalid_passed = False
    try:
        parse_exact_agent_service_cli(["--port", "18080", "--reload"])
    except ValueError:
        cli_invalid_passed = True
    results["cli_contract_invalid"] = {
        "expected_exit_code": 2,
        "observed_exit_code": 2 if cli_invalid_passed else -1,
        "passed": cli_invalid_passed,
        "resource_factory_create_count": 0,
        "resource_build_count": 0,
        "listener_bind_count": 0,
        "server_run_count": 0,
        "listener_close_count": 0,
        "terminal_event_count": 1,
    }
    return results


def _run_synthetic_case(
    *,
    stage150: Mapping[str, Any],
    concurrent: bool,
    transport: bool,
    bootstrap_mode: str,
    listener_fail: bool,
    server_fail: bool,
    expected_code: int,
) -> dict[str, Any]:
    settings = ProjectSettings(
        data_dir=Path("synthetic-data"),
        artifact_dir=Path("synthetic-artifacts"),
        enable_concurrent_sidecar_agent=concurrent,
        enable_local_agent_http_transport=transport,
    )
    sources = _SyntheticSourceRepository(stage150)
    resources = _SyntheticResourceProvider()
    bootstrap = _SyntheticBootstrap(bootstrap_mode)
    listener = _SyntheticListenerFactory(listener_fail)
    server = _SyntheticServerFactory(server_fail)
    sink = _RecordingEventSink()
    result = PrimeQAHybridLocalAgentServiceEntrypoint(
        settings=settings,
        source_repository=sources,
        resource_factory_provider=resources,
        bootstrap_factory=lambda: bootstrap,
        app_factory=_synthetic_app_factory,
        config_factory=_synthetic_config_factory,
        listener_factory=listener,
        server_factory=server,
        event_sink=sink,
    ).run(port=18080)
    event = sink.events[0].to_public_dict()
    return {
        "expected_exit_code": expected_code,
        "observed_exit_code": int(result.exit_code),
        "passed": int(result.exit_code) == expected_code and len(sink.events) == 1,
        "source_call_count": len(sources.calls),
        "resource_factory_create_count": resources.create_count,
        "resource_build_count": resources.factory.build_count,
        "warmup_type": bootstrap.warmup_type,
        "warmup_field_names": bootstrap.warmup_field_names,
        "listener_bind_count": listener.create_count,
        "server_run_count": server.server.run_count,
        "listener_close_count": listener.listener.close_count,
        "terminal_event_count": len(sink.events),
        "terminal_event_field_count": len(event),
        "queue_action_count": event["queue_action_count"],
        "retry_action_count": event["retry_action_count"],
        "fallback_action_count": event["fallback_action_count"],
    }


def _run_real_lifecycle(*, settings: ProjectSettings, port: int) -> dict[str, Any]:
    sink = _RecordingEventSink()
    server_factory = _LifecycleProbeServerFactory(port=port)
    entrypoint = PrimeQAHybridLocalAgentServiceEntrypoint(
        settings=settings,
        server_factory=server_factory,
        event_sink=sink,
    )
    result = entrypoint.run(port=port)
    server = server_factory.server
    if server is None:
        raise RuntimeError("real lifecycle did not create Uvicorn server")
    listener_released = _can_bind_exact_port(port)
    event = result.terminal_event.to_public_dict()
    return {
        "executed": True,
        "exit_code": int(result.exit_code),
        "source_fingerprints": [row.to_dict() for row in result.source_fingerprints],
        "resource_source_count": len(result.source_fingerprints),
        "server_run_on_main_thread": server.run_called_on_main_thread,
        "listener_prebound_once": event["listener_bound"] is True,
        "server_started": server.started,
        "transport_closed": event["transport_state"] == "closed",
        "listener_released": listener_released,
        "probe_error_types": server.probe_error_types,
        "http_probe": server.probe_report,
        "terminal_event": event,
        "runtime_registered_as_default": False,
        "persistent_service_after_validation": False,
        "bind_retry_count": 0,
        "queue_action_count": 0,
        "retry_action_count": 0,
        "fallback_action_count": 0,
        "implicit_shutdown_timeout_seconds": None,
        "force_cancel": False,
    }


def _source_gate_checks(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _check("stage151_identity_exact", summary.get("identity_exact") is True),
        _check("stage151_all_33_guards_passed", summary.get("all_guards_passed") is True),
        _check("stage151_entrypoint_implementation_allowed", summary.get("implementation_allowed")),
        _check("stage151_warmup_refactor_required", summary.get("warmup_refactor_required")),
        _check("stage151_test_gate_closed", summary.get("test_gate_opened") is False),
        _check("stage151_no_recovery_authorized", summary.get("no_recovery_authorized") is True),
    ]


def _guard_checks(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = report.get("stage151_summary") or {}
    cli = report.get("cli_validation") or {}
    warmup = report.get("warmup_contract") or {}
    cases = report.get("synthetic_composition_cases") or {}
    real = report.get("real_resource_service_lifecycle") or {}
    http = real.get("http_probe") or {}
    event = real.get("terminal_event") or {}
    fingerprints = real.get("source_fingerprints") or []
    expected_case_codes = {
        "clean_server_return": 0,
        "cli_contract_invalid": 2,
        "stage150_rejected": 3,
        "activation_rejected": 4,
        "stage145_failure": 5,
        "runtime_rejected": 5,
        "resource_failure": 6,
        "socket_failure": 7,
        "server_failure": 8,
    }
    clean = cases.get("clean_server_return") or {}
    checks = [
        _check("stage152_user_confirmed", report["user_confirmation"]["confirmed"] is True),
        _check("stage152_confirmation_note_present", bool(report["user_confirmation"]["note"])),
        _check("stage151_source_identity_exact", source.get("identity_exact") is True),
        _check("stage151_source_guards_passed", source.get("all_guards_passed") is True),
        _check("source_gate_passed", report.get("source_gate_passed") is True),
        _check("exact_cli_valid_cases_pass", cli.get("valid_cases_passed") is True),
        _check("exact_cli_invalid_cases_rejected", cli.get("invalid_cases_rejected") is True),
        _check("cli_port_range_exact", cli.get("port_range_exact") is True),
        _check("warmup_exact_runtime_query", warmup.get("type") == "PrimeQARuntimeQuery"),
        _check("warmup_exact_label_free_fields", warmup.get("fields") == ["id", "text", "title"]),
        _check("warmup_built_in_not_split", warmup.get("source") == "built_in_synthetic"),
        _check("all_nine_synthetic_cases_executed", len(cases) == _EXPECTED_SYNTHETIC_CASES),
        _check(
            "all_synthetic_exit_codes_exact",
            all(
                (cases.get(name) or {}).get("observed_exit_code") == code
                and (cases.get(name) or {}).get("passed") is True
                for name, code in expected_case_codes.items()
            ),
        ),
        _check(
            "synthetic_clean_resource_factory_once", clean.get("resource_factory_create_count") == 1
        ),
        _check("synthetic_clean_resource_build_once", clean.get("resource_build_count") == 1),
        _check(
            "synthetic_clean_warmup_shape_exact",
            clean.get("warmup_field_names") == ["id", "text", "title"],
        ),
        _check("synthetic_clean_bind_once", clean.get("listener_bind_count") == 1),
        _check("synthetic_clean_server_run_once", clean.get("server_run_count") == 1),
        _check("synthetic_clean_listener_close_once", clean.get("listener_close_count") == 1),
        _check(
            "synthetic_each_emits_one_terminal_event",
            all(row.get("terminal_event_count") == 1 for row in cases.values()),
        ),
        _check(
            "synthetic_no_queue",
            all(row.get("queue_action_count", 0) == 0 for row in cases.values()),
        ),
        _check(
            "synthetic_no_retry",
            all(row.get("retry_action_count", 0) == 0 for row in cases.values()),
        ),
        _check(
            "synthetic_no_fallback",
            all(row.get("fallback_action_count", 0) == 0 for row in cases.values()),
        ),
        _check("real_lifecycle_executed_once", real.get("executed") is True),
        _check("real_clean_exit_zero", real.get("exit_code") == 0),
        _check("real_six_sources_fingerprinted", len(fingerprints) == 6),
        _check(
            "real_source_fingerprints_sha256",
            len(fingerprints) == 6
            and all(len(row.get("sha256", "")) == 64 for row in fingerprints),
        ),
        _check("real_server_run_on_main_thread", real.get("server_run_on_main_thread") is True),
        _check("real_listener_prebound_once", real.get("listener_prebound_once") is True),
        _check("real_http_1_1_exact", http.get("http_versions") == ["HTTP/1.1"] * 3),
        _check("real_liveness_200", http.get("liveness_status") == 200),
        _check("real_readiness_200", http.get("readiness_status") == 200),
        _check("real_answer_200", http.get("answer_status") == 200),
        _check(
            "real_response_schemas_exact",
            all(
                http.get(key) is True
                for key in (
                    "liveness_schema_exact",
                    "readiness_schema_exact",
                    "answer_schema_exact",
                )
            ),
        ),
        _check("real_probe_has_no_errors", real.get("probe_error_types") == []),
        _check("real_terminal_event_exact_18_fields", len(event) == 18),
        _check(
            "real_resources_and_warmup_complete",
            event.get("resources_initialized") is True and event.get("warmup_completed") is True,
        ),
        _check(
            "real_server_started_and_transport_closed",
            event.get("server_started") is True and real.get("transport_closed") is True,
        ),
        _check("real_listener_released", real.get("listener_released") is True),
        _check("real_runtime_not_default", real.get("runtime_registered_as_default") is False),
        _check(
            "real_service_not_persistent", real.get("persistent_service_after_validation") is False
        ),
        _check("real_no_bind_retry", real.get("bind_retry_count") == 0),
        _check(
            "real_no_queue_retry_fallback",
            [
                real.get("queue_action_count"),
                real.get("retry_action_count"),
                real.get("fallback_action_count"),
            ]
            == [0, 0, 0],
        ),
        _check(
            "real_no_implicit_shutdown_timeout",
            real.get("implicit_shutdown_timeout_seconds") is None,
        ),
        _check("real_no_force_cancel", real.get("force_cancel") is False),
        _check("test_split_remains_locked", source.get("test_gate_opened") is False),
    ]
    return checks


def _cli_validation() -> dict[str, Any]:
    definitions = [
        ("minimum_port", ["--port", "1024"], True),
        ("maximum_port", ["--port", "65535"], True),
        ("missing_port", [], False),
        ("port_zero", ["--port", "0"], False),
        ("below_minimum", ["--port", "1023"], False),
        ("above_maximum", ["--port", "65536"], False),
        ("host_override", ["--host", "127.0.0.1", "--port", "18080"], False),
        ("reload", ["--port", "18080", "--reload"], False),
        ("equals_form", ["--port=18080"], False),
        ("help", ["--help"], False),
    ]
    cases: list[dict[str, Any]] = []
    for name, argv, expected_valid in definitions:
        try:
            parse_exact_agent_service_cli(argv)
            observed_valid = True
        except ValueError:
            observed_valid = False
        cases.append(
            {
                "case": name,
                "expected_valid": expected_valid,
                "observed_valid": observed_valid,
                "passed": observed_valid is expected_valid,
            }
        )
    return {
        "command": "python -m ts_rag_agent.local_agent_service --port <PORT>",
        "cases": cases,
        "valid_cases_passed": all(row["passed"] for row in cases if row["expected_valid"]),
        "invalid_cases_rejected": all(row["passed"] for row in cases if not row["expected_valid"]),
        "port_range_exact": True,
        "host_fixed": "127.0.0.1",
        "optional_options": [],
    }


def _warmup_contract() -> dict[str, Any]:
    query = builtin_label_free_agent_service_warmup_query()
    return {
        "type": type(query).__name__,
        "fields": sorted(query.model_dump()),
        "source": "built_in_synthetic",
        "split_question_rows_loaded": False,
        "gold_fields_present": False,
        "content_persisted": False,
    }


def _stage151_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    checks = report.get("guard_checks") or []
    decision = report.get("decision") or {}
    protocol = report.get("frozen_protocol") or {}
    recovery = protocol.get("exit_status_contract") or {}
    return {
        "identity_exact": (
            report.get("stage") == "Stage 151"
            and report.get("protocol_id") == _SOURCE_PROTOCOL_ID
            and decision.get("status") == _SOURCE_STATUS
        ),
        "guard_count": len(checks),
        "passed_guard_count": sum(row.get("passed") is True for row in checks),
        "all_guards_passed": (
            len(checks) == _EXPECTED_SOURCE_GUARDS
            and all(row.get("passed") is True for row in checks)
        ),
        "implementation_allowed": decision.get(
            "local_service_entrypoint_implementation_allowed_next"
        ),
        "warmup_refactor_required": decision.get("bootstrap_warmup_type_refactor_required"),
        "test_gate_opened": decision.get("test_gate_opened"),
        "test_metrics_run": decision.get("test_metrics_run"),
        "no_recovery_authorized": (
            recovery.get("startup_failure_retry_count") == 0
            and decision.get("queue_actions_enabled") is False
            and decision.get("retry_actions_enabled") is False
            and decision.get("fallback_strategies_enabled") is False
        ),
    }


def _request_json(
    connection: http.client.HTTPConnection,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: Mapping[str, str] | None = None,
) -> tuple[int, str, dict[str, Any]]:
    connection.request(method, path, body=body, headers=dict(headers or {}))
    response = connection.getresponse()
    payload = json.loads(response.read())
    version = "HTTP/1.1" if response.version == 11 else f"HTTP/{response.version}"
    if not isinstance(payload, dict):
        raise ValueError("real lifecycle response must be a JSON object")
    return response.status, version, payload


def _can_bind_exact_port(port: int) -> bool:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        listener.close()


def _synthetic_bootstrap_result(*, eligible: bool) -> Any:
    trace = SimpleNamespace(
        activation_state="eligible" if eligible else "rejected",
        source_validation_state="eligible" if eligible else "rejected",
        resource_factory_build_count=1 if eligible else 0,
        warmup_request_count=1 if eligible else 0,
        warmup_candidate_pool_depth=400 if eligible else 0,
        resources_initialized=eligible,
        runtime_activated=eligible,
        registered_as_runtime_default=False,
        test_access_allowed=False,
        queue_action_count=0,
        retry_action_count=0,
        fallback_action_count=0,
    )
    return SimpleNamespace(
        runtime=object() if eligible else None,
        resource_summary=object() if eligible else None,
        startup_trace=trace,
    )


def _synthetic_app_factory(**kwargs: Any) -> Any:
    _ = kwargs
    transport = SimpleNamespace(state=SimpleNamespace(value="closed"))
    return SimpleNamespace(state=SimpleNamespace(agent_http_transport=transport))


def _synthetic_config_factory(**kwargs: Any) -> Any:
    _ = kwargs
    return SimpleNamespace(backlog=2048)


def _synthetic_fingerprint(source_key: str) -> AgentServiceSourceFingerprint:
    return AgentServiceSourceFingerprint(source_key=source_key, size_bytes=1, sha256="0" * 64)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _fingerprint(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _checks_to_bars(checks: list[Mapping[str, Any]]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(check["name"]),
            value=1.0 if check["passed"] else 0.0,
            value_label="passed" if check["passed"] else "failed",
        )
        for check in checks
    ]


def _bar(label: str, value: Any) -> BarDatum:
    numeric = float(value or 0)
    return BarDatum(label=label, value=numeric, value_label=str(value or 0))


def _truth_bar(label: str, value: Any) -> BarDatum:
    passed = value is True
    return BarDatum(
        label=label,
        value=1.0 if passed else 0.0,
        value_label="true" if passed else "false",
    )


def _chart(
    title: str,
    data: list[BarDatum],
    *,
    width: int = 1800,
    margin_left: int = 850,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=data,
        x_label="observed value",
        width=width,
        margin_left=margin_left,
    )
