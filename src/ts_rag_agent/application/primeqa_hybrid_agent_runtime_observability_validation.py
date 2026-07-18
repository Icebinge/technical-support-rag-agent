from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi.testclient import TestClient

from ts_rag_agent.application.primeqa_hybrid_agent_http_transport import (
    create_primeqa_hybrid_agent_http_app,
)
from ts_rag_agent.application.primeqa_hybrid_agent_runtime_observability import (
    PublicSafeAgentWorkflowObservationEvent,
    agent_runtime_activation_observability_contract,
)
from ts_rag_agent.application.primeqa_hybrid_agent_service_entrypoint import (
    AgentServiceSourceFingerprint,
    _stage154_authorized,
)
from ts_rag_agent.application.primeqa_hybrid_agent_service_entrypoint_validation import (
    _run_real_lifecycle,
)
from ts_rag_agent.application.primeqa_hybrid_agent_tool_workflow import (
    create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset,
)
from ts_rag_agent.application.primeqa_hybrid_agent_tool_workflow_validation import (
    _active_startup_trace,
    _candidate_results,
    _FourRequestRetriever,
    _question,
    _resource_summary,
    _StaticRetriever,
    _toolset,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_activation import (
    PrimeQAHybridConcurrentRuntimeBootstrapResult,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_sidecar_agent_runtime import (
    create_primeqa_hybrid_concurrent_sidecar_agent_runtime,
)
from ts_rag_agent.application.primeqa_hybrid_online_candidate_pool_retriever import (
    CandidatePoolRetrievalConfig,
    IndependentCandidatePoolSearchChannel,
    PrimeQAHybridOnlineCandidatePoolRetriever,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    PrimeQAHybridSharedRuntimeResources,
    _forbidden_keys_found,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.config import ProjectSettings

_STAGE = "Stage 155"
_CREATED_AT = "2026-07-18"
_ANALYSIS_ID = "primeqa_hybrid_agent_runtime_activation_observability_validation_v1"
_FINAL_STATUS = "primeqa_hybrid_agent_runtime_activation_observability_validated"
_SOURCE_STAGE154_STATUS = "primeqa_hybrid_langgraph_agent_tool_workflow_implemented_and_validated"
_EXPECTED_STAGE154_GUARDS = 54
_EXPECTED_NODE_IDS = (
    "validate_request",
    "retrieve_candidate_pool",
    "prepare_context",
    "compose_grounded_answer",
    "verify_grounded_answer",
    "observe_diagnostics",
    "finalize_response",
)


@dataclass(frozen=True)
class AgentRuntimeObservabilityVisualization:
    name: str
    path: str


class RecordingAgentWorkflowObservationSink:
    """Thread-safe validation sink retaining only allowlisted events."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.events: list[PublicSafeAgentWorkflowObservationEvent] = []

    def emit(self, event: PublicSafeAgentWorkflowObservationEvent) -> None:
        event.to_public_dict()
        with self._lock:
            self.events.append(event)


RealLifecycleRunner = Callable[..., dict[str, Any]]


def run_primeqa_hybrid_agent_runtime_observability_validation(
    *,
    stage154_validation_path: Path,
    stage153_protocol_path: Path,
    pyproject_path: Path,
    workflow_source_path: Path,
    concurrent_runtime_source_path: Path,
    observability_source_path: Path,
    service_entrypoint_source_path: Path,
    settings: ProjectSettings,
    port: int,
    user_confirmed_validation: bool,
    confirmation_note: str,
    real_lifecycle_runner: RealLifecycleRunner = _run_real_lifecycle,
) -> dict[str, Any]:
    """Validate strict graph activation and content-free runtime observation."""

    started_at = time.perf_counter()
    source_paths = {
        "stage154_validation": stage154_validation_path,
        "stage153_protocol": stage153_protocol_path,
        "pyproject": pyproject_path,
        "workflow_source": workflow_source_path,
        "concurrent_runtime_source": concurrent_runtime_source_path,
        "observability_source": observability_source_path,
        "service_entrypoint_source": service_entrypoint_source_path,
    }
    source_before = {name: _fingerprint(path) for name, path in source_paths.items()}
    stage154 = _load_json_object(stage154_validation_path)
    stage154_summary = _stage154_summary(
        report=stage154,
        source_paths={
            "stage153_protocol": stage153_protocol_path,
            "pyproject": pyproject_path,
            "workflow_source": workflow_source_path,
            "concurrent_runtime_source": concurrent_runtime_source_path,
        },
    )
    loaded_at = time.perf_counter()

    complete = _workflow_case(refuse=False)
    refuse = _workflow_case(refuse=True)
    failure = _failure_case()
    concurrency = _concurrency_case()
    http = _http_case()
    activation = _activation_case(
        report=stage154,
        source_paths={
            "stage153_protocol": stage153_protocol_path,
            "pyproject": pyproject_path,
            "workflow_source": workflow_source_path,
            "concurrent_runtime_source": concurrent_runtime_source_path,
        },
    )
    synthetic_at = time.perf_counter()

    real_sink = RecordingAgentWorkflowObservationSink()
    if user_confirmed_validation:
        real_lifecycle = real_lifecycle_runner(
            settings=settings,
            port=port,
            workflow_observation_sink=real_sink,
        )
        real_observation = _observation_summary(real_sink.events)
    else:
        real_lifecycle = {
            "executed": False,
            "reason": "explicit_user_confirmation_required",
            "test_metrics_run": False,
        }
        real_observation = _observation_summary(())
    real_at = time.perf_counter()

    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "validation_scope": (
            "strict_nondefault_activation_and_public_safe_operational_observability"
        ),
        "user_confirmation": {
            "confirmed": user_confirmed_validation,
            "note": confirmation_note,
        },
        "source_files": source_before,
        "source_unchanged_after_validation": False,
        "stage154_summary": stage154_summary,
        "activation_observability_contract": (agent_runtime_activation_observability_contract()),
        "activation_validation": activation,
        "synthetic_validation": {
            "complete": complete,
            "refuse": refuse,
            "failure": failure,
            "concurrency_four": concurrency,
            "http": http,
        },
        "real_resource_service_lifecycle": real_lifecycle,
        "real_observation": real_observation,
        "guard_checks": [],
        "decision": {},
        "timing_seconds": {},
        "public_safe_contract": {
            "test_split_loaded": False,
            "test_metrics_run": False,
            "train_or_dev_rows_loaded": False,
            "request_content_saved": False,
            "request_identifiers_saved": False,
            "document_content_saved": False,
            "document_identifiers_saved": False,
            "wall_clock_timestamps_saved": False,
            "runtime_registered_as_default": False,
            "remote_exposure_authorized": False,
            "queue_action_count": 0,
            "retry_action_count": 0,
            "fallback_action_count": 0,
            "forbidden_keys_found": [],
        },
    }
    report["source_unchanged_after_validation"] = all(
        _fingerprint(path) == source_before[name] for name, path in source_paths.items()
    )
    checks = _guard_checks(report)
    failed = [row["name"] for row in checks if not row["passed"]]
    report["guard_checks"] = checks
    report["decision"] = {
        "status": _FINAL_STATUS if not failed else "stage155_validation_blocked",
        "failed_checks": failed,
        "activation_protocol_frozen": not failed,
        "stage154_current_evidence_required": True,
        "node_observability_implemented": True,
        "real_resource_service_lifecycle_validated": not failed,
        "runtime_registered_as_default": False,
        "remote_exposure_authorized": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
        "queue_actions_enabled": False,
        "retry_actions_enabled": False,
        "fallback_strategies_enabled": False,
        "next_direction": "design_local_agent_tool_selection_and_multi_turn_state_boundary",
    }
    finished_at = time.perf_counter()
    report["timing_seconds"] = {
        "load_sources": round(loaded_at - started_at, 6),
        "synthetic_validation": round(synthetic_at - loaded_at, 6),
        "real_lifecycle": round(real_at - synthetic_at, 6),
        "guard_and_report": round(finished_at - real_at, 6),
        "total": round(finished_at - started_at, 6),
    }
    forbidden = sorted(_forbidden_keys_found(report))
    report["public_safe_contract"]["forbidden_keys_found"] = forbidden
    if forbidden:
        raise ValueError(f"Stage155 report contains forbidden keys: {forbidden}")
    return report


def write_primeqa_hybrid_agent_runtime_observability_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[AgentRuntimeObservabilityVisualization]:
    """Write ten public-safe Stage155 SVG charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic = report.get("synthetic_validation") or {}
    complete = synthetic.get("complete") or {}
    refuse = synthetic.get("refuse") or {}
    failure = synthetic.get("failure") or {}
    concurrency = synthetic.get("concurrency_four") or {}
    http = synthetic.get("http") or {}
    real = report.get("real_observation") or {}
    contract = report.get("activation_observability_contract") or {}
    charts = {
        "stage155_activation_chain.svg": _chart(
            "Stage155 strict activation chain",
            [
                _truth_bar(
                    "dual explicit flags",
                    len(contract.get("required_explicit_environment_flags") or []) == 2,
                ),
                _truth_bar(
                    "Stage154 formal required", contract.get("stage154_formal_evidence_required")
                ),
                _truth_bar(
                    "current fingerprints required",
                    contract.get("stage154_current_source_fingerprints_required"),
                ),
                _truth_bar("loopback only", contract.get("binding_host") == "127.0.0.1"),
            ],
        ),
        "stage155_complete_node_latency.svg": _latency_chart(
            "Stage155 complete-path node latency", complete
        ),
        "stage155_refuse_node_latency.svg": _latency_chart(
            "Stage155 refuse-path node latency", refuse
        ),
        "stage155_failure_timeline.svg": _chart(
            "Stage155 failure timeline",
            [
                _bar("emitted events", failure.get("event_count")),
                _bar("completed nodes", failure.get("completed_node_count")),
                _bar("failed nodes", failure.get("failed_node_count")),
                _bar("retrieval calls", failure.get("retriever_call_count")),
            ],
        ),
        "stage155_concurrency_events.svg": _chart(
            "Stage155 four-request observation events",
            [
                _bar("invocations", concurrency.get("invocation_count")),
                _bar("events", concurrency.get("event_count")),
                _bar("node events", concurrency.get("node_event_count")),
                _bar("isolation failures", concurrency.get("isolation_failure_count")),
            ],
        ),
        "stage155_concurrency_inflight.svg": _chart(
            "Stage155 concurrent in-flight observation",
            [
                _bar("maximum in flight", concurrency.get("max_observed_in_flight")),
                _bar("current after run", concurrency.get("current_in_flight")),
                _bar("graph compile count", concurrency.get("graph_compile_count")),
            ],
        ),
        "stage155_http_path.svg": _chart(
            "Stage155 synthetic HTTP path",
            [
                _bar("liveness status", http.get("liveness_status")),
                _bar("readiness status", http.get("readiness_status")),
                _bar("answer status", http.get("answer_status")),
                _bar("workflow events", http.get("workflow_event_count")),
            ],
        ),
        "stage155_real_observation.svg": _chart(
            "Stage155 real service observation",
            [
                _bar("invocations", real.get("invocation_count")),
                _bar("events", real.get("event_count")),
                _bar("node events", real.get("node_event_count")),
                _bar("failed events", real.get("failed_event_count")),
            ],
        ),
        "stage155_closed_boundaries.svg": _chart(
            "Stage155 closed runtime boundaries",
            [
                _truth_bar(
                    "runtime remains nondefault", not contract.get("runtime_registered_as_default")
                ),
                _truth_bar("remote remains closed", not contract.get("remote_exposure_authorized")),
                _truth_bar("test remains closed", not contract.get("test_access_allowed")),
                _truth_bar("sampling disabled", not contract.get("sampling_enabled")),
                _truth_bar("remote export disabled", not contract.get("remote_export_enabled")),
            ],
        ),
        "stage155_guard_status.svg": _chart(
            "Stage155 guard status",
            [
                BarDatum(
                    label=str(row["name"]),
                    value=1.0 if row["passed"] else 0.0,
                    value_label="passed" if row["passed"] else "failed",
                )
                for row in report.get("guard_checks", [])
            ],
            width=3200,
            margin_left=1800,
        ),
    }
    artifacts: list[AgentRuntimeObservabilityVisualization] = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        artifacts.append(AgentRuntimeObservabilityVisualization(name=name, path=str(path)))
    return artifacts


def _workflow_case(*, refuse: bool) -> dict[str, Any]:
    sink = RecordingAgentWorkflowObservationSink()
    workflow = create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
        toolset=_toolset(_StaticRetriever(prefix="stage155"), refuse=refuse),
        observation_sink=sink,
    )
    run = workflow.run(_question("synthetic-stage155"))
    summary = _observation_summary(sink.events)
    invocation = summary["invocations"][0]
    return {
        "terminal_state": run.public_safe_trace.terminal_state,
        "event_count": summary["event_count"],
        "node_event_count": summary["node_event_count"],
        "failed_event_count": summary["failed_event_count"],
        "event_sequences_exact": invocation["event_sequences_exact"],
        "node_ids_exact": invocation["node_ids"] == list(_EXPECTED_NODE_IDS),
        "candidate_pool_depth": invocation["candidate_pool_depth"],
        "generation_context_count": invocation["generation_context_count"],
        "verification_context_count": invocation["verification_context_count"],
        "tool_call_count": invocation["tool_call_count"],
        "node_latency_ms": invocation["node_latency_ms"],
        "workflow_elapsed_ms": invocation["workflow_elapsed_ms"],
        "forbidden_keys_found": sorted(
            _forbidden_keys_found([event.to_public_dict() for event in sink.events])
        ),
    }


def _failure_case() -> dict[str, Any]:
    error = RuntimeError("private Stage155 retrieval failure")
    retriever = _StaticRetriever(prefix="stage155-failure", error=error)
    sink = RecordingAgentWorkflowObservationSink()
    workflow = create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
        toolset=_toolset(retriever, refuse=False),
        observation_sink=sink,
    )
    caught: BaseException | None = None
    try:
        workflow.run(_question("private-stage155-failure"))
    except BaseException as captured:
        caught = captured
    summary = _observation_summary(sink.events)
    return {
        "same_error_object": caught is error,
        "error_type": type(caught).__name__ if caught is not None else None,
        "event_count": summary["event_count"],
        "completed_node_count": sum(event.event_type == "node_completed" for event in sink.events),
        "failed_node_count": sum(event.event_type == "node_failed" for event in sink.events),
        "failed_node_id": next(
            (event.node_id for event in sink.events if event.event_type == "node_failed"),
            None,
        ),
        "failure_stage": sink.events[-1].failure_stage if sink.events else None,
        "retriever_call_count": retriever.call_count,
        "retry_action_count": sum(event.retry_action_count for event in sink.events),
        "fallback_action_count": sum(event.fallback_action_count for event in sink.events),
        "error_message_public": "private Stage155 retrieval failure"
        in str([event.to_public_dict() for event in sink.events]),
    }


def _concurrency_case() -> dict[str, Any]:
    retriever = _FourRequestRetriever()
    sink = RecordingAgentWorkflowObservationSink()
    workflow = create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
        toolset=_toolset(retriever, refuse=False),
        observation_sink=sink,
    )
    handles = ("stage155-alpha", "stage155-beta", "stage155-gamma", "stage155-delta")
    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(lambda handle: workflow.run(_question(handle)), handles))
    isolation_failures = sum(
        run.verified_answer.question_id != handle
        or not all(
            result.document.id.startswith(f"{handle}-") for result in run.candidate_pool_results
        )
        for handle, run in zip(handles, runs, strict=True)
    )
    summary = _observation_summary(sink.events)
    workflow_counters = workflow.counters()
    observation_counters = workflow.observation_counters()
    return {
        "invocation_count": summary["invocation_count"],
        "event_count": summary["event_count"],
        "node_event_count": summary["node_event_count"],
        "failed_event_count": summary["failed_event_count"],
        "all_event_sequences_exact": all(
            row["event_sequences_exact"] for row in summary["invocations"]
        ),
        "all_node_ids_exact": all(
            row["node_ids"] == list(_EXPECTED_NODE_IDS) for row in summary["invocations"]
        ),
        "isolation_failure_count": isolation_failures,
        "retriever_call_count": retriever.call_count,
        "max_observed_in_flight": observation_counters.max_observed_in_flight,
        "current_in_flight": observation_counters.current_in_flight,
        "graph_compile_count": workflow_counters.graph_compile_count,
        "delivery_failure_count": observation_counters.delivery_failure_count,
    }


def _http_case() -> dict[str, Any]:
    candidates = _candidate_results("stage155-http")

    def searcher(query: str, top_k: int) -> Sequence[Any]:
        _ = query
        return candidates[:top_k]

    retriever = PrimeQAHybridOnlineCandidatePoolRetriever(
        channels=(
            IndependentCandidatePoolSearchChannel(
                channel_id="stage155_synthetic",
                family="stage155_synthetic",
                weight=1.0,
                searcher=searcher,
            ),
        ),
        config=CandidatePoolRetrievalConfig(
            channel_top_k=400,
            prefix_depth=200,
            target_pool_depth=400,
            rrf_k=60,
        ),
    )
    observation_sink = RecordingAgentWorkflowObservationSink()
    summary = _resource_summary()
    runtime = create_primeqa_hybrid_concurrent_sidecar_agent_runtime(
        shared_resources=PrimeQAHybridSharedRuntimeResources(
            candidate_pool_retriever=retriever,
            summary=summary,
        ),
        observation_sink=observation_sink,
    )
    bootstrap = PrimeQAHybridConcurrentRuntimeBootstrapResult(
        runtime=runtime,
        startup_trace=_active_startup_trace(),
        resource_summary=summary,
        source_evaluation=None,
    )
    app = create_primeqa_hybrid_agent_http_app(
        settings=ProjectSettings(
            _env_file=None,
            enable_concurrent_sidecar_agent=True,
            enable_local_agent_http_transport=True,
        ),
        bootstrap_result=bootstrap,
    )
    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        answer = client.post(
            "/v1/agent/answers",
            json={
                "request_handle": "private-stage155-http",
                "title": "Adapter installation",
                "text": "How do I apply the documented adapter procedure?",
            },
        )
    observation = _observation_summary(observation_sink.events)
    return {
        "liveness_status": live.status_code,
        "readiness_status": ready.status_code,
        "answer_status": answer.status_code,
        "workflow_event_count": observation["event_count"],
        "workflow_node_event_count": observation["node_event_count"],
        "workflow_failed_event_count": observation["failed_event_count"],
        "candidate_pool_depth": observation["invocations"][0]["candidate_pool_depth"],
        "tool_call_count": observation["invocations"][0]["tool_call_count"],
        "transport_closed": app.state.agent_http_transport.state.value == "closed",
        "request_content_saved": False,
    }


def _activation_case(
    *,
    report: Mapping[str, Any],
    source_paths: Mapping[str, Path],
) -> dict[str, Any]:
    current = tuple(
        AgentServiceSourceFingerprint(
            source_key=name,
            size_bytes=int(_fingerprint(path)["size_bytes"]),
            sha256=str(_fingerprint(path)["sha256"]),
        )
        for name, path in source_paths.items()
    )
    tampered = deepcopy(report)
    tampered["decision"]["runtime_registered_as_default"] = True
    stale = list(current)
    stale[-1] = AgentServiceSourceFingerprint(
        source_key=stale[-1].source_key,
        size_bytes=stale[-1].size_bytes,
        sha256="0" * 64,
    )
    return {
        "current_stage154_authorized": _stage154_authorized(report, current),
        "tampered_stage154_rejected": not _stage154_authorized(tampered, current),
        "stale_source_rejected": not _stage154_authorized(report, stale),
        "source_fingerprint_count": len(current),
        "resource_build_before_authorization_allowed": False,
        "listener_bind_before_authorization_allowed": False,
        "alternate_source_or_port_allowed": False,
    }


def _observation_summary(
    events: Sequence[PublicSafeAgentWorkflowObservationEvent],
) -> dict[str, Any]:
    grouped: dict[int, list[PublicSafeAgentWorkflowObservationEvent]] = defaultdict(list)
    for event in events:
        grouped[event.invocation_sequence].append(event)
    invocations: list[dict[str, Any]] = []
    for invocation_sequence, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda event: event.event_sequence)
        terminal = ordered[-1]
        node_rows = [event for event in ordered if event.event_type.startswith("node_")]
        invocations.append(
            {
                "invocation_sequence": invocation_sequence,
                "event_count": len(ordered),
                "event_sequences_exact": [event.event_sequence for event in ordered]
                == list(range(1, len(ordered) + 1)),
                "node_ids": [event.node_id for event in node_rows],
                "node_latency_ms": {event.node_id: event.node_latency_ms for event in node_rows},
                "terminal_event_type": terminal.event_type,
                "workflow_elapsed_ms": terminal.workflow_elapsed_ms,
                "candidate_pool_depth": terminal.candidate_pool_depth,
                "generation_context_count": terminal.generation_context_count,
                "verification_context_count": terminal.verification_context_count,
                "tool_call_count": terminal.tool_call_count,
                "failure_stage": terminal.failure_stage,
            }
        )
    return {
        "invocation_count": len(grouped),
        "event_count": len(events),
        "node_event_count": sum(event.event_type.startswith("node_") for event in events),
        "failed_event_count": sum(event.outcome == "failed" for event in events),
        "maximum_recorded_in_flight": max(
            (event.current_in_flight for event in events),
            default=0,
        ),
        "invocations": invocations,
        "public_event_field_count": len(events[0].to_public_dict()) if events else 0,
        "forbidden_keys_found": sorted(
            _forbidden_keys_found([event.to_public_dict() for event in events])
        ),
    }


def _stage154_summary(
    *,
    report: Mapping[str, Any],
    source_paths: Mapping[str, Path],
) -> dict[str, Any]:
    checks = report.get("guard_checks") or []
    decision = report.get("decision") or {}
    current = tuple(
        AgentServiceSourceFingerprint(
            source_key=name,
            size_bytes=int(_fingerprint(path)["size_bytes"]),
            sha256=str(_fingerprint(path)["sha256"]),
        )
        for name, path in source_paths.items()
    )
    return {
        "identity_exact": (
            report.get("stage") == "Stage 154"
            and report.get("analysis_id")
            == "primeqa_hybrid_langgraph_agent_tool_workflow_validation_v1"
            and decision.get("status") == _SOURCE_STAGE154_STATUS
        ),
        "guard_count": len(checks),
        "passed_guard_count": sum(row.get("passed") is True for row in checks),
        "all_guards_passed": (
            len(checks) == _EXPECTED_STAGE154_GUARDS
            and all(row.get("passed") is True for row in checks)
        ),
        "current_sources_authorized": _stage154_authorized(report, current),
        "workflow_implemented": decision.get("workflow_implemented"),
        "real_lifecycle_validated": decision.get("real_resource_service_lifecycle_validated"),
        "runtime_registered_as_default": decision.get("runtime_registered_as_default"),
        "test_gate_opened": decision.get("test_gate_opened"),
        "test_metrics_run": decision.get("test_metrics_run"),
    }


def _guard_checks(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = report.get("stage154_summary") or {}
    contract = report.get("activation_observability_contract") or {}
    activation = report.get("activation_validation") or {}
    synthetic = report.get("synthetic_validation") or {}
    complete = synthetic.get("complete") or {}
    refuse = synthetic.get("refuse") or {}
    failure = synthetic.get("failure") or {}
    concurrency = synthetic.get("concurrency_four") or {}
    http = synthetic.get("http") or {}
    real = report.get("real_resource_service_lifecycle") or {}
    real_http = real.get("http_probe") or {}
    real_observation = report.get("real_observation") or {}
    public = report.get("public_safe_contract") or {}
    return [
        _check("stage155_user_confirmed", report["user_confirmation"]["confirmed"] is True),
        _check("stage155_confirmation_note_present", bool(report["user_confirmation"]["note"])),
        _check("stage154_identity_exact", source.get("identity_exact") is True),
        _check("stage154_all_54_guards_passed", source.get("all_guards_passed") is True),
        _check(
            "stage154_current_sources_authorized", source.get("current_sources_authorized") is True
        ),
        _check("stage154_workflow_implemented", source.get("workflow_implemented") is True),
        _check("stage154_real_lifecycle_validated", source.get("real_lifecycle_validated") is True),
        _check(
            "stage154_test_remained_closed",
            source.get("test_gate_opened") is False and source.get("test_metrics_run") is False,
        ),
        _check(
            "activation_dual_explicit_flags_exact",
            contract.get("required_explicit_environment_flags")
            == [
                "TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT",
                "TS_RAG_ENABLE_LOCAL_AGENT_HTTP_TRANSPORT",
            ],
        ),
        _check(
            "activation_stage154_formal_required",
            contract.get("stage154_formal_evidence_required") is True,
        ),
        _check(
            "activation_current_fingerprints_required",
            contract.get("stage154_current_source_fingerprints_required") is True,
        ),
        _check("activation_loopback_only", contract.get("binding_host") == "127.0.0.1"),
        _check(
            "activation_observability_not_disableable",
            contract.get("observability_disable_flag") is None,
        ),
        _check(
            "activation_current_evidence_passes",
            activation.get("current_stage154_authorized") is True,
        ),
        _check(
            "activation_tampered_evidence_rejected",
            activation.get("tampered_stage154_rejected") is True,
        ),
        _check("activation_stale_source_rejected", activation.get("stale_source_rejected") is True),
        _check(
            "activation_before_resource_and_bind",
            activation.get("resource_build_before_authorization_allowed") is False
            and activation.get("listener_bind_before_authorization_allowed") is False,
        ),
        _check("complete_terminal_exact", complete.get("terminal_state") == "complete"),
        _check(
            "complete_nine_events_exact",
            complete.get("event_count") == 9 and complete.get("node_event_count") == 7,
        ),
        _check(
            "complete_sequences_and_nodes_exact",
            complete.get("event_sequences_exact") is True
            and complete.get("node_ids_exact") is True,
        ),
        _check(
            "complete_depths_and_tools_exact",
            [
                complete.get("candidate_pool_depth"),
                complete.get("generation_context_count"),
                complete.get("verification_context_count"),
                complete.get("tool_call_count"),
            ]
            == [400, 10, 200, 3],
        ),
        _check("refuse_terminal_exact", refuse.get("terminal_state") == "refuse"),
        _check(
            "refuse_nine_events_exact",
            refuse.get("event_count") == 9 and refuse.get("node_event_count") == 7,
        ),
        _check(
            "refuse_sequences_and_nodes_exact",
            refuse.get("event_sequences_exact") is True and refuse.get("node_ids_exact") is True,
        ),
        _check(
            "workflow_events_public_safe",
            complete.get("forbidden_keys_found") == [] and refuse.get("forbidden_keys_found") == [],
        ),
        _check("failure_same_error_object", failure.get("same_error_object") is True),
        _check(
            "failure_timeline_exact",
            [
                failure.get("event_count"),
                failure.get("completed_node_count"),
                failure.get("failed_node_count"),
            ]
            == [4, 1, 1],
        ),
        _check(
            "failure_stage_exact",
            failure.get("failed_node_id") == "retrieve_candidate_pool"
            and failure.get("failure_stage") == "retrieve_candidate_pool",
        ),
        _check(
            "failure_one_retrieval_no_recovery",
            failure.get("retriever_call_count") == 1
            and failure.get("retry_action_count") == 0
            and failure.get("fallback_action_count") == 0,
        ),
        _check("failure_message_not_public", failure.get("error_message_public") is False),
        _check(
            "concurrency_four_complete",
            concurrency.get("invocation_count") == 4
            and concurrency.get("event_count") == 36
            and concurrency.get("node_event_count") == 28,
        ),
        _check(
            "concurrency_sequences_and_nodes_exact",
            concurrency.get("all_event_sequences_exact") is True
            and concurrency.get("all_node_ids_exact") is True,
        ),
        _check(
            "concurrency_request_isolated",
            concurrency.get("isolation_failure_count") == 0
            and concurrency.get("retriever_call_count") == 4,
        ),
        _check(
            "concurrency_inflight_exact",
            concurrency.get("max_observed_in_flight") == 4
            and concurrency.get("current_in_flight") == 0,
        ),
        _check("concurrency_graph_compiled_once", concurrency.get("graph_compile_count") == 1),
        _check(
            "concurrency_delivery_complete",
            concurrency.get("delivery_failure_count") == 0
            and concurrency.get("failed_event_count") == 0,
        ),
        _check(
            "http_liveness_ready_answer_200",
            [http.get("liveness_status"), http.get("readiness_status"), http.get("answer_status")]
            == [200, 200, 200],
        ),
        _check(
            "http_workflow_timeline_exact",
            http.get("workflow_event_count") == 9
            and http.get("workflow_node_event_count") == 7
            and http.get("workflow_failed_event_count") == 0,
        ),
        _check(
            "http_depth_and_tools_exact",
            http.get("candidate_pool_depth") == 400 and http.get("tool_call_count") == 3,
        ),
        _check("http_transport_closed", http.get("transport_closed") is True),
        _check("real_lifecycle_executed", real.get("executed") is True),
        _check("real_service_exit_zero", real.get("exit_code") == 0),
        _check(
            "real_eleven_sources_fingerprinted", len(real.get("source_fingerprints") or []) == 11
        ),
        _check(
            "real_http_200_exact",
            [
                real_http.get("liveness_status"),
                real_http.get("readiness_status"),
                real_http.get("answer_status"),
            ]
            == [200, 200, 200],
        ),
        _check(
            "real_service_released",
            real.get("transport_closed") is True and real.get("listener_released") is True,
        ),
        _check("real_two_invocations_observed", real_observation.get("invocation_count") == 2),
        _check(
            "real_eighteen_events_exact",
            real_observation.get("event_count") == 18
            and real_observation.get("node_event_count") == 14,
        ),
        _check("real_observation_no_failures", real_observation.get("failed_event_count") == 0),
        _check(
            "real_observation_public_safe",
            real_observation.get("public_event_field_count") == 22
            and real_observation.get("forbidden_keys_found") == [],
        ),
        _check(
            "observation_schema_exact",
            contract.get("public_event_field_count") == 22
            and contract.get("node_ids") == list(_EXPECTED_NODE_IDS),
        ),
        _check(
            "observation_has_no_wall_clock_or_content",
            contract.get("wall_clock_timestamp_recorded") is False
            and contract.get("request_content_recorded") is False
            and contract.get("request_identifiers_recorded") is False
            and contract.get("document_content_recorded") is False
            and contract.get("document_identifiers_recorded") is False,
        ),
        _check(
            "observation_no_sampling_batching_remote_export",
            contract.get("sampling_enabled") is False
            and contract.get("batching_enabled") is False
            and contract.get("remote_export_enabled") is False,
        ),
        _check(
            "runtime_remote_test_remain_closed",
            contract.get("runtime_registered_as_default") is False
            and contract.get("remote_exposure_authorized") is False
            and contract.get("test_access_allowed") is False,
        ),
        _check(
            "no_queue_retry_fallback",
            contract.get("queue_actions_allowed") is False
            and contract.get("retry_actions_allowed") is False
            and contract.get("fallback_strategies_allowed") is False,
        ),
        _check(
            "report_loaded_no_evaluation_rows",
            public.get("test_split_loaded") is False
            and public.get("test_metrics_run") is False
            and public.get("train_or_dev_rows_loaded") is False,
        ),
        _check(
            "report_saves_no_private_content",
            public.get("request_content_saved") is False
            and public.get("request_identifiers_saved") is False
            and public.get("document_content_saved") is False
            and public.get("document_identifiers_saved") is False,
        ),
        _check("source_files_unchanged", report.get("source_unchanged_after_validation") is True),
    ]


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


def _latency_chart(title: str, case: Mapping[str, Any]) -> str:
    latency = case.get("node_latency_ms") or {}
    return _chart(
        title,
        [
            BarDatum(
                label=node_id,
                value=float(latency.get(node_id) or 0.0),
                value_label=f"{float(latency.get(node_id) or 0.0):.3f}ms",
            )
            for node_id in _EXPECTED_NODE_IDS
        ],
    )


def _chart(
    title: str,
    bars: list[BarDatum],
    *,
    width: int = 1900,
    margin_left: int = 900,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label="observed value",
        width=width,
        margin_left=margin_left,
    )
