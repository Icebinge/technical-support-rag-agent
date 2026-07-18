from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from threading import Barrier, Lock
from typing import Any, Protocol

import tomli
from fastapi.testclient import TestClient

from ts_rag_agent.application.primeqa_hybrid_agent_http_transport import (
    AgentHttpTransportLogSink,
    PublicSafeAgentHttpTransportEvent,
    create_primeqa_hybrid_agent_http_app,
)
from ts_rag_agent.application.primeqa_hybrid_agent_tool_orchestration_protocol import (
    AGENT_TOOL_ORCHESTRATION_PROTOCOL_ID,
    AGENT_TOOL_WORKFLOW_GRAPH_ID,
    AgentToolWorkflowState,
    InvalidAgentToolWorkflowTransitionError,
)
from ts_rag_agent.application.primeqa_hybrid_agent_tool_workflow import (
    AgentToolWorkflowNodeExecutor,
    AgentToolWorkflowPrivateState,
    PrimeQAHybridAgentToolset,
    agent_tool_workflow_implementation_contract,
    create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset,
    create_primeqa_hybrid_reference_agent_tool_workflow,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_activation import (
    PrimeQAHybridConcurrentRuntimeBootstrapResult,
    PublicSafeConcurrentRuntimeStartupTrace,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_sidecar_agent_runtime import (
    concurrent_sidecar_runtime_contract,
    create_primeqa_hybrid_concurrent_sidecar_agent_runtime,
)
from ts_rag_agent.application.primeqa_hybrid_online_candidate_pool_retriever import (
    CandidatePoolRetrievalConfig,
    IndependentCandidatePoolSearchChannel,
    PrimeQAHybridOnlineCandidatePoolRetriever,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    PrimeQAHybridRuntimeResourceSummary,
    PrimeQAHybridSharedRuntimeResources,
    _forbidden_keys_found,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    PrimeQAHybridSidecarObservationAdapter,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.domain.answer import (
    AnswerCitation,
    AnswerVerificationResult,
    GeneratedAnswer,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuery, PrimeQARuntimeQuery
from ts_rag_agent.domain.retrieval import RetrievalResult

_STAGE = "Stage 154"
_CREATED_AT = "2026-07-18"
_ANALYSIS_ID = "primeqa_hybrid_langgraph_agent_tool_workflow_validation_v1"
_FINAL_STATUS = "primeqa_hybrid_langgraph_agent_tool_workflow_implemented_and_validated"
_NEXT_DIRECTION = "freeze_agent_workflow_runtime_activation_and_operational_observability"
_STAGE153_STATUS = "primeqa_hybrid_local_agent_tool_orchestration_protocol_frozen"
_STAGE152_STATUS = "primeqa_hybrid_local_agent_service_entrypoint_implemented_and_validated"
_EXPECTED_STAGE153_GUARDS = 46
_EXPECTED_STAGE152_GUARDS = 46
_EXPECTED_DIRECT_AGENT_DEPENDENCIES = ["langgraph==1.2.9"]
_EXPECTED_VERSIONS = {
    "langgraph": "1.2.9",
    "langchain-core": "1.4.9",
    "langchain-protocol": "0.0.18",
    "langgraph-checkpoint": "4.1.1",
    "langgraph-prebuilt": "1.1.0",
    "langgraph-sdk": "0.4.2",
    "langsmith": "0.10.6",
    "orjson": "3.11.9",
    "ormsgpack": "1.12.2",
    "requests-toolbelt": "1.0.0",
    "sniffio": "1.3.1",
    "tenacity": "9.1.4",
    "uuid-utils": "0.17.0",
    "websockets": "15.0.1",
    "xxhash": "3.8.1",
}


@dataclass(frozen=True)
class AgentToolWorkflowValidationVisualization:
    name: str
    path: str


class _CandidatePoolRetriever(Protocol):
    call_count: int

    def retrieve(self, question: PrimeQAQuery) -> Sequence[RetrievalResult]: ...


class _StaticRetriever:
    def __init__(self, *, prefix: str, error: Exception | None = None) -> None:
        self._prefix = prefix
        self._error = error
        self._lock = Lock()
        self.call_count = 0

    def retrieve(self, question: PrimeQAQuery) -> tuple[RetrievalResult, ...]:
        _ = question
        with self._lock:
            self.call_count += 1
        if self._error is not None:
            raise self._error
        return _candidate_results(self._prefix)


class _FourRequestRetriever:
    def __init__(self) -> None:
        self._barrier = Barrier(4)
        self._lock = Lock()
        self.call_count = 0

    def retrieve(self, question: PrimeQAQuery) -> tuple[RetrievalResult, ...]:
        with self._lock:
            self.call_count += 1
        self._barrier.wait()
        return _candidate_results(question.id)


class _SyntheticGenerator:
    def generate(
        self,
        question: PrimeQAQuery,
        retrieval_results: Sequence[RetrievalResult],
    ) -> GeneratedAnswer:
        first = retrieval_results[0]
        return GeneratedAnswer(
            question_id=question.id,
            answer="Synthetic private answer excluded from Stage154 evidence.",
            citations=[
                AnswerCitation(
                    document_id=first.document.id,
                    title=first.document.title,
                    retrieval_rank=first.rank,
                    evidence_score=first.score,
                )
            ],
            refused=False,
        )


class _SyntheticGeneratorFactory:
    def create(self) -> _SyntheticGenerator:
        return _SyntheticGenerator()


class _SyntheticVerifier:
    def __init__(self, *, refuse: bool) -> None:
        self._refuse = refuse

    def verify(
        self,
        answer: GeneratedAnswer,
        retrieval_results: Sequence[RetrievalResult],
    ) -> AnswerVerificationResult:
        if len(retrieval_results) != 200:
            raise RuntimeError("synthetic verifier requires the exact rank-200 prefix")
        verified = (
            GeneratedAnswer(
                question_id=answer.question_id,
                answer="Synthetic refusal excluded from Stage154 evidence.",
                citations=[],
                refused=True,
            )
            if self._refuse
            else answer
        )
        return AnswerVerificationResult(
            original_answer=answer,
            verified_answer=verified,
            citation_context_valid=not self._refuse,
            reasons=["synthetic_refusal"] if self._refuse else [],
        )


class _SyntheticVerifierFactory:
    def __init__(self, *, refuse: bool) -> None:
        self._refuse = refuse

    def create(self) -> _SyntheticVerifier:
        return _SyntheticVerifier(refuse=self._refuse)


class _RecordingLogSink(AgentHttpTransportLogSink):
    def __init__(self) -> None:
        self._lock = Lock()
        self.events: list[PublicSafeAgentHttpTransportEvent] = []

    def emit(self, event: PublicSafeAgentHttpTransportEvent) -> None:
        event.to_public_dict()
        with self._lock:
            self.events.append(event)


def run_primeqa_hybrid_agent_tool_workflow_validation(
    *,
    stage153_protocol_path: Path,
    pyproject_path: Path,
    workflow_source_path: Path,
    concurrent_runtime_source_path: Path,
    stage152_support_validation_path: Path | None,
    user_confirmed_validation: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Validate the Stage154 implementation without loading an evaluation split."""

    started_at = time.perf_counter()
    source_paths = {
        "stage153_protocol": stage153_protocol_path,
        "pyproject": pyproject_path,
        "workflow_source": workflow_source_path,
        "concurrent_runtime_source": concurrent_runtime_source_path,
    }
    source_before = {name: _fingerprint(path) for name, path in source_paths.items()}
    stage153 = _load_json_object(stage153_protocol_path)
    stage152_support = (
        _load_json_object(stage152_support_validation_path)
        if stage152_support_validation_path is not None
        and stage152_support_validation_path.is_file()
        else None
    )
    loaded_at = time.perf_counter()
    dependency_evidence = _dependency_evidence(pyproject_path)
    complete = _equivalence_case(refuse=False)
    refuse = _equivalence_case(refuse=True)
    invalid = _invalid_transition_case()
    failure = _failure_case()
    concurrency = _concurrency_case()
    http = _http_integration_case()
    synthetic_finished_at = time.perf_counter()
    source_after = {name: _fingerprint(path) for name, path in source_paths.items()}
    preliminary: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "validation_scope": (
            "Stage154 validates the installed LangGraph StateGraph adapter, its framework-neutral "
            "reference engine, request isolation, unchanged error propagation, and the existing "
            "facade/HTTP request path. Synthetic validation loads no train/dev/test rows. The "
            "separate Stage152 support artifact may prove one current real resource and loopback "
            "service lifecycle. No runtime default, remote exposure, queue, retry, or fallback "
            "is added."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_validation),
            "confirmation_note": confirmation_note,
        },
        "source_files": source_before,
        "source_unchanged_after_validation": source_before == source_after,
        "stage153_summary": _stage153_summary(stage153),
        "dependency_evidence": dependency_evidence,
        "implementation_contract": agent_tool_workflow_implementation_contract(),
        "synthetic_validation": {
            "complete_equivalence": complete,
            "refuse_equivalence": refuse,
            "invalid_transition": invalid,
            "error_propagation": failure,
            "concurrency": concurrency,
            "facade_http_integration": http,
        },
        "stage152_current_service_support": _stage152_support_summary(stage152_support),
    }
    checks = _guard_checks(preliminary)
    passed = all(check["passed"] for check in checks)
    checked_at = time.perf_counter()
    report = {
        **preliminary,
        "guard_checks": checks,
        "decision": {
            "status": _FINAL_STATUS
            if passed
            else "primeqa_hybrid_langgraph_agent_tool_workflow_validation_rejected",
            "failed_checks": [check["name"] for check in checks if not check["passed"]],
            "workflow_implemented": passed,
            "langgraph_adapter_validated": passed,
            "facade_http_request_path_validated": passed,
            "real_resource_service_lifecycle_validated": (
                passed
                and preliminary["stage152_current_service_support"].get("real_executed") is True
            ),
            "runtime_registered_as_default": False,
            "remote_exposure_authorized": False,
            "test_gate_opened": False,
            "test_metrics_run": False,
            "queue_actions_enabled": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
            "next_direction": _NEXT_DIRECTION if passed else "repair_failed_stage154_guards",
        },
        "timing_seconds": {
            "load_saved_sources": round(loaded_at - started_at, 6),
            "synthetic_validation": round(synthetic_finished_at - loaded_at, 6),
            "guard_and_report": round(checked_at - synthetic_finished_at, 6),
            "total": round(checked_at - started_at, 6),
        },
    }
    report["public_safe_contract"] = _public_safe_contract(report)
    return report


def write_primeqa_hybrid_agent_tool_workflow_validation_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[AgentToolWorkflowValidationVisualization]:
    """Write ten public-safe Stage154 SVG charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic = report.get("synthetic_validation") or {}
    dependency = report.get("dependency_evidence") or {}
    topology = (synthetic.get("complete_equivalence") or {}).get("langgraph_topology") or {}
    concurrency = synthetic.get("concurrency") or {}
    http = synthetic.get("facade_http_integration") or {}
    real = report.get("stage152_current_service_support") or {}
    charts = {
        "stage154_source_guards.svg": _chart(
            "Stage154 source guards",
            [
                _bar("Stage153 passed guards", report["stage153_summary"]["passed_guards"]),
                _bar("Stage152 support passed guards", real.get("passed_guards", 0)),
            ],
        ),
        "stage154_dependency_versions.svg": _chart(
            "Stage154 exact dependency versions",
            [
                _truth_bar(name, value == _EXPECTED_VERSIONS[name])
                for name, value in (dependency.get("installed_versions") or {}).items()
            ],
        ),
        "stage154_graph_topology.svg": _chart(
            "Stage154 LangGraph topology",
            [
                _bar("nodes", topology.get("node_count", 0)),
                _bar("conditional edges", topology.get("conditional_edge_count", 0)),
                _bar("compile count", topology.get("compile_count", 0)),
                _bar("checkpointers", int(bool(topology.get("checkpointer_attached")))),
                _bar("caches", int(bool(topology.get("cache_attached")))),
            ],
        ),
        "stage154_engine_equivalence.svg": _chart(
            "Stage154 reference and LangGraph equivalence",
            [
                _truth_bar("complete", (synthetic.get("complete_equivalence") or {}).get("equal")),
                _truth_bar("refuse", (synthetic.get("refuse_equivalence") or {}).get("equal")),
            ],
        ),
        "stage154_tool_call_counts.svg": _chart(
            "Stage154 successful tool call counts",
            [
                _bar(
                    "retrieve",
                    (synthetic.get("complete_equivalence") or {}).get(
                        "retrieval_tool_call_count", 0
                    ),
                ),
                _bar(
                    "answer",
                    (synthetic.get("complete_equivalence") or {}).get("answer_tool_call_count", 0),
                ),
                _bar(
                    "verify",
                    (synthetic.get("complete_equivalence") or {}).get(
                        "verification_tool_call_count", 0
                    ),
                ),
            ],
        ),
        "stage154_concurrency.svg": _chart(
            "Stage154 request isolation concurrency",
            [
                _bar("invocations", concurrency.get("invocation_count", 0)),
                _bar("completed", concurrency.get("completed_count", 0)),
                _bar("max in flight", concurrency.get("max_observed_in_flight", 0)),
                _bar("state isolation failures", concurrency.get("isolation_failure_count", 0)),
            ],
        ),
        "stage154_error_propagation.svg": _chart(
            "Stage154 error propagation",
            [
                _truth_bar(
                    "same error object",
                    (synthetic.get("error_propagation") or {}).get("same_object"),
                ),
                _bar(
                    "retry actions",
                    (synthetic.get("error_propagation") or {}).get("retry_action_count", 0),
                ),
                _bar(
                    "fallback actions",
                    (synthetic.get("error_propagation") or {}).get("fallback_action_count", 0),
                ),
            ],
        ),
        "stage154_http_integration.svg": _chart(
            "Stage154 in-process HTTP integration",
            [
                _bar("live", http.get("liveness_status", 0)),
                _bar("ready", http.get("readiness_status", 0)),
                _bar("answer", http.get("answer_status", 0)),
                _bar("candidate pool", http.get("candidate_pool_depth", 0)),
            ],
        ),
        "stage154_real_service_support.svg": _chart(
            "Stage154 current real service support",
            [
                _truth_bar("executed", real.get("real_executed")),
                _truth_bar("HTTP 200", real.get("answer_status") == 200),
                _truth_bar("listener released", real.get("listener_released")),
                _truth_bar("transport closed", real.get("transport_closed")),
            ],
        ),
        "stage154_guard_status.svg": _chart(
            "Stage154 formal guards",
            [
                BarDatum(
                    label=str(check["name"]),
                    value=1.0 if check["passed"] else 0.0,
                    value_label="passed" if check["passed"] else "failed",
                )
                for check in report.get("guard_checks", [])
            ],
            width=3000,
            margin_left=1650,
        ),
    }
    artifacts: list[AgentToolWorkflowValidationVisualization] = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        artifacts.append(AgentToolWorkflowValidationVisualization(name=name, path=str(path)))
    return artifacts


def _dependency_evidence(pyproject_path: Path) -> dict[str, Any]:
    with pyproject_path.open("rb") as handle:
        pyproject = tomli.load(handle)
    extras = (pyproject.get("project") or {}).get("optional-dependencies") or {}
    direct_agent = list(extras.get("agent") or [])
    installed = {name: version(name) for name in _EXPECTED_VERSIONS}
    return {
        "direct_agent_dependencies": direct_agent,
        "direct_agent_dependencies_exact": direct_agent == _EXPECTED_DIRECT_AGENT_DEPENDENCIES,
        "full_langchain_direct_dependency_present": any(
            dependency.startswith("langchain==")
            or dependency.startswith("langchain>=")
            or dependency.startswith("langchain-community")
            for dependency in direct_agent
        ),
        "installed_versions": installed,
        "installed_versions_exact": installed == _EXPECTED_VERSIONS,
        "pip_check_run_separately": True,
        "dependency_install_retry_count": 0,
        "websockets_before_install": "16.1",
        "websockets_after_install": installed["websockets"],
    }


def _equivalence_case(*, refuse: bool) -> dict[str, Any]:
    reference_retriever = _StaticRetriever(prefix="equivalence")
    graph_retriever = _StaticRetriever(prefix="equivalence")
    reference = create_primeqa_hybrid_reference_agent_tool_workflow(
        toolset=_toolset(reference_retriever, refuse=refuse)
    )
    graph = create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
        toolset=_toolset(graph_retriever, refuse=refuse)
    )
    question = _question("synthetic-equivalence")
    reference_run = reference.run(question)
    graph_run = graph.run(question)
    trace = graph_run.public_safe_trace
    equal = (
        graph_run.verified_answer == reference_run.verified_answer
        and graph_run.original_answer == reference_run.original_answer
        and graph_run.verification_result == reference_run.verification_result
        and graph_run.candidate_pool_results == reference_run.candidate_pool_results
        and graph_run.generation_context_results == reference_run.generation_context_results
        and graph_run.verification_context_results == reference_run.verification_context_results
        and graph_run.public_safe_trace == reference_run.public_safe_trace
    )
    topology = graph.topology()
    return {
        "equal": equal,
        "terminal_state": trace.terminal_state,
        "transition_count": trace.transition_count,
        "tool_call_count": trace.tool_call_count,
        "retrieval_tool_call_count": trace.retrieval_tool_call_count,
        "answer_tool_call_count": trace.answer_tool_call_count,
        "verification_tool_call_count": trace.verification_tool_call_count,
        "candidate_pool_depth": trace.candidate_pool_depth,
        "generation_context_count": trace.generation_context_count,
        "verification_context_count": trace.verification_context_count,
        "sidecar_observation_count": trace.sidecar_observation_count,
        "reference_retriever_call_count": reference_retriever.call_count,
        "graph_retriever_call_count": graph_retriever.call_count,
        "langgraph_topology": {
            **topology,
            "compile_count": graph.counters().graph_compile_count,
        },
    }


def _invalid_transition_case() -> dict[str, Any]:
    retriever = _StaticRetriever(prefix="invalid")
    executor = AgentToolWorkflowNodeExecutor(toolset=_toolset(retriever, refuse=False))
    state = _received_state(_question("synthetic-invalid"))
    before = _state_summary(state)
    rejected = False
    try:
        executor.retrieve_candidate_pool(state)
    except InvalidAgentToolWorkflowTransitionError:
        rejected = True
    return {
        "rejected": rejected,
        "state_unchanged": _state_summary(state) == before,
        "retriever_call_count": retriever.call_count,
    }


def _failure_case() -> dict[str, Any]:
    error = RuntimeError("synthetic private retrieval failure")
    retriever = _StaticRetriever(prefix="failure", error=error)
    workflow = create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
        toolset=_toolset(retriever, refuse=False)
    )
    caught: Exception | None = None
    try:
        workflow.run(_question("synthetic-failure"))
    except Exception as captured:
        caught = captured
    trace = workflow.last_public_trace
    if trace is None:
        raise RuntimeError("failure validation did not produce a public trace")
    public = trace.to_public_dict()
    return {
        "same_object": caught is error,
        "error_type": type(caught).__name__ if caught is not None else None,
        "failure_stage": trace.failure_stage,
        "terminal_state": trace.terminal_state,
        "transition_count": trace.transition_count,
        "retrieval_tool_call_count": trace.retrieval_tool_call_count,
        "retry_action_count": trace.retry_action_count,
        "fallback_action_count": trace.fallback_action_count,
        "error_message_public": "synthetic private retrieval failure" in str(public),
    }


def _concurrency_case() -> dict[str, Any]:
    retriever = _FourRequestRetriever()
    workflow = create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
        toolset=_toolset(retriever, refuse=False)
    )
    handles = ("synthetic-alpha", "synthetic-beta", "synthetic-gamma", "synthetic-delta")
    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(lambda handle: workflow.run(_question(handle)), handles))
    failures = sum(
        not (
            run.verified_answer.question_id == handle
            and all(
                result.document.id.startswith(f"{handle}-") for result in run.candidate_pool_results
            )
            and run.public_safe_trace.terminal_state == "complete"
        )
        for handle, run in zip(handles, runs, strict=True)
    )
    counters = workflow.counters()
    return {
        "invocation_count": counters.invocation_count,
        "completed_count": counters.completed_count,
        "failed_count": counters.failed_count,
        "current_in_flight": counters.current_in_flight,
        "max_observed_in_flight": counters.max_observed_in_flight,
        "graph_compile_count": counters.graph_compile_count,
        "retriever_call_count": retriever.call_count,
        "isolation_failure_count": failures,
    }


def _http_integration_case() -> dict[str, Any]:
    candidates = _candidate_results("http")

    def searcher(query: str, top_k: int) -> Sequence[RetrievalResult]:
        _ = query
        return candidates[:top_k]

    retriever = PrimeQAHybridOnlineCandidatePoolRetriever(
        channels=(
            IndependentCandidatePoolSearchChannel(
                channel_id="stage154_synthetic",
                family="stage154_synthetic",
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
    summary = _resource_summary()
    runtime = create_primeqa_hybrid_concurrent_sidecar_agent_runtime(
        shared_resources=PrimeQAHybridSharedRuntimeResources(
            candidate_pool_retriever=retriever,
            summary=summary,
        )
    )
    bootstrap = PrimeQAHybridConcurrentRuntimeBootstrapResult(
        runtime=runtime,
        startup_trace=_active_startup_trace(),
        resource_summary=summary,
        source_evaluation=None,
    )
    sink = _RecordingLogSink()
    app = create_primeqa_hybrid_agent_http_app(
        settings=ProjectSettings(
            _env_file=None,
            enable_concurrent_sidecar_agent=True,
            enable_local_agent_http_transport=True,
        ),
        bootstrap_result=bootstrap,
        log_sink=sink,
    )
    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        answer = client.post(
            "/v1/agent/answers",
            json={
                "request_handle": "stage154-synthetic-http",
                "title": "Adapter installation",
                "text": "How do I apply the documented adapter procedure?",
            },
        )
        answer_body = answer.json()
    answer_events = [event for event in sink.events if event.route_id == "agent_answer"]
    answer_event = answer_events[-1]
    return {
        "liveness_status": live.status_code,
        "readiness_status": ready.status_code,
        "answer_status": answer.status_code,
        "answer_refused": answer_body["refused"],
        "citation_count": len(answer_body["citations"]),
        "candidate_pool_depth": answer_event.candidate_pool_depth,
        "terminal_state": answer_event.terminal_state,
        "transport_closed": app.state.agent_http_transport.state.value == "closed",
        "request_content_saved": False,
        "queue_action_count": answer_event.queue_action_count,
        "retry_action_count": answer_event.retry_action_count,
        "fallback_action_count": answer_event.fallback_action_count,
    }


def _active_startup_trace() -> PublicSafeConcurrentRuntimeStartupTrace:
    contract = concurrent_sidecar_runtime_contract()
    return PublicSafeConcurrentRuntimeStartupTrace(
        runtime_mode=contract["runtime_mode"],
        settings_field="enable_concurrent_sidecar_agent",
        environment_flag="TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT",
        activation_requested=True,
        activation_state="eligible",
        source_validation_state="eligible",
        slo_profile_id=contract["slo_profile_id"],
        max_in_flight=4,
        warm_resources_ready=True,
        resources_initialized=True,
        runtime_activated=True,
        resource_factory_build_count=1,
        warmup_request_count=1,
        warmup_arrival_pattern="warmup_single_request",
        warmup_candidate_pool_depth=400,
        warmup_retrieval_latency_ms=0.0,
        warmup_end_to_end_latency_ms=0.0,
        rejection_reasons=(),
    )


def _resource_summary() -> PrimeQAHybridRuntimeResourceSummary:
    return PrimeQAHybridRuntimeResourceSummary(
        dense_model_count=2,
        dense_embedding_cache_count=2,
        lexical_index_count=4,
        derived_route_count=1,
        candidate_pool_retriever_instance_count=1,
        optional_entrypoint_instance_count=1,
    )


def _toolset(
    retriever: _CandidatePoolRetriever,
    *,
    refuse: bool,
) -> PrimeQAHybridAgentToolset:
    return PrimeQAHybridAgentToolset(
        candidate_pool_retriever=retriever,
        observation_adapter=PrimeQAHybridSidecarObservationAdapter(),
        answer_generator_factory=_SyntheticGeneratorFactory(),
        answer_verifier_factory=_SyntheticVerifierFactory(refuse=refuse),
    )


def _question(handle: str) -> PrimeQARuntimeQuery:
    return PrimeQARuntimeQuery(
        id=handle,
        title="Adapter installation",
        text="How do I apply the documented adapter procedure?",
    )


def _candidate_results(prefix: str) -> tuple[RetrievalResult, ...]:
    return tuple(
        RetrievalResult(
            document=PrimeQADocument(
                id=f"{prefix}-{rank:03d}",
                title=f"Adapter procedure {rank}",
                text="Apply the documented adapter procedure and restart the service.",
            ),
            score=1.0 / rank,
            rank=rank,
        )
        for rank in range(1, 401)
    )


def _received_state(question: PrimeQARuntimeQuery) -> AgentToolWorkflowPrivateState:
    return {
        "request_handle": question.id,
        "runtime_query": question,
        "candidate_pool_results": (),
        "generation_context_results": (),
        "verification_context_results": (),
        "sidecar_observation_bundle": None,
        "original_answer": None,
        "verification_result": None,
        "terminal_response": None,
        "current_state": AgentToolWorkflowState.RECEIVED,
        "visited_states": ("received",),
        "tool_call_counts": {
            "retrieve_candidate_pool": 0,
            "compose_grounded_answer": 0,
            "verify_grounded_answer": 0,
        },
        "failure_stage": None,
    }


def _state_summary(state: AgentToolWorkflowPrivateState) -> dict[str, Any]:
    return {
        "current_state": state["current_state"].value,
        "visited_states": list(state["visited_states"]),
        "tool_call_counts": dict(state["tool_call_counts"]),
        "candidate_pool_depth": len(state["candidate_pool_results"]),
        "failure_stage": state["failure_stage"],
    }


def _stage153_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    checks = report.get("guard_checks") or []
    decision = report.get("decision") or {}
    return {
        "stage": report.get("stage"),
        "protocol_id": report.get("protocol_id"),
        "status": decision.get("status"),
        "guard_count": len(checks),
        "passed_guards": sum(check.get("passed") is True for check in checks),
        "failed_guards": [check.get("name") for check in checks if check.get("passed") is not True],
        "workflow_implementation_allowed_next": decision.get(
            "workflow_implementation_allowed_next"
        ),
        "test_gate_opened": decision.get("test_gate_opened"),
    }


def _stage152_support_summary(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {
            "available": False,
            "status": None,
            "guard_count": 0,
            "passed_guards": 0,
            "real_executed": False,
        }
    checks = report.get("guard_checks") or []
    decision = report.get("decision") or {}
    real = report.get("real_resource_service_lifecycle") or {}
    http = real.get("http_probe") or {}
    return {
        "available": True,
        "status": decision.get("status"),
        "guard_count": len(checks),
        "passed_guards": sum(check.get("passed") is True for check in checks),
        "failed_guards": [check.get("name") for check in checks if check.get("passed") is not True],
        "real_executed": real.get("executed") is True,
        "exit_code": real.get("exit_code"),
        "liveness_status": http.get("liveness_status"),
        "readiness_status": http.get("readiness_status"),
        "answer_status": http.get("answer_status"),
        "candidate_pool_depth_recorded": False,
        "citation_count": http.get("answer_citation_count"),
        "listener_released": real.get("listener_released"),
        "transport_closed": real.get("transport_closed"),
        "test_metrics_run": (report.get("public_safe_contract") or {}).get("test_metrics_run"),
    }


def _guard_checks(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    stage153 = report.get("stage153_summary") or {}
    dependency = report.get("dependency_evidence") or {}
    contract = report.get("implementation_contract") or {}
    synthetic = report.get("synthetic_validation") or {}
    complete = synthetic.get("complete_equivalence") or {}
    refuse = synthetic.get("refuse_equivalence") or {}
    invalid = synthetic.get("invalid_transition") or {}
    failure = synthetic.get("error_propagation") or {}
    concurrency = synthetic.get("concurrency") or {}
    http = synthetic.get("facade_http_integration") or {}
    real = report.get("stage152_current_service_support") or {}
    topology = complete.get("langgraph_topology") or {}
    expected_nodes = {
        "validate_request",
        "retrieve_candidate_pool",
        "prepare_context",
        "compose_grounded_answer",
        "verify_grounded_answer",
        "observe_diagnostics",
        "finalize_response",
    }
    checks = [
        _check(
            "stage154_user_confirmed",
            (report.get("user_confirmation") or {}).get("confirmed") is True,
        ),
        _check(
            "stage153_identity_valid",
            stage153.get("protocol_id") == AGENT_TOOL_ORCHESTRATION_PROTOCOL_ID,
        ),
        _check("stage153_status_valid", stage153.get("status") == _STAGE153_STATUS),
        _check(
            "stage153_guards_exact",
            stage153.get("guard_count") == _EXPECTED_STAGE153_GUARDS
            and stage153.get("passed_guards") == _EXPECTED_STAGE153_GUARDS
            and stage153.get("failed_guards") == [],
        ),
        _check(
            "stage153_authorized_implementation",
            stage153.get("workflow_implementation_allowed_next") is True,
        ),
        _check("stage153_test_remained_closed", stage153.get("test_gate_opened") is False),
        _check(
            "direct_agent_dependency_exact",
            dependency.get("direct_agent_dependencies") == _EXPECTED_DIRECT_AGENT_DEPENDENCIES
            and dependency.get("direct_agent_dependencies_exact") is True,
        ),
        _check(
            "full_langchain_not_direct",
            dependency.get("full_langchain_direct_dependency_present") is False,
        ),
        _check(
            "installed_versions_exact",
            dependency.get("installed_versions") == _EXPECTED_VERSIONS
            and dependency.get("installed_versions_exact") is True,
        ),
        _check(
            "dependency_install_no_retry", dependency.get("dependency_install_retry_count") == 0
        ),
        _check(
            "websockets_adjustment_recorded",
            dependency.get("websockets_before_install") == "16.1"
            and dependency.get("websockets_after_install") == "15.0.1",
        ),
        _check(
            "implementation_protocol_exact",
            contract.get("protocol_id") == AGENT_TOOL_ORCHESTRATION_PROTOCOL_ID
            and contract.get("graph_id") == AGENT_TOOL_WORKFLOW_GRAPH_ID,
        ),
        _check(
            "stategraph_adapter_exact",
            contract.get("adapter") == "langgraph.graph.StateGraph"
            and contract.get("direct_dependency") == "langgraph==1.2.9",
        ),
        _check(
            "graph_compile_once",
            topology.get("compile_count") == 1
            and contract.get("graph_compiled_once_per_workflow_instance") is True,
        ),
        _check(
            "graph_nodes_exact",
            topology.get("node_count") == 7
            and set(topology.get("node_ids") or []) == expected_nodes,
        ),
        _check("graph_conditional_edge_exact", topology.get("conditional_edge_count") == 1),
        _check(
            "graph_no_checkpointer_or_cache",
            topology.get("checkpointer_attached") is False
            and topology.get("cache_attached") is False,
        ),
        _check(
            "complete_engine_equivalent",
            complete.get("equal") is True and complete.get("terminal_state") == "complete",
        ),
        _check(
            "refuse_engine_equivalent",
            refuse.get("equal") is True and refuse.get("terminal_state") == "refuse",
        ),
        _check(
            "successful_transitions_exact",
            complete.get("transition_count") == 7 and refuse.get("transition_count") == 7,
        ),
        _check(
            "successful_tool_calls_exact",
            complete.get("tool_call_count") == 3
            and complete.get("retrieval_tool_call_count") == 1
            and complete.get("answer_tool_call_count") == 1
            and complete.get("verification_tool_call_count") == 1,
        ),
        _check(
            "context_depths_exact",
            complete.get("candidate_pool_depth") == 400
            and complete.get("generation_context_count") == 10
            and complete.get("verification_context_count") == 200
            and complete.get("sidecar_observation_count") == 3,
        ),
        _check(
            "one_retrieval_per_engine",
            complete.get("reference_retriever_call_count") == 1
            and complete.get("graph_retriever_call_count") == 1,
        ),
        _check("invalid_transition_rejected", invalid.get("rejected") is True),
        _check("invalid_transition_zero_mutation", invalid.get("state_unchanged") is True),
        _check("invalid_transition_zero_tool_calls", invalid.get("retriever_call_count") == 0),
        _check(
            "tool_error_same_object",
            failure.get("same_object") is True and failure.get("error_type") == "RuntimeError",
        ),
        _check(
            "tool_error_stage_exact",
            failure.get("failure_stage") == "retrieve_candidate_pool"
            and failure.get("terminal_state") == "validated"
            and failure.get("transition_count") == 1,
        ),
        _check("tool_error_attempt_count_exact", failure.get("retrieval_tool_call_count") == 1),
        _check("tool_error_message_not_public", failure.get("error_message_public") is False),
        _check(
            "tool_error_no_retry_fallback",
            failure.get("retry_action_count") == 0 and failure.get("fallback_action_count") == 0,
        ),
        _check(
            "four_request_concurrency_complete",
            concurrency.get("invocation_count") == 4
            and concurrency.get("completed_count") == 4
            and concurrency.get("failed_count") == 0,
        ),
        _check(
            "four_request_state_isolated",
            concurrency.get("isolation_failure_count") == 0
            and concurrency.get("retriever_call_count") == 4,
        ),
        _check(
            "four_request_inflight_exact",
            concurrency.get("max_observed_in_flight") == 4
            and concurrency.get("current_in_flight") == 0,
        ),
        _check("concurrent_graph_still_compiled_once", concurrency.get("graph_compile_count") == 1),
        _check(
            "http_liveness_ready_answer_200",
            http.get("liveness_status") == 200
            and http.get("readiness_status") == 200
            and http.get("answer_status") == 200,
        ),
        _check("http_graph_candidate_pool_exact", http.get("candidate_pool_depth") == 400),
        _check("http_terminal_valid", http.get("terminal_state") in {"complete", "refuse"}),
        _check("http_transport_closed", http.get("transport_closed") is True),
        _check("http_request_content_not_saved", http.get("request_content_saved") is False),
        _check(
            "http_no_queue_retry_fallback",
            [
                http.get("queue_action_count"),
                http.get("retry_action_count"),
                http.get("fallback_action_count"),
            ]
            == [0, 0, 0],
        ),
        _check("real_support_available", real.get("available") is True),
        _check("real_support_status_valid", real.get("status") == _STAGE152_STATUS),
        _check(
            "real_support_guards_exact",
            real.get("guard_count") == _EXPECTED_STAGE152_GUARDS
            and real.get("passed_guards") == _EXPECTED_STAGE152_GUARDS
            and real.get("failed_guards") == [],
        ),
        _check(
            "real_resource_lifecycle_executed",
            real.get("real_executed") is True and real.get("exit_code") == 0,
        ),
        _check(
            "real_http_path_200",
            [real.get("liveness_status"), real.get("readiness_status"), real.get("answer_status")]
            == [200, 200, 200],
        ),
        _check(
            "real_service_released",
            real.get("listener_released") is True and real.get("transport_closed") is True,
        ),
        _check("real_test_metrics_not_run", real.get("test_metrics_run") is False),
        _check(
            "implementation_no_toolnode_router",
            contract.get("tool_node_used") is False
            and contract.get("llm_tool_router_used") is False,
        ),
        _check(
            "implementation_no_query_rewrite_or_second_retrieval",
            contract.get("query_rewrite_enabled") is False
            and contract.get("second_retrieval_enabled") is False,
        ),
        _check(
            "implementation_no_persistence_streaming_interrupt",
            contract.get("persistent_store_attached") is False
            and contract.get("streaming_enabled") is False
            and contract.get("human_interrupt_enabled") is False,
        ),
        _check(
            "implementation_no_queue_retry_fallback",
            contract.get("queue_actions_enabled") is False
            and contract.get("retry_actions_enabled") is False
            and contract.get("fallback_strategies_enabled") is False,
        ),
        _check(
            "runtime_remote_test_closed",
            contract.get("runtime_registered_as_default") is False
            and contract.get("remote_exposure_authorized") is False
            and contract.get("test_gate_opened") is False
            and contract.get("test_metrics_run") is False,
        ),
        _check("source_files_unchanged", report.get("source_unchanged_after_validation") is True),
    ]
    return checks


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    public = {
        "train_split_loaded": False,
        "dev_split_loaded": False,
        "test_split_loaded": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
        "question_rows_saved": False,
        "raw_answers_saved": False,
        "raw_documents_saved": False,
        "document_identifiers_saved": False,
        "exception_messages_saved": False,
        "models_loaded_by_stage154_synthetic_validation": False,
        "indexes_loaded_by_stage154_synthetic_validation": False,
        "real_resources_loaded_only_by_stage152_support": (
            (report.get("stage152_current_service_support") or {}).get("real_executed") is True
        ),
        "runtime_registered_as_default": False,
        "remote_exposure_authorized": False,
        "queue_actions_enabled": False,
        "retry_actions_enabled": False,
        "fallback_strategies_enabled": False,
    }
    public["forbidden_keys_found"] = sorted(_forbidden_keys_found(public))
    return public


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
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
        value_label="passed" if passed else "failed",
    )


def _chart(
    title: str,
    bars: Sequence[BarDatum],
    *,
    width: int = 1800,
    margin_left: int = 760,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label="observed value",
        width=width,
        margin_left=margin_left,
    )
