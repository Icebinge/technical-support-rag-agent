from __future__ import annotations

import hashlib
import http.client
import json
import socket
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any

import uvicorn

from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_http_transport import (
    BOUNDED_DYNAMIC_AGENT_BINDING_HOST,
    BoundedDynamicAgentHttpLogSink,
    PublicSafeBoundedDynamicAgentHttpEvent,
    create_primeqa_hybrid_bounded_dynamic_agent_http_app,
    create_primeqa_hybrid_bounded_dynamic_agent_uvicorn_config,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_runtime import (
    BoundedDynamicAgentRuntimeRun,
    PrimeQAHybridBoundedDynamicAgentRuntime,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_service_entrypoint import (
    CanonicalBoundedDynamicAgentServicePaths,
    PrimeQAHybridBoundedDynamicAgentServiceEntrypoint,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    _forbidden_keys_found,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.domain.dataset import PrimeQAQuery
from ts_rag_agent.domain.retrieval import RetrievalResult

from .primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_protocol import (
    STAGE160_DEV_SPLIT_FILENAME,
    STAGE160_DIAGNOSTIC_FOLD_COUNT,
    STAGE160_EXPECTED_DEV_SHA256,
    STAGE160_EXPECTED_GROUPING_SHA256,
    STAGE160_EXPECTED_ORDER_SHA256,
    Stage160CaseObservation,
    Stage160DiagnosticSample,
    Stage160FoldAssignment,
    Stage160WorkloadPlan,
    build_stage160_grouped_fold_assignment,
    build_stage160_workload_plan,
    canonical_json_sha256,
    load_stage160_dev_diagnostic_samples,
    query_digest,
    score_answer,
    stage160_private_report,
    summarize_stage160_observations,
)

STAGE160_EXPECTED_STAGE159_ARTIFACT_SHA256 = (
    "93eb319aeb0c2212f55df0bbb2c2b1790eeba02aa4ec20439464bc72a7f3bfe6"
)
STAGE160_EXPECTED_STAGE159_GUARD_COUNT = 65
STAGE160_EXPECTED_STAGE159_STATUS = "primeqa_hybrid_bounded_dynamic_agent_warm_service_validated"
STAGE160_EXPECTED_STAGE159_COMPOSE_COUNT = 34
STAGE160_EXPECTED_STAGE159_REFUSAL_COUNT = 87

_STAGE = "Stage 160"
_CREATED_AT = "2026-07-18"
_ANALYSIS_ID = "primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_v1"
_FINAL_STATUS = "primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_completed"


@dataclass(frozen=True)
class Stage160RuntimeCapture:
    private_identity_sha256: str
    query_digest_sha256: str
    diagnostic_group_sha256: str
    gold_document_sha256: str | None
    question_route: str
    split_subtype: str
    answerable: bool
    selected_action: str
    terminal_state: str
    refused: bool
    candidate_pool_count: int
    generation_context_count: int
    verification_context_count: int
    gold_candidate_rank: int | None
    gold_generation_rank: int | None
    gold_verification_rank: int | None
    gold_cited: bool
    citation_count: int
    answer_token_f1: float | None
    top_candidate_score: float | None
    gold_candidate_score: float | None
    router_input_token_count: int
    router_output_token_count: int
    router_generation_latency_ms: float


@dataclass(frozen=True)
class Stage160DiagnosticValidationResult:
    public_report: dict[str, Any]
    private_report: dict[str, Any]


@dataclass(frozen=True)
class Stage160Visualization:
    name: str
    path: str


class Stage160DiagnosticRuntimeObserver:
    """Validation-only decorator capturing hashed gold/runtime diagnostics."""

    def __init__(
        self,
        runtime: PrimeQAHybridBoundedDynamicAgentRuntime,
        expected_samples: Sequence[Stage160DiagnosticSample],
    ) -> None:
        self._runtime = runtime
        self._expected_samples = tuple(expected_samples)
        self._lock = Lock()
        self._next_index = 0
        self._active = False
        self._captures: list[Stage160RuntimeCapture] = []

    @property
    def last_public_trace(self):
        return self._runtime.last_public_trace

    @property
    def captures(self) -> tuple[Stage160RuntimeCapture, ...]:
        with self._lock:
            return tuple(self._captures)

    def topology(self) -> dict[str, Any]:
        return self._runtime.topology()

    def open_thread(self, opaque_thread_handle: str):
        return self._runtime.open_thread(opaque_thread_handle)

    def close_thread(self, opaque_thread_handle: str):
        return self._runtime.close_thread(opaque_thread_handle)

    def thread_summary(self, opaque_thread_handle: str):
        return self._runtime.thread_summary(opaque_thread_handle)

    def run_turn(
        self,
        *,
        opaque_thread_handle: str,
        question: PrimeQAQuery,
    ) -> BoundedDynamicAgentRuntimeRun:
        with self._lock:
            if self._active:
                raise RuntimeError("Stage160 diagnostic observer permits one active turn")
            if self._next_index >= len(self._expected_samples):
                raise RuntimeError("Stage160 diagnostic observer received an extra turn")
            sample = self._expected_samples[self._next_index]
            if query_digest(question.title, question.text) != sample.query_digest_sha256:
                raise RuntimeError("Stage160 diagnostic observer query order mismatch")
            self._active = True
        try:
            run = self._runtime.run_turn(
                opaque_thread_handle=opaque_thread_handle,
                question=question,
            )
            capture = _capture_runtime_run(sample=sample, run=run)
        finally:
            with self._lock:
                self._active = False
        with self._lock:
            self._captures.append(capture)
            self._next_index += 1
        return run


class RecordingStage160HttpLogSink(BoundedDynamicAgentHttpLogSink):
    def __init__(self) -> None:
        self._lock = Lock()
        self._events: list[PublicSafeBoundedDynamicAgentHttpEvent] = []

    def emit(self, event: PublicSafeBoundedDynamicAgentHttpEvent) -> None:
        with self._lock:
            self._events.append(event)

    def public_events(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(event.to_public_dict() for event in self._events)

    def successful_turn_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            event
            for event in self.public_events()
            if event.get("route_id") == "thread_turn" and event.get("http_status") == 200
        )


class _ObservedStage160UvicornServer(uvicorn.Server):
    def __init__(
        self,
        *,
        config: uvicorn.Config,
        lifecycle_events: Queue[tuple[str, BaseException | None]],
    ) -> None:
        super().__init__(config=config)
        self._lifecycle_events = lifecycle_events

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        self._lifecycle_events.put(("started", None))


def validate_primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics(
    *,
    settings: ProjectSettings,
    port: int,
    user_confirmed_dev_gold_diagnostics: bool,
    progress_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> Stage160DiagnosticValidationResult:
    """Run one full-dev Agent diagnostic with gold used only after runtime."""

    import torch

    if not user_confirmed_dev_gold_diagnostics:
        raise ValueError("Stage160 requires explicit dev-gold diagnostic confirmation")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage160 formal diagnostics require CUDA")
    if not _port_is_bindable(port):
        raise RuntimeError("selected Stage160 validation port is unavailable")

    started_at = time.perf_counter()
    project_root = Path(__file__).resolve().parents[3]
    artifact_dir = settings.artifact_dir.resolve()
    stage159_path = artifact_dir / "primeqa_hybrid_bounded_dynamic_agent_warm_service_stage159.json"
    dev_path = artifact_dir / "primeqa_hybrid_split_stage68_splits" / STAGE160_DEV_SPLIT_FILENAME
    current_sources = _current_source_paths(project_root)
    source_before = {key: _fingerprint(path) for key, path in current_sources.items()}
    stage159_authorization = _authorize_stage159_artifact(
        artifact_path=stage159_path,
        project_root=project_root,
    )
    authorized_at = time.perf_counter()
    _emit_progress(progress_sink, phase="source_authorized")

    diagnostic_set = load_stage160_dev_diagnostic_samples(dev_path)
    workload_plan = build_stage160_workload_plan(diagnostic_set)
    fold_assignment = build_stage160_grouped_fold_assignment(diagnostic_set.samples)
    dev_loaded_at = time.perf_counter()
    _emit_progress(
        progress_sink,
        phase="dev_diagnostic_workload_loaded",
        completed_turn_count=0,
        total_turn_count=len(diagnostic_set.samples),
        completed_thread_count=0,
        total_thread_count=len(workload_plan.threads),
    )

    torch.cuda.reset_peak_memory_stats()
    sink = RecordingStage160HttpLogSink()
    observer_holder: dict[str, Stage160DiagnosticRuntimeObserver] = {}

    def app_factory(**kwargs: Any):
        runtime = kwargs.pop("runtime")
        observer = Stage160DiagnosticRuntimeObserver(
            runtime,
            expected_samples=workload_plan.ordered_samples,
        )
        observer_holder["observer"] = observer
        return create_primeqa_hybrid_bounded_dynamic_agent_http_app(
            **kwargs,
            runtime=observer,
            log_sink=sink,
        )

    paths = CanonicalBoundedDynamicAgentServicePaths.from_settings(settings)
    entrypoint = PrimeQAHybridBoundedDynamicAgentServiceEntrypoint(
        settings=settings,
        paths=paths,
        app_factory=app_factory,
    )
    prepared = entrypoint.prepare()
    observer = observer_holder.get("observer")
    if observer is None:
        raise RuntimeError("Stage160 app composition did not expose the diagnostic observer")
    prepared_at = time.perf_counter()
    _emit_progress(progress_sink, phase="service_prepared")

    config = create_primeqa_hybrid_bounded_dynamic_agent_uvicorn_config(
        app=prepared.app,
        port=port,
    )
    lifecycle_events: Queue[tuple[str, BaseException | None]] = Queue()
    server = _ObservedStage160UvicornServer(
        config=config,
        lifecycle_events=lifecycle_events,
    )

    def run_server() -> None:
        try:
            server.run()
        except BaseException as error:
            lifecycle_events.put(("failed", error))
        finally:
            if not server.started:
                lifecycle_events.put(("exited_before_start", None))

    server_thread = Thread(
        target=run_server,
        name="stage160-real-loopback-diagnostic-service",
        daemon=False,
    )
    server_thread.start()
    lifecycle_state, lifecycle_error = lifecycle_events.get()
    if lifecycle_state != "started":
        server_thread.join()
        raise RuntimeError("Stage160 Uvicorn server exited before startup") from lifecycle_error
    server_started_at = time.perf_counter()
    _emit_progress(progress_sink, phase="server_started")

    connection = http.client.HTTPConnection(BOUNDED_DYNAMIC_AGENT_BINDING_HOST, port)
    try:
        live_status, live = _request_json(connection, "GET", "/health/live")
        ready_status, ready = _request_json(connection, "GET", "/health/ready")
        observations, dev_http = _run_full_dev_diagnostics(
            connection=connection,
            workload_plan=workload_plan,
            fold_assignment=fold_assignment,
            observer=observer,
            sink=sink,
            progress_sink=progress_sink,
        )
        dev_completed_at = time.perf_counter()
    finally:
        connection.close()
        server.should_exit = True
        server_thread.join()
    shutdown_completed_at = time.perf_counter()
    _emit_progress(progress_sink, phase="server_stopped")

    private_report = stage160_private_report(observations)
    private_sha256 = canonical_json_sha256(private_report)
    diagnostics = summarize_stage160_observations(observations)
    events = sink.public_events()
    transport = prepared.app.state.bounded_dynamic_agent_http_transport
    counters = transport.coordinator.counters()
    source_after = {key: _fingerprint(path) for key, path in current_sources.items()}
    stage159_after = _fingerprint(stage159_path)
    dev_after = _fingerprint(dev_path)
    port_released = _port_is_bindable(port)
    finished_at = time.perf_counter()

    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "One full frozen-dev replay of the nondefault bounded Agent service with "
            "dev gold joined only inside a validation observer for aggregate refusal, "
            "retrieval-layer, answer-quality, latency-tail, and grouped five-fold "
            "diagnostic stability analysis. No model or policy is selected or tuned."
        ),
        "user_confirmation": {
            "route": "A",
            "dev_gold_diagnostics_confirmed": True,
            "full_dev_replay_confirmed": True,
            "grouped_five_fold_stability_confirmed": True,
            "test_remains_locked": True,
        },
        "stage159_authorization": stage159_authorization,
        "source_files": source_before,
        "source_unchanged_after_validation": source_before == source_after,
        "stage159_artifact_unchanged_after_validation": (
            stage159_authorization["artifact"] == stage159_after
        ),
        "dev_source_unchanged_after_validation": (
            diagnostic_set.source_sha256 == dev_after["sha256"]
            and diagnostic_set.source_size_bytes == dev_after["size_bytes"]
        ),
        "dev_diagnostic_protocol": diagnostic_set.public_summary(),
        "workload_plan": workload_plan.public_summary(),
        "grouped_fold_protocol": fold_assignment.public_summary(),
        "environment": {
            "python_environment": "project_.venv",
            "torch_version": torch.__version__,
            "transformers_version": version("transformers"),
            "fastapi_version": version("fastapi"),
            "uvicorn_version": version("uvicorn"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
        },
        "startup": {
            "resource_factory_build_count": prepared.resource_factory_build_count,
            "retrieval_encoder_device": prepared.retrieval_encoder_device,
            "warmup": prepared.warmup.to_public_dict(),
            "model_generation_call_count": prepared.backend.generation_call_count,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "timing_seconds": dict(prepared.timing_seconds),
        },
        "real_service": {
            "binding_host": BOUNDED_DYNAMIC_AGENT_BINDING_HOST,
            "binding_port": port,
            "server_started": server.started,
            "server_thread_alive_after_shutdown": server_thread.is_alive(),
            "port_rebind_after_shutdown": port_released,
            "health_status": {"live": live_status, "ready": ready_status},
            "health_state": {
                "live": live.get("status"),
                "ready": ready.get("status"),
            },
            "dev_http": dev_http,
            "public_event_count": len(events),
            "public_event_route_counts": _counter_dict(
                str(event.get("route_id")) for event in events
            ),
            "coordinator_counters_after_shutdown": asdict(counters),
        },
        "aggregate_diagnostics": diagnostics,
        "private_diagnostic_artifact_contract": {
            "row_count": len(observations),
            "canonical_content_sha256": private_sha256,
            "git_policy": "ignored_local_artifact",
            "contains_raw_question": False,
            "contains_raw_answer": False,
            "contains_raw_document_id": False,
            "contains_raw_document_text": False,
            "contains_hashed_sample_identity": True,
            "public_report_contains_case_rows": False,
        },
        "closed_boundaries": {
            "train_split_loaded": False,
            "dev_split_loaded": True,
            "test_split_loaded": False,
            "test_metrics_run": False,
            "dev_gold_used_for_diagnosis": True,
            "dev_gold_projected_into_runtime": False,
            "dev_used_for_model_fit": False,
            "dev_used_for_policy_selection": False,
            "dev_used_for_threshold_tuning": False,
            "existing_answer_route_changed": False,
            "runtime_registered_as_default": False,
            "remote_exposure_authorized": False,
            "persistent_state_enabled": False,
            "query_rewrite_enabled": False,
            "second_retrieval_enabled": False,
            "queue_action_count": counters.queue_action_count,
            "retry_action_count": counters.retry_action_count,
            "fallback_action_count": counters.fallback_action_count,
        },
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "dev_gold_load_and_plan": round(dev_loaded_at - authorized_at, 6),
            "service_prepare": round(prepared_at - dev_loaded_at, 6),
            "server_start": round(server_started_at - prepared_at, 6),
            "full_dev_diagnostic_workload": round(dev_completed_at - server_started_at, 6),
            "shutdown": round(shutdown_completed_at - dev_completed_at, 6),
            "aggregate_and_final_audit": round(finished_at - shutdown_completed_at, 6),
            "total": round(finished_at - started_at, 6),
        },
    }
    report["guard_checks"] = _guard_checks(report)
    forbidden = sorted(_forbidden_keys_found(report))
    report["public_safe_contract"] = {
        "forbidden_keys_found": forbidden,
        "case_level_rows_saved_in_public_report": False,
        "raw_or_private_content_saved_in_public_report": False,
        "hashed_sample_identities_saved_in_public_report": False,
    }
    passed = all(check["passed"] for check in report["guard_checks"]) and not forbidden
    report["decision"] = _decision(report=report, passed=passed)
    return Stage160DiagnosticValidationResult(
        public_report=report,
        private_report=private_report,
    )


def write_stage160_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage160Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = report.get("aggregate_diagnostics") or {}
    overview = diagnostics.get("overview") or {}
    quality = diagnostics.get("quality_diagnostics") or {}
    refusal_flow = diagnostics.get("answerable_refusal_flow") or {}
    positions = diagnostics.get("by_turn_position") or {}
    actions = diagnostics.get("by_selected_action") or {}
    latency = diagnostics.get("latency_diagnostics") or {}
    folds = (diagnostics.get("fold_diagnostic_stability") or {}).get("folds") or {}
    correlations = latency.get("spearman_correlations_with_generation_latency") or {}
    checks = report.get("guard_checks") or []
    charts = {
        "stage160_failure_bucket_counts.svg": _chart(
            "Stage160 failure bucket counts",
            [
                _bar(label, value)
                for label, value in (diagnostics.get("failure_bucket_counts") or {}).items()
            ],
            "dev case count",
            margin_left=580,
        ),
        "stage160_answerable_refusal_flow.svg": _chart(
            "Stage160 answerable refusal flow",
            [
                _bar("answerable", refusal_flow.get("answerable_count", 0)),
                _bar("answerable refusals", refusal_flow.get("answerable_refusal_count", 0)),
                _bar(
                    "gold absent candidate pool",
                    refusal_flow.get("gold_absent_candidate_pool_refusal_count", 0),
                ),
                _bar(
                    "gold lost before Top10",
                    refusal_flow.get("gold_lost_before_generation_refusal_count", 0),
                ),
                _bar(
                    "gold visible model refusal",
                    refusal_flow.get("gold_visible_model_refusal_count", 0),
                ),
                _bar("post-compose refusal", refusal_flow.get("post_compose_refusal_count", 0)),
            ],
            "answerable dev count",
            margin_left=360,
        ),
        "stage160_quality_diagnostic_rates.svg": _chart(
            "Stage160 dev quality diagnostic rates",
            [_bar(label, value) for label, value in quality.items() if label.endswith("rate")],
            "rate",
            margin_left=560,
        ),
        "stage160_action_distribution.svg": _chart(
            "Stage160 action distribution",
            [
                _bar(label, value)
                for label, value in (overview.get("selected_action_counts") or {}).items()
            ],
            "dev turn count",
            margin_left=420,
        ),
        "stage160_turn_position_end_to_end_latency.svg": _position_chart(
            title="Stage160 end-to-end latency by turn position",
            positions=positions,
            metric="end_to_end_latency_ms",
        ),
        "stage160_turn_position_generation_latency.svg": _position_chart(
            title="Stage160 generation latency by turn position",
            positions=positions,
            metric="router_generation_latency_ms",
        ),
        "stage160_action_generation_latency.svg": _chart(
            "Stage160 generation latency by action",
            [
                _bar(label, (summary.get("router_generation_latency_ms") or {}).get("average", 0))
                for label, summary in actions.items()
            ],
            "average milliseconds",
            margin_left=440,
        ),
        "stage160_latency_correlations.svg": _chart(
            "Stage160 Spearman correlations with generation latency",
            [_bar(label, value) for label, value in correlations.items()],
            "Spearman rho",
            margin_left=460,
        ),
        "stage160_fold_answerable_refusal_rates.svg": _chart(
            "Stage160 grouped-fold answerable refusal rates",
            [
                _bar(f"fold {fold_id}", summary.get("answerable_refusal_rate", 0))
                for fold_id, summary in folds.items()
            ],
            "rate",
            margin_left=240,
        ),
        "stage160_guard_check_status.svg": _chart(
            "Stage160 guard checks",
            [_bar(str(check.get("name")), bool(check.get("passed"))) for check in checks],
            "1 means passed",
            margin_left=620,
        ),
    }
    visualizations = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        visualizations.append(Stage160Visualization(name=filename, path=str(path)))
    return tuple(visualizations)


def _capture_runtime_run(
    *,
    sample: Stage160DiagnosticSample,
    run: BoundedDynamicAgentRuntimeRun,
) -> Stage160RuntimeCapture:
    state = run.workflow_run.final_state
    candidate_pool = tuple(run.candidate_pool_results)
    generation = tuple(state["generation_context_results"])
    verification = tuple(state["verification_context_results"])
    answer = run.verified_answer
    metrics = run.workflow_run.router_metrics
    if metrics is None:
        raise RuntimeError("Stage160 runtime capture requires router metrics")
    gold_document_id = sample.gold_document_id
    gold_candidate = _first_document_result(candidate_pool, gold_document_id)
    return Stage160RuntimeCapture(
        private_identity_sha256=sample.private_identity_sha256,
        query_digest_sha256=sample.query_digest_sha256,
        diagnostic_group_sha256=sample.diagnostic_group_sha256,
        gold_document_sha256=sample.gold_document_sha256,
        question_route=sample.question_route,
        split_subtype=sample.split_subtype,
        answerable=sample.answerable,
        selected_action=str(run.public_safe_trace.selected_action),
        terminal_state=run.public_safe_trace.terminal_state,
        refused=answer.refused,
        candidate_pool_count=len(candidate_pool),
        generation_context_count=len(generation),
        verification_context_count=len(verification),
        gold_candidate_rank=_document_rank(candidate_pool, gold_document_id),
        gold_generation_rank=_document_rank(generation, gold_document_id),
        gold_verification_rank=_document_rank(verification, gold_document_id),
        gold_cited=(
            gold_document_id is not None
            and any(citation.document_id == gold_document_id for citation in answer.citations)
        ),
        citation_count=len(answer.citations),
        answer_token_f1=(
            score_answer(answer.answer, sample.gold_answer, refused=answer.refused)
            if sample.answerable
            else None
        ),
        top_candidate_score=(candidate_pool[0].score if candidate_pool else None),
        gold_candidate_score=(gold_candidate.score if gold_candidate else None),
        router_input_token_count=metrics.input_token_count,
        router_output_token_count=metrics.output_token_count,
        router_generation_latency_ms=metrics.generation_latency_ms,
    )


def _run_full_dev_diagnostics(
    *,
    connection: http.client.HTTPConnection,
    workload_plan: Stage160WorkloadPlan,
    fold_assignment: Stage160FoldAssignment,
    observer: Stage160DiagnosticRuntimeObserver,
    sink: RecordingStage160HttpLogSink,
    progress_sink: Callable[[Mapping[str, Any]], None] | None,
) -> tuple[tuple[Stage160CaseObservation, ...], dict[str, Any]]:
    observations: list[Stage160CaseObservation] = []
    open_statuses: list[int] = []
    turn_statuses: list[int] = []
    close_statuses: list[int] = []
    for thread in workload_plan.threads:
        handle = f"stage160-dev-thread-{thread.ordinal:03d}"
        open_status, _ = _request_json(
            connection,
            "POST",
            "/v1/bounded-agent/threads/open",
            {"thread_handle": handle},
        )
        open_statuses.append(open_status)
        if open_status != 201:
            raise RuntimeError("Stage160 dev thread open did not return 201")
        for turn_position, sample in enumerate(thread.samples, start=1):
            capture_cursor = len(observer.captures)
            event_cursor = len(sink.successful_turn_events())
            turn_started = time.perf_counter()
            status, response = _request_json(
                connection,
                "POST",
                "/v1/bounded-agent/threads/turn",
                {
                    "thread_handle": handle,
                    "title": sample.runtime_query.title,
                    "text": sample.runtime_query.text,
                },
            )
            end_to_end_latency_ms = round(
                (time.perf_counter() - turn_started) * 1000,
                3,
            )
            turn_statuses.append(status)
            if status != 200:
                raise RuntimeError("Stage160 dev turn did not return 200")
            captures = observer.captures
            events = sink.successful_turn_events()
            if len(captures) != capture_cursor + 1:
                raise RuntimeError("Stage160 dev turn did not produce one private capture")
            if len(events) != event_cursor + 1:
                raise RuntimeError("Stage160 dev turn did not produce one public event")
            capture = captures[-1]
            event = events[-1]
            if bool(response.get("refused")) != capture.refused:
                raise RuntimeError("Stage160 HTTP and runtime refusal states differ")
            if str(event.get("selected_action")) != capture.selected_action:
                raise RuntimeError("Stage160 HTTP and runtime selected actions differ")
            observations.append(
                Stage160CaseObservation(
                    private_identity_sha256=capture.private_identity_sha256,
                    query_digest_sha256=capture.query_digest_sha256,
                    diagnostic_group_sha256=capture.diagnostic_group_sha256,
                    gold_document_sha256=capture.gold_document_sha256,
                    fold_id=fold_assignment.fold_by_private_identity[
                        capture.private_identity_sha256
                    ],
                    thread_ordinal=thread.ordinal,
                    turn_position=turn_position,
                    question_route=capture.question_route,
                    split_subtype=capture.split_subtype,
                    answerable=capture.answerable,
                    selected_action=capture.selected_action,
                    terminal_state=capture.terminal_state,
                    refused=capture.refused,
                    candidate_pool_count=capture.candidate_pool_count,
                    generation_context_count=capture.generation_context_count,
                    verification_context_count=capture.verification_context_count,
                    gold_candidate_rank=capture.gold_candidate_rank,
                    gold_generation_rank=capture.gold_generation_rank,
                    gold_verification_rank=capture.gold_verification_rank,
                    gold_cited=capture.gold_cited,
                    citation_count=capture.citation_count,
                    answer_token_f1=capture.answer_token_f1,
                    top_candidate_score=capture.top_candidate_score,
                    gold_candidate_score=capture.gold_candidate_score,
                    router_input_token_count=capture.router_input_token_count,
                    router_output_token_count=capture.router_output_token_count,
                    router_generation_latency_ms=capture.router_generation_latency_ms,
                    end_to_end_latency_ms=end_to_end_latency_ms,
                    retained_state_bytes=int(response.get("retained_state_bytes") or 0),
                    completed_turn_count=int(response.get("completed_turn_count") or 0),
                )
            )
        close_status, _ = _request_json(
            connection,
            "POST",
            "/v1/bounded-agent/threads/close",
            {"thread_handle": handle},
        )
        close_statuses.append(close_status)
        if close_status != 200:
            raise RuntimeError("Stage160 dev thread close did not return 200")
        _emit_progress(
            progress_sink,
            phase="dev_diagnostic_thread_completed",
            completed_turn_count=len(observations),
            total_turn_count=len(workload_plan.ordered_samples),
            completed_thread_count=thread.ordinal,
            total_thread_count=len(workload_plan.threads),
        )
    return tuple(observations), {
        "open_request_count": len(open_statuses),
        "turn_request_count": len(turn_statuses),
        "close_request_count": len(close_statuses),
        "open_http_status_counts": _status_counts(open_statuses),
        "turn_http_status_counts": _status_counts(turn_statuses),
        "close_http_status_counts": _status_counts(close_statuses),
    }


def _guard_checks(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    confirmation = report.get("user_confirmation") or {}
    authorization = report.get("stage159_authorization") or {}
    dev = report.get("dev_diagnostic_protocol") or {}
    workload = report.get("workload_plan") or {}
    folds = report.get("grouped_fold_protocol") or {}
    startup = report.get("startup") or {}
    service = report.get("real_service") or {}
    dev_http = service.get("dev_http") or {}
    counters = service.get("coordinator_counters_after_shutdown") or {}
    diagnostics = report.get("aggregate_diagnostics") or {}
    overview = diagnostics.get("overview") or {}
    fold_stability = diagnostics.get("fold_diagnostic_stability") or {}
    private_contract = report.get("private_diagnostic_artifact_contract") or {}
    boundaries = report.get("closed_boundaries") or {}
    checks = [
        _check(
            "dev_gold_diagnostics_confirmed",
            confirmation.get("dev_gold_diagnostics_confirmed") is True,
        ),
        _check(
            "grouped_five_fold_confirmed",
            confirmation.get("grouped_five_fold_stability_confirmed") is True,
        ),
        _check("stage159_artifact_exact", authorization.get("artifact_identity_exact") is True),
        _check("stage159_guard_count_exact", authorization.get("guard_count") == 65),
        _check("stage159_guards_passed", authorization.get("all_guards_passed") is True),
        _check("stage159_sources_exact", authorization.get("source_fingerprint_match_count") == 4),
        _check("stage159_test_closed", authorization.get("test_gate_opened") is False),
        _check(
            "stage160_sources_unchanged", report.get("source_unchanged_after_validation") is True
        ),
        _check(
            "stage159_artifact_unchanged",
            report.get("stage159_artifact_unchanged_after_validation") is True,
        ),
        _check("dev_source_unchanged", report.get("dev_source_unchanged_after_validation") is True),
        _check("dev_source_hash_exact", dev.get("source_sha256") == STAGE160_EXPECTED_DEV_SHA256),
        _check("dev_row_count_exact", dev.get("dev_row_count") == 121),
        _check(
            "dev_answerability_counts_exact",
            [dev.get("answerable_count"), dev.get("unanswerable_count")] == [76, 45],
        ),
        _check("dev_order_exact", dev.get("stable_order_sha256") == STAGE160_EXPECTED_ORDER_SHA256),
        _check("dev_gold_diagnostic_only", dev.get("gold_fields_used_for_diagnosis") is True),
        _check("dev_gold_not_runtime", dev.get("gold_fields_projected_into_runtime") is False),
        _check(
            "dev_gold_not_selection", dev.get("gold_fields_used_for_selection_or_tuning") is False
        ),
        _check(
            "workload_grouping_exact",
            workload.get("grouping_sha256") == STAGE160_EXPECTED_GROUPING_SHA256,
        ),
        _check("workload_turn_count_exact", workload.get("turn_count") == 121),
        _check("workload_thread_count_exact", workload.get("thread_count") == 31),
        _check("fold_count_exact", folds.get("fold_count") == STAGE160_DIAGNOSTIC_FOLD_COUNT),
        _check("fold_rows_cover_dev", sum((folds.get("row_counts") or {}).values()) == 121),
        _check("fold_no_model_fit", folds.get("fit_models") is False),
        _check("fold_no_policy_selection", folds.get("select_policy") is False),
        _check("fold_no_threshold_tuning", folds.get("tune_thresholds") is False),
        _check("server_started", service.get("server_started") is True),
        _check("server_thread_joined", service.get("server_thread_alive_after_shutdown") is False),
        _check("port_released", service.get("port_rebind_after_shutdown") is True),
        _check("health_http_exact", service.get("health_status") == {"live": 200, "ready": 200}),
        _check("dev_open_http_exact", dev_http.get("open_http_status_counts") == {"201": 31}),
        _check("dev_turn_http_exact", dev_http.get("turn_http_status_counts") == {"200": 121}),
        _check("dev_close_http_exact", dev_http.get("close_http_status_counts") == {"200": 31}),
        _check("diagnostic_case_count_exact", overview.get("case_count") == 121),
        _check(
            "diagnostic_answerability_exact",
            [overview.get("answerable_count"), overview.get("unanswerable_count")] == [76, 45],
        ),
        _check(
            "stage159_action_distribution_reproduced",
            overview.get("selected_action_counts")
            == {"compose_grounded_answer": 34, "refuse_insufficient_evidence": 87},
        ),
        _check(
            "diagnostic_refusal_count_exact",
            overview.get("refusal_count") == STAGE160_EXPECTED_STAGE159_REFUSAL_COUNT,
        ),
        _check("fold_diagnostic_count_exact", fold_stability.get("fold_count") == 5),
        _check("fold_diagnostic_no_fit", fold_stability.get("fit_models") is False),
        _check("resource_factory_built_once", startup.get("resource_factory_build_count") == 1),
        _check("model_generation_count_exact", startup.get("model_generation_call_count") == 122),
        _check("gpu_memory_observed", startup.get("peak_gpu_memory_bytes", 0) > 0),
        _check("admitted_turn_count_exact", counters.get("admitted_turn_count") == 121),
        _check("completed_turn_count_exact", counters.get("completed_turn_count") == 121),
        _check("failed_turn_count_zero", counters.get("failed_turn_count") == 0),
        _check("opened_threads_zero", counters.get("opened_thread_count") == 0),
        _check("private_row_count_exact", private_contract.get("row_count") == 121),
        _check(
            "private_raw_question_closed", private_contract.get("contains_raw_question") is False
        ),
        _check("private_raw_answer_closed", private_contract.get("contains_raw_answer") is False),
        _check(
            "private_raw_document_closed",
            private_contract.get("contains_raw_document_id") is False
            and private_contract.get("contains_raw_document_text") is False,
        ),
        _check(
            "public_case_rows_closed",
            private_contract.get("public_report_contains_case_rows") is False,
        ),
        _check("test_split_closed", boundaries.get("test_split_loaded") is False),
        _check("test_metrics_closed", boundaries.get("test_metrics_run") is False),
        _check(
            "dev_no_fit_selection_tuning",
            [
                boundaries.get("dev_used_for_model_fit"),
                boundaries.get("dev_used_for_policy_selection"),
                boundaries.get("dev_used_for_threshold_tuning"),
            ]
            == [False, False, False],
        ),
        _check("runtime_nondefault", boundaries.get("runtime_registered_as_default") is False),
        _check(
            "remote_persistence_closed",
            boundaries.get("remote_exposure_authorized") is False
            and boundaries.get("persistent_state_enabled") is False,
        ),
        _check(
            "rewrite_second_retrieval_closed",
            boundaries.get("query_rewrite_enabled") is False
            and boundaries.get("second_retrieval_enabled") is False,
        ),
        _check(
            "no_queue_retry_fallback",
            [
                boundaries.get("queue_action_count"),
                boundaries.get("retry_action_count"),
                boundaries.get("fallback_action_count"),
            ]
            == [0, 0, 0],
        ),
    ]
    return checks


def _decision(*, report: Mapping[str, Any], passed: bool) -> dict[str, Any]:
    diagnostics = report.get("aggregate_diagnostics") or {}
    flow = diagnostics.get("answerable_refusal_flow") or {}
    latency = diagnostics.get("latency_diagnostics") or {}
    refusal_mechanisms = {
        "candidate_pool_miss": int(flow.get("gold_absent_candidate_pool_refusal_count") or 0),
        "generation_top10_loss": int(flow.get("gold_lost_before_generation_refusal_count") or 0),
        "gold_visible_model_refusal": int(flow.get("gold_visible_model_refusal_count") or 0),
        "post_compose_refusal": int(flow.get("post_compose_refusal_count") or 0),
    }
    dominant = max(refusal_mechanisms, key=refusal_mechanisms.get)
    correlations = latency.get("spearman_correlations_with_generation_latency") or {}
    dominant_latency_feature = (
        max(correlations, key=lambda key: abs(float(correlations[key]))) if correlations else "none"
    )
    failed = [
        str(check.get("name"))
        for check in report.get("guard_checks") or []
        if check.get("passed") is not True
    ]
    return {
        "status": _FINAL_STATUS if passed else "stage160_failure_diagnostics_rejected",
        "all_guards_passed": passed,
        "failed_checks": failed,
        "dominant_answerable_refusal_mechanism": dominant,
        "answerable_refusal_mechanism_counts": refusal_mechanisms,
        "dominant_observed_latency_correlation_feature": dominant_latency_feature,
        "generation_share_of_total_average": latency.get("generation_share_of_total_average"),
        "diagnostic_only": True,
        "policy_selected": False,
        "threshold_tuned": False,
        "runtime_registered_as_default": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
        "next_direction": "design_train_cv_router_or_context_intervention_from_stage160_mechanisms",
    }


def _authorize_stage159_artifact(
    *,
    artifact_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    artifact = _fingerprint(artifact_path)
    if artifact["sha256"] != STAGE160_EXPECTED_STAGE159_ARTIFACT_SHA256:
        raise RuntimeError("Stage160 rejected the Stage159 artifact identity")
    report = _load_json_object(artifact_path)
    checks = report.get("guard_checks") or []
    if len(checks) != STAGE160_EXPECTED_STAGE159_GUARD_COUNT or not all(
        isinstance(check, Mapping) and check.get("passed") is True for check in checks
    ):
        raise RuntimeError("Stage160 rejected the Stage159 guard evidence")
    decision = report.get("decision") or {}
    if not (
        decision.get("status") == STAGE160_EXPECTED_STAGE159_STATUS
        and decision.get("all_guards_passed") is True
        and decision.get("runtime_registered_as_default") is False
        and decision.get("test_gate_opened") is False
        and decision.get("test_metrics_run") is False
    ):
        raise RuntimeError("Stage160 rejected the Stage159 decision boundary")
    expected_sources = report.get("source_files") or {}
    matched = 0
    for key, path in _stage159_source_paths(project_root).items():
        expected = expected_sources.get(key)
        current = _fingerprint(path)
        if not isinstance(expected, Mapping) or current != dict(expected):
            raise RuntimeError(f"Stage160 rejected changed Stage159 source: {key}")
        matched += 1
    return {
        "artifact": artifact,
        "artifact_identity_exact": True,
        "guard_count": len(checks),
        "all_guards_passed": True,
        "decision_status": decision.get("status"),
        "source_fingerprint_match_count": matched,
        "runtime_registered_as_default": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
    }


def _stage159_source_paths(project_root: Path) -> dict[str, Path]:
    application = project_root / "src" / "ts_rag_agent" / "application"
    return {
        "warm_service_protocol": application
        / "primeqa_hybrid_bounded_dynamic_agent_warm_service_protocol.py",
        "warm_service_validation": application
        / "primeqa_hybrid_bounded_dynamic_agent_warm_service_validation.py",
        "validation_cli": project_root
        / "scripts"
        / "validate_primeqa_hybrid_bounded_dynamic_agent_warm_service.py",
        "pyproject": project_root / "pyproject.toml",
    }


def _current_source_paths(project_root: Path) -> dict[str, Path]:
    application = project_root / "src" / "ts_rag_agent" / "application"
    return {
        "failure_diagnostics_protocol": application
        / "primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_protocol.py",
        "failure_diagnostics_validation": application
        / "primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_validation.py",
        "validation_cli": project_root
        / "scripts"
        / "analyze_primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics.py",
        "pyproject": project_root / "pyproject.toml",
    }


def _request_json(
    connection: http.client.HTTPConnection,
    method: str,
    path: str,
    payload: Mapping[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    decoded = json.loads(raw.decode("utf-8")) if raw else {}
    if not isinstance(decoded, dict):
        raise RuntimeError("Stage160 loopback response is not a JSON object")
    return response.status, decoded


def _first_document_result(
    results: Sequence[RetrievalResult],
    document_id: str | None,
) -> RetrievalResult | None:
    if document_id is None:
        return None
    return next((item for item in results if item.document.id == document_id), None)


def _document_rank(
    results: Sequence[RetrievalResult],
    document_id: str | None,
) -> int | None:
    result = _first_document_result(results, document_id)
    return result.rank if result is not None else None


def _status_counts(statuses: Sequence[int]) -> dict[str, int]:
    return _counter_dict(str(status) for status in statuses)


def _counter_dict(values) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _port_is_bindable(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((BOUNDED_DYNAMIC_AGENT_BINDING_HOST, port))
        except OSError:
            return False
    return True


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(f"Unable to read Stage160 source artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Stage160 source artifact is not an object: {path.name}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size_bytes": resolved.stat().st_size, "sha256": digest.hexdigest()}


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit_progress(
    sink: Callable[[Mapping[str, Any]], None] | None,
    *,
    phase: str,
    **values: Any,
) -> None:
    if sink is not None:
        sink({"stage": _STAGE, "phase": phase, **values})


def _bar(label: str, value: int | float | bool) -> BarDatum:
    number = float(value)
    value_label = str(value) if isinstance(value, int | bool) else f"{number:.4f}"
    return BarDatum(label=label, value=number, value_label=value_label)


def _chart(
    title: str,
    bars: Sequence[BarDatum],
    x_label: str,
    *,
    margin_left: int,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        margin_left=margin_left,
        width=1480,
    )


def _position_chart(
    *,
    title: str,
    positions: Mapping[str, Any],
    metric: str,
) -> str:
    return _chart(
        title,
        [
            _bar(
                f"turn {position}",
                ((positions.get(str(position)) or {}).get(metric) or {}).get("average", 0),
            )
            for position in range(1, 5)
        ],
        "average milliseconds",
        margin_left=240,
    )
