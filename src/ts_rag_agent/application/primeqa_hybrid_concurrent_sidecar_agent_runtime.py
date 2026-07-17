from __future__ import annotations

import time
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from enum import Enum
from threading import BoundedSemaphore, Lock
from typing import Any, Protocol

from ts_rag_agent.application.primeqa_hybrid_nondefault_runtime_activation_protocol import (
    RuntimeActivationState,
)
from ts_rag_agent.application.primeqa_hybrid_online_candidate_pool_retriever import (
    CandidatePoolRetrievalRun,
    PrimeQAHybridOnlineCandidatePoolRetriever,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_entrypoint import (
    OptionalSidecarAgentEntrypointRun,
    PrimeQAHybridOptionalSidecarAgentEntrypoint,
    create_primeqa_hybrid_optional_sidecar_agent_entrypoint,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    PrimeQAHybridRuntimeResourceSummary,
    PrimeQAHybridSharedRuntimeResources,
    _forbidden_keys_found,
)
from ts_rag_agent.domain.dataset import PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult

_RUNTIME_MODE = "optional_sidecar_agent_concurrent_four_request"
_SLO_PROFILE_ID = "strict_practical_b_concurrency4_v1"
_MAX_IN_FLIGHT = 4
_END_TO_END_P99_LIMIT_SECONDS = 1.5
_ALLOWED_REQUEST_TRACE_FIELDS = frozenset(
    {
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
)


class ConcurrentArrivalPattern(str, Enum):
    """Frozen Stage144 arrival-pattern labels accepted by the research runtime."""

    WARMUP = "warmup_single_request"
    SYNCHRONIZED = "synchronized_four_request_burst"
    DETERMINISTIC_JITTER = "deterministic_jitter_0_to_20ms"
    OVERLOAD_PROBE = "five_request_overload_probe"


class AdmissionProbePort(Protocol):
    """Validation-only hook invoked after admission and before downstream work."""

    def on_admitted(self, in_flight_at_admission: int) -> None: ...


@dataclass(frozen=True)
class PublicSafeConcurrentRuntimeRequestTrace:
    """Stage144 allowlisted public trace for one concurrent runtime attempt."""

    runtime_mode: str
    activation_requested: bool
    activation_state: str
    slo_profile_id: str
    warm_resources_ready: bool
    concurrency_limit: int
    in_flight_at_admission: int
    admission_state: str
    arrival_pattern: str
    candidate_pool_depth: int
    retrieval_latency_ms: float
    end_to_end_latency_ms: float
    latency_budget_passed: bool
    terminal_state: str

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if set(payload) != _ALLOWED_REQUEST_TRACE_FIELDS:
            raise ValueError("concurrent runtime trace fields do not match Stage144 allowlist")
        forbidden = sorted(_forbidden_keys_found(payload))
        if forbidden:
            raise ValueError(f"concurrent runtime trace contains forbidden keys: {forbidden}")
        return payload


class PrimeQAHybridConcurrentCapacityExceededError(RuntimeError):
    """Typed nonblocking admission rejection carrying only a public-safe trace."""

    def __init__(self, trace: PublicSafeConcurrentRuntimeRequestTrace) -> None:
        super().__init__("concurrent runtime capacity four is already in use")
        self.public_safe_trace = trace


@dataclass(frozen=True)
class ConcurrentSidecarAgentRuntimeRun:
    """Private Agent result paired with the Stage144 public runtime trace."""

    entrypoint_run: OptionalSidecarAgentEntrypointRun
    public_safe_trace: PublicSafeConcurrentRuntimeRequestTrace

    @property
    def candidate_pool_results(self) -> tuple[RetrievalResult, ...]:
        return self.entrypoint_run.candidate_pool_results

    @property
    def verified_answer(self):
        return self.entrypoint_run.verified_answer


@dataclass(frozen=True)
class ConcurrentRuntimeCounters:
    """Aggregate runtime counters read atomically for validation boundaries."""

    admission_attempt_count: int
    admitted_request_count: int
    capacity_rejected_request_count: int
    downstream_request_count: int
    completed_request_count: int
    failed_request_count: int
    current_in_flight: int
    max_observed_in_flight: int


class RequestLocalProfiledCandidatePoolRetriever:
    """Share the immutable retrieval graph while isolating each request profile."""

    def __init__(self, delegate: PrimeQAHybridOnlineCandidatePoolRetriever) -> None:
        self._delegate = delegate
        self._pending_run: ContextVar[CandidatePoolRetrievalRun | None] = ContextVar(
            f"primeqa_hybrid_pending_retrieval_run_{id(self)}",
            default=None,
        )

    def prepare_request(self) -> None:
        if self._pending_run.get() is not None:
            raise RuntimeError("previous request-local retrieval profile was not consumed")

    def retrieve(self, question: PrimeQAQuestion) -> Sequence[RetrievalResult]:
        if self._pending_run.get() is not None:
            raise RuntimeError("one retrieval profile is allowed per request context")
        run = self._delegate.retrieve_profiled(question)
        self._pending_run.set(run)
        return run.results

    def take_run(self) -> CandidatePoolRetrievalRun:
        run = self._pending_run.get()
        if run is None:
            raise RuntimeError("entrypoint completed without a request-local retrieval profile")
        self._pending_run.set(None)
        return run

    def discard_run(self) -> None:
        self._pending_run.set(None)


@dataclass(frozen=True)
class PrimeQAHybridConcurrentRuntimeResourceBundle:
    """One shared entrypoint over process-owned retrieval resources."""

    profiled_retriever: RequestLocalProfiledCandidatePoolRetriever
    entrypoint: PrimeQAHybridOptionalSidecarAgentEntrypoint
    summary: PrimeQAHybridRuntimeResourceSummary


class PrimeQAHybridConcurrentSidecarAgentRuntime:
    """Explicit research runtime with four nonblocking in-flight permits."""

    def __init__(
        self,
        *,
        resources: PrimeQAHybridConcurrentRuntimeResourceBundle,
    ) -> None:
        self._resources = resources
        self._admission = BoundedSemaphore(_MAX_IN_FLIGHT)
        self._counter_lock = Lock()
        self._admission_attempt_count = 0
        self._admitted_request_count = 0
        self._capacity_rejected_request_count = 0
        self._downstream_request_count = 0
        self._completed_request_count = 0
        self._failed_request_count = 0
        self._current_in_flight = 0
        self._max_observed_in_flight = 0

    @property
    def resource_summary(self) -> PrimeQAHybridRuntimeResourceSummary:
        return self._resources.summary

    def counters(self) -> ConcurrentRuntimeCounters:
        with self._counter_lock:
            return ConcurrentRuntimeCounters(
                admission_attempt_count=self._admission_attempt_count,
                admitted_request_count=self._admitted_request_count,
                capacity_rejected_request_count=self._capacity_rejected_request_count,
                downstream_request_count=self._downstream_request_count,
                completed_request_count=self._completed_request_count,
                failed_request_count=self._failed_request_count,
                current_in_flight=self._current_in_flight,
                max_observed_in_flight=self._max_observed_in_flight,
            )

    def run(
        self,
        question: PrimeQAQuestion,
        *,
        arrival_pattern: ConcurrentArrivalPattern,
        admission_probe: AdmissionProbePort | None = None,
    ) -> ConcurrentSidecarAgentRuntimeRun:
        started_at = time.perf_counter()
        with self._counter_lock:
            self._admission_attempt_count += 1
        if not self._admission.acquire(blocking=False):
            with self._counter_lock:
                self._capacity_rejected_request_count += 1
                in_flight = self._current_in_flight
            finished_at = time.perf_counter()
            trace = _capacity_rejection_trace(
                arrival_pattern=arrival_pattern,
                in_flight=in_flight,
                duration_seconds=finished_at - started_at,
            )
            trace.to_public_dict()
            raise PrimeQAHybridConcurrentCapacityExceededError(trace)

        with self._counter_lock:
            self._admitted_request_count += 1
            self._current_in_flight += 1
            in_flight_at_admission = self._current_in_flight
            self._max_observed_in_flight = max(
                self._max_observed_in_flight,
                self._current_in_flight,
            )

        try:
            if admission_probe is not None:
                admission_probe.on_admitted(in_flight_at_admission)
            with self._counter_lock:
                self._downstream_request_count += 1
            self._resources.profiled_retriever.prepare_request()
            try:
                entrypoint_run = self._resources.entrypoint.run(question)
                retrieval_run = self._resources.profiled_retriever.take_run()
            except Exception:
                self._resources.profiled_retriever.discard_run()
                raise
            finished_at = time.perf_counter()
            end_to_end_seconds = finished_at - started_at
            trace = PublicSafeConcurrentRuntimeRequestTrace(
                runtime_mode=_RUNTIME_MODE,
                activation_requested=True,
                activation_state=RuntimeActivationState.ELIGIBLE.value,
                slo_profile_id=_SLO_PROFILE_ID,
                warm_resources_ready=True,
                concurrency_limit=_MAX_IN_FLIGHT,
                in_flight_at_admission=in_flight_at_admission,
                admission_state="admitted",
                arrival_pattern=arrival_pattern.value,
                candidate_pool_depth=len(retrieval_run.results),
                retrieval_latency_ms=round(retrieval_run.profile.total_seconds * 1000, 3),
                end_to_end_latency_ms=round(end_to_end_seconds * 1000, 3),
                latency_budget_passed=(end_to_end_seconds <= _END_TO_END_P99_LIMIT_SECONDS),
                terminal_state=entrypoint_run.public_safe_trace.terminal_state,
            )
            trace.to_public_dict()
            with self._counter_lock:
                self._completed_request_count += 1
            return ConcurrentSidecarAgentRuntimeRun(
                entrypoint_run=entrypoint_run,
                public_safe_trace=trace,
            )
        except Exception:
            with self._counter_lock:
                self._failed_request_count += 1
            raise
        finally:
            with self._counter_lock:
                self._current_in_flight -= 1
            self._admission.release()


def create_primeqa_hybrid_concurrent_sidecar_agent_runtime(
    *,
    shared_resources: PrimeQAHybridSharedRuntimeResources,
) -> PrimeQAHybridConcurrentSidecarAgentRuntime:
    """Assemble one concurrent runtime without duplicating heavy resources."""

    profiled_retriever = RequestLocalProfiledCandidatePoolRetriever(
        shared_resources.candidate_pool_retriever
    )
    entrypoint = create_primeqa_hybrid_optional_sidecar_agent_entrypoint(
        candidate_pool_retriever=profiled_retriever,
    )
    return PrimeQAHybridConcurrentSidecarAgentRuntime(
        resources=PrimeQAHybridConcurrentRuntimeResourceBundle(
            profiled_retriever=profiled_retriever,
            entrypoint=entrypoint,
            summary=shared_resources.summary,
        )
    )


def concurrent_sidecar_runtime_contract() -> dict[str, Any]:
    """Return the Stage145 implementation contract without enabling defaults."""

    return {
        "runtime_mode": _RUNTIME_MODE,
        "slo_profile_id": _SLO_PROFILE_ID,
        "max_in_flight": _MAX_IN_FLIGHT,
        "admission_mode": "nonblocking_bounded_semaphore",
        "capacity_error_type": "PrimeQAHybridConcurrentCapacityExceededError",
        "capacity_rejection_before_downstream": True,
        "request_trace_allowed_fields": sorted(_ALLOWED_REQUEST_TRACE_FIELDS),
        "shared_process_resources": True,
        "request_local_retrieval_profile": True,
        "request_local_agent_state_machine": True,
        "shared_pending_retrieval_profile_allowed": False,
        "registered_as_runtime_default": False,
        "test_access_allowed": False,
        "queue_actions_allowed": False,
        "retry_actions_allowed": False,
        "fallback_strategies_allowed": False,
        "errors_propagate": True,
    }


def _capacity_rejection_trace(
    *,
    arrival_pattern: ConcurrentArrivalPattern,
    in_flight: int,
    duration_seconds: float,
) -> PublicSafeConcurrentRuntimeRequestTrace:
    return PublicSafeConcurrentRuntimeRequestTrace(
        runtime_mode=_RUNTIME_MODE,
        activation_requested=True,
        activation_state=RuntimeActivationState.ELIGIBLE.value,
        slo_profile_id=_SLO_PROFILE_ID,
        warm_resources_ready=True,
        concurrency_limit=_MAX_IN_FLIGHT,
        in_flight_at_admission=in_flight,
        admission_state="rejected_capacity",
        arrival_pattern=arrival_pattern.value,
        candidate_pool_depth=0,
        retrieval_latency_ms=0.0,
        end_to_end_latency_ms=round(duration_seconds * 1000, 3),
        latency_budget_passed=duration_seconds <= _END_TO_END_P99_LIMIT_SECONDS,
        terminal_state="capacity_rejected",
    )
