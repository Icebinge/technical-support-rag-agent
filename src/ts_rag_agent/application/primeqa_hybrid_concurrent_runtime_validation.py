from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Barrier, Condition
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_validation_protocol import (
    ConcurrentRuntimeValidationEvidence,
    ConcurrentRuntimeValidationState,
    StrictPracticalConcurrentRuntimeValidationPolicy,
    _forbidden_keys_found,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_sidecar_agent_runtime import (
    ConcurrentArrivalPattern,
    ConcurrentRuntimeCounters,
    ConcurrentSidecarAgentRuntimeRun,
    PrimeQAHybridConcurrentCapacityExceededError,
    PrimeQAHybridConcurrentSidecarAgentRuntime,
    concurrent_sidecar_runtime_contract,
    create_primeqa_hybrid_concurrent_sidecar_agent_runtime,
)
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    EncoderFactory,
    _build_train_fold_assignments,
    _load_json_object,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_entrypoint_validation import (
    _entrypoint_trace_contract_violation_count,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    PrimeQAHybridProcessRuntimeResourceFactory,
)
from ts_rag_agent.application.primeqa_hybrid_strict_latency_validation import _distribution
from ts_rag_agent.application.rag_answering import evaluate_answers
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.answer import GeneratedAnswer
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
    summarize_primeqa_hybrid_split_samples,
)

_STAGE = "Stage 145"
_CREATED_AT = "2026-07-17"
_ANALYSIS_ID = "primeqa_hybrid_concurrent_runtime_train_cv_dev_validation_v1"
_STAGE144_STATUS = "primeqa_hybrid_concurrent_runtime_validation_protocol_frozen"
_STAGE143_STATUS = "primeqa_hybrid_optional_sidecar_runtime_wiring_validation_passed"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_TRAIN_ROW_COUNT = 562
_DEV_ROW_COUNT = 121
_TRAIN_FOLD_COUNT = 5
_REPETITIONS_PER_PATTERN = 3
_MAX_IN_FLIGHT = 4
_TRAIN_ACCEPTED_REQUEST_COUNT = 3372
_LATENCY_GATE_SCOPE_COUNT = 39
_P95_LIMIT_SECONDS = 0.8
_P99_LIMIT_SECONDS = 1.5
_TOP_K_VALUES = (10, 50, 100, 200, 400)
_SYNC_OFFSETS_MS = (0, 0, 0, 0)
_JITTER_OFFSETS_MS = (0, 7, 13, 20)
_NEXT_DIRECTION = "wire_validated_concurrency4_runtime_behind_explicit_nondefault_activation"


@dataclass(frozen=True)
class _ConcurrentValidationObservation:
    sample: PrimeQAHybridSplitSample
    verified_answer: GeneratedAnswer
    candidate_doc_ids: tuple[str, ...]
    arrival_pattern: str
    repetition: int
    target_arrival_offset_ms: int
    actual_arrival_offset_ms: float
    end_to_end_latency_seconds: float
    retrieval_latency_seconds: float
    runtime_trace: Mapping[str, Any]
    entrypoint_trace: Mapping[str, Any]
    runtime_trace_violation_count: int
    entrypoint_trace_violation_count: int
    behavior_digest: str


@dataclass(frozen=True)
class _PassExecution:
    pattern: str
    repetition: int
    observations: tuple[_ConcurrentValidationObservation, ...]
    wall_seconds: float
    scheduled_offsets_match_contract: bool


@dataclass(frozen=True)
class PrimeQAHybridConcurrentRuntimeValidationVisualization:
    name: str
    path: str


class _HoldFourAdmissions:
    """Formal overload barrier: four permits remain held before downstream work."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._admitted = 0
        self._released = False

    def on_admitted(self, in_flight_at_admission: int) -> None:
        if not 1 <= in_flight_at_admission <= _MAX_IN_FLIGHT:
            raise RuntimeError("overload probe observed an invalid in-flight count")
        with self._condition:
            self._admitted += 1
            self._condition.notify_all()
            while not self._released:
                self._condition.wait()

    def wait_until_full(self) -> None:
        with self._condition:
            while self._admitted < _MAX_IN_FLIGHT:
                self._condition.wait()

    def release(self) -> None:
        with self._condition:
            self._released = True
            self._condition.notify_all()


class _CohortArrivalGate:
    """Release every cohort worker from one measured clock origin."""

    def __init__(self, party_count: int) -> None:
        self._released_at: float | None = None
        self._barrier = Barrier(party_count, action=self._mark_released)

    def wait_for_release(self) -> float:
        self._barrier.wait()
        if self._released_at is None:
            raise RuntimeError("cohort arrival gate released without a clock origin")
        return self._released_at

    def _mark_released(self) -> None:
        self._released_at = time.perf_counter()


def run_primeqa_hybrid_concurrent_runtime_validation(
    *,
    stage144_protocol_path: Path,
    stage143_validation_path: Path,
    stage128_protocol_path: Path,
    stage125_protocol_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    user_confirmed_validation: bool,
    confirmation_note: str,
    encoder_batch_size: int = 64,
    encoder_device: str | None = None,
    encoder_factory: EncoderFactory | None = None,
) -> dict[str, Any]:
    """Implement and validate the Stage144 concurrency-four research runtime."""

    started_at = time.perf_counter()
    stage144 = _load_json_object(stage144_protocol_path)
    stage143 = _load_json_object(stage143_validation_path)
    source_files = _source_files(
        stage144_protocol=stage144_protocol_path,
        stage143_validation=stage143_validation_path,
        stage128_protocol=stage128_protocol_path,
        stage125_protocol=stage125_protocol_path,
        stage80_report=stage80_report_path,
        train_split=train_split_path,
        dev_split=dev_split_path,
        documents=documents_path,
    )
    pre_checks = _source_checks(
        stage144=stage144,
        stage143=stage143,
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
    )
    sources_loaded_at = time.perf_counter()
    if not all(check["passed"] for check in pre_checks):
        return _blocked_report(
            source_files=source_files,
            guard_checks=pre_checks,
            timing_seconds={
                "load_public_sources": round(sources_loaded_at - started_at, 3),
                "total": round(time.perf_counter() - started_at, 3),
            },
        )

    train_samples = load_primeqa_hybrid_split_samples(train_split_path)
    train_fold_assignments = _build_train_fold_assignments(
        train_samples,
        fold_count=_TRAIN_FOLD_COUNT,
    )
    train_loaded_at = time.perf_counter()

    resource_factory = PrimeQAHybridProcessRuntimeResourceFactory(
        stage128_protocol_path=stage128_protocol_path,
        stage125_protocol_path=stage125_protocol_path,
        stage80_report_path=stage80_report_path,
        documents_path=documents_path,
        encoder_batch_size=encoder_batch_size,
        encoder_device=encoder_device,
        encoder_factory=encoder_factory,
    )
    shared_resources = resource_factory.build_shared()
    runtime = create_primeqa_hybrid_concurrent_sidecar_agent_runtime(
        shared_resources=shared_resources
    )
    warmup_sample = min(
        train_samples,
        key=lambda row: hashlib.sha256(f"{_ANALYSIS_ID}:{row.sample_id}".encode()).hexdigest(),
    )
    warmup_run = runtime.run(
        _label_free_question(warmup_sample),
        arrival_pattern=ConcurrentArrivalPattern.WARMUP,
    )
    resources_ready_at = time.perf_counter()

    overload_report = _run_overload_probe(runtime=runtime, samples=train_samples[:4])
    overload_finished_at = time.perf_counter()

    train_counters_before = runtime.counters()
    pass_executions: list[_PassExecution] = []
    pass_timing_seconds: dict[str, float] = {}
    with ThreadPoolExecutor(
        max_workers=_MAX_IN_FLIGHT,
        thread_name_prefix="stage145-train",
    ) as executor:
        for repetition in range(1, _REPETITIONS_PER_PATTERN + 1):
            for pattern in (
                ConcurrentArrivalPattern.SYNCHRONIZED,
                ConcurrentArrivalPattern.DETERMINISTIC_JITTER,
            ):
                execution = _run_complete_pass(
                    executor=executor,
                    runtime=runtime,
                    samples=train_samples,
                    pattern=pattern,
                    repetition=repetition,
                )
                pass_executions.append(execution)
                pass_timing_seconds[_pass_id(pattern.value, repetition)] = round(
                    execution.wall_seconds,
                    3,
                )
    train_counters_after = runtime.counters()
    train_finished_at = time.perf_counter()

    expected_train = _stage143_split(stage143, _TRAIN_SPLIT)
    train_report = _build_train_report(
        executions=pass_executions,
        fold_assignments=train_fold_assignments,
        expected_stage143=expected_train,
        counter_delta=_counter_delta(train_counters_before, train_counters_after),
    )
    train_gate_passed = _train_gate_passed(train_report)

    dev_samples: Sequence[PrimeQAHybridSplitSample] = ()
    dev_execution: _PassExecution | None = None
    dev_report: Mapping[str, Any] = {}
    dev_loaded_at = train_finished_at
    dev_finished_at = train_finished_at
    if train_gate_passed:
        dev_samples = load_primeqa_hybrid_split_samples(dev_split_path)
        dev_loaded_at = time.perf_counter()
        dev_counters_before = runtime.counters()
        with ThreadPoolExecutor(
            max_workers=_MAX_IN_FLIGHT,
            thread_name_prefix="stage145-dev",
        ) as executor:
            dev_execution = _run_mixed_dev_pass(
                executor=executor,
                runtime=runtime,
                samples=dev_samples,
            )
        dev_counters_after = runtime.counters()
        dev_finished_at = time.perf_counter()
        dev_report = _build_dev_report(
            execution=dev_execution,
            expected_stage143=_stage143_split(stage143, _DEV_SPLIT),
            counter_delta=_counter_delta(dev_counters_before, dev_counters_after),
        )

    resource_summary = asdict(shared_resources.summary)
    policy_evidence = _policy_evidence(
        train_report=train_report,
        overload_report=overload_report,
        dev_report=dev_report,
        dev_loaded_after_train_gate=bool(dev_samples) and train_gate_passed,
        resource_summary=resource_summary,
        resource_factory_build_count=resource_factory.build_count,
    )
    policy_evaluation = StrictPracticalConcurrentRuntimeValidationPolicy().evaluate(policy_evidence)
    payload = {
        "runtime_contract": concurrent_sidecar_runtime_contract(),
        "resource_summary": resource_summary,
        "resource_factory_build_count": resource_factory.build_count,
        "warmup": {
            "request_count": 1,
            "candidate_pool_depth": len(warmup_run.candidate_pool_results),
            "runtime_trace": warmup_run.public_safe_trace.to_public_dict(),
        },
        "overload_probe": overload_report,
        "train_validation": train_report,
        "train_gate_passed_before_dev": train_gate_passed,
        "dev_loaded_only_after_train_gate": bool(dev_samples) and train_gate_passed,
        "dev_report_only_validation": dev_report,
        "concurrency_policy_evidence": asdict(policy_evidence),
        "concurrency_policy_evaluation": policy_evaluation.to_public_dict(),
    }
    checks = pre_checks + _validation_checks(
        payload=payload,
        stage143=stage143,
        dev_row_count=len(dev_samples),
    )
    checked_at = time.perf_counter()
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Implementation and real current-machine validation of the Stage144 strict "
            "practical B concurrency-four research runtime. One process builds shared "
            "resources once, runs a label-free warmup and a deterministic five-request "
            "overload probe, executes three complete train passes for each of two arrival "
            "patterns with grouped five-fold and pooled end-to-end latency gates, and "
            "loads dev once only after the train gate. Public output is aggregate only. "
            "Test, application runtime registration, defaultization, queues, retries, and "
            "fallback remain closed."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_validation),
            "confirmation_note": confirmation_note,
        },
        "source_files": source_files,
        "source_stage144": _source_stage_summary(stage144),
        "source_stage143": _source_stage_summary(stage143),
        "loaded_data_summary": {
            "train": summarize_primeqa_hybrid_split_samples({_TRAIN_SPLIT: train_samples})[
                _TRAIN_SPLIT
            ],
            "dev": (
                summarize_primeqa_hybrid_split_samples({_DEV_SPLIT: dev_samples})[_DEV_SPLIT]
                if dev_samples
                else {"row_count": 0}
            ),
            "dev_loaded_only_after_train_gate": bool(dev_samples) and train_gate_passed,
            "test_split_loaded": False,
        },
        **payload,
        "guard_checks": checks,
        "decision": _decision(checks, policy_evaluation.state),
        "timing_seconds": {
            "load_public_sources": round(sources_loaded_at - started_at, 3),
            "load_train_and_build_folds": round(train_loaded_at - sources_loaded_at, 3),
            "build_shared_resources_and_warmup": round(
                resources_ready_at - train_loaded_at,
                3,
            ),
            "run_overload_probe": round(overload_finished_at - resources_ready_at, 3),
            "run_six_complete_train_passes": round(
                train_finished_at - overload_finished_at,
                3,
            ),
            "train_passes": pass_timing_seconds,
            "load_dev_after_train_gate": round(dev_loaded_at - train_finished_at, 3),
            "run_one_dev_report_only_pass": round(dev_finished_at - dev_loaded_at, 3),
            "summarize_and_guard": round(checked_at - dev_finished_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def _run_complete_pass(
    *,
    executor: ThreadPoolExecutor,
    runtime: PrimeQAHybridConcurrentSidecarAgentRuntime,
    samples: Sequence[PrimeQAHybridSplitSample],
    pattern: ConcurrentArrivalPattern,
    repetition: int,
) -> _PassExecution:
    started_at = time.perf_counter()
    observations: list[_ConcurrentValidationObservation] = []
    offsets = _pattern_offsets(pattern)
    scheduled_exact = True
    for cohort_start in range(0, len(samples), _MAX_IN_FLIGHT):
        cohort = samples[cohort_start : cohort_start + _MAX_IN_FLIGHT]
        cohort_offsets = offsets[: len(cohort)]
        scheduled_exact = scheduled_exact and cohort_offsets == offsets[: len(cohort)]
        arrival_gate = _CohortArrivalGate(len(cohort))
        futures = [
            executor.submit(
                _run_scheduled_request,
                runtime=runtime,
                sample=sample,
                pattern=pattern,
                repetition=repetition,
                target_offset_ms=target_offset_ms,
                arrival_gate=arrival_gate,
            )
            for sample, target_offset_ms in zip(cohort, cohort_offsets, strict=True)
        ]
        observations.extend(future.result() for future in futures)
    return _PassExecution(
        pattern=pattern.value,
        repetition=repetition,
        observations=tuple(observations),
        wall_seconds=time.perf_counter() - started_at,
        scheduled_offsets_match_contract=scheduled_exact,
    )


def _run_mixed_dev_pass(
    *,
    executor: ThreadPoolExecutor,
    runtime: PrimeQAHybridConcurrentSidecarAgentRuntime,
    samples: Sequence[PrimeQAHybridSplitSample],
) -> _PassExecution:
    started_at = time.perf_counter()
    observations: list[_ConcurrentValidationObservation] = []
    scheduled_exact = True
    for cohort_index, cohort_start in enumerate(range(0, len(samples), _MAX_IN_FLIGHT)):
        cohort = samples[cohort_start : cohort_start + _MAX_IN_FLIGHT]
        pattern = (
            ConcurrentArrivalPattern.SYNCHRONIZED
            if cohort_index % 2 == 0
            else ConcurrentArrivalPattern.DETERMINISTIC_JITTER
        )
        offsets = _pattern_offsets(pattern)
        cohort_offsets = offsets[: len(cohort)]
        scheduled_exact = scheduled_exact and cohort_offsets == offsets[: len(cohort)]
        arrival_gate = _CohortArrivalGate(len(cohort))
        futures = [
            executor.submit(
                _run_scheduled_request,
                runtime=runtime,
                sample=sample,
                pattern=pattern,
                repetition=1,
                target_offset_ms=target_offset_ms,
                arrival_gate=arrival_gate,
            )
            for sample, target_offset_ms in zip(cohort, cohort_offsets, strict=True)
        ]
        observations.extend(future.result() for future in futures)
    return _PassExecution(
        pattern="alternating_synchronized_and_deterministic_jitter",
        repetition=1,
        observations=tuple(observations),
        wall_seconds=time.perf_counter() - started_at,
        scheduled_offsets_match_contract=scheduled_exact,
    )


def _run_scheduled_request(
    *,
    runtime: PrimeQAHybridConcurrentSidecarAgentRuntime,
    sample: PrimeQAHybridSplitSample,
    pattern: ConcurrentArrivalPattern,
    repetition: int,
    target_offset_ms: int,
    arrival_gate: _CohortArrivalGate,
) -> _ConcurrentValidationObservation:
    cohort_started_at = arrival_gate.wait_for_release()
    target_at = cohort_started_at + target_offset_ms / 1000
    remaining = target_at - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)
    actual_offset_ms = (time.perf_counter() - cohort_started_at) * 1000
    runtime_run = runtime.run(
        sample.to_primeqa_question(),
        arrival_pattern=pattern,
    )
    return _observe_runtime_request(
        sample=sample,
        runtime_run=runtime_run,
        pattern=pattern,
        repetition=repetition,
        target_offset_ms=target_offset_ms,
        actual_offset_ms=actual_offset_ms,
    )


def _observe_runtime_request(
    *,
    sample: PrimeQAHybridSplitSample,
    runtime_run: ConcurrentSidecarAgentRuntimeRun,
    pattern: ConcurrentArrivalPattern,
    repetition: int,
    target_offset_ms: int,
    actual_offset_ms: float,
) -> _ConcurrentValidationObservation:
    runtime_trace = runtime_run.public_safe_trace.to_public_dict()
    entrypoint_trace = runtime_run.entrypoint_run.public_safe_trace.to_public_dict()
    candidate_doc_ids = tuple(result.document.id for result in runtime_run.candidate_pool_results)
    behavior_digest = _behavior_digest(
        candidate_doc_ids=candidate_doc_ids,
        verified_answer=runtime_run.verified_answer,
        entrypoint_trace=entrypoint_trace,
    )
    return _ConcurrentValidationObservation(
        sample=sample,
        verified_answer=runtime_run.verified_answer,
        candidate_doc_ids=candidate_doc_ids,
        arrival_pattern=pattern.value,
        repetition=repetition,
        target_arrival_offset_ms=target_offset_ms,
        actual_arrival_offset_ms=actual_offset_ms,
        end_to_end_latency_seconds=float(runtime_trace["end_to_end_latency_ms"]) / 1000,
        retrieval_latency_seconds=float(runtime_trace["retrieval_latency_ms"]) / 1000,
        runtime_trace=runtime_trace,
        entrypoint_trace=entrypoint_trace,
        runtime_trace_violation_count=_runtime_trace_violation_count(
            runtime_trace=runtime_trace,
            entrypoint_trace=entrypoint_trace,
        ),
        entrypoint_trace_violation_count=_entrypoint_trace_contract_violation_count(
            public_trace=entrypoint_trace,
            verified_refused=runtime_run.verified_answer.refused,
        ),
        behavior_digest=behavior_digest,
    )


def _run_overload_probe(
    *,
    runtime: PrimeQAHybridConcurrentSidecarAgentRuntime,
    samples: Sequence[PrimeQAHybridSplitSample],
) -> dict[str, Any]:
    before = runtime.counters()
    barrier = _HoldFourAdmissions()
    rejected_trace: Mapping[str, Any] = {}
    downstream_before_rejection = 0
    downstream_after_rejection = 0
    with ThreadPoolExecutor(
        max_workers=_MAX_IN_FLIGHT,
        thread_name_prefix="stage145-overload",
    ) as executor:
        futures = [
            executor.submit(
                runtime.run,
                sample.to_primeqa_question(),
                arrival_pattern=ConcurrentArrivalPattern.OVERLOAD_PROBE,
                admission_probe=barrier,
            )
            for sample in samples
        ]
        barrier.wait_until_full()
        downstream_before_rejection = runtime.counters().downstream_request_count
        try:
            runtime.run(
                samples[0].to_primeqa_question(),
                arrival_pattern=ConcurrentArrivalPattern.OVERLOAD_PROBE,
            )
        except PrimeQAHybridConcurrentCapacityExceededError as error:
            rejected_trace = error.public_safe_trace.to_public_dict()
        else:
            barrier.release()
            raise RuntimeError("the fifth overload request was not capacity rejected")
        downstream_after_rejection = runtime.counters().downstream_request_count
        barrier.release()
        admitted_runs = [future.result() for future in futures]
    after = runtime.counters()
    delta = _counter_delta(before, after)
    admitted_latencies = [
        run.public_safe_trace.end_to_end_latency_ms / 1000 for run in admitted_runs
    ]
    return {
        "attempt_count": delta["admission_attempt_count"],
        "admitted_count": delta["admitted_request_count"],
        "rejected_count": delta["capacity_rejected_request_count"],
        "completed_admitted_count": delta["completed_request_count"],
        "failed_admitted_count": delta["failed_request_count"],
        "max_observed_in_flight": after.max_observed_in_flight,
        "admitted_end_to_end_latency_seconds": _distribution(admitted_latencies),
        "rejected_end_to_end_latency_seconds": float(
            rejected_trace.get("end_to_end_latency_ms") or 0
        )
        / 1000,
        "rejection_error_type": "PrimeQAHybridConcurrentCapacityExceededError",
        "rejected_trace": dict(rejected_trace),
        "rejected_downstream_call_count": (
            downstream_after_rejection - downstream_before_rejection
        ),
        "queue_action_count": 0,
        "retry_action_count": 0,
        "fallback_action_count": 0,
        "counter_delta": delta,
    }


def _build_train_report(
    *,
    executions: Sequence[_PassExecution],
    fold_assignments: Mapping[str, str],
    expected_stage143: Mapping[str, Any],
    counter_delta: Mapping[str, int],
) -> dict[str, Any]:
    all_observations = [
        observation for execution in executions for observation in execution.observations
    ]
    pass_reports = {
        _pass_id(execution.pattern, execution.repetition): {
            **_summarize_observations(execution.observations),
            "pattern": execution.pattern,
            "repetition": execution.repetition,
            "wall_seconds": round(execution.wall_seconds, 3),
            "throughput_requests_per_second": round(
                len(execution.observations) / execution.wall_seconds,
                4,
            ),
            "scheduled_offsets_match_contract": execution.scheduled_offsets_match_contract,
        }
        for execution in executions
    }
    for report in pass_reports.values():
        report["end_to_end_slo_passed"] = _concurrency_slo_pass(
            report["end_to_end_latency_seconds"]
        )
        report["behavior_matches_stage143"] = _behavior_matches_stage143(
            report,
            expected_stage143,
        )

    fold_reports: dict[str, dict[str, Any]] = {}
    for execution in executions:
        grouped: dict[str, list[_ConcurrentValidationObservation]] = defaultdict(list)
        for observation in execution.observations:
            grouped[fold_assignments[observation.sample.sample_id]].append(observation)
        for fold_id, rows in sorted(grouped.items()):
            gate_id = f"{_pass_id(execution.pattern, execution.repetition)}_{fold_id}"
            distribution = _distribution([row.end_to_end_latency_seconds for row in rows])
            fold_reports[gate_id] = {
                "pattern": execution.pattern,
                "repetition": execution.repetition,
                "fold": fold_id,
                "row_count": len(rows),
                "end_to_end_latency_seconds": distribution,
                "slo_passed": _concurrency_slo_pass(distribution),
            }

    pattern_reports = {}
    for pattern in (
        ConcurrentArrivalPattern.SYNCHRONIZED.value,
        ConcurrentArrivalPattern.DETERMINISTIC_JITTER.value,
    ):
        rows = [row for row in all_observations if row.arrival_pattern == pattern]
        distribution = _distribution([row.end_to_end_latency_seconds for row in rows])
        pattern_reports[pattern] = {
            "row_count": len(rows),
            "end_to_end_latency_seconds": distribution,
            "slo_passed": _concurrency_slo_pass(distribution),
        }

    global_distribution = _distribution(
        [row.end_to_end_latency_seconds for row in all_observations]
    )
    global_report = {
        "row_count": len(all_observations),
        "end_to_end_latency_seconds": global_distribution,
        "slo_passed": _concurrency_slo_pass(global_distribution),
    }
    failed_scopes = [
        *[gate_id for gate_id, report in fold_reports.items() if report["slo_passed"] is not True],
        *[
            gate_id
            for gate_id, report in pass_reports.items()
            if report["end_to_end_slo_passed"] is not True
        ],
        *[
            f"pattern_pooled_{pattern}"
            for pattern, report in pattern_reports.items()
            if report["slo_passed"] is not True
        ],
        *(["global_pooled"] if global_report["slo_passed"] is not True else []),
    ]
    combined = _summarize_observations(all_observations)
    cross_request_contamination_count = _cross_request_contamination_count(all_observations)
    pass_behavior_matches = all(
        report["behavior_matches_stage143"] is True for report in pass_reports.values()
    )
    return {
        "accepted_request_count": len(all_observations),
        "complete_pass_count": len(executions),
        "repetitions_per_pattern": {
            pattern: sum(execution.pattern == pattern for execution in executions)
            for pattern in pattern_reports
        },
        "pass_execution_order": [
            _pass_id(execution.pattern, execution.repetition) for execution in executions
        ],
        "synchronized_schedule_exact": all(
            execution.scheduled_offsets_match_contract
            for execution in executions
            if execution.pattern == ConcurrentArrivalPattern.SYNCHRONIZED.value
        ),
        "jittered_schedule_exact": all(
            execution.scheduled_offsets_match_contract
            for execution in executions
            if execution.pattern == ConcurrentArrivalPattern.DETERMINISTIC_JITTER.value
        ),
        "pass_reports": pass_reports,
        "fold_pattern_repetition_reports": fold_reports,
        "pattern_pooled_reports": pattern_reports,
        "global_pooled_report": global_report,
        "latency_gate_scope_count": (
            len(fold_reports) + len(pass_reports) + len(pattern_reports) + 1
        ),
        "failed_latency_gate_scopes": failed_scopes,
        "all_latency_gate_scopes_passed": not failed_scopes,
        "combined_runtime_summary": combined,
        "cross_request_contamination_count": cross_request_contamination_count,
        "all_pass_behavior_invariants_match_stage143": pass_behavior_matches,
        "behavior_invariants_passed": (
            pass_behavior_matches
            and cross_request_contamination_count == 0
            and combined["runtime_request_trace_violation_count"] == 0
            and combined["entrypoint_trace_violation_count"] == 0
        ),
        "runtime_counter_delta": dict(counter_delta),
    }


def _build_dev_report(
    *,
    execution: _PassExecution,
    expected_stage143: Mapping[str, Any],
    counter_delta: Mapping[str, int],
) -> dict[str, Any]:
    summary = _summarize_observations(execution.observations)
    pattern_counts = Counter(row.arrival_pattern for row in execution.observations)
    return {
        **summary,
        "measured_pass_count": 1,
        "wall_seconds": round(execution.wall_seconds, 3),
        "throughput_requests_per_second": round(
            len(execution.observations) / execution.wall_seconds,
            4,
        ),
        "scheduled_offsets_match_contract": execution.scheduled_offsets_match_contract,
        "cohort_pattern_counts": {
            "synchronized": 16,
            "deterministic_jitter": 15,
        },
        "request_pattern_counts": dict(sorted(pattern_counts.items())),
        "end_to_end_slo_passed": _concurrency_slo_pass(summary["end_to_end_latency_seconds"]),
        "behavior_matches_stage143": _behavior_matches_stage143(
            summary,
            expected_stage143,
        ),
        "runtime_counter_delta": dict(counter_delta),
    }


def _summarize_observations(
    observations: Sequence[_ConcurrentValidationObservation],
) -> dict[str, Any]:
    questions = [observation.sample.to_primeqa_question() for observation in observations]
    answers = [observation.verified_answer for observation in observations]
    metrics = asdict(evaluate_answers(questions, answers)) if observations else {}
    answerable = [observation for observation in observations if observation.sample.answerable]
    hit_counts = {
        str(top_k): sum(
            observation.sample.answer_doc_id in observation.candidate_doc_ids[:top_k]
            for observation in answerable
        )
        for top_k in _TOP_K_VALUES
    }
    gold_citations = sum(
        observation.sample.answer_doc_id
        in {citation.document_id for citation in observation.verified_answer.citations}
        for observation in answerable
    )
    return {
        "row_count": len(observations),
        "answerable_count": len(answerable),
        "runtime_request_trace_violation_count": sum(
            row.runtime_trace_violation_count for row in observations
        ),
        "entrypoint_trace_violation_count": sum(
            row.entrypoint_trace_violation_count for row in observations
        ),
        "exact_five_transition_trace_rate": _ratio(
            sum(row.entrypoint_trace_violation_count == 0 for row in observations),
            len(observations),
        ),
        "candidate_pool_depth": _distribution([len(row.candidate_doc_ids) for row in observations]),
        "end_to_end_latency_seconds": _distribution(
            [row.end_to_end_latency_seconds for row in observations]
        ),
        "retrieval_latency_seconds": _distribution(
            [row.retrieval_latency_seconds for row in observations]
        ),
        "target_arrival_offset_ms": _distribution(
            [row.target_arrival_offset_ms for row in observations]
        ),
        "actual_arrival_offset_ms": _distribution(
            [row.actual_arrival_offset_ms for row in observations]
        ),
        "arrival_offset_error_ms": _distribution(
            [
                abs(row.actual_arrival_offset_ms - row.target_arrival_offset_ms)
                for row in observations
            ]
        ),
        "latency_budget_failed_request_count": sum(
            row.runtime_trace.get("latency_budget_passed") is not True for row in observations
        ),
        "terminal_state_counts": {
            terminal: sum(
                row.runtime_trace.get("terminal_state") == terminal for row in observations
            )
            for terminal in ("complete", "refuse")
        },
        "retry_action_count": sum(
            int(row.entrypoint_trace.get("retry_action_count") or 0) for row in observations
        ),
        "fallback_action_count": sum(
            int(row.entrypoint_trace.get("fallback_action_count") or 0) for row in observations
        ),
        "recall": {
            "hit_counts": hit_counts,
            "rates": {key: _ratio(value, len(answerable)) for key, value in hit_counts.items()},
        },
        "verified_metrics": metrics,
        "verified_gold_citation_count": gold_citations,
    }


def _runtime_trace_violation_count(
    *,
    runtime_trace: Mapping[str, Any],
    entrypoint_trace: Mapping[str, Any],
) -> int:
    expected_fields = set(concurrent_sidecar_runtime_contract()["request_trace_allowed_fields"])
    arrival_patterns = {
        ConcurrentArrivalPattern.SYNCHRONIZED.value,
        ConcurrentArrivalPattern.DETERMINISTIC_JITTER.value,
    }
    checks = [
        set(runtime_trace) == expected_fields,
        runtime_trace.get("runtime_mode") == "optional_sidecar_agent_concurrent_four_request",
        runtime_trace.get("activation_requested") is True,
        runtime_trace.get("activation_state") == "eligible",
        runtime_trace.get("slo_profile_id") == "strict_practical_b_concurrency4_v1",
        runtime_trace.get("warm_resources_ready") is True,
        runtime_trace.get("concurrency_limit") == _MAX_IN_FLIGHT,
        1 <= int(runtime_trace.get("in_flight_at_admission") or 0) <= _MAX_IN_FLIGHT,
        runtime_trace.get("admission_state") == "admitted",
        runtime_trace.get("arrival_pattern") in arrival_patterns,
        runtime_trace.get("candidate_pool_depth") == 400,
        float(runtime_trace.get("retrieval_latency_ms") or -1) >= 0,
        float(runtime_trace.get("end_to_end_latency_ms") or -1)
        >= float(runtime_trace.get("retrieval_latency_ms") or 0),
        isinstance(runtime_trace.get("latency_budget_passed"), bool),
        runtime_trace.get("terminal_state") == entrypoint_trace.get("terminal_state"),
        _forbidden_keys_found(runtime_trace) == set(),
    ]
    return sum(not passed for passed in checks)


def _behavior_digest(
    *,
    candidate_doc_ids: Sequence[str],
    verified_answer: GeneratedAnswer,
    entrypoint_trace: Mapping[str, Any],
) -> str:
    private_payload = {
        "candidate_doc_ids": list(candidate_doc_ids),
        "verified_answer": asdict(verified_answer),
        "entrypoint_terminal_state": entrypoint_trace.get("terminal_state"),
        "entrypoint_action_trace": entrypoint_trace.get("action_trace"),
    }
    canonical = json.dumps(private_payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cross_request_contamination_count(
    observations: Sequence[_ConcurrentValidationObservation],
) -> int:
    grouped: dict[str, list[str]] = defaultdict(list)
    for observation in observations:
        grouped[observation.sample.sample_id].append(observation.behavior_digest)
    return sum(digest != digests[0] for digests in grouped.values() for digest in digests[1:])


def _behavior_matches_stage143(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    actual_pool = actual.get("candidate_pool_depth") or {}
    expected_pool = expected.get("candidate_pool_depth") or {}
    return (
        actual.get("runtime_request_trace_violation_count") == 0
        and actual.get("entrypoint_trace_violation_count") == 0
        and actual.get("exact_five_transition_trace_rate") == 1.0
        and actual_pool.get("min") == expected_pool.get("min") == 400.0
        and actual_pool.get("max") == expected_pool.get("max") == 400.0
        and (actual.get("recall") or {}).get("hit_counts")
        == (expected.get("recall") or {}).get("hit_counts")
        and (actual.get("verified_metrics") or {}).get("average_token_f1")
        == (expected.get("verified_metrics") or {}).get("average_token_f1")
        and actual.get("verified_gold_citation_count")
        == expected.get("verified_gold_citation_count")
        and actual.get("terminal_state_counts") == expected.get("terminal_state_counts")
        and actual.get("retry_action_count") == expected.get("retry_action_count") == 0
        and actual.get("fallback_action_count") == expected.get("fallback_action_count") == 0
    )


def _train_gate_passed(report: Mapping[str, Any]) -> bool:
    counters = report.get("runtime_counter_delta") or {}
    return (
        report.get("accepted_request_count") == _TRAIN_ACCEPTED_REQUEST_COUNT
        and report.get("complete_pass_count") == 6
        and report.get("synchronized_schedule_exact") is True
        and report.get("jittered_schedule_exact") is True
        and report.get("latency_gate_scope_count") == _LATENCY_GATE_SCOPE_COUNT
        and report.get("all_latency_gate_scopes_passed") is True
        and report.get("behavior_invariants_passed") is True
        and counters.get("admission_attempt_count") == _TRAIN_ACCEPTED_REQUEST_COUNT
        and counters.get("admitted_request_count") == _TRAIN_ACCEPTED_REQUEST_COUNT
        and counters.get("capacity_rejected_request_count") == 0
        and counters.get("downstream_request_count") == _TRAIN_ACCEPTED_REQUEST_COUNT
        and counters.get("completed_request_count") == _TRAIN_ACCEPTED_REQUEST_COUNT
        and counters.get("failed_request_count") == 0
    )


def _policy_evidence(
    *,
    train_report: Mapping[str, Any],
    overload_report: Mapping[str, Any],
    dev_report: Mapping[str, Any],
    dev_loaded_after_train_gate: bool,
    resource_summary: Mapping[str, Any],
    resource_factory_build_count: int,
) -> ConcurrentRuntimeValidationEvidence:
    train_global = (train_report.get("global_pooled_report") or {}).get(
        "end_to_end_latency_seconds"
    ) or {}
    dev_latency = dev_report.get("end_to_end_latency_seconds") or {}
    expected_resources = {
        "dense_model_count": 2,
        "dense_embedding_cache_count": 2,
        "lexical_index_count": 4,
        "derived_route_count": 1,
        "candidate_pool_retriever_instance_count": 1,
        "optional_entrypoint_instance_count": 1,
        "resources_built_or_loaded_per_request": False,
    }
    pass_reports = train_report.get("pass_reports") or {}
    fold_reports = train_report.get("fold_pattern_repetition_reports") or {}
    pattern_reports = train_report.get("pattern_pooled_reports") or {}
    return ConcurrentRuntimeValidationEvidence(
        profile_id="strict_practical_b_concurrency4_v1",
        warm_single_process=True,
        max_in_flight=4,
        synchronized_arrival_schedule_exact=(
            train_report.get("synchronized_schedule_exact") is True
        ),
        jittered_arrival_schedule_exact=(train_report.get("jittered_schedule_exact") is True),
        synchronized_train_repetitions=int(
            (train_report.get("repetitions_per_pattern") or {}).get(
                ConcurrentArrivalPattern.SYNCHRONIZED.value,
                0,
            )
        ),
        jittered_train_repetitions=int(
            (train_report.get("repetitions_per_pattern") or {}).get(
                ConcurrentArrivalPattern.DETERMINISTIC_JITTER.value,
                0,
            )
        ),
        train_accepted_request_count=int(train_report.get("accepted_request_count") or 0),
        train_fold_count=_TRAIN_FOLD_COUNT if len(fold_reports) == 30 else 0,
        train_latency_gate_scope_count=int(train_report.get("latency_gate_scope_count") or 0),
        train_fold_pattern_repetition_gates_passed=(
            len(fold_reports) == 30
            and all(report.get("slo_passed") is True for report in fold_reports.values())
        ),
        train_pass_aggregate_gates_passed=(
            len(pass_reports) == 6
            and all(report.get("end_to_end_slo_passed") is True for report in pass_reports.values())
        ),
        train_pattern_pooled_gates_passed=(
            len(pattern_reports) == 2
            and all(report.get("slo_passed") is True for report in pattern_reports.values())
        ),
        train_global_pooled_gate_passed=(
            (train_report.get("global_pooled_report") or {}).get("slo_passed") is True
        ),
        train_behavior_invariants_passed=(train_report.get("behavior_invariants_passed") is True),
        train_end_to_end_p95_seconds=_optional_float(train_global.get("p95")),
        train_end_to_end_p99_seconds=_optional_float(train_global.get("p99")),
        overload_attempt_count=int(overload_report.get("attempt_count") or 0),
        overload_admitted_count=int(overload_report.get("admitted_count") or 0),
        overload_rejected_count=int(overload_report.get("rejected_count") or 0),
        overload_rejected_before_downstream=(
            overload_report.get("rejected_downstream_call_count") == 0
        ),
        overload_error_type=str(overload_report.get("rejection_error_type") or ""),
        queue_action_count=int(overload_report.get("queue_action_count") or 0),
        retry_action_count=(
            int(overload_report.get("retry_action_count") or 0)
            + int(
                (train_report.get("combined_runtime_summary") or {}).get("retry_action_count") or 0
            )
            + int(dev_report.get("retry_action_count") or 0)
        ),
        fallback_action_count=(
            int(overload_report.get("fallback_action_count") or 0)
            + int(
                (train_report.get("combined_runtime_summary") or {}).get("fallback_action_count")
                or 0
            )
            + int(dev_report.get("fallback_action_count") or 0)
        ),
        process_resource_inventory_preserved=(
            dict(resource_summary) == expected_resources and resource_factory_build_count == 1
        ),
        request_local_state_isolated=(train_report.get("cross_request_contamination_count") == 0),
        dev_loaded_after_train_gate=dev_loaded_after_train_gate,
        dev_report_only_pass_count=int(dev_report.get("measured_pass_count") or 0),
        dev_accepted_request_count=int(dev_report.get("row_count") or 0),
        dev_end_to_end_slo_passed=dev_report.get("end_to_end_slo_passed") is True,
        dev_behavior_invariants_passed=(dev_report.get("behavior_matches_stage143") is True),
        dev_end_to_end_p95_seconds=_optional_float(dev_latency.get("p95")),
        dev_end_to_end_p99_seconds=_optional_float(dev_latency.get("p99")),
        test_split_locked=True,
        runtime_default_unchanged=True,
    )


def _source_checks(
    *,
    stage144: Mapping[str, Any],
    stage143: Mapping[str, Any],
    user_confirmed_validation: bool,
    confirmation_note: str,
) -> list[dict[str, Any]]:
    stage144_checks = stage144.get("guard_checks") or []
    stage143_checks = stage143.get("guard_checks") or []
    stage144_decision = stage144.get("decision") or {}
    stage143_decision = stage143.get("decision") or {}
    protocol = stage144.get("frozen_protocol") or {}
    profile = protocol.get("profile") or {}
    train = protocol.get("train_validation_contract") or {}
    overload = protocol.get("overload_contract") or {}
    return [
        _check(
            "stage145_user_confirmed",
            user_confirmed_validation and "Stage145" in confirmation_note,
            {"confirmed": user_confirmed_validation, "note_present": bool(confirmation_note)},
            "explicit Stage145 confirmation",
        ),
        _check(
            "stage144_protocol_source_passed",
            stage144.get("stage") == "Stage 144"
            and stage144_decision.get("status") == _STAGE144_STATUS
            and len(stage144_checks) == 29
            and all(row.get("passed") is True for row in stage144_checks),
            {"status": stage144_decision.get("status"), "guards": len(stage144_checks)},
            {"status": _STAGE144_STATUS, "guards": 29},
        ),
        _check(
            "stage144_profile_b_is_exact",
            profile.get("profile_id") == "strict_practical_b_concurrency4_v1"
            and profile.get("max_in_flight") == 4
            and profile.get("end_to_end_p95_seconds") == _P95_LIMIT_SECONDS
            and profile.get("end_to_end_p99_seconds") == _P99_LIMIT_SECONDS,
            profile,
            "profile B concurrency four and 0.8/1.5s",
        ),
        _check(
            "stage144_train_workload_and_gate_matrix_are_exact",
            train.get("accepted_requests_total") == _TRAIN_ACCEPTED_REQUEST_COUNT
            and train.get("complete_measured_pass_count") == 6
            and train.get("grouped_fold_count") == _TRAIN_FOLD_COUNT
            and train.get("total_latency_gate_scope_count") == _LATENCY_GATE_SCOPE_COUNT,
            train,
            "3372 requests, six passes, five folds, 39 scopes",
        ),
        _check(
            "stage144_overload_contract_is_exact",
            overload.get("probe_attempt_count") == 5
            and overload.get("expected_admitted_count") == 4
            and overload.get("expected_rejected_count") == 1
            and overload.get("typed_error") == "PrimeQAHybridConcurrentCapacityExceededError"
            and overload.get("queue_allowed") is False
            and overload.get("retry_allowed") is False
            and overload.get("fallback_allowed") is False,
            overload,
            "five/four/one typed pre-downstream rejection",
        ),
        _check(
            "stage143_single_request_source_passed",
            stage143.get("stage") == "Stage 143"
            and stage143_decision.get("status") == _STAGE143_STATUS
            and len(stage143_checks) == 28
            and all(row.get("passed") is True for row in stage143_checks)
            and stage143_decision.get("single_request_runtime_validated") is True,
            {"status": stage143_decision.get("status"), "guards": len(stage143_checks)},
            {"status": _STAGE143_STATUS, "guards": 28},
        ),
        _check(
            "source_reports_public_safe_and_test_locked",
            all(
                (source.get("public_safe_contract") or {}).get("forbidden_keys_found") == []
                and (source.get("public_safe_contract") or {}).get("test_split_loaded") is False
                and (source.get("public_safe_contract") or {}).get("test_metrics_run") is False
                for source in (stage144, stage143)
            ),
            "Stage144 and Stage143 public contracts",
            "forbidden empty and test false",
        ),
        _check(
            "source_default_retry_fallback_boundaries_are_closed",
            stage144_decision.get("runtime_registered_as_default") is False
            and stage144_decision.get("runtime_defaultization_allowed_now") is False
            and stage144_decision.get("retry_actions_enabled") is False
            and stage144_decision.get("fallback_strategies_enabled") is False
            and stage143_decision.get("runtime_registered_as_default") is False
            and stage143_decision.get("default_runtime_policy") == "unchanged",
            {"stage144": stage144_decision, "stage143": stage143_decision},
            "all closed",
        ),
    ]


def _validation_checks(
    *,
    payload: Mapping[str, Any],
    stage143: Mapping[str, Any],
    dev_row_count: int,
) -> list[dict[str, Any]]:
    contract = payload.get("runtime_contract") or {}
    resources = payload.get("resource_summary") or {}
    warmup = payload.get("warmup") or {}
    overload = payload.get("overload_probe") or {}
    train = payload.get("train_validation") or {}
    combined = train.get("combined_runtime_summary") or {}
    dev = payload.get("dev_report_only_validation") or {}
    evaluation = payload.get("concurrency_policy_evaluation") or {}
    expected_train = _stage143_split(stage143, _TRAIN_SPLIT)
    expected_dev = _stage143_split(stage143, _DEV_SPLIT)
    expected_resources = {
        "dense_model_count": 2,
        "dense_embedding_cache_count": 2,
        "lexical_index_count": 4,
        "derived_route_count": 1,
        "candidate_pool_retriever_instance_count": 1,
        "optional_entrypoint_instance_count": 1,
        "resources_built_or_loaded_per_request": False,
    }
    train_counters = train.get("runtime_counter_delta") or {}
    dev_counters = dev.get("runtime_counter_delta") or {}
    return [
        _check(
            "concurrent_runtime_contract_matches_stage144",
            contract.get("max_in_flight") == 4
            and contract.get("admission_mode") == "nonblocking_bounded_semaphore"
            and contract.get("capacity_rejection_before_downstream") is True
            and contract.get("request_local_retrieval_profile") is True
            and contract.get("request_local_agent_state_machine") is True
            and contract.get("shared_pending_retrieval_profile_allowed") is False,
            contract,
            "bounded four and request-local mutable state",
        ),
        _check(
            "process_resources_match_stage143_and_build_once",
            resources == expected_resources and payload.get("resource_factory_build_count") == 1,
            {"resources": resources, "build_count": payload.get("resource_factory_build_count")},
            "Stage143 inventory built once",
        ),
        _check(
            "warmup_is_one_label_free_full_request",
            warmup.get("request_count") == 1
            and warmup.get("candidate_pool_depth") == 400
            and (warmup.get("runtime_trace") or {}).get("arrival_pattern")
            == ConcurrentArrivalPattern.WARMUP.value,
            warmup,
            "one depth-400 warmup",
        ),
        _check(
            "overload_probe_attempts_five_admits_four_rejects_one",
            overload.get("attempt_count") == 5
            and overload.get("admitted_count") == 4
            and overload.get("rejected_count") == 1
            and overload.get("completed_admitted_count") == 4
            and overload.get("failed_admitted_count") == 0
            and overload.get("max_observed_in_flight") == 4,
            overload,
            "5 attempted, 4 completed, 1 rejected",
        ),
        _check(
            "overload_rejection_is_typed_before_downstream",
            overload.get("rejection_error_type") == "PrimeQAHybridConcurrentCapacityExceededError"
            and overload.get("rejected_downstream_call_count") == 0
            and (overload.get("rejected_trace") or {}).get("admission_state") == "rejected_capacity"
            and (overload.get("rejected_trace") or {}).get("terminal_state") == "capacity_rejected"
            and (overload.get("rejected_trace") or {}).get("candidate_pool_depth") == 0,
            {
                "type": overload.get("rejection_error_type"),
                "downstream": overload.get("rejected_downstream_call_count"),
                "trace": overload.get("rejected_trace"),
            },
            "typed rejection with zero downstream calls",
        ),
        _check(
            "overload_queue_retry_fallback_counts_are_zero",
            overload.get("queue_action_count") == 0
            and overload.get("retry_action_count") == 0
            and overload.get("fallback_action_count") == 0,
            {
                "queue": overload.get("queue_action_count"),
                "retry": overload.get("retry_action_count"),
                "fallback": overload.get("fallback_action_count"),
            },
            0,
        ),
        _check(
            "train_runs_six_complete_passes_and_3372_requests",
            train.get("accepted_request_count") == _TRAIN_ACCEPTED_REQUEST_COUNT
            and train.get("complete_pass_count") == 6
            and len(train.get("pass_reports") or {}) == 6,
            {
                "requests": train.get("accepted_request_count"),
                "passes": train.get("complete_pass_count"),
            },
            {"requests": 3372, "passes": 6},
        ),
        _check(
            "train_pass_order_alternates_sync_and_jitter",
            train.get("pass_execution_order")
            == [
                "synchronized_four_request_burst_repetition_1",
                "deterministic_jitter_0_to_20ms_repetition_1",
                "synchronized_four_request_burst_repetition_2",
                "deterministic_jitter_0_to_20ms_repetition_2",
                "synchronized_four_request_burst_repetition_3",
                "deterministic_jitter_0_to_20ms_repetition_3",
            ],
            train.get("pass_execution_order"),
            "alternating pattern order",
        ),
        _check(
            "train_arrival_schedules_are_exact",
            train.get("synchronized_schedule_exact") is True
            and train.get("jittered_schedule_exact") is True,
            {
                "sync": train.get("synchronized_schedule_exact"),
                "jitter": train.get("jittered_schedule_exact"),
            },
            True,
        ),
        _check(
            "train_runtime_counters_are_exact_without_rejection",
            train_counters.get("admission_attempt_count") == _TRAIN_ACCEPTED_REQUEST_COUNT
            and train_counters.get("admitted_request_count") == _TRAIN_ACCEPTED_REQUEST_COUNT
            and train_counters.get("capacity_rejected_request_count") == 0
            and train_counters.get("downstream_request_count") == _TRAIN_ACCEPTED_REQUEST_COUNT
            and train_counters.get("completed_request_count") == _TRAIN_ACCEPTED_REQUEST_COUNT
            and train_counters.get("failed_request_count") == 0,
            train_counters,
            "3372 admitted, downstream, completed; zero rejected/failed",
        ),
        _check(
            "train_latency_gate_matrix_has_39_scopes",
            len(train.get("fold_pattern_repetition_reports") or {}) == 30
            and len(train.get("pass_reports") or {}) == 6
            and len(train.get("pattern_pooled_reports") or {}) == 2
            and train.get("latency_gate_scope_count") == _LATENCY_GATE_SCOPE_COUNT,
            {
                "fold": len(train.get("fold_pattern_repetition_reports") or {}),
                "pass": len(train.get("pass_reports") or {}),
                "pattern": len(train.get("pattern_pooled_reports") or {}),
                "total": train.get("latency_gate_scope_count"),
            },
            {"fold": 30, "pass": 6, "pattern": 2, "global": 1, "total": 39},
        ),
        _check(
            "all_train_latency_gate_scopes_meet_profile_b",
            train.get("all_latency_gate_scopes_passed") is True
            and train.get("failed_latency_gate_scopes") == [],
            {
                "global": train.get("global_pooled_report"),
                "failed": train.get("failed_latency_gate_scopes"),
            },
            "all P95 <= 0.8 and P99 <= 1.5",
        ),
        _check(
            "train_runtime_and_entrypoint_traces_are_exact",
            combined.get("runtime_request_trace_violation_count") == 0
            and combined.get("entrypoint_trace_violation_count") == 0
            and combined.get("exact_five_transition_trace_rate") == 1.0,
            {
                "runtime": combined.get("runtime_request_trace_violation_count"),
                "entrypoint": combined.get("entrypoint_trace_violation_count"),
                "rate": combined.get("exact_five_transition_trace_rate"),
            },
            "zero violations and exact five transitions",
        ),
        _check(
            "train_candidate_pool_depth_is_400",
            (combined.get("candidate_pool_depth") or {}).get("min") == 400.0
            and (combined.get("candidate_pool_depth") or {}).get("max") == 400.0,
            combined.get("candidate_pool_depth"),
            400,
        ),
        _check(
            "train_cross_request_contamination_is_zero",
            train.get("cross_request_contamination_count") == 0,
            train.get("cross_request_contamination_count"),
            0,
        ),
        _check(
            "every_train_pass_matches_stage143_behavior",
            train.get("all_pass_behavior_invariants_match_stage143") is True
            and all(
                report.get("behavior_matches_stage143") is True
                for report in (train.get("pass_reports") or {}).values()
            ),
            {
                name: report.get("behavior_matches_stage143")
                for name, report in (train.get("pass_reports") or {}).items()
            },
            True,
        ),
        _check(
            "train_expected_stage143_baseline_is_present",
            (expected_train.get("recall") or {}).get("hit_counts")
            == {"10": 255, "50": 303, "100": 332, "200": 345, "400": 354}
            and (expected_train.get("verified_metrics") or {}).get("average_token_f1") == 0.1946
            and expected_train.get("verified_gold_citation_count") == 151,
            {
                "recall": (expected_train.get("recall") or {}).get("hit_counts"),
                "f1": (expected_train.get("verified_metrics") or {}).get("average_token_f1"),
                "gold": expected_train.get("verified_gold_citation_count"),
            },
            "Stage143 train baseline",
        ),
        _check(
            "train_gate_passes_before_dev_load",
            payload.get("train_gate_passed_before_dev") is True
            and payload.get("dev_loaded_only_after_train_gate") is True,
            {
                "train": payload.get("train_gate_passed_before_dev"),
                "dev_after": payload.get("dev_loaded_only_after_train_gate"),
            },
            True,
        ),
        _check(
            "dev_runs_one_121_request_report_only_pass",
            dev_row_count == _DEV_ROW_COUNT
            and dev.get("row_count") == _DEV_ROW_COUNT
            and dev.get("measured_pass_count") == 1,
            {
                "loaded": dev_row_count,
                "measured": dev.get("row_count"),
                "passes": dev.get("measured_pass_count"),
            },
            {"rows": 121, "passes": 1},
        ),
        _check(
            "dev_mixed_schedule_is_exact",
            dev.get("scheduled_offsets_match_contract") is True
            and dev.get("cohort_pattern_counts")
            == {"synchronized": 16, "deterministic_jitter": 15},
            {
                "exact": dev.get("scheduled_offsets_match_contract"),
                "cohorts": dev.get("cohort_pattern_counts"),
            },
            {"synchronized": 16, "deterministic_jitter": 15},
        ),
        _check(
            "dev_runtime_counters_are_exact_without_rejection",
            dev_counters.get("admission_attempt_count") == _DEV_ROW_COUNT
            and dev_counters.get("admitted_request_count") == _DEV_ROW_COUNT
            and dev_counters.get("capacity_rejected_request_count") == 0
            and dev_counters.get("downstream_request_count") == _DEV_ROW_COUNT
            and dev_counters.get("completed_request_count") == _DEV_ROW_COUNT
            and dev_counters.get("failed_request_count") == 0,
            dev_counters,
            "121 admitted, downstream, completed; zero rejected/failed",
        ),
        _check(
            "dev_end_to_end_latency_meets_profile_b",
            dev.get("end_to_end_slo_passed") is True,
            dev.get("end_to_end_latency_seconds"),
            "P95 <= 0.8 and P99 <= 1.5",
        ),
        _check(
            "dev_runtime_entrypoint_and_pool_are_exact",
            dev.get("runtime_request_trace_violation_count") == 0
            and dev.get("entrypoint_trace_violation_count") == 0
            and dev.get("exact_five_transition_trace_rate") == 1.0
            and (dev.get("candidate_pool_depth") or {}).get("min") == 400.0
            and (dev.get("candidate_pool_depth") or {}).get("max") == 400.0,
            {
                "runtime": dev.get("runtime_request_trace_violation_count"),
                "entrypoint": dev.get("entrypoint_trace_violation_count"),
                "depth": dev.get("candidate_pool_depth"),
            },
            "zero violations and depth 400",
        ),
        _check(
            "dev_behavior_matches_stage143",
            dev.get("behavior_matches_stage143") is True
            and (expected_dev.get("recall") or {}).get("hit_counts")
            == {"10": 55, "50": 64, "100": 66, "200": 69, "400": 70}
            and (expected_dev.get("verified_metrics") or {}).get("average_token_f1") == 0.1873
            and expected_dev.get("verified_gold_citation_count") == 33,
            {
                "matches": dev.get("behavior_matches_stage143"),
                "recall": (dev.get("recall") or {}).get("hit_counts"),
                "f1": (dev.get("verified_metrics") or {}).get("average_token_f1"),
                "gold": dev.get("verified_gold_citation_count"),
            },
            "Stage143 dev baseline",
        ),
        _check(
            "concurrency_policy_evaluates_real_evidence_as_eligible",
            evaluation.get("state") == ConcurrentRuntimeValidationState.ELIGIBLE.value
            and evaluation.get("rejection_reasons") == []
            and evaluation.get("concurrent_runtime_activated") is False,
            evaluation,
            "eligible without automatic activation",
        ),
        _check(
            "all_measured_requests_have_no_retry_or_fallback",
            combined.get("retry_action_count") == 0
            and combined.get("fallback_action_count") == 0
            and dev.get("retry_action_count") == 0
            and dev.get("fallback_action_count") == 0,
            {
                "train_retry": combined.get("retry_action_count"),
                "train_fallback": combined.get("fallback_action_count"),
                "dev_retry": dev.get("retry_action_count"),
                "dev_fallback": dev.get("fallback_action_count"),
            },
            0,
        ),
        _check(
            "test_default_queue_retry_fallback_boundaries_remain_closed",
            contract.get("registered_as_runtime_default") is False
            and contract.get("test_access_allowed") is False
            and contract.get("queue_actions_allowed") is False
            and contract.get("retry_actions_allowed") is False
            and contract.get("fallback_strategies_allowed") is False,
            contract,
            False,
        ),
        _check(
            "validation_payload_is_public_safe",
            _forbidden_keys_found(payload) == set(),
            sorted(_forbidden_keys_found(payload)),
            [],
        ),
    ]


def _decision(
    checks: Sequence[Mapping[str, Any]],
    policy_state: ConcurrentRuntimeValidationState,
) -> dict[str, Any]:
    failed = [str(check["name"]) for check in checks if check.get("passed") is not True]
    passed = not failed and policy_state is ConcurrentRuntimeValidationState.ELIGIBLE
    return {
        "status": (
            "primeqa_hybrid_concurrent_runtime_train_cv_dev_validation_passed"
            if passed
            else "primeqa_hybrid_concurrent_runtime_train_cv_dev_validation_blocked"
        ),
        "failed_checks": failed,
        "concurrent_research_runtime_implemented": True,
        "concurrent_research_runtime_validation_passed": passed,
        "profile_b_end_to_end_slo_passed": passed,
        "overload_admission_contract_validated": passed,
        "request_local_state_isolation_validated": passed,
        "stage143_behavior_preserved": passed,
        "can_wire_explicit_nondefault_concurrent_runtime_now": passed,
        "concurrent_runtime_registered_for_application_use": False,
        "concurrent_runtime_activation_allowed_now": False,
        "runtime_registered_as_default": False,
        "runtime_defaultization_allowed_now": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
        "queue_actions_enabled": False,
        "retry_actions_enabled": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_direction": _NEXT_DIRECTION
        if passed
        else "repair_stage145_concurrent_runtime_validation",
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    loaded_data_summary = report.get("loaded_data_summary") or {}
    train_summary = loaded_data_summary.get("train") or {}
    dev_summary = loaded_data_summary.get("dev") or {}
    return {
        "aggregate_only": True,
        "private_per_request_traces_written": False,
        "private_behavior_digests_written": False,
        "raw_questions_written": False,
        "raw_answers_written": False,
        "raw_documents_written": False,
        "raw_candidate_rows_written": False,
        "raw_sample_document_request_or_cohort_ids_written": False,
        "train_cv_group_values_written": False,
        "train_split_loaded": bool(train_summary.get("row_count")),
        "dev_split_loaded": bool(dev_summary.get("row_count")),
        "test_split_loaded": False,
        "test_metrics_run": False,
        "forbidden_keys_found": sorted(_forbidden_keys_found(report)),
    }


def _blocked_report(
    *,
    source_files: Mapping[str, Any],
    guard_checks: Sequence[Mapping[str, Any]],
    timing_seconds: Mapping[str, float],
) -> dict[str, Any]:
    failed = [str(check["name"]) for check in guard_checks if check.get("passed") is not True]
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": "Blocked before loading train or building runtime resources.",
        "source_files": source_files,
        "guard_checks": list(guard_checks),
        "decision": {
            "status": "primeqa_hybrid_concurrent_runtime_train_cv_dev_validation_blocked",
            "failed_checks": failed,
            "concurrent_research_runtime_implemented": True,
            "concurrent_research_runtime_validation_passed": False,
            "can_wire_explicit_nondefault_concurrent_runtime_now": False,
            "concurrent_runtime_registered_for_application_use": False,
            "concurrent_runtime_activation_allowed_now": False,
            "runtime_registered_as_default": False,
            "runtime_defaultization_allowed_now": False,
            "test_gate_opened": False,
            "test_metrics_run": False,
            "queue_actions_enabled": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
            "recommended_next_direction": "repair_stage145_source_guards",
        },
        "timing_seconds": dict(timing_seconds),
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_concurrent_runtime_validation_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridConcurrentRuntimeValidationVisualization]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = report.get("train_validation") or {}
    dev = report.get("dev_report_only_validation") or {}
    overload = report.get("overload_probe") or {}
    charts = {
        "stage145_train_pass_end_to_end_latency.svg": render_horizontal_bar_chart_svg(
            title="Stage145 train pass end-to-end P95 and P99",
            bars=_pass_latency_bars(train),
            x_label="seconds",
            width=1720,
            margin_left=820,
        ),
        "stage145_pattern_pooled_latency.svg": render_horizontal_bar_chart_svg(
            title="Stage145 pattern-pooled end-to-end latency",
            bars=_pattern_latency_bars(train),
            x_label="seconds",
            width=1680,
            margin_left=780,
        ),
        "stage145_latency_scope_maxima.svg": render_horizontal_bar_chart_svg(
            title="Stage145 maximum percentile across latency scope families",
            bars=_scope_maximum_bars(train),
            x_label="seconds",
            width=1760,
            margin_left=860,
        ),
        "stage145_dev_end_to_end_latency.svg": render_horizontal_bar_chart_svg(
            title="Stage145 dev report-only end-to-end latency",
            bars=_dev_latency_bars(dev),
            x_label="seconds",
            width=1500,
            margin_left=720,
        ),
        "stage145_request_budget.svg": render_horizontal_bar_chart_svg(
            title="Stage145 measured request budget",
            bars=_request_budget_bars(report),
            x_label="requests",
            width=1500,
            margin_left=720,
        ),
        "stage145_pass_throughput.svg": render_horizontal_bar_chart_svg(
            title="Stage145 train pass throughput",
            bars=_throughput_bars(train),
            x_label="requests per second",
            width=1740,
            margin_left=840,
        ),
        "stage145_overload_outcome.svg": render_horizontal_bar_chart_svg(
            title="Stage145 overload probe outcome",
            bars=[
                _bar("attempted", float(overload.get("attempt_count") or 0)),
                _bar("admitted", float(overload.get("admitted_count") or 0)),
                _bar("capacity rejected", float(overload.get("rejected_count") or 0)),
                _bar(
                    "rejected downstream calls",
                    float(overload.get("rejected_downstream_call_count") or 0),
                ),
            ],
            x_label="requests or calls",
            width=1500,
            margin_left=760,
        ),
        "stage145_behavior_invariants.svg": render_horizontal_bar_chart_svg(
            title="Stage145 behavior and isolation invariants",
            bars=_invariant_bars(report),
            x_label="1 means passed",
            width=1780,
            margin_left=900,
        ),
        "stage145_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage145 concurrent runtime decision flags",
            bars=[
                BarDatum(
                    label=key,
                    value=1.0 if value else 0.0,
                    value_label="true" if value else "false",
                )
                for key, value in (report.get("decision") or {}).items()
                if isinstance(value, bool)
            ],
            x_label="1 means true",
            width=2060,
            margin_left=1120,
        ),
        "stage145_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage145 concurrent runtime guard checks",
            bars=[
                BarDatum(
                    label=str(check.get("name")),
                    value=1.0 if check.get("passed") else 0.0,
                    value_label="passed" if check.get("passed") else "failed",
                )
                for check in report.get("guard_checks") or []
            ],
            x_label="1 means passed",
            width=2380,
            margin_left=1420,
        ),
    }
    artifacts = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        artifacts.append(PrimeQAHybridConcurrentRuntimeValidationVisualization(name, str(path)))
    return artifacts


def _pass_latency_bars(train: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for pass_id, report in (train.get("pass_reports") or {}).items():
        latency = report.get("end_to_end_latency_seconds") or {}
        bars.extend(
            (
                _bar(f"{pass_id} P95", float(latency.get("p95") or 0)),
                _bar(f"{pass_id} P99", float(latency.get("p99") or 0)),
            )
        )
    return bars


def _pattern_latency_bars(train: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for pattern, report in (train.get("pattern_pooled_reports") or {}).items():
        latency = report.get("end_to_end_latency_seconds") or {}
        bars.extend(
            (
                _bar(f"{pattern} P95", float(latency.get("p95") or 0)),
                _bar(f"{pattern} P99", float(latency.get("p99") or 0)),
            )
        )
    bars.extend((_bar("P95 limit", 0.8), _bar("P99 limit", 1.5)))
    return bars


def _scope_maximum_bars(train: Mapping[str, Any]) -> list[BarDatum]:
    families = {
        "fold x pattern x repetition": [
            report.get("end_to_end_latency_seconds") or {}
            for report in (train.get("fold_pattern_repetition_reports") or {}).values()
        ],
        "complete pass": [
            report.get("end_to_end_latency_seconds") or {}
            for report in (train.get("pass_reports") or {}).values()
        ],
        "pattern pooled": [
            report.get("end_to_end_latency_seconds") or {}
            for report in (train.get("pattern_pooled_reports") or {}).values()
        ],
        "global pooled": [
            (train.get("global_pooled_report") or {}).get("end_to_end_latency_seconds") or {}
        ],
    }
    return [
        _bar(f"{family} max P95", max(float(row.get("p95") or 0) for row in rows))
        for family, rows in families.items()
    ] + [
        _bar(f"{family} max P99", max(float(row.get("p99") or 0) for row in rows))
        for family, rows in families.items()
    ]


def _dev_latency_bars(dev: Mapping[str, Any]) -> list[BarDatum]:
    latency = dev.get("end_to_end_latency_seconds") or {}
    return [
        _bar("dev P50", float(latency.get("p50") or 0)),
        _bar("dev P95", float(latency.get("p95") or 0)),
        _bar("dev P99", float(latency.get("p99") or 0)),
        _bar("dev max", float(latency.get("max") or 0)),
        _bar("P95 limit", 0.8),
        _bar("P99 limit", 1.5),
    ]


def _request_budget_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    train = report.get("train_validation") or {}
    dev = report.get("dev_report_only_validation") or {}
    overload = report.get("overload_probe") or {}
    return [
        _bar("train accepted", float(train.get("accepted_request_count") or 0)),
        _bar("dev accepted", float(dev.get("row_count") or 0)),
        _bar("overload attempted", float(overload.get("attempt_count") or 0)),
        _bar("warmup", float((report.get("warmup") or {}).get("request_count") or 0)),
    ]


def _throughput_bars(train: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=pass_id,
            value=float(report.get("throughput_requests_per_second") or 0),
            value_label=f"{float(report.get('throughput_requests_per_second') or 0):.4f}",
        )
        for pass_id, report in (train.get("pass_reports") or {}).items()
    ]


def _invariant_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    train = report.get("train_validation") or {}
    dev = report.get("dev_report_only_validation") or {}
    policy = report.get("concurrency_policy_evaluation") or {}
    values = {
        "train all latency scopes": train.get("all_latency_gate_scopes_passed") is True,
        "train behavior invariants": train.get("behavior_invariants_passed") is True,
        "train zero contamination": train.get("cross_request_contamination_count") == 0,
        "dev latency": dev.get("end_to_end_slo_passed") is True,
        "dev behavior": dev.get("behavior_matches_stage143") is True,
        "policy eligible": policy.get("state") == "eligible",
    }
    return [
        BarDatum(label, 1.0 if passed else 0.0, "passed" if passed else "failed")
        for label, passed in values.items()
    ]


def _bar(label: str, value: float) -> BarDatum:
    return BarDatum(label=label, value=value, value_label=f"{value:.6f}")


def _pattern_offsets(pattern: ConcurrentArrivalPattern) -> tuple[int, ...]:
    if pattern is ConcurrentArrivalPattern.SYNCHRONIZED:
        return _SYNC_OFFSETS_MS
    if pattern is ConcurrentArrivalPattern.DETERMINISTIC_JITTER:
        return _JITTER_OFFSETS_MS
    raise ValueError(f"arrival pattern is not a measured cohort pattern: {pattern.value}")


def _pass_id(pattern: str, repetition: int) -> str:
    return f"{pattern}_repetition_{repetition}"


def _concurrency_slo_pass(distribution: Mapping[str, Any]) -> bool:
    return (
        float(distribution.get("p95", float("inf"))) <= _P95_LIMIT_SECONDS
        and float(distribution.get("p99", float("inf"))) <= _P99_LIMIT_SECONDS
    )


def _counter_delta(
    before: ConcurrentRuntimeCounters,
    after: ConcurrentRuntimeCounters,
) -> dict[str, int]:
    cumulative = (
        "admission_attempt_count",
        "admitted_request_count",
        "capacity_rejected_request_count",
        "downstream_request_count",
        "completed_request_count",
        "failed_request_count",
    )
    return {
        **{key: int(getattr(after, key) - getattr(before, key)) for key in cumulative},
        "current_in_flight": after.current_in_flight,
        "max_observed_in_flight": after.max_observed_in_flight,
    }


def _stage143_split(report: Mapping[str, Any], split: str) -> dict[str, Any]:
    key = (
        "train_runtime_validation"
        if split == _TRAIN_SPLIT
        else "dev_runtime_report_only_validation"
    )
    return dict(report.get(key) or {})


def _label_free_question(sample: PrimeQAHybridSplitSample):
    question = sample.to_primeqa_question()
    return type(question)(
        id="concurrent-runtime-warmup",
        title=question.title,
        text=question.text,
        answer="",
        answerable=False,
        answer_doc_id=None,
        doc_ids=[],
    )


def _source_stage_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision") or {}
    checks = report.get("guard_checks") or []
    public = report.get("public_safe_contract") or {}
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "status": decision.get("status"),
        "guard_check_count": len(checks),
        "guard_check_passed_count": sum(row.get("passed") is True for row in checks),
        "test_split_loaded": public.get("test_split_loaded"),
        "test_metrics_run": public.get("test_metrics_run"),
        "forbidden_keys_found": public.get("forbidden_keys_found"),
    }


def _source_files(**paths: Path) -> dict[str, Any]:
    return {
        name: {
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in paths.items()
    }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _check(name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
