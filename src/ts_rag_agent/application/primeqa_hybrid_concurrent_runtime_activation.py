from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any, Protocol

from ts_rag_agent.application.primeqa_hybrid_agent_runtime_observability import (
    AgentWorkflowObservationSink,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_validation_protocol import (
    ConcurrentRuntimeValidationEvaluation,
    ConcurrentRuntimeValidationEvidence,
    ConcurrentRuntimeValidationState,
    StrictPracticalConcurrentRuntimeValidationPolicy,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_sidecar_agent_runtime import (
    ConcurrentArrivalPattern,
    PrimeQAHybridConcurrentSidecarAgentRuntime,
    concurrent_sidecar_runtime_contract,
    create_primeqa_hybrid_concurrent_sidecar_agent_runtime,
)
from ts_rag_agent.application.primeqa_hybrid_nondefault_runtime_activation_protocol import (
    RuntimeActivationState,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    PrimeQAHybridRuntimeResourceSummary,
    PrimeQAHybridSharedRuntimeResources,
    _forbidden_keys_found,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.domain.dataset import PrimeQARuntimeQuery

_STAGE145_STATUS = "primeqa_hybrid_concurrent_runtime_train_cv_dev_validation_passed"
_STAGE145_ANALYSIS_ID = "primeqa_hybrid_concurrent_runtime_train_cv_dev_validation_v1"
_EXPECTED_STAGE145_GUARDS = 36
_SETTINGS_FIELD = "enable_concurrent_sidecar_agent"
_ENVIRONMENT_FLAG = "TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT"
_PROFILE_ID = "strict_practical_b_concurrency4_v1"
_MAX_IN_FLIGHT = 4
_TARGET_POOL_DEPTH = 400
_EXPECTED_RESOURCES = {
    "dense_model_count": 2,
    "dense_embedding_cache_count": 2,
    "lexical_index_count": 4,
    "derived_route_count": 1,
    "candidate_pool_retriever_instance_count": 1,
    "optional_entrypoint_instance_count": 1,
    "resources_built_or_loaded_per_request": False,
}


class ConcurrentRuntimeSharedResourceFactoryPort(Protocol):
    """Build one shared resource graph for the concurrent application runtime."""

    build_count: int

    def build_shared(self) -> PrimeQAHybridSharedRuntimeResources: ...


@dataclass(frozen=True)
class PublicSafeConcurrentRuntimeStartupTrace:
    """Aggregate-only application bootstrap outcome for Stage146."""

    runtime_mode: str
    settings_field: str
    environment_flag: str
    activation_requested: bool
    activation_state: str
    source_validation_state: str
    slo_profile_id: str
    max_in_flight: int
    warm_resources_ready: bool
    resources_initialized: bool
    runtime_activated: bool
    resource_factory_build_count: int
    warmup_request_count: int
    warmup_arrival_pattern: str
    warmup_candidate_pool_depth: int
    warmup_retrieval_latency_ms: float
    warmup_end_to_end_latency_ms: float
    rejection_reasons: tuple[str, ...]
    registered_as_runtime_default: bool = False
    test_access_allowed: bool = False
    queue_action_count: int = 0
    retry_action_count: int = 0
    fallback_action_count: int = 0

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        forbidden = sorted(_forbidden_keys_found(payload))
        if forbidden:
            raise ValueError(f"concurrent startup trace contains forbidden keys: {forbidden}")
        return payload


@dataclass(frozen=True)
class PrimeQAHybridConcurrentRuntimeBootstrapResult:
    """Only an eligible startup result carries an active concurrent runtime."""

    runtime: PrimeQAHybridConcurrentSidecarAgentRuntime | None
    startup_trace: PublicSafeConcurrentRuntimeStartupTrace
    resource_summary: PrimeQAHybridRuntimeResourceSummary | None
    source_evaluation: ConcurrentRuntimeValidationEvaluation | None


class PrimeQAHybridConcurrentRuntimeBootstrap:
    """One-shot fail-closed application bootstrap for the Stage145 runtime."""

    def __init__(self) -> None:
        self._start_lock = Lock()
        self._started = False

    def start(
        self,
        *,
        settings: ProjectSettings,
        stage145_report: Mapping[str, Any],
        resource_factory: ConcurrentRuntimeSharedResourceFactoryPort,
        warmup_question: PrimeQARuntimeQuery,
        observation_sink: AgentWorkflowObservationSink | None = None,
    ) -> PrimeQAHybridConcurrentRuntimeBootstrapResult:
        with self._start_lock:
            if self._started:
                raise RuntimeError("concurrent runtime bootstrap may run only once per process")
            self._started = True

        activation_requested = settings.enable_concurrent_sidecar_agent
        if not activation_requested:
            return PrimeQAHybridConcurrentRuntimeBootstrapResult(
                runtime=None,
                startup_trace=_inactive_trace(
                    activation_requested=False,
                    activation_state=RuntimeActivationState.DISABLED,
                    source_validation_state="not_evaluated_disabled",
                    rejection_reasons=("explicit_concurrent_activation_not_requested",),
                    resource_factory_build_count=resource_factory.build_count,
                ),
                resource_summary=None,
                source_evaluation=None,
            )

        evidence = concurrent_runtime_validation_evidence_from_stage145(stage145_report)
        source_evaluation = StrictPracticalConcurrentRuntimeValidationPolicy().evaluate(evidence)
        source_reasons = _stage145_source_rejection_reasons(
            report=stage145_report,
            recomputed_evidence=evidence,
            evaluation=source_evaluation,
        )
        rejection_reasons = tuple(
            dict.fromkeys((*source_reasons, *source_evaluation.rejection_reasons))
        )
        if rejection_reasons:
            return PrimeQAHybridConcurrentRuntimeBootstrapResult(
                runtime=None,
                startup_trace=_inactive_trace(
                    activation_requested=True,
                    activation_state=RuntimeActivationState.REJECTED,
                    source_validation_state=ConcurrentRuntimeValidationState.REJECTED.value,
                    rejection_reasons=rejection_reasons,
                    resource_factory_build_count=resource_factory.build_count,
                ),
                resource_summary=None,
                source_evaluation=source_evaluation,
            )

        shared_resources = resource_factory.build_shared()
        if resource_factory.build_count != 1:
            raise RuntimeError("concurrent runtime resources must be built exactly once")
        if asdict(shared_resources.summary) != _EXPECTED_RESOURCES:
            raise RuntimeError("concurrent runtime resource inventory does not match Stage145")

        runtime = create_primeqa_hybrid_concurrent_sidecar_agent_runtime(
            shared_resources=shared_resources,
            observation_sink=observation_sink,
        )
        warmup_run = runtime.run(
            warmup_question,
            arrival_pattern=ConcurrentArrivalPattern.WARMUP,
        )
        warmup_trace = warmup_run.public_safe_trace
        if warmup_trace.candidate_pool_depth != _TARGET_POOL_DEPTH:
            raise RuntimeError("concurrent runtime warmup did not produce the frozen Top400 pool")

        startup_trace = PublicSafeConcurrentRuntimeStartupTrace(
            runtime_mode=concurrent_sidecar_runtime_contract()["runtime_mode"],
            settings_field=_SETTINGS_FIELD,
            environment_flag=_ENVIRONMENT_FLAG,
            activation_requested=True,
            activation_state=RuntimeActivationState.ELIGIBLE.value,
            source_validation_state=source_evaluation.state.value,
            slo_profile_id=_PROFILE_ID,
            max_in_flight=_MAX_IN_FLIGHT,
            warm_resources_ready=True,
            resources_initialized=True,
            runtime_activated=True,
            resource_factory_build_count=resource_factory.build_count,
            warmup_request_count=1,
            warmup_arrival_pattern=warmup_trace.arrival_pattern,
            warmup_candidate_pool_depth=warmup_trace.candidate_pool_depth,
            warmup_retrieval_latency_ms=warmup_trace.retrieval_latency_ms,
            warmup_end_to_end_latency_ms=warmup_trace.end_to_end_latency_ms,
            rejection_reasons=(),
        )
        startup_trace.to_public_dict()
        return PrimeQAHybridConcurrentRuntimeBootstrapResult(
            runtime=runtime,
            startup_trace=startup_trace,
            resource_summary=shared_resources.summary,
            source_evaluation=source_evaluation,
        )


def concurrent_runtime_validation_evidence_from_stage145(
    report: Mapping[str, Any],
) -> ConcurrentRuntimeValidationEvidence:
    """Recompute Stage144 policy evidence from the saved Stage145 aggregate."""

    train = report.get("train_validation") or {}
    overload = report.get("overload_probe") or {}
    dev = report.get("dev_report_only_validation") or {}
    resources = report.get("resource_summary") or {}
    public = report.get("public_safe_contract") or {}
    decision = report.get("decision") or {}
    loaded = report.get("loaded_data_summary") or {}
    contract = report.get("runtime_contract") or {}
    pass_reports = train.get("pass_reports") or {}
    fold_reports = train.get("fold_pattern_repetition_reports") or {}
    pattern_reports = train.get("pattern_pooled_reports") or {}
    combined = train.get("combined_runtime_summary") or {}
    train_latency = (train.get("global_pooled_report") or {}).get(
        "end_to_end_latency_seconds"
    ) or {}
    dev_latency = dev.get("end_to_end_latency_seconds") or {}
    repetitions = train.get("repetitions_per_pattern") or {}
    return ConcurrentRuntimeValidationEvidence(
        profile_id=str(contract.get("slo_profile_id") or ""),
        warm_single_process=(
            report.get("resource_factory_build_count") == 1
            and resources.get("resources_built_or_loaded_per_request") is False
        ),
        max_in_flight=int(contract.get("max_in_flight") or 0),
        synchronized_arrival_schedule_exact=train.get("synchronized_schedule_exact") is True,
        jittered_arrival_schedule_exact=train.get("jittered_schedule_exact") is True,
        synchronized_train_repetitions=int(
            repetitions.get(ConcurrentArrivalPattern.SYNCHRONIZED.value) or 0
        ),
        jittered_train_repetitions=int(
            repetitions.get(ConcurrentArrivalPattern.DETERMINISTIC_JITTER.value) or 0
        ),
        train_accepted_request_count=int(train.get("accepted_request_count") or 0),
        train_fold_count=5 if len(fold_reports) == 30 else 0,
        train_latency_gate_scope_count=int(train.get("latency_gate_scope_count") or 0),
        train_fold_pattern_repetition_gates_passed=(
            len(fold_reports) == 30
            and all(row.get("slo_passed") is True for row in fold_reports.values())
        ),
        train_pass_aggregate_gates_passed=(
            len(pass_reports) == 6
            and all(row.get("end_to_end_slo_passed") is True for row in pass_reports.values())
        ),
        train_pattern_pooled_gates_passed=(
            len(pattern_reports) == 2
            and all(row.get("slo_passed") is True for row in pattern_reports.values())
        ),
        train_global_pooled_gate_passed=(
            (train.get("global_pooled_report") or {}).get("slo_passed") is True
        ),
        train_behavior_invariants_passed=train.get("behavior_invariants_passed") is True,
        train_end_to_end_p95_seconds=_optional_float(train_latency.get("p95")),
        train_end_to_end_p99_seconds=_optional_float(train_latency.get("p99")),
        overload_attempt_count=int(overload.get("attempt_count") or 0),
        overload_admitted_count=int(overload.get("admitted_count") or 0),
        overload_rejected_count=int(overload.get("rejected_count") or 0),
        overload_rejected_before_downstream=(overload.get("rejected_downstream_call_count") == 0),
        overload_error_type=str(overload.get("rejection_error_type") or ""),
        queue_action_count=int(overload.get("queue_action_count") or 0),
        retry_action_count=(
            int(overload.get("retry_action_count") or 0)
            + int(combined.get("retry_action_count") or 0)
            + int(dev.get("retry_action_count") or 0)
        ),
        fallback_action_count=(
            int(overload.get("fallback_action_count") or 0)
            + int(combined.get("fallback_action_count") or 0)
            + int(dev.get("fallback_action_count") or 0)
        ),
        process_resource_inventory_preserved=(
            dict(resources) == _EXPECTED_RESOURCES
            and report.get("resource_factory_build_count") == 1
        ),
        request_local_state_isolated=(
            train.get("cross_request_contamination_count") == 0
            and decision.get("request_local_state_isolation_validated") is True
        ),
        dev_loaded_after_train_gate=(
            loaded.get("dev_loaded_only_after_train_gate") is True
            and report.get("dev_loaded_only_after_train_gate") is True
        ),
        dev_report_only_pass_count=int(dev.get("measured_pass_count") or 0),
        dev_accepted_request_count=int(dev.get("row_count") or 0),
        dev_end_to_end_slo_passed=dev.get("end_to_end_slo_passed") is True,
        dev_behavior_invariants_passed=dev.get("behavior_matches_stage143") is True,
        dev_end_to_end_p95_seconds=_optional_float(dev_latency.get("p95")),
        dev_end_to_end_p99_seconds=_optional_float(dev_latency.get("p99")),
        test_split_locked=(
            public.get("test_split_loaded") is False
            and public.get("test_metrics_run") is False
            and decision.get("test_gate_opened") is False
            and decision.get("test_metrics_run") is False
        ),
        runtime_default_unchanged=(
            decision.get("runtime_registered_as_default") is False
            and decision.get("runtime_defaultization_allowed_now") is False
            and decision.get("default_runtime_policy") == "unchanged"
        ),
    )


def concurrent_runtime_activation_contract() -> dict[str, Any]:
    """Return the explicit disabled-by-default Stage146 activation contract."""

    return {
        "runtime_mode": concurrent_sidecar_runtime_contract()["runtime_mode"],
        "settings_field": _SETTINGS_FIELD,
        "environment_flag": _ENVIRONMENT_FLAG,
        "default_enabled": False,
        "explicit_true_required": True,
        "mutually_exclusive_with": "enable_optional_sidecar_agent",
        "activation_states": [state.value for state in RuntimeActivationState],
        "source_stage": "Stage 145",
        "source_policy": "StrictPracticalConcurrentRuntimeValidationPolicy",
        "max_in_flight": _MAX_IN_FLIGHT,
        "resources_built_for_disabled_or_rejected": False,
        "warmup_required_for_eligible": True,
        "registered_as_runtime_default": False,
        "test_access_allowed": False,
        "queue_actions_allowed": False,
        "retry_actions_allowed": False,
        "fallback_strategies_allowed": False,
        "errors_propagate": True,
    }


def _stage145_source_rejection_reasons(
    *,
    report: Mapping[str, Any],
    recomputed_evidence: ConcurrentRuntimeValidationEvidence,
    evaluation: ConcurrentRuntimeValidationEvaluation,
) -> tuple[str, ...]:
    reasons: list[str] = []
    decision = report.get("decision") or {}
    public = report.get("public_safe_contract") or {}
    checks = report.get("guard_checks") or []
    passed_names = {str(row.get("name")) for row in checks if row.get("passed") is True}
    if report.get("stage") != "Stage 145" or report.get("analysis_id") != _STAGE145_ANALYSIS_ID:
        reasons.append("stage145_source_identity_mismatch")
    if len(checks) != _EXPECTED_STAGE145_GUARDS or len(passed_names) != _EXPECTED_STAGE145_GUARDS:
        reasons.append("stage145_guard_evidence_incomplete")
    if not (
        decision.get("status") == _STAGE145_STATUS
        and decision.get("concurrent_research_runtime_validation_passed") is True
        and decision.get("can_wire_explicit_nondefault_concurrent_runtime_now") is True
    ):
        reasons.append("stage145_wiring_authorization_missing")
    if public.get("forbidden_keys_found") != []:
        reasons.append("stage145_public_safety_failed")
    saved_evidence = report.get("concurrency_policy_evidence") or {}
    if saved_evidence != asdict(recomputed_evidence):
        reasons.append("stage145_saved_evidence_mismatch")
    saved_evaluation = report.get("concurrency_policy_evaluation") or {}
    if not (
        saved_evaluation.get("state") == ConcurrentRuntimeValidationState.ELIGIBLE.value
        and saved_evaluation.get("rejection_reasons") == []
        and saved_evaluation.get("concurrent_runtime_activated") is False
        and evaluation.state is ConcurrentRuntimeValidationState.ELIGIBLE
    ):
        reasons.append("stage145_policy_evaluation_not_eligible")
    return tuple(reasons)


def _inactive_trace(
    *,
    activation_requested: bool,
    activation_state: RuntimeActivationState,
    source_validation_state: str,
    rejection_reasons: tuple[str, ...],
    resource_factory_build_count: int,
) -> PublicSafeConcurrentRuntimeStartupTrace:
    trace = PublicSafeConcurrentRuntimeStartupTrace(
        runtime_mode=concurrent_sidecar_runtime_contract()["runtime_mode"],
        settings_field=_SETTINGS_FIELD,
        environment_flag=_ENVIRONMENT_FLAG,
        activation_requested=activation_requested,
        activation_state=activation_state.value,
        source_validation_state=source_validation_state,
        slo_profile_id=_PROFILE_ID,
        max_in_flight=_MAX_IN_FLIGHT,
        warm_resources_ready=False,
        resources_initialized=False,
        runtime_activated=False,
        resource_factory_build_count=resource_factory_build_count,
        warmup_request_count=0,
        warmup_arrival_pattern="",
        warmup_candidate_pool_depth=0,
        warmup_retrieval_latency_ms=0.0,
        warmup_end_to_end_latency_ms=0.0,
        rejection_reasons=rejection_reasons,
    )
    trace.to_public_dict()
    return trace


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None
