from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from ts_rag_agent.application.primeqa_hybrid_agent_retrieval_integration_validation import (
    _DEFAULT_BM25_B,
    _DEFAULT_BM25_K1,
    _DEFAULT_ENCODER_BATCH_SIZE,
    _selected_append_config,
)
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    EncoderFactory,
    _build_dense_channels,
    _build_lexical_channels,
    _load_json_object,
)
from ts_rag_agent.application.primeqa_hybrid_nondefault_runtime_activation_protocol import (
    RuntimeActivationEvaluation,
    RuntimeActivationEvidence,
    RuntimeActivationState,
    StrictNonDefaultRuntimeActivationPolicy,
    StrictWarmLatencySlo,
)
from ts_rag_agent.application.primeqa_hybrid_online_candidate_pool_performance_validation import (
    _online_channels,
    _retrieval_config,
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
from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_orchestrator_validation import (
    _stage128_summary,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.domain.dataset import PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_RUNTIME_MODE = "optional_sidecar_agent_single_request"
_SLO_PROFILE_ID = "strict_c_warm_single_request_v1"
_STAGE142_STATUS = "primeqa_hybrid_strict_warm_latency_validation_passed"
_TARGET_POOL_DEPTH = 400
_EXPECTED_STAGE142_GUARDS = 25
_EXPECTED_TRAIN_FOLDS = 5
_SPECIAL_CHANNEL_ID = "special_token_boosted_bm25"
_ALLOWED_REQUEST_TRACE_FIELDS = frozenset(
    {
        "runtime_mode",
        "activation_requested",
        "activation_state",
        "slo_profile_id",
        "warm_resources_ready",
        "candidate_pool_depth",
        "retrieval_latency_ms",
        "latency_budget_passed",
        "terminal_state",
    }
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
        "retrieved_doc_ids",
        "runtime_content_handle",
        "sample_id",
        "source_doc_ids",
    }
)


class RuntimeResourceFactoryPort(Protocol):
    """Build the process-owned optional runtime resources exactly once."""

    def build(self) -> PrimeQAHybridRuntimeResourceBundle: ...


@dataclass(frozen=True)
class PrimeQAHybridRuntimeResourceSummary:
    """Public-safe inventory of initialized process-owned resources."""

    dense_model_count: int
    dense_embedding_cache_count: int
    lexical_index_count: int
    derived_route_count: int
    candidate_pool_retriever_instance_count: int
    optional_entrypoint_instance_count: int
    resources_built_or_loaded_per_request: bool = False


@dataclass(frozen=True)
class PublicSafeOptionalSidecarRuntimeStartupTrace:
    """Aggregate-only startup result for the optional runtime."""

    runtime_mode: str
    activation_requested: bool
    activation_state: str
    slo_profile_id: str
    concurrent_request_support_requested: bool
    warm_resources_ready: bool
    resources_initialized: bool
    runtime_activated: bool
    warmup_request_count: int
    warmup_candidate_pool_depth: int
    warmup_retrieval_latency_ms: float
    rejection_reasons: tuple[str, ...]
    retry_action_count: int = 0
    fallback_action_count: int = 0

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        _ensure_public_safe(payload)
        return payload


@dataclass(frozen=True)
class PublicSafeOptionalSidecarRuntimeRequestTrace:
    """Stage141 allowlisted public trace for one active runtime request."""

    runtime_mode: str
    activation_requested: bool
    activation_state: str
    slo_profile_id: str
    warm_resources_ready: bool
    candidate_pool_depth: int
    retrieval_latency_ms: float
    latency_budget_passed: bool
    terminal_state: str

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if set(payload) != _ALLOWED_REQUEST_TRACE_FIELDS:
            raise ValueError("runtime request trace fields do not match Stage141 allowlist")
        _ensure_public_safe(payload)
        return payload


@dataclass(frozen=True)
class OptionalSidecarAgentRuntimeRun:
    """Private Agent result paired with its public runtime trace."""

    entrypoint_run: OptionalSidecarAgentEntrypointRun
    public_safe_trace: PublicSafeOptionalSidecarRuntimeRequestTrace

    @property
    def candidate_pool_results(self) -> tuple[RetrievalResult, ...]:
        return self.entrypoint_run.candidate_pool_results

    @property
    def verified_answer(self):
        return self.entrypoint_run.verified_answer


class _ProfiledCandidatePoolRetriever:
    """Expose the existing retriever port while retaining one request profile."""

    def __init__(self, delegate: PrimeQAHybridOnlineCandidatePoolRetriever) -> None:
        self._delegate = delegate
        self._pending_run: CandidatePoolRetrievalRun | None = None

    def prepare_request(self) -> None:
        if self._pending_run is not None:
            raise RuntimeError("previous retrieval profile was not consumed")

    def retrieve(self, question: PrimeQAQuestion) -> Sequence[RetrievalResult]:
        if self._pending_run is not None:
            raise RuntimeError("only one sequential runtime request is authorized")
        self._pending_run = self._delegate.retrieve_profiled(question)
        return self._pending_run.results

    def take_run(self) -> CandidatePoolRetrievalRun:
        if self._pending_run is None:
            raise RuntimeError("entrypoint completed without a retrieval profile")
        run = self._pending_run
        self._pending_run = None
        return run

    def discard_run(self) -> None:
        self._pending_run = None


@dataclass(frozen=True)
class PrimeQAHybridRuntimeResourceBundle:
    """Process-owned retriever, entrypoint, and public resource inventory."""

    profiled_retriever: _ProfiledCandidatePoolRetriever
    entrypoint: PrimeQAHybridOptionalSidecarAgentEntrypoint
    summary: PrimeQAHybridRuntimeResourceSummary


@dataclass(frozen=True)
class PrimeQAHybridSharedRuntimeResources:
    """Long-lived retrieval graph shared by one process runtime assembly."""

    candidate_pool_retriever: PrimeQAHybridOnlineCandidatePoolRetriever
    summary: PrimeQAHybridRuntimeResourceSummary


class PrimeQAHybridOptionalSidecarAgentRuntime:
    """Explicit non-default single-request runtime around the Stage139 entrypoint."""

    def __init__(self, *, resources: PrimeQAHybridRuntimeResourceBundle) -> None:
        self._resources = resources
        self._request_lock = Lock()

    @property
    def resource_summary(self) -> PrimeQAHybridRuntimeResourceSummary:
        return self._resources.summary

    def run(self, question: PrimeQAQuestion) -> OptionalSidecarAgentRuntimeRun:
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("concurrent optional runtime requests are not authorized")
        try:
            self._resources.profiled_retriever.prepare_request()
            try:
                entrypoint_run = self._resources.entrypoint.run(question)
                retrieval_run = self._resources.profiled_retriever.take_run()
            except Exception:
                self._resources.profiled_retriever.discard_run()
                raise
            retrieval_ms = retrieval_run.profile.total_seconds * 1000
            trace = PublicSafeOptionalSidecarRuntimeRequestTrace(
                runtime_mode=_RUNTIME_MODE,
                activation_requested=True,
                activation_state=RuntimeActivationState.ELIGIBLE.value,
                slo_profile_id=_SLO_PROFILE_ID,
                warm_resources_ready=True,
                candidate_pool_depth=len(retrieval_run.results),
                retrieval_latency_ms=round(retrieval_ms, 3),
                latency_budget_passed=retrieval_ms <= StrictWarmLatencySlo().p99_seconds * 1000,
                terminal_state=entrypoint_run.public_safe_trace.terminal_state,
            )
            trace.to_public_dict()
            return OptionalSidecarAgentRuntimeRun(
                entrypoint_run=entrypoint_run,
                public_safe_trace=trace,
            )
        finally:
            self._request_lock.release()


@dataclass(frozen=True)
class PrimeQAHybridOptionalSidecarRuntimeBootstrapResult:
    """Bootstrap outcome; only an eligible result carries an active runtime."""

    runtime: PrimeQAHybridOptionalSidecarAgentRuntime | None
    startup_trace: PublicSafeOptionalSidecarRuntimeStartupTrace
    resource_summary: PrimeQAHybridRuntimeResourceSummary | None


class PrimeQAHybridOptionalSidecarRuntimeBootstrap:
    """One-shot fail-closed process bootstrap for the optional runtime."""

    def __init__(self) -> None:
        self._started = False

    def start(
        self,
        *,
        settings: ProjectSettings,
        stage142_report: Mapping[str, Any],
        resource_factory: RuntimeResourceFactoryPort,
        warmup_question: PrimeQAQuestion,
        concurrent_request_support_requested: bool = False,
    ) -> PrimeQAHybridOptionalSidecarRuntimeBootstrapResult:
        if self._started:
            raise RuntimeError("optional runtime bootstrap may run only once per process")
        self._started = True
        activation_requested = settings.enable_optional_sidecar_agent
        policy = StrictNonDefaultRuntimeActivationPolicy()

        if not activation_requested:
            evaluation = policy.evaluate(_disabled_evidence())
            return _bootstrap_result_without_runtime(
                activation_requested=False,
                concurrent_request_support_requested=concurrent_request_support_requested,
                evaluation=evaluation,
            )

        preflight = policy.evaluate(
            runtime_activation_evidence_from_stage142(
                stage142_report,
                explicit_activation_requested=True,
                concurrent_request_support_requested=concurrent_request_support_requested,
                warm_resources_ready=True,
            )
        )
        if preflight.state is not RuntimeActivationState.ELIGIBLE:
            return _bootstrap_result_without_runtime(
                activation_requested=True,
                concurrent_request_support_requested=concurrent_request_support_requested,
                evaluation=preflight,
            )

        resources = resource_factory.build()
        runtime = PrimeQAHybridOptionalSidecarAgentRuntime(resources=resources)
        final_evaluation = policy.evaluate(
            runtime_activation_evidence_from_stage142(
                stage142_report,
                explicit_activation_requested=True,
                concurrent_request_support_requested=False,
                warm_resources_ready=True,
            )
        )
        if final_evaluation.state is not RuntimeActivationState.ELIGIBLE:
            raise RuntimeError("runtime evidence changed after resource initialization")

        warmup_run = runtime.run(warmup_question)
        warmup_trace = warmup_run.public_safe_trace
        startup_trace = PublicSafeOptionalSidecarRuntimeStartupTrace(
            runtime_mode=_RUNTIME_MODE,
            activation_requested=True,
            activation_state=final_evaluation.state.value,
            slo_profile_id=_SLO_PROFILE_ID,
            concurrent_request_support_requested=False,
            warm_resources_ready=True,
            resources_initialized=True,
            runtime_activated=True,
            warmup_request_count=1,
            warmup_candidate_pool_depth=warmup_trace.candidate_pool_depth,
            warmup_retrieval_latency_ms=warmup_trace.retrieval_latency_ms,
            rejection_reasons=(),
        )
        startup_trace.to_public_dict()
        return PrimeQAHybridOptionalSidecarRuntimeBootstrapResult(
            runtime=runtime,
            startup_trace=startup_trace,
            resource_summary=resources.summary,
        )


class PrimeQAHybridProcessRuntimeResourceFactory:
    """Build the frozen Stage142 retrieval graph and Stage139 entrypoint once."""

    def __init__(
        self,
        *,
        stage128_protocol_path: Path,
        stage125_protocol_path: Path,
        stage80_report_path: Path,
        documents_path: Path,
        encoder_batch_size: int = _DEFAULT_ENCODER_BATCH_SIZE,
        encoder_device: str | None = None,
        encoder_factory: EncoderFactory | None = None,
    ) -> None:
        self._stage128_protocol_path = stage128_protocol_path
        self._stage125_protocol_path = stage125_protocol_path
        self._stage80_report_path = stage80_report_path
        self._documents_path = documents_path
        self._encoder_batch_size = encoder_batch_size
        self._encoder_device = encoder_device
        self._encoder_factory = encoder_factory
        self.build_count = 0

    def build(self) -> PrimeQAHybridRuntimeResourceBundle:
        shared = self.build_shared()
        profiled_retriever = _ProfiledCandidatePoolRetriever(shared.candidate_pool_retriever)
        entrypoint = create_primeqa_hybrid_optional_sidecar_agent_entrypoint(
            candidate_pool_retriever=profiled_retriever,
        )
        return PrimeQAHybridRuntimeResourceBundle(
            profiled_retriever=profiled_retriever,
            entrypoint=entrypoint,
            summary=shared.summary,
        )

    def build_shared(self) -> PrimeQAHybridSharedRuntimeResources:
        """Build the heavy graph once for sequential or concurrent assembly."""

        if self.build_count:
            raise RuntimeError("process runtime resources may be built only once")
        self.build_count += 1
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        stage128 = _load_json_object(self._stage128_protocol_path)
        stage125 = _load_json_object(self._stage125_protocol_path)
        stage80 = _load_json_object(self._stage80_report_path)
        selected_config = _selected_append_config(
            stage125_protocol=stage125,
            stage128_summary=_stage128_summary(stage128),
        )
        documents_by_id = load_primeqa_documents(self._documents_path)
        sections_by_document = load_primeqa_document_sections(self._documents_path)
        documents = list(documents_by_id.values())
        document_ids = tuple(document.id for document in documents)
        dense_channels, dense_summary = _build_dense_channels(
            include_dense_channels=True,
            stage80_report=stage80,
            stage80_report_path=self._stage80_report_path,
            documents=documents,
            document_ids=document_ids,
            encoder_batch_size=self._encoder_batch_size,
            encoder_device=self._encoder_device,
            encoder_factory=self._encoder_factory,
        )
        lexical_channels = _build_lexical_channels(
            documents=documents,
            sections_by_document=sections_by_document,
            bm25_k1=_DEFAULT_BM25_K1,
            bm25_b=_DEFAULT_BM25_B,
            component_depth=int(selected_config["append_generation"]["channel_top_k"]),
        )
        if len(dense_channels) != 2 or dense_summary.get("status") != "dense_channels_ready":
            raise RuntimeError("the frozen two dense runtime channels are not ready")
        independent_lexical = [
            channel for channel in lexical_channels if channel.channel_id != _SPECIAL_CHANNEL_ID
        ]
        if len(independent_lexical) != 4:
            raise RuntimeError("the frozen four lexical indexes are not ready")
        online_retriever = PrimeQAHybridOnlineCandidatePoolRetriever(
            channels=_online_channels(lexical_channels + dense_channels),
            config=_retrieval_config(selected_config),
        )
        summary = PrimeQAHybridRuntimeResourceSummary(
            dense_model_count=2,
            dense_embedding_cache_count=len(dense_summary.get("dense_cache_configs") or []),
            lexical_index_count=4,
            derived_route_count=1,
            candidate_pool_retriever_instance_count=1,
            optional_entrypoint_instance_count=1,
        )
        return PrimeQAHybridSharedRuntimeResources(
            candidate_pool_retriever=online_retriever,
            summary=summary,
        )


def runtime_activation_evidence_from_stage142(
    report: Mapping[str, Any],
    *,
    explicit_activation_requested: bool,
    concurrent_request_support_requested: bool,
    warm_resources_ready: bool,
) -> RuntimeActivationEvidence:
    """Convert the aggregate Stage142 report into Stage141 policy evidence."""

    decision = report.get("decision") or {}
    train = report.get("train_validation") or {}
    dev = report.get("dev_report_only_validation") or {}
    warmup = report.get("warmup") or {}
    public = report.get("public_safe_contract") or {}
    checks = report.get("guard_checks") or []
    passed_checks = {str(row.get("name")) for row in checks if row.get("passed") is True}
    all_guards_passed = (
        len(checks) == _EXPECTED_STAGE142_GUARDS and len(passed_checks) == _EXPECTED_STAGE142_GUARDS
    )
    candidate_identity_preserved = (
        warmup.get("candidate_pool_exact_identity_violation_count") == 0
        and train.get("total_exact_candidate_pool_identity_violation_count") == 0
        and dev.get("exact_candidate_pool_identity_violation_count") == 0
    )
    retrieval_recall_preserved = {
        "all_train_recall_counts_match_stage140",
        "dev_recall_matches_stage140",
    }.issubset(passed_checks)
    combined = train.get("combined_latency_seconds") or {}
    dev_latency = dev.get("latency_seconds") or {}
    return RuntimeActivationEvidence(
        explicit_activation_requested=explicit_activation_requested,
        concurrent_request_support_requested=concurrent_request_support_requested,
        source_performance_validated=(
            report.get("stage") == "Stage 142"
            and decision.get("status") == _STAGE142_STATUS
            and decision.get("strict_slo_validation_passed") is True
            and decision.get("strict_slo_evidence_state") == "eligible"
            and all_guards_passed
        ),
        warm_resources_ready=warm_resources_ready,
        candidate_pool_identity_preserved=candidate_identity_preserved,
        retrieval_recall_preserved=retrieval_recall_preserved,
        train_fold_count=len(train.get("combined_fold_reports") or {}),
        train_all_folds_pass=(
            train.get("all_passes_strict_slo_passed") is True
            and train.get("all_pass_folds_strict_slo_passed") is True
            and train.get("combined_strict_slo_passed") is True
            and train.get("all_combined_folds_strict_slo_passed") is True
        ),
        train_p95_seconds=_optional_float(combined.get("p95")),
        train_p99_seconds=_optional_float(combined.get("p99")),
        dev_report_only_pass=dev.get("strict_slo_passed") is True,
        dev_p95_seconds=_optional_float(dev_latency.get("p95")),
        dev_p99_seconds=_optional_float(dev_latency.get("p99")),
        test_split_locked=(
            public.get("test_split_loaded") is False
            and public.get("test_metrics_run") is False
            and decision.get("test_gate_opened") is False
        ),
    )


def optional_sidecar_runtime_contract() -> dict[str, Any]:
    """Return the explicit Stage143 runtime wiring contract."""

    return {
        "runtime_mode": _RUNTIME_MODE,
        "settings_field": "enable_optional_sidecar_agent",
        "environment_flag": "TS_RAG_ENABLE_OPTIONAL_SIDECAR_AGENT",
        "default_enabled": False,
        "explicit_true_required": True,
        "activation_states": [state.value for state in RuntimeActivationState],
        "process_scoped_resources": {
            "dense_model_count": 2,
            "dense_embedding_cache_count": 2,
            "lexical_index_count": 4,
            "derived_route_count": 1,
            "candidate_pool_retriever_instance_count": 1,
            "optional_entrypoint_instance_count": 1,
            "resources_built_or_loaded_per_request": False,
        },
        "request_trace_allowed_fields": sorted(_ALLOWED_REQUEST_TRACE_FIELDS),
        "single_request_only": True,
        "concurrent_request_support_authorized": False,
        "registered_as_runtime_default": False,
        "test_access_allowed": False,
        "retry_actions_allowed": False,
        "fallback_strategies_allowed": False,
        "errors_propagate": True,
        "request_latency_budget_interpretation": (
            "diagnostic_current_retrieval_latency_at_or_below_strict_p99_numeric_limit; "
            "does_not_replace_aggregate_percentile_validation"
        ),
    }


def _bootstrap_result_without_runtime(
    *,
    activation_requested: bool,
    concurrent_request_support_requested: bool,
    evaluation: RuntimeActivationEvaluation,
) -> PrimeQAHybridOptionalSidecarRuntimeBootstrapResult:
    trace = PublicSafeOptionalSidecarRuntimeStartupTrace(
        runtime_mode=_RUNTIME_MODE,
        activation_requested=activation_requested,
        activation_state=evaluation.state.value,
        slo_profile_id=_SLO_PROFILE_ID,
        concurrent_request_support_requested=concurrent_request_support_requested,
        warm_resources_ready=False,
        resources_initialized=False,
        runtime_activated=False,
        warmup_request_count=0,
        warmup_candidate_pool_depth=0,
        warmup_retrieval_latency_ms=0.0,
        rejection_reasons=evaluation.rejection_reasons,
    )
    trace.to_public_dict()
    return PrimeQAHybridOptionalSidecarRuntimeBootstrapResult(
        runtime=None,
        startup_trace=trace,
        resource_summary=None,
    )


def _disabled_evidence() -> RuntimeActivationEvidence:
    return RuntimeActivationEvidence(
        explicit_activation_requested=False,
        concurrent_request_support_requested=False,
        source_performance_validated=False,
        warm_resources_ready=False,
        candidate_pool_identity_preserved=False,
        retrieval_recall_preserved=False,
        train_fold_count=0,
        train_all_folds_pass=False,
        train_p95_seconds=None,
        train_p99_seconds=None,
        dev_report_only_pass=False,
        dev_p95_seconds=None,
        dev_p99_seconds=None,
        test_split_locked=True,
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _ensure_public_safe(payload: Mapping[str, Any]) -> None:
    forbidden = sorted(_forbidden_keys_found(payload))
    if forbidden:
        raise ValueError(f"runtime public trace contains forbidden keys: {forbidden}")


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
