from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest
from pydantic import ValidationError

from ts_rag_agent.application.primeqa_hybrid_online_candidate_pool_retriever import (
    CandidatePoolRetrievalConfig,
    IndependentCandidatePoolSearchChannel,
    PrimeQAHybridOnlineCandidatePoolRetriever,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_entrypoint import (
    FrozenSidecarAgentOrchestratorExecutionFactory,
    PrimeQAHybridOptionalSidecarAgentEntrypoint,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    PrimeQAHybridOptionalSidecarAgentRuntime,
    PrimeQAHybridOptionalSidecarRuntimeBootstrap,
    PrimeQAHybridRuntimeResourceBundle,
    PrimeQAHybridRuntimeResourceSummary,
    _ProfiledCandidatePoolRetriever,
    optional_sidecar_runtime_contract,
    runtime_activation_evidence_from_stage142,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


@dataclass
class RecordingResourceFactory:
    bundle: PrimeQAHybridRuntimeResourceBundle
    build_count: int = 0

    def build(self) -> PrimeQAHybridRuntimeResourceBundle:
        self.build_count += 1
        return self.bundle


class FailIfBuiltResourceFactory:
    def __init__(self) -> None:
        self.build_count = 0

    def build(self) -> PrimeQAHybridRuntimeResourceBundle:
        self.build_count += 1
        raise AssertionError("resources must not be built")


def test_project_settings_optional_runtime_defaults_to_false(monkeypatch) -> None:
    monkeypatch.delenv("TS_RAG_ENABLE_OPTIONAL_SIDECAR_AGENT", raising=False)

    settings = ProjectSettings(_env_file=None)

    assert settings.enable_optional_sidecar_agent is False


@pytest.mark.parametrize(("raw", "expected"), [("true", True), ("false", False)])
def test_project_settings_accepts_only_explicit_boolean_words(
    monkeypatch,
    raw: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("TS_RAG_ENABLE_OPTIONAL_SIDECAR_AGENT", raw)

    assert ProjectSettings(_env_file=None).enable_optional_sidecar_agent is expected


@pytest.mark.parametrize("raw", ["1", "yes", "on", "enabled", ""])
def test_project_settings_rejects_ambiguous_runtime_flag(monkeypatch, raw: str) -> None:
    monkeypatch.setenv("TS_RAG_ENABLE_OPTIONAL_SIDECAR_AGENT", raw)

    with pytest.raises(ValidationError, match="explicit true or false"):
        ProjectSettings(_env_file=None)


def test_runtime_evidence_reads_complete_stage142_aggregate() -> None:
    evidence = runtime_activation_evidence_from_stage142(
        _stage142_report(),
        explicit_activation_requested=True,
        concurrent_request_support_requested=False,
        warm_resources_ready=True,
    )

    assert evidence.source_performance_validated is True
    assert evidence.candidate_pool_identity_preserved is True
    assert evidence.retrieval_recall_preserved is True
    assert evidence.train_fold_count == 5
    assert evidence.train_all_folds_pass is True
    assert evidence.train_p95_seconds == 0.111715
    assert evidence.train_p99_seconds == 0.322262
    assert evidence.dev_report_only_pass is True
    assert evidence.test_split_locked is True


def test_disabled_bootstrap_initializes_no_resources() -> None:
    factory = FailIfBuiltResourceFactory()

    result = PrimeQAHybridOptionalSidecarRuntimeBootstrap().start(
        settings=ProjectSettings(enable_optional_sidecar_agent=False),
        stage142_report={},
        resource_factory=factory,
        warmup_question=_question(),
    )

    assert result.runtime is None
    assert result.resource_summary is None
    assert result.startup_trace.activation_state == "disabled"
    assert result.startup_trace.runtime_activated is False
    assert result.startup_trace.warmup_request_count == 0
    assert factory.build_count == 0


def test_requested_concurrent_bootstrap_is_rejected_before_resources() -> None:
    factory = FailIfBuiltResourceFactory()

    result = PrimeQAHybridOptionalSidecarRuntimeBootstrap().start(
        settings=ProjectSettings(enable_optional_sidecar_agent=True),
        stage142_report=_stage142_report(),
        resource_factory=factory,
        warmup_question=_question(),
        concurrent_request_support_requested=True,
    )

    assert result.runtime is None
    assert result.startup_trace.activation_state == "rejected"
    assert result.startup_trace.rejection_reasons == (
        "concurrent_runtime_not_authorized_by_single_request_protocol",
    )
    assert factory.build_count == 0


def test_invalid_stage142_evidence_is_rejected_before_resources() -> None:
    factory = FailIfBuiltResourceFactory()
    report = _stage142_report()
    report["train_validation"]["combined_latency_seconds"]["p95"] = 0.4

    result = PrimeQAHybridOptionalSidecarRuntimeBootstrap().start(
        settings=ProjectSettings(enable_optional_sidecar_agent=True),
        stage142_report=report,
        resource_factory=factory,
        warmup_question=_question(),
    )

    assert result.runtime is None
    assert result.startup_trace.activation_state == "rejected"
    assert "train_p95_exceeds_strict_slo" in result.startup_trace.rejection_reasons
    assert factory.build_count == 0


def test_eligible_bootstrap_builds_once_warms_and_runs() -> None:
    bundle = _runtime_bundle()
    factory = RecordingResourceFactory(bundle)
    bootstrap = PrimeQAHybridOptionalSidecarRuntimeBootstrap()

    result = bootstrap.start(
        settings=ProjectSettings(enable_optional_sidecar_agent=True),
        stage142_report=_stage142_report(),
        resource_factory=factory,
        warmup_question=_question(),
    )

    assert result.runtime is not None
    assert result.startup_trace.activation_state == "eligible"
    assert result.startup_trace.runtime_activated is True
    assert result.startup_trace.resources_initialized is True
    assert result.startup_trace.warmup_request_count == 1
    assert result.startup_trace.warmup_candidate_pool_depth == 400
    assert factory.build_count == 1

    run = result.runtime.run(_question())
    trace = run.public_safe_trace.to_public_dict()
    assert set(trace) == set(optional_sidecar_runtime_contract()["request_trace_allowed_fields"])
    assert trace["candidate_pool_depth"] == 400
    assert trace["activation_state"] == "eligible"
    assert trace["warm_resources_ready"] is True
    assert trace["terminal_state"] in {"complete", "refuse"}
    assert run.entrypoint_run.public_safe_trace.retry_action_count == 0
    assert run.entrypoint_run.public_safe_trace.fallback_action_count == 0


def test_bootstrap_rejects_a_second_start() -> None:
    bootstrap = PrimeQAHybridOptionalSidecarRuntimeBootstrap()
    factory = FailIfBuiltResourceFactory()
    bootstrap.start(
        settings=ProjectSettings(enable_optional_sidecar_agent=False),
        stage142_report={},
        resource_factory=factory,
        warmup_question=_question(),
    )

    with pytest.raises(RuntimeError, match="only once"):
        bootstrap.start(
            settings=ProjectSettings(enable_optional_sidecar_agent=False),
            stage142_report={},
            resource_factory=factory,
            warmup_question=_question(),
        )


def test_runtime_rejects_concurrent_request_without_queue_or_fallback() -> None:
    runtime = PrimeQAHybridOptionalSidecarAgentRuntime(resources=_runtime_bundle())
    assert runtime._request_lock.acquire(blocking=False) is True
    try:
        with pytest.raises(RuntimeError, match="concurrent"):
            runtime.run(_question())
    finally:
        runtime._request_lock.release()


def _runtime_bundle() -> PrimeQAHybridRuntimeResourceBundle:
    results = _candidate_results()

    def searcher(query: str, top_k: int) -> Sequence[RetrievalResult]:
        _ = query
        return results[:top_k]

    retriever = PrimeQAHybridOnlineCandidatePoolRetriever(
        channels=(
            IndependentCandidatePoolSearchChannel(
                channel_id="test",
                family="test",
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
    profiled = _ProfiledCandidatePoolRetriever(retriever)
    entrypoint = PrimeQAHybridOptionalSidecarAgentEntrypoint(
        candidate_pool_retriever=profiled,
        orchestrator_factory=FrozenSidecarAgentOrchestratorExecutionFactory(min_evidence_score=0.0),
    )
    return PrimeQAHybridRuntimeResourceBundle(
        profiled_retriever=profiled,
        entrypoint=entrypoint,
        summary=PrimeQAHybridRuntimeResourceSummary(
            dense_model_count=2,
            dense_embedding_cache_count=2,
            lexical_index_count=4,
            derived_route_count=1,
            candidate_pool_retriever_instance_count=1,
            optional_entrypoint_instance_count=1,
        ),
    )


def _question() -> PrimeQAQuestion:
    return PrimeQAQuestion(
        id="private-question",
        title="adapter token installation",
        text="How do I repair the adapter token installation failure?",
        answer="Apply the adapter token procedure.",
        answerable=True,
        answer_doc_id="doc-001",
    )


def _candidate_results() -> tuple[RetrievalResult, ...]:
    return tuple(
        RetrievalResult(
            document=PrimeQADocument(
                id=f"doc-{index:03d}",
                title="adapter token installation",
                text=f"Apply the adapter token procedure for configuration {index}.",
            ),
            score=float(np.float64(1 / index)),
            rank=index,
        )
        for index in range(1, 401)
    )


def _stage142_report() -> dict:
    check_names = [
        "all_train_recall_counts_match_stage140",
        "dev_recall_matches_stage140",
        *[f"guard_{index}" for index in range(23)],
    ]
    return {
        "stage": "Stage 142",
        "warmup": {"candidate_pool_exact_identity_violation_count": 0},
        "train_validation": {
            "total_exact_candidate_pool_identity_violation_count": 0,
            "combined_fold_reports": {f"fold_{index}": {} for index in range(1, 6)},
            "all_passes_strict_slo_passed": True,
            "all_pass_folds_strict_slo_passed": True,
            "combined_strict_slo_passed": True,
            "all_combined_folds_strict_slo_passed": True,
            "combined_latency_seconds": {"p95": 0.111715, "p99": 0.322262},
        },
        "dev_report_only_validation": {
            "exact_candidate_pool_identity_violation_count": 0,
            "strict_slo_passed": True,
            "latency_seconds": {"p95": 0.094916, "p99": 0.120182},
        },
        "guard_checks": [{"name": name, "passed": True} for name in check_names],
        "decision": {
            "status": "primeqa_hybrid_strict_warm_latency_validation_passed",
            "strict_slo_validation_passed": True,
            "strict_slo_evidence_state": "eligible",
            "test_gate_opened": False,
        },
        "public_safe_contract": {
            "test_split_loaded": False,
            "test_metrics_run": False,
        },
    }
