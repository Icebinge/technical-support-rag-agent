from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import pytest
from pydantic import ValidationError

from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_activation import (
    PrimeQAHybridConcurrentRuntimeBootstrap,
    concurrent_runtime_activation_contract,
    concurrent_runtime_validation_evidence_from_stage145,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_validation_protocol import (
    ConcurrentRuntimeValidationState,
    StrictPracticalConcurrentRuntimeValidationPolicy,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_sidecar_agent_runtime import (
    ConcurrentArrivalPattern,
)
from ts_rag_agent.application.primeqa_hybrid_online_candidate_pool_retriever import (
    CandidatePoolRetrievalConfig,
    IndependentCandidatePoolSearchChannel,
    PrimeQAHybridOnlineCandidatePoolRetriever,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    PrimeQAHybridRuntimeResourceSummary,
    PrimeQAHybridSharedRuntimeResources,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


@dataclass
class RecordingSharedResourceFactory:
    shared_resources: PrimeQAHybridSharedRuntimeResources
    build_count: int = 0

    def build_shared(self) -> PrimeQAHybridSharedRuntimeResources:
        self.build_count += 1
        return self.shared_resources


class FailIfBuiltSharedResourceFactory:
    def __init__(self) -> None:
        self.build_count = 0

    def build_shared(self) -> PrimeQAHybridSharedRuntimeResources:
        self.build_count += 1
        raise AssertionError("disabled or rejected bootstrap must not build resources")


def test_concurrent_runtime_setting_defaults_to_false(monkeypatch) -> None:
    monkeypatch.delenv("TS_RAG_ENABLE_OPTIONAL_SIDECAR_AGENT", raising=False)
    monkeypatch.delenv("TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT", raising=False)

    settings = ProjectSettings(_env_file=None)

    assert settings.enable_optional_sidecar_agent is False
    assert settings.enable_concurrent_sidecar_agent is False


@pytest.mark.parametrize(("raw", "expected"), [("true", True), ("false", False)])
def test_concurrent_runtime_setting_accepts_only_explicit_boolean_words(
    monkeypatch,
    raw: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT", raw)

    assert ProjectSettings(_env_file=None).enable_concurrent_sidecar_agent is expected


@pytest.mark.parametrize("raw", ["1", "yes", "on", "enabled", ""])
def test_concurrent_runtime_setting_rejects_ambiguous_values(monkeypatch, raw: str) -> None:
    monkeypatch.setenv("TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT", raw)

    with pytest.raises(ValidationError, match="explicit true or false"):
        ProjectSettings(_env_file=None)


def test_single_and_concurrent_runtime_settings_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        ProjectSettings(
            enable_optional_sidecar_agent=True,
            enable_concurrent_sidecar_agent=True,
        )


def test_stage145_evidence_is_recomputed_from_aggregate() -> None:
    report = _eligible_stage145_report()

    evidence = concurrent_runtime_validation_evidence_from_stage145(report)
    evaluation = StrictPracticalConcurrentRuntimeValidationPolicy().evaluate(evidence)

    assert evidence.train_accepted_request_count == 3372
    assert evidence.train_latency_gate_scope_count == 39
    assert evidence.train_end_to_end_p95_seconds == 0.7
    assert evidence.train_end_to_end_p99_seconds == 1.0
    assert evidence.overload_rejected_before_downstream is True
    assert evidence.process_resource_inventory_preserved is True
    assert evidence.test_split_locked is True
    assert evaluation.state is ConcurrentRuntimeValidationState.ELIGIBLE


def test_disabled_bootstrap_builds_no_resources_and_skips_source_evaluation() -> None:
    factory = FailIfBuiltSharedResourceFactory()

    result = PrimeQAHybridConcurrentRuntimeBootstrap().start(
        settings=ProjectSettings(enable_concurrent_sidecar_agent=False),
        stage145_report={},
        resource_factory=factory,
        warmup_question=_question(),
    )

    assert result.runtime is None
    assert result.resource_summary is None
    assert result.source_evaluation is None
    assert result.startup_trace.activation_state == "disabled"
    assert result.startup_trace.source_validation_state == "not_evaluated_disabled"
    assert result.startup_trace.resources_initialized is False
    assert result.startup_trace.runtime_activated is False
    assert factory.build_count == 0


def test_tampered_stage145_evidence_is_rejected_before_resource_build() -> None:
    report = _eligible_stage145_report()
    report["train_validation"]["global_pooled_report"]["end_to_end_latency_seconds"]["p95"] = 0.9
    factory = FailIfBuiltSharedResourceFactory()

    result = PrimeQAHybridConcurrentRuntimeBootstrap().start(
        settings=ProjectSettings(enable_concurrent_sidecar_agent=True),
        stage145_report=report,
        resource_factory=factory,
        warmup_question=_question(),
    )

    assert result.runtime is None
    assert result.startup_trace.activation_state == "rejected"
    assert "stage145_saved_evidence_mismatch" in result.startup_trace.rejection_reasons
    assert "train_end_to_end_p95_exceeds_slo" in result.startup_trace.rejection_reasons
    assert result.startup_trace.resources_initialized is False
    assert factory.build_count == 0


def test_eligible_bootstrap_builds_once_warms_and_returns_concurrent_runtime() -> None:
    factory = RecordingSharedResourceFactory(_shared_resources())

    result = PrimeQAHybridConcurrentRuntimeBootstrap().start(
        settings=ProjectSettings(enable_concurrent_sidecar_agent=True),
        stage145_report=_eligible_stage145_report(),
        resource_factory=factory,
        warmup_question=_question(),
    )

    assert result.runtime is not None
    assert result.source_evaluation is not None
    assert result.source_evaluation.state is ConcurrentRuntimeValidationState.ELIGIBLE
    assert result.startup_trace.activation_state == "eligible"
    assert result.startup_trace.runtime_activated is True
    assert result.startup_trace.resources_initialized is True
    assert result.startup_trace.resource_factory_build_count == 1
    assert result.startup_trace.warmup_request_count == 1
    assert result.startup_trace.warmup_candidate_pool_depth == 400
    assert factory.build_count == 1

    run = result.runtime.run(
        _question(),
        arrival_pattern=ConcurrentArrivalPattern.SYNCHRONIZED,
    )
    assert len(run.candidate_pool_results) == 400
    assert run.public_safe_trace.activation_state == "eligible"


def test_concurrent_bootstrap_is_one_shot() -> None:
    bootstrap = PrimeQAHybridConcurrentRuntimeBootstrap()
    factory = FailIfBuiltSharedResourceFactory()
    bootstrap.start(
        settings=ProjectSettings(enable_concurrent_sidecar_agent=False),
        stage145_report={},
        resource_factory=factory,
        warmup_question=_question(),
    )

    with pytest.raises(RuntimeError, match="only once"):
        bootstrap.start(
            settings=ProjectSettings(enable_concurrent_sidecar_agent=False),
            stage145_report={},
            resource_factory=factory,
            warmup_question=_question(),
        )


def test_concurrent_activation_contract_keeps_closed_boundaries() -> None:
    contract = concurrent_runtime_activation_contract()

    assert contract["settings_field"] == "enable_concurrent_sidecar_agent"
    assert contract["environment_flag"] == "TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT"
    assert contract["default_enabled"] is False
    assert contract["mutually_exclusive_with"] == "enable_optional_sidecar_agent"
    assert contract["max_in_flight"] == 4
    assert contract["registered_as_runtime_default"] is False
    assert contract["test_access_allowed"] is False
    assert contract["queue_actions_allowed"] is False
    assert contract["retry_actions_allowed"] is False
    assert contract["fallback_strategies_allowed"] is False


def _eligible_stage145_report() -> dict:
    pass_reports = {f"pass_{index}": {"end_to_end_slo_passed": True} for index in range(6)}
    fold_reports = {f"fold_{index}": {"slo_passed": True} for index in range(30)}
    pattern_reports = {
        "synchronized_four_request_burst": {"slo_passed": True},
        "deterministic_jitter_0_to_20ms": {"slo_passed": True},
    }
    report = {
        "stage": "Stage 145",
        "analysis_id": "primeqa_hybrid_concurrent_runtime_train_cv_dev_validation_v1",
        "runtime_contract": {
            "slo_profile_id": "strict_practical_b_concurrency4_v1",
            "max_in_flight": 4,
        },
        "resource_summary": asdict(_resource_summary()),
        "resource_factory_build_count": 1,
        "overload_probe": {
            "attempt_count": 5,
            "admitted_count": 4,
            "rejected_count": 1,
            "rejected_downstream_call_count": 0,
            "rejection_error_type": "PrimeQAHybridConcurrentCapacityExceededError",
            "queue_action_count": 0,
            "retry_action_count": 0,
            "fallback_action_count": 0,
        },
        "train_validation": {
            "accepted_request_count": 3372,
            "repetitions_per_pattern": {
                "synchronized_four_request_burst": 3,
                "deterministic_jitter_0_to_20ms": 3,
            },
            "synchronized_schedule_exact": True,
            "jittered_schedule_exact": True,
            "pass_reports": pass_reports,
            "fold_pattern_repetition_reports": fold_reports,
            "pattern_pooled_reports": pattern_reports,
            "global_pooled_report": {
                "slo_passed": True,
                "end_to_end_latency_seconds": {"p95": 0.7, "p99": 1.0},
            },
            "latency_gate_scope_count": 39,
            "behavior_invariants_passed": True,
            "cross_request_contamination_count": 0,
            "combined_runtime_summary": {
                "retry_action_count": 0,
                "fallback_action_count": 0,
            },
        },
        "dev_report_only_validation": {
            "row_count": 121,
            "measured_pass_count": 1,
            "end_to_end_latency_seconds": {"p95": 0.7, "p99": 1.0},
            "end_to_end_slo_passed": True,
            "behavior_matches_stage143": True,
            "retry_action_count": 0,
            "fallback_action_count": 0,
        },
        "loaded_data_summary": {
            "dev_loaded_only_after_train_gate": True,
            "test_split_loaded": False,
        },
        "dev_loaded_only_after_train_gate": True,
        "guard_checks": [
            {"name": f"stage145_guard_{index}", "passed": True} for index in range(36)
        ],
        "decision": {
            "status": "primeqa_hybrid_concurrent_runtime_train_cv_dev_validation_passed",
            "concurrent_research_runtime_validation_passed": True,
            "can_wire_explicit_nondefault_concurrent_runtime_now": True,
            "request_local_state_isolation_validated": True,
            "runtime_registered_as_default": False,
            "runtime_defaultization_allowed_now": False,
            "test_gate_opened": False,
            "test_metrics_run": False,
            "default_runtime_policy": "unchanged",
        },
        "public_safe_contract": {
            "test_split_loaded": False,
            "test_metrics_run": False,
            "forbidden_keys_found": [],
        },
    }
    evidence = concurrent_runtime_validation_evidence_from_stage145(report)
    evaluation = StrictPracticalConcurrentRuntimeValidationPolicy().evaluate(evidence)
    report["concurrency_policy_evidence"] = asdict(evidence)
    report["concurrency_policy_evaluation"] = evaluation.to_public_dict()
    return report


def _shared_resources() -> PrimeQAHybridSharedRuntimeResources:
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
    return PrimeQAHybridSharedRuntimeResources(
        candidate_pool_retriever=retriever,
        summary=_resource_summary(),
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


def _question() -> PrimeQAQuestion:
    return PrimeQAQuestion(
        id="private-stage146-question",
        title="Adapter installation failure",
        text="How do I repair the adapter installation failure?",
        answer="Apply the documented adapter repair procedure.",
        answerable=True,
        answer_doc_id="doc-000",
    )


def _candidate_results() -> tuple[RetrievalResult, ...]:
    return tuple(
        RetrievalResult(
            document=PrimeQADocument(
                id=f"doc-{index:03d}",
                title=f"Adapter repair procedure {index}",
                text="Apply the documented adapter repair procedure and restart the service.",
            ),
            score=float(1 / (index + 1)),
            rank=index + 1,
        )
        for index in range(400)
    )
