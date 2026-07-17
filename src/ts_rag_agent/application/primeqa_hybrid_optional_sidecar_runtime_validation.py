from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    EncoderFactory,
    _build_train_fold_assignments,
    _load_json_object,
)
from ts_rag_agent.application.primeqa_hybrid_nondefault_runtime_activation_protocol import (
    RuntimeActivationState,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_entrypoint_validation import (
    _entrypoint_trace_contract_violation_count,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    OptionalSidecarAgentRuntimeRun,
    PrimeQAHybridOptionalSidecarRuntimeBootstrap,
    PrimeQAHybridProcessRuntimeResourceFactory,
    PrimeQAHybridRuntimeResourceBundle,
    _forbidden_keys_found,
    optional_sidecar_runtime_contract,
)
from ts_rag_agent.application.primeqa_hybrid_strict_latency_validation import (
    _distribution,
    _strict_latency_pass,
)
from ts_rag_agent.application.rag_answering import evaluate_answers
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.domain.answer import GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQAQuestion
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
    summarize_primeqa_hybrid_split_samples,
)

_STAGE = "Stage 143"
_CREATED_AT = "2026-07-17"
_ANALYSIS_ID = "primeqa_hybrid_optional_sidecar_runtime_wiring_validation_v1"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_TRAIN_FOLD_COUNT = 5
_TARGET_POOL_DEPTH = 400
_TOP_K_VALUES = (10, 50, 100, 200, 400)
_STAGE141_STATUS = "primeqa_hybrid_nondefault_runtime_activation_protocol_frozen"
_STAGE142_STATUS = "primeqa_hybrid_strict_warm_latency_validation_passed"
_STAGE139_STATUS = "primeqa_hybrid_optional_sidecar_agent_entrypoint_train_cv_dev_validation_passed"
_NEXT_DIRECTION = "design_and_confirm_concurrent_runtime_validation_protocol"


@dataclass(frozen=True)
class _RuntimeValidationObservation:
    sample: PrimeQAHybridSplitSample
    verified_answer: GeneratedAnswer
    candidate_doc_ids: tuple[str, ...]
    retrieval_latency_seconds: float
    runtime_trace: Mapping[str, Any]
    entrypoint_trace: Mapping[str, Any]
    runtime_trace_violation_count: int
    entrypoint_trace_violation_count: int


@dataclass(frozen=True)
class PrimeQAHybridOptionalRuntimeValidationVisualization:
    name: str
    path: str


class _NeverBuildResourceFactory:
    def __init__(self) -> None:
        self.build_count = 0

    def build(self) -> PrimeQAHybridRuntimeResourceBundle:
        self.build_count += 1
        raise RuntimeError("disabled or rejected runtime must not initialize resources")


def run_primeqa_hybrid_optional_sidecar_runtime_validation(
    *,
    stage141_protocol_path: Path,
    stage142_validation_path: Path,
    stage139_regression_path: Path,
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
    """Validate real Stage143 optional runtime startup and train-first requests."""

    started_at = time.perf_counter()
    stage141 = _load_json_object(stage141_protocol_path)
    stage142 = _load_json_object(stage142_validation_path)
    stage139 = _load_json_object(stage139_regression_path)
    loaded_sources_at = time.perf_counter()
    source_files = _source_files(
        stage141_protocol=stage141_protocol_path,
        stage142_validation=stage142_validation_path,
        stage139_regression=stage139_regression_path,
        stage128_protocol=stage128_protocol_path,
        stage125_protocol=stage125_protocol_path,
        stage80_report=stage80_report_path,
        train_split=train_split_path,
        dev_split=dev_split_path,
        documents=documents_path,
    )
    pre_checks = _source_checks(
        stage141=stage141,
        stage142=stage142,
        stage139=stage139,
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
    )
    if not all(check["passed"] for check in pre_checks):
        return _blocked_report(
            source_files=source_files,
            guard_checks=pre_checks,
            timing_seconds={
                "load_public_sources": round(loaded_sources_at - started_at, 3),
                "total": round(time.perf_counter() - started_at, 3),
            },
        )

    train_samples = load_primeqa_hybrid_split_samples(train_split_path)
    train_fold_assignments = _build_train_fold_assignments(
        train_samples,
        fold_count=_TRAIN_FOLD_COUNT,
    )
    train_loaded_at = time.perf_counter()
    placeholder = _label_free_question(train_samples[0])

    disabled_factory = _NeverBuildResourceFactory()
    disabled = PrimeQAHybridOptionalSidecarRuntimeBootstrap().start(
        settings=ProjectSettings(enable_optional_sidecar_agent=False),
        stage142_report=stage142,
        resource_factory=disabled_factory,
        warmup_question=placeholder,
    )
    rejected_factory = _NeverBuildResourceFactory()
    rejected = PrimeQAHybridOptionalSidecarRuntimeBootstrap().start(
        settings=ProjectSettings(enable_optional_sidecar_agent=True),
        stage142_report=stage142,
        resource_factory=rejected_factory,
        warmup_question=placeholder,
        concurrent_request_support_requested=True,
    )
    startup_cases_at = time.perf_counter()

    resource_factory = PrimeQAHybridProcessRuntimeResourceFactory(
        stage128_protocol_path=stage128_protocol_path,
        stage125_protocol_path=stage125_protocol_path,
        stage80_report_path=stage80_report_path,
        documents_path=documents_path,
        encoder_batch_size=encoder_batch_size,
        encoder_device=encoder_device,
        encoder_factory=encoder_factory,
    )
    warmup_sample = min(
        train_samples,
        key=lambda row: hashlib.sha256(f"{_ANALYSIS_ID}:{row.sample_id}".encode()).hexdigest(),
    )
    eligible = PrimeQAHybridOptionalSidecarRuntimeBootstrap().start(
        settings=ProjectSettings(enable_optional_sidecar_agent=True),
        stage142_report=stage142,
        resource_factory=resource_factory,
        warmup_question=_label_free_question(warmup_sample),
    )
    resources_ready_at = time.perf_counter()
    if eligible.runtime is None:
        raise RuntimeError("eligible Stage143 bootstrap did not return an active runtime")

    train_observations = [
        _observe_runtime_request(
            sample=sample, runtime_run=eligible.runtime.run(sample.to_primeqa_question())
        )
        for sample in train_samples
    ]
    train_evaluated_at = time.perf_counter()
    train_report = _summarize_split(train_observations)
    train_folds = _summarize_train_folds(
        observations=train_observations,
        fold_assignments=train_fold_assignments,
    )
    train_gate_passed = _train_gate_passed(
        train_report=train_report,
        train_folds=train_folds,
        expected_recall=_source_recall_counts(stage142, _TRAIN_SPLIT),
        expected_agent=_source_agent_summary(stage139, _TRAIN_SPLIT),
    )

    dev_samples: Sequence[PrimeQAHybridSplitSample] = ()
    dev_observations: Sequence[_RuntimeValidationObservation] = ()
    dev_report: Mapping[str, Any] = {}
    dev_loaded_at = train_evaluated_at
    dev_evaluated_at = train_evaluated_at
    if train_gate_passed:
        dev_samples = load_primeqa_hybrid_split_samples(dev_split_path)
        dev_loaded_at = time.perf_counter()
        dev_observations = [
            _observe_runtime_request(
                sample=sample,
                runtime_run=eligible.runtime.run(sample.to_primeqa_question()),
            )
            for sample in dev_samples
        ]
        dev_evaluated_at = time.perf_counter()
        dev_report = _summarize_split(dev_observations)

    startup_cases = {
        "disabled": disabled.startup_trace.to_public_dict(),
        "rejected": rejected.startup_trace.to_public_dict(),
        "eligible": eligible.startup_trace.to_public_dict(),
    }
    resource_summary = asdict(eligible.resource_summary) if eligible.resource_summary else {}
    validation_payload = {
        "startup_cases": startup_cases,
        "resource_summary": resource_summary,
        "resource_factory_build_count": resource_factory.build_count,
        "disabled_resource_factory_build_count": disabled_factory.build_count,
        "rejected_resource_factory_build_count": rejected_factory.build_count,
        "train_runtime_validation": train_report,
        "train_fold_reports": train_folds,
        "train_gate_passed_before_dev": train_gate_passed,
        "dev_loaded_only_after_train_gate": bool(dev_samples) and train_gate_passed,
        "dev_runtime_report_only_validation": dev_report,
    }
    checks = pre_checks + _validation_checks(
        payload=validation_payload,
        stage142=stage142,
        stage139=stage139,
        dev_row_count=len(dev_samples),
    )
    checked_at = time.perf_counter()
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Explicit disabled-by-default single-request runtime wiring validation. It "
            "validates disabled and rejected startup without resource initialization, "
            "then builds eligible process-scoped resources once, executes one label-free "
            "train warmup, one complete train five-fold integrity pass, and one dev "
            "report-only pass after the train gate. Public output is aggregate only. Test, "
            "concurrent serving, defaultization, retries, and fallback remain closed."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_validation),
            "confirmation_note": confirmation_note,
        },
        "source_files": source_files,
        "runtime_contract": optional_sidecar_runtime_contract(),
        "source_stage141": _source_stage_summary(stage141),
        "source_stage142": _source_stage_summary(stage142),
        "source_stage139_regression": _source_stage_summary(stage139),
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
        **validation_payload,
        "guard_checks": checks,
        "decision": _decision(checks, train_gate_passed=train_gate_passed),
        "timing_seconds": {
            "load_public_sources": round(loaded_sources_at - started_at, 3),
            "load_train_and_build_folds": round(train_loaded_at - loaded_sources_at, 3),
            "validate_disabled_rejected_startup": round(startup_cases_at - train_loaded_at, 3),
            "build_resources_and_eligible_warmup": round(resources_ready_at - startup_cases_at, 3),
            "run_complete_train_runtime_pass": round(train_evaluated_at - resources_ready_at, 3),
            "load_dev_after_train_gate": round(dev_loaded_at - train_evaluated_at, 3),
            "run_dev_report_only_runtime_pass": round(dev_evaluated_at - dev_loaded_at, 3),
            "summarize_and_guard": round(checked_at - dev_evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def _observe_runtime_request(
    *,
    sample: PrimeQAHybridSplitSample,
    runtime_run: OptionalSidecarAgentRuntimeRun,
) -> _RuntimeValidationObservation:
    runtime_trace = runtime_run.public_safe_trace.to_public_dict()
    entrypoint_trace = runtime_run.entrypoint_run.public_safe_trace.to_public_dict()
    return _RuntimeValidationObservation(
        sample=sample,
        verified_answer=runtime_run.verified_answer,
        candidate_doc_ids=tuple(
            result.document.id for result in runtime_run.candidate_pool_results
        ),
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
    )


def _summarize_split(
    observations: Sequence[_RuntimeValidationObservation],
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
    latencies = [observation.retrieval_latency_seconds for observation in observations]
    exact_entrypoint = sum(
        observation.entrypoint_trace_violation_count == 0 for observation in observations
    )
    return {
        "row_count": len(observations),
        "answerable_count": len(answerable),
        "runtime_request_trace_violation_count": sum(
            observation.runtime_trace_violation_count for observation in observations
        ),
        "entrypoint_trace_violation_count": sum(
            observation.entrypoint_trace_violation_count for observation in observations
        ),
        "exact_five_transition_trace_count": exact_entrypoint,
        "exact_five_transition_trace_rate": _ratio(exact_entrypoint, len(observations)),
        "candidate_pool_depth": _distribution(
            [len(observation.candidate_doc_ids) for observation in observations]
        ),
        "retrieval_latency_seconds": _distribution(latencies),
        "strict_retrieval_slo_passed": _strict_latency_pass(_distribution(latencies)),
        "latency_budget_failed_request_count": sum(
            observation.runtime_trace.get("latency_budget_passed") is not True
            for observation in observations
        ),
        "terminal_state_counts": {
            terminal: sum(
                observation.runtime_trace.get("terminal_state") == terminal
                for observation in observations
            )
            for terminal in ("complete", "refuse")
        },
        "retry_action_count": sum(
            int(observation.entrypoint_trace.get("retry_action_count") or 0)
            for observation in observations
        ),
        "fallback_action_count": sum(
            int(observation.entrypoint_trace.get("fallback_action_count") or 0)
            for observation in observations
        ),
        "recall": {
            "hit_counts": hit_counts,
            "rates": {key: _ratio(value, len(answerable)) for key, value in hit_counts.items()},
        },
        "verified_metrics": metrics,
        "verified_gold_citation_count": gold_citations,
    }


def _summarize_train_folds(
    *,
    observations: Sequence[_RuntimeValidationObservation],
    fold_assignments: Mapping[str, str],
) -> dict[str, Any]:
    grouped: dict[str, list[_RuntimeValidationObservation]] = defaultdict(list)
    for observation in observations:
        grouped[fold_assignments[observation.sample.sample_id]].append(observation)
    return {
        fold_id: {
            "row_count": summary["row_count"],
            "runtime_request_trace_violation_count": summary[
                "runtime_request_trace_violation_count"
            ],
            "entrypoint_trace_violation_count": summary["entrypoint_trace_violation_count"],
            "exact_five_transition_trace_rate": summary["exact_five_transition_trace_rate"],
            "candidate_pool_depth": summary["candidate_pool_depth"],
            "retrieval_latency_seconds": summary["retrieval_latency_seconds"],
            "strict_retrieval_slo_passed": summary["strict_retrieval_slo_passed"],
            "retry_action_count": summary["retry_action_count"],
            "fallback_action_count": summary["fallback_action_count"],
        }
        for fold_id, rows in sorted(grouped.items())
        for summary in [_summarize_split(rows)]
    }


def _train_gate_passed(
    *,
    train_report: Mapping[str, Any],
    train_folds: Mapping[str, Mapping[str, Any]],
    expected_recall: Mapping[str, int],
    expected_agent: Mapping[str, Any],
) -> bool:
    return (
        train_report.get("row_count") == 562
        and len(train_folds) == _TRAIN_FOLD_COUNT
        and _split_integrity_passed(train_report)
        and all(_fold_integrity_passed(fold) for fold in train_folds.values())
        and (train_report.get("recall") or {}).get("hit_counts") == dict(expected_recall)
        and _agent_summary_matches(train_report, expected_agent)
    )


def _split_integrity_passed(report: Mapping[str, Any]) -> bool:
    pool = report.get("candidate_pool_depth") or {}
    return (
        report.get("runtime_request_trace_violation_count") == 0
        and report.get("entrypoint_trace_violation_count") == 0
        and report.get("exact_five_transition_trace_rate") == 1.0
        and pool.get("min") == _TARGET_POOL_DEPTH
        and pool.get("max") == _TARGET_POOL_DEPTH
        and report.get("strict_retrieval_slo_passed") is True
        and report.get("latency_budget_failed_request_count") == 0
        and report.get("retry_action_count") == 0
        and report.get("fallback_action_count") == 0
    )


def _fold_integrity_passed(report: Mapping[str, Any]) -> bool:
    pool = report.get("candidate_pool_depth") or {}
    return (
        report.get("runtime_request_trace_violation_count") == 0
        and report.get("entrypoint_trace_violation_count") == 0
        and report.get("exact_five_transition_trace_rate") == 1.0
        and pool.get("min") == _TARGET_POOL_DEPTH
        and pool.get("max") == _TARGET_POOL_DEPTH
        and report.get("strict_retrieval_slo_passed") is True
        and report.get("retry_action_count") == 0
        and report.get("fallback_action_count") == 0
    )


def _agent_summary_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return (
        (actual.get("verified_metrics") or {}).get("average_token_f1")
        == expected.get("verified_average_token_f1")
        and actual.get("verified_gold_citation_count")
        == expected.get("verified_gold_citation_count")
        and actual.get("terminal_state_counts") == expected.get("terminal_state_counts")
        and actual.get("exact_five_transition_trace_rate")
        == expected.get("exact_five_transition_trace_rate")
    )


def _runtime_trace_violation_count(
    *,
    runtime_trace: Mapping[str, Any],
    entrypoint_trace: Mapping[str, Any],
) -> int:
    expected_fields = set(optional_sidecar_runtime_contract()["request_trace_allowed_fields"])
    checks = [
        set(runtime_trace) == expected_fields,
        runtime_trace.get("runtime_mode") == "optional_sidecar_agent_single_request",
        runtime_trace.get("activation_requested") is True,
        runtime_trace.get("activation_state") == RuntimeActivationState.ELIGIBLE.value,
        runtime_trace.get("slo_profile_id") == "strict_c_warm_single_request_v1",
        runtime_trace.get("warm_resources_ready") is True,
        runtime_trace.get("candidate_pool_depth") == _TARGET_POOL_DEPTH,
        float(runtime_trace.get("retrieval_latency_ms") or -1) >= 0,
        isinstance(runtime_trace.get("latency_budget_passed"), bool),
        runtime_trace.get("terminal_state") == entrypoint_trace.get("terminal_state"),
        _forbidden_keys_found(runtime_trace) == set(),
    ]
    return sum(not passed for passed in checks)


def _source_checks(
    *,
    stage141: Mapping[str, Any],
    stage142: Mapping[str, Any],
    stage139: Mapping[str, Any],
    user_confirmed_validation: bool,
    confirmation_note: str,
) -> list[dict[str, Any]]:
    stage141_checks = stage141.get("guard_checks") or []
    stage142_checks = stage142.get("guard_checks") or []
    stage139_checks = stage139.get("guard_checks") or []
    stage141_decision = stage141.get("decision") or {}
    stage142_decision = stage142.get("decision") or {}
    stage139_decision = stage139.get("decision") or {}
    runtime_interface = (stage141.get("frozen_protocol") or {}).get("runtime_interface") or {}
    return [
        _check(
            "stage143_user_confirmed",
            user_confirmed_validation and "Stage143" in confirmation_note,
            {"confirmed": user_confirmed_validation, "note_present": bool(confirmation_note)},
            "explicit Stage143 confirmation",
        ),
        _check(
            "stage141_protocol_source_passed",
            stage141_decision.get("status") == _STAGE141_STATUS
            and len(stage141_checks) == 19
            and all(row.get("passed") is True for row in stage141_checks),
            {"status": stage141_decision.get("status"), "guards": len(stage141_checks)},
            {"status": _STAGE141_STATUS, "guards": 19},
        ),
        _check(
            "stage141_runtime_interface_matches_implementation",
            runtime_interface.get("future_environment_flag")
            == "TS_RAG_ENABLE_OPTIONAL_SIDECAR_AGENT"
            and runtime_interface.get("default_value") is False
            and runtime_interface.get("explicit_true_required") is True,
            runtime_interface,
            "explicit disabled-by-default Stage141 interface",
        ),
        _check(
            "stage142_strict_latency_source_passed",
            stage142_decision.get("status") == _STAGE142_STATUS
            and len(stage142_checks) == 25
            and all(row.get("passed") is True for row in stage142_checks),
            {"status": stage142_decision.get("status"), "guards": len(stage142_checks)},
            {"status": _STAGE142_STATUS, "guards": 25},
        ),
        _check(
            "stage142_authorizes_nondefault_wiring_only",
            stage142_decision.get("can_implement_nondefault_runtime_wiring_now") is True
            and stage142_decision.get("runtime_activated_now") is False
            and stage142_decision.get("concurrent_runtime_activation_allowed") is False
            and stage142_decision.get("runtime_defaultization_allowed_now") is False,
            stage142_decision,
            "wiring eligible; activation/default/concurrency closed",
        ),
        _check(
            "stage139_final_agent_regression_passed",
            stage139_decision.get("status") == _STAGE139_STATUS
            and len(stage139_checks) == 45
            and all(row.get("passed") is True for row in stage139_checks),
            {"status": stage139_decision.get("status"), "guards": len(stage139_checks)},
            {"status": _STAGE139_STATUS, "guards": 45},
        ),
        _check(
            "source_reports_are_public_safe_and_test_locked",
            all(
                (report.get("public_safe_contract") or {}).get("forbidden_keys_found") == []
                and (report.get("public_safe_contract") or {}).get("test_split_loaded") is False
                for report in (stage142, stage139)
            ),
            "Stage142 and Stage139 public contracts",
            "forbidden empty and test false",
        ),
    ]


def _validation_checks(
    *,
    payload: Mapping[str, Any],
    stage142: Mapping[str, Any],
    stage139: Mapping[str, Any],
    dev_row_count: int,
) -> list[dict[str, Any]]:
    startup = payload.get("startup_cases") or {}
    disabled = startup.get("disabled") or {}
    rejected = startup.get("rejected") or {}
    eligible = startup.get("eligible") or {}
    resources = payload.get("resource_summary") or {}
    train = payload.get("train_runtime_validation") or {}
    folds = payload.get("train_fold_reports") or {}
    dev = payload.get("dev_runtime_report_only_validation") or {}
    expected_train_recall = _source_recall_counts(stage142, _TRAIN_SPLIT)
    expected_dev_recall = _source_recall_counts(stage142, _DEV_SPLIT)
    expected_train_agent = _source_agent_summary(stage139, _TRAIN_SPLIT)
    expected_dev_agent = _source_agent_summary(stage139, _DEV_SPLIT)
    checks = [
        _check(
            "runtime_contract_is_explicit_and_nondefault",
            optional_sidecar_runtime_contract().get("default_enabled") is False
            and optional_sidecar_runtime_contract().get("explicit_true_required") is True
            and optional_sidecar_runtime_contract().get("registered_as_runtime_default") is False,
            optional_sidecar_runtime_contract(),
            "explicit true, nondefault",
        ),
        _check(
            "disabled_startup_initializes_nothing",
            disabled.get("activation_state") == "disabled"
            and disabled.get("resources_initialized") is False
            and disabled.get("runtime_activated") is False
            and disabled.get("warmup_request_count") == 0
            and payload.get("disabled_resource_factory_build_count") == 0,
            disabled,
            "disabled without resources or warmup",
        ),
        _check(
            "concurrent_startup_is_rejected_before_resources",
            rejected.get("activation_state") == "rejected"
            and rejected.get("resources_initialized") is False
            and rejected.get("runtime_activated") is False
            and tuple(rejected.get("rejection_reasons") or ())
            == ("concurrent_runtime_not_authorized_by_single_request_protocol",)
            and payload.get("rejected_resource_factory_build_count") == 0,
            rejected,
            "rejected before resource initialization",
        ),
        _check(
            "eligible_startup_activates_after_one_warmup",
            eligible.get("activation_state") == "eligible"
            and eligible.get("resources_initialized") is True
            and eligible.get("warm_resources_ready") is True
            and eligible.get("runtime_activated") is True
            and eligible.get("warmup_request_count") == 1
            and eligible.get("warmup_candidate_pool_depth") == _TARGET_POOL_DEPTH,
            eligible,
            "eligible and active after one warmup",
        ),
        _check(
            "process_resources_match_frozen_inventory_and_build_once",
            resources
            == {
                "dense_model_count": 2,
                "dense_embedding_cache_count": 2,
                "lexical_index_count": 4,
                "derived_route_count": 1,
                "candidate_pool_retriever_instance_count": 1,
                "optional_entrypoint_instance_count": 1,
                "resources_built_or_loaded_per_request": False,
            }
            and payload.get("resource_factory_build_count") == 1,
            {"resources": resources, "build_count": payload.get("resource_factory_build_count")},
            "frozen inventory built exactly once",
        ),
        _check(
            "startup_traces_are_public_safe",
            all(_forbidden_keys_found(row) == set() for row in startup.values()),
            {name: sorted(_forbidden_keys_found(row)) for name, row in startup.items()},
            [],
        ),
        _check(
            "complete_train_runtime_pass_and_five_folds",
            train.get("row_count") == 562 and len(folds) == _TRAIN_FOLD_COUNT,
            {"rows": train.get("row_count"), "folds": len(folds)},
            {"rows": 562, "folds": 5},
        ),
        _check(
            "train_runtime_and_entrypoint_traces_are_exact",
            train.get("runtime_request_trace_violation_count") == 0
            and train.get("entrypoint_trace_violation_count") == 0
            and train.get("exact_five_transition_trace_rate") == 1.0
            and all(_fold_integrity_passed(fold) for fold in folds.values()),
            {
                "runtime": train.get("runtime_request_trace_violation_count"),
                "entrypoint": train.get("entrypoint_trace_violation_count"),
                "rate": train.get("exact_five_transition_trace_rate"),
            },
            "zero violations and exact five transitions",
        ),
        _check(
            "train_candidate_pool_depth_is_400",
            (train.get("candidate_pool_depth") or {}).get("min") == _TARGET_POOL_DEPTH
            and (train.get("candidate_pool_depth") or {}).get("max") == _TARGET_POOL_DEPTH,
            train.get("candidate_pool_depth"),
            _TARGET_POOL_DEPTH,
        ),
        _check(
            "train_recall_matches_stage142",
            (train.get("recall") or {}).get("hit_counts") == dict(expected_train_recall),
            (train.get("recall") or {}).get("hit_counts"),
            expected_train_recall,
        ),
        _check(
            "train_runtime_retrieval_and_all_folds_meet_strict_slo",
            train.get("strict_retrieval_slo_passed") is True
            and all(fold.get("strict_retrieval_slo_passed") is True for fold in folds.values()),
            {
                "train": train.get("retrieval_latency_seconds"),
                "folds": {
                    name: fold.get("retrieval_latency_seconds") for name, fold in folds.items()
                },
            },
            "P95 <= 0.3 and P99 <= 1.0",
        ),
        _check(
            "train_agent_metrics_match_stage139",
            _agent_summary_matches(train, expected_train_agent),
            {
                "f1": (train.get("verified_metrics") or {}).get("average_token_f1"),
                "gold": train.get("verified_gold_citation_count"),
                "terminal": train.get("terminal_state_counts"),
            },
            expected_train_agent,
        ),
        _check(
            "train_gate_passes_before_dev_load",
            payload.get("train_gate_passed_before_dev") is True
            and payload.get("dev_loaded_only_after_train_gate") is True,
            {
                "train_gate": payload.get("train_gate_passed_before_dev"),
                "dev_after_gate": payload.get("dev_loaded_only_after_train_gate"),
            },
            True,
        ),
        _check(
            "dev_runs_once_report_only",
            dev_row_count == 121 and dev.get("row_count") == 121,
            {"loaded": dev_row_count, "measured": dev.get("row_count")},
            121,
        ),
        _check(
            "dev_runtime_and_entrypoint_traces_are_exact",
            dev.get("runtime_request_trace_violation_count") == 0
            and dev.get("entrypoint_trace_violation_count") == 0
            and dev.get("exact_five_transition_trace_rate") == 1.0,
            {
                "runtime": dev.get("runtime_request_trace_violation_count"),
                "entrypoint": dev.get("entrypoint_trace_violation_count"),
                "rate": dev.get("exact_five_transition_trace_rate"),
            },
            "zero violations and exact five transitions",
        ),
        _check(
            "dev_candidate_pool_depth_is_400",
            (dev.get("candidate_pool_depth") or {}).get("min") == _TARGET_POOL_DEPTH
            and (dev.get("candidate_pool_depth") or {}).get("max") == _TARGET_POOL_DEPTH,
            dev.get("candidate_pool_depth"),
            _TARGET_POOL_DEPTH,
        ),
        _check(
            "dev_recall_matches_stage142",
            (dev.get("recall") or {}).get("hit_counts") == dict(expected_dev_recall),
            (dev.get("recall") or {}).get("hit_counts"),
            expected_dev_recall,
        ),
        _check(
            "dev_runtime_retrieval_meets_strict_slo",
            dev.get("strict_retrieval_slo_passed") is True,
            dev.get("retrieval_latency_seconds"),
            "P95 <= 0.3 and P99 <= 1.0",
        ),
        _check(
            "dev_agent_metrics_match_stage139",
            _agent_summary_matches(dev, expected_dev_agent),
            {
                "f1": (dev.get("verified_metrics") or {}).get("average_token_f1"),
                "gold": dev.get("verified_gold_citation_count"),
                "terminal": dev.get("terminal_state_counts"),
            },
            expected_dev_agent,
        ),
        _check(
            "all_runtime_requests_have_no_retry_or_fallback",
            train.get("retry_action_count") == 0
            and train.get("fallback_action_count") == 0
            and dev.get("retry_action_count") == 0
            and dev.get("fallback_action_count") == 0,
            {
                "train_retry": train.get("retry_action_count"),
                "train_fallback": train.get("fallback_action_count"),
                "dev_retry": dev.get("retry_action_count"),
                "dev_fallback": dev.get("fallback_action_count"),
            },
            0,
        ),
        _check(
            "test_default_and_concurrency_boundaries_remain_closed",
            optional_sidecar_runtime_contract().get("test_access_allowed") is False
            and optional_sidecar_runtime_contract().get("registered_as_runtime_default") is False
            and optional_sidecar_runtime_contract().get("concurrent_request_support_authorized")
            is False,
            optional_sidecar_runtime_contract(),
            False,
        ),
    ]
    return checks


def _source_recall_counts(report: Mapping[str, Any], split: str) -> dict[str, int]:
    if split == _TRAIN_SPLIT:
        passes = (report.get("train_validation") or {}).get("pass_reports") or []
        recall = (passes[0].get("recall") or {}) if passes else {}
    else:
        recall = (report.get("dev_report_only_validation") or {}).get("recall") or {}
    return {str(key): int(value) for key, value in (recall.get("hit_counts") or {}).items()}


def _source_agent_summary(report: Mapping[str, Any], split: str) -> dict[str, Any]:
    source = (report.get("split_entrypoint_reports") or {}).get(split) or {}
    return {
        "verified_average_token_f1": (source.get("agent_verified_metrics") or {}).get(
            "average_token_f1"
        ),
        "verified_gold_citation_count": source.get("agent_verified_gold_citation_count"),
        "terminal_state_counts": {
            "complete": source.get("complete_terminal_count"),
            "refuse": source.get("refuse_terminal_count"),
        },
        "exact_five_transition_trace_rate": source.get("exact_five_transition_trace_rate"),
    }


def _label_free_question(sample: PrimeQAHybridSplitSample) -> PrimeQAQuestion:
    return PrimeQAQuestion(
        id="runtime-warmup",
        title=sample.question_title,
        text=sample.question_text,
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
        "test_metrics_run": public.get("test_metrics_run")
        if "test_metrics_run" in public
        else public.get("final_test_metrics_run"),
        "forbidden_keys_found": public.get("forbidden_keys_found"),
    }


def _decision(
    checks: Sequence[Mapping[str, Any]],
    *,
    train_gate_passed: bool,
) -> dict[str, Any]:
    failed = [str(check["name"]) for check in checks if check.get("passed") is not True]
    passed = not failed and train_gate_passed
    return {
        "status": (
            "primeqa_hybrid_optional_sidecar_runtime_wiring_validation_passed"
            if passed
            else "primeqa_hybrid_optional_sidecar_runtime_wiring_validation_blocked"
        ),
        "failed_checks": failed,
        "optional_runtime_wiring_implemented": True,
        "optional_runtime_activation_validated": passed,
        "disabled_rejected_eligible_startup_validated": passed,
        "process_scoped_resources_validated": passed,
        "single_request_runtime_validated": passed,
        "runtime_registered_as_default": False,
        "runtime_defaultization_allowed_now": False,
        "concurrent_runtime_activation_allowed": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
        "retry_actions_enabled": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_direction": _NEXT_DIRECTION
        if passed
        else "repair_stage143_runtime_wiring",
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "aggregate_only": True,
        "private_per_request_traces_written": False,
        "raw_questions_written": False,
        "raw_answers_written": False,
        "raw_documents_written": False,
        "raw_candidate_rows_written": False,
        "raw_sample_or_document_ids_written": False,
        "train_cv_group_values_written": False,
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
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": "Stage143 source preflight blocked before resource initialization.",
        "source_files": dict(source_files),
        "startup_cases": {},
        "resource_factory_build_count": 0,
        "train_runtime_validation": {},
        "train_fold_reports": {},
        "train_gate_passed_before_dev": False,
        "dev_runtime_report_only_validation": {},
        "loaded_data_summary": {
            "train": {"row_count": 0},
            "dev": {"row_count": 0},
            "test_split_loaded": False,
        },
        "guard_checks": list(guard_checks),
        "decision": _decision(guard_checks, train_gate_passed=False),
        "timing_seconds": dict(timing_seconds),
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_optional_sidecar_runtime_validation_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridOptionalRuntimeValidationVisualization]:
    output_dir.mkdir(parents=True, exist_ok=True)
    startup = report.get("startup_cases") or {}
    resources = report.get("resource_summary") or {}
    train = report.get("train_runtime_validation") or {}
    dev = report.get("dev_runtime_report_only_validation") or {}
    folds = report.get("train_fold_reports") or {}
    charts = {
        "stage143_startup_states.svg": render_horizontal_bar_chart_svg(
            title="Stage143 explicit runtime startup states",
            bars=[
                BarDatum(
                    label=name,
                    value={"disabled": 0.0, "rejected": 0.5, "eligible": 1.0}.get(
                        str(row.get("activation_state")),
                        -1.0,
                    ),
                    value_label=str(row.get("activation_state")),
                )
                for name, row in startup.items()
            ],
            x_label="disabled 0, rejected 0.5, eligible 1",
            width=1280,
            margin_left=480,
        ),
        "stage143_process_resource_inventory.svg": render_horizontal_bar_chart_svg(
            title="Stage143 process-scoped runtime resource inventory",
            bars=[
                BarDatum(label=key, value=float(value), value_label=str(value))
                for key, value in resources.items()
                if isinstance(value, int) and not isinstance(value, bool)
            ],
            x_label="initialized instance count",
            width=1560,
            margin_left=760,
        ),
        "stage143_train_fold_latency.svg": render_horizontal_bar_chart_svg(
            title="Stage143 train fold runtime retrieval P95/P99",
            bars=[
                BarDatum(
                    label=f"{fold_id} {percentile}",
                    value=_latency(row, percentile),
                    value_label=_seconds(row, percentile),
                )
                for fold_id, row in folds.items()
                for percentile in ("p95", "p99")
            ],
            x_label="seconds",
            width=1500,
            margin_left=600,
        ),
        "stage143_split_latency_vs_slo.svg": render_horizontal_bar_chart_svg(
            title="Stage143 runtime retrieval latency versus strict SLO",
            bars=[
                BarDatum(
                    label="train P95",
                    value=_latency(train, "p95"),
                    value_label=_seconds(train, "p95"),
                ),
                BarDatum(
                    label="train P99",
                    value=_latency(train, "p99"),
                    value_label=_seconds(train, "p99"),
                ),
                BarDatum(
                    label="dev P95", value=_latency(dev, "p95"), value_label=_seconds(dev, "p95")
                ),
                BarDatum(
                    label="dev P99", value=_latency(dev, "p99"), value_label=_seconds(dev, "p99")
                ),
                BarDatum(label="strict P95", value=0.3, value_label="0.300000s"),
                BarDatum(label="strict P99", value=1.0, value_label="1.000000s"),
            ],
            x_label="seconds",
            width=1360,
            margin_left=520,
        ),
        "stage143_recall_at_k.svg": render_horizontal_bar_chart_svg(
            title="Stage143 runtime candidate recall",
            bars=[
                BarDatum(
                    label=f"{split} Recall@{top_k}",
                    value=_recall_rate(row, top_k),
                    value_label=f"{_recall_rate(row, top_k):.4f}",
                )
                for split, row in ((_TRAIN_SPLIT, train), (_DEV_SPLIT, dev))
                for top_k in _TOP_K_VALUES
            ],
            x_label="gold-document recall",
            width=1500,
            margin_left=640,
        ),
        "stage143_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage143 runtime wiring decision flags",
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
            width=1900,
            margin_left=980,
        ),
        "stage143_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage143 runtime wiring guard checks",
            bars=[
                BarDatum(
                    label=str(check.get("name")),
                    value=1.0 if check.get("passed") else 0.0,
                    value_label="passed" if check.get("passed") else "failed",
                )
                for check in report.get("guard_checks") or []
            ],
            x_label="1 means passed",
            width=2280,
            margin_left=1340,
        ),
    }
    artifacts = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        artifacts.append(PrimeQAHybridOptionalRuntimeValidationVisualization(name, str(path)))
    return artifacts


def _latency(report: Mapping[str, Any], percentile: str) -> float:
    return float((report.get("retrieval_latency_seconds") or {}).get(percentile) or 0)


def _seconds(report: Mapping[str, Any], percentile: str) -> str:
    return f"{_latency(report, percentile):.6f}s"


def _recall_rate(report: Mapping[str, Any], top_k: int) -> float:
    return float(((report.get("recall") or {}).get("rates") or {}).get(str(top_k)) or 0)


def _source_files(**paths: Path) -> dict[str, Any]:
    return {
        name: {"path": str(path), "exists": path.is_file(), "size_bytes": path.stat().st_size}
        for name, path in paths.items()
    }


def _check(name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "expected": expected}


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
