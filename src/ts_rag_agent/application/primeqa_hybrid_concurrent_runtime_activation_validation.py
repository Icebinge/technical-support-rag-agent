from __future__ import annotations

import copy
import hashlib
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_activation import (
    PrimeQAHybridConcurrentRuntimeBootstrap,
    concurrent_runtime_activation_contract,
    concurrent_runtime_validation_evidence_from_stage145,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_validation import (
    _build_dev_report,
    _build_train_report,
    _counter_delta,
    _pass_id,
    _PassExecution,
    _policy_evidence,
    _run_complete_pass,
    _run_mixed_dev_pass,
    _run_overload_probe,
    _source_files,
    _train_gate_passed,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_validation import (
    _validation_checks as _stage145_runtime_checks,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_validation_protocol import (
    ConcurrentRuntimeValidationState,
    StrictPracticalConcurrentRuntimeValidationPolicy,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_sidecar_agent_runtime import (
    ConcurrentArrivalPattern,
    concurrent_sidecar_runtime_contract,
)
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    EncoderFactory,
    _build_train_fold_assignments,
    _load_json_object,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    PrimeQAHybridProcessRuntimeResourceFactory,
    PrimeQAHybridSharedRuntimeResources,
    _forbidden_keys_found,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.domain.dataset import PrimeQAQuestion
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
    summarize_primeqa_hybrid_split_samples,
)

_STAGE = "Stage 146"
_CREATED_AT = "2026-07-17"
_ANALYSIS_ID = "primeqa_hybrid_concurrent_runtime_application_activation_validation_v1"
_STAGE145_STATUS = "primeqa_hybrid_concurrent_runtime_train_cv_dev_validation_passed"
_STAGE145_ANALYSIS_ID = "primeqa_hybrid_concurrent_runtime_train_cv_dev_validation_v1"
_EXPECTED_STAGE145_GUARDS = 36
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_TRAIN_FOLD_COUNT = 5
_MAX_IN_FLIGHT = 4
_REPETITIONS_PER_PATTERN = 3
_NEXT_DIRECTION = "freeze_nondefault_application_agent_request_facade_protocol"


@dataclass(frozen=True)
class PrimeQAHybridConcurrentActivationValidationVisualization:
    name: str
    path: str


class _NeverBuildSharedResourceFactory:
    def __init__(self) -> None:
        self.build_count = 0

    def build_shared(self) -> PrimeQAHybridSharedRuntimeResources:
        self.build_count += 1
        raise RuntimeError("disabled or rejected application bootstrap built resources")


def run_primeqa_hybrid_concurrent_runtime_activation_validation(
    *,
    stage145_validation_path: Path,
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
    """Validate the explicit non-default application bootstrap on real workload."""

    started_at = time.perf_counter()
    stage145 = _load_json_object(stage145_validation_path)
    source_files = _source_files(
        stage145_validation=stage145_validation_path,
        stage128_protocol=stage128_protocol_path,
        stage125_protocol=stage125_protocol_path,
        stage80_report=stage80_report_path,
        train_split=train_split_path,
        dev_split=dev_split_path,
        documents=documents_path,
    )
    source_evidence = concurrent_runtime_validation_evidence_from_stage145(stage145)
    source_evaluation = StrictPracticalConcurrentRuntimeValidationPolicy().evaluate(source_evidence)
    pre_checks = _source_checks(
        stage145=stage145,
        source_evidence=asdict(source_evidence),
        source_evaluation=source_evaluation.to_public_dict(),
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
    )
    sources_loaded_at = time.perf_counter()
    if not all(check["passed"] for check in pre_checks):
        return _blocked_report(
            source_files=source_files,
            guard_checks=pre_checks,
            timing_seconds={
                "load_and_validate_public_sources": round(sources_loaded_at - started_at, 3),
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

    disabled_factory = _NeverBuildSharedResourceFactory()
    disabled = PrimeQAHybridConcurrentRuntimeBootstrap().start(
        settings=ProjectSettings(enable_concurrent_sidecar_agent=False),
        stage145_report=stage145,
        resource_factory=disabled_factory,
        warmup_question=placeholder,
    )

    synthetic_rejected_source = copy.deepcopy(stage145)
    synthetic_rejected_source["train_validation"]["global_pooled_report"][
        "end_to_end_latency_seconds"
    ]["p95"] = 0.800001
    rejected_factory = _NeverBuildSharedResourceFactory()
    rejected = PrimeQAHybridConcurrentRuntimeBootstrap().start(
        settings=ProjectSettings(enable_concurrent_sidecar_agent=True),
        stage145_report=synthetic_rejected_source,
        resource_factory=rejected_factory,
        warmup_question=placeholder,
    )
    source_unchanged_after_synthetic_case = stage145 == _load_json_object(stage145_validation_path)

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
    eligible = PrimeQAHybridConcurrentRuntimeBootstrap().start(
        settings=ProjectSettings(enable_concurrent_sidecar_agent=True),
        stage145_report=stage145,
        resource_factory=resource_factory,
        warmup_question=_label_free_question(warmup_sample),
    )
    startup_finished_at = time.perf_counter()
    if eligible.runtime is None or eligible.resource_summary is None:
        raise RuntimeError("eligible Stage146 bootstrap did not return an active runtime")
    runtime = eligible.runtime

    overload_report = _run_overload_probe(runtime=runtime, samples=train_samples[:4])
    overload_finished_at = time.perf_counter()

    train_counters_before = runtime.counters()
    pass_executions: list[_PassExecution] = []
    pass_timing_seconds: dict[str, float] = {}
    with ThreadPoolExecutor(
        max_workers=_MAX_IN_FLIGHT,
        thread_name_prefix="stage146-train",
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

    baseline_adapter = _stage145_baseline_adapter(stage145)
    expected_train = baseline_adapter["train_runtime_validation"]
    train_report = _build_train_report(
        executions=pass_executions,
        fold_assignments=train_fold_assignments,
        expected_stage143=expected_train,
        counter_delta=_counter_delta(train_counters_before, train_counters_after),
    )
    train_gate_passed = _train_gate_passed(train_report)

    dev_samples: Sequence[PrimeQAHybridSplitSample] = ()
    dev_report: Mapping[str, Any] = {}
    dev_loaded_at = train_finished_at
    dev_finished_at = train_finished_at
    if train_gate_passed:
        dev_samples = load_primeqa_hybrid_split_samples(dev_split_path)
        dev_loaded_at = time.perf_counter()
        dev_counters_before = runtime.counters()
        with ThreadPoolExecutor(
            max_workers=_MAX_IN_FLIGHT,
            thread_name_prefix="stage146-dev",
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
            expected_stage143=baseline_adapter["dev_runtime_report_only_validation"],
            counter_delta=_counter_delta(dev_counters_before, dev_counters_after),
        )

    resource_summary = asdict(eligible.resource_summary)
    current_policy_evidence = _policy_evidence(
        train_report=train_report,
        overload_report=overload_report,
        dev_report=dev_report,
        dev_loaded_after_train_gate=bool(dev_samples) and train_gate_passed,
        resource_summary=resource_summary,
        resource_factory_build_count=resource_factory.build_count,
    )
    current_policy_evaluation = StrictPracticalConcurrentRuntimeValidationPolicy().evaluate(
        current_policy_evidence
    )
    startup_cases = {
        "disabled": disabled.startup_trace.to_public_dict(),
        "rejected": rejected.startup_trace.to_public_dict(),
        "eligible": eligible.startup_trace.to_public_dict(),
    }
    payload = {
        "activation_contract": concurrent_runtime_activation_contract(),
        "runtime_contract": concurrent_sidecar_runtime_contract(),
        "configuration_mutual_exclusion_enforced": _configuration_mutual_exclusion_enforced(),
        "startup_cases": startup_cases,
        "synthetic_rejected_case": {
            "purpose": "fail-closed pre-resource startup validation",
            "mutation": "in-memory train global P95 set to 0.800001s",
            "source_report_modified": False,
            "persisted": False,
        },
        "source_unchanged_after_synthetic_case": source_unchanged_after_synthetic_case,
        "resource_summary": resource_summary,
        "resource_factory_build_count": resource_factory.build_count,
        "disabled_resource_factory_build_count": disabled_factory.build_count,
        "rejected_resource_factory_build_count": rejected_factory.build_count,
        "warmup": {
            "request_count": eligible.startup_trace.warmup_request_count,
            "candidate_pool_depth": eligible.startup_trace.warmup_candidate_pool_depth,
            "runtime_trace": {
                "arrival_pattern": eligible.startup_trace.warmup_arrival_pattern,
                "retrieval_latency_ms": eligible.startup_trace.warmup_retrieval_latency_ms,
                "end_to_end_latency_ms": eligible.startup_trace.warmup_end_to_end_latency_ms,
            },
        },
        "overload_probe": overload_report,
        "train_validation": train_report,
        "train_gate_passed_before_dev": train_gate_passed,
        "dev_loaded_only_after_train_gate": bool(dev_samples) and train_gate_passed,
        "dev_report_only_validation": dev_report,
        "concurrency_policy_evidence": asdict(current_policy_evidence),
        "concurrency_policy_evaluation": current_policy_evaluation.to_public_dict(),
    }
    checks = [
        *pre_checks,
        *_activation_checks(payload=payload),
        *_stage145_runtime_checks(
            payload=payload,
            stage143=baseline_adapter,
            dev_row_count=len(dev_samples),
        ),
    ]
    checked_at = time.perf_counter()
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Implementation and current-machine validation of the explicit, disabled-by-"
            "default application bootstrap for the Stage145 concurrency-four runtime. "
            "Disabled and synthetic rejected startup cases build no resources. The eligible "
            "case recomputes Stage145 evidence, builds shared resources once, warms once, "
            "and runs the complete Stage145 train/overload/dev workload through the returned "
            "application runtime. Test, defaults, network serving, queues, retries, and "
            "fallback remain closed."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_validation),
            "confirmation_note": confirmation_note,
        },
        "source_files": source_files,
        "source_stage145": _source_stage145_summary(stage145),
        "source_policy_evidence": asdict(source_evidence),
        "source_policy_evaluation": source_evaluation.to_public_dict(),
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
        "decision": _decision(
            checks=checks,
            current_policy_state=current_policy_evaluation.state,
            eligible_startup_active=eligible.startup_trace.runtime_activated,
        ),
        "timing_seconds": {
            "load_and_validate_public_sources": round(sources_loaded_at - started_at, 3),
            "load_train_and_build_folds": round(train_loaded_at - sources_loaded_at, 3),
            "run_startup_cases_build_resources_and_warmup": round(
                startup_finished_at - train_loaded_at,
                3,
            ),
            "run_overload_probe": round(overload_finished_at - startup_finished_at, 3),
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


def _source_checks(
    *,
    stage145: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    source_evaluation: Mapping[str, Any],
    user_confirmed_validation: bool,
    confirmation_note: str,
) -> list[dict[str, Any]]:
    checks = stage145.get("guard_checks") or []
    decision = stage145.get("decision") or {}
    public = stage145.get("public_safe_contract") or {}
    return [
        _check(
            "stage146_user_confirmed",
            user_confirmed_validation and "Stage146" in confirmation_note,
            {"confirmed": user_confirmed_validation, "note_present": bool(confirmation_note)},
            "explicit Stage146 confirmation",
        ),
        _check(
            "stage145_source_identity_and_guards_passed",
            stage145.get("stage") == "Stage 145"
            and stage145.get("analysis_id") == _STAGE145_ANALYSIS_ID
            and decision.get("status") == _STAGE145_STATUS
            and len(checks) == _EXPECTED_STAGE145_GUARDS
            and sum(row.get("passed") is True for row in checks) == _EXPECTED_STAGE145_GUARDS,
            {
                "stage": stage145.get("stage"),
                "analysis_id": stage145.get("analysis_id"),
                "status": decision.get("status"),
                "guards": len(checks),
            },
            "Stage145 final 36-guard result",
        ),
        _check(
            "stage145_explicit_wiring_authorization_is_present",
            decision.get("concurrent_research_runtime_validation_passed") is True
            and decision.get("can_wire_explicit_nondefault_concurrent_runtime_now") is True,
            decision,
            True,
        ),
        _check(
            "stage145_source_policy_recomputes_as_eligible",
            source_evaluation.get("state") == ConcurrentRuntimeValidationState.ELIGIBLE.value
            and source_evaluation.get("rejection_reasons") == []
            and source_evaluation.get("concurrent_runtime_activated") is False,
            source_evaluation,
            "eligible without source-side activation",
        ),
        _check(
            "stage145_saved_and_recomputed_evidence_match",
            stage145.get("concurrency_policy_evidence") == dict(source_evidence),
            {"matches": stage145.get("concurrency_policy_evidence") == dict(source_evidence)},
            True,
        ),
        _check(
            "stage145_saved_policy_evaluation_is_eligible",
            stage145.get("concurrency_policy_evaluation") == dict(source_evaluation),
            stage145.get("concurrency_policy_evaluation"),
            source_evaluation,
        ),
        _check(
            "stage145_test_default_queue_retry_fallback_are_closed",
            public.get("test_split_loaded") is False
            and public.get("test_metrics_run") is False
            and public.get("forbidden_keys_found") == []
            and decision.get("runtime_registered_as_default") is False
            and decision.get("queue_actions_enabled") is False
            and decision.get("retry_actions_enabled") is False
            and decision.get("fallback_strategies_enabled") is False,
            {
                "public": public,
                "default": decision.get("runtime_registered_as_default"),
                "queue": decision.get("queue_actions_enabled"),
                "retry": decision.get("retry_actions_enabled"),
                "fallback": decision.get("fallback_strategies_enabled"),
            },
            "all closed",
        ),
    ]


def _activation_checks(*, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    contract = payload.get("activation_contract") or {}
    startup = payload.get("startup_cases") or {}
    disabled = startup.get("disabled") or {}
    rejected = startup.get("rejected") or {}
    eligible = startup.get("eligible") or {}
    return [
        _check(
            "concurrent_application_setting_is_explicit_default_off_and_mutually_exclusive",
            contract.get("settings_field") == "enable_concurrent_sidecar_agent"
            and contract.get("environment_flag") == "TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT"
            and contract.get("default_enabled") is False
            and contract.get("explicit_true_required") is True
            and contract.get("mutually_exclusive_with") == "enable_optional_sidecar_agent"
            and payload.get("configuration_mutual_exclusion_enforced") is True,
            {
                "contract": contract,
                "enforced": payload.get("configuration_mutual_exclusion_enforced"),
            },
            "independent explicit true and mutual exclusion",
        ),
        _check(
            "disabled_startup_builds_no_resources",
            disabled.get("activation_state") == "disabled"
            and disabled.get("activation_requested") is False
            and disabled.get("source_validation_state") == "not_evaluated_disabled"
            and disabled.get("resources_initialized") is False
            and disabled.get("runtime_activated") is False
            and disabled.get("warmup_request_count") == 0
            and payload.get("disabled_resource_factory_build_count") == 0,
            disabled,
            "disabled with zero resource build",
        ),
        _check(
            "tampered_source_startup_is_rejected_before_resources",
            rejected.get("activation_state") == "rejected"
            and rejected.get("activation_requested") is True
            and rejected.get("resources_initialized") is False
            and rejected.get("runtime_activated") is False
            and rejected.get("warmup_request_count") == 0
            and payload.get("rejected_resource_factory_build_count") == 0
            and "stage145_saved_evidence_mismatch" in (rejected.get("rejection_reasons") or [])
            and "train_end_to_end_p95_exceeds_slo" in (rejected.get("rejection_reasons") or []),
            rejected,
            "rejected with zero resource build",
        ),
        _check(
            "synthetic_rejection_does_not_mutate_or_persist_source",
            payload.get("source_unchanged_after_synthetic_case") is True
            and (payload.get("synthetic_rejected_case") or {}).get("source_report_modified")
            is False
            and (payload.get("synthetic_rejected_case") or {}).get("persisted") is False,
            payload.get("synthetic_rejected_case"),
            "in-memory copy only",
        ),
        _check(
            "eligible_startup_activates_after_one_build_and_warmup",
            eligible.get("activation_state") == "eligible"
            and eligible.get("source_validation_state") == "eligible"
            and eligible.get("resources_initialized") is True
            and eligible.get("runtime_activated") is True
            and eligible.get("resource_factory_build_count") == 1
            and eligible.get("warmup_request_count") == 1
            and eligible.get("warmup_arrival_pattern") == ConcurrentArrivalPattern.WARMUP.value
            and eligible.get("warmup_candidate_pool_depth") == 400,
            eligible,
            "eligible, one build, one Top400 warmup",
        ),
        _check(
            "all_startup_traces_keep_default_test_queue_retry_fallback_closed",
            all(
                row.get("registered_as_runtime_default") is False
                and row.get("test_access_allowed") is False
                and row.get("queue_action_count") == 0
                and row.get("retry_action_count") == 0
                and row.get("fallback_action_count") == 0
                for row in startup.values()
            ),
            startup,
            "all closed",
        ),
        _check(
            "activation_contract_exposes_no_default_test_queue_retry_or_fallback",
            contract.get("registered_as_runtime_default") is False
            and contract.get("test_access_allowed") is False
            and contract.get("queue_actions_allowed") is False
            and contract.get("retry_actions_allowed") is False
            and contract.get("fallback_strategies_allowed") is False,
            contract,
            False,
        ),
        _check(
            "startup_payload_is_public_safe",
            _forbidden_keys_found(startup) == set(),
            sorted(_forbidden_keys_found(startup)),
            [],
        ),
    ]


def _stage145_baseline_adapter(stage145: Mapping[str, Any]) -> dict[str, Any]:
    train = stage145.get("train_validation") or {}
    pass_reports = train.get("pass_reports") or {}
    order = train.get("pass_execution_order") or []
    first_pass = dict(pass_reports.get(order[0]) or {}) if order else {}
    return {
        "train_runtime_validation": first_pass,
        "dev_runtime_report_only_validation": dict(
            stage145.get("dev_report_only_validation") or {}
        ),
    }


def _decision(
    *,
    checks: Sequence[Mapping[str, Any]],
    current_policy_state: ConcurrentRuntimeValidationState,
    eligible_startup_active: bool,
) -> dict[str, Any]:
    failed = [str(check["name"]) for check in checks if check.get("passed") is not True]
    passed = (
        not failed
        and current_policy_state is ConcurrentRuntimeValidationState.ELIGIBLE
        and eligible_startup_active
    )
    return {
        "status": (
            "primeqa_hybrid_concurrent_runtime_application_activation_validation_passed"
            if passed
            else "primeqa_hybrid_concurrent_runtime_application_activation_validation_blocked"
        ),
        "failed_checks": failed,
        "application_activation_bootstrap_implemented": True,
        "disabled_rejected_eligible_startup_validated": passed,
        "eligible_runtime_full_workload_validation_passed": passed,
        "explicit_nondefault_concurrent_activation_available": passed,
        "concurrent_runtime_activation_allowed_now": passed,
        "activation_requires_explicit_true_and_stage145_evidence": True,
        "single_and_concurrent_runtime_flags_mutually_exclusive": True,
        "runtime_registered_as_default": False,
        "runtime_defaultization_allowed_now": False,
        "network_service_implemented": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
        "queue_actions_enabled": False,
        "retry_actions_enabled": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_direction": (
            _NEXT_DIRECTION if passed else "repair_stage146_application_activation_validation"
        ),
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    loaded = report.get("loaded_data_summary") or {}
    train = loaded.get("train") or {}
    dev = loaded.get("dev") or {}
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
        "synthetic_rejected_source_persisted": False,
        "train_split_loaded": bool(train.get("row_count")),
        "dev_split_loaded": bool(dev.get("row_count")),
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
        "analysis_scope": "Blocked before loading train or building application resources.",
        "source_files": dict(source_files),
        "guard_checks": list(guard_checks),
        "decision": {
            "status": "primeqa_hybrid_concurrent_runtime_application_activation_validation_blocked",
            "failed_checks": failed,
            "application_activation_bootstrap_implemented": True,
            "explicit_nondefault_concurrent_activation_available": False,
            "concurrent_runtime_activation_allowed_now": False,
            "runtime_registered_as_default": False,
            "runtime_defaultization_allowed_now": False,
            "network_service_implemented": False,
            "test_gate_opened": False,
            "test_metrics_run": False,
            "queue_actions_enabled": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
            "recommended_next_direction": "repair_stage146_source_guards",
        },
        "timing_seconds": dict(timing_seconds),
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_concurrent_runtime_activation_validation_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridConcurrentActivationValidationVisualization]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = report.get("train_validation") or {}
    dev = report.get("dev_report_only_validation") or {}
    overload = report.get("overload_probe") or {}
    charts = {
        "stage146_startup_states.svg": render_horizontal_bar_chart_svg(
            title="Stage146 application startup states",
            bars=_startup_state_bars(report),
            x_label="state code",
            width=1500,
            margin_left=700,
        ),
        "stage146_startup_resource_builds.svg": render_horizontal_bar_chart_svg(
            title="Stage146 resource builds by startup case",
            bars=_startup_build_bars(report),
            x_label="build count",
            width=1500,
            margin_left=700,
        ),
        "stage146_train_pattern_latency.svg": render_horizontal_bar_chart_svg(
            title="Stage146 train pattern-pooled end-to-end latency",
            bars=_train_pattern_latency_bars(train),
            x_label="seconds",
            width=1740,
            margin_left=840,
        ),
        "stage146_train_scope_maxima.svg": render_horizontal_bar_chart_svg(
            title="Stage146 maximum P95 and P99 across 39 scopes",
            bars=_scope_maximum_bars(train),
            x_label="seconds",
            width=1740,
            margin_left=840,
        ),
        "stage146_dev_latency.svg": render_horizontal_bar_chart_svg(
            title="Stage146 dev report-only end-to-end latency",
            bars=_latency_bars(dev.get("end_to_end_latency_seconds") or {}, "dev"),
            x_label="seconds",
            width=1500,
            margin_left=700,
        ),
        "stage146_overload_outcome.svg": render_horizontal_bar_chart_svg(
            title="Stage146 five-request overload outcome",
            bars=[
                _bar("attempted", overload.get("attempt_count") or 0),
                _bar("admitted", overload.get("admitted_count") or 0),
                _bar("rejected", overload.get("rejected_count") or 0),
                _bar(
                    "rejected downstream calls",
                    overload.get("rejected_downstream_call_count") or 0,
                ),
            ],
            x_label="requests",
            width=1500,
            margin_left=720,
        ),
        "stage146_arrival_fidelity.svg": render_horizontal_bar_chart_svg(
            title="Stage146 train arrival-offset absolute error",
            bars=_arrival_error_bars(train),
            x_label="milliseconds",
            width=1640,
            margin_left=760,
        ),
        "stage146_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage146 activation decision boundaries",
            bars=_decision_bars(report),
            x_label="false=0 true=1",
            width=1760,
            margin_left=880,
        ),
        "stage146_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage146 formal guard status",
            bars=[
                _bar(
                    "passed guards",
                    sum(row.get("passed") is True for row in report.get("guard_checks") or []),
                ),
                _bar(
                    "failed guards",
                    sum(row.get("passed") is not True for row in report.get("guard_checks") or []),
                ),
            ],
            x_label="guard checks",
            width=1500,
            margin_left=700,
        ),
    }
    visualizations = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        visualizations.append(
            PrimeQAHybridConcurrentActivationValidationVisualization(
                name=name,
                path=str(path),
            )
        )
    return visualizations


def _startup_state_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    codes = {"disabled": 0.0, "rejected": 1.0, "eligible": 2.0}
    startup = report.get("startup_cases") or {}
    return [
        _bar(f"{name}: {(startup.get(name) or {}).get('activation_state')}", codes[name])
        for name in ("disabled", "rejected", "eligible")
    ]


def _startup_build_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        _bar("disabled resource builds", report.get("disabled_resource_factory_build_count") or 0),
        _bar("rejected resource builds", report.get("rejected_resource_factory_build_count") or 0),
        _bar("eligible resource builds", report.get("resource_factory_build_count") or 0),
    ]


def _train_pattern_latency_bars(train: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for name, row in (train.get("pattern_pooled_reports") or {}).items():
        latency = row.get("end_to_end_latency_seconds") or {}
        bars.extend(_latency_bars(latency, name))
    return bars


def _scope_maximum_bars(train: Mapping[str, Any]) -> list[BarDatum]:
    rows = [
        *(train.get("fold_pattern_repetition_reports") or {}).values(),
        *(train.get("pass_reports") or {}).values(),
        *(train.get("pattern_pooled_reports") or {}).values(),
        train.get("global_pooled_report") or {},
    ]
    distributions = [row.get("end_to_end_latency_seconds") or {} for row in rows]
    return [
        _bar("maximum scope P95", max((row.get("p95") or 0) for row in distributions)),
        _bar("maximum scope P99", max((row.get("p99") or 0) for row in distributions)),
        _bar("profile B P95 limit", 0.8),
        _bar("profile B P99 limit", 1.5),
    ]


def _arrival_error_bars(train: Mapping[str, Any]) -> list[BarDatum]:
    distribution = (train.get("combined_runtime_summary") or {}).get(
        "arrival_offset_error_ms"
    ) or {}
    return [
        _bar("average", distribution.get("average") or 0),
        _bar("P95", distribution.get("p95") or 0),
        _bar("P99", distribution.get("p99") or 0),
        _bar("maximum", distribution.get("max") or 0),
    ]


def _decision_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    keys = (
        "application_activation_bootstrap_implemented",
        "disabled_rejected_eligible_startup_validated",
        "eligible_runtime_full_workload_validation_passed",
        "explicit_nondefault_concurrent_activation_available",
        "runtime_registered_as_default",
        "test_gate_opened",
        "queue_actions_enabled",
        "retry_actions_enabled",
        "fallback_strategies_enabled",
    )
    return [_bar(key, float(decision.get(key) is True)) for key in keys]


def _latency_bars(distribution: Mapping[str, Any], prefix: str) -> list[BarDatum]:
    return [
        _bar(f"{prefix} P95", distribution.get("p95") or 0),
        _bar(f"{prefix} P99", distribution.get("p99") or 0),
    ]


def _bar(label: str, value: float | int) -> BarDatum:
    numeric = float(value)
    return BarDatum(label=label, value=numeric, value_label=f"{numeric:.6f}")


def _configuration_mutual_exclusion_enforced() -> bool:
    try:
        ProjectSettings(
            enable_optional_sidecar_agent=True,
            enable_concurrent_sidecar_agent=True,
        )
    except ValidationError:
        return True
    return False


def _label_free_question(sample: PrimeQAHybridSplitSample) -> PrimeQAQuestion:
    question = sample.to_primeqa_question()
    return PrimeQAQuestion(
        id="concurrent-application-runtime-warmup",
        title=question.title,
        text=question.text,
        answer="",
        answerable=False,
        answer_doc_id=None,
        doc_ids=[],
    )


def _source_stage145_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    checks = report.get("guard_checks") or []
    decision = report.get("decision") or {}
    public = report.get("public_safe_contract") or {}
    train = report.get("train_validation") or {}
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "status": decision.get("status"),
        "guard_check_count": len(checks),
        "guard_check_passed_count": sum(row.get("passed") is True for row in checks),
        "train_accepted_request_count": train.get("accepted_request_count"),
        "train_latency_gate_scope_count": train.get("latency_gate_scope_count"),
        "test_split_loaded": public.get("test_split_loaded"),
        "test_metrics_run": public.get("test_metrics_run"),
        "forbidden_keys_found": public.get("forbidden_keys_found"),
    }


def _check(name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }
