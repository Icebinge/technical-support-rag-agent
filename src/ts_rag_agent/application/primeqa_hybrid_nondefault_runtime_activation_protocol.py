from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 141"
_CREATED_AT = "2026-07-17"
_PROTOCOL_ID = "primeqa_hybrid_nondefault_runtime_activation_protocol_v1"
_SLO_PROFILE_ID = "strict_c_warm_single_request_v1"
_RUNTIME_FLAG = "TS_RAG_ENABLE_OPTIONAL_SIDECAR_AGENT"
_SOURCE_STAGE140_STATUS = "primeqa_hybrid_online_candidate_pool_performance_validation_passed"
_SOURCE_STAGE140_ANALYSIS_ID = "primeqa_hybrid_online_candidate_pool_performance_validation_v1"
_NEXT_DIRECTION = "optimize_and_validate_strict_warm_single_request_latency_on_train_cv_dev"
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
        "matched_token_strings",
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


class RuntimeActivationState(str, Enum):
    """Outcomes of the executable non-default activation policy."""

    DISABLED = "disabled"
    REJECTED = "rejected"
    ELIGIBLE = "eligible"


@dataclass(frozen=True)
class StrictWarmLatencySlo:
    """User-confirmed strict-C warm single-request latency target."""

    profile_id: str = _SLO_PROFILE_ID
    p95_seconds: float = 0.3
    p99_seconds: float = 1.0


@dataclass(frozen=True)
class RuntimeActivationEvidence:
    """Aggregate-only evidence consumed by the activation policy."""

    explicit_activation_requested: bool
    concurrent_request_support_requested: bool
    source_performance_validated: bool
    warm_resources_ready: bool
    candidate_pool_identity_preserved: bool
    retrieval_recall_preserved: bool
    train_fold_count: int
    train_all_folds_pass: bool
    train_p95_seconds: float | None
    train_p99_seconds: float | None
    dev_report_only_pass: bool
    dev_p95_seconds: float | None
    dev_p99_seconds: float | None
    test_split_locked: bool


@dataclass(frozen=True)
class RuntimeActivationEvaluation:
    """Public-safe result that never activates runtime by itself."""

    state: RuntimeActivationState
    rejection_reasons: tuple[str, ...]
    runtime_activated: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "rejection_reasons": list(self.rejection_reasons),
            "runtime_activated": self.runtime_activated,
        }


class StrictNonDefaultRuntimeActivationPolicy:
    """Fail-closed policy with no retries, fallback, or automatic activation."""

    def __init__(self, slo: StrictWarmLatencySlo | None = None) -> None:
        self._slo = slo or StrictWarmLatencySlo()

    def evaluate(self, evidence: RuntimeActivationEvidence) -> RuntimeActivationEvaluation:
        if not evidence.explicit_activation_requested:
            return RuntimeActivationEvaluation(
                state=RuntimeActivationState.DISABLED,
                rejection_reasons=("explicit_activation_not_requested",),
            )

        reasons: list[str] = []
        if evidence.concurrent_request_support_requested:
            reasons.append("concurrent_runtime_not_authorized_by_single_request_protocol")
        if not evidence.source_performance_validated:
            reasons.append("source_performance_not_validated")
        if not evidence.warm_resources_ready:
            reasons.append("warm_runtime_resources_not_ready")
        if not evidence.candidate_pool_identity_preserved:
            reasons.append("candidate_pool_identity_not_preserved")
        if not evidence.retrieval_recall_preserved:
            reasons.append("retrieval_recall_not_preserved")
        if evidence.train_fold_count != 5:
            reasons.append("train_grouped_five_fold_evidence_missing")
        if not evidence.train_all_folds_pass:
            reasons.append("train_fold_strict_slo_not_passed")
        _append_latency_reasons(
            reasons,
            split="train",
            percentile="p95",
            observed=evidence.train_p95_seconds,
            limit=self._slo.p95_seconds,
        )
        _append_latency_reasons(
            reasons,
            split="train",
            percentile="p99",
            observed=evidence.train_p99_seconds,
            limit=self._slo.p99_seconds,
        )
        if not evidence.dev_report_only_pass:
            reasons.append("dev_report_only_strict_slo_not_passed")
        _append_latency_reasons(
            reasons,
            split="dev",
            percentile="p95",
            observed=evidence.dev_p95_seconds,
            limit=self._slo.p95_seconds,
        )
        _append_latency_reasons(
            reasons,
            split="dev",
            percentile="p99",
            observed=evidence.dev_p99_seconds,
            limit=self._slo.p99_seconds,
        )
        if not evidence.test_split_locked:
            reasons.append("test_split_not_locked")

        if reasons:
            return RuntimeActivationEvaluation(
                state=RuntimeActivationState.REJECTED,
                rejection_reasons=tuple(reasons),
            )
        return RuntimeActivationEvaluation(
            state=RuntimeActivationState.ELIGIBLE,
            rejection_reasons=(),
        )


@dataclass(frozen=True)
class PrimeQAHybridRuntimeActivationProtocolVisualization:
    """One generated Stage141 protocol chart."""

    name: str
    path: str


def freeze_primeqa_hybrid_nondefault_runtime_activation_protocol(
    *,
    stage140_validation_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
    selected_slo_profile_id: str,
) -> dict[str, Any]:
    """Freeze strict-C runtime activation rules from public Stage140 aggregates."""

    started_at = time.perf_counter()
    source = _load_json_object(stage140_validation_path)
    loaded_at = time.perf_counter()
    source_summary = _stage140_summary(source)
    slo = StrictWarmLatencySlo()
    policy = StrictNonDefaultRuntimeActivationPolicy(slo)
    canonical_evaluations = _canonical_evaluations(policy, source_summary)
    frozen_protocol = _frozen_protocol(slo)
    preliminary = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "protocol_scope": (
            "Aggregate-only freeze of the user-confirmed strict-C warm single-request "
            "latency SLO and a future explicit non-default runtime activation contract. "
            "This stage reads only the saved public-safe Stage140 report. It does not load "
            "split rows, questions, documents, candidate pools, models, indexes, or test "
            "data; wire FastAPI or another runtime; activate the agent; authorize "
            "concurrent requests; change defaults; or add retries or fallback strategies."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_protocol),
            "confirmation_note": confirmation_note,
            "selected_slo_profile_id": selected_slo_profile_id,
        },
        "source_files": {"stage140_validation": _fingerprint(stage140_validation_path)},
        "stage140_summary": source_summary,
        "frozen_protocol": frozen_protocol,
        "canonical_activation_evaluations": canonical_evaluations,
    }
    guards = _guard_checks(
        report=preliminary,
        source_summary=source_summary,
        frozen_protocol=frozen_protocol,
        canonical_evaluations=canonical_evaluations,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
        selected_slo_profile_id=selected_slo_profile_id,
    )
    checked_at = time.perf_counter()
    report = {
        **preliminary,
        "guard_checks": guards,
        "decision": _decision(guards),
        "timing_seconds": {
            "load_public_stage140_aggregate": round(loaded_at - started_at, 6),
            "freeze_and_guard": round(checked_at - loaded_at, 6),
            "total": round(checked_at - started_at, 6),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_nondefault_runtime_activation_protocol_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridRuntimeActivationProtocolVisualization]:
    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage141_source_p95_vs_strict_slo.svg": render_horizontal_bar_chart_svg(
            title="Stage141 source P95 versus strict-C SLO",
            bars=_source_p95_bars(report),
            x_label="seconds",
            width=1320,
            margin_left=620,
        ),
        "stage141_percentile_evidence_availability.svg": render_horizontal_bar_chart_svg(
            title="Stage141 source percentile evidence availability",
            bars=_percentile_availability_bars(report),
            x_label="1 means available",
            width=1500,
            margin_left=760,
        ),
        "stage141_activation_case_states.svg": render_horizontal_bar_chart_svg(
            title="Stage141 canonical activation policy outcomes",
            bars=_activation_case_bars(report),
            x_label="2 eligible, 1 rejected, 0 disabled",
            width=1540,
            margin_left=760,
        ),
        "stage141_runtime_permission_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage141 runtime permission flags",
            bars=_permission_flag_bars(report),
            x_label="1 means enabled",
            width=1720,
            margin_left=920,
        ),
        "stage141_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage141 protocol guard checks",
            bars=[
                BarDatum(
                    label=str(check["name"]),
                    value=1.0 if check["passed"] else 0.0,
                    value_label="passed" if check["passed"] else "failed",
                )
                for check in report.get("guard_checks", [])
            ],
            x_label="1 means passed",
            width=2200,
            margin_left=1240,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridRuntimeActivationProtocolVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _append_latency_reasons(
    reasons: list[str],
    *,
    split: str,
    percentile: str,
    observed: float | None,
    limit: float,
) -> None:
    if observed is None:
        reasons.append(f"{split}_{percentile}_missing")
    elif observed > limit:
        reasons.append(f"{split}_{percentile}_exceeds_strict_slo")


def _stage140_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision") or {}
    guards = report.get("guard_checks") or []
    split_reports = report.get("split_reports") or {}
    public_safe = report.get("public_safe_contract") or {}
    scope = report.get("analysis_scope") or {}
    candidate_contract = report.get("selected_candidate_pool_contract") or {}
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "status": decision.get("status"),
        "guard_check_count": len(guards),
        "guard_check_passed_count": sum(bool(check.get("passed")) for check in guards),
        "online_candidate_pool_implementation_validated": decision.get(
            "online_candidate_pool_implementation_validated"
        ),
        "candidate_pool_identity_preserved": decision.get("candidate_pool_identity_preserved"),
        "retrieval_recall_preserved": decision.get("retrieval_recall_preserved"),
        "runtime_activation_allowed_now": decision.get("runtime_activation_allowed_now"),
        "runtime_defaultization_allowed_now": decision.get("runtime_defaultization_allowed_now"),
        "retry_actions_enabled": decision.get("retry_actions_enabled"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "test_split_loaded": scope.get("test_split_loaded"),
        "test_metrics_run": scope.get("test_metrics_run"),
        "train": _source_split_summary(split_reports.get("train") or {}),
        "dev": _source_split_summary(split_reports.get("dev") or {}),
        "candidate_pool_contract": {
            key: candidate_contract.get(key)
            for key in (
                "config_id",
                "channel_top_k",
                "prefix_depth",
                "target_pool_depth",
                "rrf_k",
                "channel_count",
                "independent_channel_count",
                "derived_channel_count",
                "indexes_owned_outside_request_path",
                "query_specific_candidate_pool_built_per_request",
            )
        },
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
    }


def _source_split_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    latency = report.get("latency_seconds") or {}
    candidate_size = report.get("candidate_pool_size") or {}
    return {
        "row_count": report.get("row_count"),
        "exact_candidate_pool_identity_violation_count": report.get(
            "exact_candidate_pool_identity_violation_count"
        ),
        "candidate_pool_min": candidate_size.get("min"),
        "candidate_pool_max": candidate_size.get("max"),
        "latency_p50_seconds": latency.get("p50"),
        "latency_p95_seconds": latency.get("p95"),
        "latency_p99_seconds": latency.get("p99"),
        "latency_max_seconds": latency.get("max"),
        "recall_hit_counts": (report.get("recall") or {}).get("hit_counts"),
        "fold_report_count": len(report.get("fold_reports") or {}),
    }


def _frozen_protocol(slo: StrictWarmLatencySlo) -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "implementation_status": "activation_policy_executable_runtime_not_wired",
        "runtime_interface": {
            "future_environment_flag": _RUNTIME_FLAG,
            "default_value": False,
            "explicit_true_required": True,
            "current_project_settings_field_implemented": False,
            "runtime_entrypoint_registered": False,
            "runtime_default_changed": False,
            "eligible_policy_result_auto_activates_runtime": False,
        },
        "latency_slo": {
            **asdict(slo),
            "scope": "warm_single_request_end_to_end_candidate_pool_retrieval",
            "percentile_method": "linear_interpolation_at_(n_minus_1)_times_p",
            "measurement_repetitions": 3,
            "train_protocol": "frozen_grouped_five_fold_all_folds_and_aggregate_must_pass",
            "dev_protocol": "single_locked_report_only_gate_after_train_decisions",
            "test_protocol": "locked_not_loaded_not_measured",
            "startup_cost_in_request_slo": False,
            "startup_cost_reported_separately": True,
        },
        "warmup_contract": {
            "resource_owner": "future_process_scoped_runtime_bootstrap",
            "dense_model_count": 2,
            "dense_embedding_cache_count": 2,
            "lexical_index_count": 4,
            "derived_route_count": 1,
            "candidate_pool_retriever_instance_count": 1,
            "resources_built_or_loaded_per_request": False,
            "warmup_request_count": 1,
            "warmup_request_source": "deterministic_train_only_row_selected_without_labels",
            "warmup_request_excluded_from_measurement": True,
            "same_row_included_in_each_complete_measured_train_pass": True,
        },
        "request_path_contract": {
            "query_specific_candidate_pool_built_per_request": True,
            "allowed_operations": [
                "query_encoding",
                "long_lived_index_search",
                "two_depth_rrf_fusion",
                "deduplication",
                "candidate_materialization",
            ],
            "candidate_pool_depth": 400,
            "concurrent_request_support_authorized": False,
            "single_request_research_runtime_only": True,
        },
        "activation_guards": {
            "explicit_flag_required": True,
            "warm_resources_required": True,
            "stage140_source_validation_required": True,
            "candidate_pool_exact_identity_required": True,
            "stage127_recall_exact_preservation_required": True,
            "all_five_train_folds_must_pass_slo": True,
            "train_aggregate_must_pass_slo": True,
            "dev_report_only_must_pass_slo": True,
            "test_must_remain_locked": True,
            "retry_allowed": False,
            "fallback_allowed": False,
            "fail_closed_on_any_guard_failure": True,
        },
        "public_request_trace_contract": {
            "allowed_fields": [
                "runtime_mode",
                "activation_requested",
                "activation_state",
                "slo_profile_id",
                "warm_resources_ready",
                "candidate_pool_depth",
                "retrieval_latency_ms",
                "latency_budget_passed",
                "terminal_state",
            ],
            "question_or_document_content_allowed": False,
            "question_sample_or_document_ids_allowed": False,
            "candidate_rows_allowed": False,
        },
        "rejection_contract": {
            "flag_false_behavior": "remain_disabled_without_running_optional_entrypoint",
            "requested_but_guard_failed_behavior": "reject_before_request_serving",
            "silent_route_substitution_allowed": False,
            "retry_allowed": False,
            "fallback_allowed": False,
        },
        "concurrency_boundary": {
            "concurrency_slo_confirmed": False,
            "concurrent_runtime_activation_allowed": False,
            "single_request_evidence_cannot_claim_concurrency_readiness": True,
            "future_concurrency_protocol_requires_user_confirmation": True,
        },
    }


def _canonical_evaluations(
    policy: StrictNonDefaultRuntimeActivationPolicy,
    source_summary: Mapping[str, Any],
) -> dict[str, Any]:
    train = source_summary.get("train") or {}
    dev = source_summary.get("dev") or {}
    common = {
        "source_performance_validated": source_summary.get(
            "online_candidate_pool_implementation_validated"
        )
        is True,
        "candidate_pool_identity_preserved": source_summary.get("candidate_pool_identity_preserved")
        is True,
        "retrieval_recall_preserved": source_summary.get("retrieval_recall_preserved") is True,
        "train_fold_count": int(train.get("fold_report_count") or 0),
        "test_split_locked": source_summary.get("test_split_loaded") is False
        and source_summary.get("test_metrics_run") is False,
    }
    disabled = policy.evaluate(
        RuntimeActivationEvidence(
            explicit_activation_requested=False,
            concurrent_request_support_requested=False,
            warm_resources_ready=False,
            train_all_folds_pass=False,
            train_p95_seconds=train.get("latency_p95_seconds"),
            train_p99_seconds=train.get("latency_p99_seconds"),
            dev_report_only_pass=False,
            dev_p95_seconds=dev.get("latency_p95_seconds"),
            dev_p99_seconds=dev.get("latency_p99_seconds"),
            **common,
        )
    )
    current_source = policy.evaluate(
        RuntimeActivationEvidence(
            explicit_activation_requested=True,
            concurrent_request_support_requested=False,
            warm_resources_ready=False,
            train_all_folds_pass=False,
            train_p95_seconds=train.get("latency_p95_seconds"),
            train_p99_seconds=train.get("latency_p99_seconds"),
            dev_report_only_pass=False,
            dev_p95_seconds=dev.get("latency_p95_seconds"),
            dev_p99_seconds=dev.get("latency_p99_seconds"),
            **common,
        )
    )
    compliant = policy.evaluate(
        RuntimeActivationEvidence(
            explicit_activation_requested=True,
            concurrent_request_support_requested=False,
            source_performance_validated=True,
            warm_resources_ready=True,
            candidate_pool_identity_preserved=True,
            retrieval_recall_preserved=True,
            train_fold_count=5,
            train_all_folds_pass=True,
            train_p95_seconds=0.3,
            train_p99_seconds=1.0,
            dev_report_only_pass=True,
            dev_p95_seconds=0.3,
            dev_p99_seconds=1.0,
            test_split_locked=True,
        )
    )
    concurrent = policy.evaluate(
        RuntimeActivationEvidence(
            explicit_activation_requested=True,
            concurrent_request_support_requested=True,
            source_performance_validated=True,
            warm_resources_ready=True,
            candidate_pool_identity_preserved=True,
            retrieval_recall_preserved=True,
            train_fold_count=5,
            train_all_folds_pass=True,
            train_p95_seconds=0.2,
            train_p99_seconds=0.5,
            dev_report_only_pass=True,
            dev_p95_seconds=0.2,
            dev_p99_seconds=0.5,
            test_split_locked=True,
        )
    )
    return {
        "flag_absent_or_false": disabled.to_public_dict(),
        "stage140_source_requested_now": current_source.to_public_dict(),
        "hypothetical_strict_single_request_compliant": compliant.to_public_dict(),
        "hypothetical_concurrent_request": concurrent.to_public_dict(),
    }


def _guard_checks(
    *,
    report: Mapping[str, Any],
    source_summary: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
    canonical_evaluations: Mapping[str, Any],
    user_confirmed_protocol: bool,
    confirmation_note: str,
    selected_slo_profile_id: str,
) -> list[dict[str, Any]]:
    runtime_interface = frozen_protocol.get("runtime_interface") or {}
    slo = frozen_protocol.get("latency_slo") or {}
    warmup = frozen_protocol.get("warmup_contract") or {}
    activation = frozen_protocol.get("activation_guards") or {}
    concurrency = frozen_protocol.get("concurrency_boundary") or {}
    current = canonical_evaluations.get("stage140_source_requested_now") or {}
    compliant = canonical_evaluations.get("hypothetical_strict_single_request_compliant") or {}
    concurrent = canonical_evaluations.get("hypothetical_concurrent_request") or {}
    return [
        _check(
            "user_confirmed_strict_c_protocol",
            user_confirmed_protocol
            and bool(confirmation_note.strip())
            and selected_slo_profile_id == _SLO_PROFILE_ID,
            {
                "confirmed": user_confirmed_protocol,
                "note_present": bool(confirmation_note.strip()),
                "profile": selected_slo_profile_id,
            },
            {"confirmed": True, "profile": _SLO_PROFILE_ID},
        ),
        _check(
            "stage140_source_identity_matches",
            source_summary.get("stage") == "Stage 140"
            and source_summary.get("analysis_id") == _SOURCE_STAGE140_ANALYSIS_ID,
            {
                "stage": source_summary.get("stage"),
                "analysis_id": source_summary.get("analysis_id"),
            },
            {"stage": "Stage 140", "analysis_id": _SOURCE_STAGE140_ANALYSIS_ID},
        ),
        _check(
            "stage140_validation_and_all_guards_passed",
            source_summary.get("status") == _SOURCE_STAGE140_STATUS
            and source_summary.get("guard_check_count") == 21
            and source_summary.get("guard_check_passed_count") == 21
            and source_summary.get("online_candidate_pool_implementation_validated") is True,
            {
                "status": source_summary.get("status"),
                "passed": source_summary.get("guard_check_passed_count"),
                "total": source_summary.get("guard_check_count"),
            },
            {"status": _SOURCE_STAGE140_STATUS, "passed": 21, "total": 21},
        ),
        _check(
            "stage140_identity_and_recall_preserved",
            source_summary.get("candidate_pool_identity_preserved") is True
            and source_summary.get("retrieval_recall_preserved") is True
            and all(
                (source_summary.get(split) or {}).get(
                    "exact_candidate_pool_identity_violation_count"
                )
                == 0
                for split in ("train", "dev")
            ),
            {
                "identity": source_summary.get("candidate_pool_identity_preserved"),
                "recall": source_summary.get("retrieval_recall_preserved"),
            },
            True,
        ),
        _check(
            "stage140_candidate_pool_contract_preserved",
            (source_summary.get("candidate_pool_contract") or {}).get("config_id")
            == "prefix_existing_dense_broad_append200_v1"
            and (source_summary.get("candidate_pool_contract") or {}).get("channel_top_k") == 400
            and (source_summary.get("candidate_pool_contract") or {}).get("prefix_depth") == 200
            and (source_summary.get("candidate_pool_contract") or {}).get("target_pool_depth")
            == 400
            and (source_summary.get("candidate_pool_contract") or {}).get("rrf_k") == 60
            and (source_summary.get("candidate_pool_contract") or {}).get("channel_count") == 7
            and (source_summary.get("candidate_pool_contract") or {}).get(
                "independent_channel_count"
            )
            == 6
            and (source_summary.get("candidate_pool_contract") or {}).get("derived_channel_count")
            == 1
            and (source_summary.get("candidate_pool_contract") or {}).get(
                "indexes_owned_outside_request_path"
            )
            is True
            and (source_summary.get("candidate_pool_contract") or {}).get(
                "query_specific_candidate_pool_built_per_request"
            )
            is True,
            source_summary.get("candidate_pool_contract"),
            "frozen config, Top200/400 RRF, 7=6+1 channels, and long-lived indexes",
        ),
        _check(
            "strict_c_thresholds_are_exact",
            slo.get("profile_id") == _SLO_PROFILE_ID
            and slo.get("p95_seconds") == 0.3
            and slo.get("p99_seconds") == 1.0,
            {key: slo.get(key) for key in ("profile_id", "p95_seconds", "p99_seconds")},
            {"profile_id": _SLO_PROFILE_ID, "p95_seconds": 0.3, "p99_seconds": 1.0},
        ),
        _check(
            "strict_measurement_protocol_is_repeated_train_cv_dev_report_only",
            slo.get("measurement_repetitions") == 3
            and slo.get("train_protocol")
            == "frozen_grouped_five_fold_all_folds_and_aggregate_must_pass"
            and slo.get("dev_protocol") == "single_locked_report_only_gate_after_train_decisions"
            and slo.get("test_protocol") == "locked_not_loaded_not_measured",
            {
                "repetitions": slo.get("measurement_repetitions"),
                "train": slo.get("train_protocol"),
                "dev": slo.get("dev_protocol"),
                "test": slo.get("test_protocol"),
            },
            "three warm train-CV passes then locked dev report-only; no test",
        ),
        _check(
            "startup_and_request_costs_are_separated",
            slo.get("startup_cost_in_request_slo") is False
            and slo.get("startup_cost_reported_separately") is True
            and warmup.get("resources_built_or_loaded_per_request") is False,
            {"slo": slo, "warmup": warmup},
            "startup reported separately and no per-request resource bootstrap",
        ),
        _check(
            "future_runtime_flag_is_explicit_and_disabled_by_default",
            runtime_interface.get("future_environment_flag") == _RUNTIME_FLAG
            and runtime_interface.get("default_value") is False
            and runtime_interface.get("explicit_true_required") is True,
            runtime_interface,
            {"flag": _RUNTIME_FLAG, "default": False, "explicit_true": True},
        ),
        _check(
            "runtime_wiring_is_not_falsely_claimed",
            frozen_protocol.get("implementation_status")
            == "activation_policy_executable_runtime_not_wired"
            and runtime_interface.get("current_project_settings_field_implemented") is False
            and runtime_interface.get("runtime_entrypoint_registered") is False,
            {
                "status": frozen_protocol.get("implementation_status"),
                "settings": runtime_interface.get("current_project_settings_field_implemented"),
                "entrypoint": runtime_interface.get("runtime_entrypoint_registered"),
            },
            "policy executable; runtime not wired",
        ),
        _check(
            "eligible_result_never_auto_activates",
            runtime_interface.get("eligible_policy_result_auto_activates_runtime") is False
            and compliant.get("state") == RuntimeActivationState.ELIGIBLE.value
            and compliant.get("runtime_activated") is False,
            {"interface": runtime_interface, "compliant_case": compliant},
            "eligible but not activated",
        ),
        _check(
            "current_stage140_evidence_is_rejected_honestly",
            current.get("state") == RuntimeActivationState.REJECTED.value
            and "train_p95_exceeds_strict_slo" in (current.get("rejection_reasons") or [])
            and "train_p99_missing" in (current.get("rejection_reasons") or [])
            and "dev_p99_missing" in (current.get("rejection_reasons") or [])
            and current.get("runtime_activated") is False,
            current,
            "rejected due to strict-C gaps and not activated",
        ),
        _check(
            "source_p95_relationship_is_recorded_without_invented_p99",
            (source_summary.get("train") or {}).get("latency_p95_seconds") > 0.3
            and (source_summary.get("dev") or {}).get("latency_p95_seconds") <= 0.3
            and (source_summary.get("train") or {}).get("latency_p99_seconds") is None
            and (source_summary.get("dev") or {}).get("latency_p99_seconds") is None,
            {
                split: {
                    "p95": (source_summary.get(split) or {}).get("latency_p95_seconds"),
                    "p99": (source_summary.get(split) or {}).get("latency_p99_seconds"),
                }
                for split in ("train", "dev")
            },
            "train P95 fails, dev P95 passes, both P99 unavailable",
        ),
        _check(
            "concurrent_runtime_is_explicitly_out_of_scope",
            concurrency.get("concurrency_slo_confirmed") is False
            and concurrency.get("concurrent_runtime_activation_allowed") is False
            and concurrent.get("state") == RuntimeActivationState.REJECTED.value
            and "concurrent_runtime_not_authorized_by_single_request_protocol"
            in (concurrent.get("rejection_reasons") or []),
            {"boundary": concurrency, "case": concurrent},
            "concurrent activation rejected pending a separately confirmed protocol",
        ),
        _check(
            "activation_fails_closed_without_retry_or_fallback",
            activation.get("fail_closed_on_any_guard_failure") is True
            and activation.get("retry_allowed") is False
            and activation.get("fallback_allowed") is False
            and (frozen_protocol.get("rejection_contract") or {}).get(
                "silent_route_substitution_allowed"
            )
            is False,
            {
                "activation": activation,
                "rejection": frozen_protocol.get("rejection_contract"),
            },
            "fail closed with no retry, fallback, or silent substitution",
        ),
        _check(
            "stage140_and_stage141_defaults_remain_unchanged",
            source_summary.get("runtime_activation_allowed_now") is False
            and source_summary.get("runtime_defaultization_allowed_now") is False
            and source_summary.get("default_runtime_policy") == "unchanged"
            and runtime_interface.get("runtime_default_changed") is False,
            {
                "source_activation": source_summary.get("runtime_activation_allowed_now"),
                "source_defaultization": source_summary.get("runtime_defaultization_allowed_now"),
                "protocol_default_changed": runtime_interface.get("runtime_default_changed"),
            },
            False,
        ),
        _check(
            "test_remains_locked_and_unloaded",
            source_summary.get("test_split_loaded") is False
            and source_summary.get("test_metrics_run") is False
            and activation.get("test_must_remain_locked") is True,
            {
                "loaded": source_summary.get("test_split_loaded"),
                "metrics": source_summary.get("test_metrics_run"),
            },
            False,
        ),
        _check(
            "retry_and_fallback_remain_disabled",
            source_summary.get("retry_actions_enabled") is False
            and source_summary.get("fallback_strategies_enabled") is False
            and activation.get("retry_allowed") is False
            and activation.get("fallback_allowed") is False,
            {
                "source_retry": source_summary.get("retry_actions_enabled"),
                "source_fallback": source_summary.get("fallback_strategies_enabled"),
                "protocol_retry": activation.get("retry_allowed"),
                "protocol_fallback": activation.get("fallback_allowed"),
            },
            False,
        ),
        _check(
            "source_and_protocol_are_public_safe",
            source_summary.get("public_safe_forbidden_keys_found") == []
            and not _forbidden_keys_found(report),
            {
                "source": source_summary.get("public_safe_forbidden_keys_found"),
                "protocol": sorted(_forbidden_keys_found(report)),
            },
            [],
        ),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [str(check["name"]) for check in guard_checks if not check["passed"]]
    base = {
        "protocol_id": _PROTOCOL_ID,
        "selected_slo_profile_id": _SLO_PROFILE_ID,
        "warm_single_request_p95_limit_seconds": 0.3,
        "warm_single_request_p99_limit_seconds": 1.0,
        "activation_policy_executable": True,
        "runtime_settings_flag_implemented": False,
        "runtime_entrypoint_registered": False,
        "runtime_activation_allowed_now": False,
        "runtime_activated_now": False,
        "concurrent_runtime_activation_allowed": False,
        "runtime_defaultization_allowed_now": False,
        "test_gate_opened": False,
        "retry_actions_enabled": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }
    if failed:
        return {
            **base,
            "status": "primeqa_hybrid_nondefault_runtime_activation_protocol_blocked",
            "failed_checks": failed,
            "runtime_activation_protocol_frozen": False,
            "recommended_next_direction": "review_stage141_runtime_activation_protocol_failures",
        }
    return {
        **base,
        "status": "primeqa_hybrid_nondefault_runtime_activation_protocol_frozen",
        "failed_checks": [],
        "runtime_activation_protocol_frozen": True,
        "strict_slo_currently_satisfied": False,
        "recommended_next_direction": _NEXT_DIRECTION,
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "aggregate_only": True,
        "synthetic_activation_evaluations_written": True,
        "split_files_loaded": False,
        "corpus_documents_loaded": False,
        "models_or_indexes_loaded": False,
        "runtime_requests_executed": False,
        "test_split_loaded": False,
        "test_metrics_run": False,
        "raw_questions_written": False,
        "raw_answers_written": False,
        "raw_documents_written": False,
        "raw_candidate_rows_written": False,
        "raw_sample_or_document_ids_written": False,
        "forbidden_keys_found": sorted(_forbidden_keys_found(report)),
    }


def _source_p95_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    source = report.get("stage140_summary") or {}
    slo = (report.get("frozen_protocol") or {}).get("latency_slo") or {}
    rows = [
        ("train Stage140 observed P95", (source.get("train") or {}).get("latency_p95_seconds")),
        ("dev Stage140 observed P95", (source.get("dev") or {}).get("latency_p95_seconds")),
        ("strict-C P95 limit", slo.get("p95_seconds")),
    ]
    return [
        BarDatum(label=label, value=float(value or 0.0), value_label=f"{float(value or 0):.6f}s")
        for label, value in rows
    ]


def _percentile_availability_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    source = report.get("stage140_summary") or {}
    return [
        BarDatum(
            label=f"{split} {percentile.upper()} available",
            value=1.0 if (source.get(split) or {}).get(key) is not None else 0.0,
            value_label=(
                "available" if (source.get(split) or {}).get(key) is not None else "missing"
            ),
        )
        for split in ("train", "dev")
        for percentile, key in (
            ("p95", "latency_p95_seconds"),
            ("p99", "latency_p99_seconds"),
        )
    ]


def _activation_case_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    evaluations = report.get("canonical_activation_evaluations") or {}
    values = {"disabled": 0.0, "rejected": 1.0, "eligible": 2.0}
    return [
        BarDatum(
            label=str(case_id),
            value=values.get(str((evaluation or {}).get("state")), 0.0),
            value_label=str((evaluation or {}).get("state")),
        )
        for case_id, evaluation in evaluations.items()
    ]


def _permission_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    flags = (
        "activation_policy_executable",
        "runtime_settings_flag_implemented",
        "runtime_entrypoint_registered",
        "runtime_activation_allowed_now",
        "runtime_activated_now",
        "concurrent_runtime_activation_allowed",
        "runtime_defaultization_allowed_now",
        "test_gate_opened",
        "retry_actions_enabled",
        "fallback_strategies_enabled",
    )
    return [
        BarDatum(
            label=key,
            value=1.0 if decision.get(key) else 0.0,
            value_label="true" if decision.get(key) else "false",
        )
        for key in flags
    ]


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


def _check(name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest}
