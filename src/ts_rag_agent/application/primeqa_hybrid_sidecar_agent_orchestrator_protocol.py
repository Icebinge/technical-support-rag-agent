from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_orchestrator import (
    sidecar_agent_orchestrator_contract,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 136"
_CREATED_AT = "2026-07-16"
_PROTOCOL_ID = "primeqa_hybrid_sidecar_agent_orchestrator_protocol_v1"
_SOURCE_STAGE135_STATUS = "primeqa_hybrid_sidecar_observation_validation_passed"
_SOURCE_STAGE135_ANALYSIS_ID = (
    "primeqa_hybrid_stage116_answer_context_stage128_sidecar_observation_validation_v1"
)
_NEXT_DIRECTION = "run_stage116_primary_plus_sidecar_agent_orchestrator_train_cv_dev_validation"
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
        "source_doc_ids",
    }
)


@dataclass(frozen=True)
class PrimeQAHybridSidecarAgentOrchestratorProtocolVisualization:
    """One generated Stage136 orchestrator protocol chart."""

    name: str
    path: str


def freeze_primeqa_hybrid_sidecar_agent_orchestrator_protocol(
    *,
    stage135_validation_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage136 implementation and Stage137 validation contract."""

    started_at = time.perf_counter()
    stage135_validation = _load_json_object(stage135_validation_path)
    loaded_at = time.perf_counter()

    stage135_summary = _stage135_summary(stage135_validation)
    orchestrator_contract = sidecar_agent_orchestrator_contract()
    frozen_protocol = _frozen_protocol(
        stage135_summary=stage135_summary,
        orchestrator_contract=orchestrator_contract,
    )
    preliminary = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "protocol_scope": (
            "Public-safe Stage136 implementation and protocol freeze for a "
            "train/dev-only Stage116-primary plus Stage128-sidecar agent "
            "orchestrator. This stage reads only the saved public-safe aggregate "
            "Stage135 report, freezes the executable orchestrator and trace "
            "contracts, and does not load split files, corpus documents, raw "
            "candidate rows, runtime content handles, model outputs, or test data. "
            "It does not run retrieval, answer evaluation, final metrics, runtime "
            "defaultization, or fallback strategies."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_protocol),
            "confirmation_note": confirmation_note,
        },
        "source_files": {
            "stage135_validation": _fingerprint(stage135_validation_path),
        },
        "stage135_summary": stage135_summary,
        "frozen_protocol": frozen_protocol,
    }
    guard_checks = _guard_checks(
        report=preliminary,
        stage135_summary=stage135_summary,
        orchestrator_contract=orchestrator_contract,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    checked_at = time.perf_counter()
    report = {
        **preliminary,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
        "timing_seconds": {
            "load_source": round(loaded_at - started_at, 3),
            "freeze_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_sidecar_agent_orchestrator_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridSidecarAgentOrchestratorProtocolVisualization]:
    """Write public-safe SVG charts for Stage136."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage136_stage135_safety_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage136 source safety counts",
            bars=_source_safety_bars(report),
            x_label="violation count",
            width=1660,
            margin_left=860,
        ),
        "stage136_sidecar_opportunity_capture.svg": render_horizontal_bar_chart_svg(
            title="Stage136 sidecar opportunity boundary",
            bars=_opportunity_capture_bars(report),
            x_label="answerable row count",
            width=1660,
            margin_left=860,
        ),
        "stage136_channel_permission_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage136 channel permission flags",
            bars=_channel_permission_bars(report),
            x_label="1 means allowed",
            width=1760,
            margin_left=920,
        ),
        "stage136_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage136 decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1840,
            margin_left=980,
        ),
        "stage136_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage136 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=2140,
            margin_left=1220,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridSidecarAgentOrchestratorProtocolVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage135_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision") or {}
    train = (report.get("split_observation_reports") or {}).get("train") or {}
    dev = (report.get("split_observation_reports") or {}).get("dev") or {}
    invariance = report.get("source_answer_invariance") or {}
    guard_checks = report.get("guard_checks") or []
    public_safe = report.get("public_safe_contract") or {}
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "status": decision.get("status"),
        "sidecar_observation_protocol_validated": decision.get(
            "sidecar_observation_protocol_validated"
        ),
        "can_implement_train_dev_agent_orchestrator_now": decision.get(
            "can_implement_train_dev_agent_orchestrator_now"
        ),
        "guard_check_count": len(guard_checks),
        "guard_check_passed_count": sum(bool(check.get("passed")) for check in guard_checks),
        "train": _split_summary(train),
        "dev": _split_summary(dev),
        "source_answer_invariance": {
            split: {
                "verified_average_token_f1_delta": summary.get("verified_average_token_f1_delta"),
                "verified_gold_citation_count_delta": summary.get(
                    "verified_gold_citation_count_delta"
                ),
                "changed_verified_answer_count": summary.get("changed_verified_answer_count"),
            }
            for split, summary in invariance.items()
        },
        "direct_stage128_all400_answer_context_remains_blocked": decision.get(
            "direct_stage128_all400_answer_context_remains_blocked"
        ),
        "sidecar_can_generate_answer_text": decision.get("sidecar_can_generate_answer_text"),
        "sidecar_can_replace_primary_context": decision.get("sidecar_can_replace_primary_context"),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "runtime_defaultization_allowed_now": decision.get("runtime_defaultization_allowed_now"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
    }


def _split_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "row_count": summary.get("row_count"),
        "primary_context_identity_violation_count": summary.get(
            "primary_context_identity_violation_count"
        ),
        "answer_generation_context_identity_violation_count": summary.get(
            "answer_generation_context_identity_violation_count"
        ),
        "sidecar_answer_context_leak_count": summary.get("sidecar_answer_context_leak_count"),
        "sidecar_primary_overlap_count": summary.get("sidecar_primary_overlap_count"),
        "sidecar_observation_availability_rate": summary.get(
            "sidecar_observation_availability_rate"
        ),
        "append_pool_incremental_gold_hit_count": summary.get(
            "append_pool_incremental_gold_hit_count"
        ),
        "sidecar_incremental_gold_hit_count": summary.get("sidecar_incremental_gold_hit_count"),
        "sidecar_capture_rate_of_append_gold_opportunities": summary.get(
            "sidecar_capture_rate_of_append_gold_opportunities"
        ),
    }


def _frozen_protocol(
    *,
    stage135_summary: Mapping[str, Any],
    orchestrator_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "implementation_status": "orchestrator_implemented_protocol_frozen",
        "source_stage135": {
            "status": stage135_summary.get("status"),
            "guard_checks": {
                "passed": stage135_summary.get("guard_check_passed_count"),
                "total": stage135_summary.get("guard_check_count"),
            },
            "train": stage135_summary.get("train"),
            "dev": stage135_summary.get("dev"),
        },
        "orchestrator_contract": dict(orchestrator_contract),
        "trace_contract": {
            "trace_type": "PublicSafeSidecarAgentTrace",
            "per_run_trace_allowed_in_train_dev_validation": True,
            "real_per_run_traces_written_by_stage136": False,
            "raw_private_fields_forbidden": True,
            "sidecar_selection_metadata_fields": [
                "sidecar_observation_rank",
                "candidate_pool_rank",
                "source_region",
                "route_family",
                "query_overlap_count",
                "query_overlap_ratio",
                "retrieval_prior",
                "combined_score",
                "query_overlap_present",
                "novel_query_term_count",
                "novel_query_term_ratio",
                "extends_primary_query_coverage",
                "duplicate_of_primary_context",
                "selected_for_answer_generation",
                "selected_for_answer_verification",
            ],
            "sidecar_selection_or_miss_interpretation": (
                "Runtime traces can explain selection scores and isolation, but "
                "cannot label answer-document misses without offline train/dev labels."
            ),
        },
        "effectiveness_boundary": {
            "status": "diagnostic_only_unproven",
            "source_train_append_opportunities": (stage135_summary.get("train") or {}).get(
                "append_pool_incremental_gold_hit_count"
            ),
            "source_train_sidecar_captures": (stage135_summary.get("train") or {}).get(
                "sidecar_incremental_gold_hit_count"
            ),
            "source_dev_append_opportunities": (stage135_summary.get("dev") or {}).get(
                "append_pool_incremental_gold_hit_count"
            ),
            "source_dev_sidecar_captures": (stage135_summary.get("dev") or {}).get(
                "sidecar_incremental_gold_hit_count"
            ),
            "can_claim_citation_verification_effectiveness": False,
            "can_claim_answer_quality_improvement": False,
            "can_claim_retrieval_improvement": False,
        },
        "stage137_validation_plan": {
            "next_stage": "Stage 137",
            "action": _NEXT_DIRECTION,
            "selection_split": "train",
            "selection_mode": ("fixed_orchestrator_train_grouped_cross_validation_integrity"),
            "candidate_selection_performed": False,
            "threshold_tuning_performed": False,
            "minimum_train_folds": 5,
            "validation_split": "dev",
            "dev_mode": "single_pass_report_only_no_retuning",
            "required_checks": [
                "orchestrator answer identity vs Stage116 control",
                "verification identity vs Stage116 control",
                "sidecar never enters generation or verification context",
                "public-safe trace schema and serialization",
                "sidecar selection and offline miss diagnostics",
                "test split remains unloaded",
            ],
        },
        "risk_boundaries": {
            "test_access_allowed": False,
            "final_test_metrics_allowed": False,
            "test_tuning_allowed": False,
            "runtime_defaultization_allowed": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
            "direct_stage128_all400_answer_context_blocked": True,
        },
    }


def _guard_checks(
    *,
    report: Mapping[str, Any],
    stage135_summary: Mapping[str, Any],
    orchestrator_contract: Mapping[str, Any],
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> list[dict[str, Any]]:
    train = stage135_summary.get("train") or {}
    dev = stage135_summary.get("dev") or {}
    invariance = stage135_summary.get("source_answer_invariance") or {}
    channel_routing = orchestrator_contract.get("channel_routing") or {}
    consumer_policy = orchestrator_contract.get("consumer_policy") or {}
    public_trace = orchestrator_contract.get("public_trace") or {}
    effectiveness = orchestrator_contract.get("effectiveness_boundary") or {}
    return [
        _check(
            name="user_confirmed_stage136_protocol",
            passed=user_confirmed_protocol and "Stage136" in confirmation_note,
            observed=confirmation_note,
            expected="user confirmed Stage136 protocol",
        ),
        _check(
            name="stage135_validation_passed",
            passed=stage135_summary.get("status") == _SOURCE_STAGE135_STATUS,
            observed=stage135_summary.get("status"),
            expected=_SOURCE_STAGE135_STATUS,
        ),
        _check(
            name="stage135_analysis_id_matches",
            passed=stage135_summary.get("analysis_id") == _SOURCE_STAGE135_ANALYSIS_ID,
            observed=stage135_summary.get("analysis_id"),
            expected=_SOURCE_STAGE135_ANALYSIS_ID,
        ),
        _check(
            name="stage135_allows_orchestrator_implementation",
            passed=stage135_summary.get("can_implement_train_dev_agent_orchestrator_now") is True,
            observed=stage135_summary.get("can_implement_train_dev_agent_orchestrator_now"),
            expected=True,
        ),
        _check(
            name="stage135_all_guard_checks_passed",
            passed=stage135_summary.get("guard_check_count") == 30
            and stage135_summary.get("guard_check_passed_count") == 30,
            observed={
                "passed": stage135_summary.get("guard_check_passed_count"),
                "total": stage135_summary.get("guard_check_count"),
            },
            expected={"passed": 30, "total": 30},
        ),
        _check(
            name="stage135_primary_identity_preserved",
            passed=_all_split_values_zero(
                stage135_summary,
                "primary_context_identity_violation_count",
            )
            and _all_split_values_zero(
                stage135_summary,
                "answer_generation_context_identity_violation_count",
            ),
            observed={
                split: {
                    "primary": (stage135_summary.get(split) or {}).get(
                        "primary_context_identity_violation_count"
                    ),
                    "generation": (stage135_summary.get(split) or {}).get(
                        "answer_generation_context_identity_violation_count"
                    ),
                }
                for split in ("train", "dev")
            },
            expected="all zero",
        ),
        _check(
            name="stage135_sidecar_isolation_preserved",
            passed=_all_split_values_zero(
                stage135_summary,
                "sidecar_answer_context_leak_count",
            )
            and _all_split_values_zero(
                stage135_summary,
                "sidecar_primary_overlap_count",
            ),
            observed={
                split: {
                    "leaks": (stage135_summary.get(split) or {}).get(
                        "sidecar_answer_context_leak_count"
                    ),
                    "overlaps": (stage135_summary.get(split) or {}).get(
                        "sidecar_primary_overlap_count"
                    ),
                }
                for split in ("train", "dev")
            },
            expected="all zero",
        ),
        _check(
            name="stage135_observations_available",
            passed=all(
                float(
                    (stage135_summary.get(split) or {}).get("sidecar_observation_availability_rate")
                    or 0.0
                )
                == 1.0
                for split in ("train", "dev")
            ),
            observed={
                split: (stage135_summary.get(split) or {}).get(
                    "sidecar_observation_availability_rate"
                )
                for split in ("train", "dev")
            },
            expected={"train": 1.0, "dev": 1.0},
        ),
        _check(
            name="stage135_answer_invariance_preserved",
            passed=all(
                _answer_invariance_summary_is_zero(summary) for summary in invariance.values()
            )
            and set(invariance) == {"train", "dev"},
            observed=invariance,
            expected="zero train/dev F1, citation, and changed-answer deltas",
        ),
        _check(
            name="stage135_train_opportunity_boundary_recorded",
            passed=_numeric_field_equals(train, "append_pool_incremental_gold_hit_count", 9)
            and _numeric_field_equals(train, "sidecar_incremental_gold_hit_count", 0),
            observed={
                "opportunities": train.get("append_pool_incremental_gold_hit_count"),
                "captures": train.get("sidecar_incremental_gold_hit_count"),
            },
            expected={"opportunities": 9, "captures": 0},
        ),
        _check(
            name="stage135_dev_opportunity_boundary_recorded",
            passed=_numeric_field_equals(dev, "append_pool_incremental_gold_hit_count", 1)
            and _numeric_field_equals(dev, "sidecar_incremental_gold_hit_count", 0),
            observed={
                "opportunities": dev.get("append_pool_incremental_gold_hit_count"),
                "captures": dev.get("sidecar_incremental_gold_hit_count"),
            },
            expected={"opportunities": 1, "captures": 0},
        ),
        _check(
            name="orchestrator_routes_answer_and_verification_to_stage116",
            passed=channel_routing.get("answer_generation") == "stage116_primary_answer_context"
            and channel_routing.get("answer_verification")
            == "stage116_prefix_verification_context",
            observed=channel_routing,
            expected="Stage116 generation and verification channels",
        ),
        _check(
            name="sidecar_blocked_from_answer_paths",
            passed=consumer_policy.get("answer_generation_allowed") is False
            and consumer_policy.get("answer_verification_context_allowed") is False
            and consumer_policy.get("primary_context_replacement_allowed") is False,
            observed=consumer_policy,
            expected="sidecar blocked from generation, verification, and replacement",
        ),
        _check(
            name="public_trace_excludes_private_fields",
            passed=all(
                public_trace.get(key) is False
                for key in (
                    "contains_raw_question_text",
                    "contains_raw_answer_text",
                    "contains_raw_document_text",
                    "contains_document_identifiers",
                    "contains_runtime_content_handles",
                    "contains_gold_labels",
                    "contains_test_membership",
                )
            ),
            observed=public_trace,
            expected="no private fields",
        ),
        _check(
            name="citation_verification_effectiveness_remains_unproven",
            passed=effectiveness.get("status") == "diagnostic_only_unproven"
            and effectiveness.get("citation_verification_effectiveness_claim_allowed") is False,
            observed=effectiveness,
            expected="diagnostic_only_unproven",
        ),
        _check(
            name="direct_stage128_all400_answer_context_remains_blocked",
            passed=stage135_summary.get("direct_stage128_all400_answer_context_remains_blocked")
            is True,
            observed=stage135_summary.get("direct_stage128_all400_answer_context_remains_blocked"),
            expected=True,
        ),
        _check(
            name="stage136_test_locked",
            passed=stage135_summary.get("can_open_final_test_gate_now") is False
            and stage135_summary.get("can_run_final_test_metrics_now") is False
            and stage135_summary.get("can_use_test_for_tuning") is False,
            observed={
                "open_gate": stage135_summary.get("can_open_final_test_gate_now"),
                "run_metrics": stage135_summary.get("can_run_final_test_metrics_now"),
                "tune": stage135_summary.get("can_use_test_for_tuning"),
            },
            expected="test locked",
        ),
        _check(
            name="stage136_runtime_defaults_unchanged",
            passed=stage135_summary.get("runtime_defaultization_allowed_now") is False
            and stage135_summary.get("default_runtime_policy") == "unchanged",
            observed={
                "allowed": stage135_summary.get("runtime_defaultization_allowed_now"),
                "policy": stage135_summary.get("default_runtime_policy"),
            },
            expected="unchanged",
        ),
        _check(
            name="stage136_no_fallback_strategies",
            passed=stage135_summary.get("fallback_strategies_enabled") is False
            and consumer_policy.get("fallback_strategy_allowed") is False,
            observed={
                "source": stage135_summary.get("fallback_strategies_enabled"),
                "orchestrator": consumer_policy.get("fallback_strategy_allowed"),
            },
            expected=False,
        ),
        _check(
            name="stage135_public_source_has_no_forbidden_keys",
            passed=stage135_summary.get("public_safe_forbidden_keys_found") == [],
            observed=stage135_summary.get("public_safe_forbidden_keys_found"),
            expected=[],
        ),
        _check(
            name="stage136_public_protocol_has_no_forbidden_keys",
            passed=not _forbidden_keys_found(report),
            observed=sorted(_forbidden_keys_found(report)),
            expected=[],
        ),
    ]


def _all_split_values_zero(
    stage135_summary: Mapping[str, Any],
    key: str,
) -> bool:
    return all(
        _numeric_field_equals(stage135_summary.get(split) or {}, key, 0)
        for split in ("train", "dev")
    )


def _answer_invariance_summary_is_zero(summary: Mapping[str, Any]) -> bool:
    return all(
        _numeric_field_equals(summary, key, 0)
        for key in (
            "verified_average_token_f1_delta",
            "verified_gold_citation_count_delta",
            "changed_verified_answer_count",
        )
    )


def _numeric_field_equals(summary: Mapping[str, Any], key: str, expected: int) -> bool:
    value = summary.get(key)
    return type(value) in (int, float) and value == expected


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    base = {
        "protocol_id": _PROTOCOL_ID,
        "orchestrator_implementation_available": True,
        "sidecar_effectiveness_status": "diagnostic_only_unproven",
        "can_claim_citation_verification_effectiveness": False,
        "can_claim_answer_quality_improvement": False,
        "can_claim_retrieval_improvement": False,
        "sidecar_can_generate_answer_text": False,
        "sidecar_can_enter_answer_verification_context": False,
        "sidecar_can_replace_primary_context": False,
        "direct_stage128_all400_answer_context_remains_blocked": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "runtime_defaultization_allowed_now": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }
    if failed_checks:
        return {
            **base,
            "status": "primeqa_hybrid_sidecar_agent_orchestrator_protocol_blocked",
            "failed_checks": failed_checks,
            "can_run_stage137_train_dev_validation_now": False,
            "recommended_next_direction": "review_stage136_orchestrator_protocol_failures",
        }
    return {
        **base,
        "status": "primeqa_hybrid_sidecar_agent_orchestrator_protocol_frozen",
        "failed_checks": [],
        "can_run_stage137_train_dev_validation_now": True,
        "recommended_next_direction": _NEXT_DIRECTION,
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "public_safe_summary_only": True,
        "raw_question_text_written": False,
        "raw_answer_text_written": False,
        "raw_document_text_written": False,
        "raw_document_ids_written": False,
        "raw_runtime_content_handles_written": False,
        "raw_candidate_rows_written": False,
        "raw_sample_ids_written": False,
        "split_files_loaded": False,
        "corpus_documents_loaded": False,
        "test_split_loaded": False,
        "final_test_metrics_run": False,
        "forbidden_keys_found": sorted(_forbidden_keys_found(report)),
    }


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


def _source_safety_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = report.get("stage135_summary") or {}
    keys = (
        "primary_context_identity_violation_count",
        "answer_generation_context_identity_violation_count",
        "sidecar_answer_context_leak_count",
        "sidecar_primary_overlap_count",
    )
    return [
        BarDatum(
            label=f"{split} {key}",
            value=float((summary.get(split) or {}).get(key) or 0),
            value_label=str(int((summary.get(split) or {}).get(key) or 0)),
        )
        for split in ("train", "dev")
        for key in keys
    ]


def _opportunity_capture_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = report.get("stage135_summary") or {}
    bars = []
    for split in ("train", "dev"):
        split_summary = summary.get(split) or {}
        for key, label in (
            ("append_pool_incremental_gold_hit_count", "append opportunities"),
            ("sidecar_incremental_gold_hit_count", "sidecar captures"),
        ):
            value = int(split_summary.get(key) or 0)
            bars.append(
                BarDatum(
                    label=f"{split} {label}",
                    value=float(value),
                    value_label=str(value),
                )
            )
    return bars


def _channel_permission_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    contract = (report.get("frozen_protocol") or {}).get("orchestrator_contract") or {}
    policy = contract.get("consumer_policy") or {}
    keys = (
        "observation_rendering_allowed",
        "evidence_gap_trace_allowed",
        "answer_generation_allowed",
        "answer_verification_context_allowed",
        "primary_context_replacement_allowed",
        "runtime_defaultization_allowed",
        "fallback_strategy_allowed",
    )
    return [
        BarDatum(
            label=key,
            value=1.0 if policy.get(key) else 0.0,
            value_label="allowed" if policy.get(key) else "blocked",
        )
        for key in keys
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    keys = (
        "orchestrator_implementation_available",
        "can_run_stage137_train_dev_validation_now",
        "can_claim_citation_verification_effectiveness",
        "sidecar_can_generate_answer_text",
        "sidecar_can_enter_answer_verification_context",
        "can_open_final_test_gate_now",
        "runtime_defaultization_allowed_now",
        "fallback_strategies_enabled",
    )
    return [
        BarDatum(
            label=key,
            value=1.0 if decision.get(key) else 0.0,
            value_label="true" if decision.get(key) else "false",
        )
        for key in keys
    ]


def _guard_check_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(check.get("name")),
            value=1.0 if check.get("passed") else 0.0,
            value_label="passed" if check.get("passed") else "failed",
        )
        for check in report.get("guard_checks") or []
    ]


def _check(*, name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
