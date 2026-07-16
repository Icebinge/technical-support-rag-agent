from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 134"
_CREATED_AT = "2026-07-16"
_PROTOCOL_ID = (
    "primeqa_hybrid_stage116_answer_context_stage128_sidecar_agent_protocol_v1"
)
_NEXT_DIRECTION = (
    "run_stage116_answer_context_stage128_sidecar_observation_train_cv_dev_validation"
)

_SOURCE_STAGE128_STATUS = "primeqa_hybrid_agent_retrieval_integration_protocol_frozen"
_SOURCE_STAGE128_PROTOCOL_ID = "primeqa_hybrid_agent_retrieval_integration_protocol_v1"
_SOURCE_STAGE129_STATUS = (
    "primeqa_hybrid_agent_retrieval_integration_validation_blocked_or_failed"
)
_SOURCE_STAGE129_ANALYSIS_ID = "primeqa_hybrid_agent_retrieval_integration_validation_v1"
_SOURCE_STAGE133_STATUS = (
    "primeqa_hybrid_append_candidate_evidence_shortlist_selected_config_review_completed"
)
_SOURCE_STAGE133_REVIEW_ID = (
    "primeqa_hybrid_append_candidate_evidence_shortlist_selected_config_review_v1"
)
_SOURCE_STAGE133_NEXT = "freeze_stage116_answer_context_plus_stage128_sidecar_agent_protocol"

_STAGE128_SELECTED_CONFIG_ID = "prefix_existing_dense_broad_append200_v1"
_STAGE132_SELECTED_CONFIG_ID = "prefix10_append_sidecar_probe_v1"
_STAGE132_SELECTED_PROFILE_ID = "stage132_prefix10_append_sidecar_probe_v1"
_ANSWER_CONTEXT_DEPTH = 10
_CANDIDATE_POOL_DEPTH = 400
_PREFIX_DEPTH = 200
_APPEND_START_RANK = 201
_APPEND_BUDGET = 200
_SIDECAR_OBSERVATION_SLOTS = 3
_MINIMUM_TRAIN_FOLDS = 5

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
        "source_doc_ids",
    }
)


@dataclass(frozen=True)
class PrimeQAHybridStage116AnswerContextStage128SidecarProtocolVisualization:
    """One generated Stage134 sidecar protocol chart."""

    name: str
    path: str


def freeze_primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol(
    *,
    stage128_protocol_path: Path,
    stage129_validation_path: Path,
    stage133_review_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage134 train/dev-only sidecar observation agent protocol."""

    started_at = time.perf_counter()
    stage128_protocol = _load_json_object(stage128_protocol_path)
    stage129_validation = _load_json_object(stage129_validation_path)
    stage133_review = _load_json_object(stage133_review_path)
    loaded_at = time.perf_counter()

    stage128_summary = _stage128_summary(stage128_protocol)
    stage129_summary = _stage129_summary(stage129_validation)
    stage133_summary = _stage133_summary(stage133_review)
    frozen_protocol = _frozen_protocol(
        stage128_summary=stage128_summary,
        stage129_summary=stage129_summary,
        stage133_summary=stage133_summary,
    )
    preliminary = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "protocol_scope": (
            "Public-safe protocol freeze for a Stage116 primary answer-context "
            "channel plus a Stage128/Stage132 sidecar-observation channel. This "
            "stage reads only saved aggregate public-safe Stage128, Stage129, "
            "and Stage133 reports, does not load split files, corpus documents, "
            "raw candidate rows, raw questions, raw answers, raw document "
            "identifiers, or test data, does not run retrieval, answering, "
            "validation metrics, final metrics, runtime defaultization, or "
            "fallback strategies, and does not change runtime defaults."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_protocol),
            "confirmation_note": confirmation_note,
        },
        "source_files": {
            "stage128_protocol": _fingerprint(stage128_protocol_path),
            "stage129_validation": _fingerprint(stage129_validation_path),
            "stage133_review": _fingerprint(stage133_review_path),
        },
        "stage128_summary": stage128_summary,
        "stage129_summary": stage129_summary,
        "stage133_summary": stage133_summary,
        "frozen_protocol": frozen_protocol,
    }
    guard_checks = _guard_checks(
        report=preliminary,
        stage128_summary=stage128_summary,
        stage129_summary=stage129_summary,
        stage133_summary=stage133_summary,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    checked_at = time.perf_counter()
    report = {
        **preliminary,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
        "timing_seconds": {
            "load_sources": round(loaded_at - started_at, 3),
            "freeze_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridStage116AnswerContextStage128SidecarProtocolVisualization]:
    """Write SVG charts for the Stage134 sidecar protocol freeze."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage134_protocol_components.svg": render_horizontal_bar_chart_svg(
            title="Stage134 protocol components",
            bars=_protocol_component_bars(report),
            x_label="count",
            width=1500,
            margin_left=820,
        ),
        "stage134_sidecar_train_dev_signals.svg": render_horizontal_bar_chart_svg(
            title="Stage134 sidecar train/dev signals",
            bars=_sidecar_signal_bars(report),
            x_label="delta",
            width=1540,
            margin_left=820,
        ),
        "stage134_channel_permission_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage134 channel permission flags",
            bars=_channel_permission_bars(report),
            x_label="1 means allowed",
            width=1640,
            margin_left=900,
        ),
        "stage134_risk_boundary_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage134 risk boundary flags",
            bars=_risk_boundary_bars(report),
            x_label="1 means true",
            width=1660,
            margin_left=920,
        ),
        "stage134_protocol_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage134 protocol decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1660,
            margin_left=920,
        ),
        "stage134_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage134 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=2000,
            margin_left=1160,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridStage116AnswerContextStage128SidecarProtocolVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage128_summary(stage128_protocol: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage128_protocol.get("decision") or {}
    frozen = stage128_protocol.get("frozen_protocol") or {}
    contract = frozen.get("agent_retrieval_contract") or {}
    regions = contract.get("rank_regions") or []
    prefix = _region_by_id(regions, "stage116_immutable_prefix")
    append = _region_by_id(regions, "stage128_append_expansion")
    public_safe = stage128_protocol.get("public_safe_contract") or {}
    guard_checks = stage128_protocol.get("guard_checks") or []
    return {
        "stage": stage128_protocol.get("stage"),
        "protocol_id": stage128_protocol.get("protocol_id"),
        "decision_status": decision.get("status"),
        "selected_config_id": decision.get("selected_config_id"),
        "candidate_pool_output_depth": contract.get("candidate_pool_output_depth"),
        "candidate_pool_is_not_automatic_answer_context": contract.get(
            "candidate_pool_is_not_automatic_answer_context"
        ),
        "answer_context_policy": contract.get("answer_context_policy"),
        "prefix_region": {
            "rank_start": prefix.get("rank_start"),
            "rank_end": prefix.get("rank_end"),
            "may_reorder": prefix.get("may_reorder"),
            "may_drop": prefix.get("may_drop"),
            "may_insert_expansion_candidate": prefix.get(
                "may_insert_expansion_candidate"
            ),
        },
        "append_region": {
            "rank_start": append.get("rank_start"),
            "rank_end": append.get("rank_end"),
            "source": append.get("source"),
            "append_budget": append.get("append_budget"),
            "may_insert_before_rank_201": append.get("may_insert_before_rank_201"),
        },
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "runtime_defaultization_allowed_now": decision.get(
            "runtime_defaultization_allowed_now"
        ),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "guard_check_count": len(guard_checks),
        "guard_check_passed_count": sum(1 for check in guard_checks if check.get("passed")),
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
    }


def _stage129_summary(stage129_validation: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage129_validation.get("decision") or {}
    train_cv = stage129_validation.get("train_cv_validation") or {}
    deltas = train_cv.get("deltas_vs_stage116_control") or {}
    dev_report = stage129_validation.get("dev_report_observations") or {}
    public_safe = stage129_validation.get("public_safe_contract") or {}
    guard_checks = stage129_validation.get("guard_checks") or []
    return {
        "stage": stage129_validation.get("stage"),
        "analysis_id": decision.get("analysis_id")
        or stage129_validation.get("analysis_id"),
        "decision_status": decision.get("status"),
        "selected_profile_id": decision.get("selected_profile_id"),
        "train_cv_validation_passed": decision.get("train_cv_validation_passed"),
        "train_cv_failed_checks": decision.get("train_cv_failed_checks") or [],
        "failed_checks": decision.get("failed_checks") or [],
        "train_verified_f1_delta_vs_stage116": deltas.get("verified_average_token_f1_delta"),
        "train_gold_citation_count_delta_vs_stage116": deltas.get(
            "verified_gold_citation_count_delta"
        ),
        "train_gold_hit_count_delta_vs_stage116": deltas.get(
            "gold_hit_count_at_profile_depth_delta"
        ),
        "train_changed_verified_answers_vs_stage116": deltas.get(
            "changed_verified_answers"
        ),
        "dev_validation_status": decision.get("dev_validation_status"),
        "dev_gate_status": decision.get("dev_gate_status"),
        "dev_changed_verified_answers_vs_stage116": dev_report.get(
            "dev_changed_verified_answers_vs_stage116_control"
        ),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "runtime_defaultization_allowed_now": decision.get(
            "runtime_defaultization_allowed_now"
        ),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "guard_check_count": len(guard_checks),
        "guard_check_passed_count": sum(1 for check in guard_checks if check.get("passed")),
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
    }


def _stage133_summary(stage133_review: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage133_review.get("decision") or {}
    selected = stage133_review.get("selected_config_review") or {}
    sidecar = (stage133_review.get("agent_design_review") or {}).get(
        "sidecar_contract"
    ) or {}
    public_safe = stage133_review.get("public_safe_contract") or {}
    guard_checks = stage133_review.get("guard_checks") or []
    return {
        "stage": stage133_review.get("stage"),
        "review_id": stage133_review.get("review_id") or decision.get("review_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "selected_config_id": decision.get("selected_config_id")
        or selected.get("config_id"),
        "selected_profile_id": decision.get("selected_profile_id")
        or selected.get("profile_id"),
        "selected_config_classification": decision.get(
            "selected_config_classification"
        )
        or selected.get("classification"),
        "selected_config_supported_for_agent_protocol_design": decision.get(
            "selected_config_supported_for_agent_protocol_design"
        ),
        "selected_config_supported_for_runtime_defaultization": decision.get(
            "selected_config_supported_for_runtime_defaultization"
        ),
        "selected_config_supported_for_answer_context_replacement": decision.get(
            "selected_config_supported_for_answer_context_replacement"
        ),
        "replacement_append_answer_context_route_stopped": decision.get(
            "replacement_append_answer_context_route_stopped"
        ),
        "train": selected.get("train") or {},
        "dev": selected.get("dev") or {},
        "shortlist_config": selected.get("shortlist_config") or {},
        "value_assessment": selected.get("value_assessment") or {},
        "sidecar_contract": sidecar,
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "runtime_defaultization_allowed_now": decision.get(
            "runtime_defaultization_allowed_now"
        ),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "guard_check_count": len(guard_checks),
        "guard_check_passed_count": sum(1 for check in guard_checks if check.get("passed")),
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
    }


def _frozen_protocol(
    *,
    stage128_summary: Mapping[str, Any],
    stage129_summary: Mapping[str, Any],
    stage133_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "route_name": "stage116_answer_context_plus_stage128_sidecar_observation",
        "source_reviews": {
            "stage128_protocol": {
                "status": stage128_summary.get("decision_status"),
                "selected_config_id": stage128_summary.get("selected_config_id"),
                "candidate_pool_depth": stage128_summary.get(
                    "candidate_pool_output_depth"
                ),
            },
            "stage129_direct_integration_validation": {
                "status": stage129_summary.get("decision_status"),
                "train_cv_validation_passed": stage129_summary.get(
                    "train_cv_validation_passed"
                ),
                "failed_checks": stage129_summary.get("train_cv_failed_checks"),
            },
            "stage133_selected_sidecar_review": {
                "status": stage133_summary.get("decision_status"),
                "selected_config_id": stage133_summary.get("selected_config_id"),
                "classification": stage133_summary.get(
                    "selected_config_classification"
                ),
            },
        },
        "agent_channel_contract": {
            "primary_answer_context_channel": {
                "channel_id": "stage116_primary_answer_context",
                "source": "Stage116 top200 evidence shortlist behavior",
                "answer_context_depth": _ANSWER_CONTEXT_DEPTH,
                "rank_source": "Stage116 immutable prefix ranks 1-200",
                "allowed_to_generate_answer_text": True,
                "sidecar_candidates_included": False,
                "may_be_reordered_by_sidecar": False,
                "may_be_replaced_by_sidecar": False,
            },
            "sidecar_observation_channel": {
                "channel_id": "stage128_stage132_sidecar_observation",
                "source": "Stage128 top400 candidate pool plus Stage132 selected sidecar config",
                "candidate_pool_depth": _CANDIDATE_POOL_DEPTH,
                "source_candidate_pool_config": _STAGE128_SELECTED_CONFIG_ID,
                "selected_sidecar_config": _STAGE132_SELECTED_CONFIG_ID,
                "selected_sidecar_profile": _STAGE132_SELECTED_PROFILE_ID,
                "append_region_rank_start": _APPEND_START_RANK,
                "append_region_rank_end": _CANDIDATE_POOL_DEPTH,
                "append_budget": _APPEND_BUDGET,
                "observation_slots": _SIDECAR_OBSERVATION_SLOTS,
                "allowed_to_generate_answer_text": False,
                "allowed_to_replace_primary_context": False,
                "allowed_to_support_agent_observation": True,
                "allowed_to_support_future_citation_verification": True,
            },
        },
        "agent_observation_interface": {
            "interface_id": "stage116_answer_context_with_sidecar_observation_v1",
            "public_report_contains_raw_candidate_rows": False,
            "primary_context_record_fields_to_validate": [
                "runtime_content_handle",
                "primary_context_rank",
                "primary_context_source_region",
                "retrieval_score_summary",
            ],
            "sidecar_observation_fields_to_validate": [
                "runtime_content_handle",
                "sidecar_observation_rank",
                "sidecar_source_region",
                "sidecar_route_family",
                "sidecar_score_summary",
                "citation_verification_signal",
            ],
            "allowed_runtime_visible_signals": [
                "runtime query text",
                "runtime corpus title/body/section content",
                "Stage116 primary context rank",
                "Stage128 append region rank",
                "Stage132 sidecar observation rank",
                "route-family score summaries",
            ],
            "forbidden_runtime_signals": [
                "test membership",
                "gold labels",
                "answer document labels",
                "source-provided candidate labels",
                "dev-selected thresholds",
                "raw private rows in public artifacts",
            ],
        },
        "agent_consumer_policy": {
            "allowed_train_dev_consumers": [
                {
                    "consumer_id": "sidecar_observation_rendering",
                    "allowed": True,
                    "requires_stage135_validation": True,
                },
                {
                    "consumer_id": "citation_verification_probe",
                    "allowed": True,
                    "requires_stage135_validation": True,
                },
                {
                    "consumer_id": "evidence_gap_explanation",
                    "allowed": True,
                    "requires_stage135_validation": True,
                },
            ],
            "blocked_consumers": [
                {
                    "consumer_id": "sidecar_answer_text_generation",
                    "blocked": True,
                    "reason": (
                        "Stage133 selected sidecar is safe but neutral and "
                        "cannot generate answer text."
                    ),
                },
                {
                    "consumer_id": "sidecar_primary_context_replacement",
                    "blocked": True,
                    "reason": "Replacement append routes reproduced displacement risk.",
                },
                {
                    "consumer_id": "direct_stage128_all400_answer_context",
                    "blocked": True,
                    "reason": (
                        "Stage129 direct integration failed the train-CV "
                        "gold-citation guard."
                    ),
                },
                {
                    "consumer_id": "runtime_default_retrieval_route",
                    "blocked": True,
                    "reason": "Stage134 is a train/dev protocol freeze, not defaultization.",
                },
                {
                    "consumer_id": "fallback_strategy_route",
                    "blocked": True,
                    "reason": "Fallback strategies remain disabled.",
                },
            ],
        },
        "validation_plan": {
            "next_stage": "Stage135",
            "action": _NEXT_DIRECTION,
            "selection_split": "train",
            "selection_mode": "train_grouped_cross_validation_sidecar_observation_integrity",
            "minimum_train_folds": _MINIMUM_TRAIN_FOLDS,
            "validation_split": "dev",
            "dev_mode": "single_pass_report_only_no_retuning",
            "primary_checks": [
                (
                    "Stage116 primary answer context remains byte-for-byte "
                    "unchanged at the policy level"
                ),
                "sidecar observation records are isolated from answer-text generation",
                "sidecar observation records are isolated from prefix replacement",
                "direct Stage128 all-400 answer context remains blocked",
                "test split remains unloaded",
            ],
            "metrics_to_report": [
                "primary answer-context identity status",
                "sidecar observation availability count",
                "sidecar citation-verification signal coverage",
                "answer F1 delta expected to remain zero",
                "gold citation delta expected to remain zero",
                "changed answer count expected to remain zero",
            ],
            "test_rules": {
                "test_access_allowed": False,
                "final_test_metrics_allowed": False,
                "test_tuning_allowed": False,
            },
            "runtime_rules": {
                "default_runtime_policy": "unchanged",
                "runtime_defaultization_allowed_in_stage134": False,
                "fallback_strategies_enabled": False,
            },
        },
        "risk_controls": {
            "direct_stage128_agent_integration_path_remains_blocked": True,
            "replacement_append_answer_context_route_remains_stopped": True,
            "sidecar_is_safe_but_neutral_not_quality_gain": True,
            "answer_context_replacement_allowed": False,
            "answer_text_generation_from_sidecar_allowed": False,
            "final_test_gate_allowed": False,
            "runtime_defaultization_allowed": False,
        },
    }


def _guard_checks(
    *,
    report: Mapping[str, Any],
    stage128_summary: Mapping[str, Any],
    stage129_summary: Mapping[str, Any],
    stage133_summary: Mapping[str, Any],
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> list[dict[str, Any]]:
    frozen = report["frozen_protocol"]
    channels = frozen["agent_channel_contract"]
    interface = frozen["agent_observation_interface"]
    policy = frozen["agent_consumer_policy"]
    validation = frozen["validation_plan"]
    public_safe = _public_safe_contract(report)
    sidecar = stage133_summary.get("sidecar_contract") or {}
    shortlist = stage133_summary.get("shortlist_config") or {}
    train = stage133_summary.get("train") or {}
    dev = stage133_summary.get("dev") or {}
    return [
        _check(
            name="user_confirmed_stage134_protocol",
            passed=user_confirmed_protocol and "Stage134" in confirmation_note,
            observed=confirmation_note,
            expected="user confirmed Stage134 protocol",
        ),
        _check(
            name="stage128_protocol_frozen",
            passed=stage128_summary.get("decision_status") == _SOURCE_STAGE128_STATUS
            and stage128_summary.get("protocol_id") == _SOURCE_STAGE128_PROTOCOL_ID,
            observed={
                "status": stage128_summary.get("decision_status"),
                "protocol_id": stage128_summary.get("protocol_id"),
            },
            expected={
                "status": _SOURCE_STAGE128_STATUS,
                "protocol_id": _SOURCE_STAGE128_PROTOCOL_ID,
            },
        ),
        _check(
            name="stage128_candidate_pool_contract_available_as_sidecar_source",
            passed=stage128_summary.get("selected_config_id")
            == _STAGE128_SELECTED_CONFIG_ID
            and int(stage128_summary.get("candidate_pool_output_depth") or 0)
            == _CANDIDATE_POOL_DEPTH
            and stage128_summary.get("candidate_pool_is_not_automatic_answer_context")
            is True
            and int(
                (stage128_summary.get("prefix_region") or {}).get("rank_end") or 0
            )
            == _PREFIX_DEPTH
            and int(
                (stage128_summary.get("append_region") or {}).get("rank_start") or 0
            )
            == _APPEND_START_RANK,
            observed=stage128_summary,
            expected="Stage116 top200 prefix plus Stage128 ranks 201-400 sidecar source",
        ),
        _check(
            name="stage129_direct_stage128_integration_remains_blocked",
            passed=stage129_summary.get("decision_status") == _SOURCE_STAGE129_STATUS
            and stage129_summary.get("analysis_id") == _SOURCE_STAGE129_ANALYSIS_ID
            and stage129_summary.get("train_cv_validation_passed") is False
            and "gold_citation_count_delta_vs_stage116_non_negative"
            in (stage129_summary.get("train_cv_failed_checks") or []),
            observed=stage129_summary,
            expected="Stage129 direct all-400 path failed train-CV citation guard",
        ),
        _check(
            name="stage129_test_runtime_and_fallback_boundaries_locked",
            passed=stage129_summary.get("can_open_final_test_gate_now") is False
            and stage129_summary.get("can_run_final_test_metrics_now") is False
            and stage129_summary.get("can_use_test_for_tuning") is False
            and stage129_summary.get("runtime_defaultization_allowed_now") is False
            and stage129_summary.get("fallback_strategies_enabled") is False
            and stage129_summary.get("default_runtime_policy") == "unchanged",
            observed=stage129_summary,
            expected="test locked, runtime unchanged, fallback disabled",
        ),
        _check(
            name="stage133_selected_sidecar_review_completed",
            passed=stage133_summary.get("decision_status") == _SOURCE_STAGE133_STATUS
            and stage133_summary.get("review_id") == _SOURCE_STAGE133_REVIEW_ID,
            observed={
                "status": stage133_summary.get("decision_status"),
                "review_id": stage133_summary.get("review_id"),
            },
            expected={
                "status": _SOURCE_STAGE133_STATUS,
                "review_id": _SOURCE_STAGE133_REVIEW_ID,
            },
        ),
        _check(
            name="stage133_recommends_stage134_protocol",
            passed=stage133_summary.get("recommended_next_direction")
            == _SOURCE_STAGE133_NEXT,
            observed=stage133_summary.get("recommended_next_direction"),
            expected=_SOURCE_STAGE133_NEXT,
        ),
        _check(
            name="stage133_selected_sidecar_is_safe_neutral",
            passed=stage133_summary.get("selected_config_id") == _STAGE132_SELECTED_CONFIG_ID
            and stage133_summary.get("selected_profile_id") == _STAGE132_SELECTED_PROFILE_ID
            and stage133_summary.get("selected_config_classification")
            == "safe_but_neutral_sidecar"
            and stage133_summary.get("selected_config_supported_for_agent_protocol_design")
            is True,
            observed=stage133_summary,
            expected="selected safe_but_neutral_sidecar supports protocol design",
        ),
        _check(
            name="stage133_sidecar_cannot_generate_or_replace_answer_context",
            passed=sidecar.get("append_candidates_can_generate_answer_text") is False
            and sidecar.get("append_candidates_can_replace_prefix_slots") is False
            and sidecar.get("append_candidates_can_support_agent_observation") is True
            and sidecar.get("append_candidates_can_support_future_citation_verification")
            is True
            and int(shortlist.get("replacement_append_slots") or 0) == 0
            and int(shortlist.get("append_sidecar_slots") or 0)
            == _SIDECAR_OBSERVATION_SLOTS,
            observed={"sidecar_contract": sidecar, "shortlist_config": shortlist},
            expected="sidecar only, no answer generation, no prefix replacement",
        ),
        _check(
            name="stage133_selected_sidecar_preserves_answer_metrics",
            passed=float(train.get("verified_average_token_f1_delta") or 0.0) == 0.0
            and int(train.get("verified_gold_citation_count_delta") or 0) == 0
            and float(train.get("changed_verified_answer_rate") or 0.0) == 0.0
            and float(dev.get("verified_average_token_f1_delta") or 0.0) == 0.0
            and int(dev.get("verified_gold_citation_count_delta") or 0) == 0
            and float(dev.get("changed_verified_answer_rate") or 0.0) == 0.0,
            observed={"train": train, "dev": dev},
            expected="F1 0, citation 0, changed answer rate 0 on train/dev",
        ),
        _check(
            name="stage133_replacement_route_stopped",
            passed=stage133_summary.get("replacement_append_answer_context_route_stopped")
            is True
            and stage133_summary.get(
                "selected_config_supported_for_answer_context_replacement"
            )
            is False,
            observed=stage133_summary,
            expected="replacement append answer-context route stopped",
        ),
        _check(
            name="stage134_primary_answer_context_is_stage116_only",
            passed=channels["primary_answer_context_channel"]["answer_context_depth"]
            == _ANSWER_CONTEXT_DEPTH
            and channels["primary_answer_context_channel"]["sidecar_candidates_included"]
            is False
            and channels["primary_answer_context_channel"]["may_be_reordered_by_sidecar"]
            is False
            and channels["primary_answer_context_channel"]["may_be_replaced_by_sidecar"]
            is False,
            observed=channels["primary_answer_context_channel"],
            expected="Stage116 primary answer context unchanged by sidecar",
        ),
        _check(
            name="stage134_sidecar_channel_is_observation_only",
            passed=channels["sidecar_observation_channel"]["candidate_pool_depth"]
            == _CANDIDATE_POOL_DEPTH
            and channels["sidecar_observation_channel"]["observation_slots"]
            == _SIDECAR_OBSERVATION_SLOTS
            and channels["sidecar_observation_channel"][
                "allowed_to_generate_answer_text"
            ]
            is False
            and channels["sidecar_observation_channel"][
                "allowed_to_replace_primary_context"
            ]
            is False,
            observed=channels["sidecar_observation_channel"],
            expected="sidecar observation channel cannot generate or replace",
        ),
        _check(
            name="stage134_interface_uses_runtime_visible_fields",
            passed=all(
                field not in _FORBIDDEN_PUBLIC_KEYS
                for field in interface["primary_context_record_fields_to_validate"]
                + interface["sidecar_observation_fields_to_validate"]
            ),
            observed=interface,
            expected="runtime-visible fields only",
        ),
        _check(
            name="stage134_blocks_unsafe_consumers",
            passed=_consumer_blocked(policy, "sidecar_answer_text_generation")
            and _consumer_blocked(policy, "sidecar_primary_context_replacement")
            and _consumer_blocked(policy, "direct_stage128_all400_answer_context")
            and _consumer_blocked(policy, "runtime_default_retrieval_route")
            and _consumer_blocked(policy, "fallback_strategy_route"),
            observed=policy["blocked_consumers"],
            expected="unsafe sidecar/runtime/fallback consumers blocked",
        ),
        _check(
            name="stage134_allowed_consumers_require_stage135_validation",
            passed=all(
                item["allowed"] is True and item["requires_stage135_validation"] is True
                for item in policy["allowed_train_dev_consumers"]
            ),
            observed=policy["allowed_train_dev_consumers"],
            expected="allowed consumers are Stage135-validation gated",
        ),
        _check(
            name="stage134_validation_plan_train_cv_dev_report_only",
            passed=validation["selection_split"] == "train"
            and int(validation["minimum_train_folds"]) >= _MINIMUM_TRAIN_FOLDS
            and validation["validation_split"] == "dev"
            and validation["dev_mode"] == "single_pass_report_only_no_retuning",
            observed=validation,
            expected="train grouped-CV, dev report-only",
        ),
        _check(
            name="stage134_test_runtime_and_fallback_boundaries_locked",
            passed=validation["test_rules"]["test_access_allowed"] is False
            and validation["test_rules"]["final_test_metrics_allowed"] is False
            and validation["test_rules"]["test_tuning_allowed"] is False
            and validation["runtime_rules"]["default_runtime_policy"] == "unchanged"
            and validation["runtime_rules"][
                "runtime_defaultization_allowed_in_stage134"
            ]
            is False
            and validation["runtime_rules"]["fallback_strategies_enabled"] is False,
            observed=validation,
            expected="test locked, runtime unchanged, fallback disabled",
        ),
        _check(
            name="stage134_public_safe_contract_passed",
            passed=public_safe["forbidden_keys_found"] == [],
            observed=public_safe["forbidden_keys_found"],
            expected=[],
        ),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    base = {
        "protocol_id": _PROTOCOL_ID,
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
            "status": (
                "primeqa_hybrid_stage116_answer_context_stage128_sidecar_"
                "agent_protocol_blocked"
            ),
            "failed_checks": failed_checks,
            "can_continue_agent_design_implementation": False,
            "can_build_sidecar_observation_adapter_now": False,
        }
    return {
        **base,
        "status": (
            "primeqa_hybrid_stage116_answer_context_stage128_sidecar_"
            "agent_protocol_frozen"
        ),
        "failed_checks": [],
        "recommended_next_direction": _NEXT_DIRECTION,
        "can_continue_agent_design_implementation": True,
        "can_build_sidecar_observation_adapter_now": True,
        "primary_answer_context_source": "Stage116 top200 evidence shortlist behavior",
        "sidecar_observation_source": "Stage128 top400 candidate pool via Stage132 sidecar config",
        "sidecar_can_generate_answer_text": False,
        "sidecar_can_replace_primary_context": False,
        "direct_stage128_all400_answer_context_remains_blocked": True,
        "replacement_append_answer_context_route_remains_stopped": True,
    }


def _protocol_component_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    sidecar = report["frozen_protocol"]["agent_channel_contract"][
        "sidecar_observation_channel"
    ]
    return [
        BarDatum(
            label="primary answer context depth",
            value=float(_ANSWER_CONTEXT_DEPTH),
            value_label=str(_ANSWER_CONTEXT_DEPTH),
        ),
        BarDatum(
            label="sidecar observation slots",
            value=float(sidecar["observation_slots"]),
            value_label=str(sidecar["observation_slots"]),
        ),
        BarDatum(
            label="sidecar candidate pool depth",
            value=float(sidecar["candidate_pool_depth"]),
            value_label=str(sidecar["candidate_pool_depth"]),
        ),
        BarDatum(
            label="append budget",
            value=float(sidecar["append_budget"]),
            value_label=str(sidecar["append_budget"]),
        ),
    ]


def _sidecar_signal_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = report["stage133_summary"]
    bars = []
    for split in ("train", "dev"):
        values = summary[split]
        bars.extend(
            [
                BarDatum(
                    label=f"{split} F1 delta",
                    value=float(values["verified_average_token_f1_delta"]),
                    value_label=f"{float(values['verified_average_token_f1_delta']):+.4f}",
                ),
                BarDatum(
                    label=f"{split} gold citation delta",
                    value=float(values["verified_gold_citation_count_delta"]),
                    value_label=f"{int(values['verified_gold_citation_count_delta']):+d}",
                ),
                BarDatum(
                    label=f"{split} target-depth hit delta",
                    value=float(values["gold_hit_count_at_profile_depth_delta"]),
                    value_label=f"{int(values['gold_hit_count_at_profile_depth_delta']):+d}",
                ),
                BarDatum(
                    label=f"{split} changed answer rate",
                    value=float(values["changed_verified_answer_rate"]),
                    value_label=f"{float(values['changed_verified_answer_rate']):.2%}",
                ),
            ]
        )
    return bars


def _channel_permission_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    channels = report["frozen_protocol"]["agent_channel_contract"]
    primary = channels["primary_answer_context_channel"]
    sidecar = channels["sidecar_observation_channel"]
    return [
        BarDatum(
            label="primary can generate answer text",
            value=1.0 if primary["allowed_to_generate_answer_text"] else 0.0,
            value_label=str(bool(primary["allowed_to_generate_answer_text"])).lower(),
        ),
        BarDatum(
            label="sidecar can generate answer text",
            value=1.0 if sidecar["allowed_to_generate_answer_text"] else 0.0,
            value_label=str(bool(sidecar["allowed_to_generate_answer_text"])).lower(),
        ),
        BarDatum(
            label="sidecar can replace primary context",
            value=1.0 if sidecar["allowed_to_replace_primary_context"] else 0.0,
            value_label=str(bool(sidecar["allowed_to_replace_primary_context"])).lower(),
        ),
        BarDatum(
            label="sidecar supports observation",
            value=1.0 if sidecar["allowed_to_support_agent_observation"] else 0.0,
            value_label=str(bool(sidecar["allowed_to_support_agent_observation"])).lower(),
        ),
        BarDatum(
            label="sidecar supports citation verification",
            value=1.0
            if sidecar["allowed_to_support_future_citation_verification"]
            else 0.0,
            value_label=str(
                bool(sidecar["allowed_to_support_future_citation_verification"])
            ).lower(),
        ),
    ]


def _risk_boundary_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    risks = report["frozen_protocol"]["risk_controls"]
    return [
        BarDatum(
            label=name,
            value=1.0 if value else 0.0,
            value_label=str(bool(value)).lower(),
        )
        for name, value in risks.items()
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    flags = (
        "can_continue_agent_design_implementation",
        "can_build_sidecar_observation_adapter_now",
        "can_open_final_test_gate_now",
        "can_run_final_test_metrics_now",
        "can_use_test_for_tuning",
        "runtime_defaultization_allowed_now",
        "fallback_strategies_enabled",
        "sidecar_can_generate_answer_text",
        "sidecar_can_replace_primary_context",
        "direct_stage128_all400_answer_context_remains_blocked",
        "replacement_append_answer_context_route_remains_stopped",
    )
    return [
        BarDatum(
            label=flag,
            value=1.0 if decision.get(flag) else 0.0,
            value_label=str(bool(decision.get(flag))).lower(),
        )
        for flag in flags
    ]


def _guard_check_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(check["name"]),
            value=1.0 if check["passed"] else 0.0,
            value_label="pass" if check["passed"] else "fail",
        )
        for check in report.get("guard_checks") or []
    ]


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "public_safe_summary_only": True,
        "raw_question_text_written": False,
        "raw_answer_text_written": False,
        "raw_document_text_written": False,
        "raw_document_ids_written": False,
        "raw_candidate_rows_written": False,
        "raw_sample_ids_written": False,
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


def _region_by_id(
    regions: Sequence[Mapping[str, Any]],
    region_id: str,
) -> Mapping[str, Any]:
    return next((region for region in regions if region.get("region_id") == region_id), {})


def _consumer_blocked(policy: Mapping[str, Any], consumer_id: str) -> bool:
    return any(
        item.get("consumer_id") == consumer_id and item.get("blocked") is True
        for item in policy.get("blocked_consumers") or []
    )


def _fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _check(
    *,
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }
