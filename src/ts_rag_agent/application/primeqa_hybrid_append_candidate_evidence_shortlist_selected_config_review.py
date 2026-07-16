from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 133"
_CREATED_AT = "2026-07-16"
_REVIEW_ID = (
    "primeqa_hybrid_append_candidate_evidence_shortlist_selected_config_review_v1"
)
_SOURCE_STAGE132_STATUS = (
    "primeqa_hybrid_append_candidate_evidence_shortlist_validation_completed"
)
_SOURCE_STAGE132_ANALYSIS_ID = (
    "primeqa_hybrid_append_candidate_evidence_shortlist_validation_v1"
)
_SOURCE_STAGE132_NEXT = "review_append_candidate_evidence_shortlist_selected_config"
_SELECTED_CONFIG_ID = "prefix10_append_sidecar_probe_v1"
_SELECTED_PROFILE_ID = "stage132_prefix10_append_sidecar_probe_v1"
_NEXT_DIRECTION = "freeze_stage116_answer_context_plus_stage128_sidecar_agent_protocol"
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
class PrimeQAHybridAppendCandidateEvidenceShortlistSelectedConfigReviewVisualization:
    """One generated Stage133 selected-config review chart."""

    name: str
    path: str


def review_primeqa_hybrid_append_candidate_evidence_shortlist_selected_config(
    *,
    stage132_report_path: Path,
    user_confirmed_review: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Review the Stage132 selected append-candidate sidecar config."""

    started_at = time.perf_counter()
    stage132_report = _load_json_object(stage132_report_path)
    loaded_at = time.perf_counter()

    stage132_summary = _stage132_summary(stage132_report)
    selected_config_review = _selected_config_review(stage132_report)
    replacement_route_review = _replacement_route_review(stage132_report)
    agent_design_review = _agent_design_review(
        selected_config_review=selected_config_review,
        replacement_route_review=replacement_route_review,
    )
    preliminary = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "review_id": _REVIEW_ID,
        "review_scope": (
            "Public-safe selected-config review after Stage132 validated the "
            "Stage131 append-candidate evidence shortlist configs. This stage "
            "reads only the saved Stage132 aggregate report, does not load "
            "split files, corpus documents, raw candidate rows, raw questions, "
            "raw answers, raw document identifiers, or test data, does not run "
            "retrieval or answer metrics, does not run final metrics, does not "
            "add fallback strategies, and does not change runtime defaults."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_review),
            "confirmation_note": confirmation_note,
        },
        "source_files": {
            "stage132_report": _fingerprint(stage132_report_path),
        },
        "stage132_summary": stage132_summary,
        "selected_config_review": selected_config_review,
        "replacement_route_review": replacement_route_review,
        "agent_design_review": agent_design_review,
    }
    guard_checks = _guard_checks(
        report=preliminary,
        stage132_summary=stage132_summary,
        selected_config_review=selected_config_review,
        replacement_route_review=replacement_route_review,
        agent_design_review=agent_design_review,
        user_confirmed_review=user_confirmed_review,
        confirmation_note=confirmation_note,
    )
    checked_at = time.perf_counter()
    report = {
        **preliminary,
        "guard_checks": guard_checks,
        "decision": _decision(
            guard_checks=guard_checks,
            selected_config_review=selected_config_review,
            agent_design_review=agent_design_review,
        ),
        "timing_seconds": {
            "load_stage132_report": round(loaded_at - started_at, 3),
            "review_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_append_candidate_evidence_shortlist_selected_config_review_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridAppendCandidateEvidenceShortlistSelectedConfigReviewVisualization]:
    """Write SVG charts for Stage133 selected-config review."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage133_selected_sidecar_train_dev_deltas.svg": render_horizontal_bar_chart_svg(
            title="Stage133 selected sidecar train/dev deltas",
            bars=_selected_sidecar_delta_bars(report),
            x_label="delta",
            width=1540,
            margin_left=820,
        ),
        "stage133_replacement_route_risk.svg": render_horizontal_bar_chart_svg(
            title="Stage133 replacement route risk",
            bars=_replacement_route_risk_bars(report),
            x_label="count or rate",
            width=1680,
            margin_left=920,
        ),
        "stage133_sidecar_value_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage133 sidecar value flags",
            bars=_sidecar_value_flag_bars(report),
            x_label="1 means true",
            width=1560,
            margin_left=860,
        ),
        "stage133_agent_design_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage133 agent design decision flags",
            bars=_agent_design_decision_flag_bars(report),
            x_label="1 means true",
            width=1600,
            margin_left=900,
        ),
        "stage133_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage133 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1900,
            margin_left=1080,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridAppendCandidateEvidenceShortlistSelectedConfigReviewVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage132_summary(stage132_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage132_report.get("decision") or {}
    train_selection = stage132_report.get("train_selection") or {}
    dev_report = stage132_report.get("dev_report_observations") or {}
    public_safe = stage132_report.get("public_safe_contract") or {}
    guard_checks = stage132_report.get("guard_checks") or []
    return {
        "stage": stage132_report.get("stage"),
        "analysis_id": stage132_report.get("analysis_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "selected_config_id": decision.get("selected_config_id")
        or train_selection.get("selected_config_id"),
        "selected_profile_id": decision.get("selected_profile_id")
        or train_selection.get("selected_profile_id"),
        "eligible_config_count": decision.get("eligible_config_count")
        or train_selection.get("eligible_config_count"),
        "candidate_count": train_selection.get("candidate_count"),
        "dev_used_for_selection": train_selection.get("dev_used_for_selection")
        or dev_report.get("dev_used_for_selection"),
        "dev_used_for_retuning": train_selection.get("dev_used_for_retuning")
        or dev_report.get("dev_used_for_retuning"),
        "can_continue_train_dev_development": decision.get(
            "can_continue_train_dev_development"
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


def _selected_config_review(stage132_report: Mapping[str, Any]) -> dict[str, Any]:
    train_selection = stage132_report.get("train_selection") or {}
    dev_report = stage132_report.get("dev_report_observations") or {}
    profile_reports = stage132_report.get("profile_reports") or {}
    selected_profile_id = train_selection.get("selected_profile_id")
    selected_profile = profile_reports.get(str(selected_profile_id), {})
    selected_ranking = next(
        (
            row
            for row in train_selection.get("selection_ranking") or []
            if row.get("profile_id") == selected_profile_id
        ),
        {},
    )
    selected_dev = dev_report.get("selected_dev_summary") or {}
    train_summary = train_selection.get("selected_train_summary") or {}
    shortlist_config = selected_profile.get("shortlist_config") or {}
    return {
        "selected_config_found": selected_profile_id in profile_reports,
        "config_id": train_selection.get("selected_config_id"),
        "profile_id": selected_profile_id,
        "family_id": train_selection.get("selected_family_id"),
        "classification": _classification(train_summary, selected_dev),
        "train_guard_passed": bool(selected_ranking.get("guard_passed")),
        "train_failed_checks": selected_ranking.get("failed_checks") or [],
        "train": {
            "verified_average_token_f1_delta": train_summary.get(
                "verified_average_token_f1_delta"
            ),
            "verified_gold_citation_count_delta": train_summary.get(
                "verified_gold_citation_count_delta"
            ),
            "gold_hit_count_at_profile_depth_delta": train_summary.get(
                "gold_hit_count_at_profile_depth_delta"
            ),
            "changed_verified_answer_rate": train_summary.get(
                "changed_verified_answer_rate"
            ),
        },
        "dev": {
            "verified_average_token_f1_delta": (
                selected_dev.get("deltas_vs_stage116_control") or {}
            ).get("verified_average_token_f1_delta"),
            "verified_gold_citation_count_delta": (
                selected_dev.get("deltas_vs_stage116_control") or {}
            ).get("verified_gold_citation_count_delta"),
            "gold_hit_count_at_profile_depth_delta": (
                selected_dev.get("deltas_vs_stage116_control") or {}
            ).get("gold_hit_count_at_profile_depth_delta"),
            "changed_verified_answer_rate": selected_dev.get(
                "changed_verified_answer_rate_vs_stage116"
            ),
        },
        "shortlist_config": {
            "protected_prefix_slots": shortlist_config.get("protected_prefix_slots"),
            "replacement_append_slots": shortlist_config.get("replacement_append_slots"),
            "append_sidecar_slots": shortlist_config.get("append_sidecar_slots"),
            "append_sidecar_can_generate_answer_text": shortlist_config.get(
                "append_sidecar_can_generate_answer_text"
            ),
            "append_sidecar_can_support_citation_verification": shortlist_config.get(
                "append_sidecar_can_support_citation_verification"
            ),
        },
        "value_assessment": {
            "answer_quality_improved": _positive_float(
                train_summary.get("verified_average_token_f1_delta")
            ),
            "gold_citation_improved": _positive_int(
                train_summary.get("verified_gold_citation_count_delta")
            ),
            "retrieval_coverage_improved": _positive_int(
                train_summary.get("gold_hit_count_at_profile_depth_delta")
            ),
            "answer_context_preserved": (
                float(train_summary.get("changed_verified_answer_rate") or 0.0) == 0.0
            ),
            "dev_direction_confirms_neutral_safety": (
                float(
                    (selected_dev.get("deltas_vs_stage116_control") or {}).get(
                        "verified_average_token_f1_delta"
                    )
                    or 0.0
                )
                == 0.0
                and int(
                    (selected_dev.get("deltas_vs_stage116_control") or {}).get(
                        "verified_gold_citation_count_delta"
                    )
                    or 0
                )
                == 0
                and float(
                    selected_dev.get("changed_verified_answer_rate_vs_stage116") or 0.0
                )
                == 0.0
            ),
        },
        "limitations": [
            "No answer-quality gain was observed on train-CV or dev.",
            "No gold-citation-count gain was observed on train-CV or dev.",
            "The selected sidecar config does not let append candidates generate answer text.",
            "The selected sidecar config is not evidence for runtime defaultization.",
        ],
    }


def _replacement_route_review(stage132_report: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    dev_reviews = {
        review.get("profile_id"): review
        for review in (
            (stage132_report.get("dev_report_observations") or {}).get(
                "config_dev_reviews"
            )
            or []
        )
    }
    for row in (stage132_report.get("train_selection") or {}).get(
        "selection_ranking"
    ) or []:
        if row.get("config_id") == _SELECTED_CONFIG_ID:
            continue
        dev = dev_reviews.get(row.get("profile_id")) or {}
        dev_deltas = dev.get("deltas_vs_stage116_control") or {}
        region_shift = dev.get("selected_citation_region_shift") or {}
        rows.append(
            {
                "config_id": row.get("config_id"),
                "profile_id": row.get("profile_id"),
                "protected_prefix_slots": row.get("protected_prefix_slots"),
                "replacement_append_slots": row.get("replacement_append_slots"),
                "train_guard_passed": row.get("guard_passed"),
                "train_failed_checks": row.get("failed_checks") or [],
                "train_verified_f1_delta_vs_stage116": row.get(
                    "train_verified_f1_delta_vs_stage116"
                ),
                "train_gold_citation_count_delta_vs_stage116": row.get(
                    "train_gold_citation_count_delta_vs_stage116"
                ),
                "train_changed_answer_rate_vs_stage116": row.get(
                    "train_changed_answer_rate_vs_stage116"
                ),
                "dev_verified_f1_delta_vs_stage116": dev_deltas.get(
                    "verified_average_token_f1_delta"
                ),
                "dev_gold_citation_count_delta_vs_stage116": dev_deltas.get(
                    "verified_gold_citation_count_delta"
                ),
                "dev_changed_answer_rate_vs_stage116": dev.get(
                    "changed_verified_answer_rate_vs_stage116"
                ),
                "dev_append_selected_citations": region_shift.get(
                    "append_region_selected_citation_count"
                ),
                "dev_prefix_like_selected_citation_delta": region_shift.get(
                    "prefix_like_selected_citation_delta"
                ),
            }
        )
    return {
        "replacement_configs_reviewed": len(rows),
        "replacement_configs_failed": sum(1 for row in rows if not row["train_guard_passed"]),
        "all_replacement_configs_failed": bool(rows)
        and all(not row["train_guard_passed"] for row in rows),
        "primary_failure_pattern": "append_displacement_without_gold_gain",
        "recommendation": "stop_replacement_append_answer_context_route",
        "rows": rows,
    }


def _agent_design_review(
    *,
    selected_config_review: Mapping[str, Any],
    replacement_route_review: Mapping[str, Any],
) -> dict[str, Any]:
    value = selected_config_review["value_assessment"]
    return {
        "review_status": "safe_neutral_sidecar_supported_for_agent_protocol_design",
        "selected_config_supported_for_agent_design": True,
        "selected_config_supported_for_runtime_defaultization": False,
        "selected_config_supported_for_final_test_gate": False,
        "selected_config_supported_for_answer_context_replacement": False,
        "replacement_append_answer_context_route_stopped": replacement_route_review[
            "all_replacement_configs_failed"
        ],
        "sidecar_contract": {
            "primary_answer_context_source": "Stage116 top200 evidence shortlist behavior",
            "primary_answer_context_changed": False,
            "append_candidates_can_generate_answer_text": False,
            "append_candidates_can_replace_prefix_slots": False,
            "append_candidates_can_support_agent_observation": True,
            "append_candidates_can_support_future_citation_verification": True,
            "candidate_pool_depth_available_to_agent_sidecar": 400,
        },
        "observed_value": {
            "train_retrieval_coverage_delta": selected_config_review["train"][
                "gold_hit_count_at_profile_depth_delta"
            ],
            "dev_retrieval_coverage_delta": selected_config_review["dev"][
                "gold_hit_count_at_profile_depth_delta"
            ],
            "train_answer_quality_improved": value["answer_quality_improved"],
            "train_gold_citation_improved": value["gold_citation_improved"],
            "train_answer_context_preserved": value["answer_context_preserved"],
            "dev_direction_confirms_neutral_safety": value[
                "dev_direction_confirms_neutral_safety"
            ],
        },
        "risk_review": {
            "safe_but_neutral": True,
            "no_answer_quality_gain": not value["answer_quality_improved"],
            "no_gold_citation_gain": not value["gold_citation_improved"],
            "replacement_configs_reproduce_displacement_risk": replacement_route_review[
                "all_replacement_configs_failed"
            ],
            "final_test_not_run": True,
            "runtime_default_unchanged": True,
        },
        "recommended_next_stage": {
            "stage": "Stage134",
            "action": _NEXT_DIRECTION,
            "requires_user_confirmation": True,
            "test_locked": True,
            "runtime_defaults_unchanged": True,
            "fallback_strategies_enabled": False,
        },
    }


def _guard_checks(
    *,
    report: Mapping[str, Any],
    stage132_summary: Mapping[str, Any],
    selected_config_review: Mapping[str, Any],
    replacement_route_review: Mapping[str, Any],
    agent_design_review: Mapping[str, Any],
    user_confirmed_review: bool,
    confirmation_note: str,
) -> list[dict[str, Any]]:
    public_safe = _public_safe_contract(report)
    selected_train = selected_config_review["train"]
    selected_dev = selected_config_review["dev"]
    shortlist = selected_config_review["shortlist_config"]
    return [
        _check(
            name="user_confirmed_stage133_review",
            passed=user_confirmed_review and "Stage133" in confirmation_note,
            observed=confirmation_note,
            expected="user confirmed Stage133 review",
        ),
        _check(
            name="stage132_validation_completed",
            passed=stage132_summary.get("decision_status") == _SOURCE_STAGE132_STATUS,
            observed=stage132_summary.get("decision_status"),
            expected=_SOURCE_STAGE132_STATUS,
        ),
        _check(
            name="stage132_analysis_id_matches",
            passed=stage132_summary.get("analysis_id") == _SOURCE_STAGE132_ANALYSIS_ID,
            observed=stage132_summary.get("analysis_id"),
            expected=_SOURCE_STAGE132_ANALYSIS_ID,
        ),
        _check(
            name="stage132_recommends_selected_config_review",
            passed=stage132_summary.get("recommended_next_direction")
            == _SOURCE_STAGE132_NEXT,
            observed=stage132_summary.get("recommended_next_direction"),
            expected=_SOURCE_STAGE132_NEXT,
        ),
        _check(
            name="stage132_selected_sidecar_config_present",
            passed=selected_config_review.get("selected_config_found") is True
            and selected_config_review.get("config_id") == _SELECTED_CONFIG_ID
            and selected_config_review.get("profile_id") == _SELECTED_PROFILE_ID,
            observed={
                "found": selected_config_review.get("selected_config_found"),
                "config_id": selected_config_review.get("config_id"),
                "profile_id": selected_config_review.get("profile_id"),
            },
            expected={
                "config_id": _SELECTED_CONFIG_ID,
                "profile_id": _SELECTED_PROFILE_ID,
            },
        ),
        _check(
            name="stage132_selection_is_unique_train_cv_eligible_config",
            passed=int(stage132_summary.get("eligible_config_count") or 0) == 1
            and int(stage132_summary.get("candidate_count") or 0) == 3,
            observed={
                "eligible": stage132_summary.get("eligible_config_count"),
                "candidate_count": stage132_summary.get("candidate_count"),
            },
            expected="1 / 3",
        ),
        _check(
            name="selected_sidecar_is_safe_neutral_on_train",
            passed=float(selected_train.get("verified_average_token_f1_delta") or 0.0)
            == 0.0
            and int(selected_train.get("verified_gold_citation_count_delta") or 0) == 0
            and int(selected_train.get("gold_hit_count_at_profile_depth_delta") or 0) > 0
            and float(selected_train.get("changed_verified_answer_rate") or 0.0) == 0.0,
            observed=selected_train,
            expected="F1 0, citation 0, hit gain > 0, churn 0",
        ),
        _check(
            name="selected_sidecar_is_safe_neutral_on_dev_report_only",
            passed=float(selected_dev.get("verified_average_token_f1_delta") or 0.0)
            == 0.0
            and int(selected_dev.get("verified_gold_citation_count_delta") or 0) == 0
            and int(selected_dev.get("gold_hit_count_at_profile_depth_delta") or 0) >= 0
            and float(selected_dev.get("changed_verified_answer_rate") or 0.0) == 0.0,
            observed=selected_dev,
            expected="F1 0, citation 0, hit gain >= 0, churn 0",
        ),
        _check(
            name="selected_config_is_sidecar_not_answer_replacement",
            passed=int(shortlist.get("replacement_append_slots") or 0) == 0
            and shortlist.get("append_sidecar_can_generate_answer_text") is False
            and shortlist.get("append_sidecar_can_support_citation_verification") is True,
            observed=shortlist,
            expected="0 replacement slots, no answer text generation, citation support allowed",
        ),
        _check(
            name="replacement_append_answer_context_route_should_stop",
            passed=replacement_route_review.get("all_replacement_configs_failed") is True,
            observed={
                "failed": replacement_route_review.get("replacement_configs_failed"),
                "reviewed": replacement_route_review.get("replacement_configs_reviewed"),
                "recommendation": replacement_route_review.get("recommendation"),
            },
            expected="all replacement configs failed",
        ),
        _check(
            name="stage132_dev_report_only",
            passed=stage132_summary.get("dev_used_for_selection") is False
            and stage132_summary.get("dev_used_for_retuning") is False,
            observed={
                "dev_used_for_selection": stage132_summary.get("dev_used_for_selection"),
                "dev_used_for_retuning": stage132_summary.get("dev_used_for_retuning"),
            },
            expected=False,
        ),
        _check(
            name="stage132_test_runtime_and_fallback_boundaries_locked",
            passed=stage132_summary.get("can_open_final_test_gate_now") is False
            and stage132_summary.get("can_run_final_test_metrics_now") is False
            and stage132_summary.get("can_use_test_for_tuning") is False
            and stage132_summary.get("runtime_defaultization_allowed_now") is False
            and stage132_summary.get("fallback_strategies_enabled") is False
            and stage132_summary.get("default_runtime_policy") == "unchanged",
            observed=stage132_summary,
            expected="test locked, runtime unchanged, fallback disabled",
        ),
        _check(
            name="stage133_blocks_runtime_defaultization_and_test_gate",
            passed=agent_design_review.get("selected_config_supported_for_runtime_defaultization")
            is False
            and agent_design_review.get("selected_config_supported_for_final_test_gate")
            is False
            and agent_design_review.get("selected_config_supported_for_answer_context_replacement")
            is False,
            observed={
                "runtime_defaultization": agent_design_review.get(
                    "selected_config_supported_for_runtime_defaultization"
                ),
                "final_test_gate": agent_design_review.get(
                    "selected_config_supported_for_final_test_gate"
                ),
                "answer_context_replacement": agent_design_review.get(
                    "selected_config_supported_for_answer_context_replacement"
                ),
            },
            expected=False,
        ),
        _check(
            name="stage133_public_safe_contract_passed",
            passed=public_safe["forbidden_keys_found"] == [],
            observed=public_safe["forbidden_keys_found"],
            expected=[],
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    selected_config_review: Mapping[str, Any],
    agent_design_review: Mapping[str, Any],
) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    base = {
        "review_id": _REVIEW_ID,
        "selected_config_id": selected_config_review.get("config_id"),
        "selected_profile_id": selected_config_review.get("profile_id"),
        "selected_config_classification": selected_config_review.get("classification"),
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
                "primeqa_hybrid_append_candidate_evidence_shortlist_"
                "selected_config_review_blocked"
            ),
            "failed_checks": failed_checks,
            "selected_config_supported_for_agent_protocol_design": False,
            "can_continue_train_dev_development": False,
        }
    return {
        **base,
        "status": (
            "primeqa_hybrid_append_candidate_evidence_shortlist_"
            "selected_config_review_completed"
        ),
        "failed_checks": [],
        "selected_config_supported_for_agent_protocol_design": True,
        "selected_config_supported_for_runtime_defaultization": False,
        "selected_config_supported_for_answer_context_replacement": False,
        "replacement_append_answer_context_route_stopped": agent_design_review.get(
            "replacement_append_answer_context_route_stopped"
        ),
        "can_continue_train_dev_development": True,
        "recommended_next_direction": _NEXT_DIRECTION,
    }


def _classification(
    train_summary: Mapping[str, Any],
    selected_dev: Mapping[str, Any],
) -> str:
    train_f1 = float(train_summary.get("verified_average_token_f1_delta") or 0.0)
    train_citation = int(train_summary.get("verified_gold_citation_count_delta") or 0)
    train_churn = float(train_summary.get("changed_verified_answer_rate") or 0.0)
    dev_deltas = selected_dev.get("deltas_vs_stage116_control") or {}
    dev_f1 = float(dev_deltas.get("verified_average_token_f1_delta") or 0.0)
    dev_citation = int(dev_deltas.get("verified_gold_citation_count_delta") or 0)
    dev_churn = float(selected_dev.get("changed_verified_answer_rate_vs_stage116") or 0.0)
    if (
        train_f1 == 0.0
        and train_citation == 0
        and train_churn == 0.0
        and dev_f1 == 0.0
        and dev_citation == 0
        and dev_churn == 0.0
    ):
        return "safe_but_neutral_sidecar"
    return "requires_manual_review"


def _positive_float(value: Any) -> bool:
    return float(value or 0.0) > 0.0


def _positive_int(value: Any) -> bool:
    return int(value or 0) > 0


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


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


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


def _selected_sidecar_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    selected = report["selected_config_review"]
    bars = []
    for split in ("train", "dev"):
        split_review = selected[split]
        bars.extend(
            [
                BarDatum(
                    label=f"{split} F1 delta",
                    value=float(split_review["verified_average_token_f1_delta"]),
                    value_label=f"{float(split_review['verified_average_token_f1_delta']):+.4f}",
                ),
                BarDatum(
                    label=f"{split} gold citation delta",
                    value=float(split_review["verified_gold_citation_count_delta"]),
                    value_label=f"{int(split_review['verified_gold_citation_count_delta']):+d}",
                ),
                BarDatum(
                    label=f"{split} target-depth hit delta",
                    value=float(split_review["gold_hit_count_at_profile_depth_delta"]),
                    value_label=f"{int(split_review['gold_hit_count_at_profile_depth_delta']):+d}",
                ),
                BarDatum(
                    label=f"{split} changed answer rate",
                    value=float(split_review["changed_verified_answer_rate"]),
                    value_label=f"{float(split_review['changed_verified_answer_rate']):.2%}",
                ),
            ]
        )
    return bars


def _replacement_route_risk_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for row in (report.get("replacement_route_review") or {}).get("rows") or []:
        bars.extend(
            [
                BarDatum(
                    label=f"{row['config_id']} train citation delta",
                    value=float(row["train_gold_citation_count_delta_vs_stage116"]),
                    value_label=f"{int(row['train_gold_citation_count_delta_vs_stage116']):+d}",
                ),
                BarDatum(
                    label=f"{row['config_id']} train churn",
                    value=float(row["train_changed_answer_rate_vs_stage116"]),
                    value_label=f"{float(row['train_changed_answer_rate_vs_stage116']):.2%}",
                ),
                BarDatum(
                    label=f"{row['config_id']} dev citation delta",
                    value=float(row["dev_gold_citation_count_delta_vs_stage116"]),
                    value_label=f"{int(row['dev_gold_citation_count_delta_vs_stage116']):+d}",
                ),
                BarDatum(
                    label=f"{row['config_id']} dev churn",
                    value=float(row["dev_changed_answer_rate_vs_stage116"]),
                    value_label=f"{float(row['dev_changed_answer_rate_vs_stage116']):.2%}",
                ),
            ]
        )
    return bars


def _sidecar_value_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    flags = report["selected_config_review"]["value_assessment"]
    return [
        BarDatum(
            label=flag,
            value=1.0 if value else 0.0,
            value_label=str(bool(value)).lower(),
        )
        for flag, value in flags.items()
    ]


def _agent_design_decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    review = report["agent_design_review"]
    flags = (
        "selected_config_supported_for_agent_design",
        "selected_config_supported_for_runtime_defaultization",
        "selected_config_supported_for_final_test_gate",
        "selected_config_supported_for_answer_context_replacement",
        "replacement_append_answer_context_route_stopped",
    )
    return [
        BarDatum(
            label=flag,
            value=1.0 if review.get(flag) else 0.0,
            value_label=str(bool(review.get(flag))).lower(),
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
