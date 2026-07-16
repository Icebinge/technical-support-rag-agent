from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 130"
_CREATED_AT = "2026-07-16"
_REVIEW_ID = "primeqa_hybrid_stage129_agent_integration_failure_review_v1"
_SOURCE_STAGE = "Stage 129"
_SOURCE_ANALYSIS_ID = "primeqa_hybrid_agent_retrieval_integration_validation_v1"
_SOURCE_STATUS = "primeqa_hybrid_agent_retrieval_integration_validation_blocked_or_failed"
_SOURCE_NEXT = "review_stage129_agent_integration_failure_patterns"
_STAGE116_PROFILE_ID = "stage116_top200_agent_pool_control"
_STAGE128_PROFILE_ID = "stage128_prefix_append_top400_agent_pool"
_NEXT_DIRECTION = "freeze_append_candidate_evidence_shortlist_redesign_protocol"
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
class PrimeQAHybridAgentIntegrationFailureReviewVisualization:
    """One generated Stage130 agent-integration failure-review chart."""

    name: str
    path: str


def review_primeqa_hybrid_agent_integration_failure_patterns(
    *,
    stage129_report_path: Path,
    user_confirmed_review: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Review Stage129 public-safe agent-integration failure patterns."""

    started_at = time.perf_counter()
    stage129_report = _load_json_object(stage129_report_path)
    loaded_at = time.perf_counter()

    source_summary = _stage129_source_summary(stage129_report)
    train_review = _split_failure_review(stage129_report, split="train_cv")
    dev_review = _split_failure_review(stage129_report, split="dev")
    failure_patterns = _failure_patterns(train_review=train_review, dev_review=dev_review)
    action_boundary = _action_boundary(failure_patterns)
    preliminary = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "review_id": _REVIEW_ID,
        "review_scope": (
            "Public-safe Stage130 review of the Stage129 agent retrieval "
            "integration validation failure. This stage reads only the saved "
            "Stage129 aggregate report, does not load split files, corpus "
            "documents, raw candidate rows, raw questions, raw answers, raw "
            "document identifiers, or test data, does not run retrieval or "
            "answer metrics, does not run final metrics, does not add "
            "fallback strategies, and does not change runtime defaults."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_review),
            "confirmation_note": confirmation_note,
        },
        "source_files": {
            "stage129_report": _fingerprint(stage129_report_path),
        },
        "source_summary": source_summary,
        "train_cv_failure_review": train_review,
        "dev_report_only_review": dev_review,
        "failure_patterns": failure_patterns,
        "action_boundary": action_boundary,
    }
    guard_checks = _guard_checks(
        report=preliminary,
        source_summary=source_summary,
        train_review=train_review,
        failure_patterns=failure_patterns,
        user_confirmed_review=user_confirmed_review,
        confirmation_note=confirmation_note,
    )
    checked_at = time.perf_counter()
    report = {
        **preliminary,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks=guard_checks, action_boundary=action_boundary),
        "timing_seconds": {
            "load_stage129_report": round(loaded_at - started_at, 3),
            "review_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_agent_integration_failure_review_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridAgentIntegrationFailureReviewVisualization]:
    """Write SVG charts for the Stage130 failure-pattern review."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage130_train_cv_key_deltas.svg": render_horizontal_bar_chart_svg(
            title="Stage130 train-CV Stage128 vs Stage116 deltas",
            bars=_split_delta_bars(report, split_key="train_cv_failure_review"),
            x_label="delta",
            width=1480,
            margin_left=760,
        ),
        "stage130_dev_key_deltas.svg": render_horizontal_bar_chart_svg(
            title="Stage130 dev Stage128 vs Stage116 deltas",
            bars=_split_delta_bars(report, split_key="dev_report_only_review"),
            x_label="delta",
            width=1480,
            margin_left=760,
        ),
        "stage130_changed_answer_churn.svg": render_horizontal_bar_chart_svg(
            title="Stage130 changed verified answers",
            bars=_changed_answer_bars(report),
            x_label="changed answer rate",
            width=1320,
            margin_left=620,
        ),
        "stage130_region_displacement.svg": render_horizontal_bar_chart_svg(
            title="Stage130 selected citation region displacement",
            bars=_region_displacement_bars(report),
            x_label="citation-count delta",
            width=1520,
            margin_left=820,
        ),
        "stage130_failure_pattern_scores.svg": render_horizontal_bar_chart_svg(
            title="Stage130 failure pattern scores",
            bars=_failure_pattern_score_bars(report),
            x_label="score",
            width=1560,
            margin_left=860,
        ),
        "stage130_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage130 decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1480,
            margin_left=760,
        ),
        "stage130_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage130 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1760,
            margin_left=960,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridAgentIntegrationFailureReviewVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage129_source_summary(stage129_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage129_report.get("decision") or {}
    train = stage129_report.get("train_cv_validation") or {}
    public_safe = stage129_report.get("public_safe_contract") or {}
    profile_reports = stage129_report.get("profile_reports") or {}
    return {
        "stage": stage129_report.get("stage"),
        "analysis_id": stage129_report.get("analysis_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "selected_profile_id": decision.get("selected_profile_id"),
        "train_cv_validation_passed": decision.get("train_cv_validation_passed"),
        "train_cv_failed_checks": decision.get("train_cv_failed_checks") or [],
        "decision_failed_checks": decision.get("failed_checks") or [],
        "guard_check_count": len(stage129_report.get("guard_checks") or []),
        "guard_check_passed_count": sum(
            1 for check in stage129_report.get("guard_checks") or [] if check.get("passed")
        ),
        "train_cv_selection_mode": train.get("selection_mode"),
        "train_cv_selected_profile_id": train.get("selected_profile_id"),
        "dev_gate_status": decision.get("dev_gate_status"),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "runtime_defaultization_allowed_now": decision.get(
            "runtime_defaultization_allowed_now"
        ),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
        "profile_ids": sorted(profile_reports),
    }


def _split_failure_review(stage129_report: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    profiles = stage129_report["profile_reports"]
    control = profiles[_STAGE116_PROFILE_ID]["split_reports"][split]
    candidate = profiles[_STAGE128_PROFILE_ID]["split_reports"][split]
    deltas = _split_deltas(candidate=candidate, control=control)
    control_regions = control["selected_evidence_summary"]["rank_region_counts"]
    candidate_regions = candidate["selected_evidence_summary"]["rank_region_counts"]
    region_shift = _region_shift(
        control_regions=control_regions,
        candidate_regions=candidate_regions,
    )
    row_count = int(candidate["row_count"])
    changed = int(candidate["changed_verified_answers_vs_stage116_control"])
    return {
        "split": split,
        "row_count": row_count,
        "control_profile_id": _STAGE116_PROFILE_ID,
        "candidate_profile_id": _STAGE128_PROFILE_ID,
        "control_metrics": _public_split_metrics(control),
        "candidate_metrics": _public_split_metrics(candidate),
        "candidate_vs_control_deltas": deltas,
        "changed_verified_answers_vs_control": changed,
        "changed_verified_answer_rate_vs_control": _rounded_ratio(changed, row_count),
        "selected_citation_region_shift": region_shift,
        "direct_metric_findings": _direct_metric_findings(deltas=deltas, split=split),
        "aggregate_inferences": _aggregate_inferences(
            deltas=deltas,
            region_shift=region_shift,
            changed_rate=_rounded_ratio(changed, row_count),
            split=split,
        ),
    }


def _public_split_metrics(split_report: Mapping[str, Any]) -> dict[str, Any]:
    verified = split_report["verified_metrics"]
    retrieval = split_report["retrieval_summary"]
    evidence = split_report["selected_evidence_summary"]
    return {
        "verified_average_token_f1": verified["average_token_f1"],
        "verified_gold_doc_citation_rate": verified["gold_doc_citation_rate"],
        "verified_gold_citation_count": evidence["gold_citation_count"],
        "answerable_refusal_rate": verified["answerable_refusal_rate"],
        "unanswerable_refusal_rate": verified["unanswerable_refusal_rate"],
        "gold_hit_count_at_profile_depth": retrieval["gold_hit_count_at_profile_depth"],
        "gold_hit_rate_at_profile_depth": retrieval["gold_hit_rate_at_profile_depth"],
        "gold_miss_count_at_profile_depth": retrieval["gold_miss_count_at_profile_depth"],
        "selected_citation_count": evidence["citation_count"],
        "answered_count": evidence["answered_count"],
    }


def _split_deltas(
    *,
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_metrics = _public_split_metrics(candidate)
    control_metrics = _public_split_metrics(control)
    return {
        "verified_average_token_f1_delta": _float_delta(
            candidate_metrics["verified_average_token_f1"],
            control_metrics["verified_average_token_f1"],
        ),
        "verified_gold_doc_citation_rate_delta": _float_delta(
            candidate_metrics["verified_gold_doc_citation_rate"],
            control_metrics["verified_gold_doc_citation_rate"],
        ),
        "verified_gold_citation_count_delta": int(
            candidate_metrics["verified_gold_citation_count"]
        )
        - int(control_metrics["verified_gold_citation_count"]),
        "answerable_refusal_rate_delta": _float_delta(
            candidate_metrics["answerable_refusal_rate"],
            control_metrics["answerable_refusal_rate"],
        ),
        "unanswerable_refusal_rate_delta": _float_delta(
            candidate_metrics["unanswerable_refusal_rate"],
            control_metrics["unanswerable_refusal_rate"],
        ),
        "gold_hit_count_at_profile_depth_delta": int(
            candidate_metrics["gold_hit_count_at_profile_depth"]
        )
        - int(control_metrics["gold_hit_count_at_profile_depth"]),
        "gold_hit_rate_at_profile_depth_delta": _float_delta(
            candidate_metrics["gold_hit_rate_at_profile_depth"],
            control_metrics["gold_hit_rate_at_profile_depth"],
        ),
        "selected_citation_count_delta": int(candidate_metrics["selected_citation_count"])
        - int(control_metrics["selected_citation_count"]),
    }


def _region_shift(
    *,
    control_regions: Mapping[str, Any],
    candidate_regions: Mapping[str, Any],
) -> dict[str, Any]:
    region_keys = sorted(set(control_regions) | set(candidate_regions))
    deltas = {
        region: int(candidate_regions.get(region, 0)) - int(control_regions.get(region, 0))
        for region in region_keys
    }
    append_count = int(candidate_regions.get("stage128_append_expansion_201_400", 0))
    control_prefix_like = int(control_regions.get("rank_001_010", 0)) + int(
        control_regions.get("stage116_immutable_prefix_011_200", 0)
    )
    candidate_prefix_like = int(candidate_regions.get("rank_001_010", 0)) + int(
        candidate_regions.get("stage116_immutable_prefix_011_200", 0)
    )
    prefix_like_delta = candidate_prefix_like - control_prefix_like
    return {
        "control_rank_region_counts": dict(sorted(control_regions.items())),
        "candidate_rank_region_counts": dict(sorted(candidate_regions.items())),
        "rank_region_count_deltas": deltas,
        "append_region_selected_citation_count": append_count,
        "prefix_like_selected_citation_delta": prefix_like_delta,
        "append_displacement_balance": append_count + prefix_like_delta,
    }


def _direct_metric_findings(*, deltas: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    return [
        {
            "finding_id": f"{split}_recall_gain",
            "basis": "direct_metric",
            "value": deltas["gold_hit_count_at_profile_depth_delta"],
            "summary": (
                "Stage128 candidate pool increased target-depth gold recall "
                "versus the Stage116 control."
            ),
        },
        {
            "finding_id": f"{split}_gold_citation_delta",
            "basis": "direct_metric",
            "value": deltas["verified_gold_citation_count_delta"],
            "summary": (
                "Stage128 did not preserve gold citation count versus the "
                "Stage116 control."
            ),
        },
        {
            "finding_id": f"{split}_f1_delta",
            "basis": "direct_metric",
            "value": deltas["verified_average_token_f1_delta"],
            "summary": "Stage128 token-F1 movement is small and cannot rescue citation loss.",
        },
    ]


def _aggregate_inferences(
    *,
    deltas: Mapping[str, Any],
    region_shift: Mapping[str, Any],
    changed_rate: float,
    split: str,
) -> list[dict[str, Any]]:
    append_count = int(region_shift["append_region_selected_citation_count"])
    prefix_delta = int(region_shift["prefix_like_selected_citation_delta"])
    return [
        {
            "inference_id": f"{split}_append_displaces_prefix_like_evidence",
            "basis": "aggregate_region_mix_inference",
            "observed": {
                "append_region_selected_citations": append_count,
                "prefix_like_selected_citation_delta": prefix_delta,
                "gold_citation_count_delta": deltas["verified_gold_citation_count_delta"],
            },
            "summary": (
                "The append region is being selected, and aggregate region "
                "counts show it replaces some prefix-like citations without "
                "preserving gold citation count."
            ),
        },
        {
            "inference_id": f"{split}_high_answer_churn",
            "basis": "aggregate_changed_answer_metric",
            "observed": {"changed_verified_answer_rate": changed_rate},
            "summary": (
                "The Stage128 candidate pool changes too many verified answers "
                "for a profile whose gold-citation guard failed."
            ),
        },
    ]


def _failure_patterns(
    *,
    train_review: Mapping[str, Any],
    dev_review: Mapping[str, Any],
) -> list[dict[str, Any]]:
    train_deltas = train_review["candidate_vs_control_deltas"]
    dev_deltas = dev_review["candidate_vs_control_deltas"]
    train_shift = train_review["selected_citation_region_shift"]
    dev_shift = dev_review["selected_citation_region_shift"]
    return [
        {
            "pattern_id": "recall_gain_not_citation_safe",
            "basis": "direct_metric",
            "severity": "blocking",
            "score": _pattern_score(
                abs(int(train_deltas["gold_hit_count_at_profile_depth_delta"]))
                + abs(int(train_deltas["verified_gold_citation_count_delta"])) * 10
            ),
            "evidence": {
                "train_gold_hit_count_delta": train_deltas[
                    "gold_hit_count_at_profile_depth_delta"
                ],
                "train_gold_citation_count_delta": train_deltas[
                    "verified_gold_citation_count_delta"
                ],
                "dev_gold_hit_count_delta": dev_deltas[
                    "gold_hit_count_at_profile_depth_delta"
                ],
                "dev_gold_citation_count_delta": dev_deltas[
                    "verified_gold_citation_count_delta"
                ],
            },
            "interpretation": (
                "The Stage128 candidate pool improves retrieval coverage, but "
                "the answer pipeline does not convert that recall into "
                "gold-citation-safe answers."
            ),
        },
        {
            "pattern_id": "append_region_displaces_prefix_evidence",
            "basis": "aggregate_region_mix_inference",
            "severity": "high",
            "score": _pattern_score(
                int(train_shift["append_region_selected_citation_count"])
                + abs(int(train_shift["prefix_like_selected_citation_delta"]))
            ),
            "evidence": {
                "train_append_selected_citations": train_shift[
                    "append_region_selected_citation_count"
                ],
                "train_prefix_like_selected_citation_delta": train_shift[
                    "prefix_like_selected_citation_delta"
                ],
                "dev_append_selected_citations": dev_shift[
                    "append_region_selected_citation_count"
                ],
                "dev_prefix_like_selected_citation_delta": dev_shift[
                    "prefix_like_selected_citation_delta"
                ],
            },
            "interpretation": (
                "The append region is active in selected citations, but the "
                "aggregate mix suggests it displaces stable prefix evidence "
                "without a citation-safe net gain."
            ),
        },
        {
            "pattern_id": "changed_answer_churn_too_high",
            "basis": "direct_metric",
            "severity": "high",
            "score": _pattern_score(
                train_review["changed_verified_answer_rate_vs_control"] * 100
            ),
            "evidence": {
                "train_changed_verified_answers": train_review[
                    "changed_verified_answers_vs_control"
                ],
                "train_changed_verified_answer_rate": train_review[
                    "changed_verified_answer_rate_vs_control"
                ],
                "dev_changed_verified_answers": dev_review[
                    "changed_verified_answers_vs_control"
                ],
                "dev_changed_verified_answer_rate": dev_review[
                    "changed_verified_answer_rate_vs_control"
                ],
            },
            "interpretation": (
                "The profile changes a large share of verified answers while "
                "failing the gold-citation guard, so it is not a stable "
                "runtime candidate."
            ),
        },
        {
            "pattern_id": "dev_report_confirms_risk_direction",
            "basis": "dev_report_only_metric",
            "severity": "medium",
            "score": _pattern_score(abs(int(dev_deltas["verified_gold_citation_count_delta"]))),
            "evidence": {
                "dev_gold_citation_count_delta": dev_deltas[
                    "verified_gold_citation_count_delta"
                ],
                "dev_verified_f1_delta": dev_deltas["verified_average_token_f1_delta"],
                "dev_gold_hit_count_delta": dev_deltas[
                    "gold_hit_count_at_profile_depth_delta"
                ],
            },
            "interpretation": (
                "Dev remains report-only, but its aggregate direction is "
                "consistent with the train-CV citation-risk signal."
            ),
        },
    ]


def _action_boundary(failure_patterns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blocking_patterns = [
        pattern["pattern_id"]
        for pattern in failure_patterns
        if pattern.get("severity") == "blocking"
    ]
    return {
        "stage128_runtime_defaultization_allowed_now": False,
        "stage128_final_test_gate_allowed_now": False,
        "stage128_direct_agent_integration_path_blocked": bool(blocking_patterns),
        "blocked_by_patterns": blocking_patterns,
        "test_remains_locked": True,
        "runtime_default_policy": "unchanged",
        "fallback_strategies_enabled": False,
        "next_candidate_direction": _NEXT_DIRECTION,
        "next_direction_scope": (
            "Freeze a train/dev-only protocol for a citation-preserving append "
            "candidate evidence shortlist redesign. The next protocol should "
            "keep Stage116 evidence stable and evaluate append candidates as "
            "supplemental evidence, not unrestricted replacements."
        ),
    }


def _guard_checks(
    *,
    report: Mapping[str, Any],
    source_summary: Mapping[str, Any],
    train_review: Mapping[str, Any],
    failure_patterns: Sequence[Mapping[str, Any]],
    user_confirmed_review: bool,
    confirmation_note: str,
) -> list[dict[str, Any]]:
    return [
        _check(
            name="user_confirmed_stage130_review",
            passed=user_confirmed_review and "Stage130" in confirmation_note,
            observed=confirmation_note,
            expected="user confirmed Stage130 review",
        ),
        _check(
            name="stage129_source_status_is_blocked_or_failed",
            passed=source_summary.get("decision_status") == _SOURCE_STATUS,
            observed=source_summary.get("decision_status"),
            expected=_SOURCE_STATUS,
        ),
        _check(
            name="stage129_analysis_id_matches",
            passed=source_summary.get("analysis_id") == _SOURCE_ANALYSIS_ID,
            observed=source_summary.get("analysis_id"),
            expected=_SOURCE_ANALYSIS_ID,
        ),
        _check(
            name="stage129_recommends_failure_review",
            passed=source_summary.get("recommended_next_direction") == _SOURCE_NEXT,
            observed=source_summary.get("recommended_next_direction"),
            expected=_SOURCE_NEXT,
        ),
        _check(
            name="stage129_train_cv_failed_gold_citation_guard",
            passed="gold_citation_count_delta_vs_stage116_non_negative"
            in source_summary.get("train_cv_failed_checks", []),
            observed=source_summary.get("train_cv_failed_checks"),
            expected="gold_citation_count_delta_vs_stage116_non_negative",
        ),
        _check(
            name="stage130_review_uses_public_safe_aggregate_only",
            passed=True,
            observed="saved Stage129 public-safe aggregate report only",
            expected="no raw split, corpus, candidate, question, answer, document, or test data",
        ),
        _check(
            name="stage130_direct_metrics_capture_recall_gain_and_citation_loss",
            passed=int(
                train_review["candidate_vs_control_deltas"][
                    "gold_hit_count_at_profile_depth_delta"
                ]
            )
            > 0
            and int(
                train_review["candidate_vs_control_deltas"][
                    "verified_gold_citation_count_delta"
                ]
            )
            < 0,
            observed=train_review["candidate_vs_control_deltas"],
            expected="positive recall delta and negative gold citation count delta",
        ),
        _check(
            name="stage130_failure_patterns_include_blocking_pattern",
            passed=any(pattern.get("severity") == "blocking" for pattern in failure_patterns),
            observed=[pattern["pattern_id"] for pattern in failure_patterns],
            expected="at least one blocking failure pattern",
        ),
        _check(
            name="stage130_test_locked",
            passed=source_summary.get("can_run_final_test_metrics_now") is False
            and source_summary.get("can_use_test_for_tuning") is False,
            observed={
                "can_run_final_test_metrics_now": source_summary.get(
                    "can_run_final_test_metrics_now"
                ),
                "can_use_test_for_tuning": source_summary.get("can_use_test_for_tuning"),
            },
            expected="test locked",
        ),
        _check(
            name="stage130_runtime_defaults_unchanged",
            passed=source_summary.get("default_runtime_policy") == "unchanged",
            observed=source_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage130_no_fallback_strategies",
            passed=source_summary.get("fallback_strategies_enabled") is False,
            observed=source_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="stage130_public_outputs_have_no_forbidden_keys",
            passed=not _contains_forbidden_key(report),
            observed=sorted(_forbidden_keys_found(report)),
            expected=[],
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    action_boundary: Mapping[str, Any],
) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    base = {
        "review_id": _REVIEW_ID,
        "stage128_runtime_defaultization_allowed_now": False,
        "stage128_final_test_gate_allowed_now": False,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "runtime_defaultization_allowed_now": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_direction": action_boundary.get("next_candidate_direction"),
    }
    if failed_checks:
        return {
            **base,
            "status": "primeqa_hybrid_stage129_agent_integration_failure_review_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
        }
    return {
        **base,
        "status": "primeqa_hybrid_stage129_agent_integration_failure_review_completed",
        "failed_checks": [],
        "can_continue_train_dev_development": True,
        "stage128_direct_agent_integration_path_blocked": action_boundary.get(
            "stage128_direct_agent_integration_path_blocked"
        ),
    }


def _check(*, name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


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


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _contains_forbidden_key(value: Any) -> bool:
    return bool(_forbidden_keys_found(value))


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_delta(value: Any, baseline: Any) -> float:
    return round(float(value) - float(baseline), 4)


def _rounded_ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _pattern_score(value: int | float) -> float:
    return round(float(value), 4)


def _split_delta_bars(report: Mapping[str, Any], *, split_key: str) -> list[BarDatum]:
    deltas = report[split_key]["candidate_vs_control_deltas"]
    return [
        BarDatum(
            label="verified F1",
            value=float(deltas["verified_average_token_f1_delta"]),
            value_label=f"{float(deltas['verified_average_token_f1_delta']):+.4f}",
        ),
        BarDatum(
            label="gold citation count",
            value=float(deltas["verified_gold_citation_count_delta"]),
            value_label=f"{int(deltas['verified_gold_citation_count_delta']):+d}",
        ),
        BarDatum(
            label="target-depth gold hit count",
            value=float(deltas["gold_hit_count_at_profile_depth_delta"]),
            value_label=f"{int(deltas['gold_hit_count_at_profile_depth_delta']):+d}",
        ),
        BarDatum(
            label="answerable refusal rate",
            value=float(deltas["answerable_refusal_rate_delta"]),
            value_label=f"{float(deltas['answerable_refusal_rate_delta']):+.4f}",
        ),
    ]


def _changed_answer_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label="train_cv",
            value=float(
                report["train_cv_failure_review"][
                    "changed_verified_answer_rate_vs_control"
                ]
            ),
            value_label=(
                f"{report['train_cv_failure_review']['changed_verified_answers_vs_control']}"
                f" / {report['train_cv_failure_review']['row_count']}"
            ),
        ),
        BarDatum(
            label="dev_report_only",
            value=float(
                report["dev_report_only_review"][
                    "changed_verified_answer_rate_vs_control"
                ]
            ),
            value_label=(
                f"{report['dev_report_only_review']['changed_verified_answers_vs_control']}"
                f" / {report['dev_report_only_review']['row_count']}"
            ),
        ),
    ]


def _region_displacement_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for split_key, label_prefix in (
        ("train_cv_failure_review", "train_cv"),
        ("dev_report_only_review", "dev"),
    ):
        deltas = report[split_key]["selected_citation_region_shift"][
            "rank_region_count_deltas"
        ]
        for region, delta in sorted(deltas.items()):
            bars.append(
                BarDatum(
                    label=f"{label_prefix}:{region}",
                    value=float(delta),
                    value_label=f"{int(delta):+d}",
                )
            )
    return bars


def _failure_pattern_score_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(pattern["pattern_id"]),
            value=float(pattern["score"]),
            value_label=f"{float(pattern['score']):.4f}",
        )
        for pattern in report.get("failure_patterns", [])
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    action = report.get("action_boundary") or {}
    flags = {
        "failure_review_completed": (
            decision.get("status")
            == "primeqa_hybrid_stage129_agent_integration_failure_review_completed"
        ),
        "stage128_direct_path_blocked": action.get(
            "stage128_direct_agent_integration_path_blocked"
        )
        is True,
        "final_test_gate_allowed": decision.get("can_open_final_test_gate_now") is True,
        "runtime_defaultization_allowed": (
            decision.get("runtime_defaultization_allowed_now") is True
        ),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled") is True,
    }
    return [
        BarDatum(
            label=label,
            value=1.0 if value else 0.0,
            value_label=str(bool(value)).lower(),
        )
        for label, value in flags.items()
    ]


def _guard_check_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(check["name"]),
            value=1.0 if check.get("passed") else 0.0,
            value_label="passed" if check.get("passed") else "failed",
        )
        for check in report.get("guard_checks", [])
    ]
