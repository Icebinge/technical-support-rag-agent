from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 99"
_CREATED_AT = "2026-07-15"
_STOPPED_CANDIDATE_ID = "selective_dense_sparse_low_overlap_gate_design"
_STOPPED_PROTOCOL_ID = "selective_dense_sparse_low_overlap_gate_train_dev_v1"
_STOP_DECISION_ROUTE_ID = "selective_dense_sparse_stop_decision"
_BLOCKED_CANDIDATE_ID = "source_doc_ids_oracle_union_blocked"
_PRIOR_STOPPED_CANDIDATE_IDS = frozenset(
    {
        "lexical_cluster_diversity_rerank_design",
        "structured_query_keyphrase_compaction_design",
        "section_signal_guarded_expansion_design",
        "score_margin_bm25_normalization_gate_design",
    }
)


@dataclass(frozen=True)
class PrimeQAHybridSelectiveDenseSparseStopVisualization:
    """One generated Stage99 selective dense+sparse stop-decision chart."""

    name: str
    path: str


def decide_primeqa_hybrid_selective_dense_sparse_stop(
    *,
    stage84_report_path: Path,
    stage97_report_path: Path,
    stage98_report_path: Path,
    user_confirmed_stop: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Stop selective dense+sparse after the frozen train/dev contract fails."""

    started_at = time.perf_counter()
    stage84_report = _load_json_object(stage84_report_path)
    stage97_report = _load_json_object(stage97_report_path)
    stage98_report = _load_json_object(stage98_report_path)
    loaded_at = time.perf_counter()

    stage84_candidate = _candidate_summary(stage84_report, _STOPPED_CANDIDATE_ID)
    stage97_summary = _stage97_summary(stage97_report)
    stage98_summary = _stage98_summary(stage98_report)
    route_status = _second_wave_route_status(stage84_report)
    guard_checks = _guard_checks(
        stage84_report=stage84_report,
        stage97_report=stage97_report,
        stage98_report=stage98_report,
        stage84_candidate=stage84_candidate,
        stage97_summary=stage97_summary,
        stage98_summary=stage98_summary,
        route_status=route_status,
        user_confirmed_stop=user_confirmed_stop,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "decision_scope": (
            "Train/dev-only stop decision for the Stage84 selective "
            "dense+sparse low-overlap route after Stage98 validation. This "
            "stage reads public-safe Stage84, Stage97, and Stage98 reports, "
            "does not load train/dev/test splits, does not run new retrieval "
            "metrics, does not run final metrics, does not use source DOC_IDS "
            "as runtime retrieval evidence, does not tune dev thresholds, and "
            "does not change runtime defaults."
        ),
        "user_confirmation": {
            "route_id": _STOP_DECISION_ROUTE_ID,
            "confirmed": bool(user_confirmed_stop),
            "confirmation_note": confirmation_note,
        },
        "source_files": {
            "stage84_report": _fingerprint(stage84_report_path),
            "stage97_report": _fingerprint(stage97_report_path),
            "stage98_report": _fingerprint(stage98_report_path),
        },
        "stopped_route": {
            "candidate_id": _STOPPED_CANDIDATE_ID,
            "protocol_id": _STOPPED_PROTOCOL_ID,
            "stage84_candidate_summary": stage84_candidate,
            "stage97_summary": stage97_summary,
            "stage98_summary": stage98_summary,
            "stop_reason": (
                "The train-selected selective dense+sparse policy had no dev "
                "hit@10 gain, no dev hit@1 gain, no dev not-found@50 reduction, "
                "no dev top10 improvements, no dev gate activations, and no "
                "dev promotions. Stage97 required a train-selected dev hit@10 "
                "gain and a dev not-found@50 decrease without hit@1 collapse, "
                "so the route is non-advancing."
            ),
        },
        "second_wave_route_status": route_status,
        "guard_checks": guard_checks,
        "decision": _decision(
            guard_checks=guard_checks,
            route_status=route_status,
        ),
        "timing_seconds": {
            "load_reports": round(loaded_at - started_at, 3),
            "decision_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_selective_dense_sparse_stop_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridSelectiveDenseSparseStopVisualization]:
    """Write SVG charts for Stage99 selective dense+sparse stop decision."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage99_selective_dense_sparse_train_dev_hit10_delta.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage99 selective dense+sparse train vs dev hit@10 delta",
                bars=_train_dev_hit10_delta_bars(report),
                x_label="hit@10 delta",
                width=1120,
                margin_left=380,
            )
        ),
        "stage99_selective_dense_sparse_dev_contract_deltas.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage99 selective dense+sparse dev contract deltas",
                bars=_dev_contract_delta_bars(report),
                x_label="delta or count",
                width=1220,
                margin_left=520,
            )
        ),
        "stage99_selective_dense_sparse_gate_actions.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage99 selective dense+sparse selected-policy gate actions",
                bars=_gate_action_bars(report),
                x_label="action count",
                width=1240,
                margin_left=540,
            )
        ),
        "stage99_second_wave_route_status.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage99 second-wave route status",
                bars=_route_status_bars(report),
                x_label="route count",
                width=1160,
                margin_left=500,
            )
        ),
        "stage99_selective_dense_sparse_stop_decision_flags.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage99 selective dense+sparse stop decision flags",
                bars=_decision_flag_bars(report),
                x_label="1 means true",
                width=1180,
                margin_left=520,
            )
        ),
        "stage99_selective_dense_sparse_stop_guard_check_status.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage99 selective dense+sparse stop guard checks",
                bars=_guard_check_bars(report),
                x_label="1 means passed",
                width=1400,
                margin_left=700,
            )
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridSelectiveDenseSparseStopVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage97_summary(stage97_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage97_report.get("decision") or {}
    frozen_protocol = stage97_report.get("frozen_protocol") or {}
    dense_contract = frozen_protocol.get("dense_cache_contract") or {}
    return {
        "status": decision.get("status"),
        "protocol_id": decision.get("protocol_id") or frozen_protocol.get("protocol_id"),
        "candidate_id": decision.get("candidate_id") or frozen_protocol.get("candidate_id"),
        "requires_user_confirmation_before_train_dev_run": decision.get(
            "requires_user_confirmation_before_train_dev_run"
        ),
        "can_run_train_dev_metrics_after_user_confirmation": decision.get(
            "can_run_train_dev_metrics_after_user_confirmation"
        ),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "policy_count": len(frozen_protocol.get("candidate_policy_grid") or []),
        "dense_config_count": len(dense_contract.get("allowed_dense_configs") or []),
        "download_required": dense_contract.get("download_required"),
        "document_reencoding_allowed": dense_contract.get(
            "document_reencoding_allowed"
        ),
        "train_selection_rule": frozen_protocol.get("train_selection_rule"),
        "target_metric_contract": frozen_protocol.get("target_metric_contract") or [],
    }


def _stage98_summary(stage98_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage98_report.get("decision") or {}
    train_selection = stage98_report.get("train_selection") or {}
    selected_train_comparison = (
        train_selection.get("selected_train_comparison_to_baseline") or {}
    )
    selected_policy_id = decision.get("selected_policy_id")
    selected_dev_comparison = (
        (stage98_report.get("comparisons_to_baseline") or {})
        .get("dev", {})
        .get(str(selected_policy_id), {})
    )
    artifact_safety = stage98_report.get("artifact_safety") or {}
    loaded_data_summary = stage98_report.get("loaded_data_summary") or {}
    return {
        "status": decision.get("status"),
        "protocol_id": decision.get("protocol_id"),
        "selected_policy_id": selected_policy_id,
        "train_hit10_delta": _float_metric(selected_train_comparison, "hit@10_delta"),
        "train_not_found_count_at_search_depth_delta": int(
            selected_train_comparison.get("not_found_count_at_search_depth_delta") or 0
        ),
        "train_gate_activation_count": int(
            selected_train_comparison.get("gate_activation_count") or 0
        ),
        "train_promotion_count": int(
            selected_train_comparison.get("promotion_count") or 0
        ),
        "dev_hit1_delta": _float_metric(selected_dev_comparison, "hit@1_delta"),
        "dev_hit10_delta": _float_metric(selected_dev_comparison, "hit@10_delta"),
        "dev_top10_improvement_count": int(
            selected_dev_comparison.get("top10_improvement_count") or 0
        ),
        "dev_top10_regression_count": int(
            selected_dev_comparison.get("top10_regression_count") or 0
        ),
        "dev_not_found_count_at_search_depth_delta": int(
            selected_dev_comparison.get("not_found_count_at_search_depth_delta") or 0
        ),
        "dev_gate_activation_count": int(
            selected_dev_comparison.get("gate_activation_count") or 0
        ),
        "dev_promotion_count": int(selected_dev_comparison.get("promotion_count") or 0),
        "primary_contract_passed": decision.get("primary_contract_passed"),
        "secondary_contract_passed": decision.get("secondary_contract_passed"),
        "guard_contract_passed": decision.get("guard_contract_passed"),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "test_split_loaded": loaded_data_summary.get("test_split_loaded"),
        "final_metrics_run": loaded_data_summary.get("final_metrics_run"),
        "artifact_safety": {
            key: artifact_safety.get(key)
            for key in (
                "raw_question_text_written",
                "raw_answer_text_written",
                "raw_document_text_written",
                "raw_document_title_written",
                "query_terms_written",
                "matched_token_strings_written",
                "source_doc_ids_used_as_runtime_evidence",
                "answer_doc_ids_used_as_runtime_features",
            )
        },
    }


def _second_wave_route_status(stage84_report: Mapping[str, Any]) -> dict[str, Any]:
    candidate_map = {
        str(candidate.get("candidate_id")): candidate
        for candidate in stage84_report.get("candidate_designs") or []
        if isinstance(candidate, Mapping)
    }
    original_order = [
        str(value)
        for value in stage84_report.get("recommended_execution_order") or []
    ]
    stopped_ids = set(_PRIOR_STOPPED_CANDIDATE_IDS) | {_STOPPED_CANDIDATE_ID}
    remaining_execution_order = [
        candidate_id
        for candidate_id in original_order
        if candidate_id not in stopped_ids
        and candidate_id != _BLOCKED_CANDIDATE_ID
        and (candidate_map.get(candidate_id) or {}).get("status")
        == "recommended_for_train_dev_protocol_design"
    ]
    blocked_candidate = _public_candidate(candidate_map.get(_BLOCKED_CANDIDATE_ID))
    return {
        "original_execution_order": original_order,
        "prior_stopped_candidate_ids": sorted(_PRIOR_STOPPED_CANDIDATE_IDS),
        "stopped_candidate_id": _STOPPED_CANDIDATE_ID,
        "all_stopped_candidate_ids": sorted(stopped_ids),
        "remaining_execution_order": remaining_execution_order,
        "remaining_candidate_summaries": [
            _public_candidate(candidate_map.get(candidate_id))
            for candidate_id in remaining_execution_order
        ],
        "remaining_actionable_candidate_count": len(remaining_execution_order),
        "next_candidate_id": remaining_execution_order[0]
        if remaining_execution_order
        else None,
        "blocked_candidate_id": _BLOCKED_CANDIDATE_ID,
        "blocked_candidate_summary": blocked_candidate,
        "route_family_exhausted": not remaining_execution_order,
    }


def _candidate_summary(
    stage84_report: Mapping[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    for candidate in stage84_report.get("candidate_designs") or []:
        if isinstance(candidate, Mapping) and candidate.get("candidate_id") == candidate_id:
            return _public_candidate(candidate)
    return {}


def _public_candidate(candidate: Mapping[str, Any] | None) -> dict[str, Any]:
    if not candidate:
        return {}
    return {
        "candidate_id": candidate.get("candidate_id"),
        "name": candidate.get("name"),
        "category": candidate.get("category"),
        "status": candidate.get("status"),
        "risk_level": candidate.get("risk_level"),
        "implementation_readiness": candidate.get("implementation_readiness"),
        "priority_score": candidate.get("priority_score"),
        "target_miss_count": candidate.get("target_miss_count"),
        "target_miss_count_by_split": candidate.get("target_miss_count_by_split"),
        "target_metric_contract": candidate.get("target_metric_contract"),
        "runtime_evidence_policy": candidate.get("runtime_evidence_policy"),
    }


def _guard_checks(
    *,
    stage84_report: Mapping[str, Any],
    stage97_report: Mapping[str, Any],
    stage98_report: Mapping[str, Any],
    stage84_candidate: Mapping[str, Any],
    stage97_summary: Mapping[str, Any],
    stage98_summary: Mapping[str, Any],
    route_status: Mapping[str, Any],
    user_confirmed_stop: bool,
) -> list[dict[str, Any]]:
    artifact_safety = stage98_summary.get("artifact_safety") or {}
    return [
        _check(
            name="source_stage84_report_is_stage84",
            passed=stage84_report.get("stage") == "Stage 84",
            observed=stage84_report.get("stage"),
            expected="Stage 84",
        ),
        _check(
            name="source_stage97_report_is_stage97",
            passed=stage97_report.get("stage") == "Stage 97",
            observed=stage97_report.get("stage"),
            expected="Stage 97",
        ),
        _check(
            name="source_stage98_report_is_stage98",
            passed=stage98_report.get("stage") == "Stage 98",
            observed=stage98_report.get("stage"),
            expected="Stage 98",
        ),
        _check(
            name="user_confirmed_stage99_stop_decision",
            passed=user_confirmed_stop,
            observed=user_confirmed_stop,
            expected=True,
        ),
        _check(
            name="stage97_protocol_frozen",
            passed=stage97_summary.get("status")
            == "primeqa_hybrid_selective_dense_sparse_protocol_frozen",
            observed=stage97_summary.get("status"),
            expected="primeqa_hybrid_selective_dense_sparse_protocol_frozen",
        ),
        _check(
            name="stage97_candidate_matches_selective_dense_sparse",
            passed=stage97_summary.get("candidate_id") == _STOPPED_CANDIDATE_ID,
            observed=stage97_summary.get("candidate_id"),
            expected=_STOPPED_CANDIDATE_ID,
        ),
        _check(
            name="stage97_protocol_matches_selective_dense_sparse",
            passed=stage97_summary.get("protocol_id") == _STOPPED_PROTOCOL_ID,
            observed=stage97_summary.get("protocol_id"),
            expected=_STOPPED_PROTOCOL_ID,
        ),
        _check(
            name="stage97_downloads_forbidden",
            passed=stage97_summary.get("download_required") is False,
            observed=stage97_summary.get("download_required"),
            expected=False,
        ),
        _check(
            name="stage97_train_selection_rule_forbids_dev_selection",
            passed=_stage97_rule_forbids_dev_selection(stage97_summary),
            observed=stage97_summary.get("train_selection_rule"),
            expected="dev is validation only and dev selection is forbidden",
        ),
        _check(
            name="stage98_comparison_completed",
            passed=stage98_summary.get("status")
            == "primeqa_hybrid_selective_dense_sparse_comparison_completed",
            observed=stage98_summary.get("status"),
            expected="primeqa_hybrid_selective_dense_sparse_comparison_completed",
        ),
        _check(
            name="stage98_protocol_matches_stage97",
            passed=stage98_summary.get("protocol_id") == _STOPPED_PROTOCOL_ID,
            observed=stage98_summary.get("protocol_id"),
            expected=_STOPPED_PROTOCOL_ID,
        ),
        _check(
            name="stage84_candidate_metric_contract_requires_train_selected_dev_hit10_gain",
            passed=_requires_train_selected_dev_hit10_gain(stage84_candidate),
            observed=stage84_candidate.get("target_metric_contract"),
            expected="primary metric requires train-selected dev hit@10 improvement",
        ),
        _check(
            name="stage84_candidate_metric_contract_requires_not_found_decrease",
            passed=_requires_not_found_decrease(stage84_candidate),
            observed=stage84_candidate.get("target_metric_contract"),
            expected="secondary metric requires dev not-found@50 decrease",
        ),
        _check(
            name="stage84_candidate_guard_blocks_downloads_and_dev_thresholds",
            passed=_blocks_downloads_and_dev_thresholds(stage84_candidate),
            observed=stage84_candidate.get("target_metric_contract"),
            expected="guard blocks downloads and dev-selected gate thresholds",
        ),
        _check(
            name="stage98_primary_contract_failed",
            passed=stage98_summary.get("primary_contract_passed") is False,
            observed=stage98_summary.get("primary_contract_passed"),
            expected=False,
        ),
        _check(
            name="stage98_secondary_contract_failed",
            passed=stage98_summary.get("secondary_contract_passed") is False,
            observed=stage98_summary.get("secondary_contract_passed"),
            expected=False,
        ),
        _check(
            name="stage98_guard_contract_passed",
            passed=stage98_summary.get("guard_contract_passed") is True,
            observed=stage98_summary.get("guard_contract_passed"),
            expected=True,
        ),
        _check(
            name="stage98_train_selected_policy_has_no_dev_hit10_gain",
            passed=float(stage98_summary.get("dev_hit10_delta") or 0.0) <= 0.0,
            observed=stage98_summary.get("dev_hit10_delta"),
            expected="<= 0.0",
        ),
        _check(
            name="stage98_dev_not_found_not_reduced",
            passed=int(stage98_summary.get("dev_not_found_count_at_search_depth_delta") or 0)
            >= 0,
            observed=stage98_summary.get("dev_not_found_count_at_search_depth_delta"),
            expected=">= 0",
        ),
        _check(
            name="stage98_dev_hit1_not_improved",
            passed=float(stage98_summary.get("dev_hit1_delta") or 0.0) <= 0.0,
            observed=stage98_summary.get("dev_hit1_delta"),
            expected="<= 0.0",
        ),
        _check(
            name="stage98_dev_top10_net_not_positive",
            passed=(
                int(stage98_summary.get("dev_top10_improvement_count") or 0)
                - int(stage98_summary.get("dev_top10_regression_count") or 0)
            )
            <= 0,
            observed={
                "improvements": stage98_summary.get("dev_top10_improvement_count"),
                "regressions": stage98_summary.get("dev_top10_regression_count"),
            },
            expected="net <= 0",
        ),
        _check(
            name="stage98_selected_policy_has_no_dev_gate_activation",
            passed=int(stage98_summary.get("dev_gate_activation_count") or 0) == 0,
            observed=stage98_summary.get("dev_gate_activation_count"),
            expected=0,
        ),
        _check(
            name="stage98_selected_policy_has_no_dev_promotions",
            passed=int(stage98_summary.get("dev_promotion_count") or 0) == 0,
            observed=stage98_summary.get("dev_promotion_count"),
            expected=0,
        ),
        _check(
            name="stage98_test_split_not_loaded",
            passed=stage98_summary.get("test_split_loaded") is False,
            observed=stage98_summary.get("test_split_loaded"),
            expected=False,
        ),
        _check(
            name="stage98_final_metrics_not_run",
            passed=stage98_summary.get("final_metrics_run") is False
            and stage98_summary.get("can_run_final_test_metrics_now") is False,
            observed={
                "final_metrics_run": stage98_summary.get("final_metrics_run"),
                "can_run_final_test_metrics_now": stage98_summary.get(
                    "can_run_final_test_metrics_now"
                ),
            },
            expected="final metrics locked",
        ),
        _check(
            name="stage98_final_test_gate_closed",
            passed=stage98_summary.get("can_open_final_test_gate_now") is False,
            observed=stage98_summary.get("can_open_final_test_gate_now"),
            expected=False,
        ),
        _check(
            name="stage98_forbids_test_tuning",
            passed=stage98_summary.get("can_use_test_for_tuning") is False,
            observed=stage98_summary.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage98_default_runtime_policy_unchanged",
            passed=stage98_summary.get("default_runtime_policy") == "unchanged",
            observed=stage98_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage98_artifact_safety_flags_false",
            passed=bool(artifact_safety)
            and all(value is False for value in artifact_safety.values()),
            observed=artifact_safety,
            expected="all false",
        ),
        _check(
            name="stage84_execution_order_contains_stopped_candidate",
            passed=_STOPPED_CANDIDATE_ID
            in route_status.get("original_execution_order", []),
            observed=route_status.get("original_execution_order"),
            expected=f"contains {_STOPPED_CANDIDATE_ID}",
        ),
        _check(
            name="prior_second_wave_routes_removed_from_remaining_queue",
            passed=all(
                candidate_id not in route_status.get("remaining_execution_order", [])
                for candidate_id in _PRIOR_STOPPED_CANDIDATE_IDS
            ),
            observed=route_status.get("remaining_execution_order"),
            expected="all prior stopped route ids absent",
        ),
        _check(
            name="selective_dense_sparse_removed_from_remaining_queue",
            passed=_STOPPED_CANDIDATE_ID
            not in route_status.get("remaining_execution_order", []),
            observed=route_status.get("remaining_execution_order"),
            expected=f"{_STOPPED_CANDIDATE_ID} absent",
        ),
        _check(
            name="source_doc_ids_candidate_is_blocked_not_actionable",
            passed=(
                route_status.get("blocked_candidate_summary", {}).get("status")
                == "blocked_from_train_dev_experiment"
            ),
            observed=route_status.get("blocked_candidate_summary", {}).get("status"),
            expected="blocked_from_train_dev_experiment",
        ),
        _check(
            name="no_remaining_second_wave_actionable_candidates",
            passed=int(route_status.get("remaining_actionable_candidate_count") or 0) == 0,
            observed=route_status.get("remaining_actionable_candidate_count"),
            expected=0,
        ),
        _check(
            name="stage99_no_new_retrieval_metrics_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage99_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage99_default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    route_status: Mapping[str, Any],
) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_selective_dense_sparse_stop_decision_blocked",
            "failed_checks": failed_checks,
            "stopped_candidate_id": None,
            "stopped_protocol_id": None,
            "route_family_exhausted": False,
            "can_continue_train_dev_development": False,
            "requires_user_confirmation_before_next_experiment": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        }
    route_family_exhausted = bool(route_status.get("route_family_exhausted"))
    return {
        "status": "primeqa_hybrid_selective_dense_sparse_route_stopped",
        "stopped_candidate_id": _STOPPED_CANDIDATE_ID,
        "stopped_protocol_id": _STOPPED_PROTOCOL_ID,
        "current_route_defaultization": "blocked",
        "next_candidate_id": route_status.get("next_candidate_id"),
        "remaining_actionable_candidate_count": int(
            route_status.get("remaining_actionable_candidate_count") or 0
        ),
        "route_family_exhausted": route_family_exhausted,
        "can_continue_train_dev_development": not route_family_exhausted,
        "requires_user_confirmation_before_next_experiment": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage 100: summarize second-wave retrieval route exhaustion and "
            "decide the next research direction from existing train/dev evidence; "
            "keep test locked, do not run final metrics, and keep runtime defaults "
            "unchanged."
        ),
    }


def _train_dev_hit10_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = (report.get("stopped_route") or {}).get("stage98_summary") or {}
    return [
        BarDatum(
            label="train hit@10 delta",
            value=float(summary.get("train_hit10_delta") or 0.0),
            value_label=f"{float(summary.get('train_hit10_delta') or 0.0):+.4f}",
        ),
        BarDatum(
            label="dev hit@10 delta",
            value=float(summary.get("dev_hit10_delta") or 0.0),
            value_label=f"{float(summary.get('dev_hit10_delta') or 0.0):+.4f}",
        ),
    ]


def _dev_contract_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = (report.get("stopped_route") or {}).get("stage98_summary") or {}
    return [
        BarDatum(
            label="dev hit@10 delta",
            value=float(summary.get("dev_hit10_delta") or 0.0),
            value_label=f"{float(summary.get('dev_hit10_delta') or 0.0):+.4f}",
        ),
        BarDatum(
            label="dev hit@1 delta",
            value=float(summary.get("dev_hit1_delta") or 0.0),
            value_label=f"{float(summary.get('dev_hit1_delta') or 0.0):+.4f}",
        ),
        BarDatum(
            label="dev not-found@50 delta",
            value=float(summary.get("dev_not_found_count_at_search_depth_delta") or 0),
            value_label=str(summary.get("dev_not_found_count_at_search_depth_delta") or 0),
        ),
        BarDatum(
            label="dev top10 improvements",
            value=float(summary.get("dev_top10_improvement_count") or 0),
            value_label=str(summary.get("dev_top10_improvement_count") or 0),
        ),
        BarDatum(
            label="dev top10 regressions",
            value=float(summary.get("dev_top10_regression_count") or 0),
            value_label=str(summary.get("dev_top10_regression_count") or 0),
        ),
    ]


def _gate_action_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = (report.get("stopped_route") or {}).get("stage98_summary") or {}
    return [
        BarDatum(
            label="train gate activations",
            value=float(summary.get("train_gate_activation_count") or 0),
            value_label=str(summary.get("train_gate_activation_count") or 0),
        ),
        BarDatum(
            label="train promotions",
            value=float(summary.get("train_promotion_count") or 0),
            value_label=str(summary.get("train_promotion_count") or 0),
        ),
        BarDatum(
            label="dev gate activations",
            value=float(summary.get("dev_gate_activation_count") or 0),
            value_label=str(summary.get("dev_gate_activation_count") or 0),
        ),
        BarDatum(
            label="dev promotions",
            value=float(summary.get("dev_promotion_count") or 0),
            value_label=str(summary.get("dev_promotion_count") or 0),
        ),
    ]


def _route_status_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    route_status = report.get("second_wave_route_status") or {}
    return [
        BarDatum(
            label="prior stopped routes",
            value=float(len(route_status.get("prior_stopped_candidate_ids") or [])),
            value_label=str(len(route_status.get("prior_stopped_candidate_ids") or [])),
        ),
        BarDatum(
            label="current stopped route",
            value=1.0,
            value_label="1",
        ),
        BarDatum(
            label="remaining actionable routes",
            value=float(route_status.get("remaining_actionable_candidate_count") or 0),
            value_label=str(route_status.get("remaining_actionable_candidate_count") or 0),
        ),
        BarDatum(
            label="blocked source DOC_IDS route",
            value=1.0 if route_status.get("blocked_candidate_summary") else 0.0,
            value_label="blocked" if route_status.get("blocked_candidate_summary") else "0",
        ),
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    names = [
        "route_family_exhausted",
        "can_continue_train_dev_development",
        "can_open_final_test_gate_now",
        "can_run_final_test_metrics_now",
        "can_use_test_for_tuning",
    ]
    return [
        BarDatum(
            label=name,
            value=1.0 if decision.get(name) is True else 0.0,
            value_label=str(decision.get(name)).lower(),
        )
        for name in names
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


def _stage97_rule_forbids_dev_selection(stage97_summary: Mapping[str, Any]) -> bool:
    rule = stage97_summary.get("train_selection_rule") or {}
    return bool(rule.get("dev_selection_forbidden")) and str(
        rule.get("validation_split")
    ) == "dev"


def _requires_train_selected_dev_hit10_gain(candidate: Mapping[str, Any]) -> bool:
    contract = candidate.get("target_metric_contract") or []
    return any(
        "train-selected" in str(item)
        and "dev hit@10" in str(item)
        and "improve" in str(item)
        for item in contract
    )


def _requires_not_found_decrease(candidate: Mapping[str, Any]) -> bool:
    contract = candidate.get("target_metric_contract") or []
    return any(
        "not-found@50" in str(item) and "decrease" in str(item)
        for item in contract
    )


def _blocks_downloads_and_dev_thresholds(candidate: Mapping[str, Any]) -> bool:
    contract = candidate.get("target_metric_contract") or []
    return any(
        "no downloads" in str(item) and "dev-selected gate thresholds" in str(item)
        for item in contract
    )


def _float_metric(mapping: Mapping[str, Any], key: str) -> float:
    return float(mapping.get(key) or 0.0)


def _load_json_object(path: Path) -> dict[str, Any]:
    _ensure_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    _ensure_file(path)
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")


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
