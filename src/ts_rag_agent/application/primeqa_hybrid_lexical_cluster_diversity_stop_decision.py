from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 87"
_CREATED_AT = "2026-07-15"
_STOPPED_CANDIDATE_ID = "lexical_cluster_diversity_rerank_design"
_STOPPED_PROTOCOL_ID = "lexical_cluster_diversity_rerank_train_dev_v1"
_BLOCKED_CANDIDATE_ID = "source_doc_ids_oracle_union_blocked"
_STOP_DECISION_ROUTE_ID = "lexical_cluster_diversity_stop_decision"


@dataclass(frozen=True)
class PrimeQAHybridLexicalClusterDiversityStopVisualization:
    """One generated Stage87 lexical cluster diversity stop-decision chart."""

    name: str
    path: str


def decide_primeqa_hybrid_lexical_cluster_diversity_stop(
    *,
    stage84_report_path: Path,
    stage86_report_path: Path,
    user_confirmed_stop: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Stop the LCDR route after train-selected config fails the dev hit@10 gate."""

    started_at = time.perf_counter()
    stage84_report = _load_json_object(stage84_report_path)
    stage86_report = _load_json_object(stage86_report_path)
    loaded_at = time.perf_counter()

    stage84_candidate = _candidate_summary(stage84_report, _STOPPED_CANDIDATE_ID)
    stage86_summary = _stage86_summary(stage86_report)
    candidate_queue = _candidate_queue(stage84_report)
    guard_checks = _guard_checks(
        stage84_report=stage84_report,
        stage86_report=stage86_report,
        stage84_candidate=stage84_candidate,
        stage86_summary=stage86_summary,
        candidate_queue=candidate_queue,
        user_confirmed_stop=user_confirmed_stop,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "decision_scope": (
            "Train/dev-only stop decision for the Stage84 lexical cluster "
            "diversity rerank route after Stage86 validation. This stage reads "
            "public-safe Stage84 and Stage86 reports, does not load train/dev/test "
            "splits, does not run new retrieval metrics, does not run final "
            "metrics, does not use source DOC_IDS as runtime retrieval evidence, "
            "and does not change runtime defaults."
        ),
        "user_confirmation": {
            "route_id": _STOP_DECISION_ROUTE_ID,
            "confirmed": bool(user_confirmed_stop),
            "confirmation_note": confirmation_note,
        },
        "source_files": {
            "stage84_report": _fingerprint(stage84_report_path),
            "stage86_report": _fingerprint(stage86_report_path),
        },
        "stopped_route": {
            "candidate_id": _STOPPED_CANDIDATE_ID,
            "protocol_id": _STOPPED_PROTOCOL_ID,
            "stage84_candidate_summary": stage84_candidate,
            "stage86_summary": stage86_summary,
            "stop_reason": (
                "The train-selected LCDR config improved train hit@10 but did not "
                "improve dev hit@10. Stage84 required dev hit@10 improvement over "
                "the BM25 baseline, so the route is non-advancing."
            ),
        },
        "candidate_queue": candidate_queue,
        "guard_checks": guard_checks,
        "decision": _decision(
            guard_checks=guard_checks,
            candidate_queue=candidate_queue,
        ),
        "timing_seconds": {
            "load_reports": round(loaded_at - started_at, 3),
            "decision_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_lexical_cluster_diversity_stop_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridLexicalClusterDiversityStopVisualization]:
    """Write SVG charts for Stage87 LCDR stop decision."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage87_lcdr_train_dev_hit10_delta.svg": render_horizontal_bar_chart_svg(
            title="Stage87 LCDR train vs dev hit@10 delta",
            bars=_train_dev_delta_bars(report),
            x_label="hit@10 delta",
            width=980,
            margin_left=260,
        ),
        "stage87_lcdr_dev_change_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage87 LCDR dev changed-case counts",
            bars=_dev_change_bars(report),
            x_label="case count",
            width=980,
            margin_left=360,
        ),
        "stage87_second_wave_remaining_candidate_priority.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage87 remaining second-wave candidate priority",
                bars=_remaining_candidate_priority_bars(report),
                x_label="priority score",
                width=1240,
                margin_left=540,
            )
        ),
        "stage87_lcdr_stop_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage87 LCDR stop decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1120,
            margin_left=470,
        ),
        "stage87_lcdr_stop_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage87 LCDR stop guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1260,
            margin_left=560,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridLexicalClusterDiversityStopVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage86_summary(stage86_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage86_report.get("decision") or {}
    train_selection = stage86_report.get("train_selection") or {}
    train_comparison = (
        train_selection.get("selected_train_comparison_to_baseline") or {}
    )
    dev_comparison = train_selection.get("selected_dev_comparison_to_baseline") or {}
    config = stage86_report.get("config") or {}
    return {
        "status": decision.get("status"),
        "candidate_id": config.get("candidate_id"),
        "protocol_id": config.get("protocol_id"),
        "selected_config_id": decision.get("selected_config_id"),
        "train_hit10_delta": _float_metric(train_comparison, "hit@10_delta"),
        "train_top10_improvement_count": int(
            train_comparison.get("top10_improvement_count") or 0
        ),
        "train_top10_regression_count": int(
            train_comparison.get("top10_regression_count") or 0
        ),
        "dev_hit10_delta": _float_metric(dev_comparison, "hit@10_delta"),
        "dev_top10_improvement_count": int(
            dev_comparison.get("top10_improvement_count") or 0
        ),
        "dev_top10_regression_count": int(
            dev_comparison.get("top10_regression_count") or 0
        ),
        "dev_rank_up_within_top10_count": int(
            dev_comparison.get("rank_up_within_top10_count") or 0
        ),
        "dev_rank_down_within_top10_count": int(
            dev_comparison.get("rank_down_within_top10_count") or 0
        ),
        "dev_not_found_count_at_search_depth_delta": int(
            dev_comparison.get("not_found_count_at_search_depth_delta") or 0
        ),
        "dev_rank_11_to_50_count_delta": int(
            dev_comparison.get("rank_11_to_50_count_delta") or 0
        ),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _candidate_queue(stage84_report: Mapping[str, Any]) -> dict[str, Any]:
    candidate_map = {
        str(candidate.get("candidate_id")): candidate
        for candidate in stage84_report.get("candidate_designs") or []
        if isinstance(candidate, Mapping)
    }
    original_order = [
        str(value)
        for value in stage84_report.get("recommended_execution_order") or []
    ]
    remaining_order = [
        candidate_id
        for candidate_id in original_order
        if candidate_id != _STOPPED_CANDIDATE_ID
        and candidate_id != _BLOCKED_CANDIDATE_ID
        and (candidate_map.get(candidate_id) or {}).get("status")
        == "recommended_for_train_dev_protocol_design"
    ]
    next_candidate_id = remaining_order[0] if remaining_order else None
    return {
        "original_execution_order": original_order,
        "stopped_candidate_id": _STOPPED_CANDIDATE_ID,
        "remaining_execution_order": remaining_order,
        "next_candidate_id": next_candidate_id,
        "next_candidate_summary": _public_candidate(candidate_map.get(next_candidate_id)),
        "remaining_candidate_summaries": [
            _public_candidate(candidate_map.get(candidate_id))
            for candidate_id in remaining_order
        ],
        "blocked_candidate_id": _BLOCKED_CANDIDATE_ID,
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
    stage86_report: Mapping[str, Any],
    stage84_candidate: Mapping[str, Any],
    stage86_summary: Mapping[str, Any],
    candidate_queue: Mapping[str, Any],
    user_confirmed_stop: bool,
) -> list[dict[str, Any]]:
    return [
        _check(
            name="source_stage84_report_is_stage84",
            passed=stage84_report.get("stage") == "Stage 84",
            observed=stage84_report.get("stage"),
            expected="Stage 84",
        ),
        _check(
            name="source_stage86_report_is_stage86",
            passed=stage86_report.get("stage") == "Stage 86",
            observed=stage86_report.get("stage"),
            expected="Stage 86",
        ),
        _check(
            name="user_confirmed_stage87_stop_decision",
            passed=user_confirmed_stop,
            observed=user_confirmed_stop,
            expected=True,
        ),
        _check(
            name="stage86_comparison_completed",
            passed=stage86_summary.get("status")
            == "primeqa_hybrid_lexical_cluster_diversity_comparison_completed",
            observed=stage86_summary.get("status"),
            expected="primeqa_hybrid_lexical_cluster_diversity_comparison_completed",
        ),
        _check(
            name="stage86_candidate_matches_lcdr",
            passed=stage86_summary.get("candidate_id") == _STOPPED_CANDIDATE_ID,
            observed=stage86_summary.get("candidate_id"),
            expected=_STOPPED_CANDIDATE_ID,
        ),
        _check(
            name="stage86_protocol_matches_lcdr",
            passed=stage86_summary.get("protocol_id") == _STOPPED_PROTOCOL_ID,
            observed=stage86_summary.get("protocol_id"),
            expected=_STOPPED_PROTOCOL_ID,
        ),
        _check(
            name="stage84_candidate_metric_contract_requires_dev_hit10_gain",
            passed=_requires_dev_hit10_gain(stage84_candidate),
            observed=stage84_candidate.get("target_metric_contract"),
            expected="primary metric requires dev hit@10 improvement",
        ),
        _check(
            name="stage86_train_selected_config_has_no_dev_hit10_gain",
            passed=float(stage86_summary.get("dev_hit10_delta") or 0.0) <= 0.0,
            observed=stage86_summary.get("dev_hit10_delta"),
            expected="<= 0.0",
        ),
        _check(
            name="stage86_dev_top10_net_not_positive",
            passed=(
                int(stage86_summary.get("dev_top10_improvement_count") or 0)
                - int(stage86_summary.get("dev_top10_regression_count") or 0)
            )
            <= 0,
            observed={
                "improvements": stage86_summary.get("dev_top10_improvement_count"),
                "regressions": stage86_summary.get("dev_top10_regression_count"),
            },
            expected="net <= 0",
        ),
        _check(
            name="stage86_final_test_metrics_locked",
            passed=stage86_summary.get("can_run_final_test_metrics_now") is False,
            observed=stage86_summary.get("can_run_final_test_metrics_now"),
            expected=False,
        ),
        _check(
            name="stage86_final_test_gate_closed",
            passed=stage86_summary.get("can_open_final_test_gate_now") is False,
            observed=stage86_summary.get("can_open_final_test_gate_now"),
            expected=False,
        ),
        _check(
            name="stage86_forbids_test_tuning",
            passed=stage86_summary.get("can_use_test_for_tuning") is False,
            observed=stage86_summary.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage86_default_runtime_policy_unchanged",
            passed=stage86_summary.get("default_runtime_policy") == "unchanged",
            observed=stage86_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage84_execution_order_contains_stopped_candidate",
            passed=_STOPPED_CANDIDATE_ID
            in candidate_queue.get("original_execution_order", []),
            observed=candidate_queue.get("original_execution_order"),
            expected=f"contains {_STOPPED_CANDIDATE_ID}",
        ),
        _check(
            name="stage84_next_candidate_available_after_lcdr_stop",
            passed=bool(candidate_queue.get("next_candidate_id")),
            observed=candidate_queue.get("next_candidate_id"),
            expected="non-empty next candidate id",
        ),
        _check(
            name="source_doc_ids_not_selected_as_next_candidate",
            passed=candidate_queue.get("next_candidate_id") != _BLOCKED_CANDIDATE_ID,
            observed=candidate_queue.get("next_candidate_id"),
            expected=f"not {_BLOCKED_CANDIDATE_ID}",
        ),
        _check(
            name="stage87_no_new_retrieval_metrics_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage87_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage87_default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    candidate_queue: Mapping[str, Any],
) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    next_candidate_id = candidate_queue.get("next_candidate_id")
    if failed_checks:
        return {
            "status": "primeqa_hybrid_lexical_cluster_diversity_stop_decision_blocked",
            "failed_checks": failed_checks,
            "stopped_candidate_id": None,
            "next_candidate_id": None,
            "can_continue_train_dev_development": False,
            "requires_user_confirmation_before_next_protocol": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": "primeqa_hybrid_lexical_cluster_diversity_route_stopped",
        "stopped_candidate_id": _STOPPED_CANDIDATE_ID,
        "stopped_protocol_id": _STOPPED_PROTOCOL_ID,
        "current_route_defaultization": "blocked",
        "next_candidate_id": next_candidate_id,
        "can_continue_train_dev_development": bool(next_candidate_id),
        "requires_user_confirmation_before_next_protocol": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage 88: confirm and freeze the train/dev-only protocol for "
            f"{next_candidate_id}; keep test locked, do not run final metrics, "
            "do not use source DOC_IDS, and keep runtime defaults unchanged."
        ),
    }


def _train_dev_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = (report.get("stopped_route") or {}).get("stage86_summary") or {}
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


def _dev_change_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = (report.get("stopped_route") or {}).get("stage86_summary") or {}
    return [
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
        BarDatum(
            label="dev rank-up within top10",
            value=float(summary.get("dev_rank_up_within_top10_count") or 0),
            value_label=str(summary.get("dev_rank_up_within_top10_count") or 0),
        ),
        BarDatum(
            label="dev rank-down within top10",
            value=float(summary.get("dev_rank_down_within_top10_count") or 0),
            value_label=str(summary.get("dev_rank_down_within_top10_count") or 0),
        ),
    ]


def _remaining_candidate_priority_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summaries = (report.get("candidate_queue") or {}).get("remaining_candidate_summaries") or []
    return [
        BarDatum(
            label=str(candidate.get("candidate_id")),
            value=float(candidate.get("priority_score") or 0),
            value_label=str(candidate.get("priority_score") or 0),
        )
        for candidate in summaries
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    names = [
        "can_continue_train_dev_development",
        "requires_user_confirmation_before_next_protocol",
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


def _requires_dev_hit10_gain(candidate: Mapping[str, Any]) -> bool:
    contract = candidate.get("target_metric_contract") or []
    return any("dev hit@10" in str(item) and "improve" in str(item) for item in contract)


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
