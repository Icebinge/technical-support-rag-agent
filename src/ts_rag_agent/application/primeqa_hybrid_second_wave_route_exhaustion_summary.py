from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 100"
_CREATED_AT = "2026-07-15"
_ROUTE_ID = "second_wave_retrieval_route_exhaustion_summary"
_BLOCKED_CANDIDATE_ID = "source_doc_ids_oracle_union_blocked"
_EXPECTED_SECOND_WAVE_ORDER = (
    "lexical_cluster_diversity_rerank_design",
    "structured_query_keyphrase_compaction_design",
    "section_signal_guarded_expansion_design",
    "score_margin_bm25_normalization_gate_design",
    "selective_dense_sparse_low_overlap_gate_design",
)
_STOP_REPORT_KEYS = ("stage87", "stage90", "stage93", "stage96", "stage99")


@dataclass(frozen=True)
class PrimeQAHybridSecondWaveRouteExhaustionVisualization:
    """One generated Stage100 second-wave route exhaustion chart."""

    name: str
    path: str


def summarize_primeqa_hybrid_second_wave_route_exhaustion(
    *,
    stage83_report_path: Path,
    stage84_report_path: Path,
    stage87_report_path: Path,
    stage90_report_path: Path,
    stage93_report_path: Path,
    stage96_report_path: Path,
    stage99_report_path: Path,
    user_confirmed_summary: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Summarize exhausted second-wave retrieval routes and recommend next research."""

    started_at = time.perf_counter()
    reports = {
        "stage83": _load_json_object(stage83_report_path),
        "stage84": _load_json_object(stage84_report_path),
        "stage87": _load_json_object(stage87_report_path),
        "stage90": _load_json_object(stage90_report_path),
        "stage93": _load_json_object(stage93_report_path),
        "stage96": _load_json_object(stage96_report_path),
        "stage99": _load_json_object(stage99_report_path),
    }
    loaded_at = time.perf_counter()
    route_outcomes = _route_outcomes(reports)
    blocked_diagnostic = _blocked_diagnostic(reports["stage84"])
    aggregate_summary = _aggregate_summary(
        stage83_report=reports["stage83"],
        stage99_report=reports["stage99"],
        route_outcomes=route_outcomes,
        blocked_diagnostic=blocked_diagnostic,
    )
    next_direction_options = _next_direction_options(aggregate_summary)
    guard_checks = _guard_checks(
        reports=reports,
        route_outcomes=route_outcomes,
        blocked_diagnostic=blocked_diagnostic,
        aggregate_summary=aggregate_summary,
        user_confirmed_summary=user_confirmed_summary,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "Public-safe summary of exhausted Stage84 second-wave retrieval "
            "routes. This stage reads saved Stage83, Stage84, Stage87, Stage90, "
            "Stage93, Stage96, and Stage99 reports; does not load train/dev/test "
            "split files; does not run new retrieval metrics; does not run final "
            "metrics; does not use source DOC_IDS as runtime retrieval evidence; "
            "does not tune dev thresholds; and does not change runtime defaults."
        ),
        "user_confirmation": {
            "route_id": _ROUTE_ID,
            "confirmed": bool(user_confirmed_summary),
            "confirmation_note": confirmation_note,
        },
        "source_files": {
            "stage83_report": _fingerprint(stage83_report_path),
            "stage84_report": _fingerprint(stage84_report_path),
            "stage87_report": _fingerprint(stage87_report_path),
            "stage90_report": _fingerprint(stage90_report_path),
            "stage93_report": _fingerprint(stage93_report_path),
            "stage96_report": _fingerprint(stage96_report_path),
            "stage99_report": _fingerprint(stage99_report_path),
        },
        "route_outcomes": route_outcomes,
        "blocked_diagnostic": blocked_diagnostic,
        "aggregate_summary": aggregate_summary,
        "next_direction_options": next_direction_options,
        "guard_checks": guard_checks,
        "decision": _decision(
            guard_checks=guard_checks,
            aggregate_summary=aggregate_summary,
            next_direction_options=next_direction_options,
        ),
        "timing_seconds": {
            "load_reports": round(loaded_at - started_at, 3),
            "summarize_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_second_wave_route_exhaustion_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridSecondWaveRouteExhaustionVisualization]:
    """Write SVG charts for Stage100 second-wave route exhaustion."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage100_second_wave_dev_hit10_deltas.svg": render_horizontal_bar_chart_svg(
            title="Stage100 second-wave dev hit@10 deltas",
            bars=_dev_hit10_delta_bars(report),
            x_label="dev hit@10 delta",
            width=1240,
            margin_left=560,
        ),
        "stage100_second_wave_top10_net_changes.svg": render_horizontal_bar_chart_svg(
            title="Stage100 second-wave top10 net changes",
            bars=_top10_net_bars(report),
            x_label="top10 improvements minus regressions",
            width=1240,
            margin_left=560,
        ),
        "stage100_second_wave_route_outcomes.svg": render_horizontal_bar_chart_svg(
            title="Stage100 second-wave route outcomes",
            bars=_route_outcome_bars(report),
            x_label="1 means runtime-advancing",
            width=1240,
            margin_left=560,
        ),
        "stage100_next_direction_readiness.svg": render_horizontal_bar_chart_svg(
            title="Stage100 next direction readiness",
            bars=_next_direction_bars(report),
            x_label="readiness score",
            width=1280,
            margin_left=610,
        ),
        "stage100_route_exhaustion_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage100 route exhaustion decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1180,
            margin_left=520,
        ),
        "stage100_route_exhaustion_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage100 route exhaustion guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1400,
            margin_left=720,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridSecondWaveRouteExhaustionVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _route_outcomes(reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    outcomes = []
    for key in _STOP_REPORT_KEYS:
        stop_report = reports[key]
        stopped_route = stop_report.get("stopped_route") or {}
        metric_summary = _comparison_summary(stopped_route)
        candidate_summary = stopped_route.get("stage84_candidate_summary") or {}
        decision = stop_report.get("decision") or {}
        outcomes.append(
            {
                "candidate_id": stopped_route.get("candidate_id"),
                "protocol_id": stopped_route.get("protocol_id"),
                "stop_stage": stop_report.get("stage"),
                "stop_status": decision.get("status"),
                "selected_id": _selected_id(metric_summary),
                "priority_score": candidate_summary.get("priority_score"),
                "target_miss_count": candidate_summary.get("target_miss_count"),
                "target_metric_contract": candidate_summary.get(
                    "target_metric_contract"
                )
                or [],
                "train_hit10_delta": _optional_float(
                    metric_summary.get("train_hit10_delta")
                ),
                "dev_hit10_delta": _optional_float(metric_summary.get("dev_hit10_delta")),
                "dev_hit1_delta": _optional_float(metric_summary.get("dev_hit1_delta")),
                "dev_top10_improvement_count": _optional_int(
                    metric_summary.get("dev_top10_improvement_count")
                ),
                "dev_top10_regression_count": _optional_int(
                    metric_summary.get("dev_top10_regression_count")
                ),
                "dev_top10_net": _optional_int(
                    metric_summary.get("dev_top10_improvement_count")
                )
                - _optional_int(metric_summary.get("dev_top10_regression_count")),
                "dev_not_found_at_search_depth_delta": _optional_int_or_none(
                    metric_summary.get("dev_not_found_count_at_search_depth_delta")
                ),
                "primary_contract_passed": metric_summary.get(
                    "primary_contract_passed"
                ),
                "secondary_contract_passed": metric_summary.get(
                    "secondary_contract_passed"
                ),
                "guard_contract_passed": metric_summary.get("guard_contract_passed"),
                "advanced_to_runtime_candidate": False,
                "outcome": "stopped",
                "stop_reason": stopped_route.get("stop_reason"),
                "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
                "can_run_final_test_metrics_now": decision.get(
                    "can_run_final_test_metrics_now"
                ),
                "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
                "default_runtime_policy": decision.get("default_runtime_policy"),
            }
        )
    return outcomes


def _comparison_summary(stopped_route: Mapping[str, Any]) -> Mapping[str, Any]:
    for key, value in stopped_route.items():
        if not key.endswith("_summary"):
            continue
        if key in {"stage84_candidate_summary", "stage97_summary"}:
            continue
        if isinstance(value, Mapping):
            return value
    return {}


def _selected_id(summary: Mapping[str, Any]) -> Any:
    for key in ("selected_config_id", "selected_policy_id", "selected_query_view_id"):
        if summary.get(key) is not None:
            return summary.get(key)
    return None


def _blocked_diagnostic(stage84_report: Mapping[str, Any]) -> dict[str, Any]:
    for candidate in stage84_report.get("candidate_designs") or []:
        if (
            isinstance(candidate, Mapping)
            and candidate.get("candidate_id") == _BLOCKED_CANDIDATE_ID
        ):
            return {
                "candidate_id": _BLOCKED_CANDIDATE_ID,
                "status": candidate.get("status"),
                "target_miss_count": candidate.get("target_miss_count"),
                "target_miss_count_by_split": candidate.get(
                    "target_miss_count_by_split"
                ),
                "runtime_evidence_policy": candidate.get("runtime_evidence_policy"),
                "eligible_for_train_dev_experiment": False,
                "eligible_for_runtime_defaultization": False,
            }
    return {
        "candidate_id": _BLOCKED_CANDIDATE_ID,
        "status": None,
        "eligible_for_train_dev_experiment": False,
        "eligible_for_runtime_defaultization": False,
    }


def _aggregate_summary(
    *,
    stage83_report: Mapping[str, Any],
    stage99_report: Mapping[str, Any],
    route_outcomes: Sequence[Mapping[str, Any]],
    blocked_diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    stage83_decision = stage83_report.get("decision") or {}
    stage99_decision = stage99_report.get("decision") or {}
    stopped_ids = [str(outcome["candidate_id"]) for outcome in route_outcomes]
    return {
        "first_wave_retrieval_candidates_exhausted": bool(
            stage83_decision.get("stage76_allowed_candidates_exhausted")
        ),
        "second_wave_expected_candidate_count": len(_EXPECTED_SECOND_WAVE_ORDER),
        "second_wave_stopped_candidate_count": len(stopped_ids),
        "second_wave_all_expected_candidates_stopped": tuple(stopped_ids)
        == _EXPECTED_SECOND_WAVE_ORDER,
        "runtime_advancing_second_wave_candidate_count": sum(
            bool(outcome["advanced_to_runtime_candidate"]) for outcome in route_outcomes
        ),
        "best_second_wave_dev_hit10_delta": max(
            (float(outcome["dev_hit10_delta"]) for outcome in route_outcomes),
            default=0.0,
        ),
        "best_second_wave_top10_net": max(
            (int(outcome["dev_top10_net"]) for outcome in route_outcomes),
            default=0,
        ),
        "stage99_route_family_exhausted": bool(
            stage99_decision.get("route_family_exhausted")
        ),
        "remaining_actionable_candidate_count": int(
            stage99_decision.get("remaining_actionable_candidate_count") or 0
        ),
        "blocked_source_doc_ids_diagnostic_status": blocked_diagnostic.get("status"),
        "second_wave_retrieval_route_family_exhausted": True,
    }


def _next_direction_options(
    aggregate_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    route_exhausted = bool(
        aggregate_summary.get("second_wave_retrieval_route_family_exhausted")
    )
    return [
        {
            "option_id": "answer_pipeline_error_decomposition",
            "recommended": route_exhausted,
            "requires_user_confirmation": True,
            "description": (
                "Pause retrieval-route invention and decompose remaining train/dev "
                "failures into retrieval, evidence selection, citation, and answer "
                "composition buckets before designing the next intervention."
            ),
            "readiness_score": 0.92,
            "test_policy": "locked",
            "runtime_default_policy": "unchanged",
            "next_stage": (
                "Stage101: design a train/dev-only answer-pipeline error "
                "decomposition protocol from existing public-safe artifacts."
            ),
        },
        {
            "option_id": "third_wave_retrieval_design",
            "recommended": False,
            "requires_user_confirmation": True,
            "description": (
                "Start another retrieval-candidate wave only after a new diagnostic "
                "shows a specific deployable signal not already covered by Stage76 "
                "or Stage84."
            ),
            "readiness_score": 0.42,
            "test_policy": "locked",
            "runtime_default_policy": "unchanged",
            "next_stage": "blocked until new diagnostic evidence exists",
        },
        {
            "option_id": "final_test_gate_review",
            "recommended": False,
            "requires_user_confirmation": True,
            "description": (
                "Do not open final metrics because no retrieval route produced a "
                "train-selected dev contract pass or runtime-default candidate."
            ),
            "readiness_score": 0.0,
            "test_policy": "locked",
            "runtime_default_policy": "unchanged",
            "next_stage": "blocked",
        },
        {
            "option_id": "source_doc_ids_oracle_union",
            "recommended": False,
            "requires_user_confirmation": False,
            "description": (
                "Remain blocked: source DOC_IDS are dataset metadata, not runtime "
                "retrieval evidence."
            ),
            "readiness_score": 0.0,
            "test_policy": "locked",
            "runtime_default_policy": "forbidden",
            "next_stage": "blocked",
        },
    ]


def _guard_checks(
    *,
    reports: Mapping[str, Mapping[str, Any]],
    route_outcomes: Sequence[Mapping[str, Any]],
    blocked_diagnostic: Mapping[str, Any],
    aggregate_summary: Mapping[str, Any],
    user_confirmed_summary: bool,
) -> list[dict[str, Any]]:
    expected_stages = {
        "stage83": "Stage 83",
        "stage84": "Stage 84",
        "stage87": "Stage 87",
        "stage90": "Stage 90",
        "stage93": "Stage 93",
        "stage96": "Stage 96",
        "stage99": "Stage 99",
    }
    return [
        _check(
            name="source_reports_are_expected_stages",
            passed=all(
                reports[key].get("stage") == stage
                for key, stage in expected_stages.items()
            ),
            observed={key: reports[key].get("stage") for key in expected_stages},
            expected=expected_stages,
        ),
        _check(
            name="user_confirmed_stage100_summary",
            passed=user_confirmed_summary,
            observed=user_confirmed_summary,
            expected=True,
        ),
        _check(
            name="stage83_first_wave_exhausted",
            passed=aggregate_summary.get("first_wave_retrieval_candidates_exhausted")
            is True,
            observed=aggregate_summary.get("first_wave_retrieval_candidates_exhausted"),
            expected=True,
        ),
        _check(
            name="stage84_second_wave_order_matches_expected",
            passed=tuple(
                (reports["stage84"].get("decision") or {}).get(
                    "recommended_execution_order"
                )
                or []
            )
            == _EXPECTED_SECOND_WAVE_ORDER,
            observed=(reports["stage84"].get("decision") or {}).get(
                "recommended_execution_order"
            ),
            expected=list(_EXPECTED_SECOND_WAVE_ORDER),
        ),
        _check(
            name="all_second_wave_candidates_have_stop_reports",
            passed=aggregate_summary.get("second_wave_all_expected_candidates_stopped")
            is True,
            observed=[outcome["candidate_id"] for outcome in route_outcomes],
            expected=list(_EXPECTED_SECOND_WAVE_ORDER),
        ),
        _check(
            name="no_second_wave_candidate_advanced_to_runtime",
            passed=int(
                aggregate_summary.get("runtime_advancing_second_wave_candidate_count")
                or 0
            )
            == 0,
            observed=aggregate_summary.get(
                "runtime_advancing_second_wave_candidate_count"
            ),
            expected=0,
        ),
        _check(
            name="best_second_wave_dev_hit10_delta_not_positive",
            passed=float(aggregate_summary.get("best_second_wave_dev_hit10_delta") or 0.0)
            <= 0.0,
            observed=aggregate_summary.get("best_second_wave_dev_hit10_delta"),
            expected="<= 0.0",
        ),
        _check(
            name="stage99_route_family_exhausted",
            passed=aggregate_summary.get("stage99_route_family_exhausted") is True,
            observed=aggregate_summary.get("stage99_route_family_exhausted"),
            expected=True,
        ),
        _check(
            name="no_remaining_actionable_retrieval_candidates",
            passed=int(aggregate_summary.get("remaining_actionable_candidate_count") or 0)
            == 0,
            observed=aggregate_summary.get("remaining_actionable_candidate_count"),
            expected=0,
        ),
        _check(
            name="source_doc_ids_diagnostic_remains_blocked",
            passed=blocked_diagnostic.get("status")
            == "blocked_from_train_dev_experiment"
            and blocked_diagnostic.get("eligible_for_train_dev_experiment") is False
            and blocked_diagnostic.get("eligible_for_runtime_defaultization") is False,
            observed=blocked_diagnostic,
            expected="blocked and not runtime-eligible",
        ),
        _check(
            name="all_stop_reports_have_passing_guards",
            passed=all(
                all(check.get("passed") for check in reports[key].get("guard_checks") or [])
                for key in _STOP_REPORT_KEYS
            ),
            observed={
                key: sum(
                    1
                    for check in reports[key].get("guard_checks") or []
                    if not check.get("passed")
                )
                for key in _STOP_REPORT_KEYS
            },
            expected="0 failed guards per stop report",
        ),
        _check(
            name="all_source_decisions_keep_final_test_locked",
            passed=all(
                (report.get("decision") or {}).get("can_run_final_test_metrics_now")
                is False
                for report in reports.values()
            ),
            observed={
                key: (report.get("decision") or {}).get(
                    "can_run_final_test_metrics_now"
                )
                for key, report in reports.items()
            },
            expected=False,
        ),
        _check(
            name="all_source_decisions_forbid_test_tuning",
            passed=all(
                (report.get("decision") or {}).get("can_use_test_for_tuning") is False
                for report in reports.values()
            ),
            observed={
                key: (report.get("decision") or {}).get("can_use_test_for_tuning")
                for key, report in reports.items()
            },
            expected=False,
        ),
        _check(
            name="all_source_decisions_keep_runtime_defaults_unchanged",
            passed=all(
                (report.get("decision") or {}).get("default_runtime_policy")
                == "unchanged"
                for report in reports.values()
            ),
            observed={
                key: (report.get("decision") or {}).get("default_runtime_policy")
                for key, report in reports.items()
            },
            expected="unchanged",
        ),
        _check(
            name="stage100_runs_summary_only_no_new_retrieval_metrics",
            passed=True,
            observed="summary_only",
            expected="summary_only",
        ),
        _check(
            name="stage100_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage100_default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    aggregate_summary: Mapping[str, Any],
    next_direction_options: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_second_wave_route_exhaustion_summary_blocked",
            "failed_checks": failed_checks,
            "recommended_next_direction": None,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        }
    recommended = next(option for option in next_direction_options if option["recommended"])
    return {
        "status": "primeqa_hybrid_second_wave_route_exhaustion_summary_completed",
        "first_wave_retrieval_candidates_exhausted": aggregate_summary[
            "first_wave_retrieval_candidates_exhausted"
        ],
        "second_wave_retrieval_route_family_exhausted": aggregate_summary[
            "second_wave_retrieval_route_family_exhausted"
        ],
        "runtime_advancing_second_wave_candidate_count": aggregate_summary[
            "runtime_advancing_second_wave_candidate_count"
        ],
        "remaining_actionable_candidate_count": aggregate_summary[
            "remaining_actionable_candidate_count"
        ],
        "recommended_next_direction": recommended["option_id"],
        "requires_user_confirmation_before_next_protocol": True,
        "can_continue_train_dev_development": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": recommended["next_stage"],
    }


def _dev_hit10_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(outcome["candidate_id"]),
            value=float(outcome["dev_hit10_delta"]),
            value_label=f"{float(outcome['dev_hit10_delta']):+.4f}",
        )
        for outcome in report["route_outcomes"]
    ]


def _top10_net_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(outcome["candidate_id"]),
            value=float(outcome["dev_top10_net"]),
            value_label=str(outcome["dev_top10_net"]),
        )
        for outcome in report["route_outcomes"]
    ]


def _route_outcome_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(outcome["candidate_id"]),
            value=1.0 if outcome["advanced_to_runtime_candidate"] else 0.0,
            value_label="advanced" if outcome["advanced_to_runtime_candidate"] else "stopped",
        )
        for outcome in report["route_outcomes"]
    ]


def _next_direction_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(option["option_id"]),
            value=float(option["readiness_score"]),
            value_label=f"{float(option['readiness_score']):.2f}",
        )
        for option in report["next_direction_options"]
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    names = [
        "first_wave_retrieval_candidates_exhausted",
        "second_wave_retrieval_route_family_exhausted",
        "can_continue_train_dev_development",
        "can_open_final_test_gate_now",
        "can_run_final_test_metrics_now",
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
            label=str(check["name"]),
            value=1.0 if check["passed"] else 0.0,
            value_label="passed" if check["passed"] else "failed",
        )
        for check in report["guard_checks"]
    ]


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


def _optional_int(value: Any) -> int:
    return int(value) if value is not None else 0


def _optional_int_or_none(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: Any) -> float:
    return round(float(value), 4) if value is not None else 0.0
