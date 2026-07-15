from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 83"
_CREATED_AT = "2026-07-15"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_ALLOWED_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})

_STAGE76_ALLOWED_CANDIDATES = (
    "query_view_ablation_full_title_dedup",
    "fielded_title_text_bm25_score_fusion",
    "section_bm25_doc_rollup_train_dev_probe",
    "dense_sparse_rrf_train_dev_probe",
    "bm25_k1_b_grid_train_to_dev",
)
_BLOCKED_CANDIDATE_ID = "source_doc_ids_oracle_union_blocked"


@dataclass(frozen=True)
class PrimeQAHybridRetrievalRecallExhaustionVisualization:
    """One generated Stage83 retrieval-recall exhaustion visualization."""

    name: str
    path: str


def summarize_primeqa_hybrid_retrieval_recall_exhaustion(
    *,
    stage76_report_path: Path,
    stage77_report_path: Path,
    stage78_report_path: Path,
    stage79_report_path: Path,
    stage80_report_path: Path,
    stage81_report_path: Path,
    stage82_report_path: Path,
) -> dict[str, Any]:
    """Summarize exhausted Stage76 retrieval-recall candidates from saved reports."""

    started_at = time.perf_counter()
    reports = {
        "stage76": _load_json_object(stage76_report_path),
        "stage77": _load_json_object(stage77_report_path),
        "stage78": _load_json_object(stage78_report_path),
        "stage79": _load_json_object(stage79_report_path),
        "stage80": _load_json_object(stage80_report_path),
        "stage81": _load_json_object(stage81_report_path),
        "stage82": _load_json_object(stage82_report_path),
    }
    loaded_at = time.perf_counter()
    candidate_outcomes = _candidate_outcomes(reports)
    blocked_candidate = _blocked_candidate_summary(reports["stage76"])
    dev_only_observations = _dev_only_observations(reports)
    route_options = _next_route_options()
    guard_checks = _guard_checks(
        reports=reports,
        candidate_outcomes=candidate_outcomes,
        blocked_candidate=blocked_candidate,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "Train/dev-only summary of the exhausted Stage76 retrieval-recall "
            "candidate set. This stage reads saved public-safe Stage76-Stage82 "
            "reports, does not load the frozen test split, does not run new "
            "retrieval metrics, does not run final metrics, does not use source "
            "DOC_IDS as runtime retrieval evidence, and does not change runtime "
            "defaults."
        ),
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
            "forbidden_final_splits": sorted(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage76_report": _fingerprint(stage76_report_path),
            "stage77_report": _fingerprint(stage77_report_path),
            "stage78_report": _fingerprint(stage78_report_path),
            "stage79_report": _fingerprint(stage79_report_path),
            "stage80_report": _fingerprint(stage80_report_path),
            "stage81_report": _fingerprint(stage81_report_path),
            "stage82_report": _fingerprint(stage82_report_path),
        },
        "candidate_outcomes": candidate_outcomes,
        "blocked_candidate": blocked_candidate,
        "dev_only_observations": dev_only_observations,
        "aggregate_summary": _aggregate_summary(candidate_outcomes, blocked_candidate),
        "next_route_options": route_options,
        "guard_checks": guard_checks,
        "decision": _decision(
            guard_checks=guard_checks,
            candidate_outcomes=candidate_outcomes,
            route_options=route_options,
        ),
        "timing_seconds": {
            "load_reports": round(loaded_at - started_at, 3),
            "summarize_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_retrieval_recall_exhaustion_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridRetrievalRecallExhaustionVisualization]:
    """Write SVG charts for Stage83 retrieval-recall exhaustion summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage83_candidate_dev_hit10_deltas.svg": render_horizontal_bar_chart_svg(
            title="Stage83 candidate dev hit@10 deltas",
            bars=_candidate_delta_bars(report),
            x_label="dev hit@10 delta",
            width=1180,
            margin_left=460,
        ),
        "stage83_candidate_top10_net_changes.svg": render_horizontal_bar_chart_svg(
            title="Stage83 candidate dev top10 net changes",
            bars=_candidate_net_change_bars(report),
            x_label="top10 improvements minus regressions",
            width=1180,
            margin_left=460,
        ),
        "stage83_candidate_advancement_status.svg": render_horizontal_bar_chart_svg(
            title="Stage83 candidate advancement status",
            bars=_candidate_status_bars(report),
            x_label="advanced to runtime",
            width=1180,
            margin_left=460,
        ),
        "stage83_next_route_options.svg": render_horizontal_bar_chart_svg(
            title="Stage83 next route option readiness",
            bars=_next_route_option_bars(report),
            x_label="readiness score",
            width=1180,
            margin_left=520,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridRetrievalRecallExhaustionVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _candidate_outcomes(reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _outcome(
            candidate_id="query_view_ablation_full_title_dedup",
            stage="Stage 77",
            report_key="stage77",
            selected_id=(reports["stage77"].get("decision") or {}).get(
                "train_selected_view_id"
            ),
            dev_hit10_delta=(reports["stage77"].get("decision") or {}).get(
                "train_selected_dev_hit10_delta"
            ),
            top10_improvements=(reports["stage77"].get("decision") or {}).get(
                "train_selected_dev_top10_improvements"
            ),
            top10_regressions=(reports["stage77"].get("decision") or {}).get(
                "train_selected_dev_top10_regressions"
            ),
            not_found_delta=None,
            outcome_reason="train-selected query view regressed on dev hit@10",
            report=reports["stage77"],
        ),
        _outcome(
            candidate_id="fielded_title_text_bm25_score_fusion",
            stage="Stage 78",
            report_key="stage78",
            selected_id=(reports["stage78"].get("decision") or {}).get(
                "train_selected_config_id"
            ),
            dev_hit10_delta=(reports["stage78"].get("decision") or {}).get(
                "train_selected_dev_hit10_delta"
            ),
            top10_improvements=(reports["stage78"].get("decision") or {}).get(
                "train_selected_dev_top10_improvements"
            ),
            top10_regressions=(reports["stage78"].get("decision") or {}).get(
                "train_selected_dev_top10_regressions"
            ),
            not_found_delta=None,
            outcome_reason="train-selected fielded fusion produced no dev hit@10 gain",
            report=reports["stage78"],
        ),
        _outcome(
            candidate_id="section_bm25_doc_rollup_train_dev_probe",
            stage="Stage 79",
            report_key="stage79",
            selected_id=(reports["stage79"].get("decision") or {}).get(
                "candidate_config_id"
            ),
            dev_hit10_delta=(reports["stage79"].get("decision") or {}).get(
                "candidate_dev_hit10_delta"
            ),
            top10_improvements=(reports["stage79"].get("decision") or {}).get(
                "candidate_dev_top10_improvements"
            ),
            top10_regressions=(reports["stage79"].get("decision") or {}).get(
                "candidate_dev_top10_regressions"
            ),
            not_found_delta=(reports["stage79"].get("decision") or {}).get(
                "candidate_dev_not_found_at_search_depth_delta"
            ),
            outcome_reason="section rollup regressed on dev hit@10",
            report=reports["stage79"],
        ),
        _outcome(
            candidate_id="dense_sparse_rrf_train_dev_probe",
            stage="Stage 81",
            report_key="stage81",
            selected_id=(reports["stage81"].get("decision") or {}).get(
                "selected_config_id"
            ),
            dev_hit10_delta=(reports["stage81"].get("decision") or {}).get(
                "selected_dev_hit10_delta"
            ),
            top10_improvements=(reports["stage81"].get("decision") or {}).get(
                "selected_dev_top10_improvements"
            ),
            top10_regressions=(reports["stage81"].get("decision") or {}).get(
                "selected_dev_top10_regressions"
            ),
            not_found_delta=(reports["stage81"].get("decision") or {}).get(
                "selected_dev_not_found_at_search_depth_delta"
            ),
            outcome_reason=(
                "train-selected dense+sparse RRF improved train but regressed on "
                "dev hit@10"
            ),
            report=reports["stage81"],
        ),
        _outcome(
            candidate_id="bm25_k1_b_grid_train_to_dev",
            stage="Stage 82",
            report_key="stage82",
            selected_id=(reports["stage82"].get("decision") or {}).get(
                "selected_config_id"
            ),
            dev_hit10_delta=(reports["stage82"].get("decision") or {}).get(
                "selected_dev_hit10_delta"
            ),
            top10_improvements=(reports["stage82"].get("decision") or {}).get(
                "selected_dev_top10_improvements"
            ),
            top10_regressions=(reports["stage82"].get("decision") or {}).get(
                "selected_dev_top10_regressions"
            ),
            not_found_delta=(reports["stage82"].get("decision") or {}).get(
                "selected_dev_not_found_at_search_depth_delta"
            ),
            outcome_reason="train selection stayed on the existing BM25 baseline",
            report=reports["stage82"],
        ),
    ]


def _outcome(
    *,
    candidate_id: str,
    stage: str,
    report_key: str,
    selected_id: Any,
    dev_hit10_delta: Any,
    top10_improvements: Any,
    top10_regressions: Any,
    not_found_delta: Any,
    outcome_reason: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    decision = report.get("decision") or {}
    dev_delta = _optional_float(dev_hit10_delta)
    advanced = bool(dev_delta is not None and dev_delta > 0)
    return {
        "candidate_id": candidate_id,
        "stage": stage,
        "report_key": report_key,
        "report_status": decision.get("status"),
        "selected_id": selected_id,
        "selected_dev_hit10_delta": dev_delta,
        "selected_dev_top10_improvements": _optional_int(top10_improvements),
        "selected_dev_top10_regressions": _optional_int(top10_regressions),
        "selected_dev_top10_net": _optional_int(top10_improvements)
        - _optional_int(top10_regressions),
        "selected_dev_not_found_at_search_depth_delta": _optional_int(not_found_delta)
        if not_found_delta is not None
        else None,
        "advanced_to_runtime_candidate": advanced,
        "outcome": "advanced" if advanced else "not_advanced",
        "outcome_reason": outcome_reason,
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _blocked_candidate_summary(stage76_report: Mapping[str, Any]) -> dict[str, Any]:
    candidates = stage76_report.get("candidate_designs") or []
    blocked = next(
        (
            candidate
            for candidate in candidates
            if candidate.get("candidate_id") == _BLOCKED_CANDIDATE_ID
        ),
        {},
    )
    return {
        "candidate_id": _BLOCKED_CANDIDATE_ID,
        "status": blocked.get("status"),
        "reason": (
            "source DOC_IDS are source metadata, not runtime user-query evidence; "
            "using them would be non-deployable leakage."
        ),
        "target_miss_count": blocked.get("target_miss_count"),
        "target_miss_count_by_split": blocked.get("target_miss_count_by_split"),
    }


def _dev_only_observations(reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    observations = []
    stage82 = reports["stage82"]
    stage82_dev_metrics = stage82.get("metrics_by_split", {}).get("dev", {})
    baseline_hit10 = (
        stage82_dev_metrics.get("full_document_bm25_baseline", {})
        .get("hit_at_k", {})
        .get("hit@10")
    )
    selected_config_id = (stage82.get("decision") or {}).get("selected_config_id")
    for config_id, metrics in stage82_dev_metrics.items():
        hit10 = (metrics.get("hit_at_k") or {}).get("hit@10")
        if config_id == selected_config_id:
            continue
        if baseline_hit10 is not None and hit10 is not None and hit10 > baseline_hit10:
            observations.append(
                {
                    "stage": "Stage 82",
                    "config_id": config_id,
                    "dev_hit10": hit10,
                    "baseline_dev_hit10": baseline_hit10,
                    "dev_hit10_delta": round(float(hit10) - float(baseline_hit10), 4),
                    "not_selectable_reason": (
                        "This config was not selected by the train-only rule; using "
                        "it would be dev-set selection."
                    ),
                }
            )
    return observations


def _aggregate_summary(
    candidate_outcomes: Sequence[Mapping[str, Any]],
    blocked_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "allowed_stage76_candidate_count": len(_STAGE76_ALLOWED_CANDIDATES),
        "allowed_candidate_outcome_count": len(candidate_outcomes),
        "allowed_candidates_completed": sorted(
            outcome["candidate_id"] for outcome in candidate_outcomes
        )
        == sorted(_STAGE76_ALLOWED_CANDIDATES),
        "blocked_candidate_count": 1 if blocked_candidate.get("status") else 0,
        "runtime_advancing_candidate_count": sum(
            bool(outcome["advanced_to_runtime_candidate"])
            for outcome in candidate_outcomes
        ),
        "best_selected_dev_hit10_delta": max(
            (float(outcome["selected_dev_hit10_delta"]) for outcome in candidate_outcomes),
            default=0.0,
        ),
        "stage76_retrieval_recall_set_exhausted": True,
    }


def _next_route_options() -> list[dict[str, Any]]:
    return [
        {
            "option_id": "second_wave_retrieval_candidate_design",
            "recommended": True,
            "requires_user_confirmation": True,
            "stage84_action": "design_only",
            "description": (
                "Aggregate Stage75 and Stage77-82 changed-case evidence, then "
                "design a second-wave train/dev-only retrieval candidate set "
                "before running any new metrics."
            ),
            "readiness_score": 0.9,
            "test_policy": "locked",
            "runtime_default_policy": "unchanged",
        },
        {
            "option_id": "answer_pipeline_error_decomposition",
            "recommended": False,
            "requires_user_confirmation": True,
            "stage84_action": "diagnostic_only",
            "description": (
                "Pause retrieval changes and quantify whether current failures are "
                "primarily retrieval, citation selection, or answer composition."
            ),
            "readiness_score": 0.7,
            "test_policy": "locked",
            "runtime_default_policy": "unchanged",
        },
        {
            "option_id": "freeze_retrieval_and_review_final_gate_requirements",
            "recommended": False,
            "requires_user_confirmation": True,
            "stage84_action": "gate_review_only",
            "description": (
                "Do not open final metrics; only review whether more train/dev "
                "evidence is needed before any future final-test gate discussion."
            ),
            "readiness_score": 0.5,
            "test_policy": "locked",
            "runtime_default_policy": "unchanged",
        },
    ]


def _guard_checks(
    *,
    reports: Mapping[str, Mapping[str, Any]],
    candidate_outcomes: Sequence[Mapping[str, Any]],
    blocked_candidate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected_stages = {
        "stage76": "Stage 76",
        "stage77": "Stage 77",
        "stage78": "Stage 78",
        "stage79": "Stage 79",
        "stage80": "Stage 80",
        "stage81": "Stage 81",
        "stage82": "Stage 82",
    }
    return [
        _check(
            name="source_reports_are_expected_stages",
            passed=all(
                str(reports[key].get("stage") or "") == expected
                for key, expected in expected_stages.items()
            ),
            observed={key: reports[key].get("stage") for key in expected_stages},
            expected=expected_stages,
        ),
        _check(
            name="stage76_allowed_candidates_all_accounted_for",
            passed=sorted(outcome["candidate_id"] for outcome in candidate_outcomes)
            == sorted(_STAGE76_ALLOWED_CANDIDATES),
            observed=sorted(outcome["candidate_id"] for outcome in candidate_outcomes),
            expected=sorted(_STAGE76_ALLOWED_CANDIDATES),
        ),
        _check(
            name="source_doc_ids_candidate_remains_blocked",
            passed=blocked_candidate.get("status") == "blocked_from_train_dev_experiment",
            observed=blocked_candidate.get("status"),
            expected="blocked_from_train_dev_experiment",
        ),
        _check(
            name="no_allowed_candidate_advanced_to_runtime",
            passed=not any(
                bool(outcome["advanced_to_runtime_candidate"])
                for outcome in candidate_outcomes
            ),
            observed=[
                {
                    "candidate_id": outcome["candidate_id"],
                    "advanced": outcome["advanced_to_runtime_candidate"],
                }
                for outcome in candidate_outcomes
            ],
            expected=False,
        ),
        _check(
            name="all_source_decisions_keep_final_test_locked",
            passed=all(
                (report.get("decision") or {}).get("can_run_final_test_metrics_now")
                is False
                for report in reports.values()
            ),
            observed={
                key: (report.get("decision") or {}).get("can_run_final_test_metrics_now")
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
            name="stage83_runs_summary_only_no_new_retrieval_metrics",
            passed=True,
            observed="summary_only",
            expected="summary_only",
        ),
        _check(
            name="stage83_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage83_default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    candidate_outcomes: Sequence[Mapping[str, Any]],
    route_options: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failed_checks = [check["name"] for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_retrieval_recall_exhaustion_summary_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        }
    recommended = next(option for option in route_options if option["recommended"])
    return {
        "status": "primeqa_hybrid_retrieval_recall_exhaustion_summary_completed",
        "stage76_allowed_candidates_exhausted": True,
        "runtime_advancing_candidate_count": sum(
            bool(outcome["advanced_to_runtime_candidate"])
            for outcome in candidate_outcomes
        ),
        "can_continue_train_dev_development": True,
        "requires_user_confirmation_before_next_route": True,
        "recommended_next_route_option": recommended["option_id"],
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage 84: confirm the next train/dev-only route. Recommended option "
            "is second_wave_retrieval_candidate_design; keep test locked and do "
            "not run final metrics."
        ),
    }


def _candidate_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=outcome["candidate_id"],
            value=float(outcome["selected_dev_hit10_delta"]),
            value_label=f"{outcome['selected_dev_hit10_delta']:+.4f}",
        )
        for outcome in report["candidate_outcomes"]
    ]


def _candidate_net_change_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=outcome["candidate_id"],
            value=float(outcome["selected_dev_top10_net"]),
            value_label=str(outcome["selected_dev_top10_net"]),
        )
        for outcome in report["candidate_outcomes"]
    ]


def _candidate_status_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=outcome["candidate_id"],
            value=1.0 if outcome["advanced_to_runtime_candidate"] else 0.0,
            value_label="advanced" if outcome["advanced_to_runtime_candidate"] else "no",
        )
        for outcome in report["candidate_outcomes"]
    ]


def _next_route_option_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=option["option_id"],
            value=float(option["readiness_score"]),
            value_label=f"{option['readiness_score']:.1f}",
        )
        for option in report["next_route_options"]
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


def _optional_float(value: Any) -> float:
    return round(float(value), 4) if value is not None else 0.0
