from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 106"
_CREATED_AT = "2026-07-15"
_SOURCE_STAGE105 = "Stage 105"
_STOP_DECISION_ROUTE_ID = "evidence_answerability_stop_decision"
_STOPPED_FAMILY_ID = "evidence_answerability_candidate_family"
_PROTOCOL_ID = "evidence_answerability_candidate_train_dev_comparison_v1"
_STAGE105_ANALYSIS_ID = "evidence_answerability_candidate_train_dev_comparison_v1"
_STAGE104_STATUS = "primeqa_hybrid_evidence_answerability_comparison_protocol_frozen"
_STAGE105_DEV_GUARD_FAILED_STATUS = (
    "primeqa_hybrid_evidence_answerability_comparison_completed_dev_guard_failed"
)


@dataclass(frozen=True)
class PrimeQAHybridEvidenceAnswerabilityStopVisualization:
    """One generated Stage106 evidence-answerability stop-decision chart."""

    name: str
    path: str


def decide_primeqa_hybrid_evidence_answerability_stop(
    *,
    stage104_protocol_path: Path,
    stage105_report_path: Path,
    user_confirmed_stop: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Stop the evidence-answerability candidate family after Stage105 fails dev."""

    started_at = time.perf_counter()
    stage104_report = _load_json_object(stage104_protocol_path)
    stage105_report = _load_json_object(stage105_report_path)
    loaded_at = time.perf_counter()

    stage104_summary = _stage104_summary(stage104_report)
    stage105_summary = _stage105_summary(stage105_report)
    candidate_family_summary = _candidate_family_summary(stage105_report)
    dev_better_nonselectable = _dev_better_nonselectable_configs(stage105_report)
    guard_checks = _guard_checks(
        stage104_summary=stage104_summary,
        stage105_summary=stage105_summary,
        stage105_report=stage105_report,
        dev_better_nonselectable=dev_better_nonselectable,
        user_confirmed_stop=user_confirmed_stop,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "decision_scope": (
            "Train/dev-only stop decision for the Stage103/104 "
            "evidence-answerability candidate family after Stage105 validation. "
            "This stage reads public-safe Stage104 and Stage105 reports, does "
            "not load train/dev/test split files, does not run retrieval or "
            "answer metrics, does not run final metrics, does not select from "
            "dev-only observations, does not add fallback strategies, and does "
            "not change runtime defaults."
        ),
        "user_confirmation": {
            "route_id": _STOP_DECISION_ROUTE_ID,
            "confirmed": bool(user_confirmed_stop),
            "confirmation_note": confirmation_note,
        },
        "source_files": {
            "stage104_protocol": _fingerprint(stage104_protocol_path),
            "stage105_report": _fingerprint(stage105_report_path),
        },
        "stopped_family": {
            "family_id": _STOPPED_FAMILY_ID,
            "protocol_id": _PROTOCOL_ID,
            "stage104_summary": stage104_summary,
            "stage105_summary": stage105_summary,
            "candidate_family_summary": candidate_family_summary,
            "dev_better_nonselectable_configs": dev_better_nonselectable,
            "stop_reason": (
                "The train-selected Stage105 config did not improve the train "
                "weighted target objective and failed dev validation with zero "
                "dev weighted target improvement and zero dev changed answers. "
                "The configs with better dev target deltas were not train "
                "selectable under the frozen Stage104 guards, so they cannot "
                "be chosen from dev. The current family therefore provides no "
                "justification for runtime defaultization or final-test gate "
                "opening."
            ),
        },
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks=guard_checks),
        "timing_seconds": {
            "load_reports": round(loaded_at - started_at, 3),
            "decision_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_evidence_answerability_stop_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridEvidenceAnswerabilityStopVisualization]:
    """Write SVG charts for Stage106 evidence-answerability stop decision."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage106_evidence_answerability_target_deltas.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage106 evidence-answerability target deltas",
                bars=_target_delta_bars(report),
                x_label="weighted target delta",
                width=1380,
                margin_left=700,
            )
        ),
        "stage106_train_selectability_by_family.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage106 train selectability by family",
                bars=_family_selectability_bars(report),
                x_label="config count",
                width=1320,
                margin_left=640,
            )
        ),
        "stage106_train_guard_failure_reasons.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage106 train guard failure reasons",
                bars=_guard_failure_reason_bars(report),
                x_label="failed config count",
                width=1420,
                margin_left=760,
            )
        ),
        "stage106_stop_decision_flags.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage106 stop decision flags",
                bars=_decision_flag_bars(report),
                x_label="1 means true",
                width=1260,
                margin_left=620,
            )
        ),
        "stage106_stop_guard_check_status.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage106 stop guard checks",
                bars=_guard_check_bars(report),
                x_label="1 means passed",
                width=1560,
                margin_left=820,
            )
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridEvidenceAnswerabilityStopVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage104_summary(stage104_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage104_report.get("decision") or {}
    frozen_protocol = stage104_report.get("frozen_protocol") or {}
    candidate_grid = frozen_protocol.get("candidate_config_grid") or []
    return {
        "stage": stage104_report.get("stage"),
        "protocol_id": stage104_report.get("protocol_id"),
        "decision_status": decision.get("status"),
        "recommended_direction": decision.get("recommended_direction"),
        "can_run_train_dev_candidate_comparison_after_user_confirmation": decision.get(
            "can_run_train_dev_candidate_comparison_after_user_confirmation"
        ),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get(
            "can_run_final_test_metrics_now"
        ),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "config_count": len(candidate_grid),
        "candidate_ids": sorted(
            {str(config.get("candidate_id")) for config in candidate_grid}
        ),
        "dev_validation_rule": frozen_protocol.get("dev_validation_rule") or {},
        "train_selection_rule": frozen_protocol.get("train_selection_rule") or {},
        "runtime_feature_contract": frozen_protocol.get("runtime_feature_contract") or {},
        "fallback_strategy_policy": frozen_protocol.get("fallback_strategy_policy")
        or {},
    }


def _stage105_summary(stage105_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage105_report.get("decision") or {}
    train_selection = stage105_report.get("train_selection") or {}
    dev_validation = stage105_report.get("dev_validation") or {}
    selected_config_id = decision.get("selected_config_id")
    selected_config = _config_by_id(stage105_report).get(str(selected_config_id))
    return {
        "stage": stage105_report.get("stage"),
        "analysis_id": stage105_report.get("analysis_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "selected_config_id": selected_config_id,
        "selected_candidate_id": decision.get("selected_candidate_id"),
        "selectable_config_count": decision.get("selectable_config_count"),
        "config_count": train_selection.get("config_count")
        or len(stage105_report.get("config_results") or []),
        "selection_split": train_selection.get("selection_split"),
        "selected_train_weighted_target_delta": train_selection.get(
            "selected_train_weighted_target_delta"
        ),
        "dev_validation_passed": dev_validation.get("dev_validation_passed"),
        "dev_weighted_target_delta": dev_validation.get("dev_weighted_target_delta"),
        "dev_changed_answer_count": dev_validation.get("dev_changed_answer_count"),
        "dev_target_bucket_deltas": dev_validation.get("dev_target_bucket_deltas") or {},
        "dev_metric_deltas": dev_validation.get("dev_metric_deltas") or {},
        "selected_train_selectability": (
            selected_config.get("train_selectability") if selected_config else {}
        ),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get(
            "can_run_final_test_metrics_now"
        ),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _candidate_family_summary(stage105_report: Mapping[str, Any]) -> dict[str, Any]:
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for config in stage105_report.get("config_results") or []:
        if isinstance(config, Mapping):
            by_family[str(config.get("candidate_id"))].append(config)

    return {
        family_id: _family_summary(family_configs)
        for family_id, family_configs in sorted(by_family.items())
    }


def _family_summary(configs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selectable_configs = [
        config
        for config in configs
        if (config.get("train_selectability") or {}).get("selectable") is True
    ]
    best_train = min(
        configs,
        key=lambda config: float(
            (config.get("weighted_target_score_deltas_by_split") or {}).get("train")
            or 0.0
        ),
    )
    best_dev = min(
        configs,
        key=lambda config: float(
            (config.get("weighted_target_score_deltas_by_split") or {}).get("dev")
            or 0.0
        ),
    )
    return {
        "config_count": len(configs),
        "train_selectable_config_count": len(selectable_configs),
        "best_train_delta_config_id": best_train.get("config_id"),
        "best_train_weighted_target_delta": (
            best_train.get("weighted_target_score_deltas_by_split") or {}
        ).get("train"),
        "best_dev_delta_config_id": best_dev.get("config_id"),
        "best_dev_weighted_target_delta": (
            best_dev.get("weighted_target_score_deltas_by_split") or {}
        ).get("dev"),
        "train_guard_failure_reasons": dict(
            sorted(_train_guard_failure_reasons(configs).items())
        ),
    }


def _dev_better_nonselectable_configs(
    stage105_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected_dev_delta = float(
        ((stage105_report.get("dev_validation") or {}).get("dev_weighted_target_delta"))
        or 0.0
    )
    better_configs = []
    for config in stage105_report.get("config_results") or []:
        if not isinstance(config, Mapping):
            continue
        dev_delta = float(
            (config.get("weighted_target_score_deltas_by_split") or {}).get("dev")
            or 0.0
        )
        selectable = (config.get("train_selectability") or {}).get("selectable") is True
        if dev_delta < selected_dev_delta and not selectable:
            better_configs.append(
                {
                    "config_id": config.get("config_id"),
                    "candidate_id": config.get("candidate_id"),
                    "dev_weighted_target_delta": dev_delta,
                    "train_weighted_target_delta": (
                        config.get("weighted_target_score_deltas_by_split") or {}
                    ).get("train"),
                    "train_selectable": selectable,
                    "failed_train_guards": _failed_train_guard_names(config),
                }
            )
    return sorted(
        better_configs,
        key=lambda item: (float(item["dev_weighted_target_delta"]), str(item["config_id"])),
    )


def _guard_checks(
    *,
    stage104_summary: Mapping[str, Any],
    stage105_summary: Mapping[str, Any],
    stage105_report: Mapping[str, Any],
    dev_better_nonselectable: Sequence[Mapping[str, Any]],
    user_confirmed_stop: bool,
) -> list[dict[str, Any]]:
    stage105_guard_checks = stage105_report.get("guard_checks") or []
    dev_validation_rule = stage104_summary.get("dev_validation_rule") or {}
    train_selection_rule = stage104_summary.get("train_selection_rule") or {}
    fallback_policy = stage104_summary.get("fallback_strategy_policy") or {}
    return [
        _check(
            name="source_stage104_report_is_stage104",
            passed=stage104_summary.get("stage") == "Stage 104",
            observed=stage104_summary.get("stage"),
            expected="Stage 104",
        ),
        _check(
            name="source_stage105_report_is_stage105",
            passed=stage105_summary.get("stage") == _SOURCE_STAGE105,
            observed=stage105_summary.get("stage"),
            expected=_SOURCE_STAGE105,
        ),
        _check(
            name="user_confirmed_stage106_stop_decision",
            passed=user_confirmed_stop,
            observed=user_confirmed_stop,
            expected=True,
        ),
        _check(
            name="stage104_protocol_is_frozen",
            passed=stage104_summary.get("decision_status") == _STAGE104_STATUS,
            observed=stage104_summary.get("decision_status"),
            expected=_STAGE104_STATUS,
        ),
        _check(
            name="stage104_protocol_id_matches",
            passed=stage104_summary.get("protocol_id") == _PROTOCOL_ID,
            observed=stage104_summary.get("protocol_id"),
            expected=_PROTOCOL_ID,
        ),
        _check(
            name="stage104_candidate_grid_has_nine_configs",
            passed=int(stage104_summary.get("config_count") or 0) == 9,
            observed=stage104_summary.get("config_count"),
            expected=9,
        ),
        _check(
            name="stage104_train_selection_is_train_only",
            passed=train_selection_rule.get("selection_split") == "train"
            and train_selection_rule.get("validation_split") == "dev"
            and train_selection_rule.get("dev_threshold_tuning_allowed") is False
            and train_selection_rule.get("test_access_allowed") is False,
            observed=train_selection_rule,
            expected="train selection only; dev validation only; no test access",
        ),
        _check(
            name="stage104_dev_validation_forbids_dev_selection",
            passed=dev_validation_rule.get("dev_selection_allowed") is False
            and dev_validation_rule.get("dev_retuning_allowed") is False
            and dev_validation_rule.get("test_access_allowed") is False,
            observed=dev_validation_rule,
            expected="dev selection and retuning false; test access false",
        ),
        _check(
            name="stage104_final_test_gate_locked",
            passed=stage104_summary.get("can_open_final_test_gate_now") is False
            and stage104_summary.get("can_run_final_test_metrics_now") is False,
            observed={
                "can_open_final_test_gate_now": stage104_summary.get(
                    "can_open_final_test_gate_now"
                ),
                "can_run_final_test_metrics_now": stage104_summary.get(
                    "can_run_final_test_metrics_now"
                ),
            },
            expected=False,
        ),
        _check(
            name="stage104_runtime_defaults_unchanged",
            passed=stage104_summary.get("default_runtime_policy") == "unchanged",
            observed=stage104_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage104_fallback_disabled",
            passed=stage104_summary.get("fallback_strategies_enabled") is False
            and _fallback_policy_disabled(fallback_policy),
            observed={
                "decision_flag": stage104_summary.get("fallback_strategies_enabled"),
                "fallback_policy": fallback_policy,
            },
            expected=False,
        ),
        _check(
            name="stage105_analysis_id_matches",
            passed=stage105_summary.get("analysis_id") == _STAGE105_ANALYSIS_ID,
            observed=stage105_summary.get("analysis_id"),
            expected=_STAGE105_ANALYSIS_ID,
        ),
        _check(
            name="stage105_completed_with_dev_guard_failed",
            passed=stage105_summary.get("decision_status")
            == _STAGE105_DEV_GUARD_FAILED_STATUS,
            observed=stage105_summary.get("decision_status"),
            expected=_STAGE105_DEV_GUARD_FAILED_STATUS,
        ),
        _check(
            name="stage105_recommends_stop_decision",
            passed=stage105_summary.get("recommended_next_direction")
            == _STOP_DECISION_ROUTE_ID,
            observed=stage105_summary.get("recommended_next_direction"),
            expected=_STOP_DECISION_ROUTE_ID,
        ),
        _check(
            name="stage105_all_guard_checks_passed",
            passed=all(check.get("passed") is True for check in stage105_guard_checks),
            observed={
                "passed": sum(1 for check in stage105_guard_checks if check.get("passed")),
                "total": len(stage105_guard_checks),
            },
            expected="all passed",
        ),
        _check(
            name="stage105_selection_uses_train_split",
            passed=stage105_summary.get("selection_split") == "train",
            observed=stage105_summary.get("selection_split"),
            expected="train",
        ),
        _check(
            name="stage105_selected_config_did_not_improve_train_target",
            passed=float(
                stage105_summary.get("selected_train_weighted_target_delta") or 0.0
            )
            >= 0.0,
            observed=stage105_summary.get("selected_train_weighted_target_delta"),
            expected=">= 0.0",
        ),
        _check(
            name="stage105_selected_config_failed_dev_validation",
            passed=stage105_summary.get("dev_validation_passed") is False,
            observed=stage105_summary.get("dev_validation_passed"),
            expected=False,
        ),
        _check(
            name="stage105_selected_config_did_not_improve_dev_target",
            passed=float(stage105_summary.get("dev_weighted_target_delta") or 0.0)
            >= 0.0,
            observed=stage105_summary.get("dev_weighted_target_delta"),
            expected=">= 0.0",
        ),
        _check(
            name="stage105_selected_config_changed_no_dev_answers",
            passed=int(stage105_summary.get("dev_changed_answer_count") or 0) == 0,
            observed=stage105_summary.get("dev_changed_answer_count"),
            expected=0,
        ),
        _check(
            name="stage105_dev_better_configs_are_train_nonselectable",
            passed=bool(dev_better_nonselectable)
            and all(item.get("train_selectable") is False for item in dev_better_nonselectable),
            observed=dev_better_nonselectable,
            expected="at least one dev-better config, all train non-selectable",
        ),
        _check(
            name="stage105_final_test_gate_locked",
            passed=stage105_summary.get("can_open_final_test_gate_now") is False
            and stage105_summary.get("can_run_final_test_metrics_now") is False,
            observed={
                "can_open_final_test_gate_now": stage105_summary.get(
                    "can_open_final_test_gate_now"
                ),
                "can_run_final_test_metrics_now": stage105_summary.get(
                    "can_run_final_test_metrics_now"
                ),
            },
            expected=False,
        ),
        _check(
            name="stage105_forbids_test_tuning",
            passed=stage105_summary.get("can_use_test_for_tuning") is False,
            observed=stage105_summary.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage105_default_runtime_policy_unchanged",
            passed=stage105_summary.get("default_runtime_policy") == "unchanged",
            observed=stage105_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage105_fallback_strategies_disabled",
            passed=stage105_summary.get("fallback_strategies_enabled") is False,
            observed=stage105_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="stage106_no_new_train_dev_metrics_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage106_test_split_not_loaded",
            passed=True,
            observed="not_loaded",
            expected="not_loaded",
        ),
        _check(
            name="stage106_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage106_default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
        _check(
            name="stage106_fallback_strategies_not_added",
            passed=True,
            observed=False,
            expected=False,
        ),
    ]


def _decision(*, guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_evidence_answerability_stop_decision_blocked",
            "failed_checks": failed_checks,
            "stopped_family_id": None,
            "current_route_defaultization": "blocked",
            "can_continue_train_dev_development": False,
            "requires_user_confirmation_before_next_protocol": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": "primeqa_hybrid_evidence_answerability_candidate_family_stopped",
        "stopped_family_id": _STOPPED_FAMILY_ID,
        "stopped_protocol_id": _PROTOCOL_ID,
        "current_route_defaultization": "blocked",
        "redesign_required_before_any_runtime_or_test_gate": True,
        "recommended_next_direction": "evidence_answerability_redesign_decision",
        "can_continue_train_dev_development": True,
        "requires_user_confirmation_before_next_protocol": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage107: only if the user confirms redesign, freeze a new "
            "train/dev-only evidence-answerability redesign protocol. Do not "
            "select from dev-only observations, keep test locked, keep runtime "
            "defaults unchanged, and do not add fallback strategies."
        ),
    }


def _target_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    family_summary = (
        (report.get("stopped_family") or {}).get("candidate_family_summary") or {}
    )
    bars = []
    for family_id, summary in family_summary.items():
        bars.append(
            BarDatum(
                label=f"{family_id} train best",
                value=float(summary.get("best_train_weighted_target_delta") or 0.0),
                value_label=(
                    f"{float(summary.get('best_train_weighted_target_delta') or 0.0):+.2f}"
                ),
            )
        )
        bars.append(
            BarDatum(
                label=f"{family_id} dev best",
                value=float(summary.get("best_dev_weighted_target_delta") or 0.0),
                value_label=(
                    f"{float(summary.get('best_dev_weighted_target_delta') or 0.0):+.2f}"
                ),
            )
        )
    return bars


def _family_selectability_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    family_summary = (
        (report.get("stopped_family") or {}).get("candidate_family_summary") or {}
    )
    bars = []
    for family_id, summary in family_summary.items():
        bars.append(
            BarDatum(
                label=f"{family_id} selectable",
                value=float(summary.get("train_selectable_config_count") or 0),
                value_label=str(summary.get("train_selectable_config_count") or 0),
            )
        )
        bars.append(
            BarDatum(
                label=f"{family_id} blocked",
                value=float(
                    int(summary.get("config_count") or 0)
                    - int(summary.get("train_selectable_config_count") or 0)
                ),
                value_label=str(
                    int(summary.get("config_count") or 0)
                    - int(summary.get("train_selectable_config_count") or 0)
                ),
            )
        )
    return bars


def _guard_failure_reason_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    family_summary = (
        (report.get("stopped_family") or {}).get("candidate_family_summary") or {}
    )
    counter: Counter[str] = Counter()
    for summary in family_summary.values():
        counter.update(summary.get("train_guard_failure_reasons") or {})
    return [
        BarDatum(
            label=reason,
            value=float(count),
            value_label=str(count),
        )
        for reason, count in sorted(counter.items())
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    names = [
        "redesign_required_before_any_runtime_or_test_gate",
        "can_continue_train_dev_development",
        "requires_user_confirmation_before_next_protocol",
        "can_open_final_test_gate_now",
        "can_run_final_test_metrics_now",
        "can_use_test_for_tuning",
        "fallback_strategies_enabled",
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
        for check in report.get("guard_checks") or []
    ]


def _config_by_id(stage105_report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(config.get("config_id")): config
        for config in stage105_report.get("config_results") or []
        if isinstance(config, Mapping)
    }


def _train_guard_failure_reasons(
    configs: Sequence[Mapping[str, Any]],
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for config in configs:
        counter.update(_failed_train_guard_names(config))
    return counter


def _failed_train_guard_names(config: Mapping[str, Any]) -> list[str]:
    checks = ((config.get("train_selectability") or {}).get("checks") or {})
    return [str(name) for name, passed in checks.items() if passed is False]


def _fallback_policy_disabled(fallback_policy: Mapping[str, Any]) -> bool:
    if not fallback_policy:
        return True
    if fallback_policy.get("fallback_strategies_enabled") is False:
        return True
    status = str(fallback_policy.get("status") or "").lower()
    return "disabled" in status or "not" in status


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
