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

_STAGE = "Stage 110"
_CREATED_AT = "2026-07-16"
_SOURCE_STAGE109 = "Stage 109"
_STOP_DECISION_ROUTE_ID = "failure_pattern_redesign_stop_decision"
_STOPPED_FAMILY_ID = "failure_pattern_redesign_candidate_family"
_STAGE108_PROTOCOL_ID = "primeqa_hybrid_failure_pattern_redesign_protocol_v1"
_STAGE109_ANALYSIS_ID = (
    "primeqa_hybrid_failure_pattern_redesign_train_cv_dev_validation_v1"
)
_STAGE109_NO_SELECTABLE_STATUS = (
    "primeqa_hybrid_failure_pattern_redesign_completed_no_train_cv_selectable_config"
)
_TRAIN_CV_SELECTION_MODE = "train_grouped_cross_validation_then_full_train_refit"
_FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "question_text",
        "question_title",
        "raw_question_text",
        "raw_answer_text",
        "gold_answer",
        "answer_text",
        "document_id",
        "answer_doc_id",
        "retrieved_doc_ids",
        "cited_doc_ids",
        "source_doc_ids",
        "matched_token_strings",
        "query_terms",
        "document_title",
        "document_body",
        "document_text",
    }
)


@dataclass(frozen=True)
class PrimeQAHybridFailurePatternRedesignStopVisualization:
    """One generated Stage110 failure-pattern redesign stop-decision chart."""

    name: str
    path: str


def decide_primeqa_hybrid_failure_pattern_redesign_stop(
    *,
    stage109_report_path: Path,
    user_confirmed_stop: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Stop the frozen Stage108 redesign family after Stage109 selects no config."""

    started_at = time.perf_counter()
    stage109_report = _load_json_object(stage109_report_path)
    loaded_at = time.perf_counter()

    stage109_summary = _stage109_summary(stage109_report)
    stage108_summary = _stage108_summary(stage109_report)
    config_stop_evidence = _config_stop_evidence(stage109_report)
    family_summary = _candidate_family_summary(stage109_report)
    dev_improved_nonselectable = _dev_improved_nonselectable_configs(stage109_report)
    noop_blocked_configs = _noop_blocked_configs(stage109_report)
    guard_checks = _guard_checks(
        stage109_summary=stage109_summary,
        stage108_summary=stage108_summary,
        stage109_report=stage109_report,
        config_stop_evidence=config_stop_evidence,
        dev_improved_nonselectable=dev_improved_nonselectable,
        noop_blocked_configs=noop_blocked_configs,
        user_confirmed_stop=user_confirmed_stop,
    )
    checked_at = time.perf_counter()

    stopped_family = {
        "family_id": _STOPPED_FAMILY_ID,
        "source_protocol_id": _STAGE108_PROTOCOL_ID,
        "source_analysis_id": _STAGE109_ANALYSIS_ID,
        "stage108_summary": stage108_summary,
        "stage109_summary": stage109_summary,
        "candidate_family_summary": family_summary,
        "config_stop_evidence": config_stop_evidence,
        "dev_improved_train_cv_nonselectable_configs": dev_improved_nonselectable,
        "noop_blocked_configs": noop_blocked_configs,
        "stop_reason": (
            "Stage109 found no train-CV-selectable config in the frozen "
            "Stage108 failure-pattern redesign family. Most configs reduced "
            "the weighted target score but violated the frozen train-CV "
            "answerable-refusal guard; the only no-op config was blocked by "
            "the negative train-CV delta rule. Dev cannot rescue configs that "
            "failed train-CV selectability, so this family provides no "
            "runtime-defaultization or final-test gate justification."
        ),
    }
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "decision_scope": (
            "Train/dev-only stop decision for the frozen Stage108 "
            "failure-pattern redesign family after Stage109 train grouped-CV "
            "selected no config. This stage reads only the public-safe "
            "Stage109 report, does not load train/dev/test split files, does "
            "not load corpus documents, does not run retrieval or answer "
            "metrics, does not run final metrics, does not select from dev-only "
            "observations, does not add fallback strategies, and does not "
            "change runtime defaults."
        ),
        "user_confirmation": {
            "route_id": _STOP_DECISION_ROUTE_ID,
            "confirmed": bool(user_confirmed_stop),
            "confirmation_note": confirmation_note,
        },
        "source_files": {
            "stage109_report": _fingerprint(stage109_report_path),
        },
        "stopped_family": stopped_family,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks=guard_checks),
        "timing_seconds": {
            "load_reports": round(loaded_at - started_at, 3),
            "decision_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {
        **report,
        "public_safe_contract": {
            "forbidden_keys_found": sorted(_forbidden_keys_found(report)),
            "raw_question_answer_or_document_text_written": False,
            "test_split_loaded": False,
            "final_test_metrics_run": False,
        },
    }


def write_primeqa_hybrid_failure_pattern_redesign_stop_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridFailurePatternRedesignStopVisualization]:
    """Write SVG charts for Stage110 failure-pattern redesign stop decision."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage110_train_cv_weighted_target_deltas.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage110 train-CV target deltas",
                bars=_train_cv_delta_bars(report),
                x_label="weighted target delta",
                width=1480,
                margin_left=760,
            )
        ),
        "stage110_dev_weighted_target_deltas.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage110 dev target deltas",
                bars=_dev_delta_bars(report),
                x_label="weighted target delta",
                width=1480,
                margin_left=760,
            )
        ),
        "stage110_answerable_refusal_deltas.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage110 answerable refusal deltas",
                bars=_answerable_refusal_delta_bars(report),
                x_label="train-CV answerable refusal delta",
                width=1480,
                margin_left=760,
            )
        ),
        "stage110_selectability_by_family.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage110 train-CV selectability by family",
                bars=_family_selectability_bars(report),
                x_label="config count",
                width=1440,
                margin_left=740,
            )
        ),
        "stage110_train_cv_guard_failure_reasons.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage110 train-CV guard failure reasons",
                bars=_guard_failure_reason_bars(report),
                x_label="failed config count",
                width=1440,
                margin_left=760,
            )
        ),
        "stage110_stop_decision_flags.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage110 stop decision flags",
                bars=_decision_flag_bars(report),
                x_label="1 means true",
                width=1320,
                margin_left=660,
            )
        ),
        "stage110_stop_guard_check_status.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage110 stop guard checks",
                bars=_guard_check_bars(report),
                x_label="1 means passed",
                width=1640,
                margin_left=860,
            )
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridFailurePatternRedesignStopVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage109_summary(stage109_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage109_report.get("decision") or {}
    train_selection = stage109_report.get("train_cv_selection") or {}
    dev_validation = stage109_report.get("dev_validation") or {}
    return {
        "stage": stage109_report.get("stage"),
        "analysis_id": stage109_report.get("analysis_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "selection_split": train_selection.get("selection_split"),
        "selection_mode": train_selection.get("selection_mode"),
        "selected_config_id": train_selection.get("selected_config_id"),
        "selected_candidate_family_id": train_selection.get(
            "selected_candidate_family_id"
        ),
        "selectable_config_count": train_selection.get("selectable_config_count"),
        "config_count": train_selection.get("config_count")
        or len(stage109_report.get("config_results") or []),
        "baseline_train_cv_weighted_target_score": train_selection.get(
            "baseline_train_cv_weighted_target_score"
        ),
        "selected_train_cv_weighted_target_delta": train_selection.get(
            "selected_train_cv_weighted_target_delta"
        ),
        "dev_validation_status": dev_validation.get("status"),
        "dev_validation_passed": dev_validation.get("dev_validation_passed"),
        "dev_weighted_target_delta": dev_validation.get("dev_weighted_target_delta"),
        "guard_check_count": len(stage109_report.get("guard_checks") or []),
        "guard_check_passed_count": sum(
            1 for check in stage109_report.get("guard_checks") or [] if check.get("passed")
        ),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get(
            "can_run_final_test_metrics_now"
        ),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _stage108_summary(stage109_report: Mapping[str, Any]) -> dict[str, Any]:
    stage108_summary = stage109_report.get("stage108_summary") or {}
    train_rule = stage108_summary.get("train_selection_rule") or {}
    dev_rule = stage108_summary.get("dev_validation_rule") or {}
    return {
        "stage": stage108_summary.get("stage"),
        "protocol_id": stage108_summary.get("protocol_id"),
        "decision_status": stage108_summary.get("decision_status"),
        "candidate_config_count": stage108_summary.get("candidate_config_count"),
        "selection_split": train_rule.get("selection_split"),
        "selection_mode": train_rule.get("selection_mode"),
        "train_cv_fold_count": train_rule.get("train_cv_fold_count"),
        "no_op_candidate_selectable": (
            (train_rule.get("objective") or {}).get("no_op_candidate_selectable")
        ),
        "requires_negative_train_cv_weighted_delta": (
            (train_rule.get("objective") or {}).get(
                "requires_negative_train_cv_weighted_delta"
            )
        ),
        "selectability_guards": train_rule.get("selectability_guards") or {},
        "dev_selection_allowed": dev_rule.get("dev_selection_allowed"),
        "dev_retuning_allowed": dev_rule.get("dev_retuning_allowed"),
        "dev_threshold_tuning_allowed": dev_rule.get("dev_threshold_tuning_allowed"),
        "test_access_allowed_in_dev_validation": dev_rule.get("test_access_allowed"),
        "can_open_final_test_gate_now": stage108_summary.get(
            "can_open_final_test_gate_now"
        ),
        "can_run_final_test_metrics_now": stage108_summary.get(
            "can_run_final_test_metrics_now"
        ),
        "can_use_test_for_tuning": stage108_summary.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": stage108_summary.get(
            "fallback_strategies_enabled"
        ),
        "default_runtime_policy": stage108_summary.get("default_runtime_policy"),
    }


def _config_stop_evidence(stage109_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for config in stage109_report.get("config_results") or []:
        if not isinstance(config, Mapping):
            continue
        selectability = config.get("train_cv_selectability") or {}
        observed = selectability.get("observed") or {}
        rows.append(
            {
                "config_id": config.get("config_id"),
                "candidate_family_id": config.get("candidate_family_id"),
                "component_family": config.get("component_family"),
                "train_cv_weighted_target_delta": _split_delta(config, "train_cv"),
                "dev_weighted_target_delta": _split_delta(config, "dev"),
                "train_cv_changed_answer_count": (
                    (config.get("changed_answer_counts_by_split") or {}).get("train_cv")
                ),
                "train_cv_selectable": selectability.get("selectable") is True,
                "failed_train_cv_guards": _failed_train_cv_guard_names(config),
                "observed_train_cv_selectability": {
                    "answerable_refusal_rate_delta": observed.get(
                        "answerable_refusal_rate_delta"
                    ),
                    "average_token_f1_drop": observed.get("average_token_f1_drop"),
                    "gold_doc_citation_rate_drop": observed.get(
                        "gold_doc_citation_rate_drop"
                    ),
                    "retrieval_context_miss_delta": observed.get(
                        "retrieval_context_miss_delta"
                    ),
                },
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            float(row["train_cv_weighted_target_delta"]),
            str(row["config_id"]),
        ),
    )


def _candidate_family_summary(stage109_report: Mapping[str, Any]) -> dict[str, Any]:
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for config in stage109_report.get("config_results") or []:
        if isinstance(config, Mapping):
            by_family[str(config.get("candidate_family_id"))].append(config)
    return {
        family_id: _family_summary(configs)
        for family_id, configs in sorted(by_family.items())
    }


def _family_summary(configs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selectable_configs = [
        config
        for config in configs
        if (config.get("train_cv_selectability") or {}).get("selectable") is True
    ]
    best_train = min(configs, key=lambda config: _split_delta(config, "train_cv"))
    best_dev = min(configs, key=lambda config: _split_delta(config, "dev"))
    answerable_refusal_deltas = [
        float(
            ((config.get("train_cv_selectability") or {}).get("observed") or {}).get(
                "answerable_refusal_rate_delta"
            )
            or 0.0
        )
        for config in configs
    ]
    return {
        "config_count": len(configs),
        "train_cv_selectable_config_count": len(selectable_configs),
        "best_train_cv_delta_config_id": best_train.get("config_id"),
        "best_train_cv_weighted_target_delta": _split_delta(best_train, "train_cv"),
        "best_dev_delta_config_id": best_dev.get("config_id"),
        "best_dev_weighted_target_delta": _split_delta(best_dev, "dev"),
        "max_answerable_refusal_rate_delta": max(answerable_refusal_deltas),
        "train_cv_guard_failure_reasons": dict(
            sorted(_train_cv_guard_failure_reasons(configs).items())
        ),
    }


def _dev_improved_nonselectable_configs(
    stage109_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    improved = []
    for config in stage109_report.get("config_results") or []:
        if not isinstance(config, Mapping):
            continue
        dev_delta = _split_delta(config, "dev")
        selectable = (config.get("train_cv_selectability") or {}).get("selectable") is True
        if dev_delta < 0.0 and not selectable:
            improved.append(
                {
                    "config_id": config.get("config_id"),
                    "candidate_family_id": config.get("candidate_family_id"),
                    "dev_weighted_target_delta": dev_delta,
                    "train_cv_weighted_target_delta": _split_delta(config, "train_cv"),
                    "train_cv_selectable": selectable,
                    "failed_train_cv_guards": _failed_train_cv_guard_names(config),
                }
            )
    return sorted(
        improved,
        key=lambda row: (float(row["dev_weighted_target_delta"]), str(row["config_id"])),
    )


def _noop_blocked_configs(stage109_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    blocked = []
    for config in stage109_report.get("config_results") or []:
        if not isinstance(config, Mapping):
            continue
        train_delta = _split_delta(config, "train_cv")
        changed_count = int(
            (config.get("changed_answer_counts_by_split") or {}).get("train_cv") or 0
        )
        failed_guards = _failed_train_cv_guard_names(config)
        if train_delta == 0.0 and changed_count == 0:
            blocked.append(
                {
                    "config_id": config.get("config_id"),
                    "candidate_family_id": config.get("candidate_family_id"),
                    "train_cv_weighted_target_delta": train_delta,
                    "train_cv_changed_answer_count": changed_count,
                    "failed_train_cv_guards": failed_guards,
                }
            )
    return blocked


def _guard_checks(
    *,
    stage109_summary: Mapping[str, Any],
    stage108_summary: Mapping[str, Any],
    stage109_report: Mapping[str, Any],
    config_stop_evidence: Sequence[Mapping[str, Any]],
    dev_improved_nonselectable: Sequence[Mapping[str, Any]],
    noop_blocked_configs: Sequence[Mapping[str, Any]],
    user_confirmed_stop: bool,
) -> list[dict[str, Any]]:
    stage109_guard_checks = stage109_report.get("guard_checks") or []
    split_contract = stage109_report.get("split_contract") or {}
    failed_keys = sorted(
        _forbidden_keys_found(
            {
                "stage109_summary": stage109_summary,
                "stage108_summary": stage108_summary,
                "config_stop_evidence": config_stop_evidence,
                "dev_improved_nonselectable": dev_improved_nonselectable,
                "noop_blocked_configs": noop_blocked_configs,
            }
        )
    )
    return [
        _check(
            name="source_stage109_report_is_stage109",
            passed=stage109_summary.get("stage") == _SOURCE_STAGE109,
            observed=stage109_summary.get("stage"),
            expected=_SOURCE_STAGE109,
        ),
        _check(
            name="user_confirmed_stage110_stop_decision",
            passed=user_confirmed_stop,
            observed=user_confirmed_stop,
            expected=True,
        ),
        _check(
            name="stage109_analysis_id_matches",
            passed=stage109_summary.get("analysis_id") == _STAGE109_ANALYSIS_ID,
            observed=stage109_summary.get("analysis_id"),
            expected=_STAGE109_ANALYSIS_ID,
        ),
        _check(
            name="stage109_completed_with_no_train_cv_selectable_config",
            passed=stage109_summary.get("decision_status")
            == _STAGE109_NO_SELECTABLE_STATUS,
            observed=stage109_summary.get("decision_status"),
            expected=_STAGE109_NO_SELECTABLE_STATUS,
        ),
        _check(
            name="stage109_recommends_stop_decision",
            passed=stage109_summary.get("recommended_next_direction")
            == "record_failure_pattern_redesign_stop_decision",
            observed=stage109_summary.get("recommended_next_direction"),
            expected="record_failure_pattern_redesign_stop_decision",
        ),
        _check(
            name="stage109_all_guard_checks_passed",
            passed=all(check.get("passed") is True for check in stage109_guard_checks),
            observed={
                "passed": stage109_summary.get("guard_check_passed_count"),
                "total": stage109_summary.get("guard_check_count"),
            },
            expected="all passed",
        ),
        _check(
            name="stage109_split_contract_is_train_dev_only",
            passed=split_contract.get("development_splits") == ["train", "dev"]
            and split_contract.get("selection_split") == "train"
            and split_contract.get("validation_split") == "dev"
            and split_contract.get("forbidden_final_splits") == ["test"],
            observed=split_contract,
            expected="train/dev only with test forbidden",
        ),
        _check(
            name="stage108_protocol_id_matches",
            passed=stage108_summary.get("protocol_id") == _STAGE108_PROTOCOL_ID,
            observed=stage108_summary.get("protocol_id"),
            expected=_STAGE108_PROTOCOL_ID,
        ),
        _check(
            name="stage108_train_cv_selection_rule_frozen",
            passed=stage108_summary.get("selection_split") == "train"
            and stage108_summary.get("selection_mode") == _TRAIN_CV_SELECTION_MODE
            and stage108_summary.get("requires_negative_train_cv_weighted_delta")
            is True
            and stage108_summary.get("no_op_candidate_selectable") is False,
            observed=stage108_summary,
            expected="train grouped-CV with negative delta required and no-op blocked",
        ),
        _check(
            name="stage108_dev_selection_forbidden",
            passed=stage108_summary.get("dev_selection_allowed") is False
            and stage108_summary.get("dev_retuning_allowed") is False
            and stage108_summary.get("dev_threshold_tuning_allowed") is False
            and stage108_summary.get("test_access_allowed_in_dev_validation") is False,
            observed=stage108_summary,
            expected="dev selection, retuning, threshold tuning, and test access false",
        ),
        _check(
            name="stage109_train_cv_selection_used_train_only",
            passed=stage109_summary.get("selection_split") == "train"
            and stage109_summary.get("selection_mode") == _TRAIN_CV_SELECTION_MODE,
            observed=stage109_summary,
            expected=_TRAIN_CV_SELECTION_MODE,
        ),
        _check(
            name="stage109_selected_no_config",
            passed=stage109_summary.get("selected_config_id") is None
            and stage109_summary.get("selectable_config_count") == 0,
            observed={
                "selected_config_id": stage109_summary.get("selected_config_id"),
                "selectable_config_count": stage109_summary.get(
                    "selectable_config_count"
                ),
            },
            expected="selected_config_id null and selectable_config_count 0",
        ),
        _check(
            name="all_stage109_configs_are_train_cv_nonselectable",
            passed=len(config_stop_evidence) == 7
            and all(row.get("train_cv_selectable") is False for row in config_stop_evidence),
            observed={
                "config_count": len(config_stop_evidence),
                "selectable_count": sum(
                    1 for row in config_stop_evidence if row.get("train_cv_selectable")
                ),
            },
            expected="7 configs, 0 selectable",
        ),
        _check(
            name="stage109_negative_deltas_are_blocked_not_selected",
            passed=bool(dev_improved_nonselectable)
            and all(
                row.get("train_cv_selectable") is False
                for row in dev_improved_nonselectable
            ),
            observed=dev_improved_nonselectable,
            expected="dev-improved configs exist and are train-CV nonselectable",
        ),
        _check(
            name="stage109_noop_config_blocked",
            passed=bool(noop_blocked_configs)
            and all(
                "train_cv_weighted_target_delta_negative"
                in (row.get("failed_train_cv_guards") or [])
                for row in noop_blocked_configs
            ),
            observed=noop_blocked_configs,
            expected="at least one no-op blocked by negative train-CV delta guard",
        ),
        _check(
            name="stage109_dev_validation_has_no_selected_config",
            passed=stage109_summary.get("dev_validation_status")
            == "no_train_cv_selectable_config"
            and stage109_summary.get("dev_validation_passed") is False,
            observed={
                "status": stage109_summary.get("dev_validation_status"),
                "passed": stage109_summary.get("dev_validation_passed"),
            },
            expected="no selected train-CV config to validate on dev",
        ),
        _check(
            name="stage109_final_test_gate_locked",
            passed=stage109_summary.get("can_open_final_test_gate_now") is False
            and stage109_summary.get("can_run_final_test_metrics_now") is False,
            observed={
                "can_open_final_test_gate_now": stage109_summary.get(
                    "can_open_final_test_gate_now"
                ),
                "can_run_final_test_metrics_now": stage109_summary.get(
                    "can_run_final_test_metrics_now"
                ),
            },
            expected=False,
        ),
        _check(
            name="stage109_forbids_test_tuning",
            passed=stage109_summary.get("can_use_test_for_tuning") is False,
            observed=stage109_summary.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage109_default_runtime_policy_unchanged",
            passed=stage109_summary.get("default_runtime_policy") == "unchanged",
            observed=stage109_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage109_fallback_strategies_disabled",
            passed=stage109_summary.get("fallback_strategies_enabled") is False,
            observed=stage109_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="stage110_no_new_train_dev_metrics_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage110_split_files_not_loaded",
            passed=True,
            observed="not_loaded",
            expected="not_loaded",
        ),
        _check(
            name="stage110_corpus_documents_not_loaded",
            passed=True,
            observed="not_loaded",
            expected="not_loaded",
        ),
        _check(
            name="stage110_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage110_default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
        _check(
            name="stage110_fallback_strategies_not_added",
            passed=True,
            observed=False,
            expected=False,
        ),
        _check(
            name="stage110_public_outputs_have_no_forbidden_keys",
            passed=not failed_keys,
            observed=failed_keys,
            expected=[],
        ),
    ]


def _decision(*, guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_failure_pattern_redesign_stop_decision_blocked",
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
        "status": "primeqa_hybrid_failure_pattern_redesign_family_stopped",
        "stopped_family_id": _STOPPED_FAMILY_ID,
        "stopped_protocol_id": _STAGE108_PROTOCOL_ID,
        "stopped_analysis_id": _STAGE109_ANALYSIS_ID,
        "current_route_defaultization": "blocked",
        "redesign_required_before_any_runtime_or_test_gate": True,
        "recommended_next_direction": "user_confirmed_next_research_direction_required",
        "can_continue_train_dev_development": True,
        "requires_user_confirmation_before_next_protocol": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage111: only after user confirmation, choose the next "
            "train/dev-only research direction. Do not select from dev-only "
            "observations, keep test locked, keep runtime defaults unchanged, "
            "and do not add fallback strategies."
        ),
    }


def _train_cv_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(row["config_id"]),
            value=float(row["train_cv_weighted_target_delta"]),
            value_label=f"{float(row['train_cv_weighted_target_delta']):+.2f}",
        )
        for row in (report.get("stopped_family") or {}).get("config_stop_evidence") or []
    ]


def _dev_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(row["config_id"]),
            value=float(row["dev_weighted_target_delta"]),
            value_label=f"{float(row['dev_weighted_target_delta']):+.2f}",
        )
        for row in (report.get("stopped_family") or {}).get("config_stop_evidence") or []
    ]


def _answerable_refusal_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    rows = (report.get("stopped_family") or {}).get("config_stop_evidence") or []
    bars = []
    for row in rows:
        observed = row.get("observed_train_cv_selectability") or {}
        value = float(observed.get("answerable_refusal_rate_delta") or 0.0)
        bars.append(
            BarDatum(
                label=str(row["config_id"]),
                value=value,
                value_label=f"{value:+.4f}",
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
                value=float(summary.get("train_cv_selectable_config_count") or 0),
                value_label=str(summary.get("train_cv_selectable_config_count") or 0),
            )
        )
        bars.append(
            BarDatum(
                label=f"{family_id} blocked",
                value=float(
                    int(summary.get("config_count") or 0)
                    - int(summary.get("train_cv_selectable_config_count") or 0)
                ),
                value_label=str(
                    int(summary.get("config_count") or 0)
                    - int(summary.get("train_cv_selectable_config_count") or 0)
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
        counter.update(summary.get("train_cv_guard_failure_reasons") or {})
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


def _train_cv_guard_failure_reasons(
    configs: Sequence[Mapping[str, Any]],
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for config in configs:
        counter.update(_failed_train_cv_guard_names(config))
    return counter


def _failed_train_cv_guard_names(config: Mapping[str, Any]) -> list[str]:
    checks = ((config.get("train_cv_selectability") or {}).get("checks") or {})
    return [str(name) for name, passed in checks.items() if passed is False]


def _split_delta(config: Mapping[str, Any], split: str) -> float:
    return float(
        (config.get("weighted_target_score_deltas_by_split") or {}).get(split) or 0.0
    )


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
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
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


def _forbidden_keys_found(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in _FORBIDDEN_REPORT_KEYS:
                found.add(str(key))
            found.update(_forbidden_keys_found(nested))
    elif isinstance(value, list | tuple):
        for nested in value:
            found.update(_forbidden_keys_found(nested))
    return found
