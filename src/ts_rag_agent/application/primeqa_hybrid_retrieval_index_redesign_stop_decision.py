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

_STAGE = "Stage 115"
_CREATED_AT = "2026-07-16"
_SOURCE_STAGE114 = "Stage 114"
_STOP_DECISION_ROUTE_ID = "retrieval_index_redesign_stop_decision"
_STOPPED_FAMILY_ID = "retrieval_index_redesign_candidate_family"
_STAGE113_PROTOCOL_ID = "primeqa_hybrid_retrieval_index_redesign_protocol_v1"
_STAGE114_ANALYSIS_ID = (
    "primeqa_hybrid_retrieval_index_redesign_train_cv_dev_validation_v1"
)
_STAGE114_NO_SELECTABLE_STATUS = (
    "primeqa_hybrid_retrieval_index_redesign_completed_no_train_cv_selectable_config"
)
_TRAIN_CV_SELECTION_MODE = "train_grouped_cross_validation_then_full_train_refit"
_DEV_GATE_STATUS = "report_only_no_frozen_pass_threshold"
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
class PrimeQAHybridRetrievalIndexRedesignStopVisualization:
    """One generated Stage115 retrieval/index redesign stop-decision chart."""

    name: str
    path: str


def decide_primeqa_hybrid_retrieval_index_redesign_stop(
    *,
    stage114_report_path: Path,
    user_confirmed_stop: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Stop the frozen Stage113 retrieval/index family after Stage114."""

    started_at = time.perf_counter()
    stage114_report = _load_json_object(stage114_report_path)
    loaded_at = time.perf_counter()

    stage114_summary = _stage114_summary(stage114_report)
    stage113_summary = _stage113_summary(stage114_report)
    config_stop_evidence = _config_stop_evidence(stage114_report)
    family_summary = _candidate_family_summary(stage114_report)
    train_cv_improved_blocked = _train_cv_improved_blocked_configs(stage114_report)
    dev_report_observations = _dev_report_observations(stage114_report)
    guard_checks = _guard_checks(
        stage114_summary=stage114_summary,
        stage113_summary=stage113_summary,
        stage114_report=stage114_report,
        config_stop_evidence=config_stop_evidence,
        train_cv_improved_blocked=train_cv_improved_blocked,
        user_confirmed_stop=user_confirmed_stop,
    )
    checked_at = time.perf_counter()

    stopped_family = {
        "family_id": _STOPPED_FAMILY_ID,
        "source_protocol_id": _STAGE113_PROTOCOL_ID,
        "source_analysis_id": _STAGE114_ANALYSIS_ID,
        "stage113_summary": stage113_summary,
        "stage114_summary": stage114_summary,
        "candidate_family_summary": family_summary,
        "config_stop_evidence": config_stop_evidence,
        "train_cv_improved_but_blocked_configs": train_cv_improved_blocked,
        "dev_report_observations": dev_report_observations,
        "stop_reason": (
            "Stage114 found no train-CV-selectable config in the frozen "
            "Stage113 retrieval/index redesign family. The best train-CV "
            "retrieval movement recovered only four retrieval_context_miss "
            "cases and improved recall@10 by 0.0108, but those configs "
            "violated downstream answer-quality or changed-answer guards. "
            "Dev was report-only and cannot rescue configs that failed "
            "train-CV selectability, so this family provides no "
            "runtime-defaultization or final-test gate justification."
        ),
    }
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "decision_scope": (
            "Train/dev-only stop decision for the frozen Stage113 "
            "retrieval/index redesign family after Stage114 train grouped-CV "
            "selected no config. This stage reads only the public-safe "
            "Stage114 report, does not load train/dev/test split files, does "
            "not load corpus documents, does not run retrieval or answer "
            "metrics, does not run final metrics, does not select from "
            "dev-only observations, does not add fallback strategies, and "
            "does not change runtime defaults."
        ),
        "user_confirmation": {
            "route_id": _STOP_DECISION_ROUTE_ID,
            "confirmed": bool(user_confirmed_stop),
            "confirmation_note": confirmation_note,
        },
        "source_files": {
            "stage114_report": _fingerprint(stage114_report_path),
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


def write_primeqa_hybrid_retrieval_index_redesign_stop_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridRetrievalIndexRedesignStopVisualization]:
    """Write SVG charts for Stage115 retrieval/index redesign stop decision."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage115_train_cv_retrieval_context_miss_deltas.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage115 train-CV retrieval_context_miss deltas",
                bars=_train_cv_bucket_delta_bars(report, "retrieval_context_miss_delta"),
                x_label="delta vs baseline",
                width=1540,
                margin_left=800,
            )
        ),
        "stage115_train_cv_gold_doc_recall_deltas.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage115 train-CV gold-doc recall@10 deltas",
                bars=_train_cv_recall_delta_bars(report),
                x_label="delta vs baseline",
                width=1540,
                margin_left=800,
            )
        ),
        "stage115_train_cv_changed_answer_rates.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage115 train-CV changed-answer rates",
                bars=_train_cv_changed_answer_rate_bars(report),
                x_label="changed answer rate",
                width=1540,
                margin_left=800,
            )
        ),
        "stage115_train_cv_guard_failure_reasons.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage115 train-CV guard failure reasons",
                bars=_guard_failure_reason_bars(report),
                x_label="failed config count",
                width=1480,
                margin_left=760,
            )
        ),
        "stage115_selectability_by_family.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage115 selectability by family",
                bars=_family_selectability_bars(report),
                x_label="config count",
                width=1480,
                margin_left=760,
            )
        ),
        "stage115_stop_decision_flags.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage115 stop decision flags",
                bars=_decision_flag_bars(report),
                x_label="1 means true",
                width=1360,
                margin_left=680,
            )
        ),
        "stage115_stop_guard_check_status.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage115 stop guard checks",
                bars=_guard_check_bars(report),
                x_label="1 means passed",
                width=1700,
                margin_left=880,
            )
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridRetrievalIndexRedesignStopVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage114_summary(stage114_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage114_report.get("decision") or {}
    train_selection = stage114_report.get("train_cv_selection") or {}
    dev_validation = stage114_report.get("dev_validation") or {}
    return {
        "stage": stage114_report.get("stage"),
        "analysis_id": stage114_report.get("analysis_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "selection_split": train_selection.get("selection_split"),
        "selection_mode": train_selection.get("selection_mode"),
        "selection_source": train_selection.get("selection_source"),
        "selected_config_id": train_selection.get("selected_config_id"),
        "selected_family_id": train_selection.get("selected_family_id"),
        "selectable_config_count": train_selection.get("selectable_config_count"),
        "config_count": train_selection.get("config_count")
        or len(stage114_report.get("config_results") or []),
        "baseline_train_cv_objective_score": train_selection.get(
            "baseline_train_cv_objective_score"
        ),
        "selected_train_cv_objective_delta": train_selection.get(
            "selected_train_cv_objective_delta"
        ),
        "dev_validation_status": dev_validation.get("status"),
        "dev_validation_passed": dev_validation.get("dev_validation_passed"),
        "dev_gate_status": dev_validation.get("dev_gate_status"),
        "guard_check_count": len(stage114_report.get("guard_checks") or []),
        "guard_check_passed_count": sum(
            1 for check in stage114_report.get("guard_checks") or [] if check.get("passed")
        ),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get(
            "can_run_final_test_metrics_now"
        ),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _stage113_summary(stage114_report: Mapping[str, Any]) -> dict[str, Any]:
    stage113_summary = stage114_report.get("stage113_summary") or {}
    selection_rules = stage113_summary.get("selection_rules") or {}
    dev_rules = selection_rules.get("dev_rules") or {}
    runtime_rules = selection_rules.get("runtime_rules") or {}
    test_rules = selection_rules.get("test_rules") or {}
    return {
        "stage": stage113_summary.get("stage"),
        "protocol_id": stage113_summary.get("protocol_id"),
        "protocol_status": stage113_summary.get("protocol_status"),
        "candidate_config_count": stage113_summary.get("candidate_config_count"),
        "selection_split": selection_rules.get("selection_split"),
        "selection_mode": selection_rules.get("selection_mode"),
        "minimum_train_folds": selection_rules.get("minimum_train_folds"),
        "dev_selection_allowed": dev_rules.get("dev_selection_allowed"),
        "dev_retuning_allowed": dev_rules.get("dev_retuning_allowed"),
        "dev_threshold_tuning_allowed": dev_rules.get("dev_threshold_tuning_allowed"),
        "test_access_allowed": test_rules.get("test_access_allowed"),
        "final_test_metrics_allowed": test_rules.get("final_test_metrics_allowed"),
        "test_tuning_allowed": test_rules.get("test_tuning_allowed"),
        "fallback_strategies_enabled": runtime_rules.get("fallback_strategies_enabled"),
        "default_runtime_policy": runtime_rules.get("default_runtime_policy"),
        "can_open_final_test_gate_now": stage113_summary.get(
            "can_open_final_test_gate_now"
        ),
        "can_run_final_test_metrics_now": stage113_summary.get(
            "can_run_final_test_metrics_now"
        ),
        "can_use_test_for_tuning": stage113_summary.get("can_use_test_for_tuning"),
    }


def _config_stop_evidence(stage114_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for config in stage114_report.get("config_results") or []:
        if not isinstance(config, Mapping):
            continue
        selectability = config.get("train_cv_selectability") or {}
        observed = selectability.get("observed") or {}
        rows.append(
            {
                "config_id": config.get("config_id"),
                "family_id": config.get("family_id"),
                "retrieval_mode": config.get("retrieval_mode"),
                "train_cv_objective_delta": _objective_delta(config, "train_cv"),
                "dev_objective_delta": _objective_delta(config, "dev"),
                "train_cv_retrieval_context_miss_delta": _bucket_delta(
                    config,
                    "train_cv",
                    "retrieval_context_miss",
                ),
                "dev_retrieval_context_miss_delta": _bucket_delta(
                    config,
                    "dev",
                    "retrieval_context_miss",
                ),
                "train_cv_gold_doc_recall_at_10_delta": _retrieval_delta(
                    config,
                    "train_cv",
                    "gold_doc_recall_at_10",
                ),
                "dev_gold_doc_recall_at_10_delta": _retrieval_delta(
                    config,
                    "dev",
                    "gold_doc_recall_at_10",
                ),
                "train_cv_changed_answer_rate": _changed_answer_rate(
                    config,
                    "train_cv",
                ),
                "dev_changed_answer_rate": _changed_answer_rate(config, "dev"),
                "train_cv_selectable": selectability.get("selectable") is True,
                "failed_train_cv_guards": _failed_train_cv_guard_names(config),
                "observed_train_cv_selectability": {
                    "average_token_f1_drop": observed.get(
                        "train_cv_average_token_f1_drop"
                    ),
                    "gold_doc_citation_rate_drop": observed.get(
                        "train_cv_gold_doc_citation_rate_drop"
                    ),
                    "answerable_refusal_rate_delta": observed.get(
                        "train_cv_answerable_refusal_rate_delta"
                    ),
                    "answerability_false_answer_delta": observed.get(
                        "train_cv_answerability_false_answer_delta"
                    ),
                    "evidence_selection_miss_delta": observed.get(
                        "train_cv_evidence_selection_miss_delta"
                    ),
                    "gold_span_beats_selected_delta": observed.get(
                        "train_cv_gold_span_beats_selected_delta"
                    ),
                    "changed_answer_rate": observed.get(
                        "train_cv_changed_answer_rate"
                    ),
                },
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            float(row["train_cv_objective_delta"]),
            str(row["config_id"]),
        ),
    )


def _candidate_family_summary(stage114_report: Mapping[str, Any]) -> dict[str, Any]:
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for config in stage114_report.get("config_results") or []:
        if isinstance(config, Mapping):
            by_family[str(config.get("family_id"))].append(config)
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
    best_objective = min(configs, key=lambda config: _objective_delta(config, "train_cv"))
    best_miss = min(
        configs,
        key=lambda config: _bucket_delta(config, "train_cv", "retrieval_context_miss"),
    )
    return {
        "config_count": len(configs),
        "train_cv_selectable_config_count": len(selectable_configs),
        "best_train_cv_objective_delta_config_id": best_objective.get("config_id"),
        "best_train_cv_objective_delta": _objective_delta(best_objective, "train_cv"),
        "best_train_cv_retrieval_context_miss_config_id": best_miss.get("config_id"),
        "best_train_cv_retrieval_context_miss_delta": _bucket_delta(
            best_miss,
            "train_cv",
            "retrieval_context_miss",
        ),
        "train_cv_guard_failure_reasons": dict(
            sorted(_train_cv_guard_failure_reasons(configs).items())
        ),
    }


def _train_cv_improved_blocked_configs(
    stage114_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    improved = []
    for config in stage114_report.get("config_results") or []:
        if not isinstance(config, Mapping):
            continue
        miss_delta = _bucket_delta(config, "train_cv", "retrieval_context_miss")
        recall_delta = _retrieval_delta(config, "train_cv", "gold_doc_recall_at_10")
        selectable = (config.get("train_cv_selectability") or {}).get("selectable") is True
        if miss_delta < 0 and recall_delta > 0.0 and not selectable:
            improved.append(
                {
                    "config_id": config.get("config_id"),
                    "family_id": config.get("family_id"),
                    "train_cv_retrieval_context_miss_delta": miss_delta,
                    "train_cv_gold_doc_recall_at_10_delta": recall_delta,
                    "train_cv_changed_answer_rate": _changed_answer_rate(
                        config,
                        "train_cv",
                    ),
                    "train_cv_selectable": selectable,
                    "failed_train_cv_guards": _failed_train_cv_guard_names(config),
                }
            )
    return sorted(
        improved,
        key=lambda row: (
            int(row["train_cv_retrieval_context_miss_delta"]),
            str(row["config_id"]),
        ),
    )


def _dev_report_observations(stage114_report: Mapping[str, Any]) -> dict[str, Any]:
    config_rows = []
    for config in stage114_report.get("config_results") or []:
        if not isinstance(config, Mapping):
            continue
        config_rows.append(
            {
                "config_id": config.get("config_id"),
                "family_id": config.get("family_id"),
                "dev_retrieval_context_miss_delta": _bucket_delta(
                    config,
                    "dev",
                    "retrieval_context_miss",
                ),
                "dev_gold_doc_recall_at_10_delta": _retrieval_delta(
                    config,
                    "dev",
                    "gold_doc_recall_at_10",
                ),
                "dev_average_token_f1_delta": _metric_delta(
                    config,
                    "dev",
                    "average_token_f1",
                ),
                "dev_gold_doc_citation_rate_delta": _metric_delta(
                    config,
                    "dev",
                    "gold_doc_citation_rate",
                ),
                "dev_changed_answer_rate": _changed_answer_rate(config, "dev"),
            }
        )
    best_dev_f1 = max(
        config_rows,
        key=lambda row: float(row["dev_average_token_f1_delta"]),
    )
    lowest_dev_changed = min(
        config_rows,
        key=lambda row: float(row["dev_changed_answer_rate"]),
    )
    return {
        "dev_gate_status": (
            (stage114_report.get("dev_validation") or {}).get("dev_gate_status")
        ),
        "dev_selection_used": False,
        "dev_retuning_used": False,
        "best_dev_average_token_f1_delta_config_id": best_dev_f1["config_id"],
        "best_dev_average_token_f1_delta": best_dev_f1["dev_average_token_f1_delta"],
        "lowest_dev_changed_answer_rate_config_id": lowest_dev_changed["config_id"],
        "lowest_dev_changed_answer_rate": lowest_dev_changed[
            "dev_changed_answer_rate"
        ],
        "config_observations": sorted(
            config_rows,
            key=lambda row: (float(row["dev_changed_answer_rate"]), str(row["config_id"])),
        ),
    }


def _guard_checks(
    *,
    stage114_summary: Mapping[str, Any],
    stage113_summary: Mapping[str, Any],
    stage114_report: Mapping[str, Any],
    config_stop_evidence: Sequence[Mapping[str, Any]],
    train_cv_improved_blocked: Sequence[Mapping[str, Any]],
    user_confirmed_stop: bool,
) -> list[dict[str, Any]]:
    split_contract = stage114_report.get("split_contract") or {}
    stage114_guard_checks = stage114_report.get("guard_checks") or []
    public_payload = {
        "stage114_summary": stage114_summary,
        "stage113_summary": stage113_summary,
        "config_stop_evidence": config_stop_evidence,
        "train_cv_improved_blocked": train_cv_improved_blocked,
    }
    failed_keys = sorted(_forbidden_keys_found(public_payload))
    return [
        _check(
            name="source_stage114_report_is_stage114",
            passed=stage114_summary.get("stage") == _SOURCE_STAGE114,
            observed=stage114_summary.get("stage"),
            expected=_SOURCE_STAGE114,
        ),
        _check(
            name="user_confirmed_stage115_stop_decision",
            passed=user_confirmed_stop,
            observed=user_confirmed_stop,
            expected=True,
        ),
        _check(
            name="stage114_analysis_id_matches",
            passed=stage114_summary.get("analysis_id") == _STAGE114_ANALYSIS_ID,
            observed=stage114_summary.get("analysis_id"),
            expected=_STAGE114_ANALYSIS_ID,
        ),
        _check(
            name="stage114_completed_with_no_train_cv_selectable_config",
            passed=stage114_summary.get("decision_status")
            == _STAGE114_NO_SELECTABLE_STATUS,
            observed=stage114_summary.get("decision_status"),
            expected=_STAGE114_NO_SELECTABLE_STATUS,
        ),
        _check(
            name="stage114_recommends_stop_decision",
            passed=stage114_summary.get("recommended_next_direction")
            == "record_retrieval_index_redesign_stop_decision",
            observed=stage114_summary.get("recommended_next_direction"),
            expected="record_retrieval_index_redesign_stop_decision",
        ),
        _check(
            name="stage114_all_guard_checks_passed",
            passed=all(check.get("passed") is True for check in stage114_guard_checks),
            observed={
                "passed": stage114_summary.get("guard_check_passed_count"),
                "total": stage114_summary.get("guard_check_count"),
            },
            expected="all passed",
        ),
        _check(
            name="stage114_split_contract_is_train_dev_only",
            passed=split_contract.get("development_splits") == ["train", "dev"]
            and split_contract.get("selection_split") == "train"
            and split_contract.get("validation_split") == "dev"
            and split_contract.get("forbidden_final_splits") == ["test"],
            observed=split_contract,
            expected="train/dev only with test forbidden",
        ),
        _check(
            name="stage113_protocol_id_matches",
            passed=stage113_summary.get("protocol_id") == _STAGE113_PROTOCOL_ID,
            observed=stage113_summary.get("protocol_id"),
            expected=_STAGE113_PROTOCOL_ID,
        ),
        _check(
            name="stage113_dev_test_runtime_boundaries_locked",
            passed=stage113_summary.get("dev_selection_allowed") is False
            and stage113_summary.get("dev_retuning_allowed") is False
            and stage113_summary.get("dev_threshold_tuning_allowed") is False
            and stage113_summary.get("final_test_metrics_allowed") is False
            and stage113_summary.get("test_tuning_allowed") is False
            and stage113_summary.get("fallback_strategies_enabled") is False
            and stage113_summary.get("default_runtime_policy") == "unchanged",
            observed=stage113_summary,
            expected="dev/test/runtime/fallback boundaries locked",
        ),
        _check(
            name="stage114_train_cv_selection_used_train_only",
            passed=stage114_summary.get("selection_split") == "train"
            and stage114_summary.get("selection_mode") == _TRAIN_CV_SELECTION_MODE
            and stage114_summary.get("selection_source") == "train_cv_only",
            observed=stage114_summary,
            expected="train_cv_only",
        ),
        _check(
            name="stage114_selected_no_config",
            passed=stage114_summary.get("selected_config_id") is None
            and stage114_summary.get("selectable_config_count") == 0,
            observed={
                "selected_config_id": stage114_summary.get("selected_config_id"),
                "selectable_config_count": stage114_summary.get(
                    "selectable_config_count"
                ),
            },
            expected="selected_config_id null and selectable_config_count 0",
        ),
        _check(
            name="all_stage114_configs_are_train_cv_nonselectable",
            passed=len(config_stop_evidence) == 8
            and all(row.get("train_cv_selectable") is False for row in config_stop_evidence),
            observed={
                "config_count": len(config_stop_evidence),
                "selectable_count": sum(
                    1 for row in config_stop_evidence if row.get("train_cv_selectable")
                ),
            },
            expected="8 configs, 0 selectable",
        ),
        _check(
            name="stage114_has_improved_but_blocked_train_cv_configs",
            passed=bool(train_cv_improved_blocked)
            and all(
                row.get("train_cv_selectable") is False
                for row in train_cv_improved_blocked
            ),
            observed=train_cv_improved_blocked,
            expected="train-CV improved configs exist and are nonselectable",
        ),
        _check(
            name="stage114_dev_report_has_no_selected_config",
            passed=stage114_summary.get("dev_validation_status")
            == "no_train_cv_selectable_config"
            and stage114_summary.get("dev_validation_passed") is None
            and stage114_summary.get("dev_gate_status") == _DEV_GATE_STATUS,
            observed={
                "status": stage114_summary.get("dev_validation_status"),
                "passed": stage114_summary.get("dev_validation_passed"),
                "dev_gate_status": stage114_summary.get("dev_gate_status"),
            },
            expected="no selected train-CV config; dev report-only",
        ),
        _check(
            name="stage114_final_test_gate_locked",
            passed=stage114_summary.get("can_open_final_test_gate_now") is False
            and stage114_summary.get("can_run_final_test_metrics_now") is False,
            observed={
                "can_open_final_test_gate_now": stage114_summary.get(
                    "can_open_final_test_gate_now"
                ),
                "can_run_final_test_metrics_now": stage114_summary.get(
                    "can_run_final_test_metrics_now"
                ),
            },
            expected=False,
        ),
        _check(
            name="stage114_forbids_test_tuning",
            passed=stage114_summary.get("can_use_test_for_tuning") is False,
            observed=stage114_summary.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage114_default_runtime_policy_unchanged",
            passed=stage114_summary.get("default_runtime_policy") == "unchanged",
            observed=stage114_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage114_fallback_strategies_disabled",
            passed=stage114_summary.get("fallback_strategies_enabled") is False,
            observed=stage114_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="stage115_no_new_train_dev_metrics_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage115_split_files_not_loaded",
            passed=True,
            observed="not_loaded",
            expected="not_loaded",
        ),
        _check(
            name="stage115_corpus_documents_not_loaded",
            passed=True,
            observed="not_loaded",
            expected="not_loaded",
        ),
        _check(
            name="stage115_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage115_default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
        _check(
            name="stage115_fallback_strategies_not_added",
            passed=True,
            observed=False,
            expected=False,
        ),
        _check(
            name="stage115_public_outputs_have_no_forbidden_keys",
            passed=not failed_keys,
            observed=failed_keys,
            expected=[],
        ),
    ]


def _decision(*, guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_retrieval_index_redesign_stop_decision_blocked",
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
        "status": "primeqa_hybrid_retrieval_index_redesign_family_stopped",
        "stopped_family_id": _STOPPED_FAMILY_ID,
        "stopped_protocol_id": _STAGE113_PROTOCOL_ID,
        "stopped_analysis_id": _STAGE114_ANALYSIS_ID,
        "current_route_defaultization": "blocked",
        "new_retrieval_research_required_before_any_runtime_or_test_gate": True,
        "recommended_next_direction": "user_confirmed_next_research_direction_required",
        "can_continue_train_dev_development": True,
        "requires_user_confirmation_before_next_protocol": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage116: only after user confirmation, choose the next "
            "train/dev-only research direction. Do not select from dev-only "
            "observations, keep test locked, keep runtime defaults unchanged, "
            "and do not add fallback strategies."
        ),
    }


def _objective_delta(config: Mapping[str, Any], split: str) -> float:
    return float((config.get("objective_score_deltas_by_split") or {}).get(split) or 0.0)


def _bucket_delta(config: Mapping[str, Any], split: str, bucket: str) -> int:
    return int(
        ((config.get("target_bucket_deltas_by_split") or {}).get(split) or {}).get(
            bucket
        )
        or 0
    )


def _retrieval_delta(config: Mapping[str, Any], split: str, metric: str) -> float:
    return float(
        ((config.get("retrieval_metric_deltas_by_split") or {}).get(split) or {}).get(
            metric
        )
        or 0.0
    )


def _metric_delta(config: Mapping[str, Any], split: str, metric: str) -> float:
    return float(((config.get("metric_deltas_by_split") or {}).get(split) or {}).get(metric) or 0.0)


def _changed_answer_rate(config: Mapping[str, Any], split: str) -> float:
    return float((config.get("changed_answer_rates_by_split") or {}).get(split) or 0.0)


def _failed_train_cv_guard_names(config: Mapping[str, Any]) -> list[str]:
    selectability = config.get("train_cv_selectability") or {}
    failures = selectability.get("guard_failure_reasons")
    if failures:
        return [str(name) for name in failures]
    checks = selectability.get("checks") or {}
    return [str(name) for name, passed in checks.items() if passed is False]


def _train_cv_guard_failure_reasons(
    configs: Sequence[Mapping[str, Any]],
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for config in configs:
        for reason in _failed_train_cv_guard_names(config):
            counter[reason] += 1
    return counter


def _train_cv_bucket_delta_bars(report: Mapping[str, Any], field: str) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(row["config_id"]),
            value=float(row[f"train_cv_{field}"]),
            value_label=f"{int(row[f'train_cv_{field}']):+d}",
        )
        for row in _config_stop_evidence_from_report(report)
    ]


def _train_cv_recall_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(row["config_id"]),
            value=float(row["train_cv_gold_doc_recall_at_10_delta"]),
            value_label=f"{float(row['train_cv_gold_doc_recall_at_10_delta']):+.4f}",
        )
        for row in _config_stop_evidence_from_report(report)
    ]


def _train_cv_changed_answer_rate_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(row["config_id"]),
            value=float(row["train_cv_changed_answer_rate"]),
            value_label=f"{float(row['train_cv_changed_answer_rate']):.4f}",
        )
        for row in _config_stop_evidence_from_report(report)
    ]


def _guard_failure_reason_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counter: Counter[str] = Counter()
    for row in _config_stop_evidence_from_report(report):
        for reason in row.get("failed_train_cv_guards") or []:
            counter[str(reason)] += 1
    return [
        BarDatum(label=reason, value=float(count), value_label=str(count))
        for reason, count in sorted(counter.items())
    ]


def _family_selectability_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    family_summary = (
        (report.get("stopped_family") or {}).get("candidate_family_summary") or {}
    )
    bars = []
    for family_id, summary in family_summary.items():
        selectable = int(summary.get("train_cv_selectable_config_count") or 0)
        total = int(summary.get("config_count") or 0)
        bars.append(
            BarDatum(
                label=f"{family_id} selectable",
                value=float(selectable),
                value_label=str(selectable),
            )
        )
        bars.append(
            BarDatum(
                label=f"{family_id} blocked",
                value=float(total - selectable),
                value_label=str(total - selectable),
            )
        )
    return bars


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    names = [
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
            value_label=str(decision.get(name)),
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


def _config_stop_evidence_from_report(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list((report.get("stopped_family") or {}).get("config_stop_evidence") or [])


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
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
