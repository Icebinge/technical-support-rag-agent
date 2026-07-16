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

_STAGE = "Stage 119"
_CREATED_AT = "2026-07-16"
_SOURCE_STAGE118 = "Stage 118"
_STOP_DECISION_ROUTE_ID = "second_stage_reranking_stop_decision"
_STOPPED_FAMILY_ID = "second_stage_reranking_candidate_family"
_STAGE117_PROTOCOL_ID = "primeqa_hybrid_second_stage_reranking_protocol_v1"
_STAGE118_ANALYSIS_ID = "primeqa_hybrid_second_stage_reranking_train_cv_dev_validation_v1"
_STAGE118_NO_SELECTABLE_STATUS = (
    "primeqa_hybrid_second_stage_reranking_completed_no_train_cv_selectable_config"
)
_TRAIN_CV_SELECTION_MODE = "train_grouped_cross_validation_then_full_train_refit"
_FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "answer",
        "answer_doc_id",
        "candidate_doc_ids",
        "cited_doc_ids",
        "document_id",
        "document_text",
        "document_title",
        "gold_answer",
        "matched_token_strings",
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
class PrimeQAHybridSecondStageRerankingStopVisualization:
    """One generated Stage119 second-stage reranking stop-decision chart."""

    name: str
    path: str


def decide_primeqa_hybrid_second_stage_reranking_stop(
    *,
    stage118_report_path: Path,
    user_confirmed_stop: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Stop the frozen Stage117 second-stage reranking family after Stage118."""

    started_at = time.perf_counter()
    stage118_report = _load_json_object(stage118_report_path)
    loaded_at = time.perf_counter()

    stage118_summary = _stage118_summary(stage118_report)
    stage117_summary = _stage117_summary(stage118_report)
    config_stop_evidence = _config_stop_evidence(stage118_report)
    family_summary = _candidate_family_summary(stage118_report)
    positive_signal_blocked = _positive_signal_blocked_configs(config_stop_evidence)
    dev_report_observations = _dev_report_observations(
        stage118_report=stage118_report,
        config_stop_evidence=config_stop_evidence,
    )
    guard_checks = _guard_checks(
        stage118_summary=stage118_summary,
        stage117_summary=stage117_summary,
        stage118_report=stage118_report,
        config_stop_evidence=config_stop_evidence,
        positive_signal_blocked=positive_signal_blocked,
        user_confirmed_stop=user_confirmed_stop,
    )
    checked_at = time.perf_counter()

    stopped_family = {
        "family_id": _STOPPED_FAMILY_ID,
        "source_protocol_id": _STAGE117_PROTOCOL_ID,
        "source_analysis_id": _STAGE118_ANALYSIS_ID,
        "stage117_summary": stage117_summary,
        "stage118_summary": stage118_summary,
        "candidate_family_summary": family_summary,
        "config_stop_evidence": config_stop_evidence,
        "train_cv_positive_signal_but_blocked_configs": positive_signal_blocked,
        "dev_report_observations": dev_report_observations,
        "stop_reason": (
            "Stage118 found no train-CV-selectable config in the frozen "
            "Stage117 second-stage reranking family. The top200 candidate "
            "pool was reproduced, and all configs preserved hit@200, but "
            "the best reranking signals moved too many already-good top10 "
            "or top20 cases downward. Dev remained report-only and cannot "
            "rescue configs that failed train-CV selectability, so this "
            "family provides no runtime-defaultization or final-test gate "
            "justification."
        ),
    }
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "decision_scope": (
            "Train/dev-only stop decision for the frozen Stage117 "
            "second-stage reranking family after Stage118 train grouped-CV "
            "selected no config. This stage reads only the public-safe "
            "Stage118 report, does not load train/dev/test split files, "
            "does not load corpus documents, does not rebuild candidate "
            "rows, does not run retrieval, reranking, answer, or final "
            "metrics, does not select from dev-only observations, does not "
            "add fallback strategies, and does not change runtime defaults."
        ),
        "user_confirmation": {
            "route_id": _STOP_DECISION_ROUTE_ID,
            "confirmed": bool(user_confirmed_stop),
            "confirmation_note": confirmation_note,
        },
        "source_files": {
            "stage118_report": _fingerprint(stage118_report_path),
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
            "raw_candidate_rows_written": False,
            "test_split_loaded": False,
            "final_test_metrics_run": False,
        },
    }


def write_primeqa_hybrid_second_stage_reranking_stop_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridSecondStageRerankingStopVisualization]:
    """Write SVG charts for Stage119 second-stage reranking stop decision."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage119_train_cv_objective_scores.svg": render_horizontal_bar_chart_svg(
            title="Stage119 train-CV objective scores",
            bars=_train_cv_objective_score_bars(report),
            x_label="objective score",
            width=1540,
            margin_left=760,
        ),
        "stage119_train_cv_mrr_at_20_deltas.svg": render_horizontal_bar_chart_svg(
            title="Stage119 train-CV MRR@20 deltas",
            bars=_train_cv_metric_delta_bars(report, "train_cv_mrr_at_20_delta"),
            x_label="delta vs Stage116 order",
            width=1540,
            margin_left=760,
        ),
        "stage119_train_cv_hit_at_10_deltas.svg": render_horizontal_bar_chart_svg(
            title="Stage119 train-CV hit@10 deltas",
            bars=_train_cv_metric_delta_bars(report, "train_cv_hit_at_10_delta"),
            x_label="delta vs Stage116 order",
            width=1540,
            margin_left=760,
        ),
        "stage119_train_cv_guard_failure_reasons.svg": render_horizontal_bar_chart_svg(
            title="Stage119 train-CV guard failure reasons",
            bars=_guard_failure_reason_bars(report),
            x_label="failed config count",
            width=1540,
            margin_left=820,
        ),
        "stage119_selectability_by_family.svg": render_horizontal_bar_chart_svg(
            title="Stage119 train-CV selectability by family",
            bars=_family_selectability_bars(report),
            x_label="config count",
            width=1480,
            margin_left=780,
        ),
        "stage119_stop_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage119 stop decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1360,
            margin_left=700,
        ),
        "stage119_stop_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage119 stop guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1740,
            margin_left=940,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridSecondStageRerankingStopVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage118_summary(stage118_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage118_report.get("decision") or {}
    split_contract = stage118_report.get("split_contract") or {}
    train_selection = stage118_report.get("train_cv_selection") or {}
    dev_validation = stage118_report.get("dev_validation") or {}
    candidate_pool = stage118_report.get("candidate_pool_summary") or {}
    train_pool = candidate_pool.get("train") or {}
    dev_pool = candidate_pool.get("dev") or {}
    guard_checks = stage118_report.get("guard_checks") or []
    return {
        "stage": stage118_report.get("stage"),
        "analysis_id": stage118_report.get("analysis_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "selection_split": train_selection.get("selection_split")
        or split_contract.get("selection_split"),
        "selection_source": train_selection.get("selection_source"),
        "selection_mode": train_selection.get("selection_mode")
        or split_contract.get("selection_mode"),
        "selected_config_id": train_selection.get("selected_config_id"),
        "selected_family_id": train_selection.get("selected_family_id"),
        "selectable_config_count": train_selection.get("selectable_config_count"),
        "config_count": train_selection.get("config_count")
        or len(stage118_report.get("config_results") or []),
        "dev_validation_status": dev_validation.get("status"),
        "dev_used_for_selection": dev_validation.get("dev_used_for_selection"),
        "dev_used_for_retuning": dev_validation.get("dev_used_for_retuning"),
        "train_top200_gold_present_rate": train_pool.get("gold_present_in_top200_rate"),
        "dev_top200_gold_present_rate": dev_pool.get("gold_present_in_top200_rate"),
        "train_candidate_record_count_in_memory": train_pool.get(
            "candidate_record_count_in_memory"
        ),
        "dev_candidate_record_count_in_memory": dev_pool.get(
            "candidate_record_count_in_memory"
        ),
        "raw_candidate_rows_written": train_pool.get("raw_candidate_rows_written")
        or dev_pool.get("raw_candidate_rows_written"),
        "guard_check_count": len(guard_checks),
        "guard_check_passed_count": sum(1 for check in guard_checks if check.get("passed")),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get(
            "can_run_final_test_metrics_now"
        ),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _stage117_summary(stage118_report: Mapping[str, Any]) -> dict[str, Any]:
    source = stage118_report.get("stage117_summary") or {}
    selection_rules = source.get("selection_rules") or {}
    dev_rules = selection_rules.get("dev_rules") or {}
    test_rules = selection_rules.get("test_rules") or {}
    runtime_rules = selection_rules.get("runtime_rules") or {}
    return {
        "stage": source.get("stage"),
        "protocol_id": source.get("protocol_id"),
        "decision_status": source.get("decision_status"),
        "candidate_pool_depth": source.get("candidate_pool_depth"),
        "candidate_config_count": source.get("candidate_config_count"),
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
        "can_open_final_test_gate_now": source.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": source.get(
            "can_run_final_test_metrics_now"
        ),
        "can_use_test_for_tuning": source.get("can_use_test_for_tuning"),
    }


def _config_stop_evidence(stage118_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for config in stage118_report.get("config_results") or []:
        if not isinstance(config, Mapping):
            continue
        train_comparison = (config.get("comparisons_to_baseline") or {}).get("train_cv") or {}
        dev_comparison = (config.get("comparisons_to_baseline") or {}).get("dev") or {}
        rows.append(
            {
                "config_id": config.get("config_id"),
                "family_id": config.get("family_id"),
                "ranking_method": config.get("ranking_method"),
                "training_status": config.get("training_status"),
                "training_error": config.get("training_error"),
                "train_cv_objective_score": _float(config.get("train_cv_objective_score")),
                "train_cv_selectable": config.get("train_cv_selectable") is True,
                "train_cv_mrr_at_20_delta": _float(train_comparison.get("mrr_at_20_delta")),
                "train_cv_hit_at_10_delta": _float(train_comparison.get("hit@10_delta")),
                "train_cv_hit_at_20_delta": _float(train_comparison.get("hit@20_delta")),
                "train_cv_hit_at_200_delta": _float(
                    train_comparison.get("hit@200_delta")
                ),
                "train_cv_hit_at_200_count_delta": _int(
                    train_comparison.get("hit@200_count_delta")
                ),
                "train_cv_missing_count_at_200_delta": _int(
                    train_comparison.get("missing_count_at_200_delta")
                ),
                "train_cv_bm25_top10_gold_demotions_to_below_50": _guard_observed(
                    config,
                    "train_cv_bm25_top10_gold_demotions_to_below_50_within_guard",
                ),
                "train_cv_hit_at_20_regression_rate": _guard_observed(
                    config,
                    "train_cv_hit_at_20_regression_rate_within_guard",
                ),
                "train_cv_top10_regression_count": _guard_observed(
                    config,
                    "train_cv_top10_regression_count_within_guard",
                ),
                "failed_train_cv_guards": _failed_train_cv_guard_names(config),
                "dev_mrr_at_20_delta_report_only": _float(
                    dev_comparison.get("mrr_at_20_delta")
                ),
                "dev_hit_at_10_delta_report_only": _float(dev_comparison.get("hit@10_delta")),
                "dev_hit_at_20_delta_report_only": _float(dev_comparison.get("hit@20_delta")),
                "dev_hit_at_200_delta_report_only": _float(
                    dev_comparison.get("hit@200_delta")
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -float(row["train_cv_objective_score"]),
            str(row["config_id"]),
        ),
    )


def _candidate_family_summary(stage118_report: Mapping[str, Any]) -> dict[str, Any]:
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for config in stage118_report.get("config_results") or []:
        if isinstance(config, Mapping):
            by_family[str(config.get("family_id"))].append(config)
    return {
        family_id: _family_summary(configs)
        for family_id, configs in sorted(by_family.items())
    }


def _family_summary(configs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selectable_configs = [
        config for config in configs if config.get("train_cv_selectable") is True
    ]
    best_objective = max(
        configs,
        key=lambda config: _float(config.get("train_cv_objective_score")),
    )
    best_mrr = max(
        configs,
        key=lambda config: _train_cv_delta(config, "mrr_at_20_delta"),
    )
    return {
        "config_count": len(configs),
        "train_cv_selectable_config_count": len(selectable_configs),
        "best_train_cv_objective_config_id": best_objective.get("config_id"),
        "best_train_cv_objective_score": _float(
            best_objective.get("train_cv_objective_score")
        ),
        "best_train_cv_mrr_at_20_config_id": best_mrr.get("config_id"),
        "best_train_cv_mrr_at_20_delta": _train_cv_delta(best_mrr, "mrr_at_20_delta"),
        "train_cv_guard_failure_reasons": dict(
            sorted(_train_cv_guard_failure_reasons(configs).items())
        ),
    }


def _positive_signal_blocked_configs(
    config_stop_evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for row in config_stop_evidence:
        has_positive_signal = (
            _float(row.get("train_cv_mrr_at_20_delta")) > 0.0
            or _float(row.get("train_cv_hit_at_10_delta")) > 0.0
            or _float(row.get("train_cv_hit_at_20_delta")) > 0.0
        )
        if has_positive_signal and row.get("train_cv_selectable") is False:
            rows.append(
                {
                    "config_id": row.get("config_id"),
                    "family_id": row.get("family_id"),
                    "train_cv_objective_score": row.get("train_cv_objective_score"),
                    "train_cv_mrr_at_20_delta": row.get("train_cv_mrr_at_20_delta"),
                    "train_cv_hit_at_10_delta": row.get("train_cv_hit_at_10_delta"),
                    "train_cv_hit_at_20_delta": row.get("train_cv_hit_at_20_delta"),
                    "train_cv_hit_at_200_delta": row.get("train_cv_hit_at_200_delta"),
                    "failed_train_cv_guards": row.get("failed_train_cv_guards"),
                }
            )
    return sorted(
        rows,
        key=lambda row: (-float(row["train_cv_objective_score"]), str(row["config_id"])),
    )


def _dev_report_observations(
    *,
    stage118_report: Mapping[str, Any],
    config_stop_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    dev_validation = stage118_report.get("dev_validation") or {}
    if not config_stop_evidence:
        return {
            "status": dev_validation.get("status"),
            "dev_used_for_selection": dev_validation.get("dev_used_for_selection"),
            "dev_used_for_retuning": dev_validation.get("dev_used_for_retuning"),
            "dev_observations_are_non_adoptable": True,
        }
    best_dev_mrr = max(
        config_stop_evidence,
        key=lambda row: _float(row.get("dev_mrr_at_20_delta_report_only")),
    )
    best_dev_hit10 = max(
        config_stop_evidence,
        key=lambda row: _float(row.get("dev_hit_at_10_delta_report_only")),
    )
    return {
        "status": dev_validation.get("status"),
        "selected_config_id": dev_validation.get("selected_config_id"),
        "dev_used_for_selection": dev_validation.get("dev_used_for_selection"),
        "dev_used_for_retuning": dev_validation.get("dev_used_for_retuning"),
        "dev_observations_are_non_adoptable": True,
        "best_dev_mrr_at_20_delta_config_id": best_dev_mrr.get("config_id"),
        "best_dev_mrr_at_20_delta": best_dev_mrr.get(
            "dev_mrr_at_20_delta_report_only"
        ),
        "best_dev_hit_at_10_delta_config_id": best_dev_hit10.get("config_id"),
        "best_dev_hit_at_10_delta": best_dev_hit10.get(
            "dev_hit_at_10_delta_report_only"
        ),
    }


def _guard_checks(
    *,
    stage118_summary: Mapping[str, Any],
    stage117_summary: Mapping[str, Any],
    stage118_report: Mapping[str, Any],
    config_stop_evidence: Sequence[Mapping[str, Any]],
    positive_signal_blocked: Sequence[Mapping[str, Any]],
    user_confirmed_stop: bool,
) -> list[dict[str, Any]]:
    split_contract = stage118_report.get("split_contract") or {}
    stage118_guard_checks = stage118_report.get("guard_checks") or []
    public_payload = {
        "stage118_summary": stage118_summary,
        "stage117_summary": stage117_summary,
        "config_stop_evidence": config_stop_evidence,
        "positive_signal_blocked": positive_signal_blocked,
    }
    failed_keys = sorted(_forbidden_keys_found(public_payload))
    return [
        _check(
            name="source_stage118_report_is_stage118",
            passed=stage118_summary.get("stage") == _SOURCE_STAGE118,
            observed=stage118_summary.get("stage"),
            expected=_SOURCE_STAGE118,
        ),
        _check(
            name="user_confirmed_stage119_stop_decision",
            passed=user_confirmed_stop,
            observed=user_confirmed_stop,
            expected=True,
        ),
        _check(
            name="stage118_analysis_id_matches",
            passed=stage118_summary.get("analysis_id") == _STAGE118_ANALYSIS_ID,
            observed=stage118_summary.get("analysis_id"),
            expected=_STAGE118_ANALYSIS_ID,
        ),
        _check(
            name="stage118_completed_with_no_train_cv_selectable_config",
            passed=stage118_summary.get("decision_status")
            == _STAGE118_NO_SELECTABLE_STATUS,
            observed=stage118_summary.get("decision_status"),
            expected=_STAGE118_NO_SELECTABLE_STATUS,
        ),
        _check(
            name="stage118_recommends_stop_decision",
            passed=stage118_summary.get("recommended_next_direction")
            == "record_second_stage_reranking_stop_decision",
            observed=stage118_summary.get("recommended_next_direction"),
            expected="record_second_stage_reranking_stop_decision",
        ),
        _check(
            name="stage118_all_guard_checks_passed",
            passed=all(check.get("passed") is True for check in stage118_guard_checks),
            observed={
                "passed": stage118_summary.get("guard_check_passed_count"),
                "total": stage118_summary.get("guard_check_count"),
            },
            expected="all passed",
        ),
        _check(
            name="stage118_split_contract_is_train_dev_only",
            passed=split_contract.get("development_splits") == ["train", "dev"]
            and split_contract.get("selection_split") == "train"
            and split_contract.get("validation_split") == "dev"
            and split_contract.get("forbidden_final_splits") == ["test"],
            observed=split_contract,
            expected="train/dev only with test forbidden",
        ),
        _check(
            name="stage117_protocol_id_matches",
            passed=stage117_summary.get("protocol_id") == _STAGE117_PROTOCOL_ID,
            observed=stage117_summary.get("protocol_id"),
            expected=_STAGE117_PROTOCOL_ID,
        ),
        _check(
            name="stage117_dev_test_runtime_boundaries_locked",
            passed=stage117_summary.get("dev_selection_allowed") is False
            and stage117_summary.get("dev_retuning_allowed") is False
            and stage117_summary.get("dev_threshold_tuning_allowed") is False
            and stage117_summary.get("final_test_metrics_allowed") is False
            and stage117_summary.get("test_tuning_allowed") is False
            and stage117_summary.get("fallback_strategies_enabled") is False
            and stage117_summary.get("default_runtime_policy") == "unchanged",
            observed=stage117_summary,
            expected="dev/test/runtime/fallback boundaries locked",
        ),
        _check(
            name="stage118_train_cv_selection_used_train_only",
            passed=stage118_summary.get("selection_split") == "train"
            and stage118_summary.get("selection_mode") in (None, _TRAIN_CV_SELECTION_MODE)
            and stage118_summary.get("selection_source") == "train_cv_only",
            observed=stage118_summary,
            expected="train_cv_only",
        ),
        _check(
            name="stage118_selected_no_config",
            passed=stage118_summary.get("selected_config_id") is None
            and stage118_summary.get("selectable_config_count") == 0,
            observed={
                "selected_config_id": stage118_summary.get("selected_config_id"),
                "selectable_config_count": stage118_summary.get(
                    "selectable_config_count"
                ),
            },
            expected="selected_config_id null and selectable_config_count 0",
        ),
        _check(
            name="all_stage118_configs_are_train_cv_nonselectable",
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
            name="stage118_has_positive_signal_but_blocked_configs",
            passed=bool(positive_signal_blocked)
            and all(row.get("failed_train_cv_guards") for row in positive_signal_blocked),
            observed=positive_signal_blocked,
            expected="positive train-CV signals exist but all have failed guards",
        ),
        _check(
            name="stage118_dev_report_has_no_selected_config",
            passed=stage118_summary.get("dev_validation_status")
            == "no_train_cv_selectable_config"
            and stage118_summary.get("dev_used_for_selection") is False
            and stage118_summary.get("dev_used_for_retuning") is False,
            observed={
                "status": stage118_summary.get("dev_validation_status"),
                "dev_used_for_selection": stage118_summary.get("dev_used_for_selection"),
                "dev_used_for_retuning": stage118_summary.get("dev_used_for_retuning"),
            },
            expected="no selected train-CV config; dev report-only",
        ),
        _check(
            name="stage118_top200_candidate_pool_reproduced",
            passed=stage118_summary.get("train_top200_gold_present_rate") == 0.9324
            and stage118_summary.get("dev_top200_gold_present_rate") == 0.9079,
            observed={
                "train": stage118_summary.get("train_top200_gold_present_rate"),
                "dev": stage118_summary.get("dev_top200_gold_present_rate"),
            },
            expected={"train": 0.9324, "dev": 0.9079},
        ),
        _check(
            name="stage118_candidate_rows_not_written",
            passed=stage118_summary.get("raw_candidate_rows_written") is False,
            observed=stage118_summary.get("raw_candidate_rows_written"),
            expected=False,
        ),
        _check(
            name="stage118_final_test_gate_locked",
            passed=stage118_summary.get("can_open_final_test_gate_now") is False
            and stage118_summary.get("can_run_final_test_metrics_now") is False,
            observed={
                "can_open_final_test_gate_now": stage118_summary.get(
                    "can_open_final_test_gate_now"
                ),
                "can_run_final_test_metrics_now": stage118_summary.get(
                    "can_run_final_test_metrics_now"
                ),
            },
            expected=False,
        ),
        _check(
            name="stage118_forbids_test_tuning",
            passed=stage118_summary.get("can_use_test_for_tuning") is False,
            observed=stage118_summary.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage118_default_runtime_policy_unchanged",
            passed=stage118_summary.get("default_runtime_policy") == "unchanged",
            observed=stage118_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage118_fallback_strategies_disabled",
            passed=stage118_summary.get("fallback_strategies_enabled") is False,
            observed=stage118_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="stage119_no_new_train_dev_metrics_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage119_split_files_not_loaded",
            passed=True,
            observed="not_loaded",
            expected="not_loaded",
        ),
        _check(
            name="stage119_corpus_documents_not_loaded",
            passed=True,
            observed="not_loaded",
            expected="not_loaded",
        ),
        _check(
            name="stage119_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage119_default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
        _check(
            name="stage119_fallback_strategies_not_added",
            passed=True,
            observed=False,
            expected=False,
        ),
        _check(
            name="stage119_public_outputs_have_no_forbidden_keys",
            passed=not failed_keys,
            observed=failed_keys,
            expected=[],
        ),
    ]


def _decision(*, guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_second_stage_reranking_stop_decision_blocked",
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
        "status": "primeqa_hybrid_second_stage_reranking_family_stopped",
        "stopped_family_id": _STOPPED_FAMILY_ID,
        "stopped_protocol_id": _STAGE117_PROTOCOL_ID,
        "stopped_analysis_id": _STAGE118_ANALYSIS_ID,
        "current_route_defaultization": "blocked",
        "new_research_direction_required_before_any_runtime_or_test_gate": True,
        "recommended_next_direction": "user_confirmed_next_research_direction_required",
        "can_continue_train_dev_development": True,
        "requires_user_confirmation_before_next_protocol": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage120: only after user confirmation, choose the next "
            "train/dev-only research direction. Do not select from dev-only "
            "observations, keep test locked, keep runtime defaults unchanged, "
            "and do not add fallback strategies."
        ),
    }


def _train_cv_delta(config: Mapping[str, Any], metric: str) -> float:
    comparison = (config.get("comparisons_to_baseline") or {}).get("train_cv") or {}
    return _float(comparison.get(metric))


def _guard_observed(config: Mapping[str, Any], guard_name: str) -> Any:
    for guard in config.get("train_cv_selection_guards") or []:
        if guard.get("name") == guard_name:
            return guard.get("observed")
    return None


def _failed_train_cv_guard_names(config: Mapping[str, Any]) -> list[str]:
    return [
        str(guard.get("name"))
        for guard in config.get("train_cv_selection_guards") or []
        if guard.get("passed") is not True
    ]


def _train_cv_guard_failure_reasons(
    configs: Sequence[Mapping[str, Any]],
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for config in configs:
        for reason in _failed_train_cv_guard_names(config):
            counter[reason] += 1
    return counter


def _train_cv_objective_score_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(row["config_id"]),
            value=float(row["train_cv_objective_score"]),
            value_label=f"{float(row['train_cv_objective_score']):+.4f}",
        )
        for row in _config_stop_evidence_from_report(report)
    ]


def _train_cv_metric_delta_bars(report: Mapping[str, Any], field: str) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(row["config_id"]),
            value=float(row[field]),
            value_label=f"{float(row[field]):+.4f}",
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


def _float(value: Any) -> float:
    return float(value or 0.0)


def _int(value: Any) -> int:
    return int(value or 0)
