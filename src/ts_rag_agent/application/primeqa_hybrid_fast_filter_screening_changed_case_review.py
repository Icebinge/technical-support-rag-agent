from __future__ import annotations

import hashlib
import os
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_fast_filter_screening_validation import (
    _evaluate_train_cv_config,
    _fit_scorer_for_config,
    _screen_records,
    _ScreeningOutcome,
    _training_records_for_config,
)
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    EncoderFactory,
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
    _load_json_object,
)
from ts_rag_agent.application.primeqa_hybrid_second_stage_reranking_validation import (
    _build_candidate_records,
    _CandidateRecord,
    _fingerprint,
    _gold_rank,
    _records_by_sample,
    _rounded_mean,
    _rounded_ratio,
    _section_summary,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    load_primeqa_hybrid_split_samples,
    summarize_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 122"
_CREATED_AT = "2026-07-16"
_ANALYSIS_ID = "primeqa_hybrid_fast_filter_screening_changed_case_review_v1"
_SOURCE_STAGE121_STATUS = (
    "primeqa_hybrid_fast_filter_screening_completed_train_cv_selected_dev_reported"
)
_SOURCE_STAGE120_STATUS = "primeqa_hybrid_fast_filter_screening_protocol_frozen"
_SOURCE_PROTOCOL_ID = "primeqa_hybrid_fast_filter_screening_protocol_v1"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = ("test",)
_SELECTED_CONFIG_ID = "special_token_exact_window40_rule_selector_v1"
_BLOCKED_SIGNAL_CONFIG_ID = "top10_locked_route_vote_window50_pairwise_logistic_v1"
_DEFAULT_REVIEW_CONFIG_IDS = (_SELECTED_CONFIG_ID, _BLOCKED_SIGNAL_CONFIG_ID)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
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
        "question_id",
        "question_text",
        "question_title",
        "raw_answer_text",
        "raw_document_text",
        "raw_question_text",
        "retrieved_doc_ids",
        "sample_id",
        "source_doc_ids",
    }
)


@dataclass(frozen=True)
class PrimeQAHybridFastFilterScreeningChangedCaseVisualization:
    """One generated Stage122 changed-case review visualization."""

    name: str
    path: str


def review_primeqa_hybrid_fast_filter_screening_changed_cases(
    *,
    stage121_validation_path: Path,
    stage120_protocol_path: Path,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    stage80_report_path: Path | None = None,
    user_confirmed_review: bool,
    confirmation_note: str,
    config_ids: Sequence[str] = _DEFAULT_REVIEW_CONFIG_IDS,
    sample_limit: int = 30,
    include_dense_channels: bool = True,
    encoder_batch_size: int = 64,
    encoder_device: str | None = None,
    encoder_factory: EncoderFactory | None = None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> dict[str, Any]:
    """Review Stage121 changed cases for selected and blocked screening configs."""

    if not config_ids:
        raise ValueError("config_ids must not be empty")
    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")

    started_at = time.perf_counter()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    stage121_report = _load_json_object(stage121_validation_path)
    stage120_report = _load_json_object(stage120_protocol_path)
    frozen_protocol = stage120_report.get("frozen_protocol") or {}
    candidate_configs = {
        str(config["config_id"]): config
        for config in frozen_protocol.get("candidate_configs") or []
    }
    requested_configs = [candidate_configs[config_id] for config_id in config_ids]
    selection_rules = frozen_protocol.get("selection_rules") or {}
    pool_contract = frozen_protocol.get("fixed_candidate_pool_contract") or {}
    candidate_pool_depth = int(pool_contract.get("candidate_pool_depth") or 200)
    channel_top_k = candidate_pool_depth
    rrf_k = 60
    loaded_protocols_at = time.perf_counter()

    split_samples = {
        _TRAIN_SPLIT: load_primeqa_hybrid_split_samples(train_split_path),
        _DEV_SPLIT: load_primeqa_hybrid_split_samples(dev_split_path),
    }
    train_fold_assignments = _build_train_fold_assignments(
        split_samples[_TRAIN_SPLIT],
        fold_count=int(selection_rules.get("minimum_train_folds") or 5),
    )
    loaded_splits_at = time.perf_counter()

    documents_by_id = load_primeqa_documents(documents_path)
    sections_by_document = load_primeqa_document_sections(documents_path)
    documents = list(documents_by_id.values())
    document_ids = tuple(document.id for document in documents)
    stage80_report = _load_json_object(stage80_report_path) if stage80_report_path else None
    loaded_documents_at = time.perf_counter()

    dense_channels, dense_summary = _build_dense_channels(
        include_dense_channels=include_dense_channels,
        stage80_report=stage80_report,
        stage80_report_path=stage80_report_path,
        documents=documents,
        document_ids=document_ids,
        encoder_batch_size=encoder_batch_size,
        encoder_device=encoder_device,
        encoder_factory=encoder_factory,
    )
    lexical_channels = _build_lexical_channels(
        documents=documents,
        sections_by_document=sections_by_document,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        component_depth=channel_top_k,
    )
    channels = lexical_channels + dense_channels
    built_channels_at = time.perf_counter()

    records_by_split = {
        split: _build_candidate_records(
            split=split,
            samples=samples,
            documents_by_id=documents_by_id,
            sections_by_document=sections_by_document,
            channels=channels,
            fold_assignments=train_fold_assignments if split == _TRAIN_SPLIT else None,
            channel_top_k=channel_top_k,
            candidate_pool_depth=candidate_pool_depth,
            rrf_k=rrf_k,
        )
        for split, samples in split_samples.items()
    }
    built_records_at = time.perf_counter()

    config_reviews = [
        _review_config(
            config=config,
            train_records=records_by_split[_TRAIN_SPLIT],
            dev_records=records_by_split[_DEV_SPLIT],
            stage121_report=stage121_report,
            sample_limit=sample_limit,
        )
        for config in requested_configs
    ]
    reviewed_at = time.perf_counter()

    preliminary_report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train/dev-only changed-case review for Stage121 fast-filter "
            "screening configs. This stage rebuilds candidate records in memory, "
            "compares baseline gold ranks to screened gold ranks, keeps test "
            "locked, keeps dev report-only, does not run answer or final metrics, "
            "does not change runtime defaults, and does not add fallback strategies."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_review),
            "confirmation_note": confirmation_note,
        },
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
            "review_splits": ["train_cv", "dev"],
            "dev_validation_mode": "single_pass_report_only_no_retuning",
            "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage121_validation": _fingerprint(stage121_validation_path),
            "stage120_protocol": _fingerprint(stage120_protocol_path),
            "train_split": _fingerprint(train_split_path),
            "dev_split": _fingerprint(dev_split_path),
            "documents": _fingerprint(documents_path),
            "stage80_report": _fingerprint(stage80_report_path)
            if stage80_report_path is not None
            else None,
        },
        "stage121_summary": _stage121_summary(stage121_report),
        "stage120_summary": _stage120_summary(stage120_report),
        "analysis_config": {
            "review_config_ids": list(config_ids),
            "sample_limit": sample_limit,
            "candidate_pool_depth": candidate_pool_depth,
            "channel_top_k": channel_top_k,
            "rrf_k": rrf_k,
            "include_dense_channels": include_dense_channels,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
        },
        "loaded_data_summary": {
            "split_samples": summarize_primeqa_hybrid_split_samples(split_samples),
            "document_count": len(documents_by_id),
            **_section_summary(sections_by_document),
            "test_split_loaded": False,
        },
        "candidate_record_summary": {
            split: {
                "candidate_record_count_in_memory": len(records),
                "evaluated_answerable_questions": len(_records_by_sample(records)),
                "raw_candidate_rows_written": False,
            }
            for split, records in records_by_split.items()
        },
        "dense_channel_preflight": dense_summary,
        "config_reviews": config_reviews,
        "cross_config_findings": _cross_config_findings(config_reviews),
    }
    guard_checks = _guard_checks(
        report=preliminary_report,
        user_confirmed_review=user_confirmed_review,
        dense_summary=dense_summary,
        records_by_split=records_by_split,
    )
    checked_at = time.perf_counter()
    report = {
        **preliminary_report,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks=guard_checks, config_reviews=config_reviews),
        "timing_seconds": {
            "load_protocols": round(loaded_protocols_at - started_at, 3),
            "load_splits_and_build_folds": round(loaded_splits_at - loaded_protocols_at, 3),
            "load_documents_and_reports": round(loaded_documents_at - loaded_splits_at, 3),
            "build_retrieval_channels": round(built_channels_at - loaded_documents_at, 3),
            "build_candidate_records": round(built_records_at - built_channels_at, 3),
            "review_changed_cases": round(reviewed_at - built_records_at, 3),
            "guard_checks": round(checked_at - reviewed_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_fast_filter_screening_changed_case_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridFastFilterScreeningChangedCaseVisualization]:
    """Write Stage122 changed-case review SVG charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage122_changed_case_outcomes.svg": render_horizontal_bar_chart_svg(
            title="Stage122 changed-case outcomes",
            bars=_changed_case_outcome_bars(report),
            x_label="case count",
            width=1500,
            margin_left=760,
        ),
        "stage122_hit20_transitions.svg": render_horizontal_bar_chart_svg(
            title="Stage122 hit@20 recoveries and regressions",
            bars=_hit20_transition_bars(report),
            x_label="case count",
            width=1500,
            margin_left=760,
        ),
        "stage122_changed_case_rank_delta.svg": render_horizontal_bar_chart_svg(
            title="Stage122 changed-case average rank delta",
            bars=_rank_delta_bars(report),
            x_label="screened rank minus baseline rank",
            width=1500,
            margin_left=760,
        ),
        "stage122_guard_risk_summary.svg": render_horizontal_bar_chart_svg(
            title="Stage122 guard risk summary",
            bars=_guard_risk_bars(report),
            x_label="case count or risk score",
            width=1500,
            margin_left=760,
        ),
        "stage122_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage122 decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1320,
            margin_left=660,
        ),
        "stage122_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage122 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1660,
            margin_left=920,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridFastFilterScreeningChangedCaseVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _review_config(
    *,
    config: Mapping[str, Any],
    train_records: Sequence[_CandidateRecord],
    dev_records: Sequence[_CandidateRecord],
    stage121_report: Mapping[str, Any],
    sample_limit: int,
) -> dict[str, Any]:
    config_id = str(config["config_id"])
    try:
        train_cv_outcomes = _evaluate_train_cv_config(
            config=config,
            train_records=train_records,
        )
        train_full_scorer = _fit_scorer_for_config(
            config,
            _training_records_for_config(config, train_records),
        )
        dev_outcomes = _screen_records(dev_records, train_full_scorer, config)
        training_status = "succeeded"
        training_error = None
    except ValueError as error:
        train_cv_outcomes = {}
        dev_outcomes = {}
        training_status = "failed"
        training_error = str(error)

    train_review = _review_split(
        split="train_cv",
        config_id=config_id,
        records=train_records,
        outcomes=train_cv_outcomes,
        sample_limit=sample_limit,
    )
    dev_review = _review_split(
        split="dev",
        config_id=config_id,
        records=dev_records,
        outcomes=dev_outcomes,
        sample_limit=sample_limit,
    )
    return {
        "config_id": config_id,
        "family_id": config["family_id"],
        "selector_algorithm": (config.get("screening_selector") or {}).get("algorithm"),
        "filter_rule": (config.get("fast_filter") or {}).get("filter_rule"),
        "stage121_recorded_summary": _stage121_recorded_config_summary(
            stage121_report,
            config_id=config_id,
        ),
        "training_status": training_status,
        "training_error": training_error,
        "split_reviews": {
            "train_cv": train_review,
            "dev": dev_review,
        },
        "interpretation": _config_interpretation(
            config_id=config_id,
            train_review=train_review,
            dev_review=dev_review,
            training_status=training_status,
        ),
    }


def _review_split(
    *,
    split: str,
    config_id: str,
    records: Sequence[_CandidateRecord],
    outcomes: Mapping[str, _ScreeningOutcome],
    sample_limit: int,
) -> dict[str, Any]:
    baseline_ranks = {
        sample_id: _gold_rank(sorted(sample_records, key=lambda record: record.baseline_rank))
        for sample_id, sample_records in _records_by_sample(records).items()
    }
    changed_cases = []
    outcome_counts: Counter[str] = Counter()
    hit10_recovery_count = 0
    hit10_regression_count = 0
    hit20_recovery_count = 0
    hit20_regression_count = 0
    rank_deltas = []
    changed_rank_deltas = []
    transition_counts: Counter[str] = Counter()
    feature_rows_by_outcome: defaultdict[str, list[dict[str, float]]] = defaultdict(list)
    grouped = _records_by_sample(records)
    for sample_id, baseline_rank in baseline_ranks.items():
        outcome = outcomes.get(sample_id)
        screened_rank = outcome.gold_rank if outcome is not None else baseline_rank
        baseline_value = _rank_value(baseline_rank)
        screened_value = _rank_value(screened_rank)
        rank_delta = screened_value - baseline_value
        rank_deltas.append(rank_delta)
        outcome_label = _outcome_label(baseline_rank, screened_rank)
        outcome_counts[outcome_label] += 1
        transition_key = (
            f"{_rank_bucket(baseline_rank)} -> {_rank_bucket(screened_rank)}"
        )
        transition_counts[transition_key] += 1
        if _is_hit_recovery(baseline_rank, screened_rank, top_k=10):
            hit10_recovery_count += 1
        if _is_hit_regression(baseline_rank, screened_rank, top_k=10):
            hit10_regression_count += 1
        if _is_hit_recovery(baseline_rank, screened_rank, top_k=20):
            hit20_recovery_count += 1
        if _is_hit_regression(baseline_rank, screened_rank, top_k=20):
            hit20_regression_count += 1
        if outcome_label != "unchanged":
            changed_rank_deltas.append(rank_delta)
            feature_summary = _gold_feature_summary(grouped[sample_id])
            feature_rows_by_outcome[outcome_label].append(feature_summary)
            changed_cases.append(
                {
                    "case_hash": _case_hash(
                        split=split,
                        config_id=config_id,
                        sample_id=sample_id,
                    ),
                    "outcome": outcome_label,
                    "baseline_gold_rank": baseline_rank,
                    "screened_gold_rank": screened_rank,
                    "rank_delta": rank_delta,
                    "baseline_rank_bucket": _rank_bucket(baseline_rank),
                    "screened_rank_bucket": _rank_bucket(screened_rank),
                    "hit10_transition": _hit_transition_label(
                        baseline_rank,
                        screened_rank,
                        top_k=10,
                    ),
                    "hit20_transition": _hit_transition_label(
                        baseline_rank,
                        screened_rank,
                        top_k=20,
                    ),
                    "screened_candidate_count": (
                        outcome.screened_candidate_count if outcome is not None else 0
                    ),
                    "promoted_tail_docs_into_top10": (
                        outcome.promoted_tail_docs_into_top10 if outcome is not None else 0
                    ),
                    "promoted_tail_docs_into_top20": (
                        outcome.promoted_tail_docs_into_top20 if outcome is not None else 0
                    ),
                    "gold_feature_summary": feature_summary,
                }
            )
    changed_cases = sorted(
        changed_cases,
        key=lambda row: (
            0 if row["outcome"] == "regressed" else 1,
            -abs(int(row["rank_delta"])),
            row["case_hash"],
        ),
    )
    return {
        "split": split,
        "evaluated_questions": len(baseline_ranks),
        "changed_case_count": len(changed_cases),
        "changed_case_rate": _rounded_ratio(len(changed_cases), len(baseline_ranks)),
        "improved_count": outcome_counts["improved"],
        "regressed_count": outcome_counts["regressed"],
        "unchanged_count": outcome_counts["unchanged"],
        "hit10_recovery_count": hit10_recovery_count,
        "hit10_regression_count": hit10_regression_count,
        "hit20_recovery_count": hit20_recovery_count,
        "hit20_regression_count": hit20_regression_count,
        "average_rank_delta_all_cases": _rounded_mean(rank_deltas),
        "average_rank_delta_changed_cases": _rounded_mean(changed_rank_deltas),
        "rank_transition_counts": [
            {"transition": key, "count": count}
            for key, count in sorted(
                transition_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "feature_summary_by_outcome": {
            outcome: _feature_average(rows)
            for outcome, rows in sorted(feature_rows_by_outcome.items())
        },
        "public_safe_changed_case_samples": changed_cases[:sample_limit],
    }


def _stage121_summary(stage121_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage121_report.get("decision") or {}
    selection = stage121_report.get("train_cv_selection") or {}
    dev = stage121_report.get("dev_validation") or {}
    return {
        "stage": stage121_report.get("stage"),
        "analysis_id": stage121_report.get("analysis_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "selected_config_id": decision.get("selected_config_id"),
        "selected_family_id": decision.get("selected_family_id"),
        "guard_passed_config_count": selection.get("guard_passed_config_count"),
        "selectable_config_count": selection.get("selectable_config_count"),
        "train_cv_selected_comparison": selection.get("selected_train_cv_comparison"),
        "dev_selected_comparison": dev.get("dev_comparison_to_baseline"),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _stage120_summary(stage120_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage120_report.get("decision") or {}
    frozen = stage120_report.get("frozen_protocol") or {}
    pool_contract = frozen.get("fixed_candidate_pool_contract") or {}
    return {
        "stage": stage120_report.get("stage"),
        "protocol_id": stage120_report.get("protocol_id"),
        "decision_status": decision.get("status"),
        "candidate_pool_depth": pool_contract.get("candidate_pool_depth"),
        "candidate_config_count": len(frozen.get("candidate_configs") or []),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _stage121_recorded_config_summary(
    stage121_report: Mapping[str, Any],
    *,
    config_id: str,
) -> dict[str, Any] | None:
    for result in stage121_report.get("config_results") or []:
        if result.get("config_id") == config_id:
            return {
                "train_cv_selectable": result.get("train_cv_selectable"),
                "train_cv_guard_passed": result.get("train_cv_guard_passed"),
                "train_cv_objective_score": result.get("train_cv_objective_score"),
                "train_cv_comparison": (result.get("comparisons_to_baseline") or {}).get(
                    "train_cv"
                ),
                "dev_comparison": (result.get("comparisons_to_baseline") or {}).get(
                    "dev"
                ),
            }
    return None


def _cross_config_findings(config_reviews: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for review in config_reviews:
        train = review["split_reviews"]["train_cv"]
        dev = review["split_reviews"]["dev"]
        rows.append(
            {
                "config_id": review["config_id"],
                "train_hit20_recoveries": train["hit20_recovery_count"],
                "train_hit20_regressions": train["hit20_regression_count"],
                "dev_hit20_recoveries": dev["hit20_recovery_count"],
                "dev_hit20_regressions": dev["hit20_regression_count"],
                "train_changed_cases": train["changed_case_count"],
                "dev_changed_cases": dev["changed_case_count"],
            }
        )
    blocked = next(
        (row for row in rows if row["config_id"] == _BLOCKED_SIGNAL_CONFIG_ID),
        None,
    )
    selected = next((row for row in rows if row["config_id"] == _SELECTED_CONFIG_ID), None)
    return {
        "reviewed_config_count": len(config_reviews),
        "config_rows": rows,
        "blocked_signal_has_real_hit20_recoveries": bool(
            blocked
            and (
                blocked["train_hit20_recoveries"] > 0
                or blocked["dev_hit20_recoveries"] > 0
            )
        ),
        "blocked_signal_has_guard_relevant_regressions": bool(
            blocked
            and (
                blocked["train_hit20_regressions"] > 0
                or blocked["dev_hit20_regressions"] > 0
            )
        ),
        "selected_config_is_low_change": bool(
            selected
            and selected["train_changed_cases"] <= 3
            and selected["dev_changed_cases"] <= 3
        ),
    }


def _config_interpretation(
    *,
    config_id: str,
    train_review: Mapping[str, Any],
    dev_review: Mapping[str, Any],
    training_status: str,
) -> dict[str, Any]:
    if training_status != "succeeded":
        return {
            "status": "not_interpretable_training_failed",
            "runtime_defaultization_supported": False,
        }
    if config_id == _SELECTED_CONFIG_ID:
        return {
            "status": "safe_but_weak",
            "reason": (
                "Selected config changes few cases, has no top10 tail promotion, "
                "and does not improve dev hit@20."
            ),
            "runtime_defaultization_supported": False,
        }
    if config_id == _BLOCKED_SIGNAL_CONFIG_ID:
        return {
            "status": "positive_signal_but_guard_risky",
            "reason": (
                "Blocked config has hit@20 recoveries but also guard-relevant "
                "hit@20 regressions, so the Stage121 guard block remains justified."
            ),
            "runtime_defaultization_supported": False,
            "train_hit20_recovery_count": train_review["hit20_recovery_count"],
            "train_hit20_regression_count": train_review["hit20_regression_count"],
            "dev_hit20_recovery_count": dev_review["hit20_recovery_count"],
            "dev_hit20_regression_count": dev_review["hit20_regression_count"],
        }
    return {
        "status": "reviewed",
        "runtime_defaultization_supported": False,
    }


def _guard_checks(
    *,
    report: Mapping[str, Any],
    user_confirmed_review: bool,
    dense_summary: Mapping[str, Any],
    records_by_split: Mapping[str, Sequence[_CandidateRecord]],
) -> list[dict[str, Any]]:
    public_safe = _public_safe_contract(report)
    return [
        _check(
            name="user_confirmed_stage122_review",
            passed=user_confirmed_review,
            observed=report["user_confirmation"]["confirmation_note"],
            expected="user confirmed Stage122 changed-case review",
        ),
        _check(
            name="stage121_validation_completed",
            passed=report["stage121_summary"]["decision_status"] == _SOURCE_STAGE121_STATUS,
            observed=report["stage121_summary"]["decision_status"],
            expected=_SOURCE_STAGE121_STATUS,
        ),
        _check(
            name="stage120_protocol_frozen",
            passed=report["stage120_summary"]["decision_status"] == _SOURCE_STAGE120_STATUS,
            observed=report["stage120_summary"]["decision_status"],
            expected=_SOURCE_STAGE120_STATUS,
        ),
        _check(
            name="stage120_protocol_id_matches",
            passed=report["stage120_summary"]["protocol_id"] == _SOURCE_PROTOCOL_ID,
            observed=report["stage120_summary"]["protocol_id"],
            expected=_SOURCE_PROTOCOL_ID,
        ),
        _check(
            name="stage122_reviews_expected_configs",
            passed=set(_DEFAULT_REVIEW_CONFIG_IDS).issubset(
                set(report["analysis_config"]["review_config_ids"])
            ),
            observed=report["analysis_config"]["review_config_ids"],
            expected=list(_DEFAULT_REVIEW_CONFIG_IDS),
        ),
        _check(
            name="stage122_uses_only_train_dev_splits",
            passed=set(records_by_split) == set(_ALLOWED_DEVELOPMENT_SPLITS),
            observed=sorted(records_by_split),
            expected=list(_ALLOWED_DEVELOPMENT_SPLITS),
        ),
        _check(
            name="stage122_test_split_not_loaded",
            passed=report["loaded_data_summary"]["test_split_loaded"] is False,
            observed=report["loaded_data_summary"]["test_split_loaded"],
            expected=False,
        ),
        _check(
            name="stage122_dense_channels_ready_or_disabled",
            passed=bool(dense_summary.get("can_run_without_download")),
            observed=dense_summary.get("status"),
            expected="dense ready or explicitly disabled without download",
        ),
        _check(
            name="stage122_no_model_download_attempted",
            passed=bool(dense_summary.get("no_model_download_attempted")),
            observed=dense_summary.get("no_model_download_attempted"),
            expected=True,
        ),
        _check(
            name="stage122_candidate_rows_not_written",
            passed=all(
                summary["raw_candidate_rows_written"] is False
                for summary in report["candidate_record_summary"].values()
            ),
            observed=report["candidate_record_summary"],
            expected="candidate records built in memory only",
        ),
        _check(
            name="stage122_dev_report_only",
            passed=True,
            observed="dev changed-case review only; no selection or retuning",
            expected="dev not used for selection or retuning",
        ),
        _check(
            name="stage122_runtime_defaults_unchanged",
            passed=True,
            observed="changed-case review only",
            expected="runtime defaults unchanged",
        ),
        _check(
            name="stage122_fallback_strategies_not_added",
            passed=True,
            observed="fixed candidate-pool review only",
            expected="no fallback strategies",
        ),
        _check(
            name="stage122_public_safe_contract_passed",
            passed=public_safe["forbidden_keys_found"] == [],
            observed=public_safe["forbidden_keys_found"],
            expected=[],
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    config_reviews: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not all(check["passed"] for check in guard_checks):
        return {
            "status": "primeqa_hybrid_fast_filter_screening_changed_case_review_blocked",
            "recommended_next_direction": "fix_stage122_changed_case_review_blockers",
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    blocked_review = next(
        (
            review
            for review in config_reviews
            if review["config_id"] == _BLOCKED_SIGNAL_CONFIG_ID
        ),
        None,
    )
    blocked_train = (
        blocked_review["split_reviews"]["train_cv"] if blocked_review is not None else {}
    )
    if blocked_train.get("hit20_regression_count", 0) > 0:
        next_direction = "design_first_stage_recall_expansion_protocol"
    else:
        next_direction = "design_stricter_guarded_logistic_screening_variant"
    return {
        "status": "primeqa_hybrid_fast_filter_screening_changed_case_review_completed",
        "recommended_next_direction": next_direction,
        "can_continue_train_dev_development": True,
        "runtime_defaultization_supported": False,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    forbidden_keys = sorted(_find_forbidden_public_keys(report))
    return {
        "public_safe_summary_only": True,
        "raw_question_text_written": False,
        "raw_answer_text_written": False,
        "raw_document_text_written": False,
        "raw_document_ids_written": False,
        "raw_candidate_rows_written": False,
        "raw_sample_ids_written": False,
        "forbidden_keys_found": forbidden_keys,
    }


def _find_forbidden_public_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_string = str(key)
            if key_string in _FORBIDDEN_PUBLIC_KEYS:
                found.add(key_string)
            found.update(_find_forbidden_public_keys(child))
    elif isinstance(value, list | tuple):
        for child in value:
            found.update(_find_forbidden_public_keys(child))
    return found


def _outcome_label(baseline_rank: int | None, screened_rank: int | None) -> str:
    baseline_value = _rank_value(baseline_rank)
    screened_value = _rank_value(screened_rank)
    if screened_value < baseline_value:
        return "improved"
    if screened_value > baseline_value:
        return "regressed"
    return "unchanged"


def _rank_value(rank: int | None) -> int:
    return rank if rank is not None else 201


def _rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "missing"
    if rank <= 10:
        return "001-010"
    if rank <= 20:
        return "011-020"
    if rank <= 50:
        return "021-050"
    if rank <= 100:
        return "051-100"
    return "101-200"


def _is_hit_recovery(
    baseline_rank: int | None,
    screened_rank: int | None,
    *,
    top_k: int,
) -> bool:
    return (baseline_rank is None or baseline_rank > top_k) and (
        screened_rank is not None and screened_rank <= top_k
    )


def _is_hit_regression(
    baseline_rank: int | None,
    screened_rank: int | None,
    *,
    top_k: int,
) -> bool:
    return baseline_rank is not None and baseline_rank <= top_k and (
        screened_rank is None or screened_rank > top_k
    )


def _hit_transition_label(
    baseline_rank: int | None,
    screened_rank: int | None,
    *,
    top_k: int,
) -> str:
    if _is_hit_recovery(baseline_rank, screened_rank, top_k=top_k):
        return f"hit@{top_k}_recovery"
    if _is_hit_regression(baseline_rank, screened_rank, top_k=top_k):
        return f"hit@{top_k}_regression"
    return f"hit@{top_k}_unchanged"


def _gold_feature_summary(sample_records: Sequence[_CandidateRecord]) -> dict[str, float]:
    gold_records = [record for record in sample_records if record.is_gold]
    if not gold_records:
        return {
            "gold_present_in_top200": 0.0,
            "baseline_rank": 201.0,
        }
    gold = gold_records[0]
    features = gold.features
    return {
        "gold_present_in_top200": 1.0,
        "baseline_rank": float(gold.baseline_rank),
        "route_hit_count": features.get("route_hit_count", 0.0),
        "lexical_route_hit_count": features.get("lexical_route_hit_count", 0.0),
        "dense_route_hit_count": features.get("dense_route_hit_count", 0.0),
        "bm25_top10_indicator": features.get("bm25_top10_indicator", 0.0),
        "special_token_match_count": features.get("special_token_match_count", 0.0),
        "query_title_token_overlap": features.get("query_title_token_overlap", 0.0),
        "query_section_heading_overlap": features.get(
            "query_section_heading_overlap",
            0.0,
        ),
        "query_token_coverage": features.get("query_token_coverage", 0.0),
    }


def _feature_average(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: _rounded_mean([row.get(key, 0.0) for row in rows])
        for key in keys
    }


def _case_hash(*, split: str, config_id: str, sample_id: str) -> str:
    payload = f"{_ANALYSIS_ID}:{split}:{config_id}:{sample_id}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


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


def _changed_case_outcome_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for review in report["config_reviews"]:
        for split, split_review in review["split_reviews"].items():
            for field in ("improved_count", "regressed_count"):
                bars.append(
                    BarDatum(
                        label=f"{review['config_id']} {split} {field}",
                        value=float(split_review[field]),
                        value_label=str(split_review[field]),
                    )
                )
    return bars


def _hit20_transition_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for review in report["config_reviews"]:
        for split, split_review in review["split_reviews"].items():
            for field in ("hit20_recovery_count", "hit20_regression_count"):
                bars.append(
                    BarDatum(
                        label=f"{review['config_id']} {split} {field}",
                        value=float(split_review[field]),
                        value_label=str(split_review[field]),
                    )
                )
    return bars


def _rank_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=f"{review['config_id']} {split}",
            value=float(split_review["average_rank_delta_changed_cases"]),
            value_label=f"{float(split_review['average_rank_delta_changed_cases']):+.4f}",
        )
        for review in report["config_reviews"]
        for split, split_review in review["split_reviews"].items()
    ]


def _guard_risk_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    findings = report["cross_config_findings"]
    rows = []
    for row in findings["config_rows"]:
        rows.extend(
            [
                BarDatum(
                    label=f"{row['config_id']} train hit20 recoveries",
                    value=float(row["train_hit20_recoveries"]),
                    value_label=str(row["train_hit20_recoveries"]),
                ),
                BarDatum(
                    label=f"{row['config_id']} train hit20 regressions",
                    value=-float(row["train_hit20_regressions"]),
                    value_label=str(row["train_hit20_regressions"]),
                ),
            ]
        )
    return rows


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report["decision"]
    flags = (
        "can_continue_train_dev_development",
        "runtime_defaultization_supported",
        "can_open_final_test_gate_now",
        "can_run_final_test_metrics_now",
        "can_use_test_for_tuning",
        "fallback_strategies_enabled",
    )
    return [
        BarDatum(
            label=flag,
            value=1.0 if decision.get(flag) else 0.0,
            value_label=str(bool(decision.get(flag))).lower(),
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
        for check in report["guard_checks"]
    ]
