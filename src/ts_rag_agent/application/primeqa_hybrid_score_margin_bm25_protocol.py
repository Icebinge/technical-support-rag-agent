from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 94"
_CREATED_AT = "2026-07-15"
_SOURCE_STAGE84 = "Stage 84"
_SOURCE_STAGE93 = "Stage 93"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_PROTOCOL_ID = "score_margin_bm25_normalization_gate_train_dev_v1"
_CANDIDATE_ID = "score_margin_bm25_normalization_gate_design"
_STOPPED_CANDIDATE_ID = "section_signal_guarded_expansion_design"
_BLOCKED_CANDIDATE_ID = "source_doc_ids_oracle_union_blocked"
_ALLOWED_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})


@dataclass(frozen=True)
class PrimeQAHybridScoreMarginBM25ProtocolVisualization:
    """One generated Stage94 score-margin BM25 protocol visualization."""

    name: str
    path: str


def freeze_primeqa_hybrid_score_margin_bm25_protocol(
    *,
    stage84_report_path: Path,
    stage93_report_path: Path,
    user_confirmed_candidate: bool,
    confirmed_candidate_id: str,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the train/dev protocol for score-margin BM25 normalization."""

    started_at = time.perf_counter()
    stage84_report = _load_json_object(stage84_report_path)
    stage93_report = _load_json_object(stage93_report_path)
    candidate = _selected_candidate(stage84_report)
    stage93_candidate = _stage93_next_candidate(stage93_report)
    frozen_protocol = _frozen_protocol(candidate)
    guard_checks = _guard_checks(
        stage84_report=stage84_report,
        stage93_report=stage93_report,
        candidate=candidate,
        stage93_candidate=stage93_candidate,
        frozen_protocol=frozen_protocol,
        user_confirmed_candidate=user_confirmed_candidate,
        confirmed_candidate_id=confirmed_candidate_id,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_scope": (
            "Train/dev-only protocol freeze for the score-margin BM25 "
            "normalization gate candidate selected after Stage93 stopped the "
            "section-signal route. This stage reads only public-safe Stage84 "
            "and Stage93 reports, freezes a predeclared protocol for a future "
            "train/dev metric run, does not run retrieval metrics, does not "
            "load the frozen test split, does not run final metrics, does not "
            "use source DOC_IDS as runtime retrieval evidence, does not choose "
            "runtime thresholds from dev-only observations, and does not change "
            "runtime defaults."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_candidate),
            "confirmed_candidate_id": confirmed_candidate_id,
            "confirmation_note": confirmation_note,
        },
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
            "forbidden_final_splits": sorted(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage84_report": _fingerprint(stage84_report_path),
            "stage93_report": _fingerprint(stage93_report_path),
        },
        "stage84_decision": stage84_report.get("decision") or {},
        "stage93_decision": stage93_report.get("decision") or {},
        "stage84_candidate_summary": candidate,
        "stage93_next_candidate_summary": stage93_candidate,
        "frozen_protocol": frozen_protocol,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
        "timing_seconds": {
            "load_and_freeze": round(checked_at - started_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_score_margin_bm25_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridScoreMarginBM25ProtocolVisualization]:
    """Write SVG charts for the Stage94 score-margin BM25 protocol."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage94_score_margin_bm25_config_b_values.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage94 score-margin BM25 config b values",
                bars=_config_b_value_bars(report),
                x_label="challenger BM25 b",
                width=1220,
                margin_left=560,
            )
        ),
        "stage94_score_margin_bm25_margin_thresholds.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage94 score-margin BM25 margin thresholds",
                bars=_config_margin_threshold_bars(report),
                x_label="maximum score margin to BM25 rank10",
                width=1220,
                margin_left=560,
            )
        ),
        "stage94_score_margin_bm25_length_thresholds.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage94 score-margin BM25 length threshold counts",
                bars=_config_length_threshold_bars(report),
                x_label="active document-length gates",
                width=1220,
                margin_left=560,
            )
        ),
        "stage94_score_margin_bm25_feature_group_counts.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage94 score-margin BM25 feature group counts",
                bars=_feature_group_bars(report),
                x_label="feature count",
                width=1100,
                margin_left=400,
            )
        ),
        "stage94_score_margin_bm25_protocol_decision_flags.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage94 score-margin BM25 protocol decision flags",
                bars=_decision_flag_bars(report),
                x_label="1 means true",
                width=1160,
                margin_left=500,
            )
        ),
        "stage94_score_margin_bm25_guard_check_status.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage94 score-margin BM25 guard check status",
                bars=_guard_check_bars(report),
                x_label="1 means passed",
                width=1380,
                margin_left=680,
            )
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridScoreMarginBM25ProtocolVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _selected_candidate(stage84_report: Mapping[str, Any]) -> dict[str, Any]:
    candidates = stage84_report.get("candidate_designs") or []
    if not isinstance(candidates, list):
        raise ValueError("Stage84 candidate_designs must be a list")
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate.get("candidate_id") == _CANDIDATE_ID:
            return _public_candidate_summary(candidate)
    raise ValueError(f"Stage84 report does not contain candidate {_CANDIDATE_ID!r}")


def _stage93_next_candidate(stage93_report: Mapping[str, Any]) -> dict[str, Any]:
    candidate_queue = stage93_report.get("candidate_queue") or {}
    next_candidate = candidate_queue.get("next_candidate_summary") or {}
    if isinstance(next_candidate, Mapping):
        return _public_candidate_summary(next_candidate)
    return {}


def _public_candidate_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    public_fields = [
        "candidate_id",
        "name",
        "category",
        "status",
        "risk_level",
        "implementation_readiness",
        "prior_signal_key",
        "prior_signal_score",
        "priority_score",
        "target_miss_count",
        "target_miss_count_by_split",
        "target_rank_buckets",
        "target_routes",
        "target_reason_tags",
        "rationale",
        "stage85_protocol_outline",
        "target_metric_contract",
        "runtime_evidence_policy",
    ]
    return {field: candidate[field] for field in public_fields if field in candidate}


def _frozen_protocol(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "candidate_id": _CANDIDATE_ID,
        "protocol_status": "frozen_requires_user_confirmation_before_metric_run",
        "source_stages": [_SOURCE_STAGE84, _SOURCE_STAGE93],
        "target_miss_count": int(candidate.get("target_miss_count") or 0),
        "target_miss_count_by_split": candidate.get("target_miss_count_by_split") or {},
        "target_rank_buckets": candidate.get("target_rank_buckets") or {},
        "baseline_retriever": {
            "config_id": "full_document_bm25_baseline",
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "candidate_depth": 50,
            "primary_top_k": 10,
        },
        "historical_signal_policy": {
            "source_stage": "Stage 82",
            "source_signal": "bm25_k1_b_grid",
            "dev_only_b095_observation_can_select_runtime_rule": False,
            "allowed_use": (
                "Use Stage82 only as motivation for predeclaring train-selected "
                "score-margin and document-length gates."
            ),
        },
        "candidate_config_grid": _candidate_config_grid(),
        "score_margin_feature_contract": _score_margin_feature_contract(),
        "normalization_gate_contract": _normalization_gate_contract(),
        "train_selection_rule": {
            "selection_split": "train",
            "validation_split": "dev",
            "rule": (
                "Select the candidate config on train only by hit@10, then fewer "
                "rank 11-50 near misses, fewer top10 regressions, hit@5, hit@1, "
                "MRR@10, lower top10 promotion budget, then config_id. Dev is "
                "validation only."
            ),
            "dev_selection_forbidden": True,
            "test_selection_forbidden": True,
            "stage82_dev_observation_selection_forbidden": True,
        },
        "target_metric_contract": candidate.get("target_metric_contract") or [],
        "metrics_allowed_after_confirmation": [
            "hit@1",
            "hit@5",
            "hit@10",
            "MRR@10",
            "MRR@50",
            "top10_improvement_count",
            "top10_regression_count",
            "rank_up_within_top10_count",
            "rank_down_within_top10_count",
            "not_found_count_at_50",
            "rank_11_to_50_count",
            "rank_11_to_50_count_delta",
            "score_margin_gate_promotion_count",
            "length_band_gate_count",
        ],
        "public_safe_changed_case_fields": [
            "sample_id",
            "split",
            "baseline_rank",
            "challenger_rank",
            "config_id",
            "normalization_view_id",
            "baseline_rank_bucket",
            "challenger_rank_bucket",
            "score_margin_bucket",
            "document_length_bucket",
            "promotion_reason_code",
        ],
        "explicit_exclusions": [
            "Do not use source DOC_IDS as runtime retrieval evidence.",
            "Do not use answer document IDs or gold ranks as runtime features.",
            "Do not choose candidate configs from dev-only performance.",
            "Do not choose candidate configs from Stage82 dev-only b=0.95 observations.",
            "Do not load or evaluate the frozen test split.",
            "Do not write raw question text, answer text, document titles, "
            "document body text, query terms, or matched token strings to the report.",
            "Do not change runtime defaults in this stage.",
            "Do not add behavior outside the predeclared score-margin config grid.",
        ],
    }


def _candidate_config_grid() -> list[dict[str, Any]]:
    return [
        {
            "config_id": "smbn_rank11_20_long_doc_b095_margin_v1",
            "normalization_view_id": "bm25_k1_1_5_b_0_95_long_doc",
            "challenger_bm25_k1": 1.5,
            "challenger_bm25_b": 0.95,
            "eligible_baseline_rank_min": 11,
            "eligible_baseline_rank_max": 20,
            "challenger_rank_max": 10,
            "maximum_score_margin_to_rank10": 0.05,
            "length_gate_mode": "long_document_only",
            "minimum_document_length_ratio_to_average": 1.2,
            "maximum_document_length_ratio_to_average": None,
            "maximum_top10_promotions_per_query": 1,
            "protected_bm25_top_rank_count": 8,
        },
        {
            "config_id": "smbn_rank21_50_long_doc_b095_high_confidence_v1",
            "normalization_view_id": "bm25_k1_1_5_b_0_95_long_doc",
            "challenger_bm25_k1": 1.5,
            "challenger_bm25_b": 0.95,
            "eligible_baseline_rank_min": 21,
            "eligible_baseline_rank_max": 50,
            "challenger_rank_max": 15,
            "maximum_score_margin_to_rank10": 0.03,
            "length_gate_mode": "long_document_only",
            "minimum_document_length_ratio_to_average": 1.5,
            "maximum_document_length_ratio_to_average": None,
            "maximum_top10_promotions_per_query": 1,
            "protected_bm25_top_rank_count": 8,
        },
        {
            "config_id": "smbn_rank11_20_short_doc_b055_margin_v1",
            "normalization_view_id": "bm25_k1_1_5_b_0_55_short_doc",
            "challenger_bm25_k1": 1.5,
            "challenger_bm25_b": 0.55,
            "eligible_baseline_rank_min": 11,
            "eligible_baseline_rank_max": 20,
            "challenger_rank_max": 10,
            "maximum_score_margin_to_rank10": 0.04,
            "length_gate_mode": "short_document_only",
            "minimum_document_length_ratio_to_average": None,
            "maximum_document_length_ratio_to_average": 0.85,
            "maximum_top10_promotions_per_query": 1,
            "protected_bm25_top_rank_count": 8,
        },
        {
            "config_id": "smbn_rank11_50_dual_length_band_margin_v1",
            "normalization_view_id": "bm25_k1_1_5_b_0_55_or_0_95_by_length_band",
            "challenger_bm25_k1": 1.5,
            "challenger_bm25_b": "0.55_for_short_docs_0.95_for_long_docs",
            "eligible_baseline_rank_min": 11,
            "eligible_baseline_rank_max": 50,
            "challenger_rank_max": 12,
            "maximum_score_margin_to_rank10": 0.02,
            "length_gate_mode": "outside_length_band_short_or_long",
            "normalization_branch_rule": (
                "Use b=0.55 when document_length_ratio_to_average <= 0.75; "
                "use b=0.95 when document_length_ratio_to_average >= 1.35."
            ),
            "minimum_document_length_ratio_to_average": 1.35,
            "maximum_document_length_ratio_to_average": 0.75,
            "maximum_top10_promotions_per_query": 1,
            "protected_bm25_top_rank_count": 8,
        },
    ]


def _score_margin_feature_contract() -> dict[str, Any]:
    return {
        "runtime_allowed_feature_groups": {
            "baseline_bm25_features": [
                "baseline_bm25_rank",
                "baseline_bm25_score",
                "baseline_score_margin_to_rank10",
                "baseline_score_margin_to_previous",
            ],
            "challenger_bm25_features": [
                "challenger_bm25_rank",
                "challenger_bm25_score",
                "challenger_score_margin_to_rank10",
                "normalization_view_id",
            ],
            "document_length_features": [
                "document_token_count",
                "average_document_token_count",
                "document_length_ratio_to_average",
                "document_length_bucket",
            ],
            "gate_state_features": [
                "eligible_baseline_rank_bucket",
                "challenger_rank_bucket",
                "top10_promotion_budget_remaining",
                "promotion_reason_code",
            ],
        },
        "prohibited_runtime_features": [
            "source_DOC_IDS",
            "answer document IDs",
            "gold_document_rank",
            "gold_label",
            "dev_selected_config",
            "stage82_dev_selected_b_value",
            "frozen_test_split_membership",
            "raw_question_text",
            "raw_answer_text",
            "raw_document_text",
            "raw_document_title",
            "query_terms",
            "matched_token_strings",
        ],
        "prohibited_report_fields": [
            "question text",
            "answer text",
            "document title",
            "document body text",
            "query terms",
            "matched token strings",
        ],
    }


def _normalization_gate_contract() -> dict[str, Any]:
    return {
        "candidate_pool": (
            "Start from full-document BM25 top50, compute predeclared challenger "
            "BM25 views for the same document corpus, and allow at most one gated "
            "rank 11-50 candidate to move to rank10."
        ),
        "baseline_order_source": "full_document_bm25_baseline",
        "challenger_order_sources": [
            "bm25_k1_1_5_b_0_55_short_doc",
            "bm25_k1_1_5_b_0_95_long_doc",
            "bm25_k1_1_5_b_0_55_or_0_95_by_length_band",
        ],
        "top10_protection": (
            "Promotion configs may move at most one gated candidate to rank10 "
            "and must preserve the protected BM25 top ranks declared by the config."
        ),
        "tie_breakers": [
            "lower challenger_bm25_rank",
            "smaller baseline_score_margin_to_rank10",
            "higher challenger_bm25_score",
            "higher baseline_bm25_score",
            "lower stable document id sort key",
        ],
        "non_eligible_candidate_behavior": (
            "Candidates that do not satisfy the frozen gate remain in baseline "
            "document BM25 order."
        ),
    }


def _guard_checks(
    *,
    stage84_report: Mapping[str, Any],
    stage93_report: Mapping[str, Any],
    candidate: Mapping[str, Any],
    stage93_candidate: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
    user_confirmed_candidate: bool,
    confirmed_candidate_id: str,
) -> list[dict[str, Any]]:
    stage84_decision = stage84_report.get("decision") or {}
    stage93_decision = stage93_report.get("decision") or {}
    target_metric_contract = candidate.get("target_metric_contract") or []
    explicit_exclusions = frozen_protocol.get("explicit_exclusions") or []
    feature_contract = frozen_protocol.get("score_margin_feature_contract") or {}
    prohibited_runtime_features = feature_contract.get("prohibited_runtime_features", [])
    candidate_config_grid = frozen_protocol.get("candidate_config_grid") or []
    train_selection_rule = frozen_protocol.get("train_selection_rule") or {}
    historical_signal_policy = frozen_protocol.get("historical_signal_policy") or {}
    return [
        _check(
            name="source_stage84_report_is_stage84",
            passed=stage84_report.get("stage") == _SOURCE_STAGE84,
            observed=stage84_report.get("stage"),
            expected=_SOURCE_STAGE84,
        ),
        _check(
            name="source_stage93_report_is_stage93",
            passed=stage93_report.get("stage") == _SOURCE_STAGE93,
            observed=stage93_report.get("stage"),
            expected=_SOURCE_STAGE93,
        ),
        _check(
            name="user_confirmed_score_margin_protocol",
            passed=user_confirmed_candidate,
            observed=user_confirmed_candidate,
            expected=True,
        ),
        _check(
            name="stage93_stopped_section_signal_route",
            passed=stage93_decision.get("status")
            == "primeqa_hybrid_section_signal_route_stopped"
            and stage93_decision.get("stopped_candidate_id") == _STOPPED_CANDIDATE_ID,
            observed={
                "status": stage93_decision.get("status"),
                "stopped_candidate_id": stage93_decision.get("stopped_candidate_id"),
            },
            expected=_STOPPED_CANDIDATE_ID,
        ),
        _check(
            name="confirmed_candidate_matches_stage93_next_candidate",
            passed=confirmed_candidate_id
            == stage93_decision.get("next_candidate_id")
            == _CANDIDATE_ID,
            observed={
                "confirmed_candidate_id": confirmed_candidate_id,
                "stage93_next_candidate_id": stage93_decision.get("next_candidate_id"),
            },
            expected=_CANDIDATE_ID,
        ),
        _check(
            name="stage93_next_candidate_summary_matches",
            passed=stage93_candidate.get("candidate_id") == _CANDIDATE_ID,
            observed=stage93_candidate.get("candidate_id"),
            expected=_CANDIDATE_ID,
        ),
        _check(
            name="stage93_requires_confirmation_before_next_protocol",
            passed=stage93_decision.get("requires_user_confirmation_before_next_protocol")
            is True,
            observed=stage93_decision.get(
                "requires_user_confirmation_before_next_protocol"
            ),
            expected=True,
        ),
        _check(
            name="stage93_final_test_metrics_locked",
            passed=stage93_decision.get("can_run_final_test_metrics_now") is False,
            observed=stage93_decision.get("can_run_final_test_metrics_now"),
            expected=False,
        ),
        _check(
            name="stage93_final_test_gate_closed",
            passed=stage93_decision.get("can_open_final_test_gate_now") is False,
            observed=stage93_decision.get("can_open_final_test_gate_now"),
            expected=False,
        ),
        _check(
            name="stage93_forbids_test_tuning",
            passed=stage93_decision.get("can_use_test_for_tuning") is False,
            observed=stage93_decision.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage93_runtime_default_unchanged",
            passed=stage93_decision.get("default_runtime_policy") == "unchanged",
            observed=stage93_decision.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage84_final_test_metrics_locked",
            passed=stage84_decision.get("can_run_final_test_metrics_now") is False,
            observed=stage84_decision.get("can_run_final_test_metrics_now"),
            expected=False,
        ),
        _check(
            name="stage84_forbids_test_tuning",
            passed=stage84_decision.get("can_use_test_for_tuning") is False,
            observed=stage84_decision.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage84_runtime_default_unchanged",
            passed=stage84_decision.get("default_runtime_policy") == "unchanged",
            observed=stage84_decision.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage84_candidate_is_recommended_for_protocol_design",
            passed=candidate.get("status")
            == "recommended_for_train_dev_protocol_design",
            observed=candidate.get("status"),
            expected="recommended_for_train_dev_protocol_design",
        ),
        _check(
            name="stage84_candidate_contract_requires_train_selected_dev_hit10_gain",
            passed=any(
                "train-selected rule must improve dev hit@10" in str(item)
                for item in target_metric_contract
            ),
            observed=target_metric_contract,
            expected="train-selected dev hit@10 improvement",
        ),
        _check(
            name="stage84_candidate_contract_requires_rank_11_to_50_reduction",
            passed=any(
                "rank 11-50 near misses should decrease" in str(item)
                for item in target_metric_contract
            ),
            observed=target_metric_contract,
            expected="rank 11-50 near misses should decrease",
        ),
        _check(
            name="stage84_candidate_guard_blocks_dev_only_b095_selection",
            passed=any(
                "dev-only b=0.95 observations cannot select" in str(item)
                for item in target_metric_contract
            ),
            observed=target_metric_contract,
            expected="dev-only b=0.95 cannot select runtime rule",
        ),
        _check(
            name="protocol_id_is_fixed",
            passed=frozen_protocol.get("protocol_id") == _PROTOCOL_ID,
            observed=frozen_protocol.get("protocol_id"),
            expected=_PROTOCOL_ID,
        ),
        _check(
            name="candidate_config_grid_is_predeclared",
            passed=len(candidate_config_grid) == 4
            and len({config.get("config_id") for config in candidate_config_grid}) == 4,
            observed=[config.get("config_id") for config in candidate_config_grid],
            expected=4,
        ),
        _check(
            name="candidate_config_grid_contains_length_and_margin_gates",
            passed=_config_grid_contains_length_and_margin_gates(candidate_config_grid),
            observed=candidate_config_grid,
            expected="each config has rank, score-margin, length, and promotion gates",
        ),
        _check(
            name="train_selection_rule_forbids_dev_and_test_selection",
            passed=train_selection_rule.get("dev_selection_forbidden") is True
            and train_selection_rule.get("test_selection_forbidden") is True
            and train_selection_rule.get("stage82_dev_observation_selection_forbidden")
            is True,
            observed=train_selection_rule,
            expected="train-only selection",
        ),
        _check(
            name="historical_stage82_signal_is_motivation_only",
            passed=historical_signal_policy.get(
                "dev_only_b095_observation_can_select_runtime_rule"
            )
            is False,
            observed=historical_signal_policy,
            expected="Stage82 dev-only b=0.95 cannot select runtime rule",
        ),
        _check(
            name="score_margin_feature_contract_uses_runtime_scores_only",
            passed="runtime_allowed_feature_groups" in feature_contract
            and "gold_document_rank" in prohibited_runtime_features
            and "dev_selected_config" in prohibited_runtime_features,
            observed=feature_contract,
            expected="runtime BM25 score, rank, and length features only",
        ),
        _check(
            name="source_doc_ids_forbidden_in_runtime_features",
            passed="source_DOC_IDS" in prohibited_runtime_features
            and any("source DOC_IDS" in str(item) for item in explicit_exclusions),
            observed={
                "prohibited_runtime_features": prohibited_runtime_features,
                "explicit_exclusions": explicit_exclusions,
            },
            expected="source DOC_IDS forbidden",
        ),
        _check(
            name="answer_doc_ids_forbidden_in_runtime_features",
            passed="answer document IDs" in prohibited_runtime_features
            and any("answer document IDs" in str(item) for item in explicit_exclusions),
            observed={
                "prohibited_runtime_features": prohibited_runtime_features,
                "explicit_exclusions": explicit_exclusions,
            },
            expected="answer document IDs forbidden",
        ),
        _check(
            name="report_fields_are_public_safe",
            passed=not any(
                field in frozen_protocol.get("public_safe_changed_case_fields", [])
                for field in [
                    "raw_question_text",
                    "raw_answer_text",
                    "document_title",
                    "document_body_text",
                    "query_terms",
                    "matched_token_strings",
                ]
            ),
            observed=frozen_protocol.get("public_safe_changed_case_fields"),
            expected="public-safe ids, ranks, buckets, and counts only",
        ),
        _check(
            name="source_doc_ids_oracle_blocked_candidate_not_selected",
            passed=confirmed_candidate_id != _BLOCKED_CANDIDATE_ID,
            observed=confirmed_candidate_id,
            expected=f"not {_BLOCKED_CANDIDATE_ID}",
        ),
        _check(
            name="stage94_freezes_protocol_without_metrics",
            passed=True,
            observed="protocol_freeze_only",
            expected="protocol_freeze_only",
        ),
        _check(
            name="stage94_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage94_default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
    ]


def _config_grid_contains_length_and_margin_gates(
    configs: Sequence[Mapping[str, Any]],
) -> bool:
    if not configs:
        return False
    for config in configs:
        promotion_budget = int(config.get("maximum_top10_promotions_per_query") or 0)
        protected_rank_count = int(config.get("protected_bm25_top_rank_count") or 0)
        if promotion_budget > 1 or protected_rank_count < 8:
            return False
        if config.get("maximum_score_margin_to_rank10") is None:
            return False
        has_min_length = config.get("minimum_document_length_ratio_to_average") is not None
        has_max_length = config.get("maximum_document_length_ratio_to_average") is not None
        if not has_min_length and not has_max_length:
            return False
        if config.get("eligible_baseline_rank_min") is None:
            return False
        if config.get("eligible_baseline_rank_max") is None:
            return False
        if config.get("challenger_rank_max") is None:
            return False
    return True


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_score_margin_bm25_protocol_blocked",
            "protocol_id": _PROTOCOL_ID,
            "candidate_id": _CANDIDATE_ID,
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "requires_user_confirmation_before_train_dev_run": True,
            "can_run_train_dev_metrics_after_user_confirmation": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": "primeqa_hybrid_score_margin_bm25_protocol_frozen",
        "protocol_id": _PROTOCOL_ID,
        "candidate_id": _CANDIDATE_ID,
        "can_continue_train_dev_development": True,
        "requires_user_confirmation_before_train_dev_run": True,
        "can_run_train_dev_metrics_after_user_confirmation": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage 95: after user confirmation, run the frozen train/dev-only "
            "score-margin BM25 normalization gate comparison; keep test locked, "
            "do not use source DOC_IDS, and do not run final metrics."
        ),
    }


def _config_b_value_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for config in report["frozen_protocol"]["candidate_config_grid"]:
        b_value = config["challenger_bm25_b"]
        numeric_value = _numeric_b_value(b_value)
        bars.append(
            BarDatum(
                label=config["config_id"],
                value=numeric_value,
                value_label=str(b_value),
            )
        )
    return bars


def _numeric_b_value(b_value: Any) -> float:
    if isinstance(b_value, str) and "0.95" in b_value:
        return 0.95
    if isinstance(b_value, str) and "0.55" in b_value:
        return 0.55
    return float(b_value)


def _config_margin_threshold_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=config["config_id"],
            value=float(config["maximum_score_margin_to_rank10"]),
            value_label=f"{config['maximum_score_margin_to_rank10']:.2f}",
        )
        for config in report["frozen_protocol"]["candidate_config_grid"]
    ]


def _config_length_threshold_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for config in report["frozen_protocol"]["candidate_config_grid"]:
        active_thresholds = sum(
            threshold is not None
            for threshold in (
                config.get("minimum_document_length_ratio_to_average"),
                config.get("maximum_document_length_ratio_to_average"),
            )
        )
        bars.append(
            BarDatum(
                label=config["config_id"],
                value=float(active_thresholds),
                value_label=str(active_thresholds),
            )
        )
    return bars


def _feature_group_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    groups = report["frozen_protocol"]["score_margin_feature_contract"][
        "runtime_allowed_feature_groups"
    ]
    return [
        BarDatum(label=group, value=float(len(features)), value_label=str(len(features)))
        for group, features in sorted(groups.items())
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report["decision"]
    return [
        BarDatum(
            label="can_run_train_dev_metrics_after_user_confirmation",
            value=float(decision["can_run_train_dev_metrics_after_user_confirmation"]),
            value_label="yes"
            if decision["can_run_train_dev_metrics_after_user_confirmation"]
            else "no",
        ),
        BarDatum(
            label="can_run_final_test_metrics_now",
            value=float(decision["can_run_final_test_metrics_now"]),
            value_label="yes" if decision["can_run_final_test_metrics_now"] else "no",
        ),
        BarDatum(
            label="can_use_test_for_tuning",
            value=float(decision["can_use_test_for_tuning"]),
            value_label="yes" if decision["can_use_test_for_tuning"] else "no",
        ),
    ]


def _guard_check_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=check["name"],
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
