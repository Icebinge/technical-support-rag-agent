from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 91"
_CREATED_AT = "2026-07-15"
_SOURCE_STAGE84 = "Stage 84"
_SOURCE_STAGE90 = "Stage 90"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_PROTOCOL_ID = "section_signal_guarded_expansion_train_dev_v1"
_CANDIDATE_ID = "section_signal_guarded_expansion_design"
_STRUCTURED_QUERY_CANDIDATE_ID = "structured_query_keyphrase_compaction_design"
_ALLOWED_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})


@dataclass(frozen=True)
class PrimeQAHybridSectionSignalProtocolVisualization:
    """One generated Stage91 section-signal protocol visualization."""

    name: str
    path: str


def freeze_primeqa_hybrid_section_signal_protocol(
    *,
    stage84_report_path: Path,
    stage90_report_path: Path,
    user_confirmed_candidate: bool,
    confirmed_candidate_id: str,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the train/dev protocol for section-signal guarded expansion."""

    started_at = time.perf_counter()
    stage84_report = _load_json_object(stage84_report_path)
    stage90_report = _load_json_object(stage90_report_path)
    candidate = _selected_candidate(stage84_report)
    stage90_candidate = _stage90_next_candidate(stage90_report)
    frozen_protocol = _frozen_protocol(candidate)
    guard_checks = _guard_checks(
        stage84_report=stage84_report,
        stage90_report=stage90_report,
        candidate=candidate,
        stage90_candidate=stage90_candidate,
        frozen_protocol=frozen_protocol,
        user_confirmed_candidate=user_confirmed_candidate,
        confirmed_candidate_id=confirmed_candidate_id,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_scope": (
            "Train/dev-only protocol freeze for the section signal guarded "
            "expansion candidate selected after Stage90 stopped structured "
            "query compaction. This stage reads only public-safe Stage84 and "
            "Stage90 reports, freezes a predeclared protocol for a future "
            "train/dev metric run, does not run retrieval metrics, does not "
            "load the frozen test split, does not run final metrics, does not "
            "use source DOC_IDS as runtime retrieval evidence, and does not "
            "change runtime defaults."
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
            "stage90_report": _fingerprint(stage90_report_path),
        },
        "stage84_decision": stage84_report.get("decision") or {},
        "stage90_decision": stage90_report.get("decision") or {},
        "stage84_candidate_summary": candidate,
        "stage90_next_candidate_summary": stage90_candidate,
        "frozen_protocol": frozen_protocol,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
        "timing_seconds": {
            "load_and_freeze": round(checked_at - started_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_section_signal_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridSectionSignalProtocolVisualization]:
    """Write SVG charts for Stage91 section-signal protocol."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage91_section_signal_config_promotion_budgets.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage91 section-signal config promotion budgets",
                bars=_config_promotion_budget_bars(report),
                x_label="maximum top10 promotions per query",
                width=1240,
                margin_left=520,
            )
        ),
        "stage91_section_signal_config_ratio_thresholds.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage91 section-signal config score-ratio thresholds",
                bars=_config_ratio_threshold_bars(report),
                x_label="minimum section/document score ratio",
                width=1240,
                margin_left=520,
            )
        ),
        "stage91_section_signal_feature_group_counts.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage91 section-signal feature group counts",
                bars=_feature_group_bars(report),
                x_label="feature count",
                width=1080,
                margin_left=380,
            )
        ),
        "stage91_section_signal_protocol_decision_flags.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage91 section-signal protocol decision flags",
                bars=_decision_flag_bars(report),
                x_label="1 means true",
                width=1120,
                margin_left=480,
            )
        ),
        "stage91_section_signal_guard_check_status.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage91 section-signal guard check status",
                bars=_guard_check_bars(report),
                x_label="1 means passed",
                width=1340,
                margin_left=640,
            )
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridSectionSignalProtocolVisualization(
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


def _stage90_next_candidate(stage90_report: Mapping[str, Any]) -> dict[str, Any]:
    candidate_queue = stage90_report.get("candidate_queue") or {}
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
        "source_stages": [_SOURCE_STAGE84, _SOURCE_STAGE90],
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
        "section_signal_source": {
            "source_retriever_id": "section_bm25_max_section_rollup_stage79",
            "section_score_scope": "best section BM25 score per parent document",
            "section_candidate_depth": 50,
            "document_candidate_depth": 50,
            "primary_top_k": 10,
            "raw_section_text_written_to_report": False,
            "raw_document_text_written_to_report": False,
        },
        "candidate_config_grid": _candidate_config_grid(),
        "section_signal_feature_contract": _section_signal_feature_contract(),
        "promotion_contract": _promotion_contract(),
        "train_selection_rule": {
            "selection_split": "train",
            "validation_split": "dev",
            "rule": (
                "Select the candidate config on train by hit@10, then search-depth "
                "net improvements, fewer top10 regressions, hit@5, hit@1, "
                "MRR@10, lower top10 demotion budget, then config_id. Dev is "
                "validation only."
            ),
            "dev_selection_forbidden": True,
            "test_selection_forbidden": True,
        },
        "target_metric_contract": candidate.get("target_metric_contract") or [],
        "metrics_allowed_after_confirmation": [
            "hit@1",
            "hit@5",
            "hit@10",
            "MRR@10",
            "top10_improvement_count",
            "top10_regression_count",
            "search_depth_improvement_count",
            "search_depth_regression_count",
            "rank_up_within_top10_count",
            "rank_down_within_top10_count",
            "not_found_count_at_50",
            "rank_11_to_50_count",
            "section_signal_promotion_count",
            "protected_top10_demotion_count",
        ],
        "public_safe_changed_case_fields": [
            "sample_id",
            "split",
            "baseline_rank",
            "challenger_rank",
            "config_id",
            "section_signal_bucket",
            "baseline_rank_bucket",
            "section_rank_bucket",
            "score_ratio_bucket",
            "score_margin_bucket",
            "promotion_reason_code",
            "top10_protection_action",
        ],
        "explicit_exclusions": [
            "Do not use source DOC_IDS as runtime retrieval evidence.",
            "Do not use answer document IDs or gold ranks as runtime features.",
            "Do not choose candidate configs from dev-only performance.",
            "Do not load or evaluate the frozen test split.",
            "Do not write raw question text, answer text, document titles, "
            "document body text, section text, or matched token strings to the report.",
            "Do not replace full-document BM25 with ungated section BM25.",
            "Do not change runtime defaults in this stage.",
            "Do not add behavior outside the predeclared config grid.",
        ],
    }


def _candidate_config_grid() -> list[dict[str, Any]]:
    return [
        {
            "config_id": "ssgx_shadow_no_top10_demotion_v1",
            "promotion_mode": "shadow_after_top10",
            "eligible_baseline_rank_min": 11,
            "eligible_baseline_rank_max": 50,
            "section_rank_max": 50,
            "minimum_section_to_document_score_ratio": 1.1,
            "maximum_document_score_margin_to_rank10": None,
            "maximum_top10_promotions_per_query": 0,
            "protected_bm25_top_rank_count": 10,
            "demote_existing_bm25_top10": False,
        },
        {
            "config_id": "ssgx_rank11_20_margin_guard_v1",
            "promotion_mode": "single_rank10_promotion",
            "eligible_baseline_rank_min": 11,
            "eligible_baseline_rank_max": 20,
            "section_rank_max": 30,
            "minimum_section_to_document_score_ratio": 1.2,
            "maximum_document_score_margin_to_rank10": 0.08,
            "maximum_top10_promotions_per_query": 1,
            "protected_bm25_top_rank_count": 5,
            "demote_existing_bm25_top10": True,
        },
        {
            "config_id": "ssgx_rank21_50_high_confidence_v1",
            "promotion_mode": "single_rank10_promotion",
            "eligible_baseline_rank_min": 21,
            "eligible_baseline_rank_max": 50,
            "section_rank_max": 20,
            "minimum_section_to_document_score_ratio": 1.45,
            "maximum_document_score_margin_to_rank10": 0.05,
            "maximum_top10_promotions_per_query": 1,
            "protected_bm25_top_rank_count": 8,
            "demote_existing_bm25_top10": True,
        },
        {
            "config_id": "ssgx_section_top50_injection_guard_v1",
            "promotion_mode": "single_rank10_section_candidate_injection",
            "eligible_baseline_rank_min": 51,
            "eligible_baseline_rank_max": None,
            "section_rank_max": 15,
            "minimum_section_to_document_score_ratio": 1.6,
            "maximum_document_score_margin_to_rank10": None,
            "maximum_top10_promotions_per_query": 1,
            "protected_bm25_top_rank_count": 8,
            "demote_existing_bm25_top10": True,
        },
    ]


def _section_signal_feature_contract() -> dict[str, Any]:
    return {
        "runtime_allowed_feature_groups": {
            "document_bm25_features": [
                "document_bm25_rank",
                "document_bm25_score",
                "document_score_margin_to_rank10",
                "document_score_margin_to_previous",
            ],
            "section_bm25_features": [
                "best_section_bm25_rank",
                "best_section_bm25_score",
                "section_to_document_score_ratio",
                "section_score_margin_to_previous",
            ],
            "candidate_overlap_features": [
                "query_overlap_count",
                "title_query_overlap_count",
                "section_query_overlap_count",
            ],
            "gate_state_features": [
                "eligible_baseline_rank_bucket",
                "section_rank_bucket",
                "top10_protection_budget_remaining",
                "promotion_reason_code",
            ],
        },
        "prohibited_runtime_features": [
            "source_DOC_IDS",
            "answer document IDs",
            "gold_document_rank",
            "gold_label",
            "frozen_test_split_membership",
            "raw_question_text",
            "raw_answer_text",
            "raw_document_text",
            "raw_document_title",
            "raw_section_text",
        ],
        "prohibited_report_fields": [
            "question text",
            "answer text",
            "document title",
            "document body text",
            "section text",
            "matched token strings",
        ],
    }


def _promotion_contract() -> dict[str, Any]:
    return {
        "candidate_pool": (
            "Union full-document BM25 top50 parent documents with section BM25 "
            "top50 parent documents, then apply the predeclared gate."
        ),
        "baseline_order_source": "full_document_bm25_baseline",
        "section_order_source": "section_bm25_max_section_rollup_stage79",
        "top10_protection": (
            "The shadow config demotes no baseline top10 candidates. Promotion "
            "configs may move at most one gated candidate to rank10 and must "
            "preserve the protected BM25 top ranks declared by the config."
        ),
        "tie_breakers": [
            "higher section_to_document_score_ratio",
            "higher best_section_bm25_score",
            "higher document_bm25_score",
            "lower section_bm25_rank",
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
    stage90_report: Mapping[str, Any],
    candidate: Mapping[str, Any],
    stage90_candidate: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
    user_confirmed_candidate: bool,
    confirmed_candidate_id: str,
) -> list[dict[str, Any]]:
    stage84_decision = stage84_report.get("decision") or {}
    stage90_decision = stage90_report.get("decision") or {}
    target_metric_contract = candidate.get("target_metric_contract") or []
    explicit_exclusions = frozen_protocol.get("explicit_exclusions") or []
    feature_contract = frozen_protocol.get("section_signal_feature_contract") or {}
    prohibited_runtime_features = feature_contract.get("prohibited_runtime_features", [])
    candidate_config_grid = frozen_protocol.get("candidate_config_grid") or []
    return [
        _check(
            name="source_stage84_report_is_stage84",
            passed=stage84_report.get("stage") == _SOURCE_STAGE84,
            observed=stage84_report.get("stage"),
            expected=_SOURCE_STAGE84,
        ),
        _check(
            name="source_stage90_report_is_stage90",
            passed=stage90_report.get("stage") == _SOURCE_STAGE90,
            observed=stage90_report.get("stage"),
            expected=_SOURCE_STAGE90,
        ),
        _check(
            name="user_confirmed_section_signal_protocol",
            passed=user_confirmed_candidate,
            observed=user_confirmed_candidate,
            expected=True,
        ),
        _check(
            name="stage90_stopped_structured_query_route",
            passed=stage90_decision.get("status")
            == "primeqa_hybrid_structured_query_route_stopped"
            and stage90_decision.get("stopped_candidate_id")
            == _STRUCTURED_QUERY_CANDIDATE_ID,
            observed={
                "status": stage90_decision.get("status"),
                "stopped_candidate_id": stage90_decision.get("stopped_candidate_id"),
            },
            expected=_STRUCTURED_QUERY_CANDIDATE_ID,
        ),
        _check(
            name="confirmed_candidate_matches_stage90_next_candidate",
            passed=confirmed_candidate_id
            == stage90_decision.get("next_candidate_id")
            == _CANDIDATE_ID,
            observed={
                "confirmed_candidate_id": confirmed_candidate_id,
                "stage90_next_candidate_id": stage90_decision.get("next_candidate_id"),
            },
            expected=_CANDIDATE_ID,
        ),
        _check(
            name="stage90_next_candidate_summary_matches",
            passed=stage90_candidate.get("candidate_id") == _CANDIDATE_ID,
            observed=stage90_candidate.get("candidate_id"),
            expected=_CANDIDATE_ID,
        ),
        _check(
            name="stage90_requires_confirmation_before_next_protocol",
            passed=stage90_decision.get("requires_user_confirmation_before_next_protocol")
            is True,
            observed=stage90_decision.get(
                "requires_user_confirmation_before_next_protocol"
            ),
            expected=True,
        ),
        _check(
            name="stage90_final_test_metrics_locked",
            passed=stage90_decision.get("can_run_final_test_metrics_now") is False,
            observed=stage90_decision.get("can_run_final_test_metrics_now"),
            expected=False,
        ),
        _check(
            name="stage90_forbids_test_tuning",
            passed=stage90_decision.get("can_use_test_for_tuning") is False,
            observed=stage90_decision.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage90_runtime_default_unchanged",
            passed=stage90_decision.get("default_runtime_policy") == "unchanged",
            observed=stage90_decision.get("default_runtime_policy"),
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
            name="stage84_candidate_contract_requires_dev_hit10_gain",
            passed=any("dev hit@10 must improve" in str(item) for item in target_metric_contract),
            observed=target_metric_contract,
            expected="dev hit@10 improvement contract",
        ),
        _check(
            name="stage84_candidate_contract_requires_search_depth_net_positive",
            passed=any("search-depth improvements" in str(item) for item in target_metric_contract),
            observed=target_metric_contract,
            expected="search-depth improvements exceed regressions",
        ),
        _check(
            name="stage84_candidate_guard_protects_bm25_top10",
            passed=any(
                "must not demote existing BM25 top10 hits" in str(item)
                for item in target_metric_contract
            ),
            observed=target_metric_contract,
            expected="protect existing BM25 top10 hits",
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
            name="section_signal_contract_uses_runtime_scores_only",
            passed="runtime_allowed_feature_groups" in feature_contract
            and "gold_document_rank" in prohibited_runtime_features
            and "raw_section_text" in prohibited_runtime_features,
            observed=feature_contract,
            expected="runtime section/document scores only",
        ),
        _check(
            name="promotion_configs_are_guarded",
            passed=_promotion_configs_are_guarded(candidate_config_grid),
            observed=candidate_config_grid,
            expected="top10 promotion budget <= 1 and protected top ranks declared",
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
            name="report_fields_are_public_safe",
            passed=not any(
                field in frozen_protocol.get("public_safe_changed_case_fields", [])
                for field in [
                    "raw_question_text",
                    "raw_answer_text",
                    "document_title",
                    "document_body_text",
                    "section_text",
                    "matched_token_strings",
                ]
            ),
            observed=frozen_protocol.get("public_safe_changed_case_fields"),
            expected="public-safe ids, ranks, buckets, and counts only",
        ),
        _check(
            name="stage91_freezes_protocol_without_metrics",
            passed=True,
            observed="protocol_freeze_only",
            expected="protocol_freeze_only",
        ),
        _check(
            name="stage91_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage91_default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
    ]


def _promotion_configs_are_guarded(configs: Sequence[Mapping[str, Any]]) -> bool:
    if not configs:
        return False
    for config in configs:
        promotion_budget = int(config.get("maximum_top10_promotions_per_query") or 0)
        protected_rank_count = int(config.get("protected_bm25_top_rank_count") or 0)
        if promotion_budget > 1:
            return False
        if protected_rank_count < 5:
            return False
        if config.get("config_id") == "ssgx_shadow_no_top10_demotion_v1":
            if promotion_budget != 0 or config.get("demote_existing_bm25_top10") is not False:
                return False
    return True


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_section_signal_protocol_blocked",
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
        "status": "primeqa_hybrid_section_signal_protocol_frozen",
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
            "Stage 92: after user confirmation, run the frozen train/dev-only "
            "section signal guarded expansion comparison; keep test locked and "
            "do not run final metrics."
        ),
    }


def _config_promotion_budget_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=config["config_id"],
            value=float(config["maximum_top10_promotions_per_query"]),
            value_label=str(config["maximum_top10_promotions_per_query"]),
        )
        for config in report["frozen_protocol"]["candidate_config_grid"]
    ]


def _config_ratio_threshold_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=config["config_id"],
            value=float(config["minimum_section_to_document_score_ratio"]),
            value_label=f"{config['minimum_section_to_document_score_ratio']:.2f}",
        )
        for config in report["frozen_protocol"]["candidate_config_grid"]
    ]


def _feature_group_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    groups = report["frozen_protocol"]["section_signal_feature_contract"][
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
