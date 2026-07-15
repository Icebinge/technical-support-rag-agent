from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 97"
_CREATED_AT = "2026-07-15"
_SOURCE_STAGE84 = "Stage 84"
_SOURCE_STAGE96 = "Stage 96"
_SOURCE_STAGE80 = "Stage 80"
_SOURCE_STAGE81 = "Stage 81"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_PROTOCOL_ID = "selective_dense_sparse_low_overlap_gate_train_dev_v1"
_CANDIDATE_ID = "selective_dense_sparse_low_overlap_gate_design"
_STOPPED_CANDIDATE_ID = "score_margin_bm25_normalization_gate_design"
_BLOCKED_CANDIDATE_ID = "source_doc_ids_oracle_union_blocked"
_ALLOWED_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})


@dataclass(frozen=True)
class PrimeQAHybridSelectiveDenseSparseProtocolVisualization:
    """One generated Stage97 selective dense+sparse protocol visualization."""

    name: str
    path: str


def freeze_primeqa_hybrid_selective_dense_sparse_protocol(
    *,
    stage84_report_path: Path,
    stage96_report_path: Path,
    stage80_report_path: Path,
    stage81_report_path: Path,
    user_confirmed_candidate: bool,
    confirmed_candidate_id: str,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the train/dev protocol for selective dense+sparse low-overlap gating."""

    started_at = time.perf_counter()
    stage84_report = _load_json_object(stage84_report_path)
    stage96_report = _load_json_object(stage96_report_path)
    stage80_report = _load_json_object(stage80_report_path)
    stage81_report = _load_json_object(stage81_report_path)

    candidate = _selected_candidate(stage84_report)
    stage96_candidate = _stage96_next_candidate(stage96_report)
    stage80_dense_caches = _stage80_dense_cache_summaries(stage80_report)
    stage81_dense_configs = _stage81_dense_config_summaries(stage81_report)
    frozen_protocol = _frozen_protocol(
        candidate=candidate,
        stage80_dense_caches=stage80_dense_caches,
        stage81_dense_configs=stage81_dense_configs,
    )
    guard_checks = _guard_checks(
        stage84_report=stage84_report,
        stage96_report=stage96_report,
        stage80_report=stage80_report,
        stage81_report=stage81_report,
        candidate=candidate,
        stage96_candidate=stage96_candidate,
        stage80_dense_caches=stage80_dense_caches,
        stage81_dense_configs=stage81_dense_configs,
        frozen_protocol=frozen_protocol,
        user_confirmed_candidate=user_confirmed_candidate,
        confirmed_candidate_id=confirmed_candidate_id,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_scope": (
            "Train/dev-only protocol freeze for the Stage84 selective "
            "dense+sparse low-overlap gate candidate after Stage96 stopped "
            "score-margin BM25 normalization. This stage reads only public-safe "
            "Stage84, Stage96, Stage80, and Stage81 reports, freezes a "
            "predeclared protocol for a future train/dev metric run, does not "
            "run retrieval metrics, does not load the frozen test split, does "
            "not run final metrics, does not download models, does not use "
            "source DOC_IDS as runtime retrieval evidence, and does not change "
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
            "stage96_report": _fingerprint(stage96_report_path),
            "stage80_report": _fingerprint(stage80_report_path),
            "stage81_report": _fingerprint(stage81_report_path),
        },
        "stage84_decision": stage84_report.get("decision") or {},
        "stage96_decision": stage96_report.get("decision") or {},
        "stage80_decision": stage80_report.get("decision") or {},
        "stage81_decision": stage81_report.get("decision") or {},
        "stage84_candidate_summary": candidate,
        "stage96_next_candidate_summary": stage96_candidate,
        "stage80_dense_cache_summaries": stage80_dense_caches,
        "stage81_dense_config_summaries": stage81_dense_configs,
        "frozen_protocol": frozen_protocol,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
        "timing_seconds": {
            "load_and_freeze": round(checked_at - started_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_selective_dense_sparse_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridSelectiveDenseSparseProtocolVisualization]:
    """Write SVG charts for the Stage97 selective dense+sparse protocol."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage97_selective_dense_sparse_cache_readiness.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage97 selective dense+sparse cache readiness",
                bars=_cache_readiness_bars(report),
                x_label="ready for no-download train/dev run",
                width=1280,
                margin_left=560,
            )
        ),
        "stage97_selective_dense_sparse_gate_thresholds.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage97 selective dense+sparse top1 overlap thresholds",
                bars=_gate_threshold_bars(report),
                x_label="maximum BM25 top1 query-overlap ratio",
                width=1320,
                margin_left=610,
            )
        ),
        "stage97_selective_dense_sparse_rrf_weights.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage97 selective dense+sparse dense weights",
                bars=_dense_weight_bars(report),
                x_label="dense RRF weight",
                width=1320,
                margin_left=610,
            )
        ),
        "stage97_selective_dense_sparse_feature_group_counts.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage97 selective dense+sparse feature group counts",
                bars=_feature_group_bars(report),
                x_label="feature count",
                width=1180,
                margin_left=420,
            )
        ),
        "stage97_selective_dense_sparse_protocol_decision_flags.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage97 selective dense+sparse protocol decision flags",
                bars=_decision_flag_bars(report),
                x_label="1 means true",
                width=1180,
                margin_left=520,
            )
        ),
        "stage97_selective_dense_sparse_guard_check_status.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage97 selective dense+sparse guard check status",
                bars=_guard_check_bars(report),
                x_label="1 means passed",
                width=1400,
                margin_left=700,
            )
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridSelectiveDenseSparseProtocolVisualization(
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


def _stage96_next_candidate(stage96_report: Mapping[str, Any]) -> dict[str, Any]:
    candidate_queue = stage96_report.get("candidate_queue") or {}
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


def _stage80_dense_cache_summaries(stage80_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for cache in stage80_report.get("dense_cache_candidates") or []:
        if not isinstance(cache, Mapping):
            continue
        summaries.append(
            {
                "model_name": cache.get("model_name"),
                "cache_path": cache.get("cache_path"),
                "cache_sha256": cache.get("cache_sha256"),
                "document_text_max_chars": cache.get("document_text_max_chars"),
                "document_prefix": cache.get("document_prefix"),
                "embedding_shape": cache.get("embedding_shape"),
                "document_id_count": cache.get("document_id_count"),
                "document_ids_match_current_corpus": cache.get(
                    "document_ids_match_current_corpus"
                ),
                "can_run_without_reencoding_documents": cache.get(
                    "can_run_without_reencoding_documents"
                ),
                "can_run_without_model_download": cache.get(
                    "can_run_without_model_download"
                ),
            }
        )
    return summaries


def _stage81_dense_config_summaries(stage81_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for config in stage81_report.get("dense_cache_configs") or []:
        if not isinstance(config, Mapping):
            continue
        summaries.append(
            {
                "config_id": config.get("config_id"),
                "model_name": config.get("model_name"),
                "cache_path": config.get("cache_path"),
                "cache_sha256": config.get("cache_sha256"),
                "document_text_max_chars": config.get("document_text_max_chars"),
                "document_prefix": config.get("document_prefix"),
                "query_prefix": config.get("query_prefix"),
                "query_prefix_source": config.get("query_prefix_source"),
                "embedding_shape": config.get("embedding_shape"),
                "document_id_count": config.get("document_id_count"),
                "can_run_without_model_download_in_stage80": config.get(
                    "can_run_without_model_download_in_stage80"
                ),
                "snapshot_path": config.get("snapshot_path"),
                "snapshot_status": config.get("snapshot_status"),
            }
        )
    return summaries


def _frozen_protocol(
    *,
    candidate: Mapping[str, Any],
    stage80_dense_caches: Sequence[Mapping[str, Any]],
    stage81_dense_configs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "candidate_id": _CANDIDATE_ID,
        "protocol_status": "frozen_requires_user_confirmation_before_metric_run",
        "source_stages": [_SOURCE_STAGE84, _SOURCE_STAGE96, _SOURCE_STAGE80, _SOURCE_STAGE81],
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
        "dense_cache_contract": _dense_cache_contract(
            stage80_dense_caches=stage80_dense_caches,
            stage81_dense_configs=stage81_dense_configs,
        ),
        "candidate_policy_grid": _candidate_policy_grid(stage81_dense_configs),
        "low_overlap_gate_feature_contract": _low_overlap_gate_feature_contract(),
        "train_selection_rule": {
            "selection_split": "train",
            "validation_split": "dev",
            "rule": (
                "Select the gated dense+sparse policy on train only by hit@10, "
                "then larger not-found@50 reduction, fewer top10 regressions, "
                "hit@1 non-collapse, MRR@10, lower dense promotion budget, then "
                "policy_id. Dev is validation only."
            ),
            "dev_selection_forbidden": True,
            "test_selection_forbidden": True,
            "dev_threshold_selection_forbidden": True,
            "stage81_dev_result_selection_forbidden": True,
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
            "not_found_count_at_50_delta",
            "hit@1_delta",
            "dense_sparse_gate_activation_count",
            "dense_sparse_top10_promotion_count",
            "protected_bm25_top_rank_demotion_count",
        ],
        "public_safe_changed_case_fields": [
            "sample_id",
            "split",
            "baseline_rank",
            "challenger_rank",
            "policy_id",
            "dense_config_id",
            "baseline_rank_bucket",
            "challenger_rank_bucket",
            "gate_activation_reason_code",
            "query_length_bucket",
            "bm25_top1_overlap_bucket",
            "bm25_top10_mean_overlap_bucket",
            "dense_rank_bucket",
            "promotion_budget_used",
        ],
        "explicit_exclusions": [
            "Do not use source DOC_IDS as runtime retrieval evidence.",
            "Do not use answer document IDs or gold ranks as runtime features.",
            "Do not choose gate thresholds from dev performance.",
            "Do not choose dense cache/model from dev performance.",
            "Do not download models or refresh dense caches in this protocol.",
            "Do not load or evaluate the frozen test split.",
            "Do not write raw question text, answer text, document titles, "
            "document body text, query terms, or matched token strings to the report.",
            "Do not change runtime defaults in this stage.",
        ],
    }


def _dense_cache_contract(
    *,
    stage80_dense_caches: Sequence[Mapping[str, Any]],
    stage81_dense_configs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "allowed_cache_source": "stage80_compatible_local_dense_caches_only",
        "allowed_dense_configs": list(stage81_dense_configs),
        "stage80_cache_count": len(stage80_dense_caches),
        "stage81_dense_config_count": len(stage81_dense_configs),
        "download_required": False,
        "document_reencoding_allowed": False,
        "query_encoding_mode": "local_snapshot_path_with_local_files_only",
        "model_selection_mode": "predeclared_grid_then_train_selection_only",
        "cache_preflight_required_before_metric_run": [
            "cache file exists",
            "cache sha256 matches Stage81 config",
            "document IDs match current corpus",
            "embedding rows match current corpus",
            "local Hugging Face snapshot exists",
        ],
    }


def _candidate_policy_grid(
    stage81_dense_configs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    config_ids = {str(config.get("config_id")) for config in stage81_dense_configs}
    e5_config_id = "dense_sparse_rrf__intfloat_e5_small_v2__512_passage"
    minilm_config_id = "dense_sparse_rrf__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix"
    policies = [
        {
            "policy_id": "sdsl_e5_low_overlap_balanced_v1",
            "dense_config_id": e5_config_id,
            "gate_mode": "low_bm25_lexical_overlap",
            "minimum_query_token_count": 8,
            "maximum_bm25_top1_query_overlap_ratio": 0.25,
            "maximum_bm25_top10_mean_query_overlap_ratio": 0.22,
            "dense_candidate_rank_max": 10,
            "sparse_weight": 1.0,
            "dense_weight": 1.0,
            "rrf_k": 60,
            "maximum_dense_top10_promotions_per_query": 2,
            "protected_bm25_top_rank_count": 5,
            "dense_candidate_must_be_outside_bm25_top10": True,
        },
        {
            "policy_id": "sdsl_minilm_low_overlap_balanced_v1",
            "dense_config_id": minilm_config_id,
            "gate_mode": "low_bm25_lexical_overlap",
            "minimum_query_token_count": 8,
            "maximum_bm25_top1_query_overlap_ratio": 0.25,
            "maximum_bm25_top10_mean_query_overlap_ratio": 0.22,
            "dense_candidate_rank_max": 10,
            "sparse_weight": 1.0,
            "dense_weight": 1.0,
            "rrf_k": 60,
            "maximum_dense_top10_promotions_per_query": 2,
            "protected_bm25_top_rank_count": 5,
            "dense_candidate_must_be_outside_bm25_top10": True,
        },
        {
            "policy_id": "sdsl_e5_low_overlap_dense_bias_v1",
            "dense_config_id": e5_config_id,
            "gate_mode": "strict_low_overlap_dense_bias",
            "minimum_query_token_count": 10,
            "maximum_bm25_top1_query_overlap_ratio": 0.20,
            "maximum_bm25_top10_mean_query_overlap_ratio": 0.18,
            "dense_candidate_rank_max": 8,
            "sparse_weight": 1.0,
            "dense_weight": 1.25,
            "rrf_k": 60,
            "maximum_dense_top10_promotions_per_query": 2,
            "protected_bm25_top_rank_count": 5,
            "dense_candidate_must_be_outside_bm25_top10": True,
        },
        {
            "policy_id": "sdsl_minilm_low_overlap_conservative_v1",
            "dense_config_id": minilm_config_id,
            "gate_mode": "conservative_low_overlap",
            "minimum_query_token_count": 6,
            "maximum_bm25_top1_query_overlap_ratio": 0.30,
            "maximum_bm25_top10_mean_query_overlap_ratio": 0.25,
            "dense_candidate_rank_max": 8,
            "sparse_weight": 1.0,
            "dense_weight": 0.85,
            "rrf_k": 60,
            "maximum_dense_top10_promotions_per_query": 1,
            "protected_bm25_top_rank_count": 7,
            "dense_candidate_must_be_outside_bm25_top10": True,
        },
    ]
    return [
        {**policy, "dense_config_present_in_stage81": policy["dense_config_id"] in config_ids}
        for policy in policies
    ]


def _low_overlap_gate_feature_contract() -> dict[str, Any]:
    return {
        "runtime_allowed_feature_groups": {
            "query_aggregate_features": [
                "query_token_count",
                "query_unique_token_count",
                "query_length_bucket",
            ],
            "bm25_lexical_features": [
                "bm25_top1_query_overlap_count",
                "bm25_top1_query_overlap_ratio",
                "bm25_top10_mean_query_overlap_ratio",
                "candidate_query_overlap_count",
                "candidate_query_overlap_ratio",
                "candidate_title_query_overlap_count",
                "candidate_title_query_overlap_ratio",
            ],
            "sparse_rank_score_features": [
                "bm25_rank",
                "bm25_score",
                "bm25_rank_bucket",
                "bm25_score_margin_to_rank10",
            ],
            "dense_rank_score_features": [
                "dense_config_id",
                "dense_rank",
                "dense_score",
                "dense_rank_bucket",
            ],
            "rrf_gate_features": [
                "rrf_rank",
                "rrf_score",
                "sparse_rrf_contribution",
                "dense_rrf_contribution",
                "dense_sparse_contribution_ratio",
            ],
            "action_budget_features": [
                "dense_top10_promotion_budget_remaining",
                "protected_bm25_top_rank_count",
                "gate_activation_reason_code",
            ],
        },
        "prohibited_runtime_features": [
            "source_DOC_IDS",
            "answer document IDs",
            "gold_document_rank",
            "gold_label",
            "dev_selected_gate_threshold",
            "dev_selected_dense_model",
            "stage81_dev_selected_config",
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


def _guard_checks(
    *,
    stage84_report: Mapping[str, Any],
    stage96_report: Mapping[str, Any],
    stage80_report: Mapping[str, Any],
    stage81_report: Mapping[str, Any],
    candidate: Mapping[str, Any],
    stage96_candidate: Mapping[str, Any],
    stage80_dense_caches: Sequence[Mapping[str, Any]],
    stage81_dense_configs: Sequence[Mapping[str, Any]],
    frozen_protocol: Mapping[str, Any],
    user_confirmed_candidate: bool,
    confirmed_candidate_id: str,
) -> list[dict[str, Any]]:
    stage84_decision = stage84_report.get("decision") or {}
    stage96_decision = stage96_report.get("decision") or {}
    stage80_decision = stage80_report.get("decision") or {}
    stage81_decision = stage81_report.get("decision") or {}
    target_metric_contract = candidate.get("target_metric_contract") or []
    runtime_evidence_policy = candidate.get("runtime_evidence_policy") or []
    explicit_exclusions = frozen_protocol.get("explicit_exclusions") or []
    feature_contract = frozen_protocol.get("low_overlap_gate_feature_contract") or {}
    prohibited_runtime_features = feature_contract.get("prohibited_runtime_features", [])
    candidate_policy_grid = frozen_protocol.get("candidate_policy_grid") or []
    train_selection_rule = frozen_protocol.get("train_selection_rule") or {}
    dense_cache_contract = frozen_protocol.get("dense_cache_contract") or {}
    return [
        _check(
            name="source_stage84_report_is_stage84",
            passed=stage84_report.get("stage") == _SOURCE_STAGE84,
            observed=stage84_report.get("stage"),
            expected=_SOURCE_STAGE84,
        ),
        _check(
            name="source_stage96_report_is_stage96",
            passed=stage96_report.get("stage") == _SOURCE_STAGE96,
            observed=stage96_report.get("stage"),
            expected=_SOURCE_STAGE96,
        ),
        _check(
            name="source_stage80_report_is_stage80",
            passed=stage80_report.get("stage") == _SOURCE_STAGE80,
            observed=stage80_report.get("stage"),
            expected=_SOURCE_STAGE80,
        ),
        _check(
            name="source_stage81_report_is_stage81",
            passed=stage81_report.get("stage") == _SOURCE_STAGE81,
            observed=stage81_report.get("stage"),
            expected=_SOURCE_STAGE81,
        ),
        _check(
            name="user_confirmed_selective_dense_sparse_protocol",
            passed=user_confirmed_candidate,
            observed=user_confirmed_candidate,
            expected=True,
        ),
        _check(
            name="stage96_stopped_score_margin_bm25_route",
            passed=stage96_decision.get("status")
            == "primeqa_hybrid_score_margin_bm25_route_stopped"
            and stage96_decision.get("stopped_candidate_id") == _STOPPED_CANDIDATE_ID,
            observed={
                "status": stage96_decision.get("status"),
                "stopped_candidate_id": stage96_decision.get("stopped_candidate_id"),
            },
            expected=_STOPPED_CANDIDATE_ID,
        ),
        _check(
            name="confirmed_candidate_matches_stage96_next_candidate",
            passed=confirmed_candidate_id
            == stage96_decision.get("next_candidate_id")
            == _CANDIDATE_ID,
            observed={
                "confirmed_candidate_id": confirmed_candidate_id,
                "stage96_next_candidate_id": stage96_decision.get("next_candidate_id"),
            },
            expected=_CANDIDATE_ID,
        ),
        _check(
            name="stage96_next_candidate_summary_matches",
            passed=stage96_candidate.get("candidate_id") == _CANDIDATE_ID,
            observed=stage96_candidate.get("candidate_id"),
            expected=_CANDIDATE_ID,
        ),
        _check(
            name="stage96_requires_confirmation_before_next_protocol",
            passed=stage96_decision.get("requires_user_confirmation_before_next_protocol")
            is True,
            observed=stage96_decision.get(
                "requires_user_confirmation_before_next_protocol"
            ),
            expected=True,
        ),
        _check(
            name="stage96_final_test_metrics_locked",
            passed=stage96_decision.get("can_run_final_test_metrics_now") is False,
            observed=stage96_decision.get("can_run_final_test_metrics_now"),
            expected=False,
        ),
        _check(
            name="stage96_final_test_gate_closed",
            passed=stage96_decision.get("can_open_final_test_gate_now") is False,
            observed=stage96_decision.get("can_open_final_test_gate_now"),
            expected=False,
        ),
        _check(
            name="stage96_forbids_test_tuning",
            passed=stage96_decision.get("can_use_test_for_tuning") is False,
            observed=stage96_decision.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage96_runtime_default_unchanged",
            passed=stage96_decision.get("default_runtime_policy") == "unchanged",
            observed=stage96_decision.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage80_dense_sparse_feasibility_completed",
            passed=stage80_decision.get("status")
            == "primeqa_hybrid_dense_sparse_rrf_feasibility_completed",
            observed=stage80_decision.get("status"),
            expected="primeqa_hybrid_dense_sparse_rrf_feasibility_completed",
        ),
        _check(
            name="stage80_can_run_without_download",
            passed=stage80_decision.get("can_run_dense_sparse_rrf_without_download")
            is True,
            observed=stage80_decision.get("can_run_dense_sparse_rrf_without_download"),
            expected=True,
        ),
        _check(
            name="stage80_compatible_cache_count_at_least_two",
            passed=int(stage80_decision.get("compatible_local_dense_cache_count") or 0)
            >= 2,
            observed=stage80_decision.get("compatible_local_dense_cache_count"),
            expected=">= 2",
        ),
        _check(
            name="stage80_final_test_metrics_locked",
            passed=stage80_decision.get("can_run_final_test_metrics_now") is False,
            observed=stage80_decision.get("can_run_final_test_metrics_now"),
            expected=False,
        ),
        _check(
            name="stage80_runtime_default_unchanged",
            passed=stage80_decision.get("default_runtime_policy") == "unchanged",
            observed=stage80_decision.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage81_dense_sparse_comparison_completed",
            passed=stage81_decision.get("status")
            == "primeqa_hybrid_dense_sparse_rrf_comparison_completed",
            observed=stage81_decision.get("status"),
            expected="primeqa_hybrid_dense_sparse_rrf_comparison_completed",
        ),
        _check(
            name="stage81_final_test_metrics_locked",
            passed=stage81_decision.get("can_run_final_test_metrics_now") is False,
            observed=stage81_decision.get("can_run_final_test_metrics_now"),
            expected=False,
        ),
        _check(
            name="stage81_forbids_test_tuning",
            passed=stage81_decision.get("can_use_test_for_tuning") is False,
            observed=stage81_decision.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage81_runtime_default_unchanged",
            passed=stage81_decision.get("default_runtime_policy") == "unchanged",
            observed=stage81_decision.get("default_runtime_policy"),
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
            passed=_contract_contains(
                target_metric_contract,
                "train-selected gated policy must improve dev hit@10",
            ),
            observed=target_metric_contract,
            expected="train-selected gated policy must improve dev hit@10",
        ),
        _check(
            name="stage84_candidate_contract_requires_not_found_decrease",
            passed=_contract_contains(
                target_metric_contract,
                "dev not-found@50 should decrease without hit@1 collapse",
            ),
            observed=target_metric_contract,
            expected="dev not-found@50 decrease without hit@1 collapse",
        ),
        _check(
            name="stage84_candidate_guard_blocks_downloads_and_dev_thresholds",
            passed=_contract_contains(
                target_metric_contract,
                "no downloads and no dev-selected gate thresholds",
            ),
            observed=target_metric_contract,
            expected="no downloads and no dev-selected gate thresholds",
        ),
        _check(
            name="stage84_runtime_policy_allows_only_runtime_observable_features",
            passed=any("query tokens" in str(item) for item in runtime_evidence_policy)
            and any("local dense scores" in str(item) for item in runtime_evidence_policy)
            and any("Must not use source DOC_IDS" in str(item) for item in runtime_evidence_policy),
            observed=runtime_evidence_policy,
            expected="runtime query aggregate, overlap, scores, no source DOC_IDS",
        ),
        _check(
            name="protocol_id_is_fixed",
            passed=frozen_protocol.get("protocol_id") == _PROTOCOL_ID,
            observed=frozen_protocol.get("protocol_id"),
            expected=_PROTOCOL_ID,
        ),
        _check(
            name="dense_cache_contract_uses_stage80_and_stage81_only",
            passed=dense_cache_contract.get("allowed_cache_source")
            == "stage80_compatible_local_dense_caches_only"
            and dense_cache_contract.get("download_required") is False
            and dense_cache_contract.get("document_reencoding_allowed") is False,
            observed=dense_cache_contract,
            expected="existing local dense caches only",
        ),
        _check(
            name="stage80_stage81_dense_cache_identities_match",
            passed=_cache_identities_match_stage80_stage81(
                stage80_dense_caches,
                stage81_dense_configs,
            ),
            observed={
                "stage80": [
                    cache.get("cache_path") for cache in stage80_dense_caches
                ],
                "stage81": [
                    config.get("cache_path") for config in stage81_dense_configs
                ],
            },
            expected="Stage81 dense configs must be backed by Stage80 caches",
        ),
        _check(
            name="candidate_policy_grid_is_predeclared",
            passed=len(candidate_policy_grid) == 4
            and len({policy.get("policy_id") for policy in candidate_policy_grid}) == 4,
            observed=[policy.get("policy_id") for policy in candidate_policy_grid],
            expected=4,
        ),
        _check(
            name="candidate_policy_grid_reuses_existing_dense_configs",
            passed=all(
                policy.get("dense_config_present_in_stage81") is True
                for policy in candidate_policy_grid
            ),
            observed=[
                {
                    "policy_id": policy.get("policy_id"),
                    "dense_config_id": policy.get("dense_config_id"),
                    "present": policy.get("dense_config_present_in_stage81"),
                }
                for policy in candidate_policy_grid
            ],
            expected="all policies reference Stage81 dense configs",
        ),
        _check(
            name="candidate_policy_grid_has_low_overlap_gates",
            passed=_policy_grid_has_low_overlap_gates(candidate_policy_grid),
            observed=candidate_policy_grid,
            expected="each policy has query length, overlap, dense rank, and budget gates",
        ),
        _check(
            name="train_selection_rule_forbids_dev_and_test_selection",
            passed=train_selection_rule.get("dev_selection_forbidden") is True
            and train_selection_rule.get("test_selection_forbidden") is True
            and train_selection_rule.get("dev_threshold_selection_forbidden") is True
            and train_selection_rule.get("stage81_dev_result_selection_forbidden")
            is True,
            observed=train_selection_rule,
            expected="train-only policy selection",
        ),
        _check(
            name="low_overlap_feature_contract_uses_runtime_observable_features",
            passed="runtime_allowed_feature_groups" in feature_contract
            and "gold_document_rank" in prohibited_runtime_features
            and "dev_selected_gate_threshold" in prohibited_runtime_features,
            observed=feature_contract,
            expected="runtime observable aggregate features only",
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
            name="downloads_and_cache_refresh_forbidden",
            passed=any("Do not download models" in str(item) for item in explicit_exclusions)
            and dense_cache_contract.get("download_required") is False
            and dense_cache_contract.get("document_reencoding_allowed") is False,
            observed={
                "dense_cache_contract": dense_cache_contract,
                "explicit_exclusions": explicit_exclusions,
            },
            expected="no downloads or dense cache refresh",
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
            expected="public-safe ids, buckets, and aggregate counts only",
        ),
        _check(
            name="source_doc_ids_oracle_blocked_candidate_not_selected",
            passed=confirmed_candidate_id != _BLOCKED_CANDIDATE_ID,
            observed=confirmed_candidate_id,
            expected=f"not {_BLOCKED_CANDIDATE_ID}",
        ),
        _check(
            name="stage97_freezes_protocol_without_metrics",
            passed=True,
            observed="protocol_freeze_only",
            expected="protocol_freeze_only",
        ),
        _check(
            name="stage97_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage97_default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
    ]


def _contract_contains(contract: Sequence[Any], text: str) -> bool:
    return any(text in str(item) for item in contract)


def _cache_identities_match_stage80_stage81(
    stage80_dense_caches: Sequence[Mapping[str, Any]],
    stage81_dense_configs: Sequence[Mapping[str, Any]],
) -> bool:
    stage80_paths = {str(cache.get("cache_path")) for cache in stage80_dense_caches}
    stage81_paths = {str(config.get("cache_path")) for config in stage81_dense_configs}
    return bool(stage81_paths) and stage81_paths.issubset(stage80_paths)


def _policy_grid_has_low_overlap_gates(
    policies: Sequence[Mapping[str, Any]],
) -> bool:
    if not policies:
        return False
    for policy in policies:
        if int(policy.get("minimum_query_token_count") or 0) <= 0:
            return False
        if policy.get("maximum_bm25_top1_query_overlap_ratio") is None:
            return False
        if policy.get("maximum_bm25_top10_mean_query_overlap_ratio") is None:
            return False
        if int(policy.get("dense_candidate_rank_max") or 0) <= 0:
            return False
        if int(policy.get("maximum_dense_top10_promotions_per_query") or 0) <= 0:
            return False
        if int(policy.get("protected_bm25_top_rank_count") or 0) < 5:
            return False
        if policy.get("dense_candidate_must_be_outside_bm25_top10") is not True:
            return False
    return True


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_selective_dense_sparse_protocol_blocked",
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
        "status": "primeqa_hybrid_selective_dense_sparse_protocol_frozen",
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
            "Stage 98: after user confirmation, run the frozen train/dev-only "
            "selective dense+sparse low-overlap gate comparison; keep test locked, "
            "do not use source DOC_IDS, do not download models, and do not run "
            "final metrics."
        ),
    }


def _cache_readiness_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for config in report["stage81_dense_config_summaries"]:
        ready = 1.0 if config.get("can_run_without_model_download_in_stage80") else 0.0
        bars.append(
            BarDatum(
                label=str(config.get("config_id")),
                value=ready,
                value_label="ready" if ready else "blocked",
            )
        )
    return bars


def _gate_threshold_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=policy["policy_id"],
            value=float(policy["maximum_bm25_top1_query_overlap_ratio"]),
            value_label=f"{policy['maximum_bm25_top1_query_overlap_ratio']:.2f}",
        )
        for policy in report["frozen_protocol"]["candidate_policy_grid"]
    ]


def _dense_weight_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=policy["policy_id"],
            value=float(policy["dense_weight"]),
            value_label=f"{policy['dense_weight']:.2f}",
        )
        for policy in report["frozen_protocol"]["candidate_policy_grid"]
    ]


def _feature_group_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    groups = report["frozen_protocol"]["low_overlap_gate_feature_contract"][
        "runtime_allowed_feature_groups"
    ]
    return [
        BarDatum(label=group, value=float(len(features)), value_label=str(len(features)))
        for group, features in sorted(groups.items())
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report["decision"]
    names = [
        "can_run_train_dev_metrics_after_user_confirmation",
        "can_run_final_test_metrics_now",
        "can_use_test_for_tuning",
    ]
    return [
        BarDatum(
            label=name,
            value=1.0 if decision[name] else 0.0,
            value_label="yes" if decision[name] else "no",
        )
        for name in names
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
