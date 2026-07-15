from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 104"
_CREATED_AT = "2026-07-15"
_SOURCE_STAGE103 = "Stage 103"
_SOURCE_DESIGN_ID = "evidence_selection_and_answerability_candidate_design_v1"
_PROTOCOL_ID = "evidence_answerability_candidate_train_dev_comparison_v1"
_SOURCE_RECOMMENDED_DIRECTION = "evidence_answerability_train_dev_candidate_comparison"
_NEXT_DIRECTION = "run_evidence_answerability_train_dev_candidate_comparison"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = ("test",)
_CANDIDATE_IDS = (
    "answerability_margin_gate_candidate_v1",
    "evidence_window_reselector_candidate_v1",
    "joint_gate_then_window_candidate_v1",
)
_EXPECTED_EXECUTION_ORDER = (
    "joint_gate_then_window_candidate_v1",
    "evidence_window_reselector_candidate_v1",
    "answerability_margin_gate_candidate_v1",
)
_ALLOWED_SELECTORS = frozenset(
    {
        "bm25_sentence",
        "answer_window",
        "hybrid_window",
    }
)
_FORBIDDEN_PUBLIC_FIELDS = frozenset(
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
class PrimeQAHybridEvidenceAnswerabilityComparisonProtocolVisualization:
    """One generated Stage104 evidence/answerability comparison protocol chart."""

    name: str
    path: str


def freeze_primeqa_hybrid_evidence_answerability_comparison_protocol(
    *,
    stage103_protocol_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage104 train/dev comparison grid before metric execution."""

    started_at = time.perf_counter()
    stage103_report = _load_json_object(stage103_protocol_path)
    stage103_summary = _stage103_public_summary(stage103_report)
    frozen_protocol = _frozen_protocol(stage103_summary)
    guard_checks = _guard_checks(
        stage103_summary=stage103_summary,
        frozen_protocol=frozen_protocol,
        user_confirmed_protocol=user_confirmed_protocol,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "protocol_scope": (
            "Train/dev-only comparison-grid freeze for the Stage103 "
            "evidence-answerability candidate family. This stage reads only "
            "the saved public-safe Stage103 protocol report, freezes the "
            "candidate threshold grid and selection contract, does not load "
            "train/dev/test split files, does not load corpus documents, does "
            "not run retrieval or answer metrics, does not run final metrics, "
            "does not use oracle document identifiers or gold answers as "
            "runtime evidence, does not add fallback strategies, and does not "
            "change runtime defaults."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_protocol),
            "confirmation_note": confirmation_note,
        },
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "development_splits": list(_DEVELOPMENT_SPLITS),
            "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage103_protocol": _fingerprint(stage103_protocol_path),
        },
        "stage103_summary": stage103_summary,
        "frozen_protocol": frozen_protocol,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
        "timing_seconds": {
            "load_and_freeze": round(checked_at - started_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_evidence_answerability_comparison_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridEvidenceAnswerabilityComparisonProtocolVisualization]:
    """Write SVG charts for the Stage104 comparison-grid protocol."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage104_config_counts_by_candidate.svg": render_horizontal_bar_chart_svg(
            title="Stage104 config counts by candidate",
            bars=_config_counts_by_candidate_bars(report),
            x_label="config count",
            width=1240,
            margin_left=560,
        ),
        "stage104_config_min_evidence_scores.svg": render_horizontal_bar_chart_svg(
            title="Stage104 min evidence scores",
            bars=_config_min_evidence_score_bars(report),
            x_label="min evidence score",
            width=1320,
            margin_left=660,
        ),
        "stage104_config_max_citation_ranks.svg": render_horizontal_bar_chart_svg(
            title="Stage104 max citation ranks",
            bars=_config_max_citation_rank_bars(report),
            x_label="maximum citation rank",
            width=1320,
            margin_left=660,
        ),
        "stage104_selector_mix.svg": render_horizontal_bar_chart_svg(
            title="Stage104 selector mix",
            bars=_selector_mix_bars(report),
            x_label="config count",
            width=1100,
            margin_left=360,
        ),
        "stage104_train_selection_guard_thresholds.svg": render_horizontal_bar_chart_svg(
            title="Stage104 train selection guard thresholds",
            bars=_train_selection_guard_bars(report),
            x_label="maximum allowed train regression",
            width=1320,
            margin_left=620,
        ),
        "stage104_protocol_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage104 protocol decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1320,
            margin_left=660,
        ),
        "stage104_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage104 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1560,
            margin_left=820,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridEvidenceAnswerabilityComparisonProtocolVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage103_public_summary(stage103_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage103_report.get("decision") or {}
    protocol = stage103_report.get("frozen_candidate_protocol") or {}
    stage104_contract = protocol.get("stage104_train_dev_comparison_contract") or {}
    bottleneck_summary = stage103_report.get("bottleneck_summary") or {}
    return {
        "stage": stage103_report.get("stage"),
        "design_id": stage103_report.get("design_id"),
        "decision_status": decision.get("status"),
        "recommended_direction": decision.get("recommended_direction"),
        "recommended_execution_order": decision.get("recommended_execution_order"),
        "can_continue_train_dev_development": decision.get(
            "can_continue_train_dev_development"
        ),
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
        "candidate_policies": [
            _candidate_policy_summary(candidate)
            for candidate in protocol.get("candidate_policies") or []
        ],
        "blocked_items": [
            {
                "blocked_item_id": item.get("blocked_item_id"),
                "status": item.get("status"),
            }
            for item in protocol.get("blocked_items") or []
        ],
        "stage104_contract": {
            "comparison_id": stage104_contract.get("comparison_id"),
            "run_mode": stage104_contract.get("run_mode"),
            "baseline_reference": stage104_contract.get("baseline_reference"),
            "train_selection_rule": stage104_contract.get("train_selection_rule"),
            "dev_validation_rule": stage104_contract.get("dev_validation_rule"),
            "promotion_rule": stage104_contract.get("promotion_rule"),
            "metric_contract": stage104_contract.get("metric_contract"),
        },
        "bottleneck_summary": {
            "primary_bottleneck_bucket_ids": bottleneck_summary.get(
                "primary_bottleneck_bucket_ids"
            ),
            "secondary_context_bucket_ids": bottleneck_summary.get(
                "secondary_context_bucket_ids"
            ),
            "bucket_rows": [
                {
                    "bucket_id": row.get("bucket_id"),
                    "train_count": row.get("train_count"),
                    "dev_count": row.get("dev_count"),
                    "combined_count": row.get("combined_count"),
                    "combined_priority_score": row.get("combined_priority_score"),
                    "shared_train_dev": row.get("shared_train_dev"),
                }
                for row in bottleneck_summary.get("bucket_rows") or []
            ],
        },
    }


def _candidate_policy_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "status": candidate.get("status"),
        "risk_level": candidate.get("risk_level"),
        "target_buckets": candidate.get("target_buckets"),
        "target_combined_case_count": candidate.get("target_combined_case_count"),
        "priority_score": candidate.get("priority_score"),
    }


def _frozen_protocol(stage103_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "protocol_status": "frozen_requires_user_confirmation_before_metric_run",
        "source_stages": [_SOURCE_STAGE103],
        "comparison_mode": "protocol_freeze_only_no_metrics",
        "baseline_reference": _baseline_reference(),
        "candidate_config_grid": _candidate_config_grid(),
        "grid_derivation_policy": {
            "source": (
                "Predeclared engineering grid derived from Stage103 candidate "
                "families and the Stage102 verified baseline configuration."
            ),
            "train_labels_used_to_choose_threshold_values": False,
            "dev_labels_used_to_choose_threshold_values": False,
            "test_labels_used_to_choose_threshold_values": False,
            "metric_run_performed_in_stage104": False,
        },
        "train_selection_rule": _train_selection_rule(),
        "dev_validation_rule": _dev_validation_rule(),
        "metric_contract": _metric_contract(),
        "runtime_feature_contract": _runtime_feature_contract(),
        "public_safe_output_contract": _public_safe_output_contract(),
        "explicit_exclusions": [
            "no_train_dev_split_loading_in_stage104",
            "no_test_split_loading",
            "no_final_test_metrics",
            "no_metric_run",
            "no_dev_threshold_tuning",
            "no_runtime_default_change",
            "no_fallback_strategy",
            "no_oracle_doc_id_or_gold_answer_runtime_feature",
            "no_raw_question_answer_or_document_text_in_outputs",
        ],
        "fallback_strategy_policy": {
            "fallback_strategies_enabled": False,
            "requires_user_confirmation_before_any_fallback": True,
        },
        "next_stage_contract": {
            "stage": "Stage 105",
            "recommended_direction": _NEXT_DIRECTION,
            "requires_user_confirmation_before_train_dev_metric_run": True,
            "source_protocol_id": _PROTOCOL_ID,
            "must_select_thresholds_on_train_only": True,
            "must_validate_train_selected_config_on_dev_once": True,
            "must_keep_test_locked": True,
            "must_keep_runtime_defaults_unchanged": True,
        },
        "stage103_candidate_summary": {
            "recommended_execution_order": stage103_summary.get(
                "recommended_execution_order"
            ),
            "candidate_count": len(stage103_summary.get("candidate_policies") or []),
        },
    }


def _baseline_reference() -> dict[str, Any]:
    return {
        "baseline_id": "stage102_verified_bm25_top10_answer_pipeline",
        "retriever": "BM25",
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "retrieval_top_k": 10,
        "evidence_selector_name": "bm25_sentence",
        "max_candidates_per_document": 3,
        "composition_policy_name": "top_k",
        "max_sentences": 3,
        "min_sentence_score": 2.0,
        "verifier_min_evidence_score": 7.0,
        "verifier_max_citation_rank": 3,
        "verifier_min_citations": 1,
        "source_stage": "Stage 102",
    }


def _candidate_config_grid() -> list[dict[str, Any]]:
    return [
        _config(
            config_id="amg_bm25_evidence8_rank3_v1",
            candidate_id="answerability_margin_gate_candidate_v1",
            selector_name="bm25_sentence",
            max_candidates_per_document=3,
            min_evidence_score=8.0,
            max_citation_rank=3,
            target_bucket_weights={"answerability_false_answer": 1.55},
            design_intent="moderately stricter evidence sufficiency gate",
        ),
        _config(
            config_id="amg_bm25_evidence9_rank3_v1",
            candidate_id="answerability_margin_gate_candidate_v1",
            selector_name="bm25_sentence",
            max_candidates_per_document=3,
            min_evidence_score=9.0,
            max_citation_rank=3,
            target_bucket_weights={"answerability_false_answer": 1.55},
            design_intent="stronger evidence sufficiency gate",
        ),
        _config(
            config_id="amg_bm25_evidence8_rank2_v1",
            candidate_id="answerability_margin_gate_candidate_v1",
            selector_name="bm25_sentence",
            max_candidates_per_document=3,
            min_evidence_score=8.0,
            max_citation_rank=2,
            target_bucket_weights={"answerability_false_answer": 1.55},
            design_intent="stricter citation-rank gate",
        ),
        _config(
            config_id="ewr_answer_window_mcpd3_evidence7_rank3_v1",
            candidate_id="evidence_window_reselector_candidate_v1",
            selector_name="answer_window",
            max_candidates_per_document=3,
            min_evidence_score=7.0,
            max_citation_rank=3,
            target_bucket_weights={
                "gold_span_beats_selected_answer": 1.45,
                "evidence_selection_miss": 1.70,
            },
            design_intent="answer-window selector at Stage102 verifier settings",
        ),
        _config(
            config_id="ewr_hybrid_window_mcpd3_evidence7_rank3_v1",
            candidate_id="evidence_window_reselector_candidate_v1",
            selector_name="hybrid_window",
            max_candidates_per_document=3,
            min_evidence_score=7.0,
            max_citation_rank=3,
            target_bucket_weights={
                "gold_span_beats_selected_answer": 1.45,
                "evidence_selection_miss": 1.70,
            },
            design_intent="route-aware hybrid window selector",
        ),
        _config(
            config_id="ewr_answer_window_mcpd5_evidence7_rank3_v1",
            candidate_id="evidence_window_reselector_candidate_v1",
            selector_name="answer_window",
            max_candidates_per_document=5,
            min_evidence_score=7.0,
            max_citation_rank=3,
            target_bucket_weights={
                "gold_span_beats_selected_answer": 1.45,
                "evidence_selection_miss": 1.70,
            },
            design_intent="broader answer-window candidate pool",
        ),
        _config(
            config_id="jgw_answer_window_mcpd3_evidence8_rank3_v1",
            candidate_id="joint_gate_then_window_candidate_v1",
            selector_name="answer_window",
            max_candidates_per_document=3,
            min_evidence_score=8.0,
            max_citation_rank=3,
            target_bucket_weights={
                "answerability_false_answer": 1.55,
                "gold_span_beats_selected_answer": 1.45,
                "evidence_selection_miss": 1.70,
            },
            design_intent="joint moderate gate with answer-window selector",
        ),
        _config(
            config_id="jgw_hybrid_window_mcpd3_evidence8_rank3_v1",
            candidate_id="joint_gate_then_window_candidate_v1",
            selector_name="hybrid_window",
            max_candidates_per_document=3,
            min_evidence_score=8.0,
            max_citation_rank=3,
            target_bucket_weights={
                "answerability_false_answer": 1.55,
                "gold_span_beats_selected_answer": 1.45,
                "evidence_selection_miss": 1.70,
            },
            design_intent="joint moderate gate with route-aware window selector",
        ),
        _config(
            config_id="jgw_answer_window_mcpd5_evidence8_rank2_v1",
            candidate_id="joint_gate_then_window_candidate_v1",
            selector_name="answer_window",
            max_candidates_per_document=5,
            min_evidence_score=8.0,
            max_citation_rank=2,
            target_bucket_weights={
                "answerability_false_answer": 1.55,
                "gold_span_beats_selected_answer": 1.45,
                "evidence_selection_miss": 1.70,
            },
            design_intent="joint broader window pool with stricter citation-rank gate",
        ),
    ]


def _config(
    *,
    config_id: str,
    candidate_id: str,
    selector_name: str,
    max_candidates_per_document: int,
    min_evidence_score: float,
    max_citation_rank: int,
    target_bucket_weights: Mapping[str, float],
    design_intent: str,
) -> dict[str, Any]:
    return {
        "config_id": config_id,
        "candidate_id": candidate_id,
        "selector_name": selector_name,
        "composition_policy_name": "top_k",
        "max_candidates_per_document": max_candidates_per_document,
        "max_sentences": 3,
        "min_sentence_score": 2.0,
        "verifier_min_citations": 1,
        "verifier_min_evidence_score": min_evidence_score,
        "verifier_max_citation_rank": max_citation_rank,
        "target_bucket_weights": dict(target_bucket_weights),
        "selection_split": "train",
        "validation_split": "dev",
        "dev_threshold_tuning_allowed": False,
        "test_access_allowed": False,
        "runtime_default_change_allowed": False,
        "design_intent": design_intent,
    }


def _train_selection_rule() -> dict[str, Any]:
    return {
        "selection_split": "train",
        "validation_split": "dev",
        "train_objective": (
            "Among train-selectable configs, minimize weighted target bucket "
            "score: 1.55*answerability_false_answer + "
            "1.45*gold_span_beats_selected_answer + 1.70*evidence_selection_miss."
        ),
        "selectability_guards": {
            "max_train_answerable_refusal_rate_delta": 0.05,
            "max_train_average_token_f1_drop": 0.01,
            "max_train_gold_doc_citation_rate_drop": 0.03,
        },
        "tie_breakers": [
            "lower train answerability_false_answer count",
            "lower train gold_span_beats_selected_answer count",
            "lower train evidence_selection_miss count",
            "higher train verified average token F1",
            "higher train gold document citation rate",
            "lower train changed answer count",
            "lexicographic config_id",
        ],
        "dev_threshold_tuning_allowed": False,
        "test_access_allowed": False,
    }


def _dev_validation_rule() -> dict[str, Any]:
    return {
        "dev_used_for": "single validation of train-selected config",
        "must_report_all_configs_on_dev": True,
        "dev_retuning_allowed": False,
        "dev_selection_allowed": False,
        "test_access_allowed": False,
    }


def _metric_contract() -> dict[str, list[str]]:
    return {
        "primary_metrics": [
            "answerability_false_answer count/rate",
            "gold_span_beats_selected_answer count/rate",
            "weighted target bucket score",
        ],
        "secondary_metrics": [
            "evidence_selection_miss count/rate",
            "verified average token F1",
            "verified gold document citation rate",
        ],
        "guard_metrics": [
            "answerable refusal rate",
            "unanswerable refusal rate",
            "changed answer count",
            "public-safe changed-case bucket summary",
        ],
    }


def _runtime_feature_contract() -> dict[str, list[str]]:
    return {
        "allowed_runtime_feature_groups": [
            "question route",
            "retrieval scores and rank buckets",
            "sentence/window support scores",
            "citation count and rank buckets",
            "composition candidate counts",
            "verifier reason code buckets",
        ],
        "prohibited_runtime_inputs": [
            "gold answers",
            "gold spans",
            "answer document identifiers",
            "source DOC_IDS",
            "test split labels",
            "dev-selected thresholds",
            "raw private question or answer strings written to reports",
        ],
    }


def _public_safe_output_contract() -> dict[str, Any]:
    return {
        "stage104_outputs_are_protocol_only": True,
        "stage105_allowed_aggregate_fields": [
            "split",
            "config_id",
            "candidate_id",
            "bucket_counts_by_split",
            "bucket_rates_by_split",
            "weighted_target_score",
            "metric_deltas_by_split",
            "train_selection_summary",
            "guard_checks",
        ],
        "stage105_allowed_case_fields": [
            "sample_id",
            "split",
            "config_id",
            "candidate_id",
            "baseline_bucket_id",
            "candidate_bucket_id",
            "baseline_answer_token_f1_bucket",
            "candidate_answer_token_f1_bucket",
            "baseline_citation_status",
            "candidate_citation_status",
            "answerability_action",
            "evidence_selection_action",
            "changed_case_confidence_band",
        ],
        "forbidden_fields": sorted(_FORBIDDEN_PUBLIC_FIELDS),
    }


def _guard_checks(
    *,
    stage103_summary: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
    user_confirmed_protocol: bool,
) -> list[dict[str, Any]]:
    configs = frozen_protocol.get("candidate_config_grid") or []
    config_ids = [str(config.get("config_id")) for config in configs]
    candidate_ids = [str(config.get("candidate_id")) for config in configs]
    stage104_contract = stage103_summary.get("stage104_contract") or {}
    train_rule = frozen_protocol.get("train_selection_rule") or {}
    dev_rule = frozen_protocol.get("dev_validation_rule") or {}
    output_contract = frozen_protocol.get("public_safe_output_contract") or {}
    explicit_exclusions = frozen_protocol.get("explicit_exclusions") or []
    fallback_policy = frozen_protocol.get("fallback_strategy_policy") or {}
    return [
        _check(
            name="stage103_source_is_expected_stage",
            passed=stage103_summary.get("stage") == _SOURCE_STAGE103,
            observed=stage103_summary.get("stage"),
            expected=_SOURCE_STAGE103,
        ),
        _check(
            name="user_confirmed_stage104_protocol",
            passed=user_confirmed_protocol,
            observed=user_confirmed_protocol,
            expected=True,
        ),
        _check(
            name="stage103_design_id_matches",
            passed=stage103_summary.get("design_id") == _SOURCE_DESIGN_ID,
            observed=stage103_summary.get("design_id"),
            expected=_SOURCE_DESIGN_ID,
        ),
        _check(
            name="stage103_protocol_is_frozen",
            passed=stage103_summary.get("decision_status")
            == "primeqa_hybrid_evidence_answerability_candidate_protocol_frozen",
            observed=stage103_summary.get("decision_status"),
            expected="primeqa_hybrid_evidence_answerability_candidate_protocol_frozen",
        ),
        _check(
            name="stage103_recommends_candidate_comparison",
            passed=stage103_summary.get("recommended_direction")
            == _SOURCE_RECOMMENDED_DIRECTION,
            observed=stage103_summary.get("recommended_direction"),
            expected=_SOURCE_RECOMMENDED_DIRECTION,
        ),
        _check(
            name="stage103_execution_order_matches_protocol",
            passed=tuple(stage103_summary.get("recommended_execution_order") or ())
            == _EXPECTED_EXECUTION_ORDER,
            observed=stage103_summary.get("recommended_execution_order"),
            expected=list(_EXPECTED_EXECUTION_ORDER),
        ),
        _check(
            name="stage103_candidates_present",
            passed=set(_CANDIDATE_IDS)
            == {
                str(candidate.get("candidate_id"))
                for candidate in stage103_summary.get("candidate_policies") or []
            },
            observed=[
                candidate.get("candidate_id")
                for candidate in stage103_summary.get("candidate_policies") or []
            ],
            expected=list(_CANDIDATE_IDS),
        ),
        _check(
            name="stage103_allows_train_dev_comparison_after_confirmation",
            passed=stage103_summary.get(
                "can_run_train_dev_candidate_comparison_after_user_confirmation"
            )
            is True,
            observed=stage103_summary.get(
                "can_run_train_dev_candidate_comparison_after_user_confirmation"
            ),
            expected=True,
        ),
        _check(
            name="stage103_final_test_gate_locked",
            passed=stage103_summary.get("can_open_final_test_gate_now") is False
            and stage103_summary.get("can_run_final_test_metrics_now") is False,
            observed={
                "can_open_final_test_gate_now": stage103_summary.get(
                    "can_open_final_test_gate_now"
                ),
                "can_run_final_test_metrics_now": stage103_summary.get(
                    "can_run_final_test_metrics_now"
                ),
            },
            expected=False,
        ),
        _check(
            name="stage103_forbids_test_tuning",
            passed=stage103_summary.get("can_use_test_for_tuning") is False,
            observed=stage103_summary.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage103_fallback_disabled",
            passed=stage103_summary.get("fallback_strategies_enabled") is False,
            observed=stage103_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="stage103_runtime_defaults_unchanged",
            passed=stage103_summary.get("default_runtime_policy") == "unchanged",
            observed=stage103_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage103_contract_id_matches_stage104_protocol",
            passed=stage104_contract.get("comparison_id") == _PROTOCOL_ID,
            observed=stage104_contract.get("comparison_id"),
            expected=_PROTOCOL_ID,
        ),
        _check(
            name="protocol_id_is_fixed",
            passed=frozen_protocol.get("protocol_id") == _PROTOCOL_ID,
            observed=frozen_protocol.get("protocol_id"),
            expected=_PROTOCOL_ID,
        ),
        _check(
            name="protocol_status_requires_confirmation_before_metric_run",
            passed=frozen_protocol.get("protocol_status")
            == "frozen_requires_user_confirmation_before_metric_run",
            observed=frozen_protocol.get("protocol_status"),
            expected="frozen_requires_user_confirmation_before_metric_run",
        ),
        _check(
            name="candidate_config_grid_has_nine_configs",
            passed=len(configs) == 9,
            observed=len(configs),
            expected=9,
        ),
        _check(
            name="candidate_config_ids_are_unique",
            passed=len(config_ids) == len(set(config_ids)),
            observed=config_ids,
            expected="unique config IDs",
        ),
        _check(
            name="candidate_grid_covers_all_stage103_candidates",
            passed=set(candidate_ids) == set(_CANDIDATE_IDS),
            observed=Counter(candidate_ids),
            expected=list(_CANDIDATE_IDS),
        ),
        _check(
            name="candidate_grid_has_three_configs_per_candidate",
            passed=all(count == 3 for count in Counter(candidate_ids).values()),
            observed=Counter(candidate_ids),
            expected="3 configs per candidate",
        ),
        _check(
            name="candidate_grid_uses_allowed_selectors",
            passed=all(config.get("selector_name") in _ALLOWED_SELECTORS for config in configs),
            observed=sorted({str(config.get("selector_name")) for config in configs}),
            expected=sorted(_ALLOWED_SELECTORS),
        ),
        _check(
            name="candidate_grid_keeps_composition_top_k",
            passed=all(config.get("composition_policy_name") == "top_k" for config in configs),
            observed=sorted(
                {str(config.get("composition_policy_name")) for config in configs}
            ),
            expected="top_k",
        ),
        _check(
            name="candidate_grid_forbids_dev_threshold_tuning",
            passed=all(config.get("dev_threshold_tuning_allowed") is False for config in configs),
            observed=[
                config.get("dev_threshold_tuning_allowed") for config in configs
            ],
            expected=False,
        ),
        _check(
            name="candidate_grid_forbids_test_access",
            passed=all(config.get("test_access_allowed") is False for config in configs),
            observed=[config.get("test_access_allowed") for config in configs],
            expected=False,
        ),
        _check(
            name="candidate_grid_forbids_runtime_default_change",
            passed=all(
                config.get("runtime_default_change_allowed") is False for config in configs
            ),
            observed=[
                config.get("runtime_default_change_allowed") for config in configs
            ],
            expected=False,
        ),
        _check(
            name="grid_derivation_uses_no_metric_labels",
            passed=_grid_derivation_uses_no_metric_labels(frozen_protocol),
            observed=frozen_protocol.get("grid_derivation_policy"),
            expected="train/dev/test labels not used to choose threshold values",
        ),
        _check(
            name="train_selection_rule_is_train_only",
            passed=train_rule.get("selection_split") == "train"
            and train_rule.get("validation_split") == "dev"
            and train_rule.get("dev_threshold_tuning_allowed") is False
            and train_rule.get("test_access_allowed") is False,
            observed=train_rule,
            expected="train selection, dev validation, no test",
        ),
        _check(
            name="train_selection_guards_are_frozen",
            passed=set(train_rule.get("selectability_guards") or {}) == {
                "max_train_answerable_refusal_rate_delta",
                "max_train_average_token_f1_drop",
                "max_train_gold_doc_citation_rate_drop",
            },
            observed=train_rule.get("selectability_guards"),
            expected="three frozen train selectability guards",
        ),
        _check(
            name="dev_validation_forbids_retuning",
            passed=dev_rule.get("dev_retuning_allowed") is False
            and dev_rule.get("dev_selection_allowed") is False,
            observed=dev_rule,
            expected="dev validation without retuning or selection",
        ),
        _check(
            name="public_output_contract_has_no_forbidden_fields",
            passed=_public_output_contract_is_safe(output_contract),
            observed=output_contract,
            expected="no forbidden public output fields",
        ),
        _check(
            name="stage104_exclusions_lock_test_runtime_fallback",
            passed={
                "no_test_split_loading",
                "no_final_test_metrics",
                "no_metric_run",
                "no_runtime_default_change",
                "no_fallback_strategy",
            }.issubset(set(explicit_exclusions)),
            observed=explicit_exclusions,
            expected=[
                "no_test_split_loading",
                "no_final_test_metrics",
                "no_metric_run",
                "no_runtime_default_change",
                "no_fallback_strategy",
            ],
        ),
        _check(
            name="fallback_policy_disabled",
            passed=fallback_policy.get("fallback_strategies_enabled") is False,
            observed=fallback_policy,
            expected={"fallback_strategies_enabled": False},
        ),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    base = {
        "protocol_id": _PROTOCOL_ID,
        "recommended_direction": _NEXT_DIRECTION,
        "requires_user_confirmation_before_train_dev_metric_run": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }
    if failed_checks:
        return {
            **base,
            "status": "primeqa_hybrid_evidence_answerability_comparison_protocol_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "can_run_train_dev_candidate_comparison_after_user_confirmation": False,
        }
    return {
        **base,
        "status": "primeqa_hybrid_evidence_answerability_comparison_protocol_frozen",
        "can_continue_train_dev_development": True,
        "can_run_train_dev_candidate_comparison_after_user_confirmation": True,
        "recommended_next_stage": (
            "Stage105: after user confirmation, run the frozen train/dev-only "
            "evidence-answerability candidate comparison against the Stage102 "
            "verified baseline; select thresholds on train only, validate once "
            "on dev, keep test locked, and keep runtime defaults unchanged."
        ),
    }


def _config_counts_by_candidate_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counts = Counter(
        config["candidate_id"]
        for config in report["frozen_protocol"]["candidate_config_grid"]
    )
    return [
        BarDatum(label=label, value=float(count), value_label=str(count))
        for label, count in sorted(counts.items())
    ]


def _config_min_evidence_score_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(config["config_id"]),
            value=float(config["verifier_min_evidence_score"]),
            value_label=f"{float(config['verifier_min_evidence_score']):.1f}",
        )
        for config in report["frozen_protocol"]["candidate_config_grid"]
    ]


def _config_max_citation_rank_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(config["config_id"]),
            value=float(config["verifier_max_citation_rank"]),
            value_label=str(config["verifier_max_citation_rank"]),
        )
        for config in report["frozen_protocol"]["candidate_config_grid"]
    ]


def _selector_mix_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counts = Counter(
        config["selector_name"] for config in report["frozen_protocol"]["candidate_config_grid"]
    )
    return [
        BarDatum(label=label, value=float(count), value_label=str(count))
        for label, count in sorted(counts.items())
    ]


def _train_selection_guard_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    guards = report["frozen_protocol"]["train_selection_rule"]["selectability_guards"]
    return [
        BarDatum(label=str(name), value=float(value), value_label=f"{float(value):.4f}")
        for name, value in guards.items()
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report["decision"]
    names = [
        "can_run_train_dev_candidate_comparison_after_user_confirmation",
        "can_run_final_test_metrics_now",
        "can_use_test_for_tuning",
        "fallback_strategies_enabled",
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
            label=str(check["name"]),
            value=1.0 if check["passed"] else 0.0,
            value_label="passed" if check["passed"] else "failed",
        )
        for check in report["guard_checks"]
    ]


def _grid_derivation_uses_no_metric_labels(
    frozen_protocol: Mapping[str, Any],
) -> bool:
    policy = frozen_protocol.get("grid_derivation_policy") or {}
    return (
        policy.get("train_labels_used_to_choose_threshold_values") is False
        and policy.get("dev_labels_used_to_choose_threshold_values") is False
        and policy.get("test_labels_used_to_choose_threshold_values") is False
        and policy.get("metric_run_performed_in_stage104") is False
    )


def _public_output_contract_is_safe(output_contract: Mapping[str, Any]) -> bool:
    case_fields = {
        str(field).lower()
        for field in output_contract.get("stage105_allowed_case_fields") or []
    }
    aggregate_fields = {
        str(field).lower()
        for field in output_contract.get("stage105_allowed_aggregate_fields") or []
    }
    forbidden = {field.lower() for field in _FORBIDDEN_PUBLIC_FIELDS}
    return not ((case_fields | aggregate_fields) & forbidden)


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
