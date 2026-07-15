from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 85"
_CREATED_AT = "2026-07-15"
_SOURCE_STAGE = "Stage 84"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_PROTOCOL_ID = "lexical_cluster_diversity_rerank_train_dev_v1"
_CANDIDATE_ID = "lexical_cluster_diversity_rerank_design"
_ALLOWED_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})


@dataclass(frozen=True)
class PrimeQAHybridLexicalClusterDiversityProtocolVisualization:
    """One generated Stage85 lexical cluster diversity protocol visualization."""

    name: str
    path: str


def freeze_primeqa_hybrid_lexical_cluster_diversity_protocol(
    *,
    stage84_report_path: Path,
    user_confirmed_candidate: bool,
    confirmed_candidate_id: str,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the train/dev protocol for the Stage84 recommended candidate."""

    started_at = time.perf_counter()
    stage84_report = _load_json_object(stage84_report_path)
    candidate = _selected_candidate(stage84_report)
    frozen_protocol = _frozen_protocol(candidate)
    guard_checks = _guard_checks(
        stage84_report=stage84_report,
        candidate=candidate,
        frozen_protocol=frozen_protocol,
        user_confirmed_candidate=user_confirmed_candidate,
        confirmed_candidate_id=confirmed_candidate_id,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_scope": (
            "Train/dev-only protocol freeze for the Stage84 recommended "
            "lexical cluster diversity rerank candidate. This stage reads the "
            "public-safe Stage84 design report, freezes a predeclared protocol "
            "for a future train/dev metric run, does not run retrieval metrics, "
            "does not load the frozen test split, does not run final metrics, "
            "does not use source DOC_IDS as runtime retrieval evidence, and does "
            "not change runtime defaults."
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
        },
        "stage84_decision": stage84_report.get("decision") or {},
        "stage84_candidate_summary": candidate,
        "frozen_protocol": frozen_protocol,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
        "timing_seconds": {
            "load_and_freeze": round(checked_at - started_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_lexical_cluster_diversity_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridLexicalClusterDiversityProtocolVisualization]:
    """Write SVG charts for Stage85 lexical cluster diversity protocol."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage85_lcdr_candidate_config_penalties.svg": render_horizontal_bar_chart_svg(
            title="Stage85 LCDR candidate config penalties",
            bars=_config_penalty_bars(report),
            x_label="duplicate penalty weight",
            width=1120,
            margin_left=430,
        ),
        "stage85_lcdr_feature_group_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage85 LCDR feature group counts",
            bars=_feature_group_bars(report),
            x_label="feature count",
            width=980,
            margin_left=330,
        ),
        "stage85_lcdr_protocol_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage85 LCDR protocol decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1080,
            margin_left=420,
        ),
        "stage85_lcdr_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage85 LCDR guard check status",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1220,
            margin_left=520,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridLexicalClusterDiversityProtocolVisualization(
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
            return dict(candidate)
    raise ValueError(f"Stage84 report does not contain candidate {_CANDIDATE_ID!r}")


def _frozen_protocol(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "candidate_id": _CANDIDATE_ID,
        "protocol_status": "frozen_requires_user_confirmation_before_metric_run",
        "source_stage": _SOURCE_STAGE,
        "target_miss_count": int(candidate.get("target_miss_count") or 0),
        "target_miss_count_by_split": candidate.get("target_miss_count_by_split") or {},
        "baseline_retriever": {
            "config_id": "full_document_bm25_baseline",
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "candidate_depth": 50,
            "primary_top_k": 10,
        },
        "candidate_config_grid": _candidate_config_grid(),
        "cluster_feature_contract": _cluster_feature_contract(),
        "rerank_formula": {
            "candidate_order_source": "baseline_bm25_rank",
            "cluster_duplicate_index": (
                "Count prior candidates in the same lexical_cluster_hash when "
                "traversing baseline BM25 order."
            ),
            "adjusted_score": (
                "baseline_bm25_score - duplicate_penalty_weight * "
                "top1_bm25_score * cluster_duplicate_index"
            ),
            "tie_breakers": [
                "higher baseline_bm25_score",
                "lower baseline_bm25_rank",
                "lower stable document id sort key",
            ],
            "empty_cluster_hash_behavior": (
                "cluster_duplicate_index remains 0; no alternate retriever is used."
            ),
        },
        "train_selection_rule": {
            "selection_split": "train",
            "validation_split": "dev",
            "rule": (
                "Select the candidate config on train by hit@10, then hit@5, "
                "hit@1, MRR@10, fewer top10 regressions, fewer rank-down within "
                "top10, then config_id. Dev is validation only."
            ),
            "dev_selection_forbidden": True,
            "test_selection_forbidden": True,
        },
        "metrics_allowed_after_confirmation": [
            "hit@1",
            "hit@5",
            "hit@10",
            "MRR@10",
            "top10_improvement_count",
            "top10_regression_count",
            "rank_up_within_top10_count",
            "rank_down_within_top10_count",
            "not_found_count_at_50",
        ],
        "public_safe_changed_case_fields": [
            "sample_id",
            "split",
            "baseline_rank",
            "challenger_rank",
            "baseline_cluster_duplicate_index",
            "challenger_cluster_duplicate_index",
            "config_id",
        ],
        "explicit_exclusions": [
            "Do not use source DOC_IDS as runtime retrieval evidence.",
            "Do not use answer document IDs or gold ranks as runtime features.",
            "Do not choose candidate configs from dev-only performance.",
            "Do not load or evaluate the frozen test split.",
            "Do not write raw question text, answer text, document titles, or "
            "document body text to the report.",
            "Do not change runtime defaults in this stage.",
            "Do not add alternate retrievers or replacement behavior outside the "
            "predeclared config grid.",
        ],
    }


def _candidate_config_grid() -> list[dict[str, Any]]:
    return [
        {
            "config_id": "lcdr_penalty_0_03_title_query_cluster",
            "duplicate_penalty_weight": 0.03,
            "cluster_key": "title_query_overlap_hash",
            "minimum_title_overlap_terms": 3,
            "minimum_cluster_size": 2,
        },
        {
            "config_id": "lcdr_penalty_0_06_title_query_cluster",
            "duplicate_penalty_weight": 0.06,
            "cluster_key": "title_query_overlap_hash",
            "minimum_title_overlap_terms": 3,
            "minimum_cluster_size": 2,
        },
        {
            "config_id": "lcdr_penalty_0_09_title_query_cluster",
            "duplicate_penalty_weight": 0.09,
            "cluster_key": "title_query_overlap_hash",
            "minimum_title_overlap_terms": 3,
            "minimum_cluster_size": 2,
        },
        {
            "config_id": "lcdr_penalty_0_12_title_query_cluster",
            "duplicate_penalty_weight": 0.12,
            "cluster_key": "title_query_overlap_hash",
            "minimum_title_overlap_terms": 3,
            "minimum_cluster_size": 2,
        },
    ]


def _cluster_feature_contract() -> dict[str, Any]:
    return {
        "runtime_allowed_feature_groups": {
            "query_features": [
                "query_token_count",
                "query_unique_token_count",
            ],
            "candidate_rank_score_features": [
                "baseline_bm25_rank",
                "baseline_bm25_score",
                "score_margin_to_top1",
                "score_margin_to_previous",
            ],
            "candidate_overlap_features": [
                "query_overlap_count",
                "title_query_overlap_count",
                "document_token_count",
            ],
            "cluster_features": [
                "title_query_overlap_hash",
                "lexical_cluster_hash",
                "cluster_duplicate_index",
                "cluster_size_in_candidate_depth",
            ],
        },
        "cluster_hash_contract": {
            "hash_input": (
                "sorted normalized title tokens that also appear in the normalized "
                "runtime query"
            ),
            "hash_algorithm": "sha256",
            "hash_length": 16,
            "raw_tokens_written_to_report": False,
        },
        "prohibited_runtime_features": [
            "source_DOC_IDS",
            "answer_doc_id",
            "gold_document_rank",
            "gold_label",
            "frozen_test_split_membership",
        ],
        "prohibited_report_fields": [
            "question text",
            "answer text",
            "document title",
            "document body text",
            "matched token strings",
        ],
    }


def _guard_checks(
    *,
    stage84_report: Mapping[str, Any],
    candidate: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
    user_confirmed_candidate: bool,
    confirmed_candidate_id: str,
) -> list[dict[str, Any]]:
    stage84_decision = stage84_report.get("decision") or {}
    explicit_exclusions = frozen_protocol.get("explicit_exclusions") or []
    prohibited_runtime_features = (
        frozen_protocol.get("cluster_feature_contract", {})
        .get("prohibited_runtime_features", [])
    )
    return [
        _check(
            name="source_report_is_stage84",
            passed=stage84_report.get("stage") == _SOURCE_STAGE,
            observed=stage84_report.get("stage"),
            expected=_SOURCE_STAGE,
        ),
        _check(
            name="user_confirmed_recommended_candidate",
            passed=user_confirmed_candidate,
            observed=user_confirmed_candidate,
            expected=True,
        ),
        _check(
            name="confirmed_candidate_matches_stage84_recommendation",
            passed=confirmed_candidate_id == stage84_decision.get(
                "recommended_next_candidate_id"
            )
            == _CANDIDATE_ID,
            observed={
                "confirmed_candidate_id": confirmed_candidate_id,
                "stage84_recommended_next_candidate_id": stage84_decision.get(
                    "recommended_next_candidate_id"
                ),
            },
            expected=_CANDIDATE_ID,
        ),
        _check(
            name="candidate_is_recommended_for_protocol_design",
            passed=candidate.get("status")
            == "recommended_for_train_dev_protocol_design",
            observed=candidate.get("status"),
            expected="recommended_for_train_dev_protocol_design",
        ),
        _check(
            name="stage84_requires_confirmation_before_train_dev_run",
            passed=stage84_decision.get("requires_user_confirmation_before_train_dev_run")
            is True,
            observed=stage84_decision.get(
                "requires_user_confirmation_before_train_dev_run"
            ),
            expected=True,
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
            name="protocol_id_is_fixed",
            passed=frozen_protocol.get("protocol_id") == _PROTOCOL_ID,
            observed=frozen_protocol.get("protocol_id"),
            expected=_PROTOCOL_ID,
        ),
        _check(
            name="candidate_config_grid_is_predeclared",
            passed=len(frozen_protocol.get("candidate_config_grid") or []) == 4,
            observed=[
                config.get("config_id")
                for config in frozen_protocol.get("candidate_config_grid") or []
            ],
            expected=4,
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
                ]
            ),
            observed=frozen_protocol.get("public_safe_changed_case_fields"),
            expected="public-safe ids and ranks only",
        ),
        _check(
            name="stage85_freezes_protocol_without_metrics",
            passed=True,
            observed="protocol_freeze_only",
            expected="protocol_freeze_only",
        ),
        _check(
            name="stage85_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage85_default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_lexical_cluster_diversity_protocol_blocked",
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
        "status": "primeqa_hybrid_lexical_cluster_diversity_protocol_frozen",
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
            "Stage 86: after user confirmation, run the frozen train/dev-only "
            "lexical cluster diversity rerank comparison; keep test locked and "
            "do not run final metrics."
        ),
    }


def _config_penalty_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=config["config_id"],
            value=float(config["duplicate_penalty_weight"]),
            value_label=f"{config['duplicate_penalty_weight']:.2f}",
        )
        for config in report["frozen_protocol"]["candidate_config_grid"]
    ]


def _feature_group_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    groups = report["frozen_protocol"]["cluster_feature_contract"][
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
