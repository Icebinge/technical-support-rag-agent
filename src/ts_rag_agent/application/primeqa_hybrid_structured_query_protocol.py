from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 88"
_CREATED_AT = "2026-07-15"
_SOURCE_STAGE84 = "Stage 84"
_SOURCE_STAGE87 = "Stage 87"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_PROTOCOL_ID = "structured_query_keyphrase_compaction_train_dev_v1"
_CANDIDATE_ID = "structured_query_keyphrase_compaction_design"
_STOPPED_CANDIDATE_ID = "lexical_cluster_diversity_rerank_design"
_ALLOWED_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})


@dataclass(frozen=True)
class PrimeQAHybridStructuredQueryProtocolVisualization:
    """One generated Stage88 structured-query protocol visualization."""

    name: str
    path: str


def freeze_primeqa_hybrid_structured_query_protocol(
    *,
    stage84_report_path: Path,
    stage87_report_path: Path,
    user_confirmed_candidate: bool,
    confirmed_candidate_id: str,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the train/dev protocol for structured query compaction."""

    started_at = time.perf_counter()
    stage84_report = _load_json_object(stage84_report_path)
    stage87_report = _load_json_object(stage87_report_path)
    candidate = _selected_candidate(stage84_report)
    frozen_protocol = _frozen_protocol(candidate)
    guard_checks = _guard_checks(
        stage84_report=stage84_report,
        stage87_report=stage87_report,
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
            "Train/dev-only protocol freeze for the structured query keyphrase "
            "compaction candidate selected after Stage87 stopped the lexical "
            "cluster diversity route. This stage reads only public-safe Stage84 "
            "and Stage87 reports, freezes a predeclared protocol for a future "
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
            "stage87_report": _fingerprint(stage87_report_path),
        },
        "stage84_decision": stage84_report.get("decision") or {},
        "stage87_decision": stage87_report.get("decision") or {},
        "stage84_candidate_summary": candidate,
        "stage87_next_candidate_summary": (
            stage87_report.get("candidate_queue", {}).get("next_candidate_summary")
            or {}
        ),
        "frozen_protocol": frozen_protocol,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
        "timing_seconds": {
            "load_and_freeze": round(checked_at - started_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_structured_query_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridStructuredQueryProtocolVisualization]:
    """Write SVG charts for Stage88 structured-query protocol."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage88_structured_query_config_token_limits.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage88 structured query config token limits",
                bars=_config_token_limit_bars(report),
                x_label="maximum unique query terms",
                width=1120,
                margin_left=430,
            )
        ),
        "stage88_structured_query_feature_group_counts.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage88 structured query feature group counts",
                bars=_feature_group_bars(report),
                x_label="feature count",
                width=980,
                margin_left=330,
            )
        ),
        "stage88_structured_query_protocol_decision_flags.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage88 structured query protocol decision flags",
                bars=_decision_flag_bars(report),
                x_label="1 means true",
                width=1080,
                margin_left=420,
            )
        ),
        "stage88_structured_query_guard_check_status.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage88 structured query guard check status",
                bars=_guard_check_bars(report),
                x_label="1 means passed",
                width=1240,
                margin_left=560,
            )
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridStructuredQueryProtocolVisualization(
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


def _public_candidate_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    public_fields = [
        "candidate_id",
        "name",
        "category",
        "status",
        "risk_level",
        "implementation_readiness",
        "priority_score",
        "target_miss_count",
        "target_miss_count_by_split",
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
        "source_stages": [_SOURCE_STAGE84, _SOURCE_STAGE87],
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
        "query_feature_contract": _query_feature_contract(),
        "compaction_contract": {
            "query_terms_source": "runtime question title and body text",
            "normalization": (
                "casefold, split punctuation, preserve code-like spans, remove "
                "configured stopwords, and de-duplicate by first occurrence"
            ),
            "ordering": [
                "error_code_or_log_identifier",
                "product_component_or_feature",
                "version_or_platform",
                "action_intent",
                "title_guard_terms",
                "deterministic_noun_phrase_like_terms",
            ],
            "query_text_written_to_report": False,
            "candidate_depth_unchanged": True,
        },
        "train_selection_rule": {
            "selection_split": "train",
            "validation_split": "dev",
            "rule": (
                "Select the candidate config on train by hit@10, then hit@5, "
                "hit@1, MRR@10, fewer top10 regressions, fewer rank-down within "
                "top10, lower average compacted token count, then config_id. "
                "Dev is validation only."
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
            "rank_up_within_top10_count",
            "rank_down_within_top10_count",
            "not_found_count_at_50",
            "average_compacted_query_token_count",
        ],
        "public_safe_changed_case_fields": [
            "sample_id",
            "split",
            "baseline_rank",
            "challenger_rank",
            "config_id",
            "query_view_id",
            "query_token_count",
            "compacted_query_token_count",
            "token_bucket_counts",
        ],
        "explicit_exclusions": [
            "Do not use source DOC_IDS as runtime retrieval evidence.",
            "Do not use answer document IDs or gold ranks as runtime features.",
            "Do not choose candidate configs from dev-only performance.",
            "Do not load or evaluate the frozen test split.",
            "Do not write raw question text, answer text, document titles, or "
            "document body text to the report.",
            "Do not write compacted query strings or matched token strings to "
            "the report.",
            "Do not change runtime defaults in this stage.",
            "Do not add alternate retrievers or replacement behavior outside "
            "the predeclared config grid.",
        ],
    }


def _candidate_config_grid() -> list[dict[str, Any]]:
    return [
        {
            "config_id": "sqkc_action_error_product_v1",
            "query_view_id": "action_error_product_version_terms",
            "preserved_feature_buckets": [
                "error_code_or_log_identifier",
                "product_component_or_feature",
                "version_or_platform",
                "action_intent",
                "quoted_or_code_like_terms",
            ],
            "maximum_unique_terms": 18,
            "minimum_unique_terms": 4,
        },
        {
            "config_id": "sqkc_title_guarded_action_error_v1",
            "query_view_id": "title_guarded_action_error_product_terms",
            "preserved_feature_buckets": [
                "title_guard_terms",
                "error_code_or_log_identifier",
                "product_component_or_feature",
                "version_or_platform",
                "action_intent",
            ],
            "minimum_title_terms": 3,
            "maximum_unique_terms": 16,
            "minimum_unique_terms": 4,
        },
        {
            "config_id": "sqkc_error_first_compact_v1",
            "query_view_id": "error_identifier_first_terms",
            "preserved_feature_buckets": [
                "error_code_or_log_identifier",
                "quoted_or_code_like_terms",
                "product_component_or_feature",
                "action_intent",
            ],
            "maximum_unique_terms": 14,
            "minimum_unique_terms": 3,
        },
        {
            "config_id": "sqkc_noun_phrase_compact_v1",
            "query_view_id": "deterministic_noun_phrase_like_terms",
            "preserved_feature_buckets": [
                "deterministic_noun_phrase_like_terms",
                "product_component_or_feature",
                "version_or_platform",
                "action_intent",
            ],
            "noun_phrase_window_size": 2,
            "maximum_unique_terms": 20,
            "minimum_unique_terms": 4,
        },
    ]


def _query_feature_contract() -> dict[str, Any]:
    return {
        "runtime_allowed_feature_groups": {
            "query_structure_features": [
                "query_token_count",
                "query_unique_token_count",
                "title_token_count",
                "body_token_count",
            ],
            "deterministic_token_class_features": [
                "is_error_code_or_log_identifier",
                "is_product_component_or_feature",
                "is_version_or_platform",
                "is_action_intent",
                "is_quoted_or_code_like",
            ],
            "token_position_features": [
                "first_occurrence_index",
                "appears_in_title",
                "appears_in_body",
                "bucket_order_index",
            ],
            "token_filter_features": [
                "token_length",
                "token_frequency_within_query",
                "stopword_list_membership",
            ],
        },
        "prohibited_runtime_features": [
            "source_DOC_IDS",
            "answer document IDs",
            "gold document rank",
            "gold labels",
            "frozen test split membership",
            "raw document text",
            "raw document title",
        ],
        "prohibited_report_fields": [
            "question text",
            "answer text",
            "document title",
            "document body text",
            "compacted query text",
            "matched token strings",
        ],
    }


def _guard_checks(
    *,
    stage84_report: Mapping[str, Any],
    stage87_report: Mapping[str, Any],
    candidate: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
    user_confirmed_candidate: bool,
    confirmed_candidate_id: str,
) -> list[dict[str, Any]]:
    stage84_decision = stage84_report.get("decision") or {}
    stage87_decision = stage87_report.get("decision") or {}
    target_metric_contract = candidate.get("target_metric_contract") or []
    explicit_exclusions = frozen_protocol.get("explicit_exclusions") or []
    query_feature_contract = frozen_protocol.get("query_feature_contract") or {}
    prohibited_runtime_features = query_feature_contract.get(
        "prohibited_runtime_features",
        [],
    )
    query_view_ids = [
        config.get("query_view_id")
        for config in frozen_protocol.get("candidate_config_grid") or []
    ]
    return [
        _check(
            name="source_stage84_report_is_stage84",
            passed=stage84_report.get("stage") == _SOURCE_STAGE84,
            observed=stage84_report.get("stage"),
            expected=_SOURCE_STAGE84,
        ),
        _check(
            name="source_stage87_report_is_stage87",
            passed=stage87_report.get("stage") == _SOURCE_STAGE87,
            observed=stage87_report.get("stage"),
            expected=_SOURCE_STAGE87,
        ),
        _check(
            name="user_confirmed_structured_query_protocol",
            passed=user_confirmed_candidate,
            observed=user_confirmed_candidate,
            expected=True,
        ),
        _check(
            name="stage87_stopped_lcdr_route",
            passed=stage87_decision.get("status")
            == "primeqa_hybrid_lexical_cluster_diversity_route_stopped"
            and stage87_decision.get("stopped_candidate_id") == _STOPPED_CANDIDATE_ID,
            observed={
                "status": stage87_decision.get("status"),
                "stopped_candidate_id": stage87_decision.get("stopped_candidate_id"),
            },
            expected=_STOPPED_CANDIDATE_ID,
        ),
        _check(
            name="confirmed_candidate_matches_stage87_next_candidate",
            passed=confirmed_candidate_id
            == stage87_decision.get("next_candidate_id")
            == _CANDIDATE_ID,
            observed={
                "confirmed_candidate_id": confirmed_candidate_id,
                "stage87_next_candidate_id": stage87_decision.get(
                    "next_candidate_id"
                ),
            },
            expected=_CANDIDATE_ID,
        ),
        _check(
            name="stage87_requires_confirmation_before_next_protocol",
            passed=stage87_decision.get("requires_user_confirmation_before_next_protocol")
            is True,
            observed=stage87_decision.get(
                "requires_user_confirmation_before_next_protocol"
            ),
            expected=True,
        ),
        _check(
            name="stage87_final_test_metrics_locked",
            passed=stage87_decision.get("can_run_final_test_metrics_now") is False,
            observed=stage87_decision.get("can_run_final_test_metrics_now"),
            expected=False,
        ),
        _check(
            name="stage87_forbids_test_tuning",
            passed=stage87_decision.get("can_use_test_for_tuning") is False,
            observed=stage87_decision.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage87_runtime_default_unchanged",
            passed=stage87_decision.get("default_runtime_policy") == "unchanged",
            observed=stage87_decision.get("default_runtime_policy"),
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
            name="stage84_candidate_target_contract_requires_train_selected_dev_hit10",
            passed=any("train-selected dev hit@10" in str(item) for item in target_metric_contract),
            observed=target_metric_contract,
            expected="train-selected dev hit@10 contract",
        ),
        _check(
            name="stage84_candidate_contract_forbids_dev_selection",
            passed=any("dev-only performance" in str(item) for item in target_metric_contract),
            observed=target_metric_contract,
            expected="no dev-only selection",
        ),
        _check(
            name="protocol_id_is_fixed",
            passed=frozen_protocol.get("protocol_id") == _PROTOCOL_ID,
            observed=frozen_protocol.get("protocol_id"),
            expected=_PROTOCOL_ID,
        ),
        _check(
            name="candidate_config_grid_is_predeclared",
            passed=len(frozen_protocol.get("candidate_config_grid") or []) == 4
            and len(set(query_view_ids)) == 4,
            observed=query_view_ids,
            expected=4,
        ),
        _check(
            name="query_feature_contract_is_runtime_only",
            passed="runtime_allowed_feature_groups" in query_feature_contract
            and "raw document text" in prohibited_runtime_features,
            observed=query_feature_contract,
            expected="runtime query features only",
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
                    "compacted_query_text",
                ]
            ),
            observed=frozen_protocol.get("public_safe_changed_case_fields"),
            expected="public-safe ids, ranks, and counts only",
        ),
        _check(
            name="stage88_freezes_protocol_without_metrics",
            passed=True,
            observed="protocol_freeze_only",
            expected="protocol_freeze_only",
        ),
        _check(
            name="stage88_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage88_default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_structured_query_protocol_blocked",
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
        "status": "primeqa_hybrid_structured_query_protocol_frozen",
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
            "Stage 89: after user confirmation, run the frozen train/dev-only "
            "structured query keyphrase compaction comparison; keep test "
            "locked and do not run final metrics."
        ),
    }


def _config_token_limit_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=config["config_id"],
            value=float(config["maximum_unique_terms"]),
            value_label=str(config["maximum_unique_terms"]),
        )
        for config in report["frozen_protocol"]["candidate_config_grid"]
    ]


def _feature_group_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    groups = report["frozen_protocol"]["query_feature_contract"][
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
