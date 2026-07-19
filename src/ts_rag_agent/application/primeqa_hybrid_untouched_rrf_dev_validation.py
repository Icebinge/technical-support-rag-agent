from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    EncoderFactory,
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
    _load_json_object,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    CANDIDATE_POOL_DEPTH,
    CONTEXT_DEPTH,
    records_by_sample,
    select_current_query_overlap_top10,
    select_original_rrf_top10,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector_training import (
    PrimeQAHybridCandidateDatasetBuilder,
    _candidate_pool_summary,
    _canonical_json_sha256,
    _control_selection_run,
    _evaluate_selection_run,
    _fingerprint,
    _fold_assignment_summary,
    _public_evaluation,
    _public_safe_contract,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 163"
_CREATED_AT = "2026-07-19"
_ANALYSIS_ID = "primeqa_hybrid_untouched_rrf_one_shot_dev_validation_v1"
_PROTOCOL_ID = "primeqa_hybrid_untouched_rrf_dev_protocol_v1"
_FOLD_COUNT = 5
_EXPECTED_DEV_ROWS = 121
_EXPECTED_ANSWERABLE_ROWS = 76
_EXPECTED_UNANSWERABLE_ROWS = 45
_EXPECTED_CANDIDATE_ROWS = _EXPECTED_DEV_ROWS * CANDIDATE_POOL_DEPTH
_EXPECTED_CURRENT_TOP10_GOLD_HIT_COUNT = 36
_CURRENT_CONTROL_ID = "stage160_current_query_overlap_top10_control"
_RRF_POLICY_ID = "stage163_frozen_untouched_rrf_top10_candidate"
_STAGE162_STATUS = "primeqa_hybrid_conservative_context_swap_no_train_nested_cv_safe_config"
_STAGE160_STATUS = "primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_completed"
_EXPECTED_SOURCE_HASHES = {
    "stage162": "ff126db5efc2b117ab77cf99a62ec5c399110a938b3a37ea449055e76e622d93",
    "stage160": "e17e5fe5bbc5fef4e25e41234e47b89daf19ea4ef18f3c7270601f0fee7d9377",
    "stage80": "2441bb1cb1e7888299d3f57962b18cd59df84e2086ac281105abcacfc144880f",
    "dev": "071c54f80657592bda7f8e4095afc8800a2be112362c3a275191a0fc8e28bd5f",
    "documents": "f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b",
}

ProgressSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class PrimeQAHybridUntouchedRRFDevVisualization:
    """One generated Stage163 one-shot development chart."""

    name: str
    path: str


def run_primeqa_hybrid_untouched_rrf_dev_validation(
    *,
    stage162_report_path: Path,
    stage160_report_path: Path,
    stage80_report_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    user_confirmed_validation: bool,
    confirmation_note: str,
    include_dense_channels: bool = True,
    encoder_batch_size: int = 64,
    encoder_device: str | None = None,
    encoder_factory: EncoderFactory | None = None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    progress_sink: ProgressSink | None = None,
) -> dict[str, Any]:
    """Run the frozen untouched-RRF policy's one-shot development validation."""

    started_at = time.perf_counter()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    source_authorization = _authorize_sources(
        stage162_report_path=stage162_report_path,
        stage160_report_path=stage160_report_path,
        stage80_report_path=stage80_report_path,
        dev_split_path=dev_split_path,
        documents_path=documents_path,
    )
    protocol = _frozen_protocol()
    protocol_sha256 = _canonical_json_sha256(protocol)
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="sources_authorized")

    samples = load_primeqa_hybrid_split_samples(dev_split_path)
    fold_assignments = _build_train_fold_assignments(samples, fold_count=_FOLD_COUNT)
    documents_by_id = load_primeqa_documents(documents_path)
    sections_by_document = load_primeqa_document_sections(documents_path)
    documents = list(documents_by_id.values())
    loaded_at = time.perf_counter()
    _emit(progress_sink, phase="dev_and_documents_loaded", dev_rows=len(samples))

    stage80_report = _load_json_object(stage80_report_path)
    dense_channels, dense_summary = _build_dense_channels(
        include_dense_channels=include_dense_channels,
        stage80_report=stage80_report,
        stage80_report_path=stage80_report_path,
        documents=documents,
        document_ids=tuple(document.id for document in documents),
        encoder_batch_size=encoder_batch_size,
        encoder_device=encoder_device,
        encoder_factory=encoder_factory,
    )
    lexical_channels = _build_lexical_channels(
        documents=documents,
        sections_by_document=sections_by_document,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        component_depth=CANDIDATE_POOL_DEPTH,
    )
    channels = tuple([*lexical_channels, *dense_channels])
    channels_at = time.perf_counter()
    _emit(progress_sink, phase="retrieval_channels_ready", channel_count=len(channels))

    records = PrimeQAHybridCandidateDatasetBuilder(
        documents_by_id=documents_by_id,
        sections_by_document=sections_by_document,
        channels=channels,
        fold_assignments=fold_assignments,
        progress_sink=progress_sink,
        progress_stage=_STAGE,
        progress_phase="dev_candidate_pool_build",
    ).build(samples)
    grouped_records = records_by_sample(records)
    records_at = time.perf_counter()

    current_evaluation = _evaluate_selection_run(
        samples=samples,
        grouped_records=grouped_records,
        selection_run=_control_selection_run(
            grouped_records=grouped_records,
            selector=select_current_query_overlap_top10,
        ),
        documents_by_id=documents_by_id,
    )
    rrf_selection_run = _control_selection_run(
        grouped_records=grouped_records,
        selector=select_original_rrf_top10,
    )
    rrf_evaluation = _evaluate_selection_run(
        samples=samples,
        grouped_records=grouped_records,
        selection_run=rrf_selection_run,
        documents_by_id=documents_by_id,
    )
    evaluated_at = time.perf_counter()
    comparison = _comparison(candidate=rrf_evaluation, current=current_evaluation)
    policy_structure = _policy_structure_audit(rrf_selection_run.selections)
    policy_guards = _policy_guard_results(
        candidate=rrf_evaluation,
        current=current_evaluation,
        comparison=comparison,
        policy_structure=policy_structure,
    )

    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "One-shot independent development validation of the train-frozen untouched "
            "Stage116 RRF Top10 generation-context policy against the current query-overlap "
            "Top10 control. No fitting, threshold tuning, policy search, test evaluation, "
            "Agent runtime integration, or fallback is performed."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_validation),
            "confirmation_note": confirmation_note,
        },
        "source_authorization": source_authorization,
        "frozen_protocol": protocol,
        "frozen_protocol_sha256": protocol_sha256,
        "split_contract": {
            "split_name": "primeqa_hybrid_stage68_v1",
            "protocol_version": "primeqa_hybrid_split_v1",
            "loaded_split": "dev",
            "policy_fit_split": "train_in_prior_stages_only",
            "policy_frozen_before_dev_load": True,
            "dev_used_for_fit_selection_or_tuning": False,
            "test_split_loaded": False,
            "test_metrics_run": False,
        },
        "analysis_config": {
            "candidate_pool_depth": CANDIDATE_POOL_DEPTH,
            "generation_context_depth": CONTEXT_DEPTH,
            "fold_count": _FOLD_COUNT,
            "candidate_policy_count": 1,
            "candidate_policy_id": _RRF_POLICY_ID,
            "include_dense_channels": include_dense_channels,
            "encoder_batch_size": encoder_batch_size,
            "encoder_device": encoder_device or "configured_default",
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
        },
        "loaded_data_summary": {
            "train_rows_loaded": 0,
            "dev_row_count": len(samples),
            "dev_answerable_count": sum(sample.answerable for sample in samples),
            "dev_unanswerable_count": sum(not sample.answerable for sample in samples),
            "test_rows_loaded": 0,
            "document_count": len(documents_by_id),
            "section_count": sum(len(value) for value in sections_by_document.values()),
            "raw_candidate_rows_written": False,
        },
        "grouped_fold_summary": _fold_assignment_summary(samples, fold_assignments),
        "dense_channel_preflight": dense_summary,
        "candidate_pool_summary": _candidate_pool_summary(records, samples),
        "dev_results": {
            _CURRENT_CONTROL_ID: _public_evaluation(current_evaluation),
            _RRF_POLICY_ID: _public_evaluation(rrf_evaluation),
        },
        "policy_comparison": comparison,
        "policy_structure_audit": policy_structure,
        "policy_guard_results": policy_guards,
        "policy_adoption": {
            "status": (
                "dev_safe_fixed_policy"
                if all(policy_guards.values())
                else "not_dev_safe_fixed_policy"
            ),
            "candidate_policy_id": _RRF_POLICY_ID,
            "all_strict_policy_guards_passed": all(policy_guards.values()),
            "failed_policy_guards": [name for name, passed in policy_guards.items() if not passed],
            "dev_used_for_fit_selection_or_tuning": False,
            "test_used": False,
        },
        "closed_boundaries": {
            "train_loaded": False,
            "test_loaded": False,
            "test_metrics_run": False,
            "dev_used_for_fit_selection_or_tuning": False,
            "agent_qwen_runtime_run": False,
            "runtime_registered_as_default": False,
            "runtime_integration_run": False,
            "fallback_strategies_enabled": False,
            "query_rewrite_enabled": False,
            "second_retrieval_enabled": False,
        },
        "timing_seconds": {
            "source_authorization_and_protocol": round(authorized_at - started_at, 6),
            "load_dev_and_documents": round(loaded_at - authorized_at, 6),
            "build_retrieval_channels": round(channels_at - loaded_at, 6),
            "build_dev_candidate_records": round(records_at - channels_at, 6),
            "evaluate_two_fixed_policies": round(evaluated_at - records_at, 6),
            "total": round(evaluated_at - started_at, 6),
        },
    }
    report["guard_checks"] = _guard_checks(report)
    report["public_safe_contract"] = _public_safe_contract(report)
    process_passed = all(check["passed"] for check in report["guard_checks"])
    report["decision"] = _decision(report=report, process_guards_passed=process_passed)
    return report


def write_stage163_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridUntouchedRRFDevVisualization]:
    """Write ten public-safe SVG views for Stage163."""

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_specs = {
        "stage163_context_gold_hit_count.svg": (
            "Stage163 dev context gold hit count",
            _metric_bars(report, "context_gold_hit_count"),
        ),
        "stage163_context_gold_hit_rate.svg": (
            "Stage163 dev context gold hit rate",
            _metric_bars(report, "context_gold_hit_rate"),
        ),
        "stage163_verified_f1_all.svg": (
            "Stage163 dev verified F1 over all answerable rows",
            _metric_bars(report, "average_token_f1_all_answerable"),
        ),
        "stage163_verified_f1_completed.svg": (
            "Stage163 dev verified F1 over completed answerable rows",
            _metric_bars(report, "average_token_f1_completed_answerable"),
        ),
        "stage163_gold_citation_count.svg": (
            "Stage163 dev gold citation count",
            _metric_bars(report, "gold_citation_count"),
        ),
        "stage163_answerable_refusal_count.svg": (
            "Stage163 dev answerable refusal count",
            _metric_bars(report, "answerable_refusal_count"),
        ),
        "stage163_unanswerable_false_answer_count.svg": (
            "Stage163 dev unanswerable false-answer count",
            _metric_bars(report, "unanswerable_false_answer_count"),
        ),
        "stage163_fold_hit_delta.svg": (
            "Stage163 fold context-hit delta versus current",
            _fold_delta_bars(report, "context_gold_hit_rate_delta"),
        ),
        "stage163_fold_f1_delta.svg": (
            "Stage163 fold verified-F1 delta versus current",
            _fold_delta_bars(report, "average_token_f1_all_answerable_delta"),
        ),
        "stage163_guard_status.svg": (
            "Stage163 policy and process guard status",
            [
                *[
                    _bar(f"policy:{name}", passed)
                    for name, passed in report["policy_guard_results"].items()
                ],
                *[
                    _bar(f"process:{check['name']}", check["passed"])
                    for check in report["guard_checks"]
                ],
            ],
        ),
    }
    artifacts = []
    for filename, (title, bars) in chart_specs.items():
        path = output_dir / filename
        path.write_text(
            render_horizontal_bar_chart_svg(title=title, bars=bars, x_label="value"),
            encoding="utf-8",
        )
        artifacts.append(PrimeQAHybridUntouchedRRFDevVisualization(filename, str(path)))
    return artifacts


def _comparison(
    *,
    candidate: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_metrics = candidate["aggregate"]
    current_metrics = current["aggregate"]
    candidate_cases = candidate["private_cases"]
    current_cases = current["private_cases"]
    improved = 0
    regressed = 0
    tied = 0
    changed = 0
    for sample_id, candidate_case in candidate_cases.items():
        current_case = current_cases[sample_id]
        if candidate_case.answerable:
            if candidate_case.token_f1_all > current_case.token_f1_all + 1e-12:
                improved += 1
            elif candidate_case.token_f1_all + 1e-12 < current_case.token_f1_all:
                regressed += 1
            else:
                tied += 1
        changed += candidate_case.answer_signature != current_case.answer_signature
    fold_deltas = {
        fold_id: {
            "context_gold_hit_rate_delta": round(
                float(metrics["context_gold_hit_rate"])
                - float(current["folds"][fold_id]["context_gold_hit_rate"]),
                6,
            ),
            "average_token_f1_all_answerable_delta": round(
                float(metrics["average_token_f1_all_answerable"])
                - float(current["folds"][fold_id]["average_token_f1_all_answerable"]),
                6,
            ),
        }
        for fold_id, metrics in candidate["folds"].items()
    }
    return {
        "aggregate_delta_vs_current": {
            "context_gold_hit_count": int(candidate_metrics["context_gold_hit_count"])
            - int(current_metrics["context_gold_hit_count"]),
            "context_gold_hit_rate": round(
                float(candidate_metrics["context_gold_hit_rate"])
                - float(current_metrics["context_gold_hit_rate"]),
                6,
            ),
            "average_token_f1_all_answerable": round(
                float(candidate_metrics["average_token_f1_all_answerable"])
                - float(current_metrics["average_token_f1_all_answerable"]),
                6,
            ),
            "gold_citation_count": int(candidate_metrics["gold_citation_count"])
            - int(current_metrics["gold_citation_count"]),
            "answerable_refusal_count": int(candidate_metrics["answerable_refusal_count"])
            - int(current_metrics["answerable_refusal_count"]),
            "unanswerable_false_answer_count": int(
                candidate_metrics["unanswerable_false_answer_count"]
            )
            - int(current_metrics["unanswerable_false_answer_count"]),
            "answerable_f1_improved_count": improved,
            "answerable_f1_regressed_count": regressed,
            "answerable_f1_tied_count": tied,
            "changed_verified_answer_count": changed,
        },
        "fold_deltas_vs_current": fold_deltas,
        "minimum_fold_hit_rate_delta": min(
            value["context_gold_hit_rate_delta"] for value in fold_deltas.values()
        ),
        "minimum_fold_f1_delta": min(
            value["average_token_f1_all_answerable_delta"] for value in fold_deltas.values()
        ),
    }


def _policy_structure_audit(selections: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": len(selections),
        "context_depth_minimum": min(len(item.selected) for item in selections.values()),
        "context_depth_maximum": max(len(item.selected) for item in selections.values()),
        "tail_promotion_count": sum(item.tail_promotion_count for item in selections.values()),
        "protected_prefix_violation_count": sum(
            item.protected_prefix_violation_count for item in selections.values()
        ),
        "policy_is_exact_untouched_rrf_top10": all(
            item.tail_promotion_count == 0
            and [record.baseline_rank for record in item.selected] == list(range(1, 11))
            for item in selections.values()
        ),
    }


def _policy_guard_results(
    *,
    candidate: Mapping[str, Any],
    current: Mapping[str, Any],
    comparison: Mapping[str, Any],
    policy_structure: Mapping[str, Any],
) -> dict[str, bool]:
    candidate_metrics = candidate["aggregate"]
    current_metrics = current["aggregate"]
    delta = comparison["aggregate_delta_vs_current"]
    return {
        "context_hit_strictly_improves_current": int(delta["context_gold_hit_count"]) > 0,
        "verified_f1_all_not_below_current": float(
            candidate_metrics["average_token_f1_all_answerable"]
        )
        + 1e-12
        >= float(current_metrics["average_token_f1_all_answerable"]),
        "gold_citations_not_below_current": int(candidate_metrics["gold_citation_count"])
        >= int(current_metrics["gold_citation_count"]),
        "answerable_refusals_not_above_current": int(candidate_metrics["answerable_refusal_count"])
        <= int(current_metrics["answerable_refusal_count"]),
        "unanswerable_false_answers_not_above_current": int(
            candidate_metrics["unanswerable_false_answer_count"]
        )
        <= int(current_metrics["unanswerable_false_answer_count"]),
        "every_fold_hit_not_below_current": float(comparison["minimum_fold_hit_rate_delta"]) >= 0.0,
        "every_fold_f1_not_below_current": float(comparison["minimum_fold_f1_delta"]) >= 0.0,
        "exact_untouched_rrf_top10_identity": bool(
            policy_structure["policy_is_exact_untouched_rrf_top10"]
        ),
    }


def _frozen_protocol() -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "candidate_policy_id": _RRF_POLICY_ID,
        "control_policy_id": _CURRENT_CONTROL_ID,
        "policy_frozen_from": "Stage162 train-only decision",
        "validation_split": "dev",
        "candidate_pool_source": "stage116_original_rrf_top200",
        "candidate_pool_depth": CANDIDATE_POOL_DEPTH,
        "generation_context_depth": CONTEXT_DEPTH,
        "candidate_policy": "untouched_original_rrf_ranks_1_through_10",
        "control_policy": "stage160_query_overlap_shortlist_top10",
        "evaluation_pipeline": "deterministic_answer_generator_plus_verifier",
        "agent_qwen_runtime": False,
        "policy_search_or_tuning_on_dev": False,
        "grouped_fold_role": "stability_report_and_strict_non_regression_only",
        "strict_policy_guards": [
            "context_hit_strictly_improves_current",
            "verified_f1_all_not_below_current",
            "gold_citations_not_below_current",
            "answerable_refusals_not_above_current",
            "unanswerable_false_answers_not_above_current",
            "every_fold_hit_not_below_current",
            "every_fold_f1_not_below_current",
            "exact_untouched_rrf_top10_identity",
        ],
        "blocked": {
            "train_load": True,
            "test_load": True,
            "dev_fit_selection_or_tuning": True,
            "agent_runtime": True,
            "runtime_defaultization": True,
            "fallback": True,
            "query_rewrite": True,
            "second_retrieval": True,
        },
    }


def _authorize_sources(
    *,
    stage162_report_path: Path,
    stage160_report_path: Path,
    stage80_report_path: Path,
    dev_split_path: Path,
    documents_path: Path,
) -> dict[str, Any]:
    paths = {
        "stage162": stage162_report_path,
        "stage160": stage160_report_path,
        "stage80": stage80_report_path,
        "dev": dev_split_path,
        "documents": documents_path,
    }
    fingerprints = {name: _fingerprint(path) for name, path in paths.items()}
    mismatches = {
        name: value["sha256"]
        for name, value in fingerprints.items()
        if value["sha256"] != _EXPECTED_SOURCE_HASHES[name]
    }
    if mismatches:
        raise ValueError(f"Stage163 source fingerprint mismatch: {mismatches}")
    stage162 = _load_json_object(stage162_report_path)
    stage160 = _load_json_object(stage160_report_path)
    if stage162.get("decision", {}).get("status") != _STAGE162_STATUS:
        raise ValueError("Stage163 requires the completed Stage162 no-safe-config decision")
    if stage162.get("decision", {}).get("selected_config_id") is not None:
        raise ValueError("Stage163 requires Stage162 to select no learned swap model")
    if len(stage162.get("guard_checks", [])) != 18 or not all(
        check.get("passed") is True for check in stage162["guard_checks"]
    ):
        raise ValueError("Stage163 requires all 18 Stage162 process guards")
    if stage162.get("decision", {}).get("next_direction") != (
        "freeze_untouched_rrf_as_context_baseline_and_stop_learned_swap_family"
    ):
        raise ValueError("Stage163 requires the Stage162 untouched-RRF direction")
    if stage160.get("decision", {}).get("status") != _STAGE160_STATUS:
        raise ValueError("Stage163 requires completed Stage160 diagnostics")
    if stage160.get("decision", {}).get("dominant_answerable_refusal_mechanism") != (
        "generation_top10_loss"
    ):
        raise ValueError("Stage163 requires Stage160 generation Top10 loss evidence")
    return {
        "fingerprints": fingerprints,
        "stage162_status": _STAGE162_STATUS,
        "stage162_selected_config_id": None,
        "stage162_process_guards": 18,
        "stage160_status": _STAGE160_STATUS,
        "stage160_dominant_mechanism": "generation_top10_loss",
    }


def _guard_checks(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    loaded = report["loaded_data_summary"]
    pool = report["candidate_pool_summary"]
    folds = report["grouped_fold_summary"]
    current = report["dev_results"][_CURRENT_CONTROL_ID]["aggregate"]
    boundaries = report["closed_boundaries"]
    adoption = report["policy_adoption"]
    return [
        _check("user_confirmed_stage163", report["user_confirmation"]["confirmed"] is True),
        _check(
            "frozen_protocol_identity_exact",
            report["frozen_protocol"]["protocol_id"] == _PROTOCOL_ID
            and report["frozen_protocol_sha256"]
            == _canonical_json_sha256(report["frozen_protocol"]),
        ),
        _check(
            "stage162_source_exact",
            report["source_authorization"]["fingerprints"]["stage162"]["sha256"]
            == _EXPECTED_SOURCE_HASHES["stage162"],
        ),
        _check(
            "dev_source_exact",
            report["source_authorization"]["fingerprints"]["dev"]["sha256"]
            == _EXPECTED_SOURCE_HASHES["dev"],
        ),
        _check(
            "only_dev_questions_loaded",
            loaded["train_rows_loaded"] == 0
            and loaded["dev_row_count"] == _EXPECTED_DEV_ROWS
            and loaded["test_rows_loaded"] == 0,
        ),
        _check(
            "dev_answerability_exact",
            loaded["dev_answerable_count"] == _EXPECTED_ANSWERABLE_ROWS
            and loaded["dev_unanswerable_count"] == _EXPECTED_UNANSWERABLE_ROWS,
        ),
        _check(
            "grouped_five_fold_isolation_exact",
            folds["fold_count"] == _FOLD_COUNT and folds["cross_fold_group_violation_count"] == 0,
        ),
        _check(
            "stage116_top200_candidate_pool_shape_exact",
            pool["candidate_record_count_in_memory"] == _EXPECTED_CANDIDATE_ROWS
            and pool["sample_pool_count"] == _EXPECTED_DEV_ROWS
            and pool["minimum_pool_depth"] == CANDIDATE_POOL_DEPTH
            and pool["maximum_pool_depth"] == CANDIDATE_POOL_DEPTH,
        ),
        _check(
            "stage160_current_top10_reproduced",
            current["context_gold_hit_count"] == _EXPECTED_CURRENT_TOP10_GOLD_HIT_COUNT,
        ),
        _check(
            "one_fixed_candidate_policy_only",
            report["analysis_config"]["candidate_policy_count"] == 1
            and report["analysis_config"]["candidate_policy_id"] == _RRF_POLICY_ID,
        ),
        _check(
            "policy_frozen_before_dev",
            report["split_contract"]["policy_frozen_before_dev_load"] is True
            and report["split_contract"]["dev_used_for_fit_selection_or_tuning"] is False,
        ),
        _check(
            "adoption_uses_only_frozen_policy_guards",
            adoption["candidate_policy_id"] == _RRF_POLICY_ID
            and adoption["dev_used_for_fit_selection_or_tuning"] is False
            and adoption["test_used"] is False,
        ),
        _check(
            "raw_candidate_rows_not_written",
            loaded["raw_candidate_rows_written"] is False
            and pool["raw_candidate_rows_written"] is False,
        ),
        _check(
            "train_test_closed",
            boundaries["train_loaded"] is False
            and boundaries["test_loaded"] is False
            and boundaries["test_metrics_run"] is False,
        ),
        _check(
            "dev_no_fit_selection_tuning",
            boundaries["dev_used_for_fit_selection_or_tuning"] is False,
        ),
        _check(
            "agent_and_runtime_closed",
            boundaries["agent_qwen_runtime_run"] is False
            and boundaries["runtime_registered_as_default"] is False
            and boundaries["runtime_integration_run"] is False,
        ),
        _check(
            "fallback_rewrite_second_retrieval_closed",
            boundaries["fallback_strategies_enabled"] is False
            and boundaries["query_rewrite_enabled"] is False
            and boundaries["second_retrieval_enabled"] is False,
        ),
    ]


def _decision(
    *,
    report: Mapping[str, Any],
    process_guards_passed: bool,
) -> dict[str, Any]:
    policy_passed = report["policy_adoption"]["all_strict_policy_guards_passed"]
    if not process_guards_passed:
        status = "primeqa_hybrid_untouched_rrf_dev_validation_invalid"
        next_direction = "repair_stage163_process_guards_without_reusing_dev_for_tuning"
    elif not policy_passed:
        status = "primeqa_hybrid_untouched_rrf_not_dev_safe"
        next_direction = "stop_context_policy_changes_and_analyze_gold_visible_refusals"
    else:
        status = "primeqa_hybrid_untouched_rrf_dev_validated"
        next_direction = "integrate_untouched_rrf_as_optional_nondefault_agent_context_policy"
    return {
        "status": status,
        "all_process_guards_passed": process_guards_passed,
        "failed_process_guards": [
            check["name"] for check in report["guard_checks"] if not check["passed"]
        ],
        "all_strict_policy_guards_passed": policy_passed,
        "failed_policy_guards": report["policy_adoption"]["failed_policy_guards"],
        "candidate_policy_id": _RRF_POLICY_ID,
        "dev_used_for_fit_selection_or_tuning": False,
        "test_gate_opened": False,
        "runtime_registered_as_default": False,
        "fallback_strategies_enabled": False,
        "next_direction": next_direction,
    }


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(progress_sink: ProgressSink | None, *, phase: str, **values: Any) -> None:
    if progress_sink is not None:
        progress_sink({"stage": _STAGE, "phase": phase, **values})


def _metric_bars(report: Mapping[str, Any], metric: str) -> list[BarDatum]:
    return [
        _bar(policy_id, result["aggregate"][metric])
        for policy_id, result in report["dev_results"].items()
    ]


def _fold_delta_bars(report: Mapping[str, Any], metric: str) -> list[BarDatum]:
    return [
        _bar(fold_id, values[metric])
        for fold_id, values in report["policy_comparison"]["fold_deltas_vs_current"].items()
    ]


def _bar(label: str, value: int | float | bool) -> BarDatum:
    numeric = float(value)
    if isinstance(value, bool):
        value_label = "pass" if value else "fail"
    elif isinstance(value, int):
        value_label = str(value)
    else:
        value_label = f"{numeric:.6f}"
    return BarDatum(label=str(label), value=numeric, value_label=value_label)
