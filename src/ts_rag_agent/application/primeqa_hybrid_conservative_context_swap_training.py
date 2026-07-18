from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_conservative_context_swap_selector import (
    CONSERVATIVE_SWAP_PROTOCOL_ID,
    MARGIN_THRESHOLD_QUANTILES,
    PROTECTED_PREFIX_AND_BUDGET,
    ConservativeContextSwapSelector,
    ConservativeSwapPlan,
    ConservativeSwapSelectorConfig,
    frozen_stage162_swap_configs,
    margin_threshold_candidates,
)
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
    ContextCandidateRecord,
    ContextSelection,
    ScorerFitSummary,
    create_candidate_scorer,
    records_by_sample,
    select_current_query_overlap_top10,
    select_original_rrf_top10,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector_training import (
    Stage161TrainCandidateDatasetBuilder,
    _candidate_pool_summary,
    _canonical_json_sha256,
    _control_selection_run,
    _evaluate_selection_run,
    _fingerprint,
    _fold_assignment_summary,
    _mean,
    _public_evaluation,
    _public_safe_contract,
    _SelectionRun,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 162"
_CREATED_AT = "2026-07-19"
_ANALYSIS_ID = "primeqa_hybrid_conservative_context_swap_nested_train_cv_v1"
_FOLD_COUNT = 5
_INNER_FOLD_COUNT = 4
_EXPECTED_TRAIN_ROWS = 562
_EXPECTED_ANSWERABLE_ROWS = 370
_EXPECTED_UNANSWERABLE_ROWS = 192
_EXPECTED_CANDIDATE_ROWS = _EXPECTED_TRAIN_ROWS * CANDIDATE_POOL_DEPTH
_EXPECTED_POOL_GOLD_HIT_COUNT = 345
_EXPECTED_CURRENT_TOP10_GOLD_HIT_COUNT = 175
_EXPECTED_CURRENT_COMPLETED_F1 = 0.194597
_EXPECTED_CURRENT_GOLD_CITATION_COUNT = 151
_EXPECTED_RRF_TOP10_GOLD_HIT_COUNT = 255
_EXPECTED_RRF_ALL_F1 = 0.201990
_EXPECTED_RRF_GOLD_CITATION_COUNT = 177
_QUERY_OVERLAP_CONTROL_ID = "stage160_query_overlap_top10_control"
_RRF_CONTROL_ID = "stage116_untouched_rrf_top10_primary_control"
_STAGE161_STATUS = "primeqa_hybrid_protected_context_selector_no_train_cv_safe_config"
_EXPECTED_SOURCE_HASHES = {
    "stage161": "a13b8ee5538581f0eb87a649c48fdf4ae715b6cfa8a43a97b5115001f9cd1197",
    "stage80": "2441bb1cb1e7888299d3f57962b18cd59df84e2086ac281105abcacfc144880f",
    "train": "cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155",
    "documents": "f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b",
}

ProgressSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class _NestedSelectionRun:
    selection_run: _SelectionRun
    outer_fold_summaries: tuple[dict[str, Any], ...]
    nested_fit_summaries: tuple[ScorerFitSummary, ...]
    fit_seconds: float


@dataclass(frozen=True)
class PrimeQAHybridConservativeSwapVisualization:
    """One generated Stage162 train-only nested-CV chart."""

    name: str
    path: str


def run_primeqa_hybrid_conservative_context_swap_training(
    *,
    stage161_report_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    documents_path: Path,
    user_confirmed_training: bool,
    confirmation_note: str,
    include_dense_channels: bool = True,
    encoder_batch_size: int = 64,
    encoder_device: str | None = None,
    encoder_factory: EncoderFactory | None = None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    progress_sink: ProgressSink | None = None,
) -> dict[str, Any]:
    """Run Stage162 nested train-only conservative context-swap development."""

    started_at = time.perf_counter()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    source_authorization = _authorize_sources(
        stage161_report_path=stage161_report_path,
        stage80_report_path=stage80_report_path,
        train_split_path=train_split_path,
        documents_path=documents_path,
    )
    protocol = _frozen_protocol()
    protocol_sha256 = _canonical_json_sha256(protocol)
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="sources_authorized")

    samples = load_primeqa_hybrid_split_samples(train_split_path)
    fold_assignments = _build_train_fold_assignments(samples, fold_count=_FOLD_COUNT)
    documents_by_id = load_primeqa_documents(documents_path)
    sections_by_document = load_primeqa_document_sections(documents_path)
    documents = list(documents_by_id.values())
    loaded_at = time.perf_counter()
    _emit(progress_sink, phase="train_and_documents_loaded", train_rows=len(samples))

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

    records = Stage161TrainCandidateDatasetBuilder(
        documents_by_id=documents_by_id,
        sections_by_document=sections_by_document,
        channels=channels,
        fold_assignments=fold_assignments,
        progress_sink=progress_sink,
        progress_stage=_STAGE,
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
    rrf_evaluation = _evaluate_selection_run(
        samples=samples,
        grouped_records=grouped_records,
        selection_run=_control_selection_run(
            grouped_records=grouped_records,
            selector=select_original_rrf_top10,
        ),
        documents_by_id=documents_by_id,
    )
    controls_at = time.perf_counter()
    _emit(progress_sink, phase="control_contexts_evaluated")

    configs = frozen_stage162_swap_configs()
    config_results = []
    for index, config in enumerate(configs, start=1):
        result = _evaluate_config(
            config=config,
            records=records,
            samples=samples,
            documents_by_id=documents_by_id,
            current_evaluation=current_evaluation,
            rrf_evaluation=rrf_evaluation,
        )
        config_results.append(result)
        _emit(
            progress_sink,
            phase="nested_swap_config_evaluated",
            completed=index,
            total=len(configs),
            config_id=config.config_id,
        )
    evaluated_at = time.perf_counter()

    selection = _select_config(config_results)
    selected_refit = _refit_selected_config(
        selected_config_id=selection["selected_config_id"],
        configs=configs,
        records=records,
        samples=samples,
        documents_by_id=documents_by_id,
    )
    guarded_at = time.perf_counter()

    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Nested train-only evaluation of four conservative learned swap families "
            "over the exact Stage116 RRF Top200 pool. Untouched RRF Top10 is the "
            "primary safety baseline. Development and test are not loaded; no runtime, "
            "fallback, query rewrite, or second retrieval is enabled."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_training),
            "selected_route": "Stage162 recommended conservative nested-CV route",
            "confirmation_note": confirmation_note,
        },
        "source_authorization": source_authorization,
        "frozen_protocol": protocol,
        "frozen_protocol_sha256": protocol_sha256,
        "split_contract": {
            "split_name": "primeqa_hybrid_stage68_v1",
            "protocol_version": "primeqa_hybrid_split_v1",
            "loaded_split": "train",
            "outer_evaluation": "grouped_five_fold_out_of_fold",
            "inner_threshold_selection": "four_inherited_group_folds_within_outer_train",
            "group_key": "normalized_question_plus_answer_document_or_unanswerable",
            "dev_split_loaded": False,
            "test_split_loaded": False,
            "dev_metrics_run": False,
            "test_metrics_run": False,
        },
        "analysis_config": {
            "candidate_pool_depth": CANDIDATE_POOL_DEPTH,
            "context_depth": CONTEXT_DEPTH,
            "outer_fold_count": _FOLD_COUNT,
            "inner_fold_count": _INNER_FOLD_COUNT,
            "include_dense_channels": include_dense_channels,
            "encoder_batch_size": encoder_batch_size,
            "encoder_device": encoder_device or "configured_default",
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "candidate_config_count": len(configs),
        },
        "loaded_data_summary": {
            "train_row_count": len(samples),
            "train_answerable_count": sum(sample.answerable for sample in samples),
            "train_unanswerable_count": sum(not sample.answerable for sample in samples),
            "document_count": len(documents_by_id),
            "section_count": sum(len(value) for value in sections_by_document.values()),
            "dev_rows_loaded": 0,
            "test_rows_loaded": 0,
            "raw_candidate_rows_written": False,
        },
        "grouped_fold_summary": _fold_assignment_summary(samples, fold_assignments),
        "dense_channel_preflight": dense_summary,
        "candidate_pool_summary": _candidate_pool_summary(records, samples),
        "control_results": {
            _QUERY_OVERLAP_CONTROL_ID: _public_evaluation(current_evaluation),
            _RRF_CONTROL_ID: _public_evaluation(rrf_evaluation),
        },
        "config_results": config_results,
        "train_nested_cv_selection": selection,
        "selected_full_train_refit": selected_refit,
        "closed_boundaries": {
            "dev_loaded": False,
            "test_loaded": False,
            "dev_used_for_fit_selection_or_metrics": False,
            "test_used_for_fit_selection_or_metrics": False,
            "runtime_registered_as_default": False,
            "runtime_integration_run": False,
            "model_artifact_written": False,
            "fallback_strategies_enabled": False,
            "query_rewrite_enabled": False,
            "second_retrieval_enabled": False,
        },
        "timing_seconds": {
            "source_authorization_and_protocol": round(authorized_at - started_at, 6),
            "load_train_and_documents": round(loaded_at - authorized_at, 6),
            "build_retrieval_channels": round(channels_at - loaded_at, 6),
            "build_train_candidate_records": round(records_at - channels_at, 6),
            "evaluate_controls": round(controls_at - records_at, 6),
            "evaluate_four_nested_configs": round(evaluated_at - controls_at, 6),
            "selection_and_optional_refit": round(guarded_at - evaluated_at, 6),
            "total": round(guarded_at - started_at, 6),
        },
    }
    report["guard_checks"] = _guard_checks(report)
    report["public_safe_contract"] = _public_safe_contract(report)
    passed = all(check["passed"] for check in report["guard_checks"])
    report["decision"] = _decision(report=report, guards_passed=passed)
    return report


def write_stage162_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridConservativeSwapVisualization]:
    """Write ten public-safe SVG views for Stage162."""

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_specs = {
        "stage162_context_gold_hit_count.svg": (
            "Stage162 nested OOF context gold hits",
            _metric_bars(report, "context_gold_hit_count"),
        ),
        "stage162_verified_f1_all.svg": (
            "Stage162 nested OOF verified F1 over all answerable rows",
            _metric_bars(report, "average_token_f1_all_answerable"),
        ),
        "stage162_gold_citation_count.svg": (
            "Stage162 nested OOF gold citation count",
            _metric_bars(report, "gold_citation_count"),
        ),
        "stage162_average_swap_count.svg": (
            "Stage162 average conservative swap count",
            _metric_bars(report, "average_tail_promotion_count"),
        ),
        "stage162_selector_latency.svg": (
            "Stage162 outer selector average latency",
            _metric_bars(report, "selection_latency_average_ms"),
        ),
        "stage162_selected_threshold_average.svg": (
            "Stage162 outer-fold selected margin threshold average",
            _config_summary_bars(report, "selected_threshold_average"),
        ),
        "stage162_minimum_fold_hit_delta.svg": (
            "Stage162 minimum outer-fold hit-rate delta vs RRF",
            _comparison_bars(report, "minimum_fold_hit_rate_delta_vs_rrf"),
        ),
        "stage162_minimum_fold_f1_delta.svg": (
            "Stage162 minimum outer-fold F1 delta vs RRF",
            _comparison_bars(report, "minimum_fold_f1_delta_vs_rrf"),
        ),
        "stage162_config_guard_status.svg": (
            "Stage162 strict train nested-CV config status",
            [
                _bar(result["config"]["config_id"], result["train_nested_cv_selectable"])
                for result in report["config_results"]
            ],
        ),
        "stage162_process_guard_status.svg": (
            "Stage162 process guard status",
            [_bar(check["name"], check["passed"]) for check in report["guard_checks"]],
        ),
    }
    artifacts = []
    for filename, (title, bars) in chart_specs.items():
        path = output_dir / filename
        path.write_text(
            render_horizontal_bar_chart_svg(title=title, bars=bars, x_label="value"),
            encoding="utf-8",
        )
        artifacts.append(PrimeQAHybridConservativeSwapVisualization(name=filename, path=str(path)))
    return artifacts


def _evaluate_config(
    *,
    config: ConservativeSwapSelectorConfig,
    records: Sequence[ContextCandidateRecord],
    samples: Sequence[PrimeQAHybridSplitSample],
    documents_by_id: Mapping[str, PrimeQADocument],
    current_evaluation: Mapping[str, Any],
    rrf_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        nested_run = _nested_outer_oof_selection_run(
            config=config,
            records=records,
            samples=samples,
            documents_by_id=documents_by_id,
        )
        grouped = records_by_sample(records)
        evaluation = _evaluate_selection_run(
            samples=samples,
            grouped_records=grouped,
            selection_run=nested_run.selection_run,
            documents_by_id=documents_by_id,
        )
        comparison = _comparison(
            evaluation=evaluation,
            current=current_evaluation,
            rrf=rrf_evaluation,
        )
        swap_audit = _swap_audit(
            selections=nested_run.selection_run.selections,
            promotion_budget=config.promotion_budget,
        )
        guards = _config_guard_results(
            evaluation=evaluation,
            rrf=rrf_evaluation,
            comparison=comparison,
            swap_audit=swap_audit,
            outer_fold_summaries=nested_run.outer_fold_summaries,
        )
        thresholds = [
            float(summary["selected_threshold"]) for summary in nested_run.outer_fold_summaries
        ]
        return {
            "config": asdict(config),
            "training_status": "completed",
            "nested_fit_summary": {
                "fit_count": len(nested_run.nested_fit_summaries),
                "fit_seconds": round(nested_run.fit_seconds, 6),
                "inner_fit_count": _FOLD_COUNT * _INNER_FOLD_COUNT,
                "outer_refit_count": _FOLD_COUNT,
                "selected_threshold_average": _mean(thresholds),
                "selected_threshold_minimum": round(min(thresholds), 12),
                "selected_threshold_maximum": round(max(thresholds), 12),
            },
            "outer_fold_thresholds": list(nested_run.outer_fold_summaries),
            "train_nested_oof_metrics": _public_evaluation(evaluation),
            "comparison": comparison,
            "swap_audit": swap_audit,
            "guard_results": guards,
            "train_nested_cv_selectable": all(guards.values()),
        }
    except Exception as error:
        return {
            "config": asdict(config),
            "training_status": "failed",
            "training_error": f"{type(error).__name__}: {error}",
            "nested_fit_summary": None,
            "outer_fold_thresholds": [],
            "train_nested_oof_metrics": None,
            "comparison": None,
            "swap_audit": None,
            "guard_results": {"training_completed": False},
            "train_nested_cv_selectable": False,
        }


def _nested_outer_oof_selection_run(
    *,
    config: ConservativeSwapSelectorConfig,
    records: Sequence[ContextCandidateRecord],
    samples: Sequence[PrimeQAHybridSplitSample],
    documents_by_id: Mapping[str, PrimeQADocument],
) -> _NestedSelectionRun:
    fold_ids = sorted({record.fold_id for record in records})
    sample_by_id = {sample.sample_id: sample for sample in samples}
    selections: dict[str, ContextSelection] = {}
    selection_latencies = []
    fit_summaries = []
    outer_summaries = []
    fit_seconds = 0.0
    for outer_fold_id in fold_ids:
        outer_train_records = [record for record in records if record.fold_id != outer_fold_id]
        outer_validation_records = [record for record in records if record.fold_id == outer_fold_id]
        inner_fold_ids = sorted({record.fold_id for record in outer_train_records})
        inner_plans: dict[str, ConservativeSwapPlan] = {}
        for inner_fold_id in inner_fold_ids:
            inner_fit_records = [
                record for record in outer_train_records if record.fold_id != inner_fold_id
            ]
            inner_validation_records = [
                record for record in outer_train_records if record.fold_id == inner_fold_id
            ]
            scorer = create_candidate_scorer(config.model_family)
            fit_started = time.perf_counter()
            fit_summaries.append(
                scorer.fit(
                    inner_fit_records,
                    protected_prefix_depth=config.protected_prefix_depth,
                )
            )
            fit_seconds += time.perf_counter() - fit_started
            selector = ConservativeContextSwapSelector(config=config, scorer=scorer)
            for sample_id, sample_records in records_by_sample(inner_validation_records).items():
                inner_plans[sample_id] = selector.plan(sample_records)
        outer_train_sample_ids = set(records_by_sample(outer_train_records))
        if set(inner_plans) != outer_train_sample_ids:
            raise RuntimeError("Stage162 inner OOF plans do not cover the outer-train rows")
        threshold_candidates = margin_threshold_candidates(tuple(inner_plans.values()))
        outer_train_samples = [
            sample_by_id[sample_id] for sample_id in sorted(outer_train_sample_ids)
        ]
        grouped_outer_train = records_by_sample(outer_train_records)
        threshold_results = _evaluate_threshold_candidates(
            thresholds=threshold_candidates,
            plans=inner_plans,
            samples=outer_train_samples,
            grouped_records=grouped_outer_train,
            documents_by_id=documents_by_id,
        )
        selected_threshold_result = _select_inner_threshold(threshold_results)
        selected_threshold = float(selected_threshold_result["threshold"])

        outer_scorer = create_candidate_scorer(config.model_family)
        fit_started = time.perf_counter()
        fit_summaries.append(
            outer_scorer.fit(
                outer_train_records,
                protected_prefix_depth=config.protected_prefix_depth,
            )
        )
        fit_seconds += time.perf_counter() - fit_started
        outer_selector = ConservativeContextSwapSelector(config=config, scorer=outer_scorer)
        for sample_id, sample_records in records_by_sample(outer_validation_records).items():
            selected_at = time.perf_counter()
            selections[sample_id] = outer_selector.select(
                sample_records,
                margin_threshold=selected_threshold,
            )
            selection_latencies.append((time.perf_counter() - selected_at) * 1000.0)
        outer_summaries.append(
            {
                "outer_fold_id": outer_fold_id,
                "outer_train_row_count": len(outer_train_sample_ids),
                "outer_validation_row_count": len(records_by_sample(outer_validation_records)),
                "inner_fold_count": len(inner_fold_ids),
                "inner_threshold_candidate_count": len(threshold_candidates),
                "selected_threshold": round(selected_threshold, 12),
                "inner_selected_metrics": selected_threshold_result["metrics"],
                "outer_validation_used_for_threshold_selection": False,
                "dev_used": False,
                "test_used": False,
            }
        )
    expected_ids = set(records_by_sample(records))
    if set(selections) != expected_ids:
        raise RuntimeError("Stage162 outer OOF selection did not cover every train row")
    return _NestedSelectionRun(
        selection_run=_SelectionRun(
            selections=selections,
            fit_summaries=tuple(fit_summaries),
            fit_seconds=fit_seconds,
            selection_latency_ms=tuple(selection_latencies),
        ),
        outer_fold_summaries=tuple(outer_summaries),
        nested_fit_summaries=tuple(fit_summaries),
        fit_seconds=fit_seconds,
    )


def _evaluate_threshold_candidates(
    *,
    thresholds: Sequence[float],
    plans: Mapping[str, ConservativeSwapPlan],
    samples: Sequence[PrimeQAHybridSplitSample],
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
    documents_by_id: Mapping[str, PrimeQADocument],
) -> list[dict[str, Any]]:
    results = []
    for threshold in thresholds:
        latencies = []
        selections = {}
        for sample_id, plan in plans.items():
            selected_at = time.perf_counter()
            selections[sample_id] = plan.select(margin_threshold=float(threshold))
            latencies.append((time.perf_counter() - selected_at) * 1000.0)
        evaluation = _evaluate_selection_run(
            samples=samples,
            grouped_records=grouped_records,
            selection_run=_SelectionRun(
                selections=selections,
                fit_summaries=(),
                fit_seconds=0.0,
                selection_latency_ms=tuple(latencies),
            ),
            documents_by_id=documents_by_id,
        )
        results.append(
            {
                "threshold": round(float(threshold), 12),
                "metrics": dict(evaluation["aggregate"]),
            }
        )
    return results


def _select_inner_threshold(
    threshold_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not threshold_results:
        raise ValueError("Stage162 inner threshold search has no candidates")
    selected = sorted(
        threshold_results,
        key=lambda result: (
            -int(result["metrics"]["context_gold_hit_count"]),
            -float(result["metrics"]["average_token_f1_all_answerable"]),
            -int(result["metrics"]["gold_citation_count"]),
            int(result["metrics"]["answerable_refusal_count"]),
            int(result["metrics"]["unanswerable_false_answer_count"]),
            float(result["metrics"]["average_tail_promotion_count"]),
            -float(result["threshold"]),
        ),
    )[0]
    return {"threshold": selected["threshold"], "metrics": dict(selected["metrics"])}


def _comparison(
    *,
    evaluation: Mapping[str, Any],
    current: Mapping[str, Any],
    rrf: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = evaluation["aggregate"]
    current_metrics = current["aggregate"]
    rrf_metrics = rrf["aggregate"]
    fold_deltas = {
        fold_id: {
            "context_gold_hit_rate_delta_vs_rrf": round(
                float(fold_metrics["context_gold_hit_rate"])
                - float(rrf["folds"][fold_id]["context_gold_hit_rate"]),
                6,
            ),
            "average_token_f1_all_answerable_delta_vs_rrf": round(
                float(fold_metrics["average_token_f1_all_answerable"])
                - float(rrf["folds"][fold_id]["average_token_f1_all_answerable"]),
                6,
            ),
        }
        for fold_id, fold_metrics in evaluation["folds"].items()
    }
    return {
        "vs_untouched_rrf": {
            "context_gold_hit_count_delta": int(metrics["context_gold_hit_count"])
            - int(rrf_metrics["context_gold_hit_count"]),
            "average_token_f1_all_answerable_delta": round(
                float(metrics["average_token_f1_all_answerable"])
                - float(rrf_metrics["average_token_f1_all_answerable"]),
                6,
            ),
            "gold_citation_count_delta": int(metrics["gold_citation_count"])
            - int(rrf_metrics["gold_citation_count"]),
            "answerable_refusal_count_delta": int(metrics["answerable_refusal_count"])
            - int(rrf_metrics["answerable_refusal_count"]),
            "unanswerable_false_answer_count_delta": int(metrics["unanswerable_false_answer_count"])
            - int(rrf_metrics["unanswerable_false_answer_count"]),
        },
        "vs_current_query_overlap": {
            "context_gold_hit_count_delta": int(metrics["context_gold_hit_count"])
            - int(current_metrics["context_gold_hit_count"]),
            "average_token_f1_all_answerable_delta": round(
                float(metrics["average_token_f1_all_answerable"])
                - float(current_metrics["average_token_f1_all_answerable"]),
                6,
            ),
        },
        "outer_fold_deltas_vs_rrf": fold_deltas,
        "minimum_fold_hit_rate_delta_vs_rrf": min(
            value["context_gold_hit_rate_delta_vs_rrf"] for value in fold_deltas.values()
        ),
        "minimum_fold_f1_delta_vs_rrf": min(
            value["average_token_f1_all_answerable_delta_vs_rrf"] for value in fold_deltas.values()
        ),
    }


def _swap_audit(
    *,
    selections: Mapping[str, ContextSelection],
    promotion_budget: int,
) -> dict[str, Any]:
    swap_counts = [selection.tail_promotion_count for selection in selections.values()]
    return {
        "sample_count": len(selections),
        "swap_count": sum(swap_counts),
        "average_swap_count": _mean(swap_counts),
        "maximum_swap_count": max(swap_counts),
        "zero_swap_sample_count": sum(count == 0 for count in swap_counts),
        "promotion_budget": promotion_budget,
        "promotion_budget_violation_count": sum(count > promotion_budget for count in swap_counts),
        "protected_prefix_violation_count": sum(
            selection.protected_prefix_violation_count for selection in selections.values()
        ),
    }


def _config_guard_results(
    *,
    evaluation: Mapping[str, Any],
    rrf: Mapping[str, Any],
    comparison: Mapping[str, Any],
    swap_audit: Mapping[str, Any],
    outer_fold_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    metrics = evaluation["aggregate"]
    rrf_metrics = rrf["aggregate"]
    delta = comparison["vs_untouched_rrf"]
    return {
        "training_completed": True,
        "context_hit_strictly_improves_untouched_rrf": int(delta["context_gold_hit_count_delta"])
        > 0,
        "verified_f1_all_not_below_untouched_rrf": float(metrics["average_token_f1_all_answerable"])
        + 1e-12
        >= float(rrf_metrics["average_token_f1_all_answerable"]),
        "gold_citations_not_below_untouched_rrf": int(metrics["gold_citation_count"])
        >= int(rrf_metrics["gold_citation_count"]),
        "answerable_refusals_not_above_untouched_rrf": int(metrics["answerable_refusal_count"])
        <= int(rrf_metrics["answerable_refusal_count"]),
        "unanswerable_false_answers_not_above_untouched_rrf": int(
            metrics["unanswerable_false_answer_count"]
        )
        <= int(rrf_metrics["unanswerable_false_answer_count"]),
        "protected_prefix_identity_exact": int(swap_audit["protected_prefix_violation_count"]) == 0,
        "promotion_budget_never_exceeded": int(swap_audit["promotion_budget_violation_count"]) == 0,
        "every_outer_fold_hit_not_below_untouched_rrf": float(
            comparison["minimum_fold_hit_rate_delta_vs_rrf"]
        )
        >= 0.0,
        "every_outer_fold_f1_not_below_untouched_rrf": float(
            comparison["minimum_fold_f1_delta_vs_rrf"]
        )
        >= 0.0,
        "nested_threshold_isolation_exact": len(outer_fold_summaries) == _FOLD_COUNT
        and all(
            summary["inner_fold_count"] == _INNER_FOLD_COUNT
            and summary["outer_validation_used_for_threshold_selection"] is False
            and summary["dev_used"] is False
            and summary["test_used"] is False
            and float(summary["selected_threshold"]) >= 0.0
            for summary in outer_fold_summaries
        ),
    }


def _select_config(config_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selectable = [result for result in config_results if result["train_nested_cv_selectable"]]
    if not selectable:
        return {
            "status": "no_train_nested_cv_safe_config",
            "selected_config_id": None,
            "selectable_config_count": 0,
            "selection_evidence": "train_grouped_nested_out_of_fold_only",
            "dev_used": False,
            "test_used": False,
        }
    selected = sorted(
        selectable,
        key=lambda result: (
            -int(result["train_nested_oof_metrics"]["aggregate"]["context_gold_hit_count"]),
            -float(
                result["train_nested_oof_metrics"]["aggregate"]["average_token_f1_all_answerable"]
            ),
            -int(result["train_nested_oof_metrics"]["aggregate"]["gold_citation_count"]),
            float(result["swap_audit"]["average_swap_count"]),
            float(result["train_nested_oof_metrics"]["aggregate"]["selection_latency_average_ms"]),
            str(result["config"]["config_id"]),
        ),
    )[0]
    return {
        "status": "train_nested_cv_safe_config_selected",
        "selected_config_id": selected["config"]["config_id"],
        "selectable_config_count": len(selectable),
        "selection_evidence": "train_grouped_nested_out_of_fold_only",
        "dev_used": False,
        "test_used": False,
    }


def _refit_selected_config(
    *,
    selected_config_id: str | None,
    configs: Sequence[ConservativeSwapSelectorConfig],
    records: Sequence[ContextCandidateRecord],
    samples: Sequence[PrimeQAHybridSplitSample],
    documents_by_id: Mapping[str, PrimeQADocument],
) -> dict[str, Any]:
    if selected_config_id is None:
        return {
            "status": "not_run_no_train_nested_cv_safe_config",
            "selected_config_id": None,
            "used_for_family_selection": False,
            "model_artifact_written": False,
        }
    config = next(config for config in configs if config.config_id == selected_config_id)
    fold_ids = sorted({record.fold_id for record in records})
    oof_plans: dict[str, ConservativeSwapPlan] = {}
    fit_summaries = []
    for fold_id in fold_ids:
        fit_records = [record for record in records if record.fold_id != fold_id]
        validation_records = [record for record in records if record.fold_id == fold_id]
        scorer = create_candidate_scorer(config.model_family)
        fit_summaries.append(
            scorer.fit(fit_records, protected_prefix_depth=config.protected_prefix_depth)
        )
        selector = ConservativeContextSwapSelector(config=config, scorer=scorer)
        for sample_id, sample_records in records_by_sample(validation_records).items():
            oof_plans[sample_id] = selector.plan(sample_records)
    grouped = records_by_sample(records)
    threshold_results = _evaluate_threshold_candidates(
        thresholds=margin_threshold_candidates(tuple(oof_plans.values())),
        plans=oof_plans,
        samples=samples,
        grouped_records=grouped,
        documents_by_id=documents_by_id,
    )
    selected_threshold_result = _select_inner_threshold(threshold_results)
    full_scorer = create_candidate_scorer(config.model_family)
    full_fit_summary = full_scorer.fit(
        records,
        protected_prefix_depth=config.protected_prefix_depth,
    )
    full_selector = ConservativeContextSwapSelector(config=config, scorer=full_scorer)
    selections = {
        sample_id: full_selector.select(
            sample_records,
            margin_threshold=float(selected_threshold_result["threshold"]),
        )
        for sample_id, sample_records in grouped.items()
    }
    evaluation = _evaluate_selection_run(
        samples=samples,
        grouped_records=grouped,
        selection_run=_SelectionRun(
            selections=selections,
            fit_summaries=(full_fit_summary,),
            fit_seconds=0.0,
            selection_latency_ms=(),
        ),
        documents_by_id=documents_by_id,
    )
    return {
        "status": "full_train_refit_completed_diagnostic_only",
        "selected_config_id": selected_config_id,
        "selected_threshold": selected_threshold_result["threshold"],
        "threshold_selection_metrics": selected_threshold_result["metrics"],
        "oof_threshold_fit_count": len(fit_summaries),
        "full_fit_summary": asdict(full_fit_summary),
        "in_sample_metrics": _public_evaluation(evaluation),
        "used_for_family_selection": False,
        "model_artifact_written": False,
    }


def _frozen_protocol() -> dict[str, Any]:
    return {
        "protocol_id": CONSERVATIVE_SWAP_PROTOCOL_ID,
        "selection_split": "train",
        "primary_control": _RRF_CONTROL_ID,
        "informational_control": _QUERY_OVERLAP_CONTROL_ID,
        "candidate_pool_source": "stage116_original_rrf_top200",
        "candidate_pool_depth": CANDIDATE_POOL_DEPTH,
        "generation_context_depth": CONTEXT_DEPTH,
        "outer_cv": "grouped_five_fold_out_of_fold_family_evaluation",
        "inner_cv": "four_inherited_group_folds_within_each_outer_train",
        "candidate_configs": [asdict(config) for config in frozen_stage162_swap_configs()],
        "protected_prefix_and_budget": [list(value) for value in PROTECTED_PREFIX_AND_BUDGET],
        "threshold_grid": {
            "fixed_zero": True,
            "positive_margin_quantiles": list(MARGIN_THRESHOLD_QUANTILES),
            "comparison_operator": "strictly_greater_than",
            "derived_from": "inner_out_of_fold_runtime_score_margins_only",
        },
        "inner_threshold_selection_order": [
            "context_gold_hit_count_desc",
            "verified_f1_all_desc",
            "gold_citation_count_desc",
            "answerable_refusal_count_asc",
            "unanswerable_false_answer_count_asc",
            "average_swap_count_asc",
            "threshold_desc",
        ],
        "strict_outer_guards": [
            "context_hit_strictly_improves_untouched_rrf",
            "verified_f1_all_not_below_untouched_rrf",
            "gold_citations_not_below_untouched_rrf",
            "answerable_refusals_not_above_untouched_rrf",
            "unanswerable_false_answers_not_above_untouched_rrf",
            "protected_prefix_identity_exact",
            "promotion_budget_never_exceeded",
            "every_outer_fold_hit_not_below_untouched_rrf",
            "every_outer_fold_f1_not_below_untouched_rrf",
            "nested_threshold_isolation_exact",
        ],
        "blocked": {
            "dev_load": True,
            "test_load": True,
            "fallback": True,
            "runtime_defaultization": True,
            "query_rewrite": True,
            "second_retrieval": True,
        },
    }


def _authorize_sources(
    *,
    stage161_report_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    documents_path: Path,
) -> dict[str, Any]:
    paths = {
        "stage161": stage161_report_path,
        "stage80": stage80_report_path,
        "train": train_split_path,
        "documents": documents_path,
    }
    fingerprints = {name: _fingerprint(path) for name, path in paths.items()}
    mismatches = {
        name: fingerprint["sha256"]
        for name, fingerprint in fingerprints.items()
        if fingerprint["sha256"] != _EXPECTED_SOURCE_HASHES[name]
    }
    if mismatches:
        raise ValueError(f"Stage162 source fingerprint mismatch: {mismatches}")
    stage161 = _load_json_object(stage161_report_path)
    if (stage161.get("decision") or {}).get("status") != _STAGE161_STATUS:
        raise ValueError("Stage162 requires the completed Stage161 no-safe-config decision")
    if (stage161.get("decision") or {}).get("selected_config_id") is not None:
        raise ValueError("Stage162 requires Stage161 to have selected no model")
    if not all(check.get("passed") is True for check in stage161.get("guard_checks", [])):
        raise ValueError("Stage162 requires every Stage161 process guard to pass")
    if len(stage161.get("guard_checks", [])) != 18:
        raise ValueError("Stage162 requires the exact 18 Stage161 process guards")
    if stage161.get("split_contract", {}).get("dev_split_loaded") is not False:
        raise ValueError("Stage162 requires Stage161 development to remain closed")
    if stage161.get("split_contract", {}).get("test_split_loaded") is not False:
        raise ValueError("Stage162 requires Stage161 test to remain closed")
    return {
        "fingerprints": fingerprints,
        "stage161_status": _STAGE161_STATUS,
        "stage161_selected_config_id": None,
        "stage161_process_guards": 18,
    }


def _guard_checks(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    loaded = report["loaded_data_summary"]
    pool = report["candidate_pool_summary"]
    folds = report["grouped_fold_summary"]
    current = report["control_results"][_QUERY_OVERLAP_CONTROL_ID]["aggregate"]
    rrf = report["control_results"][_RRF_CONTROL_ID]["aggregate"]
    configs = report["config_results"]
    selection = report["train_nested_cv_selection"]
    boundaries = report["closed_boundaries"]
    checks = [
        _check("user_confirmed_stage162", report["user_confirmation"]["confirmed"] is True),
        _check(
            "frozen_protocol_identity_exact",
            report["frozen_protocol"]["protocol_id"] == CONSERVATIVE_SWAP_PROTOCOL_ID
            and report["frozen_protocol_sha256"]
            == _canonical_json_sha256(report["frozen_protocol"]),
        ),
        _check(
            "stage161_source_exact",
            report["source_authorization"]["fingerprints"]["stage161"]["sha256"]
            == _EXPECTED_SOURCE_HASHES["stage161"],
        ),
        _check(
            "only_train_loaded",
            loaded["train_row_count"] == _EXPECTED_TRAIN_ROWS
            and loaded["dev_rows_loaded"] == 0
            and loaded["test_rows_loaded"] == 0,
        ),
        _check(
            "train_answerability_exact",
            loaded["train_answerable_count"] == _EXPECTED_ANSWERABLE_ROWS
            and loaded["train_unanswerable_count"] == _EXPECTED_UNANSWERABLE_ROWS,
        ),
        _check(
            "grouped_five_fold_isolation_exact",
            folds["fold_count"] == _FOLD_COUNT and folds["cross_fold_group_violation_count"] == 0,
        ),
        _check(
            "candidate_pool_exact",
            pool["candidate_record_count_in_memory"] == _EXPECTED_CANDIDATE_ROWS
            and pool["minimum_pool_depth"] == CANDIDATE_POOL_DEPTH
            and pool["maximum_pool_depth"] == CANDIDATE_POOL_DEPTH
            and pool["answerable_gold_pool_hit_count"] == _EXPECTED_POOL_GOLD_HIT_COUNT,
        ),
        _check(
            "current_query_overlap_reproduced",
            current["context_gold_hit_count"] == _EXPECTED_CURRENT_TOP10_GOLD_HIT_COUNT
            and abs(
                float(current["average_token_f1_completed_answerable"])
                - _EXPECTED_CURRENT_COMPLETED_F1
            )
            <= 0.0000005
            and current["gold_citation_count"] == _EXPECTED_CURRENT_GOLD_CITATION_COUNT,
        ),
        _check(
            "untouched_rrf_primary_control_reproduced",
            rrf["context_gold_hit_count"] == _EXPECTED_RRF_TOP10_GOLD_HIT_COUNT
            and abs(float(rrf["average_token_f1_all_answerable"]) - _EXPECTED_RRF_ALL_F1)
            <= 0.0000005
            and rrf["gold_citation_count"] == _EXPECTED_RRF_GOLD_CITATION_COUNT,
        ),
        _check(
            "four_authorized_configs_exact",
            len(configs) == 4
            and {
                (result["config"]["protected_prefix_depth"], result["config"]["promotion_budget"])
                for result in configs
            }
            == set(PROTECTED_PREFIX_AND_BUDGET)
            and {result["config"]["model_family"] for result in configs}
            == {"pairwise_logistic", "pointwise_histogram_gbdt"},
        ),
        _check(
            "four_nested_configs_completed",
            all(result["training_status"] == "completed" for result in configs),
        ),
        _check(
            "nested_outer_inner_structure_exact",
            all(
                len(result["outer_fold_thresholds"]) == _FOLD_COUNT
                and result["nested_fit_summary"]["inner_fit_count"]
                == _FOLD_COUNT * _INNER_FOLD_COUNT
                and result["nested_fit_summary"]["outer_refit_count"] == _FOLD_COUNT
                for result in configs
            ),
        ),
        _check(
            "train_nested_oof_only_selection_evidence",
            selection["selection_evidence"] == "train_grouped_nested_out_of_fold_only"
            and selection["dev_used"] is False
            and selection["test_used"] is False,
        ),
        _check(
            "selected_config_safe_or_none",
            selection["selected_config_id"] is None
            or any(
                result["config"]["config_id"] == selection["selected_config_id"]
                and result["train_nested_cv_selectable"] is True
                for result in configs
            ),
        ),
        _check(
            "dev_test_closed",
            boundaries["dev_loaded"] is False
            and boundaries["test_loaded"] is False
            and boundaries["dev_used_for_fit_selection_or_metrics"] is False
            and boundaries["test_used_for_fit_selection_or_metrics"] is False,
        ),
        _check(
            "runtime_and_fallback_closed",
            boundaries["runtime_registered_as_default"] is False
            and boundaries["runtime_integration_run"] is False
            and boundaries["fallback_strategies_enabled"] is False,
        ),
        _check(
            "rewrite_and_second_retrieval_closed",
            boundaries["query_rewrite_enabled"] is False
            and boundaries["second_retrieval_enabled"] is False,
        ),
        _check(
            "no_raw_candidate_rows_written",
            loaded["raw_candidate_rows_written"] is False
            and pool["raw_candidate_rows_written"] is False,
        ),
    ]
    return checks


def _decision(*, report: Mapping[str, Any], guards_passed: bool) -> dict[str, Any]:
    selection = report["train_nested_cv_selection"]
    selected = selection["selected_config_id"]
    if not guards_passed:
        status = "primeqa_hybrid_conservative_context_swap_training_invalid"
        next_direction = "repair_stage162_process_guards_before_any_further_evaluation"
    elif selected is None:
        status = "primeqa_hybrid_conservative_context_swap_no_train_nested_cv_safe_config"
        next_direction = "freeze_untouched_rrf_as_context_baseline_and_stop_learned_swap_family"
    else:
        status = "primeqa_hybrid_conservative_context_swap_train_nested_cv_selected"
        next_direction = "freeze_selected_swap_policy_then_run_one_shot_dev_validation"
    return {
        "status": status,
        "all_process_guards_passed": guards_passed,
        "failed_process_guards": [
            check["name"] for check in report["guard_checks"] if not check["passed"]
        ],
        "selected_config_id": selected,
        "selectable_config_count": selection["selectable_config_count"],
        "dev_gate_opened": False,
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
    bars = [
        _bar(control_id, result["aggregate"][metric])
        for control_id, result in report["control_results"].items()
    ]
    bars.extend(
        _bar(result["config"]["config_id"], result["train_nested_oof_metrics"]["aggregate"][metric])
        for result in report["config_results"]
        if result["train_nested_oof_metrics"] is not None
    )
    return bars


def _config_summary_bars(report: Mapping[str, Any], metric: str) -> list[BarDatum]:
    return [
        _bar(result["config"]["config_id"], result["nested_fit_summary"][metric])
        for result in report["config_results"]
        if result["nested_fit_summary"] is not None
    ]


def _comparison_bars(report: Mapping[str, Any], metric: str) -> list[BarDatum]:
    return [
        _bar(result["config"]["config_id"], result["comparison"][metric])
        for result in report["config_results"]
        if result["comparison"] is not None
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
