from __future__ import annotations

import gc
import os
import statistics
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ts_rag_agent.application import primeqa_hybrid_evidence_entailment_cv as stage172
from ts_rag_agent.application import primeqa_hybrid_grouped_ranking_cv as stage175
from ts_rag_agent.application import primeqa_hybrid_iterative_router_calibration as stage169
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application import primeqa_hybrid_supervised_cross_encoder_cv as stage174
from ts_rag_agent.application import primeqa_hybrid_view_calibration_cv as stage176
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
    _load_json_object,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    ContextCandidateRecord,
    records_by_sample,
    select_current_query_overlap_top10,
    select_original_rrf_top10,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector_training import (
    Stage161TrainCandidateDatasetBuilder,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 177"
_CREATED_AT = "2026-07-22"
_ANALYSIS_ID = "primeqa_hybrid_listwise_second_stage_reranker_grouped_oof_v1"
_EXPECTED_TRAIN_ROWS = 562
_EXPECTED_CANDIDATE_ROWS = 112_400
_EXPECTED_VIEW_CASES = 1_124
_EXPECTED_OOF_FITS = 5
_LISTWISE_FAMILY = "listwise_none"
_METHODS = ("original_rrf", "frozen_cross_encoder", "listwise_oof")
_BASELINES = _METHODS[:2]
_RECALL_DEPTHS = (1, 3, 5, 10)
_BOOTSTRAP_REPLICATES = 2_000
_BOOTSTRAP_SEED = 177_000
_MRR_FOLD_WIN_MINIMUM = 4
_RECALL10_NONINFERIORITY_MARGIN = -0.02
_SOURCE_HASHES = {
    "stage176": "61619c229fd786698b37f456e0cfee7568db198a0842bac4a0903d7faf5005c1",
    **stage176._SOURCE_HASHES,
}
_FORBIDDEN_PUBLIC_KEYS = stage176._FORBIDDEN_PUBLIC_KEYS | {
    "gold_document_id",
    "query_rank_row",
}

ProgressSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class QueryRankRow:
    fold_id: str
    answerable: bool
    gold_present: bool
    original_rrf_rank: int | None
    frozen_cross_encoder_rank: int | None
    listwise_oof_rank: int | None

    def rank(self, method: str) -> int | None:
        return {
            "original_rrf": self.original_rrf_rank,
            "frozen_cross_encoder": self.frozen_cross_encoder_rank,
            "listwise_oof": self.listwise_oof_rank,
        }[method]


@dataclass(frozen=True)
class Stage177Visualization:
    name: str
    path: str


def run_grouped_oof_listwise_reranking(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
    pair_rows: Sequence[stage174.PairFoldRow],
    trainer: stage175.RankingFoldTrainer,
    progress_sink: ProgressSink | None,
) -> dict[str, Any]:
    folds = sorted({row.fold_id for row in pair_rows})
    if len(folds) != 5:
        raise ValueError("Stage177 requires exactly five pair folds")
    predictions: dict[str, float] = {}
    fit_summaries = []
    for fold_index, heldout_fold in enumerate(folds, start=1):
        training_folds = frozenset(fold for fold in folds if fold != heldout_fold)
        fold_predictions, summary = trainer.fit_predict(
            family=_LISTWISE_FAMILY,
            training_rows=tuple(row for row in pair_rows if row.fold_id in training_folds),
            evaluation_rows=tuple(row for row in pair_rows if row.fold_id == heldout_fold),
            fit_id=f"oof_{fold_index}_final",
            training_fold_count=len(training_folds),
            evaluation_fold_count=1,
            progress_sink=progress_sink,
        )
        predictions.update(fold_predictions)
        fit_summaries.append(summary)
        _emit(progress_sink, phase="oof_fold_complete", completed=fold_index, total=5)
    if len(fit_summaries) != _EXPECTED_OOF_FITS:
        raise RuntimeError("Stage177 OOF fit count drifted")
    if len(predictions) != len(pair_rows):
        raise RuntimeError("Stage177 OOF pair prediction coverage is incomplete")
    return {
        "pair_logits": predictions,
        "fit_summaries": tuple(fit_summaries),
    }


def build_query_rank_rows(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
    frozen_scores: Mapping[str, float],
    listwise_logits: Mapping[str, float],
) -> tuple[QueryRankRow, ...]:
    rows = []
    for sample in samples:
        records = tuple(grouped_records[sample.sample_id])
        initial = select_current_query_overlap_top10(records).selected
        alternate = select_original_rrf_top10(records).selected
        final = stage172._deduplicate_records((*initial, *alternate))
        document_ids = {record.document_id for record in final}
        gold_present = bool(
            sample.answerable
            and sample.answer_doc_id is not None
            and sample.answer_doc_id in document_ids
        )
        ranks = {method: None for method in _METHODS}
        if gold_present:
            ranks["original_rrf"] = _gold_rank(
                records=final,
                answer_doc_id=sample.answer_doc_id,
                key=lambda record: (record.baseline_rank, record.document_id),
            )
            ranks["frozen_cross_encoder"] = _gold_rank(
                records=final,
                answer_doc_id=sample.answer_doc_id,
                key=lambda record: (
                    -frozen_scores[stage173._pair_identity(sample.sample_id, record.document_id)],
                    record.baseline_rank,
                    record.document_id,
                ),
            )
            ranks["listwise_oof"] = _gold_rank(
                records=final,
                answer_doc_id=sample.answer_doc_id,
                key=lambda record: (
                    -listwise_logits[stage173._pair_identity(sample.sample_id, record.document_id)],
                    record.baseline_rank,
                    record.document_id,
                ),
            )
        fold_id = records[0].fold_id
        rows.append(
            QueryRankRow(
                fold_id=fold_id,
                answerable=sample.answerable,
                gold_present=gold_present,
                original_rrf_rank=ranks["original_rrf"],
                frozen_cross_encoder_rank=ranks["frozen_cross_encoder"],
                listwise_oof_rank=ranks["listwise_oof"],
            )
        )
    if len(rows) != len(samples):
        raise RuntimeError("Stage177 query rank coverage is incomplete")
    return tuple(rows)


def _gold_rank(
    *,
    records: Sequence[ContextCandidateRecord],
    answer_doc_id: str | None,
    key: Callable[[ContextCandidateRecord], Any],
) -> int:
    ordered = sorted(records, key=key)
    for index, record in enumerate(ordered, start=1):
        if record.document_id == answer_doc_id:
            return index
    raise RuntimeError("Stage177 gold rank requested for a missing document")


def evaluate_reranking(
    rows: Sequence[QueryRankRow],
) -> dict[str, Any]:
    answerable_rows = tuple(row for row in rows if row.answerable)
    eligible_rows = tuple(row for row in answerable_rows if row.gold_present)
    if not answerable_rows or not eligible_rows:
        raise ValueError("Stage177 reranking metrics require answerable and eligible rows")
    method_metrics = {
        method: _method_metrics(
            rows=answerable_rows,
            eligible_rows=eligible_rows,
            method=method,
        )
        for method in _METHODS
    }
    fold_metrics = {
        fold_id: {
            method: _method_metrics(
                rows=tuple(row for row in answerable_rows if row.fold_id == fold_id),
                eligible_rows=tuple(row for row in eligible_rows if row.fold_id == fold_id),
                method=method,
            )
            for method in _METHODS
        }
        for fold_id in sorted({row.fold_id for row in rows})
    }
    comparisons = {
        baseline: _paired_bootstrap_comparison(
            rows=eligible_rows,
            candidate="listwise_oof",
            baseline=baseline,
        )
        for baseline in _BASELINES
    }
    gates = _reranking_gates(
        comparisons=comparisons,
        fold_metrics=fold_metrics,
    )
    return {
        "answerable_query_count": len(answerable_rows),
        "gold_present_query_count": len(eligible_rows),
        "candidate_pool_gold_coverage": round(len(eligible_rows) / len(answerable_rows), 6),
        "method_metrics": method_metrics,
        "fold_metrics": fold_metrics,
        "paired_bootstrap": comparisons,
        "quality_gates": gates,
        "all_quality_gates_passed": all(gate["passed"] for gate in gates),
    }


def _method_metrics(
    *,
    rows: Sequence[QueryRankRow],
    eligible_rows: Sequence[QueryRankRow],
    method: str,
) -> dict[str, Any]:
    ranks = [row.rank(method) for row in eligible_rows]
    if any(rank is None for rank in ranks):
        raise RuntimeError("Stage177 eligible rank is missing")
    integer_ranks = [int(rank) for rank in ranks if rank is not None]
    return {
        "conditional_query_count": len(integer_ranks),
        "mean_reciprocal_rank": round(statistics.fmean(1.0 / rank for rank in integer_ranks), 6),
        "mean_gold_rank": round(statistics.fmean(integer_ranks), 6),
        "median_gold_rank": round(float(statistics.median(integer_ranks)), 6),
        "conditional_recall_at": {
            str(depth): round(
                sum(rank <= depth for rank in integer_ranks) / len(integer_ranks),
                6,
            )
            for depth in _RECALL_DEPTHS
        },
        "all_answerable_recall_at": {
            str(depth): round(
                sum(row.rank(method) is not None and int(row.rank(method)) <= depth for row in rows)
                / len(rows),
                6,
            )
            for depth in _RECALL_DEPTHS
        },
    }


def _paired_bootstrap_comparison(
    *,
    rows: Sequence[QueryRankRow],
    candidate: str,
    baseline: str,
) -> dict[str, Any]:
    candidate_ranks = np.asarray([int(row.rank(candidate)) for row in rows], dtype=float)
    baseline_ranks = np.asarray([int(row.rank(baseline)) for row in rows], dtype=float)
    metric_values = {
        "mean_reciprocal_rank": (1.0 / candidate_ranks, 1.0 / baseline_ranks),
        "recall_at_3": (
            (candidate_ranks <= 3).astype(float),
            (baseline_ranks <= 3).astype(float),
        ),
        "recall_at_10": (
            (candidate_ranks <= 10).astype(float),
            (baseline_ranks <= 10).astype(float),
        ),
    }
    rng = np.random.default_rng(_BOOTSTRAP_SEED + _BASELINES.index(baseline))
    sample_indices = rng.integers(
        0,
        len(rows),
        size=(_BOOTSTRAP_REPLICATES, len(rows)),
    )
    summaries = {}
    for metric, (candidate_values, baseline_values) in metric_values.items():
        paired_values = candidate_values - baseline_values
        sampled = paired_values[sample_indices].mean(axis=1)
        summaries[metric] = {
            "observed_delta": round(float(paired_values.mean()), 6),
            "ci95_lower": round(float(np.quantile(sampled, 0.025)), 6),
            "ci95_upper": round(float(np.quantile(sampled, 0.975)), 6),
        }
    return {
        "baseline": baseline,
        "replicates": _BOOTSTRAP_REPLICATES,
        "seed": _BOOTSTRAP_SEED + _BASELINES.index(baseline),
        "metrics": summaries,
    }


def _reranking_gates(
    *,
    comparisons: Mapping[str, Mapping[str, Any]],
    fold_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    gates = []
    for baseline in _BASELINES:
        comparison = comparisons[baseline]["metrics"]
        fold_wins = sum(
            methods["listwise_oof"]["mean_reciprocal_rank"]
            > methods[baseline]["mean_reciprocal_rank"]
            for methods in fold_metrics.values()
        )
        gates.extend(
            (
                _gate(
                    f"mrr_ci_lower_positive_vs_{baseline}",
                    comparison["mean_reciprocal_rank"]["ci95_lower"],
                    0.0,
                    "gt",
                ),
                _gate(
                    f"recall3_ci_lower_nonnegative_vs_{baseline}",
                    comparison["recall_at_3"]["ci95_lower"],
                    0.0,
                    "ge",
                ),
                _gate(
                    f"recall10_ci_lower_noninferior_vs_{baseline}",
                    comparison["recall_at_10"]["ci95_lower"],
                    _RECALL10_NONINFERIORITY_MARGIN,
                    "ge",
                ),
                _gate(
                    f"mrr_fold_wins_vs_{baseline}",
                    fold_wins,
                    _MRR_FOLD_WIN_MINIMUM,
                    "ge",
                ),
            )
        )
    return gates


def _gate(name: str, observed: float | int, threshold: float | int, direction: str) -> dict:
    passed = observed > threshold if direction == "gt" else observed >= threshold
    return {
        "name": name,
        "observed": observed,
        "threshold": threshold,
        "direction": direction,
        "passed": bool(passed),
    }


def run_stage177_listwise_reranker_cv(
    *,
    stage176_report_path: Path,
    stage175_report_path: Path,
    stage174_report_path: Path,
    stage173_report_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    documents_path: Path,
    model_snapshot_path: Path,
    encoder_batch_size: int = 64,
    progress_sink: ProgressSink | None = None,
    trainer_factory: Callable[
        [Path, stage169.Stage169ResourceTracker, Any], stage175.RankingFoldTrainer
    ]
    | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    started_cpu = time.process_time()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    source_paths = {
        "stage176": stage176_report_path,
        "stage175": stage175_report_path,
        "stage174": stage174_report_path,
        "stage173": stage173_report_path,
        "stage80": stage80_report_path,
        "train": train_split_path,
        "documents": documents_path,
        **{
            source_name: model_snapshot_path / filename
            for source_name, filename in stage173._MODEL_SOURCE_FILES.items()
        },
    }
    fingerprints = {
        name: stage173._resolved_fingerprint(path) for name, path in source_paths.items()
    }
    _authorize_sources(fingerprints)
    stage176_report = _load_json_object(stage176_report_path)
    if stage176_report.get("decision", {}).get("status") != (
        "stage176_view_calibration_insufficient"
    ):
        raise ValueError("Stage176 did not authorize pure reranking evaluation")
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="sources_authorized")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage177 formal reranking requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    tracker = stage169.Stage169ResourceTracker(torch_module=torch)
    tracker.capture("analysis_start")

    samples = load_primeqa_hybrid_split_samples(train_split_path)
    if len(samples) != _EXPECTED_TRAIN_ROWS or any(
        sample.assigned_split != "train" for sample in samples
    ):
        raise ValueError("Stage177 accepts only the exact 562-row train split")
    documents_by_id = load_primeqa_documents(documents_path)
    sections_by_document = load_primeqa_document_sections(documents_path)
    documents = list(documents_by_id.values())
    fold_assignments = _build_train_fold_assignments(samples, fold_count=5)
    stage80 = _load_json_object(stage80_report_path)
    dense_channels, dense_summary = _build_dense_channels(
        include_dense_channels=True,
        stage80_report=stage80,
        stage80_report_path=stage80_report_path,
        documents=documents,
        document_ids=tuple(document.id for document in documents),
        encoder_batch_size=encoder_batch_size,
        encoder_device="cpu",
        encoder_factory=None,
    )
    if dense_summary["status"] != "dense_channels_ready":
        raise RuntimeError("Stage177 requires both authorized local dense channels")
    lexical_channels = _build_lexical_channels(
        documents=documents,
        sections_by_document=sections_by_document,
        bm25_k1=1.5,
        bm25_b=0.75,
        component_depth=200,
    )
    records = Stage161TrainCandidateDatasetBuilder(
        documents_by_id=documents_by_id,
        sections_by_document=sections_by_document,
        channels=tuple([*lexical_channels, *dense_channels]),
        fold_assignments=fold_assignments,
        progress_sink=progress_sink,
        progress_stage=_STAGE,
        progress_phase="train_candidate_replay",
    ).build(samples)
    if len(records) != _EXPECTED_CANDIDATE_ROWS:
        raise RuntimeError("Stage177 candidate replay row count drifted")
    grouped_records = records_by_sample(records)
    replay_ready_at = time.perf_counter()
    tracker.capture("candidate_replay_ready")

    frozen_scorer = stage173.LocalCrossEncoderSemanticScorer(
        snapshot_path=model_snapshot_path,
        tracker=tracker,
    )
    frozen_model_ready_at = time.perf_counter()
    tracker.capture("frozen_cross_encoder_loaded")
    base_cases, pairs, frozen_scores, frozen_summary = stage173.build_semantic_evidence_cases(
        samples=samples,
        grouped_records=grouped_records,
        documents_by_id=documents_by_id,
        scorer=frozen_scorer,
        text_policy=stage173.QueryAwareCrossEncoderTextPolicy(),
        progress_sink=progress_sink,
    )
    if len(base_cases) != _EXPECTED_VIEW_CASES:
        raise RuntimeError("Stage177 base view case count drifted")
    del frozen_scorer
    gc.collect()
    torch.cuda.empty_cache()
    tracker.capture("frozen_cross_encoder_released")
    pair_rows = stage174.build_pair_fold_rows(
        pairs=pairs,
        base_cases=base_cases,
        frozen_scores=frozen_scores,
    )
    pair_data_ready_at = time.perf_counter()

    factory = trainer_factory or (
        lambda snapshot, resource_tracker, torch_module: stage175.LocalGroupedRankingTrainer(
            snapshot_path=snapshot,
            tracker=resource_tracker,
            torch_module=torch_module,
        )
    )
    trainer = factory(model_snapshot_path, tracker, torch)
    oof = run_grouped_oof_listwise_reranking(
        samples=samples,
        grouped_records=grouped_records,
        pair_rows=pair_rows,
        trainer=trainer,
        progress_sink=progress_sink,
    )
    oof_finished_at = time.perf_counter()
    tracker.capture("listwise_oof_finished")
    fit_summaries: Sequence[stage175.RankingFitSummary] = oof["fit_summaries"]
    rank_rows = build_query_rank_rows(
        samples=samples,
        grouped_records=grouped_records,
        frozen_scores=frozen_scores,
        listwise_logits=oof["pair_logits"],
    )
    evaluation = evaluate_reranking(rank_rows)
    evaluation_finished_at = time.perf_counter()
    tracker.capture("reranking_evaluation_finished")
    candidate_selected = bool(evaluation["all_quality_gates_passed"])
    finished_at = time.perf_counter()
    snapshots = tracker.snapshots

    process_guards = [
        _check("stage176_authorized_pure_reranking", True),
        _check("exact_train_row_count", len(samples) == _EXPECTED_TRAIN_ROWS),
        _check("exact_candidate_row_count", len(records) == _EXPECTED_CANDIDATE_ROWS),
        _check("complete_pair_score_coverage", len(frozen_scores) == len(pair_rows)),
        _check("exact_grouped_oof_fit_count", len(fit_summaries) == _EXPECTED_OOF_FITS),
        _check("complete_oof_pair_coverage", len(oof["pair_logits"]) == len(pair_rows)),
        _check("complete_query_rank_coverage", len(rank_rows) == len(samples)),
        _check("three_frozen_ranking_methods", tuple(evaluation["method_metrics"]) == _METHODS),
        _check("two_thousand_paired_bootstrap_replicates", _BOOTSTRAP_REPLICATES == 2_000),
        _check(
            "listwise_family_only", all(row.family == _LISTWISE_FAMILY for row in fit_summaries)
        ),
        _check("development_not_loaded", True),
        _check("test_not_loaded", True),
        _check("answer_generation_not_run", True),
        _check("agent_turns_not_run", True),
        _check("model_checkpoint_not_written", True),
        _check("sufficiency_gate_not_run", True),
        _check("retry_count_zero", True),
        _check("fallback_count_zero", True),
        _check("default_runtime_unchanged", True),
    ]
    outer_inference_seconds = sum(summary.inference_seconds for summary in fit_summaries)
    resources = {
        "sampling_mode": "event_driven_in_process_without_monitor_polling",
        "phase_snapshot_count": len(snapshots),
        "wall_time_seconds": round(finished_at - started_at, 6),
        "process_cpu_time_seconds": round(time.process_time() - started_cpu, 6),
        "process_peak_working_set_bytes": max(
            snapshot.process_peak_working_set_bytes for snapshot in snapshots
        ),
        "process_peak_private_usage_bytes": max(
            snapshot.process_private_usage_bytes for snapshot in snapshots
        ),
        "minimum_system_available_memory_bytes": min(
            snapshot.system_available_memory_bytes for snapshot in snapshots
        ),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "gpu_model_device": torch.cuda.get_device_name(0),
        "frozen_semantic_pair_count": frozen_summary.pair_count,
        "frozen_semantic_scoring_seconds": frozen_summary.scoring_seconds,
        "oof_fit_count": len(fit_summaries),
        "optimizer_step_count": sum(summary.optimizer_step_count for summary in fit_summaries),
        "fine_tuning_seconds": round(sum(summary.fit_seconds for summary in fit_summaries), 6),
        "oof_inference_seconds": round(outer_inference_seconds, 6),
        "oof_inference_pairs_per_second": round(len(pair_rows) / outer_inference_seconds, 6),
        "estimated_twenty_pair_query_seconds": round(
            20.0 * outer_inference_seconds / len(pair_rows), 6
        ),
        "model_generation_calls": 0,
    }
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train-only five-fold grouped OOF evaluation of the frozen listwise-none "
            "configuration as a pure second-stage document reranker."
        ),
        "source_authorization": fingerprints,
        "frozen_protocol": {
            "model_snapshot_revision": model_snapshot_path.name,
            "training_family": _LISTWISE_FAMILY,
            "oof_fit_count": _EXPECTED_OOF_FITS,
            "methods": list(_METHODS),
            "recall_depths": list(_RECALL_DEPTHS),
            "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
            "bootstrap_seed": _BOOTSTRAP_SEED,
            "mrr_fold_win_minimum": _MRR_FOLD_WIN_MINIMUM,
            "recall10_noninferiority_margin": _RECALL10_NONINFERIORITY_MARGIN,
            "evidence_sufficiency_gate_enabled": False,
            "development_and_test_closed": True,
        },
        "split_contract": {
            "loaded_split": "train",
            "model_fit": "other_four_question_folds_only",
            "outer_evaluation": "one_shot_heldout_question_fold",
            "development_loaded": False,
            "test_loaded": False,
        },
        "pair_data_summary": {
            "complete_pair_count": len(pair_rows),
            "private_pair_rows_written": False,
        },
        "reranking_evaluation": evaluation,
        "training_diagnostics": _training_diagnostics(fit_summaries),
        "resource_consumption": resources,
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "candidate_replay": round(replay_ready_at - authorized_at, 6),
            "frozen_cross_encoder_load": round(frozen_model_ready_at - replay_ready_at, 6),
            "frozen_pair_build_and_score": round(pair_data_ready_at - frozen_model_ready_at, 6),
            "listwise_grouped_oof": round(oof_finished_at - pair_data_ready_at, 6),
            "reranking_evaluation": round(evaluation_finished_at - oof_finished_at, 6),
            "report_assembly": round(finished_at - evaluation_finished_at, 6),
        },
        "closed_boundaries": {
            "development_opened": False,
            "test_opened": False,
            "answer_generation_run": False,
            "agent_turns_run": False,
            "model_checkpoint_written": False,
            "sufficiency_gate_run": False,
            "retry_action_count": 0,
            "fallback_action_count": 0,
            "runtime_registered_as_default": False,
        },
        "process_guards": process_guards,
        "decision": {
            "candidate_selected": candidate_selected,
            "status": (
                "advance_to_stage178_listwise_reranker_agent_e2e"
                if candidate_selected
                else "stage177_listwise_reranker_insufficient"
            ),
            "development_opened": False,
            "test_opened": False,
            "default_runtime_activation": False,
        },
    }
    forbidden = sorted(stage176._forbidden_keys_found(report) | _forbidden_keys_found(report))
    report["public_safe_contract"] = {
        "forbidden_keys": sorted(_FORBIDDEN_PUBLIC_KEYS),
        "forbidden_keys_found": forbidden,
    }
    report["process_guards"].append(
        _check("public_report_contains_no_forbidden_keys", not forbidden)
    )
    all_process_guards = all(guard["passed"] for guard in report["process_guards"])
    report["decision"]["all_process_guards_passed"] = all_process_guards
    if not all_process_guards:
        report["decision"]["candidate_selected"] = False
        report["decision"]["status"] = "stage177_process_invalid"
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report


def write_stage177_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage177Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = report["reranking_evaluation"]
    methods = evaluation["method_metrics"]
    resources = report["resource_consumption"]
    diagnostics = report["training_diagnostics"]
    timings = report["timing_seconds"]
    charts = {
        "method_mrr.svg": _chart(
            "Stage 177 conditional mean reciprocal rank",
            tuple(
                BarDatum(method, row["mean_reciprocal_rank"], f"{row['mean_reciprocal_rank']:.3f}")
                for method, row in methods.items()
            ),
            x_label="MRR",
        ),
        "conditional_recall.svg": _chart(
            "Stage 177 gold-present conditional Recall at K",
            tuple(
                _rate_bar(f"{method} R@{depth}", row["conditional_recall_at"][str(depth)])
                for method, row in methods.items()
                for depth in _RECALL_DEPTHS
            ),
            x_label="Recall",
        ),
        "all_answerable_recall.svg": _chart(
            "Stage 177 all-answerable Recall at K",
            tuple(
                _rate_bar(f"{method} R@{depth}", row["all_answerable_recall_at"][str(depth)])
                for method, row in methods.items()
                for depth in _RECALL_DEPTHS
            ),
            x_label="Recall",
        ),
        "quality_gates.svg": _chart(
            "Stage 177 paired reranking quality gates",
            tuple(
                BarDatum(gate["name"], float(gate["passed"]), str(gate["observed"]))
                for gate in evaluation["quality_gates"]
            ),
            x_label="1 means passed",
        ),
        "fold_mrr.svg": _chart(
            "Stage 177 held-out fold MRR",
            tuple(
                BarDatum(
                    f"{fold_id} {method}",
                    row["mean_reciprocal_rank"],
                    f"{row['mean_reciprocal_rank']:.3f}",
                )
                for fold_id, fold_methods in evaluation["fold_metrics"].items()
                for method, row in fold_methods.items()
            ),
            x_label="MRR",
        ),
        "training_loss.svg": _chart(
            "Stage 177 mean listwise training loss",
            (
                BarDatum(
                    "First epoch",
                    diagnostics["first_epoch_loss_mean"],
                    f"{diagnostics['first_epoch_loss_mean']:.4f}",
                ),
                BarDatum(
                    "Final epoch",
                    diagnostics["final_epoch_loss_mean"],
                    f"{diagnostics['final_epoch_loss_mean']:.4f}",
                ),
            ),
            x_label="Listwise-none loss",
        ),
        "timing.svg": _chart(
            "Stage 177 phase wall times",
            tuple(
                BarDatum(name.replace("_", " "), value, f"{value:.2f} s")
                for name, value in timings.items()
            ),
            x_label="Seconds",
        ),
        "resources.svg": _chart(
            "Stage 177 process and GPU resource peaks",
            (
                _gib_bar("Process working set", resources["process_peak_working_set_bytes"]),
                _gib_bar("Process private usage", resources["process_peak_private_usage_bytes"]),
                _gib_bar("GPU allocated", resources["gpu_peak_allocated_bytes"]),
                _gib_bar("GPU reserved", resources["gpu_peak_reserved_bytes"]),
                _gib_bar(
                    "Minimum system available", resources["minimum_system_available_memory_bytes"]
                ),
            ),
            x_label="GiB",
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage177Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _training_diagnostics(
    summaries: Sequence[stage175.RankingFitSummary],
) -> dict[str, Any]:
    return {
        "fit_count": len(summaries),
        "first_epoch_loss_mean": round(
            statistics.fmean(summary.first_epoch_mean_loss for summary in summaries), 6
        ),
        "final_epoch_loss_mean": round(
            statistics.fmean(summary.final_epoch_mean_loss for summary in summaries), 6
        ),
        "fit_seconds_mean": round(
            statistics.fmean(summary.fit_seconds for summary in summaries), 6
        ),
        "inference_seconds_mean": round(
            statistics.fmean(summary.inference_seconds for summary in summaries), 6
        ),
        "optimizer_step_count_distribution": stage174._distribution(
            [summary.optimizer_step_count for summary in summaries]
        ),
    }


def _authorize_sources(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    for name, expected in _SOURCE_HASHES.items():
        if fingerprints[name]["sha256"] != expected:
            raise ValueError(f"Stage177 source hash mismatch: {name}")


def _forbidden_keys_found(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        found = {str(key) for key in value if str(key) in _FORBIDDEN_PUBLIC_KEYS}
        for child in value.values():
            found.update(_forbidden_keys_found(child))
        return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        found: set[str] = set()
        for child in value:
            found.update(_forbidden_keys_found(child))
        return found
    return set()


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(sink: ProgressSink | None, **event: Any) -> None:
    if sink is not None:
        sink({"stage": _STAGE, **event})


def _rate_bar(label: str, value: float) -> BarDatum:
    return BarDatum(label, value, f"{value:.1%}")


def _gib_bar(label: str, value: int) -> BarDatum:
    gib = value / (1024**3)
    return BarDatum(label, gib, f"{gib:.3f} GiB")


def _chart(title: str, bars: Sequence[BarDatum], *, x_label: str) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        width=1200,
        margin_left=440,
        margin_right=200,
    )
