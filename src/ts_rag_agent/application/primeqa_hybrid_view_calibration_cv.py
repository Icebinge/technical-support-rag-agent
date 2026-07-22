from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import statistics
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from ts_rag_agent.application import primeqa_hybrid_evidence_entailment_cv as stage172
from ts_rag_agent.application import primeqa_hybrid_grouped_ranking_cv as stage175
from ts_rag_agent.application import primeqa_hybrid_iterative_router_calibration as stage169
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application import primeqa_hybrid_supervised_cross_encoder_cv as stage174
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

_STAGE = "Stage 176"
_CREATED_AT = "2026-07-22"
_ANALYSIS_ID = "primeqa_hybrid_listwise_view_calibration_nested_cv_v1"
_EXPECTED_TRAIN_ROWS = 562
_EXPECTED_CANDIDATE_ROWS = 112_400
_EXPECTED_VIEW_CASES = 1_124
_EXPECTED_NESTED_FITS = 25
_LISTWISE_FAMILY = "listwise_none"
_LOGIT_THRESHOLDS = stage175._MARGIN_THRESHOLDS
_BOUNDED_THRESHOLDS = tuple(round(-1.0 + (index * 0.1), 1) for index in range(21))
_EXPECTED_SPEC_COUNT = 84
_SOURCE_HASHES = {
    "stage175": "27641cf6754762260a7400aa431762c5e8e34cf9f1645f4038fa8867cc04dec8",
    **stage175._SOURCE_HASHES,
}
_FORBIDDEN_PUBLIC_KEYS = stage175._FORBIDDEN_PUBLIC_KEYS | {
    "pair_logit",
    "view_candidate_logits",
}

ProgressSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class CalibrationSpec:
    policy: str
    threshold: float

    @property
    def spec_id(self) -> str:
        material = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Stage176Visualization:
    name: str
    path: str


class ViewCalibrationPolicy(Protocol):
    name: str
    thresholds: tuple[float, ...]

    def score(self, ordered_logits: Sequence[float]) -> float: ...


class AbsoluteTop1Policy:
    name = "absolute_top1"
    thresholds = _LOGIT_THRESHOLDS

    def score(self, ordered_logits: Sequence[float]) -> float:
        return float(ordered_logits[0])


class Top1Top2NoneMarginPolicy:
    name = "top1_top2_none_margin"
    thresholds = _LOGIT_THRESHOLDS

    def score(self, ordered_logits: Sequence[float]) -> float:
        return float(ordered_logits[0] - max(0.0, ordered_logits[1]))


class CandidateMassVsNonePolicy:
    name = "candidate_mass_vs_none"
    thresholds = _LOGIT_THRESHOLDS

    def score(self, ordered_logits: Sequence[float]) -> float:
        maximum = ordered_logits[0]
        return float(maximum + math.log(sum(math.exp(value - maximum) for value in ordered_logits)))


class BoundedAbsoluteRelativePolicy:
    name = "bounded_absolute_relative"
    thresholds = _BOUNDED_THRESHOLDS

    def score(self, ordered_logits: Sequence[float]) -> float:
        top1 = ordered_logits[0]
        margin = top1 - max(0.0, ordered_logits[1])
        return float(0.5 * math.tanh(top1 / 4.0) + 0.5 * math.tanh(margin / 4.0))


_POLICIES: tuple[ViewCalibrationPolicy, ...] = (
    AbsoluteTop1Policy(),
    Top1Top2NoneMarginPolicy(),
    CandidateMassVsNonePolicy(),
    BoundedAbsoluteRelativePolicy(),
)


def build_calibration_view_cases(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
    pair_logits: Mapping[str, float],
) -> tuple[
    tuple[stage172.EvidenceViewCase, ...],
    dict[str, dict[str, float]],
]:
    base_cases = stage172.build_evidence_view_cases(
        samples=samples,
        grouped_records=grouped_records,
    )
    base_by_identity = {case.private_identity: case for case in base_cases}
    cases = []
    scores_by_policy: dict[str, dict[str, float]] = {policy.name: {} for policy in _POLICIES}
    for sample in samples:
        records = tuple(grouped_records[sample.sample_id])
        initial = select_current_query_overlap_top10(records).selected
        alternate = select_original_rrf_top10(records).selected
        final = stage172._deduplicate_records((*initial, *alternate))
        for phase, visible in (("initial", initial), ("final", final)):
            identity = stage172._sha256_text(f"{sample.sample_id}:{phase}")
            ordered_logits = sorted(
                (
                    pair_logits[stage173._pair_identity(sample.sample_id, record.document_id)]
                    for record in visible
                ),
                reverse=True,
            )
            if len(ordered_logits) < 2:
                raise RuntimeError("Stage176 evidence view requires at least two candidates")
            base = base_by_identity[identity]
            cases.append(
                stage172.EvidenceViewCase(
                    private_identity=base.private_identity,
                    group_identity=base.group_identity,
                    fold_id=base.fold_id,
                    phase=base.phase,
                    stratum=base.stratum,
                    features={},
                    sufficient_label=base.sufficient_label,
                )
            )
            for policy in _POLICIES:
                scores_by_policy[policy.name][identity] = policy.score(ordered_logits)
    if any(len(scores) != len(cases) for scores in scores_by_policy.values()):
        raise RuntimeError("Stage176 calibration score coverage is incomplete")
    return tuple(cases), scores_by_policy


def run_grouped_nested_calibration(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
    pair_rows: Sequence[stage174.PairFoldRow],
    trainer: stage175.RankingFoldTrainer,
    progress_sink: ProgressSink | None,
) -> dict[str, Any]:
    folds = sorted({row.fold_id for row in pair_rows})
    if len(folds) != 5:
        raise ValueError("Stage176 requires exactly five pair folds")
    samples_by_fold = {
        fold_id: tuple(
            sample
            for sample in samples
            if tuple(grouped_records[sample.sample_id])[0].fold_id == fold_id
        )
        for fold_id in folds
    }
    outer_rows = []
    fit_summaries = []
    oof_cases: list[stage172.EvidenceViewCase] = []
    oof_scores: dict[str, float] = {}
    selected_specs: dict[str, CalibrationSpec] = {}
    policy_oof_scores: dict[str, dict[str, float]] = {policy.name: {} for policy in _POLICIES}
    policy_oof_cases: list[stage172.EvidenceViewCase] = []
    for outer_index, outer_fold in enumerate(folds, start=1):
        outer_train_folds = tuple(fold for fold in folds if fold != outer_fold)
        inner_pair_predictions: dict[str, float] = {}
        for inner_index, inner_fold in enumerate(outer_train_folds, start=1):
            training_folds = frozenset(fold for fold in outer_train_folds if fold != inner_fold)
            fit_id = f"outer_{outer_index}_inner_{inner_index}"
            predictions, summary = trainer.fit_predict(
                family=_LISTWISE_FAMILY,
                training_rows=tuple(row for row in pair_rows if row.fold_id in training_folds),
                evaluation_rows=tuple(row for row in pair_rows if row.fold_id == inner_fold),
                fit_id=fit_id,
                training_fold_count=len(training_folds),
                evaluation_fold_count=1,
                progress_sink=progress_sink,
            )
            inner_pair_predictions.update(predictions)
            fit_summaries.append(summary)
        inner_samples = tuple(
            sample for fold in outer_train_folds for sample in samples_by_fold[fold]
        )
        inner_cases, inner_scores_by_policy = build_calibration_view_cases(
            samples=inner_samples,
            grouped_records=grouped_records,
            pair_logits=inner_pair_predictions,
        )
        selected = _select_policy_threshold(
            cases=inner_cases,
            scores_by_policy=inner_scores_by_policy,
        )
        selected_spec: CalibrationSpec = selected["spec"]

        fit_id = f"outer_{outer_index}_final"
        outer_predictions, summary = trainer.fit_predict(
            family=_LISTWISE_FAMILY,
            training_rows=tuple(row for row in pair_rows if row.fold_id in outer_train_folds),
            evaluation_rows=tuple(row for row in pair_rows if row.fold_id == outer_fold),
            fit_id=fit_id,
            training_fold_count=len(outer_train_folds),
            evaluation_fold_count=1,
            progress_sink=progress_sink,
        )
        fit_summaries.append(summary)
        heldout_cases, heldout_scores_by_policy = build_calibration_view_cases(
            samples=samples_by_fold[outer_fold],
            grouped_records=grouped_records,
            pair_logits=outer_predictions,
        )
        heldout_scores = heldout_scores_by_policy[selected_spec.policy]
        heldout_metrics = stage172.evaluate_predictions(
            heldout_cases,
            heldout_scores,
            {outer_fold: selected_spec},
        )
        oof_cases.extend(heldout_cases)
        oof_scores.update(heldout_scores)
        selected_specs[outer_fold] = selected_spec
        policy_oof_cases.extend(heldout_cases)
        for policy_name, scores in heldout_scores_by_policy.items():
            policy_oof_scores[policy_name].update(scores)
        outer_rows.append(
            {
                "heldout_fold": outer_fold,
                "inner_fit_count": len(outer_train_folds),
                "outer_fit_count": 1,
                "inner_case_count": len(inner_cases),
                "heldout_case_count": len(heldout_cases),
                "policy_diagnostics": selected["policy_diagnostics"],
                "selected_policy": selected_spec.policy,
                "selected_threshold": selected_spec.threshold,
                "selected_spec_id": selected_spec.spec_id,
                "selected_inner_eligible": bool(selected["eligible"]),
                "selected_inner_metrics": selected["metrics"].public_dict(),
                "selected_inner_safe_fold_count": selected["safe_fold_count"],
                "heldout_metrics": heldout_metrics.public_dict(),
            }
        )
        _emit(progress_sink, phase="outer_fold_complete", completed=outer_index, total=5)
    if len(fit_summaries) != _EXPECTED_NESTED_FITS:
        raise RuntimeError("Stage176 nested fit count drifted")
    if len(policy_oof_cases) != len(oof_cases):
        raise RuntimeError("Stage176 policy OOF case coverage drifted")
    return {
        "outer_folds": outer_rows,
        "oof_cases": tuple(oof_cases),
        "oof_view_scores": oof_scores,
        "selected_specs": selected_specs,
        "policy_oof_cases": tuple(policy_oof_cases),
        "policy_oof_scores": policy_oof_scores,
        "fit_summaries": tuple(fit_summaries),
    }


def _select_policy_threshold(
    *,
    cases: Sequence[stage172.EvidenceViewCase],
    scores_by_policy: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    policy_rows = []
    diagnostics = []
    for policy in _POLICIES:
        rows = [
            stage174._threshold_evaluation(
                cases,
                scores_by_policy[policy.name],
                CalibrationSpec(policy.name, threshold),
            )
            for threshold in policy.thresholds
        ]
        selected = max(rows, key=stage174._threshold_selection_key)
        selected = {
            **selected,
            "eligible_threshold_count": sum(row["eligible"] for row in rows),
        }
        policy_rows.append(selected)
        diagnostics.append(_public_selection_row(selected))
    chosen = max(policy_rows, key=_policy_selection_key)
    return {**chosen, "policy_diagnostics": diagnostics}


def _policy_selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    spec: CalibrationSpec = row["spec"]
    return (*stage174._threshold_selection_key(row), spec.policy)


def _public_selection_row(row: Mapping[str, Any]) -> dict[str, Any]:
    spec: CalibrationSpec = row["spec"]
    return {
        "policy": spec.policy,
        "threshold": spec.threshold,
        "spec_id": spec.spec_id,
        "eligible_threshold_count": row["eligible_threshold_count"],
        "selected_eligible": bool(row["eligible"]),
        "safe_fold_count": row["safe_fold_count"],
        "metrics": row["metrics"].public_dict(),
    }


def run_stage176_view_calibration_cv(
    *,
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
    stage175_report = _load_json_object(stage175_report_path)
    if stage175_report.get("decision", {}).get("status") != (
        "stage175_grouped_ranking_insufficient"
    ):
        raise ValueError("Stage175 did not authorize listwise calibration research")
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="sources_authorized")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage176 formal calibration requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    tracker = stage169.Stage169ResourceTracker(torch_module=torch)
    tracker.capture("analysis_start")

    samples = load_primeqa_hybrid_split_samples(train_split_path)
    if len(samples) != _EXPECTED_TRAIN_ROWS or any(
        sample.assigned_split != "train" for sample in samples
    ):
        raise ValueError("Stage176 accepts only the exact 562-row train split")
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
        raise RuntimeError("Stage176 requires both authorized local dense channels")
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
        raise RuntimeError("Stage176 candidate replay row count drifted")
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
        raise RuntimeError("Stage176 base view case count drifted")
    del frozen_scorer
    gc.collect()
    torch.cuda.empty_cache()
    tracker.capture("frozen_cross_encoder_released")
    pair_rows = stage174.build_pair_fold_rows(
        pairs=pairs,
        base_cases=base_cases,
        frozen_scores=frozen_scores,
    )
    sampled_groups = stage175.build_sampled_training_groups(pair_rows)
    pair_data_ready_at = time.perf_counter()
    _emit(
        progress_sink,
        phase="listwise_group_data_ready",
        complete_pair_count=len(pair_rows),
        sampled_group_count=len(sampled_groups),
    )

    factory = trainer_factory or (
        lambda snapshot, resource_tracker, torch_module: stage175.LocalGroupedRankingTrainer(
            snapshot_path=snapshot,
            tracker=resource_tracker,
            torch_module=torch_module,
        )
    )
    trainer = factory(model_snapshot_path, tracker, torch)
    nested = run_grouped_nested_calibration(
        samples=samples,
        grouped_records=grouped_records,
        pair_rows=pair_rows,
        trainer=trainer,
        progress_sink=progress_sink,
    )
    nested_finished_at = time.perf_counter()
    tracker.capture("nested_calibration_finished")

    outer_folds = nested["outer_folds"]
    oof_cases = nested["oof_cases"]
    oof_scores = nested["oof_view_scores"]
    selected_specs = nested["selected_specs"]
    fit_summaries: Sequence[stage175.RankingFitSummary] = nested["fit_summaries"]
    oof_metrics = stage172.evaluate_predictions(oof_cases, oof_scores, selected_specs)
    oof_gates = stage172._quality_gates(oof_metrics)
    fold_metrics = stage174._outer_fold_metrics(oof_cases, oof_scores, selected_specs)
    all_outer_safety_passed = all(
        metrics["insufficient_final_compose_rate"]
        <= stage172._GATE_THRESHOLDS["insufficient_final_compose_rate_max"]
        for metrics in fold_metrics.values()
    )
    all_inner_selected_eligible = all(row["selected_inner_eligible"] for row in outer_folds)

    policy_diagnostics = {}
    policy_selection_rows = []
    for policy in _POLICIES:
        selected = _select_single_policy_threshold(
            policy=policy,
            cases=nested["policy_oof_cases"],
            scores=nested["policy_oof_scores"][policy.name],
        )
        policy_selection_rows.append(selected)
        policy_diagnostics[policy.name] = _public_selection_row(selected)
    final_selection = max(policy_selection_rows, key=_policy_selection_key)
    final_spec: CalibrationSpec = final_selection["spec"]
    candidate_selected = (
        all_inner_selected_eligible
        and bool(final_selection["eligible"])
        and all(gate["passed"] for gate in oof_gates)
        and all_outer_safety_passed
    )
    tracker.capture("report_assembly")
    finished_at = time.perf_counter()
    snapshots = tracker.snapshots

    process_guards = [
        _check("stage175_authorized_listwise_calibration", True),
        _check("exact_train_row_count", len(samples) == _EXPECTED_TRAIN_ROWS),
        _check("exact_candidate_row_count", len(records) == _EXPECTED_CANDIDATE_ROWS),
        _check("exact_view_case_count", len(oof_cases) == _EXPECTED_VIEW_CASES),
        _check("complete_pair_score_coverage", len(frozen_scores) == len(pair_rows)),
        _check("exact_nested_fit_count", len(fit_summaries) == _EXPECTED_NESTED_FITS),
        _check("four_frozen_calibration_policies", len(policy_diagnostics) == 4),
        _check("eighty_four_frozen_specs", _spec_count() == _EXPECTED_SPEC_COUNT),
        _check("five_grouped_outer_folds", len(outer_folds) == 5),
        _check("complete_oof_view_coverage", len(oof_scores) == len(oof_cases)),
        _check(
            "listwise_family_only", all(row.family == _LISTWISE_FAMILY for row in fit_summaries)
        ),
        _check("two_frozen_training_epochs", stage175._TRAIN_EPOCHS == 2),
        _check("development_not_loaded", True),
        _check("test_not_loaded", True),
        _check("answer_generation_not_run", True),
        _check("agent_turns_not_run", True),
        _check("model_checkpoint_not_written", True),
        _check("retry_count_zero", True),
        _check("fallback_count_zero", True),
        _check("default_runtime_unchanged", True),
    ]
    stage175_metrics = stage175_report["nested_cv"]["oof_metrics"]
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
        "nested_fit_count": len(fit_summaries),
        "optimizer_step_count": sum(summary.optimizer_step_count for summary in fit_summaries),
        "fine_tuning_seconds": round(sum(summary.fit_seconds for summary in fit_summaries), 6),
        "fold_inference_seconds": round(
            sum(summary.inference_seconds for summary in fit_summaries), 6
        ),
        "model_generation_calls": 0,
    }
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train-only listwise-none cross-encoder adaptation with 25 grouped nested-CV "
            "fits and four predeclared view calibration policies."
        ),
        "source_authorization": fingerprints,
        "frozen_protocol": {
            "model_snapshot_revision": model_snapshot_path.name,
            "training_family": _LISTWISE_FAMILY,
            "calibration_policies": [policy.name for policy in _POLICIES],
            "logit_thresholds": list(_LOGIT_THRESHOLDS),
            "bounded_thresholds": list(_BOUNDED_THRESHOLDS),
            "spec_count": _spec_count(),
            "nested_fit_count": _EXPECTED_NESTED_FITS,
            "outer_fold_count": 5,
            "inner_model_count_per_outer_fold": 4,
            "outer_model_count_per_outer_fold": 1,
            "development_and_test_closed": True,
        },
        "split_contract": {
            "loaded_split": "train",
            "policy_and_threshold_fit": "inner_oof_question_folds_only",
            "outer_evaluation": "one_shot_heldout_question_fold",
            "policy_diagnostic": "complete_outer_oof_predictions_only",
            "development_loaded": False,
            "test_loaded": False,
        },
        "pair_data_summary": {
            "complete_pair_count": len(pair_rows),
            "sampled_group_count": len(sampled_groups),
            "sampled_pair_count": sum(len(group) for group in sampled_groups),
            "private_pair_rows_written": False,
        },
        "nested_cv": {
            "outer_folds": outer_folds,
            "fit_count": len(fit_summaries),
            "fit_summaries": [asdict(summary) for summary in fit_summaries],
            "selected_spec_ids_by_fold": {
                fold_id: spec.spec_id for fold_id, spec in selected_specs.items()
            },
            "selected_policy_counts": {
                policy.name: sum(spec.policy == policy.name for spec in selected_specs.values())
                for policy in _POLICIES
            },
            "policy_full_oof_diagnostics": policy_diagnostics,
            "final_full_train_oof_selected_policy": final_spec.policy,
            "final_full_train_oof_selected_threshold": final_spec.threshold,
            "final_full_train_oof_selected_spec_id": final_spec.spec_id,
            "final_full_train_oof_selected_eligible": bool(final_selection["eligible"]),
            "all_inner_selected_specs_eligible": all_inner_selected_eligible,
            "oof_metrics": oof_metrics.public_dict(),
            "oof_quality_gates": oof_gates,
            "outer_fold_metrics": fold_metrics,
            "all_outer_folds_safety_passed": all_outer_safety_passed,
        },
        "stage175_comparison": {
            "stage175_oof_metrics": stage175_metrics,
            "stage176_oof_metrics": oof_metrics.public_dict(),
            "metric_delta": {
                name: round(getattr(oof_metrics, name) - float(stage175_metrics[name]), 6)
                for name in (
                    "balanced_accuracy",
                    "roc_auc",
                    "initial_visible_compose_rate",
                    "alternate_only_inspect_rate",
                    "alternate_only_final_compose_rate",
                    "alternate_only_path_success_rate",
                    "insufficient_final_compose_rate",
                )
            },
        },
        "training_diagnostics": _training_diagnostics(fit_summaries),
        "resource_consumption": resources,
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "candidate_replay": round(replay_ready_at - authorized_at, 6),
            "frozen_cross_encoder_load": round(frozen_model_ready_at - replay_ready_at, 6),
            "frozen_pair_build_and_score": round(pair_data_ready_at - frozen_model_ready_at, 6),
            "nested_calibration": round(nested_finished_at - pair_data_ready_at, 6),
            "report_assembly": round(finished_at - nested_finished_at, 6),
        },
        "closed_boundaries": {
            "development_opened": False,
            "test_opened": False,
            "answer_generation_run": False,
            "agent_turns_run": False,
            "model_checkpoint_written": False,
            "retry_action_count": 0,
            "fallback_action_count": 0,
            "runtime_registered_as_default": False,
        },
        "process_guards": process_guards,
        "decision": {
            "candidate_selected": candidate_selected,
            "status": (
                "advance_to_stage177_train_only_calibrated_runtime_e2e"
                if candidate_selected
                else "stage176_view_calibration_insufficient"
            ),
            "development_opened": False,
            "test_opened": False,
            "default_runtime_activation": False,
        },
    }
    forbidden = sorted(stage175._forbidden_keys_found(report) | _forbidden_keys_found(report))
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
        report["decision"]["status"] = "stage176_process_invalid"
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report


def _select_single_policy_threshold(
    *,
    policy: ViewCalibrationPolicy,
    cases: Sequence[stage172.EvidenceViewCase],
    scores: Mapping[str, float],
) -> dict[str, Any]:
    rows = [
        stage174._threshold_evaluation(
            cases,
            scores,
            CalibrationSpec(policy.name, threshold),
        )
        for threshold in policy.thresholds
    ]
    selected = max(rows, key=stage174._threshold_selection_key)
    return {
        **selected,
        "eligible_threshold_count": sum(row["eligible"] for row in rows),
    }


def write_stage176_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage176Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    nested = report["nested_cv"]
    metrics = nested["oof_metrics"]
    stage175_metrics = report["stage175_comparison"]["stage175_oof_metrics"]
    folds = nested["outer_fold_metrics"]
    diagnostics = report["training_diagnostics"]
    resources = report["resource_consumption"]
    timings = report["timing_seconds"]
    charts = {
        "oof_quality_gates.svg": _chart(
            "Stage 176 grouped OOF quality gates",
            tuple(
                BarDatum(
                    gate["name"],
                    float(gate["passed"]),
                    f"{gate['observed']:.1%} / {gate['threshold']:.1%}",
                )
                for gate in nested["oof_quality_gates"]
            ),
            x_label="1 means passed",
        ),
        "stage175_stage176_rates.svg": _chart(
            "Stage 175 versus Stage 176 grouped OOF rates",
            tuple(
                bar
                for label, name in (
                    ("initial compose", "initial_visible_compose_rate"),
                    ("final compose", "alternate_only_final_compose_rate"),
                    ("exact path", "alternate_only_path_success_rate"),
                    ("false compose", "insufficient_final_compose_rate"),
                )
                for bar in (
                    _rate_bar(f"175 {label}", stage175_metrics[name]),
                    _rate_bar(f"176 {label}", metrics[name]),
                )
            ),
            x_label="Rate",
        ),
        "policy_selection.svg": _chart(
            "Stage 176 inner-selected calibration policies",
            tuple(
                BarDatum(policy, count, str(count))
                for policy, count in nested["selected_policy_counts"].items()
            ),
            x_label="Outer-fold selections",
        ),
        "policy_oof_quality.svg": _chart(
            "Stage 176 full-OOF policy balanced accuracy",
            tuple(
                BarDatum(
                    policy,
                    row["metrics"]["balanced_accuracy"],
                    f"{row['metrics']['balanced_accuracy']:.3f}",
                )
                for policy, row in nested["policy_full_oof_diagnostics"].items()
            ),
            x_label="Balanced accuracy",
        ),
        "outer_fold_safety.svg": _chart(
            "Stage 176 outer-fold insufficient final compose",
            tuple(
                _rate_bar(fold_id, fold["insufficient_final_compose_rate"])
                for fold_id, fold in folds.items()
            ),
            x_label="False-compose rate",
        ),
        "outer_fold_path.svg": _chart(
            "Stage 176 outer-fold alternate exact path",
            tuple(
                _rate_bar(fold_id, fold["alternate_only_path_success_rate"])
                for fold_id, fold in folds.items()
            ),
            x_label="Exact-path rate",
        ),
        "training_loss.svg": _chart(
            "Stage 176 mean listwise training loss",
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
            "Stage 176 phase wall times",
            tuple(
                BarDatum(name.replace("_", " "), value, f"{value:.2f} s")
                for name, value in timings.items()
            ),
            x_label="Seconds",
        ),
        "resources.svg": _chart(
            "Stage 176 process and GPU resource peaks",
            (
                _gib_bar("Process working set", resources["process_peak_working_set_bytes"]),
                _gib_bar("Process private usage", resources["process_peak_private_usage_bytes"]),
                _gib_bar("GPU allocated", resources["gpu_peak_allocated_bytes"]),
                _gib_bar("GPU reserved", resources["gpu_peak_reserved_bytes"]),
                _gib_bar(
                    "Minimum system available",
                    resources["minimum_system_available_memory_bytes"],
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
        written.append(Stage176Visualization(filename.removesuffix(".svg"), str(path)))
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


def _spec_count() -> int:
    return sum(len(policy.thresholds) for policy in _POLICIES)


def _authorize_sources(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    for name, expected in _SOURCE_HASHES.items():
        if fingerprints[name]["sha256"] != expected:
            raise ValueError(f"Stage176 source hash mismatch: {name}")


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
