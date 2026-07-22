from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import statistics
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ts_rag_agent.application import primeqa_hybrid_evidence_entailment_cv as stage172
from ts_rag_agent.application import primeqa_hybrid_iterative_router_calibration as stage169
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
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

_STAGE = "Stage 174"
_CREATED_AT = "2026-07-22"
_ANALYSIS_ID = "primeqa_hybrid_supervised_cross_encoder_grouped_nested_cv_v1"
_EXPECTED_TRAIN_ROWS = 562
_EXPECTED_CANDIDATE_ROWS = 112_400
_EXPECTED_VIEW_CASES = 1_124
_EXPECTED_NESTED_FITS = 25
_THRESHOLDS = (
    0.05,
    *(round(value / 100, 2) for value in range(10, 91, 5)),
    0.95,
    0.975,
    0.99,
)
_TRAIN_EPOCHS = 2
_TRAIN_BATCH_SIZE = 32
_INFERENCE_BATCH_SIZE = 64
_EVENT_PAIR_BATCH_SIZE = 512
_LEARNING_RATE = 2e-5
_WEIGHT_DECAY = 0.01
_GRADIENT_CLIP_NORM = 1.0
_POSITIVE_GROUP_HARD_NEGATIVES = 4
_NEGATIVE_ONLY_GROUP_HARD_NEGATIVES = 2
_MAX_LENGTH = 512
_SOURCE_HASHES = {
    "stage173": "b75c3aea469cbe22fb5581210e0d96afb9094502aaa36d9c21aa60c22db9b366",
    "stage80": "2441bb1cb1e7888299d3f57962b18cd59df84e2086ac281105abcacfc144880f",
    "train": "cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155",
    "documents": "f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b",
    **{name: stage173._SOURCE_HASHES[name] for name in stage173._MODEL_SOURCE_FILES},
}
_FORBIDDEN_PUBLIC_KEYS = stage173._FORBIDDEN_PUBLIC_KEYS | {
    "gold_pair",
    "pair_label",
    "training_pair_identity",
}

ProgressSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class ThresholdSpec:
    threshold: float

    @property
    def spec_id(self) -> str:
        material = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class PairFoldRow:
    pair: stage173.SemanticPairInput
    fold_id: str
    group_identity: str
    frozen_score: float


@dataclass(frozen=True)
class FitSummary:
    fit_id: str
    training_fold_count: int
    evaluation_fold_count: int
    training_pair_count: int
    positive_training_pair_count: int
    negative_training_pair_count: int
    evaluation_pair_count: int
    optimizer_step_count: int
    first_epoch_mean_loss: float
    final_epoch_mean_loss: float
    fit_seconds: float
    inference_seconds: float


@dataclass(frozen=True)
class Stage174Visualization:
    name: str
    path: str


class FoldPairTrainer(Protocol):
    def fit_predict(
        self,
        *,
        training_rows: Sequence[PairFoldRow],
        evaluation_rows: Sequence[PairFoldRow],
        fit_id: str,
        training_fold_count: int,
        evaluation_fold_count: int,
        progress_sink: ProgressSink | None,
    ) -> tuple[dict[str, float], FitSummary]: ...


class LocalPointwiseCrossEncoderTrainer:
    """Fine-tune one fresh local cross-encoder per grouped fold fit."""

    def __init__(
        self,
        *,
        snapshot_path: Path,
        tracker: stage169.Stage169ResourceTracker,
        torch_module: Any,
    ) -> None:
        from transformers import AutoTokenizer

        self._snapshot_path = snapshot_path
        self._tracker = tracker
        self._torch = torch_module
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot_path),
            local_files_only=True,
        )

    def fit_predict(
        self,
        *,
        training_rows: Sequence[PairFoldRow],
        evaluation_rows: Sequence[PairFoldRow],
        fit_id: str,
        training_fold_count: int,
        evaluation_fold_count: int,
        progress_sink: ProgressSink | None,
    ) -> tuple[dict[str, float], FitSummary]:
        from transformers import AutoModelForSequenceClassification

        selected = select_hard_negative_training_rows(training_rows)
        positive_count = sum(row.pair.positive_label for row in selected)
        negative_count = len(selected) - positive_count
        if positive_count <= 0 or negative_count <= 0:
            raise ValueError("Stage174 fold training requires both pair classes")
        seed = _fit_seed(fit_id)
        random.seed(seed)
        np.random.seed(seed)
        self._torch.manual_seed(seed)
        self._torch.cuda.manual_seed_all(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(self._snapshot_path),
            local_files_only=True,
        ).to("cuda")
        optimizer = self._torch.optim.AdamW(
            model.parameters(),
            lr=_LEARNING_RATE,
            weight_decay=_WEIGHT_DECAY,
        )
        pos_weight = self._torch.tensor(
            [negative_count / positive_count],
            dtype=self._torch.float32,
            device="cuda",
        )
        loss_function = self._torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        fit_started = time.perf_counter()
        epoch_losses = []
        step_count = 0
        model.train()
        for epoch in range(_TRAIN_EPOCHS):
            generator = self._torch.Generator(device="cpu")
            generator.manual_seed(seed + epoch)
            order = self._torch.randperm(len(selected), generator=generator).tolist()
            losses = []
            for start in range(0, len(order), _TRAIN_BATCH_SIZE):
                batch = [selected[index] for index in order[start : start + _TRAIN_BATCH_SIZE]]
                encoded = self._encode(batch)
                labels = self._torch.tensor(
                    [row.pair.positive_label for row in batch],
                    dtype=self._torch.float32,
                    device="cuda",
                )
                optimizer.zero_grad(set_to_none=True)
                logits = model(**encoded).logits.reshape(-1)
                loss = loss_function(logits, labels)
                if not self._torch.isfinite(loss):
                    raise RuntimeError("Stage174 training produced a non-finite loss")
                loss.backward()
                self._torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    _GRADIENT_CLIP_NORM,
                )
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                step_count += 1
            epoch_losses.append(float(statistics.fmean(losses)))
            self._tracker.capture(f"fit_{fit_id}_epoch_{epoch + 1}")
            _emit(
                progress_sink,
                phase="cross_encoder_fit_epoch",
                fit_id=fit_id,
                epoch=epoch + 1,
                total_epochs=_TRAIN_EPOCHS,
            )
        fit_seconds = time.perf_counter() - fit_started
        inference_started = time.perf_counter()
        predictions = self._predict(
            model=model,
            rows=evaluation_rows,
            fit_id=fit_id,
            progress_sink=progress_sink,
        )
        inference_seconds = time.perf_counter() - inference_started
        summary = FitSummary(
            fit_id=fit_id,
            training_fold_count=training_fold_count,
            evaluation_fold_count=evaluation_fold_count,
            training_pair_count=len(selected),
            positive_training_pair_count=positive_count,
            negative_training_pair_count=negative_count,
            evaluation_pair_count=len(evaluation_rows),
            optimizer_step_count=step_count,
            first_epoch_mean_loss=round(epoch_losses[0], 6),
            final_epoch_mean_loss=round(epoch_losses[-1], 6),
            fit_seconds=round(fit_seconds, 6),
            inference_seconds=round(inference_seconds, 6),
        )
        del optimizer, loss_function, pos_weight, model
        gc.collect()
        self._torch.cuda.empty_cache()
        self._tracker.capture(f"fit_{fit_id}_released")
        return predictions, summary

    def _encode(self, rows: Sequence[PairFoldRow]) -> dict[str, Any]:
        encoded = self._tokenizer(
            [row.pair.question_text for row in rows],
            [row.pair.passage_text for row in rows],
            padding=True,
            truncation=True,
            max_length=_MAX_LENGTH,
            return_tensors="pt",
        )
        return {name: value.to("cuda") for name, value in encoded.items()}

    def _predict(
        self,
        *,
        model: Any,
        rows: Sequence[PairFoldRow],
        fit_id: str,
        progress_sink: ProgressSink | None,
    ) -> dict[str, float]:
        model.eval()
        predictions: dict[str, float] = {}
        event_count = (len(rows) + _EVENT_PAIR_BATCH_SIZE - 1) // _EVENT_PAIR_BATCH_SIZE
        with self._torch.inference_mode():
            for event_index, event_start in enumerate(
                range(0, len(rows), _EVENT_PAIR_BATCH_SIZE),
                start=1,
            ):
                event_rows = rows[event_start : event_start + _EVENT_PAIR_BATCH_SIZE]
                for start in range(0, len(event_rows), _INFERENCE_BATCH_SIZE):
                    batch = event_rows[start : start + _INFERENCE_BATCH_SIZE]
                    logits = model(**self._encode(batch)).logits.reshape(-1)
                    probabilities = self._torch.sigmoid(logits).detach().cpu().tolist()
                    predictions.update(
                        {
                            row.pair.private_identity: float(probability)
                            for row, probability in zip(batch, probabilities, strict=True)
                        }
                    )
                self._tracker.capture(f"fit_{fit_id}_inference_{event_index}")
                _emit(
                    progress_sink,
                    phase="cross_encoder_fit_inference",
                    fit_id=fit_id,
                    completed=event_index,
                    total=event_count,
                )
        if len(predictions) != len(rows):
            raise RuntimeError("Stage174 fold inference coverage is incomplete")
        return predictions


def select_hard_negative_training_rows(
    rows: Sequence[PairFoldRow],
) -> tuple[PairFoldRow, ...]:
    grouped: dict[str, list[PairFoldRow]] = {}
    for row in rows:
        grouped.setdefault(row.group_identity, []).append(row)
    selected = []
    for group_identity in sorted(grouped):
        group = grouped[group_identity]
        positives = [row for row in group if row.pair.positive_label]
        if len(positives) > 1:
            raise ValueError("Stage174 allows at most one positive pair per question")
        negatives = sorted(
            (row for row in group if not row.pair.positive_label),
            key=lambda row: (-row.frozen_score, row.pair.private_identity),
        )
        negative_limit = (
            _POSITIVE_GROUP_HARD_NEGATIVES if positives else _NEGATIVE_ONLY_GROUP_HARD_NEGATIVES
        )
        selected.extend(positives)
        selected.extend(negatives[:negative_limit])
    return tuple(selected)


def run_stage174_supervised_cross_encoder_cv(
    *,
    stage173_report_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    documents_path: Path,
    model_snapshot_path: Path,
    encoder_batch_size: int = 64,
    progress_sink: ProgressSink | None = None,
    trainer_factory: Callable[[Path, stage169.Stage169ResourceTracker, Any], FoldPairTrainer]
    | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    started_cpu = time.process_time()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    source_paths = {
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
    stage173_report = _load_json_object(stage173_report_path)
    if stage173_report.get("decision", {}).get("status") != (
        "stage173_frozen_cross_encoder_semantics_insufficient"
    ):
        raise ValueError("Stage173 did not authorize supervised cross-encoder adaptation")
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="sources_authorized")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage174 formal fine-tuning requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    tracker = stage169.Stage169ResourceTracker(torch_module=torch)
    tracker.capture("analysis_start")

    samples = load_primeqa_hybrid_split_samples(train_split_path)
    if len(samples) != _EXPECTED_TRAIN_ROWS or any(
        sample.assigned_split != "train" for sample in samples
    ):
        raise ValueError("Stage174 accepts only the exact 562-row train split")
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
        raise RuntimeError("Stage174 requires both authorized local dense channels")
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
        raise RuntimeError("Stage174 candidate replay row count drifted")
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
        raise RuntimeError("Stage174 base view case count drifted")
    del frozen_scorer
    gc.collect()
    torch.cuda.empty_cache()
    tracker.capture("frozen_cross_encoder_released")
    pair_rows = build_pair_fold_rows(
        pairs=pairs,
        base_cases=base_cases,
        frozen_scores=frozen_scores,
    )
    sampled_rows = select_hard_negative_training_rows(pair_rows)
    pair_data_ready_at = time.perf_counter()
    _emit(
        progress_sink,
        phase="supervised_pair_data_ready",
        complete_pair_count=len(pair_rows),
        sampled_pair_count=len(sampled_rows),
    )

    factory = trainer_factory or (
        lambda snapshot, resource_tracker, torch_module: LocalPointwiseCrossEncoderTrainer(
            snapshot_path=snapshot,
            tracker=resource_tracker,
            torch_module=torch_module,
        )
    )
    trainer = factory(model_snapshot_path, tracker, torch)
    nested = run_grouped_nested_fine_tuning(
        samples=samples,
        grouped_records=grouped_records,
        pair_rows=pair_rows,
        trainer=trainer,
        progress_sink=progress_sink,
    )
    nested_finished_at = time.perf_counter()
    tracker.capture("nested_fine_tuning_finished")

    outer_folds = nested["outer_folds"]
    oof_cases = nested["oof_cases"]
    oof_scores = nested["oof_view_scores"]
    selected_specs = nested["selected_specs"]
    fit_summaries: Sequence[FitSummary] = nested["fit_summaries"]
    final_threshold_row = _select_threshold(
        cases=oof_cases,
        scores=oof_scores,
        thresholds=_THRESHOLDS,
    )
    final_spec: ThresholdSpec = final_threshold_row["spec"]
    oof_metrics = stage172.evaluate_predictions(oof_cases, oof_scores, selected_specs)
    oof_gates = stage172._quality_gates(oof_metrics)
    fold_metrics = _outer_fold_metrics(oof_cases, oof_scores, selected_specs)
    all_outer_safety_passed = all(
        metrics["insufficient_final_compose_rate"]
        <= stage172._GATE_THRESHOLDS["insufficient_final_compose_rate_max"]
        for metrics in fold_metrics.values()
    )
    all_inner_selected_eligible = all(
        row["selected_threshold_inner_eligible"] for row in outer_folds
    )
    candidate_selected = (
        all_inner_selected_eligible
        and bool(final_threshold_row["eligible"])
        and all(gate["passed"] for gate in oof_gates)
        and all_outer_safety_passed
    )
    tracker.capture("report_assembly")
    finished_at = time.perf_counter()
    snapshots = tracker.snapshots

    process_guards = [
        _check("stage173_authorized_supervised_adaptation", True),
        _check("exact_train_row_count", len(samples) == _EXPECTED_TRAIN_ROWS),
        _check("exact_candidate_row_count", len(records) == _EXPECTED_CANDIDATE_ROWS),
        _check("exact_view_case_count", len(oof_cases) == _EXPECTED_VIEW_CASES),
        _check("complete_pair_score_coverage", len(frozen_scores) == len(pair_rows)),
        _check("exact_nested_fit_count", len(fit_summaries) == _EXPECTED_NESTED_FITS),
        _check("five_grouped_outer_folds", len(outer_folds) == 5),
        _check("complete_oof_view_coverage", len(oof_scores) == len(oof_cases)),
        _check("twenty_one_frozen_thresholds", len(_THRESHOLDS) == 21),
        _check("two_frozen_training_epochs", _TRAIN_EPOCHS == 2),
        _check("development_not_loaded", True),
        _check("test_not_loaded", True),
        _check("answer_generation_not_run", True),
        _check("agent_turns_not_run", True),
        _check("model_checkpoint_not_written", True),
        _check("retry_count_zero", True),
        _check("fallback_count_zero", True),
        _check("default_runtime_unchanged", True),
    ]
    stage173_metrics = stage173_report["nested_cv"]["oof_metrics"]
    resource_consumption = {
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
            "Train-only supervised cross-encoder adaptation with 25 grouped nested-CV "
            "fits and direct evidence-view maximum-probability thresholding."
        ),
        "source_authorization": fingerprints,
        "frozen_protocol": {
            "model_snapshot_revision": model_snapshot_path.name,
            "training_objective": "pointwise_binary_cross_entropy_with_dynamic_positive_weight",
            "train_epochs": _TRAIN_EPOCHS,
            "train_batch_size": _TRAIN_BATCH_SIZE,
            "inference_batch_size": _INFERENCE_BATCH_SIZE,
            "max_length": _MAX_LENGTH,
            "learning_rate": _LEARNING_RATE,
            "weight_decay": _WEIGHT_DECAY,
            "gradient_clip_norm": _GRADIENT_CLIP_NORM,
            "positive_group_hard_negative_count": _POSITIVE_GROUP_HARD_NEGATIVES,
            "negative_only_group_hard_negative_count": (_NEGATIVE_ONLY_GROUP_HARD_NEGATIVES),
            "thresholds": list(_THRESHOLDS),
            "nested_fit_count": _EXPECTED_NESTED_FITS,
            "outer_fold_count": 5,
            "inner_model_count_per_outer_fold": 4,
            "outer_model_count_per_outer_fold": 1,
            "gold_labels_used_only_for_training_and_evaluation": True,
            "development_and_test_closed": True,
        },
        "split_contract": {
            "loaded_split": "train",
            "pair_fit_split": "inner_or_outer_training_question_folds_only",
            "threshold_fit_split": "inner_oof_question_folds_only",
            "outer_evaluation": "one_shot_heldout_question_fold",
            "development_loaded": False,
            "test_loaded": False,
        },
        "pair_data_summary": {
            "complete_pair_count": len(pair_rows),
            "complete_positive_pair_count": sum(row.pair.positive_label for row in pair_rows),
            "complete_negative_pair_count": sum(not row.pair.positive_label for row in pair_rows),
            "full_train_sampled_pair_count": len(sampled_rows),
            "full_train_sampled_positive_count": sum(
                row.pair.positive_label for row in sampled_rows
            ),
            "full_train_sampled_negative_count": sum(
                not row.pair.positive_label for row in sampled_rows
            ),
            "private_pair_rows_written": False,
        },
        "nested_cv": {
            "outer_folds": outer_folds,
            "fit_count": len(fit_summaries),
            "fit_summaries": [asdict(summary) for summary in fit_summaries],
            "selected_threshold_ids_by_fold": {
                fold_id: spec.spec_id for fold_id, spec in selected_specs.items()
            },
            "final_full_train_oof_selected_threshold": final_spec.threshold,
            "final_full_train_oof_selected_threshold_id": final_spec.spec_id,
            "final_full_train_oof_selected_threshold_eligible": bool(
                final_threshold_row["eligible"]
            ),
            "final_full_train_oof_metrics": final_threshold_row["metrics"].public_dict(),
            "final_full_train_oof_safe_fold_count": final_threshold_row["safe_fold_count"],
            "all_inner_selected_thresholds_eligible": all_inner_selected_eligible,
            "oof_metrics": oof_metrics.public_dict(),
            "oof_quality_gates": oof_gates,
            "outer_fold_metrics": fold_metrics,
            "all_outer_folds_safety_passed": all_outer_safety_passed,
        },
        "stage173_comparison": {
            "stage173_oof_metrics": stage173_metrics,
            "stage174_oof_metrics": oof_metrics.public_dict(),
            "metric_delta": {
                name: round(getattr(oof_metrics, name) - float(stage173_metrics[name]), 6)
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
        "resource_consumption": resource_consumption,
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "candidate_replay": round(replay_ready_at - authorized_at, 6),
            "frozen_cross_encoder_load": round(frozen_model_ready_at - replay_ready_at, 6),
            "frozen_pair_build_and_score": round(pair_data_ready_at - frozen_model_ready_at, 6),
            "nested_fine_tuning": round(nested_finished_at - pair_data_ready_at, 6),
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
                "advance_to_stage175_train_only_finetuned_semantic_runtime_e2e"
                if candidate_selected
                else "stage174_supervised_cross_encoder_insufficient"
            ),
            "development_opened": False,
            "test_opened": False,
            "default_runtime_activation": False,
        },
    }
    forbidden = sorted(_forbidden_keys_found(report))
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
        report["decision"]["status"] = "stage174_process_invalid"
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report


def build_pair_fold_rows(
    *,
    pairs: Sequence[stage173.SemanticPairInput],
    base_cases: Sequence[stage172.EvidenceViewCase],
    frozen_scores: Mapping[str, float],
) -> tuple[PairFoldRow, ...]:
    fold_by_group = {
        case.group_identity: case.fold_id for case in base_cases if case.phase == "initial"
    }
    rows = []
    for pair in pairs:
        group_identity = pair.private_identity.split(":", 1)[0]
        rows.append(
            PairFoldRow(
                pair=pair,
                fold_id=fold_by_group[group_identity],
                group_identity=group_identity,
                frozen_score=frozen_scores[pair.private_identity],
            )
        )
    if len(rows) != len(pairs) or len({row.pair.private_identity for row in rows}) != len(rows):
        raise RuntimeError("Stage174 pair-fold mapping is incomplete")
    return tuple(rows)


def run_grouped_nested_fine_tuning(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
    pair_rows: Sequence[PairFoldRow],
    trainer: FoldPairTrainer,
    progress_sink: ProgressSink | None,
) -> dict[str, Any]:
    folds = sorted({row.fold_id for row in pair_rows})
    if len(folds) != 5:
        raise ValueError("Stage174 requires exactly five pair folds")
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
    oof_view_scores: dict[str, float] = {}
    selected_specs: dict[str, ThresholdSpec] = {}
    for outer_index, outer_fold in enumerate(folds, start=1):
        outer_train_folds = tuple(fold for fold in folds if fold != outer_fold)
        inner_pair_predictions: dict[str, float] = {}
        for inner_index, inner_fold in enumerate(outer_train_folds, start=1):
            training_folds = frozenset(fold for fold in outer_train_folds if fold != inner_fold)
            fit_id = f"outer_{outer_index}_inner_{inner_index}"
            predictions, summary = trainer.fit_predict(
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
        inner_cases, inner_scores = build_probability_view_cases(
            samples=inner_samples,
            grouped_records=grouped_records,
            pair_probabilities=inner_pair_predictions,
        )
        selected_row = _select_threshold(
            cases=inner_cases,
            scores=inner_scores,
            thresholds=_THRESHOLDS,
        )
        selected: ThresholdSpec = selected_row["spec"]

        fit_id = f"outer_{outer_index}_final"
        outer_predictions, summary = trainer.fit_predict(
            training_rows=tuple(row for row in pair_rows if row.fold_id in outer_train_folds),
            evaluation_rows=tuple(row for row in pair_rows if row.fold_id == outer_fold),
            fit_id=fit_id,
            training_fold_count=len(outer_train_folds),
            evaluation_fold_count=1,
            progress_sink=progress_sink,
        )
        fit_summaries.append(summary)
        heldout_cases, heldout_scores = build_probability_view_cases(
            samples=samples_by_fold[outer_fold],
            grouped_records=grouped_records,
            pair_probabilities=outer_predictions,
        )
        heldout_metrics = stage172.evaluate_predictions(
            heldout_cases,
            heldout_scores,
            {outer_fold: selected},
        )
        oof_cases.extend(heldout_cases)
        oof_view_scores.update(heldout_scores)
        selected_specs[outer_fold] = selected
        outer_rows.append(
            {
                "heldout_fold": outer_fold,
                "inner_fit_count": len(outer_train_folds),
                "outer_fit_count": 1,
                "inner_case_count": len(inner_cases),
                "heldout_case_count": len(heldout_cases),
                "inner_eligible_threshold_count": selected_row["eligible_threshold_count"],
                "selected_threshold": selected.threshold,
                "selected_threshold_id": selected.spec_id,
                "selected_threshold_inner_eligible": bool(selected_row["eligible"]),
                "selected_inner_metrics": selected_row["metrics"].public_dict(),
                "selected_inner_safe_fold_count": selected_row["safe_fold_count"],
                "heldout_metrics": heldout_metrics.public_dict(),
            }
        )
        _emit(
            progress_sink,
            phase="outer_fold_complete",
            completed=outer_index,
            total=5,
        )
    if len(fit_summaries) != _EXPECTED_NESTED_FITS:
        raise RuntimeError("Stage174 nested fit count drifted")
    return {
        "outer_folds": outer_rows,
        "oof_cases": tuple(oof_cases),
        "oof_view_scores": oof_view_scores,
        "selected_specs": selected_specs,
        "fit_summaries": tuple(fit_summaries),
    }


def build_probability_view_cases(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
    pair_probabilities: Mapping[str, float],
) -> tuple[tuple[stage172.EvidenceViewCase, ...], dict[str, float]]:
    base_cases = stage172.build_evidence_view_cases(
        samples=samples,
        grouped_records=grouped_records,
    )
    base_by_identity = {case.private_identity: case for case in base_cases}
    cases = []
    scores = {}
    for sample in samples:
        records = tuple(grouped_records[sample.sample_id])
        initial = select_current_query_overlap_top10(records).selected
        alternate = select_original_rrf_top10(records).selected
        final = stage172._deduplicate_records((*initial, *alternate))
        for phase, visible in (("initial", initial), ("final", final)):
            identity = stage172._sha256_text(f"{sample.sample_id}:{phase}")
            probability = max(
                pair_probabilities[stage173._pair_identity(sample.sample_id, record.document_id)]
                for record in visible
            )
            base = base_by_identity[identity]
            cases.append(
                stage172.EvidenceViewCase(
                    private_identity=base.private_identity,
                    group_identity=base.group_identity,
                    fold_id=base.fold_id,
                    phase=base.phase,
                    stratum=base.stratum,
                    features={"fine_tuned_max_probability": probability},
                    sufficient_label=base.sufficient_label,
                )
            )
            scores[identity] = probability
    if len(scores) != len(cases):
        raise RuntimeError("Stage174 view probability coverage is incomplete")
    return tuple(cases), scores


def _select_threshold(
    *,
    cases: Sequence[stage172.EvidenceViewCase],
    scores: Mapping[str, float],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    rows = [
        _threshold_evaluation(cases, scores, ThresholdSpec(threshold)) for threshold in thresholds
    ]
    selected = max(rows, key=_threshold_selection_key)
    return {
        **selected,
        "eligible_threshold_count": sum(row["eligible"] for row in rows),
    }


def _threshold_evaluation(
    cases: Sequence[stage172.EvidenceViewCase],
    scores: Mapping[str, float],
    spec: ThresholdSpec,
) -> dict[str, Any]:
    specs_by_fold = {fold_id: spec for fold_id in {case.fold_id for case in cases}}
    metrics = stage172.evaluate_predictions(cases, scores, specs_by_fold)
    gates = stage172._quality_gates(metrics)
    fold_metrics = _fold_metrics_for_threshold(cases, scores, spec)
    safe_fold_count = sum(
        row.insufficient_final_compose_rate
        <= stage172._GATE_THRESHOLDS["insufficient_final_compose_rate_max"]
        for row in fold_metrics.values()
    )
    return {
        "spec": spec,
        "metrics": metrics,
        "gates": gates,
        "safe_fold_count": safe_fold_count,
        "fold_count": len(fold_metrics),
        "eligible": all(gate["passed"] for gate in gates) and safe_fold_count == len(fold_metrics),
    }


def _threshold_selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics: stage172.EvidenceProxyMetrics = row["metrics"]
    spec: ThresholdSpec = row["spec"]
    return (
        int(row["safe_fold_count"] == row["fold_count"]),
        row["safe_fold_count"],
        int(
            metrics.insufficient_final_compose_rate
            <= stage172._GATE_THRESHOLDS["insufficient_final_compose_rate_max"]
        ),
        sum(gate["passed"] for gate in row["gates"]),
        metrics.alternate_only_path_success_rate,
        metrics.alternate_only_final_compose_rate,
        metrics.initial_visible_compose_rate,
        -metrics.insufficient_final_compose_rate,
        metrics.balanced_accuracy,
        -abs(spec.threshold - 0.5),
        spec.spec_id,
    )


def _outer_fold_metrics(
    cases: Sequence[stage172.EvidenceViewCase],
    scores: Mapping[str, float],
    specs_by_fold: Mapping[str, ThresholdSpec],
) -> dict[str, dict[str, int | float]]:
    result = {}
    for fold_id in sorted(specs_by_fold):
        fold_cases = tuple(case for case in cases if case.fold_id == fold_id)
        result[fold_id] = stage172.evaluate_predictions(
            fold_cases,
            {case.private_identity: scores[case.private_identity] for case in fold_cases},
            {fold_id: specs_by_fold[fold_id]},
        ).public_dict()
    return result


def _fold_metrics_for_threshold(
    cases: Sequence[stage172.EvidenceViewCase],
    scores: Mapping[str, float],
    spec: ThresholdSpec,
) -> dict[str, stage172.EvidenceProxyMetrics]:
    result = {}
    for fold_id in sorted({case.fold_id for case in cases}):
        fold_cases = tuple(case for case in cases if case.fold_id == fold_id)
        result[fold_id] = stage172.evaluate_predictions(
            fold_cases,
            {case.private_identity: scores[case.private_identity] for case in fold_cases},
            {fold_id: spec},
        )
    return result


def write_stage174_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage174Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    nested = report["nested_cv"]
    metrics = nested["oof_metrics"]
    stage173_metrics = report["stage173_comparison"]["stage173_oof_metrics"]
    folds = nested["outer_fold_metrics"]
    resources = report["resource_consumption"]
    diagnostics = report["training_diagnostics"]
    timings = report["timing_seconds"]
    charts = {
        "oof_quality_gates.svg": _chart(
            "Stage 174 grouped OOF quality gates",
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
        "stage173_stage174_rates.svg": _chart(
            "Stage 173 versus Stage 174 grouped OOF rates",
            tuple(
                bar
                for label, name in (
                    ("initial compose", "initial_visible_compose_rate"),
                    ("final compose", "alternate_only_final_compose_rate"),
                    ("exact path", "alternate_only_path_success_rate"),
                    ("false compose", "insufficient_final_compose_rate"),
                )
                for bar in (
                    _rate_bar(f"173 {label}", stage173_metrics[name]),
                    _rate_bar(f"174 {label}", metrics[name]),
                )
            ),
            x_label="Rate",
        ),
        "outer_fold_safety.svg": _chart(
            "Stage 174 outer-fold insufficient final compose",
            tuple(
                _rate_bar(fold_id, fold["insufficient_final_compose_rate"])
                for fold_id, fold in folds.items()
            ),
            x_label="False-compose rate",
        ),
        "outer_fold_path.svg": _chart(
            "Stage 174 outer-fold alternate exact path",
            tuple(
                _rate_bar(fold_id, fold["alternate_only_path_success_rate"])
                for fold_id, fold in folds.items()
            ),
            x_label="Exact-path rate",
        ),
        "selected_thresholds.svg": _chart(
            "Stage 174 inner-selected thresholds",
            tuple(
                BarDatum(
                    row["heldout_fold"],
                    row["selected_threshold"],
                    f"{row['selected_threshold']:.2f}",
                )
                for row in nested["outer_folds"]
            ),
            x_label="Probability threshold",
        ),
        "training_loss.svg": _chart(
            "Stage 174 mean fold-training loss",
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
            x_label="Weighted BCE loss",
        ),
        "timing.svg": _chart(
            "Stage 174 phase wall times",
            tuple(
                BarDatum(name.replace("_", " "), value, f"{value:.2f} s")
                for name, value in timings.items()
            ),
            x_label="Seconds",
        ),
        "resources.svg": _chart(
            "Stage 174 process and GPU resource peaks",
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
        written.append(Stage174Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _training_diagnostics(summaries: Sequence[FitSummary]) -> dict[str, Any]:
    return {
        "fit_count": len(summaries),
        "training_pair_count_distribution": _distribution(
            [summary.training_pair_count for summary in summaries]
        ),
        "positive_pair_count_distribution": _distribution(
            [summary.positive_training_pair_count for summary in summaries]
        ),
        "negative_pair_count_distribution": _distribution(
            [summary.negative_training_pair_count for summary in summaries]
        ),
        "optimizer_step_count_distribution": _distribution(
            [summary.optimizer_step_count for summary in summaries]
        ),
        "first_epoch_loss_mean": round(
            statistics.fmean(summary.first_epoch_mean_loss for summary in summaries),
            6,
        ),
        "final_epoch_loss_mean": round(
            statistics.fmean(summary.final_epoch_mean_loss for summary in summaries),
            6,
        ),
        "fit_seconds_distribution": _distribution([summary.fit_seconds for summary in summaries]),
        "inference_seconds_distribution": _distribution(
            [summary.inference_seconds for summary in summaries]
        ),
    }


def _distribution(values: Sequence[int | float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Stage174 distribution cannot be empty")
    ordered = np.asarray(sorted(values), dtype=float)
    return {
        "count": len(ordered),
        "minimum": round(float(ordered[0]), 6),
        "median": round(float(np.median(ordered)), 6),
        "maximum": round(float(ordered[-1]), 6),
        "mean": round(float(np.mean(ordered)), 6),
    }


def _fit_seed(fit_id: str) -> int:
    digest = hashlib.sha256(fit_id.encode("utf-8")).hexdigest()
    return 174_000 + (int(digest[:8], 16) % 10_000)


def _authorize_sources(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    for name, expected in _SOURCE_HASHES.items():
        if fingerprints[name]["sha256"] != expected:
            raise ValueError(f"Stage174 source hash mismatch: {name}")


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


def _chart(
    title: str,
    bars: Sequence[BarDatum],
    *,
    x_label: str,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        width=1200,
        margin_left=440,
        margin_right=200,
    )
