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

_STAGE = "Stage 175"
_CREATED_AT = "2026-07-22"
_ANALYSIS_ID = "primeqa_hybrid_grouped_pairwise_listwise_nested_cv_v1"
_EXPECTED_TRAIN_ROWS = 562
_EXPECTED_CANDIDATE_ROWS = 112_400
_EXPECTED_VIEW_CASES = 1_124
_FAMILIES = ("pairwise_anchor", "listwise_none")
_EXPECTED_NESTED_FITS = 50
_MARGIN_THRESHOLDS = (
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
)
_TRAIN_EPOCHS = 2
_TRAIN_PAIR_BUDGET = 32
_INFERENCE_BATCH_SIZE = 64
_EVENT_PAIR_BATCH_SIZE = 512
_LEARNING_RATE = 2e-5
_WEIGHT_DECAY = 0.01
_GRADIENT_CLIP_NORM = 1.0
_PAIRWISE_ANCHOR_WEIGHT = 0.5
_MAX_LENGTH = 512
_SOURCE_HASHES = {
    "stage174": "7d949a300f58f3205397c76e3accf4c9a71932d466fa4811144f3cbc04b86019",
    **stage174._SOURCE_HASHES,
}
_FORBIDDEN_PUBLIC_KEYS = stage174._FORBIDDEN_PUBLIC_KEYS | {
    "group_label",
    "positive_index",
}

ProgressSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class RankingSpec:
    family: str
    threshold: float

    @property
    def spec_id(self) -> str:
        material = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class RankingFitSummary:
    family: str
    fit_id: str
    training_fold_count: int
    evaluation_fold_count: int
    training_group_count: int
    positive_training_group_count: int
    negative_training_group_count: int
    training_pair_count: int
    evaluation_pair_count: int
    optimizer_step_count: int
    first_epoch_mean_loss: float
    final_epoch_mean_loss: float
    fit_seconds: float
    inference_seconds: float


@dataclass(frozen=True)
class Stage175Visualization:
    name: str
    path: str


class GroupedRankingObjective(Protocol):
    family: str

    def loss(self, *, logits: Any, positive_index: int | None, torch_module: Any) -> Any: ...


class PairwiseAnchorObjective:
    family = "pairwise_anchor"

    def loss(self, *, logits: Any, positive_index: int | None, torch_module: Any) -> Any:
        functional = torch_module.nn.functional
        if positive_index is None:
            return functional.softplus(logits).mean()
        positive = logits[positive_index]
        negative_mask = torch_module.ones(
            len(logits),
            dtype=torch_module.bool,
            device=logits.device,
        )
        negative_mask[positive_index] = False
        negatives = logits[negative_mask]
        rank_loss = functional.softplus(-(positive - negatives)).mean()
        anchor_loss = (functional.softplus(-positive) + functional.softplus(negatives).mean()) / 2.0
        return rank_loss + (_PAIRWISE_ANCHOR_WEIGHT * anchor_loss)


class ListwiseNoneObjective:
    family = "listwise_none"

    def loss(self, *, logits: Any, positive_index: int | None, torch_module: Any) -> Any:
        none_logit = torch_module.zeros(1, dtype=logits.dtype, device=logits.device)
        choices = torch_module.cat((logits, none_logit)).reshape(1, -1)
        target_index = len(logits) if positive_index is None else positive_index
        target = torch_module.tensor([target_index], dtype=torch_module.long, device=logits.device)
        return torch_module.nn.functional.cross_entropy(choices, target)


class RankingFoldTrainer(Protocol):
    def fit_predict(
        self,
        *,
        family: str,
        training_rows: Sequence[stage174.PairFoldRow],
        evaluation_rows: Sequence[stage174.PairFoldRow],
        fit_id: str,
        training_fold_count: int,
        evaluation_fold_count: int,
        progress_sink: ProgressSink | None,
    ) -> tuple[dict[str, float], RankingFitSummary]: ...


class LocalGroupedRankingTrainer:
    """Train one fresh cross-encoder with a selected grouped ranking objective."""

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
        self._objectives: dict[str, GroupedRankingObjective] = {
            objective.family: objective
            for objective in (PairwiseAnchorObjective(), ListwiseNoneObjective())
        }

    def fit_predict(
        self,
        *,
        family: str,
        training_rows: Sequence[stage174.PairFoldRow],
        evaluation_rows: Sequence[stage174.PairFoldRow],
        fit_id: str,
        training_fold_count: int,
        evaluation_fold_count: int,
        progress_sink: ProgressSink | None,
    ) -> tuple[dict[str, float], RankingFitSummary]:
        from transformers import AutoModelForSequenceClassification

        objective = self._objectives.get(family)
        if objective is None:
            raise ValueError(f"Unknown Stage175 ranking family: {family}")
        groups = build_sampled_training_groups(training_rows)
        positive_group_count = sum(_positive_index(group) is not None for group in groups)
        negative_group_count = len(groups) - positive_group_count
        if positive_group_count <= 0 or negative_group_count <= 0:
            raise ValueError("Stage175 fold training requires both group classes")
        seed = _fit_seed(family, fit_id)
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
        fit_started = time.perf_counter()
        epoch_losses = []
        step_count = 0
        model.train()
        for epoch in range(_TRAIN_EPOCHS):
            generator = self._torch.Generator(device="cpu")
            generator.manual_seed(seed + epoch)
            order = self._torch.randperm(len(groups), generator=generator).tolist()
            ordered_groups = tuple(groups[index] for index in order)
            losses = []
            for group_batch in _pack_group_batches(ordered_groups):
                flat_rows = tuple(row for group in group_batch for row in group)
                encoded = self._encode(flat_rows)
                optimizer.zero_grad(set_to_none=True)
                flat_logits = model(**encoded).logits.reshape(-1)
                group_losses = []
                offset = 0
                for group in group_batch:
                    group_logits = flat_logits[offset : offset + len(group)]
                    group_losses.append(
                        objective.loss(
                            logits=group_logits,
                            positive_index=_positive_index(group),
                            torch_module=self._torch,
                        )
                    )
                    offset += len(group)
                loss = self._torch.stack(group_losses).mean()
                if not self._torch.isfinite(loss):
                    raise RuntimeError("Stage175 training produced a non-finite loss")
                loss.backward()
                self._torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    _GRADIENT_CLIP_NORM,
                )
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                step_count += 1
            epoch_losses.append(float(statistics.fmean(losses)))
            self._tracker.capture(f"fit_{family}_{fit_id}_epoch_{epoch + 1}")
            _emit(
                progress_sink,
                phase="ranking_fit_epoch",
                family=family,
                fit_id=fit_id,
                epoch=epoch + 1,
                total_epochs=_TRAIN_EPOCHS,
            )
        fit_seconds = time.perf_counter() - fit_started
        inference_started = time.perf_counter()
        predictions = self._predict(
            model=model,
            rows=evaluation_rows,
            family=family,
            fit_id=fit_id,
            progress_sink=progress_sink,
        )
        inference_seconds = time.perf_counter() - inference_started
        summary = RankingFitSummary(
            family=family,
            fit_id=fit_id,
            training_fold_count=training_fold_count,
            evaluation_fold_count=evaluation_fold_count,
            training_group_count=len(groups),
            positive_training_group_count=positive_group_count,
            negative_training_group_count=negative_group_count,
            training_pair_count=sum(len(group) for group in groups),
            evaluation_pair_count=len(evaluation_rows),
            optimizer_step_count=step_count,
            first_epoch_mean_loss=round(epoch_losses[0], 6),
            final_epoch_mean_loss=round(epoch_losses[-1], 6),
            fit_seconds=round(fit_seconds, 6),
            inference_seconds=round(inference_seconds, 6),
        )
        del optimizer, model
        gc.collect()
        self._torch.cuda.empty_cache()
        self._tracker.capture(f"fit_{family}_{fit_id}_released")
        return predictions, summary

    def _encode(self, rows: Sequence[stage174.PairFoldRow]) -> dict[str, Any]:
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
        rows: Sequence[stage174.PairFoldRow],
        family: str,
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
                    logits = model(**self._encode(batch)).logits.reshape(-1).detach().cpu().tolist()
                    predictions.update(
                        {
                            row.pair.private_identity: float(logit)
                            for row, logit in zip(batch, logits, strict=True)
                        }
                    )
                self._tracker.capture(f"fit_{family}_{fit_id}_inference_{event_index}")
                _emit(
                    progress_sink,
                    phase="ranking_fit_inference",
                    family=family,
                    fit_id=fit_id,
                    completed=event_index,
                    total=event_count,
                )
        if len(predictions) != len(rows):
            raise RuntimeError("Stage175 fold inference coverage is incomplete")
        return predictions


def build_sampled_training_groups(
    rows: Sequence[stage174.PairFoldRow],
) -> tuple[tuple[stage174.PairFoldRow, ...], ...]:
    selected = stage174.select_hard_negative_training_rows(rows)
    grouped: dict[str, list[stage174.PairFoldRow]] = {}
    for row in selected:
        grouped.setdefault(row.group_identity, []).append(row)
    groups = tuple(tuple(grouped[group_identity]) for group_identity in sorted(grouped))
    if not groups or sum(len(group) for group in groups) != len(selected):
        raise RuntimeError("Stage175 sampled group construction is incomplete")
    if any(len(group) > 5 for group in groups):
        raise RuntimeError("Stage175 sampled group exceeds the frozen pair limit")
    return groups


def _positive_index(group: Sequence[stage174.PairFoldRow]) -> int | None:
    indices = [index for index, row in enumerate(group) if row.pair.positive_label]
    if len(indices) > 1:
        raise ValueError("Stage175 allows at most one positive pair per question")
    return indices[0] if indices else None


def _pack_group_batches(
    groups: Sequence[tuple[stage174.PairFoldRow, ...]],
) -> tuple[tuple[tuple[stage174.PairFoldRow, ...], ...], ...]:
    batches = []
    current: list[tuple[stage174.PairFoldRow, ...]] = []
    current_pairs = 0
    for group in groups:
        if len(group) > _TRAIN_PAIR_BUDGET:
            raise ValueError("Stage175 group exceeds the training pair budget")
        if current and current_pairs + len(group) > _TRAIN_PAIR_BUDGET:
            batches.append(tuple(current))
            current = []
            current_pairs = 0
        current.append(group)
        current_pairs += len(group)
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def build_margin_view_cases(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
    pair_logits: Mapping[str, float],
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
            logits = sorted(
                (
                    pair_logits[stage173._pair_identity(sample.sample_id, record.document_id)]
                    for record in visible
                ),
                reverse=True,
            )
            if len(logits) < 2:
                raise RuntimeError("Stage175 evidence view requires at least two candidates")
            margin = logits[0] - max(0.0, logits[1])
            base = base_by_identity[identity]
            cases.append(
                stage172.EvidenceViewCase(
                    private_identity=base.private_identity,
                    group_identity=base.group_identity,
                    fold_id=base.fold_id,
                    phase=base.phase,
                    stratum=base.stratum,
                    features={"ranking_none_margin": margin},
                    sufficient_label=base.sufficient_label,
                )
            )
            scores[identity] = margin
    if len(scores) != len(cases):
        raise RuntimeError("Stage175 view margin coverage is incomplete")
    return tuple(cases), scores


def run_grouped_nested_ranking(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
    pair_rows: Sequence[stage174.PairFoldRow],
    trainer: RankingFoldTrainer,
    progress_sink: ProgressSink | None,
) -> dict[str, Any]:
    folds = sorted({row.fold_id for row in pair_rows})
    if len(folds) != 5:
        raise ValueError("Stage175 requires exactly five pair folds")
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
    selected_oof_cases: list[stage172.EvidenceViewCase] = []
    selected_oof_scores: dict[str, float] = {}
    selected_specs: dict[str, RankingSpec] = {}
    family_oof_scores: dict[str, dict[str, float]] = {family: {} for family in _FAMILIES}
    family_oof_cases: dict[str, list[stage172.EvidenceViewCase]] = {
        family: [] for family in _FAMILIES
    }
    for outer_index, outer_fold in enumerate(folds, start=1):
        outer_train_folds = tuple(fold for fold in folds if fold != outer_fold)
        inner_samples = tuple(
            sample for fold in outer_train_folds for sample in samples_by_fold[fold]
        )
        family_inner_rows = []
        for family in _FAMILIES:
            inner_pair_predictions: dict[str, float] = {}
            for inner_index, inner_fold in enumerate(outer_train_folds, start=1):
                training_folds = frozenset(fold for fold in outer_train_folds if fold != inner_fold)
                fit_id = f"outer_{outer_index}_{family}_inner_{inner_index}"
                predictions, summary = trainer.fit_predict(
                    family=family,
                    training_rows=tuple(row for row in pair_rows if row.fold_id in training_folds),
                    evaluation_rows=tuple(row for row in pair_rows if row.fold_id == inner_fold),
                    fit_id=fit_id,
                    training_fold_count=len(training_folds),
                    evaluation_fold_count=1,
                    progress_sink=progress_sink,
                )
                inner_pair_predictions.update(predictions)
                fit_summaries.append(summary)
            inner_cases, inner_scores = build_margin_view_cases(
                samples=inner_samples,
                grouped_records=grouped_records,
                pair_logits=inner_pair_predictions,
            )
            selected = _select_threshold(
                family=family,
                cases=inner_cases,
                scores=inner_scores,
            )
            family_inner_rows.append(selected)
        chosen_inner = max(family_inner_rows, key=_family_selection_key)
        chosen_spec: RankingSpec = chosen_inner["spec"]

        heldout_by_family = {}
        for family in _FAMILIES:
            fit_id = f"outer_{outer_index}_{family}_final"
            predictions, summary = trainer.fit_predict(
                family=family,
                training_rows=tuple(row for row in pair_rows if row.fold_id in outer_train_folds),
                evaluation_rows=tuple(row for row in pair_rows if row.fold_id == outer_fold),
                fit_id=fit_id,
                training_fold_count=len(outer_train_folds),
                evaluation_fold_count=1,
                progress_sink=progress_sink,
            )
            fit_summaries.append(summary)
            heldout_cases, heldout_scores = build_margin_view_cases(
                samples=samples_by_fold[outer_fold],
                grouped_records=grouped_records,
                pair_logits=predictions,
            )
            family_oof_cases[family].extend(heldout_cases)
            family_oof_scores[family].update(heldout_scores)
            heldout_by_family[family] = (heldout_cases, heldout_scores)

        heldout_cases, heldout_scores = heldout_by_family[chosen_spec.family]
        heldout_metrics = stage172.evaluate_predictions(
            heldout_cases,
            heldout_scores,
            {outer_fold: chosen_spec},
        )
        selected_oof_cases.extend(heldout_cases)
        selected_oof_scores.update(heldout_scores)
        selected_specs[outer_fold] = chosen_spec
        outer_rows.append(
            {
                "heldout_fold": outer_fold,
                "inner_fit_count": len(outer_train_folds) * len(_FAMILIES),
                "outer_fit_count": len(_FAMILIES),
                "inner_case_count": len(inner_samples) * 2,
                "heldout_case_count": len(heldout_cases),
                "family_diagnostics": [_public_selection_row(row) for row in family_inner_rows],
                "selected_family": chosen_spec.family,
                "selected_threshold": chosen_spec.threshold,
                "selected_spec_id": chosen_spec.spec_id,
                "selected_inner_eligible": bool(chosen_inner["eligible"]),
                "selected_inner_metrics": chosen_inner["metrics"].public_dict(),
                "selected_inner_safe_fold_count": chosen_inner["safe_fold_count"],
                "heldout_metrics": heldout_metrics.public_dict(),
            }
        )
        _emit(progress_sink, phase="outer_fold_complete", completed=outer_index, total=5)
    if len(fit_summaries) != _EXPECTED_NESTED_FITS:
        raise RuntimeError("Stage175 nested fit count drifted")
    return {
        "outer_folds": outer_rows,
        "oof_cases": tuple(selected_oof_cases),
        "oof_view_scores": selected_oof_scores,
        "selected_specs": selected_specs,
        "family_oof_cases": {family: tuple(cases) for family, cases in family_oof_cases.items()},
        "family_oof_scores": family_oof_scores,
        "fit_summaries": tuple(fit_summaries),
    }


def _select_threshold(
    *,
    family: str,
    cases: Sequence[stage172.EvidenceViewCase],
    scores: Mapping[str, float],
) -> dict[str, Any]:
    rows = [
        stage174._threshold_evaluation(cases, scores, RankingSpec(family, threshold))
        for threshold in _MARGIN_THRESHOLDS
    ]
    selected = max(rows, key=stage174._threshold_selection_key)
    return {
        **selected,
        "eligible_threshold_count": sum(row["eligible"] for row in rows),
    }


def _family_selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    spec: RankingSpec = row["spec"]
    return (*stage174._threshold_selection_key(row), spec.family)


def _public_selection_row(row: Mapping[str, Any]) -> dict[str, Any]:
    spec: RankingSpec = row["spec"]
    return {
        "family": spec.family,
        "threshold": spec.threshold,
        "spec_id": spec.spec_id,
        "eligible_threshold_count": row["eligible_threshold_count"],
        "selected_eligible": bool(row["eligible"]),
        "safe_fold_count": row["safe_fold_count"],
        "metrics": row["metrics"].public_dict(),
    }


def run_stage175_grouped_ranking_cv(
    *,
    stage174_report_path: Path,
    stage173_report_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    documents_path: Path,
    model_snapshot_path: Path,
    encoder_batch_size: int = 64,
    progress_sink: ProgressSink | None = None,
    trainer_factory: Callable[[Path, stage169.Stage169ResourceTracker, Any], RankingFoldTrainer]
    | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    started_cpu = time.process_time()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    source_paths = {
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
    stage174_report = _load_json_object(stage174_report_path)
    if stage174_report.get("decision", {}).get("status") != (
        "stage174_supervised_cross_encoder_insufficient"
    ):
        raise ValueError("Stage174 did not authorize grouped ranking research")
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="sources_authorized")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage175 formal ranking requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    tracker = stage169.Stage169ResourceTracker(torch_module=torch)
    tracker.capture("analysis_start")

    samples = load_primeqa_hybrid_split_samples(train_split_path)
    if len(samples) != _EXPECTED_TRAIN_ROWS or any(
        sample.assigned_split != "train" for sample in samples
    ):
        raise ValueError("Stage175 accepts only the exact 562-row train split")
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
        raise RuntimeError("Stage175 requires both authorized local dense channels")
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
        raise RuntimeError("Stage175 candidate replay row count drifted")
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
        raise RuntimeError("Stage175 base view case count drifted")
    del frozen_scorer
    gc.collect()
    torch.cuda.empty_cache()
    tracker.capture("frozen_cross_encoder_released")
    pair_rows = stage174.build_pair_fold_rows(
        pairs=pairs,
        base_cases=base_cases,
        frozen_scores=frozen_scores,
    )
    sampled_groups = build_sampled_training_groups(pair_rows)
    pair_data_ready_at = time.perf_counter()
    _emit(
        progress_sink,
        phase="ranking_group_data_ready",
        complete_pair_count=len(pair_rows),
        sampled_group_count=len(sampled_groups),
    )

    factory = trainer_factory or (
        lambda snapshot, resource_tracker, torch_module: LocalGroupedRankingTrainer(
            snapshot_path=snapshot,
            tracker=resource_tracker,
            torch_module=torch_module,
        )
    )
    trainer = factory(model_snapshot_path, tracker, torch)
    nested = run_grouped_nested_ranking(
        samples=samples,
        grouped_records=grouped_records,
        pair_rows=pair_rows,
        trainer=trainer,
        progress_sink=progress_sink,
    )
    nested_finished_at = time.perf_counter()
    tracker.capture("nested_ranking_finished")

    outer_folds = nested["outer_folds"]
    oof_cases = nested["oof_cases"]
    oof_scores = nested["oof_view_scores"]
    selected_specs = nested["selected_specs"]
    fit_summaries: Sequence[RankingFitSummary] = nested["fit_summaries"]
    oof_metrics = stage172.evaluate_predictions(oof_cases, oof_scores, selected_specs)
    oof_gates = stage172._quality_gates(oof_metrics)
    fold_metrics = stage174._outer_fold_metrics(oof_cases, oof_scores, selected_specs)
    all_outer_safety_passed = all(
        metrics["insufficient_final_compose_rate"]
        <= stage172._GATE_THRESHOLDS["insufficient_final_compose_rate_max"]
        for metrics in fold_metrics.values()
    )
    all_inner_selected_eligible = all(row["selected_inner_eligible"] for row in outer_folds)

    family_diagnostics = {}
    family_selection_rows = []
    for family in _FAMILIES:
        cases = nested["family_oof_cases"][family]
        scores = nested["family_oof_scores"][family]
        selected = _select_threshold(family=family, cases=cases, scores=scores)
        family_selection_rows.append(selected)
        family_diagnostics[family] = _public_selection_row(selected)
    final_selection = max(family_selection_rows, key=_family_selection_key)
    final_spec: RankingSpec = final_selection["spec"]
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
        _check("stage174_authorized_grouped_ranking", True),
        _check("exact_train_row_count", len(samples) == _EXPECTED_TRAIN_ROWS),
        _check("exact_candidate_row_count", len(records) == _EXPECTED_CANDIDATE_ROWS),
        _check("exact_view_case_count", len(oof_cases) == _EXPECTED_VIEW_CASES),
        _check("complete_pair_score_coverage", len(frozen_scores) == len(pair_rows)),
        _check("exact_nested_fit_count", len(fit_summaries) == _EXPECTED_NESTED_FITS),
        _check("both_frozen_ranking_families", tuple(family_diagnostics) == _FAMILIES),
        _check("five_grouped_outer_folds", len(outer_folds) == 5),
        _check("complete_oof_view_coverage", len(oof_scores) == len(oof_cases)),
        _check("twenty_one_frozen_margin_thresholds", len(_MARGIN_THRESHOLDS) == 21),
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
    stage174_metrics = stage174_report["nested_cv"]["oof_metrics"]
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
            "Train-only comparison of anchored pairwise and explicit-none listwise "
            "cross-encoder objectives with 50 grouped nested-CV fits."
        ),
        "source_authorization": fingerprints,
        "frozen_protocol": {
            "model_snapshot_revision": model_snapshot_path.name,
            "ranking_families": list(_FAMILIES),
            "pairwise_objective": "ranknet_gold_over_negatives_plus_none_anchor",
            "pairwise_anchor_weight": _PAIRWISE_ANCHOR_WEIGHT,
            "listwise_objective": "group_softmax_with_fixed_none_logit",
            "view_score": "top1_logit_minus_max_zero_top2_logit",
            "train_epochs": _TRAIN_EPOCHS,
            "train_pair_budget": _TRAIN_PAIR_BUDGET,
            "inference_batch_size": _INFERENCE_BATCH_SIZE,
            "max_length": _MAX_LENGTH,
            "learning_rate": _LEARNING_RATE,
            "weight_decay": _WEIGHT_DECAY,
            "gradient_clip_norm": _GRADIENT_CLIP_NORM,
            "margin_thresholds": list(_MARGIN_THRESHOLDS),
            "nested_fit_count": _EXPECTED_NESTED_FITS,
            "outer_fold_count": 5,
            "inner_model_count_per_outer_fold": 8,
            "outer_model_count_per_outer_fold": 2,
            "development_and_test_closed": True,
        },
        "split_contract": {
            "loaded_split": "train",
            "family_and_threshold_fit": "inner_oof_question_folds_only",
            "outer_evaluation": "one_shot_heldout_question_fold",
            "family_diagnostic": "complete_outer_oof_predictions_only",
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
            "selected_family_counts": {
                family: sum(spec.family == family for spec in selected_specs.values())
                for family in _FAMILIES
            },
            "family_full_oof_diagnostics": family_diagnostics,
            "final_full_train_oof_selected_family": final_spec.family,
            "final_full_train_oof_selected_threshold": final_spec.threshold,
            "final_full_train_oof_selected_spec_id": final_spec.spec_id,
            "final_full_train_oof_selected_eligible": bool(final_selection["eligible"]),
            "all_inner_selected_specs_eligible": all_inner_selected_eligible,
            "oof_metrics": oof_metrics.public_dict(),
            "oof_quality_gates": oof_gates,
            "outer_fold_metrics": fold_metrics,
            "all_outer_folds_safety_passed": all_outer_safety_passed,
        },
        "stage174_comparison": {
            "stage174_oof_metrics": stage174_metrics,
            "stage175_oof_metrics": oof_metrics.public_dict(),
            "metric_delta": {
                name: round(getattr(oof_metrics, name) - float(stage174_metrics[name]), 6)
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
            "nested_ranking": round(nested_finished_at - pair_data_ready_at, 6),
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
                "advance_to_stage176_train_only_grouped_ranking_runtime_e2e"
                if candidate_selected
                else "stage175_grouped_ranking_insufficient"
            ),
            "development_opened": False,
            "test_opened": False,
            "default_runtime_activation": False,
        },
    }
    forbidden = sorted(stage174._forbidden_keys_found(report) | _forbidden_keys_found(report))
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
        report["decision"]["status"] = "stage175_process_invalid"
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report


def write_stage175_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage175Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    nested = report["nested_cv"]
    metrics = nested["oof_metrics"]
    stage174_metrics = report["stage174_comparison"]["stage174_oof_metrics"]
    folds = nested["outer_fold_metrics"]
    diagnostics = report["training_diagnostics"]
    resources = report["resource_consumption"]
    timings = report["timing_seconds"]
    charts = {
        "oof_quality_gates.svg": _chart(
            "Stage 175 grouped OOF quality gates",
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
        "stage174_stage175_rates.svg": _chart(
            "Stage 174 versus Stage 175 grouped OOF rates",
            tuple(
                bar
                for label, name in (
                    ("initial compose", "initial_visible_compose_rate"),
                    ("final compose", "alternate_only_final_compose_rate"),
                    ("exact path", "alternate_only_path_success_rate"),
                    ("false compose", "insufficient_final_compose_rate"),
                )
                for bar in (
                    _rate_bar(f"174 {label}", stage174_metrics[name]),
                    _rate_bar(f"175 {label}", metrics[name]),
                )
            ),
            x_label="Rate",
        ),
        "family_selection.svg": _chart(
            "Stage 175 inner-selected ranking families",
            tuple(
                BarDatum(family, count, str(count))
                for family, count in nested["selected_family_counts"].items()
            ),
            x_label="Outer-fold selections",
        ),
        "family_oof_quality.svg": _chart(
            "Stage 175 full-OOF family balanced accuracy",
            tuple(
                BarDatum(
                    family,
                    row["metrics"]["balanced_accuracy"],
                    f"{row['metrics']['balanced_accuracy']:.3f}",
                )
                for family, row in nested["family_full_oof_diagnostics"].items()
            ),
            x_label="Balanced accuracy",
        ),
        "outer_fold_safety.svg": _chart(
            "Stage 175 outer-fold insufficient final compose",
            tuple(
                _rate_bar(fold_id, fold["insufficient_final_compose_rate"])
                for fold_id, fold in folds.items()
            ),
            x_label="False-compose rate",
        ),
        "outer_fold_path.svg": _chart(
            "Stage 175 outer-fold alternate exact path",
            tuple(
                _rate_bar(fold_id, fold["alternate_only_path_success_rate"])
                for fold_id, fold in folds.items()
            ),
            x_label="Exact-path rate",
        ),
        "training_loss.svg": _chart(
            "Stage 175 final-epoch mean training loss by family",
            tuple(
                BarDatum(
                    family,
                    row["final_epoch_loss_mean"],
                    f"{row['final_epoch_loss_mean']:.4f}",
                )
                for family, row in diagnostics["by_family"].items()
            ),
            x_label="Objective loss",
        ),
        "timing.svg": _chart(
            "Stage 175 phase wall times",
            tuple(
                BarDatum(name.replace("_", " "), value, f"{value:.2f} s")
                for name, value in timings.items()
            ),
            x_label="Seconds",
        ),
        "resources.svg": _chart(
            "Stage 175 process and GPU resource peaks",
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
        written.append(Stage175Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _training_diagnostics(summaries: Sequence[RankingFitSummary]) -> dict[str, Any]:
    by_family = {}
    for family in _FAMILIES:
        family_rows = tuple(summary for summary in summaries if summary.family == family)
        by_family[family] = {
            "fit_count": len(family_rows),
            "first_epoch_loss_mean": round(
                statistics.fmean(row.first_epoch_mean_loss for row in family_rows), 6
            ),
            "final_epoch_loss_mean": round(
                statistics.fmean(row.final_epoch_mean_loss for row in family_rows), 6
            ),
            "fit_seconds_mean": round(statistics.fmean(row.fit_seconds for row in family_rows), 6),
            "inference_seconds_mean": round(
                statistics.fmean(row.inference_seconds for row in family_rows), 6
            ),
        }
    return {
        "fit_count": len(summaries),
        "by_family": by_family,
        "training_group_count_distribution": stage174._distribution(
            [summary.training_group_count for summary in summaries]
        ),
        "training_pair_count_distribution": stage174._distribution(
            [summary.training_pair_count for summary in summaries]
        ),
        "optimizer_step_count_distribution": stage174._distribution(
            [summary.optimizer_step_count for summary in summaries]
        ),
    }


def _fit_seed(family: str, fit_id: str) -> int:
    digest = hashlib.sha256(f"{family}:{fit_id}".encode()).hexdigest()
    return 175_000 + (int(digest[:8], 16) % 10_000)


def _authorize_sources(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    for name, expected in _SOURCE_HASHES.items():
        if fingerprints[name]["sha256"] != expected:
            raise ValueError(f"Stage175 source hash mismatch: {name}")


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
