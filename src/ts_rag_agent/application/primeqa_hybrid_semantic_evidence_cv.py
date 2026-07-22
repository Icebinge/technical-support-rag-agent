from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ts_rag_agent.application import primeqa_hybrid_evidence_entailment_cv as stage172
from ts_rag_agent.application import primeqa_hybrid_iterative_router_calibration as stage169
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
from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.infrastructure.bm25_retriever import tokenize_text
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 173"
_CREATED_AT = "2026-07-22"
_ANALYSIS_ID = "primeqa_hybrid_frozen_cross_encoder_semantic_evidence_nested_cv_v1"
_EXPECTED_TRAIN_ROWS = 562
_EXPECTED_CANDIDATE_ROWS = 112_400
_EXPECTED_VIEW_CASES = 1_124
_MODEL_FAMILIES = ("logistic", "histogram_gbdt")
_FEATURE_PROFILES = ("semantic_only", "hybrid")
_THRESHOLDS = tuple(round(value / 100, 2) for value in range(10, 91, 5))
_SEMANTIC_FEATURE_NAMES = (
    "semantic_score_max",
    "semantic_score_second",
    "semantic_score_top1_top2_margin",
    "semantic_score_top3_mean",
    "semantic_score_mean",
    "semantic_score_median",
    "semantic_score_std",
    "semantic_score_range",
    "semantic_nonnegative_fraction",
    "semantic_top1_baseline_inverse_rank",
    "semantic_gain_over_initial_max",
    "semantic_top1_new_alternate_indicator",
)
_SEMANTIC_ONLY_FEATURE_NAMES = (
    *_SEMANTIC_FEATURE_NAMES,
    "phase_final",
    "visible_document_count",
)
_HYBRID_FEATURE_NAMES = (*stage172._MODEL_FEATURE_NAMES, *_SEMANTIC_FEATURE_NAMES)
_FEATURE_NAMES_BY_PROFILE = {
    "semantic_only": _SEMANTIC_ONLY_FEATURE_NAMES,
    "hybrid": _HYBRID_FEATURE_NAMES,
}
_QUERY_AWARE_EXCERPT_CHARS = 1_600
_MAX_QUERY_TOKENS_FOR_WINDOW_SEARCH = 16
_MAX_OCCURRENCES_PER_QUERY_TOKEN = 4
_CROSS_ENCODER_MAX_LENGTH = 512
_GPU_BATCH_SIZE = 64
_EVENT_PAIR_BATCH_SIZE = 512
_SOURCE_HASHES = {
    "stage172": "48d0309e98d044f2cc89fa42526ef9c5da1c8bf9e7b2e188a60c372f8c7dd827",
    "stage80": "2441bb1cb1e7888299d3f57962b18cd59df84e2086ac281105abcacfc144880f",
    "train": "cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155",
    "documents": "f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b",
    "model_config": "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc",
    "model_weights": "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae",
    "model_tokenizer": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "model_tokenizer_config": ("a5c2e5a7b1a29a0702cd28c08a399b5ecc110c263009d17f7e3b415f25905fd8"),
    "model_special_tokens": ("3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6"),
    "model_vocab": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
_MODEL_SOURCE_FILES = {
    "model_config": "config.json",
    "model_weights": "model.safetensors",
    "model_tokenizer": "tokenizer.json",
    "model_tokenizer_config": "tokenizer_config.json",
    "model_special_tokens": "special_tokens_map.json",
    "model_vocab": "vocab.txt",
}
_FORBIDDEN_PUBLIC_KEYS = stage172._FORBIDDEN_PUBLIC_KEYS | {
    "document_text",
    "pair_identity",
    "passage_text",
    "question",
}

ProgressSink = Callable[[Mapping[str, Any]], None]
ModelFamily = Literal["logistic", "histogram_gbdt"]
FeatureProfile = Literal["semantic_only", "hybrid"]


@dataclass(frozen=True)
class SemanticModelSpec:
    feature_profile: FeatureProfile
    model_family: ModelFamily
    threshold: float

    @property
    def spec_id(self) -> str:
        material = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class SemanticPairInput:
    private_identity: str
    question_text: str
    passage_text: str
    positive_label: bool


@dataclass(frozen=True)
class SemanticScoringSummary:
    pair_count: int
    event_batch_count: int
    scoring_seconds: float
    pairs_per_second: float


@dataclass(frozen=True)
class Stage173Visualization:
    name: str
    path: str


class SemanticPairScorer(Protocol):
    def score(
        self,
        pairs: Sequence[SemanticPairInput],
        *,
        progress_sink: ProgressSink | None,
    ) -> tuple[dict[str, float], SemanticScoringSummary]: ...


class QueryAwareCrossEncoderTextPolicy:
    """Build one bounded question-passage pair without using answer labels."""

    def __init__(
        self,
        *,
        excerpt_chars: int = _QUERY_AWARE_EXCERPT_CHARS,
        max_query_tokens: int = _MAX_QUERY_TOKENS_FOR_WINDOW_SEARCH,
        max_occurrences_per_token: int = _MAX_OCCURRENCES_PER_QUERY_TOKEN,
    ) -> None:
        if excerpt_chars <= 0 or max_query_tokens <= 0 or max_occurrences_per_token <= 0:
            raise ValueError("Stage173 text-policy limits must be positive")
        self._excerpt_chars = excerpt_chars
        self._max_query_tokens = max_query_tokens
        self._max_occurrences_per_token = max_occurrences_per_token

    def passage(self, *, question: str, document: PrimeQADocument) -> str:
        query_tokens = frozenset(tokenize_text(question))
        excerpt = self._query_aware_excerpt(document.text, query_tokens=query_tokens)
        return f"{document.title}\n\n{excerpt}"

    def _query_aware_excerpt(self, text: str, *, query_tokens: frozenset[str]) -> str:
        if len(text) <= self._excerpt_chars:
            return text
        lowered = text.lower()
        last_start = max(0, len(text) - self._excerpt_chars)
        starts = {0, last_start}
        search_tokens = sorted(query_tokens, key=lambda token: (-len(token), token))[
            : self._max_query_tokens
        ]
        for token in search_tokens:
            search_from = 0
            for _ in range(self._max_occurrences_per_token):
                position = lowered.find(token, search_from)
                if position < 0:
                    break
                starts.add(min(last_start, max(0, position - (self._excerpt_chars // 3))))
                search_from = position + max(1, len(token))
        scored = []
        for start in sorted(starts):
            window = text[start : start + self._excerpt_chars]
            overlap = len(query_tokens & set(tokenize_text(window)))
            scored.append((overlap, -start, window))
        return max(scored)[2]


class LocalCrossEncoderSemanticScorer:
    """Score frozen local MiniLM pairs in event batches on one CUDA device."""

    def __init__(
        self,
        *,
        snapshot_path: Path,
        tracker: stage169.Stage169ResourceTracker,
        batch_size: int = _GPU_BATCH_SIZE,
        max_length: int = _CROSS_ENCODER_MAX_LENGTH,
        event_pair_batch_size: int = _EVENT_PAIR_BATCH_SIZE,
    ) -> None:
        if batch_size <= 0 or max_length <= 0 or event_pair_batch_size <= 0:
            raise ValueError("Stage173 scorer limits must be positive")
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            str(snapshot_path),
            max_length=max_length,
            device="cuda",
        )
        self._tracker = tracker
        self._batch_size = batch_size
        self._event_pair_batch_size = event_pair_batch_size

    def score(
        self,
        pairs: Sequence[SemanticPairInput],
        *,
        progress_sink: ProgressSink | None,
    ) -> tuple[dict[str, float], SemanticScoringSummary]:
        if len({pair.private_identity for pair in pairs}) != len(pairs):
            raise ValueError("Stage173 semantic pair identities must be unique")
        started = time.perf_counter()
        result: dict[str, float] = {}
        event_batch_count = (len(pairs) + self._event_pair_batch_size - 1) // (
            self._event_pair_batch_size
        )
        for batch_index, start in enumerate(
            range(0, len(pairs), self._event_pair_batch_size),
            start=1,
        ):
            batch = pairs[start : start + self._event_pair_batch_size]
            scores = np.asarray(
                self._model.predict(
                    [(pair.question_text, pair.passage_text) for pair in batch],
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                ),
                dtype=float,
            ).reshape(-1)
            if len(scores) != len(batch) or not np.all(np.isfinite(scores)):
                raise RuntimeError("Stage173 cross-encoder produced invalid scores")
            result.update(
                {
                    pair.private_identity: float(score)
                    for pair, score in zip(batch, scores, strict=True)
                }
            )
            self._tracker.capture(f"semantic_event_batch_{batch_index}")
            _emit(
                progress_sink,
                phase="semantic_pair_scoring",
                completed=min(start + len(batch), len(pairs)),
                total=len(pairs),
            )
        seconds = time.perf_counter() - started
        return result, SemanticScoringSummary(
            pair_count=len(pairs),
            event_batch_count=event_batch_count,
            scoring_seconds=round(seconds, 6),
            pairs_per_second=round(len(pairs) / seconds, 6),
        )


class SemanticEvidencePredictor:
    """Fit a balanced view classifier over one frozen Stage173 feature profile."""

    def __init__(self, spec: SemanticModelSpec) -> None:
        self._feature_names = _FEATURE_NAMES_BY_PROFILE[spec.feature_profile]
        if spec.model_family == "logistic":
            self._model: Any = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            max_iter=2_000,
                            random_state=173,
                        ),
                    ),
                ]
            )
        elif spec.model_family == "histogram_gbdt":
            self._model = HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=150,
                max_leaf_nodes=9,
                l2_regularization=1.0,
                random_state=173,
            )
        else:
            raise ValueError(f"unsupported Stage173 model family: {spec.model_family}")
        self._model_family = spec.model_family

    def fit(self, cases: Sequence[stage172.EvidenceViewCase]) -> None:
        matrix = _feature_matrix(cases, self._feature_names)
        labels = np.asarray([case.sufficient_label for case in cases], dtype=int)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("Stage173 fitting requires both evidence classes")
        if self._model_family == "histogram_gbdt":
            counts = np.bincount(labels, minlength=2)
            weights = np.asarray([len(labels) / (2 * counts[label]) for label in labels])
            self._model.fit(matrix, labels, sample_weight=weights)
            return
        self._model.fit(matrix, labels)

    def predict(self, cases: Sequence[stage172.EvidenceViewCase]) -> dict[str, float]:
        probabilities = self._model.predict_proba(_feature_matrix(cases, self._feature_names))[:, 1]
        return {
            case.private_identity: float(probability)
            for case, probability in zip(cases, probabilities, strict=True)
        }


def build_stage173_specs() -> tuple[SemanticModelSpec, ...]:
    return tuple(
        SemanticModelSpec(profile, family, threshold)
        for profile in _FEATURE_PROFILES
        for family in _MODEL_FAMILIES
        for threshold in _THRESHOLDS
    )


def build_semantic_evidence_cases(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
    documents_by_id: Mapping[str, PrimeQADocument],
    scorer: SemanticPairScorer,
    text_policy: QueryAwareCrossEncoderTextPolicy,
    progress_sink: ProgressSink | None,
) -> tuple[
    tuple[stage172.EvidenceViewCase, ...],
    tuple[SemanticPairInput, ...],
    Mapping[str, float],
    SemanticScoringSummary,
]:
    base_cases = stage172.build_evidence_view_cases(
        samples=samples,
        grouped_records=grouped_records,
    )
    base_by_identity = {case.private_identity: case for case in base_cases}
    pairs = []
    views_by_sample: dict[
        str,
        tuple[
            tuple[ContextCandidateRecord, ...],
            tuple[ContextCandidateRecord, ...],
        ],
    ] = {}
    for sample in samples:
        records = tuple(grouped_records[sample.sample_id])
        initial = select_current_query_overlap_top10(records).selected
        alternate = select_original_rrf_top10(records).selected
        final = stage172._deduplicate_records((*initial, *alternate))
        views_by_sample[sample.sample_id] = (initial, final)
        question = _question_text(sample)
        for record in final:
            pair_identity = _pair_identity(sample.sample_id, record.document_id)
            pairs.append(
                SemanticPairInput(
                    private_identity=pair_identity,
                    question_text=question,
                    passage_text=text_policy.passage(
                        question=question,
                        document=documents_by_id[record.document_id],
                    ),
                    positive_label=bool(
                        sample.answerable and sample.answer_doc_id == record.document_id
                    ),
                )
            )
    scores, scoring_summary = scorer.score(pairs, progress_sink=progress_sink)
    if len(scores) != len(pairs):
        raise RuntimeError("Stage173 semantic score coverage is incomplete")

    cases = []
    for sample in samples:
        initial, final = views_by_sample[sample.sample_id]
        initial_scores = {
            record.document_id: scores[_pair_identity(sample.sample_id, record.document_id)]
            for record in initial
        }
        for phase, visible in (("initial", initial), ("final", final)):
            identity = stage172._sha256_text(f"{sample.sample_id}:{phase}")
            base = base_by_identity[identity]
            semantic = summarize_semantic_view(
                visible_records=visible,
                scores={
                    record.document_id: scores[_pair_identity(sample.sample_id, record.document_id)]
                    for record in visible
                },
                initial_scores=initial_scores,
                initial_document_ids=frozenset(initial_scores),
                phase=phase,
            )
            cases.append(
                stage172.EvidenceViewCase(
                    private_identity=base.private_identity,
                    group_identity=base.group_identity,
                    fold_id=base.fold_id,
                    phase=base.phase,
                    stratum=base.stratum,
                    features={**base.features, **semantic},
                    sufficient_label=base.sufficient_label,
                )
            )
    return tuple(cases), tuple(pairs), scores, scoring_summary


def summarize_semantic_view(
    *,
    visible_records: Sequence[ContextCandidateRecord],
    scores: Mapping[str, float],
    initial_scores: Mapping[str, float],
    initial_document_ids: frozenset[str],
    phase: str,
) -> dict[str, float]:
    if phase not in {"initial", "final"}:
        raise ValueError("Stage173 phase must be initial or final")
    if not visible_records or len(scores) != len(visible_records):
        raise ValueError("Stage173 semantic summary requires one score per visible document")
    ranked = sorted(
        visible_records,
        key=lambda record: (-scores[record.document_id], record.baseline_rank),
    )
    ordered = [float(scores[record.document_id]) for record in ranked]
    top3 = ordered[: min(3, len(ordered))]
    initial_max = max(initial_scores.values())
    values = {
        "semantic_score_max": ordered[0],
        "semantic_score_second": ordered[1] if len(ordered) > 1 else ordered[0],
        "semantic_score_top1_top2_margin": (ordered[0] - ordered[1] if len(ordered) > 1 else 0.0),
        "semantic_score_top3_mean": float(statistics.fmean(top3)),
        "semantic_score_mean": float(statistics.fmean(ordered)),
        "semantic_score_median": float(statistics.median(ordered)),
        "semantic_score_std": float(statistics.pstdev(ordered)),
        "semantic_score_range": max(ordered) - min(ordered),
        "semantic_nonnegative_fraction": sum(score >= 0 for score in ordered) / len(ordered),
        "semantic_top1_baseline_inverse_rank": 1.0 / ranked[0].baseline_rank,
        "semantic_gain_over_initial_max": ordered[0] - initial_max if phase == "final" else 0.0,
        "semantic_top1_new_alternate_indicator": float(
            phase == "final" and ranked[0].document_id not in initial_document_ids
        ),
    }
    if tuple(values) != _SEMANTIC_FEATURE_NAMES:
        raise RuntimeError("Stage173 semantic feature order drifted")
    return values


def run_stage173_semantic_evidence_cv(
    *,
    stage172_report_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    documents_path: Path,
    model_snapshot_path: Path,
    encoder_batch_size: int = 64,
    progress_sink: ProgressSink | None = None,
    scorer_factory: Callable[[Path, stage169.Stage169ResourceTracker], SemanticPairScorer]
    | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    started_cpu = time.process_time()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    source_paths = {
        "stage172": stage172_report_path,
        "stage80": stage80_report_path,
        "train": train_split_path,
        "documents": documents_path,
        **{
            source_name: model_snapshot_path / filename
            for source_name, filename in _MODEL_SOURCE_FILES.items()
        },
    }
    fingerprints = {name: _resolved_fingerprint(path) for name, path in source_paths.items()}
    _authorize_sources(fingerprints)
    stage172_report = _load_json_object(stage172_report_path)
    if stage172_report.get("decision", {}).get("status") != (
        "stage172_no_grouped_oof_safe_evidence_classifier"
    ):
        raise ValueError("Stage172 did not authorize direct semantic evidence scoring")
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="sources_authorized")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage173 formal scoring requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    tracker = stage169.Stage169ResourceTracker(torch_module=torch)
    tracker.capture("analysis_start")

    samples = load_primeqa_hybrid_split_samples(train_split_path)
    if len(samples) != _EXPECTED_TRAIN_ROWS or any(
        sample.assigned_split != "train" for sample in samples
    ):
        raise ValueError("Stage173 accepts only the exact 562-row train split")
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
        raise RuntimeError("Stage173 requires both authorized local dense channels")
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
        raise RuntimeError("Stage173 candidate replay row count drifted")
    replay_ready_at = time.perf_counter()
    tracker.capture("candidate_replay_ready")

    factory = scorer_factory or (
        lambda snapshot, resource_tracker: LocalCrossEncoderSemanticScorer(
            snapshot_path=snapshot,
            tracker=resource_tracker,
        )
    )
    scorer = factory(model_snapshot_path, tracker)
    model_ready_at = time.perf_counter()
    tracker.capture("cross_encoder_loaded")
    cases, pairs, pair_scores, scoring_summary = build_semantic_evidence_cases(
        samples=samples,
        grouped_records=records_by_sample(records),
        documents_by_id=documents_by_id,
        scorer=scorer,
        text_policy=QueryAwareCrossEncoderTextPolicy(),
        progress_sink=progress_sink,
    )
    if len(cases) != _EXPECTED_VIEW_CASES:
        raise RuntimeError("Stage173 view case count drifted")
    scored_at = time.perf_counter()
    tracker.capture("semantic_cases_ready")
    _emit(
        progress_sink,
        phase="semantic_cases_ready",
        pair_count=len(pairs),
        case_count=len(cases),
    )

    specs = build_stage173_specs()
    outer_folds, oof_predictions, selected_specs = _run_nested_cv(
        cases=cases,
        specs=specs,
        progress_sink=progress_sink,
    )
    final_spec_row = _select_full_train_spec(cases=cases, specs=specs)
    final_spec: SemanticModelSpec = final_spec_row["spec"]
    cv_finished_at = time.perf_counter()
    tracker.capture("nested_cv_finished")

    oof_metrics = stage172.evaluate_predictions(cases, oof_predictions, selected_specs)
    oof_gates = stage172._quality_gates(oof_metrics)
    fold_metrics = _outer_fold_metrics(cases, oof_predictions, selected_specs)
    all_outer_safety_passed = all(
        metrics["insufficient_final_compose_rate"]
        <= stage172._GATE_THRESHOLDS["insufficient_final_compose_rate_max"]
        for metrics in fold_metrics.values()
    )
    all_inner_selected_eligible = all(row["selected_spec_inner_eligible"] for row in outer_folds)
    candidate_selected = (
        all_inner_selected_eligible
        and bool(final_spec_row["eligible"])
        and all(gate["passed"] for gate in oof_gates)
        and all_outer_safety_passed
    )
    semantic_diagnostics = _semantic_diagnostics(cases, pairs, pair_scores)
    tracker.capture("report_assembly")
    finished_at = time.perf_counter()

    snapshots = tracker.snapshots
    resource_consumption = {
        "sampling_mode": "event_driven_in_process_without_monitor_polling",
        "semantic_event_batch_count": scoring_summary.event_batch_count,
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
        "gpu_model_loaded": True,
        "gpu_model_device": torch.cuda.get_device_name(0),
        "model_generation_calls": 0,
        "semantic_pair_count": scoring_summary.pair_count,
        "semantic_scoring_seconds": scoring_summary.scoring_seconds,
        "semantic_pairs_per_second": scoring_summary.pairs_per_second,
    }
    process_guards = [
        _check("stage172_authorized_semantic_redesign", True),
        _check("exact_train_row_count", len(samples) == _EXPECTED_TRAIN_ROWS),
        _check("exact_candidate_row_count", len(records) == _EXPECTED_CANDIDATE_ROWS),
        _check("two_views_per_train_row", len(cases) == _EXPECTED_VIEW_CASES),
        _check("complete_semantic_pair_coverage", len(pair_scores) == len(pairs)),
        _check("five_grouped_outer_folds", len(outer_folds) == 5),
        _check("complete_oof_prediction_coverage", len(oof_predictions) == len(cases)),
        _check("two_frozen_feature_profiles", len(_FEATURE_PROFILES) == 2),
        _check("two_frozen_model_families", len(_MODEL_FAMILIES) == 2),
        _check("seventeen_frozen_thresholds", len(_THRESHOLDS) == 17),
        _check("frozen_candidate_spec_count", len(specs) == 68),
        _check("frozen_semantic_feature_count", len(_SEMANTIC_FEATURE_NAMES) == 12),
        _check("development_not_loaded", True),
        _check("test_not_loaded", True),
        _check("cross_encoder_not_fine_tuned", True),
        _check("answer_generation_not_run", True),
        _check("agent_turns_not_run", True),
        _check("retry_count_zero", True),
        _check("fallback_count_zero", True),
        _check("default_runtime_unchanged", True),
    ]
    stage172_metrics = stage172_report["nested_cv"]["oof_metrics"]
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train-only frozen local cross-encoder question-to-passage semantic scoring "
            "with grouped nested five-fold evidence-view classification."
        ),
        "source_authorization": fingerprints,
        "frozen_protocol": {
            "model_snapshot_revision": model_snapshot_path.name,
            "cross_encoder_max_length": _CROSS_ENCODER_MAX_LENGTH,
            "query_aware_excerpt_chars": _QUERY_AWARE_EXCERPT_CHARS,
            "gpu_batch_size": _GPU_BATCH_SIZE,
            "event_pair_batch_size": _EVENT_PAIR_BATCH_SIZE,
            "cross_encoder_fine_tuned": False,
            "semantic_features": list(_SEMANTIC_FEATURE_NAMES),
            "feature_profiles": {
                profile: list(names) for profile, names in _FEATURE_NAMES_BY_PROFILE.items()
            },
            "model_families": list(_MODEL_FAMILIES),
            "thresholds": list(_THRESHOLDS),
            "outer_fold_count": 5,
            "gold_labels_used_only_for_fit_and_evaluation": True,
            "development_and_test_closed": True,
        },
        "split_contract": {
            "loaded_split": "train",
            "fit_split": "outer_train_folds_only",
            "development_loaded": False,
            "test_loaded": False,
        },
        "case_summary": {
            "train_question_count": len(samples),
            "view_case_count": len(cases),
            "positive_view_count": sum(case.sufficient_label for case in cases),
            "negative_view_count": sum(not case.sufficient_label for case in cases),
            "semantic_pair_count": len(pairs),
            "positive_pair_count": sum(pair.positive_label for pair in pairs),
            "negative_pair_count": sum(not pair.positive_label for pair in pairs),
            "private_case_rows_written": False,
            "private_pair_rows_written": False,
        },
        "semantic_diagnostics": semantic_diagnostics,
        "nested_cv": {
            "candidate_spec_count": len(specs),
            "outer_folds": outer_folds,
            "selected_spec_ids_by_fold": {
                fold_id: spec.spec_id for fold_id, spec in selected_specs.items()
            },
            "final_full_train_oof_selected_spec": asdict(final_spec),
            "final_full_train_oof_selected_spec_id": final_spec.spec_id,
            "final_full_train_oof_selected_spec_eligible": bool(final_spec_row["eligible"]),
            "final_full_train_oof_metrics": final_spec_row["metrics"].public_dict(),
            "final_full_train_oof_safe_fold_count": final_spec_row["safe_fold_count"],
            "all_inner_selected_specs_eligible": all_inner_selected_eligible,
            "oof_metrics": oof_metrics.public_dict(),
            "oof_quality_gates": oof_gates,
            "outer_fold_metrics": fold_metrics,
            "all_outer_folds_safety_passed": all_outer_safety_passed,
        },
        "stage172_comparison": {
            "stage172_oof_metrics": stage172_metrics,
            "stage173_oof_metrics": oof_metrics.public_dict(),
            "metric_delta": {
                name: round(getattr(oof_metrics, name) - float(stage172_metrics[name]), 6)
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
        "resource_consumption": resource_consumption,
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "candidate_replay": round(replay_ready_at - authorized_at, 6),
            "cross_encoder_load": round(model_ready_at - replay_ready_at, 6),
            "semantic_pair_build_and_score": round(scored_at - model_ready_at, 6),
            "nested_cv": round(cv_finished_at - scored_at, 6),
            "report_assembly": round(finished_at - cv_finished_at, 6),
        },
        "closed_boundaries": {
            "development_opened": False,
            "test_opened": False,
            "cross_encoder_fine_tuned": False,
            "answer_generation_run": False,
            "agent_turns_run": False,
            "retry_action_count": 0,
            "fallback_action_count": 0,
            "runtime_registered_as_default": False,
        },
        "process_guards": process_guards,
        "decision": {
            "candidate_selected": candidate_selected,
            "status": (
                "advance_to_stage174_train_only_semantic_evidence_runtime_e2e"
                if candidate_selected
                else "stage173_frozen_cross_encoder_semantics_insufficient"
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
        report["decision"]["status"] = "stage173_process_invalid"
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report


def write_stage173_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage173Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    nested = report["nested_cv"]
    metrics = nested["oof_metrics"]
    stage172_metrics = report["stage172_comparison"]["stage172_oof_metrics"]
    folds = nested["outer_fold_metrics"]
    resources = report["resource_consumption"]
    timings = report["timing_seconds"]
    profile_counts = Counter(
        row["selected_spec"]["feature_profile"] for row in nested["outer_folds"]
    )
    semantic = report["semantic_diagnostics"]
    charts = {
        "oof_quality_gates.svg": _chart(
            "Stage 173 grouped OOF quality gates",
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
        "stage172_stage173_rates.svg": _chart(
            "Stage 172 versus Stage 173 grouped OOF rates",
            tuple(
                BarDatum(f"172 {label}", stage172_metrics[name], f"{stage172_metrics[name]:.1%}")
                if index % 2 == 0
                else BarDatum(f"173 {label}", metrics[name], f"{metrics[name]:.1%}")
                for label, name in (
                    ("initial compose", "initial_visible_compose_rate"),
                    ("final compose", "alternate_only_final_compose_rate"),
                    ("exact path", "alternate_only_path_success_rate"),
                    ("false compose", "insufficient_final_compose_rate"),
                )
                for index in range(2)
            ),
            x_label="Rate",
        ),
        "outer_fold_safety.svg": _chart(
            "Stage 173 outer-fold insufficient final compose",
            tuple(
                _rate_bar(fold_id, fold["insufficient_final_compose_rate"])
                for fold_id, fold in folds.items()
            ),
            x_label="False-compose rate",
        ),
        "outer_fold_path.svg": _chart(
            "Stage 173 outer-fold alternate exact path",
            tuple(
                _rate_bar(fold_id, fold["alternate_only_path_success_rate"])
                for fold_id, fold in folds.items()
            ),
            x_label="Exact-path rate",
        ),
        "selected_feature_profiles.svg": _chart(
            "Stage 173 outer-fold selected feature profiles",
            tuple(
                BarDatum(
                    profile, profile_counts.get(profile, 0), str(profile_counts.get(profile, 0))
                )
                for profile in _FEATURE_PROFILES
            ),
            x_label="Selected outer folds",
        ),
        "semantic_diagnostics.svg": _chart(
            "Stage 173 frozen cross-encoder diagnostics",
            (
                _rate_bar("Pair-level ROC AUC", semantic["pair_level_roc_auc"]),
                _rate_bar("View-max ROC AUC", semantic["view_max_roc_auc"]),
                _rate_bar("Positive pair top1", semantic["positive_pair_top1_rate"]),
            ),
            x_label="Rate",
        ),
        "timing.svg": _chart(
            "Stage 173 phase wall times",
            tuple(
                BarDatum(name.replace("_", " "), value, f"{value:.2f} s")
                for name, value in timings.items()
            ),
            x_label="Seconds",
        ),
        "resources.svg": _chart(
            "Stage 173 process and GPU resource peaks",
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
        written.append(Stage173Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _run_nested_cv(
    *,
    cases: Sequence[stage172.EvidenceViewCase],
    specs: Sequence[SemanticModelSpec],
    progress_sink: ProgressSink | None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, float],
    dict[str, SemanticModelSpec],
]:
    folds = sorted({case.fold_id for case in cases})
    if len(folds) != 5:
        raise ValueError("Stage173 requires exactly five grouped folds")
    rows = []
    oof_predictions: dict[str, float] = {}
    selected_specs: dict[str, SemanticModelSpec] = {}
    profile_families = tuple(
        (profile, family) for profile in _FEATURE_PROFILES for family in _MODEL_FAMILIES
    )
    for index, outer_fold in enumerate(folds, start=1):
        outer_train = tuple(case for case in cases if case.fold_id != outer_fold)
        heldout = tuple(case for case in cases if case.fold_id == outer_fold)
        inner_predictions: dict[tuple[str, str], dict[str, float]] = {
            key: {} for key in profile_families
        }
        for profile, family in profile_families:
            model_spec = SemanticModelSpec(profile, family, 0.5)
            for inner_fold in sorted({case.fold_id for case in outer_train}):
                inner_train = tuple(case for case in outer_train if case.fold_id != inner_fold)
                inner_heldout = tuple(case for case in outer_train if case.fold_id == inner_fold)
                predictor = SemanticEvidencePredictor(model_spec)
                predictor.fit(inner_train)
                inner_predictions[(profile, family)].update(predictor.predict(inner_heldout))
        if any(len(predictions) != len(outer_train) for predictions in inner_predictions.values()):
            raise RuntimeError("Stage173 inner OOF prediction coverage is incomplete")
        spec_rows = [
            _spec_evaluation(
                outer_train,
                inner_predictions[(spec.feature_profile, spec.model_family)],
                spec,
            )
            for spec in specs
        ]
        eligible = [row for row in spec_rows if row["eligible"]]
        selected_row = max(spec_rows, key=_spec_selection_key)
        selected: SemanticModelSpec = selected_row["spec"]
        predictor = SemanticEvidencePredictor(selected)
        predictor.fit(outer_train)
        heldout_predictions = predictor.predict(heldout)
        oof_predictions.update(heldout_predictions)
        selected_specs[outer_fold] = selected
        heldout_metrics = stage172.evaluate_predictions(
            heldout,
            heldout_predictions,
            {outer_fold: selected},
        )
        rows.append(
            {
                "heldout_fold": outer_fold,
                "outer_train_case_count": len(outer_train),
                "heldout_case_count": len(heldout),
                "candidate_spec_count": len(specs),
                "inner_eligible_spec_count": len(eligible),
                "selected_spec": asdict(selected),
                "selected_spec_id": selected.spec_id,
                "selected_spec_inner_eligible": bool(selected_row["eligible"]),
                "selected_inner_metrics": selected_row["metrics"].public_dict(),
                "selected_inner_safe_fold_count": selected_row["safe_fold_count"],
                "heldout_metrics": heldout_metrics.public_dict(),
            }
        )
        _emit(progress_sink, phase="outer_fold_complete", completed=index, total=5)
    return rows, oof_predictions, selected_specs


def _select_full_train_spec(
    *,
    cases: Sequence[stage172.EvidenceViewCase],
    specs: Sequence[SemanticModelSpec],
) -> dict[str, Any]:
    profile_families = tuple(
        (profile, family) for profile in _FEATURE_PROFILES for family in _MODEL_FAMILIES
    )
    predictions_by_model: dict[tuple[str, str], dict[str, float]] = {
        key: {} for key in profile_families
    }
    folds = sorted({case.fold_id for case in cases})
    for profile, family in profile_families:
        model_spec = SemanticModelSpec(profile, family, 0.5)
        for heldout_fold in folds:
            train = tuple(case for case in cases if case.fold_id != heldout_fold)
            heldout = tuple(case for case in cases if case.fold_id == heldout_fold)
            predictor = SemanticEvidencePredictor(model_spec)
            predictor.fit(train)
            predictions_by_model[(profile, family)].update(predictor.predict(heldout))
    if any(len(predictions) != len(cases) for predictions in predictions_by_model.values()):
        raise RuntimeError("Stage173 full-train OOF prediction coverage is incomplete")
    rows = [
        _spec_evaluation(
            cases,
            predictions_by_model[(spec.feature_profile, spec.model_family)],
            spec,
        )
        for spec in specs
    ]
    return max(rows, key=_spec_selection_key)


def _spec_evaluation(
    cases: Sequence[stage172.EvidenceViewCase],
    predictions: Mapping[str, float],
    spec: SemanticModelSpec,
) -> dict[str, Any]:
    specs_by_fold = {fold_id: spec for fold_id in {case.fold_id for case in cases}}
    metrics = stage172.evaluate_predictions(cases, predictions, specs_by_fold)
    gates = stage172._quality_gates(metrics)
    fold_metrics = _fold_metrics_for_spec(cases, predictions, spec)
    safe_fold_count = sum(
        row.insufficient_final_compose_rate
        <= stage172._GATE_THRESHOLDS["insufficient_final_compose_rate_max"]
        for row in fold_metrics.values()
    )
    eligible = all(gate["passed"] for gate in gates) and safe_fold_count == len(fold_metrics)
    return {
        "spec": spec,
        "metrics": metrics,
        "gates": gates,
        "safe_fold_count": safe_fold_count,
        "fold_count": len(fold_metrics),
        "eligible": eligible,
    }


def _spec_selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics: stage172.EvidenceProxyMetrics = row["metrics"]
    gates = row["gates"]
    spec: SemanticModelSpec = row["spec"]
    return (
        int(row["safe_fold_count"] == row["fold_count"]),
        row["safe_fold_count"],
        int(
            metrics.insufficient_final_compose_rate
            <= stage172._GATE_THRESHOLDS["insufficient_final_compose_rate_max"]
        ),
        sum(gate["passed"] for gate in gates),
        metrics.alternate_only_path_success_rate,
        metrics.alternate_only_final_compose_rate,
        metrics.initial_visible_compose_rate,
        -metrics.insufficient_final_compose_rate,
        metrics.balanced_accuracy,
        int(spec.feature_profile == "semantic_only"),
        -abs(spec.threshold - 0.5),
        spec.spec_id,
    )


def _outer_fold_metrics(
    cases: Sequence[stage172.EvidenceViewCase],
    predictions: Mapping[str, float],
    specs_by_fold: Mapping[str, SemanticModelSpec],
) -> dict[str, dict[str, int | float]]:
    result = {}
    for fold_id in sorted(specs_by_fold):
        fold_cases = tuple(case for case in cases if case.fold_id == fold_id)
        result[fold_id] = stage172.evaluate_predictions(
            fold_cases,
            {case.private_identity: predictions[case.private_identity] for case in fold_cases},
            {fold_id: specs_by_fold[fold_id]},
        ).public_dict()
    return result


def _fold_metrics_for_spec(
    cases: Sequence[stage172.EvidenceViewCase],
    predictions: Mapping[str, float],
    spec: SemanticModelSpec,
) -> dict[str, stage172.EvidenceProxyMetrics]:
    result = {}
    for fold_id in sorted({case.fold_id for case in cases}):
        fold_cases = tuple(case for case in cases if case.fold_id == fold_id)
        result[fold_id] = stage172.evaluate_predictions(
            fold_cases,
            {case.private_identity: predictions[case.private_identity] for case in fold_cases},
            {fold_id: spec},
        )
    return result


def _semantic_diagnostics(
    cases: Sequence[stage172.EvidenceViewCase],
    pairs: Sequence[SemanticPairInput],
    pair_scores: Mapping[str, float],
) -> dict[str, Any]:
    pair_labels = np.asarray([pair.positive_label for pair in pairs], dtype=int)
    scores = np.asarray([pair_scores[pair.private_identity] for pair in pairs], dtype=float)
    case_labels = np.asarray([case.sufficient_label for case in cases], dtype=int)
    case_max_scores = np.asarray(
        [case.features["semantic_score_max"] for case in cases],
        dtype=float,
    )
    positive_pair_groups: dict[str, list[SemanticPairInput]] = {}
    for pair in pairs:
        group_identity = pair.private_identity.split(":", 1)[0]
        positive_pair_groups.setdefault(group_identity, []).append(pair)
    positive_groups = [
        rows for rows in positive_pair_groups.values() if any(row.positive_label for row in rows)
    ]
    top1_count = sum(
        max(rows, key=lambda row: pair_scores[row.private_identity]).positive_label
        for rows in positive_groups
    )
    return {
        "pair_level_roc_auc": round(float(roc_auc_score(pair_labels, scores)), 6),
        "view_max_roc_auc": round(float(roc_auc_score(case_labels, case_max_scores)), 6),
        "positive_pair_group_count": len(positive_groups),
        "positive_pair_top1_count": top1_count,
        "positive_pair_top1_rate": round(top1_count / len(positive_groups), 6),
        "positive_pair_score_distribution": _distribution(scores[pair_labels == 1].tolist()),
        "negative_pair_score_distribution": _distribution(scores[pair_labels == 0].tolist()),
        "sufficient_view_max_distribution": _distribution(
            case_max_scores[case_labels == 1].tolist()
        ),
        "insufficient_view_max_distribution": _distribution(
            case_max_scores[case_labels == 0].tolist()
        ),
    }


def _feature_matrix(
    cases: Sequence[stage172.EvidenceViewCase],
    feature_names: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [[float(case.features[name]) for name in feature_names] for case in cases],
        dtype=float,
    )


def _question_text(sample: PrimeQAHybridSplitSample) -> str:
    parts = [sample.question_title.strip(), sample.question_text.strip()]
    return "\n\n".join(part for part in parts if part)


def _pair_identity(sample_id: str, document_id: str) -> str:
    return f"{stage172._sha256_text(sample_id)}:{stage172._sha256_text(document_id)}"


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Stage173 distribution cannot be empty")
    ordered = np.asarray(sorted(values), dtype=float)
    return {
        "count": len(ordered),
        "minimum": round(float(ordered[0]), 6),
        "p25": round(float(np.quantile(ordered, 0.25)), 6),
        "median": round(float(np.median(ordered)), 6),
        "p75": round(float(np.quantile(ordered, 0.75)), 6),
        "maximum": round(float(ordered[-1]), 6),
        "mean": round(float(np.mean(ordered)), 6),
    }


def _resolved_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "resolved_size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _authorize_sources(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    for name, expected in _SOURCE_HASHES.items():
        if fingerprints[name]["sha256"] != expected:
            raise ValueError(f"Stage173 source hash mismatch: {name}")


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
