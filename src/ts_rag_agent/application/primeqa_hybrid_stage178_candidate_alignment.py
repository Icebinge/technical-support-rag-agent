from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ts_rag_agent.application import primeqa_hybrid_listwise_agent_e2e as stage178
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    PrimeQAHybridProcessRuntimeResourceFactory,
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
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    PrimeQAHybridSidecarObservationAdapter,
)
from ts_rag_agent.domain.dataset import PrimeQARuntimeQuery
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 178 alignment audit"
_ANALYSIS_ID = "stage178_stage161_replay_vs_stage128_live_alignment_v1"


@dataclass(frozen=True)
class CandidateAlignmentObservation:
    prefix_sequence_exact: bool
    prefix_set_exact: bool
    prefix_symmetric_difference_count: int
    original_rrf_top10_exact: bool
    query_overlap_top10_exact: bool
    union_sequence_exact: bool
    union_set_exact: bool
    live_union_missing_from_offline_count: int
    offline_original_gold_hit: bool
    live_original_gold_hit: bool
    offline_overlap_gold_hit: bool
    live_overlap_gold_hit: bool
    offline_union_gold_hit: bool
    live_union_gold_hit: bool
    live_retrieval_seconds: float


def run_stage178_candidate_alignment_audit(
    *,
    stage128_protocol_path: Path,
    stage125_protocol_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    documents_path: Path,
    encoder_batch_size: int = 64,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    source_paths = {
        "stage128": stage128_protocol_path,
        "stage125": stage125_protocol_path,
        "stage80": stage80_report_path,
        "train": train_split_path,
        "documents": documents_path,
    }
    fingerprints = {
        name: stage178.stage173._resolved_fingerprint(path) for name, path in source_paths.items()
    }
    mismatches = [
        name
        for name, fingerprint in fingerprints.items()
        if fingerprint["sha256"] != stage178._SOURCE_HASHES[name]
    ]
    if mismatches:
        raise ValueError(f"Stage178 alignment source authorization failed: {mismatches}")

    samples = load_primeqa_hybrid_split_samples(train_split_path)
    if len(samples) != stage178._EXPECTED_TRAIN_ROWS or any(
        sample.assigned_split != "train" for sample in samples
    ):
        raise ValueError("Stage178 alignment audit accepts only the exact train split")
    documents_by_id = load_primeqa_documents(documents_path)
    sections_by_document = load_primeqa_document_sections(documents_path)
    documents = list(documents_by_id.values())
    dense_channels, dense_summary = _build_dense_channels(
        include_dense_channels=True,
        stage80_report=stage178.stage173._load_json_object(stage80_report_path),
        stage80_report_path=stage80_report_path,
        documents=documents,
        document_ids=tuple(document.id for document in documents),
        encoder_batch_size=encoder_batch_size,
        encoder_device="cpu",
        encoder_factory=None,
    )
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
        fold_assignments=_build_train_fold_assignments(samples, fold_count=5),
    ).build(samples)
    replay_ready_at = time.perf_counter()
    runtime_factory = PrimeQAHybridProcessRuntimeResourceFactory(
        stage128_protocol_path=stage128_protocol_path,
        stage125_protocol_path=stage125_protocol_path,
        stage80_report_path=stage80_report_path,
        documents_path=documents_path,
        encoder_batch_size=encoder_batch_size,
        encoder_device="cpu",
    )
    resources = runtime_factory.build_shared()
    resources_ready_at = time.perf_counter()
    observations = audit_candidate_alignment(
        samples=samples,
        grouped_records=records_by_sample(records),
        candidate_pool_retriever=resources.candidate_pool_retriever,
    )
    finished_at = time.perf_counter()
    summary = summarize_candidate_alignment(observations)
    process_guards = [
        _guard("exact_train_rows", len(samples) == stage178._EXPECTED_TRAIN_ROWS),
        _guard("exact_replay_rows", len(records) == stage178._EXPECTED_CANDIDATE_ROWS),
        _guard("dense_channels_ready", dense_summary.get("status") == "dense_channels_ready"),
        _guard("one_runtime_factory_build", runtime_factory.build_count == 1),
        _guard("development_not_loaded", True),
        _guard("test_not_loaded", True),
        _guard("raw_identifiers_not_written", True),
    ]
    return {
        "stage": _STAGE,
        "analysis_id": _ANALYSIS_ID,
        "scope": "Train-only aggregate alignment audit; no model fitting or quality selection.",
        "source_authorization": fingerprints,
        "comparison": summary,
        "timing_seconds": {
            "stage161_replay_build": round(replay_ready_at - started_at, 6),
            "stage128_runtime_resource_build": round(resources_ready_at - replay_ready_at, 6),
            "paired_alignment_audit": round(finished_at - resources_ready_at, 6),
            "wall": round(finished_at - started_at, 6),
        },
        "execution_boundaries": {
            "train_loaded": True,
            "development_loaded": False,
            "test_loaded": False,
            "model_fit_count": 0,
            "answer_quality_metrics_computed": False,
            "raw_question_or_document_ids_written": False,
        },
        "process_guards": process_guards,
        "decision": {
            "full_prefix_contract_exact": summary["prefix_sequence_exact_count"] == len(samples),
            "selection_surface_exact": all(
                summary[key] == len(samples)
                for key in (
                    "original_rrf_top10_exact_count",
                    "query_overlap_top10_exact_count",
                    "union_sequence_exact_count",
                )
            ),
            "live_union_fully_covered_by_stage177_pairs": summary[
                "live_union_missing_from_offline_pair_count"
            ]
            == 0,
            "stage178_protocol_change_authorized": False,
        },
    }


def audit_candidate_alignment(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
    candidate_pool_retriever: Any,
) -> tuple[CandidateAlignmentObservation, ...]:
    adapter = PrimeQAHybridSidecarObservationAdapter()
    observations = []
    for sample in samples:
        records = tuple(
            sorted(grouped_records[sample.sample_id], key=lambda row: row.baseline_rank)
        )
        offline_prefix = tuple(record.document_id for record in records)
        offline_original = tuple(
            record.document_id for record in select_original_rrf_top10(records).selected
        )
        offline_overlap = tuple(
            record.document_id for record in select_current_query_overlap_top10(records).selected
        )
        started_at = time.perf_counter()
        live = tuple(
            candidate_pool_retriever.retrieve(
                PrimeQARuntimeQuery(
                    id=sample.sample_id,
                    title=sample.question_title,
                    text=sample.question_text,
                )
            )
        )
        live_seconds = time.perf_counter() - started_at
        live_prefix = tuple(result.document.id for result in live[: len(offline_prefix)])
        live_original = tuple(result.document.id for result in live[:10])
        live_overlap = tuple(
            result.document.id
            for result in adapter.observe(
                question=sample.to_primeqa_question(),
                candidate_pool_results=live,
            ).answer_context_results
        )
        offline_union = _ordered_union(offline_overlap, offline_original)
        live_union = _ordered_union(live_overlap, live_original)
        gold = sample.answer_doc_id if sample.answerable else None
        observations.append(
            CandidateAlignmentObservation(
                prefix_sequence_exact=offline_prefix == live_prefix,
                prefix_set_exact=set(offline_prefix) == set(live_prefix),
                prefix_symmetric_difference_count=len(
                    set(offline_prefix).symmetric_difference(live_prefix)
                ),
                original_rrf_top10_exact=offline_original == live_original,
                query_overlap_top10_exact=offline_overlap == live_overlap,
                union_sequence_exact=offline_union == live_union,
                union_set_exact=set(offline_union) == set(live_union),
                live_union_missing_from_offline_count=len(
                    set(live_union).difference(offline_union)
                ),
                offline_original_gold_hit=gold is not None and gold in offline_original,
                live_original_gold_hit=gold is not None and gold in live_original,
                offline_overlap_gold_hit=gold is not None and gold in offline_overlap,
                live_overlap_gold_hit=gold is not None and gold in live_overlap,
                offline_union_gold_hit=gold is not None and gold in offline_union,
                live_union_gold_hit=gold is not None and gold in live_union,
                live_retrieval_seconds=live_seconds,
            )
        )
    return tuple(observations)


def summarize_candidate_alignment(
    observations: Sequence[CandidateAlignmentObservation],
) -> dict[str, Any]:
    if not observations:
        raise ValueError("Stage178 alignment audit requires observations")
    result: dict[str, Any] = {"question_count": len(observations)}
    for field in (
        "prefix_sequence_exact",
        "prefix_set_exact",
        "original_rrf_top10_exact",
        "query_overlap_top10_exact",
        "union_sequence_exact",
        "union_set_exact",
    ):
        result[f"{field}_count"] = sum(getattr(row, field) for row in observations)
    differences = np.asarray(
        [row.prefix_symmetric_difference_count for row in observations], dtype=float
    )
    latencies = np.asarray([row.live_retrieval_seconds for row in observations], dtype=float)
    result["prefix_symmetric_difference"] = _distribution(differences)
    result["live_retrieval_seconds"] = _distribution(latencies)
    result["live_union_missing_from_offline_question_count"] = sum(
        row.live_union_missing_from_offline_count > 0 for row in observations
    )
    result["live_union_missing_from_offline_pair_count"] = sum(
        row.live_union_missing_from_offline_count for row in observations
    )
    for view in ("original", "overlap", "union"):
        offline = [getattr(row, f"offline_{view}_gold_hit") for row in observations]
        live = [getattr(row, f"live_{view}_gold_hit") for row in observations]
        result[f"{view}_gold_hit"] = {
            "offline_count": sum(offline),
            "live_count": sum(live),
            "live_gain_count": sum(
                right and not left for left, right in zip(offline, live, strict=True)
            ),
            "live_loss_count": sum(
                left and not right for left, right in zip(offline, live, strict=True)
            ),
        }
    return result


def _ordered_union(*views: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(document_id for view in views for document_id in view))


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean": round(float(values.mean()), 6),
        "median": round(float(np.median(values)), 6),
        "p95": round(float(np.quantile(values, 0.95)), 6),
        "maximum": round(float(values.max()), 6),
    }


def _guard(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}
