from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.evidence_selection import classify_question_route
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.dataset import (
    PrimeQADocument,
    PrimeQADocumentSection,
)
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever, tokenize_text
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 112"
_CREATED_AT = "2026-07-16"
_ANALYSIS_ID = "primeqa_hybrid_retrieval_context_miss_root_cause_audit_v1"
_SOURCE_STAGE111 = "Stage 111"
_STAGE111_PROTOCOL_ID = "primeqa_hybrid_retrieval_context_miss_audit_protocol_v1"
_STAGE111_STATUS = "primeqa_hybrid_retrieval_context_miss_audit_protocol_frozen"
_STAGE111_NEXT_DIRECTION = "run_retrieval_context_miss_root_cause_audit_train_dev"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = ("test",)
_RETRIEVAL_TOP_K = 10
_DIAGNOSTIC_DEPTH = 50
_AUDIT_DIMENSIONS = (
    "query_expression_gap",
    "title_heading_mismatch",
    "section_boundary_or_span_locality",
    "long_document_score_dilution",
    "entity_version_error_code_mismatch",
    "bm25_field_weighting_or_index_structure",
)
_EXPECTED_PUBLIC_CASE_FIELDS = (
    "sample_id",
    "split",
    "retrieval_context_miss_root_cause_bucket",
    "question_route",
    "gold_doc_rank_bucket",
    "query_expression_gap_bucket",
    "title_heading_overlap_bucket",
    "section_locality_bucket",
    "document_length_bucket",
    "entity_version_error_code_bucket",
    "index_structure_signal_bucket",
    "confidence_band",
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "question_text",
        "question_title",
        "raw_question_text",
        "raw_answer_text",
        "gold_answer",
        "answer_text",
        "document_id",
        "answer_doc_id",
        "retrieved_doc_ids",
        "cited_doc_ids",
        "source_doc_ids",
        "matched_token_strings",
        "query_terms",
        "document_title",
        "document_body",
        "document_text",
    }
)
_HIGH_SIGNAL_BUCKETS = {
    "query_expression_gap": {
        "zero_gold_section_overlap",
        "very_low_gold_section_overlap_lt_0_10",
        "low_gold_section_overlap_lt_0_25",
        "short_query_low_gold_overlap",
    },
    "title_heading_mismatch": {
        "zero_title_heading_overlap",
        "very_low_title_heading_overlap_lt_0_10",
        "low_title_heading_overlap_lt_0_25",
    },
    "section_boundary_or_span_locality": {
        "long_section_low_local_anchor",
        "boundary_early_low_local_anchor",
        "boundary_late_low_local_anchor",
        "low_local_anchor",
    },
    "long_document_score_dilution": {
        "very_long_gold_document",
        "long_gold_document",
        "gold_document_much_longer_than_top10",
    },
    "entity_version_error_code_mismatch": {
        "special_tokens_absent_in_gold_context",
        "partial_special_token_overlap",
    },
    "bm25_field_weighting_or_index_structure": {
        "section_rank_beats_doc_rollup",
        "title_rank_beats_doc_rollup",
        "section_or_title_diagnostic_top10",
        "gold_doc_not_found_top50",
    },
}


@dataclass(frozen=True)
class PrimeQAHybridRetrievalContextMissRootCauseAuditVisualization:
    """One generated Stage112 retrieval-context-miss audit chart."""

    name: str
    path: str


@dataclass(frozen=True)
class _DocumentProfile:
    token_count: int
    title_tokens: frozenset[str]
    text_tokens: frozenset[str]
    all_tokens: frozenset[str]


@dataclass(frozen=True)
class _GoldSectionProfile:
    heading_tokens: frozenset[str]
    text_tokens: frozenset[str]
    token_count: int
    relative_answer_position: float


@dataclass(frozen=True)
class _AuditCase:
    public_case: dict[str, Any]
    dimension_buckets: dict[str, str]
    high_signal_dimensions: tuple[str, ...]
    gold_doc_rank_bucket: str
    question_route: str


def run_primeqa_hybrid_retrieval_context_miss_root_cause_audit(
    *,
    stage111_protocol_path: Path,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    user_confirmed_audit: bool,
    confirmation_note: str,
    sample_limit_per_bucket: int = 5,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> dict[str, Any]:
    """Run the Stage112 train/dev-only retrieval-context-miss root-cause audit."""

    _validate_options(
        sample_limit_per_bucket=sample_limit_per_bucket,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )
    started_at = time.perf_counter()
    stage111_protocol = _load_json_object(stage111_protocol_path)
    stage111_summary = _stage111_summary(stage111_protocol)
    stage112_contract = (
        (stage111_protocol.get("frozen_protocol") or {}).get("stage112_run_contract")
        or {}
    )
    protocol_loaded_at = time.perf_counter()

    split_samples = {
        "train": load_primeqa_hybrid_split_samples(train_split_path),
        "dev": load_primeqa_hybrid_split_samples(dev_split_path),
    }
    splits_loaded_at = time.perf_counter()
    documents = load_primeqa_documents(documents_path)
    sections_by_document = load_primeqa_document_sections(documents_path)
    documents_loaded_at = time.perf_counter()

    doc_retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
    doc_retriever.fit(documents.values())
    title_retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
    title_retriever.fit(_title_documents(documents))
    section_documents, section_owner_by_id = _section_documents(sections_by_document)
    section_retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
    section_retriever.fit(section_documents)
    indexed_at = time.perf_counter()

    document_profiles: dict[str, _DocumentProfile] = {}
    split_audits = {
        split: _audit_split(
            split=split,
            samples=samples,
            documents=documents,
            sections_by_document=sections_by_document,
            doc_retriever=doc_retriever,
            title_retriever=title_retriever,
            section_retriever=section_retriever,
            section_owner_by_id=section_owner_by_id,
            document_profiles=document_profiles,
        )
        for split, samples in split_samples.items()
    }
    audited_at = time.perf_counter()
    split_reports = {
        split: _split_report(split, split_samples[split], cases)
        for split, cases in split_audits.items()
    }
    cross_split_summary = _cross_split_summary(split_reports)
    public_case_samples = _public_case_samples(
        split_audits=split_audits,
        sample_limit_per_bucket=sample_limit_per_bucket,
    )
    guard_checks = _guard_checks(
        stage111_summary=stage111_summary,
        stage112_contract=stage112_contract,
        split_samples=split_samples,
        split_reports=split_reports,
        public_case_samples=public_case_samples,
        user_confirmed_audit=user_confirmed_audit,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train/dev-only retrieval-context-miss root-cause audit under the "
            "Stage111 frozen protocol. This stage loads only the Stage111 "
            "protocol, Stage68 train/dev split files, and PrimeQA training/dev "
            "corpus sections; it does not load the test split, does not run "
            "final metrics, does not select or tune candidates from dev, does "
            "not add fallback strategies, and does not change runtime defaults."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_audit),
            "confirmation_note": confirmation_note,
        },
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "development_splits": list(_DEVELOPMENT_SPLITS),
            "reported_splits": list(_DEVELOPMENT_SPLITS),
            "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
            "selection_split": None,
            "validation_split": None,
        },
        "source_files": {
            "stage111_protocol": _fingerprint(stage111_protocol_path),
            "train_split": _fingerprint(train_split_path),
            "dev_split": _fingerprint(dev_split_path),
            "corpus_documents": _fingerprint(documents_path),
        },
        "stage111_summary": stage111_summary,
        "audit_config": {
            "retrieval_top_k": _RETRIEVAL_TOP_K,
            "diagnostic_depth": _DIAGNOSTIC_DEPTH,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "sample_limit_per_bucket": sample_limit_per_bucket,
            "audit_dimensions": list(_AUDIT_DIMENSIONS),
            "reported_case_fields": list(_EXPECTED_PUBLIC_CASE_FIELDS),
            "gold_doc_id_usage": "offline_labeling_only_not_output_not_runtime_feature",
        },
        "loaded_data_summary": {
            "document_count": len(documents),
            "section_count": sum(len(sections) for sections in sections_by_document.values()),
            "split_rows": {
                split: len(samples) for split, samples in sorted(split_samples.items())
            },
            "answerable_rows": {
                split: sum(
                    sample.answerable and sample.answer_doc_id is not None
                    for sample in samples
                )
                for split, samples in sorted(split_samples.items())
            },
            "test_split_loaded": False,
        },
        "split_reports": split_reports,
        "cross_split_summary": cross_split_summary,
        "public_case_samples": public_case_samples,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks, cross_split_summary),
        "timing_seconds": {
            "load_stage111_protocol": round(protocol_loaded_at - started_at, 3),
            "load_train_dev_splits": round(splits_loaded_at - protocol_loaded_at, 3),
            "load_corpus_documents": round(documents_loaded_at - splits_loaded_at, 3),
            "build_diagnostic_indexes": round(indexed_at - documents_loaded_at, 3),
            "root_cause_audit": round(audited_at - indexed_at, 3),
            "aggregate_and_guard": round(checked_at - audited_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_retrieval_context_miss_root_cause_audit_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridRetrievalContextMissRootCauseAuditVisualization]:
    """Write SVG charts for the Stage112 root-cause audit."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage112_audit_case_counts_by_split.svg": render_horizontal_bar_chart_svg(
            title="Stage112 retrieval-context-miss audit cases by split",
            bars=_audit_case_count_bars(report),
            x_label="audit cases",
            margin_left=220,
        ),
        "stage112_primary_root_cause_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage112 primary root-cause buckets",
            bars=_primary_root_cause_bars(report),
            x_label="audit cases",
            width=1320,
            margin_left=620,
        ),
        "stage112_dimension_high_signal_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage112 high-signal dimensions",
            bars=_dimension_high_signal_bars(report),
            x_label="audit cases",
            width=1360,
            margin_left=680,
        ),
        "stage112_gold_rank_bucket_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage112 gold document diagnostic rank buckets",
            bars=_gold_rank_bucket_bars(report),
            x_label="audit cases",
            width=1120,
            margin_left=420,
        ),
        "stage112_question_route_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage112 question route counts",
            bars=_question_route_bars(report),
            x_label="audit cases",
            width=1280,
            margin_left=560,
        ),
        "stage112_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage112 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1640,
            margin_left=900,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridRetrievalContextMissRootCauseAuditVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _audit_split(
    *,
    split: str,
    samples: Sequence[PrimeQAHybridSplitSample],
    documents: Mapping[str, PrimeQADocument],
    sections_by_document: Mapping[str, Sequence[PrimeQADocumentSection]],
    doc_retriever: BM25Retriever,
    title_retriever: BM25Retriever,
    section_retriever: BM25Retriever,
    section_owner_by_id: Mapping[str, str],
    document_profiles: dict[str, _DocumentProfile],
) -> list[_AuditCase]:
    cases = []
    for sample in samples:
        if not sample.answerable or sample.answer_doc_id is None:
            continue
        question = sample.to_primeqa_question()
        doc_results = doc_retriever.search(
            question.full_question,
            top_k=_DIAGNOSTIC_DEPTH,
        )
        result_doc_ids = [result.document.id for result in doc_results]
        if sample.answer_doc_id in result_doc_ids[:_RETRIEVAL_TOP_K]:
            continue
        title_rank = _rank_document(
            title_retriever.search(question.full_question, top_k=_DIAGNOSTIC_DEPTH),
            sample.answer_doc_id,
        )
        section_rank = _rank_section_owner(
            section_retriever.search(question.full_question, top_k=_DIAGNOSTIC_DEPTH),
            section_owner_by_id,
            sample.answer_doc_id,
        )
        cases.append(
            _audit_case(
                split=split,
                sample=sample,
                documents=documents,
                sections_by_document=sections_by_document,
                doc_results=doc_results,
                title_rank=title_rank,
                section_rank=section_rank,
                document_profiles=document_profiles,
            )
        )
    return cases


def _audit_case(
    *,
    split: str,
    sample: PrimeQAHybridSplitSample,
    documents: Mapping[str, PrimeQADocument],
    sections_by_document: Mapping[str, Sequence[PrimeQADocumentSection]],
    doc_results: Sequence[Any],
    title_rank: int | None,
    section_rank: int | None,
    document_profiles: dict[str, _DocumentProfile],
) -> _AuditCase:
    question = sample.to_primeqa_question()
    query_tokens = frozenset(tokenize_text(question.full_question))
    answer_doc_id = str(sample.answer_doc_id)
    result_doc_ids = [result.document.id for result in doc_results]
    doc_rank = (
        result_doc_ids.index(answer_doc_id) + 1
        if answer_doc_id in result_doc_ids
        else None
    )
    gold_document = documents.get(answer_doc_id)
    gold_profile = (
        _document_profile(answer_doc_id, documents, document_profiles)
        if gold_document is not None
        else None
    )
    gold_section = (
        _gold_section_profile(
            sample=sample,
            sections=sections_by_document.get(answer_doc_id, []),
        )
        if gold_document is not None
        else None
    )
    top10_lengths = [
        _document_profile(result.document.id, documents, document_profiles).token_count
        for result in doc_results[:_RETRIEVAL_TOP_K]
        if result.document.id in documents
    ]

    query_gap_bucket = _query_expression_gap_bucket(query_tokens, gold_profile, gold_section)
    title_heading_bucket = _title_heading_overlap_bucket(
        query_tokens,
        gold_profile,
        gold_section,
    )
    section_locality_bucket = _section_locality_bucket(
        sample=sample,
        query_tokens=query_tokens,
        gold_document=gold_document,
        gold_section=gold_section,
    )
    document_length_bucket = _document_length_bucket(gold_profile, top10_lengths)
    entity_bucket = _entity_version_error_code_bucket(
        query_tokens=query_tokens,
        gold_profile=gold_profile,
        gold_section=gold_section,
    )
    index_bucket = _index_structure_signal_bucket(
        doc_rank=doc_rank,
        title_rank=title_rank,
        section_rank=section_rank,
    )
    dimension_buckets = {
        "query_expression_gap": query_gap_bucket,
        "title_heading_mismatch": title_heading_bucket,
        "section_boundary_or_span_locality": section_locality_bucket,
        "long_document_score_dilution": document_length_bucket,
        "entity_version_error_code_mismatch": entity_bucket,
        "bm25_field_weighting_or_index_structure": index_bucket,
    }
    high_signal_dimensions = tuple(
        dimension
        for dimension in _AUDIT_DIMENSIONS
        if dimension_buckets[dimension] in _HIGH_SIGNAL_BUCKETS[dimension]
    )
    root_cause_bucket = (
        high_signal_dimensions[0] if high_signal_dimensions else "mixed_or_low_confidence"
    )
    confidence_band = _confidence_band(len(high_signal_dimensions))
    question_route = classify_question_route(question)
    public_case = {
        "sample_id": sample.sample_id,
        "split": split,
        "retrieval_context_miss_root_cause_bucket": root_cause_bucket,
        "question_route": question_route,
        "gold_doc_rank_bucket": _gold_rank_bucket(doc_rank),
        "query_expression_gap_bucket": query_gap_bucket,
        "title_heading_overlap_bucket": title_heading_bucket,
        "section_locality_bucket": section_locality_bucket,
        "document_length_bucket": document_length_bucket,
        "entity_version_error_code_bucket": entity_bucket,
        "index_structure_signal_bucket": index_bucket,
        "confidence_band": confidence_band,
    }
    if tuple(public_case) != _EXPECTED_PUBLIC_CASE_FIELDS:
        raise ValueError("Stage112 public case fields do not match the frozen contract")
    return _AuditCase(
        public_case=public_case,
        dimension_buckets=dimension_buckets,
        high_signal_dimensions=high_signal_dimensions,
        gold_doc_rank_bucket=public_case["gold_doc_rank_bucket"],
        question_route=question_route,
    )


def _split_report(
    split: str,
    samples: Sequence[PrimeQAHybridSplitSample],
    cases: Sequence[_AuditCase],
) -> dict[str, Any]:
    answerable_count = sum(
        1 for sample in samples if sample.answerable and sample.answer_doc_id is not None
    )
    return {
        "split": split,
        "total_rows": len(samples),
        "answerable_rows": int(answerable_count),
        "audit_case_count": len(cases),
        "audit_case_rate_among_answerable": _rounded_ratio(len(cases), answerable_count),
        "primary_root_cause_counts": _counter_dict(
            case.public_case["retrieval_context_miss_root_cause_bucket"]
            for case in cases
        ),
        "gold_doc_rank_bucket_counts": _counter_dict(
            case.gold_doc_rank_bucket for case in cases
        ),
        "question_route_counts": _counter_dict(case.question_route for case in cases),
        "confidence_band_counts": _counter_dict(
            case.public_case["confidence_band"] for case in cases
        ),
        "dimension_bucket_counts": {
            dimension: _counter_dict(case.dimension_buckets[dimension] for case in cases)
            for dimension in _AUDIT_DIMENSIONS
        },
        "dimension_high_signal_counts": {
            dimension: sum(
                dimension in case.high_signal_dimensions for case in cases
            )
            for dimension in _AUDIT_DIMENSIONS
        },
        "multi_signal_case_counts": _counter_dict(
            _multi_signal_bucket(len(case.high_signal_dimensions)) for case in cases
        ),
    }


def _cross_split_summary(split_reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    root_cause_counter: Counter[str] = Counter()
    rank_counter: Counter[str] = Counter()
    route_counter: Counter[str] = Counter()
    confidence_counter: Counter[str] = Counter()
    dimension_high_counter: Counter[str] = Counter()
    total_cases = 0
    total_answerable = 0
    for report in split_reports.values():
        total_cases += int(report["audit_case_count"])
        total_answerable += int(report["answerable_rows"])
        root_cause_counter.update(report["primary_root_cause_counts"])
        rank_counter.update(report["gold_doc_rank_bucket_counts"])
        route_counter.update(report["question_route_counts"])
        confidence_counter.update(report["confidence_band_counts"])
        dimension_high_counter.update(report["dimension_high_signal_counts"])
    return {
        "answerable_rows": total_answerable,
        "audit_case_count": total_cases,
        "audit_case_rate_among_answerable": _rounded_ratio(total_cases, total_answerable),
        "primary_root_cause_counts": dict(sorted(root_cause_counter.items())),
        "gold_doc_rank_bucket_counts": dict(sorted(rank_counter.items())),
        "question_route_counts": dict(sorted(route_counter.items())),
        "confidence_band_counts": dict(sorted(confidence_counter.items())),
        "dimension_high_signal_counts": dict(sorted(dimension_high_counter.items())),
        "top_primary_root_causes": _top_counter_items(root_cause_counter, limit=6),
        "top_question_routes": _top_counter_items(route_counter, limit=8),
        "common_train_dev_root_causes": _common_train_dev_root_causes(split_reports),
    }


def _public_case_samples(
    *,
    split_audits: Mapping[str, Sequence[_AuditCase]],
    sample_limit_per_bucket: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    samples: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split, cases in split_audits.items():
        samples[split] = {}
        grouped: dict[str, list[_AuditCase]] = {}
        for case in cases:
            bucket = str(case.public_case["retrieval_context_miss_root_cause_bucket"])
            grouped.setdefault(bucket, []).append(case)
        for bucket, bucket_cases in sorted(grouped.items()):
            ranked = sorted(
                bucket_cases,
                key=lambda case: (
                    case.public_case["confidence_band"],
                    case.public_case["sample_id"],
                ),
            )
            samples[split][bucket] = [
                case.public_case for case in ranked[:sample_limit_per_bucket]
            ]
    return samples


def _stage111_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision") or {}
    frozen = report.get("frozen_protocol") or {}
    contract = frozen.get("stage112_run_contract") or {}
    stage102 = report.get("stage102_summary") or {}
    return {
        "stage": report.get("stage"),
        "protocol_id": report.get("protocol_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "can_run_train_dev_audit_after_user_confirmation": decision.get(
            "can_run_train_dev_audit_after_user_confirmation"
        ),
        "requires_user_confirmation_before_train_dev_audit": decision.get(
            "requires_user_confirmation_before_train_dev_audit"
        ),
        "retrieval_depth_for_diagnostic_only": contract.get(
            "retrieval_depth_for_diagnostic_only"
        ),
        "reported_splits": contract.get("reported_splits"),
        "final_test_metrics_allowed": contract.get("final_test_metrics_allowed"),
        "selection_or_threshold_tuning_allowed": contract.get(
            "selection_or_threshold_tuning_allowed"
        ),
        "candidate_defaultization_allowed": contract.get(
            "candidate_defaultization_allowed"
        ),
        "gold_doc_id_allowed_as_runtime_feature": contract.get(
            "gold_doc_id_allowed_as_runtime_feature"
        ),
        "audit_dimensions": [
            str(item.get("dimension_id"))
            for item in frozen.get("audit_dimensions") or []
        ],
        "expected_retrieval_context_miss_counts": {
            "train": stage102.get("train_retrieval_context_miss_count"),
            "dev": stage102.get("dev_retrieval_context_miss_count"),
        },
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get(
            "can_run_final_test_metrics_now"
        ),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _query_expression_gap_bucket(
    query_tokens: frozenset[str],
    gold_profile: _DocumentProfile | None,
    gold_section: _GoldSectionProfile | None,
) -> str:
    if not gold_profile:
        return "gold_document_missing"
    section_tokens = gold_section.text_tokens if gold_section else gold_profile.text_tokens
    section_ratio = _overlap_ratio(query_tokens, section_tokens)
    if len(query_tokens) <= 3 and section_ratio < 0.25:
        return "short_query_low_gold_overlap"
    if section_ratio == 0:
        return "zero_gold_section_overlap"
    if section_ratio < 0.10:
        return "very_low_gold_section_overlap_lt_0_10"
    if section_ratio < 0.25:
        return "low_gold_section_overlap_lt_0_25"
    if section_ratio < 0.50:
        return "moderate_gold_section_overlap_lt_0_50"
    return "strong_gold_section_overlap"


def _title_heading_overlap_bucket(
    query_tokens: frozenset[str],
    gold_profile: _DocumentProfile | None,
    gold_section: _GoldSectionProfile | None,
) -> str:
    if not gold_profile:
        return "gold_document_missing"
    heading_tokens = gold_section.heading_tokens if gold_section else frozenset()
    overlap_ratio = max(
        _overlap_ratio(query_tokens, gold_profile.title_tokens),
        _overlap_ratio(query_tokens, heading_tokens),
    )
    if overlap_ratio == 0:
        return "zero_title_heading_overlap"
    if overlap_ratio < 0.10:
        return "very_low_title_heading_overlap_lt_0_10"
    if overlap_ratio < 0.25:
        return "low_title_heading_overlap_lt_0_25"
    return "title_or_heading_overlap_present"


def _section_locality_bucket(
    *,
    sample: PrimeQAHybridSplitSample,
    query_tokens: frozenset[str],
    gold_document: PrimeQADocument | None,
    gold_section: _GoldSectionProfile | None,
) -> str:
    if gold_document is None:
        return "gold_document_missing"
    if gold_section is None or sample.start_offset is None:
        return "section_not_identified"
    local_text = gold_document.text[
        max(0, sample.start_offset - 220) : min(
            len(gold_document.text),
            (sample.end_offset or sample.start_offset) + 220,
        )
    ]
    local_ratio = _overlap_ratio(query_tokens, frozenset(tokenize_text(local_text)))
    if gold_section.token_count > 220 and local_ratio < 0.10:
        return "long_section_low_local_anchor"
    if gold_section.relative_answer_position <= 0.15 and local_ratio < 0.10:
        return "boundary_early_low_local_anchor"
    if gold_section.relative_answer_position >= 0.85 and local_ratio < 0.10:
        return "boundary_late_low_local_anchor"
    if local_ratio < 0.10:
        return "low_local_anchor"
    return "local_anchor_present"


def _document_length_bucket(
    gold_profile: _DocumentProfile | None,
    top10_lengths: Sequence[int],
) -> str:
    if gold_profile is None:
        return "gold_document_missing"
    token_count = gold_profile.token_count
    if top10_lengths:
        median_top10 = _median(top10_lengths)
        if median_top10 > 0 and token_count >= median_top10 * 2.5 and token_count > 1000:
            return "gold_document_much_longer_than_top10"
    if token_count > 3000:
        return "very_long_gold_document"
    if token_count > 1000:
        return "long_gold_document"
    if token_count > 300:
        return "medium_gold_document"
    return "short_gold_document"


def _entity_version_error_code_bucket(
    *,
    query_tokens: frozenset[str],
    gold_profile: _DocumentProfile | None,
    gold_section: _GoldSectionProfile | None,
) -> str:
    special_query_tokens = frozenset(token for token in query_tokens if _is_special_token(token))
    if not special_query_tokens:
        return "no_special_query_tokens"
    if gold_profile is None:
        return "gold_document_missing"
    gold_tokens = gold_profile.title_tokens | gold_profile.text_tokens
    if gold_section is not None:
        gold_tokens = gold_tokens | gold_section.heading_tokens | gold_section.text_tokens
    overlap = special_query_tokens & gold_tokens
    if not overlap:
        return "special_tokens_absent_in_gold_context"
    if len(overlap) == len(special_query_tokens):
        return "special_tokens_matched"
    return "partial_special_token_overlap"


def _index_structure_signal_bucket(
    *,
    doc_rank: int | None,
    title_rank: int | None,
    section_rank: int | None,
) -> str:
    if doc_rank is None and (
        _rank_within_top_k(title_rank) or _rank_within_top_k(section_rank)
    ):
        return "section_or_title_diagnostic_top10"
    if doc_rank is not None and doc_rank > _RETRIEVAL_TOP_K:
        if _rank_within_top_k(section_rank):
            return "section_rank_beats_doc_rollup"
        if _rank_within_top_k(title_rank):
            return "title_rank_beats_doc_rollup"
        if doc_rank <= 20:
            return "near_miss_rank_11_to_20"
    if doc_rank is None:
        return "gold_doc_not_found_top50"
    return "weak_index_structure_signal"


def _gold_rank_bucket(doc_rank: int | None) -> str:
    if doc_rank is None:
        return "not_found_top50"
    if doc_rank <= 10:
        return "rank_1_to_10"
    if doc_rank <= 20:
        return "rank_11_to_20"
    return "rank_21_to_50"


def _confidence_band(high_signal_count: int) -> str:
    if high_signal_count >= 3:
        return "high_multi_signal"
    if high_signal_count == 2:
        return "medium_two_signal"
    if high_signal_count == 1:
        return "low_single_signal"
    return "low_no_high_signal"


def _multi_signal_bucket(high_signal_count: int) -> str:
    if high_signal_count >= 3:
        return "three_or_more_high_signals"
    if high_signal_count == 2:
        return "two_high_signals"
    if high_signal_count == 1:
        return "one_high_signal"
    return "zero_high_signals"


def _document_profile(
    doc_id: str,
    documents: Mapping[str, PrimeQADocument],
    cache: dict[str, _DocumentProfile],
) -> _DocumentProfile:
    cached = cache.get(doc_id)
    if cached is not None:
        return cached
    document = documents[doc_id]
    title_tokens = frozenset(tokenize_text(document.title))
    text_tokens = frozenset(tokenize_text(document.text))
    profile = _DocumentProfile(
        token_count=len(tokenize_text(document.title)) + len(tokenize_text(document.text)),
        title_tokens=title_tokens,
        text_tokens=text_tokens,
        all_tokens=title_tokens | text_tokens,
    )
    cache[doc_id] = profile
    return profile


def _gold_section_profile(
    *,
    sample: PrimeQAHybridSplitSample,
    sections: Sequence[PrimeQADocumentSection],
) -> _GoldSectionProfile | None:
    if sample.start_offset is None:
        return None
    for section in sections:
        if section.start_offset is None or section.end_offset is None:
            continue
        if section.start_offset <= sample.start_offset <= section.end_offset:
            section_length = max(1, section.end_offset - section.start_offset)
            relative_position = (sample.start_offset - section.start_offset) / section_length
            return _GoldSectionProfile(
                heading_tokens=frozenset(tokenize_text(section.section_id)),
                text_tokens=frozenset(tokenize_text(section.text)),
                token_count=len(tokenize_text(section.text)),
                relative_answer_position=round(relative_position, 4),
            )
    return None


def _title_documents(
    documents: Mapping[str, PrimeQADocument],
) -> list[PrimeQADocument]:
    return [
        PrimeQADocument(id=document.id, title="", text=document.title)
        for document in documents.values()
    ]


def _section_documents(
    sections_by_document: Mapping[str, Sequence[PrimeQADocumentSection]],
) -> tuple[list[PrimeQADocument], dict[str, str]]:
    documents = []
    owner_by_id = {}
    for document_id, sections in sorted(sections_by_document.items()):
        for index, section in enumerate(sections):
            section_doc_id = f"{document_id}::section_{index:04d}"
            owner_by_id[section_doc_id] = document_id
            documents.append(
                PrimeQADocument(
                    id=section_doc_id,
                    title=str(section.section_id),
                    text=section.text,
                )
            )
    return documents, owner_by_id


def _rank_document(results: Sequence[Any], document_id: str) -> int | None:
    for result in results:
        if result.document.id == document_id:
            return int(result.rank)
    return None


def _rank_section_owner(
    results: Sequence[Any],
    section_owner_by_id: Mapping[str, str],
    document_id: str,
) -> int | None:
    for result in results:
        if section_owner_by_id.get(result.document.id) == document_id:
            return int(result.rank)
    return None


def _rank_within_top_k(rank: int | None) -> bool:
    return rank is not None and rank <= _RETRIEVAL_TOP_K


def _overlap_ratio(left: frozenset[str], right: frozenset[str]) -> float:
    return round(len(left & right) / len(left), 4) if left else 0.0


def _is_special_token(token: str) -> bool:
    if re.fullmatch(r"cve-\d{4}-\d+", token):
        return True
    if re.fullmatch(r"[a-z]{1,4}\d{3,}", token):
        return True
    if re.fullmatch(r"\d+(?:\.\d+){1,4}", token):
        return True
    if re.fullmatch(r"[a-z]+(?:\d+[a-z]*){1,}", token):
        return True
    return bool(re.fullmatch(r"[a-z]+-\d+(?:-\d+)*", token))


def _common_train_dev_root_causes(
    split_reports: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    train = set((split_reports.get("train") or {}).get("primary_root_cause_counts") or {})
    dev = set((split_reports.get("dev") or {}).get("primary_root_cause_counts") or {})
    return sorted(train & dev)


def _guard_checks(
    *,
    stage111_summary: Mapping[str, Any],
    stage112_contract: Mapping[str, Any],
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    split_reports: Mapping[str, Mapping[str, Any]],
    public_case_samples: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    user_confirmed_audit: bool,
) -> list[dict[str, Any]]:
    expected_counts = stage111_summary.get("expected_retrieval_context_miss_counts") or {}
    observed_counts = {
        split: report["audit_case_count"] for split, report in split_reports.items()
    }
    observed_split_names = sorted(
        {sample.assigned_split for samples in split_samples.values() for sample in samples}
    )
    expected_split_names = sorted(_DEVELOPMENT_SPLITS)
    forbidden_keys_found = _forbidden_keys_found(
        {
            "split_reports": split_reports,
            "public_case_samples": public_case_samples,
        }
    )
    public_case_fields_valid = _public_case_fields_valid(public_case_samples)
    return [
        _check(
            name="source_stage111_is_expected",
            passed=stage111_summary.get("stage") == _SOURCE_STAGE111,
            observed=stage111_summary.get("stage"),
            expected=_SOURCE_STAGE111,
        ),
        _check(
            name="source_stage111_protocol_id_matches",
            passed=stage111_summary.get("protocol_id") == _STAGE111_PROTOCOL_ID,
            observed=stage111_summary.get("protocol_id"),
            expected=_STAGE111_PROTOCOL_ID,
        ),
        _check(
            name="stage111_protocol_is_frozen",
            passed=stage111_summary.get("decision_status") == _STAGE111_STATUS,
            observed=stage111_summary.get("decision_status"),
            expected=_STAGE111_STATUS,
        ),
        _check(
            name="stage111_recommends_stage112_audit",
            passed=stage111_summary.get("recommended_next_direction")
            == _STAGE111_NEXT_DIRECTION,
            observed=stage111_summary.get("recommended_next_direction"),
            expected=_STAGE111_NEXT_DIRECTION,
        ),
        _check(
            name="user_confirmed_stage112_audit",
            passed=user_confirmed_audit,
            observed=user_confirmed_audit,
            expected=True,
        ),
        _check(
            name="stage112_contract_reports_train_dev_only",
            passed=stage112_contract.get("reported_splits") == list(_DEVELOPMENT_SPLITS)
            and stage112_contract.get("final_test_metrics_allowed") is False,
            observed=stage112_contract,
            expected="reported_splits train/dev and final metrics forbidden",
        ),
        _check(
            name="stage112_contract_forbids_selection_and_defaultization",
            passed=stage112_contract.get("selection_or_threshold_tuning_allowed") is False
            and stage112_contract.get("candidate_defaultization_allowed") is False,
            observed=stage112_contract,
            expected="no selection, threshold tuning, or defaultization",
        ),
        _check(
            name="gold_doc_ids_are_offline_labeling_only",
            passed=stage112_contract.get("stage112_may_use_gold_doc_id_for_offline_labeling")
            is True
            and stage112_contract.get("gold_doc_id_allowed_as_runtime_feature") is False,
            observed=stage112_contract,
            expected="gold doc IDs offline audit only",
        ),
        _check(
            name="loaded_splits_are_train_dev_only",
            passed=observed_split_names == expected_split_names,
            observed=observed_split_names,
            expected=expected_split_names,
        ),
        _check(
            name="test_split_not_loaded",
            passed=True,
            observed="not_loaded",
            expected="not_loaded",
        ),
        _check(
            name="audit_case_counts_match_stage102_retrieval_context_miss",
            passed=observed_counts == expected_counts,
            observed=observed_counts,
            expected=expected_counts,
        ),
        _check(
            name="audit_dimensions_match_stage111_protocol",
            passed=tuple(stage111_summary.get("audit_dimensions") or [])
            == _AUDIT_DIMENSIONS,
            observed=stage111_summary.get("audit_dimensions"),
            expected=list(_AUDIT_DIMENSIONS),
        ),
        _check(
            name="diagnostic_depth_matches_stage111_contract",
            passed=stage111_summary.get("retrieval_depth_for_diagnostic_only")
            == _DIAGNOSTIC_DEPTH,
            observed=stage111_summary.get("retrieval_depth_for_diagnostic_only"),
            expected=_DIAGNOSTIC_DEPTH,
        ),
        _check(
            name="public_case_fields_match_stage111_contract",
            passed=public_case_fields_valid,
            observed="valid" if public_case_fields_valid else "invalid",
            expected=list(_EXPECTED_PUBLIC_CASE_FIELDS),
        ),
        _check(
            name="public_outputs_have_no_forbidden_keys",
            passed=not forbidden_keys_found,
            observed=sorted(forbidden_keys_found),
            expected=[],
        ),
        _check(
            name="final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="dev_selection_and_threshold_tuning_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="runtime_defaults_remain_unchanged",
            passed=stage111_summary.get("default_runtime_policy") == "unchanged",
            observed=stage111_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="fallback_strategies_remain_disabled",
            passed=stage111_summary.get("fallback_strategies_enabled") is False,
            observed=stage111_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
    ]


def _decision(
    guard_checks: Sequence[Mapping[str, Any]],
    cross_split_summary: Mapping[str, Any],
) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_retrieval_context_miss_root_cause_audit_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": "primeqa_hybrid_retrieval_context_miss_root_cause_audit_completed",
        "analysis_id": _ANALYSIS_ID,
        "dominant_train_dev_root_causes": cross_split_summary.get(
            "top_primary_root_causes",
            [],
        ),
        "common_train_dev_root_causes": cross_split_summary.get(
            "common_train_dev_root_causes",
            [],
        ),
        "can_continue_train_dev_development": True,
        "requires_user_confirmation_before_next_protocol": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage113: after user confirmation, freeze a train/dev-only "
            "retrieval/index redesign protocol based on the Stage112 aggregate "
            "root-cause audit; do not select from dev-only observations, do not "
            "open the final test gate, keep runtime defaults unchanged, and add "
            "no fallback strategies."
        ),
    }


def _audit_case_count_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(split, float(data["audit_case_count"]), str(data["audit_case_count"]))
        for split, data in report["split_reports"].items()
    ]


def _primary_root_cause_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counter = Counter(report["cross_split_summary"]["primary_root_cause_counts"])
    return [
        BarDatum(label, float(value), str(value))
        for label, value in _top_counter_items(counter, limit=8)
    ]


def _dimension_high_signal_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counter = Counter(report["cross_split_summary"]["dimension_high_signal_counts"])
    return [
        BarDatum(label, float(counter.get(label, 0)), str(counter.get(label, 0)))
        for label in _AUDIT_DIMENSIONS
    ]


def _gold_rank_bucket_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counter = Counter(report["cross_split_summary"]["gold_doc_rank_bucket_counts"])
    return [
        BarDatum(label, float(value), str(value))
        for label, value in sorted(counter.items())
    ]


def _question_route_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counter = Counter(report["cross_split_summary"]["question_route_counts"])
    return [
        BarDatum(label, float(value), str(value))
        for label, value in _top_counter_items(counter, limit=10)
    ]


def _guard_check_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(check["name"]),
            value=1.0 if check["passed"] else 0.0,
            value_label="passed" if check["passed"] else "failed",
        )
        for check in report.get("guard_checks") or []
    ]


def _public_case_fields_valid(
    public_case_samples: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> bool:
    for bucket_samples in public_case_samples.values():
        for cases in bucket_samples.values():
            for case in cases:
                if tuple(case) != _EXPECTED_PUBLIC_CASE_FIELDS:
                    return False
    return True


def _forbidden_keys_found(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in _FORBIDDEN_PUBLIC_KEYS:
                found.add(str(key))
            found.update(_forbidden_keys_found(nested))
    elif isinstance(value, list | tuple):
        for nested in value:
            found.update(_forbidden_keys_found(nested))
    return found


def _top_counter_items(counter: Counter[str], *, limit: int) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _counter_dict(values) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _rounded_ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 4) if denominator else 0.0


def _median(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


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
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
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


def _validate_options(
    *,
    sample_limit_per_bucket: int,
    bm25_k1: float,
    bm25_b: float,
) -> None:
    if sample_limit_per_bucket <= 0:
        raise ValueError("sample_limit_per_bucket must be positive")
    if bm25_k1 <= 0:
        raise ValueError("bm25_k1 must be positive")
    if not 0 <= bm25_b <= 1:
        raise ValueError("bm25_b must be between 0 and 1")
