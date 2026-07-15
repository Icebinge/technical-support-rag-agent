from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQADocumentSection
from ts_rag_agent.infrastructure.bm25_retriever import tokenize_text
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 92"
_CREATED_AT = "2026-07-15"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_PROTOCOL_ID = "section_signal_guarded_expansion_train_dev_v1"
_CANDIDATE_ID = "section_signal_guarded_expansion_design"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})
_BASELINE_CONFIG_ID = "full_document_bm25_baseline"
_BASELINE_SECTION_SIGNAL_BUCKET = "full_document_bm25"
_PRIMARY_TOP_K = 10
_DEFAULT_SEARCH_DEPTH = 50
_BASELINE_K1 = 1.5
_BASELINE_B = 0.75
_EXPECTED_CONFIG_IDS = (
    "ssgx_shadow_no_top10_demotion_v1",
    "ssgx_rank11_20_margin_guard_v1",
    "ssgx_rank21_50_high_confidence_v1",
    "ssgx_section_top50_injection_guard_v1",
)
_EPSILON = 1e-9


@dataclass(frozen=True)
class PrimeQAHybridSectionSignalComparisonVisualization:
    """One generated Stage92 section-signal comparison visualization."""

    name: str
    path: str


@dataclass(frozen=True)
class SectionSignalConfig:
    """One frozen section-signal guarded expansion configuration."""

    config_id: str
    promotion_mode: str
    eligible_baseline_rank_min: int
    eligible_baseline_rank_max: int | None
    section_rank_max: int
    minimum_section_to_document_score_ratio: float
    maximum_document_score_margin_to_rank10: float | None
    maximum_top10_promotions_per_query: int
    protected_bm25_top_rank_count: int
    demote_existing_bm25_top10: bool


@dataclass(frozen=True)
class _RankedDocument:
    document_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class _DocumentScoreView:
    ranked: tuple[_RankedDocument, ...]
    rank_by_doc_id: dict[str, int]
    score_by_doc_id: dict[str, float]


@dataclass(frozen=True)
class _SectionSignalEvidence:
    baseline_rank: int | None
    section_rank: int | None
    document_score: float
    section_score: float
    score_ratio: float
    score_margin_to_rank10: float | None
    section_signal_bucket: str
    baseline_rank_bucket: str
    section_rank_bucket: str
    score_ratio_bucket: str
    score_margin_bucket: str
    promotion_reason_code: str
    top10_protection_action: str


class _BM25DocumentScoreIndex:
    """Vectorized full-document BM25 index that exposes public-safe scores."""

    def __init__(self, documents: Sequence[PrimeQADocument]) -> None:
        self._documents = list(documents)
        self._doc_ids = np.asarray([document.id for document in self._documents])
        self._doc_lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = {}
        for doc_index, document in enumerate(self._documents):
            tokens = tokenize_text(f"{document.title}\n\n{document.text}")
            self._doc_lengths.append(len(tokens))
            for term, term_frequency in Counter(tokens).items():
                postings.setdefault(term, []).append((doc_index, term_frequency))

        document_count = len(self._documents)
        total_length = sum(self._doc_lengths)
        self._avg_doc_length = total_length / document_count if document_count else 0.0
        self._doc_lengths_array = np.asarray(self._doc_lengths, dtype=np.float64)
        self._postings = {
            term: (
                np.asarray([row[0] for row in rows], dtype=np.int64),
                np.asarray([row[1] for row in rows], dtype=np.float64),
            )
            for term, rows in postings.items()
        }
        self._idf = {
            term: _compute_idf(document_count, len(rows))
            for term, rows in postings.items()
        }

    def score_query(
        self,
        *,
        query_terms: Sequence[str],
        search_depth: int,
    ) -> _DocumentScoreView:
        if search_depth <= 0:
            raise ValueError("search_depth must be positive")
        if not query_terms or not self._documents:
            return _DocumentScoreView(ranked=(), rank_by_doc_id={}, score_by_doc_id={})

        scores = np.zeros(len(self._documents), dtype=np.float64)
        for term in query_terms:
            idf = self._idf.get(term)
            posting = self._postings.get(term)
            if idf is None or posting is None:
                continue
            doc_indices, term_frequencies = posting
            doc_lengths = self._doc_lengths_array[doc_indices]
            scores[doc_indices] += _score_term_vector(
                idf=idf,
                term_frequencies=term_frequencies,
                doc_lengths=doc_lengths,
                avg_length=self._avg_doc_length,
                k1=_BASELINE_K1,
                b=_BASELINE_B,
            )

        scored_indices = np.flatnonzero(scores)
        if scored_indices.size == 0:
            return _DocumentScoreView(ranked=(), rank_by_doc_id={}, score_by_doc_id={})
        score_by_doc_id = {
            str(self._doc_ids[index]): float(scores[index])
            for index in scored_indices
        }
        ranked_indices = sorted(
            (int(index) for index in scored_indices),
            key=lambda index: (-float(scores[index]), str(self._doc_ids[index])),
        )[:search_depth]
        ranked = tuple(
            _RankedDocument(
                document_id=str(self._doc_ids[index]),
                rank=rank,
                score=float(scores[index]),
            )
            for rank, index in enumerate(ranked_indices, start=1)
        )
        return _DocumentScoreView(
            ranked=ranked,
            rank_by_doc_id={row.document_id: row.rank for row in ranked},
            score_by_doc_id=score_by_doc_id,
        )


class _SectionBM25ParentScoreIndex:
    """Vectorized section BM25 index rolled up to parent documents."""

    def __init__(
        self,
        documents: Sequence[PrimeQADocument],
        sections_by_document: Mapping[str, Sequence[PrimeQADocumentSection]],
    ) -> None:
        self._document_by_id = {document.id: document for document in documents}
        self._section_parent_doc_ids: list[str] = []
        self._section_lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = {}
        for document in documents:
            for section in sections_by_document.get(document.id, []):
                if not section.text.strip():
                    continue
                section_index = len(self._section_parent_doc_ids)
                self._section_parent_doc_ids.append(document.id)
                tokens = tokenize_text(
                    f"{document.title}\n\n{section.section_id}\n\n{section.text}"
                )
                self._section_lengths.append(len(tokens))
                for term, term_frequency in Counter(tokens).items():
                    postings.setdefault(term, []).append((section_index, term_frequency))

        section_count = len(self._section_parent_doc_ids)
        total_length = sum(self._section_lengths)
        self._avg_section_length = total_length / section_count if section_count else 0.0
        self._section_lengths_array = np.asarray(self._section_lengths, dtype=np.float64)
        self._postings = {
            term: (
                np.asarray([row[0] for row in rows], dtype=np.int64),
                np.asarray([row[1] for row in rows], dtype=np.float64),
            )
            for term, rows in postings.items()
        }
        self._idf = {
            term: _compute_idf(section_count, len(rows))
            for term, rows in postings.items()
        }

    def score_query(
        self,
        *,
        query_terms: Sequence[str],
        search_depth: int,
    ) -> _DocumentScoreView:
        if search_depth <= 0:
            raise ValueError("search_depth must be positive")
        if not query_terms or not self._section_parent_doc_ids:
            return _DocumentScoreView(ranked=(), rank_by_doc_id={}, score_by_doc_id={})

        scores = np.zeros(len(self._section_parent_doc_ids), dtype=np.float64)
        for term in query_terms:
            idf = self._idf.get(term)
            posting = self._postings.get(term)
            if idf is None or posting is None:
                continue
            section_indices, term_frequencies = posting
            section_lengths = self._section_lengths_array[section_indices]
            scores[section_indices] += _score_term_vector(
                idf=idf,
                term_frequencies=term_frequencies,
                doc_lengths=section_lengths,
                avg_length=self._avg_section_length,
                k1=_BASELINE_K1,
                b=_BASELINE_B,
            )

        scored_indices = np.flatnonzero(scores)
        if scored_indices.size == 0:
            return _DocumentScoreView(ranked=(), rank_by_doc_id={}, score_by_doc_id={})
        parent_scores: dict[str, float] = {}
        for section_index in scored_indices:
            document_id = self._section_parent_doc_ids[int(section_index)]
            score = float(scores[int(section_index)])
            current_score = parent_scores.get(document_id)
            if current_score is None or score > current_score:
                parent_scores[document_id] = score
        ranked_doc_ids = sorted(
            parent_scores,
            key=lambda document_id: (-parent_scores[document_id], document_id),
        )[:search_depth]
        ranked = tuple(
            _RankedDocument(
                document_id=document_id,
                rank=rank,
                score=parent_scores[document_id],
            )
            for rank, document_id in enumerate(ranked_doc_ids, start=1)
        )
        return _DocumentScoreView(
            ranked=ranked,
            rank_by_doc_id={row.document_id: row.rank for row in ranked},
            score_by_doc_id=parent_scores,
        )


def run_primeqa_hybrid_section_signal_comparison(
    *,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    stage75_report_path: Path,
    stage91_report_path: Path,
    user_confirmed_protocol: bool,
    confirmed_protocol_id: str,
    confirmation_note: str,
    top_k_values: tuple[int, ...] = (1, 5, 10),
    search_depth: int = _DEFAULT_SEARCH_DEPTH,
) -> dict[str, Any]:
    """Run the confirmed train/dev-only section signal comparison."""

    _validate_options(top_k_values=top_k_values, search_depth=search_depth)
    started_at = time.perf_counter()
    split_samples = {
        _TRAIN_SPLIT: load_primeqa_hybrid_split_samples(train_split_path),
        _DEV_SPLIT: load_primeqa_hybrid_split_samples(dev_split_path),
    }
    loaded_splits_at = time.perf_counter()
    documents = load_primeqa_documents(documents_path)
    sections_by_document = load_primeqa_document_sections(documents_path)
    document_list = list(documents.values())
    stage75_report = _load_json_object(stage75_report_path)
    stage91_report = _load_json_object(stage91_report_path)
    loaded_inputs_at = time.perf_counter()

    protocol = stage91_report.get("frozen_protocol") or {}
    protocol_configs = _configs_from_protocol(protocol)
    should_evaluate = (
        user_confirmed_protocol
        and confirmed_protocol_id == _PROTOCOL_ID
        and protocol.get("protocol_id") == _PROTOCOL_ID
        and _stage91_allows_metric_run(stage91_report)
    )
    candidate_configs = protocol_configs if should_evaluate else ()
    rank_tables: dict[str, dict[str, dict[str, Any]]] = {}
    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    train_selection: dict[str, Any] = {
        "selection_rule": _selection_rule_description(),
        "selected_config_id": None,
        "candidate_config_count": len(protocol_configs),
        "selected_train_metrics": None,
        "selected_dev_metrics": None,
        "selected_dev_comparison_to_baseline": None,
    }
    indexed_and_evaluated_at = loaded_inputs_at
    if candidate_configs:
        document_index = _BM25DocumentScoreIndex(document_list)
        section_index = _SectionBM25ParentScoreIndex(
            document_list,
            sections_by_document,
        )
        rank_tables = {
            split: _evaluate_split(
                split=split,
                samples=samples,
                document_index=document_index,
                section_index=section_index,
                candidate_configs=candidate_configs,
                top_k_values=top_k_values,
                search_depth=search_depth,
            )
            for split, samples in split_samples.items()
        }
        indexed_and_evaluated_at = time.perf_counter()
        comparisons = {
            split: {
                config.config_id: _compare_to_baseline(
                    split=split,
                    baseline=rank_tables[split][_BASELINE_CONFIG_ID],
                    challenger=rank_tables[split][config.config_id],
                    max_k=_PRIMARY_TOP_K,
                    search_depth=search_depth,
                )
                for config in candidate_configs
            }
            for split in _ALLOWED_DEVELOPMENT_SPLITS
        }
        train_selection = _select_config_on_train(
            rank_tables=rank_tables,
            comparisons=comparisons,
            candidate_configs=candidate_configs,
        )
        selected_config_id = str(train_selection["selected_config_id"])
        train_selection = {
            **train_selection,
            "selected_dev_metrics": _public_config_metrics(
                rank_tables[_DEV_SPLIT][selected_config_id]
            ),
            "selected_dev_comparison_to_baseline": comparisons[_DEV_SPLIT][
                selected_config_id
            ],
        }

    section_summary = _section_summary(sections_by_document)
    guard_checks = _guard_checks(
        split_samples=split_samples,
        rank_tables=rank_tables,
        section_summary=section_summary,
        stage75_report=stage75_report,
        stage91_report=stage91_report,
        protocol=protocol,
        protocol_configs=protocol_configs,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmed_protocol_id=confirmed_protocol_id,
        top_k_values=top_k_values,
        search_depth=search_depth,
    )
    checked_at = time.perf_counter()
    public_metrics = {
        split: {
            config_id: _public_config_metrics(config_result)
            for config_id, config_result in split_results.items()
        }
        for split, split_results in rank_tables.items()
    }
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "Train/dev-only comparison for frozen protocol "
            "section_signal_guarded_expansion_train_dev_v1. This stage uses "
            "the user-confirmed Stage91 protocol, selects a guarded section "
            "signal config on train only, validates on dev, keeps the frozen "
            "test split locked, does not run final metrics, does not use "
            "source DOC_IDS as runtime retrieval evidence, does not write raw "
            "question, answer, document, or section text, and does not change "
            "runtime defaults."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_protocol),
            "confirmed_protocol_id": confirmed_protocol_id,
            "confirmation_note": confirmation_note,
        },
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
            "forbidden_final_splits": sorted(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "train_split": _fingerprint(train_split_path),
            "dev_split": _fingerprint(dev_split_path),
            "documents": _fingerprint(documents_path),
            "stage75_report": _fingerprint(stage75_report_path),
            "stage91_report": _fingerprint(stage91_report_path),
        },
        "stage91_decision": stage91_report.get("decision") or {},
        "config": {
            "protocol_id": _PROTOCOL_ID,
            "candidate_id": _CANDIDATE_ID,
            "baseline_config_id": _BASELINE_CONFIG_ID,
            "baseline_k1": _BASELINE_K1,
            "baseline_b": _BASELINE_B,
            "top_k_values": list(top_k_values),
            "primary_top_k": _PRIMARY_TOP_K,
            "search_depth": search_depth,
            "selection_rule": _selection_rule_description(),
            "raw_section_or_document_text_written_to_report": False,
        },
        "loaded_data_summary": {
            "document_count": len(document_list),
            **section_summary,
            "split_rows": {
                split: len(samples) for split, samples in sorted(split_samples.items())
            },
            "answerable_rows": {
                split: sum(sample.answerable for sample in samples)
                for split, samples in sorted(split_samples.items())
            },
            "test_split_loaded": False,
        },
        "candidate_configs": [
            _public_candidate_config(config) for config in protocol_configs
        ],
        "metrics_by_split": public_metrics,
        "comparisons_to_baseline": comparisons,
        "train_selection": train_selection,
        "guard_checks": guard_checks,
        "decision": _decision(
            guard_checks=guard_checks,
            comparisons=comparisons,
            train_selection=train_selection,
            candidate_configs=candidate_configs,
        ),
        "timing_seconds": {
            "load_splits": round(loaded_splits_at - started_at, 3),
            "load_documents_sections_and_reports": round(
                loaded_inputs_at - loaded_splits_at,
                3,
            ),
            "bm25_section_signal_evaluate": round(
                indexed_and_evaluated_at - loaded_inputs_at,
                3,
            ),
            "guard_checks": round(checked_at - indexed_and_evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_section_signal_comparison_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridSectionSignalComparisonVisualization]:
    """Write SVG charts for Stage92 section signal comparison."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage92_section_signal_train_hit_at_10.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage92 section signal train hit@10",
                bars=_hit_at_k_bars(report, split=_TRAIN_SPLIT, top_k=_PRIMARY_TOP_K),
                x_label="hit@10",
                width=1360,
                margin_left=540,
            )
        ),
        "stage92_section_signal_dev_hit_at_10.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage92 section signal dev hit@10",
                bars=_hit_at_k_bars(report, split=_DEV_SPLIT, top_k=_PRIMARY_TOP_K),
                x_label="hit@10",
                width=1360,
                margin_left=540,
            )
        ),
        "stage92_section_signal_dev_delta_hit_at_10.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage92 section signal dev hit@10 delta vs baseline",
                bars=_delta_bars(report, split=_DEV_SPLIT, metric="hit@10_delta"),
                x_label="delta hit@10",
                width=1360,
                margin_left=540,
            )
        ),
        "stage92_section_signal_dev_search_depth_net.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage92 section signal dev search-depth net changes",
                bars=_search_depth_net_bars(report, split=_DEV_SPLIT),
                x_label="net changed cases",
                width=1360,
                margin_left=540,
            )
        ),
        "stage92_section_signal_dev_top10_changes.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage92 section signal dev top10 improvements minus regressions",
                bars=_top10_net_bars(report, split=_DEV_SPLIT),
                x_label="net changed cases",
                width=1360,
                margin_left=540,
            )
        ),
        "stage92_section_signal_guard_check_status.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage92 section signal guard checks",
                bars=_guard_check_bars(report),
                x_label="1 means passed",
                width=1360,
                margin_left=640,
            )
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridSectionSignalComparisonVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _configs_from_protocol(protocol: Mapping[str, Any]) -> tuple[SectionSignalConfig, ...]:
    configs = []
    for row in protocol.get("candidate_config_grid") or []:
        if not isinstance(row, Mapping):
            continue
        configs.append(
            SectionSignalConfig(
                config_id=str(row.get("config_id") or ""),
                promotion_mode=str(row.get("promotion_mode") or ""),
                eligible_baseline_rank_min=int(
                    row.get("eligible_baseline_rank_min") or 1
                ),
                eligible_baseline_rank_max=_optional_int(
                    row.get("eligible_baseline_rank_max")
                ),
                section_rank_max=int(row.get("section_rank_max") or 0),
                minimum_section_to_document_score_ratio=float(
                    row.get("minimum_section_to_document_score_ratio") or 0.0
                ),
                maximum_document_score_margin_to_rank10=_optional_float(
                    row.get("maximum_document_score_margin_to_rank10")
                ),
                maximum_top10_promotions_per_query=int(
                    row.get("maximum_top10_promotions_per_query") or 0
                ),
                protected_bm25_top_rank_count=int(
                    row.get("protected_bm25_top_rank_count") or 0
                ),
                demote_existing_bm25_top10=bool(
                    row.get("demote_existing_bm25_top10")
                ),
            )
        )
    return tuple(configs)


def _evaluate_split(
    *,
    split: str,
    samples: Sequence[PrimeQAHybridSplitSample],
    document_index: _BM25DocumentScoreIndex,
    section_index: _SectionBM25ParentScoreIndex,
    candidate_configs: Sequence[SectionSignalConfig],
    top_k_values: tuple[int, ...],
    search_depth: int,
) -> dict[str, dict[str, Any]]:
    answerable_samples = [
        sample
        for sample in samples
        if sample.answerable and sample.answer_doc_id is not None
    ]
    baseline_accumulator = _empty_accumulator(
        split=split,
        config_id=_BASELINE_CONFIG_ID,
        total_questions=len(samples),
    )
    candidate_accumulators = {
        config.config_id: _empty_accumulator(
            split=split,
            config_id=config.config_id,
            total_questions=len(samples),
        )
        for config in candidate_configs
    }
    for sample in answerable_samples:
        query = sample.to_primeqa_question().full_question
        query_terms = tokenize_text(query)
        document_view = document_index.score_query(
            query_terms=query_terms,
            search_depth=search_depth,
        )
        section_view = section_index.score_query(
            query_terms=query_terms,
            search_depth=search_depth,
        )
        baseline_ranked_doc_ids = [row.document_id for row in document_view.ranked]
        baseline_rank = _answer_rank(
            ranked_doc_ids=baseline_ranked_doc_ids,
            answer_doc_id=str(sample.answer_doc_id),
        )
        _record_rank(
            accumulator=baseline_accumulator,
            sample=sample,
            rank=baseline_rank,
            query_token_count=len(query_terms),
            signal_evidence=_baseline_signal_evidence(baseline_rank),
            top_k_values=top_k_values,
            search_depth=search_depth,
        )
        for config in candidate_configs:
            challenger_doc_ids, signal_evidence, action_summary = _apply_section_signal_gate(
                document_view=document_view,
                section_view=section_view,
                config=config,
                search_depth=search_depth,
            )
            challenger_rank = _answer_rank(
                ranked_doc_ids=challenger_doc_ids,
                answer_doc_id=str(sample.answer_doc_id),
            )
            _record_rank(
                accumulator=candidate_accumulators[config.config_id],
                sample=sample,
                rank=challenger_rank,
                query_token_count=len(query_terms),
                signal_evidence=signal_evidence.get(
                    str(sample.answer_doc_id),
                    _default_signal_evidence(
                        document_view=document_view,
                        section_view=section_view,
                        document_id=str(sample.answer_doc_id),
                    ),
                ),
                promotion_applied=bool(action_summary["promotion_applied"]),
                top10_demotion_applied=bool(action_summary["top10_demotion_applied"]),
                top_k_values=top_k_values,
                search_depth=search_depth,
            )
    results = {
        _BASELINE_CONFIG_ID: _finalize_accumulator(
            baseline_accumulator,
            top_k_values=top_k_values,
            search_depth=search_depth,
        )
    }
    results.update(
        {
            config_id: _finalize_accumulator(
                accumulator,
                top_k_values=top_k_values,
                search_depth=search_depth,
            )
            for config_id, accumulator in candidate_accumulators.items()
        }
    )
    return results


def _apply_section_signal_gate(
    *,
    document_view: _DocumentScoreView,
    section_view: _DocumentScoreView,
    config: SectionSignalConfig,
    search_depth: int,
) -> tuple[list[str], dict[str, _SectionSignalEvidence], dict[str, bool]]:
    baseline_doc_ids = [row.document_id for row in document_view.ranked]
    section_doc_ids = [row.document_id for row in section_view.ranked]
    union_doc_ids = _union_preserving_order(baseline_doc_ids, section_doc_ids)
    rank10_score = _rank10_score(document_view)
    evidence = {
        document_id: _section_signal_evidence(
            document_id=document_id,
            document_view=document_view,
            section_view=section_view,
            rank10_score=rank10_score,
            selected_for_promotion=False,
            top10_protection_action="not_promoted",
        )
        for document_id in union_doc_ids
    }
    eligible_doc_ids = [
        document_id
        for document_id in union_doc_ids
        if _is_eligible_for_config(
            document_id=document_id,
            document_view=document_view,
            section_view=section_view,
            config=config,
            search_depth=search_depth,
            rank10_score=rank10_score,
        )
    ]

    if config.promotion_mode == "shadow_after_top10":
        top10 = baseline_doc_ids[:_PRIMARY_TOP_K]
        after_top10 = baseline_doc_ids[_PRIMARY_TOP_K:]
        eligible_after_top10 = [
            document_id
            for document_id in sorted(
                eligible_doc_ids,
                key=lambda doc_id: (
                    section_view.rank_by_doc_id.get(doc_id, search_depth + 1),
                    -_score_ratio(document_view, section_view, doc_id),
                    doc_id,
                ),
            )
            if document_id not in top10
        ]
        ranked_doc_ids = (
            top10
            + eligible_after_top10
            + [doc_id for doc_id in after_top10 if doc_id not in eligible_after_top10]
        )[:search_depth]
        for document_id in eligible_after_top10:
            evidence[document_id] = _section_signal_evidence(
                document_id=document_id,
                document_view=document_view,
                section_view=section_view,
                rank10_score=rank10_score,
                selected_for_promotion=True,
                top10_protection_action="shadow_after_top10",
            )
        return ranked_doc_ids, evidence, {
            "promotion_applied": bool(eligible_after_top10),
            "top10_demotion_applied": False,
        }

    promoted_doc_id = _select_promoted_document(
        eligible_doc_ids=eligible_doc_ids,
        document_view=document_view,
        section_view=section_view,
    )
    if promoted_doc_id is None or config.maximum_top10_promotions_per_query <= 0:
        return baseline_doc_ids[:search_depth], evidence, {
            "promotion_applied": False,
            "top10_demotion_applied": False,
        }

    protected_prefix_count = max(_PRIMARY_TOP_K - 1, config.protected_bm25_top_rank_count)
    protected_prefix = [
        doc_id for doc_id in baseline_doc_ids[:protected_prefix_count]
        if doc_id != promoted_doc_id
    ][:_PRIMARY_TOP_K - 1]
    ranked_doc_ids = (
        protected_prefix
        + [promoted_doc_id]
        + [doc_id for doc_id in baseline_doc_ids if doc_id != promoted_doc_id]
        + [doc_id for doc_id in section_doc_ids if doc_id != promoted_doc_id]
    )
    ranked_doc_ids = _dedupe(ranked_doc_ids)[:search_depth]
    evidence[promoted_doc_id] = _section_signal_evidence(
        document_id=promoted_doc_id,
        document_view=document_view,
        section_view=section_view,
        rank10_score=rank10_score,
        selected_for_promotion=True,
        top10_protection_action="single_rank10_promotion",
    )
    return ranked_doc_ids, evidence, {
        "promotion_applied": True,
        "top10_demotion_applied": True,
    }


def _is_eligible_for_config(
    *,
    document_id: str,
    document_view: _DocumentScoreView,
    section_view: _DocumentScoreView,
    config: SectionSignalConfig,
    search_depth: int,
    rank10_score: float | None,
) -> bool:
    baseline_rank = document_view.rank_by_doc_id.get(document_id)
    section_rank = section_view.rank_by_doc_id.get(document_id)
    if section_rank is None or section_rank > config.section_rank_max:
        return False
    effective_baseline_rank = baseline_rank if baseline_rank is not None else search_depth + 1
    if effective_baseline_rank < config.eligible_baseline_rank_min:
        return False
    if (
        config.eligible_baseline_rank_max is not None
        and effective_baseline_rank > config.eligible_baseline_rank_max
    ):
        return False
    if _score_ratio(document_view, section_view, document_id) < (
        config.minimum_section_to_document_score_ratio
    ):
        return False
    if config.maximum_document_score_margin_to_rank10 is not None:
        if rank10_score is None:
            return False
        document_score = document_view.score_by_doc_id.get(document_id, 0.0)
        if rank10_score - document_score > config.maximum_document_score_margin_to_rank10:
            return False
    return True


def _select_promoted_document(
    *,
    eligible_doc_ids: Sequence[str],
    document_view: _DocumentScoreView,
    section_view: _DocumentScoreView,
) -> str | None:
    if not eligible_doc_ids:
        return None
    return sorted(
        eligible_doc_ids,
        key=lambda doc_id: (
            -_score_ratio(document_view, section_view, doc_id),
            -section_view.score_by_doc_id.get(doc_id, 0.0),
            -document_view.score_by_doc_id.get(doc_id, 0.0),
            section_view.rank_by_doc_id.get(doc_id, 10**9),
            doc_id,
        ),
    )[0]


def _baseline_signal_evidence(rank: int | None) -> _SectionSignalEvidence:
    return _SectionSignalEvidence(
        baseline_rank=rank,
        section_rank=None,
        document_score=0.0,
        section_score=0.0,
        score_ratio=0.0,
        score_margin_to_rank10=None,
        section_signal_bucket=_BASELINE_SECTION_SIGNAL_BUCKET,
        baseline_rank_bucket=_rank_bucket(rank),
        section_rank_bucket="not_evaluated",
        score_ratio_bucket="not_evaluated",
        score_margin_bucket="not_evaluated",
        promotion_reason_code="baseline",
        top10_protection_action="baseline_order",
    )


def _default_signal_evidence(
    *,
    document_view: _DocumentScoreView,
    section_view: _DocumentScoreView,
    document_id: str,
) -> _SectionSignalEvidence:
    return _section_signal_evidence(
        document_id=document_id,
        document_view=document_view,
        section_view=section_view,
        rank10_score=_rank10_score(document_view),
        selected_for_promotion=False,
        top10_protection_action="answer_doc_not_promoted",
    )


def _section_signal_evidence(
    *,
    document_id: str,
    document_view: _DocumentScoreView,
    section_view: _DocumentScoreView,
    rank10_score: float | None,
    selected_for_promotion: bool,
    top10_protection_action: str,
) -> _SectionSignalEvidence:
    baseline_rank = document_view.rank_by_doc_id.get(document_id)
    section_rank = section_view.rank_by_doc_id.get(document_id)
    document_score = document_view.score_by_doc_id.get(document_id, 0.0)
    section_score = section_view.score_by_doc_id.get(document_id, 0.0)
    score_ratio = _ratio(section_score, document_score)
    score_margin = None if rank10_score is None else rank10_score - document_score
    return _SectionSignalEvidence(
        baseline_rank=baseline_rank,
        section_rank=section_rank,
        document_score=round(document_score, 6),
        section_score=round(section_score, 6),
        score_ratio=round(score_ratio, 6) if math.isfinite(score_ratio) else score_ratio,
        score_margin_to_rank10=round(score_margin, 6)
        if score_margin is not None
        else None,
        section_signal_bucket="promoted" if selected_for_promotion else "not_promoted",
        baseline_rank_bucket=_rank_bucket(baseline_rank),
        section_rank_bucket=_rank_bucket(section_rank),
        score_ratio_bucket=_ratio_bucket(score_ratio),
        score_margin_bucket=_margin_bucket(score_margin),
        promotion_reason_code=_promotion_reason_code(
            baseline_rank=baseline_rank,
            section_rank=section_rank,
            score_ratio=score_ratio,
            selected_for_promotion=selected_for_promotion,
        ),
        top10_protection_action=top10_protection_action,
    )


def _record_rank(
    *,
    accumulator: dict[str, Any],
    sample: PrimeQAHybridSplitSample,
    rank: int | None,
    query_token_count: int,
    signal_evidence: _SectionSignalEvidence,
    promotion_applied: bool = False,
    top10_demotion_applied: bool = False,
    top_k_values: tuple[int, ...],
    search_depth: int,
) -> None:
    accumulator["evaluated_questions"] += 1
    accumulator["ranks_by_sample_id"][sample.sample_id] = rank
    accumulator["query_token_count_by_sample_id"][sample.sample_id] = query_token_count
    accumulator["signal_features_by_sample_id"][sample.sample_id] = {
        "section_signal_bucket": signal_evidence.section_signal_bucket,
        "baseline_rank_bucket": signal_evidence.baseline_rank_bucket,
        "section_rank_bucket": signal_evidence.section_rank_bucket,
        "score_ratio_bucket": signal_evidence.score_ratio_bucket,
        "score_margin_bucket": signal_evidence.score_margin_bucket,
        "promotion_reason_code": signal_evidence.promotion_reason_code,
        "top10_protection_action": signal_evidence.top10_protection_action,
    }
    accumulator["empty_query_count"] += query_token_count == 0
    accumulator["query_token_counts"].append(query_token_count)
    accumulator["promotion_count"] += promotion_applied
    accumulator["protected_top10_demotion_count"] += top10_demotion_applied
    for top_k in top_k_values:
        accumulator["hit_counts"].setdefault(top_k, 0)
        if rank is not None and rank <= top_k:
            accumulator["hit_counts"][top_k] += 1
    if rank is not None and rank <= search_depth:
        accumulator["search_depth_hit_count"] += 1
        accumulator["reciprocal_rank_sum_at_search_depth"] += 1 / rank
        if rank <= _PRIMARY_TOP_K:
            accumulator["reciprocal_rank_sum_at_10"] += 1 / rank
        else:
            accumulator["rank_11_to_50_count"] += 1


def _empty_accumulator(
    *,
    split: str,
    config_id: str,
    total_questions: int,
) -> dict[str, Any]:
    return {
        "split": split,
        "config_id": config_id,
        "total_questions": total_questions,
        "evaluated_questions": 0,
        "hit_counts": {},
        "search_depth_hit_count": 0,
        "rank_11_to_50_count": 0,
        "reciprocal_rank_sum_at_10": 0.0,
        "reciprocal_rank_sum_at_search_depth": 0.0,
        "ranks_by_sample_id": {},
        "query_token_count_by_sample_id": {},
        "signal_features_by_sample_id": {},
        "empty_query_count": 0,
        "query_token_counts": [],
        "promotion_count": 0,
        "protected_top10_demotion_count": 0,
    }


def _finalize_accumulator(
    accumulator: Mapping[str, Any],
    *,
    top_k_values: tuple[int, ...],
    search_depth: int,
) -> dict[str, Any]:
    evaluated_count = int(accumulator["evaluated_questions"])
    hit_counts = {
        top_k: int(accumulator["hit_counts"].get(top_k, 0))
        for top_k in top_k_values
    }
    search_depth_hit_count = int(accumulator["search_depth_hit_count"])
    rank_11_to_50_count = int(accumulator["rank_11_to_50_count"])
    return {
        "split": accumulator["split"],
        "config_id": accumulator["config_id"],
        "total_questions": int(accumulator["total_questions"]),
        "evaluated_questions": evaluated_count,
        "hit_counts": hit_counts,
        "hit_at_k": {
            top_k: _rounded_ratio(count, evaluated_count)
            for top_k, count in hit_counts.items()
        },
        "mrr_at_10": _rounded_ratio_float(
            float(accumulator["reciprocal_rank_sum_at_10"]),
            evaluated_count,
        ),
        "mrr_at_search_depth": _rounded_ratio_float(
            float(accumulator["reciprocal_rank_sum_at_search_depth"]),
            evaluated_count,
        ),
        "miss_count_at_primary_top_k": evaluated_count
        - hit_counts.get(_PRIMARY_TOP_K, 0),
        "miss_rate_at_primary_top_k": _rounded_ratio(
            evaluated_count - hit_counts.get(_PRIMARY_TOP_K, 0),
            evaluated_count,
        ),
        "search_depth": search_depth,
        "hit_count_at_search_depth": search_depth_hit_count,
        "not_found_count_at_search_depth": evaluated_count - search_depth_hit_count,
        "not_found_rate_at_search_depth": _rounded_ratio(
            evaluated_count - search_depth_hit_count,
            evaluated_count,
        ),
        "rank_11_to_50_count": rank_11_to_50_count,
        "rank_11_to_50_rate": _rounded_ratio(rank_11_to_50_count, evaluated_count),
        "empty_query_count": int(accumulator["empty_query_count"]),
        "average_query_token_count": _rounded_mean(accumulator["query_token_counts"]),
        "section_signal_promotion_count": int(accumulator["promotion_count"]),
        "protected_top10_demotion_count": int(
            accumulator["protected_top10_demotion_count"]
        ),
        "ranks_by_sample_id": accumulator["ranks_by_sample_id"],
        "query_token_count_by_sample_id": accumulator[
            "query_token_count_by_sample_id"
        ],
        "signal_features_by_sample_id": accumulator["signal_features_by_sample_id"],
    }


def _compare_to_baseline(
    *,
    split: str,
    baseline: Mapping[str, Any],
    challenger: Mapping[str, Any],
    max_k: int,
    search_depth: int,
) -> dict[str, Any]:
    baseline_ranks = baseline["ranks_by_sample_id"]
    challenger_ranks = challenger["ranks_by_sample_id"]
    signal_features = challenger["signal_features_by_sample_id"]
    top10_improvements = []
    top10_regressions = []
    search_depth_improvements = []
    search_depth_regressions = []
    rank_up = 0
    rank_down = 0
    both_hit = 0
    both_miss = 0
    for sample_id, baseline_rank in baseline_ranks.items():
        challenger_rank = challenger_ranks.get(sample_id)
        baseline_hit = baseline_rank is not None and baseline_rank <= max_k
        challenger_hit = challenger_rank is not None and challenger_rank <= max_k
        baseline_found = baseline_rank is not None and baseline_rank <= search_depth
        challenger_found = challenger_rank is not None and challenger_rank <= search_depth
        change_case = _change_case(
            split=split,
            sample_id=sample_id,
            baseline_rank=baseline_rank,
            challenger_rank=challenger_rank,
            config_id=str(challenger["config_id"]),
            signal_features=signal_features.get(sample_id) or {},
        )
        if not baseline_hit and challenger_hit:
            top10_improvements.append(change_case)
        elif baseline_hit and not challenger_hit:
            top10_regressions.append(change_case)
        elif baseline_hit and challenger_hit:
            both_hit += 1
            if challenger_rank < baseline_rank:
                rank_up += 1
            elif challenger_rank > baseline_rank:
                rank_down += 1
        else:
            both_miss += 1
        if not baseline_found and challenger_found:
            search_depth_improvements.append(change_case)
        elif baseline_found and not challenger_found:
            search_depth_regressions.append(change_case)
    metric_deltas = {
        f"hit@{top_k}_delta": round(
            float(challenger["hit_at_k"][top_k]) - float(baseline["hit_at_k"][top_k]),
            4,
        )
        for top_k in baseline["hit_at_k"]
    }
    metric_deltas["mrr_at_10_delta"] = round(
        float(challenger["mrr_at_10"]) - float(baseline["mrr_at_10"]),
        4,
    )
    metric_deltas["mrr_at_search_depth_delta"] = round(
        float(challenger["mrr_at_search_depth"])
        - float(baseline["mrr_at_search_depth"]),
        4,
    )
    return {
        "baseline_config_id": baseline["config_id"],
        "challenger_config_id": challenger["config_id"],
        **metric_deltas,
        "top10_improvement_count": len(top10_improvements),
        "top10_regression_count": len(top10_regressions),
        "top10_net_improvement_count": len(top10_improvements)
        - len(top10_regressions),
        "search_depth": search_depth,
        "search_depth_improvement_count": len(search_depth_improvements),
        "search_depth_regression_count": len(search_depth_regressions),
        "search_depth_net_improvement_count": len(search_depth_improvements)
        - len(search_depth_regressions),
        "not_found_count_at_search_depth_delta": int(
            challenger["not_found_count_at_search_depth"]
        )
        - int(baseline["not_found_count_at_search_depth"]),
        "rank_11_to_50_count_delta": int(challenger["rank_11_to_50_count"])
        - int(baseline["rank_11_to_50_count"]),
        "section_signal_promotion_count": int(
            challenger["section_signal_promotion_count"]
        ),
        "protected_top10_demotion_count": int(
            challenger["protected_top10_demotion_count"]
        ),
        "both_hit_count": both_hit,
        "both_miss_count": both_miss,
        "rank_up_within_top10_count": rank_up,
        "rank_down_within_top10_count": rank_down,
        "sample_top10_improvements": top10_improvements[:20],
        "sample_top10_regressions": top10_regressions[:20],
        "sample_search_depth_improvements": search_depth_improvements[:20],
        "sample_search_depth_regressions": search_depth_regressions[:20],
    }


def _select_config_on_train(
    *,
    rank_tables: Mapping[str, Mapping[str, Mapping[str, Any]]],
    comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]],
    candidate_configs: Sequence[SectionSignalConfig],
) -> dict[str, Any]:
    config_map = {config.config_id: config for config in candidate_configs}
    candidate_ids = [config.config_id for config in candidate_configs]
    train_results = rank_tables[_TRAIN_SPLIT]
    train_comparisons = comparisons[_TRAIN_SPLIT]
    selected_config_id = sorted(
        candidate_ids,
        key=lambda config_id: (
            -float(train_results[config_id]["hit_at_k"][_PRIMARY_TOP_K]),
            -int(train_comparisons[config_id]["search_depth_net_improvement_count"]),
            int(train_comparisons[config_id]["top10_regression_count"]),
            -float(train_results[config_id]["hit_at_k"].get(5, 0.0)),
            -float(train_results[config_id]["hit_at_k"].get(1, 0.0)),
            -float(train_results[config_id]["mrr_at_10"]),
            config_map[config_id].maximum_top10_promotions_per_query,
            config_id,
        ),
    )[0]
    return {
        "selection_rule": _selection_rule_description(),
        "selected_config_id": selected_config_id,
        "candidate_config_count": len(candidate_configs),
        "selected_train_metrics": _public_config_metrics(
            train_results[selected_config_id]
        ),
        "selected_train_comparison_to_baseline": train_comparisons[
            selected_config_id
        ],
    }


def _guard_checks(
    *,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    rank_tables: Mapping[str, Mapping[str, Mapping[str, Any]]],
    section_summary: Mapping[str, Any],
    stage75_report: Mapping[str, Any],
    stage91_report: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_configs: Sequence[SectionSignalConfig],
    user_confirmed_protocol: bool,
    confirmed_protocol_id: str,
    top_k_values: tuple[int, ...],
    search_depth: int,
) -> list[dict[str, Any]]:
    observed_split_names = sorted(
        {sample.assigned_split for samples in split_samples.values() for sample in samples}
    )
    expected_splits = sorted(_ALLOWED_DEVELOPMENT_SPLITS)
    stage75_split_reports = stage75_report.get("split_reports") or {}
    baseline_train_hit10 = _baseline_hit10(rank_tables, _TRAIN_SPLIT)
    baseline_dev_hit10 = _baseline_hit10(rank_tables, _DEV_SPLIT)
    stage75_train_hit10 = (
        (stage75_split_reports.get(_TRAIN_SPLIT) or {}).get("hit_at_top_k")
    )
    stage75_dev_hit10 = (
        (stage75_split_reports.get(_DEV_SPLIT) or {}).get("hit_at_top_k")
    )
    decision = stage91_report.get("decision") or {}
    public_fields = protocol.get("public_safe_changed_case_fields") or []
    return [
        _check(
            name="analysis_splits_are_train_dev_only",
            passed=observed_split_names == expected_splits,
            observed=observed_split_names,
            expected=expected_splits,
        ),
        _check(
            name="top_k_values_include_primary_top10",
            passed=_PRIMARY_TOP_K in top_k_values,
            observed=list(top_k_values),
            expected=f"contains {_PRIMARY_TOP_K}",
        ),
        _check(
            name="search_depth_covers_primary_top10",
            passed=search_depth >= _PRIMARY_TOP_K,
            observed=search_depth,
            expected=f">= {_PRIMARY_TOP_K}",
        ),
        _check(
            name="source_stage91_report_is_stage91",
            passed=stage91_report.get("stage") == "Stage 91",
            observed=stage91_report.get("stage"),
            expected="Stage 91",
        ),
        _check(
            name="stage91_protocol_id_matches",
            passed=protocol.get("protocol_id") == _PROTOCOL_ID,
            observed=protocol.get("protocol_id"),
            expected=_PROTOCOL_ID,
        ),
        _check(
            name="stage91_candidate_id_matches",
            passed=protocol.get("candidate_id") == _CANDIDATE_ID,
            observed=protocol.get("candidate_id"),
            expected=_CANDIDATE_ID,
        ),
        _check(
            name="user_confirmed_frozen_protocol",
            passed=user_confirmed_protocol,
            observed=user_confirmed_protocol,
            expected=True,
        ),
        _check(
            name="confirmed_protocol_id_matches",
            passed=confirmed_protocol_id == _PROTOCOL_ID,
            observed=confirmed_protocol_id,
            expected=_PROTOCOL_ID,
        ),
        _check(
            name="stage91_allows_train_dev_metrics_after_confirmation",
            passed=_stage91_allows_metric_run(stage91_report),
            observed=decision.get("can_run_train_dev_metrics_after_user_confirmation"),
            expected=True,
        ),
        _check(
            name="stage91_final_test_metrics_locked",
            passed=decision.get("can_run_final_test_metrics_now") is False,
            observed=decision.get("can_run_final_test_metrics_now"),
            expected=False,
        ),
        _check(
            name="stage91_forbids_test_tuning",
            passed=decision.get("can_use_test_for_tuning") is False,
            observed=decision.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage91_default_runtime_policy_unchanged",
            passed=decision.get("default_runtime_policy") == "unchanged",
            observed=decision.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="candidate_config_grid_matches_frozen_protocol",
            passed=tuple(config.config_id for config in protocol_configs)
            == _EXPECTED_CONFIG_IDS,
            observed=[config.config_id for config in protocol_configs],
            expected=list(_EXPECTED_CONFIG_IDS),
        ),
        _check(
            name="section_index_has_nonempty_sections",
            passed=int(section_summary["section_count"]) > 0,
            observed=int(section_summary["section_count"]),
            expected="> 0",
        ),
        _check(
            name="baseline_train_hit10_matches_stage75",
            passed=baseline_train_hit10 == stage75_train_hit10,
            observed=baseline_train_hit10,
            expected=stage75_train_hit10,
        ),
        _check(
            name="baseline_dev_hit10_matches_stage75",
            passed=baseline_dev_hit10 == stage75_dev_hit10,
            observed=baseline_dev_hit10,
            expected=stage75_dev_hit10,
        ),
        _check(
            name="source_doc_ids_not_used_as_runtime_evidence",
            passed=True,
            observed="not_used",
            expected="not_used",
        ),
        _check(
            name="changed_case_fields_public_safe",
            passed=not any(
                field in public_fields
                for field in [
                    "question_text",
                    "answer",
                    "document_title",
                    "document_body_text",
                    "section_text",
                    "matched_token_strings",
                ]
            ),
            observed=public_fields,
            expected="no raw question, answer, document, or section text fields",
        ),
        _check(
            name="final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]],
    train_selection: Mapping[str, Any],
    candidate_configs: Sequence[SectionSignalConfig],
) -> dict[str, Any]:
    failed_checks = [check["name"] for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_section_signal_comparison_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        }
    selected_config_id = str(train_selection["selected_config_id"])
    selected_config = {
        config.config_id: config for config in candidate_configs
    }[selected_config_id]
    selected_dev_comparison = comparisons[_DEV_SPLIT][selected_config_id]
    dev_hit10_delta = float(selected_dev_comparison["hit@10_delta"])
    search_depth_net = int(
        selected_dev_comparison["search_depth_net_improvement_count"]
    )
    dev_top10_regressions = int(selected_dev_comparison["top10_regression_count"])
    primary_contract_passed = dev_hit10_delta > 0
    secondary_contract_passed = search_depth_net > 0
    guard_contract_passed = (
        selected_config.demote_existing_bm25_top10 is False
        or dev_top10_regressions == 0
    )
    if primary_contract_passed and secondary_contract_passed and guard_contract_passed:
        recommended_next_stage = (
            "Stage 93: review section signal changed cases and runtime risk before "
            "any runtime experiment; keep test locked."
        )
    else:
        recommended_next_stage = (
            "Stage 93: stop section signal guarded expansion as a retrieval-recall "
            "route unless a new train/dev-only protocol is explicitly confirmed; "
            "keep test locked and move to the next confirmed second-wave candidate."
        )
    return {
        "status": "primeqa_hybrid_section_signal_comparison_completed",
        "selected_config_id": selected_config_id,
        "selected_dev_hit10_delta": dev_hit10_delta,
        "selected_dev_search_depth_net_improvement_count": search_depth_net,
        "selected_dev_top10_improvements": int(
            selected_dev_comparison["top10_improvement_count"]
        ),
        "selected_dev_top10_regressions": dev_top10_regressions,
        "selected_dev_rank_up_within_top10": int(
            selected_dev_comparison["rank_up_within_top10_count"]
        ),
        "selected_dev_rank_down_within_top10": int(
            selected_dev_comparison["rank_down_within_top10_count"]
        ),
        "selected_dev_not_found_at_search_depth_delta": int(
            selected_dev_comparison["not_found_count_at_search_depth_delta"]
        ),
        "selected_dev_rank_11_to_50_count_delta": int(
            selected_dev_comparison["rank_11_to_50_count_delta"]
        ),
        "selected_dev_section_signal_promotion_count": int(
            selected_dev_comparison["section_signal_promotion_count"]
        ),
        "selected_dev_protected_top10_demotion_count": int(
            selected_dev_comparison["protected_top10_demotion_count"]
        ),
        "primary_contract_passed": primary_contract_passed,
        "secondary_contract_passed": secondary_contract_passed,
        "guard_contract_passed": guard_contract_passed,
        "can_continue_train_dev_development": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": recommended_next_stage,
    }


def _public_config_metrics(config_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "total_questions": int(config_result["total_questions"]),
        "evaluated_questions": int(config_result["evaluated_questions"]),
        "hit_counts": {
            f"hit@{top_k}": count
            for top_k, count in sorted(config_result["hit_counts"].items())
        },
        "hit_at_k": {
            f"hit@{top_k}": value
            for top_k, value in sorted(config_result["hit_at_k"].items())
        },
        "mrr_at_10": float(config_result["mrr_at_10"]),
        "mrr_at_search_depth": float(config_result["mrr_at_search_depth"]),
        "miss_count_at_10": int(config_result["miss_count_at_primary_top_k"]),
        "miss_rate_at_10": float(config_result["miss_rate_at_primary_top_k"]),
        "search_depth": int(config_result["search_depth"]),
        "hit_count_at_search_depth": int(config_result["hit_count_at_search_depth"]),
        "not_found_count_at_search_depth": int(
            config_result["not_found_count_at_search_depth"]
        ),
        "not_found_rate_at_search_depth": float(
            config_result["not_found_rate_at_search_depth"]
        ),
        "rank_11_to_50_count": int(config_result["rank_11_to_50_count"]),
        "rank_11_to_50_rate": float(config_result["rank_11_to_50_rate"]),
        "empty_query_count": int(config_result["empty_query_count"]),
        "average_query_token_count": float(config_result["average_query_token_count"]),
        "section_signal_promotion_count": int(
            config_result["section_signal_promotion_count"]
        ),
        "protected_top10_demotion_count": int(
            config_result["protected_top10_demotion_count"]
        ),
    }


def _public_candidate_config(config: SectionSignalConfig) -> dict[str, Any]:
    return {
        "config_id": config.config_id,
        "promotion_mode": config.promotion_mode,
        "eligible_baseline_rank_min": config.eligible_baseline_rank_min,
        "eligible_baseline_rank_max": config.eligible_baseline_rank_max,
        "section_rank_max": config.section_rank_max,
        "minimum_section_to_document_score_ratio": (
            config.minimum_section_to_document_score_ratio
        ),
        "maximum_document_score_margin_to_rank10": (
            config.maximum_document_score_margin_to_rank10
        ),
        "maximum_top10_promotions_per_query": (
            config.maximum_top10_promotions_per_query
        ),
        "protected_bm25_top_rank_count": config.protected_bm25_top_rank_count,
        "demote_existing_bm25_top10": config.demote_existing_bm25_top10,
    }


def _change_case(
    *,
    split: str,
    sample_id: str,
    baseline_rank: int | None,
    challenger_rank: int | None,
    config_id: str,
    signal_features: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "split": split,
        "baseline_rank": baseline_rank,
        "challenger_rank": challenger_rank,
        "config_id": config_id,
        "section_signal_bucket": signal_features.get("section_signal_bucket"),
        "baseline_rank_bucket": signal_features.get("baseline_rank_bucket"),
        "section_rank_bucket": signal_features.get("section_rank_bucket"),
        "score_ratio_bucket": signal_features.get("score_ratio_bucket"),
        "score_margin_bucket": signal_features.get("score_margin_bucket"),
        "promotion_reason_code": signal_features.get("promotion_reason_code"),
        "top10_protection_action": signal_features.get("top10_protection_action"),
    }


def _hit_at_k_bars(
    report: Mapping[str, Any],
    *,
    split: str,
    top_k: int,
) -> list[BarDatum]:
    metrics = report.get("metrics_by_split", {}).get(split, {})
    metric_name = f"hit@{top_k}"
    ordered = sorted(
        metrics.items(),
        key=lambda item: (-item[1]["hit_at_k"][metric_name], item[0]),
    )
    return [
        BarDatum(
            label=config_id,
            value=float(config_metrics["hit_at_k"][metric_name]),
            value_label=f"{config_metrics['hit_at_k'][metric_name]:.4f}",
        )
        for config_id, config_metrics in ordered
    ]


def _delta_bars(
    report: Mapping[str, Any],
    *,
    split: str,
    metric: str,
) -> list[BarDatum]:
    comparisons = report.get("comparisons_to_baseline", {}).get(split, {})
    return [
        BarDatum(
            label=config_id,
            value=float(comparison[metric]),
            value_label=f"{float(comparison[metric]):+.4f}",
        )
        for config_id, comparison in sorted(comparisons.items())
    ]


def _search_depth_net_bars(report: Mapping[str, Any], *, split: str) -> list[BarDatum]:
    comparisons = report.get("comparisons_to_baseline", {}).get(split, {})
    return [
        BarDatum(
            label=config_id,
            value=float(comparison["search_depth_net_improvement_count"]),
            value_label=str(comparison["search_depth_net_improvement_count"]),
        )
        for config_id, comparison in sorted(comparisons.items())
    ]


def _top10_net_bars(report: Mapping[str, Any], *, split: str) -> list[BarDatum]:
    comparisons = report.get("comparisons_to_baseline", {}).get(split, {})
    return [
        BarDatum(
            label=config_id,
            value=float(comparison["top10_net_improvement_count"]),
            value_label=str(comparison["top10_net_improvement_count"]),
        )
        for config_id, comparison in sorted(comparisons.items())
    ]


def _guard_check_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=check["name"],
            value=1.0 if check["passed"] else 0.0,
            value_label="passed" if check["passed"] else "failed",
        )
        for check in report.get("guard_checks", [])
    ]


def _section_summary(sections_by_document: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    section_counts = [len(sections) for sections in sections_by_document.values()]
    documents_with_sections = sum(count > 0 for count in section_counts)
    section_count = sum(section_counts)
    return {
        "section_count": section_count,
        "documents_with_sections": documents_with_sections,
        "documents_without_sections": len(section_counts) - documents_with_sections,
        "average_sections_per_document": _rounded_mean(section_counts),
    }


def _selection_rule_description() -> str:
    return (
        "Select the section signal config on train only by hit@10, then "
        "search-depth net improvements, then fewer top10 regressions, then "
        "hit@5, hit@1, MRR@10, lower top10 demotion budget, then config_id; "
        "dev is validation only."
    )


def _stage91_allows_metric_run(stage91_report: Mapping[str, Any]) -> bool:
    decision = stage91_report.get("decision") or {}
    return (
        stage91_report.get("stage") == "Stage 91"
        and decision.get("status") == "primeqa_hybrid_section_signal_protocol_frozen"
        and decision.get("can_run_train_dev_metrics_after_user_confirmation") is True
        and decision.get("can_run_final_test_metrics_now") is False
        and decision.get("can_use_test_for_tuning") is False
        and decision.get("default_runtime_policy") == "unchanged"
    )


def _baseline_hit10(
    rank_tables: Mapping[str, Mapping[str, Mapping[str, Any]]],
    split: str,
) -> float | None:
    if not rank_tables:
        return None
    return rank_tables[split][_BASELINE_CONFIG_ID]["hit_at_k"][_PRIMARY_TOP_K]


def _answer_rank(*, ranked_doc_ids: Sequence[str], answer_doc_id: str) -> int | None:
    for rank, document_id in enumerate(ranked_doc_ids, start=1):
        if document_id == answer_doc_id:
            return rank
    return None


def _rank10_score(document_view: _DocumentScoreView) -> float | None:
    if len(document_view.ranked) < _PRIMARY_TOP_K:
        return None
    return document_view.ranked[_PRIMARY_TOP_K - 1].score


def _score_ratio(
    document_view: _DocumentScoreView,
    section_view: _DocumentScoreView,
    document_id: str,
) -> float:
    return _ratio(
        section_view.score_by_doc_id.get(document_id, 0.0),
        document_view.score_by_doc_id.get(document_id, 0.0),
    )


def _ratio(numerator: float, denominator: float) -> float:
    if numerator <= 0:
        return 0.0
    if denominator <= 0:
        return math.inf
    return numerator / max(denominator, _EPSILON)


def _rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "not_found_top50"
    if rank <= 1:
        return "rank_1"
    if rank <= 5:
        return "rank_2_to_5"
    if rank <= 10:
        return "rank_6_to_10"
    if rank <= 20:
        return "rank_11_to_20"
    if rank <= 50:
        return "rank_21_to_50"
    return "not_found_top50"


def _ratio_bucket(ratio: float) -> str:
    if math.isinf(ratio):
        return "ratio_document_score_zero"
    if ratio >= 1.6:
        return "ratio_gte_1_60"
    if ratio >= 1.45:
        return "ratio_1_45_to_1_59"
    if ratio >= 1.2:
        return "ratio_1_20_to_1_44"
    if ratio >= 1.1:
        return "ratio_1_10_to_1_19"
    return "ratio_lt_1_10"


def _margin_bucket(margin: float | None) -> str:
    if margin is None:
        return "margin_not_available"
    if margin <= 0.05:
        return "margin_lte_0_05"
    if margin <= 0.08:
        return "margin_0_05_to_0_08"
    return "margin_gt_0_08"


def _promotion_reason_code(
    *,
    baseline_rank: int | None,
    section_rank: int | None,
    score_ratio: float,
    selected_for_promotion: bool,
) -> str:
    prefix = "promoted" if selected_for_promotion else "not_promoted"
    return "|".join(
        [
            prefix,
            _rank_bucket(baseline_rank),
            _rank_bucket(section_rank),
            _ratio_bucket(score_ratio),
        ]
    )


def _union_preserving_order(
    first: Sequence[str],
    second: Sequence[str],
) -> list[str]:
    return _dedupe([*first, *second])


def _dedupe(values: Sequence[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _validate_options(
    *,
    top_k_values: tuple[int, ...],
    search_depth: int,
) -> None:
    if not top_k_values:
        raise ValueError("top_k_values must not be empty")
    if any(top_k <= 0 for top_k in top_k_values):
        raise ValueError("top_k_values must be positive")
    if _PRIMARY_TOP_K not in top_k_values:
        raise ValueError("top_k_values must include 10")
    if search_depth < max(top_k_values):
        raise ValueError("search_depth must cover every top_k value")


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


def _check(*, name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _compute_idf(document_count: int, document_frequency: int) -> float:
    return math.log(
        1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
    )


def _score_term_vector(
    *,
    idf: float,
    term_frequencies: np.ndarray,
    doc_lengths: np.ndarray,
    avg_length: float,
    k1: float,
    b: float,
) -> np.ndarray:
    length_normalizer = 1 - b
    if avg_length:
        length_normalizer += b * doc_lengths / avg_length
    numerator = term_frequencies * (k1 + 1)
    denominator = term_frequencies + k1 * length_normalizer
    return idf * numerator / denominator


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _rounded_ratio(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _rounded_ratio_float(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _rounded_mean(values: Sequence[int | float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
