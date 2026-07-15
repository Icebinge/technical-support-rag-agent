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
from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.infrastructure.bm25_retriever import tokenize_text
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_documents

_STAGE = "Stage 95"
_CREATED_AT = "2026-07-15"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_PROTOCOL_ID = "score_margin_bm25_normalization_gate_train_dev_v1"
_CANDIDATE_ID = "score_margin_bm25_normalization_gate_design"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})
_BASELINE_CONFIG_ID = "full_document_bm25_baseline"
_BASELINE_NORMALIZATION_VIEW_ID = "bm25_k1_1_5_b_0_75_baseline"
_PRIMARY_TOP_K = 10
_DEFAULT_SEARCH_DEPTH = 50
_BASELINE_K1 = 1.5
_BASELINE_B = 0.75
_EXPECTED_CONFIG_IDS = (
    "smbn_rank11_20_long_doc_b095_margin_v1",
    "smbn_rank21_50_long_doc_b095_high_confidence_v1",
    "smbn_rank11_20_short_doc_b055_margin_v1",
    "smbn_rank11_50_dual_length_band_margin_v1",
)
_EPSILON = 1e-9


@dataclass(frozen=True)
class PrimeQAHybridScoreMarginBM25ComparisonVisualization:
    """One generated Stage95 score-margin BM25 comparison visualization."""

    name: str
    path: str


@dataclass(frozen=True)
class ScoreMarginBM25Config:
    """One frozen score-margin BM25 normalization configuration."""

    config_id: str
    normalization_view_id: str
    challenger_bm25_k1: float
    challenger_bm25_b: float | str
    eligible_baseline_rank_min: int
    eligible_baseline_rank_max: int
    challenger_rank_max: int
    maximum_score_margin_to_rank10: float
    length_gate_mode: str
    minimum_document_length_ratio_to_average: float | None
    maximum_document_length_ratio_to_average: float | None
    maximum_top10_promotions_per_query: int
    protected_bm25_top_rank_count: int


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
class _ScoreMarginEvidence:
    baseline_rank: int | None
    challenger_rank: int | None
    baseline_score: float
    challenger_score: float
    baseline_score_margin_to_rank10: float | None
    challenger_score_margin_to_rank10: float | None
    document_length_ratio_to_average: float | None
    normalization_view_id: str
    baseline_rank_bucket: str
    challenger_rank_bucket: str
    score_margin_bucket: str
    document_length_bucket: str
    promotion_reason_code: str


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
        self._doc_id_to_index = {
            str(document_id): index for index, document_id in enumerate(self._doc_ids)
        }
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

    @property
    def average_document_length(self) -> float:
        return self._avg_doc_length

    def document_length_ratio(self, document_id: str) -> float | None:
        doc_index = self._doc_id_to_index.get(document_id)
        if doc_index is None or self._avg_doc_length <= 0:
            return None
        return float(self._doc_lengths_array[doc_index] / self._avg_doc_length)

    def score_query(
        self,
        *,
        query_terms: Sequence[str],
        search_depth: int,
        k1: float,
        b: float,
    ) -> _DocumentScoreView:
        scores = self._score_query_array(query_terms=query_terms, k1=k1, b=b)
        return self._score_view_from_scores(scores=scores, search_depth=search_depth)

    def mixed_length_band_view(
        self,
        *,
        query_terms: Sequence[str],
        search_depth: int,
        short_b: float,
        long_b: float,
        baseline_b: float,
        short_length_ratio_max: float,
        long_length_ratio_min: float,
        k1: float,
    ) -> _DocumentScoreView:
        short_scores = self._score_query_array(query_terms=query_terms, k1=k1, b=short_b)
        long_scores = self._score_query_array(query_terms=query_terms, k1=k1, b=long_b)
        baseline_scores = self._score_query_array(
            query_terms=query_terms,
            k1=k1,
            b=baseline_b,
        )
        scores = baseline_scores.copy()
        if self._avg_doc_length > 0:
            length_ratios = self._doc_lengths_array / self._avg_doc_length
            scores[length_ratios <= short_length_ratio_max] = short_scores[
                length_ratios <= short_length_ratio_max
            ]
            scores[length_ratios >= long_length_ratio_min] = long_scores[
                length_ratios >= long_length_ratio_min
            ]
        return self._score_view_from_scores(scores=scores, search_depth=search_depth)

    def _score_query_array(
        self,
        *,
        query_terms: Sequence[str],
        k1: float,
        b: float,
    ) -> np.ndarray:
        if not query_terms or not self._documents:
            return np.zeros(len(self._documents), dtype=np.float64)
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
                k1=k1,
                b=b,
            )
        return scores

    def _score_view_from_scores(
        self,
        *,
        scores: np.ndarray,
        search_depth: int,
    ) -> _DocumentScoreView:
        if search_depth <= 0:
            raise ValueError("search_depth must be positive")
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


def run_primeqa_hybrid_score_margin_bm25_comparison(
    *,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    stage75_report_path: Path,
    stage94_report_path: Path,
    user_confirmed_protocol: bool,
    confirmed_protocol_id: str,
    confirmation_note: str,
    top_k_values: tuple[int, ...] = (1, 5, 10),
    search_depth: int = _DEFAULT_SEARCH_DEPTH,
) -> dict[str, Any]:
    """Run the confirmed train/dev-only score-margin BM25 comparison."""

    _validate_options(top_k_values=top_k_values, search_depth=search_depth)
    started_at = time.perf_counter()
    split_samples = {
        _TRAIN_SPLIT: load_primeqa_hybrid_split_samples(train_split_path),
        _DEV_SPLIT: load_primeqa_hybrid_split_samples(dev_split_path),
    }
    loaded_splits_at = time.perf_counter()
    documents = load_primeqa_documents(documents_path)
    document_list = list(documents.values())
    stage75_report = _load_json_object(stage75_report_path)
    stage94_report = _load_json_object(stage94_report_path)
    loaded_inputs_at = time.perf_counter()

    protocol = stage94_report.get("frozen_protocol") or {}
    protocol_configs = _configs_from_protocol(protocol)
    should_evaluate = (
        user_confirmed_protocol
        and confirmed_protocol_id == _PROTOCOL_ID
        and protocol.get("protocol_id") == _PROTOCOL_ID
        and _stage94_allows_metric_run(stage94_report)
    )
    candidate_configs = protocol_configs if should_evaluate else ()
    rank_tables: dict[str, dict[str, dict[str, Any]]] = {}
    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    train_selection: dict[str, Any] = {
        "selection_rule": _selection_rule_description(),
        "selected_config_id": None,
        "candidate_config_count": len(protocol_configs),
        "selected_train_metrics": None,
        "selected_train_comparison_to_baseline": None,
        "selected_dev_metrics": None,
        "selected_dev_comparison_to_baseline": None,
    }
    indexed_and_evaluated_at = loaded_inputs_at
    document_index_summary = _document_index_summary(document_list)
    if candidate_configs:
        document_index = _BM25DocumentScoreIndex(document_list)
        document_index_summary = {
            "document_count": len(document_list),
            "average_document_token_count": round(
                document_index.average_document_length,
                4,
            ),
        }
        rank_tables = {
            split: _evaluate_split(
                split=split,
                samples=samples,
                document_index=document_index,
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

    guard_checks = _guard_checks(
        split_samples=split_samples,
        rank_tables=rank_tables,
        stage75_report=stage75_report,
        stage94_report=stage94_report,
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
            "score_margin_bm25_normalization_gate_train_dev_v1. This stage "
            "uses the user-confirmed Stage94 protocol, selects a score-margin "
            "BM25 normalization gate config on train only, validates on dev, "
            "keeps the frozen test split locked, does not run final metrics, "
            "does not use source DOC_IDS as runtime retrieval evidence, does "
            "not choose runtime rules from dev-only observations, does not "
            "write raw question, answer, document, query-term, or matched-token "
            "text, and does not change runtime defaults."
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
            "stage94_report": _fingerprint(stage94_report_path),
        },
        "stage94_decision": stage94_report.get("decision") or {},
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
            "raw_question_answer_document_or_query_text_written_to_report": False,
        },
        "loaded_data_summary": {
            **document_index_summary,
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
            "load_documents_and_reports": round(loaded_inputs_at - loaded_splits_at, 3),
            "bm25_score_margin_evaluate": round(
                indexed_and_evaluated_at - loaded_inputs_at,
                3,
            ),
            "guard_checks": round(checked_at - indexed_and_evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_score_margin_bm25_comparison_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridScoreMarginBM25ComparisonVisualization]:
    """Write SVG charts for Stage95 score-margin BM25 comparison."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage95_score_margin_bm25_train_hit_at_10.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage95 score-margin BM25 train hit@10",
                bars=_hit_at_k_bars(report, split=_TRAIN_SPLIT, top_k=_PRIMARY_TOP_K),
                x_label="hit@10",
                width=1400,
                margin_left=600,
            )
        ),
        "stage95_score_margin_bm25_dev_hit_at_10.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage95 score-margin BM25 dev hit@10",
                bars=_hit_at_k_bars(report, split=_DEV_SPLIT, top_k=_PRIMARY_TOP_K),
                x_label="hit@10",
                width=1400,
                margin_left=600,
            )
        ),
        "stage95_score_margin_bm25_dev_delta_hit_at_10.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage95 score-margin BM25 dev hit@10 delta vs baseline",
                bars=_delta_bars(report, split=_DEV_SPLIT, metric="hit@10_delta"),
                x_label="delta hit@10",
                width=1400,
                margin_left=600,
            )
        ),
        "stage95_score_margin_bm25_dev_rank_11_to_50_delta.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage95 score-margin BM25 dev rank 11-50 delta",
                bars=_rank_11_to_50_delta_bars(report, split=_DEV_SPLIT),
                x_label="delta count",
                width=1400,
                margin_left=600,
            )
        ),
        "stage95_score_margin_bm25_dev_top10_changes.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage95 score-margin BM25 dev top10 improvements minus regressions",
                bars=_top10_net_bars(report, split=_DEV_SPLIT),
                x_label="net changed cases",
                width=1400,
                margin_left=600,
            )
        ),
        "stage95_score_margin_bm25_dev_gate_actions.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage95 score-margin BM25 dev gate actions",
                bars=_gate_action_bars(report, split=_DEV_SPLIT),
                x_label="query-level promotion actions",
                width=1400,
                margin_left=600,
            )
        ),
        "stage95_score_margin_bm25_guard_check_status.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage95 score-margin BM25 guard checks",
                bars=_guard_check_bars(report),
                x_label="1 means passed",
                width=1460,
                margin_left=700,
            )
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridScoreMarginBM25ComparisonVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _configs_from_protocol(
    protocol: Mapping[str, Any],
) -> tuple[ScoreMarginBM25Config, ...]:
    configs = []
    for row in protocol.get("candidate_config_grid") or []:
        if not isinstance(row, Mapping):
            continue
        challenger_b = row.get("challenger_bm25_b")
        configs.append(
            ScoreMarginBM25Config(
                config_id=str(row.get("config_id") or ""),
                normalization_view_id=str(row.get("normalization_view_id") or ""),
                challenger_bm25_k1=float(row.get("challenger_bm25_k1") or _BASELINE_K1),
                challenger_bm25_b=(
                    str(challenger_b)
                    if isinstance(challenger_b, str)
                    else float(challenger_b)
                ),
                eligible_baseline_rank_min=int(
                    row.get("eligible_baseline_rank_min") or 1
                ),
                eligible_baseline_rank_max=int(
                    row.get("eligible_baseline_rank_max") or search_depth_default()
                ),
                challenger_rank_max=int(row.get("challenger_rank_max") or 0),
                maximum_score_margin_to_rank10=float(
                    row.get("maximum_score_margin_to_rank10") or 0.0
                ),
                length_gate_mode=str(row.get("length_gate_mode") or ""),
                minimum_document_length_ratio_to_average=_optional_float(
                    row.get("minimum_document_length_ratio_to_average")
                ),
                maximum_document_length_ratio_to_average=_optional_float(
                    row.get("maximum_document_length_ratio_to_average")
                ),
                maximum_top10_promotions_per_query=int(
                    row.get("maximum_top10_promotions_per_query") or 0
                ),
                protected_bm25_top_rank_count=int(
                    row.get("protected_bm25_top_rank_count") or 0
                ),
            )
        )
    return tuple(configs)


def search_depth_default() -> int:
    return _DEFAULT_SEARCH_DEPTH


def _evaluate_split(
    *,
    split: str,
    samples: Sequence[PrimeQAHybridSplitSample],
    document_index: _BM25DocumentScoreIndex,
    candidate_configs: Sequence[ScoreMarginBM25Config],
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
        normalization_view_id=_BASELINE_NORMALIZATION_VIEW_ID,
        total_questions=len(samples),
    )
    candidate_accumulators = {
        config.config_id: _empty_accumulator(
            split=split,
            config_id=config.config_id,
            normalization_view_id=config.normalization_view_id,
            total_questions=len(samples),
        )
        for config in candidate_configs
    }
    for sample in answerable_samples:
        query = sample.to_primeqa_question().full_question
        query_terms = tokenize_text(query)
        baseline_view = document_index.score_query(
            query_terms=query_terms,
            search_depth=search_depth,
            k1=_BASELINE_K1,
            b=_BASELINE_B,
        )
        view_by_normalization = _candidate_views(
            document_index=document_index,
            query_terms=query_terms,
            search_depth=search_depth,
            baseline_view=baseline_view,
            candidate_configs=candidate_configs,
        )
        baseline_ranked_doc_ids = [row.document_id for row in baseline_view.ranked]
        baseline_rank = _answer_rank(
            ranked_doc_ids=baseline_ranked_doc_ids,
            answer_doc_id=str(sample.answer_doc_id),
        )
        _record_rank(
            accumulator=baseline_accumulator,
            sample=sample,
            rank=baseline_rank,
            query_token_count=len(query_terms),
            evidence=_baseline_evidence(baseline_rank),
            top_k_values=top_k_values,
            search_depth=search_depth,
        )
        for config in candidate_configs:
            challenger_view = view_by_normalization[config.normalization_view_id]
            challenger_doc_ids, evidence, action_summary = _apply_score_margin_gate(
                baseline_view=baseline_view,
                challenger_view=challenger_view,
                document_index=document_index,
                config=config,
                search_depth=search_depth,
            )
            challenger_rank = _answer_rank(
                ranked_doc_ids=challenger_doc_ids,
                answer_doc_id=str(sample.answer_doc_id),
            )
            answer_doc_id = str(sample.answer_doc_id)
            _record_rank(
                accumulator=candidate_accumulators[config.config_id],
                sample=sample,
                rank=challenger_rank,
                query_token_count=len(query_terms),
                evidence=evidence.get(
                    answer_doc_id,
                    _default_score_margin_evidence(
                        baseline_view=baseline_view,
                        challenger_view=challenger_view,
                        document_index=document_index,
                        document_id=answer_doc_id,
                        config=config,
                    ),
                ),
                promotion_applied=bool(action_summary["promotion_applied"]),
                length_band_gate_applied=bool(action_summary["length_band_gate_applied"]),
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


def _candidate_views(
    *,
    document_index: _BM25DocumentScoreIndex,
    query_terms: Sequence[str],
    search_depth: int,
    baseline_view: _DocumentScoreView,
    candidate_configs: Sequence[ScoreMarginBM25Config],
) -> dict[str, _DocumentScoreView]:
    views: dict[str, _DocumentScoreView] = {_BASELINE_NORMALIZATION_VIEW_ID: baseline_view}
    for config in candidate_configs:
        if config.normalization_view_id in views:
            continue
        if isinstance(config.challenger_bm25_b, str):
            views[config.normalization_view_id] = document_index.mixed_length_band_view(
                query_terms=query_terms,
                search_depth=search_depth,
                short_b=0.55,
                long_b=0.95,
                baseline_b=_BASELINE_B,
                short_length_ratio_max=(
                    config.maximum_document_length_ratio_to_average or 0.0
                ),
                long_length_ratio_min=(
                    config.minimum_document_length_ratio_to_average or math.inf
                ),
                k1=config.challenger_bm25_k1,
            )
        else:
            views[config.normalization_view_id] = document_index.score_query(
                query_terms=query_terms,
                search_depth=search_depth,
                k1=config.challenger_bm25_k1,
                b=float(config.challenger_bm25_b),
            )
    return views


def _apply_score_margin_gate(
    *,
    baseline_view: _DocumentScoreView,
    challenger_view: _DocumentScoreView,
    document_index: _BM25DocumentScoreIndex,
    config: ScoreMarginBM25Config,
    search_depth: int,
) -> tuple[list[str], dict[str, _ScoreMarginEvidence], dict[str, bool]]:
    baseline_doc_ids = [row.document_id for row in baseline_view.ranked]
    rank10_score = _rank10_score(baseline_view)
    evidence = {
        document_id: _score_margin_evidence(
            document_id=document_id,
            baseline_view=baseline_view,
            challenger_view=challenger_view,
            document_index=document_index,
            config=config,
            rank10_score=rank10_score,
            selected_for_promotion=False,
        )
        for document_id in baseline_doc_ids
    }
    eligible_doc_ids = [
        document_id
        for document_id in baseline_doc_ids
        if _is_eligible_for_config(
            document_id=document_id,
            baseline_view=baseline_view,
            challenger_view=challenger_view,
            document_index=document_index,
            config=config,
            rank10_score=rank10_score,
        )
    ]
    promoted_doc_id = _select_promoted_document(
        eligible_doc_ids=eligible_doc_ids,
        baseline_view=baseline_view,
        challenger_view=challenger_view,
        rank10_score=rank10_score,
    )
    if promoted_doc_id is None or config.maximum_top10_promotions_per_query <= 0:
        return baseline_doc_ids[:search_depth], evidence, {
            "promotion_applied": False,
            "length_band_gate_applied": False,
        }

    protected_prefix_count = max(_PRIMARY_TOP_K - 1, config.protected_bm25_top_rank_count)
    protected_prefix = [
        doc_id
        for doc_id in baseline_doc_ids[:protected_prefix_count]
        if doc_id != promoted_doc_id
    ][:_PRIMARY_TOP_K - 1]
    ranked_doc_ids = (
        protected_prefix
        + [promoted_doc_id]
        + [doc_id for doc_id in baseline_doc_ids if doc_id != promoted_doc_id]
    )
    ranked_doc_ids = _dedupe(ranked_doc_ids)[:search_depth]
    evidence[promoted_doc_id] = _score_margin_evidence(
        document_id=promoted_doc_id,
        baseline_view=baseline_view,
        challenger_view=challenger_view,
        document_index=document_index,
        config=config,
        rank10_score=rank10_score,
        selected_for_promotion=True,
    )
    return ranked_doc_ids, evidence, {
        "promotion_applied": True,
        "length_band_gate_applied": True,
    }


def _is_eligible_for_config(
    *,
    document_id: str,
    baseline_view: _DocumentScoreView,
    challenger_view: _DocumentScoreView,
    document_index: _BM25DocumentScoreIndex,
    config: ScoreMarginBM25Config,
    rank10_score: float | None,
) -> bool:
    baseline_rank = baseline_view.rank_by_doc_id.get(document_id)
    challenger_rank = challenger_view.rank_by_doc_id.get(document_id)
    if baseline_rank is None:
        return False
    if baseline_rank < config.eligible_baseline_rank_min:
        return False
    if baseline_rank > config.eligible_baseline_rank_max:
        return False
    if challenger_rank is None or challenger_rank > config.challenger_rank_max:
        return False
    if rank10_score is None:
        return False
    baseline_score = baseline_view.score_by_doc_id.get(document_id, 0.0)
    if rank10_score - baseline_score > config.maximum_score_margin_to_rank10:
        return False
    return _passes_length_gate(
        document_index.document_length_ratio(document_id),
        config=config,
    )


def _passes_length_gate(
    length_ratio: float | None,
    *,
    config: ScoreMarginBM25Config,
) -> bool:
    if length_ratio is None:
        return False
    if config.length_gate_mode == "long_document_only":
        threshold = config.minimum_document_length_ratio_to_average
        return threshold is not None and length_ratio >= threshold
    if config.length_gate_mode == "short_document_only":
        threshold = config.maximum_document_length_ratio_to_average
        return threshold is not None and length_ratio <= threshold
    if config.length_gate_mode == "outside_length_band_short_or_long":
        short_threshold = config.maximum_document_length_ratio_to_average
        long_threshold = config.minimum_document_length_ratio_to_average
        return (
            short_threshold is not None
            and length_ratio <= short_threshold
            or long_threshold is not None
            and length_ratio >= long_threshold
        )
    return False


def _select_promoted_document(
    *,
    eligible_doc_ids: Sequence[str],
    baseline_view: _DocumentScoreView,
    challenger_view: _DocumentScoreView,
    rank10_score: float | None,
) -> str | None:
    if not eligible_doc_ids:
        return None
    return sorted(
        eligible_doc_ids,
        key=lambda doc_id: (
            challenger_view.rank_by_doc_id.get(doc_id, 10**9),
            _score_margin_to_rank10(
                baseline_view=baseline_view,
                document_id=doc_id,
                rank10_score=rank10_score,
            )
            or math.inf,
            -challenger_view.score_by_doc_id.get(doc_id, 0.0),
            -baseline_view.score_by_doc_id.get(doc_id, 0.0),
            doc_id,
        ),
    )[0]


def _baseline_evidence(rank: int | None) -> _ScoreMarginEvidence:
    return _ScoreMarginEvidence(
        baseline_rank=rank,
        challenger_rank=None,
        baseline_score=0.0,
        challenger_score=0.0,
        baseline_score_margin_to_rank10=None,
        challenger_score_margin_to_rank10=None,
        document_length_ratio_to_average=None,
        normalization_view_id=_BASELINE_NORMALIZATION_VIEW_ID,
        baseline_rank_bucket=_rank_bucket(rank),
        challenger_rank_bucket="not_evaluated",
        score_margin_bucket="not_evaluated",
        document_length_bucket="not_evaluated",
        promotion_reason_code="baseline",
    )


def _default_score_margin_evidence(
    *,
    baseline_view: _DocumentScoreView,
    challenger_view: _DocumentScoreView,
    document_index: _BM25DocumentScoreIndex,
    document_id: str,
    config: ScoreMarginBM25Config,
) -> _ScoreMarginEvidence:
    return _score_margin_evidence(
        document_id=document_id,
        baseline_view=baseline_view,
        challenger_view=challenger_view,
        document_index=document_index,
        config=config,
        rank10_score=_rank10_score(baseline_view),
        selected_for_promotion=False,
    )


def _score_margin_evidence(
    *,
    document_id: str,
    baseline_view: _DocumentScoreView,
    challenger_view: _DocumentScoreView,
    document_index: _BM25DocumentScoreIndex,
    config: ScoreMarginBM25Config,
    rank10_score: float | None,
    selected_for_promotion: bool,
) -> _ScoreMarginEvidence:
    baseline_rank = baseline_view.rank_by_doc_id.get(document_id)
    challenger_rank = challenger_view.rank_by_doc_id.get(document_id)
    baseline_score = baseline_view.score_by_doc_id.get(document_id, 0.0)
    challenger_score = challenger_view.score_by_doc_id.get(document_id, 0.0)
    baseline_margin = _score_margin_to_rank10(
        baseline_view=baseline_view,
        document_id=document_id,
        rank10_score=rank10_score,
    )
    challenger_rank10 = _rank10_score(challenger_view)
    challenger_margin = _score_margin_to_rank10(
        baseline_view=challenger_view,
        document_id=document_id,
        rank10_score=challenger_rank10,
    )
    length_ratio = document_index.document_length_ratio(document_id)
    return _ScoreMarginEvidence(
        baseline_rank=baseline_rank,
        challenger_rank=challenger_rank,
        baseline_score=round(baseline_score, 6),
        challenger_score=round(challenger_score, 6),
        baseline_score_margin_to_rank10=(
            None if baseline_margin is None else round(baseline_margin, 6)
        ),
        challenger_score_margin_to_rank10=(
            None if challenger_margin is None else round(challenger_margin, 6)
        ),
        document_length_ratio_to_average=(
            None if length_ratio is None else round(length_ratio, 6)
        ),
        normalization_view_id=config.normalization_view_id,
        baseline_rank_bucket=_rank_bucket(baseline_rank),
        challenger_rank_bucket=_rank_bucket(challenger_rank),
        score_margin_bucket=_margin_bucket(baseline_margin),
        document_length_bucket=_length_bucket(length_ratio),
        promotion_reason_code=_promotion_reason_code(
            baseline_rank=baseline_rank,
            challenger_rank=challenger_rank,
            baseline_margin=baseline_margin,
            length_ratio=length_ratio,
            selected_for_promotion=selected_for_promotion,
        ),
    )


def _empty_accumulator(
    *,
    split: str,
    config_id: str,
    normalization_view_id: str,
    total_questions: int,
) -> dict[str, Any]:
    return {
        "split": split,
        "config_id": config_id,
        "normalization_view_id": normalization_view_id,
        "total_questions": total_questions,
        "evaluated_questions": 0,
        "hit_counts": {},
        "search_depth_hit_count": 0,
        "rank_11_to_50_count": 0,
        "reciprocal_rank_sum_at_10": 0.0,
        "reciprocal_rank_sum_at_search_depth": 0.0,
        "ranks_by_sample_id": {},
        "empty_query_count": 0,
        "query_token_counts": [],
        "score_margin_gate_promotion_count": 0,
        "length_band_gate_count": 0,
        "signal_features_by_sample_id": {},
    }


def _record_rank(
    *,
    accumulator: dict[str, Any],
    sample: PrimeQAHybridSplitSample,
    rank: int | None,
    query_token_count: int,
    evidence: _ScoreMarginEvidence,
    top_k_values: tuple[int, ...],
    search_depth: int,
    promotion_applied: bool = False,
    length_band_gate_applied: bool = False,
) -> None:
    accumulator["evaluated_questions"] += 1
    accumulator["ranks_by_sample_id"][sample.sample_id] = rank
    accumulator["empty_query_count"] += query_token_count == 0
    accumulator["query_token_counts"].append(query_token_count)
    accumulator["signal_features_by_sample_id"][sample.sample_id] = {
        "normalization_view_id": evidence.normalization_view_id,
        "baseline_rank_bucket": evidence.baseline_rank_bucket,
        "challenger_rank_bucket": evidence.challenger_rank_bucket,
        "score_margin_bucket": evidence.score_margin_bucket,
        "document_length_bucket": evidence.document_length_bucket,
        "promotion_reason_code": evidence.promotion_reason_code,
    }
    accumulator["score_margin_gate_promotion_count"] += promotion_applied
    accumulator["length_band_gate_count"] += length_band_gate_applied
    for top_k in top_k_values:
        accumulator["hit_counts"].setdefault(top_k, 0)
        if rank is not None and rank <= top_k:
            accumulator["hit_counts"][top_k] += 1
    if rank is not None and rank <= search_depth:
        accumulator["search_depth_hit_count"] += 1
        accumulator["reciprocal_rank_sum_at_search_depth"] += 1 / rank
        if rank <= _PRIMARY_TOP_K:
            accumulator["reciprocal_rank_sum_at_10"] += 1 / rank
        elif rank <= search_depth:
            accumulator["rank_11_to_50_count"] += 1


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
        "normalization_view_id": accumulator["normalization_view_id"],
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
        "score_margin_gate_promotion_count": int(
            accumulator["score_margin_gate_promotion_count"]
        ),
        "length_band_gate_count": int(accumulator["length_band_gate_count"]),
        "ranks_by_sample_id": accumulator["ranks_by_sample_id"],
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
        "score_margin_gate_promotion_count": int(
            challenger["score_margin_gate_promotion_count"]
        ),
        "length_band_gate_count": int(challenger["length_band_gate_count"]),
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
    candidate_configs: Sequence[ScoreMarginBM25Config],
) -> dict[str, Any]:
    config_map = {config.config_id: config for config in candidate_configs}
    candidate_ids = [config.config_id for config in candidate_configs]
    train_results = rank_tables[_TRAIN_SPLIT]
    train_comparisons = comparisons[_TRAIN_SPLIT]
    selected_config_id = sorted(
        candidate_ids,
        key=lambda config_id: (
            -float(train_results[config_id]["hit_at_k"][_PRIMARY_TOP_K]),
            int(train_results[config_id]["rank_11_to_50_count"]),
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
    stage75_report: Mapping[str, Any],
    stage94_report: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_configs: Sequence[ScoreMarginBM25Config],
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
    decision = stage94_report.get("decision") or {}
    train_selection_rule = protocol.get("train_selection_rule") or {}
    historical_signal_policy = protocol.get("historical_signal_policy") or {}
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
            name="source_stage94_report_is_stage94",
            passed=stage94_report.get("stage") == "Stage 94",
            observed=stage94_report.get("stage"),
            expected="Stage 94",
        ),
        _check(
            name="stage94_protocol_id_matches",
            passed=protocol.get("protocol_id") == _PROTOCOL_ID,
            observed=protocol.get("protocol_id"),
            expected=_PROTOCOL_ID,
        ),
        _check(
            name="stage94_candidate_id_matches",
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
            name="stage94_allows_train_dev_metrics_after_confirmation",
            passed=_stage94_allows_metric_run(stage94_report),
            observed=decision.get("can_run_train_dev_metrics_after_user_confirmation"),
            expected=True,
        ),
        _check(
            name="stage94_final_test_metrics_locked",
            passed=decision.get("can_run_final_test_metrics_now") is False,
            observed=decision.get("can_run_final_test_metrics_now"),
            expected=False,
        ),
        _check(
            name="stage94_forbids_test_tuning",
            passed=decision.get("can_use_test_for_tuning") is False,
            observed=decision.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage94_default_runtime_policy_unchanged",
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
            name="stage94_train_selection_rule_forbids_dev_selection",
            passed=train_selection_rule.get("dev_selection_forbidden") is True,
            observed=train_selection_rule,
            expected="dev selection forbidden",
        ),
        _check(
            name="stage94_train_selection_rule_forbids_stage82_dev_selection",
            passed=train_selection_rule.get("stage82_dev_observation_selection_forbidden")
            is True,
            observed=train_selection_rule,
            expected="Stage82 dev observation selection forbidden",
        ),
        _check(
            name="historical_stage82_signal_is_motivation_only",
            passed=historical_signal_policy.get(
                "dev_only_b095_observation_can_select_runtime_rule"
            )
            is False,
            observed=historical_signal_policy,
            expected=False,
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
            name="answer_doc_ids_not_used_as_runtime_evidence",
            passed=True,
            observed="evaluation_label_only",
            expected="evaluation_label_only",
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
                    "query_terms",
                    "matched_token_strings",
                ]
            ),
            observed=public_fields,
            expected="no raw question, answer, document, query-term, or token fields",
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
    candidate_configs: Sequence[ScoreMarginBM25Config],
) -> dict[str, Any]:
    failed_checks = [check["name"] for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_score_margin_bm25_comparison_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        }
    selected_config_id = str(train_selection["selected_config_id"])
    selected_dev_comparison = comparisons[_DEV_SPLIT][selected_config_id]
    dev_hit10_delta = float(selected_dev_comparison["hit@10_delta"])
    rank_11_to_50_delta = int(selected_dev_comparison["rank_11_to_50_count_delta"])
    primary_contract_passed = dev_hit10_delta > 0
    secondary_contract_passed = rank_11_to_50_delta < 0
    guard_contract_passed = True
    if primary_contract_passed and secondary_contract_passed and guard_contract_passed:
        recommended_next_stage = (
            "Stage 96: review score-margin BM25 changed cases and runtime risk "
            "before any runtime experiment; keep test locked."
        )
    else:
        recommended_next_stage = (
            "Stage 96: stop score-margin BM25 normalization as a retrieval-recall "
            "route unless a new train/dev-only protocol is explicitly confirmed; "
            "keep test locked and move to the next confirmed second-wave candidate."
        )
    return {
        "status": "primeqa_hybrid_score_margin_bm25_comparison_completed",
        "selected_config_id": selected_config_id,
        "selected_dev_hit10_delta": dev_hit10_delta,
        "selected_dev_rank_11_to_50_count_delta": rank_11_to_50_delta,
        "selected_dev_top10_improvements": int(
            selected_dev_comparison["top10_improvement_count"]
        ),
        "selected_dev_top10_regressions": int(
            selected_dev_comparison["top10_regression_count"]
        ),
        "selected_dev_rank_up_within_top10": int(
            selected_dev_comparison["rank_up_within_top10_count"]
        ),
        "selected_dev_rank_down_within_top10": int(
            selected_dev_comparison["rank_down_within_top10_count"]
        ),
        "selected_dev_not_found_at_search_depth_delta": int(
            selected_dev_comparison["not_found_count_at_search_depth_delta"]
        ),
        "selected_dev_score_margin_gate_promotion_count": int(
            selected_dev_comparison["score_margin_gate_promotion_count"]
        ),
        "selected_dev_length_band_gate_count": int(
            selected_dev_comparison["length_band_gate_count"]
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
        "score_margin_gate_promotion_count": int(
            config_result["score_margin_gate_promotion_count"]
        ),
        "length_band_gate_count": int(config_result["length_band_gate_count"]),
    }


def _public_candidate_config(config: ScoreMarginBM25Config) -> dict[str, Any]:
    return {
        "config_id": config.config_id,
        "normalization_view_id": config.normalization_view_id,
        "challenger_bm25_k1": config.challenger_bm25_k1,
        "challenger_bm25_b": config.challenger_bm25_b,
        "eligible_baseline_rank_min": config.eligible_baseline_rank_min,
        "eligible_baseline_rank_max": config.eligible_baseline_rank_max,
        "challenger_rank_max": config.challenger_rank_max,
        "maximum_score_margin_to_rank10": config.maximum_score_margin_to_rank10,
        "length_gate_mode": config.length_gate_mode,
        "minimum_document_length_ratio_to_average": (
            config.minimum_document_length_ratio_to_average
        ),
        "maximum_document_length_ratio_to_average": (
            config.maximum_document_length_ratio_to_average
        ),
        "maximum_top10_promotions_per_query": (
            config.maximum_top10_promotions_per_query
        ),
        "protected_bm25_top_rank_count": config.protected_bm25_top_rank_count,
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
        "normalization_view_id": signal_features.get("normalization_view_id"),
        "baseline_rank_bucket": signal_features.get("baseline_rank_bucket"),
        "challenger_rank_bucket": signal_features.get("challenger_rank_bucket"),
        "score_margin_bucket": signal_features.get("score_margin_bucket"),
        "document_length_bucket": signal_features.get("document_length_bucket"),
        "promotion_reason_code": signal_features.get("promotion_reason_code"),
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


def _rank_11_to_50_delta_bars(
    report: Mapping[str, Any],
    *,
    split: str,
) -> list[BarDatum]:
    comparisons = report.get("comparisons_to_baseline", {}).get(split, {})
    return [
        BarDatum(
            label=config_id,
            value=float(comparison["rank_11_to_50_count_delta"]),
            value_label=str(comparison["rank_11_to_50_count_delta"]),
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


def _gate_action_bars(report: Mapping[str, Any], *, split: str) -> list[BarDatum]:
    metrics = report.get("metrics_by_split", {}).get(split, {})
    return [
        BarDatum(
            label=config_id,
            value=float(config_metrics["score_margin_gate_promotion_count"]),
            value_label=str(config_metrics["score_margin_gate_promotion_count"]),
        )
        for config_id, config_metrics in sorted(metrics.items())
        if config_id != _BASELINE_CONFIG_ID
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


def _selection_rule_description() -> str:
    return (
        "Select the score-margin BM25 config on train only by hit@10, then "
        "fewer rank 11-50 near misses, then fewer top10 regressions, then "
        "hit@5, hit@1, MRR@10, lower top10 promotion budget, then config_id; "
        "dev is validation only."
    )


def _stage94_allows_metric_run(stage94_report: Mapping[str, Any]) -> bool:
    decision = stage94_report.get("decision") or {}
    return (
        stage94_report.get("stage") == "Stage 94"
        and decision.get("status") == "primeqa_hybrid_score_margin_bm25_protocol_frozen"
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


def _document_index_summary(documents: Sequence[PrimeQADocument]) -> dict[str, Any]:
    token_counts = [
        len(tokenize_text(f"{document.title}\n\n{document.text}"))
        for document in documents
    ]
    return {
        "document_count": len(documents),
        "average_document_token_count": _rounded_mean(token_counts),
    }


def _answer_rank(*, ranked_doc_ids: Sequence[str], answer_doc_id: str) -> int | None:
    for rank, document_id in enumerate(ranked_doc_ids, start=1):
        if document_id == answer_doc_id:
            return rank
    return None


def _rank10_score(document_view: _DocumentScoreView) -> float | None:
    if len(document_view.ranked) < _PRIMARY_TOP_K:
        return None
    return document_view.ranked[_PRIMARY_TOP_K - 1].score


def _score_margin_to_rank10(
    *,
    baseline_view: _DocumentScoreView,
    document_id: str,
    rank10_score: float | None,
) -> float | None:
    if rank10_score is None:
        return None
    score = baseline_view.score_by_doc_id.get(document_id)
    if score is None:
        return None
    return rank10_score - score


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


def _margin_bucket(margin: float | None) -> str:
    if margin is None:
        return "margin_not_available"
    if margin <= 0.02:
        return "margin_lte_0_02"
    if margin <= 0.03:
        return "margin_0_02_to_0_03"
    if margin <= 0.04:
        return "margin_0_03_to_0_04"
    if margin <= 0.05:
        return "margin_0_04_to_0_05"
    return "margin_gt_0_05"


def _length_bucket(length_ratio: float | None) -> str:
    if length_ratio is None:
        return "length_not_available"
    if length_ratio <= 0.75:
        return "length_lte_0_75"
    if length_ratio <= 0.85:
        return "length_0_75_to_0_85"
    if length_ratio < 1.2:
        return "length_mid_band"
    if length_ratio < 1.35:
        return "length_1_20_to_1_35"
    if length_ratio < 1.5:
        return "length_1_35_to_1_50"
    return "length_gte_1_50"


def _promotion_reason_code(
    *,
    baseline_rank: int | None,
    challenger_rank: int | None,
    baseline_margin: float | None,
    length_ratio: float | None,
    selected_for_promotion: bool,
) -> str:
    prefix = "promoted" if selected_for_promotion else "not_promoted"
    return "|".join(
        [
            prefix,
            _rank_bucket(baseline_rank),
            _rank_bucket(challenger_rank),
            _margin_bucket(baseline_margin),
            _length_bucket(length_ratio),
        ]
    )


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
    return idf * numerator / np.maximum(denominator, _EPSILON)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _rounded_ratio(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _rounded_ratio_float(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _rounded_mean(values: Sequence[int | float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
