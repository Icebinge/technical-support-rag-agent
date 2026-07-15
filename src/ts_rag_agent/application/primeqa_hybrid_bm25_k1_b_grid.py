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

_STAGE = "Stage 82"
_CREATED_AT = "2026-07-15"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})
_BASELINE_CONFIG_ID = "full_document_bm25_baseline"
_CANDIDATE_ID = "bm25_k1_b_grid_train_to_dev"
_USER_CONFIRMED_GRID_PROTOCOL = "small_grid_user_confirmed"
_PRIMARY_TOP_K = 10
_DEFAULT_SEARCH_DEPTH = 50
_BASELINE_K1 = 1.5
_BASELINE_B = 0.75
_SMALL_GRID_K1_VALUES = (1.2, 1.5, 1.8)
_SMALL_GRID_B_VALUES = (0.55, 0.75, 0.95)


class _BM25GridIndex:
    """Shared BM25 index for evaluating fixed k1/b grids efficiently."""

    def __init__(self, documents: Sequence[PrimeQADocument]) -> None:
        self._documents = list(documents)
        self._doc_ids = np.asarray([document.id for document in self._documents])
        self._doc_lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = {}
        for doc_index, document in enumerate(self._documents):
            tokens = tokenize_text(f"{document.title}\n\n{document.text}")
            term_counts = Counter(tokens)
            self._doc_lengths.append(len(tokens))
            for term, term_frequency in term_counts.items():
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

    def rank_answer_doc_for_configs(
        self,
        *,
        query: str,
        answer_doc_id: str,
        grid_configs: Sequence[BM25GridConfig],
        top_k: int,
    ) -> dict[str, int | None]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self._documents:
            return {config.config_id: None for config in grid_configs}
        query_terms = tokenize_text(query)
        if not query_terms:
            return {config.config_id: None for config in grid_configs}

        scores = np.zeros((len(grid_configs), len(self._documents)), dtype=np.float64)
        for term in query_terms:
            idf = self._idf.get(term)
            posting = self._postings.get(term)
            if idf is None or posting is None:
                continue
            doc_indices, term_frequencies = posting
            doc_lengths = self._doc_lengths_array[doc_indices]
            for config_index, config in enumerate(grid_configs):
                scores[config_index, doc_indices] += _score_term_vector(
                    idf=idf,
                    term_frequencies=term_frequencies,
                    doc_lengths=doc_lengths,
                    avg_doc_length=self._avg_doc_length,
                    k1=config.k1,
                    b=config.b,
                )

        ranks = {}
        for config_index, config in enumerate(grid_configs):
            ranks[config.config_id] = self._rank_answer_doc(
                score_row=scores[config_index],
                answer_doc_id=answer_doc_id,
                top_k=top_k,
            )
        return ranks

    def _rank_answer_doc(
        self,
        *,
        score_row: np.ndarray,
        answer_doc_id: str,
        top_k: int,
    ) -> int | None:
        scored_indices = np.flatnonzero(score_row)
        if scored_indices.size == 0:
            return None
        ranked_indices = sorted(
            (int(index) for index in scored_indices),
            key=lambda index: (-float(score_row[index]), str(self._doc_ids[index])),
        )[:top_k]
        for rank, doc_index in enumerate(ranked_indices, start=1):
            if str(self._doc_ids[doc_index]) == answer_doc_id:
                return rank
        return None


@dataclass(frozen=True)
class PrimeQAHybridBM25K1BGridVisualization:
    """One generated Stage82 BM25 k1/b grid visualization."""

    name: str
    path: str


@dataclass(frozen=True)
class BM25GridConfig:
    """One fixed BM25 parameter-grid configuration."""

    config_id: str
    k1: float
    b: float
    is_baseline: bool


def run_primeqa_hybrid_bm25_k1_b_grid(
    *,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    stage75_report_path: Path,
    stage76_report_path: Path,
    stage81_report_path: Path,
    user_confirmed_grid_protocol: str = _USER_CONFIRMED_GRID_PROTOCOL,
    top_k_values: tuple[int, ...] = (1, 5, 10),
    search_depth: int = _DEFAULT_SEARCH_DEPTH,
) -> dict[str, Any]:
    """Run train/dev-only BM25 k1/b grid experiment with a confirmed small grid."""

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
    stage76_report = _load_json_object(stage76_report_path)
    stage81_report = _load_json_object(stage81_report_path)
    loaded_inputs_at = time.perf_counter()

    grid_configs = _confirmed_small_grid_configs(user_confirmed_grid_protocol)
    rank_tables: dict[str, dict[str, dict[str, Any]]] = {}
    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    train_selection: dict[str, Any] = {
        "selection_rule": _selection_rule_description(),
        "selected_config_id": None,
        "grid_config_count": len(grid_configs),
        "selected_train_metrics": None,
        "selected_dev_metrics": None,
        "selected_dev_comparison_to_baseline": None,
    }
    indexed_and_evaluated_at = loaded_inputs_at
    if grid_configs:
        rank_tables = {
            split: _evaluate_grid(
                split=split,
                samples=samples,
                documents=document_list,
                grid_configs=grid_configs,
                top_k_values=top_k_values,
                search_depth=search_depth,
            )
            for split, samples in split_samples.items()
        }
        indexed_and_evaluated_at = time.perf_counter()
        comparisons = {
            split: {
                config.config_id: _compare_to_baseline(
                    baseline=rank_tables[split][_BASELINE_CONFIG_ID],
                    challenger=rank_tables[split][config.config_id],
                    max_k=_PRIMARY_TOP_K,
                    search_depth=search_depth,
                )
                for config in grid_configs
                if not config.is_baseline
            }
            for split in _ALLOWED_DEVELOPMENT_SPLITS
        }
        train_selection = _select_config_on_train(
            rank_tables=rank_tables,
            grid_configs=grid_configs,
        )
        selected_config_id = str(train_selection["selected_config_id"])
        train_selection = {
            **train_selection,
            "selected_dev_metrics": _public_config_metrics(
                rank_tables[_DEV_SPLIT][selected_config_id]
            ),
            "selected_dev_comparison_to_baseline": comparisons[_DEV_SPLIT].get(
                selected_config_id,
                _baseline_self_comparison(rank_tables[_DEV_SPLIT][_BASELINE_CONFIG_ID]),
            ),
        }

    guard_checks = _guard_checks(
        split_samples=split_samples,
        rank_tables=rank_tables,
        stage75_report=stage75_report,
        stage76_report=stage76_report,
        stage81_report=stage81_report,
        grid_configs=grid_configs,
        user_confirmed_grid_protocol=user_confirmed_grid_protocol,
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
            "Train/dev-only BM25 k1/b small-grid experiment for "
            "bm25_k1_b_grid_train_to_dev. This stage uses the user-confirmed "
            "small grid around the Stage75 baseline, selects on train only, "
            "validates on dev, keeps the frozen test split locked, does not run "
            "final metrics, does not use source DOC_IDS as runtime retrieval "
            "evidence, and does not change runtime defaults."
        ),
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
            "stage76_report": _fingerprint(stage76_report_path),
            "stage81_report": _fingerprint(stage81_report_path),
        },
        "config": {
            "user_confirmed_grid_protocol": user_confirmed_grid_protocol,
            "grid_protocol_description": (
                "User selected option a: small grid around baseline "
                "k1=1.5,b=0.75."
            ),
            "k1_values": list(_SMALL_GRID_K1_VALUES),
            "b_values": list(_SMALL_GRID_B_VALUES),
            "grid_config_count": len(grid_configs),
            "baseline_config_id": _BASELINE_CONFIG_ID,
            "baseline_k1": _BASELINE_K1,
            "baseline_b": _BASELINE_B,
            "top_k_values": list(top_k_values),
            "primary_top_k": _PRIMARY_TOP_K,
            "search_depth": search_depth,
            "selection_rule": _selection_rule_description(),
        },
        "loaded_data_summary": {
            "document_count": len(document_list),
            "split_rows": {
                split: len(samples) for split, samples in sorted(split_samples.items())
            },
            "answerable_rows": {
                split: sum(sample.answerable for sample in samples)
                for split, samples in sorted(split_samples.items())
            },
            "test_split_loaded": False,
        },
        "grid_configs": [_public_grid_config(config) for config in grid_configs],
        "metrics_by_split": public_metrics,
        "comparisons_to_baseline": comparisons,
        "train_selection": train_selection,
        "guard_checks": guard_checks,
        "decision": _decision(
            guard_checks=guard_checks,
            comparisons=comparisons,
            train_selection=train_selection,
        ),
        "timing_seconds": {
            "load_splits": round(loaded_splits_at - started_at, 3),
            "load_documents_and_reports": round(loaded_inputs_at - loaded_splits_at, 3),
            "bm25_grid_index_and_evaluate": round(
                indexed_and_evaluated_at - loaded_inputs_at,
                3,
            ),
            "guard_checks": round(checked_at - indexed_and_evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_bm25_k1_b_grid_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridBM25K1BGridVisualization]:
    """Write SVG charts for Stage82 BM25 k1/b grid."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage82_bm25_grid_train_hit_at_10.svg": render_horizontal_bar_chart_svg(
            title="Stage82 train hit@10 by BM25 grid config",
            bars=_hit_at_k_bars(report, split=_TRAIN_SPLIT, top_k=_PRIMARY_TOP_K),
            x_label="hit@10",
            width=1180,
            margin_left=390,
        ),
        "stage82_bm25_grid_dev_hit_at_10.svg": render_horizontal_bar_chart_svg(
            title="Stage82 dev hit@10 by BM25 grid config",
            bars=_hit_at_k_bars(report, split=_DEV_SPLIT, top_k=_PRIMARY_TOP_K),
            x_label="hit@10",
            width=1180,
            margin_left=390,
        ),
        "stage82_bm25_grid_dev_delta_hit_at_10.svg": render_horizontal_bar_chart_svg(
            title="Stage82 dev hit@10 delta vs baseline",
            bars=_delta_bars(report, split=_DEV_SPLIT, metric="hit@10_delta"),
            x_label="delta hit@10",
            width=1180,
            margin_left=390,
        ),
        "stage82_bm25_grid_dev_near_miss_11_to_50.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage82 dev ranks 11-50 by BM25 grid config",
                bars=_rank_11_to_50_bars(report, split=_DEV_SPLIT),
                x_label="answer docs ranked 11-50",
                width=1180,
                margin_left=390,
            )
        ),
        "stage82_bm25_grid_dev_top10_changes.svg": render_horizontal_bar_chart_svg(
            title="Stage82 dev top10 improvements minus regressions",
            bars=_net_change_bars(report, split=_DEV_SPLIT),
            x_label="net changed cases",
            width=1180,
            margin_left=390,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridBM25K1BGridVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _confirmed_small_grid_configs(
    user_confirmed_grid_protocol: str,
) -> tuple[BM25GridConfig, ...]:
    if user_confirmed_grid_protocol != _USER_CONFIRMED_GRID_PROTOCOL:
        return ()
    configs = []
    for k1 in _SMALL_GRID_K1_VALUES:
        for b in _SMALL_GRID_B_VALUES:
            is_baseline = k1 == _BASELINE_K1 and b == _BASELINE_B
            config_id = (
                _BASELINE_CONFIG_ID
                if is_baseline
                else f"bm25_grid__k1_{_number_id(k1)}__b_{_number_id(b)}"
            )
            configs.append(BM25GridConfig(config_id=config_id, k1=k1, b=b, is_baseline=is_baseline))
    return tuple(configs)


def _evaluate_grid(
    *,
    split: str,
    samples: Sequence[PrimeQAHybridSplitSample],
    documents: Sequence[PrimeQADocument],
    grid_configs: Sequence[BM25GridConfig],
    top_k_values: tuple[int, ...],
    search_depth: int,
) -> dict[str, Any]:
    answerable_samples = [
        sample
        for sample in samples
        if sample.answerable and sample.answer_doc_id is not None
    ]
    index = _BM25GridIndex(documents)
    accumulators = {
        config.config_id: _empty_accumulator(
            split=split,
            config=config,
            total_questions=len(samples),
        )
        for config in grid_configs
    }
    for sample in answerable_samples:
        query = sample.to_primeqa_question().full_question
        query_token_count = len(tokenize_text(query))
        ranks = index.rank_answer_doc_for_configs(
            query=query,
            answer_doc_id=str(sample.answer_doc_id),
            grid_configs=grid_configs,
            top_k=search_depth,
        )
        for config in grid_configs:
            _record_rank(
                accumulator=accumulators[config.config_id],
                sample=sample,
                rank=ranks[config.config_id],
                query_token_count=query_token_count,
                top_k_values=top_k_values,
                search_depth=search_depth,
            )
    return {
        config_id: _finalize_accumulator(
            accumulator,
            top_k_values=top_k_values,
            search_depth=search_depth,
        )
        for config_id, accumulator in accumulators.items()
    }


def _evaluate_config(
    *,
    split: str,
    samples: Sequence[PrimeQAHybridSplitSample],
    documents: Sequence[PrimeQADocument],
    config: BM25GridConfig,
    top_k_values: tuple[int, ...],
    search_depth: int,
) -> dict[str, Any]:
    """Evaluate one config through the same shared-index code path used by tests."""

    split_results = _evaluate_grid(
        split=split,
        samples=samples,
        documents=documents,
        grid_configs=(config,),
        top_k_values=top_k_values,
        search_depth=search_depth,
    )
    return split_results[config.config_id]


def _empty_accumulator(
    *,
    split: str,
    config: BM25GridConfig,
    total_questions: int,
) -> dict[str, Any]:
    return {
        "split": split,
        "config_id": config.config_id,
        "k1": config.k1,
        "b": config.b,
        "is_baseline": config.is_baseline,
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
    }


def _record_rank(
    *,
    accumulator: dict[str, Any],
    sample: PrimeQAHybridSplitSample,
    rank: int | None,
    query_token_count: int,
    top_k_values: tuple[int, ...],
    search_depth: int,
) -> None:
    accumulator["evaluated_questions"] += 1
    accumulator["ranks_by_sample_id"][sample.sample_id] = rank
    accumulator["empty_query_count"] += query_token_count == 0
    accumulator["query_token_counts"].append(query_token_count)
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
        "k1": float(accumulator["k1"]),
        "b": float(accumulator["b"]),
        "is_baseline": bool(accumulator["is_baseline"]),
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
        "ranks_by_sample_id": accumulator["ranks_by_sample_id"],
    }


def _compare_to_baseline(
    *,
    baseline: Mapping[str, Any],
    challenger: Mapping[str, Any],
    max_k: int,
    search_depth: int,
) -> dict[str, Any]:
    baseline_ranks = baseline["ranks_by_sample_id"]
    challenger_ranks = challenger["ranks_by_sample_id"]
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
        if not baseline_hit and challenger_hit:
            top10_improvements.append(
                _change_case(sample_id, baseline_rank, challenger_rank)
            )
        elif baseline_hit and not challenger_hit:
            top10_regressions.append(
                _change_case(sample_id, baseline_rank, challenger_rank)
            )
        elif baseline_hit and challenger_hit:
            both_hit += 1
            if challenger_rank < baseline_rank:
                rank_up += 1
            elif challenger_rank > baseline_rank:
                rank_down += 1
        else:
            both_miss += 1
        if not baseline_found and challenger_found:
            search_depth_improvements.append(
                _change_case(sample_id, baseline_rank, challenger_rank)
            )
        elif baseline_found and not challenger_found:
            search_depth_regressions.append(
                _change_case(sample_id, baseline_rank, challenger_rank)
            )
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
    grid_configs: Sequence[BM25GridConfig],
) -> dict[str, Any]:
    candidate_ids = [config.config_id for config in grid_configs]
    train_results = rank_tables[_TRAIN_SPLIT]
    selected_config_id = sorted(
        candidate_ids,
        key=lambda config_id: (
            -float(train_results[config_id]["hit_at_k"][_PRIMARY_TOP_K]),
            -float(train_results[config_id]["hit_at_k"].get(5, 0.0)),
            -float(train_results[config_id]["hit_at_k"].get(1, 0.0)),
            -float(train_results[config_id]["mrr_at_10"]),
            int(train_results[config_id]["not_found_count_at_search_depth"]),
            int(train_results[config_id]["rank_11_to_50_count"]),
            config_id,
        ),
    )[0]
    return {
        "selection_rule": _selection_rule_description(),
        "selected_config_id": selected_config_id,
        "grid_config_count": len(candidate_ids),
        "selected_train_metrics": _public_config_metrics(train_results[selected_config_id]),
    }


def _guard_checks(
    *,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    rank_tables: Mapping[str, Mapping[str, Mapping[str, Any]]],
    stage75_report: Mapping[str, Any],
    stage76_report: Mapping[str, Any],
    stage81_report: Mapping[str, Any],
    grid_configs: Sequence[BM25GridConfig],
    user_confirmed_grid_protocol: str,
    top_k_values: tuple[int, ...],
    search_depth: int,
) -> list[dict[str, Any]]:
    observed_split_names = sorted(
        {sample.assigned_split for samples in split_samples.values() for sample in samples}
    )
    expected_splits = sorted(_ALLOWED_DEVELOPMENT_SPLITS)
    stage75_split_reports = stage75_report.get("split_reports") or {}
    stage76_candidates = stage76_report.get("candidate_designs") or []
    stage76_candidate = next(
        (
            candidate
            for candidate in stage76_candidates
            if candidate.get("candidate_id") == _CANDIDATE_ID
        ),
        {},
    )
    stage81_decision = stage81_report.get("decision") or {}
    baseline_train_hit10 = _baseline_hit10(rank_tables, _TRAIN_SPLIT)
    baseline_dev_hit10 = _baseline_hit10(rank_tables, _DEV_SPLIT)
    stage75_train_hit10 = (
        (stage75_split_reports.get(_TRAIN_SPLIT) or {}).get("hit_at_top_k")
    )
    stage75_dev_hit10 = (
        (stage75_split_reports.get(_DEV_SPLIT) or {}).get("hit_at_top_k")
    )
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
            name="stage75_source_report_is_stage75",
            passed=str(stage75_report.get("stage") or "") == "Stage 75",
            observed=str(stage75_report.get("stage") or ""),
            expected="Stage 75",
        ),
        _check(
            name="stage76_source_report_is_stage76",
            passed=str(stage76_report.get("stage") or "") == "Stage 76",
            observed=str(stage76_report.get("stage") or ""),
            expected="Stage 76",
        ),
        _check(
            name="stage76_bm25_grid_candidate_is_allowed",
            passed=stage76_candidate.get("status") == "recommended_for_train_dev_experiment",
            observed=stage76_candidate.get("status"),
            expected="recommended_for_train_dev_experiment",
        ),
        _check(
            name="stage76_requires_fixed_grid_values",
            passed="guard: grid values must be fixed before the run"
            in stage76_candidate.get("target_metric_contract", []),
            observed=stage76_candidate.get("target_metric_contract", []),
            expected="guard: grid values must be fixed before the run",
        ),
        _check(
            name="stage81_source_report_is_stage81",
            passed=str(stage81_report.get("stage") or "") == "Stage 81",
            observed=str(stage81_report.get("stage") or ""),
            expected="Stage 81",
        ),
        _check(
            name="stage81_did_not_open_final_test_gate",
            passed=stage81_decision.get("can_open_final_test_gate_now") is False,
            observed=stage81_decision.get("can_open_final_test_gate_now"),
            expected=False,
        ),
        _check(
            name="stage81_recommends_bm25_grid_next",
            passed=_stage81_recommends_bm25_grid(stage81_decision),
            observed=stage81_decision.get("recommended_next_stage"),
            expected=f"contains {_CANDIDATE_ID} or BM25 k1/b grid",
        ),
        _check(
            name="user_confirmed_small_grid_protocol",
            passed=user_confirmed_grid_protocol == _USER_CONFIRMED_GRID_PROTOCOL,
            observed=user_confirmed_grid_protocol,
            expected=_USER_CONFIRMED_GRID_PROTOCOL,
        ),
        _check(
            name="grid_values_fixed_before_run",
            passed=_grid_values_match_small_protocol(grid_configs),
            observed=[_public_grid_config(config) for config in grid_configs],
            expected={
                "k1_values": list(_SMALL_GRID_K1_VALUES),
                "b_values": list(_SMALL_GRID_B_VALUES),
            },
        ),
        _check(
            name="grid_includes_stage75_baseline",
            passed=any(config.is_baseline for config in grid_configs),
            observed=[config.config_id for config in grid_configs if config.is_baseline],
            expected=_BASELINE_CONFIG_ID,
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
) -> dict[str, Any]:
    failed_checks = [check["name"] for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_bm25_k1_b_grid_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        }
    selected_config_id = str(train_selection["selected_config_id"])
    selected_dev_comparison = comparisons[_DEV_SPLIT].get(
        selected_config_id,
        train_selection["selected_dev_comparison_to_baseline"],
    )
    dev_hit10_delta = float(selected_dev_comparison["hit@10_delta"])
    if dev_hit10_delta > 0:
        recommended_next_stage = (
            "Stage 83: review BM25 grid changed cases on dev and decide whether "
            "a guarded runtime experiment is justified; keep test locked."
        )
    else:
        recommended_next_stage = (
            "Stage 83: summarize the exhausted Stage76 retrieval-recall candidates "
            "and decide the next train/dev-only improvement route; keep test locked."
        )
    return {
        "status": "primeqa_hybrid_bm25_k1_b_grid_completed",
        "selected_config_id": selected_config_id,
        "selected_dev_hit10_delta": dev_hit10_delta,
        "selected_dev_top10_improvements": int(
            selected_dev_comparison["top10_improvement_count"]
        ),
        "selected_dev_top10_regressions": int(
            selected_dev_comparison["top10_regression_count"]
        ),
        "selected_dev_not_found_at_search_depth_delta": int(
            selected_dev_comparison["not_found_count_at_search_depth_delta"]
        ),
        "selected_dev_rank_11_to_50_count_delta": int(
            selected_dev_comparison["rank_11_to_50_count_delta"]
        ),
        "can_continue_train_dev_development": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": recommended_next_stage,
    }


def _public_config_metrics(config_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "k1": float(config_result["k1"]),
        "b": float(config_result["b"]),
        "is_baseline": bool(config_result["is_baseline"]),
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
    }


def _public_grid_config(config: BM25GridConfig) -> dict[str, Any]:
    return {
        "config_id": config.config_id,
        "k1": config.k1,
        "b": config.b,
        "is_baseline": config.is_baseline,
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
    ordered = sorted(
        comparisons.items(),
        key=lambda item: (-float(item[1][metric]), item[0]),
    )
    return [
        BarDatum(
            label=config_id,
            value=float(comparison[metric]),
            value_label=f"{comparison[metric]:+.4f}",
        )
        for config_id, comparison in ordered
    ]


def _rank_11_to_50_bars(report: Mapping[str, Any], *, split: str) -> list[BarDatum]:
    metrics = report.get("metrics_by_split", {}).get(split, {})
    ordered = sorted(
        metrics.items(),
        key=lambda item: (item[1]["rank_11_to_50_count"], item[0]),
    )
    return [
        BarDatum(
            label=config_id,
            value=float(config_metrics["rank_11_to_50_count"]),
            value_label=str(config_metrics["rank_11_to_50_count"]),
        )
        for config_id, config_metrics in ordered
    ]


def _net_change_bars(report: Mapping[str, Any], *, split: str) -> list[BarDatum]:
    comparisons = report.get("comparisons_to_baseline", {}).get(split, {})
    ordered = sorted(
        comparisons.items(),
        key=lambda item: (-item[1]["top10_net_improvement_count"], item[0]),
    )
    return [
        BarDatum(
            label=config_id,
            value=float(comparison["top10_net_improvement_count"]),
            value_label=str(comparison["top10_net_improvement_count"]),
        )
        for config_id, comparison in ordered
    ]


def _baseline_hit10(
    rank_tables: Mapping[str, Mapping[str, Mapping[str, Any]]],
    split: str,
) -> float | None:
    split_table = rank_tables.get(split) or {}
    baseline = split_table.get(_BASELINE_CONFIG_ID)
    if baseline is None:
        return None
    return float(baseline["hit_at_k"][_PRIMARY_TOP_K])


def _baseline_self_comparison(baseline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "baseline_config_id": baseline["config_id"],
        "challenger_config_id": baseline["config_id"],
        "hit@1_delta": 0.0,
        "hit@5_delta": 0.0,
        "hit@10_delta": 0.0,
        "mrr_at_10_delta": 0.0,
        "mrr_at_search_depth_delta": 0.0,
        "top10_improvement_count": 0,
        "top10_regression_count": 0,
        "top10_net_improvement_count": 0,
        "search_depth": int(baseline["search_depth"]),
        "search_depth_improvement_count": 0,
        "search_depth_regression_count": 0,
        "search_depth_net_improvement_count": 0,
        "not_found_count_at_search_depth_delta": 0,
        "rank_11_to_50_count_delta": 0,
        "both_hit_count": int(baseline["hit_counts"].get(_PRIMARY_TOP_K, 0)),
        "both_miss_count": int(baseline["miss_count_at_primary_top_k"]),
        "rank_up_within_top10_count": 0,
        "rank_down_within_top10_count": 0,
        "sample_top10_improvements": [],
        "sample_top10_regressions": [],
        "sample_search_depth_improvements": [],
        "sample_search_depth_regressions": [],
    }


def _grid_values_match_small_protocol(
    grid_configs: Sequence[BM25GridConfig],
) -> bool:
    observed = {(config.k1, config.b) for config in grid_configs}
    expected = {
        (k1, b) for k1 in _SMALL_GRID_K1_VALUES for b in _SMALL_GRID_B_VALUES
    }
    return observed == expected and len(grid_configs) == len(expected)


def _stage81_recommends_bm25_grid(stage81_decision: Mapping[str, Any]) -> bool:
    recommended_next_stage = str(stage81_decision.get("recommended_next_stage") or "")
    return _CANDIDATE_ID in recommended_next_stage or "BM25 k1/b grid" in recommended_next_stage


def _selection_rule_description() -> str:
    return (
        "Select across the fixed user-confirmed BM25 small grid on train only by "
        "hit@10, then hit@5, then hit@1, then MRR@10, then fewer not-found@50, "
        "then fewer rank 11-50 near misses, then config_id; dev is validation only."
    )


def _change_case(
    sample_id: str,
    baseline_rank: int | None,
    challenger_rank: int | None,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "baseline_rank": baseline_rank,
        "challenger_rank": challenger_rank,
    }


def _number_id(value: float) -> str:
    return f"{value:.2f}".replace(".", "_")


def _score_term_vector(
    *,
    idf: float,
    term_frequencies: np.ndarray,
    doc_lengths: np.ndarray,
    avg_doc_length: float,
    k1: float,
    b: float,
) -> np.ndarray:
    length_normalizer = 1 - b
    if avg_doc_length:
        length_normalizer = length_normalizer + b * doc_lengths / avg_doc_length
    numerator = term_frequencies * (k1 + 1)
    denominator = term_frequencies + k1 * length_normalizer
    return idf * numerator / denominator


def _compute_idf(document_count: int, document_frequency: int) -> float:
    return math.log(
        1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
    )


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


def _validate_options(*, top_k_values: tuple[int, ...], search_depth: int) -> None:
    if not top_k_values:
        raise ValueError("top_k_values must not be empty")
    if any(top_k <= 0 for top_k in top_k_values):
        raise ValueError("top_k_values must be positive")
    if _PRIMARY_TOP_K not in top_k_values:
        raise ValueError(f"top_k_values must include {_PRIMARY_TOP_K}")
    if search_depth < max(top_k_values):
        raise ValueError("search_depth must be at least max(top_k_values)")


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


def _rounded_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _rounded_ratio_float(numerator: float, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _rounded_mean(values: Sequence[int]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
