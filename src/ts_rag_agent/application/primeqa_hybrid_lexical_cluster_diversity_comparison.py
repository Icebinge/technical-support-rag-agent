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

_STAGE = "Stage 86"
_CREATED_AT = "2026-07-15"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_PROTOCOL_ID = "lexical_cluster_diversity_rerank_train_dev_v1"
_CANDIDATE_ID = "lexical_cluster_diversity_rerank_design"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})
_BASELINE_CONFIG_ID = "full_document_bm25_baseline"
_BASELINE_K1 = 1.5
_BASELINE_B = 0.75
_PRIMARY_TOP_K = 10
_DEFAULT_SEARCH_DEPTH = 50
_CLUSTER_HASH_EMPTY = "empty"
_EXPECTED_CONFIG_IDS = (
    "lcdr_penalty_0_03_title_query_cluster",
    "lcdr_penalty_0_06_title_query_cluster",
    "lcdr_penalty_0_09_title_query_cluster",
    "lcdr_penalty_0_12_title_query_cluster",
)


@dataclass(frozen=True)
class PrimeQAHybridLexicalClusterDiversityComparisonVisualization:
    """One generated Stage86 lexical cluster diversity comparison visualization."""

    name: str
    path: str


@dataclass(frozen=True)
class LexicalClusterDiversityConfig:
    """One frozen lexical cluster diversity rerank configuration."""

    config_id: str
    duplicate_penalty_weight: float
    cluster_key: str
    minimum_title_overlap_terms: int
    minimum_cluster_size: int


@dataclass(frozen=True)
class _RankedCandidate:
    document_index: int
    document_id: str
    baseline_rank: int
    baseline_bm25_score: float
    score_margin_to_top1: float
    score_margin_to_previous: float
    query_overlap_count: int
    title_query_overlap_count: int
    document_token_count: int
    lexical_cluster_hash: str
    cluster_duplicate_index: int
    cluster_size_in_candidate_depth: int


class _BM25CandidateIndex:
    """Shared full-document BM25 index with public-safe lexical cluster features."""

    def __init__(self, documents: Sequence[PrimeQADocument]) -> None:
        self._documents = list(documents)
        self._doc_ids = np.asarray([document.id for document in self._documents])
        self._doc_lengths: list[int] = []
        self._document_token_sets: list[set[str]] = []
        self._title_token_sets: list[set[str]] = []
        postings: dict[str, list[tuple[int, int]]] = {}
        for doc_index, document in enumerate(self._documents):
            title_tokens = set(tokenize_text(document.title))
            document_tokens = tokenize_text(f"{document.title}\n\n{document.text}")
            document_token_set = set(document_tokens)
            term_counts = Counter(document_tokens)
            self._title_token_sets.append(title_tokens)
            self._document_token_sets.append(document_token_set)
            self._doc_lengths.append(len(document_tokens))
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

    def rank_candidates(
        self,
        *,
        query: str,
        search_depth: int,
        minimum_title_overlap_terms: int,
        minimum_cluster_size: int,
    ) -> list[_RankedCandidate]:
        if search_depth <= 0:
            raise ValueError("search_depth must be positive")
        if not self._documents:
            return []
        query_terms = tokenize_text(query)
        if not query_terms:
            return []

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
                avg_doc_length=self._avg_doc_length,
                k1=_BASELINE_K1,
                b=_BASELINE_B,
            )

        scored_indices = np.flatnonzero(scores)
        if scored_indices.size == 0:
            return []
        ranked_indices = sorted(
            (int(index) for index in scored_indices),
            key=lambda index: (-float(scores[index]), str(self._doc_ids[index])),
        )[:search_depth]
        top1_score = float(scores[ranked_indices[0]]) if ranked_indices else 0.0
        query_token_set = set(query_terms)
        cluster_hashes = [
            self._cluster_hash(
                doc_index=doc_index,
                query_token_set=query_token_set,
                minimum_title_overlap_terms=minimum_title_overlap_terms,
            )
            for doc_index in ranked_indices
        ]
        cluster_sizes = Counter(
            cluster_hash for cluster_hash in cluster_hashes if cluster_hash != _CLUSTER_HASH_EMPTY
        )
        seen_clusters: Counter[str] = Counter()
        candidates = []
        previous_score = top1_score
        for rank, doc_index in enumerate(ranked_indices, start=1):
            score = float(scores[doc_index])
            cluster_hash = cluster_hashes[rank - 1]
            cluster_size = int(cluster_sizes.get(cluster_hash, 0))
            if cluster_hash == _CLUSTER_HASH_EMPTY or cluster_size < minimum_cluster_size:
                duplicate_index = 0
                effective_cluster_hash = _CLUSTER_HASH_EMPTY
                effective_cluster_size = 0
            else:
                duplicate_index = int(seen_clusters[cluster_hash])
                effective_cluster_hash = cluster_hash
                effective_cluster_size = cluster_size
                seen_clusters[cluster_hash] += 1
            title_overlap = len(self._title_token_sets[doc_index] & query_token_set)
            query_overlap = len(self._document_token_sets[doc_index] & query_token_set)
            candidates.append(
                _RankedCandidate(
                    document_index=doc_index,
                    document_id=str(self._doc_ids[doc_index]),
                    baseline_rank=rank,
                    baseline_bm25_score=score,
                    score_margin_to_top1=round(top1_score - score, 6),
                    score_margin_to_previous=round(previous_score - score, 6)
                    if rank > 1
                    else 0.0,
                    query_overlap_count=query_overlap,
                    title_query_overlap_count=title_overlap,
                    document_token_count=int(self._doc_lengths[doc_index]),
                    lexical_cluster_hash=effective_cluster_hash,
                    cluster_duplicate_index=duplicate_index,
                    cluster_size_in_candidate_depth=effective_cluster_size,
                )
            )
            previous_score = score
        return candidates

    def _cluster_hash(
        self,
        *,
        doc_index: int,
        query_token_set: set[str],
        minimum_title_overlap_terms: int,
    ) -> str:
        overlap_tokens = sorted(self._title_token_sets[doc_index] & query_token_set)
        if len(overlap_tokens) < minimum_title_overlap_terms:
            return _CLUSTER_HASH_EMPTY
        digest = hashlib.sha256("\n".join(overlap_tokens).encode("utf-8")).hexdigest()
        return digest[:16]


def run_primeqa_hybrid_lexical_cluster_diversity_comparison(
    *,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    stage75_report_path: Path,
    stage85_report_path: Path,
    user_confirmed_protocol: bool,
    confirmed_protocol_id: str,
    confirmation_note: str,
    top_k_values: tuple[int, ...] = (1, 5, 10),
    search_depth: int = _DEFAULT_SEARCH_DEPTH,
) -> dict[str, Any]:
    """Run the confirmed train/dev-only lexical cluster diversity rerank comparison."""

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
    stage85_report = _load_json_object(stage85_report_path)
    loaded_inputs_at = time.perf_counter()

    protocol = stage85_report.get("frozen_protocol") or {}
    protocol_configs = _configs_from_protocol(protocol)
    should_evaluate = (
        user_confirmed_protocol
        and confirmed_protocol_id == _PROTOCOL_ID
        and protocol.get("protocol_id") == _PROTOCOL_ID
        and _stage85_allows_metric_run(stage85_report)
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
        rank_tables = {
            split: _evaluate_split(
                split=split,
                samples=samples,
                documents=document_list,
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
        stage85_report=stage85_report,
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
            "lexical_cluster_diversity_rerank_train_dev_v1. This stage uses "
            "the user-confirmed Stage85 protocol, selects a candidate config on "
            "train only, validates on dev, keeps the frozen test split locked, "
            "does not run final metrics, does not use source DOC_IDS as runtime "
            "retrieval evidence, does not write raw question or document text, "
            "and does not change runtime defaults."
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
            "stage85_report": _fingerprint(stage85_report_path),
        },
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
            "cluster_hash_length": 16,
            "raw_tokens_written_to_report": False,
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
        ),
        "timing_seconds": {
            "load_splits": round(loaded_splits_at - started_at, 3),
            "load_documents_and_reports": round(loaded_inputs_at - loaded_splits_at, 3),
            "bm25_baseline_and_lcdr_evaluate": round(
                indexed_and_evaluated_at - loaded_inputs_at,
                3,
            ),
            "guard_checks": round(checked_at - indexed_and_evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_lexical_cluster_diversity_comparison_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridLexicalClusterDiversityComparisonVisualization]:
    """Write SVG charts for Stage86 lexical cluster diversity comparison."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage86_lcdr_train_hit_at_10.svg": render_horizontal_bar_chart_svg(
            title="Stage86 LCDR train hit@10",
            bars=_hit_at_k_bars(report, split=_TRAIN_SPLIT, top_k=_PRIMARY_TOP_K),
            x_label="hit@10",
            width=1180,
            margin_left=430,
        ),
        "stage86_lcdr_dev_hit_at_10.svg": render_horizontal_bar_chart_svg(
            title="Stage86 LCDR dev hit@10",
            bars=_hit_at_k_bars(report, split=_DEV_SPLIT, top_k=_PRIMARY_TOP_K),
            x_label="hit@10",
            width=1180,
            margin_left=430,
        ),
        "stage86_lcdr_dev_delta_hit_at_10.svg": render_horizontal_bar_chart_svg(
            title="Stage86 LCDR dev hit@10 delta vs baseline",
            bars=_delta_bars(report, split=_DEV_SPLIT, metric="hit@10_delta"),
            x_label="delta hit@10",
            width=1180,
            margin_left=430,
        ),
        "stage86_lcdr_dev_top10_changes.svg": render_horizontal_bar_chart_svg(
            title="Stage86 LCDR dev top10 improvements minus regressions",
            bars=_net_change_bars(report, split=_DEV_SPLIT),
            x_label="net changed cases",
            width=1180,
            margin_left=430,
        ),
        "stage86_lcdr_dev_answer_duplicate_buckets.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage86 LCDR dev answer duplicate buckets",
                bars=_answer_duplicate_bucket_bars(report, split=_DEV_SPLIT),
                x_label="answer docs by duplicate bucket",
                width=1040,
                margin_left=360,
            )
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridLexicalClusterDiversityComparisonVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _configs_from_protocol(
    protocol: Mapping[str, Any],
) -> tuple[LexicalClusterDiversityConfig, ...]:
    configs = []
    for row in protocol.get("candidate_config_grid") or []:
        if not isinstance(row, Mapping):
            continue
        configs.append(
            LexicalClusterDiversityConfig(
                config_id=str(row.get("config_id") or ""),
                duplicate_penalty_weight=float(row.get("duplicate_penalty_weight") or 0.0),
                cluster_key=str(row.get("cluster_key") or ""),
                minimum_title_overlap_terms=int(
                    row.get("minimum_title_overlap_terms") or 0
                ),
                minimum_cluster_size=int(row.get("minimum_cluster_size") or 0),
            )
        )
    return tuple(configs)


def _evaluate_split(
    *,
    split: str,
    samples: Sequence[PrimeQAHybridSplitSample],
    documents: Sequence[PrimeQADocument],
    candidate_configs: Sequence[LexicalClusterDiversityConfig],
    top_k_values: tuple[int, ...],
    search_depth: int,
) -> dict[str, dict[str, Any]]:
    answerable_samples = [
        sample
        for sample in samples
        if sample.answerable and sample.answer_doc_id is not None
    ]
    index = _BM25CandidateIndex(documents)
    baseline_accumulator = _empty_accumulator(
        split=split,
        config_id=_BASELINE_CONFIG_ID,
        duplicate_penalty_weight=0.0,
        is_baseline=True,
        total_questions=len(samples),
    )
    candidate_accumulators = {
        config.config_id: _empty_accumulator(
            split=split,
            config_id=config.config_id,
            duplicate_penalty_weight=config.duplicate_penalty_weight,
            is_baseline=False,
            total_questions=len(samples),
        )
        for config in candidate_configs
    }
    min_title_overlap = min(
        config.minimum_title_overlap_terms for config in candidate_configs
    )
    min_cluster_size = min(config.minimum_cluster_size for config in candidate_configs)
    for sample in answerable_samples:
        query = sample.to_primeqa_question().full_question
        query_token_count = len(tokenize_text(query))
        query_unique_token_count = len(set(tokenize_text(query)))
        ranked_candidates = index.rank_candidates(
            query=query,
            search_depth=search_depth,
            minimum_title_overlap_terms=min_title_overlap,
            minimum_cluster_size=min_cluster_size,
        )
        baseline_rank, baseline_duplicate_index = _answer_rank_and_duplicate_index(
            ranked_candidates=ranked_candidates,
            answer_doc_id=str(sample.answer_doc_id),
        )
        _record_rank(
            accumulator=baseline_accumulator,
            sample=sample,
            rank=baseline_rank,
            cluster_duplicate_index=baseline_duplicate_index,
            query_token_count=query_token_count,
            query_unique_token_count=query_unique_token_count,
            ranked_candidates=ranked_candidates,
            top_k_values=top_k_values,
            search_depth=search_depth,
        )
        for config in candidate_configs:
            reranked_candidates = _rerank_candidates(
                ranked_candidates=ranked_candidates,
                config=config,
            )
            challenger_rank, challenger_duplicate_index = (
                _answer_rank_and_duplicate_index(
                    ranked_candidates=reranked_candidates,
                    answer_doc_id=str(sample.answer_doc_id),
                )
            )
            _record_rank(
                accumulator=candidate_accumulators[config.config_id],
                sample=sample,
                rank=challenger_rank,
                cluster_duplicate_index=challenger_duplicate_index,
                query_token_count=query_token_count,
                query_unique_token_count=query_unique_token_count,
                ranked_candidates=ranked_candidates,
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


def _rerank_candidates(
    *,
    ranked_candidates: Sequence[_RankedCandidate],
    config: LexicalClusterDiversityConfig,
) -> list[_RankedCandidate]:
    top1_score = ranked_candidates[0].baseline_bm25_score if ranked_candidates else 0.0
    return sorted(
        ranked_candidates,
        key=lambda candidate: (
            -(
                candidate.baseline_bm25_score
                - config.duplicate_penalty_weight
                * top1_score
                * candidate.cluster_duplicate_index
            ),
            -candidate.baseline_bm25_score,
            candidate.baseline_rank,
            candidate.document_id,
        ),
    )


def _answer_rank_and_duplicate_index(
    *,
    ranked_candidates: Sequence[_RankedCandidate],
    answer_doc_id: str,
) -> tuple[int | None, int | None]:
    for rank, candidate in enumerate(ranked_candidates, start=1):
        if candidate.document_id == answer_doc_id:
            return rank, candidate.cluster_duplicate_index
    return None, None


def _empty_accumulator(
    *,
    split: str,
    config_id: str,
    duplicate_penalty_weight: float,
    is_baseline: bool,
    total_questions: int,
) -> dict[str, Any]:
    return {
        "split": split,
        "config_id": config_id,
        "duplicate_penalty_weight": duplicate_penalty_weight,
        "is_baseline": is_baseline,
        "total_questions": total_questions,
        "evaluated_questions": 0,
        "hit_counts": {},
        "search_depth_hit_count": 0,
        "rank_11_to_50_count": 0,
        "reciprocal_rank_sum_at_10": 0.0,
        "reciprocal_rank_sum_at_search_depth": 0.0,
        "ranks_by_sample_id": {},
        "cluster_duplicate_index_by_sample_id": {},
        "empty_query_count": 0,
        "query_token_counts": [],
        "query_unique_token_counts": [],
        "answer_cluster_duplicate_buckets": Counter(),
        "candidate_depth_clustered_count": 0,
        "candidate_depth_duplicate_count": 0,
        "candidate_depth_total": 0,
    }


def _record_rank(
    *,
    accumulator: dict[str, Any],
    sample: PrimeQAHybridSplitSample,
    rank: int | None,
    cluster_duplicate_index: int | None,
    query_token_count: int,
    query_unique_token_count: int,
    ranked_candidates: Sequence[_RankedCandidate],
    top_k_values: tuple[int, ...],
    search_depth: int,
) -> None:
    accumulator["evaluated_questions"] += 1
    accumulator["ranks_by_sample_id"][sample.sample_id] = rank
    accumulator["cluster_duplicate_index_by_sample_id"][sample.sample_id] = (
        cluster_duplicate_index
    )
    accumulator["empty_query_count"] += query_token_count == 0
    accumulator["query_token_counts"].append(query_token_count)
    accumulator["query_unique_token_counts"].append(query_unique_token_count)
    accumulator["answer_cluster_duplicate_buckets"][
        _duplicate_bucket(cluster_duplicate_index)
    ] += 1
    accumulator["candidate_depth_total"] += len(ranked_candidates)
    accumulator["candidate_depth_clustered_count"] += sum(
        candidate.lexical_cluster_hash != _CLUSTER_HASH_EMPTY
        for candidate in ranked_candidates
    )
    accumulator["candidate_depth_duplicate_count"] += sum(
        candidate.cluster_duplicate_index > 0 for candidate in ranked_candidates
    )
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
    candidate_depth_total = int(accumulator["candidate_depth_total"])
    return {
        "split": accumulator["split"],
        "config_id": accumulator["config_id"],
        "duplicate_penalty_weight": float(accumulator["duplicate_penalty_weight"]),
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
        "average_query_unique_token_count": _rounded_mean(
            accumulator["query_unique_token_counts"]
        ),
        "answer_cluster_duplicate_buckets": dict(
            sorted(accumulator["answer_cluster_duplicate_buckets"].items())
        ),
        "candidate_depth_clustered_rate": _rounded_ratio(
            int(accumulator["candidate_depth_clustered_count"]),
            candidate_depth_total,
        ),
        "candidate_depth_duplicate_rate": _rounded_ratio(
            int(accumulator["candidate_depth_duplicate_count"]),
            candidate_depth_total,
        ),
        "ranks_by_sample_id": accumulator["ranks_by_sample_id"],
        "cluster_duplicate_index_by_sample_id": accumulator[
            "cluster_duplicate_index_by_sample_id"
        ],
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
    baseline_duplicates = baseline["cluster_duplicate_index_by_sample_id"]
    challenger_duplicates = challenger["cluster_duplicate_index_by_sample_id"]
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
            baseline_cluster_duplicate_index=baseline_duplicates.get(sample_id),
            challenger_cluster_duplicate_index=challenger_duplicates.get(sample_id),
            config_id=str(challenger["config_id"]),
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
    candidate_configs: Sequence[LexicalClusterDiversityConfig],
) -> dict[str, Any]:
    candidate_ids = [config.config_id for config in candidate_configs]
    train_results = rank_tables[_TRAIN_SPLIT]
    train_comparisons = comparisons[_TRAIN_SPLIT]
    selected_config_id = sorted(
        candidate_ids,
        key=lambda config_id: (
            -float(train_results[config_id]["hit_at_k"][_PRIMARY_TOP_K]),
            -float(train_results[config_id]["hit_at_k"].get(5, 0.0)),
            -float(train_results[config_id]["hit_at_k"].get(1, 0.0)),
            -float(train_results[config_id]["mrr_at_10"]),
            int(train_comparisons[config_id]["top10_regression_count"]),
            int(train_comparisons[config_id]["rank_down_within_top10_count"]),
            config_id,
        ),
    )[0]
    return {
        "selection_rule": _selection_rule_description(),
        "selected_config_id": selected_config_id,
        "candidate_config_count": len(candidate_ids),
        "selected_train_metrics": _public_config_metrics(
            train_results[selected_config_id]
        ),
        "selected_train_comparison_to_baseline": train_comparisons[selected_config_id],
    }


def _guard_checks(
    *,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    rank_tables: Mapping[str, Mapping[str, Mapping[str, Any]]],
    stage75_report: Mapping[str, Any],
    stage85_report: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_configs: Sequence[LexicalClusterDiversityConfig],
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
    metrics_evaluated = bool(rank_tables)
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
            name="source_stage85_report_is_stage85",
            passed=stage85_report.get("stage") == "Stage 85",
            observed=stage85_report.get("stage"),
            expected="Stage 85",
        ),
        _check(
            name="stage85_protocol_id_matches",
            passed=protocol.get("protocol_id") == _PROTOCOL_ID,
            observed=protocol.get("protocol_id"),
            expected=_PROTOCOL_ID,
        ),
        _check(
            name="stage85_candidate_id_matches",
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
            name="stage85_allows_train_dev_metrics_after_confirmation",
            passed=_stage85_allows_metric_run(stage85_report),
            observed=(stage85_report.get("decision") or {}).get(
                "can_run_train_dev_metrics_after_user_confirmation"
            ),
            expected=True,
        ),
        _check(
            name="stage85_final_test_metrics_locked",
            passed=(stage85_report.get("decision") or {}).get(
                "can_run_final_test_metrics_now"
            )
            is False,
            observed=(stage85_report.get("decision") or {}).get(
                "can_run_final_test_metrics_now"
            ),
            expected=False,
        ),
        _check(
            name="stage85_default_runtime_policy_unchanged",
            passed=(stage85_report.get("decision") or {}).get(
                "default_runtime_policy"
            )
            == "unchanged",
            observed=(stage85_report.get("decision") or {}).get(
                "default_runtime_policy"
            ),
            expected="unchanged",
        ),
        _check(
            name="candidate_config_grid_matches_frozen_protocol",
            passed=_candidate_config_grid_matches_protocol(protocol_configs),
            observed=[_public_candidate_config(config) for config in protocol_configs],
            expected=list(_EXPECTED_CONFIG_IDS),
        ),
        _check(
            name="source_stage75_report_is_stage75",
            passed=stage75_report.get("stage") == "Stage 75",
            observed=stage75_report.get("stage"),
            expected="Stage 75",
        ),
        _check(
            name="baseline_train_hit10_matches_stage75",
            passed=(not metrics_evaluated) or baseline_train_hit10 == stage75_train_hit10,
            observed=baseline_train_hit10
            if metrics_evaluated
            else "not_evaluated_without_confirmation",
            expected=stage75_train_hit10,
        ),
        _check(
            name="baseline_dev_hit10_matches_stage75",
            passed=(not metrics_evaluated) or baseline_dev_hit10 == stage75_dev_hit10,
            observed=baseline_dev_hit10
            if metrics_evaluated
            else "not_evaluated_without_confirmation",
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
            passed=_changed_case_fields_public_safe(protocol),
            observed=(
                protocol.get("public_safe_changed_case_fields") or []
            ),
            expected=[
                "sample_id",
                "split",
                "baseline_rank",
                "challenger_rank",
                "baseline_cluster_duplicate_index",
                "challenger_cluster_duplicate_index",
                "config_id",
            ],
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
            "status": "primeqa_hybrid_lexical_cluster_diversity_comparison_blocked",
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
    if dev_hit10_delta > 0:
        recommended_next_stage = (
            "Stage 87: review lexical cluster diversity changed cases on dev "
            "before any runtime experiment; keep test locked."
        )
    else:
        recommended_next_stage = (
            "Stage 87: stop lexical cluster diversity as a retrieval-recall route "
            "unless a new train/dev-only protocol is explicitly confirmed; keep "
            "test locked and move to the next confirmed second-wave candidate."
        )
    return {
        "status": "primeqa_hybrid_lexical_cluster_diversity_comparison_completed",
        "selected_config_id": selected_config_id,
        "selected_dev_hit10_delta": dev_hit10_delta,
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
        "duplicate_penalty_weight": float(config_result["duplicate_penalty_weight"]),
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
        "average_query_unique_token_count": float(
            config_result["average_query_unique_token_count"]
        ),
        "answer_cluster_duplicate_buckets": config_result[
            "answer_cluster_duplicate_buckets"
        ],
        "candidate_depth_clustered_rate": float(
            config_result["candidate_depth_clustered_rate"]
        ),
        "candidate_depth_duplicate_rate": float(
            config_result["candidate_depth_duplicate_rate"]
        ),
    }


def _public_candidate_config(config: LexicalClusterDiversityConfig) -> dict[str, Any]:
    return {
        "config_id": config.config_id,
        "duplicate_penalty_weight": config.duplicate_penalty_weight,
        "cluster_key": config.cluster_key,
        "minimum_title_overlap_terms": config.minimum_title_overlap_terms,
        "minimum_cluster_size": config.minimum_cluster_size,
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


def _answer_duplicate_bucket_bars(
    report: Mapping[str, Any],
    *,
    split: str,
) -> list[BarDatum]:
    metrics = report.get("metrics_by_split", {}).get(split, {})
    baseline = metrics.get(_BASELINE_CONFIG_ID) or {}
    buckets = baseline.get("answer_cluster_duplicate_buckets") or {}
    return [
        BarDatum(label=str(bucket), value=float(count), value_label=str(count))
        for bucket, count in sorted(buckets.items())
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


def _stage85_allows_metric_run(stage85_report: Mapping[str, Any]) -> bool:
    decision = stage85_report.get("decision") or {}
    return (
        decision.get("can_run_train_dev_metrics_after_user_confirmation") is True
        and decision.get("can_run_final_test_metrics_now") is False
        and decision.get("can_use_test_for_tuning") is False
        and decision.get("default_runtime_policy") == "unchanged"
    )


def _candidate_config_grid_matches_protocol(
    candidate_configs: Sequence[LexicalClusterDiversityConfig],
) -> bool:
    if len(candidate_configs) != len(_EXPECTED_CONFIG_IDS):
        return False
    observed_ids = tuple(config.config_id for config in candidate_configs)
    if observed_ids != _EXPECTED_CONFIG_IDS:
        return False
    return all(
        config.cluster_key == "title_query_overlap_hash"
        and config.minimum_title_overlap_terms == 3
        and config.minimum_cluster_size == 2
        for config in candidate_configs
    )


def _changed_case_fields_public_safe(protocol: Mapping[str, Any]) -> bool:
    observed = protocol.get("public_safe_changed_case_fields") or []
    expected = [
        "sample_id",
        "split",
        "baseline_rank",
        "challenger_rank",
        "baseline_cluster_duplicate_index",
        "challenger_cluster_duplicate_index",
        "config_id",
    ]
    return observed == expected


def _selection_rule_description() -> str:
    return (
        "Select the lexical cluster diversity candidate config on train only by "
        "hit@10, then hit@5, then hit@1, then MRR@10, then fewer top10 "
        "regressions, then fewer rank-down cases within top10, then config_id; "
        "dev is validation only."
    )


def _change_case(
    *,
    split: str,
    sample_id: str,
    baseline_rank: int | None,
    challenger_rank: int | None,
    baseline_cluster_duplicate_index: int | None,
    challenger_cluster_duplicate_index: int | None,
    config_id: str,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "split": split,
        "baseline_rank": baseline_rank,
        "challenger_rank": challenger_rank,
        "baseline_cluster_duplicate_index": baseline_cluster_duplicate_index,
        "challenger_cluster_duplicate_index": challenger_cluster_duplicate_index,
        "config_id": config_id,
    }


def _duplicate_bucket(cluster_duplicate_index: int | None) -> str:
    if cluster_duplicate_index is None:
        return "not_found"
    if cluster_duplicate_index == 0:
        return "0"
    if cluster_duplicate_index <= 2:
        return "1_to_2"
    return "3_plus"


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
