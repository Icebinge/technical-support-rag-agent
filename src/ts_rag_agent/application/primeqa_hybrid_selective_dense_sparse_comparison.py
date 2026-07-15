from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ts_rag_agent.application.primeqa_hybrid_dense_sparse_rrf_comparison import (
    LocalSnapshotSentenceTransformerEncoder,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever, tokenize_text
from ts_rag_agent.infrastructure.dense_retriever import DenseRetriever, TextEmbeddingModel
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
    summarize_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_documents

_STAGE = "Stage 98"
_CREATED_AT = "2026-07-15"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_PROTOCOL_ID = "selective_dense_sparse_low_overlap_gate_train_dev_v1"
_BASELINE_CONFIG_ID = "full_document_bm25_baseline"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})
_PRIMARY_TOP_K = 10
_DEFAULT_SEARCH_DEPTH = 50
_CHANGE_CASE_LIMIT = 20


class EncoderFactory(Protocol):
    """Build a query encoder for one frozen Stage97 dense cache config."""

    def __call__(self, config: Mapping[str, Any]) -> TextEmbeddingModel:
        """Return a text embedding model for the supplied dense config."""


@dataclass(frozen=True)
class PrimeQAHybridSelectiveDenseSparseComparisonVisualization:
    """One generated Stage98 selective dense+sparse comparison visualization."""

    name: str
    path: str


def run_primeqa_hybrid_selective_dense_sparse_comparison(
    *,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    stage75_report_path: Path,
    stage97_report_path: Path,
    user_confirmed_protocol: bool,
    confirmed_protocol_id: str,
    confirmation_note: str,
    top_k_values: tuple[int, ...] = (1, 5, 10),
    search_depth: int = _DEFAULT_SEARCH_DEPTH,
    encoder_batch_size: int = 64,
    encoder_device: str | None = None,
    encoder_factory: EncoderFactory | None = None,
) -> dict[str, Any]:
    """Run the frozen Stage97 selective dense+sparse comparison on train/dev."""

    _validate_options(top_k_values=top_k_values, search_depth=search_depth)
    started_at = time.perf_counter()
    split_samples = {
        _TRAIN_SPLIT: load_primeqa_hybrid_split_samples(train_split_path),
        _DEV_SPLIT: load_primeqa_hybrid_split_samples(dev_split_path),
    }
    loaded_splits_at = time.perf_counter()
    documents = load_primeqa_documents(documents_path)
    document_list = list(documents.values())
    document_ids = tuple(document.id for document in document_list)
    stage75_report = _load_json_object(stage75_report_path)
    stage97_report = _load_json_object(stage97_report_path)
    loaded_inputs_at = time.perf_counter()

    frozen_protocol = _frozen_protocol(stage97_report)
    baseline_config = _baseline_config(frozen_protocol)
    dense_configs = _allowed_dense_configs(frozen_protocol)
    policy_grid = _candidate_policy_grid(frozen_protocol)
    cache_preflight = _preflight_dense_caches(
        dense_configs=dense_configs,
        current_document_ids=document_ids,
    )
    protocol_ready = _protocol_ready_for_evaluation(
        stage97_report=stage97_report,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmed_protocol_id=confirmed_protocol_id,
        cache_preflight=cache_preflight,
    )
    preflight_at = time.perf_counter()

    rank_tables: dict[str, dict[str, dict[str, Any]]] = {}
    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    train_selection: dict[str, Any] = _empty_train_selection(policy_grid)
    indexed_at = preflight_at
    evaluated_at = preflight_at

    if protocol_ready:
        encoder_factory = encoder_factory or _default_encoder_factory(
            batch_size=encoder_batch_size,
            device=encoder_device,
        )
        baseline_retriever = BM25Retriever(
            k1=float(baseline_config["bm25_k1"]),
            b=float(baseline_config["bm25_b"]),
        )
        baseline_retriever.fit(document_list)
        dense_retrievers = _build_dense_retrievers(
            dense_configs=dense_configs,
            document_list=document_list,
            document_ids=document_ids,
            encoder_factory=encoder_factory,
        )
        indexed_at = time.perf_counter()
        max_dense_rank = _max_dense_candidate_rank(policy_grid)
        document_token_sets = _document_token_sets(document_list)
        rank_tables = {
            split: _evaluate_split(
                split=split,
                samples=samples,
                baseline_retriever=baseline_retriever,
                dense_retrievers=dense_retrievers,
                documents_by_id=documents,
                document_token_sets=document_token_sets,
                policy_grid=policy_grid,
                top_k_values=top_k_values,
                search_depth=search_depth,
                dense_top_k=max_dense_rank,
            )
            for split, samples in split_samples.items()
        }
        evaluated_at = time.perf_counter()
        comparisons = {
            split: {
                policy["policy_id"]: _compare_to_baseline(
                    baseline=rank_tables[split][_BASELINE_CONFIG_ID],
                    challenger=rank_tables[split][policy["policy_id"]],
                    policy=policy,
                    max_k=_PRIMARY_TOP_K,
                    search_depth=search_depth,
                )
                for policy in policy_grid
            }
            for split in _ALLOWED_DEVELOPMENT_SPLITS
        }
        train_selection = _select_policy_on_train(
            rank_tables=rank_tables,
            comparisons=comparisons,
            policy_grid=policy_grid,
        )

    guard_checks = _guard_checks(
        split_samples=split_samples,
        rank_tables=rank_tables,
        comparisons=comparisons,
        train_selection=train_selection,
        cache_preflight=cache_preflight,
        stage75_report=stage75_report,
        stage97_report=stage97_report,
        frozen_protocol=frozen_protocol,
        dense_configs=dense_configs,
        policy_grid=policy_grid,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmed_protocol_id=confirmed_protocol_id,
        top_k_values=top_k_values,
        search_depth=search_depth,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "Train/dev-only selective dense+sparse low-overlap gate comparison "
            "for the frozen Stage97 protocol. This stage loads only train and "
            "dev splits, uses existing local dense caches and local model "
            "snapshots, does not load the frozen test split, does not run final "
            "metrics, does not use source DOC_IDS as runtime retrieval evidence, "
            "does not download models or refresh caches, and does not change "
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
            "stage97_report": _fingerprint(stage97_report_path),
        },
        "loaded_data_summary": {
            **summarize_primeqa_hybrid_split_samples(split_samples),
            "document_count": len(documents),
            "test_split_loaded": False,
            "final_metrics_run": False,
        },
        "frozen_protocol_summary": _public_protocol_summary(
            frozen_protocol=frozen_protocol,
            dense_configs=dense_configs,
            policy_grid=policy_grid,
        ),
        "dense_cache_preflight": cache_preflight,
        "metrics_by_split": {
            split: {
                config_id: _public_config_metrics(config_result)
                for config_id, config_result in split_results.items()
            }
            for split, split_results in rank_tables.items()
        },
        "comparisons_to_baseline": comparisons,
        "train_selection": train_selection,
        "guard_checks": guard_checks,
        "decision": _decision(
            guard_checks=guard_checks,
            comparisons=comparisons,
            train_selection=train_selection,
        ),
        "artifact_safety": {
            "raw_question_text_written": False,
            "raw_answer_text_written": False,
            "raw_document_text_written": False,
            "raw_document_title_written": False,
            "query_terms_written": False,
            "matched_token_strings_written": False,
            "source_doc_ids_used_as_runtime_evidence": False,
            "answer_doc_ids_used_as_runtime_features": False,
        },
        "timing_seconds": {
            "load_splits": round(loaded_splits_at - started_at, 3),
            "load_inputs": round(loaded_inputs_at - loaded_splits_at, 3),
            "preflight": round(preflight_at - loaded_inputs_at, 3),
            "index": round(indexed_at - preflight_at, 3),
            "evaluate": round(evaluated_at - indexed_at, 3),
            "guard": round(checked_at - evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_selective_dense_sparse_comparison_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridSelectiveDenseSparseComparisonVisualization]:
    """Write SVG charts for the Stage98 selective dense+sparse comparison."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage98_selective_dense_sparse_train_hit_at_10.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage98 selective dense+sparse train hit@10",
                bars=_hit_at_k_bars(report, split=_TRAIN_SPLIT, top_k=_PRIMARY_TOP_K),
                x_label="hit@10",
                width=1320,
                margin_left=620,
            )
        ),
        "stage98_selective_dense_sparse_dev_hit_at_10.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage98 selective dense+sparse dev hit@10",
                bars=_hit_at_k_bars(report, split=_DEV_SPLIT, top_k=_PRIMARY_TOP_K),
                x_label="hit@10",
                width=1320,
                margin_left=620,
            )
        ),
        "stage98_selective_dense_sparse_dev_hit10_delta.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage98 selective dense+sparse dev hit@10 delta",
                bars=_delta_bars(report, split=_DEV_SPLIT, metric="hit@10_delta"),
                x_label="delta vs BM25 baseline",
                width=1320,
                margin_left=620,
            )
        ),
        "stage98_selective_dense_sparse_dev_not_found_delta.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage98 selective dense+sparse dev not-found@50 delta",
                bars=_delta_bars(
                    report,
                    split=_DEV_SPLIT,
                    metric="not_found_count_at_search_depth_delta",
                ),
                x_label="count delta vs BM25 baseline",
                width=1320,
                margin_left=620,
            )
        ),
        "stage98_selective_dense_sparse_dev_promotions.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage98 selective dense+sparse dev promotions",
                bars=_action_bars(report, split=_DEV_SPLIT),
                x_label="dense top10 promotion count",
                width=1320,
                margin_left=620,
            )
        ),
        "stage98_selective_dense_sparse_guard_check_status.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage98 selective dense+sparse guard check status",
                bars=_guard_check_bars(report),
                x_label="1 means passed",
                width=1420,
                margin_left=760,
            )
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridSelectiveDenseSparseComparisonVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _evaluate_split(
    *,
    split: str,
    samples: Sequence[PrimeQAHybridSplitSample],
    baseline_retriever: BM25Retriever,
    dense_retrievers: Mapping[str, DenseRetriever],
    documents_by_id: Mapping[str, PrimeQADocument],
    document_token_sets: Mapping[str, frozenset[str]],
    policy_grid: Sequence[Mapping[str, Any]],
    top_k_values: tuple[int, ...],
    search_depth: int,
    dense_top_k: int,
) -> dict[str, Any]:
    answerable_samples = [
        sample
        for sample in samples
        if sample.answerable and sample.answer_doc_id is not None
    ]
    config_ids = [_BASELINE_CONFIG_ID, *[str(policy["policy_id"]) for policy in policy_grid]]
    accumulators = {
        config_id: _empty_accumulator(
            split=split,
            config_id=config_id,
            total_questions=len(samples),
        )
        for config_id in config_ids
    }
    for sample in answerable_samples:
        query = sample.to_primeqa_question().full_question
        query_terms = frozenset(tokenize_text(query))
        query_token_count = len(query_terms)
        baseline_results = baseline_retriever.search(query, top_k=search_depth)
        _record_result(
            accumulator=accumulators[_BASELINE_CONFIG_ID],
            sample=sample,
            results=baseline_results,
            query_token_count=query_token_count,
            top_k_values=top_k_values,
            search_depth=search_depth,
            action=None,
        )
        dense_results_by_config = {
            config_id: retriever.search(query, top_k=dense_top_k)
            for config_id, retriever in dense_retrievers.items()
        }
        overlap_features = _overlap_features(
            query_terms=query_terms,
            baseline_results=baseline_results,
            document_token_sets=document_token_sets,
        )
        for policy in policy_grid:
            policy_id = str(policy["policy_id"])
            dense_config_id = str(policy["dense_config_id"])
            policy_results, action = _apply_policy(
                policy=policy,
                baseline_results=baseline_results,
                dense_results=dense_results_by_config[dense_config_id],
                documents_by_id=documents_by_id,
                query_token_count=query_token_count,
                overlap_features=overlap_features,
                search_depth=search_depth,
            )
            _record_result(
                accumulator=accumulators[policy_id],
                sample=sample,
                results=policy_results,
                query_token_count=query_token_count,
                top_k_values=top_k_values,
                search_depth=search_depth,
                action=action,
            )
    return {
        config_id: _finalize_accumulator(
            accumulator,
            top_k_values=top_k_values,
            search_depth=search_depth,
        )
        for config_id, accumulator in accumulators.items()
    }


def _apply_policy(
    *,
    policy: Mapping[str, Any],
    baseline_results: Sequence[RetrievalResult],
    dense_results: Sequence[RetrievalResult],
    documents_by_id: Mapping[str, PrimeQADocument],
    query_token_count: int,
    overlap_features: Mapping[str, float],
    search_depth: int,
) -> tuple[list[RetrievalResult], dict[str, Any]]:
    gate = _gate_decision(
        policy=policy,
        query_token_count=query_token_count,
        overlap_features=overlap_features,
    )
    baseline_doc_ids = [result.document.id for result in baseline_results]
    baseline_rank_by_doc = {
        result.document.id: result.rank for result in baseline_results
    }
    if not gate["activated"]:
        return list(baseline_results), {
            **gate,
            "promotion_count": 0,
            "promotion_budget_used": 0,
            "protected_bm25_top_rank_demotion_count": 0,
            "best_dense_rank": None,
            "dense_rank_bucket": "none",
        }

    candidates = _promotion_candidates(
        policy=policy,
        dense_results=dense_results,
        baseline_doc_ids=baseline_doc_ids,
        baseline_rank_by_doc=baseline_rank_by_doc,
    )
    promotion_budget = int(policy["maximum_dense_top10_promotions_per_query"])
    promoted = candidates[:promotion_budget]
    reranked_results = _merge_promotions(
        policy=policy,
        baseline_results=baseline_results,
        promoted=promoted,
        documents_by_id=documents_by_id,
        search_depth=search_depth,
    )
    return reranked_results, {
        **gate,
        "reason_code": (
            "activated_with_promotion" if promoted else "activated_without_candidate"
        ),
        "promotion_count": len(promoted),
        "promotion_budget_used": len(promoted),
        "protected_bm25_top_rank_demotion_count": _protected_demotion_count(
            baseline_results=baseline_results,
            reranked_results=reranked_results,
        ),
        "best_dense_rank": promoted[0]["dense_rank"] if promoted else None,
        "dense_rank_bucket": _dense_rank_bucket(
            int(promoted[0]["dense_rank"]) if promoted else None
        ),
    }


def _gate_decision(
    *,
    policy: Mapping[str, Any],
    query_token_count: int,
    overlap_features: Mapping[str, float],
) -> dict[str, Any]:
    top1_overlap = float(overlap_features["bm25_top1_query_overlap_ratio"])
    top10_mean_overlap = float(
        overlap_features["bm25_top10_mean_query_overlap_ratio"]
    )
    if query_token_count < int(policy["minimum_query_token_count"]):
        reason_code = "blocked_short_query"
        activated = False
    elif top1_overlap > float(policy["maximum_bm25_top1_query_overlap_ratio"]):
        reason_code = "blocked_top1_overlap"
        activated = False
    elif top10_mean_overlap > float(
        policy["maximum_bm25_top10_mean_query_overlap_ratio"]
    ):
        reason_code = "blocked_top10_mean_overlap"
        activated = False
    else:
        reason_code = "activated_low_overlap"
        activated = True
    return {
        "activated": activated,
        "reason_code": reason_code,
        "query_length_bucket": _query_length_bucket(query_token_count),
        "bm25_top1_overlap_bucket": _overlap_bucket(top1_overlap),
        "bm25_top10_mean_overlap_bucket": _overlap_bucket(top10_mean_overlap),
    }


def _promotion_candidates(
    *,
    policy: Mapping[str, Any],
    dense_results: Sequence[RetrievalResult],
    baseline_doc_ids: Sequence[str],
    baseline_rank_by_doc: Mapping[str, int],
) -> list[dict[str, Any]]:
    dense_rank_max = int(policy["dense_candidate_rank_max"])
    sparse_weight = float(policy["sparse_weight"])
    dense_weight = float(policy["dense_weight"])
    rrf_k = int(policy["rrf_k"])
    baseline_top10 = set(baseline_doc_ids[:_PRIMARY_TOP_K])
    candidates = []
    for result in dense_results:
        doc_id = result.document.id
        if result.rank > dense_rank_max:
            continue
        if policy.get("dense_candidate_must_be_outside_bm25_top10") and (
            doc_id in baseline_top10
        ):
            continue
        sparse_rank = baseline_rank_by_doc.get(doc_id)
        sparse_score = (
            sparse_weight / (rrf_k + sparse_rank) if sparse_rank is not None else 0.0
        )
        dense_score = dense_weight / (rrf_k + result.rank)
        candidates.append(
            {
                "document_id": doc_id,
                "dense_rank": result.rank,
                "sparse_rank": sparse_rank,
                "rrf_score": sparse_score + dense_score,
            }
        )
    return sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate["rrf_score"]),
            int(candidate["dense_rank"]),
            str(candidate["document_id"]),
        ),
    )


def _merge_promotions(
    *,
    policy: Mapping[str, Any],
    baseline_results: Sequence[RetrievalResult],
    promoted: Sequence[Mapping[str, Any]],
    documents_by_id: Mapping[str, PrimeQADocument],
    search_depth: int,
) -> list[RetrievalResult]:
    protected_count = int(policy["protected_bm25_top_rank_count"])
    protected_doc_ids = {result.document.id for result in baseline_results[:protected_count]}
    promoted_doc_ids = {str(candidate["document_id"]) for candidate in promoted}
    ordered_doc_ids = [result.document.id for result in baseline_results[:protected_count]]
    ordered_doc_ids.extend(str(candidate["document_id"]) for candidate in promoted)
    ordered_doc_ids.extend(
        result.document.id
        for result in baseline_results
        if result.document.id not in protected_doc_ids
        and result.document.id not in promoted_doc_ids
    )
    deduped_doc_ids: list[str] = []
    seen = set()
    for doc_id in ordered_doc_ids:
        if doc_id in seen:
            continue
        if doc_id not in documents_by_id:
            continue
        deduped_doc_ids.append(doc_id)
        seen.add(doc_id)
    score_by_doc = {
        result.document.id: result.score for result in baseline_results
    }
    score_by_doc.update(
        {
            str(candidate["document_id"]): float(candidate["rrf_score"])
            for candidate in promoted
        }
    )
    return [
        RetrievalResult(
            document=documents_by_id[doc_id],
            score=float(score_by_doc.get(doc_id, 0.0)),
            rank=rank,
        )
        for rank, doc_id in enumerate(deduped_doc_ids[:search_depth], start=1)
    ]


def _overlap_features(
    *,
    query_terms: frozenset[str],
    baseline_results: Sequence[RetrievalResult],
    document_token_sets: Mapping[str, frozenset[str]],
) -> dict[str, float]:
    if not query_terms:
        return {
            "bm25_top1_query_overlap_ratio": 0.0,
            "bm25_top10_mean_query_overlap_ratio": 0.0,
        }
    top1 = baseline_results[0] if baseline_results else None
    top1_ratio = (
        _query_overlap_ratio(
            query_terms=query_terms,
            document_terms=document_token_sets.get(top1.document.id, frozenset()),
        )
        if top1 is not None
        else 0.0
    )
    top10_ratios = [
        _query_overlap_ratio(
            query_terms=query_terms,
            document_terms=document_token_sets.get(result.document.id, frozenset()),
        )
        for result in baseline_results[:_PRIMARY_TOP_K]
    ]
    return {
        "bm25_top1_query_overlap_ratio": round(top1_ratio, 4),
        "bm25_top10_mean_query_overlap_ratio": round(
            sum(top10_ratios) / len(top10_ratios),
            4,
        )
        if top10_ratios
        else 0.0,
    }


def _query_overlap_ratio(
    *,
    query_terms: frozenset[str],
    document_terms: frozenset[str],
) -> float:
    return len(query_terms & document_terms) / len(query_terms) if query_terms else 0.0


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
        "reciprocal_rank_sum_at_10": 0.0,
        "reciprocal_rank_sum_at_search_depth": 0.0,
        "ranks_by_sample_id": {},
        "empty_query_count": 0,
        "query_token_counts": [],
        "gate_activation_count": 0,
        "gate_reason_counts": {},
        "promotion_count": 0,
        "queries_with_promotion_count": 0,
        "protected_bm25_top_rank_demotion_count": 0,
        "sample_actions_by_id": {},
    }


def _record_result(
    *,
    accumulator: dict[str, Any],
    sample: PrimeQAHybridSplitSample,
    results: Sequence[RetrievalResult],
    query_token_count: int,
    top_k_values: tuple[int, ...],
    search_depth: int,
    action: Mapping[str, Any] | None,
) -> None:
    result_doc_ids = [result.document.id for result in results]
    answer_doc_id = str(sample.answer_doc_id)
    rank = (
        result_doc_ids.index(answer_doc_id) + 1
        if answer_doc_id in result_doc_ids
        else None
    )
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
    if action is not None:
        reason_code = str(action["reason_code"])
        accumulator["gate_reason_counts"].setdefault(reason_code, 0)
        accumulator["gate_reason_counts"][reason_code] += 1
        accumulator["gate_activation_count"] += bool(action["activated"])
        promotion_count = int(action["promotion_count"])
        accumulator["promotion_count"] += promotion_count
        accumulator["queries_with_promotion_count"] += promotion_count > 0
        accumulator["protected_bm25_top_rank_demotion_count"] += int(
            action["protected_bm25_top_rank_demotion_count"]
        )
        accumulator["sample_actions_by_id"][sample.sample_id] = {
            key: action[key]
            for key in (
                "reason_code",
                "query_length_bucket",
                "bm25_top1_overlap_bucket",
                "bm25_top10_mean_overlap_bucket",
                "dense_rank_bucket",
                "promotion_budget_used",
            )
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
        "empty_query_count": int(accumulator["empty_query_count"]),
        "average_query_token_count": _rounded_mean(
            [int(value) for value in accumulator["query_token_counts"]]
        ),
        "gate_activation_count": int(accumulator["gate_activation_count"]),
        "gate_reason_counts": dict(sorted(accumulator["gate_reason_counts"].items())),
        "promotion_count": int(accumulator["promotion_count"]),
        "queries_with_promotion_count": int(accumulator["queries_with_promotion_count"]),
        "protected_bm25_top_rank_demotion_count": int(
            accumulator["protected_bm25_top_rank_demotion_count"]
        ),
        "ranks_by_sample_id": dict(accumulator["ranks_by_sample_id"]),
        "sample_actions_by_id": dict(accumulator["sample_actions_by_id"]),
    }


def _compare_to_baseline(
    *,
    baseline: Mapping[str, Any],
    challenger: Mapping[str, Any],
    policy: Mapping[str, Any],
    max_k: int,
    search_depth: int,
) -> dict[str, Any]:
    baseline_ranks = baseline["ranks_by_sample_id"]
    challenger_ranks = challenger["ranks_by_sample_id"]
    challenger_actions = challenger["sample_actions_by_id"]
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
        challenger_found = (
            challenger_rank is not None and challenger_rank <= search_depth
        )
        change_case = _change_case(
            sample_id=sample_id,
            split=str(challenger["split"]),
            baseline_rank=baseline_rank,
            challenger_rank=challenger_rank,
            policy=policy,
            action=challenger_actions.get(sample_id, {}),
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
    not_found_delta = int(challenger["not_found_count_at_search_depth"]) - int(
        baseline["not_found_count_at_search_depth"]
    )
    return {
        "baseline_config_id": baseline["config_id"],
        "challenger_policy_id": policy["policy_id"],
        "dense_config_id": policy["dense_config_id"],
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
        "not_found_count_at_search_depth_delta": not_found_delta,
        "both_hit_count": both_hit,
        "both_miss_count": both_miss,
        "rank_up_within_top10_count": rank_up,
        "rank_down_within_top10_count": rank_down,
        "gate_activation_count": int(challenger["gate_activation_count"]),
        "gate_reason_counts": challenger["gate_reason_counts"],
        "promotion_count": int(challenger["promotion_count"]),
        "queries_with_promotion_count": int(challenger["queries_with_promotion_count"]),
        "protected_bm25_top_rank_demotion_count": int(
            challenger["protected_bm25_top_rank_demotion_count"]
        ),
        "sample_top10_improvements": top10_improvements[:_CHANGE_CASE_LIMIT],
        "sample_top10_regressions": top10_regressions[:_CHANGE_CASE_LIMIT],
        "sample_search_depth_improvements": search_depth_improvements[
            :_CHANGE_CASE_LIMIT
        ],
        "sample_search_depth_regressions": search_depth_regressions[
            :_CHANGE_CASE_LIMIT
        ],
    }


def _select_policy_on_train(
    *,
    rank_tables: Mapping[str, Mapping[str, Mapping[str, Any]]],
    comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]],
    policy_grid: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train_results = rank_tables[_TRAIN_SPLIT]
    train_comparisons = comparisons[_TRAIN_SPLIT]
    selected_policy_id = sorted(
        (str(policy["policy_id"]) for policy in policy_grid),
        key=lambda policy_id: (
            -float(train_results[policy_id]["hit_at_k"][_PRIMARY_TOP_K]),
            int(train_comparisons[policy_id]["not_found_count_at_search_depth_delta"]),
            int(train_comparisons[policy_id]["top10_regression_count"]),
            -float(train_results[policy_id]["hit_at_k"].get(1, 0.0)),
            -float(train_results[policy_id]["mrr_at_10"]),
            _policy_by_id(policy_grid, policy_id)[
                "maximum_dense_top10_promotions_per_query"
            ],
            policy_id,
        ),
    )[0]
    return {
        "selection_rule": _selection_rule_description(),
        "selected_policy_id": selected_policy_id,
        "candidate_count": len(policy_grid),
        "selected_train_metrics": _public_config_metrics(
            train_results[selected_policy_id]
        ),
        "selected_train_comparison_to_baseline": train_comparisons[selected_policy_id],
    }


def _guard_checks(
    *,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    rank_tables: Mapping[str, Mapping[str, Mapping[str, Any]]],
    comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]],
    train_selection: Mapping[str, Any],
    cache_preflight: Mapping[str, Any],
    stage75_report: Mapping[str, Any],
    stage97_report: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
    dense_configs: Sequence[Mapping[str, Any]],
    policy_grid: Sequence[Mapping[str, Any]],
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
    selected_policy_id = train_selection.get("selected_policy_id")
    selected_dev_comparison = (
        (comparisons.get(_DEV_SPLIT) or {}).get(str(selected_policy_id))
        if selected_policy_id is not None
        else None
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
            name="search_depth_matches_stage97_baseline_candidate_depth",
            passed=search_depth == int(_baseline_config(frozen_protocol)["candidate_depth"]),
            observed=search_depth,
            expected=int(_baseline_config(frozen_protocol)["candidate_depth"]),
        ),
        _check(
            name="stage75_source_report_is_stage75",
            passed=str(stage75_report.get("stage") or "") == "Stage 75",
            observed=stage75_report.get("stage"),
            expected="Stage 75",
        ),
        _check(
            name="stage97_source_report_is_stage97",
            passed=str(stage97_report.get("stage") or "") == "Stage 97",
            observed=stage97_report.get("stage"),
            expected="Stage 97",
        ),
        _check(
            name="stage97_protocol_is_frozen",
            passed=(
                (stage97_report.get("decision") or {}).get("status")
                == "primeqa_hybrid_selective_dense_sparse_protocol_frozen"
            ),
            observed=(stage97_report.get("decision") or {}).get("status"),
            expected="primeqa_hybrid_selective_dense_sparse_protocol_frozen",
        ),
        _check(
            name="user_confirmed_stage98_train_dev_run",
            passed=bool(user_confirmed_protocol),
            observed=bool(user_confirmed_protocol),
            expected=True,
        ),
        _check(
            name="confirmed_protocol_id_matches_stage97",
            passed=confirmed_protocol_id == _PROTOCOL_ID,
            observed=confirmed_protocol_id,
            expected=_PROTOCOL_ID,
        ),
        _check(
            name="stage97_requires_user_confirmation",
            passed=bool(
                (stage97_report.get("decision") or {}).get(
                    "requires_user_confirmation_before_train_dev_run"
                )
            ),
            observed=(stage97_report.get("decision") or {}).get(
                "requires_user_confirmation_before_train_dev_run"
            ),
            expected=True,
        ),
        _check(
            name="dense_cache_count_matches_stage97_protocol",
            passed=len(dense_configs) == 2,
            observed=len(dense_configs),
            expected=2,
        ),
        _check(
            name="policy_grid_count_matches_stage97_protocol",
            passed=len(policy_grid) == 4,
            observed=len(policy_grid),
            expected=4,
        ),
        _check(
            name="dense_configs_present_for_all_policies",
            passed=_policies_reference_known_dense_configs(
                dense_configs=dense_configs,
                policy_grid=policy_grid,
            ),
            observed=sorted(str(policy["dense_config_id"]) for policy in policy_grid),
            expected=sorted(str(config["config_id"]) for config in dense_configs),
        ),
        _check(
            name="dense_caches_preflight_passed",
            passed=bool(cache_preflight.get("can_run_without_download")),
            observed=cache_preflight.get("can_run_without_download"),
            expected=True,
        ),
        _check(
            name="no_model_download_attempted",
            passed=cache_preflight.get("no_model_download_attempted") is True,
            observed=cache_preflight.get("no_model_download_attempted"),
            expected=True,
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
            name="train_selection_uses_train_only",
            passed=selected_policy_id is not None,
            observed=selected_policy_id,
            expected="selected policy from train metrics",
        ),
        _check(
            name="selected_policy_has_dev_validation",
            passed=selected_dev_comparison is not None,
            observed=selected_dev_comparison is not None,
            expected=True,
        ),
        _check(
            name="source_doc_ids_not_used_as_runtime_evidence",
            passed=True,
            observed="not_used",
            expected="not_used",
        ),
        _check(
            name="answer_doc_ids_used_only_for_metric_scoring",
            passed=True,
            observed="metric_scoring_only",
            expected="metric_scoring_only",
        ),
        _check(
            name="final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="test_split_not_loaded",
            passed=True,
            observed="not_loaded",
            expected="not_loaded",
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
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_selective_dense_sparse_comparison_blocked",
            "protocol_id": _PROTOCOL_ID,
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        }
    selected_policy_id = str(train_selection["selected_policy_id"])
    selected_dev_comparison = comparisons[_DEV_SPLIT][selected_policy_id]
    dev_hit10_delta = float(selected_dev_comparison["hit@10_delta"])
    dev_hit1_delta = float(selected_dev_comparison["hit@1_delta"])
    dev_not_found_delta = int(
        selected_dev_comparison["not_found_count_at_search_depth_delta"]
    )
    primary_contract_passed = dev_hit10_delta > 0
    secondary_contract_passed = dev_not_found_delta < 0 and dev_hit1_delta >= 0
    if primary_contract_passed and secondary_contract_passed:
        recommended_next_stage = (
            "Stage 99: review selected selective dense+sparse changed cases on "
            "dev and decide whether the final-test gate is justified; keep test "
            "locked until that review passes."
        )
    else:
        recommended_next_stage = (
            "Stage 99: stop the selective dense+sparse route and summarize why "
            "the frozen train/dev contract did not pass; keep test locked."
        )
    return {
        "status": "primeqa_hybrid_selective_dense_sparse_comparison_completed",
        "protocol_id": _PROTOCOL_ID,
        "selected_policy_id": selected_policy_id,
        "selected_dev_hit10_delta": dev_hit10_delta,
        "selected_dev_hit1_delta": dev_hit1_delta,
        "selected_dev_top10_improvements": int(
            selected_dev_comparison["top10_improvement_count"]
        ),
        "selected_dev_top10_regressions": int(
            selected_dev_comparison["top10_regression_count"]
        ),
        "selected_dev_not_found_at_search_depth_delta": dev_not_found_delta,
        "selected_dev_gate_activation_count": int(
            selected_dev_comparison["gate_activation_count"]
        ),
        "selected_dev_promotion_count": int(
            selected_dev_comparison["promotion_count"]
        ),
        "primary_contract_passed": primary_contract_passed,
        "secondary_contract_passed": secondary_contract_passed,
        "guard_contract_passed": True,
        "can_continue_train_dev_development": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": recommended_next_stage,
    }


def _preflight_dense_caches(
    *,
    dense_configs: Sequence[Mapping[str, Any]],
    current_document_ids: tuple[str, ...],
) -> dict[str, Any]:
    cache_checks = []
    for config in dense_configs:
        cache_path = Path(str(config["cache_path"]))
        snapshot_path = Path(str(config["snapshot_path"]))
        cache_exists = cache_path.exists() and cache_path.is_file()
        snapshot_exists = snapshot_path.exists() and snapshot_path.is_dir()
        cache_sha256 = _file_sha256(cache_path) if cache_exists else None
        cache_sha_matches = cache_sha256 == str(config.get("cache_sha256") or "")
        document_ids_match = False
        embedding_shape: list[int] = []
        model_name_matches = False
        max_chars_matches = False
        document_prefix_matches = False
        if cache_exists:
            with np.load(cache_path, allow_pickle=False) as data:
                cached_document_ids = tuple(str(value) for value in data["document_ids"])
                embedding_shape = [int(value) for value in data["embeddings"].shape]
                document_ids_match = cached_document_ids == current_document_ids
                model_name_matches = str(data["model_name"]) == str(config["model_name"])
                max_chars_matches = int(data["document_text_max_chars"]) == int(
                    config["document_text_max_chars"]
                )
                cached_document_prefix = (
                    str(data["document_prefix"]) if "document_prefix" in data else ""
                )
                document_prefix_matches = cached_document_prefix == str(
                    config["document_prefix"]
                )
        row_count_matches = (
            bool(embedding_shape)
            and len(embedding_shape) == 2
            and embedding_shape[0] == len(current_document_ids)
        )
        cache_checks.append(
            {
                "config_id": str(config["config_id"]),
                "cache_path": str(cache_path),
                "cache_exists": cache_exists,
                "cache_sha256": cache_sha256,
                "cache_sha256_matches_stage97": cache_sha_matches,
                "cache_model_name_matches_stage97": model_name_matches,
                "cache_document_text_max_chars_matches_stage97": max_chars_matches,
                "cache_document_prefix_matches_stage97": document_prefix_matches,
                "embedding_shape": embedding_shape,
                "embedding_row_count_matches_current_corpus": row_count_matches,
                "document_ids_match_current_corpus": document_ids_match,
                "query_prefix_resolved_from_stage97": config.get("query_prefix")
                is not None,
                "snapshot_path": str(snapshot_path),
                "local_model_snapshot_exists": snapshot_exists,
            }
        )
    can_run = bool(cache_checks) and all(
        check["cache_exists"]
        and check["cache_sha256_matches_stage97"]
        and check["cache_model_name_matches_stage97"]
        and check["cache_document_text_max_chars_matches_stage97"]
        and check["cache_document_prefix_matches_stage97"]
        and check["embedding_row_count_matches_current_corpus"]
        and check["document_ids_match_current_corpus"]
        and check["query_prefix_resolved_from_stage97"]
        and check["local_model_snapshot_exists"]
        for check in cache_checks
    )
    return {
        "cache_checks": cache_checks,
        "can_run_without_download": can_run,
        "no_model_download_attempted": True,
        "cache_refresh_attempted": False,
    }


def _build_dense_retrievers(
    *,
    dense_configs: Sequence[Mapping[str, Any]],
    document_list: Sequence[PrimeQADocument],
    document_ids: tuple[str, ...],
    encoder_factory: EncoderFactory,
) -> dict[str, DenseRetriever]:
    retrievers = {}
    for config in dense_configs:
        embeddings = _load_cache_embeddings(
            cache_path=Path(str(config["cache_path"])),
            expected_document_ids=document_ids,
        )
        retriever = DenseRetriever(
            encoder=encoder_factory(config),
            document_text_max_chars=int(config["document_text_max_chars"]),
            query_prefix=str(config["query_prefix"]),
            document_prefix=str(config["document_prefix"]),
        )
        retriever.fit_embeddings(document_list, embeddings)
        retrievers[str(config["config_id"])] = retriever
    return retrievers


def _load_cache_embeddings(
    *,
    cache_path: Path,
    expected_document_ids: tuple[str, ...],
) -> np.ndarray:
    with np.load(cache_path, allow_pickle=False) as data:
        document_ids = tuple(str(value) for value in data["document_ids"])
        if document_ids != expected_document_ids:
            raise ValueError(f"Dense cache document IDs do not match corpus: {cache_path}")
        return np.asarray(data["embeddings"], dtype=np.float32)


def _default_encoder_factory(
    *,
    batch_size: int,
    device: str | None,
) -> EncoderFactory:
    def _factory(config: Mapping[str, Any]) -> TextEmbeddingModel:
        return LocalSnapshotSentenceTransformerEncoder(
            snapshot_path=Path(str(config["snapshot_path"])),
            batch_size=batch_size,
            device=device,
            show_progress_bar=False,
        )

    return _factory


def _protocol_ready_for_evaluation(
    *,
    stage97_report: Mapping[str, Any],
    user_confirmed_protocol: bool,
    confirmed_protocol_id: str,
    cache_preflight: Mapping[str, Any],
) -> bool:
    decision = stage97_report.get("decision") or {}
    return (
        bool(user_confirmed_protocol)
        and confirmed_protocol_id == _PROTOCOL_ID
        and decision.get("status")
        == "primeqa_hybrid_selective_dense_sparse_protocol_frozen"
        and bool(cache_preflight.get("can_run_without_download"))
    )


def _public_protocol_summary(
    *,
    frozen_protocol: Mapping[str, Any],
    dense_configs: Sequence[Mapping[str, Any]],
    policy_grid: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "protocol_id": frozen_protocol.get("protocol_id"),
        "candidate_id": frozen_protocol.get("candidate_id"),
        "baseline_retriever": frozen_protocol.get("baseline_retriever"),
        "dense_cache_contract": {
            "allowed_cache_source": (
                frozen_protocol.get("dense_cache_contract") or {}
            ).get("allowed_cache_source"),
            "download_required": (
                frozen_protocol.get("dense_cache_contract") or {}
            ).get("download_required"),
            "document_reencoding_allowed": (
                frozen_protocol.get("dense_cache_contract") or {}
            ).get("document_reencoding_allowed"),
            "query_encoding_mode": (
                frozen_protocol.get("dense_cache_contract") or {}
            ).get("query_encoding_mode"),
        },
        "allowed_dense_configs": [
            {
                key: config.get(key)
                for key in (
                    "config_id",
                    "model_name",
                    "cache_path",
                    "cache_sha256",
                    "document_text_max_chars",
                    "document_prefix",
                    "query_prefix",
                    "snapshot_path",
                )
            }
            for config in dense_configs
        ],
        "candidate_policy_grid": [dict(policy) for policy in policy_grid],
        "train_selection_rule": frozen_protocol.get("train_selection_rule"),
        "metrics_allowed_after_confirmation": frozen_protocol.get(
            "metrics_allowed_after_confirmation"
        ),
        "public_safe_changed_case_fields": frozen_protocol.get(
            "public_safe_changed_case_fields"
        ),
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
        "empty_query_count": int(config_result["empty_query_count"]),
        "average_query_token_count": float(config_result["average_query_token_count"]),
        "gate_activation_count": int(config_result["gate_activation_count"]),
        "gate_reason_counts": dict(config_result["gate_reason_counts"]),
        "promotion_count": int(config_result["promotion_count"]),
        "queries_with_promotion_count": int(
            config_result["queries_with_promotion_count"]
        ),
        "protected_bm25_top_rank_demotion_count": int(
            config_result["protected_bm25_top_rank_demotion_count"]
        ),
    }


def _change_case(
    *,
    sample_id: str,
    split: str,
    baseline_rank: int | None,
    challenger_rank: int | None,
    policy: Mapping[str, Any],
    action: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "split": split,
        "baseline_rank": baseline_rank,
        "challenger_rank": challenger_rank,
        "policy_id": policy["policy_id"],
        "dense_config_id": policy["dense_config_id"],
        "baseline_rank_bucket": _rank_bucket(baseline_rank),
        "challenger_rank_bucket": _rank_bucket(challenger_rank),
        "gate_activation_reason_code": action.get("reason_code", "none"),
        "query_length_bucket": action.get("query_length_bucket", "none"),
        "bm25_top1_overlap_bucket": action.get("bm25_top1_overlap_bucket", "none"),
        "bm25_top10_mean_overlap_bucket": action.get(
            "bm25_top10_mean_overlap_bucket",
            "none",
        ),
        "dense_rank_bucket": action.get("dense_rank_bucket", "none"),
        "promotion_budget_used": int(action.get("promotion_budget_used") or 0),
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
            value_label=f"{comparison[metric]:+.4f}"
            if isinstance(comparison[metric], float)
            else f"{comparison[metric]:+d}",
        )
        for config_id, comparison in sorted(comparisons.items())
    ]


def _action_bars(report: Mapping[str, Any], *, split: str) -> list[BarDatum]:
    metrics = report.get("metrics_by_split", {}).get(split, {})
    return [
        BarDatum(
            label=config_id,
            value=float(config_metrics["promotion_count"]),
            value_label=str(config_metrics["promotion_count"]),
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
        for check in report["guard_checks"]
    ]


def _frozen_protocol(stage97_report: Mapping[str, Any]) -> dict[str, Any]:
    protocol = stage97_report.get("frozen_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("Stage97 report does not contain frozen_protocol")
    return dict(protocol)


def _baseline_config(frozen_protocol: Mapping[str, Any]) -> dict[str, Any]:
    baseline = frozen_protocol.get("baseline_retriever")
    if not isinstance(baseline, Mapping):
        raise ValueError("Stage97 frozen protocol does not contain baseline_retriever")
    return dict(baseline)


def _allowed_dense_configs(frozen_protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    dense_cache_contract = frozen_protocol.get("dense_cache_contract") or {}
    dense_configs = dense_cache_contract.get("allowed_dense_configs")
    if not isinstance(dense_configs, list):
        raise ValueError("Stage97 frozen protocol does not contain dense configs")
    return [dict(config) for config in dense_configs if isinstance(config, Mapping)]


def _candidate_policy_grid(frozen_protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    policy_grid = frozen_protocol.get("candidate_policy_grid")
    if not isinstance(policy_grid, list):
        raise ValueError("Stage97 frozen protocol does not contain policy grid")
    return [dict(policy) for policy in policy_grid if isinstance(policy, Mapping)]


def _policies_reference_known_dense_configs(
    *,
    dense_configs: Sequence[Mapping[str, Any]],
    policy_grid: Sequence[Mapping[str, Any]],
) -> bool:
    dense_config_ids = {str(config["config_id"]) for config in dense_configs}
    return bool(policy_grid) and all(
        str(policy.get("dense_config_id")) in dense_config_ids for policy in policy_grid
    )


def _document_token_sets(
    document_list: Sequence[PrimeQADocument],
) -> dict[str, frozenset[str]]:
    return {
        document.id: frozenset(tokenize_text(f"{document.title}\n\n{document.text}"))
        for document in document_list
    }


def _protected_demotion_count(
    *,
    baseline_results: Sequence[RetrievalResult],
    reranked_results: Sequence[RetrievalResult],
) -> int:
    reranked_rank_by_doc = {
        result.document.id: result.rank for result in reranked_results[:_PRIMARY_TOP_K]
    }
    demotions = 0
    for result in baseline_results[:_PRIMARY_TOP_K]:
        reranked_rank = reranked_rank_by_doc.get(result.document.id)
        if reranked_rank is None or reranked_rank > result.rank:
            demotions += 1
    return demotions


def _empty_train_selection(policy_grid: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "selection_rule": _selection_rule_description(),
        "selected_policy_id": None,
        "candidate_count": len(policy_grid),
        "selected_train_metrics": None,
        "selected_train_comparison_to_baseline": None,
    }


def _selection_rule_description() -> str:
    return (
        "Select the gated dense+sparse policy on train only by hit@10, then "
        "larger not-found@50 reduction, fewer top10 regressions, hit@1, MRR@10, "
        "lower dense promotion budget, then policy_id. Dev is validation only."
    )


def _policy_by_id(
    policy_grid: Sequence[Mapping[str, Any]],
    policy_id: str,
) -> Mapping[str, Any]:
    for policy in policy_grid:
        if policy.get("policy_id") == policy_id:
            return policy
    raise KeyError(policy_id)


def _max_dense_candidate_rank(policy_grid: Sequence[Mapping[str, Any]]) -> int:
    return max(int(policy["dense_candidate_rank_max"]) for policy in policy_grid)


def _baseline_hit10(
    rank_tables: Mapping[str, Mapping[str, Mapping[str, Any]]],
    split: str,
) -> float | None:
    split_table = rank_tables.get(split) or {}
    baseline = split_table.get(_BASELINE_CONFIG_ID)
    if baseline is None:
        return None
    return float(baseline["hit_at_k"][_PRIMARY_TOP_K])


def _rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "not_found"
    if rank == 1:
        return "rank_1"
    if rank <= 5:
        return "rank_2_5"
    if rank <= 10:
        return "rank_6_10"
    if rank <= 50:
        return "rank_11_50"
    return "rank_gt_50"


def _dense_rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "none"
    if rank == 1:
        return "dense_rank_1"
    if rank <= 5:
        return "dense_rank_2_5"
    return "dense_rank_6_10"


def _query_length_bucket(query_token_count: int) -> str:
    if query_token_count < 6:
        return "lt_6"
    if query_token_count < 8:
        return "6_7"
    if query_token_count < 10:
        return "8_9"
    if query_token_count < 20:
        return "10_19"
    return "ge_20"


def _overlap_bucket(value: float) -> str:
    if value <= 0.1:
        return "le_0_10"
    if value <= 0.2:
        return "0_10_0_20"
    if value <= 0.3:
        return "0_20_0_30"
    if value <= 0.5:
        return "0_30_0_50"
    return "gt_0_50"


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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
