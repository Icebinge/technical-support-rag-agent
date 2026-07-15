from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever, tokenize_text
from ts_rag_agent.infrastructure.dense_retriever import DenseRetriever, TextEmbeddingModel
from ts_rag_agent.infrastructure.hybrid_retriever import HybridRetriever
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_documents

_STAGE = "Stage 81"
_CREATED_AT = "2026-07-15"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})
_BASELINE_CONFIG_ID = "full_document_bm25_baseline"
_CONFIRMED_COMPARE_PROTOCOL = "compare_existing_cached_dense_models"
_PRIMARY_TOP_K = 10
_DEFAULT_SEARCH_DEPTH = 50
_DEFAULT_CANDIDATE_TOP_K = 100
_DEFAULT_RRF_K = 60


class EncoderFactory(Protocol):
    """Build a query encoder for one Stage81 dense cache configuration."""

    def __call__(self, config: Mapping[str, Any]) -> TextEmbeddingModel:
        """Return a text embedding model for the supplied cache config."""


@dataclass(frozen=True)
class PrimeQAHybridDenseSparseRRFComparisonVisualization:
    """One generated Stage81 dense+sparse RRF comparison visualization."""

    name: str
    path: str


class LocalSnapshotSentenceTransformerEncoder:
    """SentenceTransformer query encoder locked to a local snapshot path."""

    def __init__(
        self,
        *,
        snapshot_path: Path,
        batch_size: int = 64,
        device: str | None = None,
        show_progress_bar: bool = False,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.snapshot_path = snapshot_path
        self._batch_size = batch_size
        self._show_progress_bar = show_progress_bar
        self._model = SentenceTransformer(
            str(snapshot_path),
            device=device,
            local_files_only=True,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=self._show_progress_bar,
        )
        return np.asarray(embeddings, dtype=np.float32)


def run_primeqa_hybrid_dense_sparse_rrf_comparison(
    *,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    stage75_report_path: Path,
    stage80_report_path: Path,
    user_confirmed_protocol: str = _CONFIRMED_COMPARE_PROTOCOL,
    top_k_values: tuple[int, ...] = (1, 5, 10),
    search_depth: int = _DEFAULT_SEARCH_DEPTH,
    candidate_top_k: int = _DEFAULT_CANDIDATE_TOP_K,
    rrf_k: int = _DEFAULT_RRF_K,
    sparse_weight: float = 1.0,
    dense_weight: float = 1.0,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    encoder_batch_size: int = 64,
    encoder_device: str | None = None,
    encoder_factory: EncoderFactory | None = None,
) -> dict[str, Any]:
    """Run train/dev-only dense+sparse RRF comparison over confirmed local caches."""

    _validate_options(
        top_k_values=top_k_values,
        search_depth=search_depth,
        candidate_top_k=candidate_top_k,
        rrf_k=rrf_k,
        sparse_weight=sparse_weight,
        dense_weight=dense_weight,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )
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
    stage80_report = _load_json_object(stage80_report_path)
    loaded_inputs_at = time.perf_counter()

    dense_cache_configs = _dense_cache_configs_from_stage80(
        stage80_report=stage80_report,
        current_document_ids=document_ids,
        user_confirmed_protocol=user_confirmed_protocol,
    )
    cache_preflight = _preflight_dense_caches(
        dense_cache_configs=dense_cache_configs,
        current_document_ids=document_ids,
    )
    preflight_at = time.perf_counter()

    rank_tables: dict[str, dict[str, dict[str, Any]]] = {}
    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    train_selection: dict[str, Any] = {
        "selection_rule": _selection_rule_description(),
        "selected_config_id": None,
        "candidate_count": len(dense_cache_configs),
        "selected_train_metrics": None,
    }
    indexed_at = preflight_at
    evaluated_at = preflight_at
    if _can_run_evaluation(cache_preflight):
        encoder_factory = encoder_factory or _default_encoder_factory(
            batch_size=encoder_batch_size,
            device=encoder_device,
        )
        baseline_retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
        baseline_retriever.fit(document_list)
        retrievers: dict[str, Any] = {_BASELINE_CONFIG_ID: baseline_retriever}
        for config in dense_cache_configs:
            embeddings = _load_cache_embeddings(
                cache_path=Path(config["cache_path"]),
                expected_document_ids=document_ids,
            )
            dense_retriever = DenseRetriever(
                encoder=encoder_factory(config),
                document_text_max_chars=int(config["document_text_max_chars"]),
                query_prefix=str(config["query_prefix"]),
                document_prefix=str(config["document_prefix"]),
            )
            dense_retriever.fit_embeddings(document_list, embeddings)
            retrievers[str(config["config_id"])] = HybridRetriever(
                sparse_retriever=baseline_retriever,
                dense_retriever=dense_retriever,
                candidate_top_k=candidate_top_k,
                rrf_k=rrf_k,
                sparse_weight=sparse_weight,
                dense_weight=dense_weight,
            )
        indexed_at = time.perf_counter()

        rank_tables = {
            split: _evaluate_split(
                split=split,
                samples=samples,
                retrievers=retrievers,
                top_k_values=top_k_values,
                search_depth=search_depth,
            )
            for split, samples in split_samples.items()
        }
        evaluated_at = time.perf_counter()
        comparisons = {
            split: {
                config_id: _compare_to_baseline(
                    baseline=rank_tables[split][_BASELINE_CONFIG_ID],
                    challenger=rank_tables[split][config_id],
                    max_k=_PRIMARY_TOP_K,
                    search_depth=search_depth,
                )
                for config_id in _candidate_config_ids(dense_cache_configs)
            }
            for split in _ALLOWED_DEVELOPMENT_SPLITS
        }
        train_selection = _select_challenger_on_train(
            rank_tables=rank_tables,
            dense_cache_configs=dense_cache_configs,
        )

    guard_checks = _guard_checks(
        split_samples=split_samples,
        rank_tables=rank_tables,
        cache_preflight=cache_preflight,
        dense_cache_configs=dense_cache_configs,
        stage75_report=stage75_report,
        stage80_report=stage80_report,
        user_confirmed_protocol=user_confirmed_protocol,
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
            "Train/dev-only dense+sparse RRF comparison for the confirmed "
            "compare_existing_cached_dense_models protocol. This stage evaluates "
            "only existing local dense caches from Stage80, encodes queries from "
            "local SentenceTransformer snapshots with local_files_only, keeps the "
            "frozen test split locked, does not run final metrics, does not use "
            "source DOC_IDS as runtime retrieval evidence, does not download "
            "models, and does not change runtime defaults."
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
            "stage80_report": _fingerprint(stage80_report_path),
        },
        "config": {
            "user_confirmed_protocol": user_confirmed_protocol,
            "top_k_values": list(top_k_values),
            "primary_top_k": _PRIMARY_TOP_K,
            "search_depth": search_depth,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "candidate_top_k": candidate_top_k,
            "rrf_k": rrf_k,
            "sparse_weight": sparse_weight,
            "dense_weight": dense_weight,
            "baseline_config_id": _BASELINE_CONFIG_ID,
            "selection_rule": _selection_rule_description(),
            "model_load_mode": "local_snapshot_path_with_local_files_only",
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
            "dense_cache_config_count": len(dense_cache_configs),
            "test_split_loaded": False,
        },
        "dense_cache_configs": _public_dense_cache_configs(dense_cache_configs),
        "dense_cache_preflight": cache_preflight,
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
            "dense_cache_preflight": round(preflight_at - loaded_inputs_at, 3),
            "bm25_and_dense_indexes": round(indexed_at - preflight_at, 3),
            "dense_sparse_rrf_evaluate": round(evaluated_at - indexed_at, 3),
            "guard_checks": round(checked_at - evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_dense_sparse_rrf_comparison_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridDenseSparseRRFComparisonVisualization]:
    """Write SVG charts for Stage81 dense+sparse RRF comparison."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage81_dense_sparse_rrf_train_hit_at_10.svg": render_horizontal_bar_chart_svg(
            title="Stage81 train hit@10 by retrieval config",
            bars=_hit_at_k_bars(report, split=_TRAIN_SPLIT, top_k=_PRIMARY_TOP_K),
            x_label="hit@10",
            width=1180,
            margin_left=510,
        ),
        "stage81_dense_sparse_rrf_dev_hit_at_10.svg": render_horizontal_bar_chart_svg(
            title="Stage81 dev hit@10 by retrieval config",
            bars=_hit_at_k_bars(report, split=_DEV_SPLIT, top_k=_PRIMARY_TOP_K),
            x_label="hit@10",
            width=1180,
            margin_left=510,
        ),
        "stage81_dense_sparse_rrf_dev_delta_hit_at_10.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage81 dev hit@10 delta vs baseline",
                bars=_delta_bars(report, split=_DEV_SPLIT, metric="hit@10_delta"),
                x_label="delta hit@10",
                width=1180,
                margin_left=510,
            )
        ),
        "stage81_dense_sparse_rrf_dev_not_found_at_50.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage81 dev not found within top50",
                bars=_not_found_bars(report, split=_DEV_SPLIT),
                x_label="answer docs not found",
                width=1180,
                margin_left=510,
            )
        ),
        "stage81_dense_sparse_rrf_dev_top10_changes.svg": render_horizontal_bar_chart_svg(
            title="Stage81 dev top10 improvements minus regressions",
            bars=_net_change_bars(report, split=_DEV_SPLIT),
            x_label="net changed cases",
            width=1180,
            margin_left=510,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridDenseSparseRRFComparisonVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _default_encoder_factory(
    *,
    batch_size: int,
    device: str | None,
) -> EncoderFactory:
    def factory(config: Mapping[str, Any]) -> TextEmbeddingModel:
        return LocalSnapshotSentenceTransformerEncoder(
            snapshot_path=Path(str(config["snapshot_path"])),
            batch_size=batch_size,
            device=device,
        )

    return factory


def _dense_cache_configs_from_stage80(
    *,
    stage80_report: Mapping[str, Any],
    current_document_ids: tuple[str, ...],
    user_confirmed_protocol: str,
) -> list[dict[str, Any]]:
    if user_confirmed_protocol != _CONFIRMED_COMPARE_PROTOCOL:
        return []
    configs = []
    for candidate in stage80_report.get("dense_cache_candidates") or []:
        if not _stage80_candidate_is_eligible(candidate):
            continue
        query_prefix, query_prefix_source = _query_prefix_from_stage80_candidate(candidate)
        snapshot_path, snapshot_status = _snapshot_path_from_stage80_candidate(candidate)
        config_id = _dense_sparse_config_id(
            model_name=str(candidate["model_name"]),
            document_text_max_chars=int(candidate["document_text_max_chars"]),
            document_prefix=str(candidate.get("document_prefix") or ""),
        )
        configs.append(
            {
                "config_id": config_id,
                "model_name": str(candidate["model_name"]),
                "cache_path": str(candidate["cache_path"]),
                "cache_sha256": str(candidate.get("cache_sha256") or ""),
                "document_text_max_chars": int(candidate["document_text_max_chars"]),
                "document_prefix": str(candidate.get("document_prefix") or ""),
                "query_prefix": query_prefix,
                "query_prefix_source": query_prefix_source,
                "embedding_shape": list(candidate.get("embedding_shape") or []),
                "document_id_count": int(candidate.get("document_id_count") or 0),
                "document_ids_match_current_corpus_in_stage80": bool(
                    candidate.get("document_ids_match_current_corpus")
                ),
                "current_document_id_count": len(current_document_ids),
                "can_run_without_model_download_in_stage80": bool(
                    candidate.get("can_run_without_model_download")
                ),
                "snapshot_path": str(snapshot_path) if snapshot_path is not None else None,
                "snapshot_status": snapshot_status,
            }
        )
    return sorted(configs, key=lambda config: str(config["config_id"]))


def _stage80_candidate_is_eligible(candidate: Mapping[str, Any]) -> bool:
    return bool(candidate.get("document_ids_match_current_corpus")) and bool(
        candidate.get("can_run_without_model_download")
    )


def _query_prefix_from_stage80_candidate(
    candidate: Mapping[str, Any],
) -> tuple[str | None, str]:
    matches = candidate.get("legacy_metric_matches") or []
    dense_matches = [match for match in matches if match.get("method") == "dense"]
    for match in dense_matches:
        if "query_prefix" in match:
            value = match.get("query_prefix")
            return "" if value is None else str(value), "stage80_legacy_dense_metric"
    for match in matches:
        if "query_prefix" in match:
            value = match.get("query_prefix")
            return "" if value is None else str(value), "stage80_legacy_metric"
    return None, "missing"


def _snapshot_path_from_stage80_candidate(
    candidate: Mapping[str, Any],
) -> tuple[Path | None, str]:
    model_cache = candidate.get("huggingface_model_cache") or {}
    model_cache_dir = model_cache.get("model_cache_dir")
    refs_main = model_cache.get("refs_main")
    snapshots = tuple(str(value) for value in model_cache.get("snapshots") or [])
    if not model_cache_dir:
        return None, "missing_model_cache_dir"
    if refs_main and refs_main in snapshots:
        return Path(str(model_cache_dir)) / "snapshots" / str(refs_main), "refs_main"
    return None, "missing_refs_main_snapshot"


def _preflight_dense_caches(
    *,
    dense_cache_configs: Sequence[Mapping[str, Any]],
    current_document_ids: tuple[str, ...],
) -> dict[str, Any]:
    cache_checks = []
    for config in dense_cache_configs:
        cache_path = Path(str(config["cache_path"]))
        snapshot_path = (
            Path(str(config["snapshot_path"]))
            if config.get("snapshot_path") is not None
            else None
        )
        cache_exists = cache_path.exists() and cache_path.is_file()
        snapshot_exists = (
            snapshot_path is not None and snapshot_path.exists() and snapshot_path.is_dir()
        )
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
        query_prefix_resolved = config.get("query_prefix") is not None
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
                "cache_model_name_matches_stage80": model_name_matches,
                "cache_document_text_max_chars_matches_stage80": max_chars_matches,
                "cache_document_prefix_matches_stage80": document_prefix_matches,
                "embedding_shape": embedding_shape,
                "embedding_row_count_matches_current_corpus": row_count_matches,
                "document_ids_match_current_corpus": document_ids_match,
                "query_prefix_resolved_from_stage80": query_prefix_resolved,
                "query_prefix_source": str(config["query_prefix_source"]),
                "snapshot_path": str(snapshot_path) if snapshot_path is not None else None,
                "local_model_snapshot_exists": snapshot_exists,
            }
        )
    can_run = bool(cache_checks) and all(
        check["cache_exists"]
        and check["cache_model_name_matches_stage80"]
        and check["cache_document_text_max_chars_matches_stage80"]
        and check["cache_document_prefix_matches_stage80"]
        and check["embedding_row_count_matches_current_corpus"]
        and check["document_ids_match_current_corpus"]
        and check["query_prefix_resolved_from_stage80"]
        and check["local_model_snapshot_exists"]
        for check in cache_checks
    )
    return {
        "cache_checks": cache_checks,
        "can_run_evaluation_without_download": can_run,
        "no_model_download_attempted": True,
    }


def _can_run_evaluation(cache_preflight: Mapping[str, Any]) -> bool:
    return bool(cache_preflight.get("can_run_evaluation_without_download"))


def _load_cache_embeddings(
    *,
    cache_path: Path,
    expected_document_ids: tuple[str, ...],
) -> np.ndarray:
    with np.load(cache_path, allow_pickle=False) as data:
        cached_document_ids = tuple(str(value) for value in data["document_ids"])
        if cached_document_ids != expected_document_ids:
            raise ValueError(f"Dense cache document IDs do not match: {cache_path}")
        return np.asarray(data["embeddings"], dtype=np.float32)


def _evaluate_split(
    *,
    split: str,
    samples: Sequence[PrimeQAHybridSplitSample],
    retrievers: Mapping[str, Any],
    top_k_values: tuple[int, ...],
    search_depth: int,
) -> dict[str, Any]:
    answerable_samples = [
        sample
        for sample in samples
        if sample.answerable and sample.answer_doc_id is not None
    ]
    accumulators = {
        config_id: _empty_accumulator(
            split=split,
            config_id=config_id,
            total_questions=len(samples),
        )
        for config_id in retrievers
    }
    for sample in answerable_samples:
        query = sample.to_primeqa_question().full_question
        query_token_count = len(tokenize_text(query))
        for config_id, retriever in retrievers.items():
            results = retriever.search(query, top_k=search_depth)
            _record_result(
                accumulator=accumulators[config_id],
                sample=sample,
                results=results,
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
    }


def _record_result(
    *,
    accumulator: dict[str, Any],
    sample: PrimeQAHybridSplitSample,
    results: Sequence[RetrievalResult],
    query_token_count: int,
    top_k_values: tuple[int, ...],
    search_depth: int,
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
    not_found_delta = int(challenger["not_found_count_at_search_depth"]) - int(
        baseline["not_found_count_at_search_depth"]
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
        "not_found_count_at_search_depth_delta": not_found_delta,
        "both_hit_count": both_hit,
        "both_miss_count": both_miss,
        "rank_up_within_top10_count": rank_up,
        "rank_down_within_top10_count": rank_down,
        "sample_top10_improvements": top10_improvements[:20],
        "sample_top10_regressions": top10_regressions[:20],
        "sample_search_depth_improvements": search_depth_improvements[:20],
        "sample_search_depth_regressions": search_depth_regressions[:20],
    }


def _select_challenger_on_train(
    *,
    rank_tables: Mapping[str, Mapping[str, Mapping[str, Any]]],
    dense_cache_configs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_ids = _candidate_config_ids(dense_cache_configs)
    train_results = rank_tables[_TRAIN_SPLIT]
    selected_config_id = sorted(
        candidate_ids,
        key=lambda config_id: (
            -float(train_results[config_id]["hit_at_k"][_PRIMARY_TOP_K]),
            -float(train_results[config_id]["hit_at_k"].get(5, 0.0)),
            -float(train_results[config_id]["hit_at_k"].get(1, 0.0)),
            -float(train_results[config_id]["mrr_at_10"]),
            config_id,
        ),
    )[0]
    return {
        "selection_rule": _selection_rule_description(),
        "candidate_count": len(candidate_ids),
        "selected_config_id": selected_config_id,
        "selected_train_metrics": _public_config_metrics(train_results[selected_config_id]),
    }


def _guard_checks(
    *,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    rank_tables: Mapping[str, Mapping[str, Mapping[str, Any]]],
    cache_preflight: Mapping[str, Any],
    dense_cache_configs: Sequence[Mapping[str, Any]],
    stage75_report: Mapping[str, Any],
    stage80_report: Mapping[str, Any],
    user_confirmed_protocol: str,
    top_k_values: tuple[int, ...],
    search_depth: int,
) -> list[dict[str, Any]]:
    observed_split_names = sorted(
        {sample.assigned_split for samples in split_samples.values() for sample in samples}
    )
    expected_splits = sorted(_ALLOWED_DEVELOPMENT_SPLITS)
    stage75_split_reports = stage75_report.get("split_reports") or {}
    stage80_decision = stage80_report.get("decision") or {}
    stage80_options = stage80_report.get("candidate_options") or []
    stage80_protocol_available = any(
        option.get("option_id") == _CONFIRMED_COMPARE_PROTOCOL
        and bool(option.get("eligible"))
        for option in stage80_options
    )
    selected_cache_count = len(dense_cache_configs)
    baseline_train_hit10 = _baseline_hit10(rank_tables, _TRAIN_SPLIT)
    baseline_dev_hit10 = _baseline_hit10(rank_tables, _DEV_SPLIT)
    stage75_train_hit10 = (
        (stage75_split_reports.get(_TRAIN_SPLIT) or {}).get("hit_at_top_k")
    )
    stage75_dev_hit10 = (
        (stage75_split_reports.get(_DEV_SPLIT) or {}).get("hit_at_top_k")
    )
    cache_checks = cache_preflight.get("cache_checks") or []
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
            name="stage80_source_report_is_stage80",
            passed=str(stage80_report.get("stage") or "") == "Stage 80",
            observed=str(stage80_report.get("stage") or ""),
            expected="Stage 80",
        ),
        _check(
            name="stage80_can_run_dense_sparse_rrf_without_download",
            passed=stage80_decision.get("can_run_dense_sparse_rrf_without_download")
            is True,
            observed=stage80_decision.get("can_run_dense_sparse_rrf_without_download"),
            expected=True,
        ),
        _check(
            name="stage80_requires_user_confirmation_before_train_dev_run",
            passed=stage80_decision.get("requires_user_confirmation_before_train_dev_run")
            is True,
            observed=stage80_decision.get(
                "requires_user_confirmation_before_train_dev_run"
            ),
            expected=True,
        ),
        _check(
            name="user_confirmed_protocol_matches_stage80_option",
            passed=(
                user_confirmed_protocol == _CONFIRMED_COMPARE_PROTOCOL
                and stage80_protocol_available
            ),
            observed=user_confirmed_protocol,
            expected=_CONFIRMED_COMPARE_PROTOCOL,
        ),
        _check(
            name="selected_cache_count_matches_compare_protocol",
            passed=selected_cache_count >= 2,
            observed=selected_cache_count,
            expected=">= 2",
        ),
        _check(
            name="dense_cache_files_exist",
            passed=bool(cache_checks)
            and all(check["cache_exists"] for check in cache_checks),
            observed=[
                {"config_id": check["config_id"], "cache_exists": check["cache_exists"]}
                for check in cache_checks
            ],
            expected=True,
        ),
        _check(
            name="dense_cache_metadata_matches_stage80",
            passed=bool(cache_checks)
            and all(
                check["cache_model_name_matches_stage80"]
                and check["cache_document_text_max_chars_matches_stage80"]
                and check["cache_document_prefix_matches_stage80"]
                for check in cache_checks
            ),
            observed=[
                {
                    "config_id": check["config_id"],
                    "model_name": check["cache_model_name_matches_stage80"],
                    "max_chars": check[
                        "cache_document_text_max_chars_matches_stage80"
                    ],
                    "document_prefix": check["cache_document_prefix_matches_stage80"],
                }
                for check in cache_checks
            ],
            expected=True,
        ),
        _check(
            name="dense_cache_document_ids_match_current_corpus",
            passed=bool(cache_checks)
            and all(check["document_ids_match_current_corpus"] for check in cache_checks),
            observed=[
                {
                    "config_id": check["config_id"],
                    "document_ids_match": check["document_ids_match_current_corpus"],
                }
                for check in cache_checks
            ],
            expected=True,
        ),
        _check(
            name="dense_cache_embedding_rows_match_current_corpus",
            passed=bool(cache_checks)
            and all(
                check["embedding_row_count_matches_current_corpus"]
                for check in cache_checks
            ),
            observed=[
                {
                    "config_id": check["config_id"],
                    "embedding_shape": check["embedding_shape"],
                    "row_count_match": check[
                        "embedding_row_count_matches_current_corpus"
                    ],
                }
                for check in cache_checks
            ],
            expected=True,
        ),
        _check(
            name="query_prefix_protocol_resolved_from_stage80",
            passed=bool(cache_checks)
            and all(
                check["query_prefix_resolved_from_stage80"] for check in cache_checks
            ),
            observed=[
                {
                    "config_id": check["config_id"],
                    "resolved": check["query_prefix_resolved_from_stage80"],
                    "source": check["query_prefix_source"],
                }
                for check in cache_checks
            ],
            expected=True,
        ),
        _check(
            name="local_model_snapshots_exist",
            passed=bool(cache_checks)
            and all(check["local_model_snapshot_exists"] for check in cache_checks),
            observed=[
                {
                    "config_id": check["config_id"],
                    "snapshot_exists": check["local_model_snapshot_exists"],
                }
                for check in cache_checks
            ],
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
            name="no_model_download_attempted",
            passed=cache_preflight.get("no_model_download_attempted") is True,
            observed=cache_preflight.get("no_model_download_attempted"),
            expected=True,
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
            "status": "primeqa_hybrid_dense_sparse_rrf_comparison_blocked",
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
            "Stage 82: review dense+sparse RRF changed cases on dev and decide "
            "whether a guarded runtime experiment is justified; keep test locked."
        )
    else:
        recommended_next_stage = (
            "Stage 82: move to the remaining Stage76 BM25 k1/b grid candidate "
            "on train/dev; keep test locked and do not open final metrics."
        )
    return {
        "status": "primeqa_hybrid_dense_sparse_rrf_comparison_completed",
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
        "empty_query_count": int(config_result["empty_query_count"]),
        "average_query_token_count": float(config_result["average_query_token_count"]),
    }


def _public_dense_cache_configs(
    dense_cache_configs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    keys = (
        "config_id",
        "model_name",
        "cache_path",
        "cache_sha256",
        "document_text_max_chars",
        "document_prefix",
        "query_prefix",
        "query_prefix_source",
        "embedding_shape",
        "document_id_count",
        "can_run_without_model_download_in_stage80",
        "snapshot_path",
        "snapshot_status",
    )
    return [{key: config.get(key) for key in keys} for config in dense_cache_configs]


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
            value_label=f"{comparison[metric]:+.4f}",
        )
        for config_id, comparison in sorted(comparisons.items())
    ]


def _not_found_bars(report: Mapping[str, Any], *, split: str) -> list[BarDatum]:
    metrics = report.get("metrics_by_split", {}).get(split, {})
    ordered = sorted(
        metrics.items(),
        key=lambda item: (item[1]["not_found_count_at_search_depth"], item[0]),
    )
    return [
        BarDatum(
            label=config_id,
            value=float(config_metrics["not_found_count_at_search_depth"]),
            value_label=str(config_metrics["not_found_count_at_search_depth"]),
        )
        for config_id, config_metrics in ordered
    ]


def _net_change_bars(report: Mapping[str, Any], *, split: str) -> list[BarDatum]:
    comparisons = report.get("comparisons_to_baseline", {}).get(split, {})
    return [
        BarDatum(
            label=config_id,
            value=float(comparison["top10_net_improvement_count"]),
            value_label=str(comparison["top10_net_improvement_count"]),
        )
        for config_id, comparison in sorted(comparisons.items())
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


def _candidate_config_ids(
    dense_cache_configs: Sequence[Mapping[str, Any]],
) -> list[str]:
    return [str(config["config_id"]) for config in dense_cache_configs]


def _dense_sparse_config_id(
    *,
    model_name: str,
    document_text_max_chars: int,
    document_prefix: str,
) -> str:
    prefix_label = "noprefix" if not document_prefix else _safe_identifier(document_prefix)
    return (
        f"dense_sparse_rrf__{_safe_identifier(model_name)}__"
        f"{document_text_max_chars}_{prefix_label}"
    )


def _safe_identifier(value: str) -> str:
    text = "".join(char if char.isalnum() else "_" for char in value).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def _selection_rule_description() -> str:
    return (
        "Select among confirmed dense+sparse RRF challengers on train only by "
        "hit@10, then hit@5, then hit@1, then MRR@10, then config_id; dev is "
        "held out for validation inside the development split boundary."
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


def _validate_options(
    *,
    top_k_values: tuple[int, ...],
    search_depth: int,
    candidate_top_k: int,
    rrf_k: int,
    sparse_weight: float,
    dense_weight: float,
    bm25_k1: float,
    bm25_b: float,
) -> None:
    if not top_k_values:
        raise ValueError("top_k_values must not be empty")
    if any(top_k <= 0 for top_k in top_k_values):
        raise ValueError("top_k_values must be positive")
    if _PRIMARY_TOP_K not in top_k_values:
        raise ValueError(f"top_k_values must include {_PRIMARY_TOP_K}")
    if search_depth < max(top_k_values):
        raise ValueError("search_depth must be at least max(top_k_values)")
    if candidate_top_k < search_depth:
        raise ValueError("candidate_top_k must be at least search_depth")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if sparse_weight < 0 or dense_weight < 0:
        raise ValueError("retriever weights must be non-negative")
    if bm25_k1 <= 0:
        raise ValueError("bm25_k1 must be positive")
    if not 0 <= bm25_b <= 1:
        raise ValueError("bm25_b must be between 0 and 1")


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
