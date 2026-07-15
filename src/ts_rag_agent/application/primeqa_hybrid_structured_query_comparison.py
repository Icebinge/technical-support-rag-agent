from __future__ import annotations

import hashlib
import json
import math
import re
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

_STAGE = "Stage 89"
_CREATED_AT = "2026-07-15"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_PROTOCOL_ID = "structured_query_keyphrase_compaction_train_dev_v1"
_CANDIDATE_ID = "structured_query_keyphrase_compaction_design"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})
_BASELINE_CONFIG_ID = "full_document_bm25_baseline"
_BASELINE_QUERY_VIEW_ID = "full_question"
_BASELINE_K1 = 1.5
_BASELINE_B = 0.75
_PRIMARY_TOP_K = 10
_DEFAULT_SEARCH_DEPTH = 50
_EXPECTED_CONFIG_IDS = (
    "sqkc_action_error_product_v1",
    "sqkc_title_guarded_action_error_v1",
    "sqkc_error_first_compact_v1",
    "sqkc_noun_phrase_compact_v1",
)
_EXPECTED_QUERY_VIEW_IDS = (
    "action_error_product_version_terms",
    "title_guarded_action_error_product_terms",
    "error_identifier_first_terms",
    "deterministic_noun_phrase_like_terms",
)

_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "being",
        "but",
        "by",
        "can",
        "cannot",
        "could",
        "customer",
        "did",
        "do",
        "does",
        "during",
        "for",
        "from",
        "get",
        "getting",
        "had",
        "has",
        "have",
        "help",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "my",
        "need",
        "needs",
        "not",
        "of",
        "on",
        "or",
        "our",
        "please",
        "problem",
        "question",
        "saw",
        "see",
        "should",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "urgent",
        "use",
        "used",
        "using",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)
_ACTION_TERMS = frozenset(
    {
        "apply",
        "change",
        "configure",
        "connect",
        "create",
        "delete",
        "disable",
        "enable",
        "fix",
        "install",
        "migrate",
        "recover",
        "remove",
        "repair",
        "reset",
        "resolve",
        "restart",
        "run",
        "set",
        "start",
        "stop",
        "troubleshoot",
        "update",
        "upgrade",
    }
)
_ERROR_TERMS = frozenset(
    {
        "abend",
        "exception",
        "failure",
        "fault",
        "stacktrace",
        "traceback",
        "warning",
    }
)
_PLATFORM_TERMS = frozenset(
    {
        "aix",
        "db2",
        "ibm",
        "java",
        "linux",
        "mq",
        "oracle",
        "unix",
        "was",
        "websphere",
        "windows",
        "zos",
        "z/os",
    }
)


@dataclass(frozen=True)
class PrimeQAHybridStructuredQueryComparisonVisualization:
    """One generated Stage89 structured-query comparison visualization."""

    name: str
    path: str


@dataclass(frozen=True)
class StructuredQueryConfig:
    """One frozen structured-query compaction configuration."""

    config_id: str
    query_view_id: str
    preserved_feature_buckets: tuple[str, ...]
    maximum_unique_terms: int
    minimum_unique_terms: int
    minimum_title_terms: int
    noun_phrase_window_size: int


@dataclass(frozen=True)
class _QueryView:
    query_terms: tuple[str, ...]
    query_token_count: int
    query_unique_token_count: int
    compacted_query_token_count: int
    token_bucket_counts: dict[str, int]
    minimum_unique_terms_satisfied: bool
    maximum_unique_terms_applied: bool


class _BM25DocumentIndex:
    """Shared full-document BM25 index for baseline and compacted query views."""

    def __init__(self, documents: Sequence[PrimeQADocument]) -> None:
        self._documents = list(documents)
        self._doc_ids = np.asarray([document.id for document in self._documents])
        self._doc_lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = {}
        for doc_index, document in enumerate(self._documents):
            document_tokens = tokenize_text(f"{document.title}\n\n{document.text}")
            self._doc_lengths.append(len(document_tokens))
            for term, term_frequency in Counter(document_tokens).items():
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

    def rank_terms(
        self,
        *,
        query_terms: Sequence[str],
        search_depth: int,
    ) -> list[str]:
        if search_depth <= 0:
            raise ValueError("search_depth must be positive")
        if not query_terms or not self._documents:
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
        return [str(self._doc_ids[index]) for index in ranked_indices]


def run_primeqa_hybrid_structured_query_comparison(
    *,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    stage75_report_path: Path,
    stage88_report_path: Path,
    user_confirmed_protocol: bool,
    confirmed_protocol_id: str,
    confirmation_note: str,
    top_k_values: tuple[int, ...] = (1, 5, 10),
    search_depth: int = _DEFAULT_SEARCH_DEPTH,
) -> dict[str, Any]:
    """Run the confirmed train/dev-only structured query comparison."""

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
    stage88_report = _load_json_object(stage88_report_path)
    loaded_inputs_at = time.perf_counter()

    protocol = stage88_report.get("frozen_protocol") or {}
    protocol_configs = _configs_from_protocol(protocol)
    should_evaluate = (
        user_confirmed_protocol
        and confirmed_protocol_id == _PROTOCOL_ID
        and protocol.get("protocol_id") == _PROTOCOL_ID
        and _stage88_allows_metric_run(stage88_report)
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
        index = _BM25DocumentIndex(document_list)
        rank_tables = {
            split: _evaluate_split(
                split=split,
                samples=samples,
                index=index,
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
        stage88_report=stage88_report,
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
            "structured_query_keyphrase_compaction_train_dev_v1. This stage "
            "uses the user-confirmed Stage88 protocol, selects a candidate "
            "query view on train only, validates on dev, keeps the frozen test "
            "split locked, does not run final metrics, does not use source "
            "DOC_IDS as runtime retrieval evidence, does not write raw "
            "question, query, answer, or document text, and does not change "
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
            "stage88_report": _fingerprint(stage88_report_path),
        },
        "stage88_decision": stage88_report.get("decision") or {},
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
            "raw_query_terms_written_to_report": False,
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
            "bm25_baseline_and_structured_query_evaluate": round(
                indexed_and_evaluated_at - loaded_inputs_at,
                3,
            ),
            "guard_checks": round(checked_at - indexed_and_evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_structured_query_comparison_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridStructuredQueryComparisonVisualization]:
    """Write SVG charts for Stage89 structured query comparison."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage89_structured_query_train_hit_at_10.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage89 structured query train hit@10",
                bars=_hit_at_k_bars(report, split=_TRAIN_SPLIT, top_k=_PRIMARY_TOP_K),
                x_label="hit@10",
                width=1260,
                margin_left=470,
            )
        ),
        "stage89_structured_query_dev_hit_at_10.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage89 structured query dev hit@10",
                bars=_hit_at_k_bars(report, split=_DEV_SPLIT, top_k=_PRIMARY_TOP_K),
                x_label="hit@10",
                width=1260,
                margin_left=470,
            )
        ),
        "stage89_structured_query_dev_delta_hit_at_10.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage89 structured query dev hit@10 delta vs baseline",
                bars=_delta_bars(report, split=_DEV_SPLIT, metric="hit@10_delta"),
                x_label="delta hit@10",
                width=1260,
                margin_left=470,
            )
        ),
        "stage89_structured_query_dev_top10_changes.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage89 structured query dev top10 improvements minus regressions",
                bars=_net_change_bars(report, split=_DEV_SPLIT),
                x_label="net changed cases",
                width=1260,
                margin_left=470,
            )
        ),
        "stage89_structured_query_average_compacted_terms.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage89 structured query average compacted terms",
                bars=_average_compacted_terms_bars(report, split=_DEV_SPLIT),
                x_label="average terms",
                width=1260,
                margin_left=470,
            )
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridStructuredQueryComparisonVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _configs_from_protocol(
    protocol: Mapping[str, Any],
) -> tuple[StructuredQueryConfig, ...]:
    configs = []
    for row in protocol.get("candidate_config_grid") or []:
        if not isinstance(row, Mapping):
            continue
        configs.append(
            StructuredQueryConfig(
                config_id=str(row.get("config_id") or ""),
                query_view_id=str(row.get("query_view_id") or ""),
                preserved_feature_buckets=tuple(
                    str(bucket) for bucket in row.get("preserved_feature_buckets") or []
                ),
                maximum_unique_terms=int(row.get("maximum_unique_terms") or 0),
                minimum_unique_terms=int(row.get("minimum_unique_terms") or 0),
                minimum_title_terms=int(row.get("minimum_title_terms") or 0),
                noun_phrase_window_size=int(row.get("noun_phrase_window_size") or 2),
            )
        )
    return tuple(configs)


def _evaluate_split(
    *,
    split: str,
    samples: Sequence[PrimeQAHybridSplitSample],
    index: _BM25DocumentIndex,
    candidate_configs: Sequence[StructuredQueryConfig],
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
        query_view_id=_BASELINE_QUERY_VIEW_ID,
        is_baseline=True,
        total_questions=len(samples),
    )
    candidate_accumulators = {
        config.config_id: _empty_accumulator(
            split=split,
            config_id=config.config_id,
            query_view_id=config.query_view_id,
            is_baseline=False,
            total_questions=len(samples),
        )
        for config in candidate_configs
    }
    for sample in answerable_samples:
        full_query = sample.to_primeqa_question().full_question
        full_query_terms = tokenize_text(full_query)
        full_query_unique_terms = _unique_terms_in_order(full_query_terms)
        baseline_ranked_doc_ids = index.rank_terms(
            query_terms=full_query_terms,
            search_depth=search_depth,
        )
        baseline_rank = _answer_rank(
            ranked_doc_ids=baseline_ranked_doc_ids,
            answer_doc_id=str(sample.answer_doc_id),
        )
        baseline_query_view = _QueryView(
            query_terms=tuple(full_query_terms),
            query_token_count=len(full_query_terms),
            query_unique_token_count=len(full_query_unique_terms),
            compacted_query_token_count=len(full_query_terms),
            token_bucket_counts={},
            minimum_unique_terms_satisfied=True,
            maximum_unique_terms_applied=False,
        )
        _record_rank(
            accumulator=baseline_accumulator,
            sample=sample,
            rank=baseline_rank,
            query_view=baseline_query_view,
            top_k_values=top_k_values,
            search_depth=search_depth,
        )
        for config in candidate_configs:
            query_view = _build_query_view(config=config, sample=sample)
            challenger_ranked_doc_ids = index.rank_terms(
                query_terms=query_view.query_terms,
                search_depth=search_depth,
            )
            challenger_rank = _answer_rank(
                ranked_doc_ids=challenger_ranked_doc_ids,
                answer_doc_id=str(sample.answer_doc_id),
            )
            _record_rank(
                accumulator=candidate_accumulators[config.config_id],
                sample=sample,
                rank=challenger_rank,
                query_view=query_view,
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


def _build_query_view(
    *,
    config: StructuredQueryConfig,
    sample: PrimeQAHybridSplitSample,
) -> _QueryView:
    title_terms = tokenize_text(sample.question_title)
    body_terms = tokenize_text(sample.question_text)
    all_terms = title_terms + body_terms
    unique_all_terms = _unique_terms_in_order(all_terms)
    title_unique_terms = _unique_terms_in_order(title_terms)
    quoted_terms = _quoted_or_code_like_terms(
        f"{sample.question_title}\n{sample.question_text}"
    )
    buckets = {
        "error_code_or_log_identifier": [
            term for term in unique_all_terms if _is_error_or_log_identifier(term)
        ],
        "product_component_or_feature": [
            term for term in unique_all_terms if _is_product_component_or_feature(term)
        ],
        "version_or_platform": [
            term for term in unique_all_terms if _is_version_or_platform(term)
        ],
        "action_intent": [term for term in unique_all_terms if term in _ACTION_TERMS],
        "quoted_or_code_like_terms": [
            term
            for term in unique_all_terms
            if term in quoted_terms or _is_code_like(term)
        ],
        "title_guard_terms": (
            [
                term
                for term in title_unique_terms
                if _is_content_term(term) or term in _ACTION_TERMS
            ]
            if len(
                [term for term in title_unique_terms if _is_content_term(term)]
            )
            >= config.minimum_title_terms
            else []
        ),
        "deterministic_noun_phrase_like_terms": _noun_phrase_like_terms(
            all_terms,
            window_size=config.noun_phrase_window_size,
        ),
    }
    selected_terms: list[str] = []
    selected_seen: set[str] = set()
    selected_bucket_counts: Counter[str] = Counter()
    for bucket_name in config.preserved_feature_buckets:
        _append_bucket_terms(
            selected_terms=selected_terms,
            selected_seen=selected_seen,
            selected_bucket_counts=selected_bucket_counts,
            bucket_name=bucket_name,
            bucket_terms=buckets.get(bucket_name, []),
            maximum_unique_terms=config.maximum_unique_terms,
        )
    if len(selected_terms) < config.minimum_unique_terms:
        completion_terms = [
            term
            for term in unique_all_terms
            if _is_content_term(term) or term in _ACTION_TERMS
        ]
        _append_bucket_terms(
            selected_terms=selected_terms,
            selected_seen=selected_seen,
            selected_bucket_counts=selected_bucket_counts,
            bucket_name="minimum_unique_terms_completion",
            bucket_terms=completion_terms,
            maximum_unique_terms=config.maximum_unique_terms,
        )
    return _QueryView(
        query_terms=tuple(selected_terms),
        query_token_count=len(all_terms),
        query_unique_token_count=len(unique_all_terms),
        compacted_query_token_count=len(selected_terms),
        token_bucket_counts=dict(sorted(selected_bucket_counts.items())),
        minimum_unique_terms_satisfied=(
            len(selected_terms) >= config.minimum_unique_terms
            or len(selected_terms) == len(unique_all_terms)
        ),
        maximum_unique_terms_applied=len(selected_terms) >= config.maximum_unique_terms,
    )


def _append_bucket_terms(
    *,
    selected_terms: list[str],
    selected_seen: set[str],
    selected_bucket_counts: Counter[str],
    bucket_name: str,
    bucket_terms: Sequence[str],
    maximum_unique_terms: int,
) -> None:
    for term in bucket_terms:
        if len(selected_terms) >= maximum_unique_terms:
            return
        if term in selected_seen:
            continue
        selected_terms.append(term)
        selected_seen.add(term)
        selected_bucket_counts[bucket_name] += 1


def _answer_rank(
    *,
    ranked_doc_ids: Sequence[str],
    answer_doc_id: str,
) -> int | None:
    for rank, document_id in enumerate(ranked_doc_ids, start=1):
        if document_id == answer_doc_id:
            return rank
    return None


def _empty_accumulator(
    *,
    split: str,
    config_id: str,
    query_view_id: str,
    is_baseline: bool,
    total_questions: int,
) -> dict[str, Any]:
    return {
        "split": split,
        "config_id": config_id,
        "query_view_id": query_view_id,
        "is_baseline": is_baseline,
        "total_questions": total_questions,
        "evaluated_questions": 0,
        "hit_counts": {},
        "search_depth_hit_count": 0,
        "rank_11_to_50_count": 0,
        "reciprocal_rank_sum_at_10": 0.0,
        "reciprocal_rank_sum_at_search_depth": 0.0,
        "ranks_by_sample_id": {},
        "query_token_count_by_sample_id": {},
        "compacted_query_token_count_by_sample_id": {},
        "token_bucket_counts_by_sample_id": {},
        "empty_query_count": 0,
        "query_token_counts": [],
        "query_unique_token_counts": [],
        "compacted_query_token_counts": [],
        "minimum_unique_terms_not_satisfied_count": 0,
        "maximum_unique_terms_applied_count": 0,
        "token_bucket_totals": Counter(),
    }


def _record_rank(
    *,
    accumulator: dict[str, Any],
    sample: PrimeQAHybridSplitSample,
    rank: int | None,
    query_view: _QueryView,
    top_k_values: tuple[int, ...],
    search_depth: int,
) -> None:
    accumulator["evaluated_questions"] += 1
    accumulator["ranks_by_sample_id"][sample.sample_id] = rank
    accumulator["query_token_count_by_sample_id"][sample.sample_id] = (
        query_view.query_token_count
    )
    accumulator["compacted_query_token_count_by_sample_id"][sample.sample_id] = (
        query_view.compacted_query_token_count
    )
    accumulator["token_bucket_counts_by_sample_id"][sample.sample_id] = (
        query_view.token_bucket_counts
    )
    accumulator["empty_query_count"] += query_view.compacted_query_token_count == 0
    accumulator["query_token_counts"].append(query_view.query_token_count)
    accumulator["query_unique_token_counts"].append(query_view.query_unique_token_count)
    accumulator["compacted_query_token_counts"].append(
        query_view.compacted_query_token_count
    )
    accumulator["minimum_unique_terms_not_satisfied_count"] += (
        not query_view.minimum_unique_terms_satisfied
    )
    accumulator["maximum_unique_terms_applied_count"] += (
        query_view.maximum_unique_terms_applied
    )
    accumulator["token_bucket_totals"].update(query_view.token_bucket_counts)
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
    avg_query_tokens = _rounded_mean(accumulator["query_token_counts"])
    avg_compacted_tokens = _rounded_mean(accumulator["compacted_query_token_counts"])
    return {
        "split": accumulator["split"],
        "config_id": accumulator["config_id"],
        "query_view_id": accumulator["query_view_id"],
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
        "average_query_token_count": avg_query_tokens,
        "average_query_unique_token_count": _rounded_mean(
            accumulator["query_unique_token_counts"]
        ),
        "average_compacted_query_token_count": avg_compacted_tokens,
        "average_compaction_ratio": _rounded_ratio_float(
            avg_compacted_tokens,
            avg_query_tokens,
        ),
        "minimum_unique_terms_not_satisfied_count": int(
            accumulator["minimum_unique_terms_not_satisfied_count"]
        ),
        "maximum_unique_terms_applied_count": int(
            accumulator["maximum_unique_terms_applied_count"]
        ),
        "token_bucket_totals": dict(sorted(accumulator["token_bucket_totals"].items())),
        "ranks_by_sample_id": accumulator["ranks_by_sample_id"],
        "query_token_count_by_sample_id": accumulator[
            "query_token_count_by_sample_id"
        ],
        "compacted_query_token_count_by_sample_id": accumulator[
            "compacted_query_token_count_by_sample_id"
        ],
        "token_bucket_counts_by_sample_id": accumulator[
            "token_bucket_counts_by_sample_id"
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
    query_token_counts = challenger["query_token_count_by_sample_id"]
    compacted_query_token_counts = challenger[
        "compacted_query_token_count_by_sample_id"
    ]
    token_bucket_counts = challenger["token_bucket_counts_by_sample_id"]
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
            query_view_id=str(challenger["query_view_id"]),
            query_token_count=query_token_counts.get(sample_id),
            compacted_query_token_count=compacted_query_token_counts.get(sample_id),
            token_bucket_counts=token_bucket_counts.get(sample_id) or {},
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
    metric_deltas["average_compacted_query_token_count_delta"] = round(
        float(challenger["average_compacted_query_token_count"])
        - float(baseline["average_compacted_query_token_count"]),
        4,
    )
    return {
        "baseline_config_id": baseline["config_id"],
        "challenger_config_id": challenger["config_id"],
        "query_view_id": challenger["query_view_id"],
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
    candidate_configs: Sequence[StructuredQueryConfig],
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
            float(train_results[config_id]["average_compacted_query_token_count"]),
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
    stage88_report: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_configs: Sequence[StructuredQueryConfig],
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
    decision = stage88_report.get("decision") or {}
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
            name="source_stage88_report_is_stage88",
            passed=stage88_report.get("stage") == "Stage 88",
            observed=stage88_report.get("stage"),
            expected="Stage 88",
        ),
        _check(
            name="stage88_protocol_id_matches",
            passed=protocol.get("protocol_id") == _PROTOCOL_ID,
            observed=protocol.get("protocol_id"),
            expected=_PROTOCOL_ID,
        ),
        _check(
            name="stage88_candidate_id_matches",
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
            name="stage88_allows_train_dev_metrics_after_confirmation",
            passed=_stage88_allows_metric_run(stage88_report),
            observed=decision.get("can_run_train_dev_metrics_after_user_confirmation"),
            expected=True,
        ),
        _check(
            name="stage88_final_test_metrics_locked",
            passed=decision.get("can_run_final_test_metrics_now") is False,
            observed=decision.get("can_run_final_test_metrics_now"),
            expected=False,
        ),
        _check(
            name="stage88_forbids_test_tuning",
            passed=decision.get("can_use_test_for_tuning") is False,
            observed=decision.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage88_default_runtime_policy_unchanged",
            passed=decision.get("default_runtime_policy") == "unchanged",
            observed=decision.get("default_runtime_policy"),
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
            observed=protocol.get("public_safe_changed_case_fields") or [],
            expected=[
                "sample_id",
                "split",
                "baseline_rank",
                "challenger_rank",
                "config_id",
                "query_view_id",
                "query_token_count",
                "compacted_query_token_count",
                "token_bucket_counts",
            ],
        ),
        _check(
            name="raw_or_compacted_query_text_not_written",
            passed=True,
            observed="not_written",
            expected="not_written",
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
            "status": "primeqa_hybrid_structured_query_comparison_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        }
    selected_config_id = str(train_selection["selected_config_id"])
    selected_dev_comparison = comparisons[_DEV_SPLIT][selected_config_id]
    selected_dev_metrics = train_selection["selected_dev_metrics"]
    dev_hit10_delta = float(selected_dev_comparison["hit@10_delta"])
    dev_top10_improvements = int(selected_dev_comparison["top10_improvement_count"])
    dev_top10_regressions = int(selected_dev_comparison["top10_regression_count"])
    primary_contract_passed = dev_hit10_delta > 0
    secondary_contract_passed = dev_top10_regressions < dev_top10_improvements
    if primary_contract_passed and secondary_contract_passed:
        recommended_next_stage = (
            "Stage 90: review structured query changed cases and runtime risk "
            "before any runtime experiment; keep test locked."
        )
    else:
        recommended_next_stage = (
            "Stage 90: stop structured query keyphrase compaction as a "
            "retrieval-recall route unless a new train/dev-only protocol is "
            "explicitly confirmed; keep test locked and move to the next "
            "confirmed second-wave candidate."
        )
    return {
        "status": "primeqa_hybrid_structured_query_comparison_completed",
        "selected_config_id": selected_config_id,
        "selected_query_view_id": selected_dev_comparison["query_view_id"],
        "selected_dev_hit10_delta": dev_hit10_delta,
        "selected_dev_top10_improvements": dev_top10_improvements,
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
        "selected_dev_average_compacted_query_token_count": float(
            selected_dev_metrics["average_compacted_query_token_count"]
        ),
        "primary_contract_passed": primary_contract_passed,
        "secondary_contract_passed": secondary_contract_passed,
        "can_continue_train_dev_development": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": recommended_next_stage,
    }


def _public_config_metrics(config_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query_view_id": str(config_result["query_view_id"]),
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
        "average_compacted_query_token_count": float(
            config_result["average_compacted_query_token_count"]
        ),
        "average_compaction_ratio": float(config_result["average_compaction_ratio"]),
        "minimum_unique_terms_not_satisfied_count": int(
            config_result["minimum_unique_terms_not_satisfied_count"]
        ),
        "maximum_unique_terms_applied_count": int(
            config_result["maximum_unique_terms_applied_count"]
        ),
        "token_bucket_totals": config_result["token_bucket_totals"],
    }


def _public_candidate_config(config: StructuredQueryConfig) -> dict[str, Any]:
    return {
        "config_id": config.config_id,
        "query_view_id": config.query_view_id,
        "preserved_feature_buckets": list(config.preserved_feature_buckets),
        "maximum_unique_terms": config.maximum_unique_terms,
        "minimum_unique_terms": config.minimum_unique_terms,
        "minimum_title_terms": config.minimum_title_terms,
        "noun_phrase_window_size": config.noun_phrase_window_size,
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


def _average_compacted_terms_bars(
    report: Mapping[str, Any],
    *,
    split: str,
) -> list[BarDatum]:
    metrics = report.get("metrics_by_split", {}).get(split, {})
    ordered = sorted(
        metrics.items(),
        key=lambda item: (float(item[1]["average_compacted_query_token_count"]), item[0]),
    )
    return [
        BarDatum(
            label=config_id,
            value=float(config_metrics["average_compacted_query_token_count"]),
            value_label=f"{config_metrics['average_compacted_query_token_count']:.2f}",
        )
        for config_id, config_metrics in ordered
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


def _stage88_allows_metric_run(stage88_report: Mapping[str, Any]) -> bool:
    decision = stage88_report.get("decision") or {}
    return (
        decision.get("can_run_train_dev_metrics_after_user_confirmation") is True
        and decision.get("can_run_final_test_metrics_now") is False
        and decision.get("can_use_test_for_tuning") is False
        and decision.get("default_runtime_policy") == "unchanged"
    )


def _candidate_config_grid_matches_protocol(
    candidate_configs: Sequence[StructuredQueryConfig],
) -> bool:
    if len(candidate_configs) != len(_EXPECTED_CONFIG_IDS):
        return False
    observed_ids = tuple(config.config_id for config in candidate_configs)
    observed_query_view_ids = tuple(config.query_view_id for config in candidate_configs)
    if observed_ids != _EXPECTED_CONFIG_IDS:
        return False
    if observed_query_view_ids != _EXPECTED_QUERY_VIEW_IDS:
        return False
    return all(
        config.maximum_unique_terms > 0 and config.minimum_unique_terms > 0
        for config in candidate_configs
    )


def _changed_case_fields_public_safe(protocol: Mapping[str, Any]) -> bool:
    observed = protocol.get("public_safe_changed_case_fields") or []
    expected = [
        "sample_id",
        "split",
        "baseline_rank",
        "challenger_rank",
        "config_id",
        "query_view_id",
        "query_token_count",
        "compacted_query_token_count",
        "token_bucket_counts",
    ]
    return observed == expected


def _selection_rule_description() -> str:
    return (
        "Select the structured query candidate config on train only by hit@10, "
        "then hit@5, then hit@1, then MRR@10, then fewer top10 regressions, "
        "then fewer rank-down cases within top10, then lower average compacted "
        "query token count, then config_id; dev is validation only."
    )


def _change_case(
    *,
    split: str,
    sample_id: str,
    baseline_rank: int | None,
    challenger_rank: int | None,
    config_id: str,
    query_view_id: str,
    query_token_count: int | None,
    compacted_query_token_count: int | None,
    token_bucket_counts: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "split": split,
        "baseline_rank": baseline_rank,
        "challenger_rank": challenger_rank,
        "config_id": config_id,
        "query_view_id": query_view_id,
        "query_token_count": query_token_count,
        "compacted_query_token_count": compacted_query_token_count,
        "token_bucket_counts": dict(sorted(token_bucket_counts.items())),
    }


def _quoted_or_code_like_terms(raw_query: str) -> set[str]:
    quoted_spans = re.findall(r"`([^`]+)`|\"([^\"]+)\"|'([^']+)'", raw_query)
    quoted_terms: set[str] = set()
    for groups in quoted_spans:
        for group in groups:
            if group:
                quoted_terms.update(tokenize_text(group))
    return quoted_terms


def _noun_phrase_like_terms(tokens: Sequence[str], *, window_size: int) -> list[str]:
    if window_size <= 0:
        return []
    selected: list[str] = []
    seen: set[str] = set()
    for index in range(0, max(0, len(tokens) - window_size + 1)):
        window = tokens[index : index + window_size]
        if all(_is_content_term(term) for term in window):
            for term in window:
                if term not in seen:
                    selected.append(term)
                    seen.add(term)
    return selected


def _is_error_or_log_identifier(term: str) -> bool:
    if term in _ERROR_TERMS:
        return True
    if term.startswith(("err", "error", "warn")):
        return True
    if term.startswith("0x"):
        return True
    has_alpha = any(char.isalpha() for char in term)
    has_digit = any(char.isdigit() for char in term)
    return has_alpha and has_digit and len(term) >= 3


def _is_version_or_platform(term: str) -> bool:
    if term in _PLATFORM_TERMS:
        return True
    if re.fullmatch(r"v?\d+(?:[._-]\d+)+", term):
        return True
    return bool(re.fullmatch(r"\d+(?:\.\d+){1,3}", term))


def _is_code_like(term: str) -> bool:
    return any(char in term for char in ("_", "+", "#", ".", "-"))


def _is_product_component_or_feature(term: str) -> bool:
    return (
        _is_content_term(term)
        and term not in _ACTION_TERMS
        and not _is_error_or_log_identifier(term)
        and not _is_version_or_platform(term)
    )


def _is_content_term(term: str) -> bool:
    return len(term) >= 3 and term not in _STOPWORDS


def _unique_terms_in_order(terms: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique_terms = []
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        unique_terms.append(term)
    return unique_terms


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


def _rounded_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _rounded_ratio_float(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _rounded_mean(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


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


def _validate_options(*, top_k_values: tuple[int, ...], search_depth: int) -> None:
    if not top_k_values:
        raise ValueError("top_k_values must not be empty")
    if any(top_k <= 0 for top_k in top_k_values):
        raise ValueError("top_k_values must all be positive")
    if _PRIMARY_TOP_K not in top_k_values:
        raise ValueError(f"top_k_values must include {_PRIMARY_TOP_K}")
    if search_depth < max(top_k_values):
        raise ValueError("search_depth must be >= max(top_k_values)")
