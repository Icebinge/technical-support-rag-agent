from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever, tokenize_text
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)
from ts_rag_agent.infrastructure.section_bm25_retriever import SectionBM25Retriever

_STAGE = "Stage 79"
_CREATED_AT = "2026-07-15"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})
_BASELINE_CONFIG_ID = "full_document_bm25_baseline"
_SECTION_CONFIG_ID = "section_bm25_max_section_rollup"
_PRIMARY_TOP_K = 10
_DEFAULT_SEARCH_DEPTH = 50


@dataclass(frozen=True)
class PrimeQAHybridSectionBM25DocRollupVisualization:
    """One generated Stage79 section BM25 doc-rollup visualization."""

    name: str
    path: str


def run_primeqa_hybrid_section_bm25_doc_rollup(
    *,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    stage75_report_path: Path,
    stage78_report_path: Path,
    top_k_values: tuple[int, ...] = (1, 5, 10),
    search_depth: int = _DEFAULT_SEARCH_DEPTH,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> dict[str, Any]:
    """Run train/dev-only section BM25 document rollup probe."""

    _validate_options(
        top_k_values=top_k_values,
        search_depth=search_depth,
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
    sections_by_document = load_primeqa_document_sections(documents_path)
    stage75_report = _load_json_object(stage75_report_path)
    stage78_report = _load_json_object(stage78_report_path)
    loaded_inputs_at = time.perf_counter()

    baseline_retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
    section_retriever = SectionBM25Retriever(k1=bm25_k1, b=bm25_b)
    baseline_retriever.fit(documents.values())
    section_retriever.fit(documents.values(), sections_by_document)
    indexed_at = time.perf_counter()

    rank_tables = {
        split: _evaluate_split(
            split=split,
            samples=samples,
            baseline_retriever=baseline_retriever,
            section_retriever=section_retriever,
            top_k_values=top_k_values,
            search_depth=search_depth,
        )
        for split, samples in split_samples.items()
    }
    evaluated_at = time.perf_counter()
    comparisons = {
        split: _compare_to_baseline(
            baseline=rank_tables[split][_BASELINE_CONFIG_ID],
            challenger=rank_tables[split][_SECTION_CONFIG_ID],
            max_k=_PRIMARY_TOP_K,
            search_depth=search_depth,
        )
        for split in _ALLOWED_DEVELOPMENT_SPLITS
    }
    section_summary = _section_summary(sections_by_document)
    guard_checks = _guard_checks(
        split_samples=split_samples,
        rank_tables=rank_tables,
        section_summary=section_summary,
        stage75_report=stage75_report,
        stage78_report=stage78_report,
        top_k_values=top_k_values,
        search_depth=search_depth,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "Train/dev-only section BM25 document-rollup probe for "
            "section_bm25_doc_rollup_train_dev_probe. This stage evaluates the "
            "existing max-section-score parent-document rollup against the "
            "full-document BM25 baseline on the frozen Stage68 train/dev splits, "
            "keeps the frozen test split locked, does not run final metrics, "
            "does not use source DOC_IDS as runtime retrieval evidence, and does "
            "not change runtime defaults."
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
            "stage78_report": _fingerprint(stage78_report_path),
        },
        "config": {
            "top_k_values": list(top_k_values),
            "primary_top_k": _PRIMARY_TOP_K,
            "search_depth": search_depth,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "baseline_config_id": _BASELINE_CONFIG_ID,
            "section_config_id": _SECTION_CONFIG_ID,
            "section_rollup": "max_section_score_per_parent_document",
            "selection_rule": (
                "No learned selection: Stage76 predeclared one section BM25 "
                "candidate. Train and dev are reported separately; dev validates "
                "whether the fixed candidate improves hit@10."
            ),
        },
        "loaded_data_summary": {
            "document_count": len(documents),
            **section_summary,
            "split_rows": {
                split: len(samples) for split, samples in sorted(split_samples.items())
            },
            "answerable_rows": {
                split: sum(sample.answerable for sample in samples)
                for split, samples in sorted(split_samples.items())
            },
        },
        "metrics_by_split": {
            split: {
                config_id: _public_config_metrics(config_result)
                for config_id, config_result in split_results.items()
            }
            for split, split_results in rank_tables.items()
        },
        "comparisons_to_baseline": comparisons,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks=guard_checks, comparisons=comparisons),
        "timing_seconds": {
            "load_splits": round(loaded_splits_at - started_at, 3),
            "load_documents_sections_and_reports": round(
                loaded_inputs_at - loaded_splits_at,
                3,
            ),
            "bm25_indexes": round(indexed_at - loaded_inputs_at, 3),
            "section_rollup_evaluate": round(evaluated_at - indexed_at, 3),
            "guard_checks": round(checked_at - evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_section_bm25_doc_rollup_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridSectionBM25DocRollupVisualization]:
    """Write SVG charts for Stage79 section BM25 doc-rollup probe."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage79_section_bm25_train_hit_at_10.svg": render_horizontal_bar_chart_svg(
            title="Stage79 train hit@10 by retrieval config",
            bars=_hit_at_k_bars(report, split=_TRAIN_SPLIT, top_k=_PRIMARY_TOP_K),
            x_label="hit@10",
            width=1120,
            margin_left=390,
        ),
        "stage79_section_bm25_dev_hit_at_10.svg": render_horizontal_bar_chart_svg(
            title="Stage79 dev hit@10 by retrieval config",
            bars=_hit_at_k_bars(report, split=_DEV_SPLIT, top_k=_PRIMARY_TOP_K),
            x_label="hit@10",
            width=1120,
            margin_left=390,
        ),
        "stage79_section_bm25_dev_delta_hit_at_10.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage79 dev hit@10 delta vs baseline",
                bars=_delta_bars(report, split=_DEV_SPLIT, metric="hit@10_delta"),
                x_label="delta hit@10",
                width=1120,
                margin_left=390,
            )
        ),
        "stage79_section_bm25_dev_not_found_at_50.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage79 dev not found within top50",
                bars=_not_found_bars(report, split=_DEV_SPLIT),
                x_label="answer docs not found",
                width=1120,
                margin_left=390,
            )
        ),
        "stage79_section_bm25_dev_top10_changes.svg": render_horizontal_bar_chart_svg(
            title="Stage79 dev top10 improvements minus regressions",
            bars=_net_change_bars(report, split=_DEV_SPLIT),
            x_label="net changed cases",
            width=1120,
            margin_left=390,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridSectionBM25DocRollupVisualization(
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
    section_retriever: SectionBM25Retriever,
    top_k_values: tuple[int, ...],
    search_depth: int,
) -> dict[str, Any]:
    answerable_samples = [
        sample
        for sample in samples
        if sample.answerable and sample.answer_doc_id is not None
    ]
    accumulators = {
        _BASELINE_CONFIG_ID: _empty_accumulator(
            split=split,
            config_id=_BASELINE_CONFIG_ID,
            total_questions=len(samples),
        ),
        _SECTION_CONFIG_ID: _empty_accumulator(
            split=split,
            config_id=_SECTION_CONFIG_ID,
            total_questions=len(samples),
        ),
    }
    for sample in answerable_samples:
        query = sample.to_primeqa_question().full_question
        query_token_count = len(tokenize_text(query))
        baseline_results = baseline_retriever.search(query, top_k=search_depth)
        section_results = section_retriever.search(query, top_k=search_depth)
        _record_result(
            accumulator=accumulators[_BASELINE_CONFIG_ID],
            sample=sample,
            results=baseline_results,
            query_token_count=query_token_count,
            top_k_values=top_k_values,
            search_depth=search_depth,
        )
        _record_result(
            accumulator=accumulators[_SECTION_CONFIG_ID],
            sample=sample,
            results=section_results,
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


def _guard_checks(
    *,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    rank_tables: Mapping[str, Mapping[str, Mapping[str, Any]]],
    section_summary: Mapping[str, Any],
    stage75_report: Mapping[str, Any],
    stage78_report: Mapping[str, Any],
    top_k_values: tuple[int, ...],
    search_depth: int,
) -> list[dict[str, Any]]:
    observed_split_names = sorted(
        {sample.assigned_split for samples in split_samples.values() for sample in samples}
    )
    expected_splits = sorted(_ALLOWED_DEVELOPMENT_SPLITS)
    stage75_split_reports = stage75_report.get("split_reports") or {}
    baseline_train_hit10 = rank_tables[_TRAIN_SPLIT][_BASELINE_CONFIG_ID]["hit_at_k"][
        _PRIMARY_TOP_K
    ]
    baseline_dev_hit10 = rank_tables[_DEV_SPLIT][_BASELINE_CONFIG_ID]["hit_at_k"][
        _PRIMARY_TOP_K
    ]
    stage75_train_hit10 = (
        (stage75_split_reports.get(_TRAIN_SPLIT) or {}).get("hit_at_top_k")
    )
    stage75_dev_hit10 = (
        (stage75_split_reports.get(_DEV_SPLIT) or {}).get("hit_at_top_k")
    )
    stage78_decision = stage78_report.get("decision") or {}
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
            name="stage78_source_report_is_stage78",
            passed=str(stage78_report.get("stage") or "") == "Stage 78",
            observed=str(stage78_report.get("stage") or ""),
            expected="Stage 78",
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
            name="stage78_did_not_open_final_test_gate",
            passed=stage78_decision.get("can_open_final_test_gate_now") is False,
            observed=stage78_decision.get("can_open_final_test_gate_now"),
            expected=False,
        ),
        _check(
            name="section_index_has_nonempty_sections",
            passed=int(section_summary["section_count"]) > 0,
            observed=int(section_summary["section_count"]),
            expected="> 0",
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
    comparisons: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    failed_checks = [check["name"] for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_section_bm25_doc_rollup_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        }
    selected_dev_comparison = comparisons[_DEV_SPLIT]
    dev_hit10_delta = float(selected_dev_comparison["hit@10_delta"])
    if dev_hit10_delta > 0:
        recommended_next_stage = (
            "Stage 80: review section BM25 changed cases on dev and decide whether "
            "a guarded runtime experiment is justified; keep test locked."
        )
    else:
        recommended_next_stage = (
            "Stage 80: check dense_sparse_rrf_train_dev_probe feasibility with "
            "local model/cache identity before any train/dev run; keep test locked "
            "and do not download or choose external models silently."
        )
    return {
        "status": "primeqa_hybrid_section_bm25_doc_rollup_completed",
        "candidate_config_id": _SECTION_CONFIG_ID,
        "candidate_dev_hit10_delta": dev_hit10_delta,
        "candidate_dev_top10_improvements": int(
            selected_dev_comparison["top10_improvement_count"]
        ),
        "candidate_dev_top10_regressions": int(
            selected_dev_comparison["top10_regression_count"]
        ),
        "candidate_dev_not_found_at_search_depth_delta": int(
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
        "hit_at_k": {
            f"hit@{top_k}": value
            for top_k, value in sorted(config_result["hit_at_k"].items())
        },
        "mrr_at_10": float(config_result["mrr_at_10"]),
        "mrr_at_search_depth": float(config_result["mrr_at_search_depth"]),
        "miss_count_at_10": int(config_result["miss_count_at_primary_top_k"]),
        "miss_rate_at_10": float(config_result["miss_rate_at_primary_top_k"]),
        "search_depth": int(config_result["search_depth"]),
        "not_found_count_at_search_depth": int(
            config_result["not_found_count_at_search_depth"]
        ),
        "not_found_rate_at_search_depth": float(
            config_result["not_found_rate_at_search_depth"]
        ),
        "empty_query_count": int(config_result["empty_query_count"]),
        "average_query_token_count": float(config_result["average_query_token_count"]),
    }


def _hit_at_k_bars(
    report: Mapping[str, Any],
    *,
    split: str,
    top_k: int,
) -> list[BarDatum]:
    metrics = report["metrics_by_split"][split]
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
    comparison = report["comparisons_to_baseline"][split]
    return [
        BarDatum(
            label=comparison["challenger_config_id"],
            value=float(comparison[metric]),
            value_label=f"{comparison[metric]:+.4f}",
        )
    ]


def _not_found_bars(report: Mapping[str, Any], *, split: str) -> list[BarDatum]:
    metrics = report["metrics_by_split"][split]
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
    comparison = report["comparisons_to_baseline"][split]
    return [
        BarDatum(
            label=comparison["challenger_config_id"],
            value=float(comparison["top10_net_improvement_count"]),
            value_label=str(comparison["top10_net_improvement_count"]),
        )
    ]


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
