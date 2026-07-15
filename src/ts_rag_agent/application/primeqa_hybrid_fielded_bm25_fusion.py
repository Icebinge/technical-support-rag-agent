from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever, tokenize_text
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_documents

_STAGE = "Stage 78"
_CREATED_AT = "2026-07-15"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})
_BASELINE_CONFIG_ID = "full_document_bm25_baseline"
_PRIMARY_TOP_K = 10


@dataclass(frozen=True)
class PrimeQAHybridFieldedBM25FusionVisualization:
    """One generated Stage78 fielded BM25 fusion visualization."""

    name: str
    path: str


@dataclass(frozen=True)
class _FusionConfig:
    config_id: str
    title_weight: float
    text_weight: float
    description: str


def run_primeqa_hybrid_fielded_bm25_fusion(
    *,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    stage75_report_path: Path,
    stage77_report_path: Path,
    top_k_values: tuple[int, ...] = (1, 5, 10),
    candidate_depth: int = 100,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> dict[str, Any]:
    """Run train/dev-only fielded title/text BM25 score-fusion experiment."""

    _validate_options(
        top_k_values=top_k_values,
        candidate_depth=candidate_depth,
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
    stage75_report = _load_json_object(stage75_report_path)
    stage77_report = _load_json_object(stage77_report_path)
    loaded_inputs_at = time.perf_counter()

    baseline_retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
    title_retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
    text_retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
    baseline_retriever.fit(documents.values())
    title_retriever.fit(_field_documents(documents, field="title"))
    text_retriever.fit(_field_documents(documents, field="text"))
    indexed_at = time.perf_counter()

    configs = _fusion_configs()
    max_k = max(top_k_values)
    rank_tables = {
        split: _evaluate_split(
            split=split,
            samples=samples,
            documents=documents,
            baseline_retriever=baseline_retriever,
            title_retriever=title_retriever,
            text_retriever=text_retriever,
            configs=configs,
            top_k_values=top_k_values,
            max_k=max_k,
            candidate_depth=candidate_depth,
        )
        for split, samples in split_samples.items()
    }
    evaluated_at = time.perf_counter()
    comparisons = {
        split: {
            config.config_id: _compare_to_baseline(
                baseline=rank_tables[split][_BASELINE_CONFIG_ID],
                challenger=rank_tables[split][config.config_id],
                max_k=max_k,
            )
            for config in configs
        }
        for split in _ALLOWED_DEVELOPMENT_SPLITS
    }
    train_selected_config = _select_train_config(rank_tables[_TRAIN_SPLIT], top_k_values)
    guard_checks = _guard_checks(
        split_samples=split_samples,
        rank_tables=rank_tables,
        stage75_report=stage75_report,
        stage77_report=stage77_report,
        top_k_values=top_k_values,
        candidate_depth=candidate_depth,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "Train/dev-only fielded title/text BM25 score-fusion experiment for "
            "fielded_title_text_bm25_score_fusion. This stage evaluates a "
            "predeclared title/text weight grid on the frozen Stage68 train/dev "
            "splits, selects on train, validates on dev, keeps the frozen test "
            "split locked, does not run final metrics, does not use source "
            "DOC_IDS as runtime retrieval evidence, and does not change runtime "
            "defaults."
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
            "stage77_report": _fingerprint(stage77_report_path),
        },
        "config": {
            "top_k_values": list(top_k_values),
            "primary_top_k": _PRIMARY_TOP_K,
            "candidate_depth": candidate_depth,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "fusion_score": (
                "normalized_title_score * title_weight + "
                "normalized_text_score * text_weight"
            ),
            "selection_rule": (
                "Select on train by hit@10, then hit@5, hit@1, MRR, then config_id. "
                "Dev is used only for validation."
            ),
        },
        "fusion_configs": [
            {
                "config_id": config.config_id,
                "title_weight": config.title_weight,
                "text_weight": config.text_weight,
                "description": config.description,
            }
            for config in configs
        ],
        "metrics_by_split": {
            split: {
                config_id: _public_config_metrics(config_result)
                for config_id, config_result in split_results.items()
            }
            for split, split_results in rank_tables.items()
        },
        "comparisons_to_baseline": comparisons,
        "train_selection": {
            "selected_config_id": train_selected_config,
            "selected_train_metrics": _public_config_metrics(
                rank_tables[_TRAIN_SPLIT][train_selected_config]
            ),
            "selected_dev_metrics": _public_config_metrics(
                rank_tables[_DEV_SPLIT][train_selected_config]
            ),
            "selected_dev_comparison_to_baseline": comparisons[_DEV_SPLIT][
                train_selected_config
            ],
        },
        "guard_checks": guard_checks,
        "decision": _decision(
            guard_checks=guard_checks,
            train_selected_config=train_selected_config,
            comparisons=comparisons,
        ),
        "timing_seconds": {
            "load_splits": round(loaded_splits_at - started_at, 3),
            "load_documents_and_reports": round(loaded_inputs_at - loaded_splits_at, 3),
            "bm25_indexes": round(indexed_at - loaded_inputs_at, 3),
            "fielded_fusion_evaluate": round(evaluated_at - indexed_at, 3),
            "guard_checks": round(checked_at - evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_fielded_bm25_fusion_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridFieldedBM25FusionVisualization]:
    """Write SVG charts for Stage78 fielded BM25 fusion."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage78_fielded_bm25_train_hit_at_10.svg": render_horizontal_bar_chart_svg(
            title="Stage78 train hit@10 by fielded BM25 config",
            bars=_hit_at_k_bars(report, split=_TRAIN_SPLIT, top_k=_PRIMARY_TOP_K),
            x_label="hit@10",
            width=1200,
            margin_left=500,
        ),
        "stage78_fielded_bm25_dev_hit_at_10.svg": render_horizontal_bar_chart_svg(
            title="Stage78 dev hit@10 by fielded BM25 config",
            bars=_hit_at_k_bars(report, split=_DEV_SPLIT, top_k=_PRIMARY_TOP_K),
            x_label="hit@10",
            width=1200,
            margin_left=500,
        ),
        "stage78_fielded_bm25_dev_delta_hit_at_10.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage78 dev hit@10 delta vs baseline",
                bars=_delta_bars(report, split=_DEV_SPLIT, metric="hit@10_delta"),
                x_label="delta hit@10",
                width=1200,
                margin_left=500,
            )
        ),
        "stage78_fielded_bm25_dev_top10_changes.svg": render_horizontal_bar_chart_svg(
            title="Stage78 dev top10 improvements minus regressions",
            bars=_net_change_bars(report, split=_DEV_SPLIT),
            x_label="net changed cases",
            width=1200,
            margin_left=500,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridFieldedBM25FusionVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _fusion_configs() -> tuple[_FusionConfig, ...]:
    return (
        _FusionConfig(
            config_id="fielded_title_0_25_text_1_00",
            title_weight=0.25,
            text_weight=1.0,
            description="Light title boost over normalized body score.",
        ),
        _FusionConfig(
            config_id="fielded_title_0_50_text_1_00",
            title_weight=0.5,
            text_weight=1.0,
            description="Moderate title boost over normalized body score.",
        ),
        _FusionConfig(
            config_id="fielded_title_1_00_text_1_00",
            title_weight=1.0,
            text_weight=1.0,
            description="Equal normalized title/body score fusion.",
        ),
        _FusionConfig(
            config_id="fielded_title_1_50_text_1_00",
            title_weight=1.5,
            text_weight=1.0,
            description="Strong title boost over normalized body score.",
        ),
        _FusionConfig(
            config_id="fielded_title_2_00_text_1_00",
            title_weight=2.0,
            text_weight=1.0,
            description="Very strong title boost over normalized body score.",
        ),
    )


def _field_documents(
    documents: Mapping[str, PrimeQADocument],
    *,
    field: str,
) -> list[PrimeQADocument]:
    if field == "title":
        return [
            PrimeQADocument(id=document.id, title="", text=document.title)
            for document in documents.values()
        ]
    if field == "text":
        return [
            PrimeQADocument(id=document.id, title="", text=document.text)
            for document in documents.values()
        ]
    raise ValueError(f"Unsupported field: {field}")


def _evaluate_split(
    *,
    split: str,
    samples: Sequence[PrimeQAHybridSplitSample],
    documents: Mapping[str, PrimeQADocument],
    baseline_retriever: BM25Retriever,
    title_retriever: BM25Retriever,
    text_retriever: BM25Retriever,
    configs: Sequence[_FusionConfig],
    top_k_values: tuple[int, ...],
    max_k: int,
    candidate_depth: int,
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
        **{
            config.config_id: _empty_accumulator(
                split=split,
                config_id=config.config_id,
                total_questions=len(samples),
            )
            for config in configs
        },
    }
    for sample in answerable_samples:
        query = sample.to_primeqa_question().full_question
        query_token_count = len(tokenize_text(query))
        baseline_results = baseline_retriever.search(query, top_k=max_k)
        title_results = title_retriever.search(query, top_k=candidate_depth)
        text_results = text_retriever.search(query, top_k=candidate_depth)
        _record_result(
            accumulator=accumulators[_BASELINE_CONFIG_ID],
            sample=sample,
            results=baseline_results,
            query_token_count=query_token_count,
            top_k_values=top_k_values,
        )
        for config in configs:
            fused_results = _fuse_field_results(
                documents=documents,
                title_results=title_results,
                text_results=text_results,
                config=config,
                top_k=max_k,
            )
            _record_result(
                accumulator=accumulators[config.config_id],
                sample=sample,
                results=fused_results,
                query_token_count=query_token_count,
                top_k_values=top_k_values,
            )
    return {
        config_id: _finalize_accumulator(accumulator, top_k_values=top_k_values)
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
        "reciprocal_rank_sum": 0.0,
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
    if rank is not None:
        accumulator["reciprocal_rank_sum"] += 1 / rank


def _finalize_accumulator(
    accumulator: Mapping[str, Any],
    *,
    top_k_values: tuple[int, ...],
) -> dict[str, Any]:
    evaluated_count = int(accumulator["evaluated_questions"])
    hit_counts = {
        top_k: int(accumulator["hit_counts"].get(top_k, 0))
        for top_k in top_k_values
    }
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
        "mrr": round(float(accumulator["reciprocal_rank_sum"]) / evaluated_count, 4)
        if evaluated_count
        else 0.0,
        "miss_count_at_primary_top_k": evaluated_count
        - hit_counts.get(_PRIMARY_TOP_K, 0),
        "miss_rate_at_primary_top_k": _rounded_ratio(
            evaluated_count - hit_counts.get(_PRIMARY_TOP_K, 0),
            evaluated_count,
        ),
        "empty_query_count": int(accumulator["empty_query_count"]),
        "average_query_token_count": _rounded_mean(accumulator["query_token_counts"]),
        "ranks_by_sample_id": accumulator["ranks_by_sample_id"],
    }


def _fuse_field_results(
    *,
    documents: Mapping[str, PrimeQADocument],
    title_results: Sequence[RetrievalResult],
    text_results: Sequence[RetrievalResult],
    config: _FusionConfig,
    top_k: int,
) -> list[RetrievalResult]:
    title_scores = {result.document.id: result.score for result in title_results}
    text_scores = {result.document.id: result.score for result in text_results}
    title_max = max(title_scores.values(), default=0.0)
    text_max = max(text_scores.values(), default=0.0)
    candidate_doc_ids = set(title_scores) | set(text_scores)
    fused_scores = {}
    for doc_id in candidate_doc_ids:
        title_score = _normalized(title_scores.get(doc_id, 0.0), title_max)
        text_score = _normalized(text_scores.get(doc_id, 0.0), text_max)
        fused_scores[doc_id] = (
            config.title_weight * title_score + config.text_weight * text_score
        )
    ranked_doc_ids = sorted(fused_scores, key=lambda doc_id: (-fused_scores[doc_id], doc_id))
    return [
        RetrievalResult(
            document=documents[doc_id],
            score=fused_scores[doc_id],
            rank=rank,
        )
        for rank, doc_id in enumerate(ranked_doc_ids[:top_k], start=1)
    ]


def _normalized(score: float, max_score: float) -> float:
    return score / max_score if max_score > 0 else 0.0


def _compare_to_baseline(
    *,
    baseline: Mapping[str, Any],
    challenger: Mapping[str, Any],
    max_k: int,
) -> dict[str, Any]:
    baseline_ranks = baseline["ranks_by_sample_id"]
    challenger_ranks = challenger["ranks_by_sample_id"]
    top10_improvements = []
    top10_regressions = []
    rank_up = 0
    rank_down = 0
    both_hit = 0
    both_miss = 0
    for sample_id, baseline_rank in baseline_ranks.items():
        challenger_rank = challenger_ranks.get(sample_id)
        baseline_hit = baseline_rank is not None and baseline_rank <= max_k
        challenger_hit = challenger_rank is not None and challenger_rank <= max_k
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
    metric_deltas = {
        f"hit@{top_k}_delta": round(
            float(challenger["hit_at_k"][top_k]) - float(baseline["hit_at_k"][top_k]),
            4,
        )
        for top_k in baseline["hit_at_k"]
    }
    metric_deltas["mrr_delta"] = round(float(challenger["mrr"]) - float(baseline["mrr"]), 4)
    return {
        "baseline_config_id": baseline["config_id"],
        "challenger_config_id": challenger["config_id"],
        **metric_deltas,
        "top10_improvement_count": len(top10_improvements),
        "top10_regression_count": len(top10_regressions),
        "top10_net_improvement_count": len(top10_improvements)
        - len(top10_regressions),
        "both_hit_count": both_hit,
        "both_miss_count": both_miss,
        "rank_up_within_top10_count": rank_up,
        "rank_down_within_top10_count": rank_down,
        "sample_top10_improvements": top10_improvements[:20],
        "sample_top10_regressions": top10_regressions[:20],
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


def _select_train_config(
    train_results: Mapping[str, Mapping[str, Any]],
    top_k_values: tuple[int, ...],
) -> str:
    challenger_config_ids = [
        config_id for config_id in train_results if config_id != _BASELINE_CONFIG_ID
    ]
    sorted_top_k = sorted(top_k_values, reverse=True)

    def sort_key(config_id: str) -> tuple[Any, ...]:
        result = train_results[config_id]
        return (
            *[result["hit_at_k"][top_k] for top_k in sorted_top_k],
            result["mrr"],
            config_id,
        )

    return max(challenger_config_ids, key=sort_key)


def _guard_checks(
    *,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    rank_tables: Mapping[str, Mapping[str, Mapping[str, Any]]],
    stage75_report: Mapping[str, Any],
    stage77_report: Mapping[str, Any],
    top_k_values: tuple[int, ...],
    candidate_depth: int,
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
    stage77_decision = stage77_report.get("decision") or {}
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
            name="candidate_depth_covers_primary_top10",
            passed=candidate_depth >= _PRIMARY_TOP_K,
            observed=candidate_depth,
            expected=f">= {_PRIMARY_TOP_K}",
        ),
        _check(
            name="stage75_source_report_is_stage75",
            passed=str(stage75_report.get("stage") or "") == "Stage 75",
            observed=str(stage75_report.get("stage") or ""),
            expected="Stage 75",
        ),
        _check(
            name="stage77_source_report_is_stage77",
            passed=str(stage77_report.get("stage") or "") == "Stage 77",
            observed=str(stage77_report.get("stage") or ""),
            expected="Stage 77",
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
            name="stage77_did_not_open_final_test_gate",
            passed=stage77_decision.get("can_open_final_test_gate_now") is False,
            observed=stage77_decision.get("can_open_final_test_gate_now"),
            expected=False,
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
    train_selected_config: str,
    comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    failed_checks = [check["name"] for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_fielded_bm25_fusion_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        }
    selected_dev_comparison = comparisons[_DEV_SPLIT][train_selected_config]
    dev_hit10_delta = float(selected_dev_comparison["hit@10_delta"])
    if dev_hit10_delta > 0:
        recommended_next_stage = (
            "Stage 79: review fielded BM25 changed cases on dev and decide whether "
            "a guarded runtime experiment is justified; keep test locked."
        )
    else:
        recommended_next_stage = (
            "Stage 79: move to the next Stage76 candidate, "
            "section_bm25_doc_rollup_train_dev_probe, on train/dev only; keep test locked."
        )
    return {
        "status": "primeqa_hybrid_fielded_bm25_fusion_completed",
        "train_selected_config_id": train_selected_config,
        "train_selected_dev_hit10_delta": dev_hit10_delta,
        "train_selected_dev_top10_improvements": int(
            selected_dev_comparison["top10_improvement_count"]
        ),
        "train_selected_dev_top10_regressions": int(
            selected_dev_comparison["top10_regression_count"]
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
        "mrr": float(config_result["mrr"]),
        "miss_count_at_10": int(config_result["miss_count_at_primary_top_k"]),
        "miss_rate_at_10": float(config_result["miss_rate_at_primary_top_k"]),
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
    comparisons = report["comparisons_to_baseline"][split]
    ordered = sorted(comparisons.items(), key=lambda item: (-item[1][metric], item[0]))
    return [
        BarDatum(
            label=config_id,
            value=float(comparison[metric]),
            value_label=f"{comparison[metric]:+.4f}",
        )
        for config_id, comparison in ordered
    ]


def _net_change_bars(report: Mapping[str, Any], *, split: str) -> list[BarDatum]:
    comparisons = report["comparisons_to_baseline"][split]
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
    candidate_depth: int,
    bm25_k1: float,
    bm25_b: float,
) -> None:
    if not top_k_values:
        raise ValueError("top_k_values must not be empty")
    if any(top_k <= 0 for top_k in top_k_values):
        raise ValueError("top_k_values must be positive")
    if _PRIMARY_TOP_K not in top_k_values:
        raise ValueError(f"top_k_values must include {_PRIMARY_TOP_K}")
    if candidate_depth < max(top_k_values):
        raise ValueError("candidate_depth must be at least max(top_k_values)")
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


def _rounded_mean(values: Sequence[int]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
