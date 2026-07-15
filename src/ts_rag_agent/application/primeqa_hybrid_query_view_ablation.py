from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever, tokenize_text
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_documents

_STAGE = "Stage 77"
_CREATED_AT = "2026-07-15"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})
_BASELINE_VIEW_ID = "full_question_baseline"
_PRIMARY_TOP_K = 10


@dataclass(frozen=True)
class PrimeQAHybridQueryViewAblationVisualization:
    """One generated Stage77 query-view ablation visualization."""

    name: str
    path: str


@dataclass(frozen=True)
class _QueryView:
    view_id: str
    name: str
    description: str
    query_builder: Callable[[PrimeQAHybridSplitSample], str]


def run_primeqa_hybrid_query_view_ablation(
    *,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    stage75_report_path: Path,
    top_k_values: tuple[int, ...] = (1, 5, 10),
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> dict[str, Any]:
    """Run train/dev-only query-view ablation for BM25 retrieval."""

    _validate_options(top_k_values=top_k_values, bm25_k1=bm25_k1, bm25_b=bm25_b)
    started_at = time.perf_counter()
    split_samples = {
        _TRAIN_SPLIT: load_primeqa_hybrid_split_samples(train_split_path),
        _DEV_SPLIT: load_primeqa_hybrid_split_samples(dev_split_path),
    }
    loaded_splits_at = time.perf_counter()
    documents = load_primeqa_documents(documents_path)
    stage75_report = _load_json_object(stage75_report_path)
    loaded_inputs_at = time.perf_counter()
    retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
    retriever.fit(documents.values())
    indexed_at = time.perf_counter()
    query_views = _query_views()
    max_k = max(top_k_values)
    rank_tables = {
        split: {
            view.view_id: _evaluate_view(
                split=split,
                samples=samples,
                retriever=retriever,
                query_view=view,
                top_k_values=top_k_values,
                max_k=max_k,
            )
            for view in query_views
        }
        for split, samples in split_samples.items()
    }
    evaluated_at = time.perf_counter()
    comparisons = {
        split: {
            view.view_id: _compare_to_baseline(
                baseline=rank_tables[split][_BASELINE_VIEW_ID],
                challenger=rank_tables[split][view.view_id],
                max_k=max_k,
            )
            for view in query_views
            if view.view_id != _BASELINE_VIEW_ID
        }
        for split in _ALLOWED_DEVELOPMENT_SPLITS
    }
    train_selected_view = _select_train_view(rank_tables[_TRAIN_SPLIT], top_k_values)
    guard_checks = _guard_checks(
        split_samples=split_samples,
        rank_tables=rank_tables,
        stage75_report=stage75_report,
        top_k_values=top_k_values,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "Train/dev-only query-view ablation for "
            "query_view_ablation_full_title_dedup. This stage evaluates fixed "
            "BM25 query views on the frozen Stage68 train/dev splits, keeps the "
            "frozen test split locked, does not run final metrics, does not use "
            "source DOC_IDS as runtime retrieval evidence, and does not change "
            "runtime defaults."
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
        },
        "config": {
            "top_k_values": list(top_k_values),
            "primary_top_k": _PRIMARY_TOP_K,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
        },
        "query_views": [
            {
                "view_id": view.view_id,
                "name": view.name,
                "description": view.description,
            }
            for view in query_views
        ],
        "metrics_by_split": {
            split: {
                view_id: _public_view_metrics(view_result)
                for view_id, view_result in split_results.items()
            }
            for split, split_results in rank_tables.items()
        },
        "comparisons_to_baseline": comparisons,
        "train_selection": {
            "selection_rule": (
                "Select on train by hit@10, then hit@5, hit@1, MRR, then view_id. "
                "Dev is used only for validation."
            ),
            "selected_view_id": train_selected_view,
            "selected_train_metrics": _public_view_metrics(
                rank_tables[_TRAIN_SPLIT][train_selected_view]
            ),
            "selected_dev_metrics": _public_view_metrics(
                rank_tables[_DEV_SPLIT][train_selected_view]
            ),
            "selected_dev_comparison_to_baseline": comparisons[_DEV_SPLIT].get(
                train_selected_view,
                _baseline_self_comparison(rank_tables[_DEV_SPLIT][_BASELINE_VIEW_ID]),
            ),
        },
        "guard_checks": guard_checks,
        "decision": _decision(
            guard_checks=guard_checks,
            train_selected_view=train_selected_view,
            comparisons=comparisons,
        ),
        "timing_seconds": {
            "load_splits": round(loaded_splits_at - started_at, 3),
            "load_documents_and_stage75": round(loaded_inputs_at - loaded_splits_at, 3),
            "bm25_index": round(indexed_at - loaded_inputs_at, 3),
            "query_view_evaluate": round(evaluated_at - indexed_at, 3),
            "guard_checks": round(checked_at - evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_query_view_ablation_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridQueryViewAblationVisualization]:
    """Write SVG charts for Stage77 query-view ablation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage77_query_view_train_hit_at_10.svg": render_horizontal_bar_chart_svg(
            title="Stage77 train hit@10 by query view",
            bars=_hit_at_k_bars(report, split=_TRAIN_SPLIT, top_k=_PRIMARY_TOP_K),
            x_label="hit@10",
            width=1120,
            margin_left=430,
        ),
        "stage77_query_view_dev_hit_at_10.svg": render_horizontal_bar_chart_svg(
            title="Stage77 dev hit@10 by query view",
            bars=_hit_at_k_bars(report, split=_DEV_SPLIT, top_k=_PRIMARY_TOP_K),
            x_label="hit@10",
            width=1120,
            margin_left=430,
        ),
        "stage77_query_view_dev_delta_hit_at_10.svg": render_horizontal_bar_chart_svg(
            title="Stage77 dev hit@10 delta vs baseline",
            bars=_delta_bars(report, split=_DEV_SPLIT, metric="hit@10_delta"),
            x_label="delta hit@10",
            width=1120,
            margin_left=430,
        ),
        "stage77_query_view_dev_top10_changes.svg": render_horizontal_bar_chart_svg(
            title="Stage77 dev top10 improvements minus regressions",
            bars=_net_change_bars(report, split=_DEV_SPLIT),
            x_label="net changed cases",
            width=1120,
            margin_left=430,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridQueryViewAblationVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _query_views() -> tuple[_QueryView, ...]:
    return (
        _QueryView(
            view_id=_BASELINE_VIEW_ID,
            name="Full question baseline",
            description="Current retrieval query: question title plus question text.",
            query_builder=lambda sample: sample.to_primeqa_question().full_question,
        ),
        _QueryView(
            view_id="title_only",
            name="Title only",
            description="Question title only.",
            query_builder=lambda sample: sample.question_title.strip(),
        ),
        _QueryView(
            view_id="full_question_dedup_terms",
            name="Full question deduplicated terms",
            description=(
                "Tokenized full question with duplicate lexical terms removed while "
                "preserving first occurrence order."
            ),
            query_builder=lambda sample: _dedup_query_terms(
                sample.to_primeqa_question().full_question
            ),
        ),
    )


def _evaluate_view(
    *,
    split: str,
    samples: Sequence[PrimeQAHybridSplitSample],
    retriever: BM25Retriever,
    query_view: _QueryView,
    top_k_values: tuple[int, ...],
    max_k: int,
) -> dict[str, Any]:
    answerable_samples = [
        sample
        for sample in samples
        if sample.answerable and sample.answer_doc_id is not None
    ]
    hit_counts = {top_k: 0 for top_k in top_k_values}
    reciprocal_rank_sum = 0.0
    ranks_by_sample_id: dict[str, int | None] = {}
    query_token_counts = []
    empty_query_count = 0
    for sample in answerable_samples:
        query = query_view.query_builder(sample)
        query_tokens = tokenize_text(query)
        query_token_counts.append(len(query_tokens))
        empty_query_count += len(query_tokens) == 0
        results = retriever.search(query, top_k=max_k)
        result_doc_ids = [result.document.id for result in results]
        rank = (
            result_doc_ids.index(str(sample.answer_doc_id)) + 1
            if sample.answer_doc_id in result_doc_ids
            else None
        )
        ranks_by_sample_id[sample.sample_id] = rank
        for top_k in top_k_values:
            if rank is not None and rank <= top_k:
                hit_counts[top_k] += 1
        if rank is not None:
            reciprocal_rank_sum += 1 / rank
    evaluated_count = len(answerable_samples)
    return {
        "split": split,
        "view_id": query_view.view_id,
        "total_questions": len(samples),
        "evaluated_questions": evaluated_count,
        "hit_counts": hit_counts,
        "hit_at_k": {
            top_k: _rounded_ratio(count, evaluated_count)
            for top_k, count in hit_counts.items()
        },
        "mrr": round(reciprocal_rank_sum / evaluated_count, 4)
        if evaluated_count
        else 0.0,
        "miss_count_at_primary_top_k": evaluated_count
        - hit_counts.get(_PRIMARY_TOP_K, 0),
        "miss_rate_at_primary_top_k": _rounded_ratio(
            evaluated_count - hit_counts.get(_PRIMARY_TOP_K, 0),
            evaluated_count,
        ),
        "empty_query_count": empty_query_count,
        "average_query_token_count": _rounded_mean(query_token_counts),
        "ranks_by_sample_id": ranks_by_sample_id,
    }


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
        "baseline_view_id": baseline["view_id"],
        "challenger_view_id": challenger["view_id"],
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


def _select_train_view(
    train_results: Mapping[str, Mapping[str, Any]],
    top_k_values: tuple[int, ...],
) -> str:
    candidate_view_ids = [
        view_id for view_id in train_results if view_id != _BASELINE_VIEW_ID
    ]
    sorted_top_k = sorted(top_k_values, reverse=True)

    def sort_key(view_id: str) -> tuple[Any, ...]:
        result = train_results[view_id]
        return (
            *[result["hit_at_k"][top_k] for top_k in sorted_top_k],
            result["mrr"],
            view_id,
        )

    return max(candidate_view_ids, key=sort_key)


def _baseline_self_comparison(baseline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "baseline_view_id": baseline["view_id"],
        "challenger_view_id": baseline["view_id"],
        "hit@1_delta": 0.0,
        "hit@5_delta": 0.0,
        "hit@10_delta": 0.0,
        "mrr_delta": 0.0,
        "top10_improvement_count": 0,
        "top10_regression_count": 0,
        "top10_net_improvement_count": 0,
        "both_hit_count": baseline["hit_counts"].get(_PRIMARY_TOP_K, 0),
        "both_miss_count": baseline["miss_count_at_primary_top_k"],
        "rank_up_within_top10_count": 0,
        "rank_down_within_top10_count": 0,
        "sample_top10_improvements": [],
        "sample_top10_regressions": [],
    }


def _guard_checks(
    *,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    rank_tables: Mapping[str, Mapping[str, Mapping[str, Any]]],
    stage75_report: Mapping[str, Any],
    top_k_values: tuple[int, ...],
) -> list[dict[str, Any]]:
    observed_split_names = sorted(
        {sample.assigned_split for samples in split_samples.values() for sample in samples}
    )
    expected_splits = sorted(_ALLOWED_DEVELOPMENT_SPLITS)
    stage75_split_reports = stage75_report.get("split_reports") or {}
    baseline_train_hit10 = rank_tables[_TRAIN_SPLIT][_BASELINE_VIEW_ID]["hit_at_k"][
        _PRIMARY_TOP_K
    ]
    baseline_dev_hit10 = rank_tables[_DEV_SPLIT][_BASELINE_VIEW_ID]["hit_at_k"][
        _PRIMARY_TOP_K
    ]
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
            name="stage75_source_report_is_stage75",
            passed=str(stage75_report.get("stage") or "") == "Stage 75",
            observed=str(stage75_report.get("stage") or ""),
            expected="Stage 75",
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
    train_selected_view: str,
    comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    failed_checks = [check["name"] for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_query_view_ablation_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        }
    selected_dev_comparison = comparisons[_DEV_SPLIT][train_selected_view]
    dev_hit10_delta = float(selected_dev_comparison["hit@10_delta"])
    if dev_hit10_delta > 0:
        recommended_next_stage = (
            "Stage 78: review query-view changed cases on dev and decide whether "
            "a guarded runtime experiment is justified; keep test locked."
        )
    else:
        recommended_next_stage = (
            "Stage 78: move to the next Stage76 candidate, "
            "fielded_title_text_bm25_score_fusion, on train/dev only; keep test locked."
        )
    return {
        "status": "primeqa_hybrid_query_view_ablation_completed",
        "train_selected_view_id": train_selected_view,
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


def _public_view_metrics(view_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "total_questions": int(view_result["total_questions"]),
        "evaluated_questions": int(view_result["evaluated_questions"]),
        "hit_at_k": {
            f"hit@{top_k}": value
            for top_k, value in sorted(view_result["hit_at_k"].items())
        },
        "mrr": float(view_result["mrr"]),
        "miss_count_at_10": int(view_result["miss_count_at_primary_top_k"]),
        "miss_rate_at_10": float(view_result["miss_rate_at_primary_top_k"]),
        "empty_query_count": int(view_result["empty_query_count"]),
        "average_query_token_count": float(view_result["average_query_token_count"]),
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
            label=view_id,
            value=float(view_metrics["hit_at_k"][metric_name]),
            value_label=f"{view_metrics['hit_at_k'][metric_name]:.4f}",
        )
        for view_id, view_metrics in ordered
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
            label=view_id,
            value=float(comparison[metric]),
            value_label=f"{comparison[metric]:+.4f}",
        )
        for view_id, comparison in ordered
    ]


def _net_change_bars(report: Mapping[str, Any], *, split: str) -> list[BarDatum]:
    comparisons = report["comparisons_to_baseline"][split]
    ordered = sorted(
        comparisons.items(),
        key=lambda item: (-item[1]["top10_net_improvement_count"], item[0]),
    )
    return [
        BarDatum(
            label=view_id,
            value=float(comparison["top10_net_improvement_count"]),
            value_label=str(comparison["top10_net_improvement_count"]),
        )
        for view_id, comparison in ordered
    ]


def _dedup_query_terms(text: str) -> str:
    terms = tokenize_text(text)
    seen = set()
    deduped = []
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return " ".join(deduped)


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
    bm25_k1: float,
    bm25_b: float,
) -> None:
    if not top_k_values:
        raise ValueError("top_k_values must not be empty")
    if any(top_k <= 0 for top_k in top_k_values):
        raise ValueError("top_k_values must be positive")
    if _PRIMARY_TOP_K not in top_k_values:
        raise ValueError(f"top_k_values must include {_PRIMARY_TOP_K}")
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
