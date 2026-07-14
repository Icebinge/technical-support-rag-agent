from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Sequence
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

_STAGE = "Stage 75"
_CREATED_AT = "2026-07-15"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})


@dataclass(frozen=True)
class PrimeQAHybridBM25MissAnalysisVisualization:
    """One generated Stage75 BM25 miss-analysis visualization."""

    name: str
    path: str


@dataclass(frozen=True)
class _DocumentTokenProfile:
    token_count: int
    unique_tokens: frozenset[str]
    title_unique_tokens: frozenset[str]
    text_unique_tokens: frozenset[str]


def run_primeqa_hybrid_bm25_miss_analysis(
    *,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    candidate_dataset_path: Path,
    top_k: int = 10,
    search_depth: int = 50,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> dict[str, Any]:
    """Analyze BM25 top-k miss cases on train/dev only."""

    _validate_options(
        top_k=top_k,
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
    loaded_documents_at = time.perf_counter()
    candidate_scan = _scan_candidate_routes(candidate_dataset_path)
    route_by_question = candidate_scan["route_by_question"]
    scanned_candidates_at = time.perf_counter()
    retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
    retriever.fit(documents.values())
    indexed_at = time.perf_counter()
    profile_cache: dict[str, _DocumentTokenProfile] = {}
    split_reports = {
        split: _analyze_split(
            split=split,
            samples=samples,
            documents=documents,
            retriever=retriever,
            route_by_question=route_by_question,
            profile_cache=profile_cache,
            top_k=top_k,
            search_depth=search_depth,
        )
        for split, samples in split_samples.items()
    }
    analyzed_at = time.perf_counter()
    guard_checks = _guard_checks(
        split_samples=split_samples,
        candidate_scan=candidate_scan,
        top_k=top_k,
        search_depth=search_depth,
    )
    checked_at = time.perf_counter()
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "Train/dev-only BM25 top10 miss analysis for "
            "primeqa_hybrid_stage68_v1. This stage diagnoses answerable "
            "questions whose gold document is not retrieved in the development "
            "top-k window. It keeps the frozen test split locked, does not run "
            "final metrics, does not tune on test, and does not change runtime "
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
            "candidate_dataset": _fingerprint(candidate_dataset_path),
        },
        "config": {
            "top_k": top_k,
            "search_depth": search_depth,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
        },
        "loaded_data_summary": {
            "document_count": len(documents),
            "candidate_rows": candidate_scan["row_count"],
            "candidate_rows_by_split": candidate_scan["rows_by_split"],
            "candidate_questions_by_split": candidate_scan["question_count_by_split"],
            "candidate_rows_with_test_split": candidate_scan["rows_with_test_split"],
            "split_rows": {
                split: len(samples) for split, samples in sorted(split_samples.items())
            },
            "answerable_rows": {
                split: sum(sample.answerable for sample in samples)
                for split, samples in sorted(split_samples.items())
            },
        },
        "split_reports": split_reports,
        "cross_split_summary": _cross_split_summary(split_reports),
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks, split_reports),
        "timing_seconds": {
            "load_splits": round(loaded_splits_at - started_at, 3),
            "load_documents": round(loaded_documents_at - loaded_splits_at, 3),
            "scan_candidate_routes": round(
                scanned_candidates_at - loaded_documents_at,
                3,
            ),
            "bm25_index": round(indexed_at - scanned_candidates_at, 3),
            "miss_analysis": round(analyzed_at - indexed_at, 3),
            "guard_checks": round(checked_at - analyzed_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return report


def write_primeqa_hybrid_bm25_miss_analysis_visualizations(
    report: dict[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridBM25MissAnalysisVisualization]:
    """Write SVG charts for Stage75 BM25 miss analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage75_bm25_miss_count_by_split.svg": render_horizontal_bar_chart_svg(
            title="Stage75 BM25 top10 miss count by split",
            bars=_miss_count_bars(report),
            x_label="missed answerable questions",
            margin_left=180,
        ),
        "stage75_bm25_miss_rate_by_split.svg": render_horizontal_bar_chart_svg(
            title="Stage75 BM25 top10 miss rate by split",
            bars=_miss_rate_bars(report),
            x_label="miss rate",
            margin_left=180,
        ),
        "stage75_bm25_miss_reason_tags.svg": render_horizontal_bar_chart_svg(
            title="Stage75 BM25 miss reason tags",
            bars=_reason_tag_bars(report),
            x_label="miss cases",
            width=1120,
            margin_left=470,
        ),
        "stage75_bm25_miss_rank_buckets.svg": render_horizontal_bar_chart_svg(
            title="Stage75 BM25 miss gold-rank buckets",
            bars=_rank_bucket_bars(report),
            x_label="miss cases",
            width=1040,
            margin_left=390,
        ),
        "stage75_bm25_dev_miss_routes.svg": render_horizontal_bar_chart_svg(
            title="Stage75 BM25 dev miss routes",
            bars=_dev_route_bars(report),
            x_label="dev miss cases",
            width=1120,
            margin_left=470,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridBM25MissAnalysisVisualization(name=filename, path=str(path))
        )
    return artifacts


def _analyze_split(
    *,
    split: str,
    samples: list[PrimeQAHybridSplitSample],
    documents: dict[str, Any],
    retriever: BM25Retriever,
    route_by_question: dict[str, str],
    profile_cache: dict[str, _DocumentTokenProfile],
    top_k: int,
    search_depth: int,
) -> dict[str, Any]:
    answerable_samples = [
        sample
        for sample in samples
        if sample.answerable and sample.answer_doc_id is not None
    ]
    miss_cases = []
    hit_count = 0
    for sample in answerable_samples:
        question = sample.to_primeqa_question()
        results = retriever.search(question.full_question, top_k=search_depth)
        result_doc_ids = [result.document.id for result in results]
        if sample.answer_doc_id in result_doc_ids[:top_k]:
            hit_count += 1
            continue
        miss_cases.append(
            _miss_case(
                split=split,
                sample=sample,
                results=results,
                documents=documents,
                route=route_by_question.get(sample.sample_id, "unknown"),
                profile_cache=profile_cache,
                top_k=top_k,
                search_depth=search_depth,
            )
        )
    evaluated_count = len(answerable_samples)
    miss_count = len(miss_cases)
    return {
        "total_questions": len(samples),
        "evaluated_questions": evaluated_count,
        "hit_at_top_k": _rounded_ratio(hit_count, evaluated_count),
        "miss_count": miss_count,
        "miss_rate": _rounded_ratio(miss_count, evaluated_count),
        "hit_count": hit_count,
        "reason_tag_counts": _counter_dict(
            tag for case in miss_cases for tag in case["reason_tags"]
        ),
        "route_miss_counts": _counter_dict(case["question_route"] for case in miss_cases),
        "source_split_miss_counts": _counter_dict(
            case["source_split"] for case in miss_cases
        ),
        "gold_rank_bucket_counts": _counter_dict(
            case["gold_rank_bucket"] for case in miss_cases
        ),
        "gold_in_source_candidate_doc_ids_counts": _counter_dict(
            str(case["gold_in_source_candidate_doc_ids"]).lower()
            for case in miss_cases
        ),
        "query_length_bucket_counts": _counter_dict(
            case["query_length_bucket"] for case in miss_cases
        ),
        "average_gold_query_overlap_ratio": _rounded_mean(
            [case["gold_query_overlap_ratio"] for case in miss_cases]
        ),
        "miss_cases": miss_cases,
    }


def _miss_case(
    *,
    split: str,
    sample: PrimeQAHybridSplitSample,
    results: Sequence[Any],
    documents: dict[str, Any],
    route: str,
    profile_cache: dict[str, _DocumentTokenProfile],
    top_k: int,
    search_depth: int,
) -> dict[str, Any]:
    query_tokens = tokenize_text(sample.to_primeqa_question().full_question)
    query_unique_tokens = frozenset(query_tokens)
    answer_doc_id = str(sample.answer_doc_id)
    result_doc_ids = [result.document.id for result in results]
    gold_rank = (
        result_doc_ids.index(answer_doc_id) + 1
        if answer_doc_id in result_doc_ids
        else None
    )
    gold_profile = (
        _document_profile(answer_doc_id, documents, profile_cache)
        if answer_doc_id in documents
        else None
    )
    gold_query_overlap = (
        len(query_unique_tokens.intersection(gold_profile.unique_tokens))
        if gold_profile
        else 0
    )
    gold_title_query_overlap = (
        len(query_unique_tokens.intersection(gold_profile.title_unique_tokens))
        if gold_profile
        else 0
    )
    gold_text_query_overlap = (
        len(query_unique_tokens.intersection(gold_profile.text_unique_tokens))
        if gold_profile
        else 0
    )
    top_results = [
        _top_result_summary(
            result=result,
            query_unique_tokens=query_unique_tokens,
            documents=documents,
            profile_cache=profile_cache,
        )
        for result in results[:top_k]
    ]
    top1_overlap = top_results[0]["query_overlap_count"] if top_results else 0
    source_candidate_doc_ids = set(sample.candidate_doc_ids)
    reason_tags = _reason_tags(
        gold_doc_present=gold_profile is not None,
        gold_rank=gold_rank,
        search_depth=search_depth,
        query_unique_count=len(query_unique_tokens),
        gold_query_overlap=gold_query_overlap,
        gold_query_overlap_ratio=_rounded_ratio(
            gold_query_overlap,
            len(query_unique_tokens),
        ),
        top1_query_overlap=top1_overlap,
        gold_in_source_candidate_doc_ids=answer_doc_id in source_candidate_doc_ids,
        top10_has_source_candidate_doc=bool(
            source_candidate_doc_ids.intersection(result_doc_ids[:top_k])
        ),
    )
    return {
        "split": split,
        "sample_id": sample.sample_id,
        "source_split": sample.source_split,
        "split_subtype": sample.split_subtype,
        "question_route": route,
        "answer_doc_id": answer_doc_id,
        "gold_doc_present_in_corpus": gold_profile is not None,
        "gold_rank_within_search_depth": gold_rank,
        "gold_rank_bucket": _gold_rank_bucket(gold_rank, search_depth),
        "gold_in_source_candidate_doc_ids": answer_doc_id in source_candidate_doc_ids,
        "source_candidate_doc_count": len(sample.candidate_doc_ids),
        "query_token_count": len(query_tokens),
        "query_unique_token_count": len(query_unique_tokens),
        "query_length_bucket": _query_length_bucket(len(query_unique_tokens)),
        "gold_document_token_count": gold_profile.token_count if gold_profile else 0,
        "gold_query_overlap_count": gold_query_overlap,
        "gold_query_overlap_ratio": _rounded_ratio(
            gold_query_overlap,
            len(query_unique_tokens),
        ),
        "gold_title_query_overlap_count": gold_title_query_overlap,
        "gold_text_query_overlap_count": gold_text_query_overlap,
        "top1_query_overlap_count": top1_overlap,
        "top10_min_score": round(top_results[-1]["score"], 4) if top_results else 0.0,
        "reason_tags": reason_tags,
        "top_results": top_results,
    }


def _top_result_summary(
    *,
    result: Any,
    query_unique_tokens: frozenset[str],
    documents: dict[str, Any],
    profile_cache: dict[str, _DocumentTokenProfile],
) -> dict[str, Any]:
    profile = _document_profile(result.document.id, documents, profile_cache)
    return {
        "rank": result.rank,
        "doc_id": result.document.id,
        "score": round(result.score, 4),
        "query_overlap_count": len(query_unique_tokens.intersection(profile.unique_tokens)),
        "title_query_overlap_count": len(
            query_unique_tokens.intersection(profile.title_unique_tokens)
        ),
        "document_token_count": profile.token_count,
    }


def _document_profile(
    doc_id: str,
    documents: dict[str, Any],
    profile_cache: dict[str, _DocumentTokenProfile],
) -> _DocumentTokenProfile:
    cached = profile_cache.get(doc_id)
    if cached is not None:
        return cached
    document = documents[doc_id]
    title_tokens = tokenize_text(document.title)
    text_tokens = tokenize_text(document.text)
    profile = _DocumentTokenProfile(
        token_count=len(title_tokens) + len(text_tokens),
        unique_tokens=frozenset([*title_tokens, *text_tokens]),
        title_unique_tokens=frozenset(title_tokens),
        text_unique_tokens=frozenset(text_tokens),
    )
    profile_cache[doc_id] = profile
    return profile


def _reason_tags(
    *,
    gold_doc_present: bool,
    gold_rank: int | None,
    search_depth: int,
    query_unique_count: int,
    gold_query_overlap: int,
    gold_query_overlap_ratio: float,
    top1_query_overlap: int,
    gold_in_source_candidate_doc_ids: bool,
    top10_has_source_candidate_doc: bool,
) -> list[str]:
    tags = []
    if not gold_doc_present:
        tags.append("gold_doc_absent_from_corpus")
    if query_unique_count <= 3:
        tags.append("short_query_lte_3_unique_terms")
    if gold_query_overlap == 0:
        tags.append("gold_doc_zero_query_overlap")
    elif gold_query_overlap <= 1:
        tags.append("gold_doc_low_query_overlap_lte_1")
    if 0 < gold_query_overlap_ratio < 0.25:
        tags.append("gold_doc_query_overlap_ratio_lt_0_25")
    if gold_rank is None:
        tags.append(f"gold_doc_not_found_within_top{search_depth}")
    elif gold_rank <= 20:
        tags.append("gold_doc_rank_11_to_20")
    else:
        tags.append(f"gold_doc_rank_21_to_{search_depth}")
    if top1_query_overlap > gold_query_overlap:
        tags.append("top1_query_overlap_exceeds_gold")
    if not gold_in_source_candidate_doc_ids:
        tags.append("gold_doc_absent_from_source_candidate_doc_ids")
    if top10_has_source_candidate_doc:
        tags.append("top10_contains_source_candidate_doc")
    return sorted(set(tags))


def _scan_candidate_routes(candidate_dataset_path: Path) -> dict[str, Any]:
    _ensure_file(candidate_dataset_path)
    rows_by_split: Counter[str] = Counter()
    question_ids_by_split: dict[str, set[str]] = {}
    route_counts_by_question: dict[str, Counter[str]] = {}
    row_count = 0
    rows_with_test_split = 0
    with candidate_dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected object on line {line_number} in {candidate_dataset_path}"
                )
            row_count += 1
            split = str(row.get("split") or "")
            question_id = str(row.get("question_id") or "")
            rows_by_split[split] += 1
            question_ids_by_split.setdefault(split, set()).add(question_id)
            rows_with_test_split += split in _FORBIDDEN_FINAL_SPLITS
            route = _route_from_candidate_row(row)
            route_counts_by_question.setdefault(question_id, Counter())[route] += 1
    route_by_question = {
        question_id: sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for question_id, counter in route_counts_by_question.items()
    }
    return {
        "row_count": row_count,
        "rows_by_split": dict(sorted(rows_by_split.items())),
        "question_count_by_split": {
            split: len(question_ids)
            for split, question_ids in sorted(question_ids_by_split.items())
        },
        "rows_with_test_split": rows_with_test_split,
        "route_by_question": route_by_question,
    }


def _route_from_candidate_row(row: dict[str, Any]) -> str:
    runtime_features = row.get("runtime_features") or {}
    metadata = row.get("metadata") or {}
    if isinstance(runtime_features, dict) and runtime_features.get("question_route"):
        return str(runtime_features["question_route"])
    if isinstance(metadata, dict) and metadata.get("question_route"):
        return str(metadata["question_route"])
    return "unknown"


def _cross_split_summary(split_reports: dict[str, Any]) -> dict[str, Any]:
    total_evaluated = sum(report["evaluated_questions"] for report in split_reports.values())
    total_hits = sum(report["hit_count"] for report in split_reports.values())
    total_misses = sum(report["miss_count"] for report in split_reports.values())
    reason_counter: Counter[str] = Counter()
    rank_counter: Counter[str] = Counter()
    route_counter: Counter[str] = Counter()
    for report in split_reports.values():
        reason_counter.update(report["reason_tag_counts"])
        rank_counter.update(report["gold_rank_bucket_counts"])
        route_counter.update(report["route_miss_counts"])
    return {
        "evaluated_questions": total_evaluated,
        "hit_count": total_hits,
        "miss_count": total_misses,
        "hit_at_top_k": _rounded_ratio(total_hits, total_evaluated),
        "miss_rate": _rounded_ratio(total_misses, total_evaluated),
        "reason_tag_counts": dict(sorted(reason_counter.items())),
        "gold_rank_bucket_counts": dict(sorted(rank_counter.items())),
        "route_miss_counts": dict(sorted(route_counter.items())),
        "top_reason_tags": _top_counter_items(reason_counter, limit=10),
        "top_miss_routes": _top_counter_items(route_counter, limit=10),
        "improvement_hypotheses": _improvement_hypotheses(reason_counter, rank_counter),
    }


def _improvement_hypotheses(
    reason_counter: Counter[str],
    rank_counter: Counter[str],
) -> list[str]:
    hypotheses = []
    if any(key.startswith("gold_doc_not_found") for key in reason_counter):
        hypotheses.append(
            "Investigate query/document lexical mismatch; many gold documents are not "
            "found within the diagnostic search depth."
        )
    if reason_counter.get("gold_doc_zero_query_overlap", 0):
        hypotheses.append(
            "Add query expansion or document-side field weighting for cases where the "
            "gold document shares no indexed query terms."
        )
    if rank_counter.get("rank_11_to_20", 0) or any(
        key.startswith("rank_21_to_") for key in rank_counter
    ):
        hypotheses.append(
            "Review document-level ranking features for near misses where the gold "
            "document appears below top10."
        )
    if reason_counter.get("top1_query_overlap_exceeds_gold", 0):
        hypotheses.append(
            "Reduce query-overlap-only dominance by adding title/type/answer-signal "
            "features before reranking documents."
        )
    return hypotheses


def _guard_checks(
    *,
    split_samples: dict[str, list[PrimeQAHybridSplitSample]],
    candidate_scan: dict[str, Any],
    top_k: int,
    search_depth: int,
) -> list[dict[str, Any]]:
    observed_split_names = sorted(
        {sample.assigned_split for samples in split_samples.values() for sample in samples}
    )
    candidate_splits = sorted(candidate_scan["rows_by_split"])
    expected_splits = sorted(_ALLOWED_DEVELOPMENT_SPLITS)
    return [
        _check(
            name="analysis_splits_are_train_dev_only",
            passed=observed_split_names == expected_splits,
            observed=observed_split_names,
            expected=expected_splits,
        ),
        _check(
            name="candidate_artifact_splits_are_train_dev_only",
            passed=candidate_splits == expected_splits,
            observed=candidate_splits,
            expected=expected_splits,
        ),
        _check(
            name="candidate_rows_have_no_test_split",
            passed=candidate_scan["rows_with_test_split"] == 0,
            observed=candidate_scan["rows_with_test_split"],
            expected=0,
        ),
        _check(
            name="search_depth_covers_top_k",
            passed=search_depth >= top_k,
            observed={"top_k": top_k, "search_depth": search_depth},
            expected="search_depth >= top_k",
        ),
        _check(
            name="stage75_report_is_public_safe_no_raw_text",
            passed=True,
            observed="raw question/answer/document text omitted",
            expected="raw question/answer/document text omitted",
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
    guard_checks: list[dict[str, Any]],
    split_reports: dict[str, Any],
) -> dict[str, Any]:
    failed_checks = [check["name"] for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_bm25_top10_miss_analysis_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": "primeqa_hybrid_bm25_top10_miss_analysis_completed",
        "can_continue_train_dev_development": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage 76: design train/dev-only retrieval-recall improvement candidates "
            "from the Stage75 miss drivers; do not use test for evaluation or tuning."
        ),
        "train_miss_count": split_reports[_TRAIN_SPLIT]["miss_count"],
        "dev_miss_count": split_reports[_DEV_SPLIT]["miss_count"],
    }


def _miss_count_bars(report: dict[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(split, data["miss_count"], str(data["miss_count"]))
        for split, data in report["split_reports"].items()
    ]


def _miss_rate_bars(report: dict[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(split, data["miss_rate"], f"{data['miss_rate']:.4f}")
        for split, data in report["split_reports"].items()
    ]


def _reason_tag_bars(report: dict[str, Any]) -> list[BarDatum]:
    counter = Counter(report["cross_split_summary"]["reason_tag_counts"])
    return [
        BarDatum(label, value, str(value))
        for label, value in _top_counter_items(counter, limit=12)
    ]


def _rank_bucket_bars(report: dict[str, Any]) -> list[BarDatum]:
    counter = report["cross_split_summary"]["gold_rank_bucket_counts"]
    return [
        BarDatum(label, value, str(value))
        for label, value in sorted(counter.items(), key=lambda item: item[0])
    ]


def _dev_route_bars(report: dict[str, Any]) -> list[BarDatum]:
    counter = Counter(report["split_reports"][_DEV_SPLIT]["route_miss_counts"])
    return [
        BarDatum(label, value, str(value))
        for label, value in _top_counter_items(counter, limit=10)
    ]


def _gold_rank_bucket(gold_rank: int | None, search_depth: int) -> str:
    if gold_rank is None:
        return f"not_found_top{search_depth}"
    if gold_rank <= 10:
        return "rank_1_to_10"
    if gold_rank <= 20:
        return "rank_11_to_20"
    return f"rank_21_to_{search_depth}"


def _query_length_bucket(query_unique_count: int) -> str:
    if query_unique_count <= 3:
        return "unique_terms_0_to_3"
    if query_unique_count <= 8:
        return "unique_terms_4_to_8"
    if query_unique_count <= 15:
        return "unique_terms_9_to_15"
    return "unique_terms_16_plus"


def _top_counter_items(counter: Counter[str], *, limit: int) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _counter_dict(values) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _rounded_ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 4) if denominator else 0.0


def _rounded_mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _check(
    *,
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "observed": observed,
        "expected": expected,
    }


def _fingerprint(path: Path) -> dict[str, Any]:
    _ensure_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")


def _validate_options(
    *,
    top_k: int,
    search_depth: int,
    bm25_k1: float,
    bm25_b: float,
) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if search_depth < top_k:
        raise ValueError("search_depth must be greater than or equal to top_k")
    if bm25_k1 <= 0:
        raise ValueError("bm25_k1 must be positive")
    if not 0 <= bm25_b <= 1:
        raise ValueError("bm25_b must be between 0 and 1")
