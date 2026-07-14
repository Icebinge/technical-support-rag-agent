from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.msqa_evaluation_split import (
    MsqaEvaluationRow,
    load_msqa_contract_rows,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.application.text_metrics import token_f1
from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever

_SPLIT_NAME = "msqa_stage57_project_eval_v1"
_ADAPTER_CONTRACT_VERSION = "msqa_eval_adapter_v1"
_SUPPORTED_CORPUS_MODES = ("answer_only", "question_answer_page_text")
_SUPPORTED_CORPUS_SCOPES = ("frozen_split_only", "all_contract_rows")


@dataclass(frozen=True)
class MsqaBaselineSample:
    """One frozen MSQA evaluation sample loaded from Stage 57 JSONL."""

    question_id: str
    question: str
    answer: str
    source_url: str


@dataclass(frozen=True)
class MsqaBaselineVisualization:
    """One generated Stage 58 MSQA baseline visualization."""

    name: str
    path: str


def evaluate_msqa_topk_baseline(
    *,
    msqa_csv_path: Path,
    split_jsonl_path: Path,
    top_k_values: Sequence[int] = (1, 3, 5, 10),
    corpus_modes: Sequence[str] = _SUPPORTED_CORPUS_MODES,
    corpus_scope: str = "frozen_split_only",
    sample_limit: int = 20,
) -> dict[str, Any]:
    """Evaluate MSQA source-row top-k baselines on the frozen Stage 57 split."""

    _ensure_file(msqa_csv_path)
    _ensure_file(split_jsonl_path)
    top_k_values = _validate_top_k_values(top_k_values)
    corpus_modes = _validate_corpus_modes(corpus_modes)
    corpus_scope = _validate_corpus_scope(corpus_scope)
    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")

    started_at = time.perf_counter()
    samples = load_msqa_baseline_samples(split_jsonl_path)
    if corpus_scope == "frozen_split_only":
        corpus_rows = _corpus_rows_from_samples(samples)
        rejected_contract_rows = {}
    else:
        corpus_rows, rejected_contract_rows = load_msqa_contract_rows(msqa_csv_path)
    loaded_at = time.perf_counter()

    variants = []
    for mode in corpus_modes:
        variant_started_at = time.perf_counter()
        retriever = BM25Retriever()
        documents = _documents_for_mode(corpus_rows, mode)
        retriever.fit(documents)
        indexed_at = time.perf_counter()
        variants.append(
            _evaluate_variant(
                mode=mode,
                corpus_rows=corpus_rows,
                samples=samples,
                retriever=retriever,
                top_k_values=top_k_values,
                sample_limit=sample_limit,
                index_seconds=round(indexed_at - variant_started_at, 3),
            )
        )
    finished_at = time.perf_counter()
    return {
        "stage": "Stage 58",
        "created_at": "2026-07-14",
        "analysis_scope": (
            "MSQA frozen-split answer-source top-k baseline. This report evaluates "
            "BM25 retrieval over MSQA Q&A source rows. It is not a PrimeQA-style "
            "document-grounded verified RAG metric, does not run Stage 51, does not "
            "tune policies, and does not change the default runtime."
        ),
        "source_files": {
            "msqa_csv": _fingerprint(msqa_csv_path),
            "split_jsonl": _fingerprint(split_jsonl_path),
        },
        "input_contract": {
            "split_name": _SPLIT_NAME,
            "adapter_contract_version": _ADAPTER_CONTRACT_VERSION,
            "query_field": "question",
            "gold_answer_field": "answer",
            "gold_source_id": "question_id",
            "gold_source_url_field": "source_url",
            "no_answer_field_fallback": True,
        },
        "baseline_definition": {
            "retriever": "BM25",
            "evaluated_top_k_values": list(top_k_values),
            "primary_variant": "answer_only",
            "diagnostic_variant": "question_answer_page_text",
            "corpus_scope": corpus_scope,
            "metric_boundary": (
                "hit@k and MRR measure whether the gold MSQA Q&A source row is "
                "retrieved. token_f1 measures retrieved row answer text against "
                "the frozen gold ProcessedAnswerText. These are answer-source "
                "baseline metrics, not document-span citation metrics."
            ),
        },
        "data": {
            "corpus_contract_rows": len(corpus_rows),
            "rejected_contract_rows": rejected_contract_rows,
            "frozen_split_samples": len(samples),
        },
        "variants": variants,
        "decision": _decision(variants),
        "timing_seconds": {
            "load": round(loaded_at - started_at, 3),
            "total": round(finished_at - started_at, 3),
        },
    }


def load_msqa_baseline_samples(split_jsonl_path: Path) -> list[MsqaBaselineSample]:
    """Load the frozen Stage 57 MSQA split JSONL."""

    _ensure_file(split_jsonl_path)
    samples = []
    for line_number, line in enumerate(
        split_jsonl_path.read_text(encoding="utf-8").split("\n"),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != _SPLIT_NAME:
            raise ValueError(
                f"Unexpected split at line {line_number}: {row.get('split')!r}"
            )
        if row.get("adapter_contract_version") != _ADAPTER_CONTRACT_VERSION:
            raise ValueError(
                "Unexpected adapter contract at line "
                f"{line_number}: {row.get('adapter_contract_version')!r}"
            )
        samples.append(
            MsqaBaselineSample(
                question_id=str(row["question_id"]),
                question=str(row["question"]),
                answer=str(row["answer"]),
                source_url=str(row["source_url"]),
            )
        )
    if not samples:
        raise ValueError(f"No MSQA samples loaded from {split_jsonl_path}")
    return samples


def write_msqa_topk_baseline_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[MsqaBaselineVisualization]:
    """Write SVG charts for Stage 58 MSQA baselines."""

    output_dir.mkdir(parents=True, exist_ok=True)
    variants = report["variants"]
    hit_at_1 = []
    hit_at_10 = []
    mrr = []
    top1_f1 = []
    for variant in variants:
        mode = variant["corpus_mode"]
        hit_at_1.append(
            BarDatum(
                label=mode,
                value=float(variant["retrieval_metrics"]["hit_at_k"].get("hit@1", 0)),
                value_label=str(variant["retrieval_metrics"]["hit_at_k"].get("hit@1", 0)),
            )
        )
        hit_at_10.append(
            BarDatum(
                label=mode,
                value=float(variant["retrieval_metrics"]["hit_at_k"].get("hit@10", 0)),
                value_label=str(variant["retrieval_metrics"]["hit_at_k"].get("hit@10", 0)),
            )
        )
        mrr.append(
            BarDatum(
                label=mode,
                value=float(variant["retrieval_metrics"]["mrr"]),
                value_label=str(variant["retrieval_metrics"]["mrr"]),
            )
        )
        top1_f1.append(
            BarDatum(
                label=mode,
                value=float(variant["answer_metrics"]["average_top1_token_f1"]),
                value_label=str(variant["answer_metrics"]["average_top1_token_f1"]),
            )
        )

    charts = {
        "stage58_msqa_hit_at_1.svg": render_horizontal_bar_chart_svg(
            title="Stage 58 MSQA answer-source hit@1",
            bars=hit_at_1,
            x_label="hit@1",
            margin_left=280,
        ),
        "stage58_msqa_hit_at_10.svg": render_horizontal_bar_chart_svg(
            title="Stage 58 MSQA answer-source hit@10",
            bars=hit_at_10,
            x_label="hit@10",
            margin_left=280,
        ),
        "stage58_msqa_mrr.svg": render_horizontal_bar_chart_svg(
            title="Stage 58 MSQA answer-source MRR",
            bars=mrr,
            x_label="MRR",
            margin_left=280,
        ),
        "stage58_msqa_top1_answer_f1.svg": render_horizontal_bar_chart_svg(
            title="Stage 58 MSQA top1 answer token F1",
            bars=top1_f1,
            x_label="average token F1",
            margin_left=280,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(MsqaBaselineVisualization(name=filename, path=str(path)))
    return artifacts


def _evaluate_variant(
    *,
    mode: str,
    corpus_rows: Sequence[MsqaEvaluationRow],
    samples: Sequence[MsqaBaselineSample],
    retriever: BM25Retriever,
    top_k_values: Sequence[int],
    sample_limit: int,
    index_seconds: float,
) -> dict[str, Any]:
    evaluate_started_at = time.perf_counter()
    rows_by_id = {row.question_id: row for row in corpus_rows}
    max_k = max(top_k_values)
    hit_counts = {top_k: 0 for top_k in top_k_values}
    reciprocal_rank_sum = 0.0
    top1_f1_values = []
    oracle_f1_sums = {top_k: 0.0 for top_k in top_k_values}
    gold_ranks = []
    failure_samples = []
    low_f1_samples = []

    for sample in samples:
        results = retriever.search(sample.question, top_k=max_k)
        result_ids = [result.document.id for result in results]
        gold_rank = (
            result_ids.index(sample.question_id) + 1
            if sample.question_id in result_ids
            else None
        )
        if gold_rank is not None:
            gold_ranks.append(gold_rank)
            reciprocal_rank_sum += 1 / gold_rank
        for top_k in top_k_values:
            if gold_rank is not None and gold_rank <= top_k:
                hit_counts[top_k] += 1
            oracle_f1_sums[top_k] += _best_answer_f1(
                result_ids=result_ids[:top_k],
                rows_by_id=rows_by_id,
                gold_answer=sample.answer,
            )
        top1_answer = rows_by_id[result_ids[0]].answer if result_ids else ""
        top1_f1 = token_f1(top1_answer, sample.answer)
        top1_f1_values.append(top1_f1)
        if gold_rank is None and len(failure_samples) < sample_limit:
            failure_samples.append(_failure_sample(sample, results, top1_f1))
        if top1_f1 < 0.3 and len(low_f1_samples) < sample_limit:
            low_f1_samples.append(_failure_sample(sample, results, top1_f1))

    total = len(samples)
    return {
        "corpus_mode": mode,
        "corpus_text_definition": _mode_description(mode),
        "retrieval_metrics": {
            "evaluated_questions": total,
            "hit_at_k": {
                f"hit@{top_k}": round(hit_counts[top_k] / total, 4)
                for top_k in top_k_values
            },
            "mrr": round(reciprocal_rank_sum / total, 4),
            "median_gold_rank_when_found": statistics.median(gold_ranks)
            if gold_ranks
            else None,
            "gold_source_not_found_at_max_k": total - hit_counts[max_k],
        },
        "answer_metrics": {
            "average_top1_token_f1": round(sum(top1_f1_values) / total, 4),
            "oracle_answer_token_f1_at_k": {
                f"oracle@{top_k}": round(oracle_f1_sums[top_k] / total, 4)
                for top_k in top_k_values
            },
            "top1_low_f1_threshold": 0.3,
            "top1_low_f1_count": sum(1 for value in top1_f1_values if value < 0.3),
        },
        "timing_seconds": {
            "index": index_seconds,
            "evaluate": round(time.perf_counter() - evaluate_started_at, 3),
        },
        "failure_mode_counts": {
            f"gold_source_missing_at_{max_k}": total - hit_counts[max_k],
            "top1_wrong_source": total - hit_counts[1],
            "top1_token_f1_below_0_3": sum(
                1 for value in top1_f1_values if value < 0.3
            ),
        },
        "samples": {
            f"gold_source_missing_at_{max_k}": failure_samples,
            "top1_token_f1_below_0_3": low_f1_samples,
        },
    }


def _documents_for_mode(
    rows: Sequence[MsqaEvaluationRow],
    mode: str,
) -> list[PrimeQADocument]:
    documents = []
    for row in rows:
        if mode == "answer_only":
            title = ""
            text = row.answer
        elif mode == "question_answer_page_text":
            title = row.question
            text = f"Question:\n{row.question}\n\nAccepted answer:\n{row.answer}"
        else:
            raise ValueError(f"Unsupported corpus mode: {mode}")
        documents.append(
            PrimeQADocument(
                id=row.question_id,
                title=title,
                text=text,
            )
        )
    return documents


def _corpus_rows_from_samples(
    samples: Sequence[MsqaBaselineSample],
) -> list[MsqaEvaluationRow]:
    rows = []
    for index, sample in enumerate(samples, start=1):
        rows.append(
            MsqaEvaluationRow(
                question_id=sample.question_id,
                answer_id="",
                source_split="msqa_stage57_project_eval_v1",
                question=sample.question,
                answer=sample.answer,
                source_url=sample.source_url,
                tags="",
                is_azure="",
                is_m365="",
                is_other="",
                is_short="",
                is_long="",
                normalized_question="",
                source_row_index=index,
            )
        )
    return rows


def _best_answer_f1(
    *,
    result_ids: Sequence[str],
    rows_by_id: Mapping[str, MsqaEvaluationRow],
    gold_answer: str,
) -> float:
    if not result_ids:
        return 0.0
    return max(token_f1(rows_by_id[result_id].answer, gold_answer) for result_id in result_ids)


def _failure_sample(
    sample: MsqaBaselineSample,
    results: Sequence[Any],
    top1_f1: float,
) -> dict[str, Any]:
    return {
        "question_id": sample.question_id,
        "question_preview": _preview(sample.question),
        "gold_answer_preview": _preview(sample.answer),
        "source_url": sample.source_url,
        "top1_token_f1": round(top1_f1, 4),
        "retrieved": [
            {
                "rank": result.rank,
                "question_id": result.document.id,
                "title_preview": _preview(result.document.title),
                "score": round(result.score, 4),
            }
            for result in results[:5]
        ],
    }


def _decision(variants: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary = next(
        variant for variant in variants if variant["corpus_mode"] == "answer_only"
    )
    return {
        "status": "msqa_topk_baseline_recorded",
        "primary_baseline_variant": "answer_only",
        "can_run_stage51_candidate_now": False,
        "can_defaultize_runtime_now": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage 59: review MSQA baseline failure modes and decide whether "
            "Stage 51 can be adapted fairly to the MSQA answer-source task"
        ),
        "reason": (
            "The frozen MSQA split now has answer-source BM25 baseline metrics. "
            f"Primary answer_only hit@10 is "
            f"{primary['retrieval_metrics']['hit_at_k'].get('hit@10')}; "
            "Stage 51 comparison remains blocked until compatibility with this "
            "MSQA answer-source task is reviewed."
        ),
    }


def _validate_top_k_values(values: Sequence[int]) -> tuple[int, ...]:
    top_k_values = tuple(int(value) for value in values)
    if not top_k_values:
        raise ValueError("top_k_values must not be empty")
    if any(value <= 0 for value in top_k_values):
        raise ValueError("top_k_values must be positive")
    return tuple(sorted(set(top_k_values)))


def _validate_corpus_modes(values: Sequence[str]) -> tuple[str, ...]:
    modes = tuple(value.strip() for value in values)
    if not modes:
        raise ValueError("corpus_modes must not be empty")
    unsupported = [mode for mode in modes if mode not in _SUPPORTED_CORPUS_MODES]
    if unsupported:
        raise ValueError(f"Unsupported corpus modes: {unsupported}")
    return modes


def _validate_corpus_scope(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in _SUPPORTED_CORPUS_SCOPES:
        raise ValueError(
            "corpus_scope must be one of: "
            + ", ".join(_SUPPORTED_CORPUS_SCOPES)
        )
    return normalized


def _mode_description(mode: str) -> str:
    if mode == "answer_only":
        return "BM25 document text is the MSQA ProcessedAnswerText only."
    if mode == "question_answer_page_text":
        return (
            "BM25 document text is the MSQA question plus ProcessedAnswerText, "
            "approximating Q&A page text and expected to be easier."
        )
    raise ValueError(f"Unsupported corpus mode: {mode}")


def _fingerprint(path: Path) -> dict[str, Any]:
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


def _preview(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."
