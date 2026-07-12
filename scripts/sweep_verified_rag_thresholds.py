from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator
from ts_rag_agent.application.verified_rag_threshold_sweep import (
    ThresholdSweepResult,
    ThresholdSweepSummary,
    VerifiedRAGThresholdSweeper,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_documents,
    load_primeqa_questions,
)

app = typer.Typer(help="扫描验证版 RAG 的拒答阈值和检索上下文参数。")


@app.command()
def main(
    split: Annotated[
        str,
        typer.Option("--split", help="要扫描的问题集合，可选值：dev、train。"),
    ] = "dev",
    retrieval_top_k_values: Annotated[
        str,
        typer.Option("--retrieval-top-k-values", help="逗号分隔的 retrieval top-k 列表。"),
    ] = "5,10,20",
    min_evidence_scores: Annotated[
        str,
        typer.Option("--min-evidence-scores", help="逗号分隔的最低证据分阈值列表。"),
    ] = "4,5,6,7,8",
    max_citation_ranks: Annotated[
        str,
        typer.Option("--max-citation-ranks", help="逗号分隔的最大引用排名列表。"),
    ] = "3,5",
    max_sentences: Annotated[
        int,
        typer.Option("--max-sentences", help="每个答案最多抽取多少个证据句。"),
    ] = 3,
    min_sentence_score: Annotated[
        float,
        typer.Option("--min-sentence-score", help="低于该分数则生成器拒答。"),
    ] = 2.0,
    min_citations: Annotated[
        int,
        typer.Option("--min-citations", help="验证器要求的最少引用数量。"),
    ] = 1,
    sample_limit_per_bucket: Annotated[
        int,
        typer.Option("--sample-limit-per-bucket", help="每组参数每类问题最多保存多少条样例。"),
    ] = 0,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help=(
                "可选 JSON 报告路径。默认写入 "
                "artifacts/verified_rag_threshold_sweep_<split>.json。"
            ),
        ),
    ] = None,
) -> None:
    """运行验证阈值网格扫描，并保存完整 JSON 报告。"""

    parsed_retrieval_top_k_values = _parse_int_values(
        retrieval_top_k_values,
        option_name="--retrieval-top-k-values",
    )
    parsed_min_evidence_scores = _parse_float_values(
        min_evidence_scores,
        option_name="--min-evidence-scores",
    )
    parsed_max_citation_ranks = _parse_int_values(
        max_citation_ranks,
        option_name="--max-citation-ranks",
    )
    _validate_options(
        retrieval_top_k_values=parsed_retrieval_top_k_values,
        min_evidence_scores=parsed_min_evidence_scores,
        max_citation_ranks=parsed_max_citation_ranks,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        min_citations=min_citations,
        sample_limit_per_bucket=sample_limit_per_bucket,
    )

    settings = ProjectSettings()
    training_dir = settings.primeqa_raw_dir / "TechQA" / "training_and_dev"
    documents_path = training_dir / "training_dev_technotes.sections.json"
    questions_path = _resolve_questions_path(training_dir, split)
    output_path = output or settings.artifact_dir / f"verified_rag_threshold_sweep_{split}.json"

    _ensure_file_exists(documents_path)
    _ensure_file_exists(questions_path)

    started_at = time.perf_counter()
    documents_by_id = load_primeqa_documents(documents_path)
    documents = list(documents_by_id.values())
    questions = load_primeqa_questions(questions_path)
    loaded_at = time.perf_counter()

    retriever = BM25Retriever()
    retriever.fit(documents)
    indexed_at = time.perf_counter()

    sweeper = VerifiedRAGThresholdSweeper(
        retriever=retriever,
        answer_generator=ExtractiveAnswerGenerator(
            max_sentences=max_sentences,
            min_sentence_score=min_sentence_score,
        ),
        min_citations=min_citations,
    )
    sweep_result = sweeper.sweep(
        questions=questions,
        retrieval_top_k_values=parsed_retrieval_top_k_values,
        min_evidence_scores=parsed_min_evidence_scores,
        max_citation_ranks=parsed_max_citation_ranks,
        sample_limit_per_bucket=sample_limit_per_bucket,
    )
    swept_at = time.perf_counter()

    report = _build_report(
        split=split,
        documents_path=documents_path,
        questions_path=questions_path,
        document_count=len(documents),
        question_count=len(questions),
        retrieval_top_k_values=parsed_retrieval_top_k_values,
        min_evidence_scores=parsed_min_evidence_scores,
        max_citation_ranks=parsed_max_citation_ranks,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        min_citations=min_citations,
        sample_limit_per_bucket=sample_limit_per_bucket,
        sweep_result=sweep_result,
        timing_seconds={
            "load_data": round(loaded_at - started_at, 3),
            "bm25_index": round(indexed_at - loaded_at, 3),
            "sweep": round(swept_at - indexed_at, 3),
            "total": round(swept_at - started_at, 3),
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(json.dumps(_summarize_report(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved verified RAG threshold sweep: {output_path}")


def _build_report(
    split: str,
    documents_path: Path,
    questions_path: Path,
    document_count: int,
    question_count: int,
    retrieval_top_k_values: list[int],
    min_evidence_scores: list[float],
    max_citation_ranks: list[int],
    max_sentences: int,
    min_sentence_score: float,
    min_citations: int,
    sample_limit_per_bucket: int,
    sweep_result: ThresholdSweepResult,
    timing_seconds: dict[str, float],
) -> dict:
    results = [
        _summary_to_dict(summary, is_pareto=index in sweep_result.pareto_candidate_indices)
        for index, summary in enumerate(sweep_result.summaries)
    ]
    return {
        "dataset": "PrimeQA/TechQA",
        "split": split,
        "paths": {
            "documents": str(documents_path),
            "questions": str(questions_path),
        },
        "grid": {
            "retrieval_top_k_values": retrieval_top_k_values,
            "min_evidence_scores": min_evidence_scores,
            "max_citation_ranks": max_citation_ranks,
            "total_configs": len(results),
        },
        "rag": {
            "retriever": "BM25",
            "answer_generator": "extractive_sentence_baseline",
            "max_sentences": max_sentences,
            "min_sentence_score": min_sentence_score,
            "answer_verifier": "citation_and_evidence_gate",
            "min_citations": min_citations,
            "sample_limit_per_bucket": sample_limit_per_bucket,
        },
        "data": {
            "documents": document_count,
            "questions": question_count,
        },
        "selection_note": (
            "Pareto candidates are configs not dominated on answerable_refusal_rate, "
            "unanswerable_refusal_rate, gold_doc_citation_rate, and average_token_f1. "
            "They are not automatically business-optimal."
        ),
        "pareto_candidate_indices": sweep_result.pareto_candidate_indices,
        "pareto_candidates": [
            results[index] for index in sweep_result.pareto_candidate_indices
        ],
        "results": results,
        "timing_seconds": timing_seconds,
    }


def _summary_to_dict(summary: ThresholdSweepSummary, is_pareto: bool) -> dict:
    original_metrics = asdict(summary.original_metrics)
    verified_metrics = asdict(summary.verified_metrics)
    return {
        "config": asdict(summary.config),
        "is_pareto_candidate": is_pareto,
        "answerable_gold_doc_in_context": summary.answerable_gold_doc_in_context,
        "metrics": {
            "original": original_metrics,
            "verified": verified_metrics,
            "delta": _build_metric_delta(original_metrics, verified_metrics),
        },
        "verification": {
            "reason_counts": summary.reason_counts,
            "newly_refused": summary.newly_refused_count,
        },
        "quality": {
            "newly_refused": summary.quality_analysis["newly_refused"],
            "remaining_risks": summary.quality_analysis["remaining_risks"],
        },
        "samples_by_newly_refused_bucket": summary.quality_analysis.get(
            "samples_by_newly_refused_bucket",
            {},
        ),
    }


def _build_metric_delta(original_metrics: dict, verified_metrics: dict) -> dict[str, float | int]:
    delta = {}
    for key, original_value in original_metrics.items():
        verified_value = verified_metrics[key]
        if isinstance(original_value, float):
            delta[key] = round(verified_value - original_value, 4)
        else:
            delta[key] = verified_value - original_value
    return delta


def _summarize_report(report: dict) -> dict:
    return {
        "dataset": report["dataset"],
        "split": report["split"],
        "grid": report["grid"],
        "rag": report["rag"],
        "data": report["data"],
        "pareto_candidates": [
            _compact_result(candidate) for candidate in report["pareto_candidates"]
        ],
        "timing_seconds": report["timing_seconds"],
    }


def _compact_result(result: dict) -> dict:
    metrics = result["metrics"]["verified"]
    quality = result["quality"]
    return {
        "config": result["config"],
        "answerable_refusal_rate": metrics["answerable_refusal_rate"],
        "unanswerable_refusal_rate": metrics["unanswerable_refusal_rate"],
        "gold_doc_citation_rate": metrics["gold_doc_citation_rate"],
        "average_token_f1": metrics["average_token_f1"],
        "newly_refused": result["verification"]["newly_refused"],
        "threshold_over_refusal_gold_cited": quality["newly_refused"]["bucket_counts"][
            "possible_threshold_over_refusal_gold_cited"
        ],
        "evidence_selection_miss_gold_available": quality["newly_refused"]["bucket_counts"][
            "evidence_selection_miss_gold_available"
        ],
        "unanswerable_still_answered": quality["remaining_risks"][
            "unanswerable_still_answered"
        ],
    }


def _parse_int_values(raw_value: str, option_name: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    except ValueError as exc:
        raise typer.BadParameter(f"{option_name} must contain comma-separated integers.") from exc
    if not values:
        raise typer.BadParameter(f"{option_name} must not be empty.")
    return values


def _parse_float_values(raw_value: str, option_name: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    except ValueError as exc:
        raise typer.BadParameter(f"{option_name} must contain comma-separated numbers.") from exc
    if not values:
        raise typer.BadParameter(f"{option_name} must not be empty.")
    return values


def _validate_options(
    retrieval_top_k_values: list[int],
    min_evidence_scores: list[float],
    max_citation_ranks: list[int],
    max_sentences: int,
    min_sentence_score: float,
    min_citations: int,
    sample_limit_per_bucket: int,
) -> None:
    if any(value <= 0 for value in retrieval_top_k_values):
        raise typer.BadParameter("--retrieval-top-k-values must contain positive integers.")
    if any(value < 0 for value in min_evidence_scores):
        raise typer.BadParameter("--min-evidence-scores must contain non-negative numbers.")
    if any(value <= 0 for value in max_citation_ranks):
        raise typer.BadParameter("--max-citation-ranks must contain positive integers.")
    if max_sentences <= 0:
        raise typer.BadParameter("--max-sentences must be positive.")
    if min_sentence_score < 0:
        raise typer.BadParameter("--min-sentence-score must be non-negative.")
    if min_citations <= 0:
        raise typer.BadParameter("--min-citations must be positive.")
    if sample_limit_per_bucket < 0:
        raise typer.BadParameter("--sample-limit-per-bucket must be non-negative.")


def _resolve_questions_path(training_dir: Path, split: str) -> Path:
    normalized = split.strip().lower()
    if normalized == "dev":
        return training_dir / "dev_Q_A.json"
    if normalized == "train":
        return training_dir / "training_Q_A.json"
    raise typer.BadParameter("--split must be either dev or train.")


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"Missing file: {path}")


if __name__ == "__main__":
    app()
