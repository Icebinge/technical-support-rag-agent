from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.answer_verification import AnswerVerifier
from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator
from ts_rag_agent.application.verified_rag_evaluation import VerifiedRAGEvaluator
from ts_rag_agent.application.verified_rag_quality_analysis import (
    analyze_verified_rag_quality,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_documents,
    load_primeqa_questions,
)

app = typer.Typer(help="分析验证层对 RAG 答案质量的影响。")


@app.command()
def main(
    split: Annotated[
        str,
        typer.Option("--split", help="要分析的问题集合，可选值：dev、train。"),
    ] = "dev",
    retrieval_top_k: Annotated[
        int,
        typer.Option("--retrieval-top-k", help="BM25 提供给回答器的上下文文档数量。"),
    ] = 5,
    max_sentences: Annotated[
        int,
        typer.Option("--max-sentences", help="每个答案最多抽取多少个证据句。"),
    ] = 3,
    min_sentence_score: Annotated[
        float,
        typer.Option("--min-sentence-score", help="低于该分数则生成器拒答。"),
    ] = 2.0,
    min_evidence_score: Annotated[
        float,
        typer.Option("--min-evidence-score", help="验证器接受答案所需的最低证据句得分。"),
    ] = 8.0,
    max_citation_rank: Annotated[
        int,
        typer.Option("--max-citation-rank", help="验证器允许引用的最差检索排名。"),
    ] = 3,
    min_citations: Annotated[
        int,
        typer.Option("--min-citations", help="验证器要求的最少引用数量。"),
    ] = 1,
    sample_limit_per_bucket: Annotated[
        int,
        typer.Option("--sample-limit-per-bucket", help="每类问题最多保存多少条样例。"),
    ] = 5,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help=(
                "可选 JSON 报告路径。默认写入 "
                "artifacts/verified_rag_quality_analysis_<split>.json。"
            ),
        ),
    ] = None,
) -> None:
    """重新运行验证版 RAG，并输出质量分析报告。"""

    _validate_options(
        retrieval_top_k=retrieval_top_k,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        min_evidence_score=min_evidence_score,
        max_citation_rank=max_citation_rank,
        min_citations=min_citations,
        sample_limit_per_bucket=sample_limit_per_bucket,
    )

    settings = ProjectSettings()
    training_dir = settings.primeqa_raw_dir / "TechQA" / "training_and_dev"
    documents_path = training_dir / "training_dev_technotes.sections.json"
    questions_path = _resolve_questions_path(training_dir, split)
    output_path = output or settings.artifact_dir / f"verified_rag_quality_analysis_{split}.json"

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

    evaluator = VerifiedRAGEvaluator(
        retriever=retriever,
        answer_generator=ExtractiveAnswerGenerator(
            max_sentences=max_sentences,
            min_sentence_score=min_sentence_score,
        ),
        answer_verifier=AnswerVerifier(
            min_citations=min_citations,
            min_evidence_score=min_evidence_score,
            max_citation_rank=max_citation_rank,
        ),
        retrieval_top_k=retrieval_top_k,
    )
    evaluation = evaluator.evaluate(questions)
    analyzed_at = time.perf_counter()

    quality_analysis = analyze_verified_rag_quality(
        evaluation,
        min_evidence_score=min_evidence_score,
        sample_limit_per_bucket=sample_limit_per_bucket,
    )
    finished_at = time.perf_counter()

    report = {
        "dataset": "PrimeQA/TechQA",
        "split": split,
        "paths": {
            "documents": str(documents_path),
            "questions": str(questions_path),
        },
        "rag": {
            "retriever": "BM25",
            "retrieval_top_k": retrieval_top_k,
            "answer_generator": "extractive_sentence_baseline",
            "max_sentences": max_sentences,
            "min_sentence_score": min_sentence_score,
            "answer_verifier": "citation_and_evidence_gate",
            "min_evidence_score": min_evidence_score,
            "max_citation_rank": max_citation_rank,
            "min_citations": min_citations,
        },
        "data": {
            "documents": len(documents),
            "questions": len(questions),
            "answerable_questions": evaluation.verified_metrics.answerable_questions,
            "unanswerable_questions": evaluation.verified_metrics.unanswerable_questions,
            "answerable_gold_doc_in_context": evaluation.answerable_gold_doc_in_context,
        },
        "metrics": {
            "original": asdict(evaluation.original_metrics),
            "verified": asdict(evaluation.verified_metrics),
        },
        "quality_analysis": quality_analysis,
        "timing_seconds": {
            "load_data": round(loaded_at - started_at, 3),
            "bm25_index": round(indexed_at - loaded_at, 3),
            "generate_verify_evaluate": round(analyzed_at - indexed_at, 3),
            "quality_analysis": round(finished_at - analyzed_at, 3),
            "total": round(finished_at - started_at, 3),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(json.dumps(_summarize_report(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved verified RAG quality analysis: {output_path}")


def _validate_options(
    retrieval_top_k: int,
    max_sentences: int,
    min_sentence_score: float,
    min_evidence_score: float,
    max_citation_rank: int,
    min_citations: int,
    sample_limit_per_bucket: int,
) -> None:
    if retrieval_top_k <= 0:
        raise typer.BadParameter("--retrieval-top-k must be positive.")
    if max_sentences <= 0:
        raise typer.BadParameter("--max-sentences must be positive.")
    if min_sentence_score < 0:
        raise typer.BadParameter("--min-sentence-score must be non-negative.")
    if min_evidence_score < 0:
        raise typer.BadParameter("--min-evidence-score must be non-negative.")
    if max_citation_rank <= 0:
        raise typer.BadParameter("--max-citation-rank must be positive.")
    if min_citations <= 0:
        raise typer.BadParameter("--min-citations must be positive.")
    if sample_limit_per_bucket < 0:
        raise typer.BadParameter("--sample-limit-per-bucket must be non-negative.")


def _summarize_report(report: dict) -> dict:
    quality_analysis = report["quality_analysis"]
    return {
        "dataset": report["dataset"],
        "split": report["split"],
        "rag": report["rag"],
        "data": report["data"],
        "metrics": report["metrics"],
        "quality_summary": {
            "newly_refused": quality_analysis["newly_refused"],
            "remaining_risks": quality_analysis["remaining_risks"],
        },
        "timing_seconds": report["timing_seconds"],
    }


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
