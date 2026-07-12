from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.answer_verification import AnswerVerifier
from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator
from ts_rag_agent.application.verified_rag_evaluation import (
    VerifiedRAGEvaluationResult,
    VerifiedRAGEvaluator,
    VerifiedRAGQuestionResult,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_documents,
    load_primeqa_questions,
)

app = typer.Typer(help="在 PrimeQA TechQA 数据集上评估带答案验证层的 RAG baseline。")


@app.command()
def main(
    split: Annotated[
        str,
        typer.Option("--split", help="要评估的问题集合，可选值：dev、train。"),
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
    sample_limit: Annotated[
        int,
        typer.Option("--sample-limit", help="报告中最多保存多少条样例。"),
    ] = 20,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="可选 JSON 报告路径。默认写入 artifacts/verified_rag_<split>_report.json。",
        ),
    ] = None,
) -> None:
    """运行 BM25 + 抽取式回答 + 引用验证的端到端评估。"""

    _validate_options(
        retrieval_top_k=retrieval_top_k,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        min_evidence_score=min_evidence_score,
        max_citation_rank=max_citation_rank,
        min_citations=min_citations,
        sample_limit=sample_limit,
    )

    settings = ProjectSettings()
    training_dir = settings.primeqa_raw_dir / "TechQA" / "training_and_dev"
    documents_path = training_dir / "training_dev_technotes.sections.json"
    questions_path = _resolve_questions_path(training_dir, split)
    output_path = output or settings.artifact_dir / f"verified_rag_{split}_report.json"

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
    evaluated_at = time.perf_counter()

    report = _build_report(
        split=split,
        documents_path=documents_path,
        questions_path=questions_path,
        document_count=len(documents),
        question_count=len(questions),
        retrieval_top_k=retrieval_top_k,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        min_evidence_score=min_evidence_score,
        max_citation_rank=max_citation_rank,
        min_citations=min_citations,
        sample_limit=sample_limit,
        evaluation=evaluation,
        timing_seconds={
            "load_data": round(loaded_at - started_at, 3),
            "bm25_index": round(indexed_at - loaded_at, 3),
            "generate_verify_evaluate": round(evaluated_at - indexed_at, 3),
            "total": round(evaluated_at - started_at, 3),
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(json.dumps(_summarize_report(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved verified RAG report: {output_path}")


def _validate_options(
    retrieval_top_k: int,
    max_sentences: int,
    min_sentence_score: float,
    min_evidence_score: float,
    max_citation_rank: int,
    min_citations: int,
    sample_limit: int,
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
    if sample_limit < 0:
        raise typer.BadParameter("--sample-limit must be non-negative.")


def _build_report(
    split: str,
    documents_path: Path,
    questions_path: Path,
    document_count: int,
    question_count: int,
    retrieval_top_k: int,
    max_sentences: int,
    min_sentence_score: float,
    min_evidence_score: float,
    max_citation_rank: int,
    min_citations: int,
    sample_limit: int,
    evaluation: VerifiedRAGEvaluationResult,
    timing_seconds: dict[str, float],
) -> dict:
    original_metrics = asdict(evaluation.original_metrics)
    verified_metrics = asdict(evaluation.verified_metrics)

    return {
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
            "documents": document_count,
            "questions": question_count,
            "answerable_questions": evaluation.verified_metrics.answerable_questions,
            "unanswerable_questions": evaluation.verified_metrics.unanswerable_questions,
            "answerable_gold_doc_in_context": evaluation.answerable_gold_doc_in_context,
            "answerable_gold_doc_in_context_rate": _safe_rate(
                evaluation.answerable_gold_doc_in_context,
                evaluation.verified_metrics.answerable_questions,
            ),
        },
        "metrics": {
            "original": original_metrics,
            "verified": verified_metrics,
            "delta": _build_metric_delta(original_metrics, verified_metrics),
        },
        "verification": {
            "reason_counts": evaluation.reason_counts,
            "newly_refused": evaluation.newly_refused_count,
        },
        "timing_seconds": timing_seconds,
        "samples": _select_samples(evaluation.question_results, sample_limit),
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


def _select_samples(
    question_results: list[VerifiedRAGQuestionResult],
    sample_limit: int,
) -> list[dict]:
    if sample_limit == 0:
        return []

    changed_results = [
        result
        for result in question_results
        if result.original_answer.refused != result.verified_answer.refused
    ]
    unchanged_results = [result for result in question_results if result not in changed_results]
    selected_results = (changed_results + unchanged_results)[:sample_limit]
    return [_build_sample(result) for result in selected_results]


def _build_sample(question_result: VerifiedRAGQuestionResult) -> dict:
    question = question_result.question
    verification = question_result.verification_result

    return {
        "question_id": question.id,
        "question_title": question.title,
        "answerable": question.answerable,
        "gold_answer_doc_id": question.answer_doc_id,
        "retrieved_documents": [
            {
                "rank": result.rank,
                "document_id": result.document.id,
                "title": result.document.title,
                "score": round(result.score, 4),
            }
            for result in question_result.retrieval_results
        ],
        "original_answer": {
            "answer": question_result.original_answer.answer,
            "refused": question_result.original_answer.refused,
            "citations": [
                asdict(citation) for citation in question_result.original_answer.citations
            ],
        },
        "verified_answer": {
            "answer": question_result.verified_answer.answer,
            "refused": question_result.verified_answer.refused,
            "citations": [
                asdict(citation) for citation in question_result.verified_answer.citations
            ],
        },
        "verification": {
            "citation_context_valid": verification.citation_context_valid,
            "reasons": verification.reasons,
        },
        "gold_answer": question.answer,
    }


def _summarize_report(report: dict) -> dict:
    return {
        "dataset": report["dataset"],
        "split": report["split"],
        "rag": report["rag"],
        "data": report["data"],
        "metrics": report["metrics"],
        "verification": report["verification"],
        "timing_seconds": report["timing_seconds"],
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


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
