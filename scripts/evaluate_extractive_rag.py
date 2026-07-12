from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator, evaluate_answers
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_documents,
    load_primeqa_questions,
)

app = typer.Typer(help="在 PrimeQA TechQA 数据集上评估抽取式 RAG answer baseline。")


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
        typer.Option("--min-sentence-score", help="低于该分数则拒答。"),
    ] = 2.0,
    sample_limit: Annotated[
        int,
        typer.Option("--sample-limit", help="报告中最多保存多少条样例。"),
    ] = 20,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="可选 JSON 报告路径。默认写入 artifacts/extractive_rag_<split>_report.json。",
        ),
    ] = None,
) -> None:
    """运行 BM25 top-k + 抽取式带引用答案的最小 RAG 评估。"""

    if retrieval_top_k <= 0:
        raise typer.BadParameter("--retrieval-top-k must be positive.")
    if sample_limit < 0:
        raise typer.BadParameter("--sample-limit must be non-negative.")

    settings = ProjectSettings()
    training_dir = settings.primeqa_raw_dir / "TechQA" / "training_and_dev"
    documents_path = training_dir / "training_dev_technotes.sections.json"
    questions_path = _resolve_questions_path(training_dir, split)
    output_path = output or settings.artifact_dir / f"extractive_rag_{split}_report.json"

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

    generator = ExtractiveAnswerGenerator(
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
    )
    answers = []
    samples = []
    answerable_gold_doc_in_context = 0

    for question in questions:
        retrieval_results = retriever.search(question.full_question, top_k=retrieval_top_k)
        if question.answerable:
            retrieved_doc_ids = {result.document.id for result in retrieval_results}
            if question.answer_doc_id in retrieved_doc_ids:
                answerable_gold_doc_in_context += 1

        answer = generator.generate(question, retrieval_results)
        answers.append(answer)
        if len(samples) < sample_limit:
            samples.append(_build_sample(question, retrieval_results, answer))

    generated_at = time.perf_counter()
    metrics = evaluate_answers(questions, answers)
    evaluated_at = time.perf_counter()

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
        },
        "data": {
            "documents": len(documents),
            "questions": len(questions),
            "answerable_questions": metrics.answerable_questions,
            "unanswerable_questions": metrics.unanswerable_questions,
            "answerable_gold_doc_in_context": answerable_gold_doc_in_context,
            "answerable_gold_doc_in_context_rate": round(
                answerable_gold_doc_in_context / metrics.answerable_questions,
                4,
            )
            if metrics.answerable_questions
            else 0.0,
        },
        "metrics": asdict(metrics),
        "timing_seconds": {
            "load_data": round(loaded_at - started_at, 3),
            "bm25_index": round(indexed_at - loaded_at, 3),
            "generate_answers": round(generated_at - indexed_at, 3),
            "evaluate_answers": round(evaluated_at - generated_at, 3),
            "total": round(evaluated_at - started_at, 3),
        },
        "samples": samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(json.dumps(_summarize_report(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved extractive RAG report: {output_path}")


def _build_sample(question, retrieval_results, answer) -> dict:
    return {
        "question_id": question.id,
        "question_title": question.title,
        "answerable": question.answerable,
        "gold_answer_doc_id": question.answer_doc_id,
        "retrieved_doc_ids": [result.document.id for result in retrieval_results],
        "generated_answer": answer.answer,
        "refused": answer.refused,
        "citations": [asdict(citation) for citation in answer.citations],
        "gold_answer": question.answer,
    }


def _summarize_report(report: dict) -> dict:
    return {
        "dataset": report["dataset"],
        "split": report["split"],
        "rag": report["rag"],
        "data": report["data"],
        "metrics": report["metrics"],
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
