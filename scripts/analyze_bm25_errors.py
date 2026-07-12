from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.domain.dataset import PrimeQAQuestion
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever, RetrievalResult
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_documents,
    load_primeqa_questions,
)

app = typer.Typer(help="分析 PrimeQA TechQA 上 BM25 检索失败案例。")


@app.command()
def main(
    split: Annotated[
        str,
        typer.Option(
            "--split",
            help="要分析的问题集合，可选值：dev、train。",
        ),
    ] = "dev",
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            help="判断失败案例时使用的检索深度。",
        ),
    ] = 10,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            help="最多保存多少条失败案例。",
        ),
    ] = 20,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="可选 JSON 输出路径。默认写入 artifacts/bm25_<split>_error_cases.json。",
        ),
    ] = None,
    k1: Annotated[
        float,
        typer.Option("--k1", help="BM25 k1 参数。"),
    ] = 1.5,
    b: Annotated[
        float,
        typer.Option("--b", help="BM25 b 参数。"),
    ] = 0.75,
) -> None:
    """保存 answer_doc_id 没有进入 top-k 的 BM25 失败案例。"""

    if top_k <= 0:
        raise typer.BadParameter("--top-k must be positive.")
    if limit <= 0:
        raise typer.BadParameter("--limit must be positive.")

    settings = ProjectSettings()
    training_dir = settings.primeqa_raw_dir / "TechQA" / "training_and_dev"
    documents_path = training_dir / "training_dev_technotes.sections.json"
    questions_path = _resolve_questions_path(training_dir, split)
    output_path = output or settings.artifact_dir / f"bm25_{split}_error_cases.json"

    _ensure_file_exists(documents_path)
    _ensure_file_exists(questions_path)

    started_at = time.perf_counter()
    documents = load_primeqa_documents(documents_path)
    questions = load_primeqa_questions(questions_path)
    retriever = BM25Retriever(k1=k1, b=b)
    retriever.fit(documents.values())

    evaluated_questions = [
        question
        for question in questions
        if question.answerable and question.answer_doc_id is not None
    ]
    error_cases = []

    for question in evaluated_questions:
        results = retriever.search(question.full_question, top_k=top_k)
        result_doc_ids = [result.document.id for result in results]
        if question.answer_doc_id in result_doc_ids:
            continue

        error_cases.append(_build_error_case(question, results))
        if len(error_cases) >= limit:
            break

    finished_at = time.perf_counter()
    report = {
        "dataset": "PrimeQA/TechQA",
        "split": split,
        "analysis": {
            "type": "bm25_top_k_miss",
            "top_k": top_k,
            "limit": limit,
            "saved_error_cases": len(error_cases),
        },
        "bm25": {
            "k1": k1,
            "b": b,
        },
        "data": {
            "documents": len(documents),
            "questions": len(questions),
            "evaluated_questions": len(evaluated_questions),
        },
        "paths": {
            "documents": str(documents_path),
            "questions": str(questions_path),
        },
        "timing_seconds": {
            "total": round(finished_at - started_at, 3),
        },
        "error_cases": error_cases,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(json.dumps(_summarize_report(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved BM25 error cases: {output_path}")


def _build_error_case(
    question: PrimeQAQuestion,
    results: list[RetrievalResult],
) -> dict:
    return {
        "question_id": question.id,
        "question_title": question.title,
        "question_text": question.text,
        "gold_answer_doc_id": question.answer_doc_id,
        "gold_answer": question.answer,
        "top_results": [
            {
                "rank": result.rank,
                "doc_id": result.document.id,
                "title": result.document.title,
                "score": round(result.score, 4),
            }
            for result in results
        ],
    }


def _summarize_report(report: dict) -> dict:
    return {
        "dataset": report["dataset"],
        "split": report["split"],
        "analysis": report["analysis"],
        "data": report["data"],
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
