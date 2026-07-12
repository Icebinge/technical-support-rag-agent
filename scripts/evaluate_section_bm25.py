from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.retrieval_evaluation import evaluate_retrieval
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
    load_primeqa_questions,
)
from ts_rag_agent.infrastructure.section_bm25_retriever import SectionBM25Retriever

app = typer.Typer(help="在 PrimeQA TechQA 数据集上评估 section 级 BM25 检索。")


@app.command()
def main(
    split: Annotated[
        str,
        typer.Option("--split", help="要评估的问题集合，可选值：dev、train。"),
    ] = "dev",
    top_k: Annotated[
        str,
        typer.Option("--top-k", help="用逗号分隔的 top-k 取值，例如：1,5,10。"),
    ] = "1,5,10",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="可选 JSON 报告路径。默认写入 artifacts/section_bm25_<split>_metrics.json。",
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
    """运行 section 级 BM25 检索评估并保存实验报告。"""

    settings = ProjectSettings()
    top_k_values = _parse_top_k_values(top_k)
    training_dir = settings.primeqa_raw_dir / "TechQA" / "training_and_dev"
    documents_path = training_dir / "training_dev_technotes.sections.json"
    questions_path = _resolve_questions_path(training_dir, split)
    output_path = output or settings.artifact_dir / f"section_bm25_{split}_metrics.json"

    _ensure_file_exists(documents_path)
    _ensure_file_exists(questions_path)

    started_at = time.perf_counter()
    documents_by_id = load_primeqa_documents(documents_path)
    sections_by_document = load_primeqa_document_sections(documents_path)
    questions = load_primeqa_questions(questions_path)
    loaded_at = time.perf_counter()

    retriever = SectionBM25Retriever(k1=k1, b=b)
    retriever.fit(documents_by_id.values(), sections_by_document)
    indexed_at = time.perf_counter()

    metrics = evaluate_retrieval(questions, retriever, top_k_values=top_k_values)
    finished_at = time.perf_counter()

    section_count = sum(len(sections) for sections in sections_by_document.values())
    report = {
        "dataset": "PrimeQA/TechQA",
        "split": split,
        "paths": {
            "documents": str(documents_path),
            "questions": str(questions_path),
        },
        "section_bm25": {
            "k1": k1,
            "b": b,
            "top_k_values": list(top_k_values),
            "aggregation": "max_section_score_per_parent_document",
        },
        "data": {
            "documents": len(documents_by_id),
            "sections": section_count,
            "questions": len(questions),
            "evaluated_questions": metrics.evaluated_questions,
        },
        "metrics": {
            "hit_at_k": {f"hit@{k}": value for k, value in metrics.hit_at_k.items()},
            "mrr": metrics.mrr,
        },
        "timing_seconds": {
            "load": round(loaded_at - started_at, 3),
            "index": round(indexed_at - loaded_at, 3),
            "evaluate": round(finished_at - indexed_at, 3),
            "total": round(finished_at - started_at, 3),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    typer.echo(f"Saved Section BM25 report: {output_path}")


def _parse_top_k_values(raw_value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    if not values:
        raise typer.BadParameter("--top-k must contain at least one integer.")
    if any(value <= 0 for value in values):
        raise typer.BadParameter("--top-k values must be positive integers.")
    return values


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
