from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.retrieval_evaluation import evaluate_retrieval
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_documents,
    load_primeqa_questions,
)
from ts_rag_agent.infrastructure.reranker import (
    CrossEncoderPairScoringModel,
    RerankingRetriever,
)

app = typer.Typer(help="在 PrimeQA TechQA 数据集上评估 BM25 + reranker 重排。")


@app.command()
def main(
    split: Annotated[
        str,
        typer.Option("--split", help="要评估的问题集合，可选值：dev、train。"),
    ] = "dev",
    model_name: Annotated[
        str,
        typer.Option("--model-name", help="CrossEncoder reranker 模型名称。"),
    ] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: Annotated[
        str,
        typer.Option("--top-k", help="用逗号分隔的 top-k 取值，例如：1,5,10。"),
    ] = "1,5,10",
    candidate_top_k: Annotated[
        int,
        typer.Option("--candidate-top-k", help="BM25 召回后交给 reranker 的候选数量。"),
    ] = 50,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="reranker 成对打分 batch size。"),
    ] = 32,
    max_length: Annotated[
        int | None,
        typer.Option("--max-length", help="CrossEncoder 最大输入长度。"),
    ] = 512,
    document_text_max_chars: Annotated[
        int,
        typer.Option("--document-text-max-chars", help="每篇候选文档输入 reranker 的最大字符数。"),
    ] = 1600,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="可选 JSON 报告路径。默认写入 artifacts/reranker_<split>_metrics.json。",
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="可选推理设备，例如 cpu、cuda。"),
    ] = None,
) -> None:
    """运行 BM25 召回 + CrossEncoder 重排评估。"""

    if candidate_top_k <= 0:
        raise typer.BadParameter("--candidate-top-k must be positive.")
    if batch_size <= 0:
        raise typer.BadParameter("--batch-size must be positive.")
    if document_text_max_chars <= 0:
        raise typer.BadParameter("--document-text-max-chars must be positive.")

    settings = ProjectSettings()
    top_k_values = _parse_top_k_values(top_k)
    training_dir = settings.primeqa_raw_dir / "TechQA" / "training_and_dev"
    documents_path = training_dir / "training_dev_technotes.sections.json"
    questions_path = _resolve_questions_path(training_dir, split)
    output_path = output or settings.artifact_dir / f"reranker_{split}_metrics.json"

    _ensure_file_exists(documents_path)
    _ensure_file_exists(questions_path)

    started_at = time.perf_counter()
    documents_by_id = load_primeqa_documents(documents_path)
    documents = list(documents_by_id.values())
    questions = load_primeqa_questions(questions_path)
    loaded_at = time.perf_counter()

    bm25_retriever = BM25Retriever()
    bm25_retriever.fit(documents)
    indexed_at = time.perf_counter()

    typer.echo(f"Loading reranker model: {model_name}")
    scorer = CrossEncoderPairScoringModel(
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    model_loaded_at = time.perf_counter()

    reranking_retriever = RerankingRetriever(
        candidate_retriever=bm25_retriever,
        scorer=scorer,
        candidate_top_k=candidate_top_k,
        document_text_max_chars=document_text_max_chars,
    )
    metrics = evaluate_retrieval(questions, reranking_retriever, top_k_values=top_k_values)
    finished_at = time.perf_counter()

    report = {
        "dataset": "PrimeQA/TechQA",
        "split": split,
        "paths": {
            "documents": str(documents_path),
            "questions": str(questions_path),
        },
        "reranker": {
            "candidate_retriever": "BM25",
            "model_name": model_name,
            "candidate_top_k": candidate_top_k,
            "batch_size": batch_size,
            "max_length": max_length,
            "document_text_max_chars": document_text_max_chars,
            "top_k_values": list(top_k_values),
        },
        "data": {
            "documents": len(documents),
            "questions": len(questions),
            "evaluated_questions": metrics.evaluated_questions,
        },
        "metrics": {
            "hit_at_k": {f"hit@{k}": value for k, value in metrics.hit_at_k.items()},
            "mrr": metrics.mrr,
        },
        "timing_seconds": {
            "load_data": round(loaded_at - started_at, 3),
            "bm25_index": round(indexed_at - loaded_at, 3),
            "load_reranker": round(model_loaded_at - indexed_at, 3),
            "evaluate": round(finished_at - model_loaded_at, 3),
            "total": round(finished_at - started_at, 3),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    typer.echo(f"Saved reranker report: {output_path}")


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
