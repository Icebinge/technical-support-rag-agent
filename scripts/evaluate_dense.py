from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.retrieval_evaluation import evaluate_retrieval
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.infrastructure.dense_embedding_cache import (
    default_dense_cache_path,
    load_or_build_document_embeddings,
)
from ts_rag_agent.infrastructure.dense_retriever import (
    DenseRetriever,
    SentenceTransformerEmbeddingModel,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_documents,
    load_primeqa_questions,
)

app = typer.Typer(help="在 PrimeQA TechQA 数据集上评估 Dense Retrieval baseline。")


@app.command()
def main(
    split: Annotated[
        str,
        typer.Option("--split", help="要评估的问题集合，可选值：dev、train。"),
    ] = "dev",
    model_name: Annotated[
        str,
        typer.Option("--model-name", help="sentence-transformers 模型名称。"),
    ] = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: Annotated[
        str,
        typer.Option("--top-k", help="用逗号分隔的 top-k 取值，例如：1,5,10。"),
    ] = "1,5,10",
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="文档向量编码 batch size。"),
    ] = 64,
    document_text_max_chars: Annotated[
        int,
        typer.Option("--document-text-max-chars", help="每篇文档用于编码的最大字符数。"),
    ] = 1600,
    query_prefix: Annotated[
        str,
        typer.Option("--query-prefix", help="查询文本编码前缀，例如 E5 使用 'query: '。"),
    ] = "",
    document_prefix: Annotated[
        str,
        typer.Option("--document-prefix", help="文档文本编码前缀，例如 E5 使用 'passage: '。"),
    ] = "",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="可选 JSON 报告路径。默认写入 artifacts/dense_<split>_metrics.json。",
        ),
    ] = None,
    cache_path: Annotated[
        Path | None,
        typer.Option("--cache-path", help="可选文档向量缓存路径。"),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="不读取也不写入文档向量缓存。"),
    ] = False,
    device: Annotated[
        str | None,
        typer.Option("--device", help="可选推理设备，例如 cpu、cuda。"),
    ] = None,
) -> None:
    """运行 Dense Retrieval 检索评估并保存可复现实验报告。"""

    if batch_size <= 0:
        raise typer.BadParameter("--batch-size must be positive.")
    if document_text_max_chars <= 0:
        raise typer.BadParameter("--document-text-max-chars must be positive.")

    settings = ProjectSettings()
    top_k_values = _parse_top_k_values(top_k)
    training_dir = settings.primeqa_raw_dir / "TechQA" / "training_and_dev"
    documents_path = training_dir / "training_dev_technotes.sections.json"
    questions_path = _resolve_questions_path(training_dir, split)
    output_path = output or settings.artifact_dir / f"dense_{split}_metrics.json"
    resolved_cache_path = cache_path or default_dense_cache_path(
        settings.data_dir,
        model_name,
        document_text_max_chars,
        document_prefix,
    )

    _ensure_file_exists(documents_path)
    _ensure_file_exists(questions_path)

    started_at = time.perf_counter()
    documents_by_id = load_primeqa_documents(documents_path)
    documents = list(documents_by_id.values())
    questions = load_primeqa_questions(questions_path)
    loaded_at = time.perf_counter()

    typer.echo(f"Loading embedding model: {model_name}")
    encoder = SentenceTransformerEmbeddingModel(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        show_progress_bar=False,
    )
    model_loaded_at = time.perf_counter()

    if not no_cache and not resolved_cache_path.exists():
        typer.echo("Encoding documents for dense retrieval. This may take a while on CPU.")
    embeddings, cache_status = load_or_build_document_embeddings(
        encoder=encoder,
        documents=documents,
        model_name=model_name,
        document_text_max_chars=document_text_max_chars,
        document_prefix=document_prefix,
        cache_path=resolved_cache_path,
        no_cache=no_cache,
    )
    encoded_at = time.perf_counter()

    retriever = DenseRetriever(
        encoder,
        document_text_max_chars=document_text_max_chars,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
    )
    retriever.fit_embeddings(documents, embeddings)
    metrics = evaluate_retrieval(questions, retriever, top_k_values=top_k_values)
    finished_at = time.perf_counter()

    report = {
        "dataset": "PrimeQA/TechQA",
        "split": split,
        "paths": {
            "documents": str(documents_path),
            "questions": str(questions_path),
            "embedding_cache": None if no_cache else str(resolved_cache_path),
        },
        "dense": {
            "model_name": model_name,
            "batch_size": batch_size,
            "document_text_max_chars": document_text_max_chars,
            "query_prefix": query_prefix,
            "document_prefix": document_prefix,
            "top_k_values": list(top_k_values),
            "cache_status": cache_status,
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
            "load_model": round(model_loaded_at - loaded_at, 3),
            "document_embeddings": round(encoded_at - model_loaded_at, 3),
            "evaluate": round(finished_at - encoded_at, 3),
            "total": round(finished_at - started_at, 3),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    typer.echo(f"Saved Dense Retrieval report: {output_path}")


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
